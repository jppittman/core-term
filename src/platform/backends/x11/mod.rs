// src/platform/backends/x11/mod.rs

//! X11 backend driver implementation for the terminal.
//!
//! This module defines `XDriver`, which serves as an orchestrator for various
//! X11 functionalities, each handled by its respective submodule:
//! - `connection`: Manages the basic connection to the X server.
//! - `window`: Handles X11 window creation, properties, and interactions.
//! - `graphics`: Manages fonts, colors, and drawing operations using Xft and Xlib.
//! - `event`: Processes X11 events and translates them into generic `BackendEvent`s.
//!
//! `XDriver` implements the `crate::platform::backends::Driver` trait, providing
//! a platform-specific layer for the terminal application's core logic.

// --- Public Constants ---

/// Default width of the terminal window in character cells, used if not otherwise specified.
pub const DEFAULT_WINDOW_WIDTH_CHARS: usize = 80;
/// Default height of the terminal window in character cells, used if not otherwise specified.
pub const DEFAULT_WINDOW_HEIGHT_CHARS: usize = 24;

use crate::platform::backends::{
    BackendEvent, CellCoords, CellRect, Driver, PlatformState, RenderCommand, TextRunStyle,
}; // Added PlatformState, RenderCommand
use anyhow::Result;
use log::{debug, error, info, trace, warn};
use std::os::unix::io::RawFd;

// Declare submodules. These contain the specialized logic for different aspects of X11 handling.
pub mod connection;
pub mod event;
pub mod font_manager;
pub mod graphics;
pub mod selection;
pub mod window; // Added selection module

use crate::platform::backends::x11::selection::SelectionAtoms; // For XDriver field
use connection::Connection;
use graphics::Graphics;
use window::{CursorVisibility, Window}; // Import CursorVisibility enum
use x11::xlib; // Added import

/// Represents the focus state of the window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FocusState {
    Focused,
    Unfocused,
}
pub const TRAIT_ATOM_ID_PRIMARY: u64 = 1;
pub const TRAIT_ATOM_ID_CLIPBOARD: u64 = 2;
pub const TRAIT_ATOM_ID_UTF8_STRING: u64 = 10;

/// Implements the `Driver` trait for the X11 windowing system.
///
/// `XDriver` coordinates X11 resources and operations through its contained submodules:
/// `Connection`, `Window`, `Graphics`, and `event` (module-level functions).
/// It handles window setup, event processing, drawing primitives, and resource cleanup.
pub struct XDriver {
    connection: Connection,
    window: Window,
    graphics: Graphics,
    has_focus: bool, // Tracks if the application window currently has input focus.
    selection_atoms: SelectionAtoms,
    selection_text: Option<String>, // Stores text if we own a selection
    framebuffer: Vec<u8>, // Dummy framebuffer (X11 uses Xft rendering currently)
}

impl XDriver {
    /// Creates and initializes all components of the X11 driver.
    ///
    /// This constructor orchestrates the setup of the X connection,
    /// font and color resources (graphics pre-initialization), the X window itself,
    /// and finally the full graphics context tied to the window.
    /// It also configures window manager hints and maps the window for display.
    ///
    /// # Returns
    /// * `Ok(XDriver)`: A fully initialized XDriver instance.
    /// * `Err(anyhow::Error)`: If any critical part of the initialization fails
    ///   (e.g., cannot connect to X server, load font, or create window).
    pub fn new() -> Result<Self> {
        info!("XDriver::new() called - initializing X11 driver components.");

        let connection = Connection::new().map_err(|e| {
            error!("Failed to establish X11 connection: {}", e);
            // Error is propagated, Drop handlers will clean up if partially initialized.
            e
        })?;
        info!("X11 connection established successfully.");

        let selection_atoms = SelectionAtoms::new(&connection).map_err(|e| {
            error!("Failed to intern selection atoms: {}", e);
            e
        })?;
        info!("X11 selection atoms interned successfully.");

        // Stage 1 of Graphics init: Load font, determine metrics, pre-allocate ANSI colors, get default bg pixel.
        // This is done before window creation as font metrics are needed for initial window sizing
        // and the initial background pixel is needed for window attributes.
        let pre_graphics_data = Graphics::load_font_and_colors(&connection).map_err(|e| {
            error!("Failed during initial font and color loading: {}", e);
            e
        })?;
        info!("Font loaded and initial colors allocated by Graphics module.");

        let font_width_px = pre_graphics_data.font_width_px;
        let font_height_px = pre_graphics_data.font_height_px;
        let initial_bg_pixel = pre_graphics_data.default_bg_pixel_value;

        debug!(
            "Using font metrics from Graphics: width={}, height={}, initial_bg_pixel={}",
            font_width_px, font_height_px, initial_bg_pixel
        );

        // Calculate initial window dimensions in pixels based on default character cell counts and loaded font metrics.
        let initial_pixel_width = (DEFAULT_WINDOW_WIDTH_CHARS as u32 * font_width_px) as u16;
        let initial_pixel_height = (DEFAULT_WINDOW_HEIGHT_CHARS as u32 * font_height_px) as u16;

        let mut window = Window::new(
            &connection,
            initial_pixel_width,
            initial_pixel_height,
            initial_bg_pixel,
        )
        .map_err(|e| {
            error!("Failed to create X11 window: {}", e);
            e
        })?;
        info!("X11 window created successfully with ID: {}", window.id());

        // Stage 2 of Graphics init: Create XftDraw for the window and the clear GC.
        // This consumes pre_graphics_data.
        let graphics = Graphics::new(&connection, window.id(), pre_graphics_data).map_err(|e| {
            error!("Failed to finalize Graphics setup: {}", e);
            e
        })?;
        info!("Graphics module fully initialized.");

        // Setup window manager protocols and hints using actual font metrics.
        window
            .setup_protocols_and_hints(&connection, font_width_px, font_height_px)
            .map_err(|e| {
                error!("Failed to setup window protocols and hints: {}", e);
                e
            })?;
        info!("Window protocols and hints configured using actual font metrics.");

        // Map the window to make it visible and flush the command buffer.
        window.map_and_flush(&connection);
        info!("Window mapped and initial flush completed.");

        Ok(XDriver {
            connection,
            window,
            graphics,
            has_focus: true, // Assume window has focus initially. Event processing will update this.
            selection_atoms,
            selection_text: None,
            framebuffer: vec![0u8; 4], // Minimal dummy buffer (X11 uses Xft, not framebuffer yet)
        })
    }

    /// Takes ownership of a given X11 selection (e.g., PRIMARY or CLIPBOARD).
    ///
    /// # Arguments
    /// * `selection_name_atom`: The atom identifying the selection to own (e.g., `self.selection_atoms.primary`).
    /// * `text`: The string content to associate with this selection.
    fn own_selection_internal(&mut self, selection_name_atom: xlib::Atom, text: String) {
        // SAFETY: FFI call. `connection.display()`, `window.id()`, and `selection_name_atom` must be valid.
        // `xlib::CurrentTime` is standard for timestamp.
        unsafe {
            xlib::XSetSelectionOwner(
                self.connection.display(),
                selection_name_atom,
                self.window.id(),
                xlib::CurrentTime, // Using CurrentTime is common
            );
            // No direct confirmation of success from XSetSelectionOwner,
            // but we can check if we are now the owner. This is usually implicit.
            // XFlush might be needed if other clients need to know about the ownership change immediately.
            xlib::XFlush(self.connection.display());
        }
        self.selection_text = Some(text);
        // It's good to log which selection we attempted to own.
        // For more robust atom-to-name, we'd need XGetAtomName, but that's another round trip.
        // For now, just use the atom value.
        info!(
            "Attempted to own selection (atom ID: {}). Stored text length: {}.",
            selection_name_atom,
            self.selection_text.as_ref().map_or(0, |s| s.len())
        );
    }

    /// Requests data from the current owner of a specified X11 selection.
    ///
    /// The data will be delivered via a `SelectionNotify` event.
    ///
    /// # Arguments
    /// * `selection_name_atom`: The atom identifying the selection to request (e.g., `self.selection_atoms.clipboard`).
    /// * `target_atom`: The desired format of the data (e.g., `self.selection_atoms.utf8_string`).
    fn request_selection_data_internal(
        &mut self,
        selection_name_atom: xlib::Atom,
        target_atom: xlib::Atom,
    ) {
        // Property atom where the selection owner should place the data.
        // Using the selection name atom itself for the property is a common convention,
        // or a dedicated atom like "XSEL_DATA". Let's use the selection name atom for simplicity here.
        // The property must be set on *our* window (`self.window.id()`).
        let property_to_set = selection_name_atom; // Or a custom atom

        // SAFETY: FFI call. Ensure all parameters are valid.
        // `xlib::CurrentTime` is standard for timestamp.
        unsafe {
            xlib::XConvertSelection(
                self.connection.display(),
                selection_name_atom,
                target_atom,
                property_to_set,  // Property on our window for the result
                self.window.id(), // Our window is the requestor
                xlib::CurrentTime,
            );
            // XFlush might be needed to ensure the request is sent promptly.
            xlib::XFlush(self.connection.display());
        }
        info!("Requested selection data (selection atom ID: {}, target atom ID: {}) for property atom ID: {}",
              selection_name_atom, target_atom, property_to_set);
    }
}

impl Driver for XDriver {
    /// Creates and initializes a new `XDriver` instance as required by the `Driver` trait.
    ///
    /// This delegates to the `XDriver`'s own `new` method.
    fn new() -> Result<Self> {
        info!("XDriver (Driver trait): new() called, creating new XDriver instance.");
        // Self::new() refers to XDriver::new() due to Rust's method resolution.
        Self::new()
    }

    /// Returns an optional raw file descriptor for event monitoring.
    ///
    /// Delegates to `Connection::get_event_fd()`. The FD can be used with `epoll` or `select`.
    fn get_event_fd(&self) -> Option<RawFd> {
        self.connection.get_event_fd()
    }

    /// Processes pending X11 events.
    ///
    /// Delegates to `event::process_pending_events`, which handles X event polling,
    /// translation to `BackendEvent`s, and updates window state (dimensions, focus).
    fn process_events(&mut self) -> Result<Vec<BackendEvent>> {
        // Logging for this can be noisy, so it's primarily within the event module.
        match event::process_pending_events(
            &self.connection,
            &mut self.window,
            &mut self.has_focus,
            &self.selection_atoms,
            self.selection_text.as_ref(),
        ) {
            Ok(events) => {
                if !events.is_empty() {
                    debug!("XDriver processed {} backend events.", events.len());
                }
                Ok(events)
            }
            Err(e) => {
                error!("Error processing X11 events in XDriver: {}", e);
                Err(e)
            }
        }
    }

    /// Retrieves the dimensions of a single character cell in pixels.
    ///
    /// Delegates to `Graphics::font_dimensions_pixels()`.
    fn get_platform_state(&self) -> PlatformState {
        let (font_w, font_h) = self.graphics.font_dimensions_pixels();
        let (display_w, display_h) = self.window.current_dimensions_pixels();
        PlatformState {
            event_fd: self.connection.get_event_fd(),
            font_cell_width_px: font_w as usize,
            font_cell_height_px: font_h as usize,
            scale_factor: 1.0, // Assuming 1.0 for X11 unless HiDPI is explicitly handled
            display_width_px: display_w,
            display_height_px: display_h,
        }
    }

    fn execute_render_commands(&mut self, commands: Vec<RenderCommand>) -> Result<()> {
        for command in commands {
            match command {
                RenderCommand::ClearAll { bg } => {
                    // Assuming graphics.clear_all is updated or adapted.
                    // For now, using existing clear_all which has a TODO for dimensions.
                    // A proper fix involves modifying graphics.clear_all.
                    // As a temporary measure, if graphics.clear_all isn't fixed,
                    // this might not clear correctly or use placeholder dimensions.
                    // Or, we can use fill_rect to clear the whole window.
                    let (w, h) = self.window.current_dimensions_pixels();
                    // self.graphics.clear_all(&self.connection, bg)?; // if clear_all is updated
                    // Using fill_rect_absolute_px to clear the entire area.
                    // The calculation for full_window_rect in cell terms is not needed here
                    // as fill_rect_absolute_px takes pixel dimensions.
                    self.graphics
                        .fill_rect_absolute_px(&self.connection, 0, 0, w, h, bg)?;
                }
                RenderCommand::DrawTextRun {
                    x,
                    y,
                    text,
                    fg,
                    bg,
                    flags,
                    is_selected,
                } => {
                    let coords = CellCoords { x, y };
                    let (actual_fg, actual_bg) = if is_selected {
                        (bg, fg) // Reverse video for selection
                    } else {
                        (fg, bg)
                    };
                    let style = TextRunStyle {
                        fg: actual_fg,
                        bg: actual_bg,
                        flags,
                    };
                    self.graphics
                        .draw_text_run(&self.connection, coords, &text, style)?;
                }
                RenderCommand::FillRect {
                    x,
                    y,
                    width,
                    height,
                    color,
                    is_selection_bg: _, // Currently not changing color based on is_selection_bg, assuming `color` is final.
                } => {
                    let rect = CellRect {
                        x,
                        y,
                        width,
                        height,
                    };
                    self.graphics.fill_rect(&self.connection, rect, color)?;
                }
                RenderCommand::SetCursorVisibility { visible } => {
                    let visibility = if visible {
                        CursorVisibility::Shown
                    } else {
                        CursorVisibility::Hidden
                    };
                    self.window
                        .set_native_cursor_visibility(&self.connection, visibility);
                }
                RenderCommand::SetWindowTitle { title } => {
                    self.window.set_title(&self.connection, &title)?;
                }
                RenderCommand::RingBell => {
                    self.window.bell(&self.connection);
                }
                RenderCommand::PresentFrame => {
                    // Call the XDriver's own present method, which handles flushing.
                    self.present()?;
                }
            }
        }
        Ok(())
    }

    /// Presents the composed frame to the display.
    ///
    /// For X11, this typically means flushing the X command buffer to ensure all
    /// drawing commands are sent to the server.
    fn present(&mut self) -> Result<()> {
        // SAFETY: XFlush is safe to call with a valid display pointer.
        // Connection::display() returns the pointer, which is non-null if connection is active.
        if !self.connection.display().is_null() {
            unsafe {
                x11::xlib::XFlush(self.connection.display());
            }
            trace!("XDriver::present() flushed X display.");
        } else {
            warn!("XDriver::present() called on a closed or invalid X display connection.");
        }
        Ok(())
    }

    /// Sets the window title.
    ///
    /// Delegates to `Window::set_title()`.
    fn set_title(&mut self, title: &str) {
        if let Err(e) = self.window.set_title(&self.connection, title) {
            error!("XDriver failed to set window title: {}", e);
        }
    }

    /// Rings the terminal bell.
    ///
    /// Delegates to `Window::bell()`.
    fn bell(&mut self) {
        self.window.bell(&self.connection);
    }

    /// Sets the visibility of the native X11 mouse pointer over the window.
    ///
    /// Adapts the boolean `visible` to the `CursorVisibility` enum required by the `Window` module.
    fn set_cursor_visibility(&mut self, visibility: CursorVisibility) {
        // Changed parameter
        self.window
            .set_native_cursor_visibility(&self.connection, visibility);
    }

    /// Informs the driver of focus changes, typically called by the application core.
    ///
    /// The X11 driver also detects focus changes via `FocusIn`/`FocusOut` events
    /// in `event::process_pending_events`, which directly updates `self.has_focus`.
    /// This method allows external setting of focus state if needed.
    fn set_focus(&mut self, focus_state: FocusState) {
        // Changed parameter
        info!(
            "XDriver::set_focus called by application core with: {:?}", // Updated log
            focus_state
        );
        self.has_focus = match focus_state {
            // Updated logic
            FocusState::Focused => true,
            FocusState::Unfocused => false,
        };
        // This state change could be used to influence rendering (e.g., cursor style),
        // though such visual changes are typically triggered by the FocusGained/FocusLost BackendEvents.
    }

    /// Cleans up all X11 resources managed by the driver.
    ///
    /// This method ensures that resources are released in the correct order:
    /// graphics resources first, then window resources, and finally the connection
    /// to the X server is closed. It is crucial for graceful shutdown.
    /// This method is idempotent.
    ///
    /// # Returns
    /// * `Ok(())`: If all cleanup operations succeed or were already performed.
    /// * `Err(anyhow::Error)`: If an error occurs during cleanup of `Graphics` or `Connection`.
    ///   Errors from `Window::cleanup` are logged but not propagated from this function.
    fn own_selection(&mut self, selection_name_atom_u64: u64, text: String) {
        // Map abstract u64 IDs from the trait to concrete X11 atoms.
        // These IDs must match those used in AppOrchestrator.
        const TRAIT_ATOM_ID_PRIMARY: u64 = 1;
        const TRAIT_ATOM_ID_CLIPBOARD: u64 = 2;

        let actual_atom = match selection_name_atom_u64 {
            TRAIT_ATOM_ID_PRIMARY => self.selection_atoms.primary,
            TRAIT_ATOM_ID_CLIPBOARD => self.selection_atoms.clipboard,
            _ => {
                warn!(
                    "XDriver::own_selection (trait): Received unknown abstract atom ID: {}",
                    selection_name_atom_u64
                );
                return;
            }
        };
        // Call the internal method that expects an xlib::Atom
        self.own_selection_internal(actual_atom, text);
    }

    fn request_selection_data(&mut self, selection_name_atom_u64: u64, target_atom_u64: u64) {
        // Map abstract u64 IDs from the trait to concrete X11 atoms.
        // Add more target mappings as needed (e.g., TARGETS)

        let actual_selection_atom = match selection_name_atom_u64 {
            TRAIT_ATOM_ID_PRIMARY => self.selection_atoms.primary,
            TRAIT_ATOM_ID_CLIPBOARD => self.selection_atoms.clipboard,
            _ => {
                warn!("XDriver::request_selection_data (trait): Received unknown abstract selection atom ID: {}", selection_name_atom_u64);
                return;
            }
        };

        let actual_target_atom = match target_atom_u64 {
            TRAIT_ATOM_ID_UTF8_STRING => self.selection_atoms.utf8_string,
            // Example: if AppOrchestrator used an abstract ID for TARGETS (e.g., 11)
            // const TRAIT_ATOM_ID_TARGETS: u64 = 11;
            // TRAIT_ATOM_ID_TARGETS => self.selection_atoms.targets,
            _ => {
                warn!("XDriver::request_selection_data (trait): Received unknown abstract target atom ID: {}", target_atom_u64);
                return;
            }
        };
        // Call the internal method that expects xlib::Atom
        self.request_selection_data_internal(actual_selection_atom, actual_target_atom);
    }

    fn cleanup(&mut self) -> Result<()> {
        info!("XDriver::cleanup() called, releasing X11 resources.");
        // Cleanup components in reverse order of dependency or creation.
        // Graphics resources (XftDraw, XftFont, GC, XftColors) depend on the Connection and Window.
        self.graphics.cleanup(&self.connection)?; // This may log errors but returns Result.

        // Window resources (the XID itself) depend on the Connection.
        self.window.cleanup(&self.connection); // This logs errors but doesn't return Result.

        // Finally, close the connection to the X server.
        self.connection.cleanup().map_err(|e| {
            error!("Error during XDriver's connection cleanup: {}", e);
            e
        })
    }

    fn get_framebuffer_mut(&mut self) -> &mut [u8] {
        // X11 currently uses Xft for rendering, not framebuffer
        // Return dummy buffer for trait compliance
        &mut self.framebuffer
    }

    fn get_framebuffer_size(&self) -> (usize, usize) {
        // X11 doesn't use framebuffer yet
        (0, 0)
    }
}

/// Ensures X11 resources are cleaned up when the `XDriver` instance is dropped.
///
/// This `Drop` implementation calls `cleanup()` to release X server resources
/// in the correct order. Errors during cleanup in `drop` are logged but not
/// propagated, as `drop` cannot return `Result`.
impl Drop for XDriver {
    fn drop(&mut self) {
        info!("Dropping XDriver, ensuring all resources are cleaned up via self.cleanup().");
        // Call the comprehensive cleanup method.
        // Order of cleanup within `self.cleanup()` is graphics -> window -> connection.
        if let Err(e) = self.cleanup() {
            error!("Error during XDriver cleanup in drop: {}", e);
        }
    }
}
