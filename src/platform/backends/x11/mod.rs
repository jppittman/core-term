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

use crate::platform::backends::{BackendEvent, CellCoords, CellRect, Driver, TextRunStyle};
use anyhow::Result;
use log::{debug, error, warn, info, trace};
use std::os::unix::io::RawFd;

// Declare submodules. These contain the specialized logic for different aspects of X11 handling.
pub mod connection;
pub mod event;
pub mod graphics;
pub mod window;

use connection::Connection;
use graphics::Graphics;
use window::{CursorVisibility, Window}; // Import CursorVisibility enum

/// Represents the focus state of the window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FocusState {
    Focused,
    Unfocused,
}

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
        })
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
        match event::process_pending_events(&self.connection, &mut self.window, &mut self.has_focus)
        {
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
    fn get_font_dimensions(&self) -> (usize, usize) {
        let (w, h) = self.graphics.font_dimensions_pixels();
        (w as usize, h as usize)
    }

    /// Retrieves the current dimensions of the display area (window client area) in pixels.
    ///
    /// Delegates to `Window::current_dimensions_pixels()`.
    fn get_display_dimensions_pixels(&self) -> (u16, u16) {
        self.window.current_dimensions_pixels()
    }

    /// Clears the entire display area with the specified background color.
    ///
    /// Delegates to `Graphics::clear_all()`.
    /// Note: `Graphics::clear_all` currently uses placeholder dimensions and needs
    /// updating to accept current window dimensions for accurate clearing.
    fn clear_all(&mut self, bg: crate::color::Color) -> Result<()> {
        // TODO: Update Graphics::clear_all to accept width & height, then pass them:
        // let (width_px, height_px) = self.window.current_dimensions_pixels();
        // self.graphics.clear_all_with_dimensions(&self.connection, bg, width_px, height_px)
        self.graphics.clear_all(&self.connection, bg)
    }

    /// Draws a run of text characters at a given cell coordinate with a specified style.
    ///
    /// Delegates to `Graphics::draw_text_run()`.
    fn draw_text_run(&mut self, coords: CellCoords, text: &str, style: TextRunStyle) -> Result<()> {
        self.graphics
            .draw_text_run(&self.connection, coords, text, style)
    }

    /// Fills a rectangular area of cells with a specified concrete color.
    ///
    /// Delegates to `Graphics::fill_rect()`.
    fn fill_rect(&mut self, rect: CellRect, color: crate::color::Color) -> Result<()> {
        self.graphics.fill_rect(&self.connection, rect, color)
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
    fn set_cursor_visibility(&mut self, visibility: CursorVisibility) { // Changed parameter
        self.window
            .set_native_cursor_visibility(&self.connection, visibility);
    }

    /// Informs the driver of focus changes, typically called by the application core.
    ///
    /// The X11 driver also detects focus changes via `FocusIn`/`FocusOut` events
    /// in `event::process_pending_events`, which directly updates `self.has_focus`.
    /// This method allows external setting of focus state if needed.
    fn set_focus(&mut self, focus_state: FocusState) { // Changed parameter
        info!(
            "XDriver::set_focus called by application core with: {:?}", // Updated log
            focus_state
        );
        self.has_focus = match focus_state { // Updated logic
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
