// src/platform/backends/x11/window.rs
#![allow(non_snake_case)] // Allow non-snake case for X11 types

use super::connection::Connection;
use anyhow::{anyhow, Context, Result}; // Combined anyhow
use log::{debug, error, info, trace, warn};
use std::ffi::CString;
use std::mem;

// X11 library imports
use libc::{c_char, c_int, c_uint};
use x11::xlib;

/// Defines the desired visibility state for the native X11 mouse cursor
/// when it is positioned over the terminal window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CursorVisibility {
    /// The native mouse cursor should be visible.
    Shown,
    /// The native mouse cursor should be hidden (invisible).
    Hidden,
}

/// Represents an X11 window and its associated properties.
///
/// This struct manages the window ID, important window manager atoms,
/// current dimensions, and native cursor visibility. It provides methods
/// for window creation, setup, and interaction, such as setting the title
/// or ringing the bell.
///
/// Cleanup of X11 resources is handled by the `cleanup` method, which should
/// be called explicitly by the owner (e.g., `XDriver`) before the `Connection`
/// is closed. The `Drop` trait provides a fallback log message if cleanup
/// was not performed.
#[derive(Debug)]
pub struct Window {
    id: xlib::Window,
    wm_delete_window: xlib::Atom, // Atom for WM_DELETE_WINDOW protocol
    protocols_atom: xlib::Atom,   // Atom for WM_PROTOCOLS
    current_pixel_width: u16,
    current_pixel_height: u16,
    is_native_cursor_visible: bool, // Tracks the current state of the X11 cursor
    initial_bg_pixel: xlib::Atom,   // Background pixel value used at window creation
}

/// Value for the Xterm cursor shape, used when making the cursor visible.
const XC_XTERM: c_uint = 152;

impl Window {
    /// Creates a new X11 window.
    ///
    /// Initializes the window on the X server with the specified dimensions and
    /// background color. The window is not yet visible (mapped) or configured
    /// with WM protocols until subsequent methods are called.
    ///
    /// # Arguments
    ///
    /// * `connection`: A reference to the active X11 `Connection`.
    /// * `width_px`: The desired initial width of the window in pixels.
    /// * `height_px`: The desired initial height of the window in pixels.
    /// * `bg_pixel_val`: The pixel value for the window's initial background color.
    ///   This is an XID (typically `u64` or `xlib::Atom`).
    ///
    /// # Returns
    ///
    /// * `Ok(Window)`: A `Window` instance representing the newly created X11 window.
    /// * `Err(anyhow::Error)`: If `XCreateWindow` fails.
    pub fn new(
        connection: &Connection,
        width_px: u16,
        height_px: u16,
        bg_pixel_val: xlib::Atom,
    ) -> Result<Self> {
        info!(
            "Creating X11 window: {}x{}px, bg_pixel: {}",
            width_px, height_px, bg_pixel_val
        );
        let display = connection.display();
        let screen = connection.screen();
        let visual = connection.visual();
        let colormap = connection.colormap(); // Needed for XSetWindowAttributes

        // SAFETY: Xlib calls involving FFI. Ensure connection and its members are valid.
        let window_id = unsafe {
            let root_window = xlib::XRootWindow(display, screen);
            let border_width = 0; // No border managed by this window itself

            let mut attributes: xlib::XSetWindowAttributes = mem::zeroed();
            attributes.colormap = colormap;
            attributes.background_pixel = bg_pixel_val;
            attributes.border_pixel = bg_pixel_val; // Though border_width is 0
            attributes.event_mask = xlib::ExposureMask // Redraw events
                | xlib::KeyPressMask       // Keyboard input
                // | xlib::KeyReleaseMask    // If needed later for specific key release events
                | xlib::StructureNotifyMask  // For ConfigureNotify (resize/move events)
                | xlib::FocusChangeMask; // For FocusIn/FocusOut events
                                         // TODO: Add ButtonPressMask, ButtonReleaseMask, PointerMotionMask for mouse event handling.

            xlib::XCreateWindow(
                display,
                root_window,
                0,                                    // x position (top-left relative to parent)
                0,                                    // y position
                width_px as c_uint,                   // width
                height_px as c_uint,                  // height
                border_width,                         // border width
                xlib::XDefaultDepth(display, screen), // depth from parent
                xlib::InputOutput as c_uint,          // class: InputOutput window
                visual,                               // visual from parent
                xlib::CWColormap | xlib::CWBackPixel | xlib::CWBorderPixel | xlib::CWEventMask, // attribute mask
                &mut attributes,
            )
        };

        if window_id == 0 {
            return Err(anyhow!("XCreateWindow failed"));
        }
        debug!(
            "X window created (ID: {}), initial size: {}x{}",
            window_id, width_px, height_px
        );

        Ok(Self {
            id: window_id,
            wm_delete_window: 0, // Initialized by setup_protocols_and_hints
            protocols_atom: 0,   // Initialized by setup_protocols_and_hints
            current_pixel_width: width_px,
            current_pixel_height: height_px,
            is_native_cursor_visible: true, // Default state
            initial_bg_pixel: bg_pixel_val,
        })
    }

    /// Sets up window manager (WM) protocols and hints for the window.
    ///
    /// This includes:
    /// - Registering interest in the `WM_DELETE_WINDOW` protocol.
    /// - Setting the initial window title to "core-term" (standard and `_NET_WM_NAME`).
    /// - Providing size hints to the WM based on font character dimensions (resize increments, min size).
    ///
    /// # Arguments
    ///
    /// * `connection`: A reference to the active X11 `Connection`.
    /// * `font_char_width`: The width of a single character cell in pixels.
    /// * `font_char_height`: The height of a single character cell in pixels.
    ///
    /// # Returns
    ///
    /// * `Ok(())`: If setup is successful.
    /// * `Err(anyhow::Error)`: If creating CString for title fails.
    pub fn setup_protocols_and_hints(
        &mut self,
        connection: &Connection,
        font_char_width: u32,
        font_char_height: u32,
    ) -> Result<()> {
        if self.id == 0 {
            warn!("setup_protocols_and_hints called on an invalid window ID (0). Skipping.");
            return Ok(());
        }
        info!(
            "Setting up WM protocols and hints for window ID: {}",
            self.id
        );
        let display = connection.display();
        // SAFETY: Xlib calls involving FFI. Ensure connection and window ID are valid.
        unsafe {
            // Intern atoms required for WM interaction.
            self.wm_delete_window = xlib::XInternAtom(
                display,
                b"WM_DELETE_WINDOW\0".as_ptr() as *const c_char,
                xlib::False, // Only return existing atom, don't create.
            );
            self.protocols_atom = xlib::XInternAtom(
                display,
                b"WM_PROTOCOLS\0".as_ptr() as *const c_char,
                xlib::False,
            );

            if self.wm_delete_window != 0 && self.protocols_atom != 0 {
                xlib::XSetWMProtocols(
                    display,
                    self.id,
                    // Pass a pointer to the atom.
                    [self.wm_delete_window].as_mut_ptr(),
                    1, // Number of protocols in the array.
                );
                debug!("WM_PROTOCOLS (WM_DELETE_WINDOW) registered.");
            } else {
                // This is not fatal but may mean the window close button doesn't work as expected.
                warn!("Failed to get WM_DELETE_WINDOW or WM_PROTOCOLS atom. Window close events might not be received.");
            }

            // Set initial window title.
            let title_cstr =
                CString::new("core-term").context("Failed to create CString for initial title")?;
            xlib::XStoreName(display, self.id, title_cstr.as_ptr() as *mut c_char);

            // Set _NET_WM_NAME for UTF-8 titles, preferred by modern WMs.
            let net_wm_name_atom = xlib::XInternAtom(
                display,
                b"_NET_WM_NAME\0".as_ptr() as *const c_char,
                xlib::False,
            );
            let utf8_string_atom = xlib::XInternAtom(
                display,
                b"UTF8_STRING\0".as_ptr() as *const c_char,
                xlib::False,
            );

            if net_wm_name_atom != 0 && utf8_string_atom != 0 {
                xlib::XChangeProperty(
                    display,
                    self.id,
                    net_wm_name_atom,
                    utf8_string_atom,
                    8, // format: 8-bit for UTF8_STRING
                    xlib::PropModeReplace,
                    title_cstr.as_ptr() as *const u8,
                    title_cstr.as_bytes().len() as c_int,
                );
                debug!("Initial window title also set via _NET_WM_NAME (UTF-8).");
            } else {
                debug!("Initial window title set via XStoreName only (_NET_WM_NAME or UTF8_STRING atom not found).");
            }

            // Provide size hints to the window manager if font dimensions are valid.
            if font_char_width > 0 && font_char_height > 0 {
                let mut size_hints: xlib::XSizeHints = mem::zeroed();
                size_hints.flags = xlib::PResizeInc | xlib::PMinSize; // We specify resize increments and min size.
                size_hints.width_inc = font_char_width as c_int;
                size_hints.height_inc = font_char_height as c_int;
                size_hints.min_width = font_char_width as c_int; // Minimum window width (1 cell)
                size_hints.min_height = font_char_height as c_int; // Minimum window height (1 cell)
                xlib::XSetWMNormalHints(display, self.id, &mut size_hints);
                debug!(
                    "WM size hints set (inc: {}x{}, min: {}x{}).",
                    font_char_width, font_char_height, font_char_width, font_char_height
                );
            } else {
                warn!("Font dimensions are zero ({}, {}), skipping WM size hints based on character cells.", font_char_width, font_char_height);
            }
        }
        Ok(())
    }

    /// Maps the window to the display, making it visible.
    ///
    /// Also flushes the X command buffer to ensure the map request is processed immediately.
    ///
    /// # Arguments
    ///
    /// * `connection`: A reference to the active X11 `Connection`.
    pub fn map_and_flush(&self, connection: &Connection) {
        if self.id == 0 {
            warn!("map_and_flush called on an invalid window ID (0).");
            return;
        }
        info!("Mapping window ID: {} and flushing display.", self.id);
        // SAFETY: Xlib calls. Ensure connection and window ID are valid.
        unsafe {
            xlib::XMapWindow(connection.display(), self.id);
            xlib::XFlush(connection.display()); // Ensure the map request is sent to the server.
        }
        debug!("Window mapped and display flushed.");
    }

    /// Sets the title of the window.
    ///
    /// Updates both the standard window title property and `_NET_WM_NAME` (for UTF-8).
    ///
    /// # Arguments
    ///
    /// * `connection`: A reference to the active X11 `Connection`.
    /// * `title`: The desired window title as a string slice.
    ///
    /// # Returns
    ///
    /// * `Ok(())`: If the title was set successfully.
    /// * `Err(anyhow::Error)`: If creating CString for the title fails.
    pub fn set_title(&self, connection: &Connection, title: &str) -> Result<()> {
        if self.id == 0 {
            warn!("set_title called on an invalid window ID (0).");
            return Ok(()); // Not an error, but operation cannot proceed.
        }
        trace!(
            "Setting window title to '{}' for window ID: {}",
            title,
            self.id
        );
        let display = connection.display();
        // SAFETY: Xlib calls. Ensure connection and window ID are valid.
        unsafe {
            let title_c_str = CString::new(title).context("Failed to create CString for title")?;
            // Set standard window title property
            xlib::XStoreName(display, self.id, title_c_str.as_ptr() as *mut c_char);

            // Also set _NET_WM_NAME for modern window managers (UTF-8)
            let net_wm_name_atom = xlib::XInternAtom(
                display,
                b"_NET_WM_NAME\0".as_ptr() as *const c_char,
                xlib::False,
            );
            let utf8_string_atom = xlib::XInternAtom(
                display,
                b"UTF8_STRING\0".as_ptr() as *const c_char,
                xlib::False,
            );

            if net_wm_name_atom != 0 && utf8_string_atom != 0 {
                xlib::XChangeProperty(
                    display,
                    self.id,
                    net_wm_name_atom,
                    utf8_string_atom,
                    8, // format is 8-bit for UTF8_STRING
                    xlib::PropModeReplace,
                    title_c_str.as_ptr() as *const u8,
                    title_c_str.as_bytes().len() as c_int,
                );
            }
            xlib::XFlush(display); // Ensure title change is processed by the server.
        }
        debug!("Window title set to: {}", title);
        Ok(())
    }

    /// Rings the X11 bell for the window.
    ///
    /// # Arguments
    ///
    /// * `connection`: A reference to the active X11 `Connection`.
    pub fn bell(&self, connection: &Connection) {
        if self.id == 0 {
            warn!("bell called on an invalid window ID (0).");
            return;
        }
        trace!("Ringing bell for window ID: {}", self.id);
        // SAFETY: Xlib calls. Ensure connection is valid.
        unsafe {
            xlib::XBell(connection.display(), 0); // 0 for default volume
            xlib::XFlush(connection.display()); // Ensure bell request is sent.
        }
    }

    /// Sets the visibility of the native X11 mouse cursor when it's over this window.
    ///
    /// This does *not* control the visibility of any terminal text cursor, which is
    /// typically drawn by the application itself.
    ///
    /// # Arguments
    ///
    /// * `connection`: A reference to the active X11 `Connection`.
    /// * `visibility`: A `CursorVisibility` enum indicating whether to show or hide the cursor.
    pub fn set_native_cursor_visibility(
        &mut self,
        connection: &Connection,
        visibility: CursorVisibility,
    ) {
        if self.id == 0 {
            warn!("set_native_cursor_visibility called on an invalid window ID (0).");
            return;
        }

        let should_be_visible = match visibility {
            CursorVisibility::Shown => true,
            CursorVisibility::Hidden => false,
        };

        if self.is_native_cursor_visible == should_be_visible {
            return; // No change needed.
        }
        self.is_native_cursor_visible = should_be_visible;
        trace!(
            "Setting native X11 cursor visibility to: {:?} for window ID: {}",
            visibility,
            self.id
        );

        let display = connection.display();
        // SAFETY: Xlib calls. Ensure connection and window ID are valid.
        unsafe {
            if should_be_visible {
                // Restore the default cursor (XC_XTERM is a common one).
                let cursor = xlib::XCreateFontCursor(display, XC_XTERM);
                xlib::XDefineCursor(display, self.id, cursor);
                // If XDefineCursor is successful, X server has a copy of the cursor.
                // We should free the cursor we created if we don't store it elsewhere.
                if cursor != 0 {
                    // Check if cursor creation was successful
                    xlib::XFreeCursor(display, cursor);
                }
            } else {
                // Create an invisible cursor.
                // This involves creating a 1x1 transparent pixmap and a cursor from it.
                let mut color: xlib::XColor = mem::zeroed(); // Dummy color, pixmap is 1-bit (mask)
                let pixmap = xlib::XCreatePixmap(display, self.id, 1, 1, 1); // 1x1, 1-bit depth (mask)
                if pixmap == 0 {
                    warn!("Failed to create 1x1 pixmap for invisible cursor.");
                    return;
                }
                let cursor = xlib::XCreatePixmapCursor(
                    display, pixmap, pixmap, &mut color, &mut color, 0, 0,
                );
                if cursor != 0 {
                    // Check if cursor creation was successful
                    xlib::XDefineCursor(display, self.id, cursor);
                    xlib::XFreeCursor(display, cursor); // Free the cursor resource
                } else {
                    warn!("Failed to create invisible pixmap cursor.");
                }
                xlib::XFreePixmap(display, pixmap); // Free the pixmap resource
            }
            xlib::XFlush(display); // Ensure cursor change is processed.
        }
        debug!("Native cursor visibility set to {:?}.", visibility);
    }

    /// Destroys the X11 window and releases associated server-side resources.
    ///
    /// This method should be called when the window is no longer needed, typically
    /// as part of the `XDriver`'s cleanup process. It is idempotent.
    ///
    /// # Arguments
    ///
    /// * `connection`: A reference to the active X11 `Connection`.
    pub fn cleanup(&mut self, connection: &Connection) {
        if self.id != 0 && !connection.display().is_null() {
            info!("Destroying X11 window (ID: {}).", self.id);
            // SAFETY: Xlib call. Ensure connection and window ID are valid.
            unsafe {
                xlib::XDestroyWindow(connection.display(), self.id);
                // It's good practice to flush after operations that change window state or destroy windows.
                xlib::XFlush(connection.display());
            }
            self.id = 0; // Mark as destroyed to prevent further operations.
            debug!("Window destroyed.");
        } else {
            info!("Window already destroyed or connection is invalid; cleanup skipped.");
        }
    }

    // --- Getter methods ---

    /// Returns the X11 ID of the window.
    #[inline]
    pub fn id(&self) -> xlib::Window {
        self.id
    }

    /// Returns the interned X11 atom for `WM_DELETE_WINDOW`.
    ///
    /// This atom is used in `ClientMessage` events to detect when the
    /// window manager requests the window to close.
    #[inline]
    pub fn wm_delete_window_atom(&self) -> xlib::Atom {
        self.wm_delete_window
    }

    /// Returns the interned X11 atom for `WM_PROTOCOLS`.
    ///
    /// This atom is used for communication about window manager protocols.
    #[inline]
    pub fn protocols_atom(&self) -> xlib::Atom {
        self.protocols_atom
    }

    /// Returns the current dimensions (width, height) of the window in pixels.
    ///
    /// These dimensions are updated internally when a resize event (`ConfigureNotify`) is processed.
    #[inline]
    pub fn current_dimensions_pixels(&self) -> (u16, u16) {
        (self.current_pixel_width, self.current_pixel_height)
    }

    /// Returns the pixel value used for the window's background at creation time.
    #[inline]
    pub fn initial_bg_pixel(&self) -> xlib::Atom {
        self.initial_bg_pixel
    }

    // --- Setter methods ---

    /// Updates the internally cached dimensions of the window.
    ///
    /// This method is typically called when a `ConfigureNotify` event is received,
    /// indicating that the window has been resized by the user or window manager.
    ///
    /// # Arguments
    ///
    /// * `width_px`: The new width of the window in pixels.
    /// * `height_px`: The new height of the window in pixels.
    pub fn update_dimensions(&mut self, width_px: u16, height_px: u16) {
        if self.current_pixel_width != width_px || self.current_pixel_height != height_px {
            debug!(
                "Updating cached window dimensions from {}x{} to {}x{}",
                self.current_pixel_width, self.current_pixel_height, width_px, height_px
            );
            self.current_pixel_width = width_px;
            self.current_pixel_height = height_px;
        }
    }
}

/// Handles resource cleanup for the `Window`.
///
/// This `Drop` implementation primarily serves as a safeguard. It logs an error
/// if the window ID is non-zero, indicating that `cleanup()` was not explicitly
/// called by the `XDriver`. **It cannot safely call Xlib functions** because
/// it doesn't have access to the `Connection`'s display pointer.
impl Drop for Window {
    fn drop(&mut self) {
        if self.id != 0 {
            // This indicates that `cleanup(&mut self, &Connection)` was not called before drop.
            // X11 resources associated with `self.id` on the server might not have been released.
            error!(
                "Window (ID: {}) dropped without explicit cleanup. Associated X server resources may be leaked if connection is already closed.", self.id
            );
        } else {
            info!("Window dropped (already cleaned up or ID was 0).");
        }
    }
}
