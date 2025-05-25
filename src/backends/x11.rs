// src/backends/x11.rs

#![allow(non_snake_case)] // Allow non-snake case for X11 types

// Import logging macros
use log::{debug, error, info, trace, warn};

// Crate-level imports
use crate::backends::{BackendEvent, CellCoords, CellRect, Driver, TextRunStyle};
use crate::color::{Color, NamedColor};
use crate::glyph::AttrFlags;
use crate::keys::{KeySymbol, Modifiers}; // Added for new key representation

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::ffi::CString;
use std::mem;
use std::os::unix::io::RawFd;
use std::ptr;

// Libc imports
use libc::{c_char, c_int, c_uint};

// X11 library imports
use x11::keysym; // Added for X11 keysym constants
use x11::xft;
use x11::xlib;
use x11::xrender::{XGlyphInfo, XRenderColor};

// --- Constants ---
const DEFAULT_FONT_NAME: &str = "Inconsolata:size=10"; // Example font
const MIN_FONT_WIDTH: u32 = 1; // Minimum reasonable font width
const MIN_FONT_HEIGHT: u32 = 1; // Minimum reasonable font height
const KEY_TEXT_BUFFER_SIZE: usize = 32; // Buffer for XLookupString text

const ANSI_COLOR_COUNT: usize = 16; // For the 16 named ANSI colors
const TOTAL_PREALLOC_COLORS: usize = ANSI_COLOR_COUNT; // Currently, only ANSI colors are preallocated

const DEFAULT_WINDOW_WIDTH_CHARS: usize = 80;
const DEFAULT_WINDOW_HEIGHT_CHARS: usize = 24;

const XRENDER_ALPHA_OPAQUE: u16 = 0xffff; // Fully opaque for XRenderColor

/// Value for the Xterm cursor shape, from <X11/cursorfont.h>.
const XC_XTERM: c_uint = 152;

/// Helper struct for RGB components as u16 (for XRenderColor).
struct Rgb16Components {
    r: u16,
    g: u16,
    b: u16,
}

/// `Driver` implementation for the X11 windowing system.
///
/// Manages an X11 window, handles X events, and uses Xft for font rendering.
pub struct XDriver {
    display: *mut xlib::Display,
    screen: c_int,
    window: xlib::Window,
    colormap: xlib::Colormap,
    visual: *mut xlib::Visual,
    xft_font: *mut xft::XftFont,
    xft_draw: *mut xft::XftDraw,
    /// Pre-allocated XftColors for the 16 standard ANSI colors.
    xft_ansi_colors: Vec<xft::XftColor>,
    /// Cache for dynamically allocated XftColors from RGB values.
    xft_color_cache_rgb: HashMap<(u8, u8, u8), xft::XftColor>,
    font_width: u32,  // Width of a single character cell in pixels.
    font_height: u32, // Height of a single character cell in pixels.
    font_ascent: u32, // Font ascent in pixels.
    current_pixel_width: u16,
    current_pixel_height: u16,
    wm_delete_window: xlib::Atom,
    protocols_atom: xlib::Atom,
    /// Graphics Context for simple operations like clearing the background.
    clear_gc: xlib::GC,
    /// Tracks if the window currently has focus (for cursor style, etc.).
    has_focus: bool,
    /// Tracks if the X11 native cursor (mouse pointer) should be visible over the window.
    /// Note: This is distinct from the terminal's text cursor.
    is_native_cursor_visible: bool,
}

impl Driver for XDriver {
    /// Creates and initializes a new `XDriver`.
    ///
    /// This involves:
    /// - Connecting to the X server.
    /// - Loading the specified font using Xft.
    /// - Calculating initial font metrics (cell width, height).
    /// - Pre-allocating XftColor structures for standard ANSI colors.
    /// - Creating the X11 window with initial dimensions.
    /// - Creating a Graphics Context (GC) for basic drawing operations.
    /// - Setting up WM protocols (like WM_DELETE_WINDOW) and window hints.
    /// - Mapping the window to make it visible.
    ///
    /// # Returns
    /// * `Result<Self>`: The initialized `XDriver` or an error if any setup step fails.
    fn new() -> Result<Self> {
        info!("Creating new XDriver");
        let display = unsafe { xlib::XOpenDisplay(ptr::null()) };
        if display.is_null() {
            return Err(anyhow::anyhow!(
                "Failed to open X display: Check DISPLAY environment variable or X server status."
            ));
        }
        debug!("X display opened successfully.");

        let screen = unsafe { xlib::XDefaultScreen(display) };
        let colormap = unsafe { xlib::XDefaultColormap(display, screen) };
        let visual = unsafe { xlib::XDefaultVisual(display, screen) };

        // Initialize with placeholder or default values before fallible operations.
        let mut driver = XDriver {
            display,
            screen,
            window: 0, // Will be set by create_window
            colormap,
            visual,
            xft_font: ptr::null_mut(), // Will be set by load_font
            xft_draw: ptr::null_mut(), // Will be set after window creation
            xft_ansi_colors: Vec::with_capacity(TOTAL_PREALLOC_COLORS),
            xft_color_cache_rgb: HashMap::new(),
            font_width: 0,  // Will be set by load_font
            font_height: 0, // Will be set by load_font
            font_ascent: 0, // Will be set by load_font
            current_pixel_width: 0,
            current_pixel_height: 0,
            wm_delete_window: 0,            // Will be set by setup_wm_protocols
            protocols_atom: 0,              // Will be set by setup_wm_protocols
            clear_gc: ptr::null_mut(),      // Will be set by create_gc
            has_focus: true, // Assume focused initially until an event says otherwise
            is_native_cursor_visible: true, // Native X cursor visible by default
        };

        // Perform fallible setup steps in a closure to handle errors and cleanup.
        if let Err(e) = (|| {
            driver.load_font().context("Failed to load font")?;
            // Calculate initial pixel dimensions based on default char cells and loaded font.
            driver.current_pixel_width =
                (DEFAULT_WINDOW_WIDTH_CHARS * driver.font_width as usize) as u16;
            driver.current_pixel_height =
                (DEFAULT_WINDOW_HEIGHT_CHARS * driver.font_height as usize) as u16;
            driver
                .init_xft_ansi_colors()
                .context("Failed to initialize Xft ANSI colors")?;
            // Determine initial background pixel value for window creation.
            let initial_bg_pixel = if !driver.xft_ansi_colors.is_empty() {
                // Use the pixel value of the pre-allocated Black color.
                driver.xft_ansi_colors[NamedColor::Black as usize].pixel
            } else {
                // Fallback if ANSI colors somehow weren't initialized (should not happen).
                warn!(
                    "ANSI colors not yet initialized for initial window background, using 0 (black)."
                );
                0 // XBlackPixel(display, screen) could also be used but requires `unsafe`
            };
            driver
                .create_window(
                    driver.current_pixel_width,
                    driver.current_pixel_height,
                    initial_bg_pixel,
                )
                .context("Failed to create window")?;
            driver
                .create_gc()
                .context("Failed to create graphics context for clearing")?;
            // Create XftDraw object associated with the window's drawable.
            driver.xft_draw = unsafe {
                xft::XftDrawCreate(
                    driver.display,
                    driver.window, // Draw onto the main window
                    driver.visual,
                    driver.colormap,
                )
            };
            if driver.xft_draw.is_null() {
                return Err(anyhow::anyhow!("Failed to create XftDraw object"));
            }
            debug!("XftDraw object created.");
            driver.setup_wm_protocols_and_hints();
            // Map the window to make it visible and flush X server requests.
            unsafe {
                xlib::XMapWindow(driver.display, driver.window);
                xlib::XFlush(driver.display);
            }
            debug!("Window mapped and flushed.");
            Ok(())
        })() {
            // If any setup step failed, log the error and ensure cleanup is attempted.
            error!("Error during XDriver setup: {:?}", e);
            // Attempt cleanup, logging any further errors during cleanup.
            if let Err(cleanup_err) = driver.cleanup() {
                error!(
                    "Error during cleanup after setup failure: {:?}",
                    cleanup_err
                );
            }
            return Err(e); // Return the original setup error.
        }
        info!("XDriver initialization complete.");
        Ok(driver)
    }

    fn get_event_fd(&self) -> Option<RawFd> {
        // The X11 connection file descriptor can be used for event polling.
        Some(unsafe { xlib::XConnectionNumber(self.display) })
    }

    /// Processes pending X11 events and translates them into `BackendEvent`s.
    fn process_events(&mut self) -> Result<Vec<BackendEvent>> {
        let mut backend_events = Vec::new();
        // Loop while there are events pending on the X display connection.
        while unsafe { xlib::XPending(self.display) } > 0 {
            let mut xevent: xlib::XEvent = unsafe { mem::zeroed() };
            unsafe { xlib::XNextEvent(self.display, &mut xevent) };

            // Extract the event type.
            let event_type = unsafe { xevent.type_ };

            match event_type {
                xlib::Expose => {
                    // An Expose event means a part of the window needs to be redrawn.
                    // The renderer handles the actual redrawing based on dirty flags.
                    // We log it here for debugging.
                    let xexpose = unsafe { xevent.expose };
                    if xexpose.count == 0 {
                        // Only process the last Expose event in a series.
                        debug!(
                            "XEvent: Expose (x:{}, y:{}, w:{}, h:{}) - redraw will be handled by renderer",
                            xexpose.x, xexpose.y, xexpose.width, xexpose.height
                        );
                        // Note: The orchestrator will call renderer.draw() which handles dirty lines.
                        // No specific BackendEvent is usually emitted for Expose itself, as it's a redraw trigger.
                    }
                }
                xlib::ConfigureNotify => {
                    // Window was resized, moved, or restacked. We only care about resize.
                    let xconfigure = unsafe { xevent.configure };
                    if self.current_pixel_width != xconfigure.width as u16
                        || self.current_pixel_height != xconfigure.height as u16
                    {
                        debug!(
                            "XEvent: ConfigureNotify (resize from {}x{} to {}x{})",
                            self.current_pixel_width,
                            self.current_pixel_height,
                            xconfigure.width,
                            xconfigure.height
                        );
                        self.current_pixel_width = xconfigure.width as u16;
                        self.current_pixel_height = xconfigure.height as u16;
                        // Create an XftDraw for the new size if it's a pixmap based approach
                        // If drawing directly to window, XftDraw might not need recreation,
                        // but internal buffers for pixmap might. Here, assuming direct or Xft handles it.
                        // If using a pixmap (self.drawable is a Pixmap), it needs recreation:
                        // unsafe {
                        //     if !self.xft_draw.is_null() { xft::XftDrawDestroy(self.xft_draw); }
                        //     // Recreate self.drawable (Pixmap)
                        //     // self.xft_draw = xft::XftDrawCreate(self.display, self.drawable, ...);
                        // }
                        backend_events.push(BackendEvent::Resize {
                            width_px: self.current_pixel_width,
                            height_px: self.current_pixel_height,
                        });
                    }
                }
                xlib::KeyPress => {
                    // A key was pressed. Translate to keysym and text.
                    let mut x_keysym: xlib::KeySym = 0; // Renamed from keysym
                    let mut key_text_buffer = [0u8; KEY_TEXT_BUFFER_SIZE]; // Buffer for XLookupString
                    let count = unsafe {
                        xlib::XLookupString(
                            &mut xevent.key, // Pass as mutable pointer
                            key_text_buffer.as_mut_ptr() as *mut c_char,
                            key_text_buffer.len() as c_int, // Use KEY_TEXT_BUFFER_SIZE via .len()
                            &mut x_keysym,   // Pass as mutable pointer, renamed
                            ptr::null_mut(), // No XComposeStatus needed
                        )
                    };
                    let text = if count > 0 {
                        // Convert the bytes from XLookupString (usually Latin-1 or UTF-8 based on locale)
                        // to a Rust String. String::from_utf8_lossy is a safe way.
                        String::from_utf8_lossy(&key_text_buffer[0..count as usize]).to_string()
                    } else {
                        String::new() // No text generated (e.g., modifier key)
                    };

                    // Determine Modifiers
                    let xkey_event = unsafe { &xevent.key };
                    let mut modifiers = Modifiers::empty();
                    if (xkey_event.state & xlib::ShiftMask) != 0 {
                        modifiers.insert(Modifiers::SHIFT);
                    }
                    if (xkey_event.state & xlib::ControlMask) != 0 {
                        modifiers.insert(Modifiers::CONTROL);
                    }
                    if (xkey_event.state & xlib::Mod1Mask) != 0 {
                        // Mod1Mask is usually Alt
                        modifiers.insert(Modifiers::ALT);
                    }
                    // Mod4Mask is usually Super/Windows/Command key
                    if (xkey_event.state & xlib::Mod4Mask) != 0 {
                        modifiers.insert(Modifiers::SUPER);
                    }
                    // Note: X11 doesn't have distinct masks for CapsLock/NumLock active state in xkey_event.state
                    // in the same way as Shift/Control/Alt/Super.
                    // These are typically handled by looking at the keysym itself or by separate Xkb state.
                    // For now, we only derive modifiers from the main modifier masks.

                    let symbol = xkeysym_to_keysymbol(x_keysym, &text);

                    debug!(
                        "XEvent: KeyPress (symbol: {:?}, modifiers: {:?}, text: '{}')",
                        symbol, modifiers, text
                    );
                    backend_events.push(BackendEvent::Key {
                        symbol,
                        modifiers,
                        text,
                    });
                }
                xlib::ClientMessage => {
                    // A client message, often from the window manager.
                    let xclient = unsafe { xevent.client_message };
                    if xclient.message_type == self.protocols_atom
                        && xclient.data.as_longs()[0] as xlib::Atom == self.wm_delete_window
                    {
                        info!("XEvent: WM_DELETE_WINDOW received from window manager.");
                        backend_events.push(BackendEvent::CloseRequested);
                    } else {
                        trace!(
                            "XEvent: Ignored ClientMessage (type: {})",
                            xclient.message_type
                        );
                    }
                }
                xlib::FocusIn => {
                    debug!("XEvent: FocusIn received.");
                    self.has_focus = true;
                    backend_events.push(BackendEvent::FocusGained);
                }
                xlib::FocusOut => {
                    debug!("XEvent: FocusOut received.");
                    self.has_focus = false;
                    backend_events.push(BackendEvent::FocusLost);
                }
                // Add other event types as needed (e.g., ButtonPress, ButtonRelease, MotionNotify for mouse)
                _ => {
                    // Log unhandled event types for debugging.
                    trace!("XEvent: Ignored (type: {})", event_type);
                }
            }
        }
        Ok(backend_events)
    }

    fn get_font_dimensions(&self) -> (usize, usize) {
        (self.font_width as usize, self.font_height as usize)
    }

    fn get_display_dimensions_pixels(&self) -> (u16, u16) {
        (self.current_pixel_width, self.current_pixel_height)
    }

    fn clear_all(&mut self, bg: Color) -> Result<()> {
        // Renderer ensures `bg` is not Color::Default.
        if matches!(bg, Color::Default) {
            error!("XDriver::clear_all received Color::Default. This is a bug in the Renderer.");
            // Use a fallback color (e.g., black) for safety, but this indicates a logic error upstream.
            let xft_bg_color = self
                .cached_rgb_to_xft_color(0, 0, 0)
                .context("Failed to resolve fallback black color for clear_all")?;
            unsafe {
                // Using XftDrawRect for consistency, even for a full clear.
                xft::XftDrawRect(
                    self.xft_draw,
                    &xft_bg_color,
                    0,
                    0,
                    self.current_pixel_width as u32,
                    self.current_pixel_height as u32,
                );
            }
            return Err(anyhow::anyhow!(
                "XDriver::clear_all received Color::Default. Renderer should resolve defaults."
            ));
        }

        // Resolve the concrete Color to an XftColor.
        let xft_bg_color = self
            .resolve_concrete_xft_color(bg)
            .context("Failed to resolve background color for clear_all")?;

        // Use the XftDraw object to fill the entire drawable area (window).
        unsafe {
            xft::XftDrawRect(
                self.xft_draw,
                &xft_bg_color,
                0,
                0,
                self.current_pixel_width as u32,
                self.current_pixel_height as u32,
            );
        }
        trace!("Window cleared with color pixel: {}", xft_bg_color.pixel);
        Ok(())
    }

    fn draw_text_run(&mut self, coords: CellCoords, text: &str, style: TextRunStyle) -> Result<()> {
        if text.is_empty() {
            return Ok(());
        }

        // Contract: Renderer ensures colors are concrete.
        if matches!(style.fg, Color::Default) || matches!(style.bg, Color::Default) {
            error!(
                "XDriver::draw_text_run received Color::Default in style. This is a bug in the Renderer."
            );
            return Err(anyhow::anyhow!(
                "XDriver::draw_text_run received Color::Default. Renderer should resolve defaults."
            ));
        }

        // Calculate pixel coordinates for the text run.
        let x_pixel = (coords.x * self.font_width as usize) as c_int;
        let y_pixel = (coords.y * self.font_height as usize) as c_int;
        // Calculate width of the background rectangle for the text run.
        let run_pixel_width = text.chars().count() * self.font_width as usize;

        // Resolve foreground and background colors to XftColor.
        let xft_fg = self
            .resolve_concrete_xft_color(style.fg)
            .context("Failed to resolve foreground color for text run")?;
        let xft_bg = self
            .resolve_concrete_xft_color(style.bg)
            .context("Failed to resolve background color for text run")?;

        // Draw the background rectangle for the text run.
        unsafe {
            xft::XftDrawRect(
                self.xft_draw,
                &xft_bg, // Use the resolved background color
                x_pixel,
                y_pixel,
                run_pixel_width as u32, // Width of the text run
                self.font_height,       // Height of a single line
            );
        }

        // Convert the Rust string to a C-compatible string for Xft.
        let c_text = CString::new(text).context("Failed to convert text to CString for Xft")?;
        // Calculate the baseline Y coordinate for XftDrawStringUtf8.
        let baseline_y_pixel = y_pixel + self.font_ascent as c_int;

        // Draw the text string.
        unsafe {
            xft::XftDrawStringUtf8(
                self.xft_draw,
                &xft_fg,                          // Use the resolved foreground color
                self.xft_font,                    // The loaded Xft font
                x_pixel,                          // X position in pixels
                baseline_y_pixel,                 // Y position of the baseline in pixels
                c_text.as_ptr() as *const u8,     // Text as C string
                c_text.as_bytes().len() as c_int, // Length of the text in bytes
            );
        }

        // Handle underline and strikethrough attributes if present.
        if style.flags.contains(AttrFlags::UNDERLINE) {
            // Simple underline: 1 pixel high, a couple of pixels below the baseline.
            let underline_y = y_pixel + self.font_height as c_int - 2; // Adjust position as needed
            unsafe {
                xft::XftDrawRect(
                    self.xft_draw,
                    &xft_fg, // Use foreground color for underline
                    x_pixel,
                    underline_y,
                    run_pixel_width as u32,
                    1, // Thickness of underline
                );
            }
        }
        if style.flags.contains(AttrFlags::STRIKETHROUGH) {
            // Simple strikethrough: 1 pixel high, roughly in the middle of the text.
            let strikethrough_y = y_pixel + (self.font_ascent / 2) as c_int; // Adjust position
            unsafe {
                xft::XftDrawRect(
                    self.xft_draw,
                    &xft_fg, // Use foreground color for strikethrough
                    x_pixel,
                    strikethrough_y,
                    run_pixel_width as u32,
                    1, // Thickness of strikethrough
                );
            }
        }
        Ok(())
    }

    fn fill_rect(&mut self, rect: CellRect, color: Color) -> Result<()> {
        if rect.width == 0 || rect.height == 0 {
            return Ok(()); // Nothing to fill.
        }
        // Contract: Renderer ensures color is concrete.
        if matches!(color, Color::Default) {
            error!("XDriver::fill_rect received Color::Default. This is a bug in the Renderer.");
            return Err(anyhow::anyhow!(
                "XDriver::fill_rect received Color::Default. Renderer should resolve defaults."
            ));
        }

        // Calculate pixel dimensions for the rectangle.
        let x_pixel = (rect.x * self.font_width as usize) as c_int;
        let y_pixel = (rect.y * self.font_height as usize) as c_int;
        let rect_pixel_width = (rect.width * self.font_width as usize) as u32;
        let rect_pixel_height = (rect.height * self.font_height as usize) as u32;

        // Resolve the concrete Color to an XftColor.
        let xft_fill_color = self
            .resolve_concrete_xft_color(color)
            .context("Failed to resolve color for fill_rect")?;

        // Use XftDrawRect to fill the area.
        unsafe {
            xft::XftDrawRect(
                self.xft_draw,
                &xft_fill_color,
                x_pixel,
                y_pixel,
                rect_pixel_width,
                rect_pixel_height,
            );
        }
        Ok(())
    }

    fn present(&mut self) -> Result<()> {
        // For X11, drawing commands are typically sent to the server as they are made.
        // XFlush ensures that all buffered commands are sent to the X server for processing.
        unsafe {
            xlib::XFlush(self.display);
        }
        trace!("XFlush called to present frame.");
        Ok(())
    }

    /// Sets the window title using X11 functions.
    fn set_title(&mut self, title: &str) {
        trace!("XDriver: Setting window title to '{}'", title);
        if self.window == 0 || self.display.is_null() {
            warn!("XDriver::set_title called on uninitialized or closed window/display.");
            return;
        }
        unsafe {
            // Standard window title property
            if let Ok(title_c_str) = CString::new(title) {
                xlib::XStoreName(self.display, self.window, title_c_str.as_ptr() as *mut _);

                // Also set _NET_WM_NAME for modern window managers (UTF-8)
                let net_wm_name_atom = xlib::XInternAtom(
                    self.display,
                    b"_NET_WM_NAME\0".as_ptr() as *const i8,
                    xlib::False,
                );
                let utf8_string_atom = xlib::XInternAtom(
                    self.display,
                    b"UTF8_STRING\0".as_ptr() as *const i8,
                    xlib::False,
                );

                if net_wm_name_atom != 0 && utf8_string_atom != 0 {
                    xlib::XChangeProperty(
                        self.display,
                        self.window,
                        net_wm_name_atom,
                        utf8_string_atom,
                        8, // format is 8-bit for UTF8_STRING
                        xlib::PropModeReplace,
                        title_c_str.as_ptr() as *const u8,
                        title_c_str.as_bytes().len() as c_int,
                    );
                }
            } else {
                error!("Failed to create CString for title: {}", title);
            }
            // Ensure changes are sent to the X server.
            xlib::XFlush(self.display);
        }
    }

    /// Rings the X11 bell.
    fn bell(&mut self) {
        trace!("XDriver: Ringing bell.");
        if self.display.is_null() {
            warn!("XDriver::bell called on closed display.");
            return;
        }
        unsafe {
            // XkbBell is generally preferred if available and XKB extension is used.
            // XBell is simpler if XKB is not explicitly managed.
            // Using XBell for simplicity here. Volume is a percentage from -100 to 100.
            xlib::XBell(self.display, 0); // 0 for default volume
            xlib::XFlush(self.display);
        }
    }

    /// Sets the visibility of the *native X11 mouse pointer* when it's over the terminal window.
    /// This does NOT control the visibility of the terminal's text cursor, which is
    /// drawn by the renderer.
    fn set_cursor_visibility(&mut self, visible: bool) {
        trace!(
            "XDriver: Setting native X11 cursor (mouse pointer) visibility to: {}",
            visible
        );
        if self.window == 0 || self.display.is_null() {
            warn!(
                "XDriver::set_cursor_visibility called on uninitialized or closed window/display."
            );
            return;
        }
        self.is_native_cursor_visible = visible;
        unsafe {
            if visible {
                // Restore the default cursor. XC_XTERM is a common default.
                let cursor = xlib::XCreateFontCursor(self.display, XC_XTERM); // Use defined constant
                xlib::XDefineCursor(self.display, self.window, cursor);
                // If XDefineCursor is successful, X server has a copy.
                // We should free the cursor we created if we don't store it.
                if cursor != 0 {
                    // Check if cursor creation was successful
                    xlib::XFreeCursor(self.display, cursor);
                }
            } else {
                // Create an invisible cursor.
                let mut color = xlib::XColor {
                    // `color` needs to be mutable for XCreatePixmapCursor
                    pixel: 0,
                    red: 0,
                    green: 0,
                    blue: 0,
                    flags: 0,
                    pad: 0,
                };
                let pixmap = xlib::XCreatePixmap(self.display, self.window, 1, 1, 1); // 1x1, 1-bit depth
                if pixmap == 0 {
                    warn!("Failed to create 1x1 pixmap for invisible cursor.");
                    return;
                }
                let cursor = xlib::XCreatePixmapCursor(
                    self.display,
                    pixmap,
                    pixmap,
                    &mut color,
                    &mut color,
                    0,
                    0,
                );
                if cursor != 0 {
                    // Check if cursor creation was successful
                    xlib::XDefineCursor(self.display, self.window, cursor);
                    xlib::XFreeCursor(self.display, cursor);
                } else {
                    warn!("Failed to create invisible pixmap cursor.");
                }
                xlib::XFreePixmap(self.display, pixmap);
            }
            xlib::XFlush(self.display);
        }
    }

    /// Updates the driver's internal focus state.
    /// This is called by the orchestrator based on `FocusIn`/`FocusOut` events.
    /// The driver can use this state, for example, to alter the appearance of the
    /// (application-drawn) text cursor if desired (e.g., block vs. hollow).
    fn set_focus(&mut self, focused: bool) {
        trace!("XDriver: Setting focus state to: {}", focused);
        self.has_focus = focused;
        // The actual visual change for the text cursor based on focus
        // would be handled by the Renderer when it gets the cursor state.
        // This driver method just records the state.
    }

    /// Cleans up X11 resources (display connection, window, GC, Xft objects).
    fn cleanup(&mut self) -> Result<()> {
        info!("Cleaning up XDriver resources...");
        unsafe {
            if !self.xft_draw.is_null() {
                trace!("Destroying XftDraw object.");
                xft::XftDrawDestroy(self.xft_draw);
                self.xft_draw = ptr::null_mut();
            }
            if !self.xft_font.is_null() {
                trace!("Closing XftFont.");
                xft::XftFontClose(self.display, self.xft_font);
                self.xft_font = ptr::null_mut();
            }

            trace!(
                "Freeing {} preallocated ANSI XftColors.",
                self.xft_ansi_colors.len()
            );
            for color_ptr in self.xft_ansi_colors.iter_mut() {
                // Check if pixel is non-zero before freeing to avoid issues with unallocated colors
                // (though init_xft_ansi_colors should ensure they are allocated).
                if color_ptr.pixel != 0
                    || color_ptr.color.red != 0
                    || color_ptr.color.green != 0
                    || color_ptr.color.blue != 0
                {
                    xft::XftColorFree(self.display, self.visual, self.colormap, color_ptr);
                }
            }
            self.xft_ansi_colors.clear();

            trace!(
                "Freeing {} cached RGB XftColors.",
                self.xft_color_cache_rgb.len()
            );
            for (_, mut cached_color) in self.xft_color_cache_rgb.drain() {
                xft::XftColorFree(self.display, self.visual, self.colormap, &mut cached_color);
            }

            if !self.clear_gc.is_null() {
                trace!("Freeing clear GC.");
                xlib::XFreeGC(self.display, self.clear_gc);
                self.clear_gc = ptr::null_mut();
            }
            if self.window != 0 {
                trace!("Destroying X window (ID: {}).", self.window);
                xlib::XDestroyWindow(self.display, self.window);
                self.window = 0;
            }
            if !self.display.is_null() {
                trace!("Closing X display connection.");
                xlib::XCloseDisplay(self.display);
                self.display = ptr::null_mut();
            }
        }
        info!("XDriver cleanup complete.");
        Ok(())
    }
}

// --- XDriver Private Helper Methods ---
impl XDriver {
    /// Loads the primary font using Xft and calculates basic font metrics.
    fn load_font(&mut self) -> Result<()> {
        debug!("Loading font: {}", DEFAULT_FONT_NAME);
        let font_name_cstr =
            CString::new(DEFAULT_FONT_NAME).context("Failed to create CString for font name")?;

        // Open the font using Xft.
        self.xft_font =
            unsafe { xft::XftFontOpenName(self.display, self.screen, font_name_cstr.as_ptr()) };
        if self.xft_font.is_null() {
            return Err(anyhow::anyhow!(
                "XftFontOpenName failed for font: '{}'. Ensure font is installed and accessible.",
                DEFAULT_FONT_NAME
            ));
        }
        debug!("Font '{}' loaded successfully.", DEFAULT_FONT_NAME);

        // Get font metrics from the loaded XftFont.
        // The ascent and descent are part of the XftFont struct.
        let font_info_ptr = self.xft_font; // Borrow for struct field access
        self.font_height = unsafe { ((*font_info_ptr).ascent + (*font_info_ptr).descent) as u32 };
        self.font_ascent = unsafe { (*font_info_ptr).ascent as u32 };

        // To get the average character width, we can measure a sample string.
        // 'M' is often used, or a string of common characters.
        let mut extents: XGlyphInfo = unsafe { mem::zeroed() };
        let sample_char_cstr = CString::new("M").expect("CString::new for 'M' failed.");
        unsafe {
            xft::XftTextExtentsUtf8(
                self.display,
                self.xft_font,
                sample_char_cstr.as_ptr() as *const u8, // Correct type for XftTextExtentsUtf8
                sample_char_cstr.as_bytes().len() as c_int,
                &mut extents, // Pass as mutable pointer
            );
        }
        self.font_width = extents.xOff as u32; // xOff is the advance width

        // Basic validation of font metrics.
        if self.font_width < MIN_FONT_WIDTH || self.font_height < MIN_FONT_HEIGHT {
            return Err(anyhow::anyhow!(
                "Font dimensions (W:{}, H:{}) below minimum (W:{}, H:{}).",
                self.font_width,
                self.font_height,
                MIN_FONT_WIDTH,
                MIN_FONT_HEIGHT
            ));
        }
        info!(
            "Font metrics: Width={}, Height={}, Ascent={}",
            self.font_width, self.font_height, self.font_ascent
        );
        Ok(())
    }

    /// Initializes the `xft_ansi_colors` vector by allocating XftColor structures
    /// for the 16 standard ANSI colors.
    fn init_xft_ansi_colors(&mut self) -> Result<()> {
        debug!(
            "Initializing {} preallocated ANSI Xft colors.",
            ANSI_COLOR_COUNT
        );
        self.xft_ansi_colors
            .resize_with(ANSI_COLOR_COUNT, || unsafe { mem::zeroed() }); // Initialize with zeroed XftColor

        for i in 0..ANSI_COLOR_COUNT {
            let named_color_enum = NamedColor::from_index(i as u8);
            // Get Color::Rgb from NamedColor
            let rgb_color = named_color_enum.to_rgb_color(); // This is now crate::color::Color
            let (r_u8, g_u8, b_u8) = match rgb_color {
                Color::Rgb(r, g, b) => (r, g, b),
                _ => {
                    // This should not happen as to_rgb_color always returns Color::Rgb
                    warn!(
                        "NamedColor::to_rgb_color did not return Color::Rgb for {:?}. Defaulting to black.",
                        named_color_enum
                    );
                    (0, 0, 0)
                }
            };

            // Convert 8-bit RGB to 16-bit for XRenderColor
            let color_components = Rgb16Components {
                r: ((r_u8 as u16) << 8) | (r_u8 as u16),
                g: ((g_u8 as u16) << 8) | (g_u8 as u16),
                b: ((b_u8 as u16) << 8) | (b_u8 as u16),
            };
            self.alloc_specific_xft_color_into_slice(
                i,
                color_components,
                &format!("ANSI {} ({:?})", i, named_color_enum),
            )?;
        }

        debug!("Preallocated ANSI Xft colors initialized.");
        Ok(())
    }

    /// Allocates a specific XftColor and stores it in the `xft_ansi_colors` slice.
    fn alloc_specific_xft_color_into_slice(
        &mut self,
        index: usize,
        color_comps: Rgb16Components,
        name_for_log: &str,
    ) -> Result<()> {
        let render_color = XRenderColor {
            red: color_comps.r,
            green: color_comps.g,
            blue: color_comps.b,
            alpha: XRENDER_ALPHA_OPAQUE, // Fully opaque
        };
        // SAFETY: FFI call to Xft.
        if unsafe {
            xft::XftColorAllocValue(
                self.display,
                self.visual,
                self.colormap,
                &render_color,                    // Pass as const pointer
                &mut self.xft_ansi_colors[index], // Pass as mutable pointer
            )
        } == 0
        {
            // XftColorAllocValue returns 0 on failure.
            return Err(anyhow::anyhow!(
                "XftColorAllocValue failed for {}",
                name_for_log
            ));
        }
        trace!(
            "Allocated XftColor for {} (idx {}, pixel: {})",
            name_for_log,
            index,
            &self.xft_ansi_colors[index].pixel
        );
        Ok(())
    }

    /// Creates the main X11 window.
    fn create_window(
        &mut self,
        pixel_width: u16,
        pixel_height: u16,
        bg_pixel_val: xlib::Atom, // Use Atom for pixel values as XID
    ) -> Result<()> {
        unsafe {
            let root_window = xlib::XRootWindow(self.display, self.screen);
            let border_width = 0; // No border managed by this window itself

            let mut attributes: xlib::XSetWindowAttributes = mem::zeroed();
            attributes.colormap = self.colormap;
            attributes.background_pixel = bg_pixel_val; // Background color
            attributes.border_pixel = bg_pixel_val; // Border color (though width is 0)
            attributes.event_mask = xlib::ExposureMask // Redraw events
                | xlib::KeyPressMask       // Keyboard input
                // | xlib::KeyReleaseMask    // If needed
                | xlib::StructureNotifyMask  // Resize/move events (ConfigureNotify)
                | xlib::FocusChangeMask; // FocusIn/FocusOut events
                                         // Add ButtonPressMask, ButtonReleaseMask, PointerMotionMask for mouse

            self.window = xlib::XCreateWindow(
                self.display,
                root_window,
                0,                                              // x position
                0,                                              // y position
                pixel_width as c_uint,                          // width
                pixel_height as c_uint,                         // height
                border_width,                                   // border width
                xlib::XDefaultDepth(self.display, self.screen), // depth
                xlib::InputOutput as c_uint,                    // class
                self.visual,                                    // visual
                xlib::CWColormap | xlib::CWBackPixel | xlib::CWBorderPixel | xlib::CWEventMask, // value mask
                &mut attributes, // attributes
            );
        }
        if self.window == 0 {
            return Err(anyhow::anyhow!("XCreateWindow failed."));
        }
        debug!(
            "X window created (ID: {}), initial size: {}x{}px",
            self.window, pixel_width, pixel_height
        );
        self.current_pixel_width = pixel_width;
        self.current_pixel_height = pixel_height;
        Ok(())
    }

    /// Creates a Graphics Context (GC) for basic drawing operations like clearing.
    fn create_gc(&mut self) -> Result<()> {
        let gc_values: xlib::XGCValues = unsafe { mem::zeroed() }; // Initialize with defaults
        self.clear_gc = unsafe {
            xlib::XCreateGC(
                self.display,
                self.window,                      // GC is for this window
                0,                                // valuemask (0 for default GC)
                &gc_values as *const _ as *mut _, // values (null for default)
            )
        };
        if self.clear_gc.is_null() {
            return Err(anyhow::anyhow!("XCreateGC failed."));
        }
        debug!("Graphics Context (GC) for clearing created.");
        Ok(())
    }

    /// Sets up WM protocols (like WM_DELETE_WINDOW) and window hints.
    fn setup_wm_protocols_and_hints(&mut self) {
        unsafe {
            // Atom for WM_DELETE_WINDOW protocol
            self.wm_delete_window = xlib::XInternAtom(
                self.display,
                b"WM_DELETE_WINDOW\0".as_ptr() as *const i8, // C-string
                xlib::False,                                 // Don't create if it doesn't exist
            );
            // Atom for WM_PROTOCOLS property
            self.protocols_atom = xlib::XInternAtom(
                self.display,
                b"WM_PROTOCOLS\0".as_ptr() as *const i8,
                xlib::False,
            );

            if self.wm_delete_window != 0 && self.protocols_atom != 0 {
                // Tell the WM that we understand the WM_DELETE_WINDOW protocol.
                xlib::XSetWMProtocols(
                    self.display,
                    self.window,
                    [self.wm_delete_window].as_mut_ptr(), // Array of atoms
                    1,                                    // Count of atoms
                );
                debug!("WM_PROTOCOLS (WM_DELETE_WINDOW) registered.");
            } else {
                warn!(
                    "Failed to get WM_DELETE_WINDOW or WM_PROTOCOLS atom. Window close button might not work as expected."
                );
            }

            // Set window title (simple version, _NET_WM_NAME is preferred for UTF-8)
            let title_cstr = CString::new("core-term").expect("CString::new for title failed.");
            xlib::XStoreName(
                self.display,
                self.window,
                title_cstr.as_ptr() as *mut c_char,
            );

            // Set _NET_WM_NAME for UTF-8 titles (modern WMs)
            let net_wm_name_atom = xlib::XInternAtom(
                self.display,
                b"_NET_WM_NAME\0".as_ptr() as *const i8,
                xlib::False,
            );
            let utf8_string_atom = xlib::XInternAtom(
                self.display,
                b"UTF8_STRING\0".as_ptr() as *const i8,
                xlib::False,
            );
            if net_wm_name_atom != 0 && utf8_string_atom != 0 {
                xlib::XChangeProperty(
                    self.display,
                    self.window,
                    net_wm_name_atom,
                    utf8_string_atom,
                    8, // format is 8-bit for UTF8_STRING
                    xlib::PropModeReplace,
                    title_cstr.as_ptr() as *const u8,
                    title_cstr.as_bytes().len() as c_int,
                );
                debug!("Window title set via XStoreName and _NET_WM_NAME.");
            } else {
                debug!(
                    "Window title set via XStoreName only (_NET_WM_NAME or UTF8_STRING atom not found)."
                );
            }

            // Set size hints for the window manager.
            let mut size_hints: xlib::XSizeHints = mem::zeroed();
            size_hints.flags = xlib::PResizeInc | xlib::PMinSize; // We provide resize increments and min size.
            size_hints.width_inc = self.font_width as c_int; // Resize step width
            size_hints.height_inc = self.font_height as c_int; // Resize step height
            size_hints.min_width = self.font_width as c_int; // Minimum window width (1 cell)
            size_hints.min_height = self.font_height as c_int; // Minimum window height (1 cell)
                                                               // PBaseSize could also be set if borderpx were non-zero.
            xlib::XSetWMNormalHints(self.display, self.window, &mut size_hints);
            debug!("WM size hints set.");
        }
    }

    /// Resolves a `crate::color::Color` to a concrete `xft::XftColor`.
    /// This function handles named, indexed (approximated to RGB), and direct RGB colors.
    /// It panics if `Color::Default` is passed, as the Renderer should resolve defaults.
    fn resolve_concrete_xft_color(&mut self, color: Color) -> Result<xft::XftColor> {
        match color {
            Color::Default => {
                // This should ideally be caught by the Renderer.
                error!(
                    "XDriver::resolve_concrete_xft_color received Color::Default. This is a bug."
                );
                // Fallback to black, but this indicates an issue in the calling code (Renderer).
                // Consider panicking in debug builds or returning a specific error.
                self.cached_rgb_to_xft_color(0, 0, 0)
                    .context("Fallback to black failed after Color::Default error")
            }
            Color::Named(named_color) => {
                // Use the pre-allocated XftColor for standard ANSI colors.
                // The index directly corresponds to NamedColor's u8 representation.
                Ok(self.xft_ansi_colors[named_color as u8 as usize])
            }
            Color::Indexed(idx) => {
                // Convert indexed color to RGB, then get/cache the XftColor.
                // crate::color::convert_to_rgb_color returns a Color::Rgb.
                let rgb_equivalent = crate::color::convert_to_rgb_color(Color::Indexed(idx));
                if let Color::Rgb(r, g, b) = rgb_equivalent {
                    trace!(
                        "XDriver: Approximating Indexed({}) to RGB({},{},{}) for XftColor.",
                        idx,
                        r,
                        g,
                        b
                    );
                    self.cached_rgb_to_xft_color(r, g, b)
                } else {
                    // This path should ideally not be reached if convert_to_rgb_color is correct.
                    error!(
                        "Failed to convert Indexed({}) to RGB. Defaulting to black.",
                        idx
                    );
                    self.cached_rgb_to_xft_color(0, 0, 0)
                }
            }
            Color::Rgb(r, g, b) => {
                // Get/cache XftColor for direct RGB values.
                self.cached_rgb_to_xft_color(r, g, b)
            }
        }
    }

    /// Retrieves an `XftColor` for an RGB value, using a cache to avoid redundant allocations.
    fn cached_rgb_to_xft_color(&mut self, r_u8: u8, g_u8: u8, b_u8: u8) -> Result<xft::XftColor> {
        // Check cache first.
        if let Some(cached_color) = self.xft_color_cache_rgb.get(&(r_u8, g_u8, b_u8)) {
            return Ok(*cached_color); // Return a copy of the cached XftColor.
        }

        // Convert 8-bit RGB to 16-bit for XRenderColor (format 0xRRGGBB -> 0xRRRRGGGGBBBB).
        let color_components = Rgb16Components {
            r: ((r_u8 as u16) << 8) | (r_u8 as u16),
            g: ((g_u8 as u16) << 8) | (g_u8 as u16),
            b: ((b_u8 as u16) << 8) | (b_u8 as u16),
        };

        let render_color = XRenderColor {
            red: color_components.r,
            green: color_components.g,
            blue: color_components.b,
            alpha: XRENDER_ALPHA_OPAQUE, // Fully opaque
        };
        let mut new_xft_color: xft::XftColor = unsafe { mem::zeroed() };

        // SAFETY: FFI call to Xft.
        if unsafe {
            xft::XftColorAllocValue(
                self.display,
                self.visual,
                self.colormap,
                &render_color,      // Pass as const pointer
                &mut new_xft_color, // Pass as mutable pointer
            )
        } == 0
        {
            // XftColorAllocValue returns 0 on failure.
            Err(anyhow::anyhow!(
                "XftColorAllocValue failed for RGB({},{},{})",
                r_u8,
                g_u8,
                b_u8
            ))
        } else {
            // Store the newly allocated color in the cache.
            self.xft_color_cache_rgb
                .insert((r_u8, g_u8, b_u8), new_xft_color);
            Ok(new_xft_color)
        }
    }
}

// Helper function to translate X11 KeySym to our KeySymbol
fn xkeysym_to_keysymbol(keysym_val: xlib::KeySym, text: &str) -> KeySymbol {
    match keysym_val as u32 {
        // Modifier Keys (when the key itself is pressed)
        keysym::XK_Shift_L | keysym::XK_Shift_R => KeySymbol::Shift,
        keysym::XK_Control_L | keysym::XK_Control_R => KeySymbol::Control,
        keysym::XK_Alt_L | keysym::XK_Alt_R | keysym::XK_Meta_L | keysym::XK_Meta_R => {
            KeySymbol::Alt
        }
        keysym::XK_Super_L | keysym::XK_Super_R | keysym::XK_Hyper_L | keysym::XK_Hyper_R => {
            KeySymbol::Super
        }
        keysym::XK_Caps_Lock => KeySymbol::CapsLock,
        keysym::XK_Num_Lock => KeySymbol::NumLock,

        // Control characters
        keysym::XK_Return => KeySymbol::Enter,
        keysym::XK_KP_Enter => KeySymbol::KeypadEnter,
        keysym::XK_Linefeed => KeySymbol::Char('\n'),
        keysym::XK_BackSpace => KeySymbol::Backspace,
        keysym::XK_Tab | keysym::XK_KP_Tab | keysym::XK_ISO_Left_Tab => KeySymbol::Tab,
        keysym::XK_Escape => KeySymbol::Escape,

        // Navigation keys
        keysym::XK_Home => KeySymbol::Home,
        keysym::XK_Left => KeySymbol::Left,
        keysym::XK_Up => KeySymbol::Up,
        keysym::XK_Right => KeySymbol::Right,
        keysym::XK_Down => KeySymbol::Down,
        keysym::XK_Page_Up | keysym::XK_Prior => KeySymbol::PageUp,
        keysym::XK_Page_Down | keysym::XK_Next => KeySymbol::PageDown,
        keysym::XK_End => KeySymbol::End,
        keysym::XK_Insert => KeySymbol::Insert,
        keysym::XK_Delete => KeySymbol::Delete,

        // Function keys
        keysym::XK_F1 => KeySymbol::F1,
        keysym::XK_F2 => KeySymbol::F2,
        keysym::XK_F3 => KeySymbol::F3,
        keysym::XK_F4 => KeySymbol::F4,
        keysym::XK_F5 => KeySymbol::F5,
        keysym::XK_F6 => KeySymbol::F6,
        keysym::XK_F7 => KeySymbol::F7,
        keysym::XK_F8 => KeySymbol::F8,
        keysym::XK_F9 => KeySymbol::F9,
        keysym::XK_F10 => KeySymbol::F10,
        keysym::XK_F11 => KeySymbol::F11,
        keysym::XK_F12 => KeySymbol::F12,
        keysym::XK_F13 => KeySymbol::F13,
        keysym::XK_F14 => KeySymbol::F14,
        keysym::XK_F15 => KeySymbol::F15,
        keysym::XK_F16 => KeySymbol::F16,
        keysym::XK_F17 => KeySymbol::F17,
        keysym::XK_F18 => KeySymbol::F18,
        keysym::XK_F19 => KeySymbol::F19,
        keysym::XK_F20 => KeySymbol::F20,
        keysym::XK_F21 => KeySymbol::F21,
        keysym::XK_F22 => KeySymbol::F22,
        keysym::XK_F23 => KeySymbol::F23,
        keysym::XK_F24 => KeySymbol::F24,

        // Keypad specific symbols (digits, operators)
        keysym::XK_KP_0 | keysym::XK_KP_Insert => KeySymbol::Keypad0,
        keysym::XK_KP_1 | keysym::XK_KP_End => KeySymbol::Keypad1,
        keysym::XK_KP_2 | keysym::XK_KP_Down => KeySymbol::Keypad2,
        keysym::XK_KP_3 | keysym::XK_KP_Page_Down | keysym::XK_KP_Next => KeySymbol::Keypad3,
        keysym::XK_KP_4 | keysym::XK_KP_Left => KeySymbol::Keypad4,
        keysym::XK_KP_5 | keysym::XK_KP_Begin => KeySymbol::Keypad5,
        keysym::XK_KP_6 | keysym::XK_KP_Right => KeySymbol::Keypad6,
        keysym::XK_KP_7 | keysym::XK_KP_Home => KeySymbol::Keypad7,
        keysym::XK_KP_8 | keysym::XK_KP_Up => KeySymbol::Keypad8,
        keysym::XK_KP_9 | keysym::XK_KP_Page_Up | keysym::XK_KP_Prior => KeySymbol::Keypad9,

        keysym::XK_KP_Decimal | keysym::XK_KP_Delete | keysym::XK_KP_Separator => {
            KeySymbol::KeypadDecimal
        }
        keysym::XK_KP_Add => KeySymbol::KeypadPlus,
        keysym::XK_KP_Subtract => KeySymbol::KeypadMinus,
        keysym::XK_KP_Multiply => KeySymbol::KeypadMultiply,
        keysym::XK_KP_Divide => KeySymbol::KeypadDivide,
        keysym::XK_KP_Equal => KeySymbol::KeypadEquals,
        keysym::XK_KP_Space => KeySymbol::Char(' '),

        // Other common keys
        keysym::XK_Print | keysym::XK_Sys_Req => KeySymbol::PrintScreen,
        keysym::XK_Scroll_Lock => KeySymbol::ScrollLock,
        keysym::XK_Pause | keysym::XK_Break => KeySymbol::Pause,
        keysym::XK_Menu => KeySymbol::Menu,

        _ => {
            if let Some(ch) = text.chars().next() {
                return KeySymbol::Char(ch);
            }
            KeySymbol::Unknown
        }
    }
}

/// Ensures X11 resources are cleaned up when the `XDriver` instance is dropped.
impl Drop for XDriver {
    fn drop(&mut self) {
        info!("Dropping XDriver instance, performing cleanup.");
        if let Err(e) = self.cleanup() {
            // Log error, but avoid panicking in drop.
            error!("Error during XDriver cleanup in drop: {}", e);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*; // To import XDriver, xkeysym_to_keysymbol, Modifiers, KeySymbol
    use x11::keysym;
    use x11::xlib; // For xlib::KeySym and event structures // For XK_* constants

    // Helper to create a basic XKeyEvent for testing modifier extraction
    fn mock_xkey_event(state: u32, keycode: u32) -> xlib::XKeyEvent {
        xlib::XKeyEvent {
            type_: xlib::KeyPress,
            serial: 0,
            send_event: 0,
            display: std::ptr::null_mut(), // Not used by current modifier logic
            window: 0,
            root: 0,
            subwindow: 0,
            time: 0,
            x: 0,
            y: 0,
            x_root: 0,
            y_root: 0,
            state: state as std::os::raw::c_uint, // XKeyEvent state is c_uint
            keycode: keycode as std::os::raw::c_uint, // keycode is c_uint
            same_screen: 0,
        }
    }

    // Tests for modifier mapping
    #[test]
    fn test_modifier_mapping() {
        let mut event = xlib::XEvent {
            key: mock_xkey_event(0, 0),
        }; // type needs to be KeyPress for key field to be valid
        event.type_ = xlib::KeyPress;

        // No modifiers
        event.key.state = 0;
        let xkey_event = &event.key;
        let mut modifiers = Modifiers::empty();
        if (xkey_event.state & xlib::ShiftMask) != 0 {
            modifiers.insert(Modifiers::SHIFT);
        }
        if (xkey_event.state & xlib::ControlMask) != 0 {
            modifiers.insert(Modifiers::CONTROL);
        }
        if (xkey_event.state & xlib::Mod1Mask) != 0 {
            modifiers.insert(Modifiers::ALT);
        }
        if (xkey_event.state & xlib::Mod4Mask) != 0 {
            modifiers.insert(Modifiers::SUPER);
        }
        assert_eq!(modifiers, Modifiers::empty());

        // Shift
        event.key.state = xlib::ShiftMask;
        let xkey_event = &event.key;
        let mut modifiers = Modifiers::empty();
        if (xkey_event.state & xlib::ShiftMask) != 0 {
            modifiers.insert(Modifiers::SHIFT);
        }
        if (xkey_event.state & xlib::ControlMask) != 0 {
            modifiers.insert(Modifiers::CONTROL);
        }
        if (xkey_event.state & xlib::Mod1Mask) != 0 {
            modifiers.insert(Modifiers::ALT);
        }
        if (xkey_event.state & xlib::Mod4Mask) != 0 {
            modifiers.insert(Modifiers::SUPER);
        }
        assert_eq!(modifiers, Modifiers::SHIFT);

        // Control
        event.key.state = xlib::ControlMask;
        let xkey_event = &event.key;
        let mut modifiers = Modifiers::empty();
        if (xkey_event.state & xlib::ShiftMask) != 0 {
            modifiers.insert(Modifiers::SHIFT);
        }
        if (xkey_event.state & xlib::ControlMask) != 0 {
            modifiers.insert(Modifiers::CONTROL);
        }
        if (xkey_event.state & xlib::Mod1Mask) != 0 {
            modifiers.insert(Modifiers::ALT);
        }
        if (xkey_event.state & xlib::Mod4Mask) != 0 {
            modifiers.insert(Modifiers::SUPER);
        }
        assert_eq!(modifiers, Modifiers::CONTROL);

        // Alt (Mod1Mask)
        event.key.state = xlib::Mod1Mask;
        let xkey_event = &event.key;
        let mut modifiers = Modifiers::empty();
        if (xkey_event.state & xlib::ShiftMask) != 0 {
            modifiers.insert(Modifiers::SHIFT);
        }
        if (xkey_event.state & xlib::ControlMask) != 0 {
            modifiers.insert(Modifiers::CONTROL);
        }
        if (xkey_event.state & xlib::Mod1Mask) != 0 {
            modifiers.insert(Modifiers::ALT);
        }
        if (xkey_event.state & xlib::Mod4Mask) != 0 {
            modifiers.insert(Modifiers::SUPER);
        }
        assert_eq!(modifiers, Modifiers::ALT);

        // Super (Mod4Mask)
        event.key.state = xlib::Mod4Mask;
        let xkey_event = &event.key;
        let mut modifiers = Modifiers::empty();
        if (xkey_event.state & xlib::ShiftMask) != 0 {
            modifiers.insert(Modifiers::SHIFT);
        }
        if (xkey_event.state & xlib::ControlMask) != 0 {
            modifiers.insert(Modifiers::CONTROL);
        }
        if (xkey_event.state & xlib::Mod1Mask) != 0 {
            modifiers.insert(Modifiers::ALT);
        }
        if (xkey_event.state & xlib::Mod4Mask) != 0 {
            modifiers.insert(Modifiers::SUPER);
        }
        assert_eq!(modifiers, Modifiers::SUPER);

        // Shift + Control
        event.key.state = xlib::ShiftMask | xlib::ControlMask;
        let xkey_event = &event.key;
        let mut modifiers = Modifiers::empty();
        if (xkey_event.state & xlib::ShiftMask) != 0 {
            modifiers.insert(Modifiers::SHIFT);
        }
        if (xkey_event.state & xlib::ControlMask) != 0 {
            modifiers.insert(Modifiers::CONTROL);
        }
        if (xkey_event.state & xlib::Mod1Mask) != 0 {
            modifiers.insert(Modifiers::ALT);
        }
        if (xkey_event.state & xlib::Mod4Mask) != 0 {
            modifiers.insert(Modifiers::SUPER);
        }
        assert_eq!(modifiers, Modifiers::SHIFT | Modifiers::CONTROL);
    }

    // Tests for xkeysym_to_keysymbol function
    #[test]
    fn test_xkeysym_to_keysymbol_special_keys() {
        assert_eq!(
            xkeysym_to_keysymbol(keysym::XK_Return, ""),
            KeySymbol::Enter
        );
        assert_eq!(
            xkeysym_to_keysymbol(keysym::XK_Escape, ""),
            KeySymbol::Escape
        );
        assert_eq!(
            xkeysym_to_keysymbol(keysym::XK_BackSpace, ""),
            KeySymbol::Backspace
        );
        assert_eq!(xkeysym_to_keysymbol(keysym::XK_Tab, ""), KeySymbol::Tab);
        assert_eq!(
            xkeysym_to_keysymbol(keysym::XK_Shift_L, ""),
            KeySymbol::Shift
        );
        assert_eq!(
            xkeysym_to_keysymbol(keysym::XK_Control_R, ""),
            KeySymbol::Control
        );
        assert_eq!(xkeysym_to_keysymbol(keysym::XK_Alt_L, ""), KeySymbol::Alt);
        assert_eq!(
            xkeysym_to_keysymbol(keysym::XK_Super_L, ""),
            KeySymbol::Super
        );
        assert_eq!(xkeysym_to_keysymbol(keysym::XK_Home, ""), KeySymbol::Home);
        assert_eq!(xkeysym_to_keysymbol(keysym::XK_Left, ""), KeySymbol::Left);
        assert_eq!(xkeysym_to_keysymbol(keysym::XK_F1, ""), KeySymbol::F1);
        assert_eq!(xkeysym_to_keysymbol(keysym::XK_F12, ""), KeySymbol::F12);
        assert_eq!(
            xkeysym_to_keysymbol(keysym::XK_Delete, ""),
            KeySymbol::Delete
        );
        assert_eq!(
            xkeysym_to_keysymbol(keysym::XK_Insert, ""),
            KeySymbol::Insert
        );
    }

    #[test]
    fn test_xkeysym_to_keysymbol_char_input() {
        // XLookupString provides the char, keysym might be generic or specific
        assert_eq!(
            xkeysym_to_keysymbol(keysym::XK_a, "a"),
            KeySymbol::Char('a')
        );
        assert_eq!(
            xkeysym_to_keysymbol(keysym::XK_A, "A"),
            KeySymbol::Char('A')
        ); // Shifted 'a'
        assert_eq!(
            xkeysym_to_keysymbol(keysym::XK_b, "b"),
            KeySymbol::Char('b')
        );
        assert_eq!(
            xkeysym_to_keysymbol(keysym::XK_eacute, "é"),
            KeySymbol::Char('é')
        ); // Char from compose/dead key
        assert_eq!(
            xkeysym_to_keysymbol(keysym::XK_plus, "+"),
            KeySymbol::Char('+')
        );
        assert_eq!(
            xkeysym_to_keysymbol(keysym::XK_space, " "),
            KeySymbol::Char(' ')
        );
        assert_eq!(
            xkeysym_to_keysymbol(keysym::XK_KP_Space, " "),
            KeySymbol::Char(' ')
        ); // Keypad space
    }

    #[test]
    fn test_xkeysym_to_keysymbol_keypad_numbers_as_char() {
        // When NumLock is on, XLookupString typically provides the digit as text
        // The current xkeysym_to_keysymbol maps XK_KP_0 to KeySymbol::Keypad0 explicitly,
        // so these tests should reflect that.
        assert_eq!(
            xkeysym_to_keysymbol(keysym::XK_KP_0, "0"),
            KeySymbol::Keypad0
        ); // Explicit mapping
        assert_eq!(
            xkeysym_to_keysymbol(keysym::XK_KP_1, "1"),
            KeySymbol::Keypad1
        ); // Explicit mapping
        assert_eq!(
            xkeysym_to_keysymbol(keysym::XK_KP_Decimal, "."),
            KeySymbol::KeypadDecimal
        );
        assert_eq!(
            xkeysym_to_keysymbol(keysym::XK_KP_Add, "+"),
            KeySymbol::KeypadPlus
        );
    }

    #[test]
    fn test_xkeysym_to_keysymbol_keypad_navigation_no_text() {
        // Simulate NumLock off: keysym is e.g. XK_KP_Home, text is empty
        // The current xkeysym_to_keysymbol maps these to KeySymbol::KeypadX
        assert_eq!(
            xkeysym_to_keysymbol(keysym::XK_KP_Home, ""),
            KeySymbol::Keypad7
        );
        assert_eq!(
            xkeysym_to_keysymbol(keysym::XK_KP_Left, ""),
            KeySymbol::Keypad4
        );
        assert_eq!(
            xkeysym_to_keysymbol(keysym::XK_KP_Up, ""),
            KeySymbol::Keypad8
        );
        // This confirms that XK_KP_Home (etc.) are treated as Keypad7 (etc.)
        // rather than KeySymbol::Home if text is empty.
    }

    #[test]
    fn test_xkeysym_to_keysymbol_unknown() {
        // A keysym not in our map, and XLookupString provides no text
        assert_eq!(xkeysym_to_keysymbol(0xFFFF, ""), KeySymbol::Unknown); // 0xFFFF is User Private Keysym Area
        assert_eq!(
            xkeysym_to_keysymbol(keysym::XK_Hangul, ""),
            KeySymbol::Unknown
        ); // Assuming Hangul not mapped and produces no simple text
    }
}
