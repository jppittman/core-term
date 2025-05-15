// src/backends/x11.rs

#![allow(non_snake_case)] // Allow non-snake case for X11 types

// Import logging macros
use log::{debug, info, warn, error, trace};

// Crate-level imports
use crate::glyph::{Color, AttrFlags, NamedColor};
// Updated to use new structs from backends::mod_rs
use crate::backends::{Driver, BackendEvent, CellCoords, TextRunStyle, CellRect}; 

use anyhow::{Context, Result};
use std::ffi::CString;
use std::os::unix::io::RawFd;
use std::ptr;
use std::mem;
use std::collections::HashMap;

// Libc imports
use libc::{c_char, c_int, c_uint, c_ulong};

// X11 library imports
use x11::xlib;
use x11::keysym;
use x11::xft;
use x11::xrender::{XGlyphInfo, XRenderColor};

// --- Constants ---
const DEFAULT_FONT_NAME: &str = "Liberation Mono:size=10";
const MIN_FONT_WIDTH: u32 = 1;
const MIN_FONT_HEIGHT: u32 = 1;

// Color indexing for pre-allocated XftColors
const ANSI_COLOR_COUNT: usize = 16; // For indices 0-15
const DEFAULT_FG_COLOR_IDX: usize = ANSI_COLOR_COUNT; // Index 16
const DEFAULT_BG_COLOR_IDX: usize = ANSI_COLOR_COUNT + 1; // Index 17
const TOTAL_PREALLOC_COLORS: usize = ANSI_COLOR_COUNT + 2; // 0-15 for ANSI, +2 for default FG/BG

const DEFAULT_WINDOW_WIDTH_CHARS: usize = 80;
const DEFAULT_WINDOW_HEIGHT_CHARS: usize = 24;

/// Alpha value for fully opaque XRenderColor.
const XRENDER_ALPHA_OPAQUE: u16 = 0xffff;

/// Helper struct to group 16-bit RGB color components for XRenderColor.
struct Rgb16Components {
    r: u16,
    g: u16,
    b: u16,
}

/// X11 driver implementation for the terminal using Xft for font rendering.
///
/// This struct manages the X11 display connection, window, fonts, colors,
/// and graphics contexts necessary to render the terminal. It translates
/// abstract drawing commands from the `Renderer` into X11/Xft calls and
/// processes X11 server events into `BackendEvent`s for the orchestrator.
pub struct XDriver {
    display: *mut xlib::Display,
    screen: c_int,
    window: xlib::Window,
    colormap: xlib::Colormap,
    visual: *mut xlib::Visual,
    xft_font: *mut xft::XftFont,
    xft_draw: *mut xft::XftDraw,
    
    /// Stores pre-allocated XftColors:
    /// Indices 0-15: Standard ANSI colors.
    /// Index DEFAULT_FG_COLOR_IDX (16): Default foreground.
    /// Index DEFAULT_BG_COLOR_IDX (17): Default background.
    xft_colors: Vec<xft::XftColor>,
    xft_color_cache_rgb: HashMap<(u8, u8, u8), xft::XftColor>,
    
    font_width: u32,
    font_height: u32,
    font_ascent: u32,

    current_pixel_width: u16,
    current_pixel_height: u16,

    wm_delete_window: xlib::Atom,
    protocols_atom: xlib::Atom,
    clear_gc: xlib::GC,
}

impl Driver for XDriver {
    /// Creates and initializes a new X11 driver instance.
    ///
    /// This involves:
    /// 1. Connecting to the X server.
    /// 2. Loading the specified Xft font and determining its metrics.
    /// 3. Initializing a palette of XftColor resources (ANSI colors + defaults).
    /// 4. Creating the main application window with appropriate X11 attributes.
    /// 5. Creating a Graphics Context (GC) for clearing operations.
    /// 6. Setting up Window Manager (WM) protocols (e.g., for handling close button).
    /// 7. Mapping the window to make it visible.
    ///
    /// If any step fails, an error is returned, and partial resources are cleaned up.
    fn new() -> Result<Self> {
        info!("Creating new XDriver");
        // Safety: FFI call to connect to X server. display must be non-null.
        let display = unsafe { xlib::XOpenDisplay(ptr::null()) };
        if display.is_null() {
            return Err(anyhow::anyhow!("Failed to open X display: Check DISPLAY environment variable or X server status."));
        }
        debug!("X display opened successfully.");

        // Safety: FFI calls, assume display is valid.
        let screen = unsafe { xlib::XDefaultScreen(display) };
        let colormap = unsafe { xlib::XDefaultColormap(display, screen) };
        let visual = unsafe { xlib::XDefaultVisual(display, screen) };

        let mut driver = XDriver {
            display,
            screen,
            window: 0,
            colormap,
            visual,
            xft_font: ptr::null_mut(),
            xft_draw: ptr::null_mut(),
            xft_colors: Vec::with_capacity(TOTAL_PREALLOC_COLORS),
            xft_color_cache_rgb: HashMap::new(),
            font_width: 0,
            font_height: 0,
            font_ascent: 0,
            current_pixel_width: 0,
            current_pixel_height: 0,
            wm_delete_window: 0,
            protocols_atom: 0,
            clear_gc: ptr::null_mut(),
        };

        // Use an IIFE-like closure for sequential initialization with error handling
        // and cleanup on failure.
        if let Err(e) = (|| {
            driver.load_font().context("Failed to load font")?;
            
            // Initial pixel dimensions are derived from default char size and font metrics.
            // These will be updated by the first ConfigureNotify if the WM resizes the window.
            driver.current_pixel_width = (DEFAULT_WINDOW_WIDTH_CHARS * driver.font_width as usize) as u16;
            driver.current_pixel_height = (DEFAULT_WINDOW_HEIGHT_CHARS * driver.font_height as usize) as u16;

            driver.init_xft_colors().context("Failed to initialize Xft colors")?;
            driver.create_window(driver.current_pixel_width, driver.current_pixel_height)
                .context("Failed to create window")?;
            driver.create_gc().context("Failed to create graphics context for clearing")?;

            // Safety: FFI call, assumes display, visual, colormap, window are valid.
            driver.xft_draw = unsafe { xft::XftDrawCreate(driver.display, driver.window, driver.visual, driver.colormap) };
            if driver.xft_draw.is_null() {
                return Err(anyhow::anyhow!("Failed to create XftDraw object"));
            }
            debug!("XftDraw object created.");

            driver.setup_wm_protocols_and_hints(); // Combined setup for WM interactions

            // Map and flush the window to make it visible.
            // Safety: FFI calls, assumes display and window are valid.
            unsafe {
                xlib::XMapWindow(driver.display, driver.window);
                xlib::XFlush(driver.display);
            }
            debug!("Window mapped and flushed.");
            Ok(())
        })() {
            error!("Error during XDriver setup: {:?}", e);
            // Explicit cleanup ensures resources are freed even if Drop is not yet called
            // (e.g., if this function returns Err early).
            let _ = driver.cleanup(); 
            return Err(e);
        }

        info!("XDriver initialization complete.");
        Ok(driver)
    }

    /// Returns the X11 connection file descriptor.
    ///
    /// This FD can be monitored by an event loop (e.g., `epoll`) for pending X events.
    fn get_event_fd(&self) -> Option<RawFd> {
        // Safety: FFI call, assumes self.display is a valid X11 Display pointer.
        Some(unsafe { xlib::XConnectionNumber(self.display) })
    }

    /// Processes pending X11 events and translates them into `BackendEvent`s.
    ///
    /// This method should be called when the event loop indicates activity on the
    /// X11 connection file descriptor. It polls for all pending X events,
    /// translates relevant ones (KeyPress, ConfigureNotify, ClientMessage for close, Focus)
    /// into `BackendEvent`s, and returns a vector of these events.
    fn process_events(&mut self) -> Result<Vec<BackendEvent>> {
        let mut backend_events = Vec::new();
        // Loop while there are pending X events.
        // Safety: FFI call, assumes self.display is valid.
        while unsafe { xlib::XPending(self.display) } > 0 {
            let mut xevent: xlib::XEvent = unsafe { mem::zeroed() };
            // Safety: FFI call to get the next event. Assumes self.display is valid.
            unsafe { xlib::XNextEvent(self.display, &mut xevent) };

            let event_type = unsafe { xevent.type_ };
            match event_type {
                xlib::Expose => {
                    let xexpose = unsafe { xevent.expose };
                    // Only handle the last Expose event in a series.
                    if xexpose.count == 0 {
                        // An Expose event signifies that a portion of the window needs to be redrawn.
                        // The orchestrator is expected to call renderer.draw() which will
                        // redraw dirty parts of the terminal or the whole terminal.
                        // No specific BackendEvent is generated here, as the need for redraw
                        // is a consequence handled by the rendering pipeline.
                        debug!("XEvent: Expose (x:{}, y:{}, w:{}, h:{}) - redraw will be handled by renderer", 
                               xexpose.x, xexpose.y, xexpose.width, xexpose.height);
                    }
                }
                xlib::ConfigureNotify => {
                    let xconfigure = unsafe { xevent.configure };
                    // Window was resized, moved, or restacked. We only care about size changes.
                    if self.current_pixel_width != xconfigure.width as u16 || self.current_pixel_height != xconfigure.height as u16 {
                        debug!(
                            "XEvent: ConfigureNotify (resize from {}x{} to {}x{})",
                            self.current_pixel_width, self.current_pixel_height,
                            xconfigure.width, xconfigure.height
                        );
                        self.current_pixel_width = xconfigure.width as u16;
                        self.current_pixel_height = xconfigure.height as u16;
                        backend_events.push(BackendEvent::Resize {
                            width_px: self.current_pixel_width,
                            height_px: self.current_pixel_height,
                        });
                    }
                }
                xlib::KeyPress => {
                    // Safety: XLookupString is an FFI call. It modifies key_text_buffer and keysym.
                    let mut keysym: xlib::KeySym = 0;
                    let mut key_text_buffer: [u8; 32] = [0; 32];
                    let count = unsafe {
                        xlib::XLookupString(
                            &mut xevent.key,
                            key_text_buffer.as_mut_ptr() as *mut c_char,
                            key_text_buffer.len() as c_int,
                            &mut keysym,
                            ptr::null_mut(), // XComposeStatus (optional)
                        )
                    };
                    let text = if count > 0 {
                        String::from_utf8_lossy(&key_text_buffer[0..count as usize]).to_string()
                    } else {
                        String::new()
                    };
                    debug!("XEvent: KeyPress (keysym: {:#X}, text: '{}')", keysym, text);
                    backend_events.push(BackendEvent::Key {
                        keysym: keysym as u32,
                        text,
                    });
                }
                xlib::ClientMessage => {
                    let xclient = unsafe { xevent.client_message };
                    // Check if this is a WM_DELETE_WINDOW message.
                    if xclient.message_type == self.protocols_atom && xclient.data.get_long(0) as xlib::Atom == self.wm_delete_window {
                        info!("XEvent: WM_DELETE_WINDOW received from window manager.");
                        backend_events.push(BackendEvent::CloseRequested);
                    } else {
                        trace!("XEvent: Ignored ClientMessage (type: {})", xclient.message_type);
                    }
                }
                xlib::FocusIn => {
                    debug!("XEvent: FocusIn received.");
                    backend_events.push(BackendEvent::FocusGained);
                }
                xlib::FocusOut => {
                    debug!("XEvent: FocusOut received.");
                    backend_events.push(BackendEvent::FocusLost);
                }
                // TODO: Handle other X events like ButtonPress, MotionNotify for mouse support.
                _ => {
                    trace!("XEvent: Ignored (type: {})", event_type);
                }
            }
        }
        Ok(backend_events)
    }

    /// Retrieves the dimensions (width, height) of a single character cell in pixels.
    /// These dimensions are determined from the loaded Xft font.
    fn get_font_dimensions(&self) -> (usize, usize) {
        (self.font_width as usize, self.font_height as usize)
    }

    /// Retrieves the current dimensions (width, height) of the display area
    /// (i.e., the window's client area) in pixels.
    /// These dimensions are updated via `ConfigureNotify` X events.
    fn get_display_dimensions_pixels(&self) -> (u16, u16) {
        (self.current_pixel_width, self.current_pixel_height)
    }

    /// Clears the entire display area (window) with the specified background color.
    /// This uses a simple X11 GC fill operation.
    fn clear_all(&mut self, bg: Color) -> Result<()> {
        let xft_bg_color = self.resolve_xft_color(bg)
            .context("Failed to resolve background color for clear_all")?;
        
        // Safety: FFI calls, assumes display, window, clear_gc are valid and initialized.
        unsafe {
            // Set the foreground of the GC to the desired background color for filling.
            xlib::XSetForeground(self.display, self.clear_gc, xft_bg_color.pixel);
            // Fill the entire window rectangle.
            xlib::XFillRectangle(
                self.display,
                self.window,
                self.clear_gc,
                0, 0, // x, y
                self.current_pixel_width as u32, // width
                self.current_pixel_height as u32, // height
            );
        }
        trace!("Window cleared with color pixel: {}", xft_bg_color.pixel);
        Ok(())
    }

    /// Draws a run of text characters at a given cell coordinate.
    ///
    /// The `Renderer` is expected to have handled `AttrFlags::REVERSE` by swapping
    /// foreground and background colors before calling this. This method handles
    /// drawing the background of the text run, the text itself, and any decorations
    /// like underline or strikethrough based on the provided `style.flags`.
    fn draw_text_run(
        &mut self,
        coords: CellCoords,
        text: &str,
        style: TextRunStyle,
    ) -> Result<()> {
        if text.is_empty() {
            return Ok(()); // Nothing to draw.
        }

        let x_pixel = (coords.x * self.font_width as usize) as c_int;
        let y_pixel = (coords.y * self.font_height as usize) as c_int;
        let run_pixel_width = text.chars().count() * self.font_width as usize;

        let xft_fg = self.resolve_xft_color(style.fg).context("Failed to resolve foreground color for text run")?;
        let xft_bg = self.resolve_xft_color(style.bg).context("Failed to resolve background color for text run")?;

        // 1. Draw background rectangle for the text run.
        // Safety: FFI call, assumes self.xft_draw is valid.
        unsafe {
            xft::XftDrawRect(
                self.xft_draw,
                &xft_bg,
                x_pixel,
                y_pixel,
                run_pixel_width as u32,
                self.font_height,
            );
        }

        // 2. Draw the text string using Xft.
        let c_text = CString::new(text).context("Failed to convert text to CString for Xft")?;
        let baseline_y_pixel = y_pixel + self.font_ascent as c_int;
        // Safety: FFI call, assumes self.xft_draw and self.xft_font are valid.
        unsafe {
            xft::XftDrawStringUtf8(
                self.xft_draw,
                &xft_fg,
                self.xft_font,
                x_pixel,
                baseline_y_pixel,
                c_text.as_ptr() as *const u8,
                c_text.as_bytes().len() as c_int,
            );
        }
        
        // 3. Draw decorations (underline, strikethrough) if specified in style.flags.
        if style.flags.contains(AttrFlags::UNDERLINE) {
            let underline_y = y_pixel + self.font_height as c_int - 2; 
             // Safety: FFI call, assumes self.xft_draw is valid.
            unsafe {
                xft::XftDrawRect(self.xft_draw, &xft_fg, x_pixel, underline_y, run_pixel_width as u32, 1);
            }
        }
        if style.flags.contains(AttrFlags::STRIKETHROUGH) {
            let strikethrough_y = y_pixel + (self.font_ascent / 2) as c_int;
             // Safety: FFI call, assumes self.xft_draw is valid.
            unsafe {
                xft::XftDrawRect(self.xft_draw, &xft_fg, x_pixel, strikethrough_y, run_pixel_width as u32, 1);
            }
        }
        Ok(())
    }

    /// Fills a rectangular area of cells with a specified color using Xft.
    fn fill_rect(
        &mut self,
        rect: CellRect,
        color: Color,
    ) -> Result<()> {
        if rect.width == 0 || rect.height == 0 {
            return Ok(()); // Nothing to fill.
        }
        let x_pixel = (rect.x * self.font_width as usize) as c_int;
        let y_pixel = (rect.y * self.font_height as usize) as c_int;
        let rect_pixel_width = (rect.width * self.font_width as usize) as u32;
        let rect_pixel_height = (rect.height * self.font_height as usize) as u32;

        let xft_fill_color = self.resolve_xft_color(color).context("Failed to resolve color for fill_rect")?;

        // Safety: FFI call, assumes self.xft_draw is valid.
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

    /// Presents the composed frame to the display.
    /// For X11, this typically means flushing the X command buffer to ensure
    /// all drawing commands are sent to and processed by the X server.
    fn present(&mut self) -> Result<()> {
        // Safety: FFI call, assumes self.display is a valid X11 Display pointer.
        unsafe {
            xlib::XFlush(self.display);
        }
        trace!("XFlush called to present frame.");
        Ok(())
    }

    /// Performs cleanup of X11 and Xft resources.
    /// This includes freeing Xft fonts, colors, drawables, GCs, destroying the window,
    /// and closing the connection to the X server.
    /// This method is crucial for releasing resources gracefully.
    fn cleanup(&mut self) -> Result<()> {
        info!("Cleaning up XDriver resources...");
        // Safety: All subsequent calls are FFI to Xlib or Xft for resource deallocation.
        // The order of operations is important: dependent resources (e.g., XftDraw associated
        // with a window) should generally be freed before the resources they depend on.
        unsafe {
            if !self.xft_draw.is_null() {
                trace!("Destroying XftDraw object.");
                xft::XftDrawDestroy(self.xft_draw);
                self.xft_draw = ptr::null_mut(); // Nullify to prevent double free on error/multiple calls
            }
            if !self.xft_font.is_null() {
                trace!("Closing XftFont.");
                xft::XftFontClose(self.display, self.xft_font);
                self.xft_font = ptr::null_mut();
            }
            
            trace!("Freeing {} preallocated XftColors.", self.xft_colors.len());
            for color_ptr in self.xft_colors.iter_mut() {
                // XftColorFree takes a pointer to the XftColor struct.
                xft::XftColorFree(self.display, self.visual, self.colormap, color_ptr);
            }
            self.xft_colors.clear(); // Clear the vector after freeing individual elements.

            trace!("Freeing {} cached RGB XftColors.", self.xft_color_cache_rgb.len());
            for (_, mut cached_color) in self.xft_color_cache_rgb.drain() {
                 xft::XftColorFree(self.display, self.visual, self.colormap, &mut cached_color);
            }
            // self.xft_color_cache_rgb is already cleared by drain().

            if !self.clear_gc.is_null() {
                trace!("Freeing clear GC.");
                xlib::XFreeGC(self.display, self.clear_gc);
                self.clear_gc = ptr::null_mut();
            }
            if self.window != 0 {
                trace!("Destroying X window (ID: {}).", self.window);
                xlib::XDestroyWindow(self.display, self.window);
                self.window = 0; // Mark as destroyed
            }
            if !self.display.is_null() {
                trace!("Closing X display connection.");
                xlib::XCloseDisplay(self.display);
                self.display = ptr::null_mut(); // Mark as closed
            }
        }
        info!("XDriver cleanup complete.");
        Ok(())
    }
}

// Private helper methods for XDriver
impl XDriver {
    /// Loads the specified Xft font and determines its metrics (width, height, ascent).
    /// Font metrics are stored in `self.font_width`, `self.font_height`, `self.font_ascent`.
    fn load_font(&mut self) -> Result<()> {
        debug!("Loading font: {}", DEFAULT_FONT_NAME);
        let font_name_cstr = CString::new(DEFAULT_FONT_NAME)
            .context("Failed to create CString for font name")?;
        
        // Safety: FFI call to XftFontOpenName. self.display and self.screen must be valid.
        self.xft_font = unsafe { xft::XftFontOpenName(self.display, self.screen, font_name_cstr.as_ptr()) };
        if self.xft_font.is_null() {
            // Provide a more helpful error message if font loading fails.
            return Err(anyhow::anyhow!(
                "XftFontOpenName failed for font: '{}'. Ensure the font is installed, accessible to Fontconfig, and the name is correct.",
                DEFAULT_FONT_NAME
            ));
        }
        debug!("Font '{}' loaded successfully via XftFontOpenName.", DEFAULT_FONT_NAME);

        // Get font metrics from the loaded XftFont.
        // Safety: Dereferencing self.xft_font (pointer to XftFont struct), which was checked for null.
        // The XftFont struct contains ascent, descent, height, etc.
        let font_info_ptr = self.xft_font;
        self.font_height = unsafe { ((*font_info_ptr).ascent + (*font_info_ptr).descent) as u32 };
        self.font_ascent = unsafe { (*font_info_ptr).ascent as u32 };
        
        // Determine font_width using XftTextExtentsUtf8 with a sample character (e.g., 'M').
        // This gives the advance width for typical characters in a monospace font.
        let mut extents: XGlyphInfo = unsafe { mem::zeroed() };
        // Using "M" as a representative character for width calculation.
        let sample_char_cstr = CString::new("M").expect("CString::new for 'M' should not fail.");
        // Safety: FFI call. self.display and self.xft_font must be valid.
        unsafe {
            xft::XftTextExtentsUtf8(
                self.display,
                self.xft_font,
                sample_char_cstr.as_ptr() as *const u8,
                sample_char_cstr.as_bytes().len() as c_int,
                &mut extents,
            );
        }
        self.font_width = extents.xOff as u32; // xOff provides the advance width.

        if self.font_width < MIN_FONT_WIDTH || self.font_height < MIN_FONT_HEIGHT {
            return Err(anyhow::anyhow!(
                "Loaded font dimensions (width:{}, height:{}) are below minimum requirements (width:{}, height:{}).",
                self.font_width, self.font_height, MIN_FONT_WIDTH, MIN_FONT_HEIGHT
            ));
        }
        info!("Font metrics determined: Width={}, Height={}, Ascent={}", self.font_width, self.font_height, self.font_ascent);
        Ok(())
    }

    /// Initializes a set of pre-allocated XftColor resources.
    /// This includes the 16 standard ANSI colors and default foreground/background colors.
    fn init_xft_colors(&mut self) -> Result<()> {
        debug!("Initializing {} preallocated Xft colors.", TOTAL_PREALLOC_COLORS);
        self.xft_colors.resize_with(TOTAL_PREALLOC_COLORS, || unsafe { mem::zeroed() });

        // Iterate through the 16 ANSI NamedColor variants to define their RGB values
        // and allocate them.
        for i in 0..ANSI_COLOR_COUNT {
            let named_color = NamedColor::from_index(i as u8); // Get NamedColor enum variant
            let (r_u8, g_u8, b_u8) = match named_color {
                NamedColor::Black         => (0x00, 0x00, 0x00),
                NamedColor::Red           => (0xCD, 0x00, 0x00),
                NamedColor::Green         => (0x00, 0xCD, 0x00),
                NamedColor::Yellow        => (0xCD, 0xCD, 0x00),
                NamedColor::Blue          => (0x00, 0x00, 0xEE),
                NamedColor::Magenta       => (0xCD, 0x00, 0xCD),
                NamedColor::Cyan          => (0x00, 0xCD, 0xCD),
                NamedColor::White         => (0xE5, 0xE5, 0xE5),
                NamedColor::BrightBlack   => (0x7F, 0x7F, 0x7F),
                NamedColor::BrightRed     => (0xFF, 0x00, 0x00),
                NamedColor::BrightGreen   => (0x00, 0xFF, 0x00),
                NamedColor::BrightYellow  => (0xFF, 0xFF, 0x00),
                NamedColor::BrightBlue    => (0x5C, 0x5C, 0xFF),
                NamedColor::BrightMagenta => (0xFF, 0x00, 0xFF),
                NamedColor::BrightCyan    => (0x00, 0xFF, 0xFF),
                NamedColor::BrightWhite   => (0xFF, 0xFF, 0xFF),
            };
            // Convert u8 RGB to u16 for XRenderColor
            let color_components = Rgb16Components {
                r: ((r_u8 as u16) << 8) | (r_u8 as u16),
                g: ((g_u8 as u16) << 8) | (g_u8 as u16),
                b: ((b_u8 as u16) << 8) | (b_u8 as u16),
            };
            self.alloc_specific_xft_color(i, color_components, &format!("ANSI {} ({:?})", i, named_color))?;
        }
        
        // Default Foreground (white)
        let fg_components = Rgb16Components { r: 0xffff, g: 0xffff, b: 0xffff };
        self.alloc_specific_xft_color(DEFAULT_FG_COLOR_IDX, fg_components, "default FG")?;
        
        // Default Background (black)
        let bg_components = Rgb16Components { r: 0x0000, g: 0x0000, b: 0x0000 };
        self.alloc_specific_xft_color(DEFAULT_BG_COLOR_IDX, bg_components, "default BG")?;
        
        debug!("Preallocated Xft colors initialized.");
        Ok(())
    }
    
    /// Helper to allocate a specific XftColor into the `xft_colors` vector at `index`.
    /// Takes `Rgb16Components` for color values.
    fn alloc_specific_xft_color(
        &mut self, 
        index: usize, 
        color_comps: Rgb16Components, 
        name_for_log: &str
    ) -> Result<()> {
        let render_color = XRenderColor { 
            red: color_comps.r, 
            green: color_comps.g, 
            blue: color_comps.b, 
            alpha: XRENDER_ALPHA_OPAQUE 
        };
        // Safety: FFI call. self.display, self.visual, self.colormap must be valid.
        // self.xft_colors[index] must be a valid mutable reference to an XftColor struct.
        if unsafe { xft::XftColorAllocValue(self.display, self.visual, self.colormap, &render_color, &mut self.xft_colors[index]) } == 0 {
            // XftColorAllocValue returns 0 on failure.
            return Err(anyhow::anyhow!("XftColorAllocValue failed for {}", name_for_log));
        }
        trace!("Allocated XftColor for {} (idx {}, pixel: {})", name_for_log, index, self.xft_colors[index].pixel);
        Ok(())
    }

    /// Creates the main application X11 window with specified pixel dimensions.
    fn create_window(&mut self, pixel_width: u16, pixel_height: u16) -> Result<()> {
        // Safety: FFI calls to Xlib for window creation.
        // Assumes self.display, self.screen, self.colormap, self.visual are valid.
        unsafe {
            let root_window = xlib::XRootWindow(self.display, self.screen);
            let border_width = 0; // Typically no border for terminal emulators.

            // Ensure default background color is initialized for the window background.
            if self.xft_colors.len() <= DEFAULT_BG_COLOR_IDX {
                 return Err(anyhow::anyhow!("Default Xft background color not initialized before window creation."));
            }
            let bg_pixel = self.xft_colors[DEFAULT_BG_COLOR_IDX].pixel;

            let mut attributes: xlib::XSetWindowAttributes = mem::zeroed();
            attributes.colormap = self.colormap;
            attributes.background_pixel = bg_pixel; // Window background
            attributes.border_pixel = bg_pixel;     // Border color (though border_width is 0)
            // Event mask for events this window should receive.
            attributes.event_mask = xlib::ExposureMask        // Window needs repainting
                | xlib::KeyPressMask                          // Keyboard input
                | xlib::StructureNotifyMask                   // Resize/move notifications (ConfigureNotify)
                | xlib::FocusChangeMask;                      // FocusIn/FocusOut events
            // TODO: Add ButtonPressMask, ButtonReleaseMask, PointerMotionMask for mouse support.

            self.window = xlib::XCreateWindow(
                self.display,
                root_window,
                0, 0, // x, y position (window manager usually overrides this)
                pixel_width as c_uint,
                pixel_height as c_uint,
                border_width, // Window border width
                xlib::XDefaultDepth(self.display, self.screen), // Window depth
                xlib::InputOutput as c_uint, // Window class
                self.visual,
                // Attribute mask: specifies which fields in 'attributes' struct are being set.
                xlib::CWColormap | xlib::CWBackPixel | xlib::CWBorderPixel | xlib::CWEventMask,
                &mut attributes,
            );
        }
        if self.window == 0 {
            // XCreateWindow returns 0 (None) on failure.
            return Err(anyhow::anyhow!("XCreateWindow failed. Could not create application window."));
        }
        debug!("X window created successfully (ID: {}), initial size: {}x{}px", self.window, pixel_width, pixel_height);
        // Store initial dimensions. These will be updated by ConfigureNotify events.
        self.current_pixel_width = pixel_width;
        self.current_pixel_height = pixel_height;
        Ok(())
    }

    /// Creates a Graphics Context (GC) used for simple drawing operations like clearing.
    fn create_gc(&mut self) -> Result<()> {
        // Safety: FFI call to XCreateGC. self.display and self.window must be valid.
        // No special GC values are needed for a basic clearing GC.
        let gc_values: xlib::XGCValues = unsafe { mem::zeroed() };
        self.clear_gc = unsafe { xlib::XCreateGC(self.display, self.window, 0, &gc_values as *const _ as *mut _) };
        if self.clear_gc.is_null() {
            return Err(anyhow::anyhow!("XCreateGC failed. Could not create graphics context."));
        }
        debug!("Graphics Context (GC) for clearing created.");
        Ok(())
    }

    /// Sets up Window Manager (WM) protocols (e.g., WM_DELETE_WINDOW) and hints (title, size).
    fn setup_wm_protocols_and_hints(&mut self) {
        // Safety: All subsequent calls are FFI to Xlib for WM interaction.
        // Assumes self.display and self.window are valid.
        unsafe {
            // Obtain atoms for WM protocols.
            self.wm_delete_window = xlib::XInternAtom(self.display, b"WM_DELETE_WINDOW\0".as_ptr() as *mut _, xlib::False);
            self.protocols_atom = xlib::XInternAtom(self.display, b"WM_PROTOCOLS\0".as_ptr() as *mut _, xlib::False);
            
            // Register to handle WM_DELETE_WINDOW if the atom was found.
            if self.wm_delete_window != 0 && self.protocols_atom != 0 {
                 xlib::XSetWMProtocols(self.display, self.window, [self.wm_delete_window].as_mut_ptr(), 1);
                 debug!("WM_PROTOCOLS (WM_DELETE_WINDOW) registered with window manager.");
            } else {
                warn!("Failed to get WM_DELETE_WINDOW or WM_PROTOCOLS atom. Close button might not work as expected.");
            }

            // Set window title (WM_NAME and _NET_WM_NAME for modern WMs).
            let title_cstr = CString::new("myterm").expect("CString::new for 'myterm' failed.");
            xlib::XStoreName(self.display, self.window, title_cstr.as_ptr() as *mut c_char);
            
            let net_wm_name_atom = xlib::XInternAtom(self.display, b"_NET_WM_NAME\0".as_ptr() as *mut _, xlib::False);
            let utf8_string_atom = xlib::XInternAtom(self.display, b"UTF8_STRING\0".as_ptr() as *mut _, xlib::False);
            if net_wm_name_atom != 0 && utf8_string_atom != 0 {
                 xlib::XChangeProperty(
                    self.display, self.window, 
                    net_wm_name_atom, utf8_string_atom, 
                    8, // format (8-bit)
                    xlib::PropModeReplace, 
                    title_cstr.as_ptr() as *const u8, 
                    title_cstr.as_bytes().len() as c_int
                );
                 debug!("Window title set to 'myterm' (XStoreName and _NET_WM_NAME).");
            } else {
                debug!("Window title set to 'myterm' (XStoreName only; _NET_WM_NAME or UTF8_STRING atom not found).");
            }

            // Set window size hints (resize increments, minimum size).
            // This tells the WM how the window should be resized in character cell steps.
            let mut size_hints: xlib::XSizeHints = mem::zeroed();
            size_hints.flags = xlib::PResizeInc | xlib::PMinSize; // Specify which fields are set
            size_hints.width_inc = self.font_width as c_int;    // Width step is one char width
            size_hints.height_inc = self.font_height as c_int;  // Height step is one char height
            size_hints.min_width = self.font_width as c_int;    // Min width is one char
            size_hints.min_height = self.font_height as c_int;  // Min height is one char
            xlib::XSetWMNormalHints(self.display, self.window, &mut size_hints);
            debug!("WM size hints set (PResizeInc, PMinSize based on font dimensions).");
        }
    }
    
    /// Resolves an internal `Color` enum to a copy of an `XftColor`.
    /// Uses pre-allocated ANSI/default colors or the RGB cache. Allocates if not found in cache.
    fn resolve_xft_color(&mut self, color: Color) -> Result<xft::XftColor> {
        match color {
            Color::Default => Ok(self.xft_colors[DEFAULT_BG_COLOR_IDX]), // Assuming default means default FG
            Color::Named(named_color) => {
                // Direct lookup for pre-allocated ANSI colors (indices 0-15)
                Ok(self.xft_colors[named_color as u8 as usize])
            }
            Color::Indexed(idx) => {
                // For indexed colors beyond the basic 16 (i.e., idx >= 16),
                // a full 256-color palette lookup and XftColor allocation/caching is needed.
                // This is a simplified placeholder for now.
                if (idx as usize) < ANSI_COLOR_COUNT { // Should have been caught by Color::Named
                     Ok(self.xft_colors[idx as usize])
                } else {
                    warn!("XDriver: Full 256-color palette for Indexed({}) not yet implemented, using default FG.", idx);
                    Ok(self.xft_colors[DEFAULT_FG_COLOR_IDX]) 
                }
            }
            Color::Rgb(r, g, b) => {
                self.cached_rgb_to_xft_color(r,g,b)
            }
        }
    }

    /// Helper to get an `XftColor` for an RGB value, using the cache or allocating a new one.
    /// If allocation is needed, the new `XftColor` is added to `self.xft_color_cache_rgb`.
    fn cached_rgb_to_xft_color(&mut self, r_u8: u8, g_u8: u8, b_u8: u8) -> Result<xft::XftColor> {
        // Check cache first
        if let Some(cached_color) = self.xft_color_cache_rgb.get(&(r_u8, g_u8, b_u8)) {
            return Ok(*cached_color); // Return a copy of the cached XftColor
        }

        // Color not in cache, need to allocate it via Xft.
        // Convert u8 (0-255) to u16 (0-65535) for XRenderColor components.
        // This is done by bit-shifting and ORing to duplicate the 8-bit value into high and low bytes.
        let color_components = Rgb16Components {
            r: ((r_u8 as u16) << 8) | (r_u8 as u16),
            g: ((g_u8 as u16) << 8) | (g_u8 as u16),
            b: ((b_u8 as u16) << 8) | (b_u8 as u16),
        };

        let render_color = XRenderColor { 
            red: color_components.r, 
            green: color_components.g, 
            blue: color_components.b, 
            alpha: XRENDER_ALPHA_OPAQUE
        };
        let mut new_xft_color: xft::XftColor = unsafe { mem::zeroed() };
        
        // Safety: FFI call. self.display, self.visual, self.colormap must be valid.
        if unsafe { xft::XftColorAllocValue(self.display, self.visual, self.colormap, &render_color, &mut new_xft_color) } == 0 {
            // Allocation failed.
            Err(anyhow::anyhow!("XftColorAllocValue failed for RGB({},{},{})", r_u8, g_u8, b_u8))
        } else {
            // Allocation successful, add to cache.
            self.xft_color_cache_rgb.insert((r_u8, g_u8, b_u8), new_xft_color);
            Ok(new_xft_color)
        }
    }
}

impl Drop for XDriver {
    /// Ensures cleanup of X11 resources when the `XDriver` instance goes out of scope.
    /// This delegates to the `cleanup` method, logging any errors encountered during cleanup.
    fn drop(&mut self) {
        info!("Dropping XDriver instance, performing cleanup via self.cleanup().");
        if let Err(e) = self.cleanup() {
            // Log error during drop, as we can't propagate it further.
            error!("Error during XDriver cleanup in drop: {}", e);
        }
    }
}
