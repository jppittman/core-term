// From src/backends/x11.rs

// Imports assumed from the full xbackend.rs file context
use crate::glyph::*;
use super::TerminalBackend;
use crate::Term;
use crate::{DEFAULT_COLS, DEFAULT_ROWS, DEFAULT_WIDTH_PX, DEFAULT_HEIGHT_PX, DEFAULT_FONT_NAME};
use libc::{
    c_int, size_t,
    EPOLLERR, EPOLLIN, EPOLLRDHUP, EPOLLHUP,
};
use std::os::unix::io::{RawFd, AsRawFd};
use std::collections::HashMap;
use std::ffi::CString;
use std::io;
use std::mem;
use std::ptr;
use anyhow::{Context, Result, Error as AnyhowError};
use x11::xlib::{
    self, Display, Window, GC, XEvent, XColor, XFontStruct, KeySym,
    XNextEvent, XConnectionNumber, XPending, XOpenDisplay, XDefaultScreen,
    XDefaultRootWindow, XCreateSimpleWindow, XMapWindow, XSelectInput,
    XDestroyWindow, XCloseDisplay, XBlackPixel, XWhitePixel, XCreateGC, XFreeGC,
    XSetForeground, XSetBackground, XLoadQueryFont, XSetFont, XFlush,
    XFillRectangle, XDrawString, XDrawLine, XAllocColor, XAllocNamedColor,
    XDefaultColormap, DoRed, DoGreen, DoBlue, XFreeColors, XKeyEvent,
    XLookupString, XComposeStatus, ExposureMask, KeyPressMask, StructureNotifyMask,
};

// --- X Backend Implementation ---
pub struct XBackend { /* ... fields as before ... */
    display: *mut Display,
    window: Window,
    x_fd: RawFd,
    gc: GC,
    font_struct: *mut XFontStruct,
    font_width: i32,
    font_height: i32,
    font_ascent: i32,
    color_cache: HashMap<ColorSpec, u64>,
    default_fg_pixel: u64,
    default_bg_pixel: u64,
}

impl XBackend {
     pub fn new() -> Self { 
        XBackend {
             display: ptr::null_mut(), window: 0, x_fd: -1, gc: ptr::null_mut(),
             font_struct: ptr::null_mut(), font_width: 0, font_height: 0, font_ascent: 0,
             color_cache: HashMap::new(), default_fg_pixel: 0, default_bg_pixel: 0,
         }
     }
     unsafe fn alloc_color(&mut self, spec: ColorSpec) -> Result<u64> {
         // 1. Check cache first (early return)
         if let Some(&pixel) = self.color_cache.get(&spec) {
             return Ok(pixel);
         }

         // 2. Determine RGB values based on spec
         let rgb_result: Option<(u8, u8, u8)> = match spec {
             ColorSpec::Default => {
                 // This case ideally shouldn't be hit if defaults are resolved beforehand.
                 eprintln!("Warning: alloc_color called with ColorSpec::Default");
                 None // Indicate failure to resolve RGB for Default spec here
             }
             ColorSpec::Idx(idx) => map_index_to_rgb(idx), // Use helper
             ColorSpec::Rgb(r, g, b) => Some((r, g, b)),
         };

         // 3. Handle cases where RGB couldn't be determined (e.g., invalid index)
         let (r, g, b) = match rgb_result {
             Some(rgb) => rgb,
             None => {
                 // Failed to get RGB (invalid index or Default spec).
                 // Cache and return the fallback pixel.
                 eprintln!("Warning: Could not determine RGB for {:?}, using fallback.", spec);
                 let fallback_pixel = self.default_bg_pixel; // Or another appropriate fallback
                 self.color_cache.insert(spec, fallback_pixel);
                 return Ok(fallback_pixel);
             }
         };

         // 4. Prepare XColor struct
         let cmap = unsafe{XDefaultColormap(self.display, XDefaultScreen(self.display))};
         let mut xcolor = XColor {
             pixel: 0, // Will be filled by XAllocColor
             red: (r as u16) << 8,
             green: (g as u16) << 8,
             blue: (b as u16) << 8,
             flags: DoRed | DoGreen | DoBlue,
             pad: 0,
         };

         // 5. Attempt to allocate the color (early return on failure)
         // SAFETY: XAllocColor is FFI.
         if unsafe{XAllocColor(self.display, cmap, &mut xcolor)} == 0 {
             // Allocation failed
             eprintln!("Warning: Failed to allocate XColor for {:?} (RGB: {},{},{})", spec, r, g, b);
             let fallback_pixel = self.default_bg_pixel; // Use fallback
             self.color_cache.insert(spec, fallback_pixel); // Cache the fallback result
             return Ok(fallback_pixel);
         }

         // 6. Allocation succeeded, cache and return the pixel value
         let pixel = xcolor.pixel;
         self.color_cache.insert(spec, pixel);
         Ok(pixel)
     }

     // --- Private Helper Methods for Event Handling ---
     fn handle_key_press(&mut self, term: &mut Term, event: &XEvent) -> Result<()> {
        // SAFETY: Accessing union field event.key and FFI calls are unsafe.
        unsafe {
            let mut key_event: XKeyEvent = event.key;
            let mut buffer: [u8; 32] = [0; 32];
            let mut keysym: KeySym = 0;
            let mut compose_status: XComposeStatus = mem::zeroed();

            // SAFETY: XLookupString is FFI.
            let count = XLookupString( &mut key_event, buffer.as_mut_ptr() as *mut i8, buffer.len() as c_int, &mut keysym, &mut compose_status );

            if count <= 0 { return Ok(()); } // Nothing to write

            let byte_slice = &buffer[..count as usize];
            // SAFETY: write is FFI.
            if libc::write(term.pty_parent.as_raw_fd(), byte_slice.as_ptr() as *const libc::c_void, count as size_t) < 0 {
                eprintln!("Error writing key press to PTY: {}", io::Error::last_os_error());
            }
        }
        Ok(())
    }

    fn handle_expose(&mut self, term: &Term, event: &XEvent) -> Result<()> {
        // SAFETY: Accessing union field event.expose is unsafe.
        let expose_event = unsafe { event.expose };
        if expose_event.count != 0 { return Ok(()); } // Only redraw on last expose

        // SAFETY: draw involves FFI calls.
        if let Err(e) = self.draw(term) { eprintln!("Error redrawing on expose: {}", e); }
        Ok(())
    }

    fn handle_configure_notify(&mut self, term: &mut Term, event: &XEvent) -> Result<()> {
        // SAFETY: Accessing union field event.configure is unsafe.
        let configure_event = unsafe { event.configure };
        let new_pixel_width = configure_event.width;
        let new_pixel_height = configure_event.height;

        if new_pixel_width <= 0 || new_pixel_height <= 0 { return Ok(()); } // Ignore invalid
        let new_pixel_width = new_pixel_width as usize;
        let new_pixel_height = new_pixel_height as usize;

        let new_cols = if self.font_width > 0 { (new_pixel_width / self.font_width as usize).max(1) } else { term.cols };
        let new_rows = if self.font_height > 0 { (new_pixel_height / self.font_height as usize).max(1) } else { term.rows };

        if new_cols == term.cols && new_rows == term.rows { return Ok(()); } // No change

        // println!( "X ConfigureNotify: Resizing from {}x{} to {}x{} ({}x{} px)", term.cols, term.rows, new_cols, new_rows, new_pixel_width, new_pixel_height ); // Debug
        if let Err(e) = term.resize(new_cols, new_rows) { eprintln!("Error resizing terminal state: {}", e); return Ok(()); }
        // SAFETY: draw involves FFI calls.
        if let Err(e) = self.draw(term) { eprintln!("Error redrawing after resize: {}", e); }
        Ok(())
    }
}

impl TerminalBackend for XBackend {
      fn init(&mut self) -> Result<()> {
         // SAFETY: All Xlib calls are FFI.
         unsafe {
             // Open display connection
             self.display = XOpenDisplay(ptr::null());
             if self.display.is_null() {
                 anyhow::bail!("Failed to open X display");
             }

             // Get screen and root window
             let screen = XDefaultScreen(self.display);
             let root_window = XDefaultRootWindow(self.display);

             // Create a simple window
             self.window = XCreateSimpleWindow(
                 self.display,
                 root_window,
                 0, 0, // x, y
                 DEFAULT_WIDTH_PX as u32, DEFAULT_HEIGHT_PX as u32, // Initial size
                 0, // border_width
                 XBlackPixel(self.display, screen), // border color
                 XWhitePixel(self.display, screen), // background color
             );
             if self.window == 0 {
                 XCloseDisplay(self.display); // Cleanup display connection
                 anyhow::bail!("Failed to create X window");
             }

             // Select input events we care about
             XSelectInput(
                 self.display,
                 self.window,
                 ExposureMask | KeyPressMask | StructureNotifyMask
             );

             // Map the window (make it visible)
             XMapWindow(self.display, self.window);

             // Get the file descriptor for the X connection (for epoll)
             self.x_fd = XConnectionNumber(self.display);
             if self.x_fd < 0 {
                 // Cleanup before erroring
                 XDestroyWindow(self.display, self.window);
                 XCloseDisplay(self.display);
                 return Err(io::Error::last_os_error()).context("Failed to get X connection file descriptor");
             }

             // Load font
             let font_name_cstring = CString::new(DEFAULT_FONT_NAME)?;
             self.font_struct = XLoadQueryFont(self.display, font_name_cstring.as_ptr());
             if self.font_struct.is_null() {
                 // Cleanup before erroring
                 XDestroyWindow(self.display, self.window);
                 XCloseDisplay(self.display);
                 anyhow::bail!("Failed to load font: {}", DEFAULT_FONT_NAME);
             }

             // Calculate and store font metrics
             if !(*self.font_struct).per_char.is_null() { self.font_width = (*self.font_struct).max_bounds.width as i32; }
             else if (*self.font_struct).min_bounds.width == (*self.font_struct).max_bounds.width { self.font_width = (*self.font_struct).max_bounds.width as i32; }
             else { self.font_width = (*self.font_struct).max_bounds.width as i32; } // Fallback width
             self.font_ascent = (*self.font_struct).ascent as i32;
             self.font_height = ((*self.font_struct).ascent + (*self.font_struct).descent) as i32;
             if self.font_width <= 0 || self.font_height <= 0 {
                 // Cleanup before erroring
                 // XFreeFontInfo might be needed here depending on XLoadQueryFont specifics
                 XDestroyWindow(self.display, self.window);
                 XCloseDisplay(self.display);
                 anyhow::bail!("Invalid font dimensions for {}", DEFAULT_FONT_NAME);
             }

             // Create Graphics Context (GC)
             self.gc = XCreateGC(self.display, self.window, 0, ptr::null_mut());
             if self.gc.is_null() {
                 // Cleanup before erroring
                 // XFreeFontInfo might be needed here
                 XDestroyWindow(self.display, self.window);
                 XCloseDisplay(self.display);
                 return Err(io::Error::last_os_error()).context("Failed to create Graphics Context (GC)");
             }

             // Initialize Default Colors and cache them
             let cmap = XDefaultColormap(self.display, screen);
             let mut xcolor_struct = XColor { pixel: 0, red: 0, green: 0, blue: 0, flags: DoRed | DoGreen | DoBlue, pad: 0 };
             // Use X defaults as initial values
             self.default_fg_pixel = XBlackPixel(self.display, screen);
             self.default_bg_pixel = XWhitePixel(self.display, screen);
             // Try to allocate named colors for better theme matching (but ignore errors)
             if XAllocNamedColor(self.display, cmap, b"black\0".as_ptr() as *const i8, &mut xcolor_struct, &mut xcolor_struct) != 0 { self.default_fg_pixel = xcolor_struct.pixel; }
             if XAllocNamedColor(self.display, cmap, b"white\0".as_ptr() as *const i8, &mut xcolor_struct, &mut xcolor_struct) != 0 { self.default_bg_pixel = xcolor_struct.pixel; }
             // Cache the default spec mapping (using FG pixel arbitrarily, could be BG too)
             self.color_cache.insert(ColorSpec::Default, self.default_fg_pixel);

             // Set initial GC state using determined defaults
             XSetForeground(self.display, self.gc, self.default_fg_pixel);
             XSetBackground(self.display, self.gc, self.default_bg_pixel);
             XSetFont(self.display, self.gc, (*self.font_struct).fid);

             // Ensure all setup commands are sent to the X server
             XFlush(self.display);
         }
         println!("X Backend initialized. Window created. Font loaded.");
         Ok(())
     }

     fn get_event_fds(&self) -> Vec<RawFd> { vec![self.x_fd] }

     fn handle_event(&mut self, term: &mut Term, event_fd: RawFd, event_kind: u32) -> Result<bool> {
        // Guard clauses for invalid FD or non-readable/error states
        if event_fd != self.x_fd { return Ok(false); } // Not our FD
        if event_kind & ((EPOLLERR | EPOLLHUP | EPOLLRDHUP) as u32) != 0 {
            eprintln!("Error or hang-up on X connection fd (event kind: 0x{:x}).", event_kind);
            return Ok(true); // Signal exit
        }
        if event_kind & (EPOLLIN as u32) == 0 { return Ok(false); } // Not readable

        let mut needs_redraw = false;
        // Process all pending X events
        // SAFETY: FFI calls within loop.
        unsafe {
            while XPending(self.display) > 0 {
                let mut event: XEvent = mem::zeroed();
                XNextEvent(self.display, &mut event);
                let event_type = event.type_; // Access type safely inside unsafe block

                // Dispatch to helper methods based on event type
                let result = match event_type {
                    xlib::KeyPress => self.handle_key_press(term, &event),
                    xlib::Expose => {
                        // Handle expose event - only mark for redraw on the last one
                        let expose_event = event.expose; // Access within unsafe
                        if expose_event.count == 0 {
                            needs_redraw = true;
                        }
                        Ok(())
                    },
                    xlib::ConfigureNotify => {
                        // Handle resize/move event
                        needs_redraw = true; // Assume redraw needed after configure
                        self.handle_configure_notify(term, &event)
                    },
                    _ => Ok(()), // Ignore other event types for this MVP
                };

                // Log errors from handlers, but don't necessarily exit
                if let Err(e) = result {
                     eprintln!("Error handling X event type {}: {}", event_type, e);
                }
            } // End while XPending
        } // End unsafe block for X event loop

        // Perform redraw *after* processing all pending events if necessary
        if needs_redraw {
             // Draw needs &mut self because alloc_color needs it
             if let Err(e) = self.draw(term) {
                 eprintln!("Error drawing after X events: {}", e);
                 // Decide if this should be fatal - maybe return Ok(true)?
             }
        }

        Ok(false) // Signal main loop to continue
    }
     // ** FIX: Needs &mut self to match trait and allow alloc_color **
     fn draw(&mut self, term: &Term) -> Result<()> {
         // SAFETY: FFI calls.
         unsafe {
             if self.display.is_null() || self.gc.is_null() || self.window == 0 { return Err(AnyhowError::msg("X draw called with invalid state")); }
             for y in 0..term.rows { for x in 0..term.cols {
                 let glyph = term.screen[y][x]; let flags = glyph.attr.flags;
                 let mut eff_fg_spec = glyph.attr.fg; let mut eff_bg_spec = glyph.attr.bg;
                 if flags.contains(AttrFlags::REVERSE) { std::mem::swap(&mut eff_fg_spec, &mut eff_bg_spec); }
                 let fg_pixel = match eff_fg_spec { ColorSpec::Default => self.default_fg_pixel, spec => self.alloc_color(spec).unwrap_or(self.default_fg_pixel) };
                 let bg_pixel = match eff_bg_spec { ColorSpec::Default => self.default_bg_pixel, spec => self.alloc_color(spec).unwrap_or(self.default_bg_pixel) };
                 let dx = (x * self.font_width as usize) as i32; let dy = (y * self.font_height as usize) as i32;
                 XSetForeground(self.display, self.gc, bg_pixel);
                 XFillRectangle(self.display, self.window, self.gc, dx, dy, self.font_width as u32, self.font_height as u32);
                 if glyph.c != ' ' {
                     let mut bytes = [0u8; 4];
                     // ** FIX: Call encode_utf8 on the char field **
                     let slice = glyph.c.encode_utf8(&mut bytes).as_bytes();
                     let draw_y_baseline = dy + self.font_ascent;
                     XSetForeground(self.display, self.gc, fg_pixel);
                     XDrawString( self.display, self.window, self.gc, dx, draw_y_baseline, slice.as_ptr() as *const i8, slice.len() as i32 );
                     if flags.contains(AttrFlags::BOLD) { XDrawString( self.display, self.window, self.gc, dx + 1, draw_y_baseline, slice.as_ptr() as *const i8, slice.len() as i32 ); }
                     if flags.contains(AttrFlags::UNDERLINE) { let line_y = draw_y_baseline + 1; XDrawLine(self.display, self.window, self.gc, dx, line_y, dx + self.font_width - 1, line_y); }
                 }
             } }
             let cx = std::cmp::min(term.cursor_x, term.cols.saturating_sub(1)); let cy = std::cmp::min(term.cursor_y, term.rows.saturating_sub(1));
             let cursor_glyph = term.screen[cy][cx]; let cursor_flags = cursor_glyph.attr.flags;
             let mut cursor_fg_spec = cursor_glyph.attr.fg; let mut cursor_bg_spec = cursor_glyph.attr.bg;
             if cursor_flags.contains(AttrFlags::REVERSE) { std::mem::swap(&mut cursor_fg_spec, &mut cursor_bg_spec); }
             let cursor_fg_pixel = match cursor_fg_spec { ColorSpec::Default => self.default_fg_pixel, spec => self.alloc_color(spec).unwrap_or(self.default_fg_pixel) };
             let cursor_bg_pixel = match cursor_bg_spec { ColorSpec::Default => self.default_bg_pixel, spec => self.alloc_color(spec).unwrap_or(self.default_bg_pixel) };
             let cx_px = (cx * self.font_width as usize) as i32; let cy_px = (cy * self.font_height as usize) as i32;
             XSetForeground(self.display, self.gc, cursor_fg_pixel);
             XFillRectangle(self.display, self.window, self.gc, cx_px, cy_px, self.font_width as u32, self.font_height as u32);
             if cursor_glyph.c != ' ' {
                 let mut bytes = [0u8; 4];
                 // ** FIX: Call encode_utf8 on the char field **
                 let slice = cursor_glyph.c.encode_utf8(&mut bytes).as_bytes();
                 let draw_y_baseline = cy_px + self.font_ascent;
                 XSetForeground(self.display, self.gc, cursor_bg_pixel);
                 XDrawString( self.display, self.window, self.gc, cx_px, draw_y_baseline, slice.as_ptr() as *const i8, slice.len() as i32 );
             }
             XFlush(self.display);
         }
         Ok(())
      }
     fn get_dimensions(&self) -> (usize, usize) { /* ... implementation as before ... */ (DEFAULT_COLS, DEFAULT_ROWS) }
}

impl Drop for XBackend {
     fn drop(&mut self) {
        // SAFETY: FFI calls.
        unsafe {
            if !self.display.is_null() {
                if !self.color_cache.is_empty() {
                    let cmap = XDefaultColormap(self.display, XDefaultScreen(self.display));
                    let mut pixels: Vec<u64> = self.color_cache.values().copied().collect();
                     if !pixels.is_empty() {
                         // **FIX: Cast pointer mutability**
                         XFreeColors(self.display, cmap, pixels.as_mut_ptr(), pixels.len() as c_int, 0);
                     }
                }
                if !self.gc.is_null() { XFreeGC(self.display, self.gc); }
                if self.window != 0 { XDestroyWindow(self.display, self.window); }
                XCloseDisplay(self.display);
            }
        }
    }
}
