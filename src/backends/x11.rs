// src/backends/x11.rs

#![allow(non_snake_case)] // Allow non-snake case for X11 types

use crate::term::Term;
use crate::glyph::{Color, AttrFlags};
use crate::backends::{TerminalBackend, BackendEvent};

use anyhow::{Context, Result};
use std::ffi::CString; // Removed CStr import
use std::os::unix::io::RawFd;
use std::ptr;
use std::mem;
use std::cmp::min;

use libc::{c_char, c_int, c_uint, c_ulong, winsize, TIOCSWINSZ};
use x11::xlib;
use x11::keysym;

// --- Constants ---
const DEFAULT_FONT_NAME: &str = "monospace:size=10";
const MIN_FONT_WIDTH: u32 = 5;
const MIN_FONT_HEIGHT: u32 = 5;
const DRAW_BUFFER_SIZE: usize = 4096;
const DEFAULT_FG_IDX: usize = 256;
const DEFAULT_BG_IDX: usize = 257;
const TOTAL_COLOR_COUNT: usize = DEFAULT_BG_IDX + 1;

/// X11 backend implementation for the terminal.
pub struct XBackend {
    display: *mut xlib::Display,
    screen: c_int,
    window: xlib::Window,
    gc: xlib::GC,
    colormap: xlib::Colormap,
    font: *mut xlib::XFontStruct,
    font_width: u32,
    font_height: u32,
    font_ascent: u32,
    colors: [xlib::XColor; TOTAL_COLOR_COUNT],
    wm_delete_window: xlib::Atom,
    protocols_atom: xlib::Atom,
}


impl TerminalBackend for XBackend {
    fn new(width: usize, height: usize) -> Result<Self> {
        let display = unsafe { xlib::XOpenDisplay(ptr::null()) };
        if display.is_null() {
            return Err(anyhow::anyhow!("Failed to open X display"));
        }

        let mut backend = XBackend {
            display,
            screen: unsafe { xlib::XDefaultScreen(display) },
            window: 0,
            gc: ptr::null_mut(),
            colormap: 0,
            font: ptr::null_mut(),
            font_width: 0,
            font_height: 0,
            font_ascent: 0,
            colors: unsafe { mem::zeroed() },
            wm_delete_window: 0,
            protocols_atom: 0,
        };

        // Safety: FFI calls
        unsafe {
            backend.colormap = xlib::XDefaultColormap(backend.display, backend.screen);
            backend.protocols_atom = xlib::XInternAtom(backend.display, b"WM_PROTOCOLS\0".as_ptr() as *mut _, xlib::False);
            backend.wm_delete_window = xlib::XInternAtom(backend.display, b"WM_DELETE_WINDOW\0".as_ptr() as *mut _, xlib::False);
        }

        backend.load_font().context("Failed to load font")?;
        backend.init_colors().context("Failed to initialize colors")?;
        backend.create_window(width, height).context("Failed to create window")?;
        backend.create_gc().context("Failed to create graphics context")?;
        backend.setup_wm();
        backend.setup_input();

        unsafe {
            xlib::XMapWindow(backend.display, backend.window);
            xlib::XFlush(backend.display);
        }

        Ok(backend)
    }

    /// Runs the X11 event loop, translating native events to `BackendEvent`s.
    fn run(&mut self, term: &mut Term, pty_fd: RawFd) -> Result<bool> {
        let mut event: xlib::XEvent = unsafe { mem::zeroed() };
        let mut key_text_buffer: [u8; 32] = [0; 32]; // Use u8 for UTF-8

        loop {
            // Safety: FFI call
            unsafe { xlib::XNextEvent(self.display, &mut event) };

            let mut backend_event: Option<BackendEvent> = None;
            let mut should_exit = false;

            // Safety: Accessing event.type_ requires unsafe block
            let event_type = unsafe { event.type_ };
            match event_type {
                xlib::Expose => {
                    // Safety: Accessing union field
                    let xexpose = unsafe { event.expose };
                    if xexpose.count == 0 {
                        self.draw(term).context("Failed during draw on expose")?;
                    }
                }
                xlib::ConfigureNotify => {
                    // Safety: Accessing union field
                    let xconfigure = unsafe { event.configure };
                    backend_event = Some(BackendEvent::Resize {
                        width_px: xconfigure.width as u16,
                        height_px: xconfigure.height as u16,
                    });
                }
                xlib::KeyPress => {
                    let mut keysym: xlib::KeySym = 0;
                    // Safety: FFI call and accessing union field requires unsafe block
                    // Pass the buffer as *mut c_char.
                    let count = unsafe {
                         xlib::XLookupString(&mut event.key, key_text_buffer.as_mut_ptr() as *mut c_char, key_text_buffer.len() as c_int, &mut keysym, ptr::null_mut())
                    };

                    // Use the count to correctly interpret the buffer
                    let text = if count > 0 {
                        // Safety: We assume XLookupString writes valid UTF-8 or compatible bytes within the count.
                        // Using from_utf8 avoids panic on invalid sequences.
                        // Use unwrap_or as a fallback, though invalid UTF-8 here would be unexpected.
                        std::str::from_utf8(&key_text_buffer[0..count as usize]).unwrap_or("")
                    } else {
                        ""
                    };

                    backend_event = Some(BackendEvent::Key {
                        keysym: keysym as u32,
                        text,
                    });
                }
                xlib::ClientMessage => {
                    // Safety: Accessing union field requires unsafe block
                    let xclient = unsafe { event.client_message }; // Use client_message field
                    if xclient.message_type == self.protocols_atom && xclient.data.get_long(0) as xlib::Atom == self.wm_delete_window {
                        should_exit = true;
                    }
                }
                 xlib::FocusIn => { backend_event = Some(BackendEvent::FocusGained); }
                 xlib::FocusOut => { backend_event = Some(BackendEvent::FocusLost); }
                _ => {}
            }

            if let Some(be) = backend_event {
                 // Clone the text slice to own the String for the event
                 let event_to_handle = match be {
                     BackendEvent::Key { keysym, text } => BackendEvent::Key { keysym, text },
                     other => other,
                 };
                self.handle_event(event_to_handle, term, pty_fd).context("Failed handling backend event")?;
            }

            if should_exit {
                return Ok(true);
            }
        }
    }

    /// Handles translated backend events.
    fn handle_event(&mut self, event: BackendEvent, term: &mut Term, pty_fd: RawFd) -> Result<()> {
        match event {
            BackendEvent::Key { text, keysym } => {
                let bytes_to_write: &[u8] = if !text.is_empty() && keysym != keysym::XK_BackSpace {
                    text.as_bytes()
                } else {
                    // Map special keysyms to escape sequences
                    match keysym {
                        keysym::XK_Return => b"\r",
                        keysym::XK_Left => b"\x1b[D",
                        keysym::XK_Right => b"\x1b[C",
                        keysym::XK_Up => b"\x1b[A",
                        keysym::XK_Down => b"\x1b[B",
                        keysym::XK_BackSpace => b"\x08", // Backspace
                        keysym::XK_Tab => b"\t",
                        keysym::XK_ISO_Left_Tab => b"\x1b[Z", // Shift+Tab
                        keysym::XK_Home => b"\x1b[H",
                        keysym::XK_End => b"\x1b[F",
                        keysym::XK_Page_Up => b"\x1b[5~",
                        keysym::XK_Page_Down => b"\x1b[6~",
                        keysym::XK_Delete => b"\x1b[3~",
                        keysym::XK_Insert => b"\x1b[2~",
                        // Add more mappings as needed
                        _ => b"", // Ignore other unmapped special keys
                    }
                };

                if bytes_to_write.is_empty() { return Ok(()); }

                let count = bytes_to_write.len();
                // Safety: FFI call to libc::write. Requires pointer cast.
                let written = unsafe { libc::write(pty_fd, bytes_to_write.as_ptr() as *const libc::c_void, count) };

                if written < 0 {
                    // Handle potential write error
                    return Err(anyhow::Error::from(std::io::Error::last_os_error())
                        .context("X11Backend: Failed to write key event to PTY"));
                }
                if (written as usize) != count {
                    // Handle partial write if necessary, though less common for TTYs
                    eprintln!("X11Backend: Warning: Partial write to PTY ({} out of {})", written, count);
                }
            }
            BackendEvent::Resize { width_px, height_px } => {
                // Avoid division by zero if font hasn't loaded yet
                if self.font_width == 0 || self.font_height == 0 { return Ok(()); }

                // Calculate new dimensions in character cells
                let new_width = (width_px as u32 / self.font_width).max(1) as usize;
                let new_height = (height_px as u32 / self.font_height).max(1) as usize;

                let (current_width, current_height) = term.get_dimensions();
                // Only resize if dimensions actually changed
                if new_width == current_width && new_height == current_height { return Ok(()); }

                // Update terminal state
                term.resize(new_width, new_height);

                // Inform the PTY slave about the new size
                let winsz = winsize { ws_row: new_height as u16, ws_col: new_width as u16, ws_xpixel: width_px, ws_ypixel: height_px };
                // Safety: FFI call to libc::ioctl
                if unsafe { libc::ioctl(pty_fd, TIOCSWINSZ, &winsz) } < 0 {
                    // Log error but don't necessarily fail the whole operation
                    eprintln!("X11Backend: Warning: ioctl(TIOCSWINSZ) failed: {}", std::io::Error::last_os_error());
                }

                // Redraw the terminal content
                self.draw(term)?;
            }
            BackendEvent::CloseRequested => {
                // This event is handled in the run loop to trigger exit
            }
            BackendEvent::FocusGained => {
                // Redraw on focus gain (e.g., to show focused cursor style)
                self.draw(term)?;
            }
            BackendEvent::FocusLost => {
                // Redraw on focus lost (e.g., to show unfocused cursor style)
                self.draw(term)?;
            }
        }
        Ok(())
    }

    /// Renders the terminal state to the X11 window.
    fn draw(&mut self, term: &Term) -> Result<()> {
        let (term_width, term_height) = term.get_dimensions();
        let mut draw_buffer: [c_char; DRAW_BUFFER_SIZE] = [0; DRAW_BUFFER_SIZE];
        let mut buffer_idx: usize = 0;
        let mut current_fg = self.colors[DEFAULT_FG_IDX].pixel;
        let mut current_bg = self.colors[DEFAULT_BG_IDX].pixel;
        let mut current_flags = AttrFlags::empty();
        let mut run_start_x = 0;
        let mut run_start_y = 0;

        // Safety: FFI calls for drawing. Assumes display/gc valid.
        unsafe {
            // Clear background
            xlib::XSetForeground(self.display, self.gc, self.colors[DEFAULT_BG_IDX].pixel);
            xlib::XFillRectangle(self.display, self.window, self.gc, 0, 0, term_width as u32 * self.font_width, term_height as u32 * self.font_height);

            // Set initial drawing colors
            xlib::XSetForeground(self.display, self.gc, current_fg);
            xlib::XSetBackground(self.display, self.gc, current_bg);

            // Iterate through each cell of the terminal grid
            for y in 0..term_height {
                for x in 0..term_width {
                    // Get glyph info, defaulting if out of bounds (shouldn't happen here)
                    let glyph = term.get_glyph(x, y).cloned().unwrap_or_default();
                    let flags = glyph.attr.flags;

                    // Determine effective foreground and background based on attributes
                    let (fg_pixel, bg_pixel) = self.resolve_colors(glyph.attr.fg, glyph.attr.bg); // Contains unsafe alloc_color
                    let (effective_fg, effective_bg) = if flags.contains(AttrFlags::REVERSE) {
                        (bg_pixel, fg_pixel) // Swap fg/bg for reverse video
                    } else {
                        (fg_pixel, bg_pixel)
                    };

                    // Check if attributes or position changed, requiring a buffer flush
                    let attr_changed = effective_fg != current_fg || effective_bg != current_bg || flags != current_flags;
                    // Position changed if it's the start of a line or not contiguous with the buffer
                    let position_changed = x == 0 || x != run_start_x + buffer_idx;

                    // Flush buffer if attributes/position changed, buffer is full, or end of line
                    if buffer_idx > 0 && (attr_changed || position_changed || buffer_idx >= DRAW_BUFFER_SIZE - 1 || x == term_width) {
                        self.flush_draw_buffer(run_start_x, run_start_y, &draw_buffer, buffer_idx)?; // Unsafe internally
                        buffer_idx = 0; // Reset buffer index
                    }

                    // Update drawing context if attributes changed
                    if attr_changed {
                        xlib::XSetForeground(self.display, self.gc, effective_fg);
                        xlib::XSetBackground(self.display, self.gc, effective_bg);
                        current_fg = effective_fg;
                        current_bg = effective_bg;
                        current_flags = flags;
                    }

                    // Start a new run if buffer is empty
                    if buffer_idx == 0 {
                        run_start_x = x;
                        run_start_y = y;
                    }

                    // Add character to buffer or draw individually
                    if glyph.c.is_ascii() && glyph.c != '\0' {
                        // Add ASCII char to buffer
                        draw_buffer[buffer_idx] = glyph.c as c_char;
                        buffer_idx += 1;
                    } else if glyph.c != '\0' {
                        // Flush buffer before drawing non-ASCII char individually
                        if buffer_idx > 0 {
                            self.flush_draw_buffer(run_start_x, run_start_y, &draw_buffer, buffer_idx)?;
                            buffer_idx = 0;
                        }
                        self.draw_single_char(x, y, glyph.c)?; // Unsafe internally
                        // Start next potential run after this char
                        run_start_x = x + 1;
                        run_start_y = y;
                    } else {
                        // Handle null/empty glyph (advance run start if needed)
                         if buffer_idx == 0 {
                             run_start_x = x + 1;
                             run_start_y = y;
                         }
                    }

                    // Draw underline/strikethrough lines if needed
                    let line_y_base = ((y as u32 + 1) * self.font_height) as c_int;
                    if flags.contains(AttrFlags::UNDERLINE) {
                        xlib::XDrawLine(self.display, self.window, self.gc,
                                        (x as u32 * self.font_width) as c_int, line_y_base - 1,
                                        ((x + 1) as u32 * self.font_width) as c_int -1, line_y_base - 1);
                    }
                    if flags.contains(AttrFlags::STRIKETHROUGH) {
                         let strike_y = line_y_base - (self.font_height / 2) as c_int;
                         xlib::XDrawLine(self.display, self.window, self.gc,
                                         (x as u32 * self.font_width) as c_int, strike_y,
                                         ((x + 1) as u32 * self.font_width) as c_int -1, strike_y);
                    }
                }
                // Flush any remaining buffer at the end of the line
                if buffer_idx > 0 {
                    self.flush_draw_buffer(run_start_x, run_start_y, &draw_buffer, buffer_idx)?;
                    buffer_idx = 0;
                }
            }

            // Draw the cursor
            self.draw_cursor(term)?; // Unsafe internally

            // Flush all drawing commands to the X server
            xlib::XFlush(self.display);
        }
        Ok(())
    }

    /// Cleans up X11 resources.
    fn cleanup(&mut self) -> Result<()> {
        // Safety: FFI calls to free X resources. Checks ensure we don't double-free.
        unsafe {
            if !self.font.is_null() { xlib::XFreeFont(self.display, self.font); self.font = ptr::null_mut(); }
            if !self.gc.is_null() { xlib::XFreeGC(self.display, self.gc); self.gc = ptr::null_mut(); }
            if self.window != 0 { xlib::XDestroyWindow(self.display, self.window); self.window = 0; }
            if !self.display.is_null() { xlib::XCloseDisplay(self.display); self.display = ptr::null_mut(); }
        }
        Ok(())
    }
}

// Private helper methods for XBackend
impl XBackend {
    /// Loads the specified font. Contains unsafe FFI calls.
    fn load_font(&mut self) -> Result<()> {
        let font_name = CString::new(DEFAULT_FONT_NAME).unwrap_or_default();
        // Safety: FFI call
        self.font = unsafe { xlib::XLoadQueryFont(self.display, font_name.as_ptr()) };
        if self.font.is_null() { anyhow::bail!("Failed to load font: {}", DEFAULT_FONT_NAME); }
        // Safety: Dereferencing raw font pointer, assumed valid after check above.
        unsafe {
            self.font_width = (*self.font).max_bounds.width as u32;
            self.font_height = ((*self.font).ascent + (*self.font).descent) as u32;
            self.font_ascent = (*self.font).ascent as u32;
        }
        if self.font_width < MIN_FONT_WIDTH || self.font_height < MIN_FONT_HEIGHT { anyhow::bail!("Font dimensions too small ({}x{})", self.font_width, self.font_height); }
        Ok(())
    }

    /// Initializes default colors. Contains unsafe FFI calls.
    fn init_colors(&mut self) -> Result<()> {
        // Safety: FFI calls to alloc_color helper
        unsafe {
            Self::alloc_color(self.display, self.colormap, &mut self.colors[DEFAULT_FG_IDX], 0xffff, 0xffff, 0xffff).context("Failed to allocate default foreground color")?;
            Self::alloc_color(self.display, self.colormap, &mut self.colors[DEFAULT_BG_IDX], 0x0000, 0x0000, 0x0000).context("Failed to allocate default background color")?;
        }
        Ok(())
    }

    /// Creates the main application window. Contains unsafe FFI calls.
    fn create_window(&mut self, initial_width_chars: usize, initial_height_chars: usize) -> Result<()> {
        // Safety: FFI calls
        unsafe {
            let root = xlib::XRootWindow(self.display, self.screen);
            let black = xlib::XBlackPixel(self.display, self.screen);
            let white = xlib::XWhitePixel(self.display, self.screen);
            let initial_pixel_width = (initial_width_chars as u32 * self.font_width) as c_uint;
            let initial_pixel_height = (initial_height_chars as u32 * self.font_height) as c_uint;
            self.window = xlib::XCreateSimpleWindow(self.display, root, 0, 0, initial_pixel_width, initial_pixel_height, 0, black, white);
        }
        Ok(())
    }

     /// Creates the Graphics Context (GC). Contains unsafe FFI calls.
     fn create_gc(&mut self) -> Result<()> {
        // Safety: FFI calls
        unsafe {
            self.gc = xlib::XCreateGC(self.display, self.window, 0, ptr::null_mut());
            if self.gc.is_null() { anyhow::bail!("Failed to create Graphics Context"); }
            if !self.font.is_null() { xlib::XSetFont(self.display, self.gc, (*self.font).fid); }
        }
        Ok(())
     }

    /// Sets up Window Manager hints and protocols. Contains unsafe FFI calls.
    fn setup_wm(&mut self) {
        // Safety: FFI calls and struct manipulation for FFI
        unsafe {
            let mut size_hints: xlib::XSizeHints = mem::zeroed();
            size_hints.flags = xlib::PResizeInc | xlib::PMinSize;
            size_hints.width_inc = self.font_width as c_int;
            size_hints.height_inc = self.font_height as c_int;
            size_hints.min_width = self.font_width as c_int;
            size_hints.min_height = self.font_height as c_int;
            xlib::XSetWMNormalHints(self.display, self.window, &mut size_hints);
            // Pass pointer directly
            xlib::XSetWMProtocols(self.display, self.window, [self.wm_delete_window].as_mut_ptr(), 1);
            let title = CString::new("myterm").unwrap_or_default();
            xlib::XStoreName(self.display, self.window, title.as_ptr() as *mut c_char);
        }
    }

    /// Sets up input event selection. Contains unsafe FFI calls.
    fn setup_input(&mut self) {
        // Safety: FFI call
        unsafe {
            xlib::XSelectInput(self.display, self.window, xlib::ExposureMask | xlib::KeyPressMask | xlib::StructureNotifyMask | xlib::FocusChangeMask);
        }
    }

    /// Allocates a specific color. Marked unsafe as it's an FFI call wrapper.
    unsafe fn alloc_color(display: *mut xlib::Display, colormap: xlib::Colormap, color: &mut xlib::XColor, r: u16, g: u16, b: u16) -> Result<()> {
        color.red = r; color.green = g; color.blue = b;
        color.flags = xlib::DoRed | xlib::DoGreen | xlib::DoBlue;
        // Safety: FFI call required by caller
        if unsafe { xlib::XAllocColor(display, colormap, color) } == 0 {
            anyhow::bail!("Failed to allocate color ({}, {}, {})", r, g, b);
        }
        Ok(())
    }

    /// Resolves internal `Color` enum to X11 pixel values. Contains unsafe calls.
    fn resolve_colors(&self, fg: Color, bg: Color) -> (c_ulong, c_ulong) {
        // Simplified: Needs proper color allocation/caching
        let fg_pixel = match fg {
            Color::Idx(idx) => if (idx as usize) < TOTAL_COLOR_COUNT { self.colors[idx as usize].pixel } else { self.colors[DEFAULT_FG_IDX].pixel },
            Color::Rgb(r, g, b) => {
                 let mut temp_color: xlib::XColor = unsafe { mem::zeroed() };
                 // Safety: FFI wrapper call
                 if unsafe { Self::alloc_color(self.display, self.colormap, &mut temp_color, (r as u16) * 257, (g as u16) * 257, (b as u16) * 257).is_ok() } { temp_color.pixel }
                 else { self.colors[DEFAULT_FG_IDX].pixel }
            },
            Color::Default => self.colors[DEFAULT_FG_IDX].pixel,
        };
        let bg_pixel = match bg {
             Color::Idx(idx) => if (idx as usize) < TOTAL_COLOR_COUNT { self.colors[idx as usize].pixel } else { self.colors[DEFAULT_BG_IDX].pixel },
             Color::Rgb(r, g, b) => {
                  let mut temp_color: xlib::XColor = unsafe { mem::zeroed() };
                  // Safety: FFI wrapper call
                  if unsafe { Self::alloc_color(self.display, self.colormap, &mut temp_color, (r as u16) * 257, (g as u16) * 257, (b as u16) * 257).is_ok() } { temp_color.pixel }
                  else { self.colors[DEFAULT_BG_IDX].pixel }
             },
            Color::Default => self.colors[DEFAULT_BG_IDX].pixel,
        };
        (fg_pixel, bg_pixel)
    }

     /// Helper to flush the drawing buffer. Marked unsafe as it calls XDrawString.
     unsafe fn flush_draw_buffer(&self, x: usize, y: usize, buffer: &[c_char], count: usize) -> Result<()> {
         if count == 0 { return Ok(()); }
         let baseline_y = ((y as u32 + 1) * self.font_height - (self.font_height - self.font_ascent)) as c_int;
         let start_x = (x as u32 * self.font_width) as c_int;
         // Safety: FFI call required by caller
         unsafe { xlib::XDrawString(self.display, self.window, self.gc, start_x, baseline_y, buffer.as_ptr(), count as c_int); }
         Ok(())
     }

      /// Helper to draw a single char. Marked unsafe as it calls XDrawString.
      unsafe fn draw_single_char(&self, x: usize, y: usize, c: char) -> Result<()> {
          let s = c.to_string();
          let c_str = CString::new(s).unwrap_or_default();
          let baseline_y = ((y as u32 + 1) * self.font_height - (self.font_height - self.font_ascent)) as c_int;
          let start_x = (x as u32 * self.font_width) as c_int;
          // Safety: FFI call required by caller
          unsafe { xlib::XDrawString(self.display, self.window, self.gc, start_x, baseline_y, c_str.as_ptr(), c_str.as_bytes().len() as c_int); }
          Ok(())
      }

      /// Draws the terminal cursor. Marked unsafe as it calls XDrawImageString.
      unsafe fn draw_cursor(&self, term: &Term) -> Result<()> {
            let (cursor_x, cursor_y) = term.get_cursor();
            let (term_width, term_height) = term.get_dimensions();
            let cx = min(cursor_x, term_width.saturating_sub(1));
            let cy = min(cursor_y, term_height.saturating_sub(1));
            let cursor_glyph = term.get_glyph(cx, cy).cloned().unwrap_or_default();
            let (glyph_fg_pixel, glyph_bg_pixel) = self.resolve_colors(cursor_glyph.attr.fg, cursor_glyph.attr.bg); // Unsafe internally
            let cursor_draw_fg = glyph_bg_pixel;
            let cursor_draw_bg = glyph_fg_pixel;

            // Safety: FFI calls required by caller
            unsafe {
                xlib::XSetForeground(self.display, self.gc, cursor_draw_fg);
                xlib::XSetBackground(self.display, self.gc, cursor_draw_bg);
            }

            let cursor_x_px = (cx as u32 * self.font_width) as c_int;
            let baseline_y = ((cy as u32 + 1) * self.font_height - (self.font_height - self.font_ascent)) as c_int;
            let cursor_char_str = cursor_glyph.c.to_string();
            let c_str = CString::new(cursor_char_str).unwrap_or_default();

            // Safety: FFI call required by caller
            unsafe { xlib::XDrawImageString(self.display, self.window, self.gc, cursor_x_px, baseline_y, c_str.as_ptr(), c_str.as_bytes().len() as c_int); }
            Ok(())
      }
}


impl Drop for XBackend {
    fn drop(&mut self) {
        let _ = self.cleanup();
    }
}

