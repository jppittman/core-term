// src/backends/x11.rs

#![allow(non_snake_case)] // Allow non-snake case for X11 types

// Import logging macros
use log::{debug, info, warn, error, trace};

use crate::term::Term;
use crate::glyph::{Color, AttrFlags};
use crate::backends::{TerminalBackend, BackendEvent};

use anyhow::{Context, Result};
use std::ffi::CString;
use std::io; // Added for io::Error
use std::os::unix::io::RawFd;
use std::ptr;
use std::mem;
use std::cmp::min;
use std::collections::HashMap;

// Added libc constants and types for epoll and read/write
use libc::{
    c_char, c_int, c_uint, c_ulong, winsize, TIOCSWINSZ,
    epoll_create1, epoll_ctl, epoll_wait, epoll_event, // epoll functions and struct
    EPOLLIN, EPOLL_CTL_ADD, EPOLL_CTL_DEL, // epoll constants
    read, // read function
    EINTR, // close function (needed for epoll fd)
    fcntl, F_GETFL, F_SETFL, O_NONBLOCK, // For setting PTY to non-blocking (optional but good practice)
};
use x11::xlib;
use x11::keysym;
use x11::xft;
use x11::xrender::{XGlyphInfo, XRenderColor};

// --- Constants ---
const DEFAULT_FONT_NAME: &str = "Liberation Mono:size=10";
const MIN_FONT_WIDTH: u32 = 5;
const MIN_FONT_HEIGHT: u32 = 5;
const DEFAULT_FG_IDX: usize = 256;
const DEFAULT_BG_IDX: usize = 257;
const TOTAL_COLOR_COUNT: usize = DEFAULT_BG_IDX + 1;
const PTY_READ_BUF_SIZE: usize = 4096; // Buffer size for reading PTY output
const MAX_EPOLL_EVENTS: usize = 2; // Max events to handle per epoll_wait call

/// X11 backend implementation for the terminal using Xft.
pub struct XBackend {
    display: *mut xlib::Display,
    screen: c_int,
    window: xlib::Window,
    colormap: xlib::Colormap,
    visual: *mut xlib::Visual,
    xft_font: *mut xft::XftFont,
    xft_draw: *mut xft::XftDraw,
    xft_colors: Vec<xft::XftColor>,
    xft_color_cache_rgb: HashMap<(u8, u8, u8), xft::XftColor>,
    font_width: u32,
    font_height: u32,
    font_ascent: u32,
    wm_delete_window: xlib::Atom,
    protocols_atom: xlib::Atom,
    clear_gc: xlib::GC,
    epoll_fd: RawFd, // Added field for epoll instance
}


impl TerminalBackend for XBackend {
    /// Creates and initializes a new X11 backend instance using Xft.
    fn new(width: usize, height: usize) -> Result<Self> {
        info!("Creating new XBackend ({}x{})", width, height);
        // Safety: FFI call to connect to X server.
        let display = unsafe { xlib::XOpenDisplay(ptr::null()) };
        if display.is_null() {
            error!("Failed to open X display");
            return Err(anyhow::anyhow!("Failed to open X display"));
        }
        debug!("X display opened successfully");

        let screen = unsafe { xlib::XDefaultScreen(display) };
        let colormap = unsafe { xlib::XDefaultColormap(display, screen) };
        let visual = unsafe { xlib::XDefaultVisual(display, screen) };

        // Create epoll instance *before* initializing backend struct
        // Safety: FFI call to epoll_create1.
        let epoll_fd = unsafe { epoll_create1(libc::EPOLL_CLOEXEC) };
        if epoll_fd < 0 {
            let err = io::Error::last_os_error();
            error!("Failed to create epoll instance: {}", err);
            // Close display connection if epoll fails early
            unsafe { xlib::XCloseDisplay(display) };
            return Err(anyhow::Error::from(err).context("Failed to create epoll instance"));
        }
        debug!("Epoll instance created: fd={}", epoll_fd);

        let mut backend = XBackend {
            display,
            screen,
            window: 0,
            colormap,
            visual,
            xft_font: ptr::null_mut(),
            xft_draw: ptr::null_mut(),
            xft_colors: Vec::with_capacity(TOTAL_COLOR_COUNT),
            xft_color_cache_rgb: HashMap::new(),
            font_width: 0,
            font_height: 0,
            font_ascent: 0,
            wm_delete_window: 0,
            protocols_atom: 0,
            clear_gc: ptr::null_mut(),
            epoll_fd, // Store the epoll fd
        };

        // Use try block or chain results to ensure cleanup on intermediate errors
        if let Err(e) = (|| { // Start of IIFE-like closure
            backend.load_font().context("Failed to load font")?;
            backend.init_colors().context("Failed to initialize colors")?;
            backend.create_window(width, height).context("Failed to create window")?;
            backend.create_gc().context("Failed to create graphics context")?;

            // Safety: FFI call assuming valid display, visual, colormap, window.
            backend.xft_draw = unsafe { xft::XftDrawCreate(backend.display, backend.window, backend.visual, backend.colormap) };
            if backend.xft_draw.is_null() {
                return Err(anyhow::anyhow!("Failed to create XftDraw"));
            }
            debug!("XftDraw created");

            backend.setup_wm();
            backend.setup_input();

            // Safety: FFI calls assuming valid display and window.
            unsafe {
                xlib::XMapWindow(backend.display, backend.window);
                xlib::XFlush(backend.display);
                debug!("Window mapped and flushed");
            }
            Ok(()) // Indicate success within the closure
        })() { // Call the closure immediately
            error!("Error during XBackend setup: {:?}", e);
            let _ = backend.cleanup(); // Ensure cleanup runs if setup fails
            return Err(e);
        }


        info!("XBackend initialization complete");
        Ok(backend)
    }

    /// Runs the event loop using epoll, handling X11 and PTY events.
    fn run(&mut self, term: &mut Term, pty_fd: RawFd) -> Result<bool> {
        info!("Starting XBackend event loop with epoll...");

        // Get the X11 connection file descriptor
        // Safety: FFI call, assumes display is valid.
        let x11_fd = unsafe { xlib::XConnectionNumber(self.display) };
        debug!("X11 fd: {}, PTY fd: {}", x11_fd, pty_fd);

        // Add X11 fd to epoll set
        let mut x11_event = epoll_event { events: EPOLLIN as u32, u64: x11_fd as u64 };
        // Safety: FFI call to epoll_ctl.
        if unsafe { epoll_ctl(self.epoll_fd, EPOLL_CTL_ADD, x11_fd, &mut x11_event) } < 0 {
            let err = io::Error::last_os_error();
            error!("Failed to add X11 fd ({}) to epoll: {}", x11_fd, err);
            return Err(anyhow::Error::from(err).context("Failed to add X11 fd to epoll"));
        }
        debug!("Added X11 fd ({}) to epoll", x11_fd);

        // Add PTY fd to epoll set
        let mut pty_event = epoll_event { events: EPOLLIN as u32, u64: pty_fd as u64 };
        // Safety: FFI call to epoll_ctl.
        if unsafe { epoll_ctl(self.epoll_fd, EPOLL_CTL_ADD, pty_fd, &mut pty_event) } < 0 {
            let err = io::Error::last_os_error();
            error!("Failed to add PTY fd ({}) to epoll: {}", pty_fd, err);
            // Attempt to remove X11 fd before returning error
            unsafe { epoll_ctl(self.epoll_fd, EPOLL_CTL_DEL, x11_fd, ptr::null_mut()) };
            return Err(anyhow::Error::from(err).context("Failed to add PTY fd to epoll"));
        }
        debug!("Added PTY fd ({}) to epoll", pty_fd);

        // Optional: Set PTY to non-blocking.
        unsafe {
            let flags = fcntl(pty_fd, F_GETFL, 0);
            if flags != -1 {
                if fcntl(pty_fd, F_SETFL, flags | O_NONBLOCK) == -1 {
                    warn!("Failed to set PTY fd to non-blocking: {}", io::Error::last_os_error());
                } else {
                    debug!("Set PTY fd ({}) to non-blocking", pty_fd);
                }
            } else {
                warn!("Failed to get PTY fd flags: {}", io::Error::last_os_error());
            }
        }

        let mut events: [epoll_event; MAX_EPOLL_EVENTS] = unsafe { mem::zeroed() };
        let mut pty_buffer = [0u8; PTY_READ_BUF_SIZE];

        loop {
            // trace!("Calling epoll_wait (timeout -1)..."); // Too verbose
            // Safety: FFI call to epoll_wait.
            let nfds = unsafe { epoll_wait(self.epoll_fd, events.as_mut_ptr(), MAX_EPOLL_EVENTS as c_int, -1) };

            if nfds < 0 {
                let err = io::Error::last_os_error();
                if err.raw_os_error() == Some(EINTR) {
                    trace!("epoll_wait interrupted, continuing");
                    continue; // Interrupted by signal, try again
                } else {
                    error!("epoll_wait failed: {}", err);
                    return Err(anyhow::Error::from(err).context("epoll_wait failed"));
                }
            }
            // trace!("epoll_wait returned {} events", nfds); // Too verbose

            for i in 0..nfds as usize {
                let current_event_fd = events[i].u64 as RawFd;

                if current_event_fd == x11_fd {
                    // trace!("Processing X11 events (fd={})", x11_fd); // Too verbose
                    // Process all pending X events non-blockingly
                    while unsafe { xlib::XPending(self.display) } > 0 {
                        let mut event: xlib::XEvent = unsafe { mem::zeroed() };
                        unsafe { xlib::XNextEvent(self.display, &mut event) };

                        let event_type = unsafe { event.type_ };
                        // trace!("Received XEvent type: {}", event_type); // Too verbose

                        let mut backend_event: Option<BackendEvent> = None;
                        let mut should_exit = false;

                        match event_type {
                            xlib::Expose => {
                                // debug!("Received Expose event"); // Can be noisy
                                let xexpose = unsafe { event.expose };
                                if xexpose.count == 0 {
                                    // Redraw immediately on expose
                                    if let Err(e) = self.draw(term) {
                                         error!("Error during expose redraw: {:?}", e);
                                     }
                                } else {
                                    // trace!("Ignoring Expose event with count > 0 ({})", xexpose.count); // Too verbose
                                }
                            }
                            xlib::ConfigureNotify => {
                                debug!("Received ConfigureNotify event");
                                let xconfigure = unsafe { event.configure };
                                backend_event = Some(BackendEvent::Resize {
                                    width_px: xconfigure.width as u16,
                                    height_px: xconfigure.height as u16,
                                });
                            }
                            xlib::KeyPress => {
                                debug!("Received KeyPress event");
                                let mut keysym: xlib::KeySym = 0;
                                let mut key_text_buffer: [u8; 32] = [0; 32];
                                let count = unsafe {
                                    xlib::XLookupString(&mut event.key, key_text_buffer.as_mut_ptr() as *mut c_char, key_text_buffer.len() as c_int, &mut keysym, ptr::null_mut())
                                };
                                let text = if count > 0 {
                                    String::from_utf8_lossy(&key_text_buffer[0..count as usize]).to_string()
                                } else {
                                    String::new()
                                };
                                // trace!("KeyPress details: keysym={}, text='{}'", keysym, text); // Moved to handle_event
                                backend_event = Some(BackendEvent::Key {
                                    keysym: keysym as u32,
                                    text,
                                });
                            }
                            xlib::ClientMessage => {
                                debug!("Received ClientMessage event");
                                let xclient = unsafe { event.client_message };
                                if xclient.message_type == self.protocols_atom && xclient.data.get_long(0) as xlib::Atom == self.wm_delete_window {
                                    info!("WM_DELETE_WINDOW received, requesting exit");
                                    should_exit = true;
                                } else {
                                    // trace!("Ignoring ClientMessage type: {}", xclient.message_type); // Too verbose
                                }
                            }
                            xlib::FocusIn => {
                                debug!("Received FocusIn event");
                                backend_event = Some(BackendEvent::FocusGained);
                            }
                            xlib::FocusOut => {
                                debug!("Received FocusOut event");
                                backend_event = Some(BackendEvent::FocusLost);
                            }
                             _ => {
                                // trace!("Ignoring XEvent type: {}", event_type); // Too verbose
                            }
                        }

                        if let Some(be) = backend_event {
                            // trace!("Handling BackendEvent from X11: {:?}", be); // Too verbose
                            if let Err(e) = self.handle_event(be, term, pty_fd) {
                                error!("Error handling BackendEvent: {:?}", e);
                            }
                        }

                        if should_exit {
                            info!("Exiting XBackend event loop due to WM_DELETE_WINDOW.");
                            return Ok(true);
                        }
                    } // End while XPending
                } else if current_event_fd == pty_fd {
                    // trace!("Processing PTY input (fd={})", pty_fd); // Too verbose
                    let bytes_read = unsafe { read(pty_fd, pty_buffer.as_mut_ptr() as *mut _, PTY_READ_BUF_SIZE) };

                    if bytes_read < 0 {
                        let err = io::Error::last_os_error();
                        if err.raw_os_error() == Some(libc::EAGAIN) || err.raw_os_error() == Some(libc::EWOULDBLOCK) {
                             // trace!("PTY read would block, continuing"); // Too verbose
                             continue;
                        } else if err.raw_os_error() == Some(EINTR) {
                            trace!("PTY read interrupted, continuing");
                            continue; // Interrupted by signal, loop again
                        } else {
                            error!("Error reading from PTY fd ({}): {}", pty_fd, err);
                            return Err(anyhow::Error::from(err).context("Error reading from PTY"));
                        }
                    } else if bytes_read == 0 {
                        info!("PTY fd ({}) closed (EOF), exiting.", pty_fd);
                        return Ok(true); // PTY closed, likely shell exited
                    } else {
                        // Debug level log moved to Term::process_bytes
                        term.process_bytes(&pty_buffer[0..bytes_read as usize]);
                        if let Err(e) = self.draw(term) {
                            error!("Error during PTY read redraw: {:?}", e);
                        }
                    }
                } else {
                    warn!("epoll_wait returned event for unknown fd: {}", current_event_fd);
                }
            } // End for nfds

        } // End loop
    }


    /// Handles translated backend events.
    fn handle_event(&mut self, event: BackendEvent, term: &mut Term, pty_fd: RawFd) -> Result<()> {
        // debug!("Handling BackendEvent: {:?}", event); // Can be noisy
        let mut needs_redraw = false;
        match event {
            BackendEvent::Key { text, keysym } => {
                trace!("handle_event Key: keysym={}, text='{}'", keysym, text);

                let bytes_to_write: &[u8] = if !text.is_empty() && keysym != keysym::XK_BackSpace {
                    trace!("  -> Using text bytes: {:?}", text.as_bytes());
                    text.as_bytes()
                } else {
                    let seq = match keysym {
                        keysym::XK_Return => b"\r" as &[u8],
                        keysym::XK_Left => b"\x1b[D",
                        keysym::XK_Right => b"\x1b[C",
                        keysym::XK_Up => b"\x1b[A",
                        keysym::XK_Down => b"\x1b[B",
                        keysym::XK_BackSpace => b"\x08",
                        keysym::XK_Tab => b"\t",
                        keysym::XK_ISO_Left_Tab => b"\x1b[Z",
                        keysym::XK_Home => b"\x1b[H",
                        keysym::XK_End => b"\x1b[F",
                        keysym::XK_Page_Up => b"\x1b[5~",
                        keysym::XK_Page_Down => b"\x1b[6~",
                        keysym::XK_Delete => b"\x1b[3~",
                        keysym::XK_Insert => b"\x1b[2~",
                        _ => b"",
                    };
                    trace!("  -> Using sequence bytes for keysym {}: {:?}", keysym, seq);
                    seq
                };

                if !bytes_to_write.is_empty() {
                    trace!("  -> Writing to PTY: {:?}", bytes_to_write);
                    let count = bytes_to_write.len();
                    let written = unsafe { libc::write(pty_fd, bytes_to_write.as_ptr() as *const libc::c_void, count) };

                    if written < 0 {
                        let err = std::io::Error::last_os_error();
                        error!("Failed to write key event to PTY: {}", err);
                        return Err(anyhow::Error::from(err)
                            .context("X11Backend: Failed to write key event to PTY"));
                    }
                    if (written as usize) != count {
                        warn!("Partial write to PTY ({} out of {})", written, count);
                    }
                } else {
                     // trace!("No bytes to write for keysym {}", keysym); // Too verbose
                }
            }
            BackendEvent::Resize { width_px, height_px } => {
                debug!("Handling Resize event: {}x{} px", width_px, height_px);
                if self.font_width == 0 || self.font_height == 0 {
                    warn!("Resize event received before font dimensions are known, skipping.");
                    return Ok(());
                 }

                let new_width = (width_px as u32 / self.font_width).max(1) as usize;
                let new_height = (height_px as u32 / self.font_height).max(1) as usize;
                debug!("Calculated new dimensions: {}x{} cells", new_width, new_height);

                let (current_width, current_height) = term.get_dimensions();
                if new_width == current_width && new_height == current_height {
                    // trace!("Resize resulted in same dimensions, skipping term resize."); // Too verbose
                    return Ok(());
                 }

                term.resize(new_width, new_height);
                info!("Resized terminal to {}x{}", new_width, new_height);

                let winsz = winsize { ws_row: new_height as u16, ws_col: new_width as u16, ws_xpixel: width_px, ws_ypixel: height_px };
                if unsafe { libc::ioctl(pty_fd, TIOCSWINSZ, &winsz) } < 0 {
                    warn!("ioctl(TIOCSWINSZ) failed: {}", std::io::Error::last_os_error());
                } else {
                     debug!("ioctl(TIOCSWINSZ) successful for {}x{}", new_width, new_height);
                }

                needs_redraw = true;
            }
            BackendEvent::CloseRequested => {
                info!("CloseRequested event handled (action taken in run loop).");
            }
            BackendEvent::FocusGained => {
                info!("FocusGained event handled, marking for redraw.");
                needs_redraw = true;
            }
            BackendEvent::FocusLost => {
                info!("FocusLost event handled, marking for redraw.");
                 needs_redraw = true;
            }
        }
        if needs_redraw {
             self.draw(term)?;
        }
        Ok(())
    }

    /// Renders the terminal state to the display using Xft.
    fn draw(&mut self, term: &Term) -> Result<()> {
        // trace!("draw() called"); // Too verbose
        let (term_width, term_height) = term.get_dimensions();
        let font_width = self.font_width;
        let font_height = self.font_height;
        let font_ascent = self.font_ascent;

        if self.xft_colors.is_empty() || DEFAULT_BG_IDX >= self.xft_colors.len() {
             error!("Attempted to draw before Xft colors were initialized or defaults are out of bounds.");
             return Err(anyhow::anyhow!("Xft colors not initialized"));
        }
        let default_bg_color_pixel = self.xft_colors[DEFAULT_BG_IDX].pixel;
        // trace!("Drawing {}x{} cells", term_width, term_height); // Too verbose


        unsafe {
            // trace!("Clearing background"); // Too verbose
            xlib::XSetForeground(self.display, self.clear_gc, default_bg_color_pixel);
            xlib::XFillRectangle(self.display, self.window, self.clear_gc, 0, 0, term_width as u32 * font_width, term_height as u32 * font_height);

            // trace!("Starting cell drawing loop"); // Too verbose
            for y in 0..term_height {
                for x in 0..term_width {
                    let glyph = term.get_glyph(x, y).cloned().unwrap_or_default();
                    let flags = glyph.attr.flags;
                    // trace!("Processing cell ({}, {}): char='{}', flags={:?}", x, y, glyph.c, flags); // Too verbose

                    if glyph.c == ' ' && flags == AttrFlags::empty() && glyph.attr.fg == Color::Default && glyph.attr.bg == Color::Default {
                         // trace!("Skipping empty default cell ({}, {})", x, y); // Too verbose
                        continue;
                    }

                    let (fg_color, bg_color) = self.resolve_xft_colors(glyph.attr.fg, glyph.attr.bg)?;
                    let (effective_fg, effective_bg) = if flags.contains(AttrFlags::REVERSE) {
                        (bg_color, fg_color)
                    } else {
                        (fg_color, bg_color)
                    };
                    // trace!("Cell ({}, {}): Effective FG pixel={}, BG pixel={}", x, y, effective_fg.pixel, effective_bg.pixel); // Too verbose

                    let cell_x = (x as u32 * font_width) as c_int;
                    let cell_y = (y as u32 * font_height) as c_int;

                    if effective_bg.pixel != default_bg_color_pixel {
                         // trace!("Drawing background rect for cell ({}, {})", x, y); // Too verbose
                        xft::XftDrawRect(self.xft_draw, &effective_bg, cell_x, cell_y, font_width, font_height);
                    }

                    if glyph.c != ' ' && glyph.c != '\0' {
                         // trace!("Drawing char '{}' for cell ({}, {})", glyph.c, x, y); // Too verbose
                        let char_str = glyph.c.to_string();
                        let c_str = CString::new(char_str).unwrap_or_default();
                        let baseline_y = (cell_y as u32 + font_ascent) as c_int;

                        xft::XftDrawStringUtf8(self.xft_draw, &effective_fg, self.xft_font,
                                               cell_x, baseline_y,
                                               c_str.as_ptr() as *const u8, c_str.as_bytes().len() as c_int);
                    }

                    let line_y_base = cell_y + font_height as c_int;
                    if flags.contains(AttrFlags::UNDERLINE) {
                         // trace!("Drawing underline for cell ({}, {})", x, y); // Too verbose
                         xft::XftDrawRect(self.xft_draw, &effective_fg, cell_x, line_y_base - 1, font_width, 1);
                    }
                    if flags.contains(AttrFlags::STRIKETHROUGH) {
                         // trace!("Drawing strikethrough for cell ({}, {})", x, y); // Too verbose
                         let strike_y = cell_y + (font_ascent / 2) as c_int;
                         xft::XftDrawRect(self.xft_draw, &effective_fg, cell_x, strike_y, font_width, 1);
                    }
                }
            }
            // trace!("Finished cell drawing loop"); // Too verbose

            // trace!("Drawing cursor"); // Too verbose
            self.draw_cursor_xft(term)?;

            // trace!("Flushing X display"); // Too verbose
            xlib::XFlush(self.display);
        }
        // trace!("draw() finished"); // Too verbose
        Ok(())
    }


    /// Cleans up X11 and Xft resources.
    fn cleanup(&mut self) -> Result<()> {
        info!("Cleaning up XBackend resources...");
        unsafe {
            if self.epoll_fd >= 0 {
                 trace!("Closing epoll fd: {}", self.epoll_fd);
                 if libc::close(self.epoll_fd) == -1 {
                     warn!("Error closing epoll fd {}: {}", self.epoll_fd, io::Error::last_os_error());
                 }
                 self.epoll_fd = -1;
            }
            if !self.xft_font.is_null() {
                trace!("Closing Xft font");
                xft::XftFontClose(self.display, self.xft_font);
                self.xft_font = ptr::null_mut();
            }
            if !self.xft_draw.is_null() {
                trace!("Destroying Xft draw");
                xft::XftDrawDestroy(self.xft_draw);
                self.xft_draw = ptr::null_mut();
            }
            trace!("Freeing {} Xft colors", self.xft_colors.len());
            for color in self.xft_colors.iter() {
                xft::XftColorFree(self.display, self.visual, self.colormap, color as *const _ as *mut _);
            }
            self.xft_colors.clear();
             trace!("Freeing {} cached RGB Xft colors", self.xft_color_cache_rgb.len());
            for (_, color) in self.xft_color_cache_rgb.drain() {
                 xft::XftColorFree(self.display, self.visual, self.colormap, &color as *const _ as *mut _);
            }
            self.xft_color_cache_rgb.clear();

            if !self.clear_gc.is_null() {
                trace!("Freeing clear GC");
                 xlib::XFreeGC(self.display, self.clear_gc);
                 self.clear_gc = ptr::null_mut();
            }
            if self.window != 0 {
                trace!("Destroying window");
                xlib::XDestroyWindow(self.display, self.window);
                self.window = 0;
             }
            if !self.display.is_null() {
                trace!("Closing display connection");
                xlib::XCloseDisplay(self.display);
                self.display = ptr::null_mut();
             }
        }
        info!("XBackend cleanup complete.");
        Ok(())
    }
}

// Private helper methods for XBackend
impl XBackend {
    /// Loads the specified font using Xft. Contains unsafe FFI calls.
    fn load_font(&mut self) -> Result<()> {
        debug!("Loading font: {}", DEFAULT_FONT_NAME);
        let font_name = CString::new(DEFAULT_FONT_NAME)?;
        self.xft_font = unsafe { xft::XftFontOpenName(self.display, self.screen, font_name.as_ptr()) };
        if self.xft_font.is_null() {
             error!("Xft: Failed to load font: {}", DEFAULT_FONT_NAME);
            anyhow::bail!("Xft: Failed to load font: {}", DEFAULT_FONT_NAME);
        }
        debug!("Font loaded successfully");

        let font_info = unsafe { *self.xft_font };
        self.font_height = (font_info.ascent + font_info.descent) as u32;
        self.font_ascent = font_info.ascent as u32;
        trace!("Font metrics: height={}, ascent={}", self.font_height, self.font_ascent);

        let mut extents: XGlyphInfo = unsafe { mem::zeroed() };
        let sample_char = CString::new("M")?;
        unsafe {
            xft::XftTextExtentsUtf8(self.display, self.xft_font, sample_char.as_ptr() as *const u8, sample_char.as_bytes().len() as c_int, &mut extents);
        }
        self.font_width = extents.xOff as u32;
        trace!("Font metrics: width={}", self.font_width);

        if self.font_width < MIN_FONT_WIDTH || self.font_height < MIN_FONT_HEIGHT {
             error!("Font dimensions too small ({}x{})", self.font_width, self.font_height);
            anyhow::bail!("Font dimensions too small ({}x{})", self.font_width, self.font_height);
        }

        Ok(())
    }

    /// Initializes default Xft colors and populates the basic color cache.
    fn init_colors(&mut self) -> Result<()> {
        debug!("Initializing {} Xft colors", TOTAL_COLOR_COUNT);
        self.xft_colors.resize(TOTAL_COLOR_COUNT, unsafe { mem::zeroed() });

        unsafe {
            self.alloc_xft_color_value(DEFAULT_FG_IDX, 0xffff, 0xffff, 0xffff)
                .context("Failed to allocate default Xft foreground color")?;
             trace!("Allocated default FG color (idx {})", DEFAULT_FG_IDX);
            self.alloc_xft_color_value(DEFAULT_BG_IDX, 0x0000, 0x0000, 0x0000)
                .context("Failed to allocate default Xft background color")?;
             trace!("Allocated default BG color (idx {})", DEFAULT_BG_IDX);
        }
        debug!("Default colors initialized");
        Ok(())
    }

    /// Creates the main application window. Contains unsafe FFI calls.
    fn create_window(&mut self, initial_width_chars: usize, initial_height_chars: usize) -> Result<()> {
        debug!("Creating window ({}x{} chars)", initial_width_chars, initial_height_chars);
        unsafe {
            let root = xlib::XRootWindow(self.display, self.screen);
            if self.xft_colors.is_empty() || DEFAULT_BG_IDX >= self.xft_colors.len() {
                 error!("Attempted to create window before Xft colors were initialized.");
                 return Err(anyhow::anyhow!("Xft colors not initialized for window creation"));
            }
            let bg_pixel = self.xft_colors[DEFAULT_BG_IDX].pixel;

            let initial_pixel_width = (initial_width_chars as u32 * self.font_width) as c_uint;
            let initial_pixel_height = (initial_height_chars as u32 * self.font_height) as c_uint;
             debug!("Initial window pixel size: {}x{}", initial_pixel_width, initial_pixel_height);

            let mut attributes: xlib::XSetWindowAttributes = mem::zeroed();
            attributes.colormap = self.colormap;
            attributes.background_pixel = bg_pixel;
            attributes.border_pixel = bg_pixel;
            attributes.event_mask = xlib::ExposureMask | xlib::KeyPressMask | xlib::StructureNotifyMask | xlib::FocusChangeMask;
             trace!("Window attributes set: event_mask={}", attributes.event_mask);

            self.window = xlib::XCreateWindow(
                self.display, root,
                0, 0,
                initial_pixel_width, initial_pixel_height,
                0,
                xlib::XDefaultDepth(self.display, self.screen),
                xlib::InputOutput as c_uint,
                self.visual,
                xlib::CWColormap | xlib::CWBackPixel | xlib::CWBorderPixel | xlib::CWEventMask,
                &mut attributes
            );
        }
        if self.window == 0 {
             error!("Failed to create X window");
            anyhow::bail!("Failed to create X window");
        }
         debug!("Window created successfully: ID={}", self.window);
        Ok(())
    }

     /// Creates the Graphics Context (GC) for clearing. Contains unsafe FFI calls.
     fn create_gc(&mut self) -> Result<()> {
         debug!("Creating graphics context (GC)");
        unsafe {
            let gc_values: xlib::XGCValues = mem::zeroed();
            self.clear_gc = xlib::XCreateGC(self.display, self.window, 0, &gc_values as *const _ as *mut _);
            if self.clear_gc.is_null() {
                 error!("Failed to create Graphics Context");
                 anyhow::bail!("Failed to create Graphics Context");
             }
        }
         debug!("GC created successfully");
        Ok(())
     }

    /// Sets up Window Manager hints and protocols. Contains unsafe FFI calls.
    fn setup_wm(&mut self) {
         debug!("Setting up window manager hints and protocols");
        unsafe {
             self.wm_delete_window = xlib::XInternAtom(self.display, b"WM_DELETE_WINDOW\0".as_ptr() as *mut _, xlib::False);
             self.protocols_atom = xlib::XInternAtom(self.display, b"WM_PROTOCOLS\0".as_ptr() as *mut _, xlib::False);

            let mut size_hints: xlib::XSizeHints = mem::zeroed();
            size_hints.flags = xlib::PResizeInc | xlib::PMinSize;
            size_hints.width_inc = self.font_width as c_int;
            size_hints.height_inc = self.font_height as c_int;
            size_hints.min_width = self.font_width as c_int;
            size_hints.min_height = self.font_height as c_int;
            xlib::XSetWMNormalHints(self.display, self.window, &mut size_hints);
             trace!("Set WM normal hints");
            xlib::XSetWMProtocols(self.display, self.window, [self.wm_delete_window].as_mut_ptr(), 1);
             trace!("Set WM protocols (WM_DELETE_WINDOW)");
            let title = CString::new("myterm").unwrap_or_default();
            xlib::XStoreName(self.display, self.window, title.as_ptr() as *mut c_char);
             trace!("Set WM_NAME");
            let utf8_string_atom = xlib::XInternAtom(self.display, b"UTF8_STRING\0".as_ptr() as *mut _, xlib::False);
            xlib::XSetTextProperty(self.display, self.window,
                                   &mut xlib::XTextProperty {
                                       value: title.as_ptr() as *mut u8,
                                       encoding: utf8_string_atom,
                                       format: 8,
                                       nitems: title.as_bytes().len() as c_ulong,
                                   },
                                   xlib::XInternAtom(self.display, b"_NET_WM_NAME\0".as_ptr() as *mut _, xlib::False));
             trace!("Set _NET_WM_NAME");

        }
         debug!("WM setup complete");
    }

    /// Sets up input event selection. Contains unsafe FFI calls.
    fn setup_input(&mut self) {
         debug!("Setting up input event mask");
        unsafe {
            xlib::XSelectInput(self.display, self.window,
                               xlib::ExposureMask | xlib::KeyPressMask | xlib::StructureNotifyMask | xlib::FocusChangeMask | xlib::ButtonPressMask | xlib::ButtonReleaseMask | xlib::PointerMotionMask );
        }
         trace!("Input mask set");
    }

    /// Allocates a specific Xft color by value. Marked unsafe as it's an FFI call wrapper.
    unsafe fn alloc_xft_color_value(&mut self, index: usize, r: u16, g: u16, b: u16) -> Result<()> {
         trace!("Allocating Xft color value: index={}, r={}, g={}, b={}", index, r, g, b);
        let color = XRenderColor { red: r, green: g, blue: b, alpha: 0xffff };
        if unsafe { xft::XftColorAllocValue(self.display, self.visual, self.colormap, &color, &mut self.xft_colors[index]) } == 0 {
             error!("Xft: Failed to allocate color value ({}, {}, {})", r, g, b);
            anyhow::bail!("Xft: Failed to allocate color value ({}, {}, {})", r, g, b);
        }
         trace!("Xft color value allocated successfully for index {}", index);
        Ok(())
    }

     /// Allocates a specific Xft color by name. Marked unsafe as it's an FFI call wrapper.
     #[allow(dead_code)] // This function is currently unused
     unsafe fn alloc_xft_color_name(&mut self, name: &str) -> Result<xft::XftColor> {
         debug!("Allocating Xft color by name: '{}'", name);
        let c_name = CString::new(name)?;
        let mut color: xft::XftColor = unsafe { mem::zeroed() };
        if unsafe { xft::XftColorAllocName(self.display, self.visual, self.colormap, c_name.as_ptr(), &mut color) } == 0 {
              error!("Xft: Failed to allocate color name '{}'", name);
             anyhow::bail!("Xft: Failed to allocate color name '{}'", name);
        }
         debug!("Xft color '{}' allocated successfully", name);
        Ok(color)
     }


    /// Resolves internal `Color` enum to XftColor structs (owned). Uses cache.
    fn resolve_xft_colors(&mut self, fg: Color, bg: Color) -> Result<(xft::XftColor, xft::XftColor)> {
         // trace!("Resolving colors: fg={:?}, bg={:?}", fg, bg); // Too verbose
         if self.xft_colors.is_empty() || DEFAULT_FG_IDX >= self.xft_colors.len() || DEFAULT_BG_IDX >= self.xft_colors.len() {
              error!("Attempted to resolve colors before Xft colors were initialized or defaults are out of bounds.");
              return Err(anyhow::anyhow!("Xft colors not initialized for resolving"));
         }

        let fg_color = match fg {
            Color::Default => self.xft_colors[DEFAULT_FG_IDX],
            Color::Idx(idx) => {
                warn!("resolve_xft_colors: Unimplemented FG color index: {}", idx);
                self.xft_colors[DEFAULT_FG_IDX]
            }
            Color::Rgb(r, g, b) => {
                if let Some(cached_color) = self.xft_color_cache_rgb.get(&(r, g, b)) {
                    // trace!("Found cached XftColor for RGB: ({}, {}, {})", r, g, b); // Too verbose
                    *cached_color
                } else {
                    trace!("Allocating new XftColor for RGB: ({}, {}, {})", r, g, b);
                    let render_color = XRenderColor {
                        red: (r as u16) << 8 | r as u16,
                        green: (g as u16) << 8 | g as u16,
                        blue: (b as u16) << 8 | b as u16,
                        alpha: 0xffff,
                    };
                    let mut xft_color: xft::XftColor = unsafe { mem::zeroed() };
                    if unsafe { xft::XftColorAllocValue(self.display, self.visual, self.colormap, &render_color, &mut xft_color) } == 0 {
                        error!("Xft: Failed to allocate RGB color value ({}, {}, {})", r, g, b);
                        self.xft_colors[DEFAULT_FG_IDX]
                    } else {
                        trace!("Successfully allocated and cached RGB: ({}, {}, {}) -> pixel {}", r, g, b, xft_color.pixel);
                        self.xft_color_cache_rgb.insert((r, g, b), xft_color);
                        xft_color
                    }
                }
            }
        };

         let bg_color = match bg {
             Color::Default => self.xft_colors[DEFAULT_BG_IDX],
             Color::Idx(idx) => {
                 warn!("resolve_xft_colors: Unimplemented BG color index: {}", idx);
                 self.xft_colors[DEFAULT_BG_IDX]
             }
             Color::Rgb(r, g, b) => {
                 if let Some(cached_color) = self.xft_color_cache_rgb.get(&(r, g, b)) {
                    // trace!("Found cached XftColor for RGB BG: ({}, {}, {})", r, g, b); // Too verbose
                    *cached_color
                 } else {
                     trace!("Allocating new XftColor for RGB BG: ({}, {}, {})", r, g, b);
                     let render_color = XRenderColor {
                         red: (r as u16) << 8 | r as u16,
                         green: (g as u16) << 8 | g as u16,
                         blue: (b as u16) << 8 | b as u16,
                         alpha: 0xffff,
                     };
                     let mut xft_color: xft::XftColor = unsafe { mem::zeroed() };
                     if unsafe { xft::XftColorAllocValue(self.display, self.visual, self.colormap, &render_color, &mut xft_color) } == 0 {
                         error!("Xft: Failed to allocate RGB BG color value ({}, {}, {})", r, g, b);
                         self.xft_colors[DEFAULT_BG_IDX]
                     } else {
                         trace!("Successfully allocated and cached RGB BG: ({}, {}, {}) -> pixel {}", r, g, b, xft_color.pixel);
                         self.xft_color_cache_rgb.insert((r, g, b), xft_color);
                         xft_color
                     }
                 }
             }
         };

         // trace!("Resolved colors: fg_pixel={}, bg_pixel={}", fg_color.pixel, bg_color.pixel); // Too verbose
         Ok((fg_color, bg_color))
    }


     /// Draws the terminal cursor using Xft. Marked unsafe as it calls XftDrawStringUtf8.
     unsafe fn draw_cursor_xft(&mut self, term: &Term) -> Result<()> {
             // trace!("draw_cursor_xft() called"); // Too verbose
            let (cursor_x, cursor_y) = term.get_cursor();
            let (term_width, term_height) = term.get_dimensions();
            let cx = min(cursor_x, term_width.saturating_sub(1));
            let cy = min(cursor_y, term_height.saturating_sub(1));
             // trace!("Cursor position: ({}, {}) (clamped from {}, {})", cx, cy, cursor_x, cursor_y); // Too verbose

            let cursor_glyph = term.get_glyph(cx, cy).cloned().unwrap_or_default();
             // trace!("Cursor glyph: char='{}', attr={:?}", cursor_glyph.c, cursor_glyph.attr); // Too verbose

            let (glyph_fg_color, glyph_bg_color) = self.resolve_xft_colors(cursor_glyph.attr.fg, cursor_glyph.attr.bg)?;
            let cursor_draw_fg = glyph_bg_color;
            let cursor_draw_bg = glyph_fg_color;
             // trace!("Cursor draw colors: FG pixel={}, BG pixel={}", cursor_draw_fg.pixel, cursor_draw_bg.pixel); // Too verbose

            let font_width = self.font_width;
            let font_height = self.font_height;
            let font_ascent = self.font_ascent;

            let cell_x = (cx as u32 * font_width) as c_int;
            let cell_y = (cy as u32 * font_height) as c_int;

            unsafe {
                 // trace!("Drawing cursor background rect"); // Too verbose
                xft::XftDrawRect(self.xft_draw, &cursor_draw_bg, cell_x, cell_y, font_width, font_height);

                if cursor_glyph.c != ' ' && cursor_glyph.c != '\0' {
                     // trace!("Drawing cursor char '{}'", cursor_glyph.c); // Too verbose
                    let cursor_char_str = cursor_glyph.c.to_string();
                    let c_str = CString::new(cursor_char_str).unwrap_or_default();
                    let baseline_y = (cell_y as u32 + font_ascent) as c_int;

                    xft::XftDrawStringUtf8(self.xft_draw, &cursor_draw_fg, self.xft_font,
                                           cell_x, baseline_y,
                                           c_str.as_ptr() as *const u8, c_str.as_bytes().len() as c_int);
                }
            }
             // trace!("draw_cursor_xft() finished"); // Too verbose
            Ok(())
      }

}


impl Drop for XBackend {
    fn drop(&mut self) {
        info!("Dropping XBackend instance");
        if let Err(e) = self.cleanup() {
            error!("Error during XBackend cleanup: {}", e);
        }
    }
}

// Unit Tests for XBackend
#[cfg(test)]
mod tests {
    use super::*;
    use crate::term::Term;
    use crate::backends::BackendEvent;
    use std::os::unix::io::RawFd;
    use std::thread;
    use std::time::Duration;
    use std::io::{self};

    // Helper to create a mock PTY pair for testing writes
    fn create_mock_pty() -> (RawFd, RawFd) {
        let mut pipe_fds = [-1; 2];
        if unsafe { libc::pipe(pipe_fds.as_mut_ptr()) } == -1 {
            panic!("Failed to create pipe for mock PTY: {}", io::Error::last_os_error());
        }
        (pipe_fds[1], pipe_fds[0])
    }

    // Helper to read bytes written to the mock PTY slave end
    fn read_from_mock_pty(read_fd: RawFd, max_bytes: usize) -> Vec<u8> {
        let mut buffer = vec![0u8; max_bytes];
        match unsafe { libc::read(read_fd, buffer.as_mut_ptr() as *mut libc::c_void, max_bytes) } {
            -1 => {
                let err = io::Error::last_os_error();
                if err.kind() == io::ErrorKind::WouldBlock {
                    Vec::new()
                } else {
                    panic!("Failed to read from mock PTY: {}", err);
                }
            },
            n if n >= 0 => {
                buffer.truncate(n as usize);
                buffer
            },
            _ => unreachable!(),
        }
    }

    // Helper to set pipe to non-blocking
    fn set_nonblocking(fd: RawFd) {
        let flags = unsafe { libc::fcntl(fd, libc::F_GETFL, 0) };
        if flags == -1 {
            panic!("fcntl(F_GETFL) failed: {}", io::Error::last_os_error());
        }
        if unsafe { libc::fcntl(fd, libc::F_SETFL, flags | libc::O_NONBLOCK) } == -1 {
            panic!("fcntl(F_SETFL, O_NONBLOCK) failed: {}", io::Error::last_os_error());
        }
    }

    // Test key event handling for basic text
    #[test]
    fn test_handle_event_key_text() {
        let (pty_fd, read_fd) = create_mock_pty();
        set_nonblocking(read_fd);

        let mut term = Term::new(80, 24);
        let mut backend = XBackend {
             display: ptr::null_mut(), screen: 0, window: 0, colormap: 0, visual: ptr::null_mut(),
             xft_font: ptr::null_mut(), xft_draw: ptr::null_mut(), xft_colors: Vec::new(),
             xft_color_cache_rgb: HashMap::new(), font_width: 10, font_height: 20,
             font_ascent: 16, wm_delete_window: 0, protocols_atom: 0, clear_gc: ptr::null_mut(),
             epoll_fd: -1,
        };

        let event = BackendEvent::Key { keysym: 0, text: "hello".to_string() };
        backend.handle_event(event, &mut term, pty_fd).unwrap();

        thread::sleep(Duration::from_millis(10));
        let written_bytes = read_from_mock_pty(read_fd, 10);

        assert_eq!(written_bytes, b"hello");

        unsafe { libc::close(pty_fd); libc::close(read_fd); }
    }

    // Test key event handling for special keysym (Return key)
    #[test]
    fn test_handle_event_key_return() {
        let (pty_fd, read_fd) = create_mock_pty();
        set_nonblocking(read_fd);
        let mut term = Term::new(80, 24);
        let mut backend = XBackend {
             display: ptr::null_mut(), screen: 0, window: 0, colormap: 0, visual: ptr::null_mut(),
             xft_font: ptr::null_mut(), xft_draw: ptr::null_mut(), xft_colors: Vec::new(),
             xft_color_cache_rgb: HashMap::new(), font_width: 10, font_height: 20,
             font_ascent: 16, wm_delete_window: 0, protocols_atom: 0, clear_gc: ptr::null_mut(),
              epoll_fd: -1,
        };

        let event = BackendEvent::Key { keysym: keysym::XK_Return as u32, text: "".to_string() };
        backend.handle_event(event, &mut term, pty_fd).unwrap();

        thread::sleep(Duration::from_millis(10));
        let written_bytes = read_from_mock_pty(read_fd, 10);

        assert_eq!(written_bytes, b"\r");

        unsafe { libc::close(pty_fd); libc::close(read_fd); }
    }

     // Test key event handling for arrow key (Up)
     #[test]
     fn test_handle_event_key_up() {
         let (pty_fd, read_fd) = create_mock_pty();
         set_nonblocking(read_fd);
         let mut term = Term::new(80, 24);
         let mut backend = XBackend {
             display: ptr::null_mut(), screen: 0, window: 0, colormap: 0, visual: ptr::null_mut(),
             xft_font: ptr::null_mut(), xft_draw: ptr::null_mut(), xft_colors: Vec::new(),
             xft_color_cache_rgb: HashMap::new(), font_width: 10, font_height: 20,
             font_ascent: 16, wm_delete_window: 0, protocols_atom: 0, clear_gc: ptr::null_mut(),
              epoll_fd: -1,
         };

         let event = BackendEvent::Key { keysym: keysym::XK_Up as u32, text: "".to_string() };
         backend.handle_event(event, &mut term, pty_fd).unwrap();

         thread::sleep(Duration::from_millis(10));
         let written_bytes = read_from_mock_pty(read_fd, 10);

         assert_eq!(written_bytes, b"\x1b[A");

         unsafe { libc::close(pty_fd); libc::close(read_fd); }
     }

    // Test resize event handling (basic check on Term state)
    #[test]
    #[ignore]
    fn test_handle_event_resize() {
        let _ = env_logger::builder().is_test(true).try_init();
        let mut backend = XBackend::new(80, 24).expect("Failed to create XBackend for resize test (requires X server)");
        let mut term = Term::new(80, 24);
        let (pty_fd, read_fd) = create_mock_pty();

        let (initial_w, initial_h) = term.get_dimensions();
        assert_eq!((initial_w, initial_h), (80, 24));

        let new_width_px = backend.font_width * 100;
        let new_height_px = backend.font_height * 30;
        let event = BackendEvent::Resize { width_px: new_width_px as u16, height_px: new_height_px as u16 };

        backend.handle_event(event, &mut term, pty_fd).unwrap();

        let (final_w, final_h) = term.get_dimensions();
        assert_eq!((final_w, final_h), (100, 30), "Terminal dimensions did not update correctly");

        unsafe { libc::close(pty_fd); libc::close(read_fd); }
        backend.cleanup().unwrap();
    }
}
