// Rust Terminal MVP (Proof of Concept)
// Basic terminal emulation concepts: PTY setup, epoll event loop,
// screen state, resizing, backend abstraction, and basic ANSI parsing.
// This is NOT a a production-ready terminal emulator.

use libc::{
    openpty, fork, execvp, ioctl, close,
    winsize, TIOCSWINSZ,
    STDIN_FILENO, STDOUT_FILENO, STDERR_FILENO,
    c_int, pid_t, size_t,
    epoll_create1, epoll_ctl, epoll_wait, epoll_event,
    EPOLL_CTL_ADD, EPOLLIN, EPOLL_CLOEXEC, EPOLLRDHUP, // Added EPOLLRDHUP
    EPOLLERR, EPOLLHUP,
    termios, tcsetattr, TCSAFLUSH,
    // Added for job control
    setsid, /*tcsetpgrp,*/ // Not using tcsetpgrp
    TIOCSCTTY,
    // Imports for non-blocking read
    fcntl, F_GETFL, F_SETFL, O_NONBLOCK,
    // TIOCSPGRP, // No longer used
};
use std::ffi::{CString, CStr};
use std::ptr;
use std::process;
use std::io::{self, Read};
// **FIX: Removed unused IntoRawFd**
use std::os::unix::io::{FromRawFd, RawFd, AsRawFd};
use std::mem;
use anyhow::{Result, Context, Error as AnyhowError};

extern crate x11;
use x11::xlib::{
    self, // Import the module itself for types like Display, Window, GC etc.
    XEvent, XNextEvent, // **FIX: Added XNextEvent**
    XConnectionNumber, XPending, // For event loop
    // Functions used in init/drop
    XOpenDisplay, XDefaultScreen, XDefaultRootWindow, XCreateSimpleWindow,
    XMapWindow, XSelectInput, XBlackPixel, XWhitePixel, XCreateGC,
    XSetForeground, XSetBackground, XLoadQueryFont, XSetFont, /*XFreeFontInfo,*/ // **FIX: Removed unused XFreeFontInfo**
    XFlush, XDestroyWindow, XFreeGC, XCloseDisplay,
    // Functions/Structs used in handle_event/draw
    XKeyEvent, XLookupString, XComposeStatus, KeySym,
    XFillRectangle, XDrawString,
    // Masks used in XSelectInput
    ExposureMask, KeyPressMask, StructureNotifyMask,
    // XConfigureEvent is implicitly part of XEvent union, access via .configure
};


// Define constants
const BUFSIZ: usize = 4096; // Increased buffer size
const MAX_EPOLL_EVENTS: c_int = 10;
const DEFAULT_COLS: usize = 80;
const DEFAULT_ROWS: usize = 24;
const DEFAULT_WIDTH_PX: usize = 640;
const DEFAULT_HEIGHT_PX: usize = 480;
const DEFAULT_FONT_NAME: &str = "fixed"; // A common fallback fixed-width font
const ESC_ARG_SIZ: usize = 16;

// Struct to hold and restore original terminal attributes
#[allow(dead_code)]
struct OriginalTermios(termios);

#[allow(dead_code)]
impl Drop for OriginalTermios {
    fn drop(&mut self) {
        // SAFETY: tcsetattr is FFI.
        unsafe {
            if tcsetattr(STDIN_FILENO, TCSAFLUSH, &self.0) < 0 {
                eprintln!("Error restoring terminal attributes: {}", io::Error::last_os_error());
            }
        }
    }
}

// Simplified Terminal State
#[derive(Debug)]
struct Term {
    _child_pid: pid_t,
    pty_parent: std::fs::File,
    screen: Vec<Vec<char>>,
    cols: usize,
    rows: usize,
    cursor_x: usize,
    cursor_y: usize,
    parser_state: ParserState,
    escape_buffer: Vec<u8>,
    // --- Basic Mode Tracking ---
    dec_modes: DecPrivateModes, // Struct to track DEC private modes
    // Add other state like SGR attributes, charsets if needed
}

// Basic struct to track common DEC private modes
#[derive(Debug, Default, Clone, Copy)]
struct DecPrivateModes {
    cursor_keys_app_mode: bool, // ?1h / ?1l (DECCKM)
    alt_screen_buffer: bool,    // ?47h / ?1047h / ?1049h (and l)
    bracketed_paste: bool,      // ?2004h / ?2004l
    mouse_reporting_btn: bool,  // ?1000h / ?1000l (X10)
    mouse_reporting_motion: bool, // ?1002h / ?1002l (BTN_EVENT)
    mouse_reporting_all: bool,  // ?1003h / ?1003l (ANY_EVENT)
    mouse_reporting_focus: bool,// ?1004h / ?1004l (FOCUS_EVENT)
    mouse_reporting_sgr: bool,  // ?1006h / ?1006l (SGR_EXT_MODE)
    // Add more modes as needed
}


// ANSI Parser States
#[derive(Debug, PartialEq, Clone, Copy)]
enum ParserState {
    Ground,          // Normal character processing
    Escape,          // Received ESC (0x1B)
    Csi,             // Received ESC [ (Control Sequence Introducer)
    Osc,             // Received ESC ] (Operating System Command)
    EscIntermediate, // Received ESC followed by intermediate char (e.g., '(', '#')
}

// Enum to represent the direction/scope of erase operations
#[derive(Debug, PartialEq, Eq, Copy, Clone)]
enum EraseDirection {
    ToEnd,       // Erase from cursor to end (0)
    ToStart,     // Erase from cursor to start (1)
    WholeLine,   // Erase whole line (2)
    Unsupported(usize), // Unsupported parameter value
}

// Enum to represent parsed CSI sequences
#[derive(Debug, PartialEq)]
enum CsiSequence {
    CursorPosition { row: usize, col: usize }, // H, f
    EraseInLine(EraseDirection),              // K
    EraseInDisplay(EraseDirection),          // J
    CursorUp { n: usize },                     // A
    CursorDown { n: usize },                   // B
    CursorForward { n: usize },                // C
    CursorBackward { n: usize },               // D
    PrivateModeSet(usize),                     // ?...h
    PrivateModeReset(usize),                   // ?...l
    // Add SGR(m), etc.
    Unsupported(u8, Vec<usize>),               // Catch-all for unsupported sequences
}

impl Term {
    // Initialize a new terminal state
    fn new(child_pid: pid_t, pty_parent: std::fs::File, cols: usize, rows: usize) -> Self {
        let cols = cols.max(1);
        let rows = rows.max(1);
        let mut screen = Vec::with_capacity(rows);
        for _ in 0..rows {
            screen.push(vec![' '; cols]);
        }
        Term {
            _child_pid: child_pid,
            pty_parent,
            screen,
            cols,
            rows,
            cursor_x: 0,
            cursor_y: 0,
            parser_state: ParserState::Ground,
            escape_buffer: Vec::new(),
            dec_modes: DecPrivateModes::default(), // Initialize modes
        }
    }

    // Resize the terminal
    fn resize(&mut self, new_cols: usize, new_rows: usize) -> Result<()> {
        if new_cols == 0 || new_rows == 0 {
            anyhow::bail!("Terminal resize dimensions must be positive (got {}x{})", new_cols, new_rows);
        }
        if self.cols == new_cols && self.rows == new_rows { return Ok(()); }

        let old_cols = self.cols;
        self.screen.resize_with(new_rows, || vec![' '; old_cols]);
        for row in self.screen.iter_mut() { row.resize(new_cols, ' '); }

        self.cols = new_cols;
        self.rows = new_rows;
        self.cursor_x = std::cmp::min(self.cursor_x, self.cols.saturating_sub(1));
        self.cursor_y = std::cmp::min(self.cursor_y, self.rows.saturating_sub(1));

        let mut win_size = winsize { ws_row: self.rows as u16, ws_col: self.cols as u16, ws_xpixel: 0, ws_ypixel: 0 };
        let fd = self.pty_parent.as_raw_fd();
        if fd >= 0 {
            // SAFETY: ioctl is FFI.
            unsafe {
                if ioctl(fd, TIOCSWINSZ, &mut win_size) < 0 && cfg!(not(test)) {
                    return Err(io::Error::last_os_error()).context("Failed to set window size on PTY parent (TIOCSWINSZ)");
                }
            }
        }
        Ok(())
    }

    // Process a single byte from the PTY based on parser state
    // This method modifies internal state and doesn't return a Result anymore.
    fn process_byte(&mut self, byte: u8) {
        // println!("Processing byte: {:02X} in state: {:?}", byte, self.parser_state); // Debug
        match self.parser_state {
            ParserState::Ground => {
                match byte {
                    0x1B => { self.parser_state = ParserState::Escape; self.escape_buffer.clear(); } // ESC
                    b'\n' => { self.cursor_y = std::cmp::min(self.cursor_y + 1, self.rows - 1); } // LF
                    b'\r' => { self.cursor_x = 0; } // CR
                    0x08 => { self.cursor_x = self.cursor_x.saturating_sub(1); } // BS
                    0x07 => { /* BEL - TODO: Ring bell */ }
                    0..=6 | 9..=12 | 14..=31 | 127 => { /* Ignore other C0/DEL */ }
                    _ => { // Printable character
                         self.cursor_y = std::cmp::min(self.cursor_y, self.rows.saturating_sub(1));
                        if self.cursor_x >= self.cols {
                             self.cursor_x = 0;
                             self.cursor_y = std::cmp::min(self.cursor_y + 1, self.rows - 1);
                        }
                        self.cursor_x = std::cmp::min(self.cursor_x, self.cols.saturating_sub(1));
                        self.cursor_y = std::cmp::min(self.cursor_y, self.rows.saturating_sub(1));
                        // Ensure indices are valid before writing
                        if self.cursor_y < self.screen.len() && self.cursor_x < self.screen[self.cursor_y].len() {
                            self.screen[self.cursor_y][self.cursor_x] = byte as char;
                        } else {
                            // This should ideally not happen if bounds checks above are correct
                            eprintln!("Warning: Attempted write outside screen bounds ({},{})", self.cursor_x, self.cursor_y);
                        }
                        self.cursor_x += 1;
                    }
                }
            }
            ParserState::Escape => {
                match byte {
                    b'[' => { self.parser_state = ParserState::Csi; self.escape_buffer.clear(); }
                    b']' => { self.parser_state = ParserState::Osc; self.escape_buffer.clear(); }
                    b'(' | b')' | b'*' | b'+' => {
                        self.escape_buffer.push(byte);
                        self.parser_state = ParserState::EscIntermediate;
                    }
                    b'=' => { /*println!("Debug: ESC = (DECKPAM) received");*/ self.parser_state = ParserState::Ground; }
                    b'>' => { /*println!("Debug: ESC > (DECKPNM) received");*/ self.parser_state = ParserState::Ground; }
                    _ => { eprintln!("Warning: Unhandled sequence after ESC: ESC {}", byte as char); self.parser_state = ParserState::Ground; }
                }
            }
             ParserState::EscIntermediate => {
                 let intermediate = self.escape_buffer.pop().unwrap_or(0);
                 match intermediate {
                     b'(' => { /*println!("Debug: Charset G0 selection: ESC ( {}", byte as char);*/ self.parser_state = ParserState::Ground; }
                     // TODO: Handle other intermediates like ')', '*', '+', '#'
                     _ => { eprintln!("Warning: Unhandled sequence ESC {}{}", intermediate as char, byte as char); self.parser_state = ParserState::Ground; }
                 }
             }
            ParserState::Csi => {
                self.escape_buffer.push(byte);
                if byte >= 0x40 && byte <= 0x7E { // Final byte
                    match self.parse_csi_sequence() {
                        Ok(sequence) => self.handle_csi_sequence(sequence),
                        Err(e) => eprintln!("Error parsing CSI sequence: {}", e),
                    }
                    self.parser_state = ParserState::Ground;
                    self.escape_buffer.clear();
                } else if byte < 0x20 || byte > 0x3F { // Invalid byte
                    eprintln!("Warning: Invalid byte in CSI sequence: 0x{:02X}", byte);
                    self.parser_state = ParserState::Ground;
                    self.escape_buffer.clear();
                }
            }
            ParserState::Osc => {
                 match byte {
                     0x07 => { /* BEL terminator */ self.parser_state = ParserState::Ground; self.escape_buffer.clear(); }
                     0x1B => { self.escape_buffer.push(byte); } // Potential ST start
                     b'\\' if self.escape_buffer.last() == Some(&0x1B) => { // ST (ESC \)
                         self.escape_buffer.pop();
                         // TODO: Process OSC: String::from_utf8_lossy(&self.escape_buffer)
                         self.parser_state = ParserState::Ground;
                         self.escape_buffer.clear();
                     }
                     _ => { self.escape_buffer.push(byte); } // Collect byte
                 }
             }
        }
    }

    // Parse a completed CSI sequence into a CsiSequence enum
    fn parse_csi_sequence(&self) -> Result<CsiSequence> {
        let sequence = &self.escape_buffer;
        if sequence.is_empty() { anyhow::bail!("Empty CSI sequence buffer"); }

        let final_byte = *sequence.last().unwrap();
        let params_bytes = &sequence[..sequence.len() - 1];

        let mut cursor = 0;
        let mut is_private = false;
        if params_bytes.get(cursor) == Some(&b'?') { is_private = true; cursor += 1; }

        let mut params: Vec<usize> = Vec::new();
        let mut current_param_start = cursor;
        while cursor < params_bytes.len() {
            if params_bytes[cursor] == b';' {
                 let param_slice = &params_bytes[current_param_start..cursor];
                 let param_str = String::from_utf8_lossy(param_slice);
                 // Default empty parameters to 0 for now, specific handlers might override
                 params.push(if param_str.is_empty() { 0 } else { param_str.parse().unwrap_or(0) });
                 current_param_start = cursor + 1;
                 if params.len() >= ESC_ARG_SIZ { break; }
             }
            cursor += 1;
        }
         if current_param_start <= params_bytes.len() {
             let param_slice = &params_bytes[current_param_start..params_bytes.len()];
             let param_str = String::from_utf8_lossy(param_slice);
             params.push(if param_str.is_empty() { 0 } else { param_str.parse().unwrap_or(0) });
         }

        // Helper to get param or default value.
        // IMPORTANT: VT defaults are tricky. Often 1 for missing/0 params in movement,
        // but 0 for missing/0 params in erase/modes. We use a provided default here.
        let get_param = |index: usize, default: usize| -> usize {
            // If param exists and is non-zero, use it. Otherwise use default.
            // This handles cases where param 0 is meaningful but missing/empty means default=1.
            match params.get(index) {
                Some(&0) => 0, // Explicit 0 is 0
                Some(&val) => val,
                None => default, // Missing param uses default
            }
        };
        let parse_erase_param = |p_idx: usize| {
            match get_param(p_idx, 0) { // Default erase param is 0
                0 => EraseDirection::ToEnd, 1 => EraseDirection::ToStart, 2 => EraseDirection::WholeLine,
                val => EraseDirection::Unsupported(val),
            }
        };

        match final_byte {
            b'H' | b'f' => Ok(CsiSequence::CursorPosition { row: get_param(0, 1), col: get_param(1, 1) }),
            b'A' => Ok(CsiSequence::CursorUp { n: get_param(0, 1) }), // Default n=1
            b'B' => Ok(CsiSequence::CursorDown { n: get_param(0, 1) }), // Default n=1
            b'C' => Ok(CsiSequence::CursorForward { n: get_param(0, 1) }), // Default n=1
            b'D' => Ok(CsiSequence::CursorBackward { n: get_param(0, 1) }), // Default n=1
            b'J' => Ok(CsiSequence::EraseInDisplay(parse_erase_param(0))),
            b'K' => Ok(CsiSequence::EraseInLine(parse_erase_param(0))),
            b'h' if is_private => Ok(CsiSequence::PrivateModeSet(get_param(0, 0))), // Default mode 0? Needs check per mode
            b'l' if is_private => Ok(CsiSequence::PrivateModeReset(get_param(0, 0))), // Default mode 0?
            // TODO: Add 'm' SGR parsing
            _ => Ok(CsiSequence::Unsupported(final_byte, params)),
        }
    }


    // Handle a parsed CSI sequence, updating terminal state
    fn handle_csi_sequence(&mut self, sequence: CsiSequence) {
        match sequence {
            CsiSequence::CursorPosition { row, col } => {
                self.cursor_y = row.saturating_sub(1).min(self.rows.saturating_sub(1));
                self.cursor_x = col.saturating_sub(1).min(self.cols.saturating_sub(1));
            }
            CsiSequence::EraseInLine(direction) => {
                if self.cursor_y >= self.rows { return; }
                match direction {
                    EraseDirection::ToEnd => { for x in self.cursor_x..self.cols { self.screen[self.cursor_y][x] = ' '; } }
                    EraseDirection::ToStart => { let end = self.cursor_x.min(self.cols.saturating_sub(1)); for x in 0..=end { self.screen[self.cursor_y][x] = ' '; } }
                    EraseDirection::WholeLine => { self.screen[self.cursor_y].fill(' '); }
                    EraseDirection::Unsupported(_) => {}
                }
            }
             CsiSequence::EraseInDisplay(direction) => {
                 match direction {
                     EraseDirection::ToEnd => {
                         if self.cursor_y < self.rows { for x in self.cursor_x..self.cols { self.screen[self.cursor_y][x] = ' '; } }
                         for y in self.cursor_y.saturating_add(1)..self.rows { self.screen[y].fill(' '); }
                     }
                     EraseDirection::ToStart => {
                          if self.cursor_y < self.rows { let end = self.cursor_x.min(self.cols.saturating_sub(1)); for x in 0..=end { self.screen[self.cursor_y][x] = ' '; } }
                         for y in 0..self.cursor_y { if y < self.rows { self.screen[y].fill(' '); } }
                     }
                     EraseDirection::WholeLine => { for y in 0..self.rows { if y < self.rows { self.screen[y].fill(' '); } } }
                    EraseDirection::Unsupported(_) => {}
                 }
             }
             // **FIX: Ensure n is at least 1 for movement, even if param was 0**
            CsiSequence::CursorUp { n } => { let n = n.max(1); self.cursor_y = self.cursor_y.saturating_sub(n).max(0); }
            CsiSequence::CursorDown { n } => { let n = n.max(1); self.cursor_y = self.cursor_y.saturating_add(n).min(self.rows.saturating_sub(1)); }
            CsiSequence::CursorForward { n } => { let n = n.max(1); self.cursor_x = self.cursor_x.saturating_add(n).min(self.cols.saturating_sub(1)); }
            CsiSequence::CursorBackward { n } => { let n = n.max(1); self.cursor_x = self.cursor_x.saturating_sub(n).max(0); }
            CsiSequence::PrivateModeSet(mode) => self.handle_dec_mode_set(mode, true),
            CsiSequence::PrivateModeReset(mode) => self.handle_dec_mode_set(mode, false),
            CsiSequence::Unsupported(_, _) => { /* Silently ignore */ }
        }
    }

    // Helper to handle DEC Private Mode setting/resetting
    fn handle_dec_mode_set(&mut self, mode: usize, set: bool) {
        // println!("DEC Mode {} {}", if set { "Set" } else { "Reset" }, mode); // Debug
        match mode {
            1 => self.dec_modes.cursor_keys_app_mode = set, // DECCKM
            // 7 => // DECAWM - Auto Wrap Mode (already handled implicitly?)
            12 => { /* att610 - Stop/Start Blinking Cursor - No-op for now */ }
            25 => { /* DECTCEM - Text Cursor Enable Mode - TODO: signal backend */ }
            47 | 1047 | 1049 => { // Alternate Screen Buffer modes
                if set != self.dec_modes.alt_screen_buffer {
                     // println!("Debug: {} Alternate Screen Buffer", if set { "Entering" } else { "Leaving" });
                     self.dec_modes.alt_screen_buffer = set;
                     // TODO: Implement screen buffer swapping
                }
            }
            1000 => self.dec_modes.mouse_reporting_btn = set, // X10 mouse
            1002 => self.dec_modes.mouse_reporting_motion = set, // Button-event mouse
            1003 => self.dec_modes.mouse_reporting_all = set, // Any-event mouse
            1004 => self.dec_modes.mouse_reporting_focus = set, // Focus reporting
            1005 => { /* UTF8 Mouse Mode - Ignore for now */ }
            1006 => self.dec_modes.mouse_reporting_sgr = set, // SGR mouse
            2004 => self.dec_modes.bracketed_paste = set, // Bracketed paste
            7727 => { /* Unknown mode, possibly from 'less' - Ignore */ }
            _ => { /* eprintln!("Warning: Unhandled DEC Private Mode: ?{} {}", mode, if set {'h'} else {'l'}); */ } // Silence warnings for now
        }
    }


    // Process incoming data from the PTY based on parser state
    fn process_pty_data(&mut self, data: &[u8]) -> Result<()> {
        for &byte in data {
            self.process_byte(byte);
        }
        Ok(())
    }

    // Get the raw file descriptor for the PTY parent
    #[allow(dead_code)]
    fn pty_parent_fd(&self) -> RawFd {
        self.pty_parent.as_raw_fd()
    }
}

// --- Terminal Backend Trait ---
trait TerminalBackend {
    fn init(&mut self) -> Result<()>;
    fn get_event_fds(&self) -> Vec<RawFd>;
    fn handle_event(&mut self, term: &mut Term, event_fd: RawFd, event_kind: u32) -> Result<bool>;
    fn draw(&mut self, term: &Term) -> Result<()> ;
    fn get_dimensions(&self) -> (usize, usize);
}

// --- X Backend Implementation ---
struct XBackend {
    display: *mut xlib::Display,
    window: xlib::Window,
    x_fd: RawFd,
    gc: xlib::GC,
    font_struct: *mut xlib::XFontStruct,
    font_width: i32,
    font_height: i32,
    font_ascent: i32,
}


impl XBackend {
     fn new() -> Self {
         XBackend {
             display: ptr::null_mut(), window: 0, x_fd: -1, gc: ptr::null_mut(),
             font_struct: ptr::null_mut(), font_width: 0, font_height: 0, font_ascent: 0,
         }
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
} // End impl XBackend

impl TerminalBackend for XBackend {
     fn init(&mut self) -> Result<()> {
         // SAFETY: All Xlib calls are FFI.
         unsafe {
             let display = XOpenDisplay(ptr::null());
             if display.is_null() { anyhow::bail!("Failed to open X display"); }
             self.display = display;
             let screen = XDefaultScreen(display);
             let root_window = XDefaultRootWindow(display);
             let window = XCreateSimpleWindow( display, root_window, 0, 0, DEFAULT_WIDTH_PX as u32, DEFAULT_HEIGHT_PX as u32, 0, XBlackPixel(display, screen), XWhitePixel(display, screen));
             if window == 0 { XCloseDisplay(display); anyhow::bail!("Failed to create X window"); }
             self.window = window;
             XSelectInput( display, window, ExposureMask | KeyPressMask | StructureNotifyMask );
             XMapWindow(display, window);
             self.x_fd = XConnectionNumber(display);
             if self.x_fd < 0 { /* ... error handling ... */ return Err(io::Error::last_os_error()).context("Failed to get X FD"); }
             let font_name_cstring = CString::new(DEFAULT_FONT_NAME)?;
             let font_struct = XLoadQueryFont(display, font_name_cstring.as_ptr());
             if font_struct.is_null() { /* ... error handling ... */ anyhow::bail!("Failed to load font"); }
             self.font_struct = font_struct;
             // Cast i16 font metrics to i32
             if !(*self.font_struct).per_char.is_null() { self.font_width = (*self.font_struct).max_bounds.width as i32; }
             else if (*self.font_struct).min_bounds.width == (*self.font_struct).max_bounds.width { self.font_width = (*self.font_struct).max_bounds.width as i32; }
             else { self.font_width = (*self.font_struct).max_bounds.width as i32; /* Warning emitted before */ }
             self.font_ascent = (*self.font_struct).ascent as i32;
             self.font_height = ((*self.font_struct).ascent + (*self.font_struct).descent) as i32;
             if self.font_width <= 0 || self.font_height <= 0 { /* ... error handling ... */ anyhow::bail!("Invalid font dimensions"); }
             let gc = XCreateGC(display, window, 0, ptr::null_mut());
             if gc.is_null() { /* ... error handling ... */ return Err(io::Error::last_os_error()).context("Failed to create GC"); }
             self.gc = gc;
             XSetForeground(display, gc, XBlackPixel(display, screen));
             XSetBackground(display, gc, XWhitePixel(display, screen));
             XSetFont(display, gc, (*self.font_struct).fid);
             XFlush(display);
         }
         println!("X Backend initialized. Window created. Font loaded.");
         Ok(())
     }

     fn get_event_fds(&self) -> Vec<RawFd> { vec![self.x_fd] }

     fn handle_event(&mut self, term: &mut Term, event_fd: RawFd, event_kind: u32) -> Result<bool> {
        if event_fd != self.x_fd { return Ok(false); }
        if event_kind & ((EPOLLERR | EPOLLHUP | EPOLLRDHUP) as u32) != 0 {
            eprintln!("Error or hang-up on X connection fd (event kind: 0x{:x}).", event_kind);
            return Ok(true); // Exit on X error
        }
        if event_kind & (EPOLLIN as u32) == 0 { return Ok(false); }

        // SAFETY: FFI calls within loop.
        unsafe {
            while XPending(self.display) > 0 {
                let mut event: XEvent = mem::zeroed();
                XNextEvent(self.display, &mut event);
                let event_type = event.type_; // Access type safely inside unsafe block

                let result = match event_type {
                    xlib::KeyPress => self.handle_key_press(term, &event),
                    xlib::Expose => self.handle_expose(term, &event),
                    xlib::ConfigureNotify => self.handle_configure_notify(term, &event),
                    _ => Ok(()),
                };
                if let Err(e) = result { eprintln!("Error handling X event type {}: {}", event_type, e); }
            }
        }
        Ok(false)
    }


     fn draw(&mut self, term: &Term) -> Result<()> {
         // SAFETY: All Xlib drawing calls are FFI.
         unsafe {
             if self.display.is_null() || self.gc.is_null() || self.window == 0 { return Err(AnyhowError::msg("X draw called with invalid state")); }
             let screen = XDefaultScreen(self.display);
             let bg = XWhitePixel(self.display, screen);
             let fg = XBlackPixel(self.display, screen);
             XSetForeground(self.display, self.gc, bg);
             let w_px = (term.cols * self.font_width as usize) as u32;
             let h_px = (term.rows * self.font_height as usize) as u32;
             XFillRectangle(self.display, self.window, self.gc, 0, 0, w_px, h_px);
             XSetForeground(self.display, self.gc, fg);
             for y in 0..term.rows {
                 for x in 0..term.cols {
                     let character = term.screen[y][x];
                     if character == ' ' { continue; }
                     let mut bytes = [0u8; 4];
                     let slice = character.encode_utf8(&mut bytes).as_bytes();
                     let dx = (x * self.font_width as usize) as i32;
                     let dy = (y * self.font_height as usize + self.font_ascent as usize) as i32;
                     XDrawString( self.display, self.window, self.gc, dx, dy, slice.as_ptr() as *const i8, slice.len() as i32 );
                 }
             }
             let cx = std::cmp::min(term.cursor_x, term.cols.saturating_sub(1));
             let cy = std::cmp::min(term.cursor_y, term.rows.saturating_sub(1));
             let cx_px = (cx * self.font_width as usize) as i32;
             let cy_px = (cy * self.font_height as usize) as i32;
             XSetForeground(self.display, self.gc, fg); // Use fg for cursor block
             XFillRectangle(self.display, self.window, self.gc, cx_px, cy_px, self.font_width as u32, self.font_height as u32);
             XFlush(self.display);
         }
         Ok(())
     }


     fn get_dimensions(&self) -> (usize, usize) {
         let cols = if self.font_width > 0 { DEFAULT_WIDTH_PX.saturating_div(self.font_width as usize).max(1) } else { DEFAULT_COLS };
         let rows = if self.font_height > 0 { DEFAULT_HEIGHT_PX.saturating_div(self.font_height as usize).max(1) } else { DEFAULT_ROWS };
         (cols, rows)
     }
} // End impl TerminalBackend for XBackend

impl Drop for XBackend {
    fn drop(&mut self) {
        // SAFETY: All Xlib calls are FFI.
        unsafe {
            if !self.display.is_null() {
                if !self.gc.is_null() { XFreeGC(self.display, self.gc); }
                if self.window != 0 { XDestroyWindow(self.display, self.window); }
                XCloseDisplay(self.display);
            }
        }
    }
}

// Create PTY, fork, and exec the shell
// Entire function is unsafe due to FFI calls
unsafe fn create_pty_and_fork(shell_path: &CStr, shell_args: &[*mut i8]) -> Result<(pid_t, std::fs::File)> {
    let mut pty_parent_fd: c_int = -1;
    let mut pty_child_fd: c_int = -1;
    // SAFETY: openpty is FFI.
    let openpty_res = unsafe { openpty(&mut pty_parent_fd, &mut pty_child_fd, ptr::null_mut(), ptr::null_mut(), ptr::null_mut()) };
    if openpty_res < 0 { return Err(AnyhowError::from(io::Error::last_os_error()).context("openpty failed")); }
    // SAFETY: fork is FFI.
    let child_pid: pid_t = unsafe { fork() };
    if child_pid < 0 { /* ... error handling ... */ return Err(AnyhowError::from(io::Error::last_os_error()).context("fork failed")); }
    if child_pid != 0 { // Parent
        // SAFETY: close is FFI.
        unsafe { close(pty_child_fd); }
        // SAFETY: from_raw_fd is unsafe.
        let pty_parent_file = unsafe { std::fs::File::from_raw_fd(pty_parent_fd) };
        return Ok((child_pid, pty_parent_file));
    }
    // Child process
    // SAFETY: FFI calls below.
    unsafe {
        close(pty_parent_fd);
        if setsid() < 0 { eprintln!("Child Error: setsid failed: {}", io::Error::last_os_error()); close(pty_child_fd); process::exit(1); }
        if libc::dup2(pty_child_fd, STDIN_FILENO) < 0 { eprintln!("Child Error: dup2 stdin failed: {}", io::Error::last_os_error()); close(pty_child_fd); process::exit(1); }
        if libc::dup2(pty_child_fd, STDOUT_FILENO) < 0 { eprintln!("Child Error: dup2 stdout failed: {}", io::Error::last_os_error()); close(pty_child_fd); process::exit(1); }
        if libc::dup2(pty_child_fd, STDERR_FILENO) < 0 { eprintln!("Child Error: dup2 stderr failed: {}", io::Error::last_os_error()); close(pty_child_fd); process::exit(1); }
        if ioctl(STDIN_FILENO, TIOCSCTTY, ptr::null_mut::<c_int>()) < 0 { eprintln!("Child Error: ioctl TIOCSCTTY failed: {}", io::Error::last_os_error()); if pty_child_fd > 2 { close(pty_child_fd); } process::exit(1); }
        if pty_child_fd > 2 { close(pty_child_fd); }
        let term_env_var = CString::new("TERM=xterm-256color").unwrap();
        libc::putenv(term_env_var.into_raw() as *mut i8); // Leaks okay before exec
        let shell_args_ptr_const: Vec<*const i8> = shell_args.iter().map(|&p| p as *const i8).collect();
        execvp(shell_path.as_ptr(), shell_args_ptr_const.as_ptr());
        // execvp only returns on error
        eprintln!("Child Error: execvp failed for '{}': {}", shell_path.to_string_lossy(), io::Error::last_os_error());
        process::exit(1);
    }
}


// Handles reading from the PTY parent and processing data.
fn handle_pty_read(term: &mut Term, buf: &mut [u8]) -> Result<bool> {
    match term.pty_parent.read(buf) {
        Ok(0) => Ok(true), // PTY closed
        Ok(nread) => { term.process_pty_data(&buf[..nread])?; Ok(false) }
        Err(ref e) if e.kind() == io::ErrorKind::WouldBlock => Ok(false),
        Err(ref e) if e.kind() == io::ErrorKind::Interrupted => Ok(false),
        Err(e) => { eprintln!("Error reading from PTY: {} (kind: {:?})", e, e.kind()); Err(e.into()) }
    }
}

// Main function implementing the epoll event loop
fn main() -> Result<()> {
    let shell_path_str = "/bin/sh";
    let shell_args_str_opt: &[Option<&str>] = &[Some("sh"), Some("-l"), None];
    let shell_path = CString::new(shell_path_str)?;
    let shell_args_c: Vec<CString> = shell_args_str_opt.iter().filter_map(|opt_s| opt_s.map(CString::new)).collect::<Result<_, _>>()?;
    let mut shell_args_ptr: Vec<*mut i8> = shell_args_c.iter().map(|cs| cs.as_ptr() as *mut i8).collect();
    shell_args_ptr.push(ptr::null_mut());

    // SAFETY: Main block contains many FFI calls.
    unsafe {
        let (_child_pid, pty_parent_file) = create_pty_and_fork(&shell_path, &shell_args_ptr)?;
        let pty_parent_fd = pty_parent_file.as_raw_fd();
        // Set non-blocking
        let flags = fcntl(pty_parent_fd, F_GETFL, 0);
        if flags < 0 || fcntl(pty_parent_fd, F_SETFL, flags | O_NONBLOCK) < 0 {
             eprintln!("Warning: Failed to set PTY master non-blocking: {}", io::Error::last_os_error());
              close(pty_parent_fd); return Err(io::Error::last_os_error().into());
        }

        let mut backend: Box<dyn TerminalBackend> = Box::new(XBackend::new());
        backend.init().context("Backend init failed")?;
        let (initial_cols, initial_rows) = backend.get_dimensions();
        let mut term = Term::new(_child_pid, pty_parent_file, initial_cols, initial_rows);
        if let Err(e) = term.resize(initial_cols, initial_rows) { eprintln!("Warning: Failed to set initial PTY size: {}", e); }

        // Epoll setup
        let epoll_fd = epoll_create1(EPOLL_CLOEXEC);
        if epoll_fd < 0 { return Err(AnyhowError::from(io::Error::last_os_error()).context("epoll_create1 failed")); }
        let mut pty_event = epoll_event { events: (EPOLLIN | EPOLLRDHUP) as u32, u64: pty_parent_fd as u64 };
        if epoll_ctl(epoll_fd, EPOLL_CTL_ADD, pty_parent_fd, &mut pty_event) < 0 { close(epoll_fd); return Err(AnyhowError::from(io::Error::last_os_error()).context("epoll_ctl PTY failed")); }
        let backend_fds = backend.get_event_fds();
        for &fd in &backend_fds {
            let mut backend_event = epoll_event { events: (EPOLLIN | EPOLLRDHUP) as u32, u64: fd as u64 };
            if epoll_ctl(epoll_fd, EPOLL_CTL_ADD, fd, &mut backend_event) < 0 { close(epoll_fd); return Err(AnyhowError::from(io::Error::last_os_error()).context(format!("epoll_ctl backend fd {} failed", fd))); }
        }

        let mut pty_buf = vec![0u8; BUFSIZ];
        let mut events: Vec<epoll_event> = vec![mem::zeroed(); MAX_EPOLL_EVENTS as usize];
        println!("Terminal MVP running. Type commands or press Ctrl+D to exit.");
        backend.draw(&term)?;

        // Event loop
        loop {
            let num_events = epoll_wait(epoll_fd, events.as_mut_ptr(), MAX_EPOLL_EVENTS, -1);
            if num_events < 0 {
                if io::Error::last_os_error().kind() == io::ErrorKind::Interrupted { continue; }
                close(epoll_fd); return Err(AnyhowError::from(io::Error::last_os_error()).context("epoll_wait failed"));
            }
            let mut should_exit = false;
            for i in 0..num_events {
                let event = &events[i as usize];
                let event_fd = event.u64 as RawFd;
                let event_kind = event.events;
                 if event_kind & (EPOLLERR | EPOLLHUP | EPOLLRDHUP) as u32 != 0 {
                     if event_fd == pty_parent_fd { eprintln!("PTY hang-up/error (event 0x{:x}). Shell exited.", event_kind); should_exit = true; break; }
                     else if backend_fds.contains(&event_fd) { eprintln!("Backend hang-up/error on fd {} (event 0x{:x}).", event_fd, event_kind); should_exit = true; break; }
                     else { eprintln!("Error/hang-up on unexpected fd {} (event 0x{:x}).", event_fd, event_kind); should_exit = true; break; }
                 }
                 if event_kind & EPOLLIN as u32 != 0 {
                    if event_fd == pty_parent_fd {
                        match handle_pty_read(&mut term, &mut pty_buf) {
                            Ok(true) => { should_exit = true; break; } // PTY closed
                            Ok(false) => { if let Err(e) = backend.draw(&term) { eprintln!("Draw error: {}", e); } }
                            Err(e) => { eprintln!("PTY read error: {}", e); should_exit = true; break; }
                        }
                    } else if backend_fds.contains(&event_fd) {
                         match backend.handle_event(&mut term, event_fd, event_kind) {
                             Ok(true) => { should_exit = true; break; } // Backend exit request
                             Ok(false) => {}
                             Err(e) => { eprintln!("Backend event error: {}", e); should_exit = true; break; }
                         }
                    }
                 }
            }
            if should_exit { break; }
        }
        close(epoll_fd);
        println!("Terminal MVP exiting.");
    } // End main unsafe block
    Ok(())
}

// --- Unit Tests ---
#[cfg(test)]
mod tests {
    use super::*; // Import items from outer module
    use std::fs::OpenOptions;

    // Helper to create a dummy Term instance for testing parser logic
    fn create_test_term(cols: usize, rows: usize) -> Term {
        let dummy_file = OpenOptions::new().read(true).write(true).open("/dev/null").expect("Failed to open /dev/null for test");
        let dummy_pid = 0;
        Term::new(dummy_pid, dummy_file, cols, rows)
    }

    // Helper to process a string of bytes through the term parser
    fn process_str(term: &mut Term, input: &str) {
        // process_pty_data now returns Result, but we ignore it in tests for now
        let _ = term.process_pty_data(input.as_bytes());
    }

    #[test]
    fn test_printable_chars() {
        let mut term = create_test_term(5, 2);
        process_str(&mut term, "abc");
        assert_eq!(term.cursor_x, 3);
        assert_eq!(term.cursor_y, 0);
        assert_eq!(term.screen[0][0], 'a');
        assert_eq!(term.screen[0][1], 'b');
        assert_eq!(term.screen[0][2], 'c');
        assert_eq!(term.screen[0][3], ' ');
    }

    #[test]
    fn test_line_wrap() {
        let mut term = create_test_term(3, 2);
        process_str(&mut term, "abcd");
        assert_eq!(term.cursor_x, 1);
        assert_eq!(term.cursor_y, 1);
        assert_eq!(term.screen[0].iter().collect::<String>(), "abc");
        assert_eq!(term.screen[1].iter().collect::<String>(), "d  ");
    }

     #[test]
    fn test_newline() {
        let mut term = create_test_term(5, 3);
        // **FIX: Test with \r\n for expected behavior**
        process_str(&mut term, "ab\r\ncd");
        assert_eq!(term.cursor_x, 2); // After 'cd'
        assert_eq!(term.cursor_y, 1); // On the second line
        assert_eq!(term.screen[0].iter().collect::<String>(), "ab   ");
        assert_eq!(term.screen[1].iter().collect::<String>(), "cd   ");
    }

    #[test]
    fn test_carriage_return() {
        let mut term = create_test_term(5, 2);
        process_str(&mut term, "abc\rde");
        assert_eq!(term.cursor_x, 2);
        assert_eq!(term.cursor_y, 0);
        assert_eq!(term.screen[0].iter().collect::<String>(), "dec  ");
    }

     #[test]
    fn test_backspace() {
        let mut term = create_test_term(5, 2);
        process_str(&mut term, "abc"); // Write abc, cursor at x=3
        term.process_byte(0x08); // Backspace moves cursor to x=2
        assert_eq!(term.cursor_x, 2);
        assert_eq!(term.cursor_y, 0);
        process_str(&mut term, "d"); // Type 'd' over 'c' at index 2
        // **FIX: Correct assertion - cursor should be at 3 after typing 'd'**
        assert_eq!(term.cursor_x, 3);
        assert_eq!(term.screen[0].iter().collect::<String>(), "abd  ");
    }


    #[test]
    fn test_parser_state_csi() {
        let mut term = create_test_term(5, 2);
        // **FIX: Remove .unwrap() calls**
        term.process_byte(0x1B); // ESC
        assert_eq!(term.parser_state, ParserState::Escape);
        term.process_byte(b'['); // [
        assert_eq!(term.parser_state, ParserState::Csi);
        term.process_byte(b'A'); // A (final byte)
        assert_eq!(term.parser_state, ParserState::Ground); // Should return to ground
    }

     #[test]
    fn test_csi_cursor_up() {
        let mut term = create_test_term(5, 3);
        term.cursor_y = 2;
        process_str(&mut term, "\x1b[A"); // ESC [ A (CUU 1)
        assert_eq!(term.cursor_y, 1);
        assert_eq!(term.cursor_x, 0); // Should remain 0
    }

     #[test]
    fn test_csi_cursor_up_param() {
        let mut term = create_test_term(5, 5);
        term.cursor_y = 4;
        process_str(&mut term, "\x1b[3A"); // ESC [ 3 A (CUU 3)
        assert_eq!(term.cursor_y, 1);
        assert_eq!(term.cursor_x, 0);
    }

    #[test]
    fn test_csi_cursor_down() {
        let mut term = create_test_term(5, 3);
        term.cursor_y = 0;
        process_str(&mut term, "\x1b[B"); // ESC [ B (CUD 1)
        assert_eq!(term.cursor_y, 1);
        assert_eq!(term.cursor_x, 0);
    }

    #[test]
    fn test_csi_cursor_forward() {
        let mut term = create_test_term(5, 3);
        process_str(&mut term, "\x1b[C"); // ESC [ C (CUF 1)
        assert_eq!(term.cursor_y, 0);
        assert_eq!(term.cursor_x, 1);
        process_str(&mut term, "\x1b[2C"); // ESC [ 2 C (CUF 2)
        assert_eq!(term.cursor_x, 3);
    }

    #[test]
    fn test_csi_cursor_backward() {
        let mut term = create_test_term(5, 3);
        term.cursor_x = 4;
        process_str(&mut term, "\x1b[D"); // ESC [ D (CUB 1)
        assert_eq!(term.cursor_y, 0);
        assert_eq!(term.cursor_x, 3);
        process_str(&mut term, "\x1b[2D"); // ESC [ 2 D (CUB 2)
        assert_eq!(term.cursor_x, 1);
    }

    #[test]
    fn test_csi_cursor_position() {
        let mut term = create_test_term(10, 5);
        process_str(&mut term, "\x1b[3;4H"); // ESC [ 3 ; 4 H (CUP row 3, col 4)
        assert_eq!(term.cursor_y, 2); // 0-based index
        assert_eq!(term.cursor_x, 3); // 0-based index
    }

     #[test]
    fn test_csi_erase_line_to_end() {
        let mut term = create_test_term(5, 2);
        process_str(&mut term, "abcde"); // Fill line
        term.cursor_x = 2; // Move cursor to 'c'
        process_str(&mut term, "\x1b[K"); // ESC [ K (EL 0 - To End)
        assert_eq!(term.screen[0].iter().collect::<String>(), "ab   ");
        assert_eq!(term.cursor_x, 2); // Cursor doesn't move
    }

    #[test]
    fn test_csi_erase_line_to_start() {
        let mut term = create_test_term(5, 2);
        process_str(&mut term, "abcde"); // Fill line
        term.cursor_x = 2; // Move cursor to 'c'
        process_str(&mut term, "\x1b[1K"); // ESC [ 1 K (EL 1 - To Start)
        assert_eq!(term.screen[0].iter().collect::<String>(), "   de");
        assert_eq!(term.cursor_x, 2); // Cursor doesn't move
    }

    #[test]
    fn test_csi_erase_whole_line() {
        let mut term = create_test_term(5, 2);
        process_str(&mut term, "abcde"); // Fill line
        term.cursor_x = 2; // Move cursor to 'c'
        process_str(&mut term, "\x1b[2K"); // ESC [ 2 K (EL 2 - Whole Line)
        assert_eq!(term.screen[0].iter().all(|&c| c == ' '), true); // Whole line is spaces
        assert_eq!(term.cursor_x, 2); // Cursor doesn't move
    }

    #[test]
    fn test_dec_mode_set_reset() {
        let mut term = create_test_term(10, 5);
        assert_eq!(term.dec_modes.cursor_keys_app_mode, false);
        process_str(&mut term, "\x1b[?1h"); // DECCKM Set
        assert_eq!(term.dec_modes.cursor_keys_app_mode, true);
        process_str(&mut term, "\x1b[?1l"); // DECCKM Reset
        assert_eq!(term.dec_modes.cursor_keys_app_mode, false);

        assert_eq!(term.dec_modes.bracketed_paste, false);
        process_str(&mut term, "\x1b[?2004h"); // Bracketed Paste Set
        assert_eq!(term.dec_modes.bracketed_paste, true);
        process_str(&mut term, "\x1b[?2004l"); // Bracketed Paste Reset
        assert_eq!(term.dec_modes.bracketed_paste, false);
    }

    #[test]
    fn test_esc_intermediate_charset() {
        let mut term = create_test_term(10, 5);
        process_str(&mut term, "\x1b(B"); // Designate G0 as US ASCII
        assert_eq!(term.parser_state, ParserState::Ground); // Should return to ground
        // TODO: Add assertion for actual charset state if implemented
    }

    #[test]
    fn test_esc_keypad_modes() {
         let mut term = create_test_term(10, 5);
         process_str(&mut term, "\x1b="); // DECKPAM
         assert_eq!(term.parser_state, ParserState::Ground);
         // TODO: Assert keypad state if tracked
         process_str(&mut term, "\x1b>"); // DECKPNM
         assert_eq!(term.parser_state, ParserState::Ground);
         // TODO: Assert keypad state if tracked
    }

    // TODO: Add tests for EraseInDisplay (J)
    // TODO: Add tests for SGR sequences ('m')
    // TODO: Add tests for OSC sequences (']')
}

