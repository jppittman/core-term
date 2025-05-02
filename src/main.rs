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
use std::os::unix::io::{FromRawFd, RawFd, AsRawFd};
use std::mem;

use anyhow::{Result, Context, Error as AnyhowError};

extern crate x11;
use x11::xlib::{
    self, // Import the module itself for types like Display, Window, GC etc.
    XEvent, XNextEvent,
    XConnectionNumber, XPending, // For event loop
    // Functions used in init/drop
    XOpenDisplay, XDefaultScreen, XDefaultRootWindow, XCreateSimpleWindow,
    XMapWindow, XSelectInput, XBlackPixel, XWhitePixel, XCreateGC,
    XSetForeground, XSetBackground, XLoadQueryFont, XSetFont, XFreeFontInfo,
    XFlush, XDestroyWindow, XFreeGC, XCloseDisplay,
    // Functions/Structs used in handle_event/draw
    XKeyEvent, XLookupString, XComposeStatus, KeySym, // Keep XFontStruct
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
const ESC_ARG_SIZ: usize = 16; // **FIX: Define missing constant**

// Struct to hold and restore original terminal attributes
// This is primarily for the ConsoleBackend and will not be used by XBackend.
#[allow(dead_code)] // Allow dead code since this struct is not used by the current backend
struct OriginalTermios(termios);

#[allow(dead_code)] // Allow dead code since this struct is not used by the current backend
impl Drop for OriginalTermios {
    fn drop(&mut self) {
        // Restore original terminal attributes when this struct goes out of scope
        // SAFETY: tcsetattr is a C FFI call. Requires unsafe block.
        unsafe {
            if tcsetattr(STDIN_FILENO, TCSAFLUSH, &self.0) < 0 {
                eprintln!("Error restoring terminal attributes: {}", io::Error::last_os_error());
            }
        }
    }
}

// Simplified Terminal State
struct Term {
    _child_pid: pid_t, // PID of the child process (shell)
    pty_parent: std::fs::File, // Parent side of the PTY

    // Basic screen buffer (simplified: just characters, no attributes yet)
    screen: Vec<Vec<char>>,
    cols: usize,
    rows: usize,

    // Cursor position
    cursor_x: usize,
    cursor_y: usize,

    // ANSI Parsing State
    parser_state: ParserState,
    escape_buffer: Vec<u8>,
}

// ANSI Parser States
#[derive(Debug, PartialEq)]
enum ParserState {
    Ground,          // Normal character processing
    Escape,          // Received ESC (0x1B)
    Csi,             // Received ESC [ (Control Sequence Introducer)
    Osc,             // Received ESC ] (Operating System Command)
    // Add other states for different sequence types if needed (DCS, etc.)
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
#[derive(Debug)]
enum CsiSequence {
    CursorPosition { row: usize, col: usize }, // H
    EraseInLine(EraseDirection),              // K
    EraseInDisplay(EraseDirection),          // J
    CursorUp { n: usize },                     // A
    CursorDown { n: usize },                   // B
    CursorForward { n: usize },                // C
    CursorBackward { n: usize },               // D
    PrivateModeSet(usize),                     // ?...h
    PrivateModeReset(usize),                   // ?...l
    Unsupported(u8, Vec<usize>),               // Catch-all for unsupported sequences
}

impl Term {
    // Initialize a new terminal state
    fn new(child_pid: pid_t, pty_parent: std::fs::File, cols: usize, rows: usize) -> Self {
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
        }
    }

    // Resize the terminal
    fn resize(&mut self, new_cols: usize, new_rows: usize) -> Result<()> {
        // Add check for zero dimensions early
        if new_cols == 0 || new_rows == 0 {
            anyhow::bail!("Terminal resize dimensions must be positive (got {}x{})", new_cols, new_rows);
        }

        if self.cols == new_cols && self.rows == new_rows {
            return Ok(()); // No actual change
        }

        let _old_rows = self.screen.len(); // **FIX: Prefix unused variable**
        let old_cols = self.cols; // Need old cols for row resizing

        // Resize rows first
        self.screen.resize_with(new_rows, || vec![' '; old_cols]); // Use old_cols for new rows initially

        // Resize columns within each row
        for row in self.screen.iter_mut() {
             row.resize(new_cols, ' ');
        }


        self.cols = new_cols;
        self.rows = new_rows;


        // Update cursor position to be within new bounds
        self.cursor_x = std::cmp::min(self.cursor_x, self.cols.saturating_sub(1));
        self.cursor_y = std::cmp::min(self.cursor_y, self.rows.saturating_sub(1));

        // Prepare winsize struct for ioctl
        let mut win_size = winsize {
            ws_row: self.rows as u16,
            ws_col: self.cols as u16,
            ws_xpixel: 0, // Placeholder - update if pixel info available
            ws_ypixel: 0, // Placeholder - update if pixel info available
        };

        // Send TIOCSWINSZ ioctl to PTY master
        // SAFETY: ioctl is a C FFI call. Requires unsafe block.
        unsafe {
            if ioctl(self.pty_parent.as_raw_fd(), TIOCSWINSZ, &mut win_size) < 0 {
                return Err(io::Error::last_os_error()).context("Failed to set window size on PTY parent (TIOCSWINSZ)");
            }
        }
        Ok(())
    }

    // Process a single byte from the PTY based on parser state
    fn process_byte(&mut self, byte: u8) {
        match self.parser_state {
            ParserState::Ground => {
                match byte {
                    0x1B => { // ESC
                        self.parser_state = ParserState::Escape;
                        self.escape_buffer.clear();
                    }
                    b'\n' => { // Newline (LF)
                         if self.cursor_y == self.rows - 1 {
                            // TODO: Handle scrolling based on scroll region
                            self.cursor_y = self.rows - 1; // Clamp for now
                         } else {
                             self.cursor_y += 1;
                         }
                    }
                    b'\r' => { // Carriage Return (CR)
                        self.cursor_x = 0;
                    }
                    0x08 => { // Backspace (BS)
                        if self.cursor_x > 0 {
                            self.cursor_x -= 1;
                            // Optionally erase char: self.screen[self.cursor_y][self.cursor_x] = ' ';
                        }
                    }
                    0x07 => { // BEL (Bell)
                        // TODO: Implement bell (visual/audio)
                        // eprintln!("<BEL>"); // Debug print
                    }
                    // Ignore most C0 controls (except handled ones) and DEL
                    0..=6 | 9..=12 | 14..=31 | 127 => {
                        // eprintln!("<CTRL:{:02X}>", byte); // Debug print ignored controls
                    }
                    _ => { // Printable character
                         // Ensure cursor is within bounds before potential wrap/write
                         self.cursor_y = std::cmp::min(self.cursor_y, self.rows.saturating_sub(1));

                        // Handle line wrap if cursor is at the end
                        if self.cursor_x >= self.cols {
                             self.cursor_x = 0;
                             if self.cursor_y == self.rows - 1 {
                                // TODO: Handle scrolling
                                 self.cursor_y = self.rows - 1; // Clamp for now
                             } else {
                                 self.cursor_y += 1;
                             }
                        }

                        // Ensure cursor is still valid after potential wrap
                        self.cursor_x = std::cmp::min(self.cursor_x, self.cols.saturating_sub(1));
                        self.cursor_y = std::cmp::min(self.cursor_y, self.rows.saturating_sub(1));

                        // Write character to screen buffer
                        self.screen[self.cursor_y][self.cursor_x] = byte as char;
                        // Advance cursor
                        self.cursor_x += 1;
                    }
                }
            }
            ParserState::Escape => {
                match byte {
                    b'[' => { // CSI
                        self.parser_state = ParserState::Csi;
                        self.escape_buffer.clear(); // Clear buffer for CSI params
                    }
                    b']' => { // OSC
                         self.parser_state = ParserState::Osc;
                         self.escape_buffer.clear(); // Clear buffer for OSC string
                    }
                     // TODO: Handle other ESC sequences (like ESC ( G for charset)
                    _ => {
                        eprintln!("Warning: Unhandled sequence after ESC: ESC {}", byte as char);
                        self.parser_state = ParserState::Ground;
                    }
                }
            }
            ParserState::Csi => {
                // Collect parameter/intermediate bytes until a final byte
                self.escape_buffer.push(byte);

                // Check for final byte (0x40-0x7E)
                if byte >= 0x40 && byte <= 0x7E {
                    // Process the completed sequence
                    match self.parse_csi_sequence() {
                        Ok(sequence) => self.handle_csi_sequence(sequence),
                        Err(e) => eprintln!("Error parsing CSI sequence: {}", e),
                    }
                    self.parser_state = ParserState::Ground;
                    self.escape_buffer.clear();
                } else if byte < 0x20 || byte > 0x3F { // Invalid byte range within CSI
                    // Includes intermediate bytes (0x20-0x2F) and parameter bytes (0x30-0x3F)
                    // If it's outside 0x20-0x3F AND not a final byte, it's invalid.
                    eprintln!("Warning: Invalid byte in CSI sequence: 0x{:02X}", byte);
                    self.parser_state = ParserState::Ground;
                    self.escape_buffer.clear();
                }
                // else: Continue collecting parameter or intermediate bytes
            }
            ParserState::Osc => {
                 // Collect bytes until BEL (0x07) or ST (ESC \)
                 match byte {
                     0x07 => { // BEL terminator
                         // Process OSC: String::from_utf8_lossy(&self.escape_buffer)
                         // Example: handle_osc_sequence(&self.escape_buffer);
                         self.parser_state = ParserState::Ground;
                         self.escape_buffer.clear();
                     }
                     0x1B => { // ESC - potential start of ST
                         self.escape_buffer.push(byte);
                     }
                     b'\\' if self.escape_buffer.last() == Some(&0x1B) => { // ST (ESC \) received
                         self.escape_buffer.pop(); // Remove ESC
                         // Process OSC: String::from_utf8_lossy(&self.escape_buffer)
                         // Example: handle_osc_sequence(&self.escape_buffer);
                         self.parser_state = ParserState::Ground;
                         self.escape_buffer.clear();
                     }
                     _ => { // Collect byte
                         // Handle previous byte being ESC but current not being '\\' if necessary
                         // If the last byte was ESC, but this isn't '\', the ESC might be literal data
                         // or an invalid sequence start. For simplicity, just append.
                         self.escape_buffer.push(byte);
                         // TODO: Add check for max OSC length?
                     }
                 }
             }
        }
    }

    // Parse a completed CSI sequence into a CsiSequence enum
    fn parse_csi_sequence(&self) -> Result<CsiSequence> {
        let sequence = &self.escape_buffer;
        if sequence.is_empty() {
            anyhow::bail!("Empty CSI sequence buffer");
        }

        let final_byte = *sequence.last().unwrap();
        let params_bytes = &sequence[..sequence.len() - 1];

        let mut cursor = 0;
        let mut is_private = false;
        // Check for standard private marker '?'
        if params_bytes.get(cursor) == Some(&b'?') {
            is_private = true;
            cursor += 1;
        }
        // TODO: Check for other leader bytes like '>', '!', ' ' if needed

        let mut params: Vec<usize> = Vec::new();
        let mut current_param_start = cursor;
        while cursor < params_bytes.len() {
            if params_bytes[cursor] == b';' {
                 let param_slice = &params_bytes[current_param_start..cursor];
                 let param_str = String::from_utf8_lossy(param_slice);
                 // Default to 0 for empty params, parse otherwise
                 params.push(if param_str.is_empty() { 0 } else { param_str.parse().unwrap_or(0) });
                 current_param_start = cursor + 1;
                 // **FIX: Use defined constant**
                 if params.len() >= ESC_ARG_SIZ { break; } // Stop if max params reached
             }
            cursor += 1;
        }
         // Parse the last (or only) parameter segment
         if current_param_start <= params_bytes.len() {
             let param_slice = &params_bytes[current_param_start..params_bytes.len()];
             let param_str = String::from_utf8_lossy(param_slice);
             // Default to 0 for empty params, parse otherwise
             params.push(if param_str.is_empty() { 0 } else { param_str.parse().unwrap_or(0) });
         }

        // Helper to get param or default value
        // Defaults often depend on the specific CSI command (e.g., 1 for movement, 0 for erase)
        let get_param = |index: usize, default: usize| -> usize {
            // Adjust default for empty parameters if needed (e.g., sometimes empty means 1)
            // For now, use the parsed value (which defaults to 0 if empty) or the provided default.
            *params.get(index).unwrap_or(&default)
        };

        // Helper to parse erase direction parameter (defaults to 0)
        let parse_erase_param = |p_idx: usize| {
            match get_param(p_idx, 0) { // Default erase param is 0
                0 => EraseDirection::ToEnd,
                1 => EraseDirection::ToStart,
                2 => EraseDirection::WholeLine,
                val => EraseDirection::Unsupported(val),
            }
        };

        match final_byte {
            // Cursor Movement
            b'H' | b'f' => { // CUP / HVP (Cursor Position)
                let row = get_param(0, 1); // Default row 1
                let col = get_param(1, 1); // Default col 1
                Ok(CsiSequence::CursorPosition { row, col })
            }
            b'A' => { // CUU (Cursor Up)
                let n = get_param(0, 1); // Default 1
                Ok(CsiSequence::CursorUp { n })
            }
            b'B' => { // CUD (Cursor Down)
                let n = get_param(0, 1); // Default 1
                Ok(CsiSequence::CursorDown { n })
            }
            b'C' => { // CUF (Cursor Forward)
                let n = get_param(0, 1); // Default 1
                Ok(CsiSequence::CursorForward { n })
            }
            b'D' => { // CUB (Cursor Backward)
                let n = get_param(0, 1); // Default 1
                Ok(CsiSequence::CursorBackward { n })
            }

            // Erasing Text
            b'J' => { // ED (Erase in Display)
                Ok(CsiSequence::EraseInDisplay(parse_erase_param(0)))
            }
            b'K' => { // EL (Erase in Line)
                Ok(CsiSequence::EraseInLine(parse_erase_param(0)))
            }

            // DEC Private Modes
            b'h' if is_private => { // DECSET (Set Mode)
                 let mode = get_param(0, 0); // Param needed to know which mode
                 Ok(CsiSequence::PrivateModeSet(mode))
            }
            b'l' if is_private => { // DECRST (Reset Mode)
                 let mode = get_param(0, 0); // Param needed
                 Ok(CsiSequence::PrivateModeReset(mode))
            }

            // Add more CSI handlers here: 'm' (SGR), 'L', 'M', 'S', 'T', etc.

            _ => { // Unsupported CSI sequence
                // eprintln!("Warning: Unsupported CSI: ESC [{}{} ({:?})", String::from_utf8_lossy(params_bytes), final_byte as char, params);
                Ok(CsiSequence::Unsupported(final_byte, params))
            }
        }
    }


    // Handle a parsed CSI sequence, updating terminal state
    fn handle_csi_sequence(&mut self, sequence: CsiSequence) {
        match sequence {
            CsiSequence::CursorPosition { row, col } => {
                // VT sequence parameters are 1-based, our screen is 0-based
                self.cursor_y = std::cmp::min(row.saturating_sub(1), self.rows.saturating_sub(1));
                self.cursor_x = std::cmp::min(col.saturating_sub(1), self.cols.saturating_sub(1));
            }
            CsiSequence::EraseInLine(direction) => {
                // Ensure cursor row is valid before indexing screen
                if self.cursor_y >= self.rows { return; }

                match direction {
                    EraseDirection::ToEnd => {
                        // Clear from cursor position to the end of the line
                        for x in self.cursor_x..self.cols {
                            self.screen[self.cursor_y][x] = ' ';
                        }
                    }
                    EraseDirection::ToStart => {
                        // Clear from the start of the line up to and including the cursor position
                        // Use saturating_sub to avoid underflow if cursor_x is 0
                        let end_x = std::cmp::min(self.cursor_x, self.cols.saturating_sub(1));
                        for x in 0..=end_x {
                             self.screen[self.cursor_y][x] = ' ';
                        }
                    }
                    EraseDirection::WholeLine => {
                        // Clear the entire line the cursor is on
                        self.screen[self.cursor_y].fill(' ');
                    }
                    EraseDirection::Unsupported(param) => {
                         eprintln!("Warning: Unsupported Erase in Line parameter: {}", param);
                    }
                }
            }
             CsiSequence::EraseInDisplay(direction) => {
                 match direction {
                     EraseDirection::ToEnd => {
                         // Erase from cursor pos to end of line
                         if self.cursor_y < self.rows {
                             for x in self.cursor_x..self.cols {
                                 self.screen[self.cursor_y][x] = ' ';
                             }
                         }
                         // Erase all lines below the cursor line
                         for y in self.cursor_y.saturating_add(1)..self.rows {
                             self.screen[y].fill(' ');
                         }
                     }
                     EraseDirection::ToStart => {
                         // Erase from start of line up to cursor pos
                          if self.cursor_y < self.rows {
                             let end_x = std::cmp::min(self.cursor_x, self.cols.saturating_sub(1));
                             for x in 0..=end_x {
                                 self.screen[self.cursor_y][x] = ' ';
                             }
                         }
                         // Erase all lines above the cursor line
                         for y in 0..self.cursor_y {
                              if y < self.rows {
                                 self.screen[y].fill(' ');
                             }
                         }
                     }
                     EraseDirection::WholeLine => { // Erase entire screen
                         for y in 0..self.rows {
                              if y < self.rows {
                                 self.screen[y].fill(' ');
                             }
                         }
                         // Optionally move cursor to top-left
                         // self.cursor_x = 0;
                         // self.cursor_y = 0;
                     }
                    EraseDirection::Unsupported(param) => {
                         eprintln!("Warning: Unsupported Erase in Display parameter: {}", param);
                    }
                 }
             }
            CsiSequence::CursorUp { n } => {
                self.cursor_y = self.cursor_y.saturating_sub(n);
                 // Clamp to top row (usually 0, TODO: consider scroll region term.top)
                 self.cursor_y = std::cmp::max(self.cursor_y, 0);
            }
            CsiSequence::CursorDown { n } => {
                self.cursor_y = self.cursor_y.saturating_add(n);
                 // Clamp to bottom row (usually rows-1, TODO: consider scroll region term.bot)
                 self.cursor_y = std::cmp::min(self.cursor_y, self.rows.saturating_sub(1));
            }
            CsiSequence::CursorForward { n } => {
                self.cursor_x = self.cursor_x.saturating_add(n);
                 // Clamp to rightmost column
                 self.cursor_x = std::cmp::min(self.cursor_x, self.cols.saturating_sub(1));
            }
            CsiSequence::CursorBackward { n } => {
                self.cursor_x = self.cursor_x.saturating_sub(n);
                 // Clamp to leftmost column (0)
                 self.cursor_x = std::cmp::max(self.cursor_x, 0);
            }
             CsiSequence::PrivateModeSet(mode) => {
                 // Handle specific DECSET modes if needed
                 match mode {
                     25 => { /* DECTCEM Show cursor - TODO: Implement in backend */ }
                     1049 => { /* Enable Alternate Screen Buffer - TODO: Implement switching logic */ }
                     2004 => { /* Enable Bracketed Paste Mode - TODO: Track state */ }
                     _ => eprintln!("Warning: Unhandled Private Mode Set: ?{}h", mode),
                 }
             }
             CsiSequence::PrivateModeReset(mode) => {
                 // Handle specific DECRST modes if needed
                 match mode {
                     25 => { /* DECTCEM Hide cursor - TODO: Implement in backend */ }
                     1049 => { /* Disable Alternate Screen Buffer - TODO: Implement switching logic */ }
                     2004 => { /* Disable Bracketed Paste Mode - TODO: Track state */ }
                     _ => eprintln!("Warning: Unhandled Private Mode Reset: ?{}l", mode),
                 }
             }
            CsiSequence::Unsupported(_final_byte, _params) => {
                 // Silently ignore unsupported sequences for now
            }
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
    #[allow(dead_code)] // Keep for potential future use
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
    font_width: i32, // **NOTE: Defined as i32**
    font_height: i32,
    font_ascent: i32,
}


impl XBackend {
     fn new() -> Self {
         XBackend {
             display: ptr::null_mut(),
             window: 0,
             x_fd: -1,
             gc: ptr::null_mut(),
             font_struct: ptr::null_mut(),
             font_width: 0,
             font_height: 0,
             font_ascent: 0,
         }
     }

     // --- Private Helper Methods for Event Handling ---

     fn handle_key_press(&mut self, term: &mut Term, event: &XEvent) -> Result<()> {
        // SAFETY: Accessing union field event.key and FFI calls are unsafe.
        unsafe {
            let mut key_event: XKeyEvent = event.key; // Access union field
            let mut buffer: [u8; 32] = [0; 32]; // Buffer for translated string (use u8)
            let mut keysym: KeySym = 0;
            let mut compose_status: XComposeStatus = mem::zeroed();

            // SAFETY: XLookupString is FFI.
            let count = XLookupString(
                &mut key_event,
                buffer.as_mut_ptr() as *mut i8, // Cast buffer
                buffer.len() as c_int,
                &mut keysym,
                &mut compose_status,
            );

            // Guard: Only proceed if XLookupString returned bytes to write.
            if count <= 0 {
                // TODO: Handle special keysyms if needed (arrows, function keys, etc.)
                // Example: if keysym == xlib::XK_Left { /* write CSI D */ }
                return Ok(()); // Nothing to write (e.g., modifier key)
            }

            let byte_slice = &buffer[..count as usize];
            // Write the key press bytes to the PTY parent
            // SAFETY: write is a syscall dealing with raw FDs. Requires unsafe block.
            if libc::write(term.pty_parent.as_raw_fd(), byte_slice.as_ptr() as *const libc::c_void, count as size_t) < 0 {
                // Log error but don't necessarily stop the terminal on write failure
                eprintln!("Error writing key press to PTY: {}", io::Error::last_os_error());
            }
        } // End unsafe block
        Ok(())
    }

    fn handle_expose(&mut self, term: &Term, event: &XEvent) -> Result<()> {
        // SAFETY: Accessing union field event.expose is unsafe.
        let expose_event = unsafe { event.expose };

        // Guard: Only redraw on the *last* expose event in a series.
        if expose_event.count != 0 {
            return Ok(());
        }

        // SAFETY: draw involves FFI calls.
        if let Err(e) = self.draw(term) {
             eprintln!("Error redrawing on expose: {}", e);
             // Decide if this should return Err(e) to stop the main loop
        }
        Ok(())
    }

    fn handle_configure_notify(&mut self, term: &mut Term, event: &XEvent) -> Result<()> {
        // SAFETY: Accessing union field event.configure is unsafe.
        let configure_event = unsafe { event.configure };
        let new_pixel_width = configure_event.width;
        let new_pixel_height = configure_event.height;

        // Guard: Check for valid dimensions from the event.
        if new_pixel_width <= 0 || new_pixel_height <= 0 {
            eprintln!("Warning: Received ConfigureNotify with non-positive dimensions ({}x{})", new_pixel_width, new_pixel_height);
            return Ok(()); // Ignore invalid event
        }
        let new_pixel_width = new_pixel_width as usize;
        let new_pixel_height = new_pixel_height as usize;

        // Calculate new cols/rows based on font metrics.
        let new_cols = if self.font_width > 0 {
            (new_pixel_width / self.font_width as usize).max(1) // Ensure at least 1 col
        } else {
            // Font not loaded or invalid? Log error, use current cols as fallback.
            eprintln!("Warning: Invalid font_width ({}) during resize.", self.font_width);
            term.cols
        };
        let new_rows = if self.font_height > 0 {
           (new_pixel_height / self.font_height as usize).max(1) // Ensure at least 1 row
        } else {
            // Font not loaded or invalid? Log error, use current rows as fallback.
            eprintln!("Warning: Invalid font_height ({}) during resize.", self.font_height);
            term.rows
        };

        // Guard: Only resize if dimensions actually changed.
        if new_cols == term.cols && new_rows == term.rows {
            return Ok(()); // No change needed
        }

        println!(
            "X ConfigureNotify: Resizing from {}x{} to {}x{} ({}x{} px)",
            term.cols, term.rows, new_cols, new_rows, new_pixel_width, new_pixel_height
        );

        // Attempt resize (updates term state and sends TIOCSWINSZ).
        if let Err(e) = term.resize(new_cols, new_rows) {
            eprintln!("Error resizing terminal state: {}", e);
            // Don't attempt draw if resize failed, but continue processing other events.
            return Ok(());
        }

        // Attempt redraw after successful resize.
        // SAFETY: draw involves FFI calls.
        if let Err(e) = self.draw(term) {
            eprintln!("Error redrawing after resize: {}", e);
            // Decide if this should return Err(e) to stop the main loop
        }

        Ok(())
    }
} // End impl XBackend

impl TerminalBackend for XBackend {
     fn init(&mut self) -> Result<()> {
         // SAFETY: All Xlib calls are FFI and inherently unsafe. Requires unsafe block.
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
             if self.x_fd < 0 {
                 let err_msg = format!("Failed to get X connection file descriptor (returned {})", self.x_fd);
                 XDestroyWindow(display, window); XCloseDisplay(display);
                 return Err(io::Error::last_os_error()).context(err_msg);
             }

             let font_name_cstring = CString::new(DEFAULT_FONT_NAME).context("Failed to create CString for font name")?;
             let font_struct = XLoadQueryFont(display, font_name_cstring.as_ptr());
             if font_struct.is_null() {
                 let err_msg = format!("Failed to load font: '{}'.", DEFAULT_FONT_NAME);
                 XDestroyWindow(display, window); XCloseDisplay(display);
                 anyhow::bail!(err_msg);
             }
             self.font_struct = font_struct;

             // **FIX: Cast i16 font metrics to i32**
             if !(*self.font_struct).per_char.is_null() {
                  // Prefer max_bounds width if per_char exists
                  self.font_width = (*self.font_struct).max_bounds.width as i32;
             } else if (*self.font_struct).min_bounds.width == (*self.font_struct).max_bounds.width {
                  // Fallback for simple fixed-width fonts
                  self.font_width = (*self.font_struct).max_bounds.width as i32;
             } else {
                   // Font might not be fixed-width, use max_bounds as approximation
                   eprintln!("Warning: Font might not be truly fixed-width. Using max_bounds.width.");
                   self.font_width = (*self.font_struct).max_bounds.width as i32;
             }
             // Cast ascent and calculate height, also as i32
             self.font_ascent = (*self.font_struct).ascent as i32;
             self.font_height = ((*self.font_struct).ascent + (*self.font_struct).descent) as i32;


             // Validate calculated font dimensions
             if self.font_width <= 0 || self.font_height <= 0 {
                 let err_msg = format!("Font '{}' loaded but has invalid dimensions (W={}, H={})", DEFAULT_FONT_NAME, self.font_width, self.font_height);
                 XFreeFontInfo(ptr::null_mut(), self.font_struct, 1); // Attempt cleanup
                 XDestroyWindow(display, window);
                 XCloseDisplay(display);
                 anyhow::bail!(err_msg);
             }

             // Create Graphics Context (GC)
             let gc = XCreateGC(display, window, 0, ptr::null_mut());
             if gc.is_null() {
                  let err_msg = "Failed to create Graphics Context (GC)";
                  XFreeFontInfo(ptr::null_mut(), self.font_struct, 1); // Attempt cleanup
                  XDestroyWindow(display, window);
                  XCloseDisplay(display);
                  return Err(io::Error::last_os_error()).context(err_msg);
             }
             self.gc = gc;

             // Set default GC properties
             let foreground_pixel = XBlackPixel(display, screen);
             let background_pixel = XWhitePixel(display, screen);
             XSetForeground(display, gc, foreground_pixel);
             XSetBackground(display, gc, background_pixel);
             XSetFont(display, gc, (*self.font_struct).fid); // Set font in GC

             // Flush requests to ensure window setup is complete
             XFlush(display);
         } // End unsafe block for init
         println!("X Backend initialized. Window created. Font loaded.");
         Ok(())
     }

     fn get_event_fds(&self) -> Vec<RawFd> { vec![self.x_fd] }

     fn handle_event(&mut self, term: &mut Term, event_fd: RawFd, event_kind: u32) -> Result<bool> {
        // Guard: Check if the event is for the X file descriptor.
        if event_fd != self.x_fd { return Ok(false); }

        // Guard: Check for errors or hangup on the X connection itself.
        if event_kind & ((EPOLLERR | EPOLLHUP | EPOLLRDHUP) as u32) != 0 {
            eprintln!("Error or hang-up on X connection file descriptor (event kind: 0x{:x}).", event_kind);
            return Ok(true); // Signal main loop to exit.
        }

        // Guard: Check if there's data to read (EPOLLIN).
        if event_kind & (EPOLLIN as u32) == 0 { return Ok(false); }

        // Process all pending X events.
        // SAFETY: FFI calls within the loop require unsafe block.
        unsafe {
            while XPending(self.display) > 0 {
                let mut event: XEvent = mem::zeroed();
                // SAFETY: XNextEvent modifies the event struct passed by pointer.
                XNextEvent(self.display, &mut event);

                // **FIX: Add unsafe block for accessing event.type_**
                let event_type = event.type_; // Read type inside unsafe

                // Dispatch to helper methods based on event type.
                let result = match event_type {
                    xlib::KeyPress => self.handle_key_press(term, &event),
                    xlib::Expose => self.handle_expose(term, &event),
                    xlib::ConfigureNotify => self.handle_configure_notify(term, &event),
                    _ => Ok(()), // Ignore other event types for this MVP
                };

                // Handle potential errors from helpers
                if let Err(e) = result {
                     // **FIX: Add unsafe block for accessing event.type_ in error**
                     eprintln!("Error handling X event type {}: {}", event_type, e);
                     // Decide whether to exit or just log
                     // return Err(e); // Example: propagate error to stop main loop
                     // return Ok(true); // Example: signal graceful exit
                }
            } // End while XPending
        } // End unsafe block for X event loop

        Ok(false) // Signal main loop to continue.
    }


     fn draw(&mut self, term: &Term) -> Result<()> {
         // SAFETY: All Xlib drawing functions are FFI calls. Requires unsafe block.
         unsafe {
             // Basic check: ensure display and GC are valid
             if self.display.is_null() || self.gc.is_null() || self.window == 0 {
                  return Err(AnyhowError::msg("X Backend draw called with invalid state (display/gc/window null)"));
             }

             let screen = XDefaultScreen(self.display);
             let background_pixel = XWhitePixel(self.display, screen);
             let foreground_pixel = XBlackPixel(self.display, screen);

             // Clear the entire window area first
             XSetForeground(self.display, self.gc, background_pixel);
             let width_px = (term.cols * self.font_width as usize) as u32;
             let height_px = (term.rows * self.font_height as usize) as u32;
             XFillRectangle(self.display, self.window, self.gc, 0, 0, width_px, height_px);

             // Set foreground color for drawing text
             XSetForeground(self.display, self.gc, foreground_pixel);

             // Draw each character from the screen buffer
             for y in 0..term.rows {
                 // Simple optimization attempt: draw contiguous non-space segments?
                 // For MVP, char-by-char is fine.
                 let row_data = &term.screen[y];
                 for x in 0..term.cols {
                     let character = row_data[x];
                     if character == ' ' { continue; } // Skip drawing spaces

                     // Convert char to a byte slice for XDrawString
                     let mut char_bytes = [0u8; 4]; // Max 4 bytes for UTF-8
                     let char_slice = character.encode_utf8(&mut char_bytes).as_bytes();

                     // Calculate position for the character baseline
                     let draw_x = (x * self.font_width as usize) as i32;
                     // Draw at baseline: y * font_height + ascent
                     let draw_y = (y * self.font_height as usize + self.font_ascent as usize) as i32;

                     // Draw the character string (single char)
                     // SAFETY: XDrawString is FFI.
                     XDrawString(
                         self.display, self.window, self.gc,
                         draw_x, draw_y,
                         char_slice.as_ptr() as *const i8, // Cast to *const i8
                         char_slice.len() as i32
                     );
                 }
             } // End loop over rows

             // --- Draw Cursor --- (Simple block cursor for now)
             // Ensure cursor position is valid
             let cursor_x = std::cmp::min(term.cursor_x, term.cols.saturating_sub(1));
             let cursor_y = std::cmp::min(term.cursor_y, term.rows.saturating_sub(1));

             let cursor_x_px = (cursor_x * self.font_width as usize) as i32;
             let cursor_y_px = (cursor_y * self.font_height as usize) as i32; // Top of cursor cell

             // Use foreground color for cursor block for simple inversion effect
             // TODO: Add focus check and cursor style handling
             XSetForeground(self.display, self.gc, foreground_pixel); // Or a specific cursor color
             // SAFETY: XFillRectangle is FFI.
             XFillRectangle(self.display, self.window, self.gc,
                                  cursor_x_px, cursor_y_px,
                                  self.font_width as u32, self.font_height as u32);

             // Flush X requests to make drawing visible
             // SAFETY: XFlush is FFI.
             XFlush(self.display);
         } // End unsafe block for draw
         Ok(())
     }


     fn get_dimensions(&self) -> (usize, usize) {
         // Calculate dimensions based on current font metrics
         // Use DEFAULT dimensions as fallback if font not loaded yet or invalid
         let cols = if self.font_width > 0 { DEFAULT_WIDTH_PX.saturating_div(self.font_width as usize).max(1) } else { DEFAULT_COLS };
         let rows = if self.font_height > 0 { DEFAULT_HEIGHT_PX.saturating_div(self.font_height as usize).max(1) } else { DEFAULT_ROWS };
         (cols, rows)
     }
} // End impl TerminalBackend for XBackend

impl Drop for XBackend {
    fn drop(&mut self) {
        // SAFETY: All Xlib calls are FFI and inherently unsafe. Requires unsafe block.
        unsafe {
            if !self.display.is_null() {
                if !self.gc.is_null() {
                    // SAFETY: XFreeGC is FFI.
                    XFreeGC(self.display, self.gc);
                }
                // Font freeing handled by XCloseDisplay implicitly for XLoadQueryFont result
                if self.window != 0 {
                    // SAFETY: XDestroyWindow is FFI.
                    XDestroyWindow(self.display, self.window);
                }
                // SAFETY: XCloseDisplay is FFI.
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

    // **FIX: Wrap FFI calls in unsafe blocks**
    let openpty_res = unsafe { openpty(&mut pty_parent_fd, &mut pty_child_fd, ptr::null_mut(), ptr::null_mut(), ptr::null_mut()) };
    if openpty_res < 0 {
        return Err(AnyhowError::from(io::Error::last_os_error()).context("Failed to open pseudo-terminal (openpty)"));
    }

    // SAFETY: fork is FFI.
    let child_pid: pid_t = unsafe { fork() };

    if child_pid < 0 {
        let io_err = AnyhowError::from(io::Error::last_os_error()).context("Failed to fork process");
        // SAFETY: close is FFI.
        unsafe { close(pty_parent_fd); }
        unsafe { close(pty_child_fd); }
        return Err(io_err);
    }

    // Parent Process
    if child_pid != 0 {
        // SAFETY: close is FFI.
        unsafe { close(pty_child_fd); }
        // SAFETY: from_raw_fd is unsafe.
        let pty_parent_file = unsafe { std::fs::File::from_raw_fd(pty_parent_fd) };
        return Ok((child_pid, pty_parent_file));
    }

    // Child Process Continues Here
    // SAFETY: close is FFI.
    unsafe { close(pty_parent_fd); }

    // SAFETY: setsid is FFI.
    if unsafe { setsid() } < 0 {
        eprintln!("Child Error: Failed to create new session (setsid): {}", io::Error::last_os_error());
        // SAFETY: close is FFI.
        unsafe { close(pty_child_fd); }
        process::exit(1);
    }

    // SAFETY: dup2 is FFI.
    if unsafe { libc::dup2(pty_child_fd, STDIN_FILENO) } < 0 {
        eprintln!("Child Error: Failed to duplicate PTY slave to stdin: {}", io::Error::last_os_error());
        // SAFETY: close is FFI.
        unsafe { close(pty_child_fd); }
        process::exit(1);
    }
    // SAFETY: dup2 is FFI.
    if unsafe { libc::dup2(pty_child_fd, STDOUT_FILENO) } < 0 {
        eprintln!("Child Error: Failed to duplicate PTY slave to stdout: {}", io::Error::last_os_error());
        // SAFETY: close is FFI.
        unsafe { close(pty_child_fd); }
        process::exit(1);
    }
    // SAFETY: dup2 is FFI.
    if unsafe { libc::dup2(pty_child_fd, STDERR_FILENO) } < 0 {
        eprintln!("Child Error: Failed to duplicate PTY slave to stderr: {}", io::Error::last_os_error());
        // SAFETY: close is FFI.
        unsafe { close(pty_child_fd); }
        process::exit(1);
    }

    // SAFETY: ioctl is FFI.
    if unsafe { ioctl(STDIN_FILENO, TIOCSCTTY, ptr::null_mut::<c_int>()) } < 0 {
         eprintln!("Child Error: Failed to set controlling terminal (ioctl TIOCSCTTY): {}", io::Error::last_os_error());
          // SAFETY: close is FFI.
          if pty_child_fd > 2 { unsafe { close(pty_child_fd); } }
         process::exit(1);
    }

    if pty_child_fd > 2 {
        // SAFETY: close is FFI.
        unsafe { close(pty_child_fd); }
    }

    // Set TERM environment variable (optional but good practice)
    let term_env_var = CString::new("TERM=xterm-256color").unwrap();
    // SAFETY: putenv is FFI and into_raw leaks memory (acceptable before exec).
    unsafe { libc::putenv(term_env_var.into_raw() as *mut i8); }

    // Execute the shell
    let shell_args_ptr_const: Vec<*const i8> = shell_args.iter().map(|&p| p as *const i8).collect();
    // SAFETY: execvp is FFI.
    unsafe { execvp(shell_path.as_ptr(), shell_args_ptr_const.as_ptr()); }

    // execvp only returns on error
    eprintln!("Child Error: Failed to execute shell '{}': {}",
              shell_path.to_string_lossy(), io::Error::last_os_error());
    process::exit(1); // Exit child if exec fails
}


// Handles reading from the PTY parent and processing data.
fn handle_pty_read(term: &mut Term, buf: &mut [u8]) -> Result<bool> {
    // Note: read itself is safe as it operates on a Rust File handle
    match term.pty_parent.read(buf) {
        Ok(0) => Ok(true), // PTY closed
        Ok(nread) => {
            term.process_pty_data(&buf[..nread])?;
            Ok(false) // Continue processing
        }
        Err(ref e) if e.kind() == io::ErrorKind::WouldBlock => Ok(false), // Expected in non-blocking
        Err(ref e) if e.kind() == io::ErrorKind::Interrupted => Ok(false), // Retry
        Err(e) => {
            // EIO (os error 5) often means PTY slave closed
            eprintln!("Error reading from PTY: {} (kind: {:?})", e, e.kind());
            Err(AnyhowError::from(e).context("Error reading from PTY parent"))
        }
    }
}

// Main function implementing the epoll event loop
fn main() -> Result<()> {
    // --- Configuration ---
    let shell_path_str = "/bin/sh";
    // Use Option<&str> for cleaner NULL termination handling
    let shell_args_str_opt: &[Option<&str>] = &[Some("sh"), Some("-l"), None]; // argv[0], arg1, NULL

    // --- Prepare Shell Path and Arguments for C ---
    let shell_path = CString::new(shell_path_str)?;
    let shell_args_c: Vec<CString> = shell_args_str_opt
        .iter()
        .filter_map(|opt_s| opt_s.map(|s| CString::new(s))) // Convert Some(&str) to CString
        .collect::<Result<Vec<CString>, _>>() // Collect into Result<Vec<CString>>
        .context("Failed to create CString for shell arguments")?;

     // Create Vec<*mut i8> ending with null pointer for execvp
     let mut shell_args_ptr: Vec<*mut i8> = shell_args_c
         .iter()
         .map(|cs| cs.as_ptr() as *mut i8) // Get pointers (casting constness away for execvp)
         .collect();
     shell_args_ptr.push(ptr::null_mut()); // Add the required NULL terminator

    // --- Main Unsafe Block --- (Keep for FFI calls inside)
    // Individual unsafe calls inside still need `unsafe {}` blocks.
    unsafe {
        // --- PTY Creation and Child Process Forking ---
        // **FIX: Remove mut if pty_parent_file is not mutated later**
        let (_child_pid, pty_parent_file) = create_pty_and_fork(&shell_path, &shell_args_ptr)?;

        // Set PTY master to non-blocking mode
        let pty_parent_fd = pty_parent_file.as_raw_fd();
        // **FIX: Wrap fcntl calls in unsafe blocks**
        let flags = fcntl(pty_parent_fd, F_GETFL, 0);
        if flags < 0 || unsafe { fcntl(pty_parent_fd, F_SETFL, flags | O_NONBLOCK) } < 0 {
             eprintln!("Warning: Failed to set PTY master to non-blocking: {}", io::Error::last_os_error());
              // SAFETY: close is FFI.
              close(pty_parent_fd);
              return Err(io::Error::last_os_error().into());
        }

        // --- Backend Initialization ---
        let mut backend: Box<dyn TerminalBackend> = Box::new(XBackend::new());
        backend.init().context("Failed to initialize terminal backend")?;

        // --- Terminal State Initialization ---
        let (initial_cols, initial_rows) = backend.get_dimensions();
         if initial_cols == 0 || initial_rows == 0 {
             eprintln!("Warning: Backend returned invalid initial dimensions ({initial_cols}x{initial_rows}). Using defaults.");
             // Consider using DEFAULT_COLS/ROWS or making it fatal
         }
        let mut term = Term::new(_child_pid, pty_parent_file, initial_cols, initial_rows); // Use _child_pid

        // --- Set Initial PTY Size ---
         if let Err(e) = term.resize(initial_cols, initial_rows) {
              eprintln!("Warning: Failed to set initial PTY size: {}", e);
         }

        // --- Epoll Setup ---
        // **FIX: Wrap epoll FFI calls in unsafe blocks**
        let epoll_fd = unsafe { epoll_create1(EPOLL_CLOEXEC) };
        if epoll_fd < 0 {
            return Err(AnyhowError::from(io::Error::last_os_error()).context("Failed to create epoll instance"));
        }

        // Register PTY master FD
        let mut pty_event = epoll_event { events: (EPOLLIN | EPOLLRDHUP) as u32, u64: pty_parent_fd as u64 };
        // SAFETY: epoll_ctl is FFI.
        if epoll_ctl(epoll_fd, EPOLL_CTL_ADD, pty_parent_fd, &mut pty_event) < 0 {
            let io_err = AnyhowError::from(io::Error::last_os_error()).context("Failed to add PTY parent to epoll");
            // SAFETY: close is FFI.
            close(epoll_fd);
            return Err(io_err);
        }

        // Register backend FDs
        let backend_fds = backend.get_event_fds();
        for &fd in &backend_fds {
            let mut backend_event = epoll_event { events: (EPOLLIN | EPOLLRDHUP) as u32, u64: fd as u64 };
            // SAFETY: epoll_ctl is FFI.
            if epoll_ctl(epoll_fd, EPOLL_CTL_ADD, fd, &mut backend_event) < 0 {
                 let io_err = AnyhowError::from(io::Error::last_os_error()).context(format!("Failed to add backend fd {} to epoll", fd));
                 // SAFETY: close is FFI.
                 close(epoll_fd);
                 return Err(io_err);
            }
        }

        // --- Event Loop ---
        let mut pty_buf = vec![0u8; BUFSIZ];
        let mut events: Vec<epoll_event> = vec![mem::zeroed(); MAX_EPOLL_EVENTS as usize];

        println!("Terminal MVP running. Type commands or press Ctrl+D to exit.");
        backend.draw(&term)?; // Initial draw

        loop {
             // **FIX: Wrap epoll_wait in unsafe block**
            let num_events = epoll_wait(epoll_fd, events.as_mut_ptr(), MAX_EPOLL_EVENTS, -1);

            if num_events < 0 {
                let epoll_err = io::Error::last_os_error();
                if epoll_err.kind() == io::ErrorKind::Interrupted { continue; } // Retry on EINTR
                // SAFETY: close is FFI.
                close(epoll_fd);
                return Err(AnyhowError::from(epoll_err).context("Error during epoll_wait"));
            }

            let mut should_exit = false;

            for i in 0..num_events {
                let event = &events[i as usize];
                let event_fd = event.u64 as RawFd;
                let event_kind = event.events;

                 // Check for errors or hangup first
                 if event_kind & (EPOLLERR | EPOLLHUP | EPOLLRDHUP) as u32 != 0 {
                     if event_fd == pty_parent_fd {
                         eprintln!("PTY hang-up or error detected (event kind: 0x{:x}). Shell likely exited.", event_kind);
                         should_exit = true; break;
                     } else if backend_fds.contains(&event_fd) {
                         eprintln!("Backend hang-up or error detected on fd {} (event kind: 0x{:x}).", event_fd, event_kind);
                         should_exit = true; break;
                     } else {
                          eprintln!("Error/hang-up on unexpected fd {} (event kind: 0x{:x}).", event_fd, event_kind);
                          should_exit = true; break; // Treat unexpected errors as fatal
                     }
                 }

                 // Process readable events if no error occurred on this FD
                 if event_kind & EPOLLIN as u32 != 0 {
                    if event_fd == pty_parent_fd {
                        // Simple: Try one read per notification in non-blocking mode
                        match handle_pty_read(&mut term, &mut pty_buf) {
                            Ok(true) => { should_exit = true; break; } // PTY closed
                            Ok(false) => { /* Read ok or WouldBlock, continue */ }
                            Err(e) => {
                                eprintln!("Fatal error during PTY read: {}", e);
                                should_exit = true; break; // Exit on read error
                            }
                        }
                         // Redraw after processing PTY data
                         if let Err(e) = backend.draw(&term) {
                              eprintln!("Error drawing after PTY read: {}", e);
                               // Consider if this should be fatal
                         }
                    } else if backend_fds.contains(&event_fd) {
                         // Handle backend event
                         match backend.handle_event(&mut term, event_fd, event_kind) {
                             Ok(true) => { should_exit = true; break; } // Backend requested exit
                             Ok(false) => { /* Continue loop */ }
                             Err(e) => {
                                 eprintln!("Fatal error during backend event handling: {}", e);
                                 should_exit = true; break; // Exit on backend error
                             }
                         }
                    }
                    // else: Readable event on unexpected FD? Ignore.
                 }
            } // End loop over num_events

            if should_exit { break; } // Exit the main loop
        } // End main loop

        // --- Cleanup ---
        // SAFETY: close is FFI.
        close(epoll_fd);
        // Backend cleanup via Drop
        // PTY parent file cleanup via Drop
        // Term cleanup via Drop (if any custom drop needed)

        println!("Terminal MVP exiting.");
    } // End main unsafe block

    Ok(())
}

// Placeholder for NULL argument CString conversion
const NULL: Option<&str> = None; // Use Option<&str> directly
