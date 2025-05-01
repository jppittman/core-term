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
    EPOLL_CTL_ADD, EPOLLIN, EPOLL_CLOEXEC,
    EPOLLERR, EPOLLHUP,
    termios, tcsetattr, TCSAFLUSH,
    // Added for job control
    setsid, TIOCSPGRP, TIOCSCTTY,
};
use std::ffi::{CString, CStr};
use std::ptr;
use std::process;
use std::io::{self, Read};
use std::os::unix::io::{FromRawFd, RawFd, AsRawFd};
use std::mem;

use anyhow::{Result, Context, Error as AnyhowError};

extern crate x11;
use x11::xlib;
use x11::xlib::{
    XEvent, XConnectionNumber, XPending,
    XKeyEvent,
};


// Define constants
const BUFSIZ: usize = 496;
const MAX_EPOLL_EVENTS: c_int = 10;
const DEFAULT_COLS: usize = 80;
const DEFAULT_ROWS: usize = 24;
const DEFAULT_WIDTH_PX: usize = 640;
const DEFAULT_HEIGHT_PX: usize = 480;
const DEFAULT_FONT_NAME: &str = "fixed";

// Struct to hold and restore original terminal attributes
// This is primarily for the ConsoleBackend and will not be used by XBackend.
#[allow(dead_code)] // Allow dead code since this struct is not used by the current backend
struct OriginalTermios(termios);

#[allow(dead_code)] // Allow dead code since this struct is not used by the current backend
impl Drop for OriginalTermios {
    fn drop(&mut self) {
        // Restore original terminal attributes when this struct goes out of scope
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
    #[allow(dead_code)] // Keep for future X backend resize handling
    fn resize(&mut self, new_cols: usize, new_rows: usize) -> Result<()> {
        if new_cols < 1 || new_rows < 1 {
            anyhow::bail!("Terminal size must be at least 1x1");
        }

        if self.cols == new_cols && self.rows == new_rows {
            return Ok(());
        }

        self.cols = new_cols;
        self.rows = new_rows;

        // Resize the screen buffer, preserving content where possible
        let old_rows = self.screen.len();
        self.screen.resize(self.rows, vec![' '; self.cols]);
        for row_idx in 0..self.rows {
             if row_idx < old_rows {
                 self.screen[row_idx].resize(self.cols, ' ');
             }
        }

        // Update cursor position to be within new bounds
        self.cursor_x = std::cmp::min(self.cursor_x, self.cols.saturating_sub(1));
        self.cursor_y = std::cmp::min(self.cursor_y, self.rows.saturating_sub(1));

        let mut win_size = winsize {
            ws_row: self.rows as u16,
            ws_col: self.cols as u16,
            ws_xpixel: 0,
            ws_ypixel: 0,
        };

        unsafe {
            if ioctl(self.pty_parent.as_raw_fd(), TIOCSWINSZ, &mut win_size) < 0 {
                return Err(io::Error::last_os_error()).context("Failed to set window size on PTY parent");
            }
        }

        println!("Terminal resized to {}x{}", self.cols, self.rows);
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
                    b'\n' => { // Newline
                        self.cursor_y += 1;
                        self.cursor_x = 0;
                    }
                    b'\r' => { // Carriage Return
                        self.cursor_x = 0;
                    }
                    0x08 => { // Backspace (BS)
                        if self.cursor_x > 0 {
                            self.cursor_x -= 1;
                            self.screen[self.cursor_y][self.cursor_x] = ' ';
                        }
                    }
                    0x07 => { // BEL (Bell) - often used to terminate OSC
                        // Ignore BEL in Ground state
                    }
                    0..=31 | 127 => {
                        // Ignore most C0 and DEL control characters for this MVP
                    }
                    _ => {
                        // Treat as printable character
                        if self.cursor_x >= self.cols {
                            self.cursor_y += 1;
                            self.cursor_x = 0;
                        }

                        if self.cursor_y >= self.rows {
                            // Handle scrolling
                            let mut new_screen = Vec::with_capacity(self.rows);
                            for r in 1..self.rows {
                                new_screen.push(self.screen[r].clone());
                            }
                            new_screen.push(vec![' '; self.cols]);
                            self.screen = new_screen;
                            self.cursor_y = self.rows - 1;
                        }

                        self.cursor_x = std::cmp::min(self.cursor_x, self.cols.saturating_sub(1));
                        self.cursor_y = std::cmp::min(self.cursor_y, self.rows.saturating_sub(1));

                        self.screen[self.cursor_y][self.cursor_x] = byte as char;
                        self.cursor_x += 1;
                    }
                }
            }
            ParserState::Escape => {
                match byte {
                    b'[' => { // CSI
                        self.parser_state = ParserState::Csi;
                    }
                    b']' => { // OSC
                         self.parser_state = ParserState::Osc;
                         self.escape_buffer.clear();
                    }
                    _ => {
                        eprintln!("Warning: Unknown escape sequence: ESC {}", byte as char);
                        self.parser_state = ParserState::Ground;
                    }
                }
            }
            ParserState::Csi => {
                self.escape_buffer.push(byte);

                // Check for the final byte of a CSI sequence (usually >= 0x40 and <= 0x7E)
                // Or check for intermediate bytes (0x20-0x2F) or parameter bytes (0x30-0x3F)
                if byte >= 0x40 && byte <= 0x7E {
                    // Process the CSI sequence
                    match self.parse_csi_sequence() {
                        Ok(sequence) => self.handle_csi_sequence(sequence),
                        Err(e) => eprintln!("Error parsing CSI sequence: {}", e),
                    }
                    self.parser_state = ParserState::Ground;
                    self.escape_buffer.clear();
                } else if byte < 0x20 || byte > 0x7E {
                    // Invalid byte in CSI sequence, reset parser
                    eprintln!("Warning: Invalid byte in CSI sequence: {}", byte);
                    self.parser_state = ParserState::Ground;
                    self.escape_buffer.clear();
                }
                // Otherwise, it's an intermediate or parameter byte, continue collecting
            }
            ParserState::Osc => {
                // Collect bytes for the OSC sequence until a terminator (ST or BEL)
                if byte == 0x07 { // BEL
                    // Process OSC sequence (e.g., set window title - not implemented yet)
                    // println!("Received OSC sequence: {}", String::from_utf8_lossy(&self.escape_buffer)); // Optional debug
                    self.parser_state = ParserState::Ground;
                    self.escape_buffer.clear();
                } else if byte == 0x1B { // ESC (might be followed by \)
                    // Potentially the start of ST (ESC \)
                    self.escape_buffer.push(byte);
                } else if byte == b'\\' && self.escape_buffer.last() == Some(&0x1B) {
                    // Received ST (ESC \)
                    self.escape_buffer.pop(); // Remove the ESC
                    // Process OSC sequence (e.g., set window title - not implemented yet)
                    // println!("Received OSC sequence: {}", String::from_utf8_lossy(&self.escape_buffer)); // Optional debug
                    self.parser_state = ParserState::Ground;
                    self.escape_buffer.clear();
                } else {
                    // Collect other bytes in the OSC sequence
                    self.escape_buffer.push(byte);
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
        let mut params_bytes = &sequence[..sequence.len() - 1];

        // Check for private mode indicator '?'
        let is_private = if let Some((&first, rest)) = params_bytes.split_first() {
            if first == b'?' {
                params_bytes = rest; // Consume the '?'
                true
            } else {
                false
            }
        } else {
            false
        };


        let params_str = String::from_utf8_lossy(params_bytes);
        let params: Vec<usize> = params_str
            .split(';')
            .filter_map(|s| s.parse().ok())
            .collect();

        let parse_erase_param = |p: Option<&usize>| {
            match p.copied().unwrap_or(0) {
                0 => EraseDirection::ToEnd,
                1 => EraseDirection::ToStart,
                2 => EraseDirection::WholeLine,
                val => EraseDirection::Unsupported(val),
            }
        };

        match final_byte {
            b'H' => { // CSI n ; m H - Cursor Position
                let row = params.get(0).copied().unwrap_or(1);
                let col = params.get(1).copied().unwrap_or(1);
                Ok(CsiSequence::CursorPosition { row, col })
            }
            b'K' => { // CSI n K - Erase in Line
                let direction = parse_erase_param(params.get(0));
                Ok(CsiSequence::EraseInLine(direction))
            }
            b'J' => { // CSI n J - Erase in Display
                 let direction = parse_erase_param(params.get(0));
                Ok(CsiSequence::EraseInDisplay(direction))
            }
            b'A' => { // CSI n A - Cursor Up
                let n = params.get(0).copied().unwrap_or(1);
                Ok(CsiSequence::CursorUp { n })
            }
            b'B' => { // CSI n B - Cursor Down
                let n = params.get(0).copied().unwrap_or(1);
                Ok(CsiSequence::CursorDown { n })
            }
            b'C' => { // CSI n C - Cursor Forward
                let n = params.get(0).copied().unwrap_or(1);
                Ok(CsiSequence::CursorForward { n })
            }
            b'D' => { // CSI n D - Cursor Backward
                let n = params.get(0).copied().unwrap_or(1);
                Ok(CsiSequence::CursorBackward { n })
            }
            b'h' if is_private => { // CSI ? n h - Set Private Mode
                 let mode = params.get(0).copied().unwrap_or(0);
                 Ok(CsiSequence::PrivateModeSet(mode))
            }
            b'l' if is_private => { // CSI ? n l - Reset Private Mode
                 let mode = params.get(0).copied().unwrap_or(0);
                 Ok(CsiSequence::PrivateModeReset(mode))
            }
            _ => {
                eprintln!("Warning: Unknown CSI sequence: ESC [{}{}",
                          String::from_utf8_lossy(params_bytes), final_byte as char);
                Ok(CsiSequence::Unsupported(final_byte, params))
            }
        }
    }

    // Handle a parsed CSI sequence, updating terminal state
    fn handle_csi_sequence(&mut self, sequence: CsiSequence) {
        match sequence {
            CsiSequence::CursorPosition { row, col } => {
                self.cursor_y = std::cmp::min(row.saturating_sub(1), self.rows.saturating_sub(1));
                self.cursor_x = std::cmp::min(col.saturating_sub(1), self.cols.saturating_sub(1));
            }
            CsiSequence::EraseInLine(direction) => {
                if self.cursor_y >= self.rows { return; }

                match direction {
                    EraseDirection::ToEnd => {
                        for x in self.cursor_x..self.cols {
                            self.screen[self.cursor_y][x] = ' ';
                        }
                    }
                    EraseDirection::ToStart => {
                        for x in 0..=self.cursor_x.saturating_sub(1) {
                            self.screen[self.cursor_y][x] = ' ';
                        }
                         if self.cursor_x < self.cols {
                             self.screen[self.cursor_y][self.cursor_x] = ' ';
                         }
                    }
                    EraseDirection::WholeLine => {
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
                        if self.cursor_y < self.rows {
                            for x in self.cursor_x..self.cols {
                                self.screen[self.cursor_y][x] = ' ';
                            }
                        }
                        for y in self.cursor_y.saturating_add(1)..self.rows {
                            self.screen[y].fill(' ');
                        }
                    }
                    EraseDirection::ToStart => {
                         if self.cursor_y < self.rows {
                            for x in 0..=self.cursor_x.saturating_sub(1) {
                                self.screen[self.cursor_y][x] = ' ';
                            }
                             if self.cursor_x < self.cols {
                                 self.screen[self.cursor_y][self.cursor_x] = ' ';
                             }
                        }
                        for y in 0..self.cursor_y {
                             if y < self.rows {
                                self.screen[y].fill(' ');
                            }
                        }
                    }
                    EraseDirection::WholeLine => {
                        for y in 0..self.rows {
                             if y < self.rows {
                                self.screen[y].fill(' ');
                            }
                        }
                        self.cursor_x = 0;
                        self.cursor_y = 0;
                    }
                    EraseDirection::Unsupported(param) => {
                         eprintln!("Warning: Unsupported Erase in Display parameter: {}", param);
                    }
                }
            }
            CsiSequence::CursorUp { n } => {
                self.cursor_y = self.cursor_y.saturating_sub(n);
            }
            CsiSequence::CursorDown { n } => {
                self.cursor_y = std::cmp::min(self.cursor_y.saturating_add(n), self.rows.saturating_sub(1));
            }
            CsiSequence::CursorForward { n } => {
                self.cursor_x = std::cmp::min(self.cursor_x.saturating_add(n), self.cols.saturating_sub(1));
            }
            CsiSequence::CursorBackward { n } => {
                self.cursor_x = self.cursor_x.saturating_sub(n);
            }
            CsiSequence::PrivateModeSet(mode) => {
                // Basic handling for bracketed paste mode (2004)
                if mode == 2004 {
                    // println!("Debug: Set bracketed paste mode"); // Optional debug
                } else {
                    eprintln!("Warning: Unhandled Private Mode Set: {}", mode);
                }
            }
            CsiSequence::PrivateModeReset(mode) => {
                // Basic handling for bracketed paste mode (2004)
                 if mode == 2004 {
                    // println!("Debug: Reset bracketed paste mode"); // Optional debug
                } else {
                    eprintln!("Warning: Unhandled Private Mode Reset: {}", mode);
                }
            }
            CsiSequence::Unsupported(_final_byte, _params) => {}
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
// Defines the interface for different terminal display backends (Console, X, Wayland, etc.)
trait TerminalBackend {
    // Perform any backend-specific initialization
    fn init(&mut self) -> Result<()>;

    // Get a list of file descriptors the backend wants to be polled
    fn get_event_fds(&self) -> Vec<RawFd>;

    // Handle an event for a specific file descriptor managed by the backend.
    // Returns true if the main loop should exit.
    fn handle_event(&mut self, term: &mut Term, event_fd: RawFd, event_kind: u32) -> Result<bool>;

    // Draw the current state of the Term to the backend's display
    fn draw(&mut self, term: &Term) -> Result<()> ;

    // Get the current dimensions (cols, rows) from the backend
    fn get_dimensions(&self) -> (usize, usize);
}

// --- X Backend Implementation ---
// Implements the TerminalBackend trait for an X window.
struct XBackend {
    display: *mut xlib::Display,
    window: xlib::Window,
    x_fd: RawFd,
    gc: xlib::GC,
    font_struct: *mut xlib::XFontStruct,
    font_width: i32,
    font_height: i32,
    font_ascent: i32,
    font_descent: i32,
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
             font_descent: 0,
         }
     }
}

impl TerminalBackend for XBackend {
     fn init(&mut self) -> Result<()> {
         let display = unsafe { xlib::XOpenDisplay(ptr::null()) };
         if display.is_null() {
             anyhow::bail!("Failed to open X display");
         }
         self.display = display;

         let screen = unsafe { xlib::XDefaultScreen(display) };
         let root_window = unsafe { xlib::XDefaultRootWindow(display) };

         let window = unsafe {
             xlib::XCreateSimpleWindow(
                 display,
                 root_window,
                 0, 0, // x, y
                 DEFAULT_WIDTH_PX as u32, DEFAULT_HEIGHT_PX as u32, // width, height (initial size)
                 0, // border_width
                 xlib::XBlackPixel(display, screen), // border color
                 xlib::XWhitePixel(display, screen), // background color
             )
         };
         self.window = window;

         unsafe {
             xlib::XSelectInput(
                 display,
                 window,
                 xlib::ExposureMask | xlib::KeyPressMask | xlib::StructureNotifyMask, // Redraw, KeyPress, Resize
             );
         }

         unsafe { xlib::XMapWindow(display, window) };

         self.x_fd = unsafe { XConnectionNumber(display) };
         if self.x_fd < 0 {
              return Err(io::Error::last_os_error()).context("Failed to get X connection file descriptor");
         }

         // --- Load Font and Create Graphics Context (Using XLoadQueryFont) ---
         let font_name_cstring = CString::new(DEFAULT_FONT_NAME).context("Failed to create CString for font name")?;
         let font_struct = unsafe { xlib::XLoadQueryFont(display, font_name_cstring.as_ptr()) };
         if font_struct.is_null() {
             let io_err = AnyhowError::from(io::Error::last_os_error()).context(format!("Failed to load and query font: {}", DEFAULT_FONT_NAME));
             return Err(io_err);
         }
         self.font_struct = font_struct;

         // Calculate font metrics from the queried structure
         unsafe {
             self.font_width = (*self.font_struct).max_bounds.width as i32;
             self.font_height = (*self.font_struct).ascent + (*self.font_struct).descent;
             self.font_ascent = (*self.font_struct).ascent;
             self.font_descent = (*self.font_struct).descent;
         }

         let gc = unsafe { xlib::XCreateGC(display, window, 0, ptr::null_mut()) };
         if gc.is_null() {
              let io_err = AnyhowError::from(io::Error::last_os_error()).context("Failed to create Graphics Context");
              unsafe {
                  xlib::XFreeFontInfo(ptr::null_mut(), self.font_struct, 1);
              }
              return Err(io_err);
         }
         self.gc = gc;

         // Set default foreground and background colors and font in the GC
         unsafe {
             let foreground_pixel = xlib::XBlackPixel(display, screen);
             let background_pixel = xlib::XWhitePixel(display, screen);
             xlib::XSetForeground(display, gc, foreground_pixel);
             xlib::XSetBackground(display, gc, background_pixel);
             xlib::XSetFont(display, gc, (*self.font_struct).fid);
         }

         println!("X Backend initialized. Window created. Font loaded.");
         Ok(())
     }

     fn get_event_fds(&self) -> Vec<RawFd> {
         vec![self.x_fd]
     }

     fn handle_event(&mut self, term: &mut Term, event_fd: RawFd, event_kind: u32) -> Result<bool> {
         // --- Xlib Event Handling ---
         if event_fd == self.x_fd && event_kind & (EPOLLIN as u32) != 0 {
             let mut event: XEvent = unsafe { mem::zeroed() };
             while unsafe { XPending(self.display) } > 0 {
                 unsafe { xlib::XNextEvent(self.display, &mut event) };
                 unsafe {
                     match event.type_ {
                         xlib::KeyPress => {
                             // Handle key press
                             let mut key_event: XKeyEvent = From::from(event); // Convert XEvent to XKeyEvent
                             let mut buffer: [c_int; 20] = [0; 20]; // Buffer for translated string
                             let mut keysym: xlib::KeySym = 0; // KeySym value
                             // XLookupString takes a *mut XComposeStatus, we can pass null if we don't need it
                             // A real terminal would likely need to manage this state.
                             let mut compose_status: xlib::XComposeStatus = mem::zeroed();

                             // Translate the key press event into a string
                             let count = xlib::XLookupString(
                                 &mut key_event, // Pass mutable reference cast to mutable pointer
                                 buffer.as_mut_ptr() as *mut i8, // Cast to *mut i8 for the buffer
                                 buffer.len() as c_int,
                                 &mut keysym,
                                 &mut compose_status, // Pass mutable reference
                             );
                             // We are not storing compose_status in XBackend state in this version

                             if count > 0 {
                                 // Successfully translated to a string, write to PTY
                                 let byte_slice = std::slice::from_raw_parts(
                                     buffer.as_ptr() as *const u8, // Cast to *const u8 for the slice
                                     count as usize,
                                 );
                                 // Write the key press bytes to the PTY parent
                                 if libc::write(term.pty_parent.as_raw_fd(), byte_slice.as_ptr() as *const libc::c_void, count as size_t) < 0 {
                                     eprintln!("Error writing key press to PTY: {}", io::Error::last_os_error());
                                 } else {
                                     // Redraw the terminal after input is sent to the shell
                                     // The shell's output will cause a PTY read event, which will trigger a redraw.
                                     // No need to redraw immediately here unless we want local echo.
                                 }
                             } else {
                                 // No string translation (e.g., modifier keys, function keys)
                                 // You might handle specific KeySyms here for control sequences (e.g., arrow keys)
                                 // println!("KeyPress with no string: KeySym = {}", keysym); // Optional debug print
                             }
                         }
                         xlib::Expose => {
                             if event.expose.count == 0 {
                                 self.draw(term)?;
                             }
                         }
                         xlib::ConfigureNotify => {
                             // Handle resize event - update terminal dimensions and notify PTY
                             println!("X ConfigureNotify event (handling not implemented)");
                         }
                         _ => {}
                     }
                 }
             }
         }
         Ok(false)
     }

     fn draw(&mut self, term: &Term) -> Result<()> {
         unsafe {
              // Clear the entire window area
             let screen = xlib::XDefaultScreen(self.display);
             let background_pixel = xlib::XWhitePixel(self.display, screen);
             xlib::XSetForeground(self.display, self.gc, background_pixel);
             xlib::XFillRectangle(self.display, self.window, self.gc,
                                  0, 0,
                                  (term.cols * self.font_width as usize) as u32,
                                  (term.rows * self.font_height as usize) as u32);

             // Set foreground color for drawing text
             let foreground_pixel = xlib::XBlackPixel(self.display, screen);
             xlib::XSetForeground(self.display, self.gc, foreground_pixel);

             // Draw each character from the screen buffer
             for y in 0..term.rows {
                 for x in 0..term.cols {
                     let character = term.screen[y][x];
                     // Convert char to a byte slice for XDrawString
                     let mut char_bytes = [0u8; 4];
                     let char_slice = character.encode_utf8(&mut char_bytes).as_bytes();

                     // Calculate position for the character
                     let draw_x = (x * self.font_width as usize) as i32;
                     let draw_y = (y * self.font_height as usize + self.font_ascent as usize) as i32;

                     // Draw the character
                     if character != ' ' {
                         xlib::XDrawString(self.display, self.window, self.gc,
                                           draw_x, draw_y,
                                           char_slice.as_ptr() as *const i8, char_slice.len() as i32);
                     }
                 }
             }

             // --- Draw Cursor ---
             let cursor_x_px = (term.cursor_x * self.font_width as usize) as i32;
             let cursor_y_px = (term.cursor_y * self.font_height as usize) as i32;

             // Set cursor color (e.g., inverse of foreground)
             let cursor_color = xlib::XWhitePixel(self.display, screen);
             xlib::XSetForeground(self.display, self.gc, cursor_color);

             xlib::XFillRectangle(self.display, self.window, self.gc,
                                  cursor_x_px, cursor_y_px,
                                  self.font_width as u32, self.font_height as u32);

             xlib::XFlush(self.display);
         }

         Ok(())
     }

     fn get_dimensions(&self) -> (usize, usize) {
         let cols = if self.font_width > 0 { DEFAULT_WIDTH_PX / self.font_width as usize } else { DEFAULT_COLS };
         let rows = if self.font_height > 0 { DEFAULT_HEIGHT_PX / self.font_height as usize } else { DEFAULT_ROWS };
         (cols, rows)
     }
}

impl Drop for XBackend {
    fn drop(&mut self) {
        unsafe {
            if !self.display.is_null() {
                if !self.gc.is_null() {
                    xlib::XFreeGC(self.display, self.gc);
                    self.gc = ptr::null_mut();
                }
                if !self.font_struct.is_null() {
                    xlib::XFreeFontInfo(ptr::null_mut(), self.font_struct, 1);
                    self.font_struct = ptr::null_mut();
                }
                if self.window != 0 {
                    xlib::XDestroyWindow(self.display, self.window);
                    self.window = 0;
                }
                xlib::XCloseDisplay(self.display);
                self.display = ptr::null_mut();
            }
            println!("X Backend cleaned up.");
        }
    }
}

// Create PTY, fork, and exec the shell
unsafe fn create_pty_and_fork(shell_path: &CStr, shell_args: &mut [*mut i8]) -> Result<(pid_t, std::fs::File)> {
    let mut pty_parent_fd: c_int = -1;
    let mut pty_child_fd: c_int = -1;

    // Attempt to open a new pseudo-terminal pair
    let openpty_res = unsafe { openpty(&mut pty_parent_fd, &mut pty_child_fd, ptr::null_mut(), ptr::null_mut(), ptr::null_mut()) };
    if openpty_res < 0 {
        return Err(AnyhowError::from(io::Error::last_os_error()).context("Failed to open pseudo-terminal"));
    }

    // Fork the process
    let child_pid: pid_t = unsafe { fork() };

    if child_pid < 0 {
        let io_err = AnyhowError::from(io::Error::last_os_error()).context("Failed to fork process");
        unsafe { close(pty_parent_fd) };
        unsafe { close(pty_child_fd) };
        return Err(io_err);
    }

    if child_pid != 0 { // Parent process
        unsafe { close(pty_child_fd) };
        let pty_parent_file = unsafe { std::fs::File::from_raw_fd(pty_parent_fd) };
        return Ok((child_pid, pty_parent_file));
    }

    // Child process continues here

    // Close the master side of the PTY in the child
    unsafe { close(pty_parent_fd) };

    // Create a new session and process group, making the child the session leader
    // and process group leader. This is necessary for job control.
    let setsid_res = unsafe { setsid() };
    if setsid_res < 0 {
        eprintln!("Failed to create new session: {}", io::Error::last_os_error());
        unsafe { close(pty_child_fd) }; // Clean up file descriptor before exiting
        process::exit(1);
    }

    // Duplicate the slave PTY file descriptor to stdin, stdout, and stderr
    // dup2(oldfd, newfd)
    // This should happen *before* TIOCSCTTY, matching st.c.
    let dup2_stdin_res = unsafe { libc::dup2(pty_child_fd, STDIN_FILENO) };
    if dup2_stdin_res < 0 {
        eprintln!("Failed to duplicate PTY slave to stdin in child: {}", io::Error::last_os_error());
        unsafe { close(pty_child_fd) }; // Close the original slave fd on failure
        process::exit(1);
    }

    let dup2_stdout_res = unsafe { libc::dup2(pty_child_fd, STDOUT_FILENO) };
    if dup2_stdout_res < 0 {
        eprintln!("Failed to duplicate PTY slave to stdout in child: {}", io::Error::last_os_error());
        // STDIN might be duplicated, so we only close the original slave fd
        unsafe { close(pty_child_fd) };
        process::exit(1);
    }

    let dup2_stderr_res = unsafe { libc::dup2(pty_child_fd, STDERR_FILENO) };
    if dup2_stderr_res < 0 {
        eprintln!("Failed to duplicate PTY slave to stderr in child: {}", io::Error::last_os_error());
        // STDIN/STDOUT might be duplicated, so we only close the original slave fd
        unsafe { close(pty_child_fd) };
        process::exit(1);
    }

    // Make the PTY child (now STDIN_FILENO) the controlling terminal for the new session.
    // This is done by an ioctl call on the PTY child's file descriptor.
    // This call now happens *after* dup2, matching st.c's sequence.
    // Explicitly specify the type for ptr::null_mut()
    // Note: st.c calls ioctl(s, TIOCSCTTY, NULL) where s is the original slave fd.
    // After dup2, STDIN_FILENO points to the same file description as s.
    // Calling ioctl on STDIN_FILENO here should be equivalent and is more common
    // after dup2. Let's stick to STDIN_FILENO for now.
    if pty_child_fd >= 0 { // Ensure pty_child_fd is valid before calling ioctl on it
        let tiocsctty_res = unsafe { ioctl(STDIN_FILENO, TIOCSCTTY, ptr::null_mut::<c_int>()) };
        if tiocsctty_res < 0 {
                 eprintln!("Failed to set controlling terminal: {}", io::Error::last_os_error());
                 // Clean up file descriptor before exiting - pty_child_fd is already closed by dup2 if it was 0,1,2
                 // If pty_child_fd was > 2, we need to close it here.
                 if pty_child_fd > 2 {
                     unsafe { close(pty_child_fd) };
                 }
                 process::exit(1);
        }
    } else {
         eprintln!("Error: Invalid pty_child_fd before TIOCSCTTY");
         // pty_child_fd should be valid here if openpty succeeded, but defensive check
         process::exit(1);
    }


    // Close the original slave PTY file descriptor if its value was > 2.
    // If it was 0, 1, or 2, dup2 effectively closed the originals and s now points to them,
    // so we shouldn't close it again. This matches st.c's logic.
    if pty_child_fd > 2 {
        unsafe { close(pty_child_fd) };
    }


    let shell_args_ptr_const: Vec<*const i8> = shell_args.iter().map(|&s| unsafe { *s as *const i8 }).collect();
    unsafe { execvp(shell_path.as_ptr(), shell_args_ptr_const.as_ptr()) };

    eprintln!("Error executing shell: {}", io::Error::last_os_error());
    process::exit(1);
}

// Handles reading from the PTY parent and processing data.
// Returns true if the PTY was closed, false otherwise.
fn handle_pty_read(term: &mut Term, buf: &mut [u8]) -> Result<bool> {
    match term.pty_parent.read(buf) {
        Ok(0) => {
            println!("PTY closed. Shell likely exited.");
            Ok(true)
        }
        Ok(nread) => {
            let slice = &buf[..nread];
            term.process_pty_data(slice)?;
            Ok(false)
        }
        Err(ref e) if e.kind() == io::ErrorKind::Interrupted => Ok(false),
        Err(e) => return Err(AnyhowError::from(e).context("Error reading from PTY")),
    }
}

// Main function implementing the epoll event loop
fn main() -> Result<()> {
    let shell_path_str = "/bin/sh";
    let shell_path = CString::new(shell_path_str).context("Failed to create CString for shell path")?;
    let mut shell_args_c: Vec<CString> = vec![CString::new("-l").context("Failed to create CString for shell arg")?];
    let mut shell_args_ptr: Vec<*mut i8> = shell_args_c.iter_mut().map(|s| s.as_ptr() as *mut i8).collect();
    shell_args_ptr.push(ptr::null_mut());

    // Main unsafe block for raw file descriptor operations and libc calls
    unsafe {
        let (child_pid, mut pty_parent_file) = create_pty_and_fork(&shell_path, &mut shell_args_ptr)
            .context("Failed during PTY creation or shell fork/exec")?;

        let child_pid_ptr: *const pid_t = &child_pid; // Pointer to child PID
        if ioctl(pty_parent_file.as_raw_fd(), TIOCSPGRP, child_pid_ptr) < 0 {
             eprintln!("Warning: Failed to set foreground process group using ioctl(TIOCSPGRP): {}", io::Error::last_os_error());
        }

        // --- Diagnostic Read ---
        // Attempt to read any immediate output from the shell after launch.
        // This might capture error messages before the epoll loop starts.
        let mut initial_output_buf = vec![0u8; 1024];
        match pty_parent_file.read(&mut initial_output_buf) {
            Ok(0) => {
                 eprintln!("Diagnostic: PTY closed immediately after fork/exec.");
            }
            Ok(nread) => {
                 eprintln!("Diagnostic: Received initial output from shell:");
                 eprintln!("{}", String::from_utf8_lossy(&initial_output_buf[..nread]));
            }
            Err(e) => {
                 eprintln!("Diagnostic: Error reading initial output from PTY: {}", e);
            }
        }
        // --- End Diagnostic Read ---


        let mut backend: Box<dyn TerminalBackend> = Box::new(XBackend::new());

        backend.init()?;

        let (initial_cols, initial_rows) = backend.get_dimensions();
        let mut term = Term::new(child_pid, pty_parent_file, initial_cols, initial_rows);

        let pty_parent_fd = term.pty_parent.as_raw_fd();
        let backend_fds = backend.get_event_fds();

        let epoll_fd = epoll_create1(EPOLL_CLOEXEC);
        if epoll_fd < 0 {
            return Err(AnyhowError::from(io::Error::last_os_error()).context("Failed to create epoll instance"));
        }

        let mut pty_event = epoll_event { events: EPOLLIN as u32, u64: pty_parent_fd as u64 };
        if epoll_ctl(epoll_fd, EPOLL_CTL_ADD, pty_parent_fd, &mut pty_event) < 0 {
            let io_err = AnyhowError::from(io::Error::last_os_error()).context("Failed to add PTY parent to epoll");
            close(epoll_fd);
            return Err(io_err);
        }

        for &fd in &backend_fds {
            let mut backend_event = epoll_event { events: EPOLLIN as u32, u64: fd as u64 };
            if epoll_ctl(epoll_fd, EPOLL_CTL_ADD, fd, &mut backend_event) < 0 {
                 let io_err = AnyhowError::from(io::Error::last_os_error()).context(format!("Failed to add backend fd {} to epoll", fd));
                 close(epoll_fd);
                 return Err(io_err);
            }
        }

        let mut pty_buf = vec![0u8; BUFSIZ];
        let mut events: Vec<epoll_event> = vec![std::mem::zeroed(); MAX_EPOLL_EVENTS as usize];

        println!("Terminal MVP running. Type commands or press Ctrl+D to exit.");
        backend.draw(&term)?; // Initial draw

        loop {
            let num_events = epoll_wait(epoll_fd, events.as_mut_ptr(), MAX_EPOLL_EVENTS, -1);

            if num_events < 0 {
                let epoll_err = io::Error::last_os_error();
                if epoll_err.kind() == io::ErrorKind::Interrupted {
                    continue;
                }
                return Err(AnyhowError::from(epoll_err).context("Error during epoll_wait"));
            }

            let mut should_exit = false;

            for i in 0..num_events {
                let event = &events[i as usize];
                let event_fd = event.u64 as RawFd;
                let event_kind = event.events;

                // Process read events
                if event_kind & (EPOLLIN as u32) != 0 {
                    if event_fd == pty_parent_fd {
                        if handle_pty_read(&mut term, &mut pty_buf)? {
                            should_exit = true;
                            break;
                        }
                        backend.draw(&term)?; // Redraw after PTY data
                    } else if backend_fds.contains(&event_fd) {
                         if backend.handle_event(&mut term, event_fd, event_kind)? {
                             should_exit = true;
                             break;
                         }
                    }
                }

                // Process error/hang-up events
                if event_kind & ((EPOLLERR | EPOLLHUP) as u32) != 0 {
                    if event_fd == pty_parent_fd {
                        eprintln!("Error or hang-up on PTY parent file descriptor.");
                        should_exit = true;
                        break;
                    } else if backend_fds.contains(&event_fd) {
                        eprintln!("Error or hang-up on backend file descriptor {}.", event_fd);
                        should_exit = true;
                        break;
                    } else {
                         eprintln!("Received error/hang-up event for unknown file descriptor: {}", event_fd);
                         should_exit = true;
                         break;
                    }
                }
            }

            if should_exit {
                break;
            }
        }

        close(epoll_fd);

        println!("Terminal MVP exiting.");
    }

    Ok(())
}
