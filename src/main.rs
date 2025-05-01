// Rust Terminal MVP (Proof of Concept)
// Basic terminal emulation concepts: PTY setup, epoll event loop,
// screen state, resizing, backend abstraction, and basic ANSI parsing.
// This is NOT a a production-ready terminal emulator.

// Temporarily allow unused imports for constants related to TIOCSPTLCK,
// which is commented out to work around environment issues.
#[allow(unused_imports)]
use libc::{
    openpty, fork, execvp, ioctl, close, // Removed setsid, tcsetpgrp, TIOCSCTTY
    winsize, TIOCSWINSZ, TIOCSPTLCK, // Keep TIOCSPTLCK for context, but commented out
    STDIN_FILENO, STDOUT_FILENO, STDERR_FILENO,
    c_int, pid_t, size_t,
    epoll_create1, epoll_ctl, epoll_wait, epoll_event,
    EPOLL_CTL_ADD, EPOLLIN, EPOLL_CLOEXEC,
    EPOLLERR, EPOLLHUP,
    termios, tcgetattr, tcsetattr, TCSAFLUSH, // Imports for raw mode
    ICANON, ECHO, ISIG, IXON, ICRNL, OPOST, // Flags for raw mode
};
use std::ffi::{CString, CStr};
use std::ptr;
use std::process;
use std::io::{self, Read, Write};
use std::os::unix::io::{FromRawFd, RawFd, AsRawFd}; // Import AsRawFd trait
use std::mem; // For mem::zeroed

use anyhow::{Result, Context};

// --- Xlib Imports ---
// To use these, you need to add the 'x11' crate to your Cargo.toml:
// [dependencies]
// x11 = "2.18.2" // Use the latest version available
// You also need the X11 development libraries installed on your system
// (e.g., libx11-dev on Debian/Ubuntu, xorg-devel on Fedora/CentOS, or libx11 on Arch Linux).
// Ensure pkg-config can find your X11 installation (check with `pkg-config --libs x11`).
// If `pkg-config --cflags x11` is empty, you may need to manually specify include/library paths.
// Try building with:
// CFLAGS="-I/usr/include/X11" cargo build
// Or more explicitly with RUSTFLAGS:
// RUSTFLAGS="-C link-arg=-L/usr/lib -C link-arg=-lX11 -C link-arg=-lXext -C link-arg=-lXft -C link-arg=-lfontconfig -C link-arg=-lfreetype" cargo build
extern crate x11; // Link the x11 crate
use x11::xlib;
// Comment out specific unused imports for now to reduce warnings
use x11::xlib::{XEvent, XOpenDisplay, XConnectionNumber, XPending, XNextEvent, XCloseDisplay,
                 XDefaultScreen, XDefaultRootWindow, XCreateSimpleWindow, XMapWindow, XSelectInput,
                 XBlackPixel, XWhitePixel, XSetForeground, XDefaultGC, XFillRectangle, XFlush, XDestroyWindow,
                 /* XDrawString, XFontStruct, XLoadFont, XFreeFont, XTextExtents, XSetFont, XGCValues, XCreateGC, XFreeGC, XClearWindow */
                };


// Define constants
const BUFSIZ: usize = 4096;
const MAX_EPOLL_EVENTS: c_int = 10;
const DEFAULT_COLS: usize = 80;
const DEFAULT_ROWS: usize = 24;
const DEFAULT_WIDTH_PX: usize = 640; // Example window width
const DEFAULT_HEIGHT_PX: usize = 480; // Example window height

// Struct to hold and restore original terminal attributes
// This is primarily for the ConsoleBackend and will not be used by XBackend.
struct OriginalTermios(termios);

impl Drop for OriginalTermios {
    fn drop(&mut self) {
        // Restore original terminal attributes when this struct goes out of scope
        unsafe {
            if tcsetattr(STDIN_FILENO, TCSAFLUSH, &self.0) < 0 {
                // We can't return a Result from Drop, so just print an error
                eprintln!("Error restoring terminal attributes: {}", io::Error::last_os_error());
            }
        }
    }
}


// Simplified Terminal State
struct Term {
    _child_pid: pid_t, // PID of the child process (shell) - prefixed with _ as it's unused in MVP
    pty_parent: std::fs::File, // Parent side of the PTY

    // Basic screen buffer (simplified: just characters, no attributes yet)
    screen: Vec<Vec<char>>,
    cols: usize,
    rows: usize,

    // Cursor position
    cursor_x: usize,
    cursor_y: usize,

    // ANSI Parsing State (Simplified)
    parser_state: ParserState,
    escape_buffer: Vec<u8>,
}

// Simplified ANSI Parser States
#[derive(Debug, PartialEq)]
enum ParserState {
    Ground,          // Normal character processing
    Escape,          // Received ESC (0x1B)
    Csi,             // Received ESC [ (Control Sequence Introducer)
    // Add other states for different sequence types if needed (OSC, DCS, etc.)
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
            _child_pid: child_pid, // Store pid, but currently unused in MVP logic
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
    // This method is currently unused in the ConsoleBackend but is part of the
    // Term's API for future backend implementations that handle window resizing.
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
                 // Resize existing rows
                 self.screen[row_idx].resize(self.cols, ' ');
             } else {
                 // New rows are already initialized with spaces by resize above
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
            // Use as_raw_fd() from the imported trait
            // Corrected context usage
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
                        // Move cursor back, clear character, move cursor back again
                        if self.cursor_x > 0 {
                            self.cursor_x -= 1;
                            self.screen[self.cursor_y][self.cursor_x] = ' ';
                            // No second cursor move back needed for simple overwrite
                        }
                    }
                    0..=31 | 127 => {
                        // Ignore most C0 and DEL control characters for this MVP
                        // Note: 0x7F (DEL) could also be handled for deletion
                    }
                    _ => {
                        // Treat as printable character
                        if self.cursor_x >= self.cols {
                            self.cursor_y += 1;
                            self.cursor_x = 0;
                        }

                        if self.cursor_y >= self.rows {
                            // Handle scrolling by creating a new screen vector
                            let mut new_screen = Vec::with_capacity(self.rows);
                            // Copy all rows except the first one
                            for r in 1..self.rows {
                                new_screen.push(self.screen[r].clone()); // Clone the row data
                            }
                            // Add a new, empty row at the bottom
                            new_screen.push(vec![' '; self.cols]);
                            self.screen = new_screen; // Replace the old screen
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
                    // Add handling for other sequences starting with ESC if needed
                    _ => {
                        // Unknown escape sequence, return to ground state
                        eprintln!("Warning: Unknown escape sequence: ESC {}", byte as char);
                        self.parser_state = ParserState::Ground;
                    }
                }
            }
            ParserState::Csi => {
                // Collect bytes for the CSI sequence
                self.escape_buffer.push(byte);

                // Check for the final byte of a CSI sequence (usually >= 0x40 and <= 0x7E)
                if byte >= 0x40 && byte <= 0x7E {
                    // Process the CSI sequence
                    match self.parse_csi_sequence() {
                        Ok(sequence) => self.handle_csi_sequence(sequence),
                        Err(e) => eprintln!("Error parsing CSI sequence: {}", e),
                    }
                    self.parser_state = ParserState::Ground; // Return to ground state
                    self.escape_buffer.clear();
                }
                // Handle potential errors or sequences that exceed buffer size
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

        // Parse parameters (simplified: assumes semicolon-separated numbers)
        let params_str = String::from_utf8_lossy(params_bytes);
        let params: Vec<usize> = params_str
            .split(';')
            .filter_map(|s| s.parse().ok())
            .collect();

        // Helper to parse the single parameter for erase operations
        let parse_erase_param = |p: Option<&usize>| {
            match p.copied().unwrap_or(0) { // Default is 0
                0 => EraseDirection::ToEnd,
                1 => EraseDirection::ToStart,
                2 => EraseDirection::WholeLine,
                val => EraseDirection::Unsupported(val),
            }
        };


        match final_byte {
            b'H' => { // CSI n ; m H - Cursor Position
                let row = params.get(0).copied().unwrap_or(1); // Default 1
                let col = params.get(1).copied().unwrap_or(1); // Default 1
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
                let n = params.get(0).copied().unwrap_or(1); // Default 1
                Ok(CsiSequence::CursorUp { n })
            }
            b'B' => { // CSI n B - Cursor Down
                let n = params.get(0).copied().unwrap_or(1); // Default 1
                Ok(CsiSequence::CursorDown { n })
            }
            b'C' => { // CSI n C - Cursor Forward
                let n = params.get(0).copied().unwrap_or(1); // Default 1
                Ok(CsiSequence::CursorForward { n })
            }
            b'D' => { // CSI n D - Cursor Backward
                let n = params.get(0).copied().unwrap_or(1); // Default 1
                Ok(CsiSequence::CursorBackward { n })
            }
            // Add parsing for other CSI sequences here (SGR for colors, etc.)
            _ => {
                // Unknown CSI sequence
                eprintln!("Warning: Unknown CSI sequence: ESC [{}{}",
                          String::from_utf8_lossy(params_bytes), final_byte as char);
                Ok(CsiSequence::Unsupported(final_byte, params)) // Return as unsupported
            }
        }
    }

    // Handle a parsed CSI sequence, updating terminal state
    fn handle_csi_sequence(&mut self, sequence: CsiSequence) {
        match sequence {
            CsiSequence::CursorPosition { row, col } => {
                // CursorPosition is 1-indexed in ANSI, convert to 0-indexed
                self.cursor_y = std::cmp::min(row.saturating_sub(1), self.rows.saturating_sub(1));
                self.cursor_x = std::cmp::min(col.saturating_sub(1), self.cols.saturating_sub(1));
            }
            CsiSequence::EraseInLine(direction) => {
                if self.cursor_y >= self.rows { return; } // Ensure cursor is within bounds

                match direction {
                    EraseDirection::ToEnd => { // Erase from cursor to end of line
                        for x in self.cursor_x..self.cols {
                            self.screen[self.cursor_y][x] = ' ';
                        }
                    }
                    EraseDirection::ToStart => { // Erase from cursor to start of line
                        for x in 0..=self.cursor_x.saturating_sub(1) {
                            self.screen[self.cursor_y][x] = ' ';
                        }
                         // Also clear the character at the cursor position itself, as per spec
                         if self.cursor_x < self.cols {
                             self.screen[self.cursor_y][self.cursor_x] = ' ';
                         }
                    }
                    EraseDirection::WholeLine => { // Erase whole line
                        self.screen[self.cursor_y].fill(' ');
                    }
                    EraseDirection::Unsupported(param) => {
                         eprintln!("Warning: Unsupported Erase in Line parameter: {}", param);
                    }
                }
            }
            CsiSequence::EraseInDisplay(direction) => {
                match direction {
                    EraseDirection::ToEnd => { // Erase from cursor to end of display
                        // Clear from cursor to end of current line
                        if self.cursor_y < self.rows {
                            for x in self.cursor_x..self.cols {
                                self.screen[self.cursor_y][x] = ' ';
                            }
                        }
                        // Clear all lines below the current line
                        for y in self.cursor_y.saturating_add(1)..self.rows {
                            self.screen[y].fill(' ');
                        }
                    }
                    EraseDirection::ToStart => { // Erase from cursor to start of display
                         // Clear from start of current line to cursor
                         if self.cursor_y < self.rows {
                            for x in 0..=self.cursor_x.saturating_sub(1) {
                                self.screen[self.cursor_y][x] = ' ';
                            }
                             if self.cursor_x < self.cols {
                                 self.screen[self.cursor_y][self.cursor_x] = ' ';
                             }
                        }
                        // Clear all lines above the current line
                        for y in 0..self.cursor_y {
                             if y < self.rows {
                                self.screen[y].fill(' ');
                            }
                        }
                    }
                    EraseDirection::WholeLine => { // Erase whole display
                        for y in 0..self.rows {
                             if y < self.rows {
                                self.screen[y].fill(' ');
                            }
                        }
                        // Move cursor to home position (top-left)
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
            CsiSequence::Unsupported(_final_byte, _params) => { // Prefix unused variables
                // Warning already printed in parse_csi_sequence
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
    // This method is used implicitly by the AsRawFd trait implementation for File.
    // Keeping it explicit for clarity and potential future direct use.
    fn pty_parent_fd(&self) -> RawFd {
        // Use as_raw_fd() from the imported trait
        self.pty_parent.as_raw_fd()
    }
}

// --- Terminal Backend Trait ---
// Defines the interface for different terminal display backends (Console, X, Wayland, etc.)
trait TerminalBackend {
    // Perform any backend-specific initialization (e.g., open window, set up raw mode)
    fn init(&mut self) -> Result<()>;

    // Get a list of file descriptors the backend wants to be polled by the event loop
    fn get_event_fds(&self) -> Vec<RawFd>;

    // Handle an event for a specific file descriptor managed by the backend.
    // Returns true if the main loop should exit.
    fn handle_event(&mut self, term: &mut Term, event_fd: RawFd, event_kind: u32) -> Result<bool>;

    // Draw the current state of the Term to the backend's display
    fn draw(&mut self, term: &Term) -> Result<()> ;

    // Get the current dimensions (cols, rows) from the backend
    fn get_dimensions(&self) -> (usize, usize);
}

// --- Console Backend Implementation ---
// Implements the TerminalBackend trait for a standard console/TTY
struct ConsoleBackend {
    original_termios: Option<OriginalTermios>, // Store original termios settings
}

impl ConsoleBackend {
    fn new() -> Self {
        ConsoleBackend {
            original_termios: None, // Initialize as None
        }
    }

    // Helper to clear the console screen using ANSI escape codes
    fn clear_screen(&mut self) -> Result<()> { // Needs mutable self to write to stdout
        print!("\x1B[2J\x1B[H"); // ANSI escape to clear screen and move cursor home
        io::stdout().flush().context("Failed to flush stdout after clear")?;
        Ok(())
    }

    // Helper to set the cursor position on the console using ANSI escape codes
    // ANSI cursor position is 1-indexed (row, col)
    fn set_cursor_position(&mut self, row: usize, col: usize) -> Result<()> { // Needs mutable self
         print!("\x1B[{};{}H", row + 1, col + 1); // Convert 0-indexed to 1-indexed
         io::stdout().flush().context("Failed to flush stdout after set cursor")?;
         Ok(())
    }
}

impl TerminalBackend for ConsoleBackend {
    fn init(&mut self) -> Result<()> {
        // Set terminal to raw mode
        unsafe {
            let mut original_termios: termios = mem::zeroed();
            // Get current terminal attributes
            if tcgetattr(STDIN_FILENO, &mut original_termios) < 0 {
                return Err(io::Error::last_os_error()).context("Failed to get terminal attributes");
            }

            // Save original settings
            self.original_termios = Some(OriginalTermios(original_termios));

            let mut raw_termios = original_termios;
            // Disable canonical mode (ICANON), echoing (ECHO), and signal characters (ISIG)
            raw_termios.c_lflag &= !(ICANON | ECHO | ISIG);
            // Disable IXON (Ctrl+S, Ctrl+Q flow control) and ICRNL (CR to NL translation)
            raw_termios.c_iflag &= !(IXON | ICRNL);
            // Disable output processing (OPOST)
            raw_termios.c_oflag &= !(OPOST);

            // Set minimum number of input characters (VMIN) and timeout (VTIME)
            raw_termios.c_cc[libc::VMIN] = 0; // Read returns immediately if no bytes are available
            raw_termios.c_cc[libc::VTIME] = 0; // No timeout

            // Set the new terminal attributes
            if tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw_termios) < 0 {
                return Err(io::Error::last_os_error()).context("Failed to set raw terminal attributes");
            }
        }

        Ok(())
    }

    fn get_event_fds(&self) -> Vec<RawFd> {
        vec![STDIN_FILENO]
    }

    fn handle_event(&mut self, term: &mut Term, event_fd: RawFd, event_kind: u32) -> Result<bool> {
        if event_fd != STDIN_FILENO || event_kind & (EPOLLIN as u32) == 0 { // Cast EPOLLIN
             // Should not happen with this backend/event type, but handle defensively
             eprintln!("ConsoleBackend received unexpected event for fd {}", event_fd);
             return Ok(false);
        }

        let mut stdin_buf = vec![0u8; BUFSIZ];
        match io::stdin().read(&mut stdin_buf) {
            Ok(0) => {
                println!("Stdin closed. Exiting.");
                Ok(true)
            }
            Ok(nread) => {
                let slice = &stdin_buf[..nread];
                unsafe { // Still need unsafe for libc::write
                     // Use as_raw_fd() from the imported trait
                     if libc::write(term.pty_parent.as_raw_fd(), slice.as_ptr() as *const libc::c_void, nread as size_t) < 0 {
                         // Corrected context usage
                         return Err(io::Error::last_os_error()).context("Error writing stdin to PTY");
                     }
                }
                Ok(false)
            }
             Err(ref e) if e.kind() == io::ErrorKind::Interrupted => {
                Ok(false)
            }
            Err(e) => {
                // Corrected context usage
                return Err(e).context("Error reading from stdin");
            }
        }
    }

    // Draw the current state of the Term to the backend's display
    fn draw(&mut self, term: &Term) -> Result<()> {
        self.clear_screen()?; // Clear the entire screen

        // Draw each line
        for y in 0..term.rows {
            // Move cursor to the start of the line before printing
            self.set_cursor_position(y, 0)?;
            let line: String = term.screen[y].iter().collect();
            // Print the line content
            print!("{}", line);
        }

        // After drawing the whole screen, set the cursor to the correct position
        self.set_cursor_position(term.cursor_y, term.cursor_x)?;
        io::stdout().flush().context("Failed to flush stdout after drawing")?; // Flush to ensure changes are visible

        Ok(())
    }

    fn get_dimensions(&self) -> (usize, usize) {
        (DEFAULT_COLS, DEFAULT_ROWS)
    }
}

// --- X Backend Implementation ---
// Implements the TerminalBackend trait for an X window.
struct XBackend {
    // Xlib Display connection
    display: *mut xlib::Display,

    // The X window ID
    window: xlib::Window,

    // The X connection file descriptor for epoll
    x_fd: RawFd,
}

impl XBackend {
    fn new() -> Self {
        // Initialize with null/zero values
        XBackend {
            display: ptr::null_mut(),
            window: 0,
            x_fd: -1,
        }
    }
}

impl TerminalBackend for XBackend {
    fn init(&mut self) -> Result<()> {
        // Open the X display connection
        let display = unsafe { XOpenDisplay(ptr::null()) };
        if display.is_null() {
            anyhow::bail!("Failed to open X display");
        }
        self.display = display;

        let screen = unsafe { xlib::XDefaultScreen(display) };
        let root_window = unsafe { xlib::XDefaultRootWindow(display) };

        // Create a simple window
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

        // Select input events we are interested in
        unsafe {
            xlib::XSelectInput(
                display,
                window,
                xlib::ExposureMask | xlib::KeyPressMask | xlib::StructureNotifyMask, // Redraw, KeyPress, Resize
            );
        }

        // Map the window to make it visible
        unsafe { xlib::XMapWindow(display, window) };

        // Get the file descriptor for the X connection for use with epoll
        self.x_fd = unsafe { XConnectionNumber(display) };
        if self.x_fd < 0 {
             // Corrected context usage
             return Err(io::Error::last_os_error()).context("Failed to get X connection file descriptor");
        }

        println!("X Backend initialized. Window created.");
        Ok(())
    }

    fn get_event_fds(&self) -> Vec<RawFd> {
        // Return the X connection file descriptor for epoll
        vec![self.x_fd]
    }

    fn handle_event(&mut self, term: &mut Term, event_fd: RawFd, event_kind: u32) -> Result<bool> {
        // --- Xlib Event Handling Placeholder ---
        // This is where we would process X events from the X connection file descriptor.
        // For example, read key presses, handle window expose/resize events.
        if event_fd == self.x_fd && event_kind & (EPOLLIN as u32) != 0 {
            let mut event: XEvent = unsafe { mem::zeroed() };
            while unsafe { XPending(self.display) } > 0 {
                unsafe { xlib::XNextEvent(self.display, &mut event) };
                unsafe { // Unsafe block for accessing event.type_
                    match event.type_ {
                        xlib::KeyPress => {
                            // Handle key press - translate X event to bytes and write to PTY
                            // This is complex and requires keycode to keysym to character mapping
                            println!("X KeyPress event (handling not implemented)");
                            // For now, just print a message. Later, translate key and write to term.pty_parent
                            // let key_bytes = ... // Translate X event to bytes
                            // unsafe { libc::write(term.pty_parent.as_raw_fd(), key_bytes.as_ptr() as *const libc::c_void, key_bytes.len() as size_t) };
                        }
                        xlib::Expose => {
                            // Handle expose event - redraw the window content
                            // We need to draw the current state of the terminal buffer
                            self.draw(term)?; // Call the draw method
                            println!("X Expose event (handling implemented basic redraw)");
                        }
                        xlib::ConfigureNotify => {
                            // Handle resize event - update terminal dimensions and notify PTY
                            // This is complex and requires calculating cols/rows based on font size
                            // and then calling term.resize() and notifying the PTY.
                            // let new_width = event.configure2.width;
                            // let new_height = event.configure2.height;
                            // Calculate new cols/rows based on font size (not implemented yet)
                            // term.resize(new_cols, new_rows)?;
                            println!("X ConfigureNotify event (handling not implemented)");
                        }
                        _ => {
                            // Ignore other events for now
                        }
                    }
                } // End unsafe block for accessing event.type_
            }
        }
        // Return false to continue the main loop
        Ok(false)
    }

    // Draw the current state of the Term to the backend's display
    fn draw(&mut self, _term: &Term) -> Result<()> { // Prefix term with _
        // --- Xlib Drawing Placeholder ---
        // This is where we would use Xlib drawing functions (XDrawString, XFillRectangle, etc.)
        // to render the contents of term.screen to the X window.
        // This also involves handling colors and attributes (not implemented in Term yet).

        // For this basic implementation, we'll just clear the window and print a message.
        // A real implementation would draw characters from term.screen.

        unsafe {
             // Clear the window
            let screen = xlib::XDefaultScreen(self.display);
            let background_pixel = xlib::XWhitePixel(self.display, screen);
            xlib::XSetForeground(self.display, xlib::XDefaultGC(self.display, screen), background_pixel);
            xlib::XFillRectangle(self.display, self.window, xlib::XDefaultGC(self.display, screen),
                                 0, 0, // x, y
                                 DEFAULT_WIDTH_PX as u32, DEFAULT_HEIGHT_PX as u32); // width, height

            // For now, let's draw some text to indicate it's the X window
            let _text = "X Backend Window (Drawing Placeholder)"; // Prefix text with _
            let foreground_pixel = xlib::XBlackPixel(self.display, screen);
            xlib::XSetForeground(self.display, xlib::XDefaultGC(self.display, screen), foreground_pixel);
            // Note: Drawing text properly requires a font and calculating positions,
            // which is more complex. This is a very basic placeholder.
            // xlib::XDrawString(self.display, self.window, xlib::XDefaultGC(self.display, screen),
            //                    10, 20, // x, y (approximate position)
            //                    text.as_ptr() as *const i8, text.len() as i32);

            xlib::XFlush(self.display); // Ensure drawing commands are sent to the X server
        }

        println!("X Backend drawing (placeholder)."); // Console message for debugging
        Ok(())
    }

    fn get_dimensions(&self) -> (usize, usize) {
        // --- Xlib Dimensions Placeholder ---
        // Get window dimensions and calculate cols/rows based on font size.
        // (80, 24) // Default size for now
        // In a real implementation, query the window size:
        // let mut attributes: xlib::XWindowAttributes = unsafe { mem::zeroed() };
        // unsafe { xlib::XGetWindowAttributes(self.display, self.window, &mut attributes) };
        // let pixel_width = attributes.width;
        // let pixel_height = attributes.height;
        // Calculate cols/rows based on font size (not implemented yet)
        (DEFAULT_COLS, DEFAULT_ROWS) // Use default for now
    }
}

impl Drop for XBackend {
    fn drop(&mut self) {
        // Clean up Xlib resources when the XBackend is dropped
        if !self.display.is_null() {
            unsafe {
                // Destroy the window
                if self.window != 0 {
                    xlib::XDestroyWindow(self.display, self.window);
                }
                // Close the display connection
                xlib::XCloseDisplay(self.display);
            }
            println!("X Backend cleaned up.");
        }
    }
}


// Create PTY, fork, and exec the shell
unsafe fn create_pty_and_fork(shell_path: &CStr, shell_args: &mut [*mut i8]) -> Result<(pid_t, std::fs::File)> {
    let mut pty_parent_fd: c_int = -1;
    let mut pty_child_fd: c_int = -1;

    // Wrap libc calls in explicit unsafe blocks within the unsafe fn
    unsafe { // Explicit unsafe block for openpty
        if openpty(&mut pty_parent_fd, &mut pty_child_fd, ptr::null_mut(), ptr::null_mut(), ptr::null_mut()) < 0 {
            // Corrected context usage
            return Err(io::Error::last_os_error()).context("Failed to open pseudo-terminal");
        }
    }


    let _one: c_int = 1; // Prefixed with _ to mark as unused
    // Temporarily commenting out TIOCSPTLCK as it seems to be causing issues in your environment
    // In a real terminal, this is used to lock the PTY child before the shell takes over.
    // unsafe { // Explicit unsafe block for ioctl
    //     if ioctl(pty_child_fd, TIOCSPTLCK, &_one as *const c_int) < 0 { // Use _one
    //         // Corrected context usage
    //         let io_err = io::Error::last_os_error();
    //         close(pty_parent_fd); // Removed unnecessary unsafe
    //         close(pty_child_fd); // Removed unnecessary unsafe
    //         return Err(io_err).context("Failed to lock PTY child");
    //     }
    // }


    // Wrap libc calls in explicit unsafe blocks within the unsafe fn
    let child_pid: pid_t = unsafe { fork() }; // Explicit unsafe block for fork

    if child_pid < 0 {
        // Corrected context usage
        let io_err = io::Error::last_os_error();
        unsafe { close(pty_parent_fd) }; // Explicit unsafe block for close
        unsafe { close(pty_child_fd) }; // Explicit unsafe block for close
        return Err(io_err).context("Failed to fork process");
    }

    if child_pid != 0 { // Parent process
        unsafe { close(pty_child_fd) }; // Close the PTY child in the parent - Explicit unsafe block for close
        // Use FromRawFd from the imported trait
        let pty_parent_file = unsafe { std::fs::File::from_raw_fd(pty_parent_fd) }; // Explicit unsafe block for from_raw_fd
        return Ok((child_pid, pty_parent_file)); // Early return for the parent
    }

    // Child process continues here

    // --- Job Control Setup (Removed for Console Backend) ---
    // The following calls were removed because achieving reliable job control
    // when running this program within another terminal emulator proved problematic.
    // For the X backend, job control will be handled differently, if needed.

    // // Create a new session and process group, making the child the session leader
    // // and process group leader. This is necessary for job control.
    // unsafe { // Explicit unsafe block for setsid
    //     if setsid() < 0 {
    //         eprintln!("Failed to create new session: {}", io::Error::last_os_error());
    //         close(pty_parent_fd); // Clean up file descriptors before exiting - Removed unnecessary unsafe
    //         close(pty_child_fd); // Removed unnecessary unsafe
    //         process::exit(1);
    //     }
    // }

    // // Make the PTY child the controlling terminal for the new session.
    // // This is done by an ioctl call on the PTY child's file descriptor.
    // // We use STDIN_FILENO here because we've already dup2'd the PTY child
    // // to stdin, stdout, and stderr.
    // // Explicitly specify the type for ptr::null_mut()
    // unsafe { // Explicit unsafe block for ioctl
    //     if ioctl(STDIN_FILENO, TIOCSCTTY, ptr::null_mut::<c_int>()) < 0 {
    //          eprintln!("Failed to set controlling terminal: {}", io::Error::last_os_error());
    //          close(pty_parent_fd); // Clean up file descriptors before exiting - Removed unnecessary unsafe
    //          close(pty_child_fd); // Removed unnecessary unsafe
    //          process::exit(1);
    //     }
    // }

    unsafe { close(pty_parent_fd) }; // Close the PTY parent in the child - Explicit unsafe block for close


    // Temporarily commenting out TIOCSPTLCK unlock in child
    // let zero: c_int = 0;
    // unsafe { // Explicit unsafe block for ioctl
    //     if ioctl(pty_child_fd, TIOCSPTLCK, &zero as *const c_int) < 0 { // Removed unnecessary unsafe
    //          eprintln!("Failed to unlock PTY child in child: {}", io::Error::last_os_error());
    //          process::exit(1);
    //     }
    // }

    // Wrap libc calls in explicit unsafe blocks within the unsafe fn
    unsafe { // Explicit unsafe block for dup2 calls
        if libc::dup2(pty_child_fd, STDIN_FILENO) < 0 ||
           libc::dup2(pty_child_fd, STDOUT_FILENO) < 0 ||
           libc::dup2(pty_child_fd, STDERR_FILENO) < 0 {
            eprintln!("Failed to duplicate PTY child descriptor in child: {}", io::Error::last_os_error());
            close(pty_child_fd); // Wrap close in unsafe
            process::exit(1);
        }
    }


    unsafe { close(pty_child_fd) }; // Close the original PTY child descriptor - Explicit unsafe block for close

    // Corrected type for argv to match execvp signature (*const *const i8)
    let shell_args_ptr_const: Vec<*const i8> = shell_args.iter().map(|&s| s as *const i8).collect();
    // Wrap libc calls in explicit unsafe blocks within the unsafe fn
    unsafe { execvp(shell_path.as_ptr(), shell_args_ptr_const.as_ptr()) }; // Explicit unsafe block for execvp

    // If execvp returns, it failed
    eprintln!("Error executing shell: {}", io::Error::last_os_error());
    process::exit(1);

    // Note: This point should not be reached if execvp succeeds.
    // The process::exit(1) calls handle the failure case.
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
        Err(e) => return Err(e).context("Error reading from PTY"), // Corrected context usage
    }
}


// Main function implementing the epoll event loop
fn main() -> Result<()> {
    let shell_path_str = "/bin/sh";
    let shell_path = CString::new(shell_path_str).context("Failed to create CString for shell path")?; // Use .context()
    let mut shell_args_c: Vec<CString> = vec![CString::new("-l").context("Failed to create CString for shell arg")?]; // Use .context()
    let mut shell_args_ptr: Vec<*mut i8> = shell_args_c.iter_mut().map(|s| s.as_ptr() as *mut i8).collect();
    shell_args_ptr.push(ptr::null_mut());

    unsafe { // Main unsafe block for raw file descriptor operations and libc calls
        let (child_pid, pty_parent_file) = create_pty_and_fork(&shell_path, &mut shell_args_ptr)
            .context("Failed during PTY creation or shell fork/exec")?; // Use .context()

        // --- Job Control Setup (Removed for Console Backend) ---
        // The call to tcsetpgrp was removed here because it requires the PTY parent
        // to be the controlling terminal of the calling process, which is not
        // reliably the case when running within another terminal emulator.
        // Job control will be handled differently in the X backend.
        // if tcsetpgrp(pty_parent_file.as_raw_fd(), child_pid) < 0 {
        //      eprintln!("Warning: Failed to set foreground process group: {}", io::Error::last_os_error());
        // }


        // --- Backend Selection ---
        // Choose the backend to use (ConsoleBackend or XBackend)
        // let mut backend: Box<dyn TerminalBackend> = Box::new(ConsoleBackend::new()); // Use ConsoleBackend for console testing
        let mut backend: Box<dyn TerminalBackend> = Box::new(XBackend::new()); // Switch to XBackend for X development


        backend.init()?;

        let (initial_cols, initial_rows) = backend.get_dimensions();
        let mut term = Term::new(child_pid, pty_parent_file, initial_cols, initial_rows);

        // Use as_raw_fd() from the imported trait
        let pty_parent_fd = term.pty_parent.as_raw_fd();
        let backend_fds = backend.get_event_fds(); // Removed mut


        let epoll_fd = unsafe { epoll_create1(EPOLL_CLOEXEC) }; // Explicit unsafe block
        if epoll_fd < 0 {
            // Corrected context usage
            return Err(io::Error::last_os_error()).context("Failed to create epoll instance");
        }

        let mut pty_event = epoll_event { events: EPOLLIN as u32, u64: pty_parent_fd as u64 }; // Cast EPOLLIN
        unsafe { // Explicit unsafe block
            if epoll_ctl(epoll_fd, EPOLL_CTL_ADD, pty_parent_fd, &mut pty_event) < 0 {
                // Corrected context usage
                let io_err = io::Error::last_os_error();
                close(epoll_fd);
                return Err(io_err).context("Failed to add PTY parent to epoll");
            }
        }


        for &fd in &backend_fds {
            let mut backend_event = epoll_event { events: EPOLLIN as u32, u64: fd as u64 }; // Cast EPOLLIN
             unsafe { // Explicit unsafe block
                 if epoll_ctl(epoll_fd, EPOLL_CTL_ADD, fd, &mut backend_event) < 0 { // Corrected typo back to EPOOL_CTL_ADD - this should be EPOLL_CTL_ADD, fixing it now
                      let io_err = io::Error::last_os_error();
                      close(epoll_fd);
                      return Err(io_err).context(format!("Failed to add backend fd {} to epoll", fd));
                 }
             }
        }

        let mut pty_buf = vec![0u8; BUFSIZ];
        let mut events: Vec<epoll_event> = vec![std::mem::zeroed(); MAX_EPOLL_EVENTS as usize];

        println!("Terminal MVP running. Type commands or press Ctrl+D to exit.");
        backend.draw(&term)?; // Initial draw

        loop {
            let num_events = unsafe { epoll_wait(epoll_fd, events.as_mut_ptr(), MAX_EPOLL_EVENTS, -1) }; // Explicit unsafe block

            if num_events < 0 {
                let epoll_err = io::Error::last_os_error();
                if epoll_err.kind() == io::ErrorKind::Interrupted {
                    continue;
                }
                // Corrected context usage
                return Err(epoll_err).context("Error during epoll_wait");
            }

            let mut should_exit = false;

            for i in 0..num_events {
                let event = &events[i as usize];
                let event_fd = event.u64 as RawFd;
                let event_kind = event.events;

                // Use match to handle events based on file descriptor
                match event_fd {
                    fd if fd == pty_parent_fd => {
                        if event_kind & (EPOLLIN as u32) != 0 { // Cast EPOLLIN
                            if handle_pty_read(&mut term, &mut pty_buf)? {
                                should_exit = true;
                                break; // Exit event processing loop
                            }
                            backend.draw(&term)?; // Redraw after PTY data
                        }
                        if event_kind & ((EPOLLERR | EPOLLHUP) as u32) != 0 { // Cast EPOLLERR | EPOLLHUP
                            eprintln!("Error or hang-up on PTY parent file descriptor.");
                            should_exit = true;
                            break; // Exit event processing loop
                        }
                    }
                    fd if backend_fds.contains(&fd) => {
                         if event_kind & (EPOLLIN as u32) != 0 { // Cast EPOLLIN
                             if backend.handle_event(&mut term, event_fd, event_kind)? {
                                should_exit = true;
                                break; // Exit event processing loop
                            }
                            // Redraw after backend event if needed (e.g., resize)
                            // backend.draw(&term)?; // Drawing is handled within handle_event for X Expose
                         }
                         if event_kind & ((EPOLLERR | EPOLLHUP) as u32) != 0 { // Cast EPOLLERR | EPOLLHUP
                             eprintln!("Error or hang-up on backend file descriptor {}.", event_fd);
                             should_exit = true;
                             break; // Exit event processing loop
                         }
                    }
                    _ => {
                        // Unexpected file descriptor
                        eprintln!("Received event for unknown file descriptor: {}", event_fd);
                         if event_kind & ((EPOLLERR | EPOLLHUP) as u32) != 0 { // Cast EPOLLERR | EPOLLHUP
                             eprintln!("Error or hang-up on unknown file descriptor {}.", event_fd);
                             should_exit = true;
                             break; // Exit event processing loop
                         }
                    }
                }
            }

            if should_exit {
                break;
            }
        }

        unsafe { close(epoll_fd) }; // Explicit unsafe block for close
        // libc::waitpid(term.child_pid, ..., 0); // Wait for child process exit

        println!("Terminal MVP exiting.");
    } // End of unsafe block

    Ok(())
}
