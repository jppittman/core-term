// src/backends/console.rs
// Provides a ConsoleBackend that interacts with the host terminal.

// Use items from other modules in the crate
use crate::glyph::{Attributes, ColorSpec, AttrFlags}; // Use items from glyph module
use crate::backends::TerminalBackend;
use crate::Term; // Use the Term struct definition
use crate::{DEFAULT_COLS, DEFAULT_ROWS}; // Use constants from main/lib
use std::os::unix::io::{RawFd, AsRawFd};
use std::fmt::Write as FmtWrite;
use std::io::{self, Write};
use std::mem;

// Necessary imports for this file
use anyhow::{Context, Result};
use libc::{
    // Termios functions
    tcgetattr, tcsetattr, cfmakeraw, termios, TCSANOW,
    // IOCTL for size
    ioctl, winsize, TIOCGWINSZ,
    // Standard FDs
    STDIN_FILENO, STDOUT_FILENO,
    // Read/Write
    read, write,
    // Other types
    /*c_int, c_ulong,*/ c_void, size_t
};
// ** FIX: Removed unused ptr **
// use std::ptr;


// Struct to hold original terminal settings for restoration
// Kept private to this module as it's an implementation detail
struct OriginalTermios(termios);

// --- Console Backend Implementation ---
pub struct ConsoleBackend { // Make struct public
    original_termios: Option<OriginalTermios>,
}

impl ConsoleBackend {
    pub fn new() -> Self { // Make constructor public
        ConsoleBackend { original_termios: None }
    }

    // Helper to generate SGR escape sequence string (remains private)
        fn generate_sgr_sequence(current_attr: &mut Attributes, new_attr: Attributes) -> String {
        if *current_attr == new_attr { return String::new(); }

        let mut seq = String::from("\x1b[");
        let mut codes: Vec<u8> = Vec::new();

        // Check if a full reset (0) is the most efficient sequence
        if new_attr == Attributes::default() {
            codes.push(0);
        } else {
            // --- Color Changes ---
            // Check foreground
            if current_attr.fg != new_attr.fg {
                match new_attr.fg {
                    ColorSpec::Default => codes.push(39),
                    ColorSpec::Idx(idx) => {
                        if idx < 8 { codes.push(30 + idx); } // 30-37
                        else if idx < 16 { codes.push(90 + (idx - 8)); } // 90-97
                        else { codes.extend(&[38, 5, idx]); } // 38;5;idx
                    }
                    ColorSpec::Rgb(r, g, b) => codes.extend(&[38, 2, r, g, b]), // 38;2;r;g;b
                }
            }
            // Check background
            if current_attr.bg != new_attr.bg {
                 match new_attr.bg {
                    ColorSpec::Default => codes.push(49),
                    ColorSpec::Idx(idx) => {
                        if idx < 8 { codes.push(40 + idx); } // 40-47
                        else if idx < 16 { codes.push(100 + (idx - 8)); } // 100-107
                        else { codes.extend(&[48, 5, idx]); } // 48;5;idx
                    }
                    ColorSpec::Rgb(r, g, b) => codes.extend(&[48, 2, r, g, b]), // 48;2;r;g;b
                }
            }

            // --- Flag Changes ---
            // Flags to turn ON (present in new but not current)
            let flags_to_set = new_attr.flags - current_attr.flags;
            if flags_to_set.contains(AttrFlags::BOLD) { codes.push(1); }
            if flags_to_set.contains(AttrFlags::UNDERLINE) { codes.push(4); }
            if flags_to_set.contains(AttrFlags::REVERSE) { codes.push(7); }
            // Add other flags to set...

            // Flags to turn OFF (present in current but not new)
            let flags_to_reset = current_attr.flags - new_attr.flags;
            if flags_to_reset.contains(AttrFlags::BOLD) { codes.push(22); } // 22 resets bold/faint
            if flags_to_reset.contains(AttrFlags::UNDERLINE) { codes.push(24); }
            if flags_to_reset.contains(AttrFlags::REVERSE) { codes.push(27); }
            // Add other flags to reset...

            // If only flag resets occurred (no sets, no color changes)
            // and the attributes are different, we might need a full reset (0)
            // because just sending reset codes might not cover all state changes.
            // However, for simplicity now, if codes is empty but state differs,
            // we force a reset. A more optimal approach would track individual resets.
            if codes.is_empty() && *current_attr != new_attr {
                 codes.push(0); // Use reset code as a catch-all
            }
        }

        // If after all checks, no codes were generated, attributes must be equivalent
        if codes.is_empty() {
            *current_attr = new_attr; // Update state anyway
            return String::new();
        }

        // Build the final sequence string like "1;31;44"
        for (i, code) in codes.iter().enumerate() {
            // Using write! macro which is efficient for building strings
            let _ = write!(seq, "{}", code); // Ignore result for String write
            if i < codes.len() - 1 {
                seq.push(';');
            }
        }
        seq.push('m'); // Final 'm'

        *current_attr = new_attr; // Update the current state
        seq
    }
}

// ** FIX: Use correct trait path **
impl TerminalBackend for ConsoleBackend {
    fn init(&mut self) -> Result<()> {
        // SAFETY: isatty is FFI.
        if unsafe { libc::isatty(STDIN_FILENO) } != 1 { anyhow::bail!("Standard input is not a terminal."); }
        let mut termios_settings: termios = unsafe { mem::zeroed() };
        // SAFETY: tcgetattr is FFI.
        if unsafe { tcgetattr(STDIN_FILENO, &mut termios_settings) } < 0 { return Err(io::Error::last_os_error()).context("tcgetattr failed"); }
        self.original_termios = Some(OriginalTermios(termios_settings));
        let mut raw_settings = termios_settings;
        // SAFETY: cfmakeraw is FFI.
        unsafe { cfmakeraw(&mut raw_settings) };
        // SAFETY: tcsetattr is FFI.
        if unsafe { tcsetattr(STDIN_FILENO, TCSANOW, &raw_settings) } < 0 { self.original_termios = None; return Err(io::Error::last_os_error()).context("tcsetattr raw failed"); }
        print!("\x1b[?25l\x1b[2J\x1b[H"); // Hide cursor, Clear screen, Move home
        io::stdout().flush()?;
        Ok(())
     }

    fn get_event_fds(&self) -> Vec<RawFd> { vec![STDIN_FILENO] }

    fn handle_event(&mut self, term: &mut Term, event_fd: RawFd, _event_kind: u32) -> Result<bool> {
         if event_fd == STDIN_FILENO {
            let mut buf = [0u8; 64];
            // SAFETY: read is FFI.
            let nread = unsafe { read(STDIN_FILENO, buf.as_mut_ptr() as *mut c_void, buf.len()) };
            if nread < 0 { let err = io::Error::last_os_error(); if err.kind() != io::ErrorKind::Interrupted { return Err(err).context("stdin read error"); } return Ok(false); }
            if nread == 0 { return Ok(true); } // Stdin closed, exit
            // SAFETY: write is FFI.
            let pty_fd = term.pty_parent.as_raw_fd();
            if unsafe { write(pty_fd, buf.as_ptr() as *const c_void, nread as size_t) } < 0 { return Err(io::Error::last_os_error()).context("pty write error"); }
        }
        Ok(false)
     }

    // ** FIX: draw needs &mut self because generate_sgr_sequence takes &mut **
    fn draw(&mut self, term: &Term) -> Result<()> {
        let mut output_buffer = String::new();
        output_buffer.push_str("\x1b[?25l"); // Hide cursor
        output_buffer.push_str("\x1b[H");   // Move home
        let mut current_attr = Attributes::default(); // Track current attributes
        for y in 0..term.rows {
            for x in 0..term.cols {
                let glyph = term.screen[y][x];
                // ** FIX: Call the associated function correctly **
                output_buffer.push_str(&Self::generate_sgr_sequence(&mut current_attr, glyph.attr));
                output_buffer.push(glyph.c);
            }
             if y < term.rows - 1 { output_buffer.push_str("\n\r"); } // Newline + CR
        }
        output_buffer.push_str(&format!("\x1b[{};{}H", term.cursor_y + 1, term.cursor_x + 1)); // Move cursor
        output_buffer.push_str("\x1b[?25h"); // Show cursor
        print!("{}", output_buffer);
        io::stdout().flush()?;
        Ok(())
     }

    fn get_dimensions(&self) -> (usize, usize) {
        let mut winsize: winsize = unsafe { mem::zeroed() };
        // SAFETY: ioctl is FFI.
        if unsafe { ioctl(STDOUT_FILENO, TIOCGWINSZ, &mut winsize) } < 0 {
             eprintln!("Warning: ioctl TIOCGWINSZ failed: {}. Using defaults.", io::Error::last_os_error()); // Print warning
             (DEFAULT_COLS, DEFAULT_ROWS)
        } else {
            let cols = if winsize.ws_col > 0 { winsize.ws_col as usize } else { DEFAULT_COLS };
            let rows = if winsize.ws_row > 0 { winsize.ws_row as usize } else { DEFAULT_ROWS };
            (cols, rows)
        }
     }
}

impl Drop for ConsoleBackend {
    fn drop(&mut self) {
        if let Some(original) = self.original_termios.take() {
            // SAFETY: tcsetattr is FFI.
            if unsafe { tcsetattr(STDIN_FILENO, TCSANOW, &original.0) } < 0 {
                eprintln!("Error restoring terminal attributes: {}", io::Error::last_os_error());
            }
        }
        print!("\x1b[?25h"); // Ensure cursor is visible
        let _ = io::stdout().flush();
    }
}
