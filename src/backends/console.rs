// src/backends/console.rs

use crate::term::Term;
// Removed unused Glyph import
use crate::glyph::{Color, AttrFlags};
use crate::backends::{TerminalBackend, BackendEvent};

use anyhow::{Context, Result};
use std::io::{self, Write, Read, stdout, stdin};
// Removed unused AsRawFd import
use std::os::unix::io::RawFd;
// Removed unused tcgetattr import
use termios::{Termios, TCSANOW, ECHO, ICANON, ISIG, VMIN, VTIME, tcsetattr};
// Removed unused STDOUT_FILENO import
use libc::{winsize, TIOCGWINSZ, TIOCSWINSZ, STDIN_FILENO};
use std::mem;
// Removed unused min import (std::cmp::min used implicitly below)

// --- ANSI Escape Code Constants ---
const CURSOR_HIDE: &str = "\x1b[?25l";
const CURSOR_SHOW: &str = "\x1b[?25h";
const CURSOR_HOME: &str = "\x1b[H";
const SGR_PREFIX: &str = "\x1b[";
const SGR_SUFFIX: char = 'm';
const SGR_SEPARATOR: char = ';';
const SGR_RESET: u16 = 0;
const CUP_SUFFIX: char = 'H';

/// Backend that interacts directly with the controlling terminal via ANSI escape codes.
pub struct ConsoleBackend {
    original_termios: Option<Termios>,
    last_known_width: u16,
    last_known_height: u16,
}

impl TerminalBackend for ConsoleBackend {
    /// Creates a new ConsoleBackend instance.
    fn new(_width: usize, _height: usize) -> Result<Self> {
        let (initial_width, initial_height) = get_terminal_size(STDIN_FILENO)?;
        Ok(ConsoleBackend {
            original_termios: None,
            last_known_width: initial_width,
            last_known_height: initial_height,
        })
    }

    /// Sets up raw mode and potentially handles input (simplified).
    fn run(&mut self, term: &mut Term, pty_fd: RawFd) -> Result<bool> {
        // Get current terminal settings *before* modifying
        let original_termios_result = Termios::from_fd(STDIN_FILENO);
        if let Ok(original_termios) = original_termios_result {
            self.original_termios = Some(original_termios); // Store for restoration

            // Enter raw mode
            let mut raw_termios = original_termios; // Clone to modify
            raw_termios.c_lflag &= !(ECHO | ICANON | ISIG);
            raw_termios.c_cc[VMIN] = 1;
            raw_termios.c_cc[VTIME] = 0;
            tcsetattr(STDIN_FILENO, TCSANOW, &raw_termios)
                .context("Failed to set raw terminal attributes")?;
        } else {
             // Fix E0283: Explicitly convert error to anyhow::Error
             return Err(anyhow::Error::from(original_termios_result.unwrap_err())
                 .context("Failed to get initial terminal attributes"));
        }

        // Hide cursor for cleaner initial draw
        print!("{}", CURSOR_HIDE);
        stdout().flush()?;

        // --- Initial Draw ---
        term.resize(self.last_known_width as usize, self.last_known_height as usize);
        self.draw(term)?;

        // --- Simplified Event Loop ---
        let mut buf = [0u8; 128];
        loop {
            // Check for terminal resize (polling approach)
             match get_terminal_size(STDIN_FILENO) {
                 Ok((w, h)) if w != self.last_known_width || h != self.last_known_height => {
                     self.last_known_width = w;
                     self.last_known_height = h;
                     self.handle_event(BackendEvent::Resize { width_px: 0, height_px: 0 }, term, pty_fd)?;
                 }
                 Err(e) => eprintln!("Warning: Failed to get terminal size: {}", e),
                 _ => {}
             }

            // Blocking read from stdin
            match stdin().read(&mut buf) {
                Ok(0) => return Ok(true), // EOF
                Ok(n) => {
                    // Simulate Key events from raw stdin bytes
                    for byte in &buf[..n] {
                        let text = std::str::from_utf8(std::slice::from_ref(byte)).unwrap_or("");
                        self.handle_event(BackendEvent::Key { keysym: 0, text: text.to_string() }, term, pty_fd)?;
                    }
                }
                 Err(ref e) if e.kind() == io::ErrorKind::Interrupted => continue,
                 Err(e) => return Err(e.into()), // Propagate other errors
            }
        }
    }

    /// Handles backend events for the console.
    fn handle_event(&mut self, event: BackendEvent, term: &mut Term, pty_fd: RawFd) -> Result<()> {
        match event {
            BackendEvent::Key { text, .. } => {
                if !text.is_empty() {
                    let bytes_to_write = text.as_bytes();
                    let count = bytes_to_write.len();
                    let written = unsafe { libc::write(pty_fd, bytes_to_write.as_ptr() as *const libc::c_void, count) };

                    if written < 0 {
                        return Err(anyhow::Error::from(std::io::Error::last_os_error())
                            .context("ConsoleBackend: Failed to write key event to PTY"));
                    }
                    if (written as usize) != count {
                        eprintln!("ConsoleBackend: Warning: Partial write to PTY ({} out of {})", written, count);
                    }
                }
            }
            BackendEvent::Resize { .. } => {
                let (new_width, new_height) = get_terminal_size(STDIN_FILENO)?;
                self.last_known_width = new_width;
                self.last_known_height = new_height;

                term.resize(new_width as usize, new_height as usize);

                let winsz = winsize { ws_row: new_height, ws_col: new_width, ws_xpixel: 0, ws_ypixel: 0 };
                if unsafe { libc::ioctl(pty_fd, TIOCSWINSZ, &winsz) } < 0 {
                    eprintln!("ConsoleBackend: Warning: ioctl(TIOCSWINSZ) failed: {}", std::io::Error::last_os_error());
                }
                self.draw(term)?; // Redraw after resize
            }
            BackendEvent::CloseRequested | BackendEvent::FocusGained | BackendEvent::FocusLost => {}
        }
        Ok(())
    }

    /// Renders the terminal state to stdout using ANSI escape codes.
    fn draw(&mut self, term: &Term) -> Result<()> {
        let (term_width, term_height) = term.get_dimensions();
        let mut output_buffer = String::with_capacity(term_width * term_height * 5);

        output_buffer.push_str(CURSOR_HIDE);

        let mut current_fg = Color::Default;
        let mut current_bg = Color::Default;
        let mut current_flags = AttrFlags::empty();

        output_buffer.push_str(CURSOR_HOME);

        for y in 0..term_height {
            for x in 0..term_width {
                let glyph = term.get_glyph(x, y).cloned().unwrap_or_default();

                let (eff_fg, eff_bg, eff_flags) = if glyph.attr.flags.contains(AttrFlags::REVERSE) {
                    (glyph.attr.bg, glyph.attr.fg, glyph.attr.flags.difference(AttrFlags::REVERSE))
                } else {
                    (glyph.attr.fg, glyph.attr.bg, glyph.attr.flags)
                };

                let mut sgr_codes = Vec::new();
                if eff_fg != current_fg || eff_bg != current_bg || eff_flags != current_flags {
                    let needs_reset = eff_flags != current_flags ||
                                      matches!(eff_fg, Color::Default | Color::Rgb(_,_,_)) != matches!(current_fg, Color::Default | Color::Rgb(_,_,_)) ||
                                      matches!(eff_bg, Color::Default | Color::Rgb(_,_,_)) != matches!(current_bg, Color::Default | Color::Rgb(_,_,_));

                    if needs_reset || (current_flags != AttrFlags::empty() || current_fg != Color::Default || current_bg != Color::Default) {
                         sgr_codes.push(SGR_RESET);
                    }

                    Self::append_sgr_codes(&mut sgr_codes, eff_fg, eff_bg, eff_flags);

                    current_fg = eff_fg;
                    current_bg = eff_bg;
                    current_flags = eff_flags;
                }

                if !sgr_codes.is_empty() {
                    output_buffer.push_str(SGR_PREFIX);
                    let code_str = sgr_codes.iter().map(|c| c.to_string()).collect::<Vec<_>>().join(&SGR_SEPARATOR.to_string());
                    output_buffer.push_str(&code_str);
                    output_buffer.push(SGR_SUFFIX);
                }

                output_buffer.push(if glyph.c == '\0' { ' ' } else { glyph.c });
            }
        }

        let (cursor_x, cursor_y) = term.get_cursor();
        output_buffer.push_str(&Self::format_cursor_position(cursor_y + 1, cursor_x + 1));

        output_buffer.push_str(CURSOR_SHOW);

        stdout().write_all(output_buffer.as_bytes())?;
        stdout().flush()?;

        Ok(())
    }

    /// Restores original terminal settings.
    fn cleanup(&mut self) -> Result<()> {
        print!("{}", CURSOR_SHOW);
        stdout().flush()?;

        if let Some(original_termios) = self.original_termios.take() {
            tcsetattr(STDIN_FILENO, TCSANOW, &original_termios)
                .context("Failed to restore original terminal attributes")?;
        }
        Ok(())
    }
}

impl ConsoleBackend {
    /// Helper to generate SGR codes based on desired attributes.
    fn append_sgr_codes(codes: &mut Vec<u16>, fg: Color, bg: Color, flags: AttrFlags) {
        if flags != AttrFlags::empty() {
            if flags.contains(AttrFlags::BOLD) { codes.push(1); }
            if flags.contains(AttrFlags::ITALIC) { codes.push(3); }
            if flags.contains(AttrFlags::UNDERLINE) { codes.push(4); }
            if flags.contains(AttrFlags::HIDDEN) { codes.push(8); }
            if flags.contains(AttrFlags::STRIKETHROUGH) { codes.push(9); }
        }

        match fg {
            Color::Default => { if !codes.is_empty() { codes.push(39) } }
            Color::Idx(idx) => {
                if idx < 8 { codes.push(30 + idx as u16); }
                else if idx < 16 { codes.push(90 + (idx - 8) as u16); }
                else { codes.extend(&[38, 5, idx as u16]); }
            }
            Color::Rgb(r, g, b) => codes.extend(&[38, 2, r as u16, g as u16, b as u16]),
        }

        match bg {
            Color::Default => { if !codes.is_empty() { codes.push(49) } }
            Color::Idx(idx) => {
                if idx < 8 { codes.push(40 + idx as u16); }
                else if idx < 16 { codes.push(100 + (idx - 8) as u16); }
                else { codes.extend(&[48, 5, idx as u16]); }
            }
            Color::Rgb(r, g, b) => codes.extend(&[48, 2, r as u16, g as u16, b as u16]),
        }
    }

    /// Helper function to format a CUP (Cursor Position) ANSI sequence.
    fn format_cursor_position(row: usize, col: usize) -> String {
        format!("\x1b[{};{}{}", row, col, CUP_SUFFIX)
    }
}

/// Helper function to get the terminal size using ioctl.
fn get_terminal_size(fd: RawFd) -> Result<(u16, u16)> {
    unsafe {
        let mut winsz: winsize = mem::zeroed();
        if libc::ioctl(fd, TIOCGWINSZ, &mut winsz) == -1 {
            return Err(anyhow::Error::from(std::io::Error::last_os_error())
                .context("Failed to get terminal size via ioctl(TIOCGWINSZ)"));
        }
        let cols = if winsz.ws_col == 0 { 80 } else { winsz.ws_col };
        let rows = if winsz.ws_row == 0 { 24 } else { winsz.ws_row };
        Ok((cols, rows))
    }
}


impl Drop for ConsoleBackend {
    fn drop(&mut self) {
        let _ = self.cleanup();
    }
}
