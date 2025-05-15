// src/backends/console.rs

// Added use for log::error and removed unused NamedColor
use crate::glyph::{Color, AttrFlags};
// Updated to use new structs from backends::mod_rs
// Also importing the default char dimension constants from the parent module
use crate::backends::{
    Driver, BackendEvent, CellCoords, TextRunStyle, CellRect,
    DEFAULT_WINDOW_WIDTH_CHARS, DEFAULT_WINDOW_HEIGHT_CHARS
};


use anyhow::{Context, Result};
use std::io::{self, Write, Read, stdout, stdin};
use std::os::unix::io::RawFd;
// Removed unused tcgetattr
use termios::{Termios, TCSANOW, ECHO, ICANON, ISIG, VMIN, VTIME, tcsetattr};
use libc::{winsize, TIOCGWINSZ, STDIN_FILENO}; // Removed TIOCSWINSZ as PTY resizing is orchestrator's job
use std::mem;

// Import the error macro from the log crate
use log::{debug, info, warn, trace, error};


// --- ANSI Escape Code Constants ---
const CURSOR_HIDE: &str = "\x1b[?25l";
const CURSOR_SHOW: &str = "\x1b[?25h";
// const CURSOR_HOME: &str = "\x1b[H"; // Not used directly by primitives, Renderer controls cursor pos
const SGR_PREFIX: &str = "\x1b[";
const SGR_SUFFIX: char = 'm';
const SGR_SEPARATOR: char = ';';
const SGR_RESET_ALL: u16 = 0; // Renamed for clarity
const CLEAR_SCREEN_AND_HOME: &str = "\x1b[2J\x1b[H"; // Clears screen and moves cursor to home

// Default "font" dimensions for console, used for get_font_dimensions
// and to calculate "pixel" dimensions for get_display_dimensions_pixels.
// These are arbitrary placeholders as console doesn't have true pixels.
const DEFAULT_CONSOLE_FONT_WIDTH_PX: u16 = 8;
const DEFAULT_CONSOLE_FONT_HEIGHT_PX: u16 = 16;


/// Driver that interacts directly with the controlling terminal via ANSI escape codes.
pub struct ConsoleDriver {
    original_termios: Option<Termios>,
    last_known_width_cells: u16,
    last_known_height_cells: u16,
    // Assumed pixel dimensions for a character cell.
    font_width_px: u16,
    font_height_px: u16,
    // Buffer for reading from stdin
    input_buffer: [u8; 128],
}

impl Driver for ConsoleDriver {
    /// Creates a new `ConsoleDriver` instance.
    /// It attempts to set the terminal to raw mode and queries initial dimensions.
    fn new() -> Result<Self> {
        info!("Creating new ConsoleDriver.");
        let original_termios = match Termios::from_fd(STDIN_FILENO) {
            Ok(ts) => Some(ts),
            Err(e) => {
                warn!("Failed to get initial terminal attributes: {}. Proceeding without raw mode or restore.", e);
                None
            }
        };

        if let Some(ref ots) = original_termios {
            let mut raw_termios = *ots; // Clone to modify
            // Set terminal to raw mode: no echo, no canonical mode (line buffering),
            // no signal generation (Ctrl-C). VMIN=0, VTIME=0 for non-blocking read.
            raw_termios.c_lflag &= !(ECHO | ICANON | ISIG);
            raw_termios.c_iflag &= !(libc::IXON | libc::IXOFF | libc::ICRNL | libc::INLCR | libc::IGNCR);
            raw_termios.c_oflag &= !libc::OPOST;
            raw_termios.c_cc[VMIN] = 0;  // Non-blocking read: return immediately
            raw_termios.c_cc[VTIME] = 0; // No timeout
            tcsetattr(STDIN_FILENO, TCSANOW, &raw_termios)
                .context("ConsoleDriver: Failed to set raw terminal attributes")?;
            debug!("ConsoleDriver: Terminal set to raw mode.");
        }

        // Hide cursor
        // Note: print! and stdout().flush() can error.
        print!("{}", CURSOR_HIDE);
        stdout().flush().context("ConsoleDriver: Failed to flush stdout for CURSOR_HIDE")?;

        let (initial_width, initial_height) = get_terminal_size_cells(STDIN_FILENO)
            .context("ConsoleDriver: Failed to get initial terminal size")?;
        info!("ConsoleDriver: Initial terminal size: {}x{} cells.", initial_width, initial_height);

        Ok(ConsoleDriver {
            original_termios,
            last_known_width_cells: initial_width,
            last_known_height_cells: initial_height,
            font_width_px: DEFAULT_CONSOLE_FONT_WIDTH_PX,
            font_height_px: DEFAULT_CONSOLE_FONT_HEIGHT_PX,
            input_buffer: [0u8; 128],
        })
    }

    /// Returns `STDIN_FILENO` to be monitored for input events.
    fn get_event_fd(&self) -> Option<RawFd> {
        Some(STDIN_FILENO)
    }

    /// Processes pending input events from stdin and checks for terminal resize.
    fn process_events(&mut self) -> Result<Vec<BackendEvent>> {
        let mut backend_events = Vec::new();

        // 1. Check for terminal resize by polling current size.
        match get_terminal_size_cells(STDIN_FILENO) {
            Ok((current_width_cells, current_height_cells)) => {
                if current_width_cells != self.last_known_width_cells || current_height_cells != self.last_known_height_cells {
                    info!(
                        "ConsoleDriver: Terminal resized from {}x{} to {}x{} cells.",
                        self.last_known_width_cells, self.last_known_height_cells,
                        current_width_cells, current_height_cells
                    );
                    self.last_known_width_cells = current_width_cells;
                    self.last_known_height_cells = current_height_cells;

                    // Calculate "pixel" dimensions for the BackendEvent::Resize
                    let width_px = current_width_cells.saturating_mul(self.font_width_px);
                    let height_px = current_height_cells.saturating_mul(self.font_height_px);

                    backend_events.push(BackendEvent::Resize { width_px, height_px });
                }
            }
            Err(e) => {
                warn!("ConsoleDriver: Failed to get terminal size during event processing: {}. Last known size used.", e);
                // Continue with last known size, but log the error.
            }
        }

        // 2. Process keyboard input from stdin.
        // Since VMIN=0, VTIME=0, read should be non-blocking.
        // Orchestrator (main.rs) calls this when epoll signals STDIN_FILENO is ready.
        match stdin().read(&mut self.input_buffer) {
            Ok(0) => {
                // EOF on stdin, could mean the terminal was closed or input stream ended.
                info!("ConsoleDriver: EOF received on stdin. Requesting close.");
                backend_events.push(BackendEvent::CloseRequested);
            }
            Ok(bytes_read) => {
                trace!("ConsoleDriver: Read {} bytes from stdin.", bytes_read);
                for i in 0..bytes_read {
                    let byte = self.input_buffer[i];
                    // Very basic input handling: treat each byte as a character.
                    // TODO: Implement proper escape sequence parsing for arrow keys, function keys, etc.
                    // For now, keysym is 0, and text is the char.
                    let text = String::from_utf8_lossy(&[byte]).to_string();
                    trace!("ConsoleDriver: Processed byte {} as char '{}'", byte, text);
                    backend_events.push(BackendEvent::Key { keysym: 0, text });
                }
            }
            Err(ref e) if e.kind() == io::ErrorKind::WouldBlock => {
                // No data available to read, which is fine for non-blocking.
                trace!("ConsoleDriver: stdin read WouldBlock, no input to process.");
            }
            Err(ref e) if e.kind() == io::ErrorKind::Interrupted => {
                // Interrupted by a signal, try again later.
                trace!("ConsoleDriver: stdin read Interrupted.");
            }
            Err(e) => {
                // Other read error.
                return Err(e).context("ConsoleDriver: Error reading from stdin");
            }
        }
        Ok(backend_events)
    }

    /// Returns the estimated fixed pixel dimensions of a single character cell.
    fn get_font_dimensions(&self) -> (usize, usize) {
        (self.font_width_px as usize, self.font_height_px as usize)
    }

    /// Returns the estimated total display dimensions in "pixels".
    /// This is calculated from cell dimensions multiplied by estimated font dimensions.
    fn get_display_dimensions_pixels(&self) -> (u16, u16) {
        let width_px = self.last_known_width_cells.saturating_mul(self.font_width_px);
        let height_px = self.last_known_height_cells.saturating_mul(self.font_height_px);
        (width_px, height_px)
    }

    /// Clears the entire display area using ANSI escape codes.
    /// Sets the background color if specified (not fully supported by basic clear).
    fn clear_all(&mut self, bg: Color) -> Result<()> {
        let mut cmd = String::new();
        // Standard clear screen and move cursor to home.
        cmd.push_str(CLEAR_SCREEN_AND_HOME);

        // Attempt to set background color for the cleared screen.
        // Note: \x1b[2J typically clears with current background.
        // To ensure a specific background, we might need to apply SGR after clear.
        if bg != Color::Default {
            let mut sgr_codes = Vec::new();
            Self::sgr_append_bg_color(&mut sgr_codes, bg);
            if !sgr_codes.is_empty() {
                cmd.push_str(SGR_PREFIX);
                cmd.push_str(&sgr_codes.iter().map(|c| c.to_string()).collect::<Vec<_>>().join(&SGR_SEPARATOR.to_string()));
                cmd.push(SGR_SUFFIX);
                // After setting BG, a full screen repaint or fill with spaces might be needed
                // if the terminal doesn't apply new BG to already cleared areas.
                // For simplicity, we assume `\x1b[2J` uses the *current* background or a default.
                // A more robust way is to fill with spaces after setting BG.
            }
        }

        print!("{}", cmd);
        // No immediate flush here; `present()` will handle flushing.
        trace!("ConsoleDriver: clear_all command prepared: {:?}", cmd);
        Ok(())
    }

    /// Draws a run of text characters at a given cell coordinate using ANSI codes.
    fn draw_text_run(
        &mut self,
        coords: CellCoords,
        text: &str,
        style: TextRunStyle,
    ) -> Result<()> {
        if text.is_empty() {
            return Ok(());
        }

        let mut cmd = String::new();
        // 1. Move cursor to position (ANSI coordinates are 1-based).
        cmd.push_str(&Self::format_cursor_position(coords.y + 1, coords.x + 1));

        // 2. Build and append SGR sequence for the style.
        let mut sgr_codes = Vec::new();
        // Always reset to ensure clean state before applying new attributes,
        // unless no attributes are being set and current state is already default.
        // For simplicity in driver primitive, always reset then apply.
        sgr_codes.push(SGR_RESET_ALL);
        Self::sgr_append_attributes(&mut sgr_codes, style.fg, style.bg, style.flags);

        if !sgr_codes.is_empty() {
            cmd.push_str(SGR_PREFIX);
            cmd.push_str(&sgr_codes.iter().map(|c| c.to_string()).collect::<Vec<_>>().join(&SGR_SEPARATOR.to_string()));
            cmd.push(SGR_SUFFIX);
        }

        // 3. Append the text.
        cmd.push_str(text);

        // 4. Optional: Reset SGR attributes after the text run.
        // This makes each draw_text_run call more self-contained visually.
        // cmd.push_str(SGR_PREFIX);
        // cmd.push_str(&SGR_RESET_ALL.to_string());
        // cmd.push(SGR_SUFFIX);
        // Decided against auto-reset here; Renderer should manage this if needed by drawing spaces.

        print!("{}", cmd);
        trace!("ConsoleDriver: draw_text_run at ({},{}) text '{}' style {:?} cmd: {:?}", coords.x, coords.y, text, style, cmd);
        Ok(())
    }

    /// Fills a rectangular area of cells with a specified color using ANSI codes.
    fn fill_rect(
        &mut self,
        rect: CellRect,
        color: Color,
    ) -> Result<()> {
        if rect.width == 0 || rect.height == 0 {
            return Ok(());
        }

        let mut cmd = String::new();
        let mut sgr_codes = Vec::new();
        sgr_codes.push(SGR_RESET_ALL); // Reset first
        Self::sgr_append_bg_color(&mut sgr_codes, color); // Only set background

        if !sgr_codes.is_empty() {
            cmd.push_str(SGR_PREFIX);
            cmd.push_str(&sgr_codes.iter().map(|c| c.to_string()).collect::<Vec<_>>().join(&SGR_SEPARATOR.to_string()));
            cmd.push(SGR_SUFFIX);
        }

        let spaces: String = std::iter::repeat(' ').take(rect.width).collect();

        for y_offset in 0..rect.height {
            let current_y = rect.y + y_offset;
            // Move cursor to the start of the current line segment of the rectangle.
            cmd.push_str(&Self::format_cursor_position(current_y + 1, rect.x + 1));
            // Print spaces with the set background color.
            cmd.push_str(&spaces);
        }

        // Optional: Reset SGR after filling.
        // cmd.push_str(SGR_PREFIX);
        // cmd.push_str(&SGR_RESET_ALL.to_string());
        // cmd.push(SGR_SUFFIX);

        print!("{}", cmd);
        trace!("ConsoleDriver: fill_rect at ({},{}, w:{}, h:{}) color {:?} cmd: {:?}", rect.x, rect.y, rect.width, rect.height, color, cmd);
        Ok(())
    }

    /// Flushes `stdout` to ensure all buffered output is written to the terminal.
    fn present(&mut self) -> Result<()> {
        stdout().flush().context("ConsoleDriver: Failed to flush stdout during present")
    }

    /// Restores original terminal settings and shows the cursor.
    fn cleanup(&mut self) -> Result<()> {
        info!("ConsoleDriver: Cleaning up...");
        // Show cursor
        print!("{}", CURSOR_SHOW);
        stdout().flush().context("ConsoleDriver: Failed to flush stdout for CURSOR_SHOW during cleanup")?;

        // Restore original terminal settings
        if let Some(original_termios) = self.original_termios.take() {
            debug!("ConsoleDriver: Restoring original terminal attributes.");
            tcsetattr(STDIN_FILENO, TCSANOW, &original_termios)
                .context("ConsoleDriver: Failed to restore original terminal attributes")?;
        } else {
            warn!("ConsoleDriver: No original termios settings to restore.");
        }
        info!("ConsoleDriver: Cleanup complete.");
        Ok(())
    }
}

impl ConsoleDriver {
    /// Appends SGR codes for foreground, background, and attribute flags.
    fn sgr_append_attributes(codes: &mut Vec<u16>, fg: Color, bg: Color, flags: AttrFlags) {
        // Attribute flags (bold, italic, underline, etc.)
        if flags.contains(AttrFlags::BOLD) { codes.push(1); }
        // Note: Faint (2) is often not supported or same as normal.
        if flags.contains(AttrFlags::ITALIC) { codes.push(3); }
        if flags.contains(AttrFlags::UNDERLINE) { codes.push(4); }
        if flags.contains(AttrFlags::BLINK) { codes.push(5); }
        // REVERSE is handled by Renderer by swapping fg/bg before calling driver.
        if flags.contains(AttrFlags::HIDDEN) { codes.push(8); }
        if flags.contains(AttrFlags::STRIKETHROUGH) { codes.push(9); }

        // Foreground color
        Self::sgr_append_fg_color(codes, fg);
        // Background color
        Self::sgr_append_bg_color(codes, bg);
    }

    /// Appends SGR codes for foreground color.
    fn sgr_append_fg_color(codes: &mut Vec<u16>, fg: Color) {
        match fg {
            Color::Default => codes.push(39), // Default foreground
            Color::Named(nc) => {
                if (nc as u8) < 8 { codes.push(30 + nc as u8 as u16); } // 30-37
                else { codes.push(90 + (nc as u8 - 8) as u16); } // 90-97 (bright)
            }
            Color::Indexed(idx) => {
                codes.extend_from_slice(&[38, 5, idx as u16]); // 256-color mode
            }
            Color::Rgb(r, g, b) => {
                codes.extend_from_slice(&[38, 2, r as u16, g as u16, b as u16]); // True color
            }
        }
    }

    /// Appends SGR codes for background color.
    fn sgr_append_bg_color(codes: &mut Vec<u16>, bg: Color) {
        match bg {
            Color::Default => codes.push(49), // Default background
            Color::Named(nc) => {
                if (nc as u8) < 8 { codes.push(40 + nc as u8 as u16); } // 40-47
                else { codes.push(100 + (nc as u8 - 8) as u16); } // 100-107 (bright)
            }
            Color::Indexed(idx) => {
                codes.extend_from_slice(&[48, 5, idx as u16]); // 256-color mode
            }
            Color::Rgb(r, g, b) => {
                codes.extend_from_slice(&[48, 2, r as u16, g as u16, b as u16]); // True color
            }
        }
    }

    /// Helper function to format a CUP (Cursor Position) ANSI sequence.
    /// Row and column are 1-based.
    fn format_cursor_position(row_1_based: usize, col_1_based: usize) -> String {
        format!("\x1b[{};{}H", row_1_based, col_1_based)
    }
}

/// Helper function to get the terminal size in character cells using ioctl.
fn get_terminal_size_cells(fd: RawFd) -> Result<(u16, u16)> {
    // Safety: FFI call to ioctl. fd must be a valid file descriptor for a terminal.
    unsafe {
        let mut winsz: winsize = mem::zeroed();
        if libc::ioctl(fd, TIOCGWINSZ, &mut winsz) == -1 {
            return Err(anyhow::Error::from(std::io::Error::last_os_error())
                .context("ConsoleDriver: Failed to get terminal size via ioctl(TIOCGWINSZ)"));
        }
        // Provide default values if ioctl returns 0 for cols/rows (can happen in some contexts)
        // Using the constants imported from the parent `backends` module.
        let cols = if winsz.ws_col == 0 { DEFAULT_WINDOW_WIDTH_CHARS as u16 } else { winsz.ws_col };
        let rows = if winsz.ws_row == 0 { DEFAULT_WINDOW_HEIGHT_CHARS as u16 } else { winsz.ws_row };
        Ok((cols, rows))
    }
}

impl Drop for ConsoleDriver {
    fn drop(&mut self) {
        info!("ConsoleDriver: Dropping instance, attempting cleanup.");
        // Attempt cleanup. Errors are logged within cleanup itself.
        if let Err(e) = self.cleanup() {
            // Use the error! macro from the log crate.
            error!("ConsoleDriver: Error during cleanup in drop: {}", e);
        }
    }
}

