// src/backends/console.rs

//! Provides a `Driver` implementation for rendering to a standard Unix console
//! using ANSI escape codes. This backend is useful for running `myterm` in
//! environments without a graphical display server or as a fallback.

// Corrected: Import Color directly from the color module.
use crate::backends::{
    BackendEvent, CellCoords, CellRect, Driver, TextRunStyle, DEFAULT_WINDOW_HEIGHT_CHARS,
    DEFAULT_WINDOW_WIDTH_CHARS,
};
use crate::color::Color;
use crate::glyph::AttrFlags; // NamedColor is still useful here for panic messages

use anyhow::{Context, Result};
use libc::{winsize, STDIN_FILENO, TIOCGWINSZ}; // For terminal size and raw mode
use std::io::{self, stdin, stdout, Read, Write};
use std::mem;
use std::os::unix::io::RawFd;
use termios::{tcsetattr, Termios, ECHO, ICANON, ISIG, TCSANOW, VMIN, VTIME}; // For raw mode

// Logging
use log::{debug, error, info, trace, warn};

// --- ANSI Escape Code Constants ---
const CURSOR_HIDE: &str = "\x1b[?25l"; // Hide cursor
const CURSOR_SHOW: &str = "\x1b[?25h"; // Show cursor
const SGR_PREFIX: &str = "\x1b["; // Start of Select Graphic Rendition sequence
const SGR_SUFFIX: char = 'm'; // End of SGR sequence
const SGR_SEPARATOR: char = ';'; // Separator for multiple SGR codes
const SGR_RESET_ALL: u16 = 0; // SGR code to reset all attributes
const CLEAR_SCREEN_AND_HOME: &str = "\x1b[2J\x1b[H"; // Clear entire screen and move cursor to home

// Default font dimensions for console mode. These are approximations as true
// pixel dimensions are not typically available or relevant for a console backend
// in the same way as for a graphical backend. They are used for calculating
// an initial pixel-based size if requested by the orchestrator, but the primary
// unit of operation for the console driver is character cells.
const DEFAULT_CONSOLE_FONT_WIDTH_PX: u16 = 8;
const DEFAULT_CONSOLE_FONT_HEIGHT_PX: u16 = 16;

/// A `Driver` implementation for rendering to a standard Unix console.
///
/// It uses ANSI escape codes for drawing, cursor manipulation, and color setting.
/// It also attempts to set the terminal to raw mode for direct input handling.
pub struct ConsoleDriver {
    /// Stores the original terminal attributes to restore them on cleanup.
    original_termios: Option<Termios>,
    /// Last known width of the terminal in character cells.
    last_known_width_cells: u16,
    /// Last known height of the terminal in character cells.
    last_known_height_cells: u16,
    /// Nominal font width in pixels (used for `get_font_dimensions`).
    font_width_px: u16,
    /// Nominal font height in pixels (used for `get_font_dimensions`).
    font_height_px: u16,
    /// Buffer for reading input from stdin.
    input_buffer: [u8; 128], // Small buffer for typical key press sequences
    /// Tracks if the cursor is intended to be visible.
    is_cursor_logically_visible: bool,
}

impl Driver for ConsoleDriver {
    /// Creates a new `ConsoleDriver`.
    ///
    /// This attempts to:
    /// 1. Get the current terminal attributes.
    /// 2. Set the terminal to raw mode (disabling echo, canonical mode, signals).
    /// 3. Hide the console's native cursor initially (it's managed by `set_cursor_visibility`).
    /// 4. Get the initial terminal size in character cells.
    ///
    /// If setting raw mode fails, it logs a warning and proceeds, which might
    /// lead to suboptimal input handling.
    fn new() -> Result<Self> {
        info!("Creating new ConsoleDriver.");
        let original_termios = match Termios::from_fd(STDIN_FILENO) {
            Ok(ts) => Some(ts),
            Err(e) => {
                warn!(
                    "ConsoleDriver: Failed to get initial termios: {}. Proceeding without raw mode.",
                    e
                );
                None
            }
        };

        if let Some(ref ots) = original_termios {
            let mut raw_termios = *ots;
            // Disable echo, canonical mode (line buffering), and signal generation (Ctrl-C).
            raw_termios.c_lflag &= !(ECHO | ICANON | ISIG);
            // Disable software flow control (IXON/IXOFF), CR-to-NL mapping, etc.
            raw_termios.c_iflag &=
                !(libc::IXON | libc::IXOFF | libc::ICRNL | libc::INLCR | libc::IGNCR);
            // Disable output processing (OPOST).
            raw_termios.c_oflag &= !libc::OPOST;
            // Set VMIN and VTIME for non-blocking reads (read returns immediately).
            raw_termios.c_cc[VMIN] = 0; // Minimum number of bytes for read
            raw_termios.c_cc[VTIME] = 0; // Timeout in deciseconds for read

            // Attempt to set the terminal to raw mode.
            // SAFETY: tcsetattr is an FFI call.
            if let Err(e) = tcsetattr(STDIN_FILENO, TCSANOW, &raw_termios) {
                // Use context for better error reporting as per anyhow guidelines.
                warn!(
                    "ConsoleDriver: Failed to set raw terminal attributes: {}. Input might not work as expected.",
                    e
                );
                // Proceeding without raw mode is a degraded state, but might still be partially functional.
            } else {
                debug!("ConsoleDriver: Terminal set to raw mode.");
            }
        }

        // Hide the console's native cursor by default.
        // The orchestrator will call set_cursor_visibility based on terminal state.
        print!("{}", CURSOR_HIDE);
        stdout()
            .flush()
            .context("ConsoleDriver: Failed to flush stdout for initial CURSOR_HIDE")?;

        let (initial_width, initial_height) = get_terminal_size_cells(STDIN_FILENO)
            .context("ConsoleDriver: Failed to get initial terminal size")?;
        info!(
            "ConsoleDriver: Initial terminal size: {}x{} cells.",
            initial_width, initial_height
        );

        Ok(ConsoleDriver {
            original_termios,
            last_known_width_cells: initial_width,
            last_known_height_cells: initial_height,
            font_width_px: DEFAULT_CONSOLE_FONT_WIDTH_PX,
            font_height_px: DEFAULT_CONSOLE_FONT_HEIGHT_PX,
            input_buffer: [0u8; 128],           // Initialize buffer
            is_cursor_logically_visible: false, // Start hidden, orchestrator will set.
        })
    }

    /// Returns the standard input file descriptor (`STDIN_FILENO`) for event monitoring.
    fn get_event_fd(&self) -> Option<RawFd> {
        Some(STDIN_FILENO)
    }

    /// Processes events for the console driver.
    ///
    /// This includes:
    /// 1. Checking for terminal resize events (via `ioctl TIOCGWINSZ`).
    /// 2. Reading keyboard input from stdin.
    ///
    /// # Returns
    /// * `Result<Vec<BackendEvent>>`: A list of detected events, or an error if stdin read fails.
    fn process_events(&mut self) -> Result<Vec<BackendEvent>> {
        let mut backend_events = Vec::new();

        // Check for terminal resize.
        match get_terminal_size_cells(STDIN_FILENO) {
            Ok((current_width_cells, current_height_cells)) => {
                if current_width_cells != self.last_known_width_cells
                    || current_height_cells != self.last_known_height_cells
                {
                    info!(
                        "ConsoleDriver: Terminal resized from {}x{} to {}x{} cells.",
                        self.last_known_width_cells,
                        self.last_known_height_cells,
                        current_width_cells,
                        current_height_cells
                    );
                    self.last_known_width_cells = current_width_cells;
                    self.last_known_height_cells = current_height_cells;
                    // Calculate pixel dimensions based on nominal font size for the Resize event.
                    let width_px = current_width_cells.saturating_mul(self.font_width_px);
                    let height_px = current_height_cells.saturating_mul(self.font_height_px);
                    backend_events.push(BackendEvent::Resize {
                        width_px,
                        height_px,
                    });
                }
            }
            Err(e) => {
                // Log warning but continue with last known size if ioctl fails.
                warn!(
                    "ConsoleDriver: Failed to get terminal size: {}. Using last known.",
                    e
                );
            }
        }

        // Process keyboard input.
        // In raw mode with VMIN=0, VTIME=0, read should be non-blocking.
        match stdin().read(&mut self.input_buffer) {
            Ok(0) => {
                // EOF on stdin, usually means the controlling terminal was closed or PTY master closed.
                info!("ConsoleDriver: EOF on stdin. Requesting close.");
                backend_events.push(BackendEvent::CloseRequested);
            }
            Ok(bytes_read) => {
                trace!("ConsoleDriver: Read {} bytes from stdin.", bytes_read);
                // Convert each byte to a Key event. This is a simplification;
                // proper handling would involve parsing multi-byte sequences (e.g., for arrows).
                for i in 0..bytes_read {
                    let byte = self.input_buffer[i];
                    // For console, keysym is often just the byte value or 0 if it's part of a sequence.
                    // Text is the byte converted to a char.
                    let text = String::from_utf8_lossy(&[byte]).to_string();
                    trace!("ConsoleDriver: Processed byte {} as char '{}'", byte, text);
                    // Using byte as keysym is a placeholder. A more robust solution would map
                    // common control chars or escape sequences to more meaningful keysyms.
                    backend_events.push(BackendEvent::Key {
                        keysym: byte as u32,
                        text,
                    });
                }
            }
            Err(ref e) if e.kind() == io::ErrorKind::WouldBlock => {
                // No data available to read, which is normal for non-blocking.
                trace!("ConsoleDriver: stdin read WouldBlock.");
            }
            Err(ref e) if e.kind() == io::ErrorKind::Interrupted => {
                // Read was interrupted by a signal, typically safe to retry.
                trace!("ConsoleDriver: stdin read Interrupted.");
            }
            Err(e) => {
                // Other read errors are potentially critical.
                return Err(e).context("ConsoleDriver: Error reading from stdin");
            }
        }
        Ok(backend_events)
    }

    /// Returns the nominal font dimensions used by the console driver.
    fn get_font_dimensions(&self) -> (usize, usize) {
        (self.font_width_px as usize, self.font_height_px as usize)
    }

    /// Returns the display dimensions in pixels, calculated from cell dimensions and nominal font size.
    fn get_display_dimensions_pixels(&self) -> (u16, u16) {
        let width_px = self
            .last_known_width_cells
            .saturating_mul(self.font_width_px);
        let height_px = self
            .last_known_height_cells
            .saturating_mul(self.font_height_px);
        (width_px, height_px)
    }

    /// Clears the entire console screen using ANSI codes.
    /// The `bg` color argument (guaranteed concrete by Renderer) is used to set the
    /// background color before clearing.
    fn clear_all(&mut self, bg: Color) -> Result<()> {
        // Renderer ensures `bg` is not Color::Default.
        // If it were, this would be an internal logic error.
        if matches!(bg, Color::Default) {
            error!(
                "ConsoleDriver::clear_all received Color::Default. This is a bug in the Renderer."
            );
            // Fallback or panic, as this violates the contract with Renderer.
            panic!(
                "ConsoleDriver::clear_all received Color::Default. Renderer should resolve defaults."
            );
        }

        let mut cmd = String::new();
        // Set the specific background color for the whole screen using SGR.
        let mut sgr_codes = Vec::new();
        Self::sgr_append_concrete_bg_color(&mut sgr_codes, bg);
        if !sgr_codes.is_empty() {
            cmd.push_str(SGR_PREFIX);
            cmd.push_str(
                &sgr_codes
                    .iter()
                    .map(u16::to_string) // Use direct method reference
                    .collect::<Vec<_>>()
                    .join(&SGR_SEPARATOR.to_string()),
            );
            cmd.push(SGR_SUFFIX);
        }
        // Standard clear screen and home cursor sequence.
        // This should apply the SGR background set above to the cleared area.
        cmd.push_str(CLEAR_SCREEN_AND_HOME);

        print!("{}", cmd);
        trace!("ConsoleDriver: clear_all command prepared: {:?}", cmd);
        // No flush needed here, as `present` will handle flushing.
        Ok(())
    }

    /// Draws a run of text on the console.
    /// `style.fg` and `style.bg` are guaranteed concrete by the Renderer.
    fn draw_text_run(&mut self, coords: CellCoords, text: &str, style: TextRunStyle) -> Result<()> {
        if text.is_empty() {
            return Ok(());
        }
        // Contract: Renderer ensures colors are concrete.
        if matches!(style.fg, Color::Default) || matches!(style.bg, Color::Default) {
            error!(
                "ConsoleDriver::draw_text_run received Color::Default in style. This is a bug in the Renderer."
            );
            panic!(
                "ConsoleDriver::draw_text_run received Color::Default. Renderer should resolve defaults."
            );
        }

        let mut cmd = String::new();
        // Move cursor to position (1-based for ANSI CUP).
        cmd.push_str(&Self::format_cursor_position(coords.y + 1, coords.x + 1));

        // Build SGR sequence.
        let mut sgr_codes = Vec::new();
        sgr_codes.push(SGR_RESET_ALL); // Reset first for a clean slate.
        Self::sgr_append_concrete_attributes(&mut sgr_codes, style.fg, style.bg, style.flags);

        if !sgr_codes.is_empty() {
            cmd.push_str(SGR_PREFIX);
            cmd.push_str(
                &sgr_codes
                    .iter()
                    .map(u16::to_string)
                    .collect::<Vec<_>>()
                    .join(&SGR_SEPARATOR.to_string()),
            );
            cmd.push(SGR_SUFFIX);
        }
        cmd.push_str(text); // Append the text to draw.
        print!("{}", cmd);
        trace!(
            "ConsoleDriver: draw_text_run at ({},{}) text '{}' style {:?} cmd: {:?}",
            coords.x,
            coords.y,
            text,
            style,
            cmd
        );
        Ok(())
    }

    /// Fills a rectangular area of cells with a specified concrete color.
    /// This is done by setting the background color and printing spaces.
    fn fill_rect(&mut self, rect: CellRect, color: Color) -> Result<()> {
        if rect.width == 0 || rect.height == 0 {
            return Ok(());
        }
        // Contract: Renderer ensures color is concrete.
        if matches!(color, Color::Default) {
            error!(
                "ConsoleDriver::fill_rect received Color::Default. This is a bug in the Renderer."
            );
            panic!(
                "ConsoleDriver::fill_rect received Color::Default. Renderer should resolve defaults."
            );
        }

        let mut cmd = String::new();
        // Set background color using SGR.
        let mut sgr_codes = Vec::new();
        sgr_codes.push(SGR_RESET_ALL); // Reset for clean application of background.
        Self::sgr_append_concrete_bg_color(&mut sgr_codes, color);

        if !sgr_codes.is_empty() {
            cmd.push_str(SGR_PREFIX);
            cmd.push_str(
                &sgr_codes
                    .iter()
                    .map(u16::to_string)
                    .collect::<Vec<_>>()
                    .join(&SGR_SEPARATOR.to_string()),
            );
            cmd.push(SGR_SUFFIX);
        }

        // Create a string of spaces with the required width.
        let spaces: String = vec![' '; rect.width].into_iter().collect();
        // Iterate over each row in the rectangle.
        for y_offset in 0..rect.height {
            let current_y = rect.y + y_offset;
            // Move cursor to the start of the segment for this row.
            cmd.push_str(&Self::format_cursor_position(current_y + 1, rect.x + 1));
            // Print spaces to fill the segment with the set background color.
            cmd.push_str(&spaces);
        }
        print!("{}", cmd);
        trace!(
            "ConsoleDriver: fill_rect at ({},{}, w:{}, h:{}) color {:?} cmd: {:?}",
            rect.x,
            rect.y,
            rect.width,
            rect.height,
            color,
            cmd
        );
        Ok(())
    }

    /// Flushes stdout to ensure all buffered commands are sent to the console.
    fn present(&mut self) -> Result<()> {
        stdout()
            .flush()
            .context("ConsoleDriver: Failed to flush stdout during present")
    }

    /// Sets the window title using an OSC sequence.
    /// This is a common way to set titles in terminal emulators that support it.
    fn set_title(&mut self, title: &str) {
        // OSC 0 sets icon name and window title. OSC 2 sets window title only.
        // Using OSC 0 for wider compatibility.
        print!("\x1b]0;{}\x07", title); // \x07 is the BEL character, often used as ST for OSC
                                        // No flush here, assume present() will be called.
        trace!("ConsoleDriver: Set window title to '{}'", title);
    }

    /// Rings the terminal bell by printing the BEL character.
    fn bell(&mut self) {
        print!("\x07");
        // No flush here.
        trace!("ConsoleDriver: Rang bell.");
    }

    /// Sets the visibility of the console's native cursor using ANSI codes.
    fn set_cursor_visibility(&mut self, visible: bool) {
        trace!(
            "ConsoleDriver: Setting native cursor visibility to: {}",
            visible
        );
        if visible {
            print!("{}", CURSOR_SHOW);
        } else {
            print!("{}", CURSOR_HIDE);
        }
        // No flush here, assume present() will be called.
        self.is_cursor_logically_visible = visible;
    }

    /// Informs the driver about focus changes.
    /// For a console driver, this is typically a no-op as focus is managed
    /// by the terminal emulator application itself or the OS, not via ANSI codes
    /// that the driver would send.
    fn set_focus(&mut self, focused: bool) {
        trace!(
            "ConsoleDriver: Focus event received (focused: {}). No specific action taken by console driver.",
            focused
        );
        // No action needed for console driver regarding focus typically.
    }

    /// Restores original terminal attributes and shows the cursor.
    fn cleanup(&mut self) -> Result<()> {
        info!("ConsoleDriver: Cleaning up...");
        // Ensure cursor is shown.
        print!("{}", CURSOR_SHOW);
        stdout()
            .flush()
            .context("ConsoleDriver: Failed to flush for CURSOR_SHOW cleanup")?;

        // Restore original terminal attributes if they were saved.
        if let Some(original_termios_val) = self.original_termios.take() {
            debug!("ConsoleDriver: Restoring original terminal attributes.");
            // SAFETY: tcsetattr is an FFI call.
            tcsetattr(STDIN_FILENO, TCSANOW, &original_termios_val)
                .context("ConsoleDriver: Failed to restore original terminal attributes")?;
        } else {
            warn!("ConsoleDriver: No original termios to restore.");
        }
        info!("ConsoleDriver: Cleanup complete.");
        Ok(())
    }
}

// --- ConsoleDriver Private Helper Methods ---
impl ConsoleDriver {
    /// Appends SGR codes for concrete foreground, background, and attribute flags.
    /// Panics if `fg` or `bg` is `Color::Default`, as these should be resolved by the Renderer.
    fn sgr_append_concrete_attributes(
        codes: &mut Vec<u16>,
        fg: Color,
        bg: Color,
        flags: AttrFlags,
    ) {
        // Apply standard attribute flags.
        if flags.contains(AttrFlags::BOLD) {
            codes.push(1);
        }
        if flags.contains(AttrFlags::ITALIC) {
            codes.push(3);
        } // Often not supported or styled differently
        if flags.contains(AttrFlags::UNDERLINE) {
            codes.push(4);
        }
        if flags.contains(AttrFlags::BLINK) {
            codes.push(5);
        }
        if flags.contains(AttrFlags::HIDDEN) {
            codes.push(8);
        }
        if flags.contains(AttrFlags::STRIKETHROUGH) {
            codes.push(9);
        }
        // Note: FAINT (2) is omitted as its support is inconsistent. REVERSE is handled by Renderer.

        // Append resolved foreground and background colors.
        Self::sgr_append_concrete_fg_color(codes, fg);
        Self::sgr_append_concrete_bg_color(codes, bg);
    }

    /// Appends SGR codes for a concrete foreground color.
    /// Panics on `Color::Default`.
    fn sgr_append_concrete_fg_color(codes: &mut Vec<u16>, fg: Color) {
        match fg {
            Color::Default => panic!(
                "ConsoleDriver received Color::Default for foreground. Renderer should resolve defaults."
            ),
            Color::Named(nc) => {
                // Standard (0-7) and bright (8-15) named colors.
                if (nc as u8) < 8 {
                    codes.push(30 + nc as u16);
                }
                // 30-37 for normal
                else {
                    codes.push(90 + (nc as u8 - 8) as u16);
                } // 90-97 for bright
            }
            Color::Indexed(idx) => {
                // 256-color palette.
                codes.extend_from_slice(&[38, 5, idx as u16]);
            }
            Color::Rgb(r, g, b) => {
                // True color (24-bit).
                codes.extend_from_slice(&[38, 2, r as u16, g as u16, b as u16]);
            }
        }
    }

    /// Appends SGR codes for a concrete background color.
    /// Panics on `Color::Default`.
    fn sgr_append_concrete_bg_color(codes: &mut Vec<u16>, bg: Color) {
        match bg {
            Color::Default => panic!(
                "ConsoleDriver received Color::Default for background. Renderer should resolve defaults."
            ),
            Color::Named(nc) => {
                if (nc as u8) < 8 {
                    codes.push(40 + nc as u16);
                }
                // 40-47 for normal
                else {
                    codes.push(100 + (nc as u8 - 8) as u16);
                } // 100-107 for bright
            }
            Color::Indexed(idx) => {
                codes.extend_from_slice(&[48, 5, idx as u16]);
            }
            Color::Rgb(r, g, b) => {
                codes.extend_from_slice(&[48, 2, r as u16, g as u16, b as u16]);
            }
        }
    }

    /// Formats an ANSI CUP (Cursor Position) sequence.
    /// Row and column are 1-based.
    fn format_cursor_position(row_1_based: usize, col_1_based: usize) -> String {
        format!("\x1b[{};{}H", row_1_based, col_1_based)
    }
}

/// Retrieves the terminal size in character cells using an `ioctl` call.
///
/// # Arguments
/// * `fd`: The file descriptor of the terminal (e.g., `STDIN_FILENO`).
///
/// # Returns
/// * `Result<(u16, u16)>`: A tuple `(columns, rows)` or an error if `ioctl` fails.
///   If `ioctl` returns 0 for cols/rows, defaults from `backends` module are used.
fn get_terminal_size_cells(fd: RawFd) -> Result<(u16, u16)> {
    // SAFETY: `ioctl` is an FFI call. `winsz` must be valid.
    unsafe {
        let mut winsz: winsize = mem::zeroed();
        if libc::ioctl(fd, TIOCGWINSZ, &mut winsz) == -1 {
            // Convert OS error to anyhow::Error for context.
            return Err(anyhow::Error::from(std::io::Error::last_os_error())
                .context("ConsoleDriver: ioctl(TIOCGWINSZ) failed"));
        }
        // Use default dimensions if ioctl reports zero, which can happen in some contexts.
        let cols = if winsz.ws_col == 0 {
            DEFAULT_WINDOW_WIDTH_CHARS as u16
        } else {
            winsz.ws_col
        };
        let rows = if winsz.ws_row == 0 {
            DEFAULT_WINDOW_HEIGHT_CHARS as u16
        } else {
            winsz.ws_row
        };
        Ok((cols, rows))
    }
}

/// Ensures cleanup is attempted when `ConsoleDriver` is dropped.
impl Drop for ConsoleDriver {
    fn drop(&mut self) {
        info!("ConsoleDriver: Dropping instance, attempting cleanup.");
        if let Err(e) = self.cleanup() {
            // Log error, but don't panic in drop.
            error!("ConsoleDriver: Error during cleanup in drop: {}", e);
        }
    }
}
