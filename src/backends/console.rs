// src/backends/console.rs

use crate::backends::{
    BackendEvent, CellCoords, CellRect, DEFAULT_WINDOW_HEIGHT_CHARS, DEFAULT_WINDOW_WIDTH_CHARS,
    Driver, TextRunStyle,
};
use crate::glyph::{AttrFlags, Color}; // Added NamedColor for panic message clarity

use anyhow::{Context, Result};
use libc::{STDIN_FILENO, TIOCGWINSZ, winsize};
use std::io::{self, Read, Write, stdin, stdout};
use std::mem;
use std::os::unix::io::RawFd;
use termios::{ECHO, ICANON, ISIG, TCSANOW, Termios, VMIN, VTIME, tcsetattr};

use log::{debug, error, info, trace, warn};

const CURSOR_HIDE: &str = "\x1b[?25l";
const CURSOR_SHOW: &str = "\x1b[?25h";
const SGR_PREFIX: &str = "\x1b[";
const SGR_SUFFIX: char = 'm';
const SGR_SEPARATOR: char = ';';
const SGR_RESET_ALL: u16 = 0;
const CLEAR_SCREEN_AND_HOME: &str = "\x1b[2J\x1b[H";

const DEFAULT_CONSOLE_FONT_WIDTH_PX: u16 = 8;
const DEFAULT_CONSOLE_FONT_HEIGHT_PX: u16 = 16;

pub struct ConsoleDriver {
    original_termios: Option<Termios>,
    last_known_width_cells: u16,
    last_known_height_cells: u16,
    font_width_px: u16,
    font_height_px: u16,
    input_buffer: [u8; 128],
}

impl Driver for ConsoleDriver {
    fn new() -> Result<Self> {
        info!("Creating new ConsoleDriver.");
        let original_termios = match Termios::from_fd(STDIN_FILENO) {
            Ok(ts) => Some(ts),
            Err(e) => {
                warn!(
                    "Failed to get initial termios: {}. Proceeding without raw mode.",
                    e
                );
                None
            }
        };

        if let Some(ref ots) = original_termios {
            let mut raw_termios = *ots;
            raw_termios.c_lflag &= !(ECHO | ICANON | ISIG);
            raw_termios.c_iflag &=
                !(libc::IXON | libc::IXOFF | libc::ICRNL | libc::INLCR | libc::IGNCR);
            raw_termios.c_oflag &= !libc::OPOST;
            raw_termios.c_cc[VMIN] = 0;
            raw_termios.c_cc[VTIME] = 0;
            tcsetattr(STDIN_FILENO, TCSANOW, &raw_termios)
                .context("ConsoleDriver: Failed to set raw terminal attributes")?;
            debug!("ConsoleDriver: Terminal set to raw mode.");
        }

        print!("{}", CURSOR_HIDE);
        stdout()
            .flush()
            .context("ConsoleDriver: Failed to flush stdout for CURSOR_HIDE")?;

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
            input_buffer: [0u8; 128],
        })
    }

    fn get_event_fd(&self) -> Option<RawFd> {
        Some(STDIN_FILENO)
    }

    fn process_events(&mut self) -> Result<Vec<BackendEvent>> {
        let mut backend_events = Vec::new();

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
                    let width_px = current_width_cells.saturating_mul(self.font_width_px);
                    let height_px = current_height_cells.saturating_mul(self.font_height_px);
                    backend_events.push(BackendEvent::Resize {
                        width_px,
                        height_px,
                    });
                }
            }
            Err(e) => {
                warn!(
                    "ConsoleDriver: Failed to get terminal size: {}. Using last known.",
                    e
                );
            }
        }

        match stdin().read(&mut self.input_buffer) {
            Ok(0) => {
                info!("ConsoleDriver: EOF on stdin. Requesting close.");
                backend_events.push(BackendEvent::CloseRequested);
            }
            Ok(bytes_read) => {
                trace!("ConsoleDriver: Read {} bytes from stdin.", bytes_read);
                for i in 0..bytes_read {
                    let byte = self.input_buffer[i];
                    let text = String::from_utf8_lossy(&[byte]).to_string();
                    trace!("ConsoleDriver: Processed byte {} as char '{}'", byte, text);
                    backend_events.push(BackendEvent::Key { keysym: 0, text });
                }
            }
            Err(ref e) if e.kind() == io::ErrorKind::WouldBlock => {
                trace!("ConsoleDriver: stdin read WouldBlock.");
            }
            Err(ref e) if e.kind() == io::ErrorKind::Interrupted => {
                trace!("ConsoleDriver: stdin read Interrupted.");
            }
            Err(e) => {
                return Err(e).context("ConsoleDriver: Error reading from stdin");
            }
        }
        Ok(backend_events)
    }

    fn get_font_dimensions(&self) -> (usize, usize) {
        (self.font_width_px as usize, self.font_height_px as usize)
    }

    fn get_display_dimensions_pixels(&self) -> (u16, u16) {
        let width_px = self
            .last_known_width_cells
            .saturating_mul(self.font_width_px);
        let height_px = self
            .last_known_height_cells
            .saturating_mul(self.font_height_px);
        (width_px, height_px)
    }

    /// Clears the entire display area. `bg` color MUST be concrete.
    fn clear_all(&mut self, bg: Color) -> Result<()> {
        if matches!(bg, Color::Default) {
            error!(
                "ConsoleDriver::clear_all received Color::Default. This is a bug in the Renderer."
            );
            // Fallback to terminal's default clear or panic
            panic!(
                "ConsoleDriver::clear_all received Color::Default. Renderer should resolve defaults."
            );
        }

        let mut cmd = String::new();
        cmd.push_str(CLEAR_SCREEN_AND_HOME); // Standard clear

        // Set the specific background color for the whole screen.
        // This typically involves setting the SGR background and then printing spaces
        // or relying on the terminal to apply the current BG to cleared areas.
        // For simplicity, we set SGR BG, then clear. Some terminals might clear
        // with the *new* BG, others with the *old* or a fixed default.
        // A more robust clear_all would fill the screen with spaces with the new BG.
        let mut sgr_codes = Vec::new();
        Self::sgr_append_concrete_bg_color(&mut sgr_codes, bg);
        if !sgr_codes.is_empty() {
            cmd.push_str(SGR_PREFIX);
            cmd.push_str(
                &sgr_codes
                    .iter()
                    .map(|c| c.to_string())
                    .collect::<Vec<_>>()
                    .join(&SGR_SEPARATOR.to_string()),
            );
            cmd.push(SGR_SUFFIX);
            // To be certain, one might need to print spaces across the whole screen here.
            // For now, relying on the SGR BG + 2J.
        }

        print!("{}", cmd);
        trace!("ConsoleDriver: clear_all command prepared: {:?}", cmd);
        Ok(())
    }

    /// Draws a run of text. `style.fg` and `style.bg` MUST be concrete.
    fn draw_text_run(&mut self, coords: CellCoords, text: &str, style: TextRunStyle) -> Result<()> {
        if text.is_empty() {
            return Ok(());
        }
        if matches!(style.fg, Color::Default) || matches!(style.bg, Color::Default) {
            error!(
                "ConsoleDriver::draw_text_run received Color::Default in style. This is a bug in the Renderer."
            );
            panic!(
                "ConsoleDriver::draw_text_run received Color::Default. Renderer should resolve defaults."
            );
        }

        let mut cmd = String::new();
        cmd.push_str(&Self::format_cursor_position(coords.y + 1, coords.x + 1));

        let mut sgr_codes = Vec::new();
        sgr_codes.push(SGR_RESET_ALL); // Always reset for clean application
        Self::sgr_append_concrete_attributes(&mut sgr_codes, style.fg, style.bg, style.flags);

        if !sgr_codes.is_empty() {
            cmd.push_str(SGR_PREFIX);
            cmd.push_str(
                &sgr_codes
                    .iter()
                    .map(|c| c.to_string())
                    .collect::<Vec<_>>()
                    .join(&SGR_SEPARATOR.to_string()),
            );
            cmd.push(SGR_SUFFIX);
        }
        cmd.push_str(text);
        print!("{}", cmd);
        trace!(
            "ConsoleDriver: draw_text_run at ({},{}) text '{}' style {:?} cmd: {:?}",
            coords.x, coords.y, text, style, cmd
        );
        Ok(())
    }

    /// Fills a rectangular area. `color` MUST be concrete.
    fn fill_rect(&mut self, rect: CellRect, color: Color) -> Result<()> {
        if rect.width == 0 || rect.height == 0 {
            return Ok(());
        }
        if matches!(color, Color::Default) {
            error!(
                "ConsoleDriver::fill_rect received Color::Default. This is a bug in the Renderer."
            );
            panic!(
                "ConsoleDriver::fill_rect received Color::Default. Renderer should resolve defaults."
            );
        }

        let mut cmd = String::new();
        let mut sgr_codes = Vec::new();
        sgr_codes.push(SGR_RESET_ALL);
        Self::sgr_append_concrete_bg_color(&mut sgr_codes, color);

        if !sgr_codes.is_empty() {
            cmd.push_str(SGR_PREFIX);
            cmd.push_str(
                &sgr_codes
                    .iter()
                    .map(|c| c.to_string())
                    .collect::<Vec<_>>()
                    .join(&SGR_SEPARATOR.to_string()),
            );
            cmd.push(SGR_SUFFIX);
        }

        let spaces: String = std::iter::repeat(' ').take(rect.width).collect();
        for y_offset in 0..rect.height {
            let current_y = rect.y + y_offset;
            cmd.push_str(&Self::format_cursor_position(current_y + 1, rect.x + 1));
            cmd.push_str(&spaces);
        }
        print!("{}", cmd);
        trace!(
            "ConsoleDriver: fill_rect at ({},{}, w:{}, h:{}) color {:?} cmd: {:?}",
            rect.x, rect.y, rect.width, rect.height, color, cmd
        );
        Ok(())
    }

    fn present(&mut self) -> Result<()> {
        stdout()
            .flush()
            .context("ConsoleDriver: Failed to flush stdout during present")
    }

    fn cleanup(&mut self) -> Result<()> {
        info!("ConsoleDriver: Cleaning up...");
        print!("{}", CURSOR_SHOW);
        stdout()
            .flush()
            .context("ConsoleDriver: Failed to flush for CURSOR_SHOW cleanup")?;
        if let Some(original_termios) = self.original_termios.take() {
            debug!("ConsoleDriver: Restoring original terminal attributes.");
            tcsetattr(STDIN_FILENO, TCSANOW, &original_termios)
                .context("ConsoleDriver: Failed to restore original terminal attributes")?;
        } else {
            warn!("ConsoleDriver: No original termios to restore.");
        }
        info!("ConsoleDriver: Cleanup complete.");
        Ok(())
    }
}

impl ConsoleDriver {
    /// Appends SGR codes for concrete foreground, background, and flags. Panics on Color::Default.
    fn sgr_append_concrete_attributes(
        codes: &mut Vec<u16>,
        fg: Color,
        bg: Color,
        flags: AttrFlags,
    ) {
        if flags.contains(AttrFlags::BOLD) {
            codes.push(1);
        }
        if flags.contains(AttrFlags::ITALIC) {
            codes.push(3);
        }
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

        Self::sgr_append_concrete_fg_color(codes, fg);
        Self::sgr_append_concrete_bg_color(codes, bg);
    }

    /// Appends SGR codes for a concrete foreground color. Panics on Color::Default.
    fn sgr_append_concrete_fg_color(codes: &mut Vec<u16>, fg: Color) {
        match fg {
            Color::Default => panic!(
                "ConsoleDriver received Color::Default for foreground. Renderer should resolve defaults."
            ),
            Color::Named(nc) => {
                if (nc as u8) < 8 {
                    codes.push(30 + nc as u8 as u16);
                } else {
                    codes.push(90 + (nc as u8 - 8) as u16);
                }
            }
            Color::Indexed(idx) => {
                codes.extend_from_slice(&[38, 5, idx as u16]);
            }
            Color::Rgb(r, g, b) => {
                codes.extend_from_slice(&[38, 2, r as u16, g as u16, b as u16]);
            }
        }
    }

    /// Appends SGR codes for a concrete background color. Panics on Color::Default.
    fn sgr_append_concrete_bg_color(codes: &mut Vec<u16>, bg: Color) {
        match bg {
            Color::Default => panic!(
                "ConsoleDriver received Color::Default for background. Renderer should resolve defaults."
            ),
            Color::Named(nc) => {
                if (nc as u8) < 8 {
                    codes.push(40 + nc as u8 as u16);
                } else {
                    codes.push(100 + (nc as u8 - 8) as u16);
                }
            }
            Color::Indexed(idx) => {
                codes.extend_from_slice(&[48, 5, idx as u16]);
            }
            Color::Rgb(r, g, b) => {
                codes.extend_from_slice(&[48, 2, r as u16, g as u16, b as u16]);
            }
        }
    }

    fn format_cursor_position(row_1_based: usize, col_1_based: usize) -> String {
        format!("\x1b[{};{}H", row_1_based, col_1_based)
    }
}

fn get_terminal_size_cells(fd: RawFd) -> Result<(u16, u16)> {
    unsafe {
        let mut winsz: winsize = mem::zeroed();
        if libc::ioctl(fd, TIOCGWINSZ, &mut winsz) == -1 {
            return Err(anyhow::Error::from(std::io::Error::last_os_error())
                .context("ConsoleDriver: ioctl(TIOCGWINSZ) failed"));
        }
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

impl Drop for ConsoleDriver {
    fn drop(&mut self) {
        info!("ConsoleDriver: Dropping instance, attempting cleanup.");
        if let Err(e) = self.cleanup() {
            error!("ConsoleDriver: Error during cleanup in drop: {}", e);
        }
    }
}
