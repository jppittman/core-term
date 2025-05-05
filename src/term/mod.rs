// src/term/mod.rs

//! Defines the core `Term` struct, representing the terminal emulator's state,
//! and handles the overall processing of input bytes by dispatching parsed ANSI commands.

// Declare screen submodule
mod screen;

// Use necessary items from other modules
use crate::{
    ansi::{
        commands::{AnsiCommand, Attribute, C0Control, CsiCommand, EscCommand}, // Import command types
        AnsiProcessor, // Import the new processor
    },
    glyph::{Attributes, Glyph, REPLACEMENT_CHARACTER}, // Keep Glyph related imports
};
use log::{debug, trace, warn}; // Keep logging imports
use std::cmp::{max, min}; // Keep cmp imports

// --- Constants ---
/// Default tab stop interval.
pub(super) const DEFAULT_TAB_INTERVAL: usize = 8;
// Removed parser-specific constants (MAX_CSI_PARAMS, etc.) as they are handled by ansi crate

// --- Helper Structs/Enums ---
// Removed Utf8Decoder struct

/// Represents the cursor position (0-based).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Cursor {
    pub x: usize,
    pub y: usize,
}

/// Holds DEC Private Mode settings.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub(super) struct DecModes {
    /// DECCKM state (false = normal, true = application cursor keys)
    cursor_keys_app_mode: bool,
    /// DECOM state (false = absolute coords, true = relative to scroll region)
    origin_mode: bool,
    // Add other modes as needed (e.g., autowrap, reverse video, cursor visibility)
    // autowrap: bool, // Example: managed by MODE_WRAP in Term::mode now? Check consistency.
    // reverse_video: bool, // Example: managed by MODE_REVERSE in Term::mode now? Check consistency.
    cursor_visible: bool, // DECTCEM
}

// --- Main Term Struct Definition ---
/// Represents the state of the terminal emulator.
pub struct Term {
    /// Terminal width in columns.
    pub(super) width: usize,
    /// Terminal height in rows.
    pub(super) height: usize,
    /// Flag indicating if the cursor is currently wrapped to the next line logically.
    /// Set when a character is written in the last column. Reset by explicit cursor movement.
    pub(super) wrap_next: bool,
    /// Current cursor position.
    pub(super) cursor: Cursor,
    /// Primary screen buffer.
    pub(super) screen: Vec<Vec<Glyph>>,
    /// Alternate screen buffer.
    pub(super) alt_screen: Vec<Vec<Glyph>>,
    /// Flag indicating if the alternate screen buffer is currently active.
    pub(super) using_alt_screen: bool,
    /// Saved cursor position for the primary screen (used by DECSC/DECRC, SCOSC/SCORC).
    pub(super) saved_cursor: Cursor,
    /// Saved attributes for the primary screen.
    #[allow(dead_code)] // Potentially useful later, silence warning for now
    pub(super) saved_attributes: Attributes,
    /// Saved cursor position for the alternate screen.
    pub(super) saved_cursor_alt: Cursor,
    /// Saved attributes for the alternate screen.
    #[allow(dead_code)] // Potentially useful later, silence warning for now
    pub(super) saved_attributes_alt: Attributes,
    /// Current character attributes (bold, colors, etc.) to apply to new glyphs.
    pub(super) current_attributes: Attributes,
    /// Default character attributes (used for clearing, reset).
    pub(super) default_attributes: Attributes,
    /// DEC private mode settings.
    pub(super) dec_modes: DecModes,
    /// Tab stop positions (true if a tab stop exists at the index).
    pub(super) tabs: Vec<bool>,
    /// Top line of the scrolling region (0-based, inclusive).
    pub(super) scroll_top: usize,
    /// Bottom line of the scrolling region (0-based, inclusive).
    pub(super) scroll_bot: usize,
    /// Tracks which lines need redrawing (implementation detail, may change).
    #[allow(dead_code)] // Potentially useful later, silence warning for now
    pub(super) dirty: Vec<u8>,

    // --- New field for ANSI processing ---
    /// Processes byte stream into ANSI commands.
    ansi_processor: AnsiProcessor,

    // --- Removed fields ---
    // parser_state: ParserState, // Replaced by ansi_processor internal state
    // csi_params: Vec<u16>, // Handled by ansi_processor
    // csi_intermediates: Vec<char>, // Handled by ansi_processor
    // osc_string: String, // Handled by ansi_processor
    // utf8_decoder: Utf8Decoder, // Handled by ansi_processor's lexer
}

// --- Public API Implementation ---
impl Term {
    /// Creates a new terminal state with given dimensions.
    /// Minimum dimensions are 1x1.
    pub fn new(width: usize, height: usize) -> Self {
        let default_attributes = Attributes::default();
        let default_glyph = Glyph {
            c: ' ',
            attr: default_attributes,
        };
        let width = max(1, width);
        let height = max(1, height);

        let screen = vec![vec![default_glyph; width]; height];
        let alt_screen = vec![vec![default_glyph; width]; height];
        let tabs = (0..width).map(|i| i % DEFAULT_TAB_INTERVAL == 0).collect();
        let dirty = vec![1u8; height];

        Term {
            width,
            height,
            screen,
            alt_screen,
            using_alt_screen: false,
            cursor: Cursor::default(),
            wrap_next: false,
            saved_cursor: Cursor::default(),
            saved_attributes: default_attributes,
            saved_cursor_alt: Cursor::default(),
            saved_attributes_alt: default_attributes,
            current_attributes: default_attributes,
            default_attributes,
            dec_modes: DecModes {
                cursor_keys_app_mode: false, // Default state
                origin_mode: false,          // Default state
                cursor_visible: true,        // Default state (visible)
            },
            tabs,
            scroll_top: 0,
            scroll_bot: height.saturating_sub(1),
            dirty,
            ansi_processor: AnsiProcessor::new(), // Initialize the processor
        }
    }

    /// Gets the glyph at the specified coordinates (0-based).
    /// Returns `None` if coordinates are out of bounds.
    pub fn get_glyph(&self, x: usize, y: usize) -> Option<&Glyph> {
        let current_screen = if self.using_alt_screen {
            &self.alt_screen
        } else {
            &self.screen
        };
        current_screen.get(y).and_then(|row| row.get(x))
    }

    /// Gets the current cursor position (0-based column, 0-based row).
    pub fn get_cursor(&self) -> (usize, usize) {
        (self.cursor.x, self.cursor.y)
    }

    /// Gets the terminal dimensions (width, height) in cells.
    pub fn get_dimensions(&self) -> (usize, usize) {
        (self.width, self.height)
    }

    /// Processes a slice of bytes received from the PTY or input source.
    /// Uses the internal `AnsiProcessor` to parse bytes into commands and then dispatches them.
    pub fn process_bytes(&mut self, bytes: &[u8]) {
        // Log input if debug level is enabled
        if log::log_enabled!(log::Level::Debug) {
            let printable_bytes: String = bytes
                .iter()
                .map(|&b| {
                    if b.is_ascii_graphic() || b == b' ' {
                        b as char
                    } else if b == 0x1B {
                        '␛' // Represent ESC clearly
                    } else if b < 0x20 || b == 0x7F {
                        '.' // Represent C0/DEL as '.'
                    } else {
                        '?' // Represent other non-printable as '?'
                    }
                })
                .collect();
            debug!(
                "Processing {} bytes from PTY: '{}'",
                bytes.len(),
                printable_bytes
            );
        }

        // Feed bytes into the ANSI processor's lexer
        self.ansi_processor.process_bytes(bytes);

        // Retrieve and process the parsed commands
        let commands = self.ansi_processor.parser.take_commands();
        for command in commands {
            trace!("Dispatching command: {:?}", command);
            match command {
                AnsiCommand::Print(c) => {
                    // The ANSI processor handles UTF-8 decoding.
                    // Handle printable chars, including replacement chars for errors.
                    if c == REPLACEMENT_CHARACTER {
                        warn!("ANSI processor emitted REPLACEMENT_CHARACTER");
                    }
                    screen::handle_printable(self, c);
                }
                AnsiCommand::C0Control(ctrl) => {
                    // Handle C0 controls based on the enum variant
                    match ctrl {
                        C0Control::BS => screen::backspace(self),
                        C0Control::HT => screen::tab(self),
                        C0Control::LF | C0Control::VT | C0Control::FF => screen::newline(self), // Treat VT/FF as LF
                        C0Control::CR => screen::carriage_return(self),
                        C0Control::BEL => {
                            debug!("BEL received (bell not implemented via backend yet)");
                            // TODO: Call backend's bell function if available
                        }
                        C0Control::ESC => {
                            // This indicates an ESC *outside* a sequence, which is unusual
                            // but the parser handles state transitions correctly.
                            trace!("Standalone ESC processed (parser state handled)");
                        }
                        C0Control::SO | C0Control::SI => {
                            trace!("Ignoring C0 charset shift SO/SI ({:?})", ctrl);
                        }
                        C0Control::NUL => { /* Ignore NUL */ }
                        C0Control::DEL => { /* Ignore DEL */ }
                        // TODO: Handle other C0s like ENQ, ACK, NAK if needed
                        _ => {
                            trace!("Ignoring unhandled C0 control: {:?}", ctrl);
                        }
                    }
                }
                AnsiCommand::Csi(csi_cmd) => {
                    // Handle CSI commands based on the CsiCommand enum variant
                    match csi_cmd {
                        CsiCommand::CursorUp(n) => screen::move_cursor(self, 0, -(n as isize)),
                        CsiCommand::CursorDown(n) => screen::move_cursor(self, 0, n as isize),
                        CsiCommand::CursorForward(n) => screen::move_cursor(self, n as isize, 0),
                        CsiCommand::CursorBackward(n) => screen::move_cursor(self, -(n as isize), 0),
                        CsiCommand::CursorNextLine(n) => {
                            // Move down n lines, then to column 0
                            screen::move_cursor(self, 0, n as isize);
                            self.cursor.x = 0;
                            self.wrap_next = false;
                        }
                        CsiCommand::CursorPrevLine(n) => {
                            // Move up n lines, then to column 0
                            screen::move_cursor(self, 0, -(n as isize));
                            self.cursor.x = 0;
                            self.wrap_next = false;
                        }
                        CsiCommand::CursorCharacterAbsolute(n) => {
                            let target_x = n.saturating_sub(1) as usize; // 1-based to 0-based
                            self.cursor.x = min(target_x, self.width.saturating_sub(1));
                            self.wrap_next = false; // Explicit positioning resets wrap
                        }
                        CsiCommand::CursorPosition(row, col) => {
                            // Params are 1-based
                            screen::set_cursor_pos(self, col, row);
                        }
                        CsiCommand::EraseInDisplay(mode) => {
                            screen::handle_erase_in_display(self, mode)
                        }
                        CsiCommand::EraseInLine(mode) => screen::handle_erase_in_line(self, mode),
                        CsiCommand::InsertLine(n) => screen::insert_blank_lines(self, n as usize),
                        CsiCommand::DeleteLine(n) => screen::delete_lines(self, n as usize),
                        CsiCommand::DeleteCharacter(n) => screen::delete_chars(self, n as usize),
                        CsiCommand::InsertCharacter(n) => {
                            screen::insert_blank_chars(self, n as usize)
                        }
                        CsiCommand::ScrollUp(n) => screen::scroll_up(self, n as usize),
                        CsiCommand::ScrollDown(n) => screen::scroll_down(self, n as usize),
                        CsiCommand::SetGraphicsRendition(attrs) => {
                            screen::handle_sgr(self, &attrs)
                        }
                        CsiCommand::SetModePrivate(mode) => {
                            screen::handle_dec_mode_set(self, mode, true)
                        }
                        CsiCommand::ResetModePrivate(mode) => {
                            screen::handle_dec_mode_set(self, mode, false)
                        }
                        CsiCommand::SetMode(mode) => {
                            warn!("Ignoring standard Set Mode (SM): CSI {} h", mode);
                        }
                        CsiCommand::ResetMode(mode) => {
                            warn!("Ignoring standard Reset Mode (RM): CSI {} l", mode);
                        }
                        CsiCommand::SaveCursor => screen::save_cursor(self), // SCOSC
                        CsiCommand::RestoreCursor => screen::restore_cursor(self), // SCORC
                        CsiCommand::DeviceStatusReport(mode) => {
                            screen::handle_device_status_report(self, mode)
                        }
                        CsiCommand::ClearTabStops(mode) => screen::handle_clear_tab_stops(self, mode),
                        // Add cases for other handled CSI commands
                        _ => {
                            warn!("Unhandled parsed CSI command: {:?}", csi_cmd);
                        }
                    }
                }
                AnsiCommand::Esc(esc_cmd) => {
                    // Handle simple ESC sequences parsed by the ansi crate
                    match esc_cmd {
                        EscCommand::Index => screen::index(self),
                        EscCommand::NextLine => screen::newline(self),
                        EscCommand::SetTabStop => screen::set_horizontal_tabstop(self),
                        EscCommand::ReverseIndex => screen::reverse_index(self),
                        EscCommand::SaveCursor => screen::save_cursor(self), // DECSC
                        EscCommand::RestoreCursor => screen::restore_cursor(self), // DECRC
                        EscCommand::ResetToInitialState => screen::reset(self),
                        EscCommand::SelectCharacterSet(inter, final_char) => {
                            screen::handle_select_character_set(self, inter, final_char)
                        }
                        // Handle other ESC commands if needed
                        _ => {
                            warn!("Unhandled parsed ESC command: {:?}", esc_cmd);
                        }
                    }
                }
                AnsiCommand::Osc(data) => screen::handle_osc(self, &data),
                AnsiCommand::Dcs(data) => {
                    warn!("Unhandled DCS sequence, data len: {}", data.len());
                    // screen::handle_dcs(self, &data) // Define if needed
                }
                AnsiCommand::Pm(data) => {
                    warn!("Unhandled PM sequence, data len: {}", data.len());
                    // screen::handle_pm(self, &data) // Define if needed
                }
                AnsiCommand::Apc(data) => {
                    warn!("Unhandled APC sequence, data len: {}", data.len());
                    // screen::handle_apc(self, &data) // Define if needed
                }
                AnsiCommand::StringTerminator => { /* Usually no action needed here */ }
                AnsiCommand::C1Control(byte) => {
                    // Most C1 are handled via ESC or specific command types now.
                    // Handle any remaining C1 codes if necessary.
                    warn!("Unhandled C1 control byte: 0x{:02X}", byte);
                }
                AnsiCommand::Ignore(byte) => {
                    trace!("ANSI processor ignored byte: 0x{:02X}", byte);
                }
                AnsiCommand::Error(byte) => {
                    warn!("ANSI processor reported error at byte: 0x{:02X}", byte);
                }
            }
        }
    }

    // Removed process_byte_in_parser as it's handled by AnsiProcessor now

    /// Resizes the terminal emulator state, including screen buffers and cursor position.
    /// Resets the scrolling region to the full screen size.
    pub fn resize(&mut self, new_width: usize, new_height: usize) {
        let old_height = self.height;
        let old_width = self.width;
        // Ensure minimum dimensions of 1x1.
        let new_height = max(1, new_height);
        let new_width = max(1, new_width);

        // Reset scroll region *before* calling screen::resize, as screen::resize
        // might clamp cursor positions based on the old region if called first.
        self.scroll_top = 0;
        self.scroll_bot = new_height.saturating_sub(1);

        if old_height == new_height && old_width == new_width {
            debug!(
                "Resize called with same dimensions ({}, {}), only reset scroll region.",
                new_width, new_height
            );
            return;
        }
        debug!(
            "Resizing terminal from {}x{} to {}x{}",
            old_width, old_height, new_width, new_height
        );

        // Call the screen submodule's resize logic to handle buffer adjustments.
        screen::resize(self, new_width, new_height);

        // Update the main Term struct's dimensions *after* screen::resize.
        self.width = new_width;
        self.height = new_height;

        // Mark all lines dirty after resize
        self.dirty.resize(new_height, 1);
        self.dirty.fill(1);
    }
}

// --- Private Helper Functions (Moved to screen module or removed) ---
// Removed clear_csi_params, clear_osc_string, push_csi_param, next_csi_param,
// get_csi_param, get_csi_param_or_0 as these are handled by AnsiProcessor/commands.

// --- State Transition Handlers (Removed) ---
// Removed handle_ground_control_or_esc, handle_escape_byte, handle_csi_entry_byte,
// handle_csi_param_byte, handle_csi_intermediate_byte, handle_csi_ignore_byte,
// handle_osc_string_byte as these are replaced by the AnsiProcessor logic.

// --- Dispatch Functions (Removed) ---
// Removed csi_dispatch, handle_osc_dispatch, handle_sgr, handle_dec_mode_enable,
// handle_dec_mode_disable, apply_dec_mode_actions. Logic moved to screen module
// or handled directly in process_bytes match statement.

// Keep tests module declaration
#[cfg(test)]
mod tests;

// Remove parser submodule declaration if the file is removed
// mod parser;

