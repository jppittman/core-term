//! This module defines the core terminal emulator logic.
//! It acts as a state machine processing inputs and producing actions.

mod screen;
pub mod unicode;

// Standard library imports
use std::cmp::min;

// Crate-level imports
use crate::{
    ansi::commands::{
            AnsiCommand, C0Control, CsiCommand, EscCommand, Attribute,
            Color as AnsiColor,
    },
    term::unicode::get_char_display_width,
    backends::BackendEvent,
    glyph::{Glyph, Attributes, AttrFlags, Color as GlyphColor, NamedColor},
    term::screen::{Screen, Cursor}, // Ensure Cursor is imported if used for type hints
};

// Logging
use log::{debug, trace, warn};

/// Default tab interval.
pub const DEFAULT_TAB_INTERVAL: u8 = 8;

// --- Locally Defined Constants for Modes ---
const ERASE_MODE_TO_END: u16 = 0;
const ERASE_MODE_TO_START: u16 = 1;
const ERASE_MODE_ALL: u16 = 2;
const ERASE_MODE_SCROLLBACK: u16 = 3; // Typically specific to ED

// DEC Private Mode numbers (from common terminal documentation like ECMA-48, VT100/VT220 manuals)
const DECCKM_MODE: u16 = 1;    // Cursor Keys Mode (Application vs. Normal)
const DECOM_MODE: u16 = 6;     // Origin Mode (Absolute vs. Relative to scroll region)
const DECTCEM_MODE: u16 = 25;  // Text Cursor Enable Mode (Show/Hide cursor)
// Alternate Screen Buffer modes
const ALT_SCREEN_BUF_1047_MODE: u16 = 1047; // Use Alternate Screen Buffer, clears on switch (obsolete by 1049)
const CURSOR_SAVE_RESTORE_1048_MODE: u16 = 1048; // Save/Restore cursor (used with 1049)
const ALT_SCREEN_SAVE_RESTORE_1049_MODE: u16 = 1049; // Use Alternate Screen Buffer, save/restore cursor, clear on switch


/// Actions that the terminal emulator signals to the orchestrator.
#[derive(Debug, Clone, PartialEq)]
pub enum EmulatorAction {
    WritePty(Vec<u8>),
    SetTitle(String),
    RingBell,
    RequestRedraw,
    SetCursorVisibility(bool),
    // Potentially others: SetClipboard, etc.
}

/// Inputs that the terminal emulator processes.
#[derive(Debug, Clone)]
pub enum EmulatorInput {
    Ansi(AnsiCommand),
    User(BackendEvent), // e.g., keyboard input from the driver
    RawChar(char),      // For direct character printing if needed outside ANSI
}

/// Represents the terminal's private modes (DECSET/DECRST).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DecPrivateModes {
    pub origin_mode: bool,          // DECOM: Cursor addressing relative to margins
    pub cursor_visible: bool,       // DECTCEM: Show/Hide cursor
    pub cursor_keys_app_mode: bool, // DECCKM: Application vs. Normal cursor keys
    pub using_alt_screen: bool,     // Tracks if 1047/1049 mode is active
    // Add other DEC modes as needed, e.g., mouse reporting modes, auto-wrap (DECAWM)
}

/// Represents the character sets G0, G1, G2, G3.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CharacterSet {
    Ascii,
    UkNational,
    DecLineDrawing,
    // Add others as needed, e.g., DecSpecialGraphicsAndLineDrawing
}

impl CharacterSet {
    /// Maps a character designator (from an ESC sequence) to a CharacterSet enum.
    fn from_char(ch: char) -> Self {
        match ch {
            'B' => CharacterSet::Ascii,         // US ASCII
            'A' => CharacterSet::UkNational,    // UK National
            '0' => CharacterSet::DecLineDrawing,// DEC Line Drawing Set
            // TODO: Add more mappings as per standards (e.g., '1', '2', 'U', 'K')
            _ => {
                warn!("Unsupported character set designator: {}", ch);
                CharacterSet::Ascii // Default to ASCII if unknown
            }
        }
    }
}

/// Represents the mode types for SM/RM sequences (Set Mode / Reset Mode).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    DecPrivate(u16), // For CSI ? Pm h/l sequences
    Standard(u16),   // For CSI Pm h/l sequences (less common for terminal setup)
}


/// The core terminal emulator.
///
/// This struct acts as a state machine, processing `EmulatorInput` (derived from
/// PTY output or user actions) and producing `EmulatorAction`s (side effects
/// like writing to PTY or requesting a redraw). It manages the terminal grid,
/// cursor state, attributes, and various modes.
///
/// The `TerminalEmulator` is designed to be backend-agnostic. It does not perform
/// direct I/O or rendering; these tasks are delegated to the `Orchestrator` and
/// `Driver`/`Renderer` components based on the `EmulatorAction`s it signals.
pub struct TerminalEmulator {
    /// The screen state, including grids, scrollback, cursor, etc.
    screen: Screen,
    /// The logical cursor state (position and attributes).
    /// Position can be relative to scroll margins if origin_mode is active.
    cursor: Cursor, // This is the logical cursor. Screen also has its own absolute cursor.
    /// Current SGR attributes for new characters.
    current_attributes: Attributes,
    /// Saved cursor state for DECSC/DECRC.
    saved_cursor_state: Option<Cursor>, // Stores logical cursor
    /// Saved SGR attributes for DECSC/DECRC.
    saved_attributes_state: Option<Attributes>,
    /// Tracks various DEC private modes.
    dec_modes: DecPrivateModes,
    /// Active character sets (G0, G1, G2, G3).
    active_charsets: [CharacterSet; 4],
    /// Currently selected G-level (0-3) for character interpretation.
    active_charset_g_level: usize,
    /// Tracks which lines (absolute indices) have been modified and need redraw.
    dirty_lines: Vec<usize>, // TODO: Consider a more efficient structure like a BitVec or HashSet.
    /// Flag to track if the last printed character was wide, for cursor positioning.
    last_char_was_wide: bool,
}

impl TerminalEmulator {
    /// Creates a new `TerminalEmulator`.
    ///
    /// # Arguments
    /// * `width` - Initial width of the terminal in columns.
    /// * `height` - Initial height of the terminal in rows.
    /// * `scrollback_limit` - Maximum number of lines for the scrollback buffer.
    pub fn new(width: usize, height: usize, scrollback_limit: usize) -> Self {
        let default_attributes = Attributes::default();
        let screen = Screen::new(width, height, scrollback_limit);
        // Logical cursor starts at (0,0) with default attributes.
        // If origin mode is on, (0,0) logical is screen.scroll_top absolute.
        let cursor = Cursor { x: 0, y: 0, attributes: default_attributes };

        TerminalEmulator {
            screen,
            cursor,
            current_attributes: default_attributes,
            saved_cursor_state: None,
            saved_attributes_state: None,
            dec_modes: DecPrivateModes {
                cursor_visible: true, // Cursor is visible by default
                ..Default::default()
            },
            active_charsets: [
                CharacterSet::Ascii, CharacterSet::Ascii, // G0, G1
                CharacterSet::Ascii, CharacterSet::Ascii, // G2, G3
            ],
            active_charset_g_level: 0, // Default to G0
            dirty_lines: (0..height).collect(), // Initially, all lines are dirty.
            last_char_was_wide: false,
        }
    }

    /// Interprets an `EmulatorInput` and updates the terminal state.
    ///
    /// This is the main entry point for processing data from the PTY or user actions.
    /// It may return an `EmulatorAction` if the input requires an external side effect.
    /// It also manages dirty line tracking for rendering optimization.
    pub fn interpret_input(&mut self, input: EmulatorInput) -> Option<EmulatorAction> {
        // Before processing, sync the emulator's logical cursor and modes to the screen sub-component.
        // This ensures screen operations (like scrolling, clearing) use the correct context.
        self.sync_emulator_state_to_screen_for_processing();

        let mut action = match input {
            EmulatorInput::Ansi(command) => self.handle_ansi_command(command),
            EmulatorInput::User(event) => self.handle_backend_event(event),
            EmulatorInput::RawChar(ch) => {
                self.print_char(ch); // Directly print a character
                None // Printing itself usually just dirties lines, handled below.
            }
        };

        // After processing, sync the screen's cursor state back to the emulator's logical cursor.
        self.sync_screen_state_to_emulator_after_processing();

        // If no specific action was generated but lines were dirtied,
        // signal a general RequestRedraw.
        if action.is_none() && !self.dirty_lines.is_empty() {
            action = Some(EmulatorAction::RequestRedraw);
        }
        action
    }

    /// Handles a parsed `AnsiCommand`.
    fn handle_ansi_command(&mut self, command: AnsiCommand) -> Option<EmulatorAction> {
        // ... (rest of the function remains the same as in term_mod_rs_fix_1, no changes needed here for wcwidth) ...
        // For brevity, I'm omitting the identical parts of this function.
        // The important change is in `print_char` below.
        match command {
            AnsiCommand::C0Control(c0) => match c0 {
                C0Control::BS => { self.backspace(); None }
                C0Control::HT => { self.horizontal_tab(); None }
                C0Control::LF | C0Control::VT | C0Control::FF => { self.newline(true); None } // LF, VT, FF often treated similarly
                C0Control::CR => { self.carriage_return(); None }
                C0Control::SO => { self.set_g_level(1); None } // Shift Out (LS1) -> G1
                C0Control::SI => { self.set_g_level(0); None } // Shift In (LS0) -> G0
                C0Control::BEL => Some(EmulatorAction::RingBell),
                // TODO: Handle other C0 controls if necessary (e.g., NUL, DEL, ENQ)
                _ => { debug!("Unhandled C0 control: {:?}", c0); None }
            },
            AnsiCommand::Esc(esc) => match esc {
                EscCommand::SetTabStop => { self.screen.set_tabstop(self.cursor.x); None } // HTS
                EscCommand::Index => { self.index(); None } // IND
                EscCommand::NextLine => { self.newline(true); self.carriage_return(); None } // NEL
                EscCommand::ReverseIndex => { self.reverse_index(); None } // RI
                EscCommand::SaveCursor => { self.save_cursor_dec(); None } // DECSC
                EscCommand::RestoreCursor => { self.restore_cursor_dec(); None } // DECRC
                EscCommand::SelectCharacterSet(intermediate_char, final_char) => {
                    // Determine G-level based on intermediate: ( -> G0, ) -> G1, * -> G2, + -> G3
                    let g_idx = match intermediate_char {
                        '(' => 0,
                        ')' => 1,
                        '*' => 2,
                        '+' => 3,
                        _ => {
                            warn!("Unsupported G-set designator intermediate: {}", intermediate_char);
                            0 // Default to G0 on error
                        }
                    };
                    self.designate_character_set(g_idx, CharacterSet::from_char(final_char));
                    None
                }
                // TODO: Handle other Esc commands (e.g., RIS, SS2, SS3)
                _ => { debug!("Unhandled Esc command: {:?}", esc); None }
            },
            AnsiCommand::Csi(csi) => match csi {
                // Cursor movement
                CsiCommand::CursorUp(n) => { self.cursor_up(n.max(1) as usize); None }
                CsiCommand::CursorDown(n) => { self.cursor_down(n.max(1) as usize); None }
                CsiCommand::CursorForward(n) => { self.cursor_forward(n.max(1) as usize); None }
                CsiCommand::CursorBackward(n) => { self.cursor_backward(n.max(1) as usize); None }
                CsiCommand::CursorNextLine(n) => { self.cursor_down(n.max(1) as usize); self.carriage_return(); None }
                CsiCommand::CursorPrevLine(n) => { self.cursor_up(n.max(1) as usize); self.carriage_return(); None }
                CsiCommand::CursorCharacterAbsolute(n) => { self.cursor_to_column(n.saturating_sub(1) as usize); None }
                CsiCommand::CursorPosition(r, c) => { self.cursor_to_pos(r.saturating_sub(1) as usize, c.saturating_sub(1) as usize); None }
                // Erasing
                CsiCommand::EraseInDisplay(mode) => { self.erase_in_display(mode); None }
                CsiCommand::EraseInLine(mode) => { self.erase_in_line(mode); None }
                // Editing
                CsiCommand::InsertCharacter(n) => { self.insert_blank_chars(n.max(1) as usize); None }
                CsiCommand::DeleteCharacter(n) => { self.delete_chars(n.max(1) as usize); None }
                CsiCommand::InsertLine(n) => { self.insert_lines(n.max(1) as usize); None }
                CsiCommand::DeleteLine(n) => { self.delete_lines(n.max(1) as usize); None }
                // Attributes
                CsiCommand::SetGraphicsRendition(attrs_vec) => { self.handle_sgr_attributes(attrs_vec); None }
                // Mode setting
                CsiCommand::SetMode(mode_num) => self.handle_set_mode(Mode::Standard(mode_num), true),
                CsiCommand::ResetMode(mode_num) => self.handle_set_mode(Mode::Standard(mode_num), false),
                CsiCommand::SetModePrivate(mode_num) => self.handle_set_mode(Mode::DecPrivate(mode_num), true),
                CsiCommand::ResetModePrivate(mode_num) => self.handle_set_mode(Mode::DecPrivate(mode_num), false),
                // Status reports
                CsiCommand::DeviceStatusReport(dsr_param) => {
                    // DSR 6 (CPR - Cursor Position Report)
                    if dsr_param == 0 || dsr_param == 6 { // DSR 0 is often treated as an alias for DSR 6 by some terminals/apps
                        let (cur_x, cur_y_logical) = self.cursor_pos(); // Gets logical cursor position
                        // CPR reports 1-based absolute screen coordinates.
                        // If origin mode is on, logical y is relative to scroll_top.
                        let cur_y_abs = if self.dec_modes.origin_mode {
                            self.screen.scroll_top() + cur_y_logical
                        } else {
                            cur_y_logical
                        };
                        let response = format!("\x1B[{};{}R", cur_y_abs + 1, cur_x + 1);
                        Some(EmulatorAction::WritePty(response.into_bytes()))
                    } else if dsr_param == 5 { // DSR 5 (Status Report "OK")
                        Some(EmulatorAction::WritePty(b"\x1B[0n".to_vec())) // Response is CSI 0 n
                    } else {
                        warn!("Unhandled DSR parameter: {}", dsr_param);
                        None
                    }
                }
                CsiCommand::EraseCharacter(n) => { self.erase_chars(n.max(1) as usize); None }
                CsiCommand::ScrollUp(n) => { self.scroll_up(n.max(1) as usize); None } // SU
                CsiCommand::ScrollDown(n) => { self.scroll_down(n.max(1) as usize); None } // SD
                // Cursor saving/restoring (DECSC/DECRC are usually via Esc 7 / Esc 8)
                // SCOSC/SCORC are CSI s / CSI u
                CsiCommand::SaveCursor => { self.save_cursor_dec(); None} // CSI s (SCOSC)
                CsiCommand::RestoreCursor => { self.restore_cursor_dec(); None} // CSI u (SCORC)
                // TODO: Add more CSI commands (e.g., TBC, CHA, VPA, REP)
                _ => { debug!("Unhandled CSI command: {:?}", csi); None }
            },
            AnsiCommand::Osc(data) => self.handle_osc(data),
            AnsiCommand::Print(ch) => { self.print_char(ch); None }
            // TODO: Handle Dcs, Pm, Apc, StringTerminator, Ignore, Error variants
            _ => { debug!("Unhandled ANSI command type: {:?}", command); None }
        }
    }

    /// Handles a `BackendEvent` (typically user input).
    fn handle_backend_event(&mut self, event: BackendEvent) -> Option<EmulatorAction> {
        // Define some common key codes if not importing from a specific library here
        // These are illustrative; actual values depend on what the backend sends.
        const KEY_RETURN: u32 = 0xFF0D;       // Enter/Return
        const KEY_BACKSPACE: u32 = 0xFF08;    // Backspace
        const KEY_TAB: u32 = 0xFF09;          // Tab
        const KEY_ISO_LEFT_TAB: u32 = 0xFE20; // Shift+Tab (common X11 keysym)
        const KEY_ESCAPE: u32 = 0xFF1B;       // Escape

        // Arrow keys (common X11 keysyms)
        const KEY_UP_ARROW: u32 = 0xFF52;
        const KEY_DOWN_ARROW: u32 = 0xFF54;
        const KEY_RIGHT_ARROW: u32 = 0xFF53;
        const KEY_LEFT_ARROW: u32 = 0xFF51;
        // Add more for KP_ keys, Home, End, etc. if needed.
        // Example: const KEY_KP_ENTER: u32 = 0xFF8D;

        match event {
            BackendEvent::Key { keysym, text } => {
                let mut bytes_to_send: Vec<u8> = Vec::new();

                // Prioritize text if it's a simple printable character and not a special key.
                // This check helps differentiate between, e.g., 'a' and Ctrl+A.
                // A more robust solution might involve checking modifier states from BackendEvent.
                if !text.is_empty() && text.chars().all(|c| !c.is_control()) && !(
                    // List keysyms where we prefer to send a specific sequence over the text.
                    keysym == KEY_RETURN || // keysym == KEY_KP_ENTER ||
                    keysym == KEY_BACKSPACE ||
                    keysym == KEY_TAB || keysym == KEY_ISO_LEFT_TAB ||
                    keysym == KEY_ESCAPE ||
                    (keysym >= KEY_LEFT_ARROW && keysym <= KEY_UP_ARROW) // Basic arrows
                    // Add other function/navigation keysym ranges here if they produce text
                ) {
                    bytes_to_send.extend(text.as_bytes());
                } else {
                    // Handle special keys by keysym, potentially sending ANSI sequences.
                    match keysym {
                        KEY_RETURN /* | KEY_KP_ENTER */ => bytes_to_send.push(b'\r'), // CR
                        KEY_BACKSPACE => bytes_to_send.push(0x08), // BS
                        KEY_TAB /* | KEY_KP_TAB */ => bytes_to_send.push(b'\t'), // HT
                        KEY_ISO_LEFT_TAB => bytes_to_send.extend_from_slice(b"\x1b[Z"), // CSI Z (Shift Tab)
                        KEY_ESCAPE => bytes_to_send.push(0x1B), // ESC

                        // Arrow keys: send appropriate CSI sequences based on DECCKM mode
                        KEY_UP_ARROW /* | KEY_KP_UP */ => {
                            bytes_to_send.extend_from_slice(if self.dec_modes.cursor_keys_app_mode { b"\x1bOA" } else { b"\x1b[A" });
                        }
                        KEY_DOWN_ARROW /* | KEY_KP_DOWN */ => {
                            bytes_to_send.extend_from_slice(if self.dec_modes.cursor_keys_app_mode { b"\x1bOB" } else { b"\x1b[B" });
                        }
                        KEY_RIGHT_ARROW /* | KEY_KP_RIGHT */ => {
                            bytes_to_send.extend_from_slice(if self.dec_modes.cursor_keys_app_mode { b"\x1bOC" } else { b"\x1b[C" });
                        }
                        KEY_LEFT_ARROW /* | KEY_KP_LEFT */ => {
                            bytes_to_send.extend_from_slice(if self.dec_modes.cursor_keys_app_mode { b"\x1bOD" } else { b"\x1b[D" });
                        }
                        // TODO: Add Home, End, PgUp, PgDn, Insert, Delete, F-keys, etc.
                        // These will also need to check DECCKM or other modes.
                        _ => {
                            // Fallback for other keysyms: if text is a single non-control char, send it.
                            if text.chars().count() == 1 && text.chars().next().map_or(false, |c| !c.is_control()) {
                                bytes_to_send.extend(text.as_bytes());
                            } else if !text.is_empty() && text.chars().all(|c| c.is_control() || c.is_ascii_alphanumeric() || c.is_ascii_punctuation()) {
                                // If text contains control characters or simple ASCII, send it.
                                // This might be for Ctrl+<key> combinations that produce a single control char.
                                bytes_to_send.extend(text.as_bytes());
                            } else {
                                trace!("Unhandled keysym: {} (text: '{}')", keysym, text);
                            }
                        }
                    }
                }

                if !bytes_to_send.is_empty() {
                    return Some(EmulatorAction::WritePty(bytes_to_send));
                }
            }
            BackendEvent::Resize { width_px, height_px, .. } => {
                // This event is typically handled by the Orchestrator, which then calls term_emulator.resize().
                // No direct action from TerminalEmulator itself, but log it.
                debug!("BackendEvent::Resize received by TerminalEmulator: {}px x {}px. Orchestrator should call term.resize() with char dimensions.", width_px, height_px);
            }
            BackendEvent::FocusGained => {
                debug!("Focus gained by terminal.");
                // Some terminals send sequences on focus change if a mode is set (e.g., CSI I / CSI O)
                // For now, just log. This might become an EmulatorAction if needed.
            }
            BackendEvent::FocusLost => {
                debug!("Focus lost by terminal.");
            }
            BackendEvent::CloseRequested => {
                // This is primarily for the Orchestrator to handle application exit.
                debug!("Backend signaled close/quit. Orchestrator should handle.");
            }
        }
        None
    }

    /// Resizes the terminal display grid.
    ///
    /// # Arguments
    /// * `cols` - New width in columns.
    /// * `rows` - New height in rows.
    pub fn resize(&mut self, cols: usize, rows: usize) {
        let current_scrollback_limit = self.screen.scrollback_limit();
        self.screen.resize(cols, rows, current_scrollback_limit);

        // Clamp logical cursor to new dimensions, considering origin mode.
        let max_logical_y = if self.dec_modes.origin_mode {
            self.screen.scroll_bot() - self.screen.scroll_top() // Max y is relative to scroll region size
        } else {
            self.screen.height.saturating_sub(1) // Max y is absolute screen height
        };
        self.cursor.x = min(self.cursor.x, self.screen.width.saturating_sub(1));
        self.cursor.y = min(self.cursor.y, max_logical_y);

        self.mark_all_lines_dirty();
        debug!("Terminal resized to {}x{}. Logical cursor at ({}, {}). All lines marked dirty.", cols, rows, self.cursor.x, self.cursor.y);
    }

    /// Marks all lines on the screen as dirty, forcing a full redraw.
    fn mark_all_lines_dirty(&mut self) {
        self.dirty_lines = (0..self.screen.height).collect();
        self.screen.mark_all_dirty(); // Also tell the screen sub-component
    }

    /// Marks a single absolute line `y_abs` as dirty.
    fn mark_line_dirty(&mut self, y_abs: usize) {
        if y_abs < self.screen.height && !self.dirty_lines.contains(&y_abs) {
            self.dirty_lines.push(y_abs);
        }
        self.screen.mark_line_dirty(y_abs); // Also tell the screen sub-component
    }

    /// Returns a list of dirty line indices and clears the internal dirty list.
    /// This also incorporates dirty flags from the `Screen` sub-component.
    pub fn take_dirty_lines(&mut self) -> Vec<usize> {
        // Use a HashSet to collect unique dirty line indices from both sources.
        let mut all_dirty_indices: std::collections::HashSet<usize> = self.dirty_lines.drain(..).collect();

        // Add lines marked dirty directly in the screen component.
        for (idx, &is_dirty_flag) in self.screen.dirty.iter().enumerate() {
            if is_dirty_flag != 0 { // Assuming 0 means not dirty, non-zero means dirty.
                all_dirty_indices.insert(idx);
            }
        }
        self.screen.clear_dirty_flags(); // Clear screen's internal dirty flags.

        let mut sorted_dirty_lines: Vec<usize> = all_dirty_indices.into_iter().collect();
        sorted_dirty_lines.sort_unstable(); // Sort for consistent processing order.
        sorted_dirty_lines
    }

    /// Gets a clone of the glyph at the specified absolute `(x_abs, y_abs)` coordinates.
    pub fn get_glyph(&self, x_abs: usize, y_abs: usize) -> Glyph {
        self.screen.get_glyph(x_abs, y_abs)
    }

    /// Returns the terminal's current dimensions (columns, rows).
    pub fn dimensions(&self) -> (usize, usize) {
        (self.screen.width, self.screen.height)
    }

    /// Returns the logical cursor position `(x, y)`.
    /// `y` is relative to scroll margins if origin mode is active.
    pub fn cursor_pos(&self) -> (usize, usize) {
        (self.cursor.x, self.cursor.y)
    }

    /// Returns the absolute on-screen cursor position.
    /// This is needed by the Renderer.
    pub fn get_screen_cursor_pos(&self) -> (usize, usize) {
        // self.screen.cursor already holds the absolute position
        // after sync_emulator_state_to_screen_for_processing has been called.
        // However, to be safe and always provide the current state, we can recalculate.
        let abs_x = self.cursor.x;
        let abs_y = if self.dec_modes.origin_mode {
            (self.screen.scroll_top() + self.cursor.y).min(self.screen.scroll_bot())
        } else {
            self.cursor.y.min(self.screen.height.saturating_sub(1))
        };
        (abs_x.min(self.screen.width.saturating_sub(1)), abs_y)
    }

    /// Returns true if the cursor should be visible.
    pub fn is_cursor_visible(&self) -> bool {
        self.dec_modes.cursor_visible
    }


    /// Returns true if the alternate screen buffer is currently active.
    pub fn is_alt_screen_active(&self) -> bool {
        self.screen.alt_screen_active // Or self.dec_modes.using_alt_screen if that's the source of truth
    }

    /// Syncs the emulator's logical cursor state and relevant modes to the `Screen` sub-component.
    /// This is done before `Screen` performs operations that depend on cursor position or modes.
    fn sync_emulator_state_to_screen_for_processing(&mut self) {
        self.screen.cursor.x = self.cursor.x.min(self.screen.width.saturating_sub(1));

        // Convert logical cursor y to absolute screen y for screen.cursor
        if self.dec_modes.origin_mode {
            let scroll_top = self.screen.scroll_top();
            let max_relative_y = self.screen.scroll_bot().saturating_sub(scroll_top);
            let clamped_relative_y = self.cursor.y.min(max_relative_y);
            self.screen.cursor.y = scroll_top + clamped_relative_y;
        } else {
            self.screen.cursor.y = self.cursor.y.min(self.screen.height.saturating_sub(1));
        }
        // Final clamp to ensure screen.cursor.y is always within physical screen bounds.
        self.screen.cursor.y = self.screen.cursor.y.min(self.screen.height.saturating_sub(1));


        self.screen.cursor.attributes = self.current_attributes;
        self.screen.origin_mode = self.dec_modes.origin_mode; // Inform screen about origin mode
    }

    /// Syncs the `Screen` sub-component's cursor state back to the emulator's logical cursor.
    /// This is done after `Screen` operations that might have modified the cursor.
    fn sync_screen_state_to_emulator_after_processing(&mut self) {
        self.cursor.x = self.screen.cursor.x; // X is always absolute

        // Convert absolute screen y back to logical y for emulator.cursor
        if self.dec_modes.origin_mode {
            let scroll_top = self.screen.scroll_top();
            // Ensure screen.cursor.y is at least scroll_top before subtracting
            self.cursor.y = self.screen.cursor.y.saturating_sub(scroll_top);
        } else {
            self.cursor.y = self.screen.cursor.y;
        }
        // current_attributes are managed by TerminalEmulator, not synced back from screen.cursor.attributes
    }


    /// Returns the default glyph (space with current attributes) for filling cleared areas.
    fn default_fill_glyph(&self) -> Glyph {
        Glyph { c: ' ', attr: self.current_attributes }
    }

    /// Prints a character to the screen at the current cursor position.
    /// Handles character width, wrapping, and scrolling.
    fn print_char(&mut self, ch: char) {
        let ch_to_print = self.map_char_to_active_charset(ch);
        let char_width = get_char_display_width(ch_to_print); // Uses wcwidth
        let current_screen_width = self.screen.width;

        // Line wrapping logic
        if self.last_char_was_wide && self.cursor.x == 0 {
            // Cursor already wrapped due to previous wide char, do nothing extra here.
        } else if self.cursor.x >= current_screen_width || // Cursor is at or past the end
                  (char_width == 2 && self.cursor.x >= current_screen_width.saturating_sub(1)) { // Wide char won't fit
            // Perform line feed (move to next line, column 0)
            let current_logical_y = self.cursor.y; // Before potential scroll
            let max_logical_y = if self.dec_modes.origin_mode {
                self.screen.scroll_bot() - self.screen.scroll_top()
            } else {
                self.screen.height.saturating_sub(1)
            };

            if self.cursor.y == max_logical_y { // If at the bottom of screen/region
                self.screen.scroll_up_serial(1, self.default_fill_glyph());
                // Mark all lines within the scrolling region as dirty after scroll
                for y_dirty in self.screen.scroll_top()..=self.screen.scroll_bot() { self.mark_line_dirty(y_dirty); }
            } else {
                self.cursor.y += 1;
            }
            self.cursor.x = 0;
            // Mark the original line (before potential scroll/newline) and the new line as dirty
            let original_abs_y = (if self.dec_modes.origin_mode { self.screen.scroll_top() + current_logical_y } else { current_logical_y }).min(self.screen.height.saturating_sub(1));
            let new_abs_y = (if self.dec_modes.origin_mode { self.screen.scroll_top() + self.cursor.y } else { self.cursor.y }).min(self.screen.height.saturating_sub(1));
            self.mark_line_dirty(original_abs_y);
            if original_abs_y != new_abs_y { self.mark_line_dirty(new_abs_y); }
        }

        // Determine absolute Y position for writing to screen grid
        let target_y_abs = if self.dec_modes.origin_mode {
            (self.screen.scroll_top() + self.cursor.y).min(self.screen.scroll_bot())
        } else {
            self.cursor.y.min(self.screen.height.saturating_sub(1))
        };

        // Write glyph to screen
        if target_y_abs < self.screen.height { // Ensure y is within physical bounds
            self.screen.set_glyph(self.cursor.x, target_y_abs, Glyph { c: ch_to_print, attr: self.current_attributes });
            self.mark_line_dirty(target_y_abs); // Mark the line as dirty

            if char_width == 2 && self.cursor.x + 1 < current_screen_width {
                // Set placeholder for the second cell of a wide character
                 self.screen.set_glyph(self.cursor.x + 1, target_y_abs, Glyph { c: '\0', attr: self.current_attributes });
                 // The line is already marked dirty.
            }
        } else {
            warn!("print_char: Attempted to print at y_abs {} out of bounds (height {})", target_y_abs, self.screen.height);
        }

        self.last_char_was_wide = char_width == 2;
        self.cursor.x += char_width; // Advance cursor by character display width
    }


    /// Maps a character to its representation in the currently active character set.
    fn map_char_to_active_charset(&self, ch: char) -> char {
        let current_set = self.active_charsets[self.active_charset_g_level];
        match current_set {
            CharacterSet::Ascii => ch,
            CharacterSet::UkNational => if ch == '#' { '£' } else { ch }, // Pound sign for '#'
            CharacterSet::DecLineDrawing => map_to_dec_line_drawing(ch),
            // Add other character set mappings here
        }
    }

    // --- C0 Control Handlers ---
    fn backspace(&mut self) { if self.cursor.x > 0 { self.cursor.x -= 1; } self.last_char_was_wide = false; }
    fn horizontal_tab(&mut self) {
        let next_stop = self.screen.get_next_tabstop(self.cursor.x).unwrap_or(self.screen.width.saturating_sub(1));
        self.cursor.x = min(next_stop, self.screen.width.saturating_sub(1)); self.last_char_was_wide = false;
    }
    fn newline(&mut self, is_line_feed: bool) { // LF, FF, VT
        let current_logical_y = self.cursor.y;
        let max_logical_y = if self.dec_modes.origin_mode { self.screen.scroll_bot() - self.screen.scroll_top() } else { self.screen.height.saturating_sub(1) };

        if current_logical_y == max_logical_y { // If at the bottom of screen/region
            self.screen.scroll_up_serial(1, self.default_fill_glyph());
            for y_dirty in self.screen.scroll_top()..=self.screen.scroll_bot() { self.mark_line_dirty(y_dirty); }
        } else if self.cursor.y < max_logical_y { // If not at bottom, move down
            self.cursor.y += 1;
        }
        if is_line_feed { self.cursor.x = 0; } // LF and FF also imply CR usually
        self.last_char_was_wide = false;
        let original_abs_y = (if self.dec_modes.origin_mode { self.screen.scroll_top() + current_logical_y } else { current_logical_y }).min(self.screen.height.saturating_sub(1));
        let new_abs_y = (if self.dec_modes.origin_mode { self.screen.scroll_top() + self.cursor.y } else { self.cursor.y }).min(self.screen.height.saturating_sub(1));
        self.mark_line_dirty(original_abs_y); if original_abs_y != new_abs_y { self.mark_line_dirty(new_abs_y); }

    }
    fn carriage_return(&mut self) { self.cursor.x = 0; self.last_char_was_wide = false; }

    // --- Character Set Selection ---
    fn set_g_level(&mut self, g_level: usize) {
        if g_level < self.active_charsets.len() {
            self.active_charset_g_level = g_level;
            trace!("Switched to G{} character set mapping.", g_level);
        } else {
            warn!("Attempted to set invalid G-level: {}", g_level);
        }
    }
    fn designate_character_set(&mut self, g_set_index: usize, charset: CharacterSet) {
        if g_set_index < self.active_charsets.len() {
            self.active_charsets[g_set_index] = charset;
            trace!("Designated G{} to {:?}", g_set_index, charset);
        } else {
            warn!("Invalid G-set index for designate charset: {}", g_set_index);
        }
    }

    // --- ESC Command Handlers ---
    fn index(&mut self) { // IND
        let current_logical_y = self.cursor.y;
        let max_logical_y = if self.dec_modes.origin_mode { self.screen.scroll_bot() - self.screen.scroll_top() } else { self.screen.height.saturating_sub(1) };
        if current_logical_y == max_logical_y { // If at bottom of scroll region/screen
            self.screen.scroll_up_serial(1, self.default_fill_glyph());
            for y_dirty in self.screen.scroll_top()..=self.screen.scroll_bot() { self.mark_line_dirty(y_dirty); }
        } else if self.cursor.y < max_logical_y {
            self.cursor.y += 1;
        }
        self.last_char_was_wide = false;
        let original_abs_y = (if self.dec_modes.origin_mode { self.screen.scroll_top() + current_logical_y } else { current_logical_y }).min(self.screen.height.saturating_sub(1));
        let new_abs_y = (if self.dec_modes.origin_mode { self.screen.scroll_top() + self.cursor.y } else { self.cursor.y }).min(self.screen.height.saturating_sub(1));
        self.mark_line_dirty(original_abs_y); if original_abs_y != new_abs_y { self.mark_line_dirty(new_abs_y); }
    }
    fn reverse_index(&mut self) { // RI
        let current_logical_y = self.cursor.y;
        // If origin mode is on, logical y=0 is screen.scroll_top.
        // If origin mode is off, logical y=0 is screen top (0).
        if current_logical_y == 0 { // If at top of scroll region/screen
            self.screen.scroll_down_serial(1, self.default_fill_glyph());
            for y_dirty in self.screen.scroll_top()..=self.screen.scroll_bot() { self.mark_line_dirty(y_dirty); }
        } else {
            self.cursor.y -= 1;
        }
        self.last_char_was_wide = false;
        let original_abs_y = (if self.dec_modes.origin_mode { self.screen.scroll_top() + current_logical_y } else { current_logical_y }).min(self.screen.height.saturating_sub(1));
        let new_abs_y = (if self.dec_modes.origin_mode { self.screen.scroll_top() + self.cursor.y } else { self.cursor.y }).min(self.screen.height.saturating_sub(1));
        self.mark_line_dirty(original_abs_y); if original_abs_y != new_abs_y { self.mark_line_dirty(new_abs_y); }
    }
    fn save_cursor_dec(&mut self) { // DECSC
        self.saved_cursor_state = Some(self.cursor); // Save logical cursor
        self.saved_attributes_state = Some(self.current_attributes);
        trace!("DECSC: Saved logical cursor {:?} and attributes {:?}", self.cursor, self.current_attributes);
    }
    fn restore_cursor_dec(&mut self) { // DECRC
        if let Some(saved_cursor) = self.saved_cursor_state {
            self.cursor = saved_cursor; // Restore logical cursor
            // Clamp restored logical cursor to current screen/region bounds
            let max_logical_y = if self.dec_modes.origin_mode {
                self.screen.scroll_bot() - self.screen.scroll_top()
            } else {
                self.screen.height.saturating_sub(1)
            };
            self.cursor.x = min(self.cursor.x, self.screen.width.saturating_sub(1));
            self.cursor.y = min(self.cursor.y, max_logical_y);

            if let Some(saved_attrs) = self.saved_attributes_state {
                self.current_attributes = saved_attrs;
            }
            trace!("DECRC: Restored logical cursor {:?} and attributes {:?}", self.cursor, self.current_attributes);
        } else {
            // If no saved state, VT100 typically homes cursor (0,0 logical) with default attributes.
            self.cursor = Cursor { x: 0, y: 0, attributes: Attributes::default() };
            self.current_attributes = Attributes::default();
            warn!("DECRC: No cursor state saved, restored to default (0,0 logical).");
        }
        self.last_char_was_wide = false;
    }

    // --- CSI Command Handlers ---
    fn cursor_up(&mut self, n: usize) { self.cursor.y = self.cursor.y.saturating_sub(n); self.last_char_was_wide = false; }
    fn cursor_down(&mut self, n: usize) { let max_y = if self.dec_modes.origin_mode { self.screen.scroll_bot() - self.screen.scroll_top() } else { self.screen.height.saturating_sub(1) }; self.cursor.y = min(self.cursor.y.saturating_add(n), max_y); self.last_char_was_wide = false; }
    fn cursor_forward(&mut self, n: usize) { self.cursor.x = min(self.cursor.x.saturating_add(n), self.screen.width.saturating_sub(1)); self.last_char_was_wide = false; }
    fn cursor_backward(&mut self, n: usize) { self.cursor.x = self.cursor.x.saturating_sub(n); self.last_char_was_wide = false; }
    fn cursor_to_column(&mut self, col: usize) { self.cursor.x = min(col, self.screen.width.saturating_sub(1)); self.last_char_was_wide = false; } // CHA, HPA
    fn cursor_to_pos(&mut self, row_param: usize, col_param: usize) { // CUP, HVP
        // Parameters are 1-based, convert to 0-based logical coordinates.
        let target_col = min(col_param, self.screen.width.saturating_sub(1));
        let target_row_logical = if self.dec_modes.origin_mode {
            min(row_param, self.screen.scroll_bot() - self.screen.scroll_top())
        } else {
            min(row_param, self.screen.height.saturating_sub(1))
        };
        self.cursor.x = target_col;
        self.cursor.y = target_row_logical;
        self.last_char_was_wide = false;
    }
    fn erase_in_display(&mut self, mode: u16) { // ED
        let (cx_abs, cy_abs) = (self.screen.cursor.x, self.screen.cursor.y); // Use absolute screen cursor after sync
        match mode {
            ERASE_MODE_TO_END => { // 0: Erase from cursor to end of screen
                self.screen.clear_line_segment(cy_abs, cx_abs, self.screen.width); // Clear rest of current line
                for y in (cy_abs + 1)..self.screen.height { self.screen.clear_line_segment(y, 0, self.screen.width); }
            }
            ERASE_MODE_TO_START => { // 1: Erase from start of screen to cursor
                for y in 0..cy_abs { self.screen.clear_line_segment(y, 0, self.screen.width); }
                self.screen.clear_line_segment(cy_abs, 0, cx_abs + 1); // Clear start of current line up to cursor
            }
            ERASE_MODE_ALL => { // 2: Erase entire screen
                for y in 0..self.screen.height { self.screen.clear_line_segment(y, 0, self.screen.width); }
                // ED 2 often implies cursor moves to home (0,0 logical)
                // self.cursor.x = 0; self.cursor.y = 0;
            }
            ERASE_MODE_SCROLLBACK => { // 3: Erase scrollback buffer (xterm extension)
                self.screen.scrollback.clear();
                debug!("ED 3 (Erase Scrollback) requested.");
                return; // No screen lines dirtied by this.
            }
            _ => warn!("Unknown ED mode: {}", mode),
        }
        self.mark_all_lines_dirty(); // ED usually dirties a large portion or all of the screen.
    }
    fn erase_in_line(&mut self, mode: u16) { // EL
        let (cx_abs, cy_abs) = (self.screen.cursor.x, self.screen.cursor.y); // Use absolute screen cursor
        match mode {
            ERASE_MODE_TO_END => self.screen.clear_line_segment(cy_abs, cx_abs, self.screen.width), // 0: Erase from cursor to end of line
            ERASE_MODE_TO_START => self.screen.clear_line_segment(cy_abs, 0, cx_abs + 1),      // 1: Erase from start of line to cursor
            ERASE_MODE_ALL => self.screen.clear_line_segment(cy_abs, 0, self.screen.width),       // 2: Erase entire line
            _ => warn!("Unknown EL mode: {}", mode),
        }
        // Mark only the current line dirty. Screen's clear_line_segment should do this.
    }
    fn erase_chars(&mut self, n: usize) { // ECH
        let (cx_abs, cy_abs) = (self.screen.cursor.x, self.screen.cursor.y);
        let end_x = min(cx_abs + n, self.screen.width);
        self.screen.fill_region(cy_abs, cx_abs, end_x, self.default_fill_glyph());
        // Screen's fill_region should mark the line dirty.
    }

    fn insert_blank_chars(&mut self, n: usize) { // ICH
        let cy_abs = self.screen.cursor.y; // Use absolute screen cursor y
        self.screen.insert_blank_chars_in_line(cy_abs, self.cursor.x, n, self.default_fill_glyph());
        // Screen's method should mark line dirty.
    }
    fn delete_chars(&mut self, n: usize) { // DCH
        let cy_abs = self.screen.cursor.y;
        self.screen.delete_chars_in_line(cy_abs, self.cursor.x, n, self.default_fill_glyph());
    }
    fn insert_lines(&mut self, n: usize) { // IL
        let cy_abs = self.screen.cursor.y; // Absolute cursor Y within the screen
        // IL operates within scrolling margins if cursor is inside them.
        if cy_abs >= self.screen.scroll_top() && cy_abs <= self.screen.scroll_bot() {
            // Temporarily adjust scroll region for the operation as per VT* behavior for IL
            let original_scroll_top = self.screen.scroll_top();
            let original_scroll_bottom = self.screen.scroll_bot();
            self.screen.set_scrolling_region(cy_abs + 1, original_scroll_bottom + 1); // 1-based for set_scrolling_region
            self.screen.scroll_down_serial(n, self.default_fill_glyph());
            self.screen.set_scrolling_region(original_scroll_top + 1, original_scroll_bottom + 1); // Restore
            // All lines in the original scroll region could be affected.
            for y_dirty in original_scroll_top..=original_scroll_bottom { self.mark_line_dirty(y_dirty); }
        }
    }
    fn delete_lines(&mut self, n: usize) { // DL
        let cy_abs = self.screen.cursor.y;
        if cy_abs >= self.screen.scroll_top() && cy_abs <= self.screen.scroll_bot() {
            let original_scroll_top = self.screen.scroll_top();
            let original_scroll_bottom = self.screen.scroll_bot();
            self.screen.set_scrolling_region(cy_abs + 1, original_scroll_bottom + 1);
            self.screen.scroll_up_serial(n, self.default_fill_glyph());
            self.screen.set_scrolling_region(original_scroll_top + 1, original_scroll_bottom + 1);
            for y_dirty in original_scroll_top..=original_scroll_bottom { self.mark_line_dirty(y_dirty); }
        }
    }
    fn scroll_up(&mut self, n: usize) { // SU
        // SU scrolls the current scrolling region up. New lines at bottom.
        self.screen.scroll_up_serial(n, self.default_fill_glyph());
        for y_dirty in self.screen.scroll_top()..=self.screen.scroll_bot() { self.mark_line_dirty(y_dirty); }
    }
    fn scroll_down(&mut self, n: usize) { // SD
        // SD scrolls the current scrolling region down. New lines at top.
        self.screen.scroll_down_serial(n, self.default_fill_glyph());
        for y_dirty in self.screen.scroll_top()..=self.screen.scroll_bot() { self.mark_line_dirty(y_dirty); }
    }


    /// Handles SGR (Select Graphic Rendition) attributes.
    fn handle_sgr_attributes(&mut self, attributes_vec: Vec<Attribute>) {
        for attr_cmd in attributes_vec {
            match attr_cmd {
                Attribute::Reset => self.current_attributes = Attributes::default(),
                Attribute::Bold => self.current_attributes.flags.insert(AttrFlags::BOLD),
                Attribute::Faint => self.current_attributes.flags.insert(AttrFlags::FAINT), // Often same as not bold
                Attribute::Italic => self.current_attributes.flags.insert(AttrFlags::ITALIC),
                Attribute::Underline => self.current_attributes.flags.insert(AttrFlags::UNDERLINE),
                Attribute::BlinkSlow | Attribute::BlinkRapid => self.current_attributes.flags.insert(AttrFlags::BLINK),
                Attribute::Reverse => self.current_attributes.flags.insert(AttrFlags::REVERSE),
                Attribute::Conceal => self.current_attributes.flags.insert(AttrFlags::HIDDEN),
                Attribute::Strikethrough => self.current_attributes.flags.insert(AttrFlags::STRIKETHROUGH),
                Attribute::Foreground(color) => self.current_attributes.fg = map_ansi_color_to_glyph_color(color),
                Attribute::Background(color) => self.current_attributes.bg = map_ansi_color_to_glyph_color(color),
                Attribute::NoBold => { self.current_attributes.flags.remove(AttrFlags::BOLD); self.current_attributes.flags.remove(AttrFlags::FAINT); }
                Attribute::NoItalic => self.current_attributes.flags.remove(AttrFlags::ITALIC),
                Attribute::NoUnderline => self.current_attributes.flags.remove(AttrFlags::UNDERLINE),
                Attribute::NoBlink => self.current_attributes.flags.remove(AttrFlags::BLINK),
                Attribute::NoReverse => self.current_attributes.flags.remove(AttrFlags::REVERSE),
                Attribute::NoConceal => self.current_attributes.flags.remove(AttrFlags::HIDDEN),
                Attribute::NoStrikethrough => self.current_attributes.flags.remove(AttrFlags::STRIKETHROUGH),
                // TODO: Support UnderlineColor, Overlined, UnderlineDouble properly if needed
                Attribute::UnderlineColor(color) => { warn!("SGR UnderlineColor not yet fully supported: {:?}", color); /* Potentially store in Attributes */ },
                Attribute::Overlined => { warn!("SGR Overlined not yet supported."); /* Potentially add to AttrFlags */ },
                Attribute::UnderlineDouble => {self.current_attributes.flags.insert(AttrFlags::UNDERLINE); warn!("SGR Double Underline treated as single underline.");} // TODO: Distinct double underline
            }
        }
    }

    /// Handles SM (Set Mode) and RM (Reset Mode) sequences.
    fn handle_set_mode(&mut self, mode_type: Mode, enable: bool) -> Option<EmulatorAction> {
        let mut action = None;
        match mode_type {
            Mode::DecPrivate(mode_num) => {
                trace!("Setting DEC Private Mode {} to {}", mode_num, enable);
                match mode_num {
                    DECCKM_MODE => self.dec_modes.cursor_keys_app_mode = enable,
                    DECOM_MODE => { // Origin Mode
                        self.dec_modes.origin_mode = enable;
                        self.screen.origin_mode = enable; // Inform screen component
                        // When DECOM is set/reset, cursor moves to (0,0) logical within the new context.
                        self.cursor.x = 0;
                        self.cursor.y = 0; // Logical (0,0)
                        // Screen cursor will be updated by sync_emulator_state_to_screen_for_processing
                        // to be (0, scroll_top) if origin_mode is true, or (0,0) absolute if false.
                    }
                    DECTCEM_MODE => { // Cursor visibility
                        self.dec_modes.cursor_visible = enable;
                        action = Some(EmulatorAction::SetCursorVisibility(enable));
                    }
                    ALT_SCREEN_BUF_1047_MODE => { // Use Alt Screen, clear on switch (like 1049 but no cursor save/restore)
                        if enable {
                            if !self.dec_modes.using_alt_screen {
                                self.screen.enter_alt_screen(true); // Clear alt screen
                                self.dec_modes.using_alt_screen = true;
                                action = Some(EmulatorAction::RequestRedraw);
                            }
                        } else {
                            if self.dec_modes.using_alt_screen {
                                self.screen.exit_alt_screen();
                                self.dec_modes.using_alt_screen = false;
                                action = Some(EmulatorAction::RequestRedraw);
                            }
                        }
                    }
                    ALT_SCREEN_SAVE_RESTORE_1049_MODE => { // Use Alt Screen, save/restore cursor, clear on switch
                        if enable {
                            if !self.dec_modes.using_alt_screen {
                                self.save_cursor_dec(); // Save cursor before switching
                                self.screen.enter_alt_screen(true); // Clear alt screen
                                self.dec_modes.using_alt_screen = true;
                                action = Some(EmulatorAction::RequestRedraw);
                            }
                        } else {
                            if self.dec_modes.using_alt_screen {
                                self.screen.exit_alt_screen();
                                self.restore_cursor_dec(); // Restore cursor after switching back
                                self.dec_modes.using_alt_screen = false;
                                action = Some(EmulatorAction::RequestRedraw);
                            }
                        }
                    }
                    CURSOR_SAVE_RESTORE_1048_MODE => { // DECSCURS/DECRCURS (used with 1049)
                        if enable { self.save_cursor_dec(); } else { self.restore_cursor_dec(); }
                    }
                    // TODO: Add other DEC private modes (e.g., mouse reporting, DECAWM)
                    _ => warn!("Unknown DEC private mode {} to set/reset: {}", mode_num, enable),
                }
            }
            Mode::Standard(mode_num) => {
                // Standard modes (e.g., IRM, LNM) are less common for terminal setup via CSI h/l
                // but can be implemented here if needed.
                warn!("Standard mode set/reset not fully implemented yet: {} (enable: {})", mode_num, enable);
            }
        }
        action
    }

    /// Handles OSC (Operating System Command) sequences.
    fn handle_osc(&mut self, data: Vec<u8>) -> Option<EmulatorAction> {
        let osc_str = String::from_utf8_lossy(&data);
        let parts: Vec<&str> = osc_str.splitn(2, ';').collect();

        if parts.len() == 2 {
            let ps = parts[0].parse::<u32>().unwrap_or(u32::MAX); // Parameter selector
            let pt = parts[1].to_string(); // Text parameter

            match ps {
                0 | 2 => { // Set window title (OSC 0 for icon & window, OSC 2 for window only)
                    return Some(EmulatorAction::SetTitle(pt));
                }
                // TODO: Handle other OSC commands (e.g., color setting OSC 4, 10, 11, 12)
                _ => debug!("Unhandled OSC command: Ps={}, Pt='{}'", ps, pt),
            }
        } else {
            warn!("Malformed OSC sequence: {}", osc_str);
        }
        None
    }
}

/// Maps an `AnsiColor` (from parser) to a `GlyphColor` (for internal representation).
fn map_ansi_color_to_glyph_color(ansi_color: AnsiColor) -> GlyphColor {
    match ansi_color {
        AnsiColor::Default => GlyphColor::Default,
        AnsiColor::Black => GlyphColor::Named(NamedColor::Black),
        AnsiColor::Red => GlyphColor::Named(NamedColor::Red),
        AnsiColor::Green => GlyphColor::Named(NamedColor::Green),
        AnsiColor::Yellow => GlyphColor::Named(NamedColor::Yellow),
        AnsiColor::Blue => GlyphColor::Named(NamedColor::Blue),
        AnsiColor::Magenta => GlyphColor::Named(NamedColor::Magenta),
        AnsiColor::Cyan => GlyphColor::Named(NamedColor::Cyan),
        AnsiColor::White => GlyphColor::Named(NamedColor::White),
        AnsiColor::BrightBlack => GlyphColor::Named(NamedColor::BrightBlack),
        AnsiColor::BrightRed => GlyphColor::Named(NamedColor::BrightRed),
        AnsiColor::BrightGreen => GlyphColor::Named(NamedColor::BrightGreen),
        AnsiColor::BrightYellow => GlyphColor::Named(NamedColor::BrightYellow),
        AnsiColor::BrightBlue => GlyphColor::Named(NamedColor::BrightBlue),
        AnsiColor::BrightMagenta => GlyphColor::Named(NamedColor::BrightMagenta),
        AnsiColor::BrightCyan => GlyphColor::Named(NamedColor::BrightCyan),
        AnsiColor::BrightWhite => GlyphColor::Named(NamedColor::BrightWhite),
        AnsiColor::Indexed(idx) => {
            // NamedColor enum covers 0-15. For 16-255, use GlyphColor::Indexed.
            if idx < 16 {
                GlyphColor::Named(NamedColor::from_index(idx))
            } else {
                GlyphColor::Indexed(idx)
            }
        }
        AnsiColor::Rgb(r, g, b) => GlyphColor::Rgb(r, g, b),
    }
}

/// Maps ASCII characters to DEC Line Drawing characters.
fn map_to_dec_line_drawing(ch: char) -> char {
    // This mapping is based on common VT100/VT220 line drawing sets.
    // Refer to VT100 manual or ECMA-48 for precise mappings if needed.
    match ch {
        // Standard box drawing characters
        '`' => '◆', // Diamond (often used as solid block)
        'a' => '▒', // Checkerboard/shaded block
        'b' => '\u{2409}', // HT symbol
        'c' => '\u{240C}', // FF symbol
        'd' => '\u{240D}', // CR symbol
        'e' => '\u{240A}', // LF symbol
        'f' => '°', // Degree sign
        'g' => '±', // Plus/minus
        'h' => '\u{2424}', // NL symbol (sometimes HT)
        'i' => '\u{240B}', // VT symbol
        'j' => '┘', // Box drawings light up and left
        'k' => '┐', // Box drawings light down and left
        'l' => '┌', // Box drawings light down and right
        'm' => '└', // Box drawings light up and right
        'n' => '┼', // Box drawings light vertical and horizontal
        'o' => '─', // Box drawings light horizontal (scan line 1)
        'p' => '─', // Scan line 3 (often same as 'o')
        'q' => '─', // Scan line 5 (often same as 'o')
        'r' => '─', // Scan line 7 (often same as 'o')
        's' => '─', // Scan line 9 (often same as 'o')
        't' => '├', // Box drawings light vertical and right
        'u' => '┤', // Box drawings light vertical and left
        'v' => '┴', // Box drawings light up and horizontal
        'w' => '┬', // Box drawings light down and horizontal
        'x' => '│', // Box drawings light vertical
        'y' => '≤', // Less-than-or-equal-to
        'z' => '≥', // Greater-than-or-equal-to
        '{' => 'π', // Greek pi
        '|' => '≠', // Not equal to
        '}' => '£', // Pound sterling
        '~' => '·', // Bullet / Interpunct (sometimes right arrow or DEL symbol)
        _ => ch,    // If not in mapping, return original character
    }
}

#[cfg(test)]
mod tests;
