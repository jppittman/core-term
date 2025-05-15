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
    term::screen::{Screen, Cursor},
};

// Logging
use log::{debug, trace, warn};

/// Default tab interval.
pub const DEFAULT_TAB_INTERVAL: u8 = 8;

// --- Locally Defined Constants for Modes ---
const ERASE_MODE_TO_END: u16 = 0;
const ERASE_MODE_TO_START: u16 = 1;
const ERASE_MODE_ALL: u16 = 2;
const ERASE_MODE_SCROLLBACK: u16 = 3;

const DECCKM_MODE: u16 = 1;
const DECOM_MODE: u16 = 6;
const DECTCEM_MODE: u16 = 25;
const ALT_SCREEN_BUF_1047_MODE: u16 = 1047;
const CURSOR_SAVE_RESTORE_1048_MODE: u16 = 1048;
const ALT_SCREEN_SAVE_RESTORE_1049_MODE: u16 = 1049;


/// Actions that the terminal emulator signals to the orchestrator.
#[derive(Debug, Clone, PartialEq)]
pub enum EmulatorAction {
    WritePty(Vec<u8>),
    SetTitle(String),
    RingBell,
    RequestRedraw,
    SetCursorVisibility(bool),
}

/// Inputs that the terminal emulator processes.
#[derive(Debug, Clone)]
pub enum EmulatorInput {
    Ansi(AnsiCommand),
    User(BackendEvent),
    RawChar(char),
}

/// Represents the terminal's private modes (DECSET/DECRST).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DecPrivateModes {
    pub origin_mode: bool,
    pub cursor_visible: bool,
    pub cursor_keys_app_mode: bool,
    pub using_alt_screen: bool,
}

/// Represents the character sets G0, G1, G2, G3.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CharacterSet {
    Ascii,
    UkNational,
    DecLineDrawing,
}

impl CharacterSet {
    fn from_char(ch: char) -> Self {
        match ch {
            'B' => CharacterSet::Ascii,
            'A' => CharacterSet::UkNational,
            '0' => CharacterSet::DecLineDrawing,
            _ => {
                warn!("Unsupported character set designator: {}", ch);
                CharacterSet::Ascii
            }
        }
    }
}

/// Represents the mode types for SM/RM sequences (Set Mode / Reset Mode).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    DecPrivate(u16),
    Standard(u16),
}


/// The core terminal emulator.
pub struct TerminalEmulator {
    screen: Screen,
    cursor: Cursor,
    current_attributes: Attributes,
    saved_cursor_state: Option<Cursor>,
    saved_attributes_state: Option<Attributes>,
    dec_modes: DecPrivateModes,
    active_charsets: [CharacterSet; 4],
    active_charset_g_level: usize,
    dirty_lines: Vec<usize>,
    last_char_was_wide: bool,
}

impl TerminalEmulator {
    pub fn new(width: usize, height: usize, scrollback_limit: usize) -> Self {
        let default_attributes = Attributes::default();
        let screen = Screen::new(width, height, scrollback_limit);
        let cursor = Cursor { x: 0, y: 0, attributes: default_attributes };

        TerminalEmulator {
            screen,
            cursor,
            current_attributes: default_attributes,
            saved_cursor_state: None,
            saved_attributes_state: None,
            dec_modes: DecPrivateModes {
                cursor_visible: true,
                ..Default::default()
            },
            active_charsets: [
                CharacterSet::Ascii, CharacterSet::Ascii,
                CharacterSet::Ascii, CharacterSet::Ascii,
            ],
            active_charset_g_level: 0,
            dirty_lines: (0..height).collect(),
            last_char_was_wide: false,
        }
    }

    pub fn interpret_input(&mut self, input: EmulatorInput) -> Option<EmulatorAction> {
        self.sync_emulator_state_to_screen_for_processing();
        let mut action = match input {
            EmulatorInput::Ansi(command) => self.handle_ansi_command(command),
            EmulatorInput::User(event) => self.handle_backend_event(event),
            EmulatorInput::RawChar(ch) => {
                self.print_char(ch);
                None
            }
        };
        self.sync_screen_state_to_emulator_after_processing();
        if action.is_none() && !self.dirty_lines.is_empty() {
            action = Some(EmulatorAction::RequestRedraw);
        }
        action
    }

    fn handle_ansi_command(&mut self, command: AnsiCommand) -> Option<EmulatorAction> {
        // ... (rest of the function remains the same as in term_mod_rs_fix_1, no changes needed here for wcwidth) ...
        // For brevity, I'm omitting the identical parts of this function.
        // The important change is in `print_char` below.
        match command {
            AnsiCommand::C0Control(c0) => match c0 {
                C0Control::BS => { self.backspace(); None }
                C0Control::HT => { self.horizontal_tab(); None }
                C0Control::LF | C0Control::VT | C0Control::FF => { self.newline(true); None }
                C0Control::CR => { self.carriage_return(); None }
                C0Control::SO => { self.set_g_level(1); None }
                C0Control::SI => { self.set_g_level(0); None }
                C0Control::BEL => Some(EmulatorAction::RingBell),
                _ => { debug!("Unhandled C0 control: {:?}", c0); None }
            },
            AnsiCommand::Esc(esc) => match esc {
                EscCommand::SetTabStop => { self.screen.set_tabstop(self.cursor.x); None }
                EscCommand::Index => { self.index(); None }
                EscCommand::NextLine => { self.newline(true); self.carriage_return(); None }
                EscCommand::ReverseIndex => { self.reverse_index(); None }
                EscCommand::SaveCursor => { self.save_cursor_dec(); None }
                EscCommand::RestoreCursor => { self.restore_cursor_dec(); None }
                EscCommand::SelectCharacterSet(intermediate_char, final_char) => {
                    let g_idx = match intermediate_char {
                        '(' => 0, ')' => 1, '*' => 2, '+' => 3,
                        _ => { warn!("Unsupported G-set designator intermediate: {}", intermediate_char); 0 }
                    };
                    self.designate_character_set(g_idx, CharacterSet::from_char(final_char));
                    None
                }
                _ => { debug!("Unhandled Esc command: {:?}", esc); None }
            },
            AnsiCommand::Csi(csi) => match csi {
                CsiCommand::CursorUp(n) => { self.cursor_up(n.max(1) as usize); None }
                CsiCommand::CursorDown(n) => { self.cursor_down(n.max(1) as usize); None }
                CsiCommand::CursorForward(n) => { self.cursor_forward(n.max(1) as usize); None }
                CsiCommand::CursorBackward(n) => { self.cursor_backward(n.max(1) as usize); None }
                CsiCommand::CursorNextLine(n) => { self.cursor_down(n.max(1) as usize); self.carriage_return(); None }
                CsiCommand::CursorPrevLine(n) => { self.cursor_up(n.max(1) as usize); self.carriage_return(); None }
                CsiCommand::CursorCharacterAbsolute(n) => { self.cursor_to_column(n.saturating_sub(1) as usize); None }
                CsiCommand::CursorPosition(r, c) => { self.cursor_to_pos(r.saturating_sub(1) as usize, c.saturating_sub(1) as usize); None }
                CsiCommand::EraseInDisplay(mode) => { self.erase_in_display(mode); None }
                CsiCommand::EraseInLine(mode) => { self.erase_in_line(mode); None }
                CsiCommand::InsertCharacter(n) => { self.insert_blank_chars(n.max(1) as usize); None }
                CsiCommand::DeleteCharacter(n) => { self.delete_chars(n.max(1) as usize); None }
                CsiCommand::InsertLine(n) => { self.insert_lines(n.max(1) as usize); None }
                CsiCommand::DeleteLine(n) => { self.delete_lines(n.max(1) as usize); None }
                CsiCommand::SetGraphicsRendition(attrs_vec) => { self.handle_sgr_attributes(attrs_vec); None }
                CsiCommand::SetMode(mode_num) => self.handle_set_mode(Mode::Standard(mode_num), true),
                CsiCommand::ResetMode(mode_num) => self.handle_set_mode(Mode::Standard(mode_num), false),
                CsiCommand::SetModePrivate(mode_num) => self.handle_set_mode(Mode::DecPrivate(mode_num), true),
                CsiCommand::ResetModePrivate(mode_num) => self.handle_set_mode(Mode::DecPrivate(mode_num), false),
                CsiCommand::DeviceStatusReport(dsr_param) => {
                    if dsr_param == 0 || dsr_param == 6 {
                        let (cur_x, cur_y_logical) = self.cursor_pos();
                        let cur_y_abs = if self.dec_modes.origin_mode { self.screen.scroll_top() + cur_y_logical } else { cur_y_logical };
                        let response = format!("\x1B[{};{}R", cur_y_abs + 1, cur_x + 1);
                        Some(EmulatorAction::WritePty(response.into_bytes()))
                    } else if dsr_param == 5 {
                        Some(EmulatorAction::WritePty(b"\x1B[0n".to_vec()))
                    } else { warn!("Unhandled DSR parameter: {}", dsr_param); None }
                }
                CsiCommand::EraseCharacter(n) => { self.erase_chars(n.max(1) as usize); None }
                CsiCommand::ScrollUp(n) => { self.scroll_up(n.max(1) as usize); None }
                CsiCommand::ScrollDown(n) => { self.scroll_down(n.max(1) as usize); None }
                CsiCommand::SaveCursor => { self.save_cursor_dec(); None}
                CsiCommand::RestoreCursor => { self.restore_cursor_dec(); None}
                _ => { debug!("Unhandled CSI command: {:?}", csi); None }
            },
            AnsiCommand::Osc(data) => self.handle_osc(data),
            AnsiCommand::Print(ch) => { self.print_char(ch); None }
            _ => { debug!("Unhandled ANSI command type: {:?}", command); None }
        }
    }


    fn handle_backend_event(&mut self, event: BackendEvent) -> Option<EmulatorAction> {
        // Define some common key codes if not importing from a specific library here
        // These are illustrative; actual values depend on what the backend sends.
        const KEY_RETURN: u32 = 0xFF0D;
        const KEY_BACKSPACE: u32 = 0xFF08;
        const KEY_TAB: u32 = 0xFF09;
        const KEY_ISO_LEFT_TAB: u32 = 0xFE20; // Often used for Shift+Tab
        const KEY_ESCAPE: u32 = 0xFF1B;
        const KEY_UP_ARROW: u32 = 0xFF52;
        const KEY_DOWN_ARROW: u32 = 0xFF54;
        const KEY_RIGHT_ARROW: u32 = 0xFF53;
        const KEY_LEFT_ARROW: u32 = 0xFF51;
        // Add more for KP_ keys, Home, End, etc. if needed.
        // Example: const KEY_KP_ENTER: u32 = 0xFF8D;

        match event {
            BackendEvent::Key { keysym, text } => {
                let mut bytes_to_send: Vec<u8> = Vec::new();
                // Check if text is preferred (printable char not shadowed by a special keysym action)
                if !text.is_empty() && !(
                    // List keysyms where we prefer to send a specific sequence over the text.
                    // This list might need adjustment based on how backends report keys.
                    keysym == KEY_RETURN || // keysym == KEY_KP_ENTER ||
                    keysym == KEY_BACKSPACE ||
                    keysym == KEY_TAB || keysym == KEY_ISO_LEFT_TAB ||
                    keysym == KEY_ESCAPE ||
                    (keysym >= KEY_LEFT_ARROW && keysym <= KEY_UP_ARROW) // Basic arrows
                    // Add other function/navigation keysym ranges here
                ) {
                    bytes_to_send.extend(text.as_bytes());
                } else {
                    // Handle special keys by keysym
                    match keysym {
                        KEY_RETURN /* | KEY_KP_ENTER */ => bytes_to_send.push(b'\r'),
                        KEY_BACKSPACE => bytes_to_send.push(0x08),
                        KEY_TAB /* | KEY_KP_TAB */ => bytes_to_send.push(b'\t'),
                        KEY_ISO_LEFT_TAB => bytes_to_send.extend_from_slice(b"\x1b[Z"), // Shift Tab
                        KEY_ESCAPE => bytes_to_send.push(0x1B),

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
                        _ => {
                            if text.chars().count() == 1 { // Fallback for other keysyms if text is a single char
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
                debug!("BackendEvent::Resize received: {}px x {}px. Orchestrator should call term.resize() with char dimensions.", width_px, height_px);
            }
            BackendEvent::FocusGained => { debug!("Focus gained"); }
            BackendEvent::FocusLost => { debug!("Focus lost"); }
            BackendEvent::CloseRequested => {
                debug!("Backend signaled close/quit.");
            }
        }
        None
    }

    pub fn resize(&mut self, cols: usize, rows: usize) {
        let current_scrollback_limit = self.screen.scrollback_limit();
        self.screen.resize(cols, rows, current_scrollback_limit);
        let max_logical_y = if self.dec_modes.origin_mode {
            self.screen.scroll_bot() - self.screen.scroll_top()
        } else {
            self.screen.height.saturating_sub(1)
        };
        self.cursor.x = min(self.cursor.x, self.screen.width.saturating_sub(1));
        self.cursor.y = min(self.cursor.y, max_logical_y);
        self.mark_all_lines_dirty();
        debug!("Terminal resized to {}x{}. Logical cursor at ({}, {}).", cols, rows, self.cursor.x, self.cursor.y);
    }

    fn mark_all_lines_dirty(&mut self) {
        self.dirty_lines = (0..self.screen.height).collect();
        self.screen.mark_all_dirty();
    }

    fn mark_line_dirty(&mut self, y_abs: usize) {
        if y_abs < self.screen.height && !self.dirty_lines.contains(&y_abs) {
            self.dirty_lines.push(y_abs);
        }
        self.screen.mark_line_dirty(y_abs);
    }

    pub fn take_dirty_lines(&mut self) -> Vec<usize> {
        let mut all_dirty_indices: std::collections::HashSet<usize> = self.dirty_lines.drain(..).collect();
        for (idx, &is_dirty_flag) in self.screen.dirty.iter().enumerate() {
            if is_dirty_flag != 0 {
                all_dirty_indices.insert(idx);
            }
        }
        self.screen.clear_dirty_flags();
        all_dirty_indices.into_iter().collect()
    }

    pub fn get_glyph(&self, x_abs: usize, y_abs: usize) -> Glyph {
        self.screen.get_glyph(x_abs, y_abs)
    }

    pub fn dimensions(&self) -> (usize, usize) {
        (self.screen.width, self.screen.height)
    }

    pub fn cursor_pos(&self) -> (usize, usize) {
        (self.cursor.x, self.cursor.y)
    }

    pub fn is_alt_screen_active(&self) -> bool {
        self.screen.alt_screen_active
    }

    fn sync_emulator_state_to_screen_for_processing(&mut self) {
        self.screen.cursor.x = self.cursor.x;
        if self.dec_modes.origin_mode {
            let scroll_top = self.screen.scroll_top();
            let max_relative_y = self.screen.scroll_bot().saturating_sub(scroll_top);
            let clamped_relative_y = self.cursor.y.min(max_relative_y);
            self.screen.cursor.y = scroll_top + clamped_relative_y;
        } else {
            self.screen.cursor.y = self.cursor.y.min(self.screen.height.saturating_sub(1));
        }
        self.screen.cursor.y = self.screen.cursor.y.min(self.screen.height.saturating_sub(1));
        self.screen.cursor.attributes = self.current_attributes;
        self.screen.origin_mode = self.dec_modes.origin_mode;
    }

    fn sync_screen_state_to_emulator_after_processing(&mut self) {
        self.cursor.x = self.screen.cursor.x;
        if self.dec_modes.origin_mode {
            let scroll_top = self.screen.scroll_top();
            self.cursor.y = self.screen.cursor.y.saturating_sub(scroll_top);
        } else {
            self.cursor.y = self.screen.cursor.y;
        }
    }

    fn default_fill_glyph(&self) -> Glyph {
        Glyph { c: ' ', attr: self.current_attributes }
    }

    fn print_char(&mut self, ch: char) {
        let ch_to_print = self.map_char_to_active_charset(ch);
        // Use the new char_display_width function from the unicode module
        let char_width = get_char_display_width(ch_to_print);
        let current_screen_width = self.screen.width;

        if self.last_char_was_wide && self.cursor.x == 0 {
            // Already wrapped
        } else if self.cursor.x >= current_screen_width ||
                  (char_width == 2 && self.cursor.x >= current_screen_width.saturating_sub(1)) {
            let max_logical_y = if self.dec_modes.origin_mode {
                self.screen.scroll_bot() - self.screen.scroll_top()
            } else {
                self.screen.height.saturating_sub(1)
            };
            if self.cursor.y == max_logical_y {
                self.screen.scroll_up_serial(1, self.default_fill_glyph());
                for y_dirty in self.screen.scroll_top()..=self.screen.scroll_bot() { self.mark_line_dirty(y_dirty); }
            } else {
                self.cursor.y += 1;
            }
            self.cursor.x = 0;
        }

        let target_y_abs = if self.dec_modes.origin_mode {
            self.screen.scroll_top() + self.cursor.y
        } else {
            self.cursor.y
        }.min(self.screen.height.saturating_sub(1));

        if target_y_abs < self.screen.height {
            self.screen.set_glyph(self.cursor.x, target_y_abs, Glyph { c: ch_to_print, attr: self.current_attributes });
            if char_width == 2 && self.cursor.x + 1 < current_screen_width {
                 self.screen.set_glyph(self.cursor.x + 1, target_y_abs, Glyph { c: '\0', attr: self.current_attributes });
            }
        } else {
            warn!("print_char: Attempted to print at y_abs {} out of bounds (height {})", target_y_abs, self.screen.height);
        }

        self.last_char_was_wide = char_width == 2;
        self.cursor.x += char_width;
    }

    fn map_char_to_active_charset(&self, ch: char) -> char {
        let current_set = self.active_charsets[self.active_charset_g_level];
        match current_set {
            CharacterSet::Ascii => ch,
            CharacterSet::UkNational => if ch == '#' { '£' } else { ch },
            CharacterSet::DecLineDrawing => map_to_dec_line_drawing(ch),
        }
    }

    fn backspace(&mut self) { if self.cursor.x > 0 { self.cursor.x -= 1; } self.last_char_was_wide = false; }
    fn horizontal_tab(&mut self) {
        let next_stop = self.screen.get_next_tabstop(self.cursor.x).unwrap_or(self.screen.width.saturating_sub(1));
        self.cursor.x = min(next_stop, self.screen.width.saturating_sub(1)); self.last_char_was_wide = false;
    }
    fn newline(&mut self, is_line_feed: bool) {
        let current_logical_y = self.cursor.y;
        let max_logical_y = if self.dec_modes.origin_mode { self.screen.scroll_bot() - self.screen.scroll_top() } else { self.screen.height.saturating_sub(1) };
        if current_logical_y == max_logical_y {
            self.screen.scroll_up_serial(1, self.default_fill_glyph());
            for y_dirty in self.screen.scroll_top()..=self.screen.scroll_bot() { self.mark_line_dirty(y_dirty); }
        } else if self.cursor.y < max_logical_y { self.cursor.y += 1; }
        if is_line_feed { self.cursor.x = 0; } self.last_char_was_wide = false;
        let original_abs_y = (if self.dec_modes.origin_mode { self.screen.scroll_top() + current_logical_y } else { current_logical_y }).min(self.screen.height.saturating_sub(1));
        let new_abs_y = (if self.dec_modes.origin_mode { self.screen.scroll_top() + self.cursor.y } else { self.cursor.y }).min(self.screen.height.saturating_sub(1));
        self.mark_line_dirty(original_abs_y); if original_abs_y != new_abs_y { self.mark_line_dirty(new_abs_y); }
    }
    fn carriage_return(&mut self) { self.cursor.x = 0; self.last_char_was_wide = false; }
    fn set_g_level(&mut self, g_level: usize) { if g_level < self.active_charsets.len() { self.active_charset_g_level = g_level; } else { warn!("Attempted to set invalid G-level: {}", g_level); } }
    fn designate_character_set(&mut self, g_set_index: usize, charset: CharacterSet) {
        if g_set_index < self.active_charsets.len() { self.active_charsets[g_set_index] = charset; trace!("Designated G{} to {:?}", g_set_index, charset); }
        else { warn!("Invalid G-set index for designate charset: {}", g_set_index); }
    }
    fn index(&mut self) {
        let current_logical_y = self.cursor.y;
        let max_logical_y = if self.dec_modes.origin_mode { self.screen.scroll_bot() - self.screen.scroll_top() } else { self.screen.height.saturating_sub(1) };
        if current_logical_y == max_logical_y { self.screen.scroll_up_serial(1, self.default_fill_glyph()); for y_dirty in self.screen.scroll_top()..=self.screen.scroll_bot() { self.mark_line_dirty(y_dirty); } }
        else if self.cursor.y < max_logical_y { self.cursor.y += 1; } self.last_char_was_wide = false;
        let original_abs_y = (if self.dec_modes.origin_mode { self.screen.scroll_top() + current_logical_y } else { current_logical_y }).min(self.screen.height.saturating_sub(1));
        let new_abs_y = (if self.dec_modes.origin_mode { self.screen.scroll_top() + self.cursor.y } else { self.cursor.y }).min(self.screen.height.saturating_sub(1));
        self.mark_line_dirty(original_abs_y); if original_abs_y != new_abs_y { self.mark_line_dirty(new_abs_y); }
    }
    fn reverse_index(&mut self) {
        let current_logical_y = self.cursor.y;
        if current_logical_y == 0 { self.screen.scroll_down_serial(1, self.default_fill_glyph()); for y_dirty in self.screen.scroll_top()..=self.screen.scroll_bot() { self.mark_line_dirty(y_dirty); } }
        else { self.cursor.y -= 1; } self.last_char_was_wide = false;
        let original_abs_y = (if self.dec_modes.origin_mode { self.screen.scroll_top() + current_logical_y } else { current_logical_y }).min(self.screen.height.saturating_sub(1));
        let new_abs_y = (if self.dec_modes.origin_mode { self.screen.scroll_top() + self.cursor.y } else { self.cursor.y }).min(self.screen.height.saturating_sub(1));
        self.mark_line_dirty(original_abs_y); if original_abs_y != new_abs_y { self.mark_line_dirty(new_abs_y); }
    }
    fn save_cursor_dec(&mut self) { self.saved_cursor_state = Some(self.cursor); self.saved_attributes_state = Some(self.current_attributes); trace!("DECSC: Saved logical cursor {:?} and attributes {:?}", self.cursor, self.current_attributes); }
    fn restore_cursor_dec(&mut self) {
        if let Some(saved_cursor) = self.saved_cursor_state {
            self.cursor = saved_cursor;
            let max_logical_y = if self.dec_modes.origin_mode { self.screen.scroll_bot() - self.screen.scroll_top() } else { self.screen.height.saturating_sub(1) };
            self.cursor.x = min(self.cursor.x, self.screen.width.saturating_sub(1)); self.cursor.y = min(self.cursor.y, max_logical_y);
            if let Some(saved_attrs) = self.saved_attributes_state { self.current_attributes = saved_attrs; }
            trace!("DECRC: Restored logical cursor {:?} and attributes {:?}", self.cursor, self.current_attributes);
        } else {
            self.cursor = Cursor { x: 0, y: 0, attributes: Attributes::default() }; self.current_attributes = Attributes::default();
            warn!("DECRC: No cursor state saved, restored to default.");
        }
        self.last_char_was_wide = false;
    }
    fn cursor_up(&mut self, n: usize) { self.cursor.y = self.cursor.y.saturating_sub(n); self.last_char_was_wide = false; }
    fn cursor_down(&mut self, n: usize) { let max_y = if self.dec_modes.origin_mode { self.screen.scroll_bot() - self.screen.scroll_top() } else { self.screen.height.saturating_sub(1) }; self.cursor.y = min(self.cursor.y.saturating_add(n), max_y); self.last_char_was_wide = false; }
    fn cursor_forward(&mut self, n: usize) { self.cursor.x = min(self.cursor.x.saturating_add(n), self.screen.width.saturating_sub(1)); self.last_char_was_wide = false; }
    fn cursor_backward(&mut self, n: usize) { self.cursor.x = self.cursor.x.saturating_sub(n); self.last_char_was_wide = false; }
    fn cursor_to_column(&mut self, col: usize) { self.cursor.x = min(col, self.screen.width.saturating_sub(1)); self.last_char_was_wide = false; }
    fn cursor_to_pos(&mut self, row_param: usize, col_param: usize) {
        let target_col = min(col_param, self.screen.width.saturating_sub(1));
        let target_row_logical = if self.dec_modes.origin_mode { min(row_param, self.screen.scroll_bot() - self.screen.scroll_top()) } else { min(row_param, self.screen.height.saturating_sub(1)) };
        self.cursor.x = target_col; self.cursor.y = target_row_logical; self.last_char_was_wide = false;
    }
    fn erase_in_display(&mut self, mode: u16) {
        let (cx_abs, cy_abs) = (self.screen.cursor.x, self.screen.cursor.y);
        match mode {
            ERASE_MODE_TO_END => { self.screen.clear_line_segment(cy_abs, cx_abs, self.screen.width); for y in (cy_abs + 1)..self.screen.height { self.screen.clear_line_segment(y, 0, self.screen.width); } }
            ERASE_MODE_TO_START => { for y in 0..cy_abs { self.screen.clear_line_segment(y, 0, self.screen.width); } self.screen.clear_line_segment(cy_abs, 0, cx_abs + 1); }
            ERASE_MODE_ALL => { for y in 0..self.screen.height { self.screen.clear_line_segment(y, 0, self.screen.width); } }
            ERASE_MODE_SCROLLBACK => { self.screen.scrollback.clear(); debug!("ED 3 (Erase Scrollback) requested."); return; }
            _ => warn!("Unknown ED mode: {}", mode),
        }
        self.mark_all_lines_dirty();
    }
    fn erase_in_line(&mut self, mode: u16) {
        let (cx_abs, cy_abs) = (self.screen.cursor.x, self.screen.cursor.y);
        match mode {
            ERASE_MODE_TO_END => self.screen.clear_line_segment(cy_abs, cx_abs, self.screen.width),
            ERASE_MODE_TO_START => self.screen.clear_line_segment(cy_abs, 0, cx_abs + 1),
            ERASE_MODE_ALL => self.screen.clear_line_segment(cy_abs, 0, self.screen.width),
            _ => warn!("Unknown EL mode: {}", mode),
        }
    }
    fn erase_chars(&mut self, n: usize) { let (cx_abs, cy_abs) = (self.screen.cursor.x, self.screen.cursor.y); let end_x = min(cx_abs + n, self.screen.width); self.screen.fill_region(cy_abs, cx_abs, end_x, self.default_fill_glyph()); }
    fn insert_blank_chars(&mut self, n: usize) { let cy_abs = self.screen.cursor.y; self.screen.insert_blank_chars_in_line(cy_abs, self.cursor.x, n, self.default_fill_glyph()); }
    fn delete_chars(&mut self, n: usize) { let cy_abs = self.screen.cursor.y; self.screen.delete_chars_in_line(cy_abs, self.cursor.x, n, self.default_fill_glyph()); }
    fn insert_lines(&mut self, n: usize) {
        let cy_abs = self.screen.cursor.y;
        if cy_abs >= self.screen.scroll_top() && cy_abs <= self.screen.scroll_bot() {
            let original_scroll_top = self.screen.scroll_top(); let original_scroll_bottom = self.screen.scroll_bot();
            self.screen.set_scrolling_region(cy_abs + 1, original_scroll_bottom + 1);
            self.screen.scroll_down_serial(n, self.default_fill_glyph());
            self.screen.set_scrolling_region(original_scroll_top + 1, original_scroll_bottom + 1);
            self.mark_all_lines_dirty();
        }
    }
    fn delete_lines(&mut self, n: usize) {
        let cy_abs = self.screen.cursor.y;
        if cy_abs >= self.screen.scroll_top() && cy_abs <= self.screen.scroll_bot() {
            let original_scroll_top = self.screen.scroll_top(); let original_scroll_bottom = self.screen.scroll_bot();
            self.screen.set_scrolling_region(cy_abs + 1, original_scroll_bottom + 1);
            self.screen.scroll_up_serial(n, self.default_fill_glyph());
            self.screen.set_scrolling_region(original_scroll_top + 1, original_scroll_bottom + 1);
            self.mark_all_lines_dirty();
        }
    }
    fn scroll_up(&mut self, n: usize) { self.screen.scroll_up_serial(n, self.default_fill_glyph()); }
    fn scroll_down(&mut self, n: usize) { self.screen.scroll_down_serial(n, self.default_fill_glyph()); }

    fn handle_sgr_attributes(&mut self, attributes_vec: Vec<Attribute>) {
        for attr_cmd in attributes_vec {
            match attr_cmd {
                Attribute::Reset => self.current_attributes = Attributes::default(),
                Attribute::Bold => self.current_attributes.flags.insert(AttrFlags::BOLD),
                Attribute::Faint => self.current_attributes.flags.insert(AttrFlags::FAINT),
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
                Attribute::UnderlineColor(color) => { warn!("SGR UnderlineColor not yet fully supported: {:?}", color); },
                Attribute::Overlined => { warn!("SGR Overlined not yet supported."); },
                Attribute::UnderlineDouble => {self.current_attributes.flags.insert(AttrFlags::UNDERLINE); warn!("SGR Double Underline treated as single underline.");} // TODO: Distinct double underline
            }
        }
    }

    fn handle_set_mode(&mut self, mode_type: Mode, enable: bool) -> Option<EmulatorAction> {
        let mut action = None;
        match mode_type {
            Mode::DecPrivate(mode_num) => {
                match mode_num {
                    DECCKM_MODE => self.dec_modes.cursor_keys_app_mode = enable,
                    DECOM_MODE => {
                        self.dec_modes.origin_mode = enable;
                        self.screen.origin_mode = enable;
                        self.cursor.x = 0; self.cursor.y = 0;
                        self.sync_emulator_state_to_screen_for_processing();
                        self.sync_screen_state_to_emulator_after_processing();
                    }
                    DECTCEM_MODE => {
                        self.dec_modes.cursor_visible = enable;
                        action = Some(EmulatorAction::SetCursorVisibility(enable));
                    }
                    ALT_SCREEN_BUF_1047_MODE => {
                        if enable {
                            if !self.dec_modes.using_alt_screen {
                                self.screen.enter_alt_screen(true); self.dec_modes.using_alt_screen = true;
                                action = Some(EmulatorAction::RequestRedraw);
                            }
                        } else {
                            if self.dec_modes.using_alt_screen {
                                self.screen.exit_alt_screen(); self.dec_modes.using_alt_screen = false;
                                action = Some(EmulatorAction::RequestRedraw);
                            }
                        }
                    }
                    ALT_SCREEN_SAVE_RESTORE_1049_MODE => {
                        if enable {
                            if !self.dec_modes.using_alt_screen {
                                self.save_cursor_dec(); self.screen.enter_alt_screen(true);
                                self.dec_modes.using_alt_screen = true; action = Some(EmulatorAction::RequestRedraw);
                            }
                        } else {
                            if self.dec_modes.using_alt_screen {
                                self.screen.exit_alt_screen(); self.restore_cursor_dec();
                                self.dec_modes.using_alt_screen = false; action = Some(EmulatorAction::RequestRedraw);
                            }
                        }
                    }
                    CURSOR_SAVE_RESTORE_1048_MODE => {
                        if enable { self.save_cursor_dec(); } else { self.restore_cursor_dec(); }
                    }
                    _ => warn!("Unknown DEC private mode {} to set/reset: {}", mode_num, enable),
                }
            }
            Mode::Standard(mode_num) => {
                warn!("Standard mode set/reset not fully implemented yet: {} (enable: {})", mode_num, enable);
            }
        }
        action
    }

    fn handle_osc(&mut self, data: Vec<u8>) -> Option<EmulatorAction> {
        let osc_str = String::from_utf8_lossy(&data);
        let parts: Vec<&str> = osc_str.splitn(2, ';').collect();
        if parts.len() == 2 {
            let ps = parts[0].parse::<u32>().unwrap_or(u32::MAX);
            let pt = parts[1].to_string();
            match ps {
                0 | 2 => return Some(EmulatorAction::SetTitle(pt)),
                _ => debug!("Unhandled OSC command: Ps={}, Pt='{}'", ps, pt),
            }
        } else { warn!("Malformed OSC sequence: {}", osc_str); }
        None
    }
}

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
            if idx < 16 { GlyphColor::Named(NamedColor::from_index(idx)) }
            else { GlyphColor::Indexed(idx) }
        }
        AnsiColor::Rgb(r, g, b) => GlyphColor::Rgb(r, g, b),
    }
}

fn map_to_dec_line_drawing(ch: char) -> char {
    match ch {
        '`' => '◆', 'a' => '▒', 'b' => '\u{2409}', 'c' => '\u{240C}', 'd' => '\u{240D}',
        'e' => '\u{240A}', 'f' => '°', 'g' => '±', 'h' => '\u{2424}', 'i' => '\u{240B}',
        'j' => '┘', 'k' => '┐', 'l' => '┌', 'm' => '└', 'n' => '┼',
        'o' => '─', 'p' => '─', 'q' => '─', 'r' => '─', 's' => '─',
        't' => '├', 'u' => '┤', 'v' => '┴', 'w' => '┬', 'x' => '│',
        'y' => '≤', 'z' => '≥', '{' => 'π', '|' => '≠', '}' => '£', '~' => '·',
        _ => ch,
    }
}

