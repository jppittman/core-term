//! This module defines the core terminal emulator logic.
//! It acts as a state machine processing inputs and producing actions.

// Standard library imports
use std::cmp::min;

// Crate-level imports
use crate::{
    ansi::{ // Import constants directly from the ansi module root
        self,
        commands::{
            AnsiCommand, C0Control, CsiCommand, EscCommand, Attribute,
            Color as AnsiColor, // Alias Ansi's Color to avoid clash with glyph::Color
            Mode, CharacterSet, // These should be in ansi::commands
        },
        ERASE_MODE_TO_END, ERASE_MODE_TO_START, ERASE_MODE_ALL, ERASE_MODE_SCROLLBACK,
        DECTCEM_MODE, DECOM_MODE, DECCKM_MODE, ALT_SCREEN_BUF_1047_MODE,
        CURSOR_SAVE_RESTORE_1048_MODE, ALT_SCREEN_SAVE_RESTORE_1049_MODE,
    },
    backends::{self, BackendEvent, KeyEvent}, // Import KeyEvent
    glyph::{self, Glyph, Attributes, AttrFlags, Color as GlyphColor, NamedColor},
    // Use `super::screen` if screen.rs is in the same directory as this mod.rs
    // and this mod.rs is for the `term` module.
    // Or `crate::term::screen` if `term` is a top-level module and `screen` is a submodule.
    // Assuming `mod screen;` is declared in this file (or this is `term/lib.rs` and screen is `term/screen.rs`)
    screen::{Screen, Cursor},
};

// External Crate
// NOTE: Add `unicode-width = "0.1.x"` (e.g., "0.1.11") to your Cargo.toml dependencies
use unicode_width::UnicodeWidthChar;

// Logging
use log::{debug, trace, warn};

/// Default tab interval.
pub const DEFAULT_TAB_INTERVAL: u8 = 8;

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

/// The core terminal emulator.
pub struct TerminalEmulator {
    screen: Screen,
    cursor: Cursor, 
    current_attributes: Attributes,
    saved_cursor_state: Option<Cursor>, 
    saved_attributes_state: Option<Attributes>,
    dec_modes: DecPrivateModes,
    active_charsets: [CharacterSet; 2], 
    active_charset_g_level: usize, 
    dirty_lines: Vec<usize>, 
    last_char_was_wide: bool,
}

impl TerminalEmulator {
    pub fn new(width: usize, height: usize, scrollback_limit: usize) -> Self {
        let default_attributes = Attributes::default();
        // scrollback_limit is now correctly passed to Screen::new
        let screen = Screen::new(width, height, scrollback_limit);
        let cursor = Cursor { x: 0, y: 0, attributes: default_attributes };

        TerminalEmulator {
            screen,
            cursor,
            current_attributes: default_attributes,
            saved_cursor_state: None,
            saved_attributes_state: None,
            dec_modes: DecPrivateModes::default(),
            active_charsets: [CharacterSet::Ascii, CharacterSet::Ascii],
            active_charset_g_level: 0,
            dirty_lines: (0..height).collect(),
            last_char_was_wide: false,
        }
    }

    pub fn interpret_input(&mut self, input: EmulatorInput) -> Option<EmulatorAction> {
        let mut action = match input {
            EmulatorInput::Ansi(command) => self.handle_ansi_command(command),
            EmulatorInput::User(event) => self.handle_backend_event(event),
            EmulatorInput::RawChar(ch) => {
                self.print_char(ch);
                None 
            }
        };
        
        self.sync_screen_cursor();

        if action.is_none() && !self.dirty_lines.is_empty() {
            action = Some(EmulatorAction::RequestRedraw);
        }
        action
    }

    fn handle_ansi_command(&mut self, command: AnsiCommand) -> Option<EmulatorAction> {
        self.sync_cursor_to_screen_for_processing();

        let action = match command {
            AnsiCommand::C0Control(c0) => match c0 { // Corrected from AnsiCommand::Control
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
                EscCommand::NextLine => { self.newline(true); None }
                EscCommand::ReverseIndex => { self.reverse_index(); None }
                EscCommand::SaveCursor => { self.save_cursor_dec(); None }
                EscCommand::RestoreCursor => { self.restore_cursor_dec(); None }
                // Assuming EscCommand::CharSet is (designator_byte: u8, charset_char: char)
                // This needs to align with your actual EscCommand::CharSet variant definition
                EscCommand::DesignateCharset(g_set_byte_as_u8, charset_char_code) => {
                    let g_idx = match g_set_byte_as_u8 { // This was g_set_byte in previous version, ensure type
                        b'(' => 0, 
                        b')' => 1, 
                        b'*' => 2, 
                        b'+' => 3, 
                        _ => { warn!("Unsupported G-set designator byte: {}", g_set_byte_as_u8); 0 }
                    };
                    let charset = CharacterSet::from_char(charset_char_code);
                    self.designate_character_set(g_idx, charset);
                    None
                }
                _ => { debug!("Unhandled Esc command: {:?}", esc); None }
            },
            AnsiCommand::Csi(csi) => match csi {
                // Ensure these variant names match your ansi::commands::CsiCommand
                CsiCommand::CursorUp(n) => { self.cursor_up(n.max(1) as usize); None }
                CsiCommand::CursorDown(n) => { self.cursor_down(n.max(1) as usize); None }
                CsiCommand::CursorForward(n) => { self.cursor_forward(n.max(1) as usize); None }
                CsiCommand::CursorBackward(n) => { self.cursor_backward(n.max(1) as usize); None }
                CsiCommand::CursorNextLine(n) => { self.cursor_down(n.max(1) as usize); self.carriage_return(); None }
                CsiCommand::CursorPrevLine(n) => { self.cursor_up(n.max(1) as usize); self.carriage_return(); None }
                CsiCommand::CursorCharacterAbsolute(n) => { self.cursor_to_column(n.saturating_sub(1) as usize); None } // Corrected
                CsiCommand::CursorPosition(r, c) => { self.cursor_to_pos(r.saturating_sub(1) as usize, c.saturating_sub(1) as usize); None }
                CsiCommand::EraseInDisplay(mode) => { self.erase_in_display(mode); None } // Corrected
                CsiCommand::EraseInLine(mode) => { self.erase_in_line(mode); None } // Corrected
                CsiCommand::InsertChars(n) => { self.insert_blank_chars(n.max(1) as usize); None } // Corrected
                CsiCommand::DeleteChars(n) => { self.delete_chars(n.max(1) as usize); None } // Corrected
                CsiCommand::InsertLine(n) => { self.insert_lines(n.max(1) as usize); None } // Corrected (was InsertLines)
                CsiCommand::DeleteLine(n) => { self.delete_lines(n.max(1) as usize); None } // Corrected (was DeleteLines)
                CsiCommand::SetGraphicsRendition(attrs) => { self.handle_sgr_attributes(attrs); None } // Corrected
                CsiCommand::SetMode(mode_cmd) => self.handle_set_mode(mode_cmd, true),
                CsiCommand::ResetMode(mode_cmd) => self.handle_set_mode(mode_cmd, false),
                CsiCommand::SetTopBottomMargin { top, bottom } => { // Assuming this is the variant for DECSTBM
                    self.screen.set_scrolling_region(top as usize, bottom as usize); None
                }
                CsiCommand::DeviceAttributes => { Some(EmulatorAction::WritePty(b"\x1B[?6c".to_vec())) }
                CsiCommand::EraseCharacter(n) => { self.erase_chars(n.max(1) as usize); None } // Corrected
                CsiCommand::ScrollUp(n) => { self.scroll_up(n.max(1) as usize); None }
                CsiCommand::ScrollDown(n) => { self.scroll_down(n.max(1) as usize); None }
                _ => { debug!("Unhandled CSI command: {:?}", csi); None }
            },
            AnsiCommand::Osc(data) => self.handle_osc(data),
            AnsiCommand::Print(ch) => { self.print_char(ch); None }
        };
        action
    }

    fn handle_backend_event(&mut self, event: BackendEvent) -> Option<EmulatorAction> {
        match event {
            // Assuming KeyEvent has `text: Option<String>` or similar, and `keysym` for non-text keys
            BackendEvent::Key(key_event) => {
                // This is highly dependent on KeyEvent's structure.
                // For now, let's assume key_event.text gives us a char if it's simple.
                if let Some(text) = &key_event.text { // Assuming text is Option<String>
                    if !text.is_empty() {
                        // This simplified logic only handles single char text.
                        // Proper handling needs to consider control keys, modifiers, etc.
                        let ch = text.chars().next().unwrap_or_default(); // Example
                        let mut bytes_to_send = vec![0; 4];
                        let len = ch.encode_utf8(&mut bytes_to_send).len();
                        bytes_to_send.truncate(len);
                        return Some(EmulatorAction::WritePty(bytes_to_send));
                    }
                }
                // TODO: Handle key_event.keysym for arrow keys, function keys, etc.
                // considering self.dec_modes.cursor_keys_app_mode
            }
            BackendEvent::Resize { cols, rows, .. } => {
                self.resize(cols as usize, rows as usize);
                return Some(EmulatorAction::RequestRedraw);
            }
            BackendEvent::FocusGained => { // Assuming FocusGained and FocusLost
                debug!("Focus gained");
            }
            BackendEvent::FocusLost => {
                debug!("Focus lost");
            }
            BackendEvent::Closed => {
                debug!("Backend signaled close/quit.");
            }
             _ => { debug!("Unhandled backend event: {:?}", event); }
        }
        None
    }

    pub fn resize(&mut self, cols: usize, rows: usize) {
        let current_scrollback_limit = self.screen.scrollback_limit();
        self.screen.resize(cols, rows, current_scrollback_limit);
        self.cursor.x = min(self.cursor.x, cols.saturating_sub(1));
        self.cursor.y = min(self.cursor.y, rows.saturating_sub(1));
        self.mark_all_lines_dirty();
    }

    fn mark_all_lines_dirty(&mut self) {
        self.dirty_lines = (0..self.screen.height).collect();
    }

    fn mark_line_dirty(&mut self, y: usize) {
        if y < self.screen.height && !self.dirty_lines.contains(&y) {
            self.dirty_lines.push(y);
        }
        self.screen.mark_line_dirty(y);
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

    pub fn get_glyph(&self, x: usize, y: usize) -> Glyph {
        self.screen.get_glyph(x, y)
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

    fn sync_cursor_to_screen_for_processing(&mut self) {
        self.screen.cursor.x = self.cursor.x;
        if self.dec_modes.origin_mode {
            let scroll_top = self.screen.scroll_top();
            let logical_y_clamped = self.cursor.y.max(scroll_top).min(self.screen.scroll_bot());
            self.screen.cursor.y = logical_y_clamped - scroll_top;
        } else {
            self.screen.cursor.y = self.cursor.y;
        }
        self.screen.cursor.attributes = self.current_attributes;
        self.screen.origin_mode = self.dec_modes.origin_mode;
    }

    fn sync_screen_cursor(&mut self) {
        self.cursor.x = self.screen.cursor.x;
        if self.dec_modes.origin_mode {
            self.cursor.y = self.screen.scroll_top() + self.screen.cursor.y;
        } else {
            self.cursor.y = self.screen.cursor.y;
        }
    }

    fn default_fill_glyph(&self) -> Glyph {
        Glyph { c: ' ', attr: self.current_attributes }
    }

    fn print_char(&mut self, ch: char) {
        let ch_to_print = self.map_char_to_active_charset(ch);
        let char_width = UnicodeWidthChar::width(ch_to_print).unwrap_or(1);

        if self.last_char_was_wide && self.cursor.x == self.screen.width.saturating_sub(1) {
             self.cursor.x = 0;
             self.newline(false); 
        } else if char_width == 2 && self.cursor.x >= self.screen.width.saturating_sub(1) {
            self.cursor.x = 0;
            self.newline(false);
        }
        
        let target_x = self.cursor.x;
        let target_y_abs = if self.dec_modes.origin_mode { self.screen.scroll_top() + self.cursor.y } else { self.cursor.y };

        self.screen.set_glyph(target_x, target_y_abs, Glyph { c: ch_to_print, attr: self.current_attributes });
        self.mark_line_dirty(target_y_abs);

        self.last_char_was_wide = char_width == 2;

        self.cursor.x += char_width;
        if self.cursor.x >= self.screen.width {
            self.cursor.x = 0;
            self.newline(false); 
        }
    }
    
    fn map_char_to_active_charset(&self, ch: char) -> char {
        let current_set = self.active_charsets[self.active_charset_g_level];
        match current_set {
            CharacterSet::Ascii => ch,
            CharacterSet::UkNational => if ch == '#' { '£' } else { ch },
            CharacterSet::DecLineDrawing => map_to_dec_line_drawing(ch),
            _ => ch,
        }
    }

    fn backspace(&mut self) {
        if self.cursor.x > 0 { self.cursor.x -= 1; }
        self.last_char_was_wide = false;
    }

    fn horizontal_tab(&mut self) {
        let next_stop = self.screen.get_next_tabstop(self.cursor.x)
            .unwrap_or(self.screen.width.saturating_sub(1));
        self.cursor.x = min(next_stop, self.screen.width.saturating_sub(1));
        self.last_char_was_wide = false;
    }

    fn newline(&mut self, is_line_feed: bool) {
        let effective_y_abs = if self.dec_modes.origin_mode { self.screen.scroll_top() + self.cursor.y } else { self.cursor.y };
        let scroll_bot_abs = self.screen.scroll_bot();

        if effective_y_abs == scroll_bot_abs {
            self.screen.scroll_up_serial(1, self.default_fill_glyph());
        } else {
            let max_logical_y = if self.dec_modes.origin_mode {
                self.screen.scroll_bot() - self.screen.scroll_top()
            } else {
                self.screen.height.saturating_sub(1)
            };
            if self.cursor.y < max_logical_y {
                self.cursor.y += 1;
            }
        }

        if is_line_feed { self.cursor.x = 0; }
        self.last_char_was_wide = false;
        self.mark_line_dirty(effective_y_abs);
        let new_effective_y_abs = if self.dec_modes.origin_mode { self.screen.scroll_top() + self.cursor.y } else { self.cursor.y };
        if effective_y_abs != new_effective_y_abs { self.mark_line_dirty(new_effective_y_abs); }
    }

    fn carriage_return(&mut self) {
        self.cursor.x = 0;
        self.last_char_was_wide = false;
    }
    
    fn set_g_level(&mut self, g_level: usize) {
        if g_level < self.active_charsets.len() {
            self.active_charset_g_level = g_level;
        } else { warn!("Attempted to set invalid G-level: {}", g_level); }
    }

    fn designate_character_set(&mut self, g_set_index: usize, charset: CharacterSet) {
        if g_set_index < self.active_charsets.len() {
            self.active_charsets[g_set_index] = charset;
            trace!("Designated G{} to {:?}", g_set_index, charset);
        } else { warn!("Invalid G-set index for designate charset: {}", g_set_index); }
    }

    fn index(&mut self) {
        let current_logical_y = self.cursor.y;
        let effective_y_abs = if self.dec_modes.origin_mode { self.screen.scroll_top() + current_logical_y } else { current_logical_y };

        if effective_y_abs == self.screen.scroll_bot() {
            self.screen.scroll_up_serial(1, self.default_fill_glyph());
        } else {
            let max_logical_y = if self.dec_modes.origin_mode { self.screen.scroll_bot() - self.screen.scroll_top() } else { self.screen.height.saturating_sub(1) };
            if self.cursor.y < max_logical_y { self.cursor.y += 1; }
        }
        self.last_char_was_wide = false;
        self.mark_line_dirty(effective_y_abs);
        let new_effective_y_abs = if self.dec_modes.origin_mode { self.screen.scroll_top() + self.cursor.y } else { self.cursor.y };
        if effective_y_abs != new_effective_y_abs { self.mark_line_dirty(new_effective_y_abs); }
    }

    fn reverse_index(&mut self) {
        let current_logical_y = self.cursor.y;
        let effective_y_abs = if self.dec_modes.origin_mode { self.screen.scroll_top() + current_logical_y } else { current_logical_y };

        if effective_y_abs == self.screen.scroll_top() {
            self.screen.scroll_down_serial(1, self.default_fill_glyph());
        } else if self.cursor.y > 0 {
            self.cursor.y -= 1;
        }
        self.last_char_was_wide = false;
        self.mark_line_dirty(effective_y_abs);
        let new_effective_y_abs = if self.dec_modes.origin_mode { self.screen.scroll_top() + self.cursor.y } else { self.cursor.y };
        if effective_y_abs != new_effective_y_abs { self.mark_line_dirty(new_effective_y_abs); }
    }

    fn save_cursor_dec(&mut self) {
        self.saved_cursor_state = Some(self.cursor);
        self.saved_attributes_state = Some(self.current_attributes);
        trace!("DECSC: Saved cursor {:?} and attributes {:?}", self.cursor, self.current_attributes);
    }

    fn restore_cursor_dec(&mut self) {
        if let Some(saved_cursor) = self.saved_cursor_state {
            self.cursor = saved_cursor;
            if let Some(saved_attrs) = self.saved_attributes_state {
                self.current_attributes = saved_attrs;
            }
            trace!("DECRC: Restored cursor {:?} and attributes {:?}", self.cursor, self.current_attributes);
        } else {
            self.cursor = Cursor { x: 0, y: 0, attributes: Attributes::default() };
            self.current_attributes = Attributes::default();
            warn!("DECRC: No cursor state saved, restored to default.");
        }
        self.last_char_was_wide = false;
    }

    fn cursor_up(&mut self, n: usize) {
        self.cursor.y = self.cursor.y.saturating_sub(n);
        // Clamping to 0 for relative y is fine. Absolute position is handled by effective_y.
        self.last_char_was_wide = false;
    }

    fn cursor_down(&mut self, n: usize) {
        let max_y_logical = if self.dec_modes.origin_mode {
            self.screen.scroll_bot() - self.screen.scroll_top()
        } else {
            self.screen.height.saturating_sub(1)
        };
        self.cursor.y = min(self.cursor.y.saturating_add(n), max_y_logical);
        self.last_char_was_wide = false;
    }

    fn cursor_forward(&mut self, n: usize) {
        self.cursor.x = min(self.cursor.x.saturating_add(n), self.screen.width.saturating_sub(1));
        self.last_char_was_wide = false;
    }

    fn cursor_backward(&mut self, n: usize) {
        self.cursor.x = self.cursor.x.saturating_sub(n);
        self.last_char_was_wide = false;
    }

    fn cursor_to_column(&mut self, col: usize) {
        self.cursor.x = min(col, self.screen.width.saturating_sub(1));
        self.last_char_was_wide = false;
    }

    fn cursor_to_pos(&mut self, row_abs_or_rel: usize, col: usize) {
        if self.dec_modes.origin_mode {
            // row_abs_or_rel is relative to margins (0-based from scroll_top)
            let max_rel_y = self.screen.scroll_bot() - self.screen.scroll_top();
            self.cursor.y = min(row_abs_or_rel, max_rel_y);
        } else {
            // row_abs_or_rel is absolute screen row (0-based)
            self.cursor.y = min(row_abs_or_rel, self.screen.height.saturating_sub(1));
        }
        self.cursor.x = min(col, self.screen.width.saturating_sub(1));
        self.last_char_was_wide = false;
    }

    fn erase_in_display(&mut self, mode: u16) {
        let fill_glyph = self.default_fill_glyph();
        let (cx_abs, cy_abs) = (self.cursor.x, if self.dec_modes.origin_mode { self.screen.scroll_top() + self.cursor.y } else { self.cursor.y });

        match mode {
            ERASE_MODE_TO_END => {
                self.screen.fill_region(cy_abs, cx_abs, self.screen.width, fill_glyph.clone());
                for y in (cy_abs + 1)..self.screen.height {
                    self.screen.fill_region(y, 0, self.screen.width, fill_glyph.clone());
                }
            }
            ERASE_MODE_TO_START => {
                for y in 0..cy_abs {
                    self.screen.fill_region(y, 0, self.screen.width, fill_glyph.clone());
                }
                self.screen.fill_region(cy_abs, 0, cx_abs + 1, fill_glyph.clone());
            }
            ERASE_MODE_ALL => {
                for y in 0..self.screen.height {
                    self.screen.fill_region(y, 0, self.screen.width, fill_glyph.clone());
                }
            }
            ERASE_MODE_SCROLLBACK => {
                self.screen.scrollback.clear();
                debug!("ED 3 (Erase Scrollback) requested.");
            }
            _ => warn!("Unknown ED mode: {}", mode),
        }
        self.mark_all_lines_dirty();
    }

    fn erase_in_line(&mut self, mode: u16) {
        let fill_glyph = self.default_fill_glyph();
        let (cx_abs, cy_abs) = (self.cursor.x, if self.dec_modes.origin_mode { self.screen.scroll_top() + self.cursor.y } else { self.cursor.y });

        match mode {
            ERASE_MODE_TO_END => self.screen.fill_region(cy_abs, cx_abs, self.screen.width, fill_glyph),
            ERASE_MODE_TO_START => self.screen.fill_region(cy_abs, 0, cx_abs + 1, fill_glyph),
            ERASE_MODE_ALL => self.screen.fill_region(cy_abs, 0, self.screen.width, fill_glyph),
            _ => warn!("Unknown EL mode: {}", mode),
        }
        self.mark_line_dirty(cy_abs);
    }
    
    fn erase_chars(&mut self, n: usize) {
        let fill_glyph = self.default_fill_glyph();
        let (cx_abs, cy_abs) = (self.cursor.x, if self.dec_modes.origin_mode { self.screen.scroll_top() + self.cursor.y } else { self.cursor.y });
        let end_x = min(cx_abs + n, self.screen.width);
        self.screen.fill_region(cy_abs, cx_abs, end_x, fill_glyph);
        self.mark_line_dirty(cy_abs);
    }

    fn insert_blank_chars(&mut self, n: usize) {
        let cy_abs = if self.dec_modes.origin_mode { self.screen.scroll_top() + self.cursor.y } else { self.cursor.y };
        self.screen.insert_blank_chars_in_line(cy_abs, self.cursor.x, n, self.default_fill_glyph());
        self.mark_line_dirty(cy_abs);
    }

    fn delete_chars(&mut self, n: usize) {
        let cy_abs = if self.dec_modes.origin_mode { self.screen.scroll_top() + self.cursor.y } else { self.cursor.y };
        self.screen.delete_chars_in_line(cy_abs, self.cursor.x, n, self.default_fill_glyph());
        self.mark_line_dirty(cy_abs);
    }

    fn insert_lines(&mut self, n: usize) {
        let cy_abs = if self.dec_modes.origin_mode { self.screen.scroll_top() + self.cursor.y } else { self.cursor.y };
        if cy_abs >= self.screen.scroll_top() && cy_abs <= self.screen.scroll_bot() {
            let original_scroll_top = self.screen.scroll_top();
            let original_scroll_bottom = self.screen.scroll_bot();
            self.screen.set_scrolling_region(cy_abs + 1, original_scroll_bottom + 1);
            self.screen.scroll_down_serial(n, self.default_fill_glyph());
            self.screen.set_scrolling_region(original_scroll_top + 1, original_scroll_bottom + 1);
            self.mark_all_lines_dirty();
        }
    }

    fn delete_lines(&mut self, n: usize) {
        let cy_abs = if self.dec_modes.origin_mode { self.screen.scroll_top() + self.cursor.y } else { self.cursor.y };
        if cy_abs >= self.screen.scroll_top() && cy_abs <= self.screen.scroll_bot() {
            let original_scroll_top = self.screen.scroll_top();
            let original_scroll_bottom = self.screen.scroll_bot();
            self.screen.set_scrolling_region(cy_abs + 1, original_scroll_bottom + 1);
            self.screen.scroll_up_serial(n, self.default_fill_glyph());
            self.screen.set_scrolling_region(original_scroll_top + 1, original_scroll_bottom + 1);
            self.mark_all_lines_dirty();
        }
    }
    
    fn scroll_up(&mut self, n: usize) {
        self.screen.scroll_up_serial(n, self.default_fill_glyph());
        self.mark_all_lines_dirty();
    }

    fn scroll_down(&mut self, n: usize) {
        self.screen.scroll_down_serial(n, self.default_fill_glyph());
        self.mark_all_lines_dirty();
    }

    fn handle_sgr_attributes(&mut self, attributes_vec: Vec<Attribute>) {
        for attr_cmd in attributes_vec {
            match attr_cmd {
                Attribute::Reset => self.current_attributes = Attributes::default(),
                Attribute::Bold => self.current_attributes.flags.insert(AttrFlags::BOLD),
                Attribute::Faint => self.current_attributes.flags.insert(AttrFlags::FAINT),
                Attribute::Italic => self.current_attributes.flags.insert(AttrFlags::ITALIC),
                Attribute::Underline => self.current_attributes.flags.insert(AttrFlags::UNDERLINE),
                Attribute::SlowBlink | Attribute::RapidBlink => self.current_attributes.flags.insert(AttrFlags::BLINK),
                Attribute::Reverse => self.current_attributes.flags.insert(AttrFlags::REVERSE), // Corrected
                Attribute::Hidden => self.current_attributes.flags.insert(AttrFlags::HIDDEN),  // Corrected
                Attribute::Strike => self.current_attributes.flags.insert(AttrFlags::STRIKETHROUGH), // Corrected
                Attribute::Font(_font_num) => { /* TODO: Font selection */ }
                Attribute::DefaultForeground => self.current_attributes.fg = GlyphColor::Default,
                Attribute::DefaultBackground => self.current_attributes.bg = GlyphColor::Default,
                Attribute::Foreground(color) => self.current_attributes.fg = map_ansi_color_to_glyph_color(color),
                Attribute::Background(color) => self.current_attributes.bg = map_ansi_color_to_glyph_color(color),
                Attribute::CancelBoldFaint => {
                    self.current_attributes.flags.remove(AttrFlags::BOLD);
                    self.current_attributes.flags.remove(AttrFlags::FAINT);
                }
                Attribute::CancelItalic => self.current_attributes.flags.remove(AttrFlags::ITALIC),
                Attribute::CancelUnderline => self.current_attributes.flags.remove(AttrFlags::UNDERLINE),
                Attribute::CancelBlink => self.current_attributes.flags.remove(AttrFlags::BLINK),
                Attribute::CancelReverse => self.current_attributes.flags.remove(AttrFlags::REVERSE),
                Attribute::CancelHidden => self.current_attributes.flags.remove(AttrFlags::HIDDEN),
                Attribute::CancelStrike => self.current_attributes.flags.remove(AttrFlags::STRIKETHROUGH),
            }
        }
    }
    
    fn handle_set_mode(&mut self, mode_cmd: Mode, enable: bool) -> Option<EmulatorAction> {
        match mode_cmd {
            Mode::DecPrivate(mode_num) => self.set_dec_private_mode(mode_num, enable),
            Mode::Standard(_mode_num) => {
                warn!("Standard mode set/reset not fully implemented yet: {:?}", mode_cmd);
                None
            }
        }
    }

    fn set_dec_private_mode(&mut self, mode: u16, enable: bool) -> Option<EmulatorAction> {
        let mut action = None;
        match mode {
            DECCKM_MODE => self.dec_modes.cursor_keys_app_mode = enable,
            DECOM_MODE => {
                self.dec_modes.origin_mode = enable;
                self.screen.origin_mode = enable;
                if enable { self.cursor.x = 0; self.cursor.y = 0; } 
                else { self.cursor.x = 0; self.cursor.y = 0; }
            }
            DECTCEM_MODE => {
                self.dec_modes.cursor_visible = enable;
                action = Some(EmulatorAction::SetCursorVisibility(enable));
            }
            ALT_SCREEN_BUF_1047_MODE => {
                if enable {
                    if !self.dec_modes.using_alt_screen {
                        self.screen.enter_alt_screen(true);
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
            ALT_SCREEN_SAVE_RESTORE_1049_MODE => {
                 if enable {
                    if !self.dec_modes.using_alt_screen {
                        self.save_cursor_dec(); 
                        self.screen.enter_alt_screen(true); 
                        self.dec_modes.using_alt_screen = true;
                        action = Some(EmulatorAction::RequestRedraw);
                    }
                } else {
                    if self.dec_modes.using_alt_screen {
                        self.screen.exit_alt_screen(); 
                        self.restore_cursor_dec(); 
                        self.dec_modes.using_alt_screen = false;
                        action = Some(EmulatorAction::RequestRedraw);
                    }
                }
            }
            CURSOR_SAVE_RESTORE_1048_MODE => {
                if enable { self.save_cursor_dec(); }
                else { self.restore_cursor_dec(); }
            }
            _ => warn!("Unknown DEC private mode {} to set/reset: {}", mode, enable),
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
            if idx < 16 {
                GlyphColor::Named(NamedColor::from_index(idx))
            } else {
                GlyphColor::Indexed(idx) // Use the new Indexed variant for 16-255
            }
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

// Placeholder - needs proper implementation based on backends::KeyEvent
fn key_event_to_char(_event: &KeyEvent) -> Option<char> {
    None
}

