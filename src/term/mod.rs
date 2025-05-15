// myterm/src/term/mod.rs

//! This module defines the core terminal emulator logic.
//! It acts as a state machine processing inputs and producing actions.

// Sub-modules
mod screen;
mod cursor; // New cursor module
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
    glyph::{Glyph, Attributes, AttrFlags, Color as GlyphColor, NamedColor, DEFAULT_GLYPH},
    term::screen::Screen,
    term::cursor::{CursorController, ScreenContext}, // Using new CursorController
};

// Logging
use log::{debug, trace, warn};

/// Default tab interval.
pub const DEFAULT_TAB_INTERVAL: u8 = 8; // Still used by screen.rs

// --- Enums for Modes ---

/// Defines the modes for erase operations (ED, EL).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u16)]
pub enum EraseMode {
    /// Erase from the current position to the end of the line/display. (Ps = 0)
    ToEnd = 0,
    /// Erase from the start of the line/display to the current position. (Ps = 1)
    ToStart = 1,
    /// Erase the entire line/display. (Ps = 2)
    All = 2,
    /// Erase scrollback buffer (xterm extension for ED). (Ps = 3)
    Scrollback = 3,
    /// Represents an unsupported or unknown erase mode.
    Unknown = u16::MAX, // Or some other sentinel value
}

impl From<u16> for EraseMode {
    fn from(value: u16) -> Self {
        match value {
            0 => EraseMode::ToEnd,
            1 => EraseMode::ToStart,
            2 => EraseMode::All,
            3 => EraseMode::Scrollback,
            _ => {
                warn!("Unknown erase mode value: {}", value);
                EraseMode::Unknown
            }
        }
    }
}

/// Defines specific DEC private mode numbers.
/// These are used in CSI ? Pm h/l sequences.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u16)]
pub enum DecModeConstant {
    /// DECCKM: Cursor Keys Mode (Application vs. Normal)
    CursorKeys = 1,
    /// DECOM: Origin Mode (Absolute vs. Relative to scroll region)
    Origin = 6,
    /// DECTCEM: Text Cursor Enable Mode (Show/Hide cursor)
    TextCursorEnable = 25,
    /// Use Alternate Screen Buffer, clears on switch (older, often superseded by 1049)
    AltScreenBufferClear = 1047,
    /// Save/Restore cursor (often used in conjunction with 1049)
    SaveRestoreCursor = 1048,
    /// Use Alternate Screen Buffer, save/restore cursor, clear on switch (most common)
    AltScreenBufferSaveRestore = 1049,
    // Add other DEC private mode constants here if they are specifically handled.
    // For unhandled ones, the u16 value will be used directly.
}

// Attempt to convert u16 to DecModeConstant, falling back if unknown.
// This is useful if you want to specifically handle known modes
// but still have the raw u16 for others.
impl DecModeConstant {
    fn from_u16(value: u16) -> Option<Self> {
        match value {
            1 => Some(DecModeConstant::CursorKeys),
            6 => Some(DecModeConstant::Origin),
            25 => Some(DecModeConstant::TextCursorEnable),
            1047 => Some(DecModeConstant::AltScreenBufferClear),
            1048 => Some(DecModeConstant::SaveRestoreCursor),
            1049 => Some(DecModeConstant::AltScreenBufferSaveRestore),
            _ => None,
        }
    }
}


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
    pub cursor_keys_app_mode: bool,
    pub using_alt_screen: bool,
    // cursor_visible is now managed by CursorController but influenced by DECTCEM
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

/// Represents the mode types for SM/RM sequences.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    DecPrivate(u16), // The u16 here is the raw mode number from the ANSI sequence
    Standard(u16),
}

/// The core terminal emulator.
pub struct TerminalEmulator {
    screen: Screen,
    cursor_controller: CursorController, // Manages logical cursor, attributes, visibility
    dec_modes: DecPrivateModes,
    active_charsets: [CharacterSet; 4],
    active_charset_g_level: usize,
    dirty_lines: Vec<usize>,
    last_char_was_wide: bool,
}

impl TerminalEmulator {
    /// Creates a new `TerminalEmulator`.
    pub fn new(width: usize, height: usize, scrollback_limit: usize) -> Self {
        let initial_attributes = DEFAULT_GLYPH.attr; // Use attributes from a default glyph
        let mut screen = Screen::new(width, height, scrollback_limit);
        // Ensure screen's default attributes match the initial cursor attributes
        screen.default_attributes = initial_attributes;

        TerminalEmulator {
            screen,
            cursor_controller: CursorController::new(initial_attributes),
            dec_modes: DecPrivateModes {
                // cursor_visible is true by default in CursorController
                ..Default::default()
            },
            active_charsets: [
                CharacterSet::Ascii, CharacterSet::Ascii,
                CharacterSet::Ascii, CharacterSet::Ascii,
            ],
            active_charset_g_level: 0,
            dirty_lines: (0..height.max(1)).collect(),
            last_char_was_wide: false,
        }
    }

    /// Helper to create the current `ScreenContext` for `CursorController`.
    fn current_screen_context(&self) -> ScreenContext {
        ScreenContext {
            width: self.screen.width,
            height: self.screen.height,
            scroll_top: self.screen.scroll_top(),
            scroll_bot: self.screen.scroll_bot(),
            origin_mode_active: self.dec_modes.origin_mode,
        }
    }

    /// Interprets an `EmulatorInput` and updates the terminal state.
    pub fn interpret_input(&mut self, input: EmulatorInput) -> Option<EmulatorAction> {
        let mut action = match input {
            EmulatorInput::Ansi(command) => self.handle_ansi_command(command),
            EmulatorInput::User(event) => self.handle_backend_event(event),
            EmulatorInput::RawChar(ch) => {
                self.print_char(ch);
                None
            }
        };

        if action.is_none() && !self.dirty_lines.is_empty() {
            action = Some(EmulatorAction::RequestRedraw);
        }
        action
    }

    /// Handles a parsed `AnsiCommand`.
    fn handle_ansi_command(&mut self, command: AnsiCommand) -> Option<EmulatorAction> {
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
                EscCommand::SetTabStop => {
                    let (cursor_x, _) = self.cursor_controller.logical_pos();
                    self.screen.set_tabstop(cursor_x);
                    None
                }
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
                CsiCommand::EraseInDisplay(mode_val) => { self.erase_in_display(EraseMode::from(mode_val)); None }
                CsiCommand::EraseInLine(mode_val) => { self.erase_in_line(EraseMode::from(mode_val)); None }
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
                        let screen_ctx = self.current_screen_context();
                        let (abs_x, abs_y) = self.cursor_controller.physical_screen_pos(&screen_ctx);
                        let response = format!("\x1B[{};{}R", abs_y + 1, abs_x + 1);
                        Some(EmulatorAction::WritePty(response.into_bytes()))
                    } else if dsr_param == 5 {
                        Some(EmulatorAction::WritePty(b"\x1B[0n".to_vec()))
                    } else {
                        warn!("Unhandled DSR parameter: {}", dsr_param); None
                    }
                }
                CsiCommand::EraseCharacter(n) => { self.erase_chars(n.max(1) as usize); None }
                CsiCommand::ScrollUp(n) => { self.scroll_up(n.max(1) as usize); None }
                CsiCommand::ScrollDown(n) => { self.scroll_down(n.max(1) as usize); None }
                CsiCommand::SaveCursor => { self.save_cursor_dec(); None}
                CsiCommand::RestoreCursor => { self.restore_cursor_dec(); None}
                CsiCommand::ClearTabStops(mode_val) => {
                    let (cursor_x, _) = self.cursor_controller.logical_pos();
                    self.screen.clear_tabstops(cursor_x, screen::TabClearMode::from(mode_val));
                    None
                }
                CsiCommand::SetScrollingRegion { top, bottom } => {
                    self.screen.set_scrolling_region(top as usize, bottom as usize);
                    self.cursor_controller.move_to_logical(0, 0, &self.current_screen_context());
                    None
                }
                _ => { debug!("Unhandled CSI command: {:?}", csi); None }
            },
            AnsiCommand::Osc(data) => self.handle_osc(data),
            AnsiCommand::Print(ch) => { self.print_char(ch); None }
            _ => { debug!("Unhandled ANSI command type: {:?}", command); None }
        }
    }

    /// Handles a `BackendEvent`.
    fn handle_backend_event(&mut self, event: BackendEvent) -> Option<EmulatorAction> {
        const KEY_RETURN: u32 = 0xFF0D;
        const KEY_BACKSPACE: u32 = 0xFF08;
        const KEY_TAB: u32 = 0xFF09;
        const KEY_ISO_LEFT_TAB: u32 = 0xFE20;
        const KEY_ESCAPE: u32 = 0xFF1B;
        const KEY_UP_ARROW: u32 = 0xFF52;
        const KEY_DOWN_ARROW: u32 = 0xFF54;
        const KEY_RIGHT_ARROW: u32 = 0xFF53;
        const KEY_LEFT_ARROW: u32 = 0xFF51;

        match event {
            BackendEvent::Key { keysym, text } => {
                let mut bytes_to_send: Vec<u8> = Vec::new();
                if !text.is_empty() && text.chars().all(|c| !c.is_control()) && !(
                    keysym == KEY_RETURN || keysym == KEY_BACKSPACE ||
                    keysym == KEY_TAB || keysym == KEY_ISO_LEFT_TAB ||
                    keysym == KEY_ESCAPE ||
                    (keysym >= KEY_LEFT_ARROW && keysym <= KEY_UP_ARROW)
                ) {
                    bytes_to_send.extend(text.as_bytes());
                } else {
                    match keysym {
                        KEY_RETURN => bytes_to_send.push(b'\r'),
                        KEY_BACKSPACE => bytes_to_send.push(0x08),
                        KEY_TAB => bytes_to_send.push(b'\t'),
                        KEY_ISO_LEFT_TAB => bytes_to_send.extend_from_slice(b"\x1b[Z"),
                        KEY_ESCAPE => bytes_to_send.push(0x1B),
                        KEY_UP_ARROW => bytes_to_send.extend_from_slice(if self.dec_modes.cursor_keys_app_mode { b"\x1bOA" } else { b"\x1b[A" }),
                        KEY_DOWN_ARROW => bytes_to_send.extend_from_slice(if self.dec_modes.cursor_keys_app_mode { b"\x1bOB" } else { b"\x1b[B" }),
                        KEY_RIGHT_ARROW => bytes_to_send.extend_from_slice(if self.dec_modes.cursor_keys_app_mode { b"\x1bOC" } else { b"\x1b[C" }),
                        KEY_LEFT_ARROW => bytes_to_send.extend_from_slice(if self.dec_modes.cursor_keys_app_mode { b"\x1bOD" } else { b"\x1b[D" }),
                        _ => {
                            if text.chars().count() == 1 && text.chars().next().map_or(false, |c| !c.is_control()) {
                                bytes_to_send.extend(text.as_bytes());
                            } else if !text.is_empty() && text.chars().all(|c| c.is_control() || c.is_ascii_alphanumeric() || c.is_ascii_punctuation()) {
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
            _ => debug!("BackendEvent {:?} passed to TerminalEmulator, typically handled by Orchestrator.", event),
        }
        None
    }

    /// Resizes the terminal display grid.
    pub fn resize(&mut self, cols: usize, rows: usize) {
        let current_scrollback_limit = self.screen.scrollback_limit();
        self.screen.resize(cols, rows, current_scrollback_limit);
        let (log_x, log_y) = self.cursor_controller.logical_pos();
        self.cursor_controller.move_to_logical(log_x, log_y, &self.current_screen_context());
        self.screen.default_attributes = self.cursor_controller.attributes();
        self.mark_all_lines_dirty();
        debug!("Terminal resized to {}x{}. Cursor re-clamped. All lines marked dirty.", cols, rows);
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
        let mut sorted_dirty_lines: Vec<usize> = all_dirty_indices.into_iter().collect();
        sorted_dirty_lines.sort_unstable();
        sorted_dirty_lines
    }

    pub fn get_glyph(&self, x_abs: usize, y_abs: usize) -> Glyph {
        self.screen.get_glyph(x_abs, y_abs)
    }

    pub fn dimensions(&self) -> (usize, usize) {
        (self.screen.width, self.screen.height)
    }

    pub fn cursor_pos(&self) -> (usize, usize) {
        self.cursor_controller.logical_pos()
    }

    pub fn get_screen_cursor_pos(&self) -> (usize, usize) {
        self.cursor_controller.physical_screen_pos(&self.current_screen_context())
    }

    pub fn is_cursor_visible(&self) -> bool {
        self.cursor_controller.is_visible()
    }

    pub fn is_alt_screen_active(&self) -> bool {
        self.screen.alt_screen_active
    }

    fn print_char(&mut self, ch: char) {
        let ch_to_print = self.map_char_to_active_charset(ch);
        let char_width = get_char_display_width(ch_to_print);
        let mut screen_ctx = self.current_screen_context();
        let (mut current_cursor_x, _) = self.cursor_controller.logical_pos();
        let did_wrap_due_to_wide_char_at_edge = self.last_char_was_wide && current_cursor_x == 0;

        if !did_wrap_due_to_wide_char_at_edge &&
           (current_cursor_x >= screen_ctx.width || (char_width == 2 && current_cursor_x >= screen_ctx.width.saturating_sub(1))) {
            let (prev_log_x, prev_log_y) = self.cursor_controller.logical_pos();
            let pre_newline_phys_y = self.cursor_controller.physical_screen_pos(&screen_ctx).1;
            self.newline(true);
            screen_ctx = self.current_screen_context();
            current_cursor_x = self.cursor_controller.logical_pos().0;
            self.mark_line_dirty(pre_newline_phys_y);
            let post_newline_phys_y = self.cursor_controller.physical_screen_pos(&screen_ctx).1;
            if pre_newline_phys_y != post_newline_phys_y {
                 self.mark_line_dirty(post_newline_phys_y);
            }
        }

        let (physical_x, physical_y) = self.cursor_controller.physical_screen_pos(&screen_ctx);
        let glyph_attrs = self.cursor_controller.attributes();

        if physical_y < self.screen.height {
            self.screen.set_glyph(physical_x, physical_y, Glyph { c: ch_to_print, attr: glyph_attrs });
            if char_width == 2 && physical_x + 1 < self.screen.width {
                self.screen.set_glyph(physical_x + 1, physical_y, Glyph { c: '\0', attr: glyph_attrs });
            }
        } else {
            warn!("print_char: Attempted to print at physical_y {} out of bounds (height {})", physical_y, self.screen.height);
        }
        self.last_char_was_wide = char_width == 2;
        self.cursor_controller.move_right(char_width, &screen_ctx);
    }

    fn map_char_to_active_charset(&self, ch: char) -> char {
        let current_set = self.active_charsets[self.active_charset_g_level];
        match current_set {
            CharacterSet::Ascii => ch,
            CharacterSet::UkNational => if ch == '#' { '£' } else { ch },
            CharacterSet::DecLineDrawing => map_to_dec_line_drawing(ch),
        }
    }

    fn backspace(&mut self) { 
        self.cursor_controller.move_left(1); self.last_char_was_wide = false; 
    }

    fn horizontal_tab(&mut self) {
        let (current_x, _) = self.cursor_controller.logical_pos();
        let screen_ctx = self.current_screen_context();
        let next_stop = self.screen.get_next_tabstop(current_x).unwrap_or(screen_ctx.width.saturating_sub(1));
        self.cursor_controller.move_to_logical_col(next_stop, &screen_ctx);
        self.last_char_was_wide = false;
    }

    fn newline(&mut self, is_line_feed: bool) {
        let screen_ctx = self.current_screen_context();
        let (_, current_logical_y) = self.cursor_controller.logical_pos();
        let (_, current_physical_y) = self.cursor_controller.physical_screen_pos(&screen_ctx);
        let max_logical_y = if screen_ctx.origin_mode_active {
            screen_ctx.scroll_bot.saturating_sub(screen_ctx.scroll_top)
        } else {
            screen_ctx.height.saturating_sub(1)
        };

        if current_logical_y == max_logical_y {
            self.screen.scroll_up_serial(1);
        } else if current_logical_y < max_logical_y {
            self.cursor_controller.move_down(1, &screen_ctx);
        }
        if is_line_feed { self.cursor_controller.carriage_return(); }
        self.last_char_was_wide = false;
        self.mark_line_dirty(current_physical_y);
        let (_, new_physical_y) = self.cursor_controller.physical_screen_pos(&self.current_screen_context());
        if current_physical_y != new_physical_y {
            self.mark_line_dirty(new_physical_y);
        }
    }
    fn carriage_return(&mut self) { self.cursor_controller.carriage_return(); self.last_char_was_wide = false; }

    fn set_g_level(&mut self, g_level: usize) {
        if g_level < self.active_charsets.len() {
            self.active_charset_g_level = g_level;
            trace!("Switched to G{} character set mapping.", g_level);
        } else { warn!("Attempted to set invalid G-level: {}", g_level); }
    }
    fn designate_character_set(&mut self, g_set_index: usize, charset: CharacterSet) {
        if g_set_index < self.active_charsets.len() {
            self.active_charsets[g_set_index] = charset;
            trace!("Designated G{} to {:?}", g_set_index, charset);
        } else { warn!("Invalid G-set index for designate charset: {}", g_set_index); }
    }

    fn index(&mut self) {
        let screen_ctx = self.current_screen_context();
        let (_, current_logical_y) = self.cursor_controller.logical_pos();
        let (_, current_physical_y) = self.cursor_controller.physical_screen_pos(&screen_ctx);
        let max_logical_y = if screen_ctx.origin_mode_active { screen_ctx.scroll_bot.saturating_sub(screen_ctx.scroll_top) } else { screen_ctx.height.saturating_sub(1) };

        if current_logical_y == max_logical_y {
            self.screen.scroll_up_serial(1);
        } else if current_logical_y < max_logical_y {
            self.cursor_controller.move_down(1, &screen_ctx);
        }
        self.last_char_was_wide = false;
        self.mark_line_dirty(current_physical_y);
        let (_, new_physical_y) = self.cursor_controller.physical_screen_pos(&self.current_screen_context());
        if current_physical_y != new_physical_y { self.mark_line_dirty(new_physical_y); }
    }
    fn reverse_index(&mut self) {
        let screen_ctx = self.current_screen_context();
        let (_, current_logical_y) = self.cursor_controller.logical_pos();
        let (_, current_physical_y) = self.cursor_controller.physical_screen_pos(&screen_ctx);

        if current_logical_y == 0 {
            self.screen.scroll_down_serial(1);
        } else {
            self.cursor_controller.move_up(1);
        }
        self.last_char_was_wide = false;
        self.mark_line_dirty(current_physical_y);
        let (_, new_physical_y) = self.cursor_controller.physical_screen_pos(&self.current_screen_context());
        if current_physical_y != new_physical_y { self.mark_line_dirty(new_physical_y); }
    }
    fn save_cursor_dec(&mut self) { self.cursor_controller.save_state(); }
    fn restore_cursor_dec(&mut self) {
        let default_attrs = Attributes::default();
        self.cursor_controller.restore_state(&self.current_screen_context(), default_attrs);
        self.screen.default_attributes = self.cursor_controller.attributes();
        self.last_char_was_wide = false;
    }

    fn cursor_up(&mut self, n: usize) { self.cursor_controller.move_up(n); self.last_char_was_wide = false; }
    fn cursor_down(&mut self, n: usize) { self.cursor_controller.move_down(n, &self.current_screen_context()); self.last_char_was_wide = false; }
    fn cursor_forward(&mut self, n: usize) { self.cursor_controller.move_right(n, &self.current_screen_context()); self.last_char_was_wide = false; }
    fn cursor_backward(&mut self, n: usize) { self.cursor_controller.move_left(n); self.last_char_was_wide = false; }
    fn cursor_to_column(&mut self, col: usize) { self.cursor_controller.move_to_logical_col(col, &self.current_screen_context()); self.last_char_was_wide = false; }
    fn cursor_to_pos(&mut self, row_param: usize, col_param: usize) {
        self.cursor_controller.move_to_logical(col_param, row_param, &self.current_screen_context());
        self.last_char_was_wide = false;
    }

    fn erase_in_display(&mut self, mode: EraseMode) {
        let screen_ctx = self.current_screen_context();
        let (cx_phys, cy_phys) = self.cursor_controller.physical_screen_pos(&screen_ctx);
        match mode {
            EraseMode::ToEnd => {
                self.screen.clear_line_segment(cy_phys, cx_phys, screen_ctx.width);
                for y in (cy_phys + 1)..screen_ctx.height { self.screen.clear_line_segment(y, 0, screen_ctx.width); }
            }
            EraseMode::ToStart => {
                for y in 0..cy_phys { self.screen.clear_line_segment(y, 0, screen_ctx.width); }
                self.screen.clear_line_segment(cy_phys, 0, cx_phys + 1);
            }
            EraseMode::All => {
                for y in 0..screen_ctx.height { self.screen.clear_line_segment(y, 0, screen_ctx.width); }
            }
            EraseMode::Scrollback => { self.screen.scrollback.clear(); return; }
            EraseMode::Unknown => warn!("Unknown ED mode used."),
        }
        if mode != EraseMode::Scrollback {
            self.mark_all_lines_dirty();
        }
    }
    fn erase_in_line(&mut self, mode: EraseMode) {
        let screen_ctx = self.current_screen_context();
        let (cx_phys, cy_phys) = self.cursor_controller.physical_screen_pos(&screen_ctx);
        match mode {
            EraseMode::ToEnd => self.screen.clear_line_segment(cy_phys, cx_phys, screen_ctx.width),
            EraseMode::ToStart => self.screen.clear_line_segment(cy_phys, 0, cx_phys + 1),
            EraseMode::All => self.screen.clear_line_segment(cy_phys, 0, screen_ctx.width),
            EraseMode::Scrollback => warn!("EraseMode::Scrollback is not applicable to EraseInLine (EL)."),
            EraseMode::Unknown => warn!("Unknown EL mode used."),
        }
    }
    fn erase_chars(&mut self, n: usize) {
        let screen_ctx = self.current_screen_context();
        let (cx_phys, cy_phys) = self.cursor_controller.physical_screen_pos(&screen_ctx);
        let end_x = min(cx_phys + n, screen_ctx.width);
        self.screen.clear_line_segment(cy_phys, cx_phys, end_x);
    }

    fn insert_blank_chars(&mut self, n: usize) {
        let screen_ctx = self.current_screen_context();
        let (cx_log, _) = self.cursor_controller.logical_pos();
        let (_, cy_phys) = self.cursor_controller.physical_screen_pos(&screen_ctx);
        self.screen.insert_blank_chars_in_line(cy_phys, cx_log, n);
    }
    fn delete_chars(&mut self, n: usize) {
        let screen_ctx = self.current_screen_context();
        let (cx_log, _) = self.cursor_controller.logical_pos();
        let (_, cy_phys) = self.cursor_controller.physical_screen_pos(&screen_ctx);
        self.screen.delete_chars_in_line(cy_phys, cx_log, n);
    }
    fn insert_lines(&mut self, n: usize) {
        let screen_ctx = self.current_screen_context();
        let (_, cy_phys) = self.cursor_controller.physical_screen_pos(&screen_ctx);
        if cy_phys >= screen_ctx.scroll_top && cy_phys <= screen_ctx.scroll_bot {
            let original_scroll_top = self.screen.scroll_top();
            let original_scroll_bottom = self.screen.scroll_bot();
            self.screen.set_scrolling_region(cy_phys + 1, original_scroll_bottom + 1);
            self.screen.scroll_down_serial(n);
            self.screen.set_scrolling_region(original_scroll_top + 1, original_scroll_bottom + 1);
            for y_dirty in original_scroll_top..=original_scroll_bottom { self.mark_line_dirty(y_dirty); }
        }
    }
    fn delete_lines(&mut self, n: usize) {
        let screen_ctx = self.current_screen_context();
        let (_, cy_phys) = self.cursor_controller.physical_screen_pos(&screen_ctx);
        if cy_phys >= screen_ctx.scroll_top && cy_phys <= screen_ctx.scroll_bot {
            let original_scroll_top = self.screen.scroll_top();
            let original_scroll_bottom = self.screen.scroll_bot();
            self.screen.set_scrolling_region(cy_phys + 1, original_scroll_bottom + 1);
            self.screen.scroll_up_serial(n);
            self.screen.set_scrolling_region(original_scroll_top + 1, original_scroll_bottom + 1);
            for y_dirty in original_scroll_top..=original_scroll_bottom { self.mark_line_dirty(y_dirty); }
        }
    }
    fn scroll_up(&mut self, n: usize) {
        self.screen.scroll_up_serial(n);
    }
    fn scroll_down(&mut self, n: usize) {
        self.screen.scroll_down_serial(n);
    }

    fn handle_sgr_attributes(&mut self, attributes_vec: Vec<Attribute>) {
        let mut current_attrs = self.cursor_controller.attributes();
        for attr_cmd in attributes_vec {
            match attr_cmd {
                Attribute::Reset => current_attrs = Attributes::default(),
                Attribute::Bold => current_attrs.flags.insert(AttrFlags::BOLD),
                Attribute::Faint => current_attrs.flags.insert(AttrFlags::FAINT),
                Attribute::Italic => current_attrs.flags.insert(AttrFlags::ITALIC),
                Attribute::Underline => current_attrs.flags.insert(AttrFlags::UNDERLINE),
                Attribute::BlinkSlow | Attribute::BlinkRapid => current_attrs.flags.insert(AttrFlags::BLINK),
                Attribute::Reverse => current_attrs.flags.insert(AttrFlags::REVERSE),
                Attribute::Conceal => current_attrs.flags.insert(AttrFlags::HIDDEN),
                Attribute::Strikethrough => current_attrs.flags.insert(AttrFlags::STRIKETHROUGH),
                Attribute::Foreground(color) => current_attrs.fg = map_ansi_color_to_glyph_color(color),
                Attribute::Background(color) => current_attrs.bg = map_ansi_color_to_glyph_color(color),
                Attribute::NoBold => { current_attrs.flags.remove(AttrFlags::BOLD); current_attrs.flags.remove(AttrFlags::FAINT); }
                Attribute::NoItalic => current_attrs.flags.remove(AttrFlags::ITALIC),
                Attribute::NoUnderline => current_attrs.flags.remove(AttrFlags::UNDERLINE),
                Attribute::NoBlink => current_attrs.flags.remove(AttrFlags::BLINK),
                Attribute::NoReverse => current_attrs.flags.remove(AttrFlags::REVERSE),
                Attribute::NoConceal => current_attrs.flags.remove(AttrFlags::HIDDEN),
                Attribute::NoStrikethrough => current_attrs.flags.remove(AttrFlags::STRIKETHROUGH),
                Attribute::UnderlineColor(color) => warn!("SGR UnderlineColor not yet fully supported: {:?}", color),
                Attribute::Overlined => warn!("SGR Overlined not yet supported."),
                Attribute::UnderlineDouble =>{current_attrs.flags.insert(AttrFlags::UNDERLINE); warn!("SGR Double Underline treated as single.");}
            }
        }
        self.cursor_controller.set_attributes(current_attrs);
        self.screen.default_attributes = current_attrs;
    }

    fn handle_set_mode(&mut self, mode_type: Mode, enable: bool) -> Option<EmulatorAction> {
        let mut action = None;
        match mode_type {
            Mode::DecPrivate(mode_num) => {
                trace!("Setting DEC Private Mode {} to {}", mode_num, enable);
                // Try to match known DecModeConstant values first
                match DecModeConstant::from_u16(mode_num) {
                    Some(DecModeConstant::CursorKeys) => self.dec_modes.cursor_keys_app_mode = enable,
                    Some(DecModeConstant::Origin) => {
                        self.dec_modes.origin_mode = enable;
                        self.screen.origin_mode = enable;
                        self.cursor_controller.move_to_logical(0, 0, &self.current_screen_context());
                    }
                    Some(DecModeConstant::TextCursorEnable) => {
                        self.cursor_controller.set_visible(enable);
                        action = Some(EmulatorAction::SetCursorVisibility(enable));
                    }
                    Some(DecModeConstant::AltScreenBufferClear) => {
                        if enable {
                            if !self.dec_modes.using_alt_screen {
                                self.screen.enter_alt_screen(true);
                                self.dec_modes.using_alt_screen = true;
                                self.cursor_controller.move_to_logical(0,0, &self.current_screen_context());
                                self.screen.default_attributes = self.cursor_controller.attributes();
                                action = Some(EmulatorAction::RequestRedraw);
                            }
                        } else {
                            if self.dec_modes.using_alt_screen {
                                self.screen.exit_alt_screen();
                                self.dec_modes.using_alt_screen = false;
                                self.cursor_controller.move_to_logical(0,0, &self.current_screen_context());
                                self.screen.default_attributes = self.cursor_controller.attributes();
                                action = Some(EmulatorAction::RequestRedraw);
                            }
                        }
                    }
                    Some(DecModeConstant::AltScreenBufferSaveRestore) => {
                        if enable {
                            if !self.dec_modes.using_alt_screen {
                                self.cursor_controller.save_state();
                                self.screen.enter_alt_screen(true);
                                self.dec_modes.using_alt_screen = true;
                                self.cursor_controller.move_to_logical(0,0, &self.current_screen_context());
                                self.screen.default_attributes = self.cursor_controller.attributes();
                                action = Some(EmulatorAction::RequestRedraw);
                            }
                        } else {
                            if self.dec_modes.using_alt_screen {
                                self.screen.exit_alt_screen();
                                self.dec_modes.using_alt_screen = false;
                                self.cursor_controller.restore_state(&self.current_screen_context(), Attributes::default());
                                self.screen.default_attributes = self.cursor_controller.attributes();
                                action = Some(EmulatorAction::RequestRedraw);
                            }
                        }
                    }
                    Some(DecModeConstant::SaveRestoreCursor) => {
                        if enable { self.cursor_controller.save_state(); }
                        else { self.cursor_controller.restore_state(&self.current_screen_context(), Attributes::default()); }
                        self.screen.default_attributes = self.cursor_controller.attributes();
                    }
                    None => { // Mode number not in DecModeConstant enum
                        warn!("Unknown DEC private mode {} to set/reset: {}", mode_num, enable);
                    }
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
        'j' => '┘', 'k' => '┐', 'l' => '┌', 'm' => '└', 'n' => '┼', 'o' => '─',
        'p' => '─', 'q' => '─', 'r' => '─', 's' => '─', 't' => '├', 'u' => '┤',
        'v' => '┴', 'w' => '┬', 'x' => '│', 'y' => '≤', 'z' => '≥', '{' => 'π',
        '|' => '≠', '}' => '£', '~' => '·',
        _ => ch,
    }
}

#[cfg(test)]
mod tests;
