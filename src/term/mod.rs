// myterm/src/term/mod.rs

//! This module defines the core terminal emulator logic.
//! It acts as a state machine processing inputs and producing actions.

// Sub-modules - existing and new
pub mod screen;     // Existing
pub mod cursor;     // Existing
pub mod unicode;    // Existing

pub mod action;     // New
pub mod charset;    // New
pub mod input;      // New
pub mod modes;      // New
// pub mod utils;   // User opted out of this for now

// Re-export items for easier use by other modules and within this module
pub use action::EmulatorAction;
pub use charset::{CharacterSet, map_to_dec_line_drawing};
pub use input::EmulatorInput;
pub use modes::{EraseMode, DecModeConstant, DecPrivateModes, Mode};


// Standard library imports
use std::cmp::min;

// Crate-level imports (adjust paths based on where items are moved)
use crate::{
    ansi::commands::{
            AnsiCommand, C0Control, CsiCommand, Attribute,
            Color as AnsiColor, // Keep EscCommand if used by AnsiCommand::from_c1 or from_esc
    },
    term::unicode::get_char_display_width, // unicode is an existing submodule
    backends::BackendEvent,
    glyph::{Glyph, Attributes, AttrFlags, Color as GlyphColor, NamedColor},
    term::screen::Screen, // screen is an existing submodule
    term::cursor::{CursorController, ScreenContext}, // cursor is an existing submodule
};

// Logging
use log::{debug, trace, warn};

/// Default tab interval.
pub const DEFAULT_TAB_INTERVAL: u8 = 8;

// Definitions for EraseMode, DecModeConstant, DecPrivateModes, Mode, CharacterSet,
// EmulatorAction, and EmulatorInput have been moved to their respective submodules.
// The map_to_dec_line_drawing function has been moved to charset.rs.

/// The core terminal emulator.
pub struct TerminalEmulator {
    screen: Screen,
    cursor_controller: CursorController,
    dec_modes: DecPrivateModes, // This type is now from modes.rs
    active_charsets: [CharacterSet; 4], // This type is now from charset.rs
    active_charset_g_level: usize,
    dirty_lines: Vec<usize>,
    cursor_wrap_next: bool,
    current_cursor_shape: u16, // For DECSCUSR, default 1 or 2 (block)
}

impl TerminalEmulator {
    /// Creates a new `TerminalEmulator`.
    pub fn new(width: usize, height: usize, scrollback_limit: usize) -> Self {
        let initial_attributes = Attributes::default();
        let mut screen = Screen::new(width, height, scrollback_limit);
        screen.default_attributes = initial_attributes;

        TerminalEmulator {
            screen,
            cursor_controller: CursorController::new(initial_attributes),
            dec_modes: DecPrivateModes::default(),
            active_charsets: [
                CharacterSet::Ascii, CharacterSet::Ascii,
                CharacterSet::Ascii, CharacterSet::Ascii,
            ],
            active_charset_g_level: 0,
            dirty_lines: (0..height.max(1)).collect(),
            cursor_wrap_next: false,
            current_cursor_shape: 2, // Default to steady block (common default)
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

    // --- Public Accessor Methods for Tests ---
    pub fn is_origin_mode_active(&self) -> bool { self.dec_modes.origin_mode }
    pub fn is_cursor_keys_app_mode_active(&self) -> bool { self.dec_modes.cursor_keys_app_mode }
    pub fn is_bracketed_paste_mode_active(&self) -> bool { self.dec_modes.bracketed_paste_mode }
    pub fn is_focus_event_mode_active(&self) -> bool { self.dec_modes.focus_event_mode }
    
    pub fn is_mouse_mode_active(&self, mode_num: u16) -> bool {
        match DecModeConstant::from_u16(mode_num) { // DecModeConstant is now from modes.rs
            Some(DecModeConstant::MouseX10) => self.dec_modes.mouse_x10_mode,
            Some(DecModeConstant::MouseVt200) => self.dec_modes.mouse_vt200_mode,
            Some(DecModeConstant::MouseVt200Highlight) => self.dec_modes.mouse_vt200_highlight_mode,
            Some(DecModeConstant::MouseButtonEvent) => self.dec_modes.mouse_button_event_mode,
            Some(DecModeConstant::MouseAnyEvent) => self.dec_modes.mouse_any_event_mode,
            Some(DecModeConstant::MouseUtf8) => self.dec_modes.mouse_utf8_mode,
            Some(DecModeConstant::MouseSgr) => self.dec_modes.mouse_sgr_mode,
            Some(DecModeConstant::MouseUrxvt) => {
                warn!("is_mouse_mode_active check for MouseUrxvt (1015): Not fully implemented.");
                false 
            }
            Some(DecModeConstant::MousePixelPosition) => {
                warn!("is_mouse_mode_active check for MousePixelPosition (1016): Not fully implemented.");
                false
            }
            _ => {
                warn!("is_mouse_mode_active called with non-mouse mode or unhandled mouse mode: {}", mode_num);
                false
            }
        }
    }
    pub fn get_cursor_shape(&self) -> u16 { self.current_cursor_shape }


    /// Interprets an `EmulatorInput` and updates the terminal state.
    pub fn interpret_input(&mut self, input: EmulatorInput) -> Option<EmulatorAction> { // EmulatorInput and EmulatorAction are from new submodules
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
        if !matches!(command, AnsiCommand::Print(_)) {
            self.cursor_wrap_next = false;
        }

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
            AnsiCommand::Esc(esc_cmd) => match esc_cmd {
                crate::ansi::commands::EscCommand::SetTabStop => {
                    let (cursor_x, _) = self.cursor_controller.logical_pos();
                    self.screen.set_tabstop(cursor_x);
                    None
                }
                crate::ansi::commands::EscCommand::Index => { self.index(); None }
                crate::ansi::commands::EscCommand::NextLine => { self.newline(true); self.carriage_return(); None }
                crate::ansi::commands::EscCommand::ReverseIndex => { self.reverse_index(); None }
                crate::ansi::commands::EscCommand::SaveCursor => { self.save_cursor_dec(); None }
                crate::ansi::commands::EscCommand::RestoreCursor => { self.restore_cursor_dec(); None }
                crate::ansi::commands::EscCommand::SelectCharacterSet(intermediate_char, final_char) => {
                    let g_idx = match intermediate_char {
                        '(' => 0, ')' => 1, '*' => 2, '+' => 3,
                        _ => { warn!("Unsupported G-set designator intermediate: {}", intermediate_char); 0 }
                    };
                    self.designate_character_set(g_idx, CharacterSet::from_char(final_char)); // CharacterSet from charset.rs
                    None
                }
                _ => { debug!("Unhandled Esc command: {:?}", esc_cmd); None }
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
                CsiCommand::EraseInDisplay(mode_val) => { self.erase_in_display(EraseMode::from(mode_val)); None } // EraseMode from modes.rs
                CsiCommand::EraseInLine(mode_val) => { self.erase_in_line(EraseMode::from(mode_val)); None } // EraseMode from modes.rs
                CsiCommand::InsertCharacter(n) => { self.insert_blank_chars(n.max(1) as usize); None }
                CsiCommand::DeleteCharacter(n) => { self.delete_chars(n.max(1) as usize); None }
                CsiCommand::InsertLine(n) => { self.insert_lines(n.max(1) as usize); None }
                CsiCommand::DeleteLine(n) => { self.delete_lines(n.max(1) as usize); None }
                CsiCommand::SetGraphicsRendition(attrs_vec) => { self.handle_sgr_attributes(attrs_vec); None }
                CsiCommand::SetMode(mode_num) => self.handle_set_mode(Mode::Standard(mode_num), true), // Mode from modes.rs
                CsiCommand::ResetMode(mode_num) => self.handle_set_mode(Mode::Standard(mode_num), false), // Mode from modes.rs
                CsiCommand::SetModePrivate(mode_num) => self.handle_set_mode(Mode::DecPrivate(mode_num), true), // Mode from modes.rs
                CsiCommand::ResetModePrivate(mode_num) => self.handle_set_mode(Mode::DecPrivate(mode_num), false), // Mode from modes.rs
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
                CsiCommand::SetCursorStyle { shape } => {
                    self.current_cursor_shape = shape;
                    debug!("Set cursor style to shape: {}", shape);
                    None
                }
                CsiCommand::WindowManipulation { ps1, ps2, ps3 } => {
                    self.handle_window_manipulation(ps1, ps2, ps3)
                }
                CsiCommand::Unsupported(intermediates, final_byte_opt) => {
                    warn!("TerminalEmulator received CsiCommand::Unsupported: intermediates={:?}, final={:?}", intermediates, final_byte_opt);
                    None
                }
                _ => { debug!("Unhandled CsiCommand variant: {:?}", csi); None }
            },
            AnsiCommand::Osc(data) => self.handle_osc(data),
            AnsiCommand::Print(ch) => { self.print_char(ch); None }
            _ => { debug!("Unhandled ANSI command type: {:?}", command); None }
        }
    }

    /// Handles a `BackendEvent`.
    fn handle_backend_event(&mut self, event: BackendEvent) -> Option<EmulatorAction> {
        self.cursor_wrap_next = false;
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
        self.cursor_wrap_next = false;
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

        if char_width == 0 { return; } 

        let mut screen_ctx = self.current_screen_context();

        if self.cursor_wrap_next {
            self.newline(true);
            screen_ctx = self.current_screen_context(); 
            self.cursor_wrap_next = false; 
        }

        let (mut physical_x, mut physical_y) = self.cursor_controller.physical_screen_pos(&screen_ctx);

        if physical_x + char_width > screen_ctx.width {
            if char_width == 2 && physical_x == screen_ctx.width.saturating_sub(1) {
                let fill_glyph = Glyph { c: ' ', attr: self.cursor_controller.attributes() };
                if physical_y < self.screen.height {
                    self.screen.set_glyph(physical_x, physical_y, fill_glyph);
                    self.mark_line_dirty(physical_y);
                }
            }
            self.newline(true);
            screen_ctx = self.current_screen_context(); 
            (physical_x, physical_y) = self.cursor_controller.physical_screen_pos(&screen_ctx);
        }

        let glyph_attrs = self.cursor_controller.attributes();
        if physical_y < self.screen.height { 
            self.screen.set_glyph(physical_x, physical_y, Glyph { c: ch_to_print, attr: glyph_attrs });
            self.mark_line_dirty(physical_y);

            if char_width == 2 {
                if physical_x + 1 < screen_ctx.width { 
                    self.screen.set_glyph(physical_x + 1, physical_y, Glyph { c: '\0', attr: glyph_attrs });
                } else {
                    warn!("Wide char placeholder for '{}' at ({},{}) could not be placed due to edge of screen.", ch_to_print, physical_x, physical_y);
                }
            }
        } else {
            warn!("print_char: Attempted to print at physical_y {} out of bounds (height {})", physical_y, self.screen.height);
        }

        self.cursor_controller.move_right(char_width, &screen_ctx);

        let (final_logical_x, _) = self.cursor_controller.logical_pos();
        if final_logical_x == screen_ctx.width {
            self.cursor_wrap_next = true;
        } else {
            self.cursor_wrap_next = false; 
        }
    }


    fn map_char_to_active_charset(&self, ch: char) -> char {
        let current_set = self.active_charsets[self.active_charset_g_level];
        match current_set { // CharacterSet is from charset.rs
            CharacterSet::Ascii => ch,
            CharacterSet::UkNational => if ch == '#' { '£' } else { ch },
            CharacterSet::DecLineDrawing => map_to_dec_line_drawing(ch), // map_to_dec_line_drawing from charset.rs
        }
    }

    fn backspace(&mut self) { 
        self.cursor_wrap_next = false;
        self.cursor_controller.move_left(1); 
    }

    fn horizontal_tab(&mut self) {
        self.cursor_wrap_next = false;
        let (current_x, _) = self.cursor_controller.logical_pos();
        let screen_ctx = self.current_screen_context();
        let next_stop = self.screen.get_next_tabstop(current_x).unwrap_or(screen_ctx.width.saturating_sub(1));
        self.cursor_controller.move_to_logical_col(next_stop, &screen_ctx);
    }

    fn newline(&mut self, is_line_feed: bool) {
        self.cursor_wrap_next = false; 
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
        
        self.mark_line_dirty(current_physical_y);
        let (_, new_physical_y) = self.cursor_controller.physical_screen_pos(&self.current_screen_context());
        if current_physical_y != new_physical_y {
            self.mark_line_dirty(new_physical_y);
        }
    }
    fn carriage_return(&mut self) { 
        self.cursor_wrap_next = false;
        self.cursor_controller.carriage_return(); 
    }

    fn set_g_level(&mut self, g_level: usize) {
        if g_level < self.active_charsets.len() {
            self.active_charset_g_level = g_level;
            trace!("Switched to G{} character set mapping.", g_level);
        } else { warn!("Attempted to set invalid G-level: {}", g_level); }
    }
    fn designate_character_set(&mut self, g_set_index: usize, charset: CharacterSet) { // CharacterSet from charset.rs
        if g_set_index < self.active_charsets.len() {
            self.active_charsets[g_set_index] = charset;
            trace!("Designated G{} to {:?}", g_set_index, charset);
        } else { warn!("Invalid G-set index for designate charset: {}", g_set_index); }
    }

    fn index(&mut self) {
        self.cursor_wrap_next = false;
        let screen_ctx = self.current_screen_context();
        let (_, current_logical_y) = self.cursor_controller.logical_pos();
        let (_, current_physical_y) = self.cursor_controller.physical_screen_pos(&screen_ctx);
        let max_logical_y = if screen_ctx.origin_mode_active { screen_ctx.scroll_bot.saturating_sub(screen_ctx.scroll_top) } else { screen_ctx.height.saturating_sub(1) };

        if current_logical_y == max_logical_y {
            self.screen.scroll_up_serial(1);
        } else if current_logical_y < max_logical_y {
            self.cursor_controller.move_down(1, &screen_ctx);
        }
        self.mark_line_dirty(current_physical_y);
        let (_, new_physical_y) = self.cursor_controller.physical_screen_pos(&self.current_screen_context());
        if current_physical_y != new_physical_y { self.mark_line_dirty(new_physical_y); }
    }
    fn reverse_index(&mut self) {
        self.cursor_wrap_next = false;
        let screen_ctx = self.current_screen_context();
        let (_, current_logical_y) = self.cursor_controller.logical_pos();
        let (_, current_physical_y) = self.cursor_controller.physical_screen_pos(&screen_ctx);

        if current_logical_y == 0 {
            self.screen.scroll_down_serial(1);
        } else {
            self.cursor_controller.move_up(1);
        }
        self.mark_line_dirty(current_physical_y);
        let (_, new_physical_y) = self.cursor_controller.physical_screen_pos(&self.current_screen_context());
        if current_physical_y != new_physical_y { self.mark_line_dirty(new_physical_y); }
    }
    fn save_cursor_dec(&mut self) { self.cursor_controller.save_state(); }
    fn restore_cursor_dec(&mut self) {
        self.cursor_wrap_next = false;
        let default_attrs = Attributes::default();
        self.cursor_controller.restore_state(&self.current_screen_context(), default_attrs);
        self.screen.default_attributes = self.cursor_controller.attributes();
    }

    fn cursor_up(&mut self, n: usize) { self.cursor_wrap_next = false; self.cursor_controller.move_up(n); }
    fn cursor_down(&mut self, n: usize) { self.cursor_wrap_next = false; self.cursor_controller.move_down(n, &self.current_screen_context()); }
    fn cursor_forward(&mut self, n: usize) { self.cursor_wrap_next = false; self.cursor_controller.move_right(n, &self.current_screen_context()); }
    fn cursor_backward(&mut self, n: usize) { self.cursor_wrap_next = false; self.cursor_controller.move_left(n); }
    fn cursor_to_column(&mut self, col: usize) { self.cursor_wrap_next = false; self.cursor_controller.move_to_logical_col(col, &self.current_screen_context()); }
    fn cursor_to_pos(&mut self, row_param: usize, col_param: usize) {
        self.cursor_wrap_next = false;
        self.cursor_controller.move_to_logical(col_param, row_param, &self.current_screen_context());
    }

    fn erase_in_display(&mut self, mode: EraseMode) { // EraseMode from modes.rs
        self.cursor_wrap_next = false;
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
    fn erase_in_line(&mut self, mode: EraseMode) { // EraseMode from modes.rs
        self.cursor_wrap_next = false;
        let screen_ctx = self.current_screen_context();
        let (cx_phys, cy_phys) = self.cursor_controller.physical_screen_pos(&screen_ctx);
        match mode {
            EraseMode::ToEnd => self.screen.clear_line_segment(cy_phys, cx_phys, screen_ctx.width),
            EraseMode::ToStart => self.screen.clear_line_segment(cy_phys, 0, cx_phys + 1),
            EraseMode::All => self.screen.clear_line_segment(cy_phys, 0, screen_ctx.width),
            EraseMode::Scrollback => warn!("EraseMode::Scrollback is not applicable to EraseInLine (EL)."),
            EraseMode::Unknown => warn!("Unknown EL mode used."),
        }
        self.mark_line_dirty(cy_phys);
    }
    fn erase_chars(&mut self, n: usize) {
        self.cursor_wrap_next = false;
        let screen_ctx = self.current_screen_context();
        let (cx_phys, cy_phys) = self.cursor_controller.physical_screen_pos(&screen_ctx);
        let end_x = min(cx_phys + n, screen_ctx.width);
        self.screen.clear_line_segment(cy_phys, cx_phys, end_x);
        self.mark_line_dirty(cy_phys);
    }

    fn insert_blank_chars(&mut self, n: usize) {
        self.cursor_wrap_next = false;
        let screen_ctx = self.current_screen_context();
        let (cx_log, _) = self.cursor_controller.logical_pos();
        let (_, cy_phys) = self.cursor_controller.physical_screen_pos(&screen_ctx);
        self.screen.insert_blank_chars_in_line(cy_phys, cx_log, n);
        self.mark_line_dirty(cy_phys);
    }
    fn delete_chars(&mut self, n: usize) {
        self.cursor_wrap_next = false;
        let screen_ctx = self.current_screen_context();
        let (cx_log, _) = self.cursor_controller.logical_pos();
        let (_, cy_phys) = self.cursor_controller.physical_screen_pos(&screen_ctx);
        self.screen.delete_chars_in_line(cy_phys, cx_log, n);
        self.mark_line_dirty(cy_phys);
    }
    fn insert_lines(&mut self, n: usize) {
        self.cursor_wrap_next = false;
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
        self.cursor_wrap_next = false;
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
        self.cursor_wrap_next = false;
        self.screen.scroll_up_serial(n);
        for y_dirty in self.screen.scroll_top()..=self.screen.scroll_bot() {
            self.mark_line_dirty(y_dirty);
        }
    }
    fn scroll_down(&mut self, n: usize) {
        self.cursor_wrap_next = false;
        self.screen.scroll_down_serial(n);
        for y_dirty in self.screen.scroll_top()..=self.screen.scroll_bot() {
            self.mark_line_dirty(y_dirty);
        }
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
                Attribute::UnderlineDouble => current_attrs.flags.insert(AttrFlags::UNDERLINE), 
                Attribute::NoBold => { current_attrs.flags.remove(AttrFlags::BOLD); current_attrs.flags.remove(AttrFlags::FAINT); }
                Attribute::NoItalic => current_attrs.flags.remove(AttrFlags::ITALIC),
                Attribute::NoUnderline => current_attrs.flags.remove(AttrFlags::UNDERLINE),
                Attribute::NoBlink => current_attrs.flags.remove(AttrFlags::BLINK),
                Attribute::NoReverse => current_attrs.flags.remove(AttrFlags::REVERSE),
                Attribute::NoConceal => current_attrs.flags.remove(AttrFlags::HIDDEN),
                Attribute::NoStrikethrough => current_attrs.flags.remove(AttrFlags::STRIKETHROUGH),
                Attribute::Foreground(color) => current_attrs.fg = map_ansi_color_to_glyph_color(color),
                Attribute::Background(color) => current_attrs.bg = map_ansi_color_to_glyph_color(color),
                Attribute::Overlined => warn!("SGR Overlined not yet visually supported."),
                Attribute::NoOverlined => warn!("SGR NoOverlined not yet visually supported."),
                Attribute::UnderlineColor(color) => warn!("SGR UnderlineColor not yet fully supported: {:?}", color),
            }
        }
        self.cursor_controller.set_attributes(current_attrs);
        self.screen.default_attributes = current_attrs;
    }

    fn handle_set_mode(&mut self, mode_type: Mode, enable: bool) -> Option<EmulatorAction> { // Mode from modes.rs
        self.cursor_wrap_next = false;
        let mut action = None;
        match mode_type {
            Mode::DecPrivate(mode_num) => {
                trace!("Setting DEC Private Mode {} to {}", mode_num, enable);
                match DecModeConstant::from_u16(mode_num) { // DecModeConstant from modes.rs
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
                    Some(DecModeConstant::BracketedPaste) => self.dec_modes.bracketed_paste_mode = enable,
                    Some(DecModeConstant::FocusEvent) => self.dec_modes.focus_event_mode = enable,
                    Some(DecModeConstant::MouseX10) => self.dec_modes.mouse_x10_mode = enable,
                    Some(DecModeConstant::MouseVt200) => self.dec_modes.mouse_vt200_mode = enable,
                    Some(DecModeConstant::MouseVt200Highlight) => self.dec_modes.mouse_vt200_highlight_mode = enable,
                    Some(DecModeConstant::MouseButtonEvent) => self.dec_modes.mouse_button_event_mode = enable,
                    Some(DecModeConstant::MouseAnyEvent) => self.dec_modes.mouse_any_event_mode = enable,
                    Some(DecModeConstant::MouseUtf8) => self.dec_modes.mouse_utf8_mode = enable,
                    Some(DecModeConstant::MouseSgr) => self.dec_modes.mouse_sgr_mode = enable,
                    Some(DecModeConstant::MouseUrxvt) => {
                        warn!("DEC Private Mode {} (MouseUrxvt) set to {} - not fully implemented.", mode_num, enable);
                    }
                    Some(DecModeConstant::MousePixelPosition) => {
                        warn!("DEC Private Mode {} (MousePixelPosition) set to {} - not fully implemented.", mode_num, enable);
                    }
                    Some(DecModeConstant::Att610CursorBlink) => {
                        warn!("DEC Private Mode 12 (ATT610 Cursor Blink) set to {} - blink animation not implemented.", enable);
                    }
                    Some(DecModeConstant::Unknown7727) => {
                         warn!("DEC Private Mode 7727 set to {} - behavior undefined.", enable);
                    }
                    None => { 
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
    
    fn handle_window_manipulation(&mut self, ps1: u16, _ps2: Option<u16>, _ps3: Option<u16>) -> Option<EmulatorAction> {
        match ps1 {
            14 => { 
                warn!("WindowManipulation: Report text area size in pixels (14) requested, but not fully implemented to report actual pixels.");
                None
            }
            18 => { 
                let (cols, rows) = self.dimensions();
                let response = format!("\x1b[8;{};{}t", rows, cols);
                Some(EmulatorAction::WritePty(response.into_bytes()))
            }
            22 | 23 => { 
                warn!("WindowManipulation: Save/Restore window title (22/23) not implemented.");
                None
            }
            _ => {
                warn!("Unhandled WindowManipulation: ps1={}, ps2={:?}, ps3={:?}", ps1, _ps2, _ps3);
                None
            }
        }
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

// map_to_dec_line_drawing function is now in charset.rs

#[cfg(test)]
mod tests;

