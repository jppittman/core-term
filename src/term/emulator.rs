// src/term/emulator.rs

// Re-export items for easier use by other modules and within this module
use super::modes::{DecModeConstant, DecPrivateModes, EraseMode, Mode};
use super::*;
use action::EmulatorAction;
use charset::{map_to_dec_line_drawing, CharacterSet};

// Standard library imports
use std::cmp::{max, min}; // Added max

// Crate-level imports (adjust paths based on where items are moved)
use crate::{
    ansi::commands::{
        AnsiCommand,
        Attribute, // Keep Attribute as it's used in handle_sgr_attributes
        C0Control,
        CsiCommand,
        EscCommand,
    },
    backends::BackendEvent,
    glyph::{AttrFlags, Attributes, Glyph, WIDE_CHAR_PLACEHOLDER}, // Added WIDE_CHAR_PLACEHOLDER
    term::unicode::get_char_display_width,
    term::ControlEvent,
};

// Logging
use log::{debug, trace, warn};

// Constants (ensure these are defined, e.g., in term/mod.rs or config.rs if not already)
const DEFAULT_CURSOR_SHAPE: u16 = 1; // Example default shape
const DEFAULT_TAB_INTERVAL: u8 = 8;

/// The core terminal emulator.
pub struct TerminalEmulator {
    pub(super) screen: Screen,
    pub(super) cursor_controller: CursorController,
    pub(super) dec_modes: DecPrivateModes,
    pub(super) active_charsets: [CharacterSet; 4],
    pub(super) active_charset_g_level: usize,
    pub(super) cursor_wrap_next: bool,
    pub(super) current_cursor_shape: u16, // Stores the current cursor shape code
    pub(super) selection: Option<SelectionState>, // Added selection field
}

impl TerminalEmulator {
    /// Creates a new `TerminalEmulator`.
    pub fn new(width: usize, height: usize, scrollback_limit: usize) -> Self {
        let initial_attributes = Attributes::default(); // SGR Reset attributes
        let mut screen = Screen::new(width, height, scrollback_limit);
        // Ensure the screen's default_attributes are initialized correctly.
        // This is crucial for clearing operations.
        screen.default_attributes = initial_attributes;

        TerminalEmulator {
            screen,
            cursor_controller: CursorController::new(initial_attributes),
            dec_modes: DecPrivateModes::default(),
            active_charsets: [
                CharacterSet::Ascii, // G0
                CharacterSet::Ascii, // G1
                CharacterSet::Ascii, // G2
                CharacterSet::Ascii, // G3
            ],
            active_charset_g_level: 0, // Default to G0
            cursor_wrap_next: false,
            current_cursor_shape: DEFAULT_CURSOR_SHAPE, // Use constant for default
            selection: None, // Initialize selection to None
        }
    }

    /// Helper to create the current `ScreenContext` for `CursorController`.
    pub(super) fn current_screen_context(&self) -> ScreenContext {
        ScreenContext {
            width: self.screen.width,
            height: self.screen.height,
            scroll_top: self.screen.scroll_top(),
            scroll_bot: self.screen.scroll_bot(),
            origin_mode_active: self.dec_modes.origin_mode,
        }
    }

    // --- Public Accessor Methods for Tests ---
    #[allow(dead_code)]
    pub(super) fn is_origin_mode_active(&self) -> bool {
        self.dec_modes.origin_mode
    }
    #[allow(dead_code)]
    pub(super) fn is_cursor_keys_app_mode_active(&self) -> bool {
        self.dec_modes.cursor_keys_app_mode
    }
    #[allow(dead_code)]
    pub(super) fn is_bracketed_paste_mode_active(&self) -> bool {
        self.dec_modes.bracketed_paste_mode
    }
    #[allow(dead_code)]
    pub(super) fn is_focus_event_mode_active(&self) -> bool {
        self.dec_modes.focus_event_mode
    }

    #[allow(dead_code)]
    pub(super) fn is_mouse_mode_active(&self, mode_num: u16) -> bool {
        match DecModeConstant::from_u16(mode_num) {
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
                warn!(
                    "is_mouse_mode_active check for MousePixelPosition (1016): Not fully implemented."
                );
                false
            }
            _ => {
                warn!(
                    "is_mouse_mode_active called with non-mouse mode or unhandled mouse mode: {}",
                    mode_num
                );
                false
            }
        }
    }
    #[allow(dead_code)]
    pub(super) fn get_cursor_shape(&self) -> u16 {
        self.current_cursor_shape
    }

    /// Handles a parsed `AnsiCommand`.
    pub(super) fn handle_ansi_command(&mut self, command: AnsiCommand) -> Option<EmulatorAction> {
        if !matches!(command, AnsiCommand::Print(_)) {
            // Most non-Print commands should reset the cursor_wrap_next flag.
            // Specific commands like BS, HT might also reset it within their handlers if they affect wrapping state.
            self.cursor_wrap_next = false;
        }

        match command {
            AnsiCommand::C0Control(c0) => match c0 {
                C0Control::BS => {
                    self.backspace();
                    None
                }
                C0Control::HT => {
                    self.horizontal_tab();
                    None
                }
                C0Control::LF | C0Control::VT | C0Control::FF => {
                    self.perform_line_feed();
                    None
                }
                C0Control::CR => {
                    self.carriage_return();
                    None
                }
                C0Control::SO => {
                    self.set_g_level(1);
                    None
                }
                C0Control::SI => {
                    self.set_g_level(0);
                    None
                }
                C0Control::BEL => Some(EmulatorAction::RingBell),
                _ => {
                    debug!("Unhandled C0 control: {:?}", c0);
                    None
                }
            },
            AnsiCommand::Esc(esc_cmd) => match esc_cmd {
                EscCommand::SetTabStop => {
                    let (cursor_x, _) = self.cursor_controller.logical_pos();
                    self.screen.set_tabstop(cursor_x);
                    None
                }
                EscCommand::Index => {
                    self.index();
                    None
                }
                EscCommand::NextLine => {
                    // NEL is CR then LF.
                    // perform_line_feed handles LNM, so CR first ensures it's always CR+LF like.
                    self.carriage_return();
                    self.perform_line_feed();
                    None
                }
                EscCommand::ReverseIndex => {
                    self.reverse_index();
                    None
                }
                EscCommand::SaveCursor => {
                    self.save_cursor_dec();
                    None
                }
                EscCommand::RestoreCursor => {
                    self.restore_cursor_dec();
                    None
                }
                EscCommand::SelectCharacterSet(intermediate_char, final_char) => {
                    let g_idx = match intermediate_char {
                        '(' => 0,
                        ')' => 1,
                        '*' => 2,
                        '+' => 3,
                        _ => {
                            warn!(
                                "Unsupported G-set designator intermediate: {}",
                                intermediate_char
                            );
                            0
                        }
                    };
                    self.designate_character_set(g_idx, CharacterSet::from_char(final_char));
                    None
                }
                EscCommand::ResetToInitialState => {
                    trace!("Processing ESC c (ResetToInitialState)");
                    if self.screen.alt_screen_active {
                        self.screen.exit_alt_screen();
                    }
                    let default_attrs = Attributes::default();
                    self.cursor_controller.set_attributes(default_attrs);
                    self.screen.default_attributes = default_attrs;
                    self.erase_in_display(EraseMode::All);
                    self.cursor_controller
                        .move_to_logical(0, 0, &self.current_screen_context());
                    self.dec_modes = DecPrivateModes::default();
                    self.screen.origin_mode = self.dec_modes.origin_mode;
                    let (_, h) = self.dimensions();
                    self.screen.set_scrolling_region(1, h);
                    self.active_charsets = [CharacterSet::Ascii; 4];
                    self.active_charset_g_level = 0;
                    self.screen.clear_tabstops(0, screen::TabClearMode::All);
                    let (w, _) = self.dimensions();
                    for i in
                        (DEFAULT_TAB_INTERVAL as usize..w).step_by(DEFAULT_TAB_INTERVAL as usize)
                    {
                        self.screen.set_tabstop(i);
                    }
                    self.cursor_wrap_next = false;
                    self.current_cursor_shape = DEFAULT_CURSOR_SHAPE;
                    self.cursor_controller.set_visible(true);
                    if self.dec_modes.text_cursor_enable_mode {
                        return Some(EmulatorAction::SetCursorVisibility(true));
                    }
                    None
                }
                _ => {
                    debug!("Unhandled Esc command: {:?}", esc_cmd);
                    None
                }
            },
            AnsiCommand::Csi(csi) => match csi {
                CsiCommand::CursorUp(n) => {
                    self.cursor_up(n.max(1) as usize);
                    None
                }
                CsiCommand::CursorDown(n) => {
                    self.cursor_down(n.max(1) as usize);
                    None
                }
                CsiCommand::CursorForward(n) => {
                    self.cursor_forward(n.max(1) as usize);
                    None
                }
                CsiCommand::CursorBackward(n) => {
                    self.cursor_backward(n.max(1) as usize);
                    None
                }
                CsiCommand::CursorNextLine(n) => {
                    self.cursor_down(n.max(1) as usize);
                    self.carriage_return();
                    None
                }
                CsiCommand::CursorPrevLine(n) => {
                    self.cursor_up(n.max(1) as usize);
                    self.carriage_return();
                    None
                }
                CsiCommand::CursorCharacterAbsolute(n) => {
                    self.cursor_to_column(n.saturating_sub(1) as usize);
                    None
                }
                CsiCommand::CursorPosition(r, c) => {
                    self.cursor_to_pos(r.saturating_sub(1) as usize, c.saturating_sub(1) as usize);
                    None
                }
                CsiCommand::EraseInDisplay(mode_val) => {
                    self.erase_in_display(EraseMode::from(mode_val));
                    None
                }
                CsiCommand::EraseInLine(mode_val) => {
                    self.erase_in_line(EraseMode::from(mode_val));
                    None
                }
                CsiCommand::InsertCharacter(n) => {
                    self.insert_blank_chars(n.max(1) as usize);
                    None
                }
                CsiCommand::DeleteCharacter(n) => {
                    self.delete_chars(n.max(1) as usize);
                    None
                }
                CsiCommand::InsertLine(n) => {
                    self.insert_lines(n.max(1) as usize);
                    None
                }
                CsiCommand::DeleteLine(n) => {
                    self.delete_lines(n.max(1) as usize);
                    None
                }
                CsiCommand::SetGraphicsRendition(attrs_vec) => {
                    self.handle_sgr_attributes(attrs_vec);
                    None
                }
                CsiCommand::SetMode(mode_num) => {
                    self.handle_set_mode(Mode::Standard(mode_num), true)
                }
                CsiCommand::ResetMode(mode_num) => {
                    self.handle_set_mode(Mode::Standard(mode_num), false)
                }
                CsiCommand::SetModePrivate(mode_num) => {
                    self.handle_set_mode(Mode::DecPrivate(mode_num), true)
                }
                CsiCommand::ResetModePrivate(mode_num) => {
                    self.handle_set_mode(Mode::DecPrivate(mode_num), false)
                }
                CsiCommand::DeviceStatusReport(dsr_param) => {
                    if dsr_param == 0 || dsr_param == 6 {
                        let screen_ctx = self.current_screen_context();
                        let (abs_x, abs_y) =
                            self.cursor_controller.physical_screen_pos(&screen_ctx);
                        let response = format!("\x1B[{};{}R", abs_y + 1, abs_x + 1);
                        Some(EmulatorAction::WritePty(response.into_bytes()))
                    } else if dsr_param == 5 {
                        Some(EmulatorAction::WritePty(b"\x1B[0n".to_vec()))
                    } else {
                        warn!("Unhandled DSR parameter: {}", dsr_param);
                        None
                    }
                }
                CsiCommand::EraseCharacter(n) => {
                    self.erase_chars(n.max(1) as usize);
                    None
                }
                CsiCommand::ScrollUp(n) => {
                    self.scroll_up(n.max(1) as usize);
                    None
                }
                CsiCommand::ScrollDown(n) => {
                    self.scroll_down(n.max(1) as usize);
                    None
                }
                CsiCommand::SaveCursor => {
                    self.save_cursor_dec();
                    None
                }
                CsiCommand::RestoreCursor => {
                    self.restore_cursor_dec();
                    None
                }
                CsiCommand::ClearTabStops(mode_val) => {
                    let (cursor_x, _) = self.cursor_controller.logical_pos();
                    self.screen
                        .clear_tabstops(cursor_x, screen::TabClearMode::from(mode_val));
                    None
                }
                CsiCommand::SetScrollingRegion { top, bottom } => {
                    self.screen
                        .set_scrolling_region(top as usize, bottom as usize);
                    self.cursor_controller
                        .move_to_logical(0, 0, &self.current_screen_context());
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
                    warn!(
                        "TerminalEmulator received CsiCommand::Unsupported: intermediates={:?}, final={:?}. This is usually an error from the parser.",
                        intermediates, final_byte_opt
                    );
                    None
                }
                _ => {
                    debug!(
                        "Unhandled CsiCommand variant in TerminalEmulator: {:?}",
                        csi
                    );
                    None
                }
            },
            AnsiCommand::Osc(data) => self.handle_osc(data),
            AnsiCommand::Print(ch) => {
                self.print_char(ch);
                None
            }
            _ => {
                debug!(
                    "Unhandled ANSI command type in TerminalEmulator: {:?}",
                    command
                );
                None
            }
        }
    }

    /// Handles a `BackendEvent`.
    pub(super) fn handle_backend_event(&mut self, event: BackendEvent) -> Option<EmulatorAction> {
        self.cursor_wrap_next = false;

        const KEY_RETURN: u32 = 0xFF0D;
        const KEY_BACKSPACE: u32 = 0xFF08;
        const KEY_TAB: u32 = 0xFF09;
        const KEY_ISO_LEFT_TAB: u32 = 0xFE20;
        const KEY_ESCAPE: u32 = 0xFF1B;

        const KEY_LEFT_ARROW: u32 = 0xFF51;
        const KEY_UP_ARROW: u32 = 0xFF52;
        const KEY_RIGHT_ARROW: u32 = 0xFF53;
        const KEY_DOWN_ARROW: u32 = 0xFF54;

        match event {
            BackendEvent::Key { keysym, text } => {
                let mut bytes_to_send: Vec<u8> = Vec::new();
                match keysym {
                    KEY_RETURN => bytes_to_send.push(b'\r'),
                    KEY_BACKSPACE => bytes_to_send.push(0x08),
                    KEY_TAB => bytes_to_send.push(b'\t'),
                    KEY_ISO_LEFT_TAB => bytes_to_send.extend_from_slice(b"\x1b[Z"),
                    KEY_ESCAPE => bytes_to_send.push(0x1B),
                    KEY_UP_ARROW => {
                        bytes_to_send.extend_from_slice(if self.dec_modes.cursor_keys_app_mode {
                            b"\x1bOA"
                        } else {
                            b"\x1b[A"
                        })
                    }
                    KEY_DOWN_ARROW => {
                        bytes_to_send.extend_from_slice(if self.dec_modes.cursor_keys_app_mode {
                            b"\x1bOB"
                        } else {
                            b"\x1b[B"
                        })
                    }
                    KEY_RIGHT_ARROW => {
                        bytes_to_send.extend_from_slice(if self.dec_modes.cursor_keys_app_mode {
                            b"\x1bOC"
                        } else {
                            b"\x1b[C"
                        })
                    }
                    KEY_LEFT_ARROW => {
                        bytes_to_send.extend_from_slice(if self.dec_modes.cursor_keys_app_mode {
                            b"\x1bOD"
                        } else {
                            b"\x1b[D"
                        })
                    }
                    _ => {
                        if !text.is_empty() {
                            bytes_to_send.extend(text.as_bytes());
                        } else {
                            trace!("Unhandled keysym: {:#X} with empty text", keysym);
                        }
                    }
                }

                if !bytes_to_send.is_empty() {
                    return Some(EmulatorAction::WritePty(bytes_to_send));
                }
            }
            BackendEvent::FocusGained => {
                if self.dec_modes.focus_event_mode {
                    return Some(EmulatorAction::WritePty(b"\x1b[I".to_vec()));
                }
            }
            BackendEvent::FocusLost => {
                if self.dec_modes.focus_event_mode {
                    return Some(EmulatorAction::WritePty(b"\x1b[O".to_vec()));
                }
            }
            _ => debug!(
                "BackendEvent {:?} passed to TerminalEmulator's handle_backend_event, usually handled by Orchestrator or translated.",
                event
            ),
        }
        None
    }

    /// Handles a `UserInputAction`.
    /// This is a new method to specifically handle translated user inputs.
    pub(super) fn handle_user_input_action(
        &mut self,
        action: UserInputAction,
    ) -> Option<EmulatorAction> {
        self.cursor_wrap_next = false; // Reset wrap on any user action

        match action {
            UserInputAction::KeyInput {
                symbol,
                modifiers: _, // Modifiers might be used later for keybindings
                text,
            } => {
                // This largely replicates the old BackendEvent::Key logic.
                // Consider abstracting key-to-byte mapping if it grows complex.
                let mut bytes_to_send: Vec<u8> = Vec::new();
                match symbol {
                    KeySymbol::Return => bytes_to_send.push(b'\r'),
                    KeySymbol::Backspace => bytes_to_send.push(0x08),
                    KeySymbol::Tab => bytes_to_send.push(b'\t'),
                    // KeySymbol::IsoLeftTab => bytes_to_send.extend_from_slice(b"\x1b[Z"), // TODO: Add IsoLeftTab to KeySymbol
                    KeySymbol::Escape => bytes_to_send.push(0x1B),
                    KeySymbol::Up => {
                        bytes_to_send.extend_from_slice(if self.dec_modes.cursor_keys_app_mode {
                            b"\x1bOA"
                        } else {
                            b"\x1b[A"
                        })
                    }
                    KeySymbol::Down => {
                        bytes_to_send.extend_from_slice(if self.dec_modes.cursor_keys_app_mode {
                            b"\x1bOB"
                        } else {
                            b"\x1b[B"
                        })
                    }
                    KeySymbol::Right => {
                        bytes_to_send.extend_from_slice(if self.dec_modes.cursor_keys_app_mode {
                            b"\x1bOC"
                        } else {
                            b"\x1b[C"
                        })
                    }
                    KeySymbol::Left => {
                        bytes_to_send.extend_from_slice(if self.dec_modes.cursor_keys_app_mode {
                            b"\x1bOD"
                        } else {
                            b"\x1b[D"
                        })
                    }
                    KeySymbol::Char(ch) => {
                        // Use the provided text if available (e.g., from IME), otherwise char.
                        if let Some(text_str) = text {
                            bytes_to_send.extend(text_str.as_bytes());
                        } else {
                            bytes_to_send.push(ch as u8); // Simple char to byte conversion
                        }
                    }
                    // TODO: Handle other KeySymbols (F-keys, PageUp/Down, etc.)
                    _ => {
                        if let Some(text_str) = text {
                            bytes_to_send.extend(text_str.as_bytes());
                        } else {
                            trace!("Unhandled KeySymbol: {:?} with no text", symbol);
                        }
                    }
                }

                if !bytes_to_send.is_empty() {
                    return Some(EmulatorAction::WritePty(bytes_to_send));
                }
                None
            }
            UserInputAction::MouseInput {
                event_type,
                col,
                row,
                button,
                modifiers: _, // Modifiers might be used for advanced selection later
            } => {
                // TODO: Implement mouse reporting if a mouse mode is active (e.g., X10, SGR).
                // For now, focus on selection.

                let (clamped_col, clamped_row) = self.screen.clamp_coords(col, row);

                match event_type {
                    MouseEventType::Press => {
                        if button == MouseButton::Left {
                            // Start or restart selection
                            self.selection = Some(SelectionState {
                                start: (clamped_col, clamped_row),
                                end: (clamped_col, clamped_row),
                                mode: SelectionMode::Normal, // Default to Normal for now
                            });
                            // Mark the line of selection start as dirty
                            if clamped_row < self.screen.height {
                                self.screen.mark_line_dirty(clamped_row);
                            }
                            return Some(EmulatorAction::RequestRedraw); // Request redraw for selection highlight
                        }
                        // Handle other mouse buttons if needed in the future
                    }
                    MouseEventType::Move => {
                        if let Some(ref mut selection) = self.selection {
                            let old_end_row = selection.end.1;
                            if selection.end != (clamped_col, clamped_row) {
                                selection.end = (clamped_col, clamped_row);

                                // Mark lines covered by old and new selection end as dirty
                                if old_end_row < self.screen.height {
                                    self.screen.mark_line_dirty(old_end_row);
                                }
                                if clamped_row < self.screen.height && clamped_row != old_end_row {
                                    self.screen.mark_line_dirty(clamped_row);
                                }
                                // Also mark lines between old_end_row and clamped_row if selection spans multiple lines
                                // This is a simplified version; more precise dirtying might be needed for block selection
                                let (min_r, max_r) = if old_end_row < clamped_row {
                                    (old_end_row, clamped_row)
                                } else {
                                    (clamped_row, old_end_row)
                                };
                                for r in min_r..=max_r {
                                    if r < self.screen.height {
                                        self.screen.mark_line_dirty(r);
                                    }
                                }
                                return Some(EmulatorAction::RequestRedraw);
                            }
                        }
                    }
                    MouseEventType::Release => {
                        if button == MouseButton::Left {
                            // Selection is finalized.
                            // If selection is very small (e.g. start == end), could clear it here
                            // to treat it as a click, but current spec is just to finalize.
                            // For now, no specific action other than redraw if state changed.
                            if self.selection.is_some() {
                                // Ensure the last line of selection is marked dirty
                                let (sel_start, sel_end, _) = self.get_normalized_selection_bounds();
                                if sel_start.1 < self.screen.height { self.screen.mark_line_dirty(sel_start.1); }
                                if sel_end.1 < self.screen.height { self.screen.mark_line_dirty(sel_end.1); }
                                return Some(EmulatorAction::RequestRedraw);
                            }
                        }
                    }
                }
                None
            }
            UserInputAction::InitiateCopy => {
                if let Some(text_to_copy) = self.get_selected_text() {
                    // Clear selection after copying
                    if let Some(selection_state) = self.selection.take() {
                         let (start_coords, end_coords, _) = self.normalize_selection_coords(selection_state.start, selection_state.end, selection_state.mode);
                         self.mark_selection_dirty(Some(start_coords), Some(end_coords));
                    }
                    return Some(EmulatorAction::CopyToClipboard(text_to_copy));
                }
                None
            }
            UserInputAction::InitiatePaste => {
                // Request content from orchestrator. Orchestrator will send PasteText back.
                Some(EmulatorAction::RequestClipboardContent)
            }
            UserInputAction::PasteText(text) => {
                // This is how text from clipboard arrives
                // Bracketed paste mode check
                let mut bytes_to_send = Vec::new();
                if self.dec_modes.bracketed_paste_mode {
                    bytes_to_send.extend_from_slice(b"\x1b[200~");
                    bytes_to_send.extend_from_slice(text.as_bytes());
                    bytes_to_send.extend_from_slice(b"\x1b[201~");
                } else {
                    bytes_to_send.extend_from_slice(text.as_bytes());
                }
                if !bytes_to_send.is_empty() {
                    return Some(EmulatorAction::WritePty(bytes_to_send));
                }
                None
            }
        }
    }


    /// Handles an internal `ControlEvent`.
    pub(super) fn handle_control_event(&mut self, event: ControlEvent) -> Option<EmulatorAction> {
        self.cursor_wrap_next = false;
        match event {
            ControlEvent::FrameRendered => {
                trace!("TerminalEmulator: FrameRendered event received.");
                None
            }
            ControlEvent::Resize { cols, rows } => {
                trace!(
                    "TerminalEmulator: ControlEvent::Resize to {}x{} received.",
                    cols,
                    rows
                );
                self.resize(cols, rows);
                None
            }
        }
    }

    /// Resizes the terminal display grid.
    pub(super) fn resize(&mut self, cols: usize, rows: usize) {
        self.cursor_wrap_next = false;
        // When resizing, active selection should probably be cleared or adjusted.
        // For now, let's clear it to avoid complex coordinate recalculations.
        if self.selection.is_some() {
            let old_selection = self.selection.take();
            if let Some(sel) = old_selection {
                 let (start_coords, end_coords, _) = self.normalize_selection_coords(sel.start, sel.end, sel.mode);
                 self.mark_selection_dirty(Some(start_coords), Some(end_coords));
            }
        }

        let current_scrollback_limit = self.screen.scrollback_limit();
        self.screen.resize(cols, rows, current_scrollback_limit);
        let (log_x, log_y) = self.cursor_controller.logical_pos();
        self.cursor_controller
            .move_to_logical(log_x, log_y, &self.current_screen_context());
        debug!(
            "Terminal resized to {}x{}. Cursor re-clamped. All lines marked dirty by screen.resize().",
            cols, rows
        );
    }

    /// Marks lines covered by the selection (old or new) as dirty.
    fn mark_selection_dirty(&mut self, old_start: Option<(usize, usize)>, old_end: Option<(usize, usize)>) {
        let mut dirty_rows: std::collections::HashSet<usize> = std::collections::HashSet::new();

        if let Some(sel_state) = self.selection {
            let (current_start, current_end, _) = self.normalize_selection_coords(sel_state.start, sel_state.end, sel_state.mode);
            for r in current_start.1..=current_end.1 {
                if r < self.screen.height { dirty_rows.insert(r); }
            }
        }
        if let (Some(start), Some(end)) = (old_start, old_end) {
             let (norm_start, norm_end, _) = self.normalize_selection_coords(start, end, SelectionMode::Normal); // Mode doesn't matter for row range
            for r in norm_start.1..=norm_end.1 {
                if r < self.screen.height { dirty_rows.insert(r); }
            }
        }

        for row_idx in dirty_rows {
            self.screen.mark_line_dirty(row_idx);
        }
    }


    /// Returns the current logical cursor position (0-based column, row).
    pub fn cursor_pos(&self) -> (usize, usize) {
        self.cursor_controller.logical_pos()
    }

    /// Returns `true` if the alternate screen buffer is currently active.
    pub fn is_alt_screen_active(&self) -> bool {
        self.screen.alt_screen_active
    }

    /// Normalizes selection coordinates so start is before or at end.
    /// Returns ((start_col, start_row), (end_col, end_row), mode)
    fn normalize_selection_coords(
        &self,
        coord1: (usize, usize),
        coord2: (usize, usize),
        mode: SelectionMode,
    ) -> ((usize, usize), (usize, usize), SelectionMode) {
        let (r1, c1) = (coord1.1, coord1.0);
        let (r2, c2) = (coord2.1, coord2.0);

        if r1 < r2 || (r1 == r2 && c1 <= c2) {
            ((c1, r1), (c2, r2), mode)
        } else {
            ((c2, r2), (c1, r1), mode)
        }
    }
    
    /// Helper to get normalized selection bounds if a selection exists.
    fn get_normalized_selection_bounds(
        &self,
    ) -> ((usize, usize), (usize, usize), SelectionMode) {
        if let Some(sel_state) = self.selection {
            self.normalize_selection_coords(sel_state.start, sel_state.end, sel_state.mode)
        } else {
            // Default to a non-sensical range if no selection, or handle error
            ((0,0), (0,0), SelectionMode::Normal) 
        }
    }


    /// Retrieves the text currently selected.
    pub(super) fn get_selected_text(&self) -> Option<String> {
        if let Some(selection_state) = self.selection {
            let ((start_col, start_row), (end_col, end_row), mode) =
                self.normalize_selection_coords(selection_state.start, selection_state.end, selection_state.mode);

            if mode == SelectionMode::Normal {
                let mut selected_text = String::new();
                for r_idx in start_row..=end_row {
                    if r_idx >= self.screen.height {
                        continue;
                    }
                    let line = &self.screen.grid[r_idx];
                    let line_start_col = if r_idx == start_row { start_col } else { 0 };
                    let line_end_col = if r_idx == end_row {
                        end_col
                    } else {
                        self.screen.width.saturating_sub(1)
                    };

                    let mut line_text = String::new();
                    let mut current_col = 0;
                    while current_col <= line_end_col && current_col < self.screen.width {
                        if current_col >= line_start_col {
                            let glyph = &line[current_col];
                            if !glyph.attr.flags.contains(AttrFlags::WIDE_CHAR_SPACER) {
                                line_text.push(glyph.c);
                            }
                        }
                        // Check for wide char primary and advance by 2 if so, otherwise 1
                        if line[current_col].attr.flags.contains(AttrFlags::WIDE_CHAR_PRIMARY) && current_col + 1 < self.screen.width {
                            current_col += 2;
                        } else {
                            current_col += 1;
                        }
                    }
                    
                    // Trim trailing whitespace from lines that are not the last line of selection,
                    // or if the selection does not extend to the end of the line.
                    // This emulates typical terminal copy behavior.
                    if r_idx < end_row || end_col < self.screen.width.saturating_sub(1) {
                         selected_text.push_str(line_text.trim_end_matches(' '));
                    } else {
                        selected_text.push_str(&line_text);
                    }


                    if r_idx < end_row {
                        selected_text.push('\n');
                    }
                }
                return Some(selected_text);
            } else {
                // TODO: Implement Block selection text extraction
                warn!("Block selection mode text extraction not yet implemented.");
                return None;
            }
        }
        None
    }


    // --- Character Printing and Low-Level Operations ---

    /// Prints a single character to the terminal at the current cursor position.
    /// Handles character width, line wrapping, and updates cursor position.
    pub(super) fn print_char(&mut self, ch: char) {
        // Map character to the active G0/G1/G2/G3 character set.
        let ch_to_print = self.map_char_to_active_charset(ch);
        let char_width = get_char_display_width(ch_to_print);

        // Zero-width characters are complex. For now, if wcwidth reports 0, skip for cursor advancement.
        if char_width == 0 {
            trace!(
                "print_char: Encountered zero-width char '{}'. No cursor advancement.",
                ch_to_print
            );
            // TODO: Potentially handle combining characters by placing them in the current cell
            // without advancing, if the renderer and font support it.
            return;
        }

        let mut screen_ctx = self.current_screen_context();

        // Handle line wrap if cursor_wrap_next was set by the previous character.
        // This flag indicates that the cursor is at the end of the line and the next
        // character should wrap to the beginning of the next line.
        if self.cursor_wrap_next {
            self.carriage_return(); // Move to column 0 of the current line.
            self.move_down_one_line_and_dirty(); // Move to the next line, handles scrolling.
                                                 // move_down_one_line_and_dirty also resets self.cursor_wrap_next to false.
            screen_ctx = self.current_screen_context(); // Update context after potential scroll/cursor move.
                                                        // self.cursor_wrap_next is now false.
        }

        // Get current physical cursor position for placing the glyph.
        // This position is now correctly at the start of the line if a wrap just occurred.
        let (mut physical_x, mut physical_y) =
            self.cursor_controller.physical_screen_pos(&screen_ctx);

        // Check if the character (considering its width) would exceed the line width
        // from the current physical_x. This handles cases where the character is wider
        // than the remaining space on the line, even if cursor_wrap_next was false initially.
        if physical_x + char_width > screen_ctx.width {
            // If a wide char (width 2) is at the very last column (e.g. col 79 of 80), it can't fit.
            // Standard behavior: print a space in the last cell, then wrap.
            if char_width == 2 && physical_x == screen_ctx.width.saturating_sub(1) {
                let fill_glyph = Glyph {
                    c: ' ', // Fill with a space
                    attr: Attributes {
                        flags: AttrFlags::empty(),
                        ..self.cursor_controller.attributes()
                    }, // Ensure flags are clean for the space
                };
                if physical_y < self.screen.height {
                    // Bounds check
                    self.screen.set_glyph(physical_x, physical_y, fill_glyph);
                    self.screen.mark_line_dirty(physical_y);
                }
            }

            // Perform wrap: CR then effectively LF.
            self.carriage_return();
            self.move_down_one_line_and_dirty(); // This moves cursor down and handles scrolling.
                                                 // It also resets self.cursor_wrap_next.
            screen_ctx = self.current_screen_context(); // Update context
                                                        // Get new physical cursor position after this wrap.
            (physical_x, physical_y) = self.cursor_controller.physical_screen_pos(&screen_ctx);
        }

        // Place the character glyph on the screen.
        let glyph_attrs = self.cursor_controller.attributes();
        if physical_y < self.screen.height {
            // Ensure y is within bounds before writing
            let glyph_to_set = Glyph {
                c: ch_to_print,
                attr: glyph_attrs, // Use the full attributes from the cursor
            };

            self.screen.set_glyph(
                physical_x,
                physical_y,
                glyph_to_set.clone(), // Use clone if Glyph is not Copy
            );
            self.screen.mark_line_dirty(physical_y); // Mark line dirty via screen method.

            // If it's a wide character, place a placeholder and set flags.
            if char_width == 2 {
                // Mark the primary part of the wide character.
                let mut primary_glyph_attrs = glyph_attrs; // Start with cursor attributes
                primary_glyph_attrs
                    .flags
                    .insert(AttrFlags::WIDE_CHAR_PRIMARY);
                primary_glyph_attrs
                    .flags
                    .remove(AttrFlags::WIDE_CHAR_SPACER); // Ensure spacer flag is not present
                let primary_glyph = Glyph {
                    c: ch_to_print,
                    attr: primary_glyph_attrs,
                };
                self.screen.set_glyph(physical_x, physical_y, primary_glyph);

                if physical_x + 1 < screen_ctx.width {
                    let mut spacer_attrs = glyph_attrs; // Start with cursor attributes
                    spacer_attrs.flags.remove(AttrFlags::WIDE_CHAR_PRIMARY); // Ensure primary flag is not on spacer
                    spacer_attrs.flags.insert(AttrFlags::WIDE_CHAR_SPACER);
                    let placeholder_glyph = Glyph {
                        c: WIDE_CHAR_PLACEHOLDER, // Defined in glyph.rs
                        attr: spacer_attrs,
                    };
                    self.screen
                        .set_glyph(physical_x + 1, physical_y, placeholder_glyph);
                    // Line is already marked dirty from the primary character.
                } else {
                    // This case implies a wide char was printed at the exact last column.
                    // The WIDE_CHAR_PRIMARY flag is set, but no spacer is placed.
                    // The cursor advancement logic below will handle cursor_wrap_next.
                    trace!(
                        "Wide char placeholder for '{}' at ({},{}) could not be placed as it's at the edge of screen (width {}). Only primary part written.",
                        ch_to_print, physical_x, physical_y, screen_ctx.width
                    );
                }
            }
        } else {
            warn!(
                "print_char: Attempted to print at physical_y {} out of bounds (height {})",
                physical_y, self.screen.height
            );
        }

        // Advance the logical cursor position by the character's width.
        // self.cursor_controller.move_right uses the current logical position and advances it.
        // The logical position should be correct after any wrapping.
        self.cursor_controller.move_right(char_width, &screen_ctx);

        // Check if the new logical cursor position requires a wrap on the *next* character.
        let (final_logical_x, _) = self.cursor_controller.logical_pos();
        // Set cursor_wrap_next if the cursor is exactly at or beyond the width.
        // e.g., width 80 (cols 0-79). If final_logical_x is 80, it's at the wrap position.
        self.cursor_wrap_next = final_logical_x >= screen_ctx.width;
    }

    /// Maps a character to its equivalent in the currently active G0/G1/G2/G3 character set.
    fn map_char_to_active_charset(&self, ch: char) -> char {
        let current_set = self.active_charsets[self.active_charset_g_level];
        match current_set {
            CharacterSet::Ascii => ch,
            CharacterSet::UkNational => {
                if ch == '#' {
                    '£'
                } else {
                    ch
                }
            }
            CharacterSet::DecLineDrawing => map_to_dec_line_drawing(ch),
        }
    }

    /// Handles BS (Backspace).
    fn backspace(&mut self) {
        self.cursor_wrap_next = false;
        self.cursor_controller.move_left(1);
    }

    /// Handles HT (Horizontal Tab).
    fn horizontal_tab(&mut self) {
        self.cursor_wrap_next = false;
        let (current_x, _) = self.cursor_controller.logical_pos();
        let screen_ctx = self.current_screen_context();
        let next_stop = self
            .screen
            .get_next_tabstop(current_x)
            .unwrap_or(screen_ctx.width.saturating_sub(1).max(current_x));
        self.cursor_controller
            .move_to_logical_col(next_stop, &screen_ctx);
    }

    /// Performs a line feed operation.
    fn perform_line_feed(&mut self) {
        log::trace!("perform_line_feed called");
        // self.cursor_wrap_next should be managed by the caller or by move_down_one_line_and_dirty
        self.move_down_one_line_and_dirty();
        self.carriage_return(); // Always perform carriage return
    }

    /// Helper for LF, IND, NEL: moves cursor down one line, scrolling if necessary.
    fn move_down_one_line_and_dirty(&mut self) {
        self.cursor_wrap_next = false; // Critical: any vertical movement resets pending wrap.
        let screen_ctx = self.current_screen_context();
        let (_, current_logical_y) = self.cursor_controller.logical_pos();
        let (_current_physical_x, current_physical_y) =
            self.cursor_controller.physical_screen_pos(&screen_ctx);

        let max_logical_y_in_region = if screen_ctx.origin_mode_active {
            screen_ctx.scroll_bot.saturating_sub(screen_ctx.scroll_top)
        } else {
            screen_ctx.height.saturating_sub(1)
        };

        let physical_effective_bottom = if screen_ctx.origin_mode_active {
            screen_ctx.scroll_bot
        } else {
            screen_ctx.height.saturating_sub(1)
        };

        if current_physical_y == physical_effective_bottom {
            log::trace!(
                "move_down_one_line: Scrolling up. Cursor at physical_y: {}, effective_bottom: {}",
                current_physical_y,
                physical_effective_bottom
            );
            self.screen.scroll_up_serial(1);
        } else if current_logical_y < max_logical_y_in_region {
            log::trace!(
                "move_down_one_line: Moving cursor down. logical_y: {}, max_logical_y_in_region: {}",
                current_logical_y,
                max_logical_y_in_region
            );
            self.cursor_controller.move_down(1, &screen_ctx);
        } else if !screen_ctx.origin_mode_active
            && current_physical_y < screen_ctx.height.saturating_sub(1)
        {
            log::trace!(
                "move_down_one_line: Moving cursor down (below scroll region, origin mode off). physical_y: {}, screen_height: {}",
                current_physical_y,
                screen_ctx.height
            );
            self.cursor_controller.move_down(1, &screen_ctx);
        } else {
            log::trace!(
                "move_down_one_line: Cursor at bottom, no scroll or move_down. physical_y: {}, logical_y: {}, max_logical_y: {}",
                current_physical_y,
                current_logical_y,
                max_logical_y_in_region
            );
        }

        log::trace!(
            "move_down_one_line: Marking old line dirty. current_physical_y: {}, screen_height: {}",
            current_physical_y,
            self.screen.height
        );
        if current_physical_y < self.screen.height {
            // Bounds check before marking dirty
            self.screen.mark_line_dirty(current_physical_y);
        }

        let (_, new_physical_y) = self
            .cursor_controller
            .physical_screen_pos(&self.current_screen_context()); // Re-fetch context as it might have changed

        if current_physical_y != new_physical_y {
            log::trace!(
                "move_down_one_line: Marking new line dirty. new_physical_y: {}, screen_height: {}",
                new_physical_y,
                self.screen.height
            );
            if new_physical_y < self.screen.height {
                // Bounds check
                self.screen.mark_line_dirty(new_physical_y);
            }
        } else {
            log::trace!(
                "move_down_one_line: New physical y ({}) is same as current ({}), not marking new line again.",
                new_physical_y,
                current_physical_y
            );
        }
    }

    /// Handles CR (Carriage Return).
    fn carriage_return(&mut self) {
        self.cursor_wrap_next = false;
        self.cursor_controller.carriage_return();
    }

    /// Sets the active G-level.
    fn set_g_level(&mut self, g_level: usize) {
        if g_level < self.active_charsets.len() {
            self.active_charset_g_level = g_level;
            trace!("Switched to G{} character set mapping.", g_level);
        } else {
            warn!("Attempted to set invalid G-level: {}", g_level);
        }
    }

    /// Designates a character set.
    fn designate_character_set(&mut self, g_set_index: usize, charset: CharacterSet) {
        if g_set_index < self.active_charsets.len() {
            self.active_charsets[g_set_index] = charset;
            trace!("Designated G{} to {:?}", g_set_index, charset);
        } else {
            warn!("Invalid G-set index for designate charset: {}", g_set_index);
        }
    }

    /// Handles IND (Index).
    fn index(&mut self) {
        self.cursor_wrap_next = false;
        let screen_ctx = self.current_screen_context();
        let (_, current_physical_y) = self.cursor_controller.physical_screen_pos(&screen_ctx);

        if current_physical_y == screen_ctx.scroll_bot {
            self.screen.scroll_up_serial(1);
        } else if current_physical_y < screen_ctx.height.saturating_sub(1) {
            self.cursor_controller.move_down(1, &screen_ctx);
        }
        if current_physical_y < self.screen.height {
            self.screen.mark_line_dirty(current_physical_y);
        }
        let (_, new_physical_y) = self
            .cursor_controller
            .physical_screen_pos(&self.current_screen_context());
        if current_physical_y != new_physical_y && new_physical_y < self.screen.height {
            self.screen.mark_line_dirty(new_physical_y);
        }
    }

    /// Handles RI (Reverse Index).
    fn reverse_index(&mut self) {
        self.cursor_wrap_next = false;
        let screen_ctx = self.current_screen_context();
        let (_, current_physical_y) = self.cursor_controller.physical_screen_pos(&screen_ctx);

        if current_physical_y == screen_ctx.scroll_top {
            self.screen.scroll_down_serial(1);
        } else if current_physical_y > 0 {
            self.cursor_controller.move_up(1);
        }
        if current_physical_y < self.screen.height {
            self.screen.mark_line_dirty(current_physical_y);
        }
        let (_, new_physical_y) = self
            .cursor_controller
            .physical_screen_pos(&self.current_screen_context());
        if current_physical_y != new_physical_y && new_physical_y < self.screen.height {
            self.screen.mark_line_dirty(new_physical_y);
        }
    }

    /// Saves cursor state (DECSC).
    fn save_cursor_dec(&mut self) {
        self.cursor_controller.save_state();
    }

    /// Restores cursor state (DECRC).
    fn restore_cursor_dec(&mut self) {
        self.cursor_wrap_next = false;
        self.cursor_controller
            .restore_state(&self.current_screen_context(), Attributes::default());
        self.screen.default_attributes = self.cursor_controller.attributes();
    }

    // --- CSI Handler Implementations ---
    fn cursor_up(&mut self, n: usize) {
        self.cursor_wrap_next = false;
        self.cursor_controller.move_up(n);
    }

    fn cursor_down(&mut self, n: usize) {
        self.cursor_wrap_next = false;
        log::trace!("cursor_down: n = {}", n);
        for i in 0..n {
            log::trace!(
                "cursor_down: iteration {} calling index (was perform_index)",
                i
            ); // Changed comment
            self.index(); // index handles scrolling and dirtying
        }
    }

    fn cursor_forward(&mut self, n: usize) {
        self.cursor_wrap_next = false;
        self.cursor_controller
            .move_right(n, &self.current_screen_context());
    }

    fn cursor_backward(&mut self, n: usize) {
        self.cursor_wrap_next = false;
        self.cursor_controller.move_left(n);
    }

    fn cursor_to_column(&mut self, col: usize) {
        self.cursor_wrap_next = false;
        self.cursor_controller
            .move_to_logical_col(col, &self.current_screen_context());
    }

    fn cursor_to_pos(&mut self, row_param: usize, col_param: usize) {
        self.cursor_wrap_next = false;
        self.cursor_controller.move_to_logical(
            col_param,
            row_param,
            &self.current_screen_context(),
        );
    }

    fn erase_in_display(&mut self, mode: EraseMode) {
        self.cursor_wrap_next = false;
        let screen_ctx = self.current_screen_context();
        let (cx_phys, cy_phys) = self.cursor_controller.physical_screen_pos(&screen_ctx);
        self.screen.default_attributes = self.cursor_controller.attributes();

        match mode {
            EraseMode::ToEnd => {
                self.screen
                    .clear_line_segment(cy_phys, cx_phys, screen_ctx.width);
                for y in (cy_phys + 1)..screen_ctx.height {
                    self.screen.clear_line_segment(y, 0, screen_ctx.width);
                }
            }
            EraseMode::ToStart => {
                for y in 0..cy_phys {
                    self.screen.clear_line_segment(y, 0, screen_ctx.width);
                }
                self.screen.clear_line_segment(cy_phys, 0, cx_phys + 1);
            }
            EraseMode::All => {
                for y in 0..screen_ctx.height {
                    self.screen.clear_line_segment(y, 0, screen_ctx.width);
                }
            }
            EraseMode::Scrollback => {
                self.screen.scrollback.clear();
                return;
            }
            EraseMode::Unknown => warn!("Unknown ED mode used."),
        }
        if mode != EraseMode::Scrollback {
            self.screen.mark_all_dirty();
        }
    }

    fn erase_in_line(&mut self, mode: EraseMode) {
        self.cursor_wrap_next = false;
        let screen_ctx = self.current_screen_context();
        let (cx_phys, cy_phys) = self.cursor_controller.physical_screen_pos(&screen_ctx);
        self.screen.default_attributes = self.cursor_controller.attributes();

        match mode {
            EraseMode::ToEnd => self
                .screen
                .clear_line_segment(cy_phys, cx_phys, screen_ctx.width),
            EraseMode::ToStart => self.screen.clear_line_segment(cy_phys, 0, cx_phys + 1),
            EraseMode::All => self.screen.clear_line_segment(cy_phys, 0, screen_ctx.width),
            EraseMode::Scrollback => {
                warn!("EraseMode::Scrollback is not applicable to EraseInLine (EL).")
            }
            EraseMode::Unknown => warn!("Unknown EL mode used."),
        }
    }

    fn erase_chars(&mut self, n: usize) {
        self.cursor_wrap_next = false;
        let screen_ctx = self.current_screen_context();
        let (cx_phys, cy_phys) = self.cursor_controller.physical_screen_pos(&screen_ctx);
        let end_x = min(cx_phys + n, screen_ctx.width);
        self.screen.default_attributes = self.cursor_controller.attributes();
        self.screen.clear_line_segment(cy_phys, cx_phys, end_x);
    }

    fn insert_blank_chars(&mut self, n: usize) {
        self.cursor_wrap_next = false;
        let screen_ctx = self.current_screen_context();
        let (cx_log, _) = self.cursor_controller.logical_pos();
        let (_, cy_phys) = self.cursor_controller.physical_screen_pos(&screen_ctx);
        self.screen.default_attributes = self.cursor_controller.attributes();
        self.screen.insert_blank_chars_in_line(cy_phys, cx_log, n);
    }

    fn delete_chars(&mut self, n: usize) {
        self.cursor_wrap_next = false;
        let screen_ctx = self.current_screen_context();
        let (cx_log, _) = self.cursor_controller.logical_pos();
        let (_, cy_phys) = self.cursor_controller.physical_screen_pos(&screen_ctx);
        self.screen.default_attributes = self.cursor_controller.attributes();
        self.screen.delete_chars_in_line(cy_phys, cx_log, n);
    }

    fn insert_lines(&mut self, n: usize) {
        self.cursor_wrap_next = false;
        let screen_ctx = self.current_screen_context();
        let (_, cy_phys) = self.cursor_controller.physical_screen_pos(&screen_ctx);
        self.screen.default_attributes = self.cursor_controller.attributes();

        if cy_phys >= screen_ctx.scroll_top && cy_phys <= screen_ctx.scroll_bot {
            let original_scroll_top = self.screen.scroll_top();
            let original_scroll_bottom = self.screen.scroll_bot();

            self.screen
                .set_scrolling_region(cy_phys + 1, original_scroll_bottom + 1);
            self.screen.scroll_down_serial(n);

            self.screen
                .set_scrolling_region(original_scroll_top + 1, original_scroll_bottom + 1);

            for y_dirty in cy_phys..=original_scroll_bottom {
                if y_dirty < self.screen.height {
                    self.screen.mark_line_dirty(y_dirty);
                }
            }
        }
    }

    fn delete_lines(&mut self, n: usize) {
        self.cursor_wrap_next = false;
        let screen_ctx = self.current_screen_context();
        let (_, cy_phys) = self.cursor_controller.physical_screen_pos(&screen_ctx);
        self.screen.default_attributes = self.cursor_controller.attributes();

        if cy_phys >= screen_ctx.scroll_top && cy_phys <= screen_ctx.scroll_bot {
            let original_scroll_top = self.screen.scroll_top();
            let original_scroll_bottom = self.screen.scroll_bot();

            self.screen
                .set_scrolling_region(cy_phys + 1, original_scroll_bottom + 1);
            self.screen.scroll_up_serial(n);

            self.screen
                .set_scrolling_region(original_scroll_top + 1, original_scroll_bottom + 1);
            for y_dirty in cy_phys..=original_scroll_bottom {
                if y_dirty < self.screen.height {
                    self.screen.mark_line_dirty(y_dirty);
                }
            }
        }
    }

    fn scroll_up(&mut self, n: usize) {
        self.cursor_wrap_next = false;
        self.screen.default_attributes = self.cursor_controller.attributes();
        self.screen.scroll_up_serial(n);
    }

    fn scroll_down(&mut self, n: usize) {
        self.cursor_wrap_next = false;
        self.screen.default_attributes = self.cursor_controller.attributes();
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
                Attribute::BlinkSlow | Attribute::BlinkRapid => {
                    current_attrs.flags.insert(AttrFlags::BLINK)
                }
                Attribute::Reverse => current_attrs.flags.insert(AttrFlags::REVERSE),
                Attribute::Conceal => current_attrs.flags.insert(AttrFlags::HIDDEN),
                Attribute::Strikethrough => current_attrs.flags.insert(AttrFlags::STRIKETHROUGH),
                Attribute::UnderlineDouble => current_attrs.flags.insert(AttrFlags::UNDERLINE),
                Attribute::NoBold => {
                    current_attrs.flags.remove(AttrFlags::BOLD);
                    current_attrs.flags.remove(AttrFlags::FAINT);
                }
                Attribute::NoItalic => current_attrs.flags.remove(AttrFlags::ITALIC),
                Attribute::NoUnderline => current_attrs.flags.remove(AttrFlags::UNDERLINE),
                Attribute::NoBlink => current_attrs.flags.remove(AttrFlags::BLINK),
                Attribute::NoReverse => current_attrs.flags.remove(AttrFlags::REVERSE),
                Attribute::NoConceal => current_attrs.flags.remove(AttrFlags::HIDDEN),
                Attribute::NoStrikethrough => current_attrs.flags.remove(AttrFlags::STRIKETHROUGH),
                Attribute::Foreground(color) => {
                    current_attrs.fg = color;
                }
                Attribute::Background(color) => {
                    current_attrs.bg = color;
                }
                Attribute::Overlined => warn!("SGR Overlined not yet visually supported."),
                Attribute::NoOverlined => warn!("SGR NoOverlined not yet visually supported."),
                Attribute::UnderlineColor(color) => {
                    warn!("SGR UnderlineColor not yet fully supported: {:?}", color)
                }
            }
        }
        self.cursor_controller.set_attributes(current_attrs);
        self.screen.default_attributes = current_attrs;
    }

    fn handle_set_mode(&mut self, mode_type: Mode, enable: bool) -> Option<EmulatorAction> {
        self.cursor_wrap_next = false;
        let mut action_to_return = None;

        match mode_type {
            Mode::DecPrivate(mode_num) => {
                trace!("Setting DEC Private Mode {} to {}", mode_num, enable);
                match DecModeConstant::from_u16(mode_num) {
                    Some(DecModeConstant::CursorKeys) => {
                        self.dec_modes.cursor_keys_app_mode = enable
                    }
                    Some(DecModeConstant::Origin) => {
                        self.dec_modes.origin_mode = enable;
                        self.screen.origin_mode = enable;
                        self.cursor_controller.move_to_logical(
                            0,
                            0,
                            &self.current_screen_context(),
                        );
                    }
                    Some(DecModeConstant::TextCursorEnable) => {
                        self.dec_modes.text_cursor_enable_mode = enable;
                        self.cursor_controller.set_visible(enable);
                        action_to_return = Some(EmulatorAction::SetCursorVisibility(enable));
                    }
                    Some(DecModeConstant::AltScreenBufferClear)
                    | Some(DecModeConstant::AltScreenBufferSaveRestore) => {
                        if !self.dec_modes.allow_alt_screen {
                            // Assuming this flag exists in DecPrivateModes
                            warn!(
                                "Alternate screen disabled by configuration, ignoring mode {}.",
                                mode_num
                            );
                            return None;
                        }
                        let clear_on_entry = mode_num
                            == DecModeConstant::AltScreenBufferClear as u16
                            || mode_num == DecModeConstant::AltScreenBufferSaveRestore as u16;

                        if enable {
                            if !self.dec_modes.using_alt_screen {
                                // Assuming this flag exists
                                if mode_num == DecModeConstant::AltScreenBufferSaveRestore as u16 {
                                    self.save_cursor_dec();
                                }
                                self.screen.default_attributes =
                                    self.cursor_controller.attributes();
                                self.screen.enter_alt_screen(clear_on_entry);
                                self.dec_modes.using_alt_screen = true;
                                self.cursor_controller.move_to_logical(
                                    0,
                                    0,
                                    &self.current_screen_context(),
                                );
                                self.screen.mark_all_dirty(); // Changed from self.mark_all_lines_dirty()
                                action_to_return = Some(EmulatorAction::RequestRedraw);
                            }
                        } else if self.dec_modes.using_alt_screen {
                            self.screen.exit_alt_screen();
                            self.dec_modes.using_alt_screen = false;
                            if mode_num == DecModeConstant::AltScreenBufferSaveRestore as u16 {
                                self.restore_cursor_dec();
                            } else {
                                self.cursor_controller.move_to_logical(
                                    0,
                                    0,
                                    &self.current_screen_context(),
                                );
                                self.screen.default_attributes =
                                    self.cursor_controller.attributes();
                            }
                            self.screen.mark_all_dirty(); // Changed from self.mark_all_lines_dirty()
                            action_to_return = Some(EmulatorAction::RequestRedraw);
                        }
                    }
                    Some(DecModeConstant::SaveRestoreCursor) => {
                        if enable {
                            self.save_cursor_dec();
                        } else {
                            self.restore_cursor_dec();
                        }
                    }
                    Some(DecModeConstant::BracketedPaste) => {
                        self.dec_modes.bracketed_paste_mode = enable
                    }
                    Some(DecModeConstant::FocusEvent) => self.dec_modes.focus_event_mode = enable,
                    Some(DecModeConstant::MouseX10) => self.dec_modes.mouse_x10_mode = enable,
                    Some(DecModeConstant::MouseVt200) => self.dec_modes.mouse_vt200_mode = enable,
                    Some(DecModeConstant::MouseVt200Highlight) => {
                        self.dec_modes.mouse_vt200_highlight_mode = enable
                    }
                    Some(DecModeConstant::MouseButtonEvent) => {
                        self.dec_modes.mouse_button_event_mode = enable
                    }
                    Some(DecModeConstant::MouseAnyEvent) => {
                        self.dec_modes.mouse_any_event_mode = enable
                    }
                    Some(DecModeConstant::MouseUtf8) => self.dec_modes.mouse_utf8_mode = enable,
                    Some(DecModeConstant::MouseSgr) => self.dec_modes.mouse_sgr_mode = enable,
                    Some(DecModeConstant::MouseUrxvt) => {
                        warn!(
                            "DEC Private Mode {} (MouseUrxvt) set to {} - not fully implemented.",
                            mode_num, enable
                        );
                    }
                    Some(DecModeConstant::MousePixelPosition) => {
                        warn!(
                            "DEC Private Mode {} (MousePixelPosition) set to {} - not fully implemented.",
                            mode_num, enable
                        );
                    }
                    Some(DecModeConstant::Att610CursorBlink) => {
                        self.dec_modes.cursor_blink_mode = enable; // Assuming this flag exists
                        warn!(
                            "DEC Private Mode 12 (ATT610 Cursor Blink) set to {}. Visual blink not implemented.",
                            enable
                        );
                    }
                    Some(DecModeConstant::Unknown7727) => {
                        warn!(
                            "DEC Private Mode 7727 set to {} - behavior undefined.",
                            enable
                        );
                    }
                    None => {
                        warn!(
                            "Unknown DEC private mode {} to set/reset: {}",
                            mode_num, enable
                        );
                    }
                }
            }
            Mode::Standard(mode_num) => match mode_num {
                4 => {
                    self.dec_modes.insert_mode = enable;
                }
                20 => {
                    self.dec_modes.linefeed_newline_mode = enable;
                }
                _ => {
                    warn!(
                        "Standard mode {} set/reset to {} - not fully implemented yet.",
                        mode_num, enable
                    );
                }
            },
        }
        action_to_return
    }

    fn handle_window_manipulation(
        &mut self,
        ps1: u16,
        _ps2: Option<u16>,
        _ps3: Option<u16>,
    ) -> Option<EmulatorAction> {
        match ps1 {
            14 => {
                warn!(
                    "WindowManipulation: Report text area size in pixels (14) requested, but not implemented."
                );
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
                warn!(
                    "Unhandled WindowManipulation: ps1={}, ps2={:?}, ps3={:?}",
                    ps1, _ps2, _ps3
                );
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
                0 | 2 => {
                    return Some(EmulatorAction::SetTitle(pt));
                }
                _ => debug!("Unhandled OSC command: Ps={}, Pt='{}'", ps, pt),
            }
        } else {
            warn!(
                "Malformed OSC sequence (expected Ptext;Pstring): {}",
                osc_str
            );
        }
        None
    }
}


// --- Implement TerminalInterface for TerminalEmulator ---
// This was previously in src/term/mod.rs, moved here as it's specific to TerminalEmulator.
impl TerminalInterface for TerminalEmulator {
    fn dimensions(&self) -> (usize, usize) {
        (self.screen.width, self.screen.height)
    }

    /// Interprets an `EmulatorInput` and updates the terminal state.
    fn interpret_input(&mut self, input: EmulatorInput) -> Option<EmulatorAction> {
        let mut action = match input {
            EmulatorInput::Ansi(command) => self.handle_ansi_command(command),
            EmulatorInput::Control(event) => self.handle_control_event(event),
            EmulatorInput::User(user_input_action) => { // Changed from BackendEvent
                self.handle_user_input_action(user_input_action)
            }
            EmulatorInput::RawChar(ch) => {
                self.print_char(ch);
                None
            }
        };

        if action.is_none() && !self.screen.dirty.iter().all(|&d| d == 0) {
            if !matches!(action, Some(EmulatorAction::RequestRedraw)) {
                action = Some(EmulatorAction::RequestRedraw);
            }
        }
        action
    }

    fn get_glyph(&self, x: usize, y: usize) -> Glyph {
        self.screen.get_glyph(x, y)
    }

    fn is_cursor_visible(&self) -> bool {
        self.dec_modes.text_cursor_enable_mode
    }

    fn get_screen_cursor_pos(&self) -> (usize, usize) {
        self.cursor_controller
            .physical_screen_pos(&self.current_screen_context())
    }
    
    fn get_render_snapshot(&self) -> RenderSnapshot {
        let (width, height) = self.dimensions();
        let mut lines = Vec::with_capacity(height);

        for r_idx in 0..height {
            let screen_line = &self.screen.grid[r_idx];
            // Simple dirty check for now: if the screen's dirty flag for this line is set.
            // More sophisticated checks could involve comparing with a previous snapshot.
            let is_line_dirty = self.screen.dirty[r_idx] != 0;
            
            lines.push(SnapshotLine {
                cells: screen_line.clone(), // Clone the glyphs for the snapshot
                is_dirty: is_line_dirty,
            });
        }

        let cursor_state = if self.is_cursor_visible() {
            let (cursor_col, cursor_row) = self.get_screen_cursor_pos();
            Some(CursorRenderState {
                col: cursor_col,
                row: cursor_row,
                shape: self.current_cursor_shape, // From config or internal state
                is_visible: true, // Already checked by self.is_cursor_visible()
            })
        } else {
            None
        };

        let selection_render_state = self.selection.map(|sel_state| {
            // Normalize coordinates for rendering if necessary, though renderer might also do this.
            // For now, directly pass what's stored.
            SelectionRenderState {
                start_coords: sel_state.start,
                end_coords: sel_state.end,
                mode: sel_state.mode,
            }
        });

        RenderSnapshot {
            dimensions: (width, height),
            lines,
            cursor_state,
            selection_state: selection_render_state,
        }
    }


    fn take_dirty_lines(&mut self) -> Vec<usize> {
        let mut all_dirty_indices: std::collections::HashSet<usize> =
            std::collections::HashSet::new();

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
}
