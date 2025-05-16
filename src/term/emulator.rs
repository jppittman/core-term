// Re-export items for easier use by other modules and within this module
use action::EmulatorAction;
use charset::{CharacterSet, map_to_dec_line_drawing};
use super::modes::{DecModeConstant, DecPrivateModes, EraseMode, Mode};
use super::*;

// Standard library imports
use std::cmp::min;

// Crate-level imports (adjust paths based on where items are moved)
use crate::{
    ansi::commands::{
        AnsiCommand,
        Attribute,
        C0Control,
        CsiCommand,
        EscCommand, // Added EscCommand
    },
    backends::BackendEvent,
    glyph::{AttrFlags, Attributes},
    term::unicode::get_char_display_width,
};

// Logging
use log::{debug, trace, warn};

/// The core terminal emulator.
pub struct TerminalEmulator {
    pub(super) screen: Screen,
    pub(super) cursor_controller: CursorController,
    pub(super) dec_modes: DecPrivateModes,
    pub(super) active_charsets: [CharacterSet; 4],
    pub(super) active_charset_g_level: usize,
    pub(super) dirty_lines: Vec<usize>,
    pub(super) cursor_wrap_next: bool,
    pub(super) current_cursor_shape: u16,
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
                CharacterSet::Ascii,
                CharacterSet::Ascii,
                CharacterSet::Ascii,
                CharacterSet::Ascii,
            ],
            active_charset_g_level: 0,
            dirty_lines: (0..height.max(1)).collect(),
            cursor_wrap_next: false,
            current_cursor_shape: DEFAULT_CURSOR_SHAPE,
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
    pub(super) fn is_origin_mode_active(&self) -> bool {
        self.dec_modes.origin_mode
    }
    pub(super) fn is_cursor_keys_app_mode_active(&self) -> bool {
        self.dec_modes.cursor_keys_app_mode
    }
    pub(super) fn is_bracketed_paste_mode_active(&self) -> bool {
        self.dec_modes.bracketed_paste_mode
    }
    pub(super) fn is_focus_event_mode_active(&self) -> bool {
        self.dec_modes.focus_event_mode
    }

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
    pub(super) fn get_cursor_shape(&self) -> u16 {
        self.current_cursor_shape
    }

    /// Handles a parsed `AnsiCommand`.
    pub(super) fn handle_ansi_command(&mut self, command: AnsiCommand) -> Option<EmulatorAction> {
        if !matches!(command, AnsiCommand::Print(_)) {
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
                    self.perform_line_feed();
                    self.carriage_return();
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
                            0 // Default to G0 if intermediate is unknown
                        }
                    };
                    self.designate_character_set(g_idx, CharacterSet::from_char(final_char));
                    None
                }
                EscCommand::ResetToInitialState => {
                    trace!("Processing ESC c (ResetToInitialState)");

                    // 1. Ensure on primary screen
                    if self.screen.alt_screen_active {
                        self.screen.exit_alt_screen(); // This marks lines dirty
                        // dec_modes.using_alt_screen will be reset by DecPrivateModes::default() later
                    }

                    // 2. Reset attributes for clearing and cursor
                    let default_attrs = Attributes::default();
                    self.cursor_controller.set_attributes(default_attrs);
                    self.screen.default_attributes = default_attrs;

                    // 3. Clear screen content (uses self.screen.default_attributes)
                    self.erase_in_display(EraseMode::All); // This marks all lines dirty

                    // 4. Home cursor (attributes are already default)
                    self.cursor_controller
                        .move_to_logical(0, 0, &self.current_screen_context());

                    // 5. Reset DEC Private Modes (also resets origin_mode, cursor visibility via DECTCEM default)
                    self.dec_modes = DecPrivateModes::default();
                    self.screen.origin_mode = self.dec_modes.origin_mode; // Sync screen's copy

                    // 6. Reset scrolling region to full screen
                    let (_, h) = self.dimensions(); // Get current height
                    self.screen.set_scrolling_region(1, h); // 1-based params

                    // 7. Reset character sets
                    self.active_charsets = [CharacterSet::Ascii; 4];
                    self.active_charset_g_level = 0;

                    // 8. Reset tab stops
                    self.screen.clear_tabstops(0, screen::TabClearMode::All);
                    let (w, _) = self.dimensions();
                    for i in
                        (DEFAULT_TAB_INTERVAL as usize..w).step_by(DEFAULT_TAB_INTERVAL as usize)
                    {
                        self.screen.set_tabstop(i);
                    }

                    // 9. Reset miscellaneous state
                    self.cursor_wrap_next = false;
                    self.current_cursor_shape = DEFAULT_CURSOR_SHAPE;
                    self.cursor_controller.set_visible(true); // RIS makes cursor visible

                    // 10. Mark all lines dirty (already done by erase_in_display and exit_alt_screen)
                    // self.mark_all_lines_dirty(); // Can be called again, it's idempotent.

                    // 11. Signal actions
                    // RequestRedraw is implicit. SetCursorVisibility is important.
                    // SetTitle could also be added here.
                    // For now, only SetCursorVisibility to match st's behavior of making cursor visible.
                    Some(EmulatorAction::SetCursorVisibility(true))
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
                        // DSR 6 is cursor position report
                        let screen_ctx = self.current_screen_context();
                        let (abs_x, abs_y) =
                            self.cursor_controller.physical_screen_pos(&screen_ctx);
                        let response = format!("\x1B[{};{}R", abs_y + 1, abs_x + 1);
                        Some(EmulatorAction::WritePty(response.into_bytes()))
                    } else if dsr_param == 5 {
                        // DSR 5 is status report (OK)
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
                    // This is DECSC, typically. ANSI s/u are less common.
                    self.save_cursor_dec();
                    None
                }
                CsiCommand::RestoreCursor => {
                    // This is DECRC.
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
                    // After setting scrolling region, cursor moves to home within that region if DECOM is on,
                    // or to absolute (0,0) if DECOM is off.
                    // st.c moves to (0,0) of the new margin (if DECOM) or physical (0,0).
                    // Our CursorController.move_to_logical already handles DECOM.
                    self.cursor_controller
                        .move_to_logical(0, 0, &self.current_screen_context());
                    None
                }
                CsiCommand::SetCursorStyle { shape } => {
                    self.current_cursor_shape = shape;
                    debug!("Set cursor style to shape: {}", shape);
                    // Note: This might need an EmulatorAction if the driver needs to change cursor visuals.
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
                // Catch-all for CsiCommand variants not explicitly handled above
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
            // Catch-all for AnsiCommand variants not explicitly handled
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
        // Key constants for common keys. Values might need adjustment based on X11 KeySyms or other backend specifics.
        const KEY_RETURN: u32 = 0xFF0D; // Enter/Return
        const KEY_BACKSPACE: u32 = 0xFF08; // Backspace
        const KEY_TAB: u32 = 0xFF09; // Tab
        const KEY_ISO_LEFT_TAB: u32 = 0xFE20; // Shift+Tab often produces this
        const KEY_ESCAPE: u32 = 0xFF1B; // Escape

        // Arrow keys (common X11 KeySym values)
        const KEY_LEFT_ARROW: u32 = 0xFF51;
        const KEY_UP_ARROW: u32 = 0xFF52;
        const KEY_RIGHT_ARROW: u32 = 0xFF53;
        const KEY_DOWN_ARROW: u32 = 0xFF54;

        // TODO: Add more key constants (Home, End, PgUp, PgDn, F-keys, etc.)

        match event {
            BackendEvent::Key { keysym, text } => {
                let mut bytes_to_send: Vec<u8> = Vec::new();
                // Prioritize keysym for control/special keys, then text for printable chars.
                match keysym {
                    KEY_RETURN => bytes_to_send.push(b'\r'), // Or b'\n' depending on desired behavior / LNM mode
                    KEY_BACKSPACE => bytes_to_send.push(0x08), // BS
                    KEY_TAB => bytes_to_send.push(b'\t'),    // HT
                    KEY_ISO_LEFT_TAB => bytes_to_send.extend_from_slice(b"\x1b[Z"), // CSI Z (Shift Tab)
                    KEY_ESCAPE => bytes_to_send.push(0x1B),                         // ESC

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
                    // Add more special keys here (Home, End, F-keys, etc.)
                    _ => {
                        // Fallback to text if keysym is not a special handled one
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
            // Orchestrator handles Resize, CloseRequested, FocusGained, FocusLost directly.
            // They are not typically fed back into interpret_input as EmulatorInput::User.
            // If they were, the match would need to handle them here.
            _ => debug!(
                "BackendEvent {:?} passed to TerminalEmulator's handle_backend_event, usually handled by Orchestrator.",
                event
            ),
        }
        None
    }

    /// Resizes the terminal display grid.
    pub(super) fn resize(&mut self, cols: usize, rows: usize) {
        self.cursor_wrap_next = false;
        let current_scrollback_limit = self.screen.scrollback_limit();
        self.screen.resize(cols, rows, current_scrollback_limit);
        let (log_x, log_y) = self.cursor_controller.logical_pos();
        self.cursor_controller
            .move_to_logical(log_x, log_y, &self.current_screen_context());
        self.mark_all_lines_dirty();
        debug!(
            "Terminal resized to {}x{}. Cursor re-clamped. All lines marked dirty.",
            cols, rows
        );
    }

    fn mark_all_lines_dirty(&mut self) {
        self.dirty_lines = (0..self.screen.height).collect(); // Legacy, for initial full draw
        self.screen.mark_all_dirty(); // Primary mechanism
    }

    #[allow(dead_code)] // Potentially useful later, but screen.mark_line_dirty is primary
    fn mark_line_dirty(&mut self, y_abs: usize) {
        if y_abs < self.screen.height && !self.dirty_lines.contains(&y_abs) {
            // self.dirty_lines.push(y_abs);
        }
        self.screen.mark_line_dirty(y_abs);
    }

    pub fn cursor_pos(&self) -> (usize, usize) {
        self.cursor_controller.logical_pos()
    }

    pub fn is_alt_screen_active(&self) -> bool {
        self.screen.alt_screen_active
    }

    pub(super) fn print_char(&mut self, ch: char) {
        let ch_to_print = self.map_char_to_active_charset(ch);
        let char_width = get_char_display_width(ch_to_print);

        if char_width == 0 {
            // True zero-width character (e.g. combining mark not precomposed)
            // Behavior for standalone ZWCs is complex.
            // Option 1: Overstrike the previous character (difficult to manage generically).
            // Option 2: Advance cursor by 0, effectively ignoring it for layout if not combined.
            // Option 3: If at line start, or after space, might be treated as width 1 (like some terminals).
            // For now, if it's a ZWC and we don't have a preceding char to combine with,
            // we might just ignore it or let it overstrike (which means not advancing cursor).
            // The current `get_char_display_width` should handle this.
            // If it returns 0, we simply don't advance the cursor below.
            // The character might still be stored in the grid if needed for rendering complex scripts.
            // For simplicity, if width is 0, we might just return.
            trace!(
                "print_char: Encountered zero-width char '{}'. Behavior might be minimal.",
                ch_to_print
            );
            // If we want to place it without advancing:
            // let (physical_x, physical_y) = self.cursor_controller.physical_screen_pos(&self.current_screen_context());
            // self.screen.set_glyph(physical_x, physical_y, Glyph { c: ch_to_print, attr: self.cursor_controller.attributes() });
            return; // Or handle by overstriking/combining if supported
        }

        let mut screen_ctx = self.current_screen_context();

        if self.cursor_wrap_next {
            self.perform_line_feed(); // This also does carriage_return
            screen_ctx = self.current_screen_context(); // Update context after newline
            self.cursor_wrap_next = false;
        }

        let (mut physical_x, mut physical_y) =
            self.cursor_controller.physical_screen_pos(&screen_ctx);

        // Line wrapping for wide characters needs careful handling.
        // If a wide char (width 2) is at the second to last column, it should print.
        // If it's at the last column, it should wrap then print.
        if physical_x + char_width > screen_ctx.width {
            // If a single char (even narrow) doesn't fit, wrap.
            // Or if a wide char would be split.
            if char_width == 2 && physical_x == screen_ctx.width.saturating_sub(1) {
                // Special case: Wide char at the very last column.
                // Some terminals might place a space, then wrap. Others just wrap.
                // st.c behavior: if wide char at last col, it wraps.
                // Let's fill the last cell with a space (using current attributes) before wrapping.
                let fill_glyph = Glyph {
                    c: ' ',
                    attr: self.cursor_controller.attributes(),
                };
                if physical_y < self.screen.height {
                    // Check bounds before setting glyph
                    self.screen.set_glyph(physical_x, physical_y, fill_glyph);
                }
            }
            self.perform_line_feed(); // Wrap to next line, cursor to col 0
            screen_ctx = self.current_screen_context(); // Update context
            (physical_x, physical_y) = self.cursor_controller.physical_screen_pos(&screen_ctx);
        }

        let glyph_attrs = self.cursor_controller.attributes();
        if physical_y < self.screen.height {
            // Ensure physical_y is within bounds
            self.screen.set_glyph(
                physical_x,
                physical_y,
                Glyph {
                    c: ch_to_print,
                    attr: glyph_attrs,
                },
            );

            if char_width == 2 {
                // Place placeholder for the second half of the wide character
                if physical_x + 1 < screen_ctx.width {
                    self.screen.set_glyph(
                        physical_x + 1,
                        physical_y,
                        Glyph {
                            c: '\0',           // Placeholder character
                            attr: glyph_attrs, // Inherit attributes
                        },
                    );
                } else {
                    // This case (wide char perfectly fitting then needing wrap for placeholder)
                    // should have been handled by the wrap logic above.
                    // If it occurs, it implies a logic error or an extremely narrow terminal.
                    warn!(
                        "Wide char placeholder for '{}' at ({},{}) could not be placed due to edge of screen (width {}). This might indicate a wrap logic issue.",
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

        self.cursor_controller.move_right(char_width, &screen_ctx);

        // Check for wrap-next condition *after* moving right
        let (final_logical_x, _) = self.cursor_controller.logical_pos();
        if final_logical_x >= screen_ctx.width {
            // Use >= because logical_x can be == width
            self.cursor_wrap_next = true;
        } else {
            self.cursor_wrap_next = false;
        }
    }

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

    fn backspace(&mut self) {
        self.cursor_wrap_next = false;
        self.cursor_controller.move_left(1);
    }

    fn horizontal_tab(&mut self) {
        self.cursor_wrap_next = false;
        let (current_x, _) = self.cursor_controller.logical_pos();
        let screen_ctx = self.current_screen_context();
        let next_stop = self
            .screen
            .get_next_tabstop(current_x)
            .unwrap_or(screen_ctx.width.saturating_sub(1).max(current_x)); // Ensure forward or stay
        self.cursor_controller
            .move_to_logical_col(next_stop, &screen_ctx);
    }

    fn move_down_one_line_and_dirty(&mut self) {
        self.cursor_wrap_next = false; // Reset wrap flag on any newline-like operation
        let screen_ctx = self.current_screen_context();
        let (_, current_logical_y) = self.cursor_controller.logical_pos();
        let (_current_physical_x, current_physical_y) = // Store current physical X too
        self.cursor_controller.physical_screen_pos(&screen_ctx);

        let max_logical_y_in_region = if screen_ctx.origin_mode_active {
            screen_ctx.scroll_bot.saturating_sub(screen_ctx.scroll_top)
        } else {
            screen_ctx.height.saturating_sub(1)
        };

        let physical_scroll_bot = screen_ctx.scroll_bot;

        if current_physical_y == physical_scroll_bot {
            // At the bottom of the scrolling region, scroll
            log::trace!(
                "move_down_one_line: Scrolling up. Cursor at physical_y: {}, scroll_bot: {}",
                current_physical_y,
                physical_scroll_bot
            );
            self.screen.scroll_up_serial(1);
        } else if current_logical_y < max_logical_y_in_region {
            // Within scroll region (or full screen if no region) but not at the bottom edge of it
            log::trace!(
                "move_down_one_line: Moving cursor down. logical_y: {}, max_logical_y_in_region: {}",
                current_logical_y,
                max_logical_y_in_region
            );
            self.cursor_controller.move_down(1, &screen_ctx);
        } else if !screen_ctx.origin_mode_active
            && current_physical_y < screen_ctx.height.saturating_sub(1)
        {
            // This case handles cursor movement below a scroll region when origin mode is off.
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
            // Cursor might be at the very last line of the screen, outside any scroll region,
            // or already scrolled. No further downward movement.
        }

        // Mark lines dirty
        // current_physical_y is the line the cursor was on *before* any potential move_down or scroll.
        log::trace!(
            "move_down_one_line: Marking old line dirty. current_physical_y: {}, screen_height: {}",
            current_physical_y,
            self.screen.height
        );
        self.screen.mark_line_dirty(current_physical_y);
        // This gets the cursor position *after* move_down or scroll might have occurred.
        let (_, new_physical_y) = self
            .cursor_controller
            .physical_screen_pos(&self.current_screen_context()); // Crucial: use updated context if screen might have changed

        if current_physical_y != new_physical_y {
            log::trace!(
                "move_down_one_line: Marking new line dirty. new_physical_y: {}, screen_height: {}",
                new_physical_y,
                self.screen.height
            );
            self.screen.mark_line_dirty(new_physical_y);
        } else {
            // If the physical line didn't change (e.g., couldn't move down further, and no scroll happened),
            // current_physical_y was already marked. If it scrolled, new_physical_y will be different if
            // cursor effectively stays on the same line number but content shifted.
            // If it just moved down, new_physical_y is different.
            // If it was at the bottom and couldn't move and didn't scroll, physical_y is the same.
            log::trace!(
                "move_down_one_line: New physical y ({}) is same as current ({}), not marking new line again.",
                new_physical_y,
                current_physical_y
            );
        }
    }

    /// Performs a line feed operation: moves cursor down one line,
    /// performing a scroll if at the bottom of the scrolling region.
    /// Also performs a carriage return.
    /// Corresponds to `LF`, `VT`, `FF` control codes, or `NEL` if LNM is not a factor.
    fn perform_line_feed(&mut self) {
        log::trace!("perform_line_feed called");
        self.move_down_one_line_and_dirty();
        self.carriage_return(); // LF, VT, FF typically include CR
    }

    /// Performs an index operation: moves cursor down one line,
    /// performing a scroll if at the bottom of the scrolling region.
    /// Does NOT perform a carriage return.
    /// Corresponds to `IND` (ESC D).
    fn perform_index(&mut self) {
        log::trace!("perform_index called");
        self.move_down_one_line_and_dirty();
    }

    fn carriage_return(&mut self) {
        self.cursor_wrap_next = false;
        self.cursor_controller.carriage_return();
    }

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

    fn index(&mut self) {
        // Equivalent to LF without CR part
        self.cursor_wrap_next = false;
        let screen_ctx = self.current_screen_context();
        let (_, current_physical_y) = self.cursor_controller.physical_screen_pos(&screen_ctx);

        if current_physical_y == screen_ctx.scroll_bot {
            // At bottom of scroll region
            self.screen.scroll_up_serial(1);
        } else if current_physical_y < screen_ctx.height.saturating_sub(1) {
            // Not at physical bottom
            self.cursor_controller.move_down(1, &screen_ctx);
        }

        self.screen.mark_line_dirty(current_physical_y);
        let (_, new_physical_y) = self
            .cursor_controller
            .physical_screen_pos(&self.current_screen_context());
        if current_physical_y != new_physical_y {
            self.screen.mark_line_dirty(new_physical_y);
        }
    }
    fn reverse_index(&mut self) {
        // Mapped to ESC M
        self.cursor_wrap_next = false;
        let screen_ctx = self.current_screen_context();
        let (_, current_physical_y) = self.cursor_controller.physical_screen_pos(&screen_ctx);

        if current_physical_y == screen_ctx.scroll_top {
            // At top of scroll region
            self.screen.scroll_down_serial(1);
        } else if current_physical_y > 0 {
            // Not at physical top
            self.cursor_controller.move_up(1);
        }

        self.screen.mark_line_dirty(current_physical_y);
        let (_, new_physical_y) = self
            .cursor_controller
            .physical_screen_pos(&self.current_screen_context());
        if current_physical_y != new_physical_y {
            self.screen.mark_line_dirty(new_physical_y);
        }
    }
    fn save_cursor_dec(&mut self) {
        // DECSC
        self.cursor_controller.save_state();
    }
    fn restore_cursor_dec(&mut self) {
        // DECRC
        self.cursor_wrap_next = false;
        // Attributes are restored by CursorController.restore_state
        self.cursor_controller
            .restore_state(&self.current_screen_context(), Attributes::default());
        // Ensure screen's default_attributes matches the potentially restored cursor attributes
        self.screen.default_attributes = self.cursor_controller.attributes();
    }

    fn cursor_up(&mut self, n: usize) {
        self.cursor_wrap_next = false;
        self.cursor_controller.move_up(n);
    }

    // Ensure cursor_down calls perform_index
    fn cursor_down(&mut self, n: usize) {
        self.cursor_wrap_next = false;
        log::trace!("cursor_down: n = {}", n);
        for i in 0..n {
            log::trace!("cursor_down: iteration {} calling perform_index", i);
            self.perform_index();
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
        // CHA
        self.cursor_wrap_next = false;
        self.cursor_controller
            .move_to_logical_col(col, &self.current_screen_context());
    }
    fn cursor_to_pos(&mut self, row_param: usize, col_param: usize) {
        // CUP, HVP
        self.cursor_wrap_next = false;
        self.cursor_controller.move_to_logical(
            col_param, // col is first in st.c for tmoveato, but second in CUP sequence
            row_param, // row is second in st.c for tmoveato, but first in CUP sequence
            &self.current_screen_context(),
        );
    }

    fn erase_in_display(&mut self, mode: EraseMode) {
        // ED
        self.cursor_wrap_next = false;
        let screen_ctx = self.current_screen_context();
        let (cx_phys, cy_phys) = self.cursor_controller.physical_screen_pos(&screen_ctx);
        // Ensure screen uses current SGR attributes for clearing, as per standard behavior
        self.screen.default_attributes = self.cursor_controller.attributes();

        match mode {
            EraseMode::ToEnd => {
                // 0
                self.screen
                    .clear_line_segment(cy_phys, cx_phys, screen_ctx.width);
                for y in (cy_phys + 1)..screen_ctx.height {
                    self.screen.clear_line_segment(y, 0, screen_ctx.width);
                }
            }
            EraseMode::ToStart => {
                // 1
                for y in 0..cy_phys {
                    self.screen.clear_line_segment(y, 0, screen_ctx.width);
                }
                self.screen.clear_line_segment(cy_phys, 0, cx_phys + 1); // Inclusive of cursor pos
            }
            EraseMode::All => {
                // 2
                for y in 0..screen_ctx.height {
                    self.screen.clear_line_segment(y, 0, screen_ctx.width);
                }
            }
            EraseMode::Scrollback => {
                // 3 (xterm extension)
                self.screen.scrollback.clear();
                return; // No screen dirtying for scrollback clear
            }
            EraseMode::Unknown => warn!("Unknown ED mode used."),
        }
        // screen.clear_line_segment marks lines dirty. For multi-line ED, ensure all affected are.
        // For simplicity, if not scrollback, mark_all_dirty is safe, though could be optimized.
        if mode != EraseMode::Scrollback {
            self.screen.mark_all_dirty(); // Over-marks for ToEnd/ToStart but ensures correctness
        }
    }
    fn erase_in_line(&mut self, mode: EraseMode) {
        // EL
        self.cursor_wrap_next = false;
        let screen_ctx = self.current_screen_context();
        let (cx_phys, cy_phys) = self.cursor_controller.physical_screen_pos(&screen_ctx);
        self.screen.default_attributes = self.cursor_controller.attributes();

        match mode {
            EraseMode::ToEnd => self
                .screen
                .clear_line_segment(cy_phys, cx_phys, screen_ctx.width),
            EraseMode::ToStart => self.screen.clear_line_segment(cy_phys, 0, cx_phys + 1), // Inclusive
            EraseMode::All => self.screen.clear_line_segment(cy_phys, 0, screen_ctx.width),
            EraseMode::Scrollback => {
                warn!("EraseMode::Scrollback is not applicable to EraseInLine (EL).")
            }
            EraseMode::Unknown => warn!("Unknown EL mode used."),
        }
    }
    fn erase_chars(&mut self, n: usize) {
        // ECH
        self.cursor_wrap_next = false;
        let screen_ctx = self.current_screen_context();
        let (cx_phys, cy_phys) = self.cursor_controller.physical_screen_pos(&screen_ctx);
        let end_x = min(cx_phys + n, screen_ctx.width); // Erase n chars, or up to line end
        self.screen.default_attributes = self.cursor_controller.attributes();
        self.screen.clear_line_segment(cy_phys, cx_phys, end_x);
    }

    fn insert_blank_chars(&mut self, n: usize) {
        // ICH
        self.cursor_wrap_next = false;
        let screen_ctx = self.current_screen_context();
        let (cx_log, _) = self.cursor_controller.logical_pos();
        let (_, cy_phys) = self.cursor_controller.physical_screen_pos(&screen_ctx);
        self.screen.default_attributes = self.cursor_controller.attributes();
        self.screen.insert_blank_chars_in_line(cy_phys, cx_log, n);
    }
    fn delete_chars(&mut self, n: usize) {
        // DCH
        self.cursor_wrap_next = false;
        let screen_ctx = self.current_screen_context();
        let (cx_log, _) = self.cursor_controller.logical_pos();
        let (_, cy_phys) = self.cursor_controller.physical_screen_pos(&screen_ctx);
        self.screen.default_attributes = self.cursor_controller.attributes();
        self.screen.delete_chars_in_line(cy_phys, cx_log, n);
    }
    fn insert_lines(&mut self, n: usize) {
        // IL
        self.cursor_wrap_next = false;
        let screen_ctx = self.current_screen_context();
        let (_, cy_phys) = self.cursor_controller.physical_screen_pos(&screen_ctx);
        self.screen.default_attributes = self.cursor_controller.attributes();

        if cy_phys >= screen_ctx.scroll_top && cy_phys <= screen_ctx.scroll_bot {
            // IL operates within the scrolling margins.
            // Lines from cy_phys to scroll_bot scroll down.
            let original_scroll_top = self.screen.scroll_top();
            let original_scroll_bottom = self.screen.scroll_bot();

            // Temporarily set scroll region from current line to bottom of original region
            // to make scroll_down_serial operate correctly on the affected part.
            // Add 1 because set_scrolling_region takes 1-based params.
            self.screen
                .set_scrolling_region(cy_phys + 1, original_scroll_bottom + 1);
            self.screen.scroll_down_serial(n);

            self.screen
                .set_scrolling_region(original_scroll_top + 1, original_scroll_bottom + 1);

            // Ensure all lines in the original scroll region that were affected are marked dirty.
            // scroll_down_serial marks dirty lines within its *current* (temporary) scroll region.
            // We need to ensure lines from original_scroll_top to original_scroll_bottom are considered.
            for y_dirty in cy_phys..=original_scroll_bottom {
                self.screen.mark_line_dirty(y_dirty);
            }
        }
    }
    fn delete_lines(&mut self, n: usize) {
        // DL
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
                self.screen.mark_line_dirty(y_dirty);
            }
        }
    }
    fn scroll_up(&mut self, n: usize) {
        // SU
        self.cursor_wrap_next = false;
        self.screen.default_attributes = self.cursor_controller.attributes();
        self.screen.scroll_up_serial(n);
    }
    fn scroll_down(&mut self, n: usize) {
        // SD
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
                Attribute::UnderlineDouble => current_attrs.flags.insert(AttrFlags::UNDERLINE), // Treat as single for now
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
                    current_attrs.fg = map_ansi_color_to_glyph_color(color)
                }
                Attribute::Background(color) => {
                    current_attrs.bg = map_ansi_color_to_glyph_color(color)
                }
                Attribute::Overlined => warn!("SGR Overlined not yet visually supported."),
                Attribute::NoOverlined => warn!("SGR NoOverlined not yet visually supported."),
                Attribute::UnderlineColor(color) => {
                    warn!("SGR UnderlineColor not yet fully supported: {:?}", color)
                    // TODO: Store underline color in Attributes struct if desired
                }
            }
        }
        self.cursor_controller.set_attributes(current_attrs);
        // Update screen's default_attributes for subsequent clearing operations *if SGR changes it*.
        // This is a key point: SGR changes the *current* attributes for printing AND for clearing.
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
                        // ?25
                        self.dec_modes.text_cursor_enable_mode = enable; // Update mode state
                        self.cursor_controller.set_visible(enable);
                        action_to_return = Some(EmulatorAction::SetCursorVisibility(enable));
                    }
                    Some(DecModeConstant::AltScreenBufferClear)
                    | Some(DecModeConstant::AltScreenBufferSaveRestore) => {
                        // 1047, 1049
                        if !self.dec_modes.allow_alt_screen {
                            // Check global allowaltscreen equivalent
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
                            // Enter alt screen
                            if !self.dec_modes.using_alt_screen {
                                if mode_num == DecModeConstant::AltScreenBufferSaveRestore as u16 {
                                    self.save_cursor_dec(); // Save cursor state (DECSC-like)
                                }
                                self.screen.default_attributes =
                                    self.cursor_controller.attributes(); // Use current for clearing alt
                                self.screen.enter_alt_screen(clear_on_entry);
                                self.dec_modes.using_alt_screen = true;
                                self.cursor_controller.move_to_logical(
                                    0,
                                    0,
                                    &self.current_screen_context(),
                                );
                                self.mark_all_lines_dirty();
                                action_to_return = Some(EmulatorAction::RequestRedraw);
                            }
                        } else {
                            // Exit alt screen
                            if self.dec_modes.using_alt_screen {
                                self.screen.exit_alt_screen();
                                self.dec_modes.using_alt_screen = false;
                                if mode_num == DecModeConstant::AltScreenBufferSaveRestore as u16 {
                                    self.restore_cursor_dec(); // Restore cursor state (DECRC-like)
                                } else {
                                    // For 1047, cursor typically goes to home of primary screen
                                    self.cursor_controller.move_to_logical(
                                        0,
                                        0,
                                        &self.current_screen_context(),
                                    );
                                }
                                self.screen.default_attributes =
                                    self.cursor_controller.attributes(); // Sync after potential restore
                                self.mark_all_lines_dirty();
                                action_to_return = Some(EmulatorAction::RequestRedraw);
                            }
                        }
                    }
                    Some(DecModeConstant::SaveRestoreCursor) => {
                        // 1048 (often used with 1049)
                        // This mode itself doesn't directly switch screens in st.c,
                        // it enables the DECSC/DECRC behavior for 1049.
                        // For now, we can treat it as a separate save/restore if needed,
                        // but its main utility is in conjunction with 1049.
                        // Let's assume it's for the current screen.
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
                        // ?12
                        self.dec_modes.cursor_blink_mode = enable;
                        // TODO: This might need an EmulatorAction to signal blink state change to renderer/driver
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
            Mode::Standard(mode_num) => {
                // Standard modes (SM/RM)
                match mode_num {
                    4 => {
                        // IRM - Insert/Replace Mode
                        self.dec_modes.insert_mode = enable;
                    }
                    20 => {
                        // LNM - Linefeed/Newline Mode
                        self.dec_modes.linefeed_newline_mode = enable;
                        // If LNM is set (true), LF, FF, VT should also perform CR.
                        // If LNM is reset (false), they act as line feeds only.
                    }
                    _ => {
                        warn!(
                            "Standard mode {} set/reset to {} - not fully implemented yet.",
                            mode_num, enable
                        );
                    }
                }
            }
        }
        action_to_return
    }

    fn handle_window_manipulation(
        &mut self,
        ps1: u16,
        _ps2: Option<u16>, // placeholder for ps2
        _ps3: Option<u16>, // placeholder for ps3
    ) -> Option<EmulatorAction> {
        match ps1 {
            14 => {
                // Report text area size in pixels (not implemented)
                warn!(
                    "WindowManipulation: Report text area size in pixels (14) requested, but not implemented."
                );
                None
            }
            18 => {
                // Report text area size in characters
                let (cols, rows) = self.dimensions();
                let response = format!("\x1b[8;{};{}t", rows, cols);
                Some(EmulatorAction::WritePty(response.into_bytes()))
            }
            22 | 23 => {
                // Save/Restore window title (not implemented)
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
            let ps = parts[0].parse::<u32>().unwrap_or(u32::MAX); // Use u32::MAX for unparsable
            let pt = parts[1].to_string();
            match ps {
                0 | 2 => {
                    // Set icon name and window title (0), Set window title (2)
                    return Some(EmulatorAction::SetTitle(pt));
                }
                // TODO: Handle other OSC commands like color setting (4, 10, 11, 12, etc.)
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
