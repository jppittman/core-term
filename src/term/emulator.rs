// Re-export items for easier use by other modules and within this module
use super::modes::{DecModeConstant, DecPrivateModes, EraseMode, Mode};
use super::*;
use action::EmulatorAction;
use charset::{CharacterSet, map_to_dec_line_drawing};

// Standard library imports
use std::cmp::min;

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
    // Import crate::color::Color if needed for explicit type annotations,
    // but it's mostly handled by the Attribute enum now.
    // use crate::color::Color;
    glyph::{AttrFlags, Attributes},
    term::ControlEvent,
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
    pub(super) dirty_lines: Vec<usize>, // To be deprecated in favor of screen.dirty
    pub(super) cursor_wrap_next: bool,
    pub(super) current_cursor_shape: u16, // Stores the current cursor shape code
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
            active_charset_g_level: 0,                 // Default to G0
            dirty_lines: (0..height.max(1)).collect(), // Mark all lines dirty initially
            cursor_wrap_next: false,
            current_cursor_shape: DEFAULT_CURSOR_SHAPE, // Use constant for default
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
    // These methods are marked pub(super) and are used in tests.
    // Allowing dead_code for them as they might not be used outside the `term` module's tests.
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
    /// This is the primary entry point for commands originating from the PTY.
    pub(super) fn handle_ansi_command(&mut self, command: AnsiCommand) -> Option<EmulatorAction> {
        // If the command is not a Print command, reset the cursor_wrap_next flag.
        // This flag is only relevant for consecutive printable characters.
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
                    // LF, VT, FF are often treated similarly: move down one line.
                    // If Linefeed/Newline Mode (LNM) is set, also perform a CR.
                    self.perform_line_feed(); // This handles LF and potential LNM CR.
                    None
                }
                C0Control::CR => {
                    self.carriage_return();
                    None
                }
                C0Control::SO => {
                    // Shift Out (LS1) - Invoke G1 character set.
                    self.set_g_level(1);
                    None
                }
                C0Control::SI => {
                    // Shift In (LS0) - Invoke G0 character set.
                    self.set_g_level(0);
                    None
                }
                C0Control::BEL => Some(EmulatorAction::RingBell),
                // Other C0 controls might be ignored or handled specifically if needed.
                _ => {
                    debug!("Unhandled C0 control: {:?}", c0);
                    None
                }
            },
            AnsiCommand::Esc(esc_cmd) => match esc_cmd {
                EscCommand::SetTabStop => {
                    // HTS - Horizontal Tabulation Set
                    let (cursor_x, _) = self.cursor_controller.logical_pos();
                    self.screen.set_tabstop(cursor_x);
                    None
                }
                EscCommand::Index => {
                    // IND - Index (move cursor down one line, scroll if at bottom)
                    self.index();
                    None
                }
                EscCommand::NextLine => {
                    // NEL - Next Line (like CR then LF)
                    self.perform_line_feed(); // Handles LF and potential LNM CR.
                    self.carriage_return(); // Ensure CR if LNM wasn't active.
                    None
                }
                EscCommand::ReverseIndex => {
                    // RI - Reverse Index (move cursor up one line, scroll if at top)
                    self.reverse_index();
                    None
                }
                EscCommand::SaveCursor => {
                    // DECSC - Save Cursor (DEC private)
                    self.save_cursor_dec();
                    None
                }
                EscCommand::RestoreCursor => {
                    // DECRC - Restore Cursor (DEC private)
                    self.restore_cursor_dec();
                    None
                }
                EscCommand::SelectCharacterSet(intermediate_char, final_char) => {
                    // Designate G0, G1, G2, or G3 character set.
                    // intermediate_char: '(', ')', '*', '+' for G0, G1, G2, G3 respectively.
                    // final_char: Defines the character set (e.g., 'B' for ASCII, '0' for DEC Special Graphics).
                    let g_idx = match intermediate_char {
                        '(' => 0, // G0
                        ')' => 1, // G1
                        '*' => 2, // G2
                        '+' => 3, // G3
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
                    // RIS - Reset to Initial State
                    trace!("Processing ESC c (ResetToInitialState)");

                    // 1. Ensure on primary screen
                    if self.screen.alt_screen_active {
                        self.screen.exit_alt_screen(); // This also marks all lines dirty
                    }

                    // 2. Reset attributes for clearing and cursor
                    let default_attrs = Attributes::default();
                    self.cursor_controller.set_attributes(default_attrs);
                    self.screen.default_attributes = default_attrs;

                    // 3. Clear screen content (uses self.screen.default_attributes)
                    self.erase_in_display(EraseMode::All); // This also marks lines dirty

                    // 4. Home cursor (attributes are already default)
                    self.cursor_controller
                        .move_to_logical(0, 0, &self.current_screen_context());

                    // 5. Reset DEC Private Modes (also resets origin_mode, cursor visibility via DECTCEM default)
                    self.dec_modes = DecPrivateModes::default();
                    self.screen.origin_mode = self.dec_modes.origin_mode; // Sync screen's copy

                    // 6. Reset scrolling region to full screen
                    let (_, h) = self.dimensions();
                    self.screen.set_scrolling_region(1, h); // 1-based params

                    // 7. Reset character sets
                    self.active_charsets = [CharacterSet::Ascii; 4]; // G0-G3 to ASCII
                    self.active_charset_g_level = 0; // Select G0

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
                    self.current_cursor_shape = DEFAULT_CURSOR_SHAPE; // Reset to default shape
                    self.cursor_controller.set_visible(true); // Cursor visible by default

                    // 10. All lines are already dirty from erase_in_display or exit_alt_screen.
                    // self.mark_all_lines_dirty(); // Redundant if erase_in_display does it.

                    // 11. Signal cursor visibility change if DECTCEM default is true
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
                    // Move cursor down N lines, then to column 1.
                    self.cursor_down(n.max(1) as usize);
                    self.carriage_return();
                    None
                }
                CsiCommand::CursorPrevLine(n) => {
                    // Move cursor up N lines, then to column 1.
                    self.cursor_up(n.max(1) as usize);
                    self.carriage_return();
                    None
                }
                CsiCommand::CursorCharacterAbsolute(n) => {
                    // CHA - Cursor Character Absolute (move to column N).
                    self.cursor_to_column(n.saturating_sub(1) as usize); // 1-based to 0-based
                    None
                }
                CsiCommand::CursorPosition(r, c) => {
                    // CUP - Cursor Position.
                    self.cursor_to_pos(r.saturating_sub(1) as usize, c.saturating_sub(1) as usize); // 1-based to 0-based
                    None
                }
                CsiCommand::EraseInDisplay(mode_val) => {
                    // ED - Erase in Display.
                    self.erase_in_display(EraseMode::from(mode_val));
                    None
                }
                CsiCommand::EraseInLine(mode_val) => {
                    // EL - Erase in Line.
                    self.erase_in_line(EraseMode::from(mode_val));
                    None
                }
                CsiCommand::InsertCharacter(n) => {
                    // ICH - Insert Character.
                    self.insert_blank_chars(n.max(1) as usize);
                    None
                }
                CsiCommand::DeleteCharacter(n) => {
                    // DCH - Delete Character.
                    self.delete_chars(n.max(1) as usize);
                    None
                }
                CsiCommand::InsertLine(n) => {
                    // IL - Insert Line.
                    self.insert_lines(n.max(1) as usize);
                    None
                }
                CsiCommand::DeleteLine(n) => {
                    // DL - Delete Line.
                    self.delete_lines(n.max(1) as usize);
                    None
                }
                CsiCommand::SetGraphicsRendition(attrs_vec) => {
                    // SGR - Select Graphic Rendition.
                    self.handle_sgr_attributes(attrs_vec);
                    None
                }
                CsiCommand::SetMode(mode_num) => {
                    // SM - Set Mode.
                    self.handle_set_mode(Mode::Standard(mode_num), true)
                }
                CsiCommand::ResetMode(mode_num) => {
                    // RM - Reset Mode.
                    self.handle_set_mode(Mode::Standard(mode_num), false)
                }
                CsiCommand::SetModePrivate(mode_num) => {
                    // DECSET - Set DEC Private Mode.
                    self.handle_set_mode(Mode::DecPrivate(mode_num), true)
                }
                CsiCommand::ResetModePrivate(mode_num) => {
                    // DECRST - Reset DEC Private Mode.
                    self.handle_set_mode(Mode::DecPrivate(mode_num), false)
                }
                CsiCommand::DeviceStatusReport(dsr_param) => {
                    // DSR - Device Status Report.
                    if dsr_param == 0 || dsr_param == 6 {
                        // DSR 6 is cursor position report (CPR).
                        // Format: ESC [ <row> ; <col> R
                        let screen_ctx = self.current_screen_context();
                        let (abs_x, abs_y) =
                            self.cursor_controller.physical_screen_pos(&screen_ctx);
                        let response = format!("\x1B[{};{}R", abs_y + 1, abs_x + 1); // 1-based
                        Some(EmulatorAction::WritePty(response.into_bytes()))
                    } else if dsr_param == 5 {
                        // DSR 5 is status report (OK).
                        // Format: ESC [ 0 n
                        Some(EmulatorAction::WritePty(b"\x1B[0n".to_vec()))
                    } else {
                        warn!("Unhandled DSR parameter: {}", dsr_param);
                        None
                    }
                }
                CsiCommand::EraseCharacter(n) => {
                    // ECH - Erase Character.
                    self.erase_chars(n.max(1) as usize);
                    None
                }
                CsiCommand::ScrollUp(n) => {
                    // SU - Scroll Up.
                    self.scroll_up(n.max(1) as usize);
                    None
                }
                CsiCommand::ScrollDown(n) => {
                    // SD - Scroll Down.
                    self.scroll_down(n.max(1) as usize);
                    None
                }
                CsiCommand::SaveCursor => {
                    // DECSC - Save Cursor (DEC private sequence, often also mapped to non-private 's').
                    self.save_cursor_dec();
                    None
                }
                CsiCommand::RestoreCursor => {
                    // DECRC - Restore Cursor (DEC private sequence, often also mapped to non-private 'u').
                    self.restore_cursor_dec();
                    None
                }
                CsiCommand::ClearTabStops(mode_val) => {
                    // TBC - Tabulation Clear.
                    let (cursor_x, _) = self.cursor_controller.logical_pos();
                    self.screen
                        .clear_tabstops(cursor_x, screen::TabClearMode::from(mode_val));
                    None
                }
                CsiCommand::SetScrollingRegion { top, bottom } => {
                    // DECSTBM - Set Top and Bottom Margins (Scrolling Region).
                    // Parameters are 1-based. If bottom is 0 or omitted, it defaults to last line.
                    self.screen
                        .set_scrolling_region(top as usize, bottom as usize);
                    // After setting scrolling region, cursor moves to home within the new region
                    // if origin mode is active, or to absolute (0,0) if not.
                    // For simplicity, always move to logical (0,0) within the current origin context.
                    self.cursor_controller
                        .move_to_logical(0, 0, &self.current_screen_context());
                    None
                }
                CsiCommand::SetCursorStyle { shape } => {
                    // DECSCUSR - Set Cursor Style.
                    self.current_cursor_shape = shape;
                    debug!("Set cursor style to shape: {}", shape);
                    // Note: Visual update of cursor shape is handled by the renderer/driver
                    // based on this state when the cursor is drawn.
                    None
                }
                CsiCommand::WindowManipulation { ps1, ps2, ps3 } => {
                    // DTTERM Window Manipulation (often used for window/icon titles, size reports).
                    self.handle_window_manipulation(ps1, ps2, ps3)
                }
                CsiCommand::Unsupported(intermediates, final_byte_opt) => {
                    warn!(
                        "TerminalEmulator received CsiCommand::Unsupported: intermediates={:?}, final={:?}. This is usually an error from the parser.",
                        intermediates, final_byte_opt
                    );
                    None
                }
                // Catch-all for other CsiCommand variants not explicitly handled above.
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
            // Other AnsiCommand variants (Dcs, Pm, Apc, StringTerminator, Ignore, Error)
            // might be logged or handled specifically if needed.
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
    /// This is where user input (like key presses) or platform events are translated
    /// into actions, often resulting in bytes being sent to the PTY.
    pub(super) fn handle_backend_event(&mut self, event: BackendEvent) -> Option<EmulatorAction> {
        self.cursor_wrap_next = false; // User input typically breaks character wrapping sequences.

        // Define common keysyms for better readability.
        // These values are typically from X11/keysymdef.h or similar.
        const KEY_RETURN: u32 = 0xFF0D;
        const KEY_BACKSPACE: u32 = 0xFF08;
        const KEY_TAB: u32 = 0xFF09;
        const KEY_ISO_LEFT_TAB: u32 = 0xFE20; // Shift+Tab often produces this
        const KEY_ESCAPE: u32 = 0xFF1B;

        const KEY_LEFT_ARROW: u32 = 0xFF51;
        const KEY_UP_ARROW: u32 = 0xFF52;
        const KEY_RIGHT_ARROW: u32 = 0xFF53;
        const KEY_DOWN_ARROW: u32 = 0xFF54;
        // Add other keysyms as needed (Home, End, PgUp, PgDn, Delete, Insert, F-keys)

        match event {
            BackendEvent::Key { keysym, text } => {
                let mut bytes_to_send: Vec<u8> = Vec::new();
                // Prioritize keysym for special keys, then text for printable characters.
                match keysym {
                    KEY_RETURN => bytes_to_send.push(b'\r'), // Send Carriage Return
                    KEY_BACKSPACE => bytes_to_send.push(0x08), // ASCII BS
                    KEY_TAB => bytes_to_send.push(b'\t'),    // ASCII HT
                    KEY_ISO_LEFT_TAB => bytes_to_send.extend_from_slice(b"\x1b[Z"), // Common sequence for Shift+Tab
                    KEY_ESCAPE => bytes_to_send.push(0x1B),                         // ASCII ESC

                    // Cursor Keys: Check Application Cursor Keys Mode (DECCKM)
                    KEY_UP_ARROW => {
                        bytes_to_send.extend_from_slice(if self.dec_modes.cursor_keys_app_mode {
                            b"\x1bOA" // SS3 A
                        } else {
                            b"\x1b[A" // CSI A
                        })
                    }
                    KEY_DOWN_ARROW => {
                        bytes_to_send.extend_from_slice(if self.dec_modes.cursor_keys_app_mode {
                            b"\x1bOB" // SS3 B
                        } else {
                            b"\x1b[B" // CSI B
                        })
                    }
                    KEY_RIGHT_ARROW => {
                        bytes_to_send.extend_from_slice(if self.dec_modes.cursor_keys_app_mode {
                            b"\x1bOC" // SS3 C
                        } else {
                            b"\x1b[C" // CSI C
                        })
                    }
                    KEY_LEFT_ARROW => {
                        bytes_to_send.extend_from_slice(if self.dec_modes.cursor_keys_app_mode {
                            b"\x1bOD" // SS3 D
                        } else {
                            b"\x1b[D" // CSI D
                        })
                    }
                    // Add other special keys here (Home, End, F-keys, etc.)

                    // If not a special keysym handled above, use the provided text.
                    _ => {
                        if !text.is_empty() {
                            bytes_to_send.extend(text.as_bytes());
                        } else {
                            // Log unhandled keysyms if they don't produce text, for debugging.
                            trace!("Unhandled keysym: {:#X} with empty text", keysym);
                        }
                    }
                }

                if !bytes_to_send.is_empty() {
                    return Some(EmulatorAction::WritePty(bytes_to_send));
                }
            }
            BackendEvent::FocusGained => {
                // Handle FocusIn event if focus reporting mode is active.
                if self.dec_modes.focus_event_mode {
                    return Some(EmulatorAction::WritePty(b"\x1b[I".to_vec())); // CSI I
                }
            }
            BackendEvent::FocusLost => {
                // Handle FocusOut event if focus reporting mode is active.
                if self.dec_modes.focus_event_mode {
                    return Some(EmulatorAction::WritePty(b"\x1b[O".to_vec())); // CSI O
                }
            }
            // Resize and CloseRequested events are typically handled directly by the orchestrator
            // before reaching the TerminalEmulator as a User event.
            // If they do reach here, it's likely an architectural oversight or a specific design choice.
            _ => debug!(
                "BackendEvent {:?} passed to TerminalEmulator's handle_backend_event, usually handled by Orchestrator or translated.",
                event
            ),
        }
        None // No action to take for unhandled or non-PTY-writing events
    }

    /// Handles an internal `ControlEvent`.
    /// These are events generated by the orchestrator or other parts of the terminal
    /// application itself, not directly from user input or PTY output.
    pub(super) fn handle_control_event(&mut self, event: ControlEvent) -> Option<EmulatorAction> {
        self.cursor_wrap_next = false; // Control events usually reset wrap state.
        match event {
            ControlEvent::FrameRendered => {
                // This event signifies that the renderer has completed drawing a frame.
                // The terminal might use this to clear per-frame dirty flags or other
                // post-render bookkeeping if necessary.
                trace!("TerminalEmulator: FrameRendered event received.");
                // Currently, dirty line clearing is handled by `take_dirty_lines`.
                // No specific action needed here for now.
                None
            }
            ControlEvent::Resize { cols, rows } => {
                // This event signals that the orchestrator has determined new terminal
                // dimensions (e.g., from a window resize).
                trace!(
                    "TerminalEmulator: ControlEvent::Resize to {}x{} received.",
                    cols, rows
                );
                self.resize(cols, rows);
                // Resize itself marks lines dirty, orchestrator will trigger redraw.
                None
            }
        }
    }

    /// Resizes the terminal display grid.
    /// This involves adjusting the screen's internal buffers and potentially
    /// re-clamping the cursor position.
    pub(super) fn resize(&mut self, cols: usize, rows: usize) {
        self.cursor_wrap_next = false; // Resize invalidates wrap state.
        let current_scrollback_limit = self.screen.scrollback_limit();
        // The screen's resize method handles adjusting grids and marking all lines dirty.
        self.screen.resize(cols, rows, current_scrollback_limit);

        // After screen resize, ensure the cursor's logical position is still valid
        // and re-clamp it if necessary. The ScreenContext will now reflect new dimensions.
        let (log_x, log_y) = self.cursor_controller.logical_pos();
        self.cursor_controller
            .move_to_logical(log_x, log_y, &self.current_screen_context());

        // `screen.resize` already marks all lines dirty.
        // self.mark_all_lines_dirty(); // This would be redundant.
        debug!(
            "Terminal resized to {}x{}. Cursor re-clamped. All lines marked dirty by screen.resize().",
            cols, rows
        );
    }

    /// Marks all lines on the screen as dirty, forcing a full redraw.
    /// This method updates both the legacy `dirty_lines` vector (for now)
    /// and the `screen.dirty` bitmap.
    fn mark_all_lines_dirty(&mut self) {
        // TODO: Deprecate self.dirty_lines in favor of screen.dirty.
        self.dirty_lines = (0..self.screen.height).collect();
        self.screen.mark_all_dirty(); // Ensure screen's internal flags are also set.
    }

    /// Marks a single line `y_abs` (absolute, 0-based) as dirty.
    /// This method updates both the legacy `dirty_lines` vector (for now)
    /// and the `screen.dirty` bitmap.
    #[allow(dead_code)] // Keep for potential direct use or future refactor.
    fn mark_line_dirty(&mut self, y_abs: usize) {
        if y_abs < self.screen.height && !self.dirty_lines.contains(&y_abs) {
            // self.dirty_lines.push(y_abs); // TODO: Remove if fully relying on screen.dirty
        }
        self.screen.mark_line_dirty(y_abs); // Use screen's method.
    }

    // --- Public query methods used by TerminalInterface ---

    /// Returns the current logical cursor position (0-based column, row).
    pub fn cursor_pos(&self) -> (usize, usize) {
        self.cursor_controller.logical_pos()
    }

    /// Returns `true` if the alternate screen buffer is currently active.
    pub fn is_alt_screen_active(&self) -> bool {
        self.screen.alt_screen_active
    }

    // --- Character Printing and Low-Level Operations ---

    /// Prints a single character to the terminal at the current cursor position.
    /// Handles character width, line wrapping, and updates cursor position.
    pub(super) fn print_char(&mut self, ch: char) {
        // Map character to the active G0/G1/G2/G3 character set.
        let ch_to_print = self.map_char_to_active_charset(ch);
        let char_width = get_char_display_width(ch_to_print);

        // Zero-width characters (like combining marks not handled by precomposition)
        // are complex. For now, if wcwidth reports 0, we might simply not advance
        // the cursor, or try to overstrike. True handling requires grapheme clustering.
        // Current behavior: if width is 0, we effectively "skip" it for cursor advancement,
        // but it might be part of a grapheme cluster that the font handles.
        if char_width == 0 {
            trace!(
                "print_char: Encountered zero-width char '{}'. Behavior might be minimal.",
                ch_to_print
            );
            // Potentially, one could try to place it in the current cell without advancing,
            // but this depends on how the renderer and font handle combining characters.
            // For simplicity, we might just not advance the cursor for a true ZWC.
            // However, the current `get_char_display_width` might return 1 for unprintable controls
            // that wcwidth returns -1 for, so this path might not be hit often for those.
            return; // Or handle by overstriking/combining if renderer supports.
        }

        let mut screen_ctx = self.current_screen_context(); // Get current screen context

        // Handle line wrap if cursor_wrap_next was set by the previous character.
        if self.cursor_wrap_next {
            self.perform_line_feed(); // This also does CR if LNM is set.
            screen_ctx = self.current_screen_context(); // Update context after potential scroll
            self.cursor_wrap_next = false; // Reset wrap flag
        }

        // Get current physical cursor position for placing the glyph.
        let (mut physical_x, mut physical_y) =
            self.cursor_controller.physical_screen_pos(&screen_ctx);

        // Check if the character (considering its width) would exceed the line width.
        if physical_x + char_width > screen_ctx.width {
            // If a wide char (width 2) is at the very last column, it can't fit.
            // Some terminals might print a space or '?' in the last cell.
            // Here, we'll clear the last cell if it's a wide char that would overflow.
            if char_width == 2 && physical_x == screen_ctx.width.saturating_sub(1) {
                let fill_glyph = Glyph {
                    c: ' ',                                    // Fill with a space
                    attr: self.cursor_controller.attributes(), // Use current attributes for the space
                };
                if physical_y < self.screen.height {
                    // Ensure y is within bounds
                    self.screen.set_glyph(physical_x, physical_y, fill_glyph);
                }
            }
            // Perform line feed (and potential CR if LNM)
            self.perform_line_feed();
            screen_ctx = self.current_screen_context(); // Update context
            // Get new physical cursor position after line feed.
            (physical_x, physical_y) = self.cursor_controller.physical_screen_pos(&screen_ctx);
        }

        // Place the character glyph on the screen.
        let glyph_attrs = self.cursor_controller.attributes();
        if physical_y < self.screen.height {
            // Ensure y is within bounds
            self.screen.set_glyph(
                physical_x,
                physical_y,
                Glyph {
                    c: ch_to_print,
                    attr: glyph_attrs,
                },
            );

            // If it's a wide character, place a null placeholder in the next cell.
            if char_width == 2 {
                if physical_x + 1 < screen_ctx.width {
                    self.screen.set_glyph(
                        physical_x + 1,
                        physical_y,
                        Glyph {
                            c: '\0', // Placeholder for wide char
                            attr: glyph_attrs,
                        },
                    );
                } else {
                    // This case should ideally be prevented by the wrap logic above.
                    // If a wide char is printed at the last column, it should have wrapped.
                    warn!(
                        "Wide char placeholder for '{}' at ({},{}) could not be placed due to edge of screen (width {}). This might indicate a wrap logic issue.",
                        ch_to_print, physical_x, physical_y, screen_ctx.width
                    );
                }
            }
        } else {
            // This should ideally not happen if scrolling and cursor logic is correct.
            warn!(
                "print_char: Attempted to print at physical_y {} out of bounds (height {})",
                physical_y, self.screen.height
            );
        }

        // Advance the logical cursor position by the character's width.
        self.cursor_controller.move_right(char_width, &screen_ctx);

        // Check if the new logical cursor position requires a wrap on the *next* character.
        let (final_logical_x, _) = self.cursor_controller.logical_pos();
        self.cursor_wrap_next = final_logical_x >= screen_ctx.width;
    }

    /// Maps a character to its equivalent in the currently active G0/G1/G2/G3 character set.
    fn map_char_to_active_charset(&self, ch: char) -> char {
        let current_set = self.active_charsets[self.active_charset_g_level];
        match current_set {
            CharacterSet::Ascii => ch, // No mapping needed for ASCII
            CharacterSet::UkNational => {
                // UK pound sign mapping for '#'
                if ch == '#' { '£' } else { ch }
            }
            CharacterSet::DecLineDrawing => map_to_dec_line_drawing(ch),
            // Add other character set mappings here if supported.
        }
    }

    // --- Basic Cursor Movements and Control Code Handlers ---

    /// Handles BS (Backspace). Moves cursor left by one, stopping at column 0.
    fn backspace(&mut self) {
        self.cursor_wrap_next = false; // Backspace cancels wrap.
        self.cursor_controller.move_left(1);
    }

    /// Handles HT (Horizontal Tab). Moves cursor to the next tab stop.
    /// If no more tab stops, moves to the last column.
    fn horizontal_tab(&mut self) {
        self.cursor_wrap_next = false; // Tab cancels wrap.
        let (current_x, _) = self.cursor_controller.logical_pos();
        let screen_ctx = self.current_screen_context();
        // Find next tab stop, or go to last column if none.
        let next_stop = self
            .screen
            .get_next_tabstop(current_x)
            .unwrap_or(screen_ctx.width.saturating_sub(1).max(current_x)); // Ensure it doesn't go past current_x if no tabs
        self.cursor_controller
            .move_to_logical_col(next_stop, &screen_ctx);
    }

    /// Performs a line feed operation: moves cursor down one line.
    /// If at the bottom of the scrolling region, the region scrolls up.
    /// If Linefeed/Newline Mode (LNM) is active, also performs a carriage return.
    fn perform_line_feed(&mut self) {
        log::trace!("perform_line_feed called");
        self.move_down_one_line_and_dirty(); // Handles scrolling and y-movement
        if self.dec_modes.linefeed_newline_mode {
            self.carriage_return();
        }
    }

    /// Helper for LF, IND, NEL: moves cursor down one line, scrolling if necessary.
    /// Marks affected lines as dirty.
    fn move_down_one_line_and_dirty(&mut self) {
        self.cursor_wrap_next = false;
        let screen_ctx = self.current_screen_context();
        let (_, current_logical_y) = self.cursor_controller.logical_pos();
        let (_current_physical_x, current_physical_y) =
            self.cursor_controller.physical_screen_pos(&screen_ctx);

        // Determine the maximum logical Y within the current context (screen or scroll region)
        let max_logical_y_in_region = if screen_ctx.origin_mode_active {
            screen_ctx.scroll_bot.saturating_sub(screen_ctx.scroll_top)
        } else {
            screen_ctx.height.saturating_sub(1)
        };

        // Physical bottom of the scrolling region (if origin mode) or screen.
        let physical_effective_bottom = if screen_ctx.origin_mode_active {
            screen_ctx.scroll_bot
        } else {
            screen_ctx.height.saturating_sub(1)
        };

        // If cursor is at the bottom of its effective region (scrolling region or screen)
        if current_physical_y == physical_effective_bottom {
            log::trace!(
                "move_down_one_line: Scrolling up. Cursor at physical_y: {}, effective_bottom: {}",
                current_physical_y,
                physical_effective_bottom
            );
            // Scroll the content of the active scrolling region up by one line.
            // Note: scroll_up_serial uses scroll_top/scroll_bot from screen state.
            self.screen.scroll_up_serial(1);
            // Cursor's logical_y might not change if origin_mode is on and it was at the bottom of the region.
            // If origin_mode is off and it was at screen bottom, it stays at the new bottom (logically).
        } else if current_logical_y < max_logical_y_in_region {
            // If not at the bottom, simply move the cursor down one logical line.
            log::trace!(
                "move_down_one_line: Moving cursor down. logical_y: {}, max_logical_y_in_region: {}",
                current_logical_y,
                max_logical_y_in_region
            );
            self.cursor_controller.move_down(1, &screen_ctx);
        } else if !screen_ctx.origin_mode_active
            && current_physical_y < screen_ctx.height.saturating_sub(1)
        {
            // This case handles cursor below a defined scrolling region when origin mode is off.
            log::trace!(
                "move_down_one_line: Moving cursor down (below scroll region, origin mode off). physical_y: {}, screen_height: {}",
                current_physical_y,
                screen_ctx.height
            );
            self.cursor_controller.move_down(1, &screen_ctx);
        } else {
            // Cursor is at the logical bottom and physical bottom, no scroll needed, no move down possible.
            log::trace!(
                "move_down_one_line: Cursor at bottom, no scroll or move_down. physical_y: {}, logical_y: {}, max_logical_y: {}",
                current_physical_y,
                current_logical_y,
                max_logical_y_in_region
            );
        }

        // Mark the original cursor line as dirty.
        log::trace!(
            "move_down_one_line: Marking old line dirty. current_physical_y: {}, screen_height: {}",
            current_physical_y,
            self.screen.height
        );
        self.screen.mark_line_dirty(current_physical_y); // Mark old line

        // Mark the new cursor line as dirty if it changed.
        let (_, new_physical_y) = self
            .cursor_controller
            .physical_screen_pos(&self.current_screen_context());

        if current_physical_y != new_physical_y {
            log::trace!(
                "move_down_one_line: Marking new line dirty. new_physical_y: {}, screen_height: {}",
                new_physical_y,
                self.screen.height
            );
            self.screen.mark_line_dirty(new_physical_y); // Mark new line
        } else {
            log::trace!(
                "move_down_one_line: New physical y ({}) is same as current ({}), not marking new line again.",
                new_physical_y,
                current_physical_y
            );
        }
    }

    /// Handles CR (Carriage Return). Moves cursor to column 0 of the current line.
    fn carriage_return(&mut self) {
        self.cursor_wrap_next = false; // CR cancels wrap.
        self.cursor_controller.carriage_return();
    }

    /// Sets the active G-level (G0, G1, G2, G3) for character set mapping.
    fn set_g_level(&mut self, g_level: usize) {
        if g_level < self.active_charsets.len() {
            self.active_charset_g_level = g_level;
            trace!("Switched to G{} character set mapping.", g_level);
        } else {
            warn!("Attempted to set invalid G-level: {}", g_level);
        }
    }

    /// Designates a character set for one of the G-levels (G0-G3).
    fn designate_character_set(&mut self, g_set_index: usize, charset: CharacterSet) {
        if g_set_index < self.active_charsets.len() {
            self.active_charsets[g_set_index] = charset;
            trace!("Designated G{} to {:?}", g_set_index, charset);
        } else {
            warn!("Invalid G-set index for designate charset: {}", g_set_index);
        }
    }

    /// Handles IND (Index). Moves cursor down one line, scrolling if at bottom of region.
    fn index(&mut self) {
        self.cursor_wrap_next = false;
        let screen_ctx = self.current_screen_context();
        let (_, current_physical_y) = self.cursor_controller.physical_screen_pos(&screen_ctx);

        // If cursor is at the bottom of the scrolling region (or screen if no specific region).
        if current_physical_y == screen_ctx.scroll_bot {
            self.screen.scroll_up_serial(1); // Scroll content of region up.
        } else if current_physical_y < screen_ctx.height.saturating_sub(1) {
            // If not at the very bottom of the physical screen, move cursor down.
            self.cursor_controller.move_down(1, &screen_ctx);
        }
        // Mark lines dirty: original cursor line and new cursor line if different.
        self.screen.mark_line_dirty(current_physical_y);
        let (_, new_physical_y) = self
            .cursor_controller
            .physical_screen_pos(&self.current_screen_context());
        if current_physical_y != new_physical_y {
            self.screen.mark_line_dirty(new_physical_y);
        }
    }

    /// Handles RI (Reverse Index). Moves cursor up one line, scrolling if at top of region.
    fn reverse_index(&mut self) {
        self.cursor_wrap_next = false;
        let screen_ctx = self.current_screen_context();
        let (_, current_physical_y) = self.cursor_controller.physical_screen_pos(&screen_ctx);

        // If cursor is at the top of the scrolling region (or screen).
        if current_physical_y == screen_ctx.scroll_top {
            self.screen.scroll_down_serial(1); // Scroll content of region down.
        } else if current_physical_y > 0 {
            // If not at the very top of the physical screen, move cursor up.
            self.cursor_controller.move_up(1);
        }
        // Mark lines dirty.
        self.screen.mark_line_dirty(current_physical_y);
        let (_, new_physical_y) = self
            .cursor_controller
            .physical_screen_pos(&self.current_screen_context());
        if current_physical_y != new_physical_y {
            self.screen.mark_line_dirty(new_physical_y);
        }
    }

    /// Saves cursor state (DECSC).
    fn save_cursor_dec(&mut self) {
        self.cursor_controller.save_state();
    }

    /// Restores cursor state (DECRC).
    fn restore_cursor_dec(&mut self) {
        self.cursor_wrap_next = false; // Restoring cursor usually cancels wrap.
        // Restore using current screen context and current default attributes as fallback.
        self.cursor_controller
            .restore_state(&self.current_screen_context(), Attributes::default());
        // Ensure the screen's default attributes are also updated to the restored cursor's attributes.
        self.screen.default_attributes = self.cursor_controller.attributes();
    }

    // --- CSI Handler Implementations ---
    // These methods correspond to specific CSI command variants.

    /// Moves cursor up N lines (CUU).
    fn cursor_up(&mut self, n: usize) {
        self.cursor_wrap_next = false;
        self.cursor_controller.move_up(n);
    }

    /// Moves cursor down N lines (CUD).
    /// This is implemented by calling `perform_index` N times, which handles scrolling.
    fn cursor_down(&mut self, n: usize) {
        self.cursor_wrap_next = false;
        log::trace!("cursor_down: n = {}", n);
        for i in 0..n {
            log::trace!("cursor_down: iteration {} calling perform_index", i);
            self.index();
        }
    }

    /// Moves cursor forward N columns (CUF).
    fn cursor_forward(&mut self, n: usize) {
        self.cursor_wrap_next = false;
        self.cursor_controller
            .move_right(n, &self.current_screen_context());
    }

    /// Moves cursor backward N columns (CUB).
    fn cursor_backward(&mut self, n: usize) {
        self.cursor_wrap_next = false;
        self.cursor_controller.move_left(n);
    }

    /// Moves cursor to column (CHA). 1-based parameter.
    fn cursor_to_column(&mut self, col: usize) {
        self.cursor_wrap_next = false;
        self.cursor_controller
            .move_to_logical_col(col, &self.current_screen_context());
    }

    /// Moves cursor to position (row, col) (CUP). 1-based parameters.
    fn cursor_to_pos(&mut self, row_param: usize, col_param: usize) {
        self.cursor_wrap_next = false;
        self.cursor_controller.move_to_logical(
            col_param, // col is x
            row_param, // row is y
            &self.current_screen_context(),
        );
    }

    /// Erases parts of the display (ED).
    fn erase_in_display(&mut self, mode: EraseMode) {
        self.cursor_wrap_next = false;
        let screen_ctx = self.current_screen_context();
        let (cx_phys, cy_phys) = self.cursor_controller.physical_screen_pos(&screen_ctx);
        // Ensure erase operations use the currently active SGR attributes for filling.
        self.screen.default_attributes = self.cursor_controller.attributes();

        match mode {
            EraseMode::ToEnd => {
                // Clear from cursor to end of current line.
                self.screen
                    .clear_line_segment(cy_phys, cx_phys, screen_ctx.width);
                // Clear all lines below the current line.
                for y in (cy_phys + 1)..screen_ctx.height {
                    self.screen.clear_line_segment(y, 0, screen_ctx.width);
                }
            }
            EraseMode::ToStart => {
                // Clear all lines above the current line.
                for y in 0..cy_phys {
                    self.screen.clear_line_segment(y, 0, screen_ctx.width);
                }
                // Clear from start of current line to cursor (inclusive).
                self.screen.clear_line_segment(cy_phys, 0, cx_phys + 1);
            }
            EraseMode::All => {
                // Clear entire screen.
                for y in 0..screen_ctx.height {
                    self.screen.clear_line_segment(y, 0, screen_ctx.width);
                }
            }
            EraseMode::Scrollback => {
                // Clear scrollback buffer only.
                self.screen.scrollback.clear();
                return; // No screen lines are dirtied by this.
            }
            EraseMode::Unknown => warn!("Unknown ED mode used."),
        }
        // ED operations (except Scrollback) dirty the screen.
        if mode != EraseMode::Scrollback {
            self.screen.mark_all_dirty(); // Mark all lines dirty as ED affects large areas.
        }
    }

    /// Erases parts of the current line (EL).
    fn erase_in_line(&mut self, mode: EraseMode) {
        self.cursor_wrap_next = false;
        let screen_ctx = self.current_screen_context();
        let (cx_phys, cy_phys) = self.cursor_controller.physical_screen_pos(&screen_ctx);
        // Ensure erase operations use the currently active SGR attributes.
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
        // EL operations dirty the current line.
        // screen.clear_line_segment already marks the line dirty.
    }

    /// Erases N characters from cursor position (ECH).
    fn erase_chars(&mut self, n: usize) {
        self.cursor_wrap_next = false;
        let screen_ctx = self.current_screen_context();
        let (cx_phys, cy_phys) = self.cursor_controller.physical_screen_pos(&screen_ctx);
        let end_x = min(cx_phys + n, screen_ctx.width); // Don't erase past end of line.
        // Ensure erase operations use the currently active SGR attributes.
        self.screen.default_attributes = self.cursor_controller.attributes();
        self.screen.clear_line_segment(cy_phys, cx_phys, end_x);
        // screen.clear_line_segment marks the line dirty.
    }

    /// Inserts N blank characters at cursor (ICH).
    fn insert_blank_chars(&mut self, n: usize) {
        self.cursor_wrap_next = false;
        let screen_ctx = self.current_screen_context();
        let (cx_log, _) = self.cursor_controller.logical_pos(); // Use logical X for insertion point
        let (_, cy_phys) = self.cursor_controller.physical_screen_pos(&screen_ctx);
        // Ensure inserted blanks use current SGR attributes.
        self.screen.default_attributes = self.cursor_controller.attributes();
        self.screen.insert_blank_chars_in_line(cy_phys, cx_log, n);
        // screen.insert_blank_chars_in_line marks the line dirty.
    }

    /// Deletes N characters at cursor (DCH).
    fn delete_chars(&mut self, n: usize) {
        self.cursor_wrap_next = false;
        let screen_ctx = self.current_screen_context();
        let (cx_log, _) = self.cursor_controller.logical_pos(); // Use logical X for deletion point
        let (_, cy_phys) = self.cursor_controller.physical_screen_pos(&screen_ctx);
        // Ensure newly exposed cells use current SGR attributes.
        self.screen.default_attributes = self.cursor_controller.attributes();
        self.screen.delete_chars_in_line(cy_phys, cx_log, n);
        // screen.delete_chars_in_line marks the line dirty.
    }

    /// Inserts N blank lines at cursor (IL).
    fn insert_lines(&mut self, n: usize) {
        self.cursor_wrap_next = false;
        let screen_ctx = self.current_screen_context();
        let (_, cy_phys) = self.cursor_controller.physical_screen_pos(&screen_ctx);
        // Ensure new lines use current SGR attributes.
        self.screen.default_attributes = self.cursor_controller.attributes();

        // IL operates within the scrolling margins if cursor is inside them.
        if cy_phys >= screen_ctx.scroll_top && cy_phys <= screen_ctx.scroll_bot {
            // Temporarily adjust scrolling region for the operation, as per VT102 manual (DECSTD003)
            // The region for IL/DL is from current line to bottom margin.
            let original_scroll_top = self.screen.scroll_top();
            let original_scroll_bottom = self.screen.scroll_bot();

            self.screen
                .set_scrolling_region(cy_phys + 1, original_scroll_bottom + 1); // 1-based for API
            self.screen.scroll_down_serial(n); // Scroll down this sub-region

            // Restore original scrolling region
            self.screen
                .set_scrolling_region(original_scroll_top + 1, original_scroll_bottom + 1);

            // Mark all lines from cursor to bottom of original region as dirty.
            for y_dirty in cy_phys..=original_scroll_bottom {
                self.screen.mark_line_dirty(y_dirty);
            }
        }
    }

    /// Deletes N lines at cursor (DL).
    fn delete_lines(&mut self, n: usize) {
        self.cursor_wrap_next = false;
        let screen_ctx = self.current_screen_context();
        let (_, cy_phys) = self.cursor_controller.physical_screen_pos(&screen_ctx);
        // Ensure new lines use current SGR attributes.
        self.screen.default_attributes = self.cursor_controller.attributes();

        // DL operates within the scrolling margins if cursor is inside them.
        if cy_phys >= screen_ctx.scroll_top && cy_phys <= screen_ctx.scroll_bot {
            // Similar to IL, adjust scrolling region for the operation.
            let original_scroll_top = self.screen.scroll_top();
            let original_scroll_bottom = self.screen.scroll_bot();

            self.screen
                .set_scrolling_region(cy_phys + 1, original_scroll_bottom + 1); // 1-based for API
            self.screen.scroll_up_serial(n); // Scroll up this sub-region

            // Restore original scrolling region
            self.screen
                .set_scrolling_region(original_scroll_top + 1, original_scroll_bottom + 1);
            // Mark all lines from cursor to bottom of original region as dirty.
            for y_dirty in cy_phys..=original_scroll_bottom {
                self.screen.mark_line_dirty(y_dirty);
            }
        }
    }

    /// Scrolls display up N lines (SU).
    fn scroll_up(&mut self, n: usize) {
        self.cursor_wrap_next = false;
        // Ensure new lines use current SGR attributes.
        self.screen.default_attributes = self.cursor_controller.attributes();
        self.screen.scroll_up_serial(n); // Uses screen's current scroll_top/bot
    }

    /// Scrolls display down N lines (SD).
    fn scroll_down(&mut self, n: usize) {
        self.cursor_wrap_next = false;
        // Ensure new lines use current SGR attributes.
        self.screen.default_attributes = self.cursor_controller.attributes();
        self.screen.scroll_down_serial(n); // Uses screen's current scroll_top/bot
    }

    /// Handles SGR (Select Graphic Rendition) attributes.
    fn handle_sgr_attributes(&mut self, attributes_vec: Vec<Attribute>) {
        let mut current_attrs = self.cursor_controller.attributes();
        for attr_cmd in attributes_vec {
            match attr_cmd {
                Attribute::Reset => current_attrs = Attributes::default(), // Reset to default
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
                // SGR 21 (Doubly underlined) is often treated as single underline by terminals.
                Attribute::UnderlineDouble => current_attrs.flags.insert(AttrFlags::UNDERLINE),
                Attribute::NoBold => {
                    // SGR 22: Normal intensity (neither bold nor faint).
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
                    // `color` is already `crate::color::Color` due to parser update.
                    current_attrs.fg = color;
                }
                Attribute::Background(color) => {
                    // `color` is already `crate::color::Color`.
                    current_attrs.bg = color;
                }
                Attribute::Overlined => warn!("SGR Overlined not yet visually supported."), // Typically SGR 53
                Attribute::NoOverlined => warn!("SGR NoOverlined not yet visually supported."), // Typically SGR 55
                Attribute::UnderlineColor(color) => {
                    // SGR 58 ; 2 ; r;g;b m  OR SGR 58 ; 5 ; idx m
                    // This attribute is complex; for now, we just log it.
                    // A full implementation would store this underline color separately.
                    warn!("SGR UnderlineColor not yet fully supported: {:?}", color)
                }
            }
        }
        self.cursor_controller.set_attributes(current_attrs);
        // Crucially, update the screen's default_attributes for clearing operations.
        self.screen.default_attributes = current_attrs;
    }

    /// Handles SM (Set Mode) and RM (Reset Mode) sequences for ANSI and DEC private modes.
    fn handle_set_mode(&mut self, mode_type: Mode, enable: bool) -> Option<EmulatorAction> {
        self.cursor_wrap_next = false; // Mode changes usually reset wrap state.
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
                        self.screen.origin_mode = enable; // Sync screen's understanding
                        // When DECOM is set/reset, cursor moves to (0,0) within the new context.
                        self.cursor_controller.move_to_logical(
                            0,
                            0,
                            &self.current_screen_context(),
                        );
                    }
                    Some(DecModeConstant::TextCursorEnable) => {
                        // DECTCEM - Show/Hide Cursor
                        self.dec_modes.text_cursor_enable_mode = enable;
                        self.cursor_controller.set_visible(enable);
                        action_to_return = Some(EmulatorAction::SetCursorVisibility(enable));
                    }
                    Some(DecModeConstant::AltScreenBufferClear)
                    | Some(DecModeConstant::AltScreenBufferSaveRestore) => {
                        if !self.dec_modes.allow_alt_screen {
                            warn!(
                                "Alternate screen disabled by configuration, ignoring mode {}.",
                                mode_num
                            );
                            return None; // Do nothing if alt screen is disallowed
                        }
                        // Determine if the alternate screen should be cleared upon entry.
                        // 1047 and 1049 both clear.
                        let clear_on_entry = mode_num
                            == DecModeConstant::AltScreenBufferClear as u16
                            || mode_num == DecModeConstant::AltScreenBufferSaveRestore as u16;

                        if enable {
                            // Entering alternate screen
                            if !self.dec_modes.using_alt_screen {
                                // Save cursor state if mode is 1049 (or 1048, though 1048 is just save/restore)
                                if mode_num == DecModeConstant::AltScreenBufferSaveRestore as u16 {
                                    self.save_cursor_dec();
                                }
                                // Ensure new alt screen uses current attributes for clearing
                                self.screen.default_attributes =
                                    self.cursor_controller.attributes();
                                self.screen.enter_alt_screen(clear_on_entry);
                                self.dec_modes.using_alt_screen = true;
                                // Cursor moves to (0,0) on alt screen.
                                self.cursor_controller.move_to_logical(
                                    0,
                                    0,
                                    &self.current_screen_context(),
                                );
                                self.mark_all_lines_dirty(); // Entire screen content changed
                                action_to_return = Some(EmulatorAction::RequestRedraw);
                            }
                        } else if self.dec_modes.using_alt_screen {
                            // Exiting alternate screen
                            self.screen.exit_alt_screen();
                            self.dec_modes.using_alt_screen = false;
                            // Restore cursor state if mode was 1049 (or 1048)
                            if mode_num == DecModeConstant::AltScreenBufferSaveRestore as u16 {
                                self.restore_cursor_dec(); // This updates screen.default_attributes
                            } else {
                                // For 1047, cursor typically goes to (0,0) on primary screen.
                                self.cursor_controller.move_to_logical(
                                    0,
                                    0,
                                    &self.current_screen_context(),
                                );
                                // Ensure screen default attributes match current cursor for primary screen.
                                self.screen.default_attributes =
                                    self.cursor_controller.attributes();
                            }
                            self.mark_all_lines_dirty(); // Entire screen content changed
                            action_to_return = Some(EmulatorAction::RequestRedraw);
                        }
                    }
                    Some(DecModeConstant::SaveRestoreCursor) => {
                        // DECSC/DECRC (often 1048) - just save/restore cursor state
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
                    // Mouse modes: update the flags. Orchestrator/Driver might use these.
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
                        // urxvt extended mouse mode (?1015)
                        warn!(
                            "DEC Private Mode {} (MouseUrxvt) set to {} - not fully implemented.",
                            mode_num, enable
                        );
                        // Potentially set a flag if this mode needs specific handling.
                    }
                    Some(DecModeConstant::MousePixelPosition) => {
                        // SGR Pixel Positioning Mouse Mode (?1016)
                        warn!(
                            "DEC Private Mode {} (MousePixelPosition) set to {} - not fully implemented.",
                            mode_num, enable
                        );
                    }
                    Some(DecModeConstant::Att610CursorBlink) => {
                        // This mode (often ?12l/h) controls cursor blinking.
                        // The visual effect is handled by the renderer based on this state.
                        self.dec_modes.cursor_blink_mode = enable;
                        warn!(
                            "DEC Private Mode 12 (ATT610 Cursor Blink) set to {}. Visual blink not implemented.",
                            enable
                        );
                        // No immediate EmulatorAction needed, renderer will query state.
                    }
                    Some(DecModeConstant::Unknown7727) => {
                        // A mode seen in logs, specific behavior might be unknown.
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
                // Handle standard ANSI modes (SM/RM)
                match mode_num {
                    4 => {
                        // IRM - Insert/Replace Mode
                        self.dec_modes.insert_mode = enable;
                    }
                    20 => {
                        // LNM - Linefeed/Newline Mode
                        self.dec_modes.linefeed_newline_mode = enable;
                    }
                    // Add other standard modes as needed (e.g., KAM, HEM, PUM, SRM, etc.)
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

    /// Handles Window Manipulation sequences (often from dtterm or xterm).
    /// CSI Ps ; Ps ; Ps t
    fn handle_window_manipulation(
        &mut self,
        ps1: u16,
        _ps2: Option<u16>, // Often used for row/height
        _ps3: Option<u16>, // Often used for col/width
    ) -> Option<EmulatorAction> {
        match ps1 {
            14 => {
                // Report text area size in pixels.
                // This would require getting pixel dimensions from the driver.
                // For now, we log it as not implemented.
                warn!(
                    "WindowManipulation: Report text area size in pixels (14) requested, but not implemented."
                );
                None
            }
            18 => {
                // Report text area size in characters (rows;cols).
                // Format: CSI 8 ; <rows> ; <cols> t
                let (cols, rows) = self.dimensions();
                let response = format!("\x1b[8;{};{}t", rows, cols);
                Some(EmulatorAction::WritePty(response.into_bytes()))
            }
            22 | 23 => {
                // Save (22) / Restore (23) window title from internal stack.
                // Not typically implemented in basic terminals.
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

    /// Handles OSC (Operating System Command) sequences.
    /// Primarily used for setting window/icon titles.
    fn handle_osc(&mut self, data: Vec<u8>) -> Option<EmulatorAction> {
        // OSC sequences are typically of the form: OSC Ps ; Pt ST
        // Ps is a numeric parameter, Pt is the string. ST is the string terminator (BEL or ESC \)
        // For simplicity, we parse up to the first ';'.
        let osc_str = String::from_utf8_lossy(&data);
        let parts: Vec<&str> = osc_str.splitn(2, ';').collect();

        if parts.len() == 2 {
            let ps = parts[0].parse::<u32>().unwrap_or(u32::MAX); // Use u32::MAX for unparsed
            let pt = parts[1].to_string(); // The rest is the string parameter

            match ps {
                0 | 2 => {
                    // OSC 0 ; Pt ST: Set icon name and window title.
                    // OSC 2 ; Pt ST: Set window title.
                    // We'll treat both as setting the window title.
                    return Some(EmulatorAction::SetTitle(pt));
                }
                // Add other OSC commands as needed (e.g., color setting OSC 4, 10, 11, 12)
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
