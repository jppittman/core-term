// src/term/emulator/ansi_handler.rs

use super::TerminalEmulator;
use crate::{
    ansi::commands::{AnsiCommand, C0Control, CsiCommand, EscCommand},
    glyph::Attributes,
    term::{
        action::EmulatorAction,
        charset::CharacterSet,
        cursor::CursorShape,
        emulator::FocusState,
        modes::{DecPrivateModes, EraseMode, Mode},
        screen::TabClearMode,
        DEFAULT_TAB_INTERVAL,
    }, // For Attributes::default() in ResetToInitialState
};

use log::{debug, trace, warn}; // Assuming logging is still desired

// --- Helper functions moved from TerminalEmulator ---
// These are now private to the ansi_handler module, or are methods on TerminalEmulator called via `emulator.`
// They take `emulator: &mut TerminalEmulator` as their first argument if they are local helpers.

fn backspace(emulator: &mut TerminalEmulator) {
    emulator.cursor_wrap_next = false;
    emulator.cursor_controller.move_left(1);
}

fn horizontal_tab(emulator: &mut TerminalEmulator) {
    emulator.cursor_wrap_next = false;
    let (current_x, _) = emulator.cursor_controller.logical_pos();
    let screen_ctx = emulator.current_screen_context();
    let next_stop = emulator
        .screen
        .get_next_tabstop(current_x)
        .unwrap_or(screen_ctx.width.saturating_sub(1).max(current_x));
    emulator
        .cursor_controller
        .move_to_logical_col(next_stop, &screen_ctx);
}

fn perform_line_feed(emulator: &mut TerminalEmulator) {
    log::trace!("perform_line_feed called in ansi_handler");
    emulator.move_down_one_line_and_dirty(); // Call as method
    if emulator.dec_modes.linefeed_newline_mode { // Check LNM mode
        emulator.carriage_return(); // Call as method
    }
}

// move_down_one_line_and_dirty is now a method on TerminalEmulator in methods.rs
// carriage_return is now a method on TerminalEmulator in methods.rs

fn set_g_level(emulator: &mut TerminalEmulator, g_level: usize) {
    if g_level < emulator.active_charsets.len() {
        emulator.active_charset_g_level = g_level;
        trace!("Switched to G{} character set mapping.", g_level);
    } else {
        warn!("Attempted to set invalid G-level: {}", g_level);
    }
}

// index is now a method on TerminalEmulator in methods.rs

fn reverse_index(emulator: &mut TerminalEmulator) {
    emulator.cursor_wrap_next = false;
    let screen_ctx = emulator.current_screen_context();
    let (_, current_physical_y) = emulator.cursor_controller.physical_screen_pos(&screen_ctx);

    if current_physical_y == screen_ctx.scroll_top {
        emulator.screen.scroll_down_serial(1);
    } else if current_physical_y > 0 {
        emulator.cursor_controller.move_up(1);
    }
    if current_physical_y < emulator.screen.height {
        emulator.screen.mark_line_dirty(current_physical_y);
    }
    let (_, new_physical_y) = emulator
        .cursor_controller
        .physical_screen_pos(&emulator.current_screen_context());
    if current_physical_y != new_physical_y && new_physical_y < emulator.screen.height {
        emulator.screen.mark_line_dirty(new_physical_y);
    }
}

// save_cursor_dec is now a method on TerminalEmulator in methods.rs
// restore_cursor_dec is now a method on TerminalEmulator in methods.rs

fn designate_character_set(
    emulator: &mut TerminalEmulator,
    g_set_index: usize,
    charset: CharacterSet,
) {
    if g_set_index < emulator.active_charsets.len() {
        emulator.active_charsets[g_set_index] = charset;
        trace!("Designated G{} to {:?}", g_set_index, charset);
    } else {
        warn!("Invalid G-set index for designate charset: {}", g_set_index);
    }
}

// --- CSI Handler Small Helpers ---
// These functions (cursor_up, cursor_down, cursor_forward, cursor_backward,
// cursor_to_column, cursor_to_pos) have been moved to cursor_handler.rs
// and are now methods on TerminalEmulator.

fn reset(emulator: &mut TerminalEmulator) -> Option<EmulatorAction> {
    if emulator.screen.alt_screen_active {
        emulator.screen.exit_alt_screen();
    }
    let default_attrs = Attributes::default();
    emulator.cursor_controller.reset();
    emulator.screen.default_attributes = default_attrs;
    emulator.erase_in_display(EraseMode::All); // Call as method on emulator
    emulator.dec_modes = DecPrivateModes::default();
    emulator.screen.origin_mode = emulator.dec_modes.origin_mode;
    let (_, h) = emulator.dimensions();
    emulator.screen.set_scrolling_region(1, h);
    emulator.active_charsets = [CharacterSet::Ascii; 4];
    emulator.active_charset_g_level = 0;
    emulator.screen.clear_tabstops(0, TabClearMode::All);
    let (w, _) = emulator.dimensions();
    for i in (DEFAULT_TAB_INTERVAL as usize..w).step_by(DEFAULT_TAB_INTERVAL as usize) {
        emulator.screen.set_tabstop(i);
    }
    emulator.cursor_wrap_next = false;
    if emulator.dec_modes.text_cursor_enable_mode {
        return Some(EmulatorAction::SetCursorVisibility(true));
    }
    None
}

// The main ANSI command processing function
#[allow(clippy::too_many_lines)] // To be addressed by further refactoring if needed
pub(super) fn process_ansi_command(
    emulator: &mut TerminalEmulator,
    command: AnsiCommand,
) -> Option<EmulatorAction> {
    if !matches!(command, AnsiCommand::Print(_)) {
        emulator.cursor_wrap_next = false;
    }

    match command {
        AnsiCommand::C0Control(c0) => match c0 {
            C0Control::BS => {
                backspace(emulator);
                None
            }
            C0Control::HT => {
                horizontal_tab(emulator);
                None
            }
            C0Control::LF | C0Control::VT | C0Control::FF => {
                perform_line_feed(emulator); // This local fn calls emulator.methods
                None
            }
            C0Control::CR => {
                emulator.carriage_return(); // Call as method
                None
            }
            C0Control::SO => {
                set_g_level(emulator, 1);
                None
            }
            C0Control::SI => {
                set_g_level(emulator, 0);
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
                let (cursor_x, _) = emulator.cursor_controller.logical_pos();
                emulator.screen.set_tabstop(cursor_x);
                None
            }
            EscCommand::Index => {
                emulator.index(); // Call as method
                None
            }
            EscCommand::NextLine => {
                emulator.carriage_return(); // Call as method
                perform_line_feed(emulator); // This local fn calls emulator.methods
                None
            }
            EscCommand::ReverseIndex => {
                reverse_index(emulator); // This local fn can stay if only used here
                None
            }
            EscCommand::SaveCursor => {
                emulator.save_cursor(); // Call as method
                None
            }
            EscCommand::RestoreCursor => {
                emulator.restore_cursor(); // Call as method
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
                designate_character_set(emulator, g_idx, CharacterSet::from_char(final_char));
                None
            }
            EscCommand::ResetToInitialState => reset(emulator),
            _ => {
                debug!("Unhandled Esc command: {:?}", esc_cmd);
                None
            }
        },
        AnsiCommand::Csi(csi) => match csi {
            CsiCommand::CursorUp(n) => {
                emulator.cursor_up(n.max(1) as usize);
                None
            }
            CsiCommand::CursorDown(n) => {
                emulator.cursor_down(n.max(1) as usize);
                None
            }
            CsiCommand::CursorForward(n) => {
                emulator.cursor_forward(n.max(1) as usize);
                None
            }
            CsiCommand::CursorBackward(n) => {
                emulator.cursor_backward(n.max(1) as usize);
                None
            }
            CsiCommand::CursorNextLine(n) => {
                emulator.cursor_down(n.max(1) as usize);
                emulator.carriage_return(); // Call as method
                None
            }
            CsiCommand::CursorPrevLine(n) => {
                emulator.cursor_up(n.max(1) as usize);
                emulator.carriage_return(); // Call as method
                None
            }
            CsiCommand::CursorCharacterAbsolute(n) => {
                emulator.cursor_to_column(n.saturating_sub(1) as usize);
                None
            }
            CsiCommand::CursorPosition(r, c) => {
                emulator.cursor_to_pos(r.saturating_sub(1) as usize, c.saturating_sub(1) as usize);
                None
            }
            CsiCommand::EraseInDisplay(mode_val) => {
                emulator.erase_in_display(EraseMode::from(mode_val)); // Call as method
                None
            }
            CsiCommand::EraseInLine(mode_val) => {
                emulator.erase_in_line(EraseMode::from(mode_val)); // Call as method
                None
            }
            CsiCommand::InsertCharacter(n) => {
                emulator.insert_blank_chars(n.max(1) as usize); // Call as method
                None
            }
            CsiCommand::DeleteCharacter(n) => {
                emulator.delete_chars(n.max(1) as usize); // Call as method
                None
            }
            CsiCommand::InsertLine(n) => {
                emulator.insert_lines(n.max(1) as usize); // Call as method
                None
            }
            CsiCommand::DeleteLine(n) => {
                emulator.delete_lines(n.max(1) as usize); // Call as method
                None
            }
            CsiCommand::SetGraphicsRendition(attrs_vec) => {
                emulator.handle_sgr_attributes(attrs_vec); // Call as method
                None
            }
            CsiCommand::SetMode(mode_num) => {
                emulator.handle_set_mode(Mode::Standard(mode_num), true) // Call as method
            }
            CsiCommand::ResetMode(mode_num) => {
                emulator.handle_set_mode(Mode::Standard(mode_num), false) // Call as method
            }
            CsiCommand::SetModePrivate(mode_num) => {
                emulator.handle_set_mode(Mode::DecPrivate(mode_num), true) // Call as method
            }
            CsiCommand::ResetModePrivate(mode_num) => {
                emulator.handle_set_mode(Mode::DecPrivate(mode_num), false) // Call as method
            }
            CsiCommand::DeviceStatusReport(dsr_param) => {
                if dsr_param == 0 || dsr_param == 6 {
                    let screen_ctx = emulator.current_screen_context();
                    let (abs_x, abs_y) =
                        emulator.cursor_controller.physical_screen_pos(&screen_ctx);
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
                emulator.erase_chars(n.max(1) as usize); // Call as method
                None
            }
            CsiCommand::ScrollUp(n) => {
                emulator.scroll_up(n.max(1) as usize); // Call as method
                None
            }
            CsiCommand::ScrollDown(n) => {
                emulator.scroll_down(n.max(1) as usize); // Call as method
                None
            }
            CsiCommand::SaveCursor | CsiCommand::SaveCursorAnsi => {
                emulator.save_cursor();
                None
            }
            CsiCommand::RestoreCursor | CsiCommand::RestoreCursorAnsi => {
                emulator.restore_cursor();
                None
            }
            CsiCommand::ClearTabStops(mode_val) => {
                let (cursor_x, _) = emulator.cursor_controller.logical_pos();
                emulator
                    .screen
                    .clear_tabstops(cursor_x, TabClearMode::from(mode_val));
                None
            }
            CsiCommand::SetScrollingRegion { top, bottom } => {
                emulator
                    .screen
                    .set_scrolling_region(top as usize, bottom as usize);
                emulator.cursor_controller.move_to_logical(
                    0,
                    0,
                    &emulator.current_screen_context(),
                );
                None
            }
            CsiCommand::SetCursorStyle { shape } => {
                let focus_state = emulator.focus_state;
                let cursor_shape = CursorShape::from_decscusr_code(shape);
                match focus_state {
                    FocusState::Focused => emulator.cursor_controller.cursor.shape = cursor_shape,
                    FocusState::Unfocused => {
                        emulator.cursor_controller.cursor.unfocused_shape = cursor_shape
                    }
                }
                debug!("Set cursor style to shape: {}", shape);
                None
            }
            CsiCommand::WindowManipulation { ps1, ps2, ps3 } => {
                emulator.handle_window_manipulation(ps1, ps2, ps3)
            }
            CsiCommand::PrimaryDeviceAttributes => {
                let response = "\x1b[?6c".to_string().into_bytes();
                Some(EmulatorAction::WritePty(response))
            }
            CsiCommand::SetTabStop => {
                todo!("ansi set tabstop");
            }
            CsiCommand::Reset => reset(emulator),
            CsiCommand::Unsupported(intermediates, final_byte_opt) => {
                warn!(
                    "TerminalEmulator received CsiCommand::Unsupported: intermediates={:?}, final={:?}. This is usually an error from the parser.",
                    intermediates, final_byte_opt
                );
                None
            }
        },
        AnsiCommand::Osc(data) => emulator.handle_osc(data), // Call as method
        AnsiCommand::Print(ch) => {
            emulator.print_char(ch); // Call as method
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
