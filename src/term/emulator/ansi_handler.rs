// src/term/emulator/ansi_handler.rs

use super::TerminalEmulator;
use crate::{
    ansi::commands::{AnsiCommand, C0Control, CsiCommand, EscCommand},
    term::{
        action::EmulatorAction,
        charset::CharacterSet,
        cursor::CursorShape,
        emulator::FocusState,
        modes::{EraseMode, Mode},
        screen::TabClearMode,
    },
};

use log::{debug, warn}; // Assuming logging is still desired

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
                emulator.backspace();
                None
            }
            C0Control::HT => {
                emulator.horizontal_tab();
                None
            }
            C0Control::LF | C0Control::VT | C0Control::FF => {
                emulator.line_feed();
                None
            }
            C0Control::CR => {
                emulator.carriage_return();
                None
            }
            C0Control::SO => {
                emulator.set_g_level(1);
                None
            }
            C0Control::SI => {
                emulator.set_g_level(0);
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
                emulator.index();
                None
            }
            EscCommand::NextLine => {
                emulator.carriage_return();
                emulator.line_feed();
                None
            }
            EscCommand::ReverseIndex => {
                emulator.reverse_index();
                None
            }
            EscCommand::SaveCursor => {
                emulator.save_cursor();
                None
            }
            EscCommand::RestoreCursor => {
                emulator.restore_cursor();
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
                emulator.designate_character_set(g_idx, CharacterSet::from_char(final_char));
                None
            }
            EscCommand::ResetToInitialState => emulator.reset(),
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
            CsiCommand::Reset => emulator.reset(),
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
