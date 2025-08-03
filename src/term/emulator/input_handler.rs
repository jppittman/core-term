// src/term/emulator/input_handler.rs

use super::{key_translator, FocusState, TerminalEmulator};
use crate::term::{
    action::{EmulatorAction, UserInputAction},
    snapshot::{Point, SelectionMode},
    ControlEvent,
};
use log::{debug, trace};

#[allow(clippy::too_many_lines)]
pub(super) fn process_user_input_action(
    emulator: &mut TerminalEmulator,
    action: UserInputAction,
) -> Option<EmulatorAction> {
    emulator.cursor_wrap_next = false;

    match action {
        UserInputAction::FocusLost => emulator.focus_state = FocusState::Unfocused,
        UserInputAction::FocusGained => emulator.focus_state = FocusState::Focused,
        UserInputAction::KeyInput {
            symbol,
            modifiers,
            text,
        } => {
            let bytes_to_send =
                key_translator::translate_key_input(symbol, modifiers, text, &emulator.dec_modes);
            if !bytes_to_send.is_empty() {
                return Some(EmulatorAction::WritePty(bytes_to_send));
            }
        }
        UserInputAction::StartSelection { x, y } => {
            emulator.start_selection(Point { x, y }, SelectionMode::Cell);
            return Some(EmulatorAction::RequestRedraw);
        }
        UserInputAction::ExtendSelection { x, y } => {
            emulator.extend_selection(Point { x, y });
            return Some(EmulatorAction::RequestRedraw);
        }
        UserInputAction::ApplySelectionClear => {
            emulator.apply_selection_clear();
            return Some(EmulatorAction::RequestRedraw);
        }
        UserInputAction::RequestClipboardPaste => {
            debug!(
                "UserInputAction: RequestClipboardPaste received. Requesting clipboard content."
            );
            return Some(EmulatorAction::RequestClipboardContent);
        }
        UserInputAction::RequestPrimaryPaste => {
            debug!("UserInputAction: RequestPrimaryPaste received. (Currently not fully implemented, forwarding to RequestClipboardContent)");
            return Some(EmulatorAction::RequestClipboardContent);
        }
        UserInputAction::InitiateCopy => {
            if let Some(text) = emulator.get_selected_text() {
                if !text.is_empty() {
                    return Some(EmulatorAction::CopyToClipboard(text));
                }
            }
            debug!("UserInputAction: InitiateCopy called but no text selected or selection empty.");
        }
        UserInputAction::PasteText(text_to_paste) => {
            if emulator.dec_modes.bracketed_paste_mode {
                log::debug!("InputHandler: Bracketed paste mode ON. Wrapping and sending to PTY.");
                let mut pasted_bytes = Vec::new();
                pasted_bytes.extend_from_slice(b"\x1b[200~");
                pasted_bytes.extend_from_slice(text_to_paste.as_bytes());
                pasted_bytes.extend_from_slice(b"\x1b[201~");
                return Some(EmulatorAction::WritePty(pasted_bytes));
            } else {
                log::debug!("InputHandler: Bracketed paste mode OFF. Calling emulator.paste_text.");
                for char_val in text_to_paste.chars() {
                    // Iterate over chars and process them
                    emulator.print_char(char_val);
                }
                return Some(EmulatorAction::RequestRedraw);
            }
        }
        // Add catch-all for other UserInputAction variants to satisfy exhaustiveness
        _ => {
            log::debug!(
                "Unhandled UserInputAction variant in input_handler: {:?}",
                action
            );
        }
    }
    None
}

pub(super) fn process_control_event(
    emulator: &mut TerminalEmulator,
    event: ControlEvent,
) -> Option<EmulatorAction> {
    emulator.cursor_wrap_next = false;
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
            emulator.resize(cols, rows);
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::term::emulator::TerminalEmulator;

    fn create_test_emu_for_input() -> TerminalEmulator {
        TerminalEmulator::new(80, 24)
    }

    #[test]
    fn test_paste_text_action_bracketed_on() {
        let mut emu = create_test_emu_for_input();
        emu.dec_modes.bracketed_paste_mode = true;

        let text_to_paste = "Hello\nWorld".to_string();
        let action = UserInputAction::PasteText(text_to_paste.clone());

        let result = process_user_input_action(&mut emu, action);

        let expected_bytes = format!("\x1b[200~{}\x1b[201~", text_to_paste).into_bytes();
        assert_eq!(result, Some(EmulatorAction::WritePty(expected_bytes)));
    }

    #[test]
    fn test_paste_text_action_bracketed_off() {
        let mut emu = create_test_emu_for_input();
        assert!(!emu.dec_modes.bracketed_paste_mode);

        let text_to_paste = "Hello\nWorld".to_string();
        let action = UserInputAction::PasteText(text_to_paste.clone());

        let result = process_user_input_action(&mut emu, action);
        assert_eq!(result, Some(EmulatorAction::RequestRedraw));

        let snapshot_option = emu.get_render_snapshot();
        let snapshot = snapshot_option.as_ref().expect("Snapshot was None");
        // Print the actual screen content for debugging
        for (r, line) in snapshot.lines.iter().enumerate() {
            let line_str: String = line
                .cells
                .iter()
                .map(|glyph_wrapper| match glyph_wrapper {
                    crate::glyph::Glyph::Single(cell) | crate::glyph::Glyph::WidePrimary(cell) => {
                        cell.c
                    }
                    crate::glyph::Glyph::WideSpacer { .. } => crate::glyph::WIDE_CHAR_PLACEHOLDER,
                })
                .collect();
            println!("Actual line {}: '{}'", r, line_str);
        }
        println!(
            "Actual cursor pos: {:?}",
            snapshot.cursor_state.as_ref().map(|cs| (cs.y, cs.x))
        );

        match snapshot.lines[0].cells[0] {
            crate::glyph::Glyph::Single(cell) | crate::glyph::Glyph::WidePrimary(cell) => {
                assert_eq!(cell.c, 'H')
            }
            _ => panic!("Expected H at [0][0]"),
        }
        match snapshot.lines[0].cells[1] {
            crate::glyph::Glyph::Single(cell) | crate::glyph::Glyph::WidePrimary(cell) => {
                assert_eq!(cell.c, 'e')
            }
            _ => panic!("Expected e at [0][1]"),
        }
        match snapshot.lines[0].cells[2] {
            crate::glyph::Glyph::Single(cell) | crate::glyph::Glyph::WidePrimary(cell) => {
                assert_eq!(cell.c, 'l')
            }
            _ => panic!("Expected l at [0][2]"),
        }
        match snapshot.lines[0].cells[3] {
            crate::glyph::Glyph::Single(cell) | crate::glyph::Glyph::WidePrimary(cell) => {
                assert_eq!(cell.c, 'l')
            }
            _ => panic!("Expected l at [0][3]"),
        }
        match snapshot.lines[0].cells[4] {
            crate::glyph::Glyph::Single(cell) | crate::glyph::Glyph::WidePrimary(cell) => {
                assert_eq!(cell.c, 'o')
            }
            _ => panic!("Expected o at [0][4]"),
        }
        match snapshot.lines[1].cells[0] {
            crate::glyph::Glyph::Single(cell) | crate::glyph::Glyph::WidePrimary(cell) => {
                assert_eq!(cell.c, 'W')
            }
            _ => panic!("Expected W at [1][0]"),
        }
    }
}
