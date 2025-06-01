// src/term/emulator/input_handler.rs

use super::{FocusState, TerminalEmulator};
use crate::keys::{KeySymbol, Modifiers};
use crate::term::{
    action::{EmulatorAction, UserInputAction}, // Removed MouseButton, MouseEventType
    snapshot::{Point, SelectionMode}, // Removed SelectionMode as it's handled in TerminalEmulator methods
    ControlEvent,
};
use log::{debug, trace}; // Added debug log

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
            let mut bytes_to_send: Vec<u8> = Vec::new();
            if let Some(txt_val) = &text {
                // Priority 1: Explicit text from driver.
                // Specific Ctrl+Alpha generation is needed if `text` doesn't already contain the control char.
                if modifiers.contains(Modifiers::CONTROL)
                    && matches!(symbol, KeySymbol::Char(c) if c.is_ascii_alphabetic())
                {
                    if let KeySymbol::Char(c_val) = symbol {
                        bytes_to_send.push((c_val.to_ascii_lowercase() as u8) - b'a' + 1);
                    }
                } else if modifiers.contains(Modifiers::CONTROL) && symbol == KeySymbol::Char('[') {
                    bytes_to_send.push(0x1b); // Ctrl+[ is ESC
                } else if modifiers.contains(Modifiers::CONTROL) && symbol == KeySymbol::Char('\\')
                {
                    bytes_to_send.push(0x1c); // Ctrl+\ is FS
                } else if modifiers.contains(Modifiers::CONTROL) && symbol == KeySymbol::Char(']') {
                    bytes_to_send.push(0x1d); // Ctrl+] is GS
                } else if modifiers.contains(Modifiers::CONTROL) && symbol == KeySymbol::Char('^') {
                    // Typically Ctrl+6 for US layout
                    bytes_to_send.push(0x1e); // Ctrl+^ is RS
                } else if modifiers.contains(Modifiers::CONTROL) && symbol == KeySymbol::Char('_') {
                    // Typically Ctrl+- for US layout
                    bytes_to_send.push(0x1f); // Ctrl+_ is US
                } else if modifiers.contains(Modifiers::CONTROL) && symbol == KeySymbol::Char(' ') {
                    bytes_to_send.push(0x00); // Ctrl+Space is NUL
                } else if modifiers.contains(Modifiers::CONTROL) && symbol == KeySymbol::Char('?') {
                    // This is tricky. Ctrl+? often generates DEL (0x7f) or depends on terminal settings.
                    // Or, if symbol is KeySymbol::Backspace and modifier is CONTROL, it might be DEL.
                    // For now, assuming if KeySymbol::Char('?') with CONTROL, it's DEL.
                    bytes_to_send.push(0x7f); // Ctrl+? is DEL
                } else {
                    bytes_to_send.extend(txt_val.as_bytes());
                }
            } else {
                // Priority 2: Text is None. Use KeySymbol and Modifiers to generate sequence.
                match symbol {
                    KeySymbol::Enter | KeySymbol::KeypadEnter => bytes_to_send.push(b'\r'), // Send CR
                    KeySymbol::Backspace => bytes_to_send.push(0x08),                       // BS
                    KeySymbol::Tab => {
                        if modifiers.contains(Modifiers::SHIFT) {
                            bytes_to_send.extend_from_slice(b"\x1b[Z"); // CSI Z (Shift+Tab)
                        } else {
                            bytes_to_send.push(b'\t'); // HT
                        }
                    }
                    KeySymbol::Escape => bytes_to_send.push(0x1B), // ESC

                    KeySymbol::Up => bytes_to_send.extend_from_slice(
                        if emulator.dec_modes.cursor_keys_app_mode {
                            b"\x1bOA"
                        } else {
                            b"\x1b[A"
                        },
                    ),
                    KeySymbol::Down => bytes_to_send.extend_from_slice(
                        if emulator.dec_modes.cursor_keys_app_mode {
                            b"\x1bOB"
                        } else {
                            b"\x1b[B"
                        },
                    ),
                    KeySymbol::Right => bytes_to_send.extend_from_slice(
                        if emulator.dec_modes.cursor_keys_app_mode {
                            b"\x1bOC"
                        } else {
                            b"\x1b[C"
                        },
                    ),
                    KeySymbol::Left => bytes_to_send.extend_from_slice(
                        if emulator.dec_modes.cursor_keys_app_mode {
                            b"\x1bOD"
                        } else {
                            b"\x1b[D"
                        },
                    ),

                    KeySymbol::Home => bytes_to_send.extend_from_slice(b"\x1b[1~"), // Standard: ESC [ 1 ~
                    KeySymbol::End => bytes_to_send.extend_from_slice(b"\x1b[4~"), // Standard: ESC [ 4 ~
                    KeySymbol::PageUp => bytes_to_send.extend_from_slice(b"\x1b[5~"),
                    KeySymbol::PageDown => bytes_to_send.extend_from_slice(b"\x1b[6~"),
                    KeySymbol::Insert => bytes_to_send.extend_from_slice(b"\x1b[2~"),
                    KeySymbol::Delete => bytes_to_send.extend_from_slice(b"\x1b[3~"),

                    // Common function key sequences (SS3 for F1-F4, CSI for others)
                    // These can vary; this is a common set.
                    KeySymbol::F1 => bytes_to_send.extend_from_slice(b"\x1bOP"), // SS3 P
                    KeySymbol::F2 => bytes_to_send.extend_from_slice(b"\x1bOQ"), // SS3 Q
                    KeySymbol::F3 => bytes_to_send.extend_from_slice(b"\x1bOR"), // SS3 R
                    KeySymbol::F4 => bytes_to_send.extend_from_slice(b"\x1bOS"), // SS3 S
                    KeySymbol::F5 => bytes_to_send.extend_from_slice(b"\x1b[15~"),
                    KeySymbol::F6 => bytes_to_send.extend_from_slice(b"\x1b[17~"),
                    KeySymbol::F7 => bytes_to_send.extend_from_slice(b"\x1b[18~"),
                    KeySymbol::F8 => bytes_to_send.extend_from_slice(b"\x1b[19~"),
                    KeySymbol::F9 => bytes_to_send.extend_from_slice(b"\x1b[20~"),
                    KeySymbol::F10 => bytes_to_send.extend_from_slice(b"\x1b[21~"),
                    KeySymbol::F11 => bytes_to_send.extend_from_slice(b"\x1b[23~"),
                    KeySymbol::F12 => bytes_to_send.extend_from_slice(b"\x1b[24~"),
                    // F13-F24 would follow similar CSI patterns if needed

                    // Keypad keys (application mode usually sends SS3 or specific codes,
                    // numeric mode sends digits/operators directly)
                    // This simplified version sends digits/operators directly for now.
                    // A full implementation would check emulator.dec_modes.keypad_app_mode
                    KeySymbol::KeypadPlus => bytes_to_send.push(b'+'),
                    KeySymbol::KeypadMinus => bytes_to_send.push(b'-'),
                    KeySymbol::KeypadMultiply => bytes_to_send.push(b'*'),
                    KeySymbol::KeypadDivide => bytes_to_send.push(b'/'),
                    KeySymbol::KeypadDecimal => bytes_to_send.push(b'.'),
                    KeySymbol::Keypad0 => bytes_to_send.push(b'0'),
                    KeySymbol::Keypad1 => bytes_to_send.push(b'1'),
                    KeySymbol::Keypad2 => bytes_to_send.push(b'2'),
                    KeySymbol::Keypad3 => bytes_to_send.push(b'3'),
                    KeySymbol::Keypad4 => bytes_to_send.push(b'4'),
                    KeySymbol::Keypad5 => bytes_to_send.push(b'5'),
                    KeySymbol::Keypad6 => bytes_to_send.push(b'6'),
                    KeySymbol::Keypad7 => bytes_to_send.push(b'7'),
                    KeySymbol::Keypad8 => bytes_to_send.push(b'8'),
                    KeySymbol::Keypad9 => bytes_to_send.push(b'9'),
                    // KeySymbol::KeypadEnter is handled by the KeySymbol::Enter case above.
                    KeySymbol::Char(c) => {
                        // Handle Alt+Char as ESC prefix + char
                        if modifiers.contains(Modifiers::ALT) {
                            bytes_to_send.push(0x1B); // ESC
                        }
                        let mut buf = [0; 4];
                        bytes_to_send.extend(c.encode_utf8(&mut buf).as_bytes());
                    }
                    _ => {
                        trace!(
                            "Unhandled KeySymbol (with no text): {:?}, Modifiers: {:?}",
                            symbol,
                            modifiers
                        );
                    }
                }
            }

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
            debug!("UserInputAction: RequestClipboardPaste received. Requesting clipboard content.");
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
                for char_val in text_to_paste.chars() { // Iterate over chars and process them
                    emulator.print_char(char_val);
                }
                return Some(EmulatorAction::RequestRedraw);
            }
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
    use crate::term::DecPrivateModes;

    fn create_test_emu_for_input() -> TerminalEmulator {
        TerminalEmulator::new(80, 24, 100)
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

        let snapshot = emu.get_render_snapshot();
        // Print the actual screen content for debugging
        for (r, line) in snapshot.lines.iter().enumerate() {
            let line_str: String = line.cells.iter().map(|cell| cell.c).collect();
            println!("Actual line {}: '{}'", r, line_str);
        }
        println!("Actual cursor pos: {:?}", snapshot.cursor_state.map(|cs| (cs.y, cs.x)));

        assert_eq!(snapshot.lines[0].cells[0].c, 'H');
        assert_eq!(snapshot.lines[0].cells[1].c, 'e');
        assert_eq!(snapshot.lines[0].cells[2].c, 'l');
        assert_eq!(snapshot.lines[0].cells[3].c, 'l');
        assert_eq!(snapshot.lines[0].cells[4].c, 'o');
        assert_eq!(snapshot.lines[1].cells[0].c, 'W');
    }
}
