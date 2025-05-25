// src/term/emulator/input_handler.rs

use super::TerminalEmulator;
use crate::keys::{KeySymbol, Modifiers};
use crate::term::{
    action::{EmulatorAction, UserInputAction}, // UserInputAction from action.rs
    ControlEvent,                              // ControlEvent from term::mod.rs (re-exported from action.rs)
};
use log::trace;

#[allow(clippy::too_many_lines)]
pub(super) fn process_user_input_action(
    emulator: &mut TerminalEmulator,
    action: UserInputAction,
) -> Option<EmulatorAction> {
    emulator.cursor_wrap_next = false;

    match action {
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
                } else if modifiers.contains(Modifiers::CONTROL) && symbol == KeySymbol::Char(']')
                {
                    bytes_to_send.push(0x1d); // Ctrl+] is GS
                } else if modifiers.contains(Modifiers::CONTROL) && symbol == KeySymbol::Char('^')
                {
                    // Typically Ctrl+6 for US layout
                    bytes_to_send.push(0x1e); // Ctrl+^ is RS
                } else if modifiers.contains(Modifiers::CONTROL) && symbol == KeySymbol::Char('_')
                {
                    // Typically Ctrl+- for US layout
                    bytes_to_send.push(0x1f); // Ctrl+_ is US
                } else if modifiers.contains(Modifiers::CONTROL) && symbol == KeySymbol::Char(' ')
                {
                    bytes_to_send.push(0x00); // Ctrl+Space is NUL
                } else if modifiers.contains(Modifiers::CONTROL) && symbol == KeySymbol::Char('?') 
                { 
                    // This is tricky. Ctrl+? often generates DEL (0x7f) or depends on terminal settings.
                    // Or, if symbol is KeySymbol::Backspace and modifier is CONTROL, it might be DEL.
                    // For now, assuming if KeySymbol::Char('?') with CONTROL, it's DEL.
                    bytes_to_send.push(0x7f); // Ctrl+? is DEL
                }
                else {
                    bytes_to_send.extend(txt_val.as_bytes());
                }
            } else {
                // Priority 2: Text is None. Use KeySymbol and Modifiers to generate sequence.
                match symbol {
                    KeySymbol::Enter | KeySymbol::KeypadEnter => bytes_to_send.push(b'\r'), // Send CR
                    KeySymbol::Backspace => bytes_to_send.push(0x08), // BS
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
                    KeySymbol::End => bytes_to_send.extend_from_slice(b"\x1b[4~"),   // Standard: ESC [ 4 ~
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
        UserInputAction::MouseInput { .. } => {
            trace!("TODO: Handle UserInputAction::MouseInput");
            // Placeholder for mouse input processing
        }
        UserInputAction::InitiateCopy => {
            // TODO: Implement actual text extraction from TerminalEmulator selection state.
            // For now, sending placeholder text.
            return Some(EmulatorAction::CopyToClipboard(
                "NEEDS_IMPLEMENTATION: Get selected text from TerminalEmulator".to_string(),
            ));
        }
        UserInputAction::InitiatePaste => {
            return Some(EmulatorAction::RequestClipboardContent);
        }
        UserInputAction::PasteText(text_to_paste) => {
            // Convert pasted text to bytes and send to PTY.
            // This might need further processing (e.g., bracketed paste mode wrapping).
            if emulator.dec_modes.bracketed_paste_mode {
                let mut pasted_bytes = Vec::new();
                pasted_bytes.extend_from_slice(b"\x1b[200~"); // Start bracketed paste
                pasted_bytes.extend_from_slice(text_to_paste.as_bytes());
                pasted_bytes.extend_from_slice(b"\x1b[201~"); // End bracketed paste
                return Some(EmulatorAction::WritePty(pasted_bytes));
            } else {
                return Some(EmulatorAction::WritePty(text_to_paste.into_bytes()));
            }
        }
    }
    None
}

// Preserving process_control_event as it's called from TerminalEmulator::interpret_input
pub(super) fn process_control_event(
    emulator: &mut TerminalEmulator,
    event: ControlEvent,
) -> Option<EmulatorAction> {
    emulator.cursor_wrap_next = false;
    match event {
        ControlEvent::FrameRendered => {
            trace!("TerminalEmulator: FrameRendered event received.");
            // Potentially reset some per-frame state here if needed in the future.
            None
        }
        ControlEvent::Resize { cols, rows } => {
            trace!(
                "TerminalEmulator: ControlEvent::Resize to {}x{} received.",
                cols,
                rows
            );
            emulator.resize(cols, rows); // Call the existing resize method on TerminalEmulator
            None // Resize itself doesn't directly cause an EmulatorAction to be returned.
                 // Redraw is handled implicitly or by the orchestrator.
        }
    }
}
