// src/term/emulator/input_handler.rs

use super::{FocusState, TerminalEmulator};
use crate::backends::{MouseButton, MouseEventType};
use crate::keys::{KeySymbol, Modifiers};
use crate::term::{
    snapshot::SelectionMode, // Import SelectionMode
    ControlEvent,            // ControlEvent from term::mod.rs (re-exported from action.rs)
    action::{EmulatorAction, UserInputAction}, // UserInputAction from action.rs
};
use log::{debug, trace}; // Added debug

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
                            symbol, modifiers
                        );
                    }
                }
            }

            if !bytes_to_send.is_empty() {
                return Some(EmulatorAction::WritePty(bytes_to_send));
            }
        }
        UserInputAction::MouseInput {
            col,
            row,
            event_type,
            button,
            modifiers,
        } => {
            // Local mutable reference to selection for convenience
            let selection = &mut emulator.selection;
            let mut pty_bytes: Option<Vec<u8>> = None;

            match event_type {
                MouseEventType::Press => {
                    if button == MouseButton::Left {
                        // Differentiate between normal and block selection mode based on modifiers (e.g., Alt)
                        let mode = if modifiers.contains(Modifiers::ALT) {
                            // Alt for block selection
                            SelectionMode::Block
                        } else {
                            SelectionMode::Normal
                        };
                        selection.start_selection(col, row, mode);
                        debug!("MousePress: Started selection at ({}, {}) mode {:?}", col, row, mode);
                    }
                    // Handle other button presses if needed (e.g., paste on Middle click)
                }
                MouseEventType::Move => {
                    if selection.is_active {
                        // Only update if a selection is active (e.g. dragging)
                        selection.update_selection(col, row);
                        // trace!("MouseDrag: Updated selection to ({}, {})", col, row); // Can be too verbose
                    }
                    // Mouse reporting for motion might also be needed if a mode is active
                }
                MouseEventType::Release => {
                    if button == MouseButton::Left && selection.is_active {
                        selection.end_selection();
                        debug!("MouseRelease: Ended selection. Start: {:?}, End: {:?}", selection.start, selection.end);

                        // Potentially copy to clipboard on selection end if configured (e.g. primary selection)
                        // For now, this is handled by UserInputAction::InitiateCopy explicitly.
                    }
                    // Handle other button releases if needed
                }
            }

            // Mouse Reporting logic
            // Check if any mouse reporting mode is active.
            // This uses a simplified check; a real implementation would check specific modes.
            if emulator.dec_modes.mouse_x10_mode
                || emulator.dec_modes.mouse_vt200_mode
                || emulator.dec_modes.mouse_vt200_highlight_mode
                || emulator.dec_modes.mouse_button_event_mode
                || emulator.dec_modes.mouse_any_event_mode
                || emulator.dec_modes.mouse_sgr_mode // SGR mode (1006)
                || emulator.dec_modes.mouse_sgr_pixels_mode // SGR-Pixels mode (1016) - assuming similar reporting structure
            {
                let mut final_button_code = 0;
                let mut release_event = false;

                // Map our MouseButton to X11-like button codes for reporting
                match button {
                    MouseButton::Left => final_button_code = 0,
                    MouseButton::Middle => final_button_code = 1,
                    MouseButton::Right => final_button_code = 2,
                    MouseButton::ScrollUp => final_button_code = 64,
                    MouseButton::ScrollDown => final_button_code = 65,
                    MouseButton::ScrollLeft => final_button_code = 66, // Common extension for SGR/newer modes
                    MouseButton::ScrollRight => final_button_code = 67, // Common extension for SGR/newer modes
                    MouseButton::Unknown => { /* No specific button code for unknown in standard reporting */ }
                }
                
                if event_type == MouseEventType::Move && selection.is_active {
                    // For drag events, X10/VT200 motion notify adds 32 to the button code.
                    // SGR mode handles motion differently (reports it as part of the button code itself).
                    if !emulator.dec_modes.mouse_sgr_mode && !emulator.dec_modes.mouse_sgr_pixels_mode {
                        final_button_code += 32; // Motion flag for non-SGR modes
                    }
                }


                if event_type == MouseEventType::Release {
                    release_event = true;
                    // For X10 compatibility and some VT200 modes, button release is often encoded as 3 for all buttons.
                    // SGR mode can distinguish releases with its own 'm' event type.
                    if !emulator.dec_modes.mouse_sgr_mode && !emulator.dec_modes.mouse_sgr_pixels_mode {
                        // For non-SGR modes that report release (like ButtonEvent mode 1002, AnyEvent mode 1003)
                        // the button code itself might be 3 for release, or use the actual button code.
                        // For simplicity and common X10 behavior, we'll use 3 if not SGR.
                        // However, if it's a scroll wheel release, those buttons are special.
                        if button != MouseButton::ScrollUp && button != MouseButton::ScrollDown &&
                           button != MouseButton::ScrollLeft && button != MouseButton::ScrollRight {
                           final_button_code = 3; // All non-scroll buttons release as '3' in X10/Normal tracking
                        }
                        // Scroll wheel releases are often not reported or reported as separate button events.
                        // For this example, we'll assume scroll release uses its original code if reported.
                    }
                }

                // Add modifier flags to the button code.
                // These are standard values for XTerm mouse reporting.
                if modifiers.contains(Modifiers::SHIFT) { final_button_code += 4; }
                if modifiers.contains(Modifiers::ALT) { final_button_code += 8; } // Meta/Alt
                if modifiers.contains(Modifiers::CONTROL) { final_button_code += 16; }


                // Clamp coordinates: 1-based for reporting.
                // Max is 223 for X10 (255-32), or higher for SGR/UTF-8 extensions.
                // SGR mode (1006) can go up to 2015 (2047-32).
                let max_coord_val = if emulator.dec_modes.mouse_sgr_mode || emulator.dec_modes.mouse_sgr_pixels_mode { 2015 } else { 223 };
                let report_col = (col + 1).min(max_coord_val);
                let report_row = (row + 1).min(max_coord_val);

                let mut seq = Vec::new();
                seq.push(0x1B); // ESC
                seq.push(b'['); // CSI

                if emulator.dec_modes.mouse_sgr_mode || emulator.dec_modes.mouse_sgr_pixels_mode {
                    // SGR Mouse Mode (1006): CSI < Pb ; Px ; Py M (press) or m (release)
                    // Pb: button code + modifiers.
                    // Px: column. (Note: SGR sends 1-based coords directly, not +32 offset)
                    // Py: row.
                    seq.extend_from_slice(format!("<{};{};{}", final_button_code, report_col, report_row).as_bytes());
                    if release_event {
                        seq.push(b'm'); // Release
                    } else {
                        seq.push(b'M'); // Press or drag
                    }
                } else if emulator.dec_modes.mouse_any_event_mode || emulator.dec_modes.mouse_button_event_mode || emulator.dec_modes.mouse_vt200_mode || emulator.dec_modes.mouse_x10_mode {
                    // Normal Mouse Tracking (X10, VT200, ButtonEvent, AnyEvent)
                    // Format: CSI M Cb Cx Cy
                    // Cb: button code + 32.
                    // Cx: column + 32.
                    // Cy: row + 32.
                    
                    // Determine if this event should be reported for the active non-SGR modes
                    let mut should_report = false;
                    match event_type {
                        MouseEventType::Press => should_report = true,
                        MouseEventType::Release => {
                            // X10 mode (1000) only reports press.
                            // VT200 (1001), ButtonEvent (1002), AnyEvent (1003) report release.
                            if emulator.dec_modes.mouse_vt200_mode || emulator.dec_modes.mouse_button_event_mode || emulator.dec_modes.mouse_any_event_mode {
                                should_report = true;
                            }
                        },
                        MouseEventType::Move => {
                            // Only AnyEvent (1003) reports motion without buttons.
                            // Other modes (X10, VT200, ButtonEvent) report motion only during drag.
                            if emulator.dec_modes.mouse_any_event_mode {
                                should_report = true;
                            } else if selection.is_active { // Dragging
                                should_report = true;
                            }
                        }
                    }

                    if should_report {
                        seq.push(b'M');
                        seq.push((final_button_code + 32) as u8);
                        seq.push((report_col + 32) as u8);
                        seq.push((report_row + 32) as u8);
                    } else {
                        seq.clear(); // Don't send anything
                    }
                } else {
                    seq.clear(); // Not in a mode that reports this event type
                }
                
                if !seq.is_empty() {
                    debug!("Sending mouse report sequence: {:?}", String::from_utf8_lossy(&seq));
                    pty_bytes = Some(seq);
                }
            }

            if pty_bytes.is_some() {
                return pty_bytes.map(EmulatorAction::WritePty);
            }
        }
        UserInputAction::InitiateCopy => {
            // Get selected text from the emulator's selection state and screen buffer.
            let (term_cols, term_rows) = emulator.dimensions();
            let lines = emulator.screen.snapshot_buffer(); // Need a way to get current screen lines
            let selected_text = emulator.selection.get_selected_text(&lines, term_cols, term_rows);
            if !selected_text.is_empty() {
                debug!("InitiateCopy: Selected text: '{}'", selected_text);
                return Some(EmulatorAction::CopyToClipboard(selected_text));
            }
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
                cols, rows
            );
            emulator.resize(cols, rows); // Call the existing resize method on TerminalEmulator
            None // Resize itself doesn't directly cause an EmulatorAction to be returned.
            // Redraw is handled implicitly or by the orchestrator.
        }
    }
}
