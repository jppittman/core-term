// src/term/emulator/input_handler.rs

use super::TerminalEmulator;
use crate::{
    backends::BackendEvent,
    term::{action::EmulatorAction, ControlEvent},
};
use log::{debug, trace}; // Assuming logging is still desired

#[allow(clippy::too_many_lines)] // To be addressed by further refactoring if needed
pub(super) fn process_backend_event(
    emulator: &mut TerminalEmulator,
    event: BackendEvent,
) -> Option<EmulatorAction> {
    emulator.cursor_wrap_next = false;

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
            let mut bytes_to_send: Vec<u8> = Vec::<u8>::new();
            match keysym {
                KEY_RETURN => bytes_to_send.push(b'\r'),
                KEY_BACKSPACE => bytes_to_send.push(0x08),
                KEY_TAB => bytes_to_send.push(b'\t'),
                KEY_ISO_LEFT_TAB => bytes_to_send.extend_from_slice(b"\x1b[Z"),
                KEY_ESCAPE => bytes_to_send.push(0x1B),
                KEY_UP_ARROW => {
                    bytes_to_send.extend_from_slice(if emulator.dec_modes.cursor_keys_app_mode {
                        b"\x1bOA"
                    } else {
                        b"\x1b[A"
                    })
                }
                KEY_DOWN_ARROW => {
                    bytes_to_send.extend_from_slice(if emulator.dec_modes.cursor_keys_app_mode {
                        b"\x1bOB"
                    } else {
                        b"\x1b[B"
                    })
                }
                KEY_RIGHT_ARROW => {
                    bytes_to_send.extend_from_slice(if emulator.dec_modes.cursor_keys_app_mode {
                        b"\x1bOC"
                    } else {
                        b"\x1b[C"
                    })
                }
                KEY_LEFT_ARROW => {
                    bytes_to_send.extend_from_slice(if emulator.dec_modes.cursor_keys_app_mode {
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
            if emulator.dec_modes.focus_event_mode {
                return Some(EmulatorAction::WritePty(b"\x1b[I".to_vec()));
            }
        }
        BackendEvent::FocusLost => {
            if emulator.dec_modes.focus_event_mode {
                return Some(EmulatorAction::WritePty(b"\x1b[O".to_vec()));
            }
        }
        _ => debug!(
            "BackendEvent {:?} passed to TerminalEmulator's process_backend_event, usually handled by Orchestrator or translated.",
            event
        ),
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
            // Call the resize method on TerminalEmulator.
            // This method is defined in src/term/emulator/mod.rs
            emulator.resize(cols, rows);
            None
        }
    }
}
