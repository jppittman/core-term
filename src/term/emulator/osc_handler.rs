// src/term/emulator/osc_handler.rs

use super::TerminalEmulator;
use crate::term::action::EmulatorAction;
use log::{debug, warn};

impl TerminalEmulator {
    pub(super) fn handle_osc(&mut self, data: Vec<u8>) -> Option<EmulatorAction> {
        let osc_str = String::from_utf8_lossy(&data);
        let parts: Vec<&str> = osc_str.splitn(2, ';').collect();

        if parts.len() == 2 {
            let ps_str = parts[0];
            let content_str = parts[1]; // This is the raw string, possibly with terminators

            let ps = ps_str.parse::<u32>().unwrap_or(u32::MAX);

            match ps {
                0 | 2 => { // OSC Set Icon Name or Set Window Title
                    // Directly use content_str, assuming it's pre-processed by parser
                    return Some(EmulatorAction::SetTitle(content_str.to_string()));
                }
                _ => debug!("Unhandled OSC command: Ps={}, Pt='{}'", ps_str, content_str),
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
