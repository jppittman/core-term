// src/term/emulator/osc_handler.rs

use super::TerminalEmulator;
use crate::term::action::EmulatorAction;
use log::{debug, warn};

impl TerminalEmulator {
    pub(super) fn handle_osc(&mut self, data: Vec<u8>) -> Option<EmulatorAction> {
        let osc_str = String::from_utf8_lossy(&data);
        let parts: Vec<&str> = osc_str.splitn(2, ';').collect();

        if parts.len() == 2 {
            let ps = parts[0].parse::<u32>().unwrap_or(u32::MAX);
            let pt = parts[1].to_string();

            match ps {
                0 | 2 => {
                    return Some(EmulatorAction::SetTitle(pt));
                }
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
