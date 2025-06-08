// src/ansi/parser.rs

//! ANSI escape sequence parser.
//! Takes individual `AnsiToken`s and accumulates `AnsiCommand`s internally.

// Import necessary items from commands and lexer modules
// Removed unused EscCommand import
use super::commands::{AnsiCommand, C0Control};
use super::lexer::AnsiToken;
use log::{error, trace, warn};
use std::mem;

// Define maximum buffer sizes to prevent excessive memory use
const MAX_PARAMS: usize = 16;
const MAX_INTERMEDIATES: usize = 2;
const MAX_OSC_LEN: usize = 1024; // Limit OSC/DCS/PM/APC string length

/// Represents the current state of the ANSI parser state machine.
#[derive(Debug, Clone, PartialEq, Eq)]
enum State {
    Ground,
    Escape,
    CsiEntry,
    CsiParam,
    CsiIntermediate,
    OscString,
    DcsEntry,
    PmString,
    ApcString,
    EscInString,
    EscIntermediate,
}

/// The ANSI parser structure.
#[derive(Debug)]
pub struct AnsiParser {
    state: State,
    commands: Vec<AnsiCommand>,
    params: Vec<u16>,
    intermediates: Vec<u8>,
    string_buffer: Vec<u8>,
    is_private_csi: bool,
    current_param_value: u16,
    is_param_empty: bool,
    string_state_origin: Option<State>,
    esc_intermediate: Option<char>,
}

impl AnsiParser {
    pub fn new() -> Self {
        AnsiParser {
            state: State::Ground,
            commands: Vec::new(),
            params: Vec::with_capacity(MAX_PARAMS),
            intermediates: Vec::with_capacity(MAX_INTERMEDIATES),
            string_buffer: Vec::with_capacity(MAX_OSC_LEN / 4),
            is_private_csi: false,
            current_param_value: 0,
            is_param_empty: true,
            string_state_origin: None,
            esc_intermediate: None,
        }
    }

    pub fn take_commands(&mut self) -> Vec<AnsiCommand> {
        mem::take(&mut self.commands)
    }

    fn clear_csi_state(&mut self) {
        self.params.clear();
        self.intermediates.clear();
        self.is_private_csi = false;
        self.current_param_value = 0;
        self.is_param_empty = true;
    }

    fn clear_string_buffer(&mut self) {
        self.string_buffer.clear();
        self.string_state_origin = None;
    }

    fn clear_esc_state(&mut self) {
        self.esc_intermediate = None;
    }

    fn add_param(&mut self, param: u16) {
        if self.params.len() < MAX_PARAMS {
            self.params.push(param);
        } else {
            warn!("Exceeded maximum CSI parameters ({})", MAX_PARAMS);
        }
    }

    fn finalize_param(&mut self) {
        if self.is_param_empty {
            self.add_param(0);
        } else {
            self.add_param(self.current_param_value);
        }
        self.current_param_value = 0;
        self.is_param_empty = true;
    }

    fn add_intermediate(&mut self, intermediate: u8) {
        if self.intermediates.len() < MAX_INTERMEDIATES {
            self.intermediates.push(intermediate);
        } else {
            warn!(
                "Exceeded maximum CSI intermediate bytes ({})",
                MAX_INTERMEDIATES
            );
        }
    }

    fn add_string_byte(&mut self, byte: u8) {
        if self.string_buffer.len() < MAX_OSC_LEN {
            self.string_buffer.push(byte);
        } else {
            warn!("Exceeded maximum string length ({})", MAX_OSC_LEN);
        }
    }

    fn dispatch_c0(&mut self, byte: u8) {
        trace!("Dispatching C0 Control: {}", byte);
        if let Some(command) = AnsiCommand::from_c0(byte) {
            self.commands.push(command);
        } else {
            error!("Unhandled C0 control byte: {}", byte);
            self.commands.push(AnsiCommand::Error(byte));
        }
        self.clear_esc_state();
        self.state = State::Ground;
    }

    fn dispatch_print(&mut self, c: char) {
        self.commands.push(AnsiCommand::Print(c));
        self.clear_esc_state();
        self.state = State::Ground;
    }

    fn dispatch_csi(&mut self, final_byte: u8) {
        trace!(
            "Dispatching CSI: Private={}, Params={:?}, Intermediates={:?}, Final={}({})",
            self.is_private_csi,
            self.params,
            self.intermediates,
            final_byte as char,
            final_byte
        );
        // These are the actual parameters and intermediates collected by the parser state machine
        let params_vec = mem::take(&mut self.params);
        let intermediates_vec = mem::take(&mut self.intermediates);
        let is_private_csi_flag = self.is_private_csi;

        if let Some(command) = AnsiCommand::from_csi(
            params_vec,
            intermediates_vec,
            is_private_csi_flag,
            final_byte,
        ) {
            // Check if the command is the specific Unsupported variant we want to remap
            if let AnsiCommand::Csi(super::commands::CsiCommand::Unsupported(
                _,
                Some(unsupported_final_byte),
            )) = command
            {
                trace!(
                    "Remapping CsiCommand::Unsupported with final byte {} to AnsiCommand::Error",
                    unsupported_final_byte
                );
                self.commands
                    .push(AnsiCommand::Error(unsupported_final_byte));
            } else {
                // It's a different, valid CSI command
                self.commands.push(command);
            }
        } else {
            // AnsiCommand::from_csi returned None, meaning it's not just unsupported but perhaps malformed.
            warn!(
                "AnsiCommand::from_csi returned None for final_byte {}. Reporting as AnsiCommand::Error.",
                final_byte
            );
            self.commands.push(AnsiCommand::Error(final_byte));
        }
        self.clear_csi_state();
        self.state = State::Ground;
    }

    fn dispatch_osc(&mut self, consume_st: bool) {
        let data = mem::take(&mut self.string_buffer);
        trace!("Dispatching OSC: Data length {}", data.len());
        self.commands.push(AnsiCommand::Osc(data));
        self.clear_string_buffer();
        if !consume_st { /* ST handled separately */ }
        self.state = State::Ground;
    }

    fn dispatch_dcs(&mut self, consume_st: bool) {
        let data = mem::take(&mut self.string_buffer);
        trace!("Dispatching DCS: Data length {}", data.len());
        self.commands.push(AnsiCommand::Dcs(data));
        self.clear_string_buffer();
        if !consume_st { /* ST handled separately */ }
        self.state = State::Ground;
    }

    fn dispatch_pm(&mut self, consume_st: bool) {
        let data = mem::take(&mut self.string_buffer);
        trace!("Dispatching PM: Data length {}", data.len());
        self.commands.push(AnsiCommand::Pm(data));
        self.clear_string_buffer();
        if !consume_st { /* ST handled separately */ }
        self.state = State::Ground;
    }

    fn dispatch_apc(&mut self, consume_st: bool) {
        let data = mem::take(&mut self.string_buffer);
        trace!("Dispatching APC: Data length {}", data.len());
        self.commands.push(AnsiCommand::Apc(data));
        self.clear_string_buffer();
        if !consume_st { /* ST handled separately */ }
        self.state = State::Ground;
    }

    fn dispatch_st_standalone(&mut self) {
        trace!("Dispatching Standalone String Terminator (ST)");
        self.commands.push(AnsiCommand::StringTerminator);
        self.clear_string_buffer();
        self.clear_esc_state();
        self.state = State::Ground;
    }

    fn dispatch_ignore(&mut self, byte: u8) {
        trace!("Dispatching Ignore: {}", byte);
        self.commands.push(AnsiCommand::Ignore(byte));
    }

    fn dispatch_error(&mut self, byte: u8) {
        trace!("Dispatching Error: {}", byte);
        self.commands.push(AnsiCommand::Error(byte));
        self.clear_esc_state();
        self.state = State::Ground;
    }

    fn enter_string_state(&mut self, next_state: State) {
        self.clear_string_buffer();
        self.string_state_origin = Some(self.state.clone());
        self.state = next_state;
    }

    fn enter_esc_in_string_state(&mut self) {
        self.string_state_origin = Some(self.state.clone());
        self.state = State::EscInString;
    }

    pub fn process_token(&mut self, token: AnsiToken) {
        match self.state {
            State::Ground => match token {
                AnsiToken::Print(c) => self.dispatch_print(c),
                AnsiToken::C0Control(0x1B) => {
                    self.clear_esc_state();
                    self.state = State::Escape;
                }
                AnsiToken::C0Control(byte) => self.dispatch_c0(byte),
            },
            State::Escape => match token {
                AnsiToken::C0Control(0x1B) => self.state = State::Escape,
                AnsiToken::Print('[') => {
                    self.clear_csi_state();
                    self.state = State::CsiEntry;
                }
                AnsiToken::Print(']') => self.enter_string_state(State::OscString),
                AnsiToken::Print('P') => self.enter_string_state(State::DcsEntry),
                AnsiToken::Print('^') => self.enter_string_state(State::PmString),
                AnsiToken::Print('_') => self.enter_string_state(State::ApcString),
                AnsiToken::Print('\\') => self.dispatch_st_standalone(),
                AnsiToken::Print(inter @ ('(' | ')' | '*' | '+')) => {
                    self.esc_intermediate = Some(inter);
                    self.state = State::EscIntermediate;
                }
                AnsiToken::Print(c) => {
                    if let Some(command) = AnsiCommand::from_esc(c) {
                        // AnsiCommand::from_esc is from commands.rs
                        self.commands.push(command);
                    } else {
                        // If 'c' does not form a valid ESC sequence, treat 'c' as a printable character.
                        self.commands.push(AnsiCommand::Print(c));
                    }
                    self.state = State::Ground; // Ensure state transitions back to Ground
                }
                AnsiToken::C0Control(byte) => self.dispatch_c0(byte),
            },
            State::EscIntermediate => {
                if let Some(inter) = self.esc_intermediate {
                    match token {
                        AnsiToken::Print(final_char) => {
                            // Use the specific helper for ESC intermediate sequences
                            if let Some(command) =
                                AnsiCommand::from_esc_intermediate(inter, final_char)
                            {
                                self.commands.push(command);
                            } else {
                                self.dispatch_ignore(inter as u8);
                                self.dispatch_ignore(final_char as u8);
                            }
                        }
                        AnsiToken::C0Control(byte) => {
                            self.dispatch_ignore(inter as u8);
                            self.dispatch_c0(byte);
                        }
                    }
                } else {
                    error!("Invalid EscIntermediate state");
                    self.dispatch_error(token.to_byte_lossy());
                }
                self.clear_esc_state();
                self.state = State::Ground;
            }

            State::CsiEntry => match token {
                AnsiToken::C0Control(0x1B) => {
                    self.clear_csi_state();
                    self.clear_esc_state();
                    self.state = State::Escape;
                }
                AnsiToken::C0Control(byte) => {
                    self.dispatch_c0(byte);
                    self.clear_csi_state();
                }
                AnsiToken::Print(c @ '0'..='9') => {
                    self.current_param_value = (c as u16) - ('0' as u16);
                    self.is_param_empty = false;
                    self.state = State::CsiParam;
                }
                AnsiToken::Print(';') => {
                    self.add_param(0);
                    self.is_param_empty = true;
                    self.state = State::CsiParam;
                }
                AnsiToken::Print(p @ ('?' | '>' | '!' | '$' | '\'')) => {
                    self.is_private_csi = true;
                    self.add_intermediate(p as u8);
                    self.state = State::CsiParam;
                }
                AnsiToken::Print(i @ ' '..='/') => {
                    self.add_intermediate(i as u8);
                    self.state = State::CsiIntermediate;
                }
                AnsiToken::Print(f @ '@'..='~') => self.dispatch_csi(f as u8),
                AnsiToken::Print(c) => {
                    warn!("Unexpected char '{}' in CsiEntry", c);
                    self.dispatch_error(c as u8);
                    self.clear_csi_state();
                }
            },
            State::CsiParam => match token {
                AnsiToken::C0Control(0x1B) => {
                    self.clear_csi_state();
                    self.clear_esc_state();
                    self.state = State::Escape;
                }
                AnsiToken::C0Control(byte) => {
                    self.dispatch_c0(byte);
                    self.clear_csi_state();
                }
                AnsiToken::Print(c @ '0'..='9') => {
                    if let Some(next_val) = self.current_param_value.checked_mul(10) {
                        if let Some(final_val) = next_val.checked_add((c as u16) - ('0' as u16)) {
                            self.current_param_value = final_val;
                        } else {
                            warn!("CSI param overflow");
                            self.current_param_value = u16::MAX;
                        }
                    } else {
                        warn!("CSI param overflow");
                        self.current_param_value = u16::MAX;
                    }
                    self.is_param_empty = false;
                }
                AnsiToken::Print(';') => self.finalize_param(),
                AnsiToken::Print(i @ ' '..='/') => {
                    self.finalize_param();
                    self.add_intermediate(i as u8);
                    self.state = State::CsiIntermediate;
                }
                AnsiToken::Print(f @ '@'..='~') => {
                    self.finalize_param();
                    self.dispatch_csi(f as u8);
                }
                AnsiToken::Print(c) => {
                    warn!("Unexpected char '{}' in CsiParam", c);
                    self.dispatch_error(c as u8);
                    self.clear_csi_state();
                }
            },
            State::CsiIntermediate => match token {
                AnsiToken::C0Control(0x1B) => {
                    self.clear_csi_state();
                    self.clear_esc_state();
                    self.state = State::Escape;
                }
                AnsiToken::C0Control(byte) => {
                    self.dispatch_c0(byte);
                    self.clear_csi_state();
                }
                AnsiToken::Print(i @ ' '..='/') => self.add_intermediate(i as u8),
                AnsiToken::Print(f @ '@'..='~') => self.dispatch_csi(f as u8),
                AnsiToken::Print(c) => {
                    warn!("Unexpected char '{}' in CsiIntermediate", c);
                    self.dispatch_error(c as u8);
                    self.clear_csi_state();
                }
            },
            State::OscString | State::DcsEntry | State::PmString | State::ApcString => {
                match token {
                    AnsiToken::C0Control(0x1B) => self.enter_esc_in_string_state(),
                    AnsiToken::C0Control(0x07) if self.state == State::OscString => {
                        self.dispatch_osc(false)
                    }
                    AnsiToken::C0Control(0x18) | AnsiToken::C0Control(0x1A) => {
                        self.clear_string_buffer();
                        self.state = State::Ground;
                    }
                    AnsiToken::C0Control(byte) => self.add_string_byte(byte),
                    AnsiToken::Print(c) => {
                        let mut buf = [0; 4];
                        c.encode_utf8(&mut buf)
                            .as_bytes()
                            .iter()
                            .for_each(|&b| self.add_string_byte(b));
                    }
                }
            }
            State::EscInString => match token {
                AnsiToken::Print('\\') => match self.string_state_origin {
                    Some(State::OscString) => self.dispatch_osc(true),
                    Some(State::DcsEntry) => self.dispatch_dcs(true),
                    Some(State::PmString) => self.dispatch_pm(true),
                    Some(State::ApcString) => self.dispatch_apc(true),
                    _ => {
                        error!("EscInString state missing origin!");
                        self.dispatch_st_standalone();
                    }
                },
                _ => {
                    trace!(
                        "ESC followed by {:?} inside string - aborting string, processing ESC sequence",
                        token
                    );
                    self.clear_string_buffer();
                    self.commands.push(AnsiCommand::C0Control(C0Control::ESC));
                    self.state = State::Ground;
                    self.process_token(token);
                }
            },
        }
    }
}

impl Default for AnsiParser {
    fn default() -> Self {
        Self::new()
    }
}

// Helper trait
trait AnsiTokenByte {
    fn to_byte_lossy(&self) -> u8;
}

impl AnsiTokenByte for AnsiToken {
    fn to_byte_lossy(&self) -> u8 {
        match self {
            AnsiToken::Print(c) => (*c as u32).try_into().unwrap_or(b'?'),
            AnsiToken::C0Control(b) => *b,
        }
    }
}
