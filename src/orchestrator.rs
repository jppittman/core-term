// src/orchestrator.rs
//! Orchestrates the main application flow, coordinating between the PTY,
//! terminal emulator, renderer, and backend driver. This module aims to encapsulate
//! the core event processing logic, making it testable and maintainable by
//! abstracting away direct OS calls and backend specifics.

use crate::{
    ansi::AnsiParser,
    backends::{BackendEvent, Driver}, // Removed BackendModifiers, Keycode
    config::{Config, KeySymbol, Modifiers as ConfigModifiers},
    os::pty::PtyChannel,
    renderer::Renderer,
    term::{ControlEvent, EmulatorAction, EmulatorInput, TerminalInterface, UserInputAction},
};
use anyhow::Error as AnyhowError;
use arboard; // Import arboard
use std::io::ErrorKind as IoErrorKind;

const PTY_READ_BUFFER_SIZE: usize = 4096;

/// Represents the status of the orchestrator after processing an event or an iteration of its loop.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum OrchestratorStatus {
    /// The orchestrator processed events successfully and should continue running.
    Running,
    /// A shutdown signal was received (e.g., PTY EOF, Quit event from driver).
    /// The application should terminate gracefully.
    Shutdown,
}

/// Encapsulates the main application state and logic for orchestrating terminal components.
///
/// This struct uses trait objects for its main dependencies (`PtyChannel`, `TerminalInterface`,
/// `Driver`) to allow for mocking in tests and flexibility in choosing
/// concrete implementations. The `Renderer` is a concrete type.
pub struct AppOrchestrator<'a> {
    pty_channel: &'a mut dyn PtyChannel,
    term: &'a mut dyn TerminalInterface,
    parser: &'a mut dyn AnsiParser,
    pub renderer: Renderer,
    pub driver: &'a mut dyn Driver,
    config: &'a Config, // Added config field
    pty_read_buffer: [u8; PTY_READ_BUFFER_SIZE],
}

impl<'a> AppOrchestrator<'a> {
    /// Creates a new `AppOrchestrator`.
    pub fn new(
        pty_channel: &'a mut dyn PtyChannel,
        term: &'a mut dyn TerminalInterface,
        parser: &'a mut dyn AnsiParser,
        renderer: Renderer,
        driver: &'a mut dyn Driver,
        config: &'a Config, // Added config parameter
    ) -> Self {
        AppOrchestrator {
            pty_channel,
            term,
            parser,
            renderer,
            driver,
            config, // Initialize config field
            pty_read_buffer: [0; PTY_READ_BUFFER_SIZE],
        }
    }

    // Removed translate_modifiers and translate_keysym helper functions

    pub fn process_pty_events(&mut self) -> Result<OrchestratorStatus, AnyhowError> {
        log::trace!("Orchestrator: Processing available PTY data...");
        match self.pty_channel.read(&mut self.pty_read_buffer) {
            Ok(0) => {
                log::info!("Orchestrator: PTY EOF received. Signaling shutdown.");
                Ok(OrchestratorStatus::Shutdown)
            }
            Ok(count) => {
                log::debug!("Orchestrator: Read {} bytes from PTY.", count);
                let data_slice = &self.pty_read_buffer[..count];
                let pty_data_copy = data_slice.to_vec();
                self.interpret_pty_bytes_mut_access(&pty_data_copy);
                Ok(OrchestratorStatus::Running)
            }
            Err(e) if e.kind() == IoErrorKind::WouldBlock => {
                log::trace!("Orchestrator: PTY read would block (no new data available).");
                Ok(OrchestratorStatus::Running)
            }
            Err(e) => {
                log::error!("Orchestrator: Unrecoverable error reading from PTY: {}", e);
                Err(AnyhowError::from(e).context("PTY read error"))
            }
        }
    }

    fn interpret_pty_bytes_mut_access(&mut self, pty_data_slice: &[u8]) {
        let commands = self
            .parser
            .process_bytes(pty_data_slice)
            .into_iter()
            .map(EmulatorInput::Ansi);

        for command_input in commands {
            if let Some(action) = self.term.interpret_input(command_input) {
                self.handle_emulator_action(action);
            }
        }
    }

    pub fn process_driver_events(&mut self) -> Result<OrchestratorStatus, String> {
        log::trace!("Orchestrator: Processing available driver events...");
        let events = self.driver.process_events().map_err(|e| {
            let err_msg = format!("Orchestrator: Driver error processing events: {}", e);
            log::error!("{}", err_msg);
            err_msg
        })?;

        if events.is_empty() {
            log::trace!("Orchestrator: No new driver events.");
            return Ok(OrchestratorStatus::Running);
        }

        for event in events {
            log::debug!("Orchestrator: Handling BackendEvent: {:?}", event);
            if event == BackendEvent::CloseRequested {
                log::info!("Orchestrator: CloseRequested event received. Signaling shutdown.");
                return Ok(OrchestratorStatus::Shutdown);
            }
            self.handle_specific_driver_event(event);
        }
        Ok(OrchestratorStatus::Running)
    }

    fn handle_specific_driver_event(&mut self, event: BackendEvent) {
        match event {
            BackendEvent::Key {
                symbol, // Directly use the translated symbol from BackendEvent
                modifiers, // Directly use the translated modifiers from BackendEvent
                text,
            } => {
                let copy_binding = self.config.keybindings.copy;
                let paste_binding = self.config.keybindings.paste;

                let mut user_input_action = None;

                // Compare directly with config::KeySymbol and config::Modifiers
                if symbol == copy_binding.symbol && modifiers == copy_binding.modifiers {
                    user_input_action = Some(UserInputAction::InitiateCopy);
                } else if symbol == paste_binding.symbol && modifiers == paste_binding.modifiers {
                    user_input_action = Some(UserInputAction::InitiatePaste);
                } else {
                    // Regular key input
                    user_input_action = Some(UserInputAction::KeyInput {
                        symbol,    // Use directly
                        modifiers, // Use directly
                        text,      // Pass along the text from backend
                    });
                }

                if let Some(uia) = user_input_action {
                    if let Some(action) = self.term.interpret_input(EmulatorInput::User(uia)) {
                        self.handle_emulator_action(action);
                    }
                }
            }
            BackendEvent::Resize {
                width_px,
                height_px,
            } => {
                let (char_width, char_height) = self.driver.get_font_dimensions();
                if char_width == 0 || char_height == 0 {
                    log::warn!(
                        "Orchestrator: Received resize but driver reported zero char dimensions ({}, {}). Ignoring resize.",
                        char_width,
                        char_height
                    );
                    return;
                }

                let new_cols = (width_px as usize / char_width.max(1)).max(1);
                let new_rows = (height_px as usize / char_height.max(1)).max(1);

                log::info!(
                    "Orchestrator: Resizing to {}x{} cells ({}x{} px, char_size: {}x{})",
                    new_cols,
                    new_rows,
                    width_px,
                    height_px,
                    char_width,
                    char_height
                );

                if let Err(e) = self.pty_channel.resize(new_cols as u16, new_rows as u16) {
                    log::warn!(
                        "Orchestrator: Failed to resize PTY to {}x{}: {}",
                        new_cols,
                        new_rows,
                        e
                    );
                }

                let resize_event = EmulatorInput::Control(ControlEvent::Resize {
                    cols: new_cols,
                    rows: new_rows,
                });
                if let Some(action) = self.term.interpret_input(resize_event) {
                    self.handle_emulator_action(action);
                }
            }
            BackendEvent::FocusGained => {
                log::debug!("Orchestrator: FocusGained event.");
                self.driver.set_focus(true);
                if let Some(action) = self
                    .term
                    .interpret_input(EmulatorInput::User(BackendEvent::FocusGained))
                {
                    self.handle_emulator_action(action);
                }
            }
            BackendEvent::FocusLost => {
                log::debug!("Orchestrator: FocusLost event.");
                self.driver.set_focus(false);
                if let Some(action) = self
                    .term
                    .interpret_input(EmulatorInput::User(BackendEvent::FocusLost))
                {
                    self.handle_emulator_action(action);
                }
            }
            BackendEvent::CloseRequested => {
                log::warn!(
                    "Orchestrator: CloseRequested event unexpectedly reached handle_specific_driver_event."
                );
            }
        }
    }

    /// Handles actions signaled by the `TerminalInterface` implementation.
    fn handle_emulator_action(&mut self, action: EmulatorAction) {
        log::debug!("Orchestrator: Handling EmulatorAction: {:?}", action);
        match action {
            EmulatorAction::WritePty(data) => {
                if let Err(e) = self.pty_channel.write_all(&data) {
                    log::error!(
                        "Orchestrator: Failed to write_all {} bytes to PTY: {}",
                        data.len(),
                        e
                    );
                } else {
                    log::trace!("Orchestrator: Wrote {} bytes to PTY.", data.len());
                }
            }
            EmulatorAction::SetTitle(title) => {
                self.driver.set_title(&title);
            }
            EmulatorAction::RingBell => {
                self.driver.bell();
            }
            EmulatorAction::RequestRedraw => {
                log::trace!("Orchestrator: EmulatorAction::RequestRedraw received (now implicit).");
            }
            EmulatorAction::SetCursorVisibility(visible) => {
                log::trace!(
                    "Orchestrator: Setting driver cursor visibility to: {}",
                    visible
                );
                self.driver.set_cursor_visibility(visible);
            }
            EmulatorAction::CopyToClipboard(text) => {
                match arboard::Clipboard::new() {
                    Ok(mut clipboard) => {
                        if let Err(e) = clipboard.set_text(text) {
                            log::error!("Orchestrator: Failed to copy to clipboard: {}", e);
                        } else {
                            log::info!("Orchestrator: Copied text to clipboard.");
                        }
                    }
                    Err(e) => {
                        log::error!("Orchestrator: Failed to initialize clipboard for copy: {}", e);
                    }
                }
            }
            EmulatorAction::RequestClipboardContent => {
                match arboard::Clipboard::new() {
                    Ok(mut clipboard) => {
                        match clipboard.get_text() {
                            Ok(text) => {
                                log::info!("Orchestrator: Pasting text from clipboard.");
                                let paste_input =
                                    EmulatorInput::User(UserInputAction::PasteText(text));
                                // Recursively call interpret_input -> handle_emulator_action
                                // This is okay for one level. Deeper recursion could be an issue.
                                if let Some(action) = self.term.interpret_input(paste_input) {
                                    self.handle_emulator_action(action);
                                }
                            }
                            Err(e) => {
                                log::error!("Orchestrator: Failed to get text from clipboard: {}", e);
                            }
                        }
                    }
                    Err(e) => {
                        log::error!(
                            "Orchestrator: Failed to initialize clipboard for paste: {}",
                            e
                        );
                    }
                }
            }
        }
    }

    pub fn render_if_needed(&mut self) -> anyhow::Result<()> {
        log::trace!("Orchestrator: Calling renderer.draw().");
        self.renderer.draw(&mut *self.term, &mut *self.driver)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests;
