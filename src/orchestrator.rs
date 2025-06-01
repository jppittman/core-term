// src/orchestrator.rs
//! Orchestrates the main application flow, coordinating between the PTY,
//! terminal emulator, renderer, and backend driver. This module aims to encapsulate
//! the core event processing logic, making it testable and maintainable by
//! abstracting away direct OS calls and backend specifics.

use crate::{
    ansi::AnsiParser,
    platform::backends::{BackendEvent, Driver, RenderCommand}, // Removed PlatformState
    platform::os::pty::PtyChannel,
    renderer::Renderer,
    term::{ControlEvent, EmulatorAction, EmulatorInput, TerminalInterface, UserInputAction},
};
use anyhow::Error as AnyhowError;
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
    pty_read_buffer: [u8; PTY_READ_BUFFER_SIZE],
    // Store relevant parts of PlatformState
    font_cell_width_px: usize,
    font_cell_height_px: usize,
    pending_render_actions: Vec<EmulatorAction>,
}

impl<'a> AppOrchestrator<'a> {
    /// Creates a new `AppOrchestrator`.
    pub fn new(
        pty_channel: &'a mut dyn PtyChannel,
        term: &'a mut dyn TerminalInterface,
        parser: &'a mut dyn AnsiParser,
        renderer: Renderer,
        driver: &'a mut dyn Driver, // driver is already mutable
    ) -> Self {
        // Get initial platform state
        let initial_platform_state = driver.get_platform_state();
        let font_cell_width_px = initial_platform_state.font_cell_width_px.max(1); // Ensure not zero
        let font_cell_height_px = initial_platform_state.font_cell_height_px.max(1); // Ensure not zero

        // Calculate initial dimensions for the terminal
        let initial_cols = (initial_platform_state.display_width_px as usize / font_cell_width_px).max(1);
        let initial_rows = (initial_platform_state.display_height_px as usize / font_cell_height_px).max(1);

        // Inform the terminal about its initial size and cell pixel size
        term.interpret_input(EmulatorInput::Control(ControlEvent::Resize {
            cols: initial_cols,
            rows: initial_rows,
        }));
        // Assuming TerminalEmulator has a way to receive cell pixel size,
        // or it recalculates internally based on new grid dimensions and display pixel size.
        // For now, let's assume `ControlEvent::Resize` is enough for the terminal to adjust.
        // If `term.update_cell_pixel_size` existed, it would be called here:
        // term.update_cell_pixel_size(font_cell_width_px, font_cell_height_px);

        // Inform the PTY of its initial size (cells only)
        if let Err(e) = pty_channel.resize(
            initial_cols as u16,
            initial_rows as u16
        ) {
            log::warn!("Orchestrator: Failed to set initial PTY size: {}", e);
        }

        AppOrchestrator {
            pty_channel,
            term,
            parser,
            renderer,
            driver,
            pty_read_buffer: [0; PTY_READ_BUFFER_SIZE],
            font_cell_width_px,
            font_cell_height_px,
            pending_render_actions: Vec::new(),
        }
    }

    /// Processes input for the terminal emulator and routes resulting actions.
    fn process_emulator_input(&mut self, input: EmulatorInput) {
        if let Some(action) = self.term.interpret_input(input) {
            match action {
                EmulatorAction::WritePty(_) |
                EmulatorAction::CopyToClipboard(_) |
                EmulatorAction::RequestClipboardContent => {
                    self.handle_non_render_emulator_action(action);
                }
                EmulatorAction::SetTitle(_) |
                EmulatorAction::RingBell |
                EmulatorAction::RequestRedraw |
                EmulatorAction::SetCursorVisibility(_) => {
                    self.pending_render_actions.push(action);
                }
            }
        }
    }

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
        for command_input in self.parser.process_bytes(pty_data_slice).into_iter().map(EmulatorInput::Ansi) {
            self.process_emulator_input(command_input);
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
                symbol,
                modifiers,
                text,
            } => {
                // New signature
                // Translate BackendEvent::Key to UserInputAction::KeyInput
                let key_input_action = UserInputAction::KeyInput {
                    symbol,                                                // from BackendEvent
                    modifiers,                                             // from BackendEvent
                    text: if text.is_empty() { None } else { Some(text) }, // Convert String to Option<String>
                };
                let user_input = EmulatorInput::User(key_input_action);
                self.process_emulator_input(user_input);
            }
            BackendEvent::Resize { .. } => { // width_px, height_px from event are noted but platform_state is source of truth
                let platform_state = self.driver.get_platform_state();

                // Update stored font dimensions, they might change (e.g. DPI change on X11)
                self.font_cell_width_px = platform_state.font_cell_width_px.max(1);
                self.font_cell_height_px = platform_state.font_cell_height_px.max(1);

                // Use dimensions from platform_state as it's the most current from the driver after resize
                let new_cols = (platform_state.display_width_px as usize / self.font_cell_width_px).max(1);
                let new_rows = (platform_state.display_height_px as usize / self.font_cell_height_px).max(1);

                log::info!(
                    "Orchestrator: Resizing to {}x{} cells ({}x{} px, char_size: {}x{})",
                    new_cols,
                    new_rows,
                    platform_state.display_width_px,
                    platform_state.display_height_px,
                    self.font_cell_width_px,
                    self.font_cell_height_px
                );

                // Resize PTY (cells only)
                if let Err(e) = self.pty_channel.resize(
                    new_cols as u16,
                    new_rows as u16
                ) {
                    log::warn!("Orchestrator: Failed to resize PTY: {}", e);
                }

                // Update terminal emulator dimensions
                let resize_emulator_input = EmulatorInput::Control(ControlEvent::Resize {
                    cols: new_cols,
                    rows: new_rows,
                });
                self.process_emulator_input(resize_emulator_input);
                // TODO: If TerminalEmulator needs explicit cell pixel size updates:
                // self.term.update_cell_pixel_size(self.font_cell_width_px, self.font_cell_height_px);
            }
            BackendEvent::FocusGained => {
                log::debug!("Orchestrator: FocusGained event.");
                self.driver.set_focus(crate::platform::backends::FocusState::Focused); // Corrected FocusState path
                self.process_emulator_input(EmulatorInput::User(UserInputAction::FocusGained));
            }
            BackendEvent::FocusLost => {
                log::debug!("Orchestrator: FocusLost event.");
                self.driver.set_focus(crate::platform::backends::FocusState::Unfocused); // Corrected FocusState path
                self.process_emulator_input(EmulatorInput::User(UserInputAction::FocusLost));
            }
            BackendEvent::CloseRequested => {
                log::warn!(
                    "Orchestrator: CloseRequested event unexpectedly reached handle_specific_driver_event."
                );
            }
            BackendEvent::PasteData { text } => {
                log::debug!("Orchestrator: PasteData event received with text length: {}", text.len());
                let paste_input_action = UserInputAction::PasteText(text);
                let user_input = EmulatorInput::User(paste_input_action);
                self.process_emulator_input(user_input);
            }
            BackendEvent::MouseButtonPress { button, x, y, modifiers: _ } => {
                if self.font_cell_width_px == 0 || self.font_cell_height_px == 0 {
                    log::warn!("Font dimensions are zero, cannot process mouse click.");
                    return;
                }
                let cell_x = (x as usize) / self.font_cell_width_px;
                let cell_y = (y as usize) / self.font_cell_height_px;
                log::debug!("MouseButtonPress: {:?} at pixel ({}, {}), cell ({}, {})", button, x, y, cell_x, cell_y);

                let user_action = match button {
                    crate::platform::backends::MouseButton::Left => {
                        Some(UserInputAction::StartSelection { x: cell_x, y: cell_y })
                    }
                    crate::platform::backends::MouseButton::Middle => {
                        Some(UserInputAction::RequestPrimaryPaste)
                    }
                    _ => None, // Other buttons ignored for now
                };

                if let Some(action) = user_action {
                    self.process_emulator_input(EmulatorInput::User(action));
                }
            }
            BackendEvent::MouseButtonRelease { button, x, y, modifiers: _ } => {
                if self.font_cell_width_px == 0 || self.font_cell_height_px == 0 {
                    log::warn!("Font dimensions are zero, cannot process mouse release.");
                    return;
                }
                let cell_x = (x as usize) / self.font_cell_width_px; // Though x,y might not be used for release action itself
                let cell_y = (y as usize) / self.font_cell_height_px;
                log::debug!("MouseButtonRelease: {:?} at pixel ({}, {}), cell ({}, {})", button, x, y, cell_x, cell_y);


                let user_action = match button {
                    crate::platform::backends::MouseButton::Left => {
                        Some(UserInputAction::ApplySelectionClear)
                    }
                    _ => None, // Other buttons ignored for now
                };

                if let Some(action) = user_action {
                    self.process_emulator_input(EmulatorInput::User(action));
                }
            }
            BackendEvent::MouseMove { x, y, modifiers: _ } => {
                if self.font_cell_width_px == 0 || self.font_cell_height_px == 0 {
                    log::warn!("Font dimensions are zero, cannot process mouse move.");
                    return;
                }
                let cell_x = (x as usize) / self.font_cell_width_px;
                let cell_y = (y as usize) / self.font_cell_height_px;
                log::trace!("MouseMove: at pixel ({}, {}), cell ({}, {})", x, y, cell_x, cell_y);


                let user_action = UserInputAction::ExtendSelection { x: cell_x, y: cell_y };
                self.process_emulator_input(EmulatorInput::User(user_action));
            }
        }
    }

    /// Handles emulator actions that do not directly generate render commands,
    /// such as PTY writes or clipboard operations.
    fn handle_non_render_emulator_action(&mut self, action: EmulatorAction) {
        log::debug!("Orchestrator: Handling non-render EmulatorAction: {:?}", action);
        match action {
            EmulatorAction::WritePty(data) => {
                if let Err(e) = self.pty_channel.write_all(&data) {
                    log::error!("Orchestrator: Failed to write_all {} bytes to PTY: {}", data.len(), e);
                } else {
                    log::trace!("Orchestrator: Wrote {} bytes to PTY.", data.len());
                }
            }
            EmulatorAction::CopyToClipboard(text) => {
                // TODO: Consider relocating these constants to platform::backends or as associated constants on the Driver trait.
                const TRAIT_ATOM_ID_PRIMARY: u64 = 1; // Example abstract ID for Primary selection
                const TRAIT_ATOM_ID_CLIPBOARD: u64 = 2; // Example abstract ID for Clipboard selection

                self.driver.own_selection(TRAIT_ATOM_ID_CLIPBOARD, text.clone());
                self.driver.own_selection(TRAIT_ATOM_ID_PRIMARY, text);
                log::info!("Orchestrator: Requested driver to own PRIMARY and CLIPBOARD selections.");
            }
            EmulatorAction::RequestClipboardContent => {
                // TODO: Consider relocating these constants to platform::backends or as associated constants on the Driver trait.
                const TRAIT_ATOM_ID_CLIPBOARD: u64 = 2; // Example abstract ID for Clipboard
                const TRAIT_ATOM_ID_UTF8_STRING: u64 = 10; // Example abstract ID for UTF8_STRING target

                self.driver.request_selection_data(TRAIT_ATOM_ID_CLIPBOARD, TRAIT_ATOM_ID_UTF8_STRING);
                log::info!("Orchestrator: Requested clipboard content from driver (target UTF8_STRING).");
            }
            _ => {
                log::warn!("Orchestrator: handle_non_render_emulator_action received an unexpected action: {:?}", action);
                // Or use unimplemented!("Action {:?} not handled by handle_non_render_emulator_action", action);
            }
        }
    }

    /// Handles emulator actions that generate render commands.
    fn handle_render_generating_emulator_action(&mut self, action: EmulatorAction, commands: &mut Vec<RenderCommand>) {
        log::debug!("Orchestrator: Handling render-generating EmulatorAction: {:?}", action);
        match action {
            EmulatorAction::SetTitle(title) => {
                commands.push(RenderCommand::SetWindowTitle { title });
            }
            EmulatorAction::RingBell => {
                commands.push(RenderCommand::RingBell);
            }
            EmulatorAction::RequestRedraw => {
                log::trace!("Orchestrator: EmulatorAction::RequestRedraw received (handled by render_if_needed).");
                // This action is a signal; actual redrawing is managed by the main loop's call to render_if_needed.
                // No direct command is generated here, but it's categorized as render-related.
            }
            EmulatorAction::SetCursorVisibility(visible) => {
                // This refers to the *native* OS cursor, not the terminal's drawn cursor.
                commands.push(RenderCommand::SetCursorVisibility { visible });
            }
            _ => {
                log::warn!("Orchestrator: handle_render_generating_emulator_action received an unexpected action: {:?}", action);
                // Or use unimplemented!("Action {:?} not handled by handle_render_generating_emulator_action", action);
            }
        }
    }

    pub fn render_if_needed(&mut self) -> anyhow::Result<()> {
        log::trace!("Orchestrator: Preparing render commands for frame.");
        let snapshot = self.term.get_render_snapshot();
        let mut final_render_commands = self.renderer.draw(snapshot)?;

        // Process any pending emulator actions that translate to render commands
        let actions_to_process: Vec<EmulatorAction> = self.pending_render_actions.drain(..).collect();
        for action in actions_to_process {
            // Non-render actions were already handled immediately and won't be in this list.
            self.handle_render_generating_emulator_action(action, &mut final_render_commands);
        }

        // Always add PresentFrame as the last command for this frame.
        final_render_commands.push(RenderCommand::PresentFrame);

        log::debug!("Orchestrator: Executing {} render commands.", final_render_commands.len());
        self.driver.execute_render_commands(final_render_commands)?;

        // Optionally, notify the terminal that the frame was rendered.
        // This can be useful for synchronization or for features like DA1 "Send Primary Device Attributes".
        // self.term.interpret_input(EmulatorInput::Control(ControlEvent::FrameRendered));

        Ok(())
    }
}

#[cfg(test)]
mod tests;
