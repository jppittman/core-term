// src/orchestrator.rs
//! Orchestrates the main application flow, coordinating between the PTY,
//! terminal emulator, renderer, and backend driver. This module aims to encapsulate
//! the core event processing logic, making it testable and maintainable by
//! abstracting away direct OS calls and backend specifics.

use crate::{
    ansi::AnsiParser, // AnsiProcessor import removed as only AnsiParser trait is used.
    backends::{BackendEvent, Driver},
    // Changed PtyError to anyhow::Error as PtyError is not defined in os/pty.rs
    os::pty::PtyChannel,
    renderer::Renderer, // Using the concrete Renderer struct.
    term::{EmulatorAction, EmulatorInput, TerminalInterface},
};
use anyhow::Error as AnyhowError; // Using anyhow::Error for PTY errors.
use std::io::ErrorKind as IoErrorKind; // For checking PTY read WouldBlock.

const PTY_READ_BUFFER_SIZE: usize = 4096; // Default buffer size for PTY reads.

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
    pub renderer: Renderer, // Renderer is now a concrete type.
    pub driver: &'a mut dyn Driver,

    // `needs_render` flag removed. Rendering decisions are now primarily driven by
    // the Renderer checking the TerminalEmulator's dirty state.
    pty_read_buffer: [u8; PTY_READ_BUFFER_SIZE], // Reusable buffer for PTY reads.
}

// This constructor takes 5 essential dependencies.
impl<'a> AppOrchestrator<'a> {
    /// Creates a new `AppOrchestrator`.
    ///
    /// # Arguments
    /// * `pty_channel` - A mutable reference to an initialized PTY channel.
    /// * `term` - A mutable reference to an initialized terminal emulator.
    /// * `parser` - An `AnsiParser` instance, typically new or reset for the session.
    /// * `renderer` - An initialized `Renderer` instance.
    /// * `driver` - A mutable reference to an initialized backend driver.
    pub fn new(
        pty_channel: &'a mut dyn PtyChannel,
        term: &'a mut dyn TerminalInterface,
        parser: &'a mut dyn AnsiParser,
        renderer: Renderer,
        driver: &'a mut dyn Driver,
    ) -> Self {
        AppOrchestrator {
            pty_channel,
            term,
            parser,
            renderer,
            driver,
            pty_read_buffer: [0; PTY_READ_BUFFER_SIZE],
            // `needs_render` field removed.
        }
    }

    /// Processes data that has become available for reading on the PTY channel.
    ///
    /// Reads from the PTY, parses the data into ANSI commands, interprets them
    /// with the terminal emulator, and handles any resulting actions.
    ///
    /// # Returns
    /// * `Ok(OrchestratorStatus::Running)` if PTY is active and data (or WouldBlock) was handled.
    /// * `Ok(OrchestratorStatus::Shutdown)` if PTY EOF is received, indicating the child process exited.
    /// * `Err(AnyhowError)` if a PTY read error (other than WouldBlock) occurs.
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
                self.interpret_pty_bytes(data_slice);
                Ok(OrchestratorStatus::Running)
            }
            Err(e) if e.kind() == IoErrorKind::WouldBlock => {
                log::trace!("Orchestrator: PTY read would block (no new data available).");
                Ok(OrchestratorStatus::Running) // Not an error for non-blocking PTY.
            }
            Err(e) => {
                log::error!("Orchestrator: Unrecoverable error reading from PTY: {}", e);
                // Wrap the std::io::Error into an anyhow::Error for consistent error handling.
                Err(AnyhowError::from(e).context("PTY read error"))
            }
        }
    }

    /// Parses a slice of bytes (typically from PTY) and interprets them.
    /// (Internal helper for `process_pty_events`)
    fn interpret_pty_bytes(&mut self, pty_data_slice: &[u8]) {
        let commands = self
            .parser
            .process_bytes(pty_data_slice)
            .into_iter()
            .map(EmulatorInput::Ansi); // Convert AnsiCommand to EmulatorInput::Ansi

        for command_input in commands {
            if let Some(action) = self.term.interpret_input(command_input) {
                self.handle_emulator_action(action);
            }
        }
        // `needs_render` flag removed. The terminal emulator now manages its dirty state internally.
        // The renderer will query this state.
    }

    /// Polls for and processes events from the backend driver (e.g., keyboard, resize).
    ///
    /// # Returns
    /// * `Ok(OrchestratorStatus)`: `Running` if events were handled or no events occurred,
    ///   `Shutdown` if a quit/close event was received from the driver.
    /// * `Err(String)`: If the driver reports an unrecoverable error during its event processing.
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

    /// Handles a specific `BackendEvent` payload (e.g., Key, Resize, Focus).
    /// (Internal helper for `process_driver_events`, called after checking for CloseRequested).
    fn handle_specific_driver_event(&mut self, event: BackendEvent) {
        match event {
            BackendEvent::Key { keysym, text } => {
                let user_input = EmulatorInput::User(BackendEvent::Key { keysym, text });
                if let Some(action) = self.term.interpret_input(user_input) {
                    self.handle_emulator_action(action);
                }
            }
            BackendEvent::Resize {
                width_px,
                height_px,
            } => {
                self.handle_resize_event(width_px, height_px);
            }
            BackendEvent::FocusGained => {
                log::debug!("Orchestrator: FocusGained event.");
                self.driver.set_focus(true); // Notify driver
                // TerminalEmulator might update its internal state if needed (e.g., for focus reporting sequences).
                // The renderer will query cursor visibility and draw appropriately.
            }
            BackendEvent::FocusLost => {
                log::debug!("Orchestrator: FocusLost event.");
                self.driver.set_focus(false); // Notify driver
            }
            BackendEvent::CloseRequested => {
                // This case should be handled by the caller (process_driver_events)
                // to immediately signal shutdown. Log if it reaches here.
                log::warn!(
                    "Orchestrator: CloseRequested event unexpectedly reached handle_specific_driver_event."
                );
            }
        }
    }

    /// Handles a resize event.
    fn handle_resize_event(&mut self, width_px: u16, height_px: u16) {
        let (char_width, char_height) = self.driver.get_font_dimensions();
        if char_width == 0 || char_height == 0 {
            log::warn!(
                "Orchestrator: Received resize but driver reported zero char dimensions ({}, {}). Ignoring resize.",
                char_width,
                char_height
            );
            return;
        }

        // Ensure dimensions are at least 1x1.
        let new_cols = (width_px as usize / char_width.max(1)).max(1);
        let new_rows = (height_px as usize / char_height.max(1)).max(1);

        log::info!(
            "Orchestrator: Resizing terminal to {}x{} cells ({}x{} px, char_size: {}x{})",
            new_cols,
            new_rows,
            width_px,
            height_px,
            char_width,
            char_height
        );

        // Assuming TerminalInterface and Renderer will have `resize` methods.
        // These calls will internally mark everything as dirty or handle necessary updates.
        self.term.resize(new_cols, new_rows);
        self.renderer.resize(new_cols, new_rows);
        // `needs_render` flag removed. Resize inherently means a full redraw will be
        // picked up by the renderer.
    }

    /// Handles actions signaled by the `TerminalInterface` implementation.
    fn handle_emulator_action(&mut self, action: EmulatorAction) {
        log::debug!("Orchestrator: Handling EmulatorAction: {:?}", action);
        match action {
            // EmulatorAction::RequestRedraw case removed.
            EmulatorAction::WritePty(data) => {
                if let Err(e) = self.pty_channel.write_all(&data) {
                    log::error!(
                        "Orchestrator: Failed to write_all {} bytes to PTY: {}",
                        data.len(),
                        e
                    );
                    // TODO: Consider how to handle critical PTY write failures.
                    // Options: Signal shutdown, attempt to continue, etc.
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
        }
    }

    /// Performs rendering.
    /// This is called once per event loop iteration after all inputs
    /// for that iteration have been processed. The `Renderer` will internally
    /// check the terminal's dirty state.
    ///
    /// # Returns
    /// `Ok(())` on successful render.
    /// `Err(anyhow::Error)` if the driver fails to present the frame.
    pub fn render_if_needed(&mut self) -> anyhow::Result<()> {
        // `needs_render` flag removed. Renderer::draw is called unconditionally.
        // The renderer itself will determine if actual drawing operations are needed
        // by querying the terminal's dirty state (e.g., via `term.take_dirty_lines()`).
        log::trace!("Orchestrator: Calling renderer.draw().");
        self.renderer.draw(&mut *self.term, &mut *self.driver)?; // Propagate driver errors
        Ok(())
    }

    // --- Test-only helper methods ---
    // These are useful for setting up specific states or asserting conditions in unit tests.
    #[cfg(test)]
    pub(crate) fn term_for_test(&mut self) -> &mut (dyn TerminalInterface + 'a) {
        self.term
    }
    #[cfg(test)]
    pub(crate) fn pty_for_test(&mut self) -> &mut (dyn PtyChannel + 'a) {
        self.pty_channel
    }
}
