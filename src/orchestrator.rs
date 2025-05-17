// src/orchestrator.rs
// STYLE GUIDE: Rustdoc for module explaining its purpose.
//! Orchestrates the main application flow, coordinating between the PTY,
//! terminal emulator, renderer, and backend driver. This module aims to encapsulate
//! the core event processing logic, making it testable and maintainable by
//! abstracting away direct OS calls and backend specifics.

use crate::{
    ansi::{AnsiParser, AnsiProcessor},
    backends::{BackendEvent, Driver},
    os::pty::{PtyChannel, PtyError},
    renderer::*, // Using the Renderer abstraction.
    term::{EmulatorAction, EmulatorInput, TerminalInterface}, // Using the Terminal abstraction.
};
use std::io::ErrorKind as IoErrorKind; // For checking PTY read WouldBlock.

// STYLE GUIDE: Define Constants for magic numbers or common literals.
const PTY_READ_BUFFER_SIZE: usize = 4096; // Default buffer size for PTY reads.
// Consider making this configurable if needed.

/// Represents the status of the orchestrator after processing an event or an iteration of its loop.
// STYLE GUIDE: Return Values: Use dedicated enums for clarity, especially for status/control flow.
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
/// `RendererInterface`, `Driver`) to allow for mocking in tests and flexibility in choosing
/// concrete implementations.
// STYLE GUIDE: Rustdoc for public structs.
pub struct AppOrchestrator<'a> {
    // STYLE GUIDE: Fields use snake_case. Comments explain purpose if not obvious.
    // Using mutable references to allow these components to be owned elsewhere (e.g., in main)
    // and to facilitate testing with mocks that might also need mutable access.
    pty_channel: &'a mut dyn PtyChannel,
    term: &'a mut dyn TerminalInterface,
    parser: &'a mut dyn AnsiParser,
    pub renderer: Renderer,
    pub driver: &'a mut dyn Driver,

    needs_render: bool, // Internal state: true if a redraw is pending.
    pty_read_buffer: [u8; PTY_READ_BUFFER_SIZE], // Reusable buffer for PTY reads.
}

// STYLE GUIDE: Function argument count for `new`.
// This constructor takes 5 essential dependencies. For such a core orchestrating struct,
// this is generally acceptable. If configuration grows, a separate Config struct would be advised.
impl<'a> AppOrchestrator<'a> {
    /// Creates a new `AppOrchestrator`.
    ///
    /// # Arguments
    /// * `pty_channel` - A mutable reference to an initialized PTY channel.
    /// * `term` - A mutable reference to an initialized terminal emulator.
    /// * `parser` - An `AnsiParser` instance, typically new or reset for the session.
    /// * `renderer` - A mutable reference to an initialized renderer.
    /// * `driver` - A mutable reference to an initialized backend driver.
    // STYLE GUIDE: Rustdoc for public functions.
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
            needs_render: true, // Assume an initial render is always required.
            pty_read_buffer: [0; PTY_READ_BUFFER_SIZE],
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
    /// * `Err(PtyError)` if a PTY read error (other than WouldBlock) occurs.
    // STYLE GUIDE: Function argument count (0 excluding self) - OK.
    // STYLE GUIDE: Return value is a clear Result with a status enum.
    // STYLE GUIDE: Avoid deep nesting - match with guard clauses.
    pub fn process_pty_events(&mut self) -> Result<OrchestratorStatus, PtyError> {
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
                // Assuming PtyError can be created from std::io::Error via `From` trait.
                Err(PtyError::Io(e))
            }
        }
    }

    /// Parses a slice of bytes (typically from PTY) and interprets them.
    /// (Internal helper for `process_pty_events`)
    fn interpret_pty_bytes(&mut self, pty_data_slice: &[u8]) {
        // Assuming AnsiParser is stateful and its `advance` method consumes bytes
        // and makes `next_command` yield new commands.
        let commands = self
            .parser
            .process_bytes(pty_data_slice)
            .into_iter()
            .map(|cmd| EmulatorInput::Ansi(cmd));

        for command in commands {
            if let Some(action) = self.term.interpret_input(command) {
                self.handle_emulator_action(action);
            }
        }
        // Any processed PTY input implies a potential visual change.
        // EmulatorAction::RequestRedraw should ideally be the primary trigger.
        if !pty_data_slice.is_empty() {
            self.needs_render = true;
        }
    }

    /// Polls for and processes events from the backend driver (e.g., keyboard, resize).
    ///
    /// # Returns
    /// * `Ok(OrchestratorStatus)`: `Running` if events were handled or no events occurred,
    ///   `Shutdown` if a quit/close event was received from the driver.
    /// * `Err(String)`: If the driver reports an unrecoverable error during its event processing.
    // STYLE GUIDE: Function argument count - OK.
    pub fn process_driver_events(&mut self) -> Result<OrchestratorStatus, String> {
        log::trace!("Orchestrator: Processing available driver events...");
        // STYLE GUIDE: Loop with match and break is flatter than nested ifs for multiple events.
        // Assumes `self.driver.process_events()` returns a Vec of events.
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
            // STYLE GUIDE: Use early return for terminal conditions like CloseRequested.
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
        // STYLE GUIDE: Match is preferred for enums.
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
                // self.term.focus_changed(true); // If TerminalInterface supports this
                self.needs_render = true; // Cursor style might change, requiring redraw.
            }
            BackendEvent::FocusLost => {
                log::debug!("Orchestrator: FocusLost event.");
                self.driver.set_focus(false); // Notify driver
                // self.term.focus_changed(false);
                self.needs_render = true;
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
    // STYLE GUIDE: Extracted into helper to keep `handle_specific_driver_event` cleaner.
    fn handle_resize_event(&mut self, width_px: u16, height_px: u16) {
        let (char_width, char_height) = self.driver.get_font_dimensions();
        // STYLE GUIDE: Guard clause for invalid char dimensions.
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

        self.term.resize(new_cols, new_rows); // TerminalInterface::resize
        self.renderer.resize(new_cols, new_rows); // RendererInterface::resize
        self.needs_render = true; // Resize always requires a full redraw.
    }

    /// Handles actions signaled by the `TerminalInterface` implementation.
    // STYLE GUIDE: Function argument count - OK.
    fn handle_emulator_action(&mut self, action: EmulatorAction) {
        log::debug!("Orchestrator: Handling EmulatorAction: {:?}", action);
        match action {
            EmulatorAction::RequestRedraw => {
                self.needs_render = true;
            }
            EmulatorAction::WritePty(data) => {
                // STYLE GUIDE: Early return/clear error handling for I/O.
                // `write_all` attempts to write the entire buffer.
                if let Err(e) = self.pty_channel.write_all(&data) {
                    log::error!(
                        "Orchestrator: Failed to write_all {} bytes to PTY: {}",
                        data.len(),
                        e
                    );
                    // TODO: Critical PTY write failure. Consider signaling error to main loop for shutdown.
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
            EmulatorAction::SetCursorVisibility(visible) => {
                self.driver.set_cursor_visibility(visible);
                self.needs_render = true; // Cursor visibility change requires redraw.
            }
        }
    }

    /// Performs rendering if the `needs_render` flag is set.
    /// This should be called once per event loop iteration after all inputs
    /// for that iteration have been processed.
    ///
    /// # Returns
    /// `Ok(())` on successful render or if no render was needed.
    /// `Err(anyhow::Error)` if the driver fails to present the frame.
    // STYLE GUIDE: Function argument count - OK.
    pub fn render_if_needed(&mut self) -> anyhow::Result<()> {
        // STYLE GUIDE: Guard clause for early return.
        if !self.needs_render {
            log::trace!("Orchestrator: No render needed for this iteration.");
            return Ok(());
        }

        log::info!("Orchestrator: Performing render.");
        // This is the critical point:
        // `renderer.draw` will call `term.take_dirty_lines()`.
        // No other calls to `term.take_dirty_lines()` should occur in the orchestrator's
        // flow between PTY/driver event processing and this render call for those events.
        self.renderer.draw(&mut *self.term, &mut *self.driver)?; // Propagate driver errors
        self.needs_render = false; // Reset flag *after* successful rendering.
        Ok(())
    }

    // --- Test-only helper methods ---
    // These are useful for setting up specific states or asserting conditions in unit tests.
    // STYLE GUIDE: Test public API. These helpers are `cfg(test)`.
    #[cfg(test)]
    pub(crate) fn flag_render_as_needed_for_test(&mut self) {
        self.needs_render = true;
    }

    #[cfg(test)]
    pub(crate) fn consume_render_flag_for_test(&mut self) -> bool {
        let needed = self.needs_render;
        self.needs_render = false;
        needed
    }

    #[cfg(test)]
    pub(crate) fn term_for_test(&mut self) -> &mut (dyn TerminalInterface + 'a) {
        self.term
    }
    #[cfg(test)]
    pub(crate) fn pty_for_test(&mut self) -> &mut (dyn PtyChannel + 'a) {
        self.pty_channel
    }
}
