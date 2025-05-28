// src/orchestrator.rs
//! Orchestrates the main application flow, coordinating between the PTY,
//! terminal emulator, renderer, and backend driver. This module aims to encapsulate
//! the core event processing logic, making it testable and maintainable by
//! abstracting away direct OS calls and backend specifics.

use crate::{
    ansi::AnsiParser,
    backends::{BackendEvent, Driver},
    os::pty::PtyChannel,
    renderer::Renderer,
    term::{action::{MouseInput, KeyInput}, ControlEvent, EmulatorAction, EmulatorInput, TerminalInterface, UserInputAction},
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
}

impl<'a> AppOrchestrator<'a> {
    /// Creates a new `AppOrchestrator`.
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
                symbol,
                modifiers,
                text,
            } => {
                let key_input_action = UserInputAction::KeyInput(KeyInput{
                    symbol,
                    modifiers,
                    text: if text.is_empty() { None } else { Some(text) },
                });
                let user_input = EmulatorInput::User(key_input_action);
                if let Some(action) = self.term.interpret_input(user_input) {
                    self.handle_emulator_action(action);
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
                    .interpret_input(EmulatorInput::User(UserInputAction::FocusGained))
                {
                    self.handle_emulator_action(action);
                }
            }
            BackendEvent::FocusLost => {
                log::debug!("Orchestrator: FocusLost event.");
                self.driver.set_focus(false);
                if let Some(action) = self
                    .term
                    .interpret_input(EmulatorInput::User(UserInputAction::FocusLost))
                {
                    self.handle_emulator_action(action);
                }
            }
            BackendEvent::CloseRequested => {
                log::warn!(
                    "Orchestrator: CloseRequested event unexpectedly reached handle_specific_driver_event."
                );
            }
            BackendEvent::Mouse {
                col,
                row,
                event_type,
                button,
                modifiers,
            } => {
                log::debug!(
                    "Orchestrator: Processing BackendEvent::Mouse: col={}, row={}, type={:?}, button={:?}, mods={:?}",
                    col,
                    row,
                    event_type,
                    button,
                    modifiers
                );
                let mouse_input_action = UserInputAction::MouseInput(MouseInput{
                    col,
                    row,
                    event_type,
                    button,
                    modifiers,
                });
                let user_input = EmulatorInput::User(mouse_input_action);
                if let Some(action) = self.term.interpret_input(user_input) {
                    self.handle_emulator_action(action);
                }
            }
        }
    }

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
            EmulatorAction::CopyToClipboard(_) => {
                unimplemented!("clipboard feature not yet implemented") // Corrected: remove !
            }
            EmulatorAction::RequestClipboardContent => {
                unimplemented!("clipboard feature not yet implemented") // Corrected: remove !
            }
        }
    }

    pub fn render_if_needed(&mut self) -> anyhow::Result<()> {
        log::trace!("Orchestrator: Calling renderer.draw().");
        let snapshot = self.term.get_render_snapshot();
        self.renderer.draw(snapshot, &mut *self.driver)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ansi::commands::AnsiCommand;
    use crate::backends::{BackendEvent, Driver, MouseButton, MouseEventType};
    use crate::keys::Modifiers;
    use crate::os::pty::PtyChannel;
    use crate::renderer::Renderer;
    use crate::term::snapshot::RenderSnapshot;
    use crate::term::{
        EmulatorAction, EmulatorInput, TerminalInterface, UserInputAction,
    };
    use anyhow::Result;
    use std::cell::RefCell;
    use std::collections::VecDeque;
    use std::io;
    use std::os::unix::io::RawFd;
    use crate::color::Color;

    struct MockPtyChannel {
        read_data: RefCell<VecDeque<Vec<u8>>>,
        written_data: RefCell<Vec<u8>>,
        resize_calls: RefCell<Vec<(u16, u16)>>,
    }

    impl MockPtyChannel {
        fn new() -> Self {
            MockPtyChannel {
                read_data: RefCell::new(VecDeque::new()),
                written_data: RefCell::new(Vec::new()),
                resize_calls: RefCell::new(Vec::new()),
            }
        }
        #[allow(dead_code)]
        fn expect_read_data(&self, data: Vec<u8>) {
            self.read_data.borrow_mut().push_back(data);
        }
    }

    impl PtyChannel for MockPtyChannel {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            if let Some(data) = self.read_data.borrow_mut().pop_front() {
                let len = std::cmp::min(buf.len(), data.len());
                buf[..len].copy_from_slice(&data[..len]);
                Ok(len)
            } else {
                Err(io::Error::new(io::ErrorKind::WouldBlock, "No data"))
            }
        }

        fn write_all(&mut self, buf: &[u8]) -> io::Result<()> {
            self.written_data.borrow_mut().extend_from_slice(buf);
            Ok(())
        }

        fn resize(&self, cols: u16, rows: u16) -> io::Result<()> {
            self.resize_calls.borrow_mut().push((cols, rows));
            Ok(())
        }
        fn get_raw_fd(&self) -> Option<RawFd> { Some(-1) }
    }

    struct MockTerminalInterface {
        inputs_received: RefCell<Vec<EmulatorInput>>,
        snapshot_to_return: Option<RenderSnapshot>,
    }

    impl MockTerminalInterface {
        fn new(snapshot: Option<RenderSnapshot>) -> Self {
            MockTerminalInterface {
                inputs_received: RefCell::new(Vec::new()),
                snapshot_to_return: snapshot,
            }
        }
    }

    impl TerminalInterface for MockTerminalInterface {
        fn interpret_input(&mut self, input: EmulatorInput) -> Option<EmulatorAction> {
            self.inputs_received.borrow_mut().push(input);
            None
        }

        fn get_render_snapshot(&self) -> RenderSnapshot {
             self.snapshot_to_return.clone().unwrap_or_else(|| RenderSnapshot {
                dimensions: (80, 24),
                lines: Vec::new(),
                cursor_state: None,
                selection_state: None,
            })
        }
    }

    struct MockAnsiParser {
        commands_to_return: RefCell<VecDeque<AnsiCommand>>,
    }

    impl MockAnsiParser {
        fn new() -> Self {
            MockAnsiParser {
                commands_to_return: RefCell::new(VecDeque::new()),
            }
        }
        #[allow(dead_code)]
        fn expect_commands(&self, commands: Vec<AnsiCommand>) {
            for cmd in commands {
                self.commands_to_return.borrow_mut().push_back(cmd);
            }
        }
    }

    impl AnsiParser for MockAnsiParser {
        fn process_bytes(&mut self, bytes: &[u8]) -> Vec<AnsiCommand> {
            let mut returned_cmds = Vec::new();
            if !bytes.is_empty() {
                while let Some(cmd) = self.commands_to_return.borrow_mut().pop_front() {
                    returned_cmds.push(cmd);
                }
            }
            returned_cmds
        }
    }

    struct MockDriver {
        events_to_process: RefCell<VecDeque<BackendEvent>>,
        font_dims: (usize, usize),
        focus_set_calls: RefCell<Vec<bool>>,
        title_set_calls: RefCell<Vec<String>>,
        bell_rung_count: RefCell<usize>,
        cursor_visibility_calls: RefCell<Vec<bool>>,
    }

    impl MockDriver {
        fn new(font_width: usize, font_height: usize) -> Self {
            MockDriver {
                events_to_process: RefCell::new(VecDeque::new()),
                font_dims: (font_width, font_height),
                focus_set_calls: RefCell::new(Vec::new()),
                title_set_calls: RefCell::new(Vec::new()),
                bell_rung_count: RefCell::new(0),
                cursor_visibility_calls: RefCell::new(Vec::new()),
            }
        }
        #[allow(dead_code)]
        fn queue_event(&self, event: BackendEvent) {
            self.events_to_process.borrow_mut().push_back(event);
        }
    }

    impl Driver for MockDriver {
        fn new() -> Result<Self> {
            Ok(Self::new(8,16))
        }
        fn get_event_fd(&self) -> Option<RawFd> { Some(-1) }
        fn process_events(&mut self) -> Result<Vec<BackendEvent>> {
            Ok(self.events_to_process.borrow_mut().drain(..).collect())
        }
        fn get_font_dimensions(&self) -> (usize, usize) { self.font_dims }
        fn get_display_dimensions_pixels(&self) -> (u16, u16) { (800, 600) }
        fn clear_all(&mut self, _bg: Color) -> Result<()> { Ok(()) }
        fn draw_text_run(&mut self, _coords: crate::backends::CellCoords, _text: &str, _style: crate::backends::TextRunStyle) -> Result<()> { Ok(()) }
        fn fill_rect(&mut self, _rect: crate::backends::CellRect, _color: Color) -> Result<()> { Ok(()) }
        fn present(&mut self) -> Result<()> { Ok(()) }
        fn set_title(&mut self, title: &str) { self.title_set_calls.borrow_mut().push(title.to_string()); }
        fn bell(&mut self) { *self.bell_rung_count.borrow_mut() += 1; }
        fn set_cursor_visibility(&mut self, visible: bool) { self.cursor_visibility_calls.borrow_mut().push(visible); }
        fn set_focus(&mut self, focused: bool) { self.focus_set_calls.borrow_mut().push(focused); }
        fn cleanup(&mut self) -> Result<()> { Ok(()) }
    }

    #[test]
    fn test_orchestrator_backend_mouse_event_to_emulator_input() {
        let mut mock_pty = MockPtyChannel::new();
        let mock_term_snapshot = RenderSnapshot {
            dimensions: (80, 24), lines: Vec::new(), cursor_state: None, selection_state: None,
        };
        let mut mock_term = MockTerminalInterface::new(Some(mock_term_snapshot));
        let mut mock_parser = MockAnsiParser::new();
        let renderer = Renderer::new();
        let mut mock_driver = MockDriver::new(8, 16);

        let mut orchestrator = AppOrchestrator::new(
            &mut mock_pty,
            &mut mock_term,
            &mut mock_parser,
            renderer,
            &mut mock_driver,
        );

        let backend_mouse_event = BackendEvent::Mouse {
            col: 5,
            row: 10,
            event_type: MouseEventType::Press,
            button: MouseButton::Left,
            modifiers: Modifiers::SHIFT,
        };
        
        mock_driver.queue_event(backend_mouse_event.clone());
        
        match orchestrator.process_driver_events() {
            Ok(status) => assert_eq!(status, OrchestratorStatus::Running),
            Err(e) => panic!("process_driver_events failed: {}", e),
        }

        let inputs = mock_term.inputs_received.borrow();
        assert_eq!(inputs.len(), 1, "Expected one input to be sent to terminal emulator");

        let expected_user_input_action = UserInputAction::MouseInput {
            col: 5,
            row: 10,
            event_type: MouseEventType::Press,
            button: MouseButton::Left,
            modifiers: Modifiers::SHIFT,
        };
        let expected_emulator_input = EmulatorInput::User(expected_user_input_action);

        assert_eq!(inputs[0], expected_emulator_input, "EmulatorInput does not match expected");
    }
}

