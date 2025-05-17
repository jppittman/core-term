// core-term/orchestrator/tests.rs

//! Unit tests for the AppOrchestrator.
//! These tests focus on the orchestrator's interactions with its dependencies:
//! PtyChannel, TerminalInterface (TerminalEmulator), AnsiParser, Renderer, and Driver.
//! Tests aim for robustness by verifying exact call sequences and parameters on mocks.

#[cfg(test)]
mod orchestrator_tests {
    use crate::ansi::{AnsiCommand, AnsiParser as AnsiParserTrait};
    use crate::backends::{BackendEvent, CellCoords, CellRect, Driver, TextRunStyle};
    use crate::color::{Color, NamedColor};
    use crate::glyph::{AttrFlags, Attributes, Glyph};
    use crate::orchestrator::{AppOrchestrator, OrchestratorStatus};
    use crate::os::pty::PtyChannel;
    use crate::renderer::{Renderer, RENDERER_DEFAULT_BG, RENDERER_DEFAULT_FG};
    use crate::term::{
        ControlEvent, EmulatorAction, EmulatorInput, TerminalInterface,
    };

    use anyhow::Result;
    use nix::unistd::Pid;
    use std::collections::{HashSet, VecDeque};
    use std::io::{self, ErrorKind, Read, Write};
    use std::os::unix::io::{AsRawFd, RawFd};
    use std::sync::{Arc, Mutex};
    use test_log::test; // For logging within tests

    // --- Mock PtyChannel ---
    #[derive(Debug)]
    struct MockPtyChannel {
        read_buffer: Arc<Mutex<VecDeque<u8>>>,
        write_buffer: Arc<Mutex<Vec<u8>>>,
        child_pid_val: Pid,
        is_eof: Arc<Mutex<bool>>,
        resize_calls: Arc<Mutex<Vec<(u16, u16)>>>,
    }

    impl MockPtyChannel {
        fn new() -> Self {
            Self {
                read_buffer: Arc::new(Mutex::new(VecDeque::new())),
                write_buffer: Arc::new(Mutex::new(Vec::new())),
                child_pid_val: Pid::from_raw(1234),
                is_eof: Arc::new(Mutex::new(false)),
                resize_calls: Arc::new(Mutex::new(Vec::new())),
            }
        }

        fn push_bytes_to_read(&self, bytes: &[u8]) {
            self.read_buffer.lock().unwrap().extend(bytes);
        }

        fn set_eof(&self, eof_state: bool) {
            *self.is_eof.lock().unwrap() = eof_state;
        }

        fn get_written_data(&self) -> Vec<u8> {
            self.write_buffer.lock().unwrap().clone()
        }

        fn clear_written_data(&self) {
            self.write_buffer.lock().unwrap().clear();
        }

        fn get_resize_calls(&self) -> Vec<(u16, u16)> {
            self.resize_calls.lock().unwrap().clone()
        }
    }

    impl Read for MockPtyChannel {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            let mut read_buf = self.read_buffer.lock().unwrap();
            if *self.is_eof.lock().unwrap() && read_buf.is_empty() {
                return Ok(0);
            }
            if read_buf.is_empty() {
                return Err(io::Error::new(ErrorKind::WouldBlock, "No data"));
            }
            let len = std::cmp::min(buf.len(), read_buf.len());
            for i in 0..len {
                buf[i] = read_buf.pop_front().unwrap();
            }
            Ok(len)
        }
    }

    impl Write for MockPtyChannel {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            self.write_buffer.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    impl AsRawFd for MockPtyChannel {
        fn as_raw_fd(&self) -> RawFd {
            -1
        }
    }

    impl PtyChannel for MockPtyChannel {
        fn resize(&self, cols: u16, rows: u16) -> Result<()> {
            self.resize_calls.lock().unwrap().push((cols, rows));
            Ok(())
        }
        fn child_pid(&self) -> Pid {
            self.child_pid_val
        }
    }

    // --- Mock AnsiParser ---
    struct MockAnsiParser {
        commands_to_return_on_next_call: VecDeque<AnsiCommand>,
        bytes_processed_log: Arc<Mutex<Vec<Vec<u8>>>>,
    }

    impl MockAnsiParser {
        fn new() -> Self {
            Self {
                commands_to_return_on_next_call: VecDeque::new(),
                bytes_processed_log: Arc::new(Mutex::new(Vec::new())),
            }
        }

        fn expect_commands_for_next_call(&mut self, cmds: Vec<AnsiCommand>) {
            self.commands_to_return_on_next_call = cmds.into();
        }

         fn get_processed_bytes_log(&self) -> Vec<Vec<u8>> {
            self.bytes_processed_log.lock().unwrap().clone()
        }
        #[allow(dead_code)]
         fn clear_processed_bytes_log(&self) {
            self.bytes_processed_log.lock().unwrap().clear();
        }
    }

    impl AnsiParserTrait for MockAnsiParser {
        fn process_bytes(&mut self, bytes: &[u8]) -> Vec<AnsiCommand> {
            self.bytes_processed_log.lock().unwrap().push(bytes.to_vec());
            self.commands_to_return_on_next_call.drain(..).collect()
        }
    }

    // --- Mock TerminalInterface ---
    #[derive(Clone)]
    struct MockTerminal {
        width: Arc<Mutex<usize>>,
        height: Arc<Mutex<usize>>,
        cursor_x: Arc<Mutex<usize>>,
        cursor_y: Arc<Mutex<usize>>,
        cursor_visible: Arc<Mutex<bool>>,
        grid: Arc<Mutex<Vec<Vec<Glyph>>>>,
        dirty_lines: Arc<Mutex<HashSet<usize>>>,
        inputs_received: Arc<Mutex<Vec<EmulatorInput>>>,
        actions_to_return_on_next_call: Arc<Mutex<VecDeque<Option<EmulatorAction>>>>,
        #[allow(dead_code)] // May not be used in all tests directly
        scrollback_limit: usize,
        default_attributes: Attributes,
    }

    impl MockTerminal {
        fn new(width: usize, height: usize, scrollback_limit: usize) -> Self {
            let default_attrs = Attributes {
                fg: RENDERER_DEFAULT_FG,
                bg: RENDERER_DEFAULT_BG,
                flags: AttrFlags::empty(),
            };
            let initial_dirty_lines: HashSet<usize> = (0..height).collect();
            Self {
                width: Arc::new(Mutex::new(width)),
                height: Arc::new(Mutex::new(height)),
                cursor_x: Arc::new(Mutex::new(0)),
                cursor_y: Arc::new(Mutex::new(0)),
                cursor_visible: Arc::new(Mutex::new(true)),
                grid: Arc::new(Mutex::new(vec![
                    vec![
                        Glyph {
                            c: ' ',
                            attr: default_attrs
                        };
                        width
                    ];
                    height
                ])),
                dirty_lines: Arc::new(Mutex::new(initial_dirty_lines)),
                inputs_received: Arc::new(Mutex::new(Vec::new())),
                actions_to_return_on_next_call: Arc::new(Mutex::new(VecDeque::new())),
                scrollback_limit,
                default_attributes: default_attrs,
            }
        }

        fn expect_action_for_next_call(&mut self, action: Option<EmulatorAction>) {
            self.actions_to_return_on_next_call.lock().unwrap().push_back(action);
        }

         fn get_inputs_received(&self) -> Vec<EmulatorInput> {
            self.inputs_received.lock().unwrap().clone()
        }

        fn clear_inputs_received(&self) {
            self.inputs_received.lock().unwrap().clear();
        }

        fn mark_line_dirty(&mut self, y: usize) {
            if y < *self.height.lock().unwrap() {
                self.dirty_lines.lock().unwrap().insert(y);
            }
        }
    }

    impl TerminalInterface for MockTerminal {
        fn dimensions(&self) -> (usize, usize) {
            (*self.width.lock().unwrap(), *self.height.lock().unwrap())
        }
        fn get_glyph(&self, x: usize, y: usize) -> Glyph {
            let grid = self.grid.lock().unwrap();
            grid.get(y)
                .and_then(|row| row.get(x))
                .cloned()
                .unwrap_or_else(|| Glyph {
                    c: ' ',
                    attr: self.default_attributes,
                })
        }
        fn is_cursor_visible(&self) -> bool {
            *self.cursor_visible.lock().unwrap()
        }
        fn get_screen_cursor_pos(&self) -> (usize, usize) {
            (*self.cursor_x.lock().unwrap(), *self.cursor_y.lock().unwrap())
        }
        fn take_dirty_lines(&mut self) -> Vec<usize> {
            let mut lines: Vec<usize> = self.dirty_lines.lock().unwrap().drain().collect();
            lines.sort_unstable();
            lines
        }
        fn interpret_input(&mut self, input: EmulatorInput) -> Option<EmulatorAction> {
            self.inputs_received.lock().unwrap().push(input.clone());
            match input {
                EmulatorInput::Control(ControlEvent::Resize{cols, rows}) => {
                    let mut w = self.width.lock().unwrap();
                    let mut h = self.height.lock().unwrap();
                    *w = cols;
                    *h = rows;
                    let mut grid_mut = self.grid.lock().unwrap();
                    *grid_mut = vec![vec![Glyph {c: ' ', attr: self.default_attributes}; cols]; rows];
                    let mut dirty = self.dirty_lines.lock().unwrap();
                    dirty.clear();
                    for i in 0..rows {
                        dirty.insert(i);
                    }
                }
                _ => {}
            }
            self.actions_to_return_on_next_call.lock().unwrap().pop_front().unwrap_or(None)
        }
    }

    // --- Mock Driver ---
    #[derive(Debug, Clone, PartialEq)]
    enum MockDriverCall {
        ClearAll { bg: Color },
        DrawTextRun { coords: CellCoords, text: String, style: TextRunStyle },
        FillRect { rect: CellRect, color: Color },
        Present,
        SetTitle { title: String },
        Bell,
        SetCursorVisibility { visible: bool },
        SetFocus { focused: bool },
        Cleanup,
        GetFontDimensions,
        GetDisplayDimensionsPixels,
        ProcessEvents,
    }

    struct MockDriver {
        calls: Arc<Mutex<Vec<MockDriverCall>>>,
        events_to_return_on_next_call: Arc<Mutex<VecDeque<BackendEvent>>>,
        font_dims: (usize, usize),
        display_dims_px: (u16, u16),
        event_fd: Option<RawFd>,
        focus_state: Arc<Mutex<bool>>,
    }

    impl MockDriver {
        fn new() -> Self {
            Self {
                calls: Arc::new(Mutex::new(Vec::new())),
                events_to_return_on_next_call: Arc::new(Mutex::new(VecDeque::new())),
                font_dims: (8, 16),
                display_dims_px: (640, 480),
                event_fd: None,
                focus_state: Arc::new(Mutex::new(true)),
            }
        }

        fn expect_events_for_next_call(&mut self, events: Vec<BackendEvent>) {
            *self.events_to_return_on_next_call.lock().unwrap() = events.into();
        }

        fn get_calls(&self) -> Vec<MockDriverCall> {
            self.calls.lock().unwrap().clone()
        }

        fn clear_calls(&self) {
            self.calls.lock().unwrap().clear();
        }
    }

    impl Driver for MockDriver {
        fn new() -> Result<Self> { Ok(Self::new()) }
        fn get_event_fd(&self) -> Option<RawFd> { self.event_fd }
        fn process_events(&mut self) -> Result<Vec<BackendEvent>> {
            self.calls.lock().unwrap().push(MockDriverCall::ProcessEvents);
            Ok(self.events_to_return_on_next_call.lock().unwrap().drain(..).collect())
        }
        fn get_font_dimensions(&self) -> (usize, usize) {
            self.calls.lock().unwrap().push(MockDriverCall::GetFontDimensions);
            self.font_dims
        }
        fn get_display_dimensions_pixels(&self) -> (u16, u16) {
            self.calls.lock().unwrap().push(MockDriverCall::GetDisplayDimensionsPixels);
            self.display_dims_px
        }
        fn clear_all(&mut self, bg: Color) -> Result<()> {
            self.calls.lock().unwrap().push(MockDriverCall::ClearAll { bg }); Ok(())
        }
        fn draw_text_run(&mut self, coords: CellCoords, text: &str, style: TextRunStyle) -> Result<()> {
            self.calls.lock().unwrap().push(MockDriverCall::DrawTextRun { coords, text: text.to_string(), style }); Ok(())
        }
        fn fill_rect(&mut self, rect: CellRect, color: Color) -> Result<()> {
            self.calls.lock().unwrap().push(MockDriverCall::FillRect { rect, color }); Ok(())
        }
        fn present(&mut self) -> Result<()> {
            self.calls.lock().unwrap().push(MockDriverCall::Present); Ok(())
        }
        fn set_title(&mut self, title: &str) {
            self.calls.lock().unwrap().push(MockDriverCall::SetTitle { title: title.to_string() });
        }
        fn bell(&mut self) {
            self.calls.lock().unwrap().push(MockDriverCall::Bell);
        }
        fn set_cursor_visibility(&mut self, visible: bool) {
            self.calls.lock().unwrap().push(MockDriverCall::SetCursorVisibility { visible });
        }
        fn set_focus(&mut self, focused: bool) {
            self.calls.lock().unwrap().push(MockDriverCall::SetFocus { focused });
            *self.focus_state.lock().unwrap() = focused;
        }
        fn cleanup(&mut self) -> Result<()> {
            self.calls.lock().unwrap().push(MockDriverCall::Cleanup); Ok(())
        }
    }

    // Helper function to create orchestrator for tests that need to re-instantiate it.
    // Note: Renderer needs to be Clone if we pass it by value and re-create orchestrator.
    // Assuming Renderer is Cloneable for the interleaved test.
    fn create_orchestrator<'a>(
        pty: &'a mut dyn PtyChannel,
        term: &'a mut dyn TerminalInterface,
        parser: &'a mut dyn AnsiParserTrait,
        renderer: Renderer, // Takes ownership, will be moved
        driver: &'a mut dyn Driver,
    ) -> AppOrchestrator<'a> {
        AppOrchestrator::new(pty, term, parser, renderer, driver)
    }


    #[test]
    fn test_orchestrator_pty_data_flow_exact_verification() {
        let mut mock_pty = MockPtyChannel::new();
        let mut mock_term = MockTerminal::new(10, 1, 0);
        let mut mock_parser = MockAnsiParser::new();
        let renderer = Renderer::new();
        let mut mock_driver = MockDriver::new();

        let pty_input_data = b"Hi";
        mock_pty.push_bytes_to_read(pty_input_data);

        let expected_parsed_commands = vec![AnsiCommand::Print('H'), AnsiCommand::Print('i')];
        mock_parser.expect_commands_for_next_call(expected_parsed_commands.clone());

        let pty_action_response = EmulatorAction::WritePty(b"ack_H".to_vec());
        mock_term.expect_action_for_next_call(Some(pty_action_response.clone()));
        mock_term.expect_action_for_next_call(None);
        
        let status;
        // Scope for orchestrator to release borrows
        {
            let mut orchestrator = create_orchestrator(
                &mut mock_pty, &mut mock_term, &mut mock_parser, renderer, &mut mock_driver
            );
            status = orchestrator.process_pty_events().unwrap();
        }
        assert_eq!(status, OrchestratorStatus::Running, "Orchestrator should be running");

        assert_eq!(mock_parser.get_processed_bytes_log(), vec![pty_input_data.to_vec()], "Parser did not receive exact PTY data");
        let expected_inputs_to_term: Vec<EmulatorInput> = expected_parsed_commands.into_iter().map(EmulatorInput::Ansi).collect();
        assert_eq!(mock_term.get_inputs_received(), expected_inputs_to_term, "Terminal did not receive exact parsed commands");
        assert_eq!(mock_pty.get_written_data(), b"ack_H", "PTY did not receive exact action response");
    }

    #[test]
    fn test_orchestrator_driver_key_event_flow_exact_verification() {
        let mut mock_pty = MockPtyChannel::new();
        let mut mock_term = MockTerminal::new(10, 1, 0);
        let mut mock_parser = MockAnsiParser::new();
        let renderer = Renderer::new();
        let mut mock_driver = MockDriver::new();

        let key_event = BackendEvent::Key { keysym: 'X' as u32, text: "X".to_string() };
        mock_driver.expect_events_for_next_call(vec![key_event.clone()]);

        let pty_response_action = EmulatorAction::WritePty(b"X_response".to_vec());
        mock_term.expect_action_for_next_call(Some(pty_response_action.clone()));

        let status;
        {
            let mut orchestrator = create_orchestrator(
                &mut mock_pty, &mut mock_term, &mut mock_parser, renderer, &mut mock_driver
            );
            status = orchestrator.process_driver_events().unwrap();
        }
        assert_eq!(status, OrchestratorStatus::Running, "Orchestrator should be running");

        assert_eq!(mock_term.get_inputs_received(), vec![EmulatorInput::User(key_event)], "Terminal did not receive exact key event");
        assert_eq!(mock_pty.get_written_data(), b"X_response", "PTY did not receive exact response for key event");
        assert!(mock_driver.get_calls().contains(&MockDriverCall::ProcessEvents));
    }

    #[test]
    fn test_orchestrator_driver_resize_event_flow_exact_verification() {
        let mut mock_pty = MockPtyChannel::new();
        let mut mock_term = MockTerminal::new(10, 5, 0);
        let mut mock_parser = MockAnsiParser::new();
        let renderer = Renderer::new();
        let mut mock_driver = MockDriver::new();
        mock_driver.font_dims = (10, 20);

        let resize_event_px = BackendEvent::Resize { width_px: 800, height_px: 600 };
        mock_driver.expect_events_for_next_call(vec![resize_event_px.clone()]);
        mock_term.expect_action_for_next_call(None);

        let status;
        {
            let mut orchestrator = create_orchestrator(
                 &mut mock_pty, &mut mock_term, &mut mock_parser, renderer, &mut mock_driver
            );
            status = orchestrator.process_driver_events().unwrap();
        }
        assert_eq!(status, OrchestratorStatus::Running);

        let expected_cols = 80;
        let expected_rows = 30;

        assert_eq!(mock_pty.get_resize_calls(), vec![(expected_cols as u16, expected_rows as u16)], "PTY resize call mismatch");
        assert_eq!(mock_term.get_inputs_received(), vec![EmulatorInput::Control(ControlEvent::Resize {
            cols: expected_cols,
            rows: expected_rows,
        })], "Terminal did not receive correct resize control event");
        assert_eq!(mock_term.dimensions(), (expected_cols, expected_rows), "MockTerminal dimensions not updated as expected");

        let driver_calls = mock_driver.get_calls();
        assert!(driver_calls.contains(&MockDriverCall::ProcessEvents));
        assert!(driver_calls.contains(&MockDriverCall::GetFontDimensions));
    }

    #[test]
    fn test_render_if_needed_no_dirty_lines_no_first_draw() {
        let mut mock_pty = MockPtyChannel::new();
        let mut mock_term = MockTerminal::new(1, 1, 0);
        let _ = mock_term.take_dirty_lines(); // Clear initial dirty lines
        let mut mock_parser = MockAnsiParser::new();
        let mut renderer = Renderer::new();
        renderer.first_draw = false;
        let mut mock_driver = MockDriver::new();
        
        {
            let mut orchestrator = create_orchestrator(
                &mut mock_pty, &mut mock_term, &mut mock_parser, renderer, &mut mock_driver
            );
            orchestrator.render_if_needed().unwrap();
        }
        let driver_calls = mock_driver.get_calls();
        assert!(driver_calls.is_empty(), "Expected no driver calls when no rendering is needed. Got: {:?}", driver_calls);
    }

    #[test]
    fn test_render_if_needed_first_draw_scenario() {
        let mut mock_pty = MockPtyChannel::new();
        let mut mock_term = MockTerminal::new(2, 1, 0);
        let mut mock_parser = MockAnsiParser::new();
        let mut renderer_instance = Renderer::new(); // first_draw is true by default
        let mut mock_driver = MockDriver::new();
        let first_draw_flag_after_render;

        {
            let mut orchestrator = create_orchestrator(
                &mut mock_pty, &mut mock_term, &mut mock_parser, renderer_instance.clone(), &mut mock_driver
            );
            orchestrator.render_if_needed().unwrap();
            first_draw_flag_after_render = orchestrator.renderer.first_draw;
        }

        let driver_calls = mock_driver.get_calls();
        let expected_calls = vec![
            MockDriverCall::ClearAll { bg: RENDERER_DEFAULT_BG },
            MockDriverCall::FillRect {
                rect: CellRect { x: 0, y: 0, width: 2, height: 1 },
                color: RENDERER_DEFAULT_BG,
            },
            MockDriverCall::DrawTextRun {
                coords: CellCoords { x: 0, y: 0 },
                text: " ".to_string(),
                style: TextRunStyle {
                    fg: RENDERER_DEFAULT_BG,
                    bg: RENDERER_DEFAULT_FG,
                    flags: AttrFlags::empty(),
                },
            },
            MockDriverCall::Present,
        ];
        assert_eq!(driver_calls, expected_calls, "Driver call sequence mismatch for first_draw");
        assert!(!first_draw_flag_after_render, "first_draw flag should be false after initial render");
    }


    #[test]
    fn test_orchestrator_pty_eof_signals_shutdown() {
        let mut mock_pty = MockPtyChannel::new();
        mock_pty.set_eof(true);
        let mut mock_term = MockTerminal::new(1, 1, 0);
        let mut mock_parser = MockAnsiParser::new();
        let renderer = Renderer::new();
        let mut mock_driver = MockDriver::new();
        
        let status;
        {
             let mut orchestrator = create_orchestrator(
                &mut mock_pty, &mut mock_term, &mut mock_parser, renderer, &mut mock_driver
            );
            status = orchestrator.process_pty_events().unwrap();
        }
        assert_eq!(status, OrchestratorStatus::Shutdown);
    }

    #[test]
    fn test_orchestrator_driver_close_requested_signals_shutdown() {
        let mut mock_pty = MockPtyChannel::new();
        let mut mock_term = MockTerminal::new(1, 1, 0);
        let mut mock_parser = MockAnsiParser::new();
        let renderer = Renderer::new();
        let mut mock_driver = MockDriver::new();

        mock_driver.expect_events_for_next_call(vec![BackendEvent::CloseRequested]);
        
        let status;
        {
            let mut orchestrator = create_orchestrator(
                &mut mock_pty, &mut mock_term, &mut mock_parser, renderer, &mut mock_driver
            );
            status = orchestrator.process_driver_events().unwrap();
        }
        assert_eq!(status, OrchestratorStatus::Shutdown);
    }

    #[test]
    fn test_orchestrator_handles_emulator_action_set_title_exact() {
        let mut mock_pty = MockPtyChannel::new();
        mock_pty.push_bytes_to_read(b"T");
        let mut mock_term = MockTerminal::new(1, 1, 0);
        let mut mock_parser = MockAnsiParser::new();
        let renderer = Renderer::new();
        let mut mock_driver = MockDriver::new();

        mock_parser.expect_commands_for_next_call(vec![AnsiCommand::Print('T')]);
        mock_term.expect_action_for_next_call(Some(EmulatorAction::SetTitle("Exact Title".to_string())));
        
        {
            let mut orchestrator = create_orchestrator(
                &mut mock_pty, &mut mock_term, &mut mock_parser, renderer, &mut mock_driver
            );
            orchestrator.process_pty_events().unwrap();
        }
        let driver_calls = mock_driver.get_calls();
        assert_eq!(driver_calls, vec![MockDriverCall::SetTitle { title: "Exact Title".to_string() }]);
    }

    #[test]
    fn test_orchestrator_handles_emulator_action_ring_bell_exact() {
        let mut mock_pty = MockPtyChannel::new();
        mock_pty.push_bytes_to_read(b"\x07");
        let mut mock_term = MockTerminal::new(1, 1, 0);
        let mut mock_parser = MockAnsiParser::new();
        let renderer = Renderer::new();
        let mut mock_driver = MockDriver::new();

        mock_parser.expect_commands_for_next_call(vec![AnsiCommand::C0Control(crate::ansi::commands::C0Control::BEL)]);
        mock_term.expect_action_for_next_call(Some(EmulatorAction::RingBell));
        
        {
            let mut orchestrator = create_orchestrator(
                 &mut mock_pty, &mut mock_term, &mut mock_parser, renderer, &mut mock_driver
            );
            orchestrator.process_pty_events().unwrap();
        }
        assert_eq!(mock_driver.get_calls(), vec![MockDriverCall::Bell]);
    }

    #[test]
    fn test_orchestrator_handles_focus_events_exact() {
        let mut mock_pty = MockPtyChannel::new();
        let mut mock_term = MockTerminal::new(10, 1, 0);
        let mut mock_parser = MockAnsiParser::new();
        // Renderer needs to be mutable if we want to check its state or pass it to multiple orchestrators
        let mut renderer = Renderer::new(); 
        let mut mock_driver = MockDriver::new();

        // Test FocusGained
        mock_driver.expect_events_for_next_call(vec![BackendEvent::FocusGained]);
        mock_term.expect_action_for_next_call(None);
        let status_gain;
        {
            let mut orchestrator = create_orchestrator(
                &mut mock_pty, &mut mock_term, &mut mock_parser, renderer.clone(), &mut mock_driver
            );
            status_gain = orchestrator.process_driver_events().unwrap();
        }
        assert_eq!(status_gain, OrchestratorStatus::Running);
        assert_eq!(mock_driver.get_calls(), vec![MockDriverCall::ProcessEvents, MockDriverCall::SetFocus{focused: true}]);
        assert_eq!(mock_term.get_inputs_received(), vec![EmulatorInput::User(BackendEvent::FocusGained)]);

        mock_driver.clear_calls();
        mock_term.clear_inputs_received();

        // Test FocusLost
        mock_driver.expect_events_for_next_call(vec![BackendEvent::FocusLost]);
        mock_term.expect_action_for_next_call(None);
        let status_lost;
        {
            // Re-assign renderer if it was moved, or clone if it's Cloneable
             renderer = Renderer::new(); // Re-create or clone
             let mut orchestrator = create_orchestrator(
                &mut mock_pty, &mut mock_term, &mut mock_parser, renderer.clone(), &mut mock_driver
            );
            status_lost = orchestrator.process_driver_events().unwrap();
        }
        assert_eq!(status_lost, OrchestratorStatus::Running);
        assert_eq!(mock_driver.get_calls(), vec![MockDriverCall::ProcessEvents, MockDriverCall::SetFocus{focused: false}]);
        assert_eq!(mock_term.get_inputs_received(), vec![EmulatorInput::User(BackendEvent::FocusLost)]);
    }

    #[test]
    fn test_orchestrator_interleaved_pty_and_driver_events() {
        let mut mock_pty = MockPtyChannel::new();
        let mut mock_term = MockTerminal::new(10, 1, 0);
        let mut mock_parser = MockAnsiParser::new();
        let mut renderer = Renderer::new(); // Make it mutable to be re-used or cloned
        let mut mock_driver = MockDriver::new();

        // --- Phase 1: PTY data ---
        mock_pty.push_bytes_to_read(b"P1");
        mock_parser.expect_commands_for_next_call(vec![AnsiCommand::Print('P'), AnsiCommand::Print('1')]);
        mock_term.expect_action_for_next_call(Some(EmulatorAction::WritePty(b"AckP1".to_vec())));
        mock_term.expect_action_for_next_call(None);
        {
            let mut orchestrator = create_orchestrator(
                &mut mock_pty, &mut mock_term, &mut mock_parser, renderer.clone(), &mut mock_driver
            );
            assert_eq!(orchestrator.process_pty_events().unwrap(), OrchestratorStatus::Running);
        }
        assert_eq!(mock_parser.get_processed_bytes_log(), vec![b"P1".to_vec()]);
        assert_eq!(mock_term.get_inputs_received(), vec![EmulatorInput::Ansi(AnsiCommand::Print('P')), EmulatorInput::Ansi(AnsiCommand::Print('1'))]);
        assert_eq!(mock_pty.get_written_data(), b"AckP1");

        mock_parser.clear_processed_bytes_log();
        mock_term.clear_inputs_received();
        mock_pty.clear_written_data();
        mock_driver.clear_calls();

        // --- Phase 2: Driver key event ---
        let key_event = BackendEvent::Key { keysym: 'K' as u32, text: "K".to_string() };
        mock_driver.expect_events_for_next_call(vec![key_event.clone()]);
        mock_term.expect_action_for_next_call(Some(EmulatorAction::WritePty(b"AckK".to_vec())));
        {
            // Re-initialize renderer if it was moved, or clone
            renderer = Renderer::new(); // Or renderer.clone() if you modify it
            let mut orchestrator = create_orchestrator(
                 &mut mock_pty, &mut mock_term, &mut mock_parser, renderer.clone(), &mut mock_driver
            );
            assert_eq!(orchestrator.process_driver_events().unwrap(), OrchestratorStatus::Running);
        }
        assert_eq!(mock_driver.get_calls(), vec![MockDriverCall::ProcessEvents]);
        assert_eq!(mock_term.get_inputs_received(), vec![EmulatorInput::User(key_event)]);
        assert_eq!(mock_pty.get_written_data(), b"AckK");

        mock_driver.clear_calls();
        mock_term.clear_inputs_received();
        mock_pty.clear_written_data();

        // --- Phase 3: More PTY data ---
        mock_pty.push_bytes_to_read(b"P2");
        mock_parser.expect_commands_for_next_call(vec![AnsiCommand::Print('P'), AnsiCommand::Print('2')]);
        mock_term.expect_action_for_next_call(None);
        mock_term.expect_action_for_next_call(Some(EmulatorAction::SetTitle("TitleP2".to_string())));
        {
            renderer = Renderer::new(); // Or renderer.clone()
            let mut orchestrator = create_orchestrator(
                &mut mock_pty, &mut mock_term, &mut mock_parser, renderer.clone(), &mut mock_driver
            );
            assert_eq!(orchestrator.process_pty_events().unwrap(), OrchestratorStatus::Running);
        }
        assert_eq!(mock_parser.get_processed_bytes_log(), vec![b"P2".to_vec()]);
        assert_eq!(mock_term.get_inputs_received(), vec![EmulatorInput::Ansi(AnsiCommand::Print('P')), EmulatorInput::Ansi(AnsiCommand::Print('2'))]);
        assert!(mock_pty.get_written_data().is_empty());
        assert_eq!(mock_driver.get_calls(), vec![MockDriverCall::SetTitle{title: "TitleP2".to_string()}]);
    }
}

