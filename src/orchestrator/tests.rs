// src/orchestrator/tests.rs

use crate::ansi::{AnsiCommand, AnsiParser};
use crate::color::Color;
use crate::glyph::{AttrFlags, Attributes, Glyph};
use crate::keys::{KeySymbol, Modifiers}; // Added for driver event tests
use crate::orchestrator::{AppOrchestrator, OrchestratorStatus};
use crate::platform::backends::{
    BackendEvent, CellCoords, CellRect, CursorVisibility, Driver, FocusState, MouseButton, // Added MouseButton
    PlatformState, RenderCommand, TextRunStyle,
};
use crate::platform::os::pty::PtyChannel;
use crate::renderer::{Renderer, RENDERER_DEFAULT_BG, RENDERER_DEFAULT_FG};
use crate::term::{
    ControlEvent, // Added for resize event
    CursorRenderState, CursorShape, EmulatorAction, EmulatorInput, RenderSnapshot, Selection,
    SnapshotLine, TerminalInterface, UserInputAction, // Added UserInputAction
};

use anyhow::Result;
use nix::unistd::Pid;
use std::cell::RefCell;
use std::io::{Error as IoError, ErrorKind as IoErrorKind, Read, Write}; // Added for mocks
use std::os::unix::io::RawFd; // Added for mocks
use std::sync::Mutex;

// --- Mock PtyChannel ---
#[derive(Debug)]
pub struct MockPtyChannel {
    read_data: RefCell<Vec<u8>>,
    write_buffer: RefCell<Vec<u8>>,
    resize_calls: RefCell<Vec<(u16, u16)>>,
    child_pid_val: Pid,
    read_error: RefCell<Option<IoError>>,
    raw_fd: RawFd,
    eof: RefCell<bool>, // Added for EOF simulation
}

impl MockPtyChannel {
    pub fn new() -> Self {
        Self {
            read_data: RefCell::new(Vec::new()),
            write_buffer: RefCell::new(Vec::new()),
            resize_calls: RefCell::new(Vec::new()),
            child_pid_val: Pid::from_raw(1000), // Default mock PID
            read_error: RefCell::new(None),
            raw_fd: -1, // Default mock FD
            eof: RefCell::new(false),
        }
    }

    pub fn expect_read_data(&self, data: Vec<u8>) {
        *self.eof.borrow_mut() = false; // Reading data means not EOF
        self.read_data.borrow_mut().extend(data);
    }

    pub fn expect_read_error(&self, error: IoError) {
        *self.eof.borrow_mut() = false; // Error also means not EOF for this call
        *self.read_error.borrow_mut() = Some(error);
    }

    pub fn expect_eof(&self) {
        *self.eof.borrow_mut() = true;
        self.read_data.borrow_mut().clear(); // EOF means no more data
        *self.read_error.borrow_mut() = None; // No error if EOF
    }

    pub fn get_written_data(&self) -> Vec<u8> {
        self.write_buffer.borrow().clone()
    }

    pub fn get_resize_calls(&self) -> Vec<(u16, u16)> {
        self.resize_calls.borrow().clone()
    }
}

impl Read for MockPtyChannel {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if let Some(err) = self.read_error.borrow_mut().take() {
            return Err(err);
        }
        if *self.eof.borrow() && self.read_data.borrow().is_empty() {
            *self.eof.borrow_mut() = false; // Consume EOF state
            return Ok(0); // Simulate EOF
        }
        let mut read_data_ref = self.read_data.borrow_mut();
        if read_data_ref.is_empty() {
            return Err(IoError::new(IoErrorKind::WouldBlock, "MockPty WouldBlock"));
        }
        let len = std::cmp::min(buf.len(), read_data_ref.len());
        buf[..len].copy_from_slice(&read_data_ref[..len]);
        read_data_ref.drain(..len);
        Ok(len)
    }
}

impl Write for MockPtyChannel {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.write_buffer.borrow_mut().extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(()) // No-op
    }
}

impl PtyChannel for MockPtyChannel {
    fn as_raw_fd(&self) -> RawFd {
        self.raw_fd
    }

    fn resize(&mut self, cols: u16, rows: u16) -> anyhow::Result<()> {
        self.resize_calls.borrow_mut().push((cols, rows));
        Ok(())
    }

    fn child_pid(&self) -> Pid {
        self.child_pid_val
    }
}

// --- Mock TerminalInterface ---
#[derive(Default)]
pub struct MockTerminalInterface {
    inputs_processed: RefCell<Vec<EmulatorInput>>,
    actions_to_return: RefCell<Vec<Option<EmulatorAction>>>,
    snapshot_to_return: RefCell<Option<RenderSnapshot>>,
}

impl MockTerminalInterface {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn expect_action(&self, action: Option<EmulatorAction>) {
        self.actions_to_return.borrow_mut().push(action);
    }

    pub fn expect_snapshot(&self, snapshot: RenderSnapshot) {
        *self.snapshot_to_return.borrow_mut() = Some(snapshot);
    }

    pub fn get_inputs_processed(&self) -> Vec<EmulatorInput> {
        self.inputs_processed.borrow().clone()
    }

    pub fn clear_inputs_processed(&self) { // Added for multi-stage tests
        self.inputs_processed.borrow_mut().clear();
    }
}

impl TerminalInterface for MockTerminalInterface {
    fn interpret_input(&mut self, input: EmulatorInput) -> Option<EmulatorAction> {
        self.inputs_processed.borrow_mut().push(input);
        if self.actions_to_return.borrow().is_empty() {
            None
        } else {
            self.actions_to_return.borrow_mut().remove(0)
        }
    }

    fn get_render_snapshot(&self) -> RenderSnapshot {
        self.snapshot_to_return
            .borrow_mut()
            .take()
            .unwrap_or_else(|| {
                // Return a minimal default snapshot if none was expected
                RenderSnapshot {
                    dimensions: (80, 24),
                    lines: vec![SnapshotLine {
                        is_dirty: true,
                        cells: vec![
                            Glyph {
                                c: ' ',
                                attr: default_attrs()
                            };
                            80
                        ],
                    }; 24],
                    cursor_state: None,
                    selection: Selection::default(),
                }
            })
    }
}

// --- MockAnsiParser ---
#[derive(Default)]
pub struct MockAnsiParser {
    bytes_processed: RefCell<Vec<u8>>,
    commands_to_return: RefCell<Vec<AnsiCommand>>,
}

impl MockAnsiParser {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn expect_commands(&self, commands: Vec<AnsiCommand>) {
        self.commands_to_return.borrow_mut().extend(commands);
    }

    pub fn get_bytes_processed(&self) -> Vec<u8> {
        self.bytes_processed.borrow().clone()
    }
}

impl AnsiParser for MockAnsiParser {
    fn process_bytes(&mut self, bytes: &[u8]) -> Vec<AnsiCommand> {
        self.bytes_processed.borrow_mut().extend_from_slice(bytes);
        self.commands_to_return.borrow_mut().drain(..).collect()
    }
}

// --- Enhanced MockDriver ---
// (Existing MockDriver from tests.rs, to be enhanced)
struct MockDriver {
    commands: Mutex<Vec<RenderCommand>>,
    // New fields for enhanced mocking
    events_to_return: RefCell<Vec<BackendEvent>>,
    platform_state_val: RefCell<PlatformState>, // Changed to RefCell for interior mutability
    set_title_calls: RefCell<Vec<String>>,
    bell_calls: RefCell<usize>,
    set_cursor_visibility_calls: RefCell<Vec<CursorVisibility>>,
    set_focus_calls: RefCell<Vec<FocusState>>,
    own_selection_calls: RefCell<Vec<(u64, String)>>,
    request_selection_data_calls: RefCell<Vec<(u64, u64)>>,
    event_fd_val: Option<RawFd>,
    // Original fields for basic rendering mock, now part of platform_state_val
    // font_width: usize,
    // font_height: usize,
    // display_width_px: u16,
    // display_height_px: u16,
}

impl MockDriver {
    // Updated new method
    fn new(initial_platform_state: PlatformState, event_fd: Option<RawFd>) -> Self {
        Self {
            commands: Mutex::new(Vec::new()),
            events_to_return: RefCell::new(Vec::new()),
            platform_state_val: RefCell::new(initial_platform_state),
            set_title_calls: RefCell::new(Vec::new()),
            bell_calls: RefCell::new(0),
            set_cursor_visibility_calls: RefCell::new(Vec::new()),
            set_focus_calls: RefCell::new(Vec::new()),
            own_selection_calls: RefCell::new(Vec::new()),
            request_selection_data_calls: RefCell::new(Vec::new()),
            event_fd_val: event_fd,
        }
    }

    // Default constructor for simpler test setups if specific platform state isn't critical
    fn default() -> Self {
        Self::new(
            PlatformState {
                event_fd: None, // This will be overridden by event_fd_val if Some
                font_cell_width_px: 8,
                font_cell_height_px: 16,
                display_width_px: 80 * 8,
                display_height_px: 24 * 16,
                scale_factor: 1.0,
            },
            None,
        )
    }


    fn commands(&self) -> Vec<RenderCommand> {
        self.commands.lock().unwrap().clone()
    }

    fn clear_commands(&self) {
        self.commands.lock().unwrap().clear();
    }

    // New methods for controlling mock behavior
    pub fn expect_backend_event(&self, event: BackendEvent) {
        self.events_to_return.borrow_mut().push(event);
    }

    pub fn set_platform_state(&self, state: PlatformState) {
        *self.platform_state_val.borrow_mut() = state;
    }

    pub fn get_platform_state_val(&self) -> PlatformState { // Renamed from get_platform_state to avoid conflict with trait
        self.platform_state_val.borrow().clone()
    }

    pub fn get_set_title_calls(&self) -> Vec<String> {
        self.set_title_calls.borrow().clone()
    }

    pub fn get_bell_calls_count(&self) -> usize {
        *self.bell_calls.borrow()
    }

    pub fn get_set_cursor_visibility_calls(&self) -> Vec<CursorVisibility> {
        self.set_cursor_visibility_calls.borrow().clone()
    }

    pub fn get_set_focus_calls(&self) -> Vec<FocusState> {
        self.set_focus_calls.borrow().clone()
    }

    pub fn get_own_selection_calls(&self) -> Vec<(u64, String)> {
        self.own_selection_calls.borrow().clone()
    }

    pub fn get_request_selection_data_calls(&self) -> Vec<(u64, u64)> {
        self.request_selection_data_calls.borrow().clone()
    }
}

impl Driver for MockDriver {
    fn new() -> anyhow::Result<Self>
    where
        Self: Sized,
    {
        // Provide a default initial state for the Driver::new() case
        Ok(MockDriver::default())
    }

    fn get_event_fd(&self) -> Option<RawFd> {
        self.event_fd_val
    }

    fn process_events(&mut self) -> anyhow::Result<Vec<BackendEvent>> {
        Ok(self.events_to_return.borrow_mut().drain(..).collect())
    }

    fn get_platform_state(&self) -> PlatformState {
        let mut state = self.platform_state_val.borrow().clone();
        state.event_fd = self.event_fd_val; // Ensure event_fd in state matches mock's
        state
    }

    fn execute_render_commands(&mut self, commands: Vec<RenderCommand>) -> Result<()> {
        self.commands.lock().unwrap().extend(commands);
        Ok(())
    }

    // present() was removed from Driver trait, execute_render_commands now includes PresentFrame
    // fn present(&mut self) -> anyhow::Result<()> {
    //     self.commands.lock().unwrap().push(RenderCommand::PresentFrame);
    //     Ok(())
    // }

    fn set_title(&mut self, title: &str) {
        self.set_title_calls.borrow_mut().push(title.to_string());
    }

    fn bell(&mut self) {
        *self.bell_calls.borrow_mut() += 1;
    }

    fn set_cursor_visibility(&mut self, visibility: CursorVisibility) {
        self.set_cursor_visibility_calls.borrow_mut().push(visibility);
    }

    fn set_focus(&mut self, focus_state: FocusState) {
        self.set_focus_calls.borrow_mut().push(focus_state);
    }

    fn cleanup(&mut self) -> anyhow::Result<()> {
        Ok(()) // No-op
    }

    fn own_selection(&mut self, selection_name_atom_u64: u64, text: String) {
        self.own_selection_calls.borrow_mut().push((selection_name_atom_u64, text));
    }

    fn request_selection_data(&mut self, selection_name_atom_u64: u64, target_atom_u64: u64) {
        self.request_selection_data_calls.borrow_mut().push((selection_name_atom_u64, target_atom_u64));
    }
}

// Helper to create a RenderSnapshot for tests (existing, might need minor adjustments)
fn create_test_snapshot(
    lines_data: Vec<SnapshotLine>,
    cursor_state: Option<CursorRenderState>,
    num_cols: usize,
    num_rows: usize,
    selection: Selection,
) -> RenderSnapshot {
    RenderSnapshot {
        dimensions: (num_cols, num_rows),
        lines: lines_data,
        cursor_state,
        selection,
    }
}

// Helper to create default attributes (existing)
fn default_attrs() -> Attributes {
    Attributes {
        fg: RENDERER_DEFAULT_FG,
        bg: RENDERER_DEFAULT_BG,
        flags: AttrFlags::empty(),
    }
}

// Existing tests for Renderer... These should still pass if MockDriver is correctly adapted.
// It seems `present()` was removed from the Driver trait, and PresentFrame is now a RenderCommand.
// The existing tests might need slight adjustment if they call adapter.present() directly.

#[test]
fn test_render_empty_screen_with_cursor() {
    let font_width = 8;
    let font_height = 16;
    let num_cols = 3;
    let num_rows = 2;

    let initial_platform_state = PlatformState {
        event_fd: None,
        font_cell_width_px: font_width,
        font_cell_height_px: font_height,
        display_width_px: (num_cols * font_width) as u16,
        display_height_px: (num_rows * font_height) as u16,
        scale_factor: 1.0,
    };

    let renderer = Renderer::new();
    let mut adapter = MockDriver::new(initial_platform_state, None);

    let default_glyph = Glyph {
        c: ' ',
        attr: default_attrs(),
    };
    let lines = vec![
        SnapshotLine {
            is_dirty: true,
            cells: vec![default_glyph; num_cols]
        };
        num_rows
    ];

    let cursor_render_state = Some(CursorRenderState {
        x: 0,
        y: 0,
        shape: CursorShape::Block,
        cell_char_underneath: ' ',
        cell_attributes_underneath: default_attrs(),
    });

    let snapshot = create_test_snapshot(lines, cursor_render_state, num_cols, num_rows, Selection::default());
    let mut render_commands = renderer
        .draw(snapshot)
        .expect("Render draw failed");
    render_commands.push(RenderCommand::PresentFrame); // Manually add PresentFrame as it's expected by tests
    adapter.execute_render_commands(render_commands).expect("Execute render commands failed");
    // adapter.present().expect("Adapter present failed"); // present() removed from trait

    let commands = adapter.commands();
    assert!(!commands.is_empty(), "Should have drawing commands");

    let expected_bg_fill_for_line0 = RenderCommand::FillRect {
        x: 0,
        y: 0,
        width: num_cols,
        height: 1,
        color: RENDERER_DEFAULT_BG,
        is_selection_bg: false,
    };
    let expected_bg_fill_for_line1 = RenderCommand::FillRect {
        x: 0,
        y: 1,
        width: num_cols,
        height: 1,
        color: RENDERER_DEFAULT_BG,
        is_selection_bg: false,
    };

    assert!(
        commands.contains(&expected_bg_fill_for_line0),
        "Missing background fill for line 0"
    );
    assert!(
        commands.contains(&expected_bg_fill_for_line1),
        "Missing background fill for line 1"
    );

    let cursor_draw_style = TextRunStyle {
        fg: RENDERER_DEFAULT_BG,
        bg: RENDERER_DEFAULT_FG,
        flags: AttrFlags::empty(),
    };
    let expected_cursor_draw = RenderCommand::DrawTextRun {
        x: 0,
        y: 0,
        text: " ".to_string(),
        fg: cursor_draw_style.fg,
        bg: cursor_draw_style.bg,
        flags: cursor_draw_style.flags,
        is_selected: false,
    };

    assert!(
        commands.contains(&expected_cursor_draw),
        "Missing cursor draw command. Commands: {:?}",
        commands
    );
    assert_eq!(
        commands.last().unwrap(),
        &RenderCommand::PresentFrame,
        "Last command should be PresentFrame"
    );
}

// --- Tests for Emulator Action Translation ---

fn setup_orchestrator_for_action_translation() -> (
    AppOrchestrator<'static>, // This lifetime might be tricky if mocks are not 'static
    MockPtyChannel,          // Return mocks to inspect them later if needed, though owned by test function
    MockTerminalInterface,
    MockAnsiParser,
    MockDriver,
) {
    // Due to lifetime issues with passing &mut to AppOrchestrator::new,
    // we'll leak the mocks for the duration of the test. This is common in test setups.
    // Alternatively, use Box::leak(Box::new(mock_component)).
    let mock_pty_channel = Box::leak(Box::new(MockPtyChannel::new()));
    let mock_term = Box::leak(Box::new(MockTerminalInterface::new()));
    let mock_parser = Box::leak(Box::new(MockAnsiParser::new()));
    let mock_driver = Box::leak(Box::new(MockDriver::default()));
    let renderer = Renderer::new();

    let orchestrator = AppOrchestrator::new(
        mock_pty_channel,
        mock_term,
        mock_parser,
        renderer,
        mock_driver,
    );

    // Return owned mocks by cloning or by ensuring they are not moved if not needed outside setup
    // For simplicity here, we assume we might want to inspect them, so we'd ideally return them
    // or make assertions on the leaked static refs.
    // However, the function signature expects owned types. This setup is problematic.
    // Let's simplify: tests will own their mocks and pass mutable refs.
    // The helper is not strictly necessary if each test does its own setup.
    // For now, let's assume each test sets up its own mocks and orchestrator.
    // This helper structure is more complex than needed for these tests.
    // I will do the setup directly in each test.
    unimplemented!("Helper not used, setup in each test directly")
}


#[test]
fn it_should_handle_set_title_action_and_generate_render_command() {
    let mut mock_pty = MockPtyChannel::new();
    let mut mock_term = MockTerminalInterface::new();
    let mut mock_parser = MockAnsiParser::new();
    let mut mock_driver = MockDriver::default();
    let renderer = Renderer::new(); // Actual renderer

    let mut orchestrator = AppOrchestrator::new(
        &mut mock_pty, &mut mock_term, &mut mock_parser, renderer, &mut mock_driver
    );
    // Clear any commands from initialization or previous interactions
    mock_driver.clear_commands();
    mock_term.clear_inputs_processed();


    // Trigger: PTY event leads to EmulatorAction
    mock_pty.expect_read_data(b"trigger_title".to_vec());
    mock_parser.expect_commands(vec![AnsiCommand::Print("trigger_title".to_string())]);
    let title_to_set = "Test Window Title".to_string();
    mock_term.expect_action(Some(EmulatorAction::SetTitle(title_to_set.clone())));

    let status = orchestrator.process_pty_events().unwrap();
    assert_eq!(status, OrchestratorStatus::Running);
    // SetTitle is a render action, so it should be in pending_render_actions now.
    // No commands should be on the driver yet from this action.
    assert!(mock_driver.commands().is_empty());


    // Action: Call render_if_needed to process pending actions
    // Provide a default snapshot for rendering to proceed
    mock_term.expect_snapshot(create_test_snapshot(vec![], None, 80, 24, Selection::default()));
    orchestrator.render_if_needed().unwrap();

    // Assertions
    let rendered_commands = mock_driver.commands();
    let expected_command = RenderCommand::SetWindowTitle { title: title_to_set };
    assert!(
        rendered_commands.contains(&expected_command),
        "Render commands should contain SetWindowTitle. Got: {:?}",
        rendered_commands
    );
}

#[test]
fn it_should_handle_ring_bell_action_and_generate_render_command() {
    let mut mock_pty = MockPtyChannel::new();
    let mut mock_term = MockTerminalInterface::new();
    let mut mock_parser = MockAnsiParser::new();
    let mut mock_driver = MockDriver::default();
    let renderer = Renderer::new();

    let mut orchestrator = AppOrchestrator::new(
        &mut mock_pty, &mut mock_term, &mut mock_parser, renderer, &mut mock_driver
    );
    mock_driver.clear_commands();
    mock_term.clear_inputs_processed();

    mock_pty.expect_read_data(b"trigger_bell".to_vec());
    mock_parser.expect_commands(vec![AnsiCommand::Print("trigger_bell".to_string())]);
    mock_term.expect_action(Some(EmulatorAction::RingBell));

    orchestrator.process_pty_events().unwrap();
    assert!(mock_driver.commands().is_empty()); // Should be pending

    mock_term.expect_snapshot(create_test_snapshot(vec![], None, 80, 24, Selection::default()));
    orchestrator.render_if_needed().unwrap();

    let rendered_commands = mock_driver.commands();
    assert!(rendered_commands.contains(&RenderCommand::RingBell));
}

#[test]
fn it_should_handle_set_cursor_visibility_action_and_generate_render_command() {
    let mut mock_pty = MockPtyChannel::new();
    let mut mock_term = MockTerminalInterface::new();
    let mut mock_parser = MockAnsiParser::new();
    let mut mock_driver = MockDriver::default();
    let renderer = Renderer::new();

    let mut orchestrator = AppOrchestrator::new(
        &mut mock_pty, &mut mock_term, &mut mock_parser, renderer, &mut mock_driver
    );
    mock_driver.clear_commands();
    mock_term.clear_inputs_processed();

    mock_pty.expect_read_data(b"trigger_cursor".to_vec());
    mock_parser.expect_commands(vec![AnsiCommand::Print("trigger_cursor".to_string())]);
    mock_term.expect_action(Some(EmulatorAction::SetCursorVisibility(false)));

    orchestrator.process_pty_events().unwrap();
    assert!(mock_driver.commands().is_empty()); // Should be pending

    mock_term.expect_snapshot(create_test_snapshot(vec![], None, 80, 24, Selection::default()));
    orchestrator.render_if_needed().unwrap();

    let rendered_commands = mock_driver.commands();
    let expected_command = RenderCommand::SetCursorVisibility { visible: false };
    assert!(rendered_commands.contains(&expected_command));
}

#[test]
fn it_should_handle_copy_to_clipboard_action_and_call_driver() {
    let mut mock_pty = MockPtyChannel::new();
    let mut mock_term = MockTerminalInterface::new();
    let mut mock_parser = MockAnsiParser::new();
    let mut mock_driver = MockDriver::default();
    let renderer = Renderer::new();

    let mut orchestrator = AppOrchestrator::new(
        &mut mock_pty, &mut mock_term, &mut mock_parser, renderer, &mut mock_driver
    );
    mock_term.clear_inputs_processed();

    let clipboard_text = "clipboard text".to_string();
    mock_pty.expect_read_data(b"trigger_copy".to_vec());
    mock_parser.expect_commands(vec![AnsiCommand::Print("trigger_copy".to_string())]);
    mock_term.expect_action(Some(EmulatorAction::CopyToClipboard(clipboard_text.clone())));

    orchestrator.process_pty_events().unwrap(); // Action is handled directly

    let own_selection_calls = mock_driver.get_own_selection_calls();
    // Orchestrator calls own_selection for both CLIPBOARD and PRIMARY
    const TRAIT_ATOM_ID_PRIMARY: u64 = 1;
    const TRAIT_ATOM_ID_CLIPBOARD: u64 = 2;

    assert!(
        own_selection_calls.contains(&(TRAIT_ATOM_ID_CLIPBOARD, clipboard_text.clone())),
        "CopyToClipboard did not call driver.own_selection for CLIPBOARD. Calls: {:?}", own_selection_calls
    );
    assert!(
        own_selection_calls.contains(&(TRAIT_ATOM_ID_PRIMARY, clipboard_text.clone())),
        "CopyToClipboard did not call driver.own_selection for PRIMARY. Calls: {:?}", own_selection_calls
    );
    assert!(own_selection_calls.len() >= 2, "Expected at least two calls to own_selection for PRIMARY and CLIPBOARD");
}

#[test]
fn it_should_handle_request_clipboard_content_action_and_call_driver() {
    let mut mock_pty = MockPtyChannel::new();
    let mut mock_term = MockTerminalInterface::new();
    let mut mock_parser = MockAnsiParser::new();
    let mut mock_driver = MockDriver::default();
    let renderer = Renderer::new();

    let mut orchestrator = AppOrchestrator::new(
        &mut mock_pty, &mut mock_term, &mut mock_parser, renderer, &mut mock_driver
    );
    mock_term.clear_inputs_processed();

    mock_pty.expect_read_data(b"trigger_paste_request".to_vec());
    mock_parser.expect_commands(vec![AnsiCommand::Print("trigger_paste_request".to_string())]);
    mock_term.expect_action(Some(EmulatorAction::RequestClipboardContent));

    orchestrator.process_pty_events().unwrap(); // Action is handled directly

    let request_calls = mock_driver.get_request_selection_data_calls();
    const TRAIT_ATOM_ID_CLIPBOARD: u64 = 2;
    const TRAIT_ATOM_ID_UTF8_STRING: u64 = 10;

    assert_eq!(request_calls.len(), 1, "Expected one call to request_selection_data");
    assert_eq!(
        request_calls[0],
        (TRAIT_ATOM_ID_CLIPBOARD, TRAIT_ATOM_ID_UTF8_STRING),
        "RequestClipboardContent did not call driver.request_selection_data with correct atoms. Calls: {:?}", request_calls
    );
}
#[test]
fn test_render_simple_text() {
    let font_width = 8;
    let font_height = 16;
    let num_cols = 5;
    let num_rows = 1;

    let initial_platform_state = PlatformState {
        event_fd: None,
        font_cell_width_px: font_width,
        font_cell_height_px: font_height,
        display_width_px: (num_cols * font_width) as u16,
        display_height_px: (num_rows * font_height) as u16,
        scale_factor: 1.0,
    };

    let renderer = Renderer::new();
    let mut adapter = MockDriver::new(initial_platform_state, None);

    let mut line_cells = vec![
        Glyph {
            c: ' ',
            attr: default_attrs()
        };
        num_cols
    ];
    line_cells[0] = Glyph {
        c: 'H',
        attr: default_attrs(),
    };
    line_cells[1] = Glyph {
        c: 'i',
        attr: default_attrs(),
    };

    let lines = vec![SnapshotLine {
        is_dirty: true,
        cells: line_cells,
    }];

    let cursor_render_state = Some(CursorRenderState {
        x: 2,
        y: 0,
        shape: CursorShape::Block,
        cell_char_underneath: ' ',
        cell_attributes_underneath: default_attrs(),
    });

    let snapshot = create_test_snapshot(lines, cursor_render_state, num_cols, num_rows, Selection::default());
    let mut render_commands = renderer
        .draw(snapshot)
        .expect("Render draw failed");
    render_commands.push(RenderCommand::PresentFrame); // Manually add PresentFrame
    adapter.execute_render_commands(render_commands).expect("Execute render commands failed");
    // adapter.present().expect("Adapter present failed");

    let commands = adapter.commands();

    let text_style = TextRunStyle {
        fg: RENDERER_DEFAULT_FG,
        bg: RENDERER_DEFAULT_BG,
        flags: AttrFlags::empty(),
    };

    let expected_text_hi = RenderCommand::DrawTextRun {
        x: 0,
        y: 0,
        text: "Hi".to_string(),
        fg: text_style.fg,
        bg: text_style.bg,
        flags: text_style.flags,
        is_selected: false,
    };
    assert!(
        commands.contains(&expected_text_hi),
        "Missing text 'Hi'. Commands: {:?}",
        commands
    );

    let expected_fill_spaces = RenderCommand::FillRect {
        x: 2,
        y: 0,
        width: num_cols - 2,
        height: 1,
        color: RENDERER_DEFAULT_BG,
        is_selection_bg: false,
    };
    assert!(
        commands.contains(&expected_fill_spaces),
        "Missing fill for remaining spaces. Commands: {:?}",
        commands
    );

    let cursor_draw_style = TextRunStyle {
        fg: RENDERER_DEFAULT_BG,
        bg: RENDERER_DEFAULT_FG,
        flags: AttrFlags::empty(),
    };
    let expected_cursor_draw = RenderCommand::DrawTextRun {
        x: 2,
        y: 0,
        text: " ".to_string(),
        fg: cursor_draw_style.fg,
        bg: cursor_draw_style.bg,
        flags: cursor_draw_style.flags,
        is_selected: false,
    };
    assert!(
        commands.contains(&expected_cursor_draw),
        "Missing cursor draw. Commands: {:?}",
        commands
    );
    assert_eq!(commands.last().unwrap(), &RenderCommand::PresentFrame);
}

#[test]
fn test_render_dirty_line_only() {
    let font_width = 8;
    let font_height = 16;
    let num_cols = 3;
    let num_rows = 2;

    let initial_platform_state = PlatformState {
        event_fd: None,
        font_cell_width_px: font_width,
        font_cell_height_px: font_height,
        display_width_px: (num_cols * font_width) as u16,
        display_height_px: (num_rows * font_height) as u16,
        scale_factor: 1.0,
    };

    let renderer = Renderer::new();
    let mut adapter = MockDriver::new(initial_platform_state.clone(), None); // Clone state for reuse

    let mut line0_cells = vec![
        Glyph {
            c: ' ',
            attr: default_attrs()
        };
        num_cols
    ];
    line0_cells[0] = Glyph {
        c: 'A',
        attr: default_attrs(),
    };

    let line1_cells = vec![
        Glyph {
            c: 'B',
            attr: default_attrs()
        };
        num_cols
    ];

    let lines_frame1 = vec![
        SnapshotLine {
            is_dirty: true,
            cells: line0_cells.clone(),
        },
        SnapshotLine {
            is_dirty: true,
            cells: line1_cells.clone(),
        },
    ];
    let cursor_state_frame1 = Some(CursorRenderState {
        x: 1,
        y: 0,
        shape: CursorShape::Block,
        cell_char_underneath: ' ',
        cell_attributes_underneath: default_attrs(),
    });
    let snapshot1 = create_test_snapshot(
        lines_frame1,
        cursor_state_frame1.clone(),
        num_cols,
        num_rows,
        Selection::default(),
    );
    let render_commands1 = renderer
        .draw(snapshot1)
        .expect("Render draw failed");
    adapter.execute_render_commands(render_commands1).expect("Execute render commands failed");
    adapter.clear_commands(); // Clear commands after first frame

    // Re-initialize adapter or ensure its state is suitable for second render call
    // For this test, reusing adapter and just clearing commands is fine.
    // If MockDriver had more complex internal state affected by rendering, might need new instance.

    let mut line0_cells_frame2 = vec![
        Glyph {
            c: ' ',
            attr: default_attrs()
        };
        num_cols
    ];
    line0_cells_frame2[0] = Glyph {
        c: 'A',
        attr: default_attrs(),
    };

    let mut line1_cells_frame2 = vec![
        Glyph {
            c: ' ',
            attr: default_attrs()
        };
        num_cols
    ];
    line1_cells_frame2[0] = Glyph {
        c: 'C',
        attr: default_attrs(),
    };

    let lines_frame2 = vec![
        SnapshotLine {
            is_dirty: false, // Line 0 is NOT dirty
            cells: line0_cells_frame2,
        },
        SnapshotLine {
            is_dirty: true, // Line 1 IS dirty
            cells: line1_cells_frame2,
        },
    ];
    let cursor_state_frame2 = Some(CursorRenderState {
        x: 0,
        y: 1,
        shape: CursorShape::Block,
        cell_char_underneath: 'C', // Underneath the cursor on line 1
        cell_attributes_underneath: default_attrs(),
    });

    let snapshot2 =
        create_test_snapshot(lines_frame2, cursor_state_frame2, num_cols, num_rows, Selection::default());
    let mut render_commands2 = renderer
        .draw(snapshot2)
        .expect("Render draw failed");
    render_commands2.push(RenderCommand::PresentFrame); // Manually add PresentFrame
    adapter.execute_render_commands(render_commands2).expect("Execute render commands failed");
    // adapter.present().expect("Adapter present failed");
    let commands_frame2 = adapter.commands();

    let text_style = TextRunStyle {
        fg: RENDERER_DEFAULT_FG,
        bg: RENDERER_DEFAULT_BG,
        flags: AttrFlags::empty(),
    };

    // Check that 'A' from line 0 (not dirty) was NOT redrawn
    let draw_a_cmd = RenderCommand::DrawTextRun {
        x: 0,
        y: 0,
        text: "A".to_string(),
        fg: text_style.fg,
        bg: text_style.bg,
        flags: text_style.flags,
        is_selected: false,
    };
    assert!(
        !commands_frame2.contains(&draw_a_cmd),
        "Text 'A' from non-dirty line 0 should not be redrawn. Commands: {:?}",
        commands_frame2
    );

    // Check that the background fill for line 0 (not dirty) was NOT redrawn
    let fill_line0_bg = RenderCommand::FillRect {
        x:0, y:0, width: num_cols, height: 1, color: RENDERER_DEFAULT_BG, is_selection_bg: false
    };
    assert!(
        !commands_frame2.contains(&fill_line0_bg),
        "Background for non-dirty line 0 should not be redrawn. Commands: {:?}",
        commands_frame2
    );


    // Check that 'C' from line 1 (dirty) WAS redrawn
    let draw_c_cmd = RenderCommand::DrawTextRun {
        x: 0,
        y: 1,
        text: "C".to_string(), // This is the char at (0,1)
        fg: text_style.fg,
        bg: text_style.bg,
        flags: text_style.flags,
        is_selected: false,
    };
     assert!(
        commands_frame2.iter().any(|cmd| matches!(cmd, RenderCommand::DrawTextRun { y, text, .. } if *y == 1 && text == "C")),
        "Text 'C' from dirty line 1 should be redrawn. Commands: {:?}",
        commands_frame2
    );


    // Check that the cursor on 'C' at (0,1) was drawn
    let cursor_draw_style = TextRunStyle {
        fg: RENDERER_DEFAULT_BG, // Swapped FG/BG for cursor
        bg: RENDERER_DEFAULT_FG,
        flags: AttrFlags::empty(),
    };
    let expected_cursor_draw = RenderCommand::DrawTextRun {
        x: 0, // Cursor x position
        y: 1, // Cursor y position
        text: "C".to_string(), // Character under cursor
        fg: cursor_draw_style.fg,
        bg: cursor_draw_style.bg,
        flags: cursor_draw_style.flags,
        is_selected: false,
    };
    assert!(
        commands_frame2.contains(&expected_cursor_draw),
        "Missing cursor draw on line 1. Commands: {:?}",
        commands_frame2
    );

    assert_eq!(commands_frame2.last().unwrap(), &RenderCommand::PresentFrame);
}

// TODO: Add tests for AppOrchestrator using these mocks.
// Example test structure:
// #[test]
// fn test_orchestrator_pty_to_terminal_flow() {
//     let mut mock_pty = MockPtyChannel::new();
//     let mut mock_term = MockTerminalInterface::new();
//     let mut mock_parser = MockAnsiParser::new();
//     let mut mock_driver = MockDriver::default(); // Or new with specific state
//     let renderer = Renderer::new();
//
//     let mut orchestrator = AppOrchestrator::new(
//         &mut mock_pty,
//         &mut mock_term,
//         &mut mock_parser,
//         renderer,
//         &mut mock_driver,
//     );
//
//     // Setup expectations
//     mock_pty.expect_read_data(b"hello".to_vec());
//     mock_parser.expect_commands(vec![AnsiCommand::Print('h'), AnsiCommand::Print('e'), /* ... */]);
//     mock_term.expect_action(Some(EmulatorAction::RequestRedraw)); // Example
//
//     // Run orchestrator logic
//     let status = orchestrator.process_pty_events().unwrap();
//     assert_eq!(status, OrchestratorStatus::Running);
//     orchestrator.render_if_needed().unwrap();
//
//     // Assertions
//     assert_eq!(mock_parser.get_bytes_processed(), b"hello");
//     assert!(!mock_term.get_inputs_processed().is_empty());
//     // assert!(mock_driver.commands().contains(&RenderCommand::PresentFrame));
// }

#[test]
fn it_should_initialize_and_resize_terminal_and_pty_based_on_driver_state() {
    // 1. Setup Mock Components
    let mut mock_driver = MockDriver::default(); // Start with default
    let platform_state = PlatformState {
        display_width_px: 800,
        display_height_px: 600,
        font_cell_width_px: 8,
        font_cell_height_px: 16,
        event_fd: None, // Not relevant for this test
        scale_factor: 1.0, // Not directly relevant but part of the struct
    };
    mock_driver.set_platform_state(platform_state.clone());

    let mut mock_pty_channel = MockPtyChannel::new();
    let mut mock_term = MockTerminalInterface::new();
    let mut mock_parser = MockAnsiParser::new(); // Not directly used in `new` but required
    let renderer = Renderer::new(); // Actual renderer, its interaction isn't the focus here

    // 2. Expected Dimensions Calculation
    let expected_cols = platform_state.display_width_px as usize / platform_state.font_cell_width_px;
    let expected_rows = platform_state.display_height_px as usize / platform_state.font_cell_height_px;

    // 3. Instantiate AppOrchestrator
    let _orchestrator = AppOrchestrator::new(
        &mut mock_pty_channel,
        &mut mock_term,
        &mut mock_parser,
        renderer,
        &mut mock_driver,
    );

    // 4. Assert Interactions
    // Terminal Resize
    let terminal_inputs = mock_term.get_inputs_processed();
    assert!(
        !terminal_inputs.is_empty(),
        "Terminal should have received inputs."
    );

    use crate::term::ControlEvent; // Import for pattern matching
    let expected_resize_event = EmulatorInput::Control(ControlEvent::Resize {
        cols: expected_cols,
        rows: expected_rows,
    });
    assert!(
        terminal_inputs.contains(&expected_resize_event),
        "Terminal did not receive expected Resize event. Received: {:?}",
        terminal_inputs
    );

    // PTY Resize
    let pty_resize_calls = mock_pty_channel.get_resize_calls();
    assert!(
        !pty_resize_calls.is_empty(),
        "PTY resize should have been called."
    );
    assert!(
        pty_resize_calls.contains(&(expected_cols as u16, expected_rows as u16)),
        "PTY was not resized to expected dimensions. Received: {:?}",
        pty_resize_calls
    );
}

// --- Tests for PTY Input Processing ---

#[test]
fn it_should_process_simple_text_from_pty_and_update_terminal() {
    let mut mock_pty = MockPtyChannel::new();
    let mut mock_term = MockTerminalInterface::new();
    let mut mock_parser = MockAnsiParser::new();
    let mut mock_driver = MockDriver::default();
    let renderer = Renderer::new();

    let mut orchestrator = AppOrchestrator::new(
        &mut mock_pty,
        &mut mock_term,
        &mut mock_parser,
        renderer,
        &mut mock_driver,
    );

    let pty_input_data = b"hello".to_vec();
    let expected_ansi_command = AnsiCommand::Print("hello".to_string());

    mock_pty.expect_read_data(pty_input_data.clone());
    mock_parser.expect_commands(vec![expected_ansi_command.clone()]);

    let status = orchestrator.process_pty_events().unwrap();
    assert_eq!(status, OrchestratorStatus::Running);

    assert_eq!(mock_parser.get_bytes_processed(), pty_input_data);
    let term_inputs = mock_term.get_inputs_processed();
    assert_eq!(term_inputs.len(), 1);
    assert_eq!(term_inputs[0], EmulatorInput::Ansi(expected_ansi_command));
}

#[test]
fn it_should_process_ansi_escape_codes_from_pty_and_update_terminal() {
    let mut mock_pty = MockPtyChannel::new();
    let mut mock_term = MockTerminalInterface::new();
    let mut mock_parser = MockAnsiParser::new();
    let mut mock_driver = MockDriver::default();
    let renderer = Renderer::new();

    let mut orchestrator = AppOrchestrator::new(
        &mut mock_pty,
        &mut mock_term,
        &mut mock_parser,
        renderer,
        &mut mock_driver,
    );

    let pty_input_data = b"\x1b[A".to_vec(); // CSI A - Cursor Up
    let expected_ansi_command = AnsiCommand::CursorUp(1);

    mock_pty.expect_read_data(pty_input_data.clone());
    mock_parser.expect_commands(vec![expected_ansi_command.clone()]);

    let status = orchestrator.process_pty_events().unwrap();
    assert_eq!(status, OrchestratorStatus::Running);

    assert_eq!(mock_parser.get_bytes_processed(), pty_input_data);
    let term_inputs = mock_term.get_inputs_processed();
    assert_eq!(term_inputs.len(), 1);
    assert_eq!(term_inputs[0], EmulatorInput::Ansi(expected_ansi_command));
}

#[test]
fn it_should_handle_pty_eof_and_signal_shutdown() {
    let mut mock_pty = MockPtyChannel::new();
    let mut mock_term = MockTerminalInterface::new(); // Not directly used but required
    let mut mock_parser = MockAnsiParser::new();    // Not directly used but required
    let mut mock_driver = MockDriver::default();
    let renderer = Renderer::new();

    let mut orchestrator = AppOrchestrator::new(
        &mut mock_pty,
        &mut mock_term,
        &mut mock_parser,
        renderer,
        &mut mock_driver,
    );

    mock_pty.expect_eof(); // Configure mock PTY to return Ok(0) on next read

    let status = orchestrator.process_pty_events().unwrap();
    assert_eq!(status, OrchestratorStatus::Shutdown);
}

#[test]
fn it_should_write_emulator_responses_back_to_pty() {
    let mut mock_pty = MockPtyChannel::new();
    let mut mock_term = MockTerminalInterface::new();
    let mut mock_parser = MockAnsiParser::new();
    let mut mock_driver = MockDriver::default();
    let renderer = Renderer::new();

    let mut orchestrator = AppOrchestrator::new(
        &mut mock_pty,
        &mut mock_term,
        &mut mock_parser,
        renderer,
        &mut mock_driver,
    );

    let pty_input_data = b"input".to_vec();
    let parsed_command = AnsiCommand::Print("input".to_string());
    let terminal_action_input = EmulatorInput::Ansi(parsed_command.clone());
    let pty_response_data = b"response".to_vec();

    mock_pty.expect_read_data(pty_input_data.clone());
    mock_parser.expect_commands(vec![parsed_command]);
    // Expect that when the terminal processes the input, it will request a PTY write
    mock_term.expect_action(Some(EmulatorAction::WritePty(pty_response_data.clone())));

    let status = orchestrator.process_pty_events().unwrap();
    assert_eq!(status, OrchestratorStatus::Running);

    // Check that the PTY received the data to write
    assert_eq!(mock_pty.get_written_data(), pty_response_data);
    // Also verify terminal received the input
    assert!(mock_term.get_inputs_processed().contains(&terminal_action_input));
}

// --- Tests for Driver Event Processing ---

#[test]
fn it_should_process_key_input_event_and_send_to_terminal() {
    let mut mock_pty = MockPtyChannel::new();
    let mut mock_term = MockTerminalInterface::new();
    let mut mock_parser = MockAnsiParser::new();
    let mut mock_driver = MockDriver::default();
    let renderer = Renderer::new();

    let mut orchestrator = AppOrchestrator::new(
        &mut mock_pty,
        &mut mock_term,
        &mut mock_parser,
        renderer,
        &mut mock_driver,
    );

    let key_event = BackendEvent::Key {
        symbol: KeySymbol::Char('a'),
        modifiers: Modifiers::NONE,
        text: "a".to_string(),
    };
    mock_driver.expect_backend_event(key_event);

    let status = orchestrator.process_driver_events().unwrap();
    assert_eq!(status, OrchestratorStatus::Running);

    let expected_term_input = EmulatorInput::User(UserInputAction::KeyInput {
        symbol: KeySymbol::Char('a'),
        modifiers: Modifiers::NONE,
        text: Some("a".to_string()),
    });
    assert!(mock_term.get_inputs_processed().contains(&expected_term_input));
}

#[test]
fn it_should_process_resize_event_and_update_terminal_and_pty() {
    let initial_font_width = 8;
    let initial_font_height = 16;
    let initial_platform_state = PlatformState {
        display_width_px: 80 * initial_font_width, // 640
        display_height_px: 24 * initial_font_height, // 384
        font_cell_width_px: initial_font_width,
        font_cell_height_px: initial_font_height,
        event_fd: None,
        scale_factor: 1.0,
    };

    let mut mock_pty = MockPtyChannel::new();
    let mut mock_term = MockTerminalInterface::new();
    let mut mock_parser = MockAnsiParser::new();
    let mut mock_driver = MockDriver::new(initial_platform_state.clone(), None);
    let renderer = Renderer::new();

    // Orchestrator is initialized with initial_platform_state via mock_driver
    let mut orchestrator = AppOrchestrator::new(
        &mut mock_pty,
        &mut mock_term,
        &mut mock_parser,
        renderer,
        &mut mock_driver,
    );

    // Clear initial resize event from terminal inputs due to AppOrchestrator::new()
    mock_term.clear_inputs_processed();
    mock_pty.resize_calls.borrow_mut().clear();


    let new_display_width_px = 640; // Different from initial to test change
    let new_display_height_px = 400; // e.g. 80x25 cells
    let new_font_cell_width_px = 8;
    let new_font_cell_height_px = 16;

    let resize_event = BackendEvent::Resize {
        width_px: new_display_width_px,
        height_px: new_display_height_px,
    };
    mock_driver.expect_backend_event(resize_event);

    // IMPORTANT: Update the driver's platform state that will be queried *during* event processing
    let new_platform_state = PlatformState {
        display_width_px: new_display_width_px,
        display_height_px: new_display_height_px,
        font_cell_width_px: new_font_cell_width_px,
        font_cell_height_px: new_font_cell_height_px,
        event_fd: None,
        scale_factor: 1.0,
    };
    mock_driver.set_platform_state(new_platform_state.clone());


    let status = orchestrator.process_driver_events().unwrap();
    assert_eq!(status, OrchestratorStatus::Running);

    let expected_cols = new_display_width_px as usize / new_font_cell_width_px;
    let expected_rows = new_display_height_px as usize / new_font_cell_height_px;

    let expected_term_resize = EmulatorInput::Control(ControlEvent::Resize {
        cols: expected_cols,
        rows: expected_rows,
    });
    assert!(mock_term.get_inputs_processed().contains(&expected_term_resize), "Terminal inputs: {:?}", mock_term.get_inputs_processed());

    let pty_resizes = mock_pty.get_resize_calls();
    assert!(pty_resizes.contains(&(expected_cols as u16, expected_rows as u16)), "PTY resizes: {:?}", pty_resizes);
}


#[test]
fn it_should_process_focus_gained_event_and_notify_terminal_and_driver() {
    let mut mock_pty = MockPtyChannel::new();
    let mut mock_term = MockTerminalInterface::new();
    let mut mock_parser = MockAnsiParser::new();
    let mut mock_driver = MockDriver::default();
    let renderer = Renderer::new();

    let mut orchestrator = AppOrchestrator::new(
        &mut mock_pty,
        &mut mock_term,
        &mut mock_parser,
        renderer,
        &mut mock_driver,
    );

    mock_driver.expect_backend_event(BackendEvent::FocusGained);

    let status = orchestrator.process_driver_events().unwrap();
    assert_eq!(status, OrchestratorStatus::Running);

    assert!(mock_driver.get_set_focus_calls().contains(&FocusState::Focused));
    assert!(mock_term.get_inputs_processed().contains(&EmulatorInput::User(UserInputAction::FocusGained)));
}

#[test]
fn it_should_process_focus_lost_event_and_notify_terminal_and_driver() {
    let mut mock_pty = MockPtyChannel::new();
    let mut mock_term = MockTerminalInterface::new();
    let mut mock_parser = MockAnsiParser::new();
    let mut mock_driver = MockDriver::default();
    let renderer = Renderer::new();

    let mut orchestrator = AppOrchestrator::new(
        &mut mock_pty,
        &mut mock_term,
        &mut mock_parser,
        renderer,
        &mut mock_driver,
    );

    mock_driver.expect_backend_event(BackendEvent::FocusLost);

    let status = orchestrator.process_driver_events().unwrap();
    assert_eq!(status, OrchestratorStatus::Running);

    assert!(mock_driver.get_set_focus_calls().contains(&FocusState::Unfocused));
    assert!(mock_term.get_inputs_processed().contains(&EmulatorInput::User(UserInputAction::FocusLost)));
}

#[test]
fn it_should_process_paste_data_event_and_send_to_terminal() {
    let mut mock_pty = MockPtyChannel::new();
    let mut mock_term = MockTerminalInterface::new();
    let mut mock_parser = MockAnsiParser::new();
    let mut mock_driver = MockDriver::default();
    let renderer = Renderer::new();

    let mut orchestrator = AppOrchestrator::new(
        &mut mock_pty,
        &mut mock_term,
        &mut mock_parser,
        renderer,
        &mut mock_driver,
    );

    let paste_text = "pasted text".to_string();
    mock_driver.expect_backend_event(BackendEvent::PasteData { text: paste_text.clone() });

    let status = orchestrator.process_driver_events().unwrap();
    assert_eq!(status, OrchestratorStatus::Running);

    let expected_term_input = EmulatorInput::User(UserInputAction::PasteText(paste_text));
    assert!(mock_term.get_inputs_processed().contains(&expected_term_input));
}

#[test]
fn it_should_process_mouse_button_press_and_send_to_terminal() {
    let mut mock_pty = MockPtyChannel::new();
    let mut mock_term = MockTerminalInterface::new();
    let mut mock_parser = MockAnsiParser::new();
    // MockDriver uses 8x16 font cells by default
    let mut mock_driver = MockDriver::default();
    let renderer = Renderer::new();

    let mut orchestrator = AppOrchestrator::new(
        &mut mock_pty,
        &mut mock_term,
        &mut mock_parser,
        renderer,
        &mut mock_driver,
    );
    mock_term.clear_inputs_processed(); // Clear initial resize

    // Sub-Test 1: Left Button
    mock_driver.expect_backend_event(BackendEvent::MouseButtonPress {
        button: MouseButton::Left,
        x: 16, // cell x = 16 / 8 = 2
        y: 32, // cell y = 32 / 16 = 2
        modifiers: Modifiers::NONE
    });
    let status = orchestrator.process_driver_events().unwrap();
    assert_eq!(status, OrchestratorStatus::Running);
    let expected_left_click_input = EmulatorInput::User(UserInputAction::StartSelection { x: 2, y: 2 });
    assert!(mock_term.get_inputs_processed().contains(&expected_left_click_input), "Inputs: {:?}", mock_term.get_inputs_processed());

    mock_term.clear_inputs_processed(); // Clear for next sub-test

    // Sub-Test 2: Middle Button
    mock_driver.expect_backend_event(BackendEvent::MouseButtonPress {
        button: MouseButton::Middle,
        x: 24, // cell x = 24 / 8 = 3
        y: 48, // cell y = 48 / 16 = 3
        modifiers: Modifiers::NONE
    });
    let status = orchestrator.process_driver_events().unwrap();
    assert_eq!(status, OrchestratorStatus::Running);
    let expected_middle_click_input = EmulatorInput::User(UserInputAction::RequestPrimaryPaste);
    assert!(mock_term.get_inputs_processed().contains(&expected_middle_click_input), "Inputs: {:?}", mock_term.get_inputs_processed());
}

#[test]
fn it_should_process_mouse_button_release_and_send_to_terminal() {
    let mut mock_pty = MockPtyChannel::new();
    let mut mock_term = MockTerminalInterface::new();
    let mut mock_parser = MockAnsiParser::new();
    let mut mock_driver = MockDriver::default();
    let renderer = Renderer::new();

    let mut orchestrator = AppOrchestrator::new(
        &mut mock_pty,
        &mut mock_term,
        &mut mock_parser,
        renderer,
        &mut mock_driver,
    );
    mock_term.clear_inputs_processed();

    mock_driver.expect_backend_event(BackendEvent::MouseButtonRelease {
        button: MouseButton::Left,
        x: 16, // cell x = 2
        y: 32, // cell y = 2
        modifiers: Modifiers::NONE
    });
    let status = orchestrator.process_driver_events().unwrap();
    assert_eq!(status, OrchestratorStatus::Running);
    let expected_input = EmulatorInput::User(UserInputAction::ApplySelectionClear);
    assert!(mock_term.get_inputs_processed().contains(&expected_input));
}

#[test]
fn it_should_process_mouse_move_event_and_send_to_terminal() {
    let mut mock_pty = MockPtyChannel::new();
    let mut mock_term = MockTerminalInterface::new();
    let mut mock_parser = MockAnsiParser::new();
    let mut mock_driver = MockDriver::default(); // Font 8x16
    let renderer = Renderer::new();

    let mut orchestrator = AppOrchestrator::new(
        &mut mock_pty,
        &mut mock_term,
        &mut mock_parser,
        renderer,
        &mut mock_driver,
    );
    mock_term.clear_inputs_processed();

    mock_driver.expect_backend_event(BackendEvent::MouseMove {
        x: 40, // cell x = 40 / 8 = 5
        y: 64, // cell y = 64 / 16 = 4
        modifiers: Modifiers::NONE
    });
    let status = orchestrator.process_driver_events().unwrap();
    assert_eq!(status, OrchestratorStatus::Running);
    let expected_input = EmulatorInput::User(UserInputAction::ExtendSelection { x: 5, y: 4 });
    assert!(mock_term.get_inputs_processed().contains(&expected_input));
}

#[test]
fn it_should_handle_driver_close_request_and_signal_shutdown() {
    let mut mock_pty = MockPtyChannel::new();
    let mut mock_term = MockTerminalInterface::new();
    let mut mock_parser = MockAnsiParser::new();
    let mut mock_driver = MockDriver::default();
    let renderer = Renderer::new();

    let mut orchestrator = AppOrchestrator::new(
        &mut mock_pty,
        &mut mock_term,
        &mut mock_parser,
        renderer,
        &mut mock_driver,
    );

    mock_driver.expect_backend_event(BackendEvent::CloseRequested);
    let result = orchestrator.process_driver_events().unwrap();
    assert_eq!(result, OrchestratorStatus::Shutdown);
}

// --- Tests for Rendering Logic ---

#[test]
fn it_should_collect_render_commands_from_terminal_snapshot_and_pending_actions() {
    let mut mock_pty = MockPtyChannel::new();
    let mut mock_term = MockTerminalInterface::new();
    let mut mock_parser = MockAnsiParser::new();
    let mut mock_driver = MockDriver::default(); // Uses 8x16 font cells, 80x24 grid
    let renderer = Renderer::new(); // Actual renderer

    let mut orchestrator = AppOrchestrator::new(
        &mut mock_pty,
        &mut mock_term,
        &mut mock_parser,
        renderer, // renderer is moved into orchestrator here
        &mut mock_driver,
    );

    // Clear setup-related calls from AppOrchestrator::new()
    mock_term.clear_inputs_processed();
    mock_driver.clear_commands();

    // 1. Setup: Populate pending_render_actions via PTY event processing
    let title_text = "Test Title".to_string();
    // OSC sequence for setting title: \x1b]2;Test Title\x07
    let pty_input_for_title = format!("\x1b]2;{}\x07", title_text).into_bytes();
    let parsed_osc_command = AnsiCommand::OscDeviceControl(format!("2;{}", title_text));
    let terminal_action_for_title = EmulatorAction::SetTitle(title_text.clone());

    mock_pty.expect_read_data(pty_input_for_title.clone());
    mock_parser.expect_commands(vec![parsed_osc_command.clone()]);
    mock_term.expect_action(Some(terminal_action_for_title)); // This action will be put into pending_render_actions

    let status = orchestrator.process_pty_events().unwrap();
    assert_eq!(status, OrchestratorStatus::Running);

    // At this point, SetTitle should be in orchestrator.pending_render_actions
    // Clear any commands that might have been generated if SetTitle was handled immediately (it shouldn't for render commands)
    mock_driver.clear_commands();


    // 2. Setup: Terminal Snapshot
    let num_cols = 1;
    let num_rows = 1;
    let snapshot_lines = vec![SnapshotLine {
        is_dirty: true,
        cells: vec![Glyph {
            c: 'A',
            attr: default_attrs(),
        }],
    }];
    let snapshot = create_test_snapshot(snapshot_lines, None, num_cols, num_rows, Selection::default());
    mock_term.expect_snapshot(snapshot); // Configure mock_term to return this snapshot

    // 3. Action: Call render_if_needed
    orchestrator.render_if_needed().unwrap();

    // 4. Assertions
    let rendered_commands = mock_driver.commands();
    assert!(!rendered_commands.is_empty(), "No render commands were generated.");

    // Check for command from pending action (SetTitle)
    let expected_title_command = RenderCommand::SetWindowTitle { title: title_text };
    assert!(
        rendered_commands.contains(&expected_title_command),
        "Missing SetWindowTitle command. Rendered: {:?}",
        rendered_commands
    );

    // Check for commands from snapshot rendering (e.g., FillRect for background, DrawTextRun for 'A')
    // Specifics depend on Renderer implementation, but we expect some.
    // For a 1x1 cell with 'A':
    // - A FillRect for the background of the line/cell.
    // - A DrawTextRun for the character 'A'.
    let has_fill_rect = rendered_commands.iter().any(|cmd| matches!(cmd, RenderCommand::FillRect { .. }));
    let has_draw_text_run_A = rendered_commands.iter().any(|cmd| matches!(cmd, RenderCommand::DrawTextRun { text, .. } if text == "A"));

    assert!(has_fill_rect, "Missing FillRect command for snapshot. Rendered: {:?}", rendered_commands);
    assert!(has_draw_text_run_A, "Missing DrawTextRun for 'A' from snapshot. Rendered: {:?}", rendered_commands);

    // Check for PresentFrame as the last command
    assert_eq!(
        rendered_commands.last().unwrap(),
        &RenderCommand::PresentFrame,
        "Last command should be PresentFrame. Rendered: {:?}",
        rendered_commands
    );

    // Check order: snapshot commands, then pending action commands (like SetTitle), then PresentFrame.
    // Renderer::draw produces commands from snapshot. These are extended by commands from pending_render_actions.
    let fill_rect_pos = rendered_commands.iter().position(|cmd| matches!(cmd, RenderCommand::FillRect { .. }));
    let draw_text_pos = rendered_commands.iter().position(|cmd| matches!(cmd, RenderCommand::DrawTextRun { text, .. } if text == "A"));
    let title_cmd_pos = rendered_commands.iter().position(|cmd| cmd == &expected_title_command);
    let present_pos = rendered_commands.iter().rposition(|cmd| cmd == &RenderCommand::PresentFrame);

    assert!(fill_rect_pos.is_some(), "FillRect from snapshot missing");
    assert!(draw_text_pos.is_some(), "DrawTextRun 'A' from snapshot missing");
    assert!(title_cmd_pos.is_some(), "SetWindowTitle command missing");
    assert!(present_pos.is_some(), "PresentFrame command missing");

    // Snapshot commands (fill, draw) should come before title command from pending_render_actions
    // And title command should come before PresentFrame.
    // The exact order of fill_rect vs draw_text is renderer-dependent but both are part of snapshot rendering.
    if let (Some(draw_idx), Some(title_idx)) = (draw_text_pos, title_cmd_pos) {
         assert!(draw_idx < title_idx, "Snapshot draw command should appear before pending action command (SetTitle)");
    }
     if let (Some(fill_idx), Some(title_idx)) = (fill_rect_pos, title_cmd_pos) {
         assert!(fill_idx < title_idx, "Snapshot fill command should appear before pending action command (SetTitle)");
    }
    if let (Some(title_idx), Some(p_idx)) = (title_cmd_pos, present_pos) {
        assert!(title_idx < p_idx, "SetTitle command should appear before PresentFrame");
    }
}


#[test]
fn it_should_always_add_present_frame_as_last_render_command() {
    let mut mock_pty = MockPtyChannel::new();
    let mut mock_term = MockTerminalInterface::new();
    let mut mock_parser = MockAnsiParser::new();
    let mut mock_driver = MockDriver::default();
    let renderer = Renderer::new();

    let mut orchestrator = AppOrchestrator::new(
        &mut mock_pty,
        &mut mock_term,
        &mut mock_parser,
        renderer,
        &mut mock_driver,
    );
    mock_driver.clear_commands(); // Clear commands from init

    // Configure a very simple snapshot
    let snapshot = create_test_snapshot(vec![], None, 1, 1, Selection::default()); // Empty 1x1 grid
    mock_term.expect_snapshot(snapshot);

    orchestrator.render_if_needed().unwrap();

    let commands = mock_driver.commands();
    assert!(!commands.is_empty(), "Render commands should not be empty.");
    assert_eq!(
        commands.last().unwrap(),
        &RenderCommand::PresentFrame,
        "The last command must be PresentFrame."
    );
}
