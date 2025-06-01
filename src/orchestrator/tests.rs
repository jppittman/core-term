// src/orchestrator/tests.rs

// Updated imports based on current crate structure
use crate::color::Color;
use crate::platform::backends::{CellCoords, CellRect, CursorVisibility, Driver, FocusState, TextRunStyle, PlatformState, BackendEvent, RenderCommand}; // Driver was RenderAdapter, TextRunStyle was ResolvedCellAttrs
use crate::renderer::{Renderer, RENDERER_DEFAULT_BG, RENDERER_DEFAULT_FG}; // Rgb is now a variant Color::Rgb(...). Colors struct removed.
                                                                           // FontDesc is now FontConfig. Using ColorScheme for palette.
use crate::glyph::{AttrFlags, Attributes, Glyph}; // Cell -> Glyph, CellAttrs -> Attributes, Flags -> AttrFlags
use crate::term::{
    CursorRenderState, CursorShape, RenderSnapshot, Selection, SnapshotLine, // Changed SelectionRenderState to Selection
}; // CursorStyle removed, SelectionRange -> SelectionRenderState

use std::sync::Mutex;
use anyhow::Result;
// Rc is not used for colors directly in the renderer anymore, but might be useful for shared config.
// For simplicity here, we'll manage colors more directly or via Config.

// Mocking structures for testing the renderer's interaction with a Driver.
#[derive(Debug, Clone, PartialEq)]
enum DrawCommand {
    ClearAll(Color), // Represents Driver::clear_all
    FillRect {
        rect: CellRect,
        color: Color,
    }, // Represents Driver::fill_rect
    DrawTextRun {
        coords: CellCoords,
        text: String,
        style: TextRunStyle,
    }, // Represents Driver::draw_text_run
    Present,
}

// Helper for float comparisons (if needed for pixel calculations, though tests focus on cell ops)
// fn float_eq(a: f32, b: f32) -> bool { (a - b).abs() < 0.001 }

// Mock Driver implementation
struct MockDriver {
    commands: Mutex<Vec<RenderCommand>>, // Changed from DrawCommand to RenderCommand
    font_width: usize,
    font_height: usize,
    display_width_px: u16,
    display_height_px: u16,
}

impl MockDriver {
    fn new(
        font_width: usize,
        font_height: usize,
        display_width_px: u16,
        display_height_px: u16,
    ) -> Self {
        Self {
            commands: Mutex::new(Vec::new()),
            font_width,
            font_height,
            display_width_px,
            display_height_px,
        }
    }

    fn commands(&self) -> Vec<RenderCommand> { // Changed from DrawCommand to RenderCommand
        self.commands.lock().unwrap().clone()
    }

    fn clear_commands(&self) {
        self.commands.lock().unwrap().clear();
    }
}

impl Driver for MockDriver {
    fn new() -> anyhow::Result<Self>
    where
        Self: Sized,
    {
        Ok(MockDriver::new(8, 16, 80 * 8, 24 * 16))
    }

    fn get_event_fd(&self) -> Option<std::os::unix::io::RawFd> {
        None
    }
    fn process_events(&mut self) -> anyhow::Result<Vec<BackendEvent>> {
        Ok(Vec::new())
    }

    fn get_platform_state(&self) -> PlatformState { // Added implementation
        PlatformState {
            event_fd: None,
            font_cell_width_px: self.font_width,
            font_cell_height_px: self.font_height,
            display_width_px: self.display_width_px,
            display_height_px: self.display_height_px,
            scale_factor: 1.0,
        }
    }

    fn execute_render_commands(&mut self, commands: Vec<RenderCommand>) -> Result<()> { // Added implementation
        self.commands.lock().unwrap().extend(commands);
        Ok(())
    }

    fn present(&mut self) -> anyhow::Result<()> {
        self.commands.lock().unwrap().push(RenderCommand::PresentFrame); // Changed to PresentFrame
        Ok(())
    }

    fn set_title(&mut self, _title: &str) {}
    fn bell(&mut self) {}
    fn set_cursor_visibility(&mut self, _visibility: CursorVisibility) {
    }
    fn set_focus(&mut self, _focus_state: FocusState) { // Changed parameter name
    }
    fn cleanup(&mut self) -> anyhow::Result<()> {
        Ok(())
    }
    fn own_selection(&mut self, _selection_name_atom_u64: u64, _text: String) {} // Added own_selection
    fn request_selection_data(&mut self, _selection_name_atom_u64: u64, _target_atom_u64: u64) {} // Added request_selection_data
}

// Helper to create a RenderSnapshot for tests
fn create_test_snapshot(
    lines_data: Vec<SnapshotLine>,
    cursor_state: Option<CursorRenderState>,
    num_cols: usize,
    num_rows: usize,
    selection: Selection, // Changed from Option<Selection> to Selection
) -> RenderSnapshot {
    RenderSnapshot {
        dimensions: (num_cols, num_rows),
        lines: lines_data,
        cursor_state,
        selection, // Changed from selection_state
    }
}

// Helper to create default attributes
fn default_attrs() -> Attributes {
    Attributes {
        fg: RENDERER_DEFAULT_FG,
        bg: RENDERER_DEFAULT_BG,
        flags: AttrFlags::empty(),
    }
}

#[test]
fn test_render_empty_screen_with_cursor() {
    let font_width = 8;
    let font_height = 16;
    let num_cols = 3;
    let num_rows = 2;
    let display_w_px = (num_cols * font_width) as u16;
    let display_h_px = (num_rows * font_height) as u16;

    let renderer = Renderer::new();
    let mut adapter = MockDriver::new(font_width, font_height, display_w_px, display_h_px);

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

    let snapshot = create_test_snapshot(lines, cursor_render_state, num_cols, num_rows, Selection::default()); // Pass Selection::default()
    let render_commands = renderer // Changed variable name
        .draw(snapshot) // Removed adapter from draw call
        .expect("Render draw failed");
    adapter.execute_render_commands(render_commands).expect("Execute render commands failed"); // Execute commands
    adapter.present().expect("Adapter present failed"); // Call present

    let commands = adapter.commands();
    assert!(!commands.is_empty(), "Should have drawing commands");

    let expected_bg_fill_for_line0 = RenderCommand::FillRect { // Changed to RenderCommand
        x: 0,
        y: 0,
        width: num_cols,
        height: 1,
        color: RENDERER_DEFAULT_BG,
        is_selection_bg: false, // Added is_selection_bg
    };
    let expected_bg_fill_for_line1 = RenderCommand::FillRect { // Changed to RenderCommand
        x: 0,
        y: 1,
        width: num_cols,
        height: 1,
        color: RENDERER_DEFAULT_BG,
        is_selection_bg: false, // Added is_selection_bg
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
    let expected_cursor_draw = RenderCommand::DrawTextRun { // Changed to RenderCommand
        x: 0,
        y: 0,
        text: " ".to_string(),
        fg: cursor_draw_style.fg, // Pass individual fields
        bg: cursor_draw_style.bg,
        flags: cursor_draw_style.flags,
        is_selected: false, // Added is_selected
    };

    assert!(
        commands.contains(&expected_cursor_draw),
        "Missing cursor draw command. Commands: {:?}",
        commands
    );
    assert_eq!(
        commands.last().unwrap(),
        &RenderCommand::PresentFrame, // Changed to PresentFrame
        "Last command should be Present"
    );
}

#[test]
fn test_render_simple_text() {
    let font_width = 8;
    let font_height = 16;
    let num_cols = 5;
    let num_rows = 1;
    let display_w_px = (num_cols * font_width) as u16;
    let display_h_px = (num_rows * font_height) as u16;

    let renderer = Renderer::new();
    let mut adapter = MockDriver::new(font_width, font_height, display_w_px, display_h_px);

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

    let snapshot = create_test_snapshot(lines, cursor_render_state, num_cols, num_rows, Selection::default()); // Pass Selection::default()
    let render_commands = renderer // Changed variable name
        .draw(snapshot) // Removed adapter from draw call
        .expect("Render draw failed");
    adapter.execute_render_commands(render_commands).expect("Execute render commands failed"); // Execute commands
    adapter.present().expect("Adapter present failed"); // Call present

    let commands = adapter.commands();

    let text_style = TextRunStyle {
        fg: RENDERER_DEFAULT_FG,
        bg: RENDERER_DEFAULT_BG,
        flags: AttrFlags::empty(),
    };

    let expected_text_hi = RenderCommand::DrawTextRun { // Changed to RenderCommand
        x: 0,
        y: 0,
        text: "Hi".to_string(),
        fg: text_style.fg, // Pass individual fields
        bg: text_style.bg,
        flags: text_style.flags,
        is_selected: false, // Added is_selected
    };
    assert!(
        commands.contains(&expected_text_hi),
        "Missing text 'Hi'. Commands: {:?}",
        commands
    );

    let expected_fill_spaces = RenderCommand::FillRect { // Changed to RenderCommand
        x: 2,
        y: 0,
        width: num_cols - 2,
        height: 1,
        color: RENDERER_DEFAULT_BG,
        is_selection_bg: false, // Added is_selection_bg
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
    let expected_cursor_draw = RenderCommand::DrawTextRun { // Changed to RenderCommand
        x: 2,
        y: 0,
        text: " ".to_string(),
        fg: cursor_draw_style.fg, // Pass individual fields
        bg: cursor_draw_style.bg,
        flags: cursor_draw_style.flags,
        is_selected: false, // Added is_selected
    };
    assert!(
        commands.contains(&expected_cursor_draw),
        "Missing cursor draw. Commands: {:?}",
        commands
    );
    assert_eq!(commands.last().unwrap(), &RenderCommand::PresentFrame); // Changed to PresentFrame
}

#[test]
fn test_render_dirty_line_only() {
    let font_width = 8;
    let font_height = 16;
    let num_cols = 3;
    let num_rows = 2;
    let display_w_px = (num_cols * font_width) as u16;
    let display_h_px = (num_rows * font_height) as u16;

    let renderer = Renderer::new();
    let mut adapter = MockDriver::new(font_width, font_height, display_w_px, display_h_px);

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
        Selection::default(), // Pass Selection::default()
    );
    let render_commands1 = renderer // Changed variable name
        .draw(snapshot1) // Removed adapter from draw call
        .expect("Render draw failed");
    adapter.execute_render_commands(render_commands1).expect("Execute render commands failed"); // Execute commands
    adapter.clear_commands();

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
            is_dirty: false,
            cells: line0_cells_frame2,
        },
        SnapshotLine {
            is_dirty: true,
            cells: line1_cells_frame2,
        },
    ];
    let cursor_state_frame2 = Some(CursorRenderState {
        x: 0,
        y: 1,
        shape: CursorShape::Block,
        cell_char_underneath: 'C',
        cell_attributes_underneath: default_attrs(),
    });

    let snapshot2 =
        create_test_snapshot(lines_frame2, cursor_state_frame2, num_cols, num_rows, Selection::default()); // Pass Selection::default()
    let render_commands2 = renderer // Changed variable name
        .draw(snapshot2) // Removed adapter from draw call
        .expect("Render draw failed");
    adapter.execute_render_commands(render_commands2).expect("Execute render commands failed"); // Execute commands
    adapter.present().expect("Adapter present failed"); // Call present
    let commands_frame2 = adapter.commands();

    let text_style = TextRunStyle {
        fg: RENDERER_DEFAULT_FG,
        bg: RENDERER_DEFAULT_BG,
        flags: AttrFlags::empty(),
    };

    let draw_a_cmd = RenderCommand::DrawTextRun { // Changed to RenderCommand
        x: 0,
        y: 0,
        text: "A".to_string(),
        fg: text_style.fg, // Pass individual fields
        bg: text_style.bg,
        flags: text_style.flags,
        is_selected: false, // Added is_selected
    };
    assert!(
        !commands_frame2.contains(&draw_a_cmd),
        "Text 'A' from non-dirty line 0 should not be redrawn. Commands: {:?}",
        commands_frame2
    );

    let draw_c_cmd = RenderCommand::DrawTextRun { // Changed to RenderCommand
        x: 0,
        y: 1,
        text: "C".to_string(),
        fg: text_style.fg, // Pass individual fields
        bg: text_style.bg,
        flags: text_style.flags,
        is_selected: false, // Added is_selected
    };
    assert!(
        commands_frame2.contains(&draw_c_cmd),
        "Text 'C' from dirty line 1 should be redrawn. Commands: {:?}",
        commands_frame2
    );

    let cursor_draw_style = TextRunStyle {
        fg: RENDERER_DEFAULT_BG,
        bg: RENDERER_DEFAULT_FG,
        flags: AttrFlags::empty(),
    };
    let expected_cursor_draw = RenderCommand::DrawTextRun { // Changed to RenderCommand
        x: 0,
        y: 1,
        text: "C".to_string(),
        fg: cursor_draw_style.fg, // Pass individual fields
        bg: cursor_draw_style.bg,
        flags: cursor_draw_style.flags,
        is_selected: false, // Added is_selected
    };
    assert!(
        commands_frame2.contains(&expected_cursor_draw),
        "Missing cursor draw on line 1. Commands: {:?}",
        commands_frame2
    );

    assert_eq!(commands_frame2.last().unwrap(), &RenderCommand::PresentFrame); // Changed to PresentFrame
}

// --- AppOrchestrator Tests ---
use crate::orchestrator::{AppOrchestrator, OrchestratorStatus};
use crate::term::{EmulatorAction, EmulatorInput, TerminalInterface, UserInputAction};
use crate::ansi::{AnsiParser, ControlCode, EscCode, SgrCode, StandardAnsiChar}; // Assuming AnsiParser and related types are needed for mock
use crate::platform::os::pty::PtyChannel;
use crate::keys::{KeySymbol, Modifiers}; // For keybinding tests

use std::io;
use std::collections::VecDeque;
use std::cell::RefCell;

// Mock PtyChannel
struct MockPtyChannel {
    read_data: VecDeque<u8>,
    written_data: RefCell<Vec<u8>>,
    resize_calls: RefCell<Vec<(u16, u16)>>,
}

impl MockPtyChannel {
    fn new() -> Self {
        MockPtyChannel {
            read_data: VecDeque::new(),
            written_data: RefCell::new(Vec::new()),
            resize_calls: RefCell::new(Vec::new()),
        }
    }
}

impl PtyChannel for MockPtyChannel {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let mut count = 0;
        for i in 0..buf.len() {
            if let Some(byte) = self.read_data.pop_front() {
                buf[i] = byte;
                count += 1;
            } else {
                break;
            }
        }
        if count == 0 && self.read_data.is_empty() {
            // Simulate WouldBlock if no data was ever added, or all has been read.
            // This behavior might need adjustment depending on specific test needs.
            // If read_data starts empty, it will immediately return WouldBlock.
            // If data is added and then depleted, subsequent calls yield WouldBlock.
            return Err(io::Error::new(io::ErrorKind::WouldBlock, "No data"));
        }
        Ok(count)
    }

    fn write_all(&mut self, buf: &[u8]) -> io::Result<()> {
        self.written_data.borrow_mut().extend_from_slice(buf);
        Ok(())
    }

    fn resize(&mut self, cols: u16, rows: u16) -> io::Result<()> {
        self.resize_calls.borrow_mut().push((cols, rows));
        Ok(())
    }

    fn get_raw_fd(&self) -> Option<std::os::unix::io::RawFd> {
        None
    }
}


// Mock TerminalInterface
#[derive(Clone)]
struct MockTerminal {
    inputs: RefCell<Vec<EmulatorInput>>,
    actions_to_return: RefCell<VecDeque<EmulatorAction>>, // Actions it will return on interpret_input
}

impl MockTerminal {
    fn new() -> Self {
        MockTerminal {
            inputs: RefCell::new(Vec::new()),
            actions_to_return: RefCell::new(VecDeque::new()),
        }
    }

    fn get_inputs(&self) -> Vec<EmulatorInput> {
        self.inputs.borrow().clone()
    }

    #[allow(dead_code)] // May be used in future tests
    fn add_action_to_return(&self, action: EmulatorAction) {
        self.actions_to_return.borrow_mut().push_back(action);
    }
}

impl TerminalInterface for MockTerminal {
    fn interpret_input(&mut self, input: EmulatorInput) -> Option<EmulatorAction> {
        self.inputs.borrow_mut().push(input);
        self.actions_to_return.borrow_mut().pop_front()
    }

    fn get_render_snapshot(&self) -> RenderSnapshot {
        // Provide a minimal default snapshot, tests can customize if needed by wrapping MockTerminal
        create_test_snapshot(Vec::new(), None, 1, 1, Selection::default())
    }

    // Other methods can be default or panic if not expected to be called
    fn get_current_title(&self) -> String {
        "MockTerminal".to_string()
    }
    fn handle_action(&mut self, _action: EmulatorAction) {} // Simplified
    fn get_cell_size_px(&self) -> (usize, usize) { (8,16) }
    fn get_dimensions_cells(&self) -> (usize, usize) { (80,24) }
}

// Mock AnsiParser
struct MockAnsiParser;

impl AnsiParser for MockAnsiParser {
    fn process_bytes(&mut self, _bytes: &[u8]) -> Vec<StandardAnsiChar> {
        Vec::new() // Return no commands for simplicity in these tests
    }
}


// Adapter for MockDriver to make it usable in AppOrchestrator tests
// The existing MockDriver is fine for capturing render commands.
// We need a way to feed BackendEvents into the orchestrator.
struct TestHarness {
    pty: MockPtyChannel,
    term: MockTerminal,
    parser: MockAnsiParser,
    driver: MockDriver, // Use existing MockDriver for rendering, add event feeding if needed
}

impl TestHarness {
    fn new(font_w: usize, font_h: usize, display_w: u16, display_h: u16) -> Self {
        TestHarness {
            pty: MockPtyChannel::new(),
            term: MockTerminal::new(),
            parser: MockAnsiParser,
            driver: MockDriver::new(font_w, font_h, display_w, display_h),
        }
    }

    // Helper to create orchestrator with current mocks
    fn create_orchestrator<'a>(&'a mut self, renderer: Renderer) -> AppOrchestrator<'a> {
        AppOrchestrator::new(
            &mut self.pty,
            &mut self.term,
            &mut self.parser,
            renderer,
            &mut self.driver,
        )
    }
}


#[test]
fn test_orchestrator_keybinding_copy_default_config() {
    let mut harness = TestHarness::new(8, 16, 80 * 8, 24 * 16);
    let renderer = Renderer::new(); // Renderer isn't the focus here
    let mut orchestrator = harness.create_orchestrator(renderer);

    // Default copy: Ctrl+Shift+C
    let key_event = BackendEvent::Key {
        symbol: KeySymbol::Char('C'),
        modifiers: Modifiers::CONTROL | Modifiers::SHIFT,
        text: "C".to_string(), // Text might or might not be there for shortcuts
    };

    // Simulate this event coming from the driver
    orchestrator.handle_specific_driver_event(key_event);

    let terminal_inputs = harness.term.get_inputs();
    assert_eq!(terminal_inputs.len(), 1, "Terminal should have received one input");

    match &terminal_inputs[0] {
        EmulatorInput::User(UserInputAction::InitiateCopy) => {
            // Correct action received
        }
        other => panic!("Expected InitiateCopy, got {:?}", other),
    }
}

#[test]
fn test_orchestrator_keybinding_paste_default_config() {
    let mut harness = TestHarness::new(8, 16, 80 * 8, 24 * 16);
    let renderer = Renderer::new();
    let mut orchestrator = harness.create_orchestrator(renderer);

    // Default paste: Ctrl+Shift+V
    let key_event = BackendEvent::Key {
        symbol: KeySymbol::Char('V'),
        modifiers: Modifiers::CONTROL | Modifiers::SHIFT,
        text: "V".to_string(),
    };

    orchestrator.handle_specific_driver_event(key_event);

    let terminal_inputs = harness.term.get_inputs();
    assert_eq!(terminal_inputs.len(), 1, "Terminal should have received one input");

    match &terminal_inputs[0] {
        EmulatorInput::User(UserInputAction::RequestClipboardPaste) => {
            // Correct action received
        }
        other => panic!("Expected RequestClipboardPaste, got {:?}", other),
    }
}

#[test]
fn test_orchestrator_keybinding_no_match_falls_through() {
    let mut harness = TestHarness::new(8, 16, 80 * 8, 24 * 16);
    let renderer = Renderer::new();
    let mut orchestrator = harness.create_orchestrator(renderer);

    // A key event that doesn't match default copy/paste
    let key_event = BackendEvent::Key {
        symbol: KeySymbol::Char('A'),
        modifiers: Modifiers::NONE,
        text: "a".to_string(),
    };

    orchestrator.handle_specific_driver_event(key_event);

    let terminal_inputs = harness.term.get_inputs();
    assert_eq!(terminal_inputs.len(), 1, "Terminal should have received one input");

    match &terminal_inputs[0] {
        EmulatorInput::User(UserInputAction::KeyInput { symbol, modifiers, text }) => {
            assert_eq!(*symbol, KeySymbol::Char('A'));
            assert_eq!(*modifiers, Modifiers::NONE);
            assert_eq!(text.as_deref(), Some("a"));
        }
        other => panic!("Expected KeyInput, got {:?}", other),
    }
}

#[test]
fn test_orchestrator_keybinding_match_prevents_fallback() {
    let mut harness = TestHarness::new(8, 16, 80 * 8, 24 * 16);
    let renderer = Renderer::new();
    let mut orchestrator = harness.create_orchestrator(renderer);

    // Default copy: Ctrl+Shift+C
    let key_event = BackendEvent::Key {
        symbol: KeySymbol::Char('C'),
        modifiers: Modifiers::CONTROL | Modifiers::SHIFT,
        text: "C".to_string(),
    };

    orchestrator.handle_specific_driver_event(key_event);

    let terminal_inputs = harness.term.get_inputs();
    assert_eq!(terminal_inputs.len(), 1, "Terminal should have received one input"); // Crucial: only one input

    // Already verified it's InitiateCopy in another test. Here, focus is on *not* also getting KeyInput.
    if let EmulatorInput::User(UserInputAction::KeyInput { .. }) = &terminal_inputs[0] {
        panic!("KeyInput should not be sent when a keybinding matches.");
    }
    // Ensure it's the copy action
     match &terminal_inputs[0] {
        EmulatorInput::User(UserInputAction::InitiateCopy) => { /* Correct */ }
        other => panic!("Expected InitiateCopy, got {:?}", other),
    }
}

// TODO: Test for paste_primary if a default is added or if config can be easily mocked.
// For now, paste_primary is None by default, so no key event will trigger it without config changes.

// Example of how a test for paste_primary might look if CONFIG could be mocked:
/*
#[test]
fn test_orchestrator_keybinding_paste_primary_custom_config() {
    // --- ARRANGE ---
    // 1. Setup Mocking for CONFIG to return custom keybindings
    // This is the hard part. One way could be to have a global static AtomicRefCell
    // for the config, but that's a larger refactor.
    // For this example, let's assume we have a way:
    // `override_global_config_for_test(custom_config);`

    let custom_bindings = CopyPasteKeybindings {
        copy: vec![], // Don't care for this test
        paste_clipboard: vec![], // Don't care
        paste_primary: Some(vec![Keybinding {
            key: KeySymbol::Char('P'),
            mods: Modifiers::ALT,
        }]),
    };
    // This would require `CONFIG` to be mutable or for `AppOrchestrator` to take config:
    // For now, this test is hypothetical.
    // If AppOrchestrator could take config: `orchestrator.override_keybindings(custom_bindings);`

    let mut harness = TestHarness::new(8, 16, 80 * 8, 24 * 16);
    let renderer = Renderer::new();
    let mut orchestrator = harness.create_orchestrator(renderer);

    // --- ACT ---
    let key_event = BackendEvent::Key {
        symbol: KeySymbol::Char('P'),
        modifiers: Modifiers::ALT,
        text: "P".to_string(),
    };
    orchestrator.handle_specific_driver_event(key_event);

    // --- ASSERT ---
    let terminal_inputs = harness.term.get_inputs();
    assert_eq!(terminal_inputs.len(), 1);
    match &terminal_inputs[0] {
        EmulatorInput::User(UserInputAction::RequestPrimaryPaste) => {}
        other => panic!("Expected RequestPrimaryPaste, got {:?}", other),
    }

    // `restore_global_config_after_test();`
}
*/
