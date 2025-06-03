// src/orchestrator/tests.rs

use crate::config::{self, Config, Keybinding, KeybindingsConfig};
use crate::glyph::{AttrFlags, Attributes, ContentCell, Glyph};
use crate::keys::{KeySymbol, Modifiers}; // Added for KeySymbol, Modifiers
use crate::platform::backends::{
    BackendEvent, CursorVisibility, Driver, FocusState, PlatformState, RenderCommand, TextRunStyle,
};
use crate::platform::os::pty::PtyChannel; // Added for PtyChannel trait
use crate::renderer::Renderer;
use crate::term::{
    AnsiCommand, CursorRenderState, CursorShape, EmulatorAction, EmulatorInput, RenderSnapshot,
    Selection, SnapshotLine, TerminalInterface, UserInputAction,
}; // Added for TerminalInterface, UserInputAction, EmulatorInput, AnsiCommand, EmulatorAction
use crate::ansi::AnsiParser; // Added for AnsiParser trait
use crate::orchestrator::AppOrchestrator; // Added for AppOrchestrator

use anyhow::Result;
use std::io;
use std::sync::{Arc, Mutex}; // Added Arc

// --- Mock Implementations ---

// Mock PtyChannel
struct MockPtyChannel {
    written_data: Mutex<Vec<u8>>,
    read_data: Mutex<Vec<u8>>, // Data to be returned by read
}

impl MockPtyChannel {
    fn new() -> Self {
        Self {
            written_data: Mutex::new(Vec::new()),
            read_data: Mutex::new(Vec::new()),
        }
    }
}

impl PtyChannel for MockPtyChannel {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let mut read_data_guard = self.read_data.lock().unwrap();
        if read_data_guard.is_empty() {
            return Err(io::Error::new(io::ErrorKind::WouldBlock, "No data"));
        }
        let len = std::cmp::min(buf.len(), read_data_guard.len());
        buf[..len].copy_from_slice(&read_data_guard[..len]);
        *read_data_guard = read_data_guard[len..].to_vec(); // Consume data
        Ok(len)
    }

    fn write_all(&mut self, buf: &[u8]) -> io::Result<()> {
        self.written_data.lock().unwrap().extend_from_slice(buf);
        Ok(())
    }
    fn resize(&mut self, _cols: u16, _rows: u16) -> io::Result<()> {
        Ok(())
    } // Minimal implementation
    fn get_reader_fd(&self) -> Option<std::os::unix::io::RawFd> {
        None
    } // Minimal implementation
}

// Mock TerminalInterface
#[derive(Clone)] // Added Clone for Arc<Mutex<MockTerminalInterface>> if needed by test_setup
struct MockTerminalInterface {
    last_input: Arc<Mutex<Option<EmulatorInput>>>,
    snapshot_to_return: Arc<Mutex<RenderSnapshot>>, // Added for get_render_snapshot
}

impl MockTerminalInterface {
    fn new() -> Self {
        Self {
            last_input: Arc::new(Mutex::new(None)),
            // Provide a default snapshot for get_render_snapshot
            snapshot_to_return: Arc::new(Mutex::new(RenderSnapshot {
                dimensions: (80, 24),
                lines: Vec::new(),
                cursor_state: None,
                selection: Selection::default(),
            })),
        }
    }
    fn last_input(&self) -> Option<EmulatorInput> {
        self.last_input.lock().unwrap().clone()
    }
}

impl TerminalInterface for MockTerminalInterface {
    fn interpret_input(&mut self, input: EmulatorInput) -> Option<EmulatorAction> {
        *self.last_input.lock().unwrap() = Some(input);
        None // For simplicity, assume no action is returned unless specifically testing for it
    }
    fn get_render_snapshot(&self) -> RenderSnapshot {
        self.snapshot_to_return.lock().unwrap().clone()
    }
    // Add other methods if AppOrchestrator calls them
}

// Mock AnsiParser
struct MockAnsiParser;
impl AnsiParser for MockAnsiParser {
    fn process_bytes(&mut self, _bytes: &[u8]) -> Vec<AnsiCommand> {
        Vec::new() // Return no commands for simplicity
    }
}

// Mock Driver implementation (already exists, ensure it's suitable)
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

    fn commands(&self) -> Vec<RenderCommand> {
        // Changed from DrawCommand to RenderCommand
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

    fn get_platform_state(&self) -> PlatformState {
        // Added implementation
        PlatformState {
            event_fd: None,
            font_cell_width_px: self.font_width,
            font_cell_height_px: self.font_height,
            display_width_px: self.display_width_px,
            display_height_px: self.display_height_px,
            scale_factor: 1.0,
        }
    }

    fn execute_render_commands(&mut self, commands: Vec<RenderCommand>) -> Result<()> {
        // Added implementation
        self.commands.lock().unwrap().extend(commands);
        Ok(())
    }

    fn present(&mut self) -> anyhow::Result<()> {
        self.commands
            .lock()
            .unwrap()
            .push(RenderCommand::PresentFrame); // Changed to PresentFrame
        Ok(())
    }

    fn set_title(&mut self, _title: &str) {}
    fn bell(&mut self) {}
    fn set_cursor_visibility(&mut self, _visibility: CursorVisibility) {}
    fn set_focus(&mut self, _focus_state: FocusState) { // Changed parameter name
    }
    fn cleanup(&mut self) -> anyhow::Result<()> {
        Ok(())
    }
    fn own_selection(&mut self, _selection_name_atom_u64: u64, _text: String) {} // Added own_selection
    fn request_selection_data(&mut self, _selection_name_atom_u64: u64, _target_atom_u64: u64) {}
}


// --- Test Setup Helper ---

// Helper structure to hold mocks for convenience in tests
struct TestMocks {
    pty: MockPtyChannel,
    term: MockTerminalInterface,
    parser: MockAnsiParser,
    driver: MockDriver,
}

fn setup_orchestrator_with_mocks() -> (AppOrchestrator<'static>, TestMocks) {
    // Leak the mocks to get 'static references. This is common in test setups
    // where actual cleanup isn't critical.
    let pty = Box::leak(Box::new(MockPtyChannel::new()));
    let term = Box::leak(Box::new(MockTerminalInterface::new()));
    let parser = Box::leak(Box::new(MockAnsiParser));
    let driver = Box::leak(Box::new(MockDriver::new(8, 16, 800, 600))); // Default dimensions

    let renderer = Renderer::new(); // Real renderer
    let orchestrator = AppOrchestrator::new(pty, term, parser, renderer, driver);

    (
        orchestrator,
        TestMocks {
            pty: pty, // This is problematic, we pass &mut to orchestrator.
                      // We need a way to inspect these after they are moved or borrowed.
                      // For now, MockTerminalInterface uses Arc<Mutex> for last_input,
                      // which works around this for checking terminal inputs.
                      // If we need to inspect pty or driver, they'd need similar Arc<Mutex> fields.
                      // Let's refine TestMocks or how we access them.
                      // For now, we'll primarily rely on MockTerminalInterface's Arc for assertions.
            // The leaked boxes are fine, but direct access to their fields after passing &mut
            // to AppOrchestrator::new is not.
            // The solution is to have the test setup return the orchestrator AND the means to inspect.
            // The MockTerminalInterface is already designed for this with its Arc<Mutex<Option<EmulatorInput>>>.
            // So, we actually need to return the Arc<Mutex<Option<EmulatorInput>>> or the MockTerminalInterface clone.
            // Let's simplify: setup will return the orchestrator and the inspectable parts of mocks.
            // We don't need to return the whole TestMocks struct if we return inspectable parts.
            // For now, we'll just return the orchestrator and the term's inspectable field.
            // This part of the helper needs careful thought on what tests need to assert.
            // Let's assume tests will primarily check terminal input for these keybinding tests.

            // This TestMocks struct is not really usable as is because AppOrchestrator takes mutable borrows.
            // The important part is that the mocks given to AppOrchestrator are the ones we can later inspect.
            // The `term` mock is designed for this.
            pty: MockPtyChannel::new(), // Placeholder, not the one used by orchestrator
            term: term.clone(), // Clone the interface that has the Arc, so we can inspect it.
            parser: MockAnsiParser, // Placeholder
            driver: MockDriver::new(1,1,1,1), // Placeholder
        },
    )
}
// Revised setup to return the inspectable mock part directly
fn setup_orchestrator_for_key_tests() -> (AppOrchestrator<'static>, Arc<Mutex<Option<EmulatorInput>>>) {
    let pty = Box::leak(Box::new(MockPtyChannel::new()));
    let term_mock = Box::leak(Box::new(MockTerminalInterface::new()));
    let parser = Box::leak(Box::new(MockAnsiParser));
    let driver = Box::leak(Box::new(MockDriver::new(8, 16, 800, 600)));

    let renderer = Renderer::new();
    let orchestrator = AppOrchestrator::new(pty, term_mock, parser, renderer, driver);
    (orchestrator, term_mock.last_input.clone())
}


// Helper to create a RenderSnapshot for tests (already exists)
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
        fg: config::CONFIG.colors.foreground,
        bg: config::CONFIG.colors.background,
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

    let default_glyph = Glyph::Single(ContentCell {
        c: ' ',
        attr: default_attrs(),
    });
    let lines = vec![
        SnapshotLine {
            is_dirty: true,
            cells: vec![default_glyph.clone(); num_cols]
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

    let snapshot = create_test_snapshot(
        lines,
        cursor_render_state,
        num_cols,
        num_rows,
        Selection::default(),
    );
    let render_commands = renderer // Changed variable name
        .draw(snapshot) // Removed adapter from draw call
        .expect("Render draw failed");
    adapter
        .execute_render_commands(render_commands)
        .expect("Execute render commands failed");
    adapter.present().expect("Adapter present failed");

    let commands = adapter.commands();
    assert!(!commands.is_empty(), "Should have drawing commands");

    let expected_bg_fill_for_line0 = RenderCommand::FillRect {
        x: 0,
        y: 0,
        width: num_cols,
        height: 1,
        color: config::CONFIG.colors.background,
        is_selection_bg: false,
    };
    let expected_bg_fill_for_line1 = RenderCommand::FillRect {
        x: 0,
        y: 1,
        width: num_cols,
        height: 1,
        color: config::CONFIG.colors.background,
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
        fg: config::CONFIG.colors.background,
        bg: config::CONFIG.colors.foreground,
        flags: AttrFlags::empty(),
    };
    let expected_cursor_draw = RenderCommand::DrawTextRun {
        // Changed to RenderCommand
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
        Glyph::Single(ContentCell {
            c: ' ',
            attr: default_attrs()
        });
        num_cols
    ];
    line_cells[0] = Glyph::Single(ContentCell {
        c: 'H',
        attr: default_attrs(),
    });
    line_cells[1] = Glyph::Single(ContentCell {
        c: 'i',
        attr: default_attrs(),
    });

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

    let snapshot = create_test_snapshot(
        lines,
        cursor_render_state,
        num_cols,
        num_rows,
        Selection::default(),
    ); // Pass Selection::default()
    let render_commands = renderer // Changed variable name
        .draw(snapshot) // Removed adapter from draw call
        .expect("Render draw failed");
    adapter
        .execute_render_commands(render_commands)
        .expect("Execute render commands failed"); // Execute commands
    adapter.present().expect("Adapter present failed"); // Call present

    let commands = adapter.commands();

    let text_style = TextRunStyle {
        fg: config::CONFIG.colors.foreground,
        bg: config::CONFIG.colors.background,
        flags: AttrFlags::empty(),
    };

    let expected_text_hi = RenderCommand::DrawTextRun {
        // Changed to RenderCommand
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

    let expected_fill_spaces = RenderCommand::FillRect {
        // Changed to RenderCommand
        x: 2,
        y: 0,
        width: num_cols - 2,
        height: 1,
        color: config::CONFIG.colors.background,
        is_selection_bg: false, // Added is_selection_bg
    };
    assert!(
        commands.contains(&expected_fill_spaces),
        "Missing fill for remaining spaces. Commands: {:?}",
        commands
    );

    let cursor_draw_style = TextRunStyle {
        fg: config::CONFIG.colors.background,
        bg: config::CONFIG.colors.foreground,
        flags: AttrFlags::empty(),
    };
    let expected_cursor_draw = RenderCommand::DrawTextRun {
        // Changed to RenderCommand
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
        Glyph::Single(ContentCell {
            c: ' ',
            attr: default_attrs()
        });
        num_cols
    ];
    line0_cells[0] = Glyph::Single(ContentCell {
        c: 'A',
        attr: default_attrs(),
    });

    let line1_cells = vec![
        Glyph::Single(ContentCell {
            c: 'B',
            attr: default_attrs()
        });
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
    adapter
        .execute_render_commands(render_commands1)
        .expect("Execute render commands failed"); // Execute commands
    adapter.clear_commands();

    let mut line0_cells_frame2 = vec![
        Glyph::Single(ContentCell {
            c: ' ',
            attr: default_attrs()
        });
        num_cols
    ];
    line0_cells_frame2[0] = Glyph::Single(ContentCell {
        c: 'A',
        attr: default_attrs(),
    });

    let mut line1_cells_frame2 = vec![
        Glyph::Single(ContentCell {
            c: ' ',
            attr: default_attrs()
        });
        num_cols
    ];
    line1_cells_frame2[0] = Glyph::Single(ContentCell {
        c: 'C',
        attr: default_attrs(),
    });

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

    let snapshot2 = create_test_snapshot(
        lines_frame2,
        cursor_state_frame2,
        num_cols,
        num_rows,
        Selection::default(),
    ); // Pass Selection::default()
    let render_commands2 = renderer // Changed variable name
        .draw(snapshot2) // Removed adapter from draw call
        .expect("Render draw failed");
    adapter
        .execute_render_commands(render_commands2)
        .expect("Execute render commands failed"); // Execute commands
    adapter.present().expect("Adapter present failed"); // Call present
    let commands_frame2 = adapter.commands();

    let text_style = TextRunStyle {
        fg: config::CONFIG.colors.foreground,
        bg: config::CONFIG.colors.background,
        flags: AttrFlags::empty(),
    };

    let draw_a_cmd = RenderCommand::DrawTextRun {
        // Changed to RenderCommand
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

    let draw_c_cmd = RenderCommand::DrawTextRun {
        // Changed to RenderCommand
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
        fg: config::CONFIG.colors.background,
        bg: config::CONFIG.colors.foreground,
        flags: AttrFlags::empty(),
    };
    let expected_cursor_draw = RenderCommand::DrawTextRun {
        // Changed to RenderCommand
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

    assert_eq!(
        commands_frame2.last().unwrap(),
        &RenderCommand::PresentFrame
    );
}

// --- Keybinding Tests ---

#[test]
fn test_orchestrator_handles_default_copy_binding() {
    let (mut orchestrator, last_term_input) = setup_orchestrator_for_key_tests();
    // Default is Ctrl+Shift+C -> InitiateCopy
    // We assume KeySymbol and Modifiers from `crate::keys` (used in config)
    // are directly compatible/translatable with `crate::platform::backends` versions.
    let event = BackendEvent::Key {
        symbol: KeySymbol::Char('C'), // This is platform::backends::KeySymbol
        modifiers: Modifiers::CONTROL | Modifiers::SHIFT, // This is platform::backends::Modifiers
        text: "".to_string(),
    };
    orchestrator.handle_specific_driver_event(event);

    let input_received = last_term_input.lock().unwrap().clone();
    match input_received.as_ref() {
        Some(EmulatorInput::User(UserInputAction::InitiateCopy)) => { /* success */ }
        other => panic!("Expected InitiateCopy, got {:?}", other),
    }
}

#[test]
fn test_orchestrator_handles_default_paste_binding() {
    let (mut orchestrator, last_term_input) = setup_orchestrator_for_key_tests();
    // Default is Ctrl+Shift+V -> RequestClipboardPaste
    let event = BackendEvent::Key {
        symbol: KeySymbol::Char('V'),
        modifiers: Modifiers::CONTROL | Modifiers::SHIFT,
        text: "".to_string(),
    };
    orchestrator.handle_specific_driver_event(event);

    let input_received = last_term_input.lock().unwrap().clone();
    match input_received.as_ref() {
        Some(EmulatorInput::User(UserInputAction::RequestClipboardPaste)) => { /* success */ }
        other => panic!("Expected RequestClipboardPaste, got {:?}", other),
    }
}

#[test]
fn test_orchestrator_handles_unbound_key_as_keyinput() {
    let (mut orchestrator, last_term_input) = setup_orchestrator_for_key_tests();
    // Assuming F12 with no modifiers is not bound by default
    let key_sym = KeySymbol::F12;
    let key_mods = Modifiers::empty();
    let event = BackendEvent::Key {
        symbol: key_sym,
        modifiers: key_mods,
        text: "test_text".to_string(), // Add some text to check if it's passed correctly
    };
    orchestrator.handle_specific_driver_event(event);

    let input_received = last_term_input.lock().unwrap().clone();
    match input_received.as_ref() {
        Some(EmulatorInput::User(UserInputAction::KeyInput { symbol, modifiers, text })) => {
            assert_eq!(*symbol, key_sym);
            assert_eq!(*modifiers, key_mods);
            assert_eq!(*text, Some("test_text".to_string()));
        }
        other => panic!("Expected KeyInput, got {:?}", other),
    }
}

#[test]
fn test_orchestrator_handles_unbound_key_as_keyinput_empty_text() {
    let (mut orchestrator, last_term_input) = setup_orchestrator_for_key_tests();
    let key_sym = KeySymbol::Home; // Another unbound key example
    let key_mods = Modifiers::ALT;
    let event = BackendEvent::Key {
        symbol: key_sym,
        modifiers: key_mods,
        text: "".to_string(), // Empty text
    };
    orchestrator.handle_specific_driver_event(event);

    let input_received = last_term_input.lock().unwrap().clone();
    match input_received.as_ref() {
        Some(EmulatorInput::User(UserInputAction::KeyInput { symbol, modifiers, text })) => {
            assert_eq!(*symbol, key_sym);
            assert_eq!(*modifiers, key_mods);
            assert_eq!(*text, None); // Expect None for empty string
        }
        other => panic!("Expected KeyInput with None text, got {:?}", other),
    }
}
