// src/orchestrator/tests.rs

use crate::config;
use crate::platform::backends::{CursorVisibility, Driver, FocusState, TextRunStyle, PlatformState, BackendEvent, RenderCommand};
use crate::renderer::Renderer;
use crate::glyph::{AttrFlags, Attributes, Glyph};
use crate::term::{CursorRenderState, CursorShape, RenderSnapshot, Selection, SnapshotLine};

use std::sync::Mutex;
use anyhow::Result;

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
    let render_commands = renderer // Changed variable name
        .draw(snapshot) // Removed adapter from draw call
        .expect("Render draw failed");
    adapter.execute_render_commands(render_commands).expect("Execute render commands failed");
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
        fg: config::CONFIG.colors.foreground,
        bg: config::CONFIG.colors.background,
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
        fg: config::CONFIG.colors.foreground,
        bg: config::CONFIG.colors.background,
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
        fg: config::CONFIG.colors.background,
        bg: config::CONFIG.colors.foreground,
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
