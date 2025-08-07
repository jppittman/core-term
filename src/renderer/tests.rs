// src/renderer/tests.rs

// Imports from the main crate
use crate::color::Color;
use crate::config::Config; // Added Config, removed self
use crate::glyph::{AttrFlags, Attributes, ContentCell, Glyph};
use crate::platform::backends::{
    BackendEvent, CellCoords, CellRect, CursorVisibility, Driver, FocusState, PlatformState,
    RenderCommand as ActualRenderCommand, TextRunStyle,
};
use crate::renderer::Renderer;
use crate::term::{CursorRenderState, CursorShape, RenderSnapshot, Selection, SnapshotLine};

use anyhow::Result;
use std::sync::Mutex;

// Mocking structures for testing the renderer's interaction with a Driver.
#[derive(Debug, Clone, PartialEq)]
#[allow(dead_code)] // Allow dead code for variants not explicitly constructed in tests
enum MockDrawCommand {
    ClearAll {
        bg: Color,
    },
    FillRect {
        rect: CellRect,
        color: Color,
        is_selection_bg: bool,
    },
    DrawTextRun {
        coords: CellCoords,
        text: String,
        style: TextRunStyle,
        is_selected: bool,
    },
    Present,
}

// Mock Driver implementation
struct MockDriver {
    commands: Mutex<Vec<ActualRenderCommand>>,
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

    fn commands(&self) -> Vec<ActualRenderCommand> {
        self.commands.lock().unwrap().clone()
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
        PlatformState {
            event_fd: None,
            font_cell_width_px: self.font_width,
            font_cell_height_px: self.font_height,
            display_width_px: self.display_width_px,
            display_height_px: self.display_height_px,
            scale_factor: 1.0,
        }
    }

    fn execute_render_commands(&mut self, commands: Vec<ActualRenderCommand>) -> Result<()> {
        self.commands.lock().unwrap().extend(commands);
        Ok(())
    }

    fn present(&mut self) -> anyhow::Result<()> {
        self.commands
            .lock()
            .unwrap()
            .push(ActualRenderCommand::PresentFrame);
        Ok(())
    }

    fn set_title(&mut self, _title: &str) {}
    fn bell(&mut self) {}
    fn set_cursor_visibility(&mut self, _visible: CursorVisibility) {}
    fn set_focus(&mut self, _focus_state: FocusState) {}
    fn cleanup(&mut self) -> anyhow::Result<()> {
        Ok(())
    }
    fn own_selection(&mut self, _selection_name_atom_u64: u64, _text: String) {}
    fn request_selection_data(&mut self, _selection_name_atom_u64: u64, _target_atom_u64: u64) {}
}

// Helper to create a Renderer and MockDriver
fn create_test_renderer_and_driver(
    num_cols: usize,
    num_rows: usize,
    font_width: usize,
    font_height: usize,
) -> (Renderer, MockDriver) {
    let renderer = Renderer::new();
    let display_width_px = (num_cols * font_width) as u16;
    let display_height_px = (num_rows * font_height) as u16;
    let driver = MockDriver::new(font_width, font_height, display_width_px, display_height_px);
    (renderer, driver)
}

// Helper to create a RenderSnapshot for tests
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

fn default_attrs() -> Attributes {
    Attributes::default()
}

#[test]
fn test_render_empty_screen_with_cursor() {
    let font_width = 8;
    let font_height = 16;
    let num_cols = 3;
    let num_rows = 2;

    let (renderer, mut driver) =
        create_test_renderer_and_driver(num_cols, num_rows, font_width, font_height);

    let test_config = Config::default();
    let test_platform_state = driver.get_platform_state(); // Use platform state from mock driver

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
    let render_commands =
        renderer.prepare_render_commands(&snapshot, &test_config, &test_platform_state);
    driver
        .execute_render_commands(render_commands)
        .expect("Execute render commands failed");
    driver.present().expect("Driver present failed");

    let commands = driver.commands();
    assert!(!commands.is_empty(), "Should have drawing commands");

    let expected_bg_fill_line0 = ActualRenderCommand::FillRect {
        x: 0,
        y: 0,
        width: num_cols,
        height: 1,
        color: test_config.colors.background,
        is_selection_bg: false,
    };
    let expected_bg_fill_line1 = ActualRenderCommand::FillRect {
        x: 0,
        y: 1,
        width: num_cols,
        height: 1,
        color: test_config.colors.background,
        is_selection_bg: false,
    };

    assert!(
        commands.contains(&expected_bg_fill_line0),
        "Missing background fill for line 0. Commands: {:?}",
        commands
    );
    assert!(
        commands.contains(&expected_bg_fill_line1),
        "Missing background fill for line 1. Commands: {:?}",
        commands
    );

    let expected_cursor_draw = ActualRenderCommand::DrawTextRun {
        x: 0,
        y: 0,
        text: " ".to_string(),
        fg: test_config.colors.background,
        bg: test_config.colors.foreground,
        flags: AttrFlags::empty(),
        is_selected: false,
    };

    assert!(
        commands.contains(&expected_cursor_draw),
        "Missing cursor draw command. Commands: {:?}",
        commands
    );
    assert_eq!(
        commands.last().unwrap(),
        &ActualRenderCommand::PresentFrame,
        "Last command should be Present"
    );
}

#[test]
fn test_render_simple_text() {
    let font_width = 8;
    let font_height = 16;
    let num_cols = 5;
    let num_rows = 1;

    let (renderer, mut driver) =
        create_test_renderer_and_driver(num_cols, num_rows, font_width, font_height);

    let test_config = Config::default();
    let test_platform_state = driver.get_platform_state();

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

    let lines_data = vec![SnapshotLine {
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
        lines_data,
        cursor_render_state,
        num_cols,
        num_rows,
        Selection::default(),
    );
    let render_commands =
        renderer.prepare_render_commands(&snapshot, &test_config, &test_platform_state);
    driver
        .execute_render_commands(render_commands)
        .expect("Execute render commands failed");
    driver.present().expect("Driver present failed");

    let commands = driver.commands();

    let expected_text_hi = ActualRenderCommand::DrawTextRun {
        x: 0,
        y: 0,
        text: "Hi".to_string(),
        fg: test_config.colors.foreground,
        bg: test_config.colors.background,
        flags: AttrFlags::empty(),
        is_selected: false,
    };
    assert!(
        commands.contains(&expected_text_hi),
        "Missing text 'Hi'. Commands: {:?}",
        commands
    );

    let expected_fill_spaces = ActualRenderCommand::FillRect {
        x: 2,
        y: 0,
        width: num_cols - 2,
        height: 1,
        color: test_config.colors.background,
        is_selection_bg: false,
    };
    assert!(
        commands.contains(&expected_fill_spaces),
        "Missing fill for remaining spaces. Commands: {:?}",
        commands
    );

    let expected_cursor_draw = ActualRenderCommand::DrawTextRun {
        x: 2,
        y: 0,
        text: " ".to_string(),
        fg: test_config.colors.background,
        bg: test_config.colors.foreground,
        flags: AttrFlags::empty(),
        is_selected: false,
    };
    assert!(
        commands.contains(&expected_cursor_draw),
        "Missing cursor draw. Commands: {:?}",
        commands
    );
    assert_eq!(commands.last().unwrap(), &ActualRenderCommand::PresentFrame);
}

#[test]
fn test_dirty_line_processing() {
    let font_width = 8;
    let font_height = 16;
    let num_cols = 3;
    let num_rows = 2;

    let (renderer, mut driver) =
        create_test_renderer_and_driver(num_cols, num_rows, font_width, font_height);

    let test_config = Config::default();
    let test_platform_state = driver.get_platform_state();

    let line0_dirty_cells = vec![
        Glyph::Single(ContentCell {
            c: 'A',
            attr: default_attrs(),
        }),
        Glyph::Single(ContentCell {
            c: 'B',
            attr: default_attrs(),
        }),
        Glyph::Single(ContentCell {
            c: 'C',
            attr: default_attrs(),
        }),
    ];
    let line1_clean_cells = vec![
        Glyph::Single(ContentCell {
            c: 'X',
            attr: default_attrs(),
        }),
        Glyph::Single(ContentCell {
            c: 'Y',
            attr: default_attrs(),
        }),
        Glyph::Single(ContentCell {
            c: 'Z',
            attr: default_attrs(),
        }),
    ];

    let lines_data = vec![
        SnapshotLine {
            is_dirty: true,
            cells: line0_dirty_cells.clone(),
        },
        SnapshotLine {
            is_dirty: false,
            cells: line1_clean_cells.clone(),
        },
    ];

    let cursor_state = Some(CursorRenderState {
        x: 1,
        y: 0,
        shape: CursorShape::Block,
        cell_char_underneath: 'B',
        cell_attributes_underneath: default_attrs(),
    });

    let snapshot = create_test_snapshot(
        lines_data,
        cursor_state,
        num_cols,
        num_rows,
        Selection::default(),
    );
    let render_commands =
        renderer.prepare_render_commands(&snapshot, &test_config, &test_platform_state);
    driver
        .execute_render_commands(render_commands)
        .expect("Execute render commands failed");
    driver.present().expect("Driver present failed");

    let commands = driver.commands();

    let expected_text_abc = ActualRenderCommand::DrawTextRun {
        x: 0,
        y: 0,
        text: "ABC".to_string(),
        fg: test_config.colors.foreground,
        bg: test_config.colors.background,
        flags: AttrFlags::empty(),
        is_selected: false,
    };
    assert!(
        commands.contains(&expected_text_abc),
        "Text 'ABC' from dirty line 0 should be drawn. Commands: {:?}",
        commands
    );

    let non_expected_text_xyz = ActualRenderCommand::DrawTextRun {
        x: 0,
        y: 1,
        text: "XYZ".to_string(),
        fg: test_config.colors.foreground,
        bg: test_config.colors.background,
        flags: AttrFlags::empty(),
        is_selected: false,
    };
    assert!(
        !commands.contains(&non_expected_text_xyz),
        "Text 'XYZ' from clean line 1 should NOT be drawn. Commands: {:?}",
        commands
    );

    let fill_rect_line1_found = commands.iter().any(|cmd| match cmd {
        ActualRenderCommand::FillRect { y, height, .. } => *y == 1 && *height == 1,
        _ => false,
    });
    assert!(
        !fill_rect_line1_found,
        "No FillRect command should exist for clean line 1. Commands: {:?}",
        commands
    );

    let expected_cursor_draw = ActualRenderCommand::DrawTextRun {
        x: 1,
        y: 0,
        text: "B".to_string(),
        fg: test_config.colors.background,
        bg: test_config.colors.foreground,
        flags: AttrFlags::empty(),
        is_selected: false,
    };
    assert!(
        commands.contains(&expected_cursor_draw),
        "Missing cursor draw. Commands: {:?}",
        commands
    );

    assert_eq!(commands.last().unwrap(), &ActualRenderCommand::PresentFrame);
}

#[test]
fn test_render_wide_character() {
    let font_width = 8;
    let font_height = 16;
    let num_cols = 5;
    let num_rows = 1;

    let (renderer, mut driver) =
        create_test_renderer_and_driver(num_cols, num_rows, font_width, font_height);

    let test_config = Config::default();
    let test_platform_state = driver.get_platform_state();

    let mut line_cells = vec![
        Glyph::Single(ContentCell {
            c: ' ',
            attr: default_attrs()
        });
        num_cols
    ];
    line_cells[0] = Glyph::WidePrimary(ContentCell {
        c: '世',
        attr: default_attrs(),
    });
    line_cells[1] = Glyph::WideSpacer {
        primary_column_on_line: 0,
    };

    let lines_data = vec![SnapshotLine {
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
        lines_data,
        cursor_render_state,
        num_cols,
        num_rows,
        Selection::default(),
    );
    let render_commands =
        renderer.prepare_render_commands(&snapshot, &test_config, &test_platform_state);
    driver
        .execute_render_commands(render_commands)
        .expect("Execute render commands failed");
    driver.present().expect("Driver present failed");

    let commands = driver.commands();

    let expected_text_wide = ActualRenderCommand::DrawTextRun {
        x: 0,
        y: 0,
        text: "世".to_string(),
        fg: test_config.colors.foreground,
        bg: test_config.colors.background,
        flags: AttrFlags::empty(),
        is_selected: false,
    };
    assert!(
        commands.contains(&expected_text_wide),
        "Missing wide char text. Commands: {:?}",
        commands
    );

    let expected_fill_spaces = ActualRenderCommand::FillRect {
        x: 2,
        y: 0,
        width: num_cols - 2,
        height: 1,
        color: test_config.colors.background,
        is_selection_bg: false,
    };
    assert!(
        commands.contains(&expected_fill_spaces),
        "Missing fill for remaining spaces. Commands: {:?}",
        commands
    );

    let expected_cursor_draw = ActualRenderCommand::DrawTextRun {
        x: 2,
        y: 0,
        text: " ".to_string(),
        fg: test_config.colors.background,
        bg: test_config.colors.foreground,
        flags: AttrFlags::empty(),
        is_selected: false,
    };
    assert!(
        commands.contains(&expected_cursor_draw),
        "Missing cursor draw. Commands: {:?}",
        commands
    );
    assert_eq!(commands.last().unwrap(), &ActualRenderCommand::PresentFrame);
}
