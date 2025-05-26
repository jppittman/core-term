// src/renderer/tests.rs
#![cfg(test)]

// Imports from the main crate
use crate::backends::{CellCoords, CellRect, Driver, TextRunStyle}; // Driver was RenderAdapter
use crate::color::Color; // Rgb is now Color::Rgb, Colors struct removed
// FontDesc is now FontConfig
use crate::glyph::{AttrFlags, Attributes, Glyph}; // Cell -> Glyph, CellAttrs -> Attributes, Flags -> AttrFlags
use crate::renderer::{RENDERER_DEFAULT_BG, RENDERER_DEFAULT_FG, Renderer};
use crate::term::{
    CursorRenderState, CursorShape, RenderSnapshot, SelectionRenderState, SnapshotLine,
};

use std::sync::Mutex;

// Mocking structures for testing the renderer's interaction with a Driver.
#[derive(Debug, Clone, PartialEq)]
enum MockDrawCommand {
    ClearAll {
        bg: Color,
    },
    FillRect {
        rect: CellRect,
        color: Color,
    },
    DrawTextRun {
        coords: CellCoords,
        text: String,
        style: TextRunStyle,
    },
    Present,
    // MockDriver specific commands, if any, or just stick to Driver trait
}

// Mock Driver implementation
struct MockDriver {
    commands: Mutex<Vec<MockDrawCommand>>,
    font_width: usize,
    font_height: usize,
    display_width_px: u16,
    display_height_px: u16,
    // Add other necessary fields like focus state if Driver expects them
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

    fn commands(&self) -> Vec<MockDrawCommand> {
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
        // For mock, provide some defaults. Real drivers would initialize properly.
        Ok(MockDriver::new(8, 16, 80 * 8, 24 * 16))
    }

    fn get_event_fd(&self) -> Option<std::os::unix::io::RawFd> {
        None
    }
    fn process_events(&mut self) -> anyhow::Result<Vec<crate::backends::BackendEvent>> {
        Ok(Vec::new())
    }

    fn get_font_dimensions(&self) -> (usize, usize) {
        (self.font_width, self.font_height)
    }
    fn get_display_dimensions_pixels(&self) -> (u16, u16) {
        (self.display_width_px, self.display_height_px)
    }

    fn clear_all(&mut self, bg: Color) -> anyhow::Result<()> {
        self.commands
            .lock()
            .unwrap()
            .push(MockDrawCommand::ClearAll { bg });
        Ok(())
    }

    fn draw_text_run(
        &mut self,
        coords: CellCoords,
        text: &str,
        style: TextRunStyle,
    ) -> anyhow::Result<()> {
        self.commands
            .lock()
            .unwrap()
            .push(MockDrawCommand::DrawTextRun {
                coords,
                text: text.to_string(),
                style,
            });
        Ok(())
    }

    fn fill_rect(&mut self, rect: CellRect, color: Color) -> anyhow::Result<()> {
        self.commands
            .lock()
            .unwrap()
            .push(MockDrawCommand::FillRect { rect, color });
        Ok(())
    }

    fn present(&mut self) -> anyhow::Result<()> {
        self.commands.lock().unwrap().push(MockDrawCommand::Present);
        Ok(())
    }

    fn set_title(&mut self, _title: &str) {}
    fn bell(&mut self) {}
    fn set_cursor_visibility(&mut self, _visible: bool) {}
    fn set_focus(&mut self, _focused: bool) {}
    fn cleanup(&mut self) -> anyhow::Result<()> {
        Ok(())
    }
}

// Helper to create a Renderer and MockDriver
fn create_test_renderer_and_driver(
    num_cols: usize,
    num_rows: usize,
    font_width: usize,
    font_height: usize,
) -> (Renderer, MockDriver) {
    let renderer = Renderer::new(); // Renderer::new takes no args
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
    selection_state: Option<SelectionRenderState>,
) -> RenderSnapshot {
    RenderSnapshot {
        dimensions: (num_cols, num_rows),
        lines: lines_data,
        cursor_state,
        selection_state,
    }
}

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

    let (renderer, mut driver) =
        create_test_renderer_and_driver(num_cols, num_rows, font_width, font_height);

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

    let snapshot = create_test_snapshot(lines, cursor_render_state, num_cols, num_rows, None);
    renderer
        .draw(snapshot, &mut driver)
        .expect("Renderer draw failed");

    let commands = driver.commands();
    assert!(!commands.is_empty(), "Should have drawing commands");

    // Expected: FillRect for each dirty line's background, then cursor, then Present.
    // The renderer should coalesce fills if possible.
    let expected_bg_fill_line0 = MockDrawCommand::FillRect {
        rect: CellRect {
            x: 0,
            y: 0,
            width: num_cols,
            height: 1,
        },
        color: RENDERER_DEFAULT_BG,
    };
    let expected_bg_fill_line1 = MockDrawCommand::FillRect {
        rect: CellRect {
            x: 0,
            y: 1,
            width: num_cols,
            height: 1,
        },
        color: RENDERER_DEFAULT_BG,
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

    // Cursor drawing: The renderer draws the character under the cursor with fg/bg swapped.
    let cursor_style = TextRunStyle {
        fg: RENDERER_DEFAULT_BG,   // Original BG becomes FG for cursor char
        bg: RENDERER_DEFAULT_FG,   // Original FG becomes BG for cursor cell
        flags: AttrFlags::empty(), // Assuming no special flags for cursor text itself
    };
    let expected_cursor_draw = MockDrawCommand::DrawTextRun {
        coords: CellCoords { x: 0, y: 0 },
        text: " ".to_string(), // Character under cursor
        style: cursor_style,
    };

    assert!(
        commands.contains(&expected_cursor_draw),
        "Missing cursor draw command. Commands: {:?}",
        commands
    );
    assert_eq!(
        commands.last().unwrap(),
        &MockDrawCommand::Present,
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

    let lines_data = vec![SnapshotLine {
        is_dirty: true,
        cells: line_cells,
    }];

    let cursor_render_state = Some(CursorRenderState {
        x: 2, // Cursor after "Hi"
        y: 0,
        shape: CursorShape::Block,
        cell_char_underneath: ' ', // Char under cursor at (0,2) is space
        cell_attributes_underneath: default_attrs(),
    });

    let snapshot = create_test_snapshot(lines_data, cursor_render_state, num_cols, num_rows, None);
    renderer
        .draw(snapshot, &mut driver)
        .expect("Renderer draw failed");

    let commands = driver.commands();

    let default_text_style = TextRunStyle {
        fg: RENDERER_DEFAULT_FG,
        bg: RENDERER_DEFAULT_BG,
        flags: AttrFlags::empty(),
    };

    // Renderer should coalesce "H" and "i" into one DrawTextRun
    let expected_text_hi = MockDrawCommand::DrawTextRun {
        coords: CellCoords { x: 0, y: 0 },
        text: "Hi".to_string(),
        style: default_text_style,
    };
    assert!(
        commands.contains(&expected_text_hi),
        "Missing text 'Hi'. Commands: {:?}",
        commands
    );

    // Renderer should fill the rest of the line (3 cells: "   ")
    let expected_fill_spaces = MockDrawCommand::FillRect {
        rect: CellRect {
            x: 2, // Starts after "Hi"
            y: 0,
            width: num_cols - 2, // 3 cells
            height: 1,
        },
        color: RENDERER_DEFAULT_BG,
    };
    assert!(
        commands.contains(&expected_fill_spaces),
        "Missing fill for remaining spaces. Commands: {:?}",
        commands
    );

    let cursor_style = TextRunStyle {
        fg: RENDERER_DEFAULT_BG,
        bg: RENDERER_DEFAULT_FG,
        flags: AttrFlags::empty(),
    };
    let expected_cursor_draw = MockDrawCommand::DrawTextRun {
        coords: CellCoords { x: 2, y: 0 }, // Cursor at (0,2)
        text: " ".to_string(),             // Char under cursor is space
        style: cursor_style,
    };
    assert!(
        commands.contains(&expected_cursor_draw),
        "Missing cursor draw. Commands: {:?}",
        commands
    );
    assert_eq!(commands.last().unwrap(), &MockDrawCommand::Present);
}

#[test]
fn test_dirty_line_processing() {
    let font_width = 8;
    let font_height = 16;
    let num_cols = 3;
    let num_rows = 2;

    let (renderer, mut driver) =
        create_test_renderer_and_driver(num_cols, num_rows, font_width, font_height);

    let line0_dirty_cells = vec![
        Glyph {
            c: 'A',
            attr: default_attrs(),
        },
        Glyph {
            c: 'B',
            attr: default_attrs(),
        },
        Glyph {
            c: 'C',
            attr: default_attrs(),
        },
    ];
    let line1_clean_cells = vec![
        Glyph {
            c: 'X',
            attr: default_attrs(),
        },
        Glyph {
            c: 'Y',
            attr: default_attrs(),
        },
        Glyph {
            c: 'Z',
            attr: default_attrs(),
        },
    ];

    let lines_data = vec![
        SnapshotLine {
            is_dirty: true,
            cells: line0_dirty_cells.clone(),
        },
        SnapshotLine {
            is_dirty: false, // Line 1 is NOT dirty
            cells: line1_clean_cells.clone(),
        },
    ];

    // Cursor on the dirty line
    let cursor_state = Some(CursorRenderState {
        x: 1,
        y: 0,
        shape: CursorShape::Block,
        cell_char_underneath: 'B',
        cell_attributes_underneath: default_attrs(),
    });

    let snapshot = create_test_snapshot(lines_data, cursor_state, num_cols, num_rows, None);
    renderer
        .draw(snapshot, &mut driver)
        .expect("Renderer draw failed");

    let commands = driver.commands();

    let default_text_style = TextRunStyle {
        fg: RENDERER_DEFAULT_FG,
        bg: RENDERER_DEFAULT_BG,
        flags: AttrFlags::empty(),
    };

    // Check that "ABC" from dirty line 0 was drawn
    let expected_text_abc = MockDrawCommand::DrawTextRun {
        coords: CellCoords { x: 0, y: 0 },
        text: "ABC".to_string(),
        style: default_text_style,
    };
    assert!(
        commands.contains(&expected_text_abc),
        "Text 'ABC' from dirty line 0 should be drawn. Commands: {:?}",
        commands
    );

    // Check that "XYZ" from clean line 1 was NOT drawn as a text run
    let non_expected_text_xyz = MockDrawCommand::DrawTextRun {
        coords: CellCoords { x: 0, y: 1 },
        text: "XYZ".to_string(),
        style: default_text_style,
    };
    assert!(
        !commands.contains(&non_expected_text_xyz),
        "Text 'XYZ' from clean line 1 should NOT be drawn. Commands: {:?}",
        commands
    );

    // Check if there's any FillRect for line 1 (it shouldn't be, as it's not dirty)
    let fill_rect_line1_found = commands.iter().any(|cmd| match cmd {
        MockDrawCommand::FillRect { rect, .. } => rect.y == 1,
        _ => false,
    });
    assert!(
        !fill_rect_line1_found,
        "No FillRect command should exist for clean line 1. Commands: {:?}",
        commands
    );

    let cursor_style = TextRunStyle {
        fg: RENDERER_DEFAULT_BG,
        bg: RENDERER_DEFAULT_FG,
        flags: AttrFlags::empty(),
    };
    let expected_cursor_draw = MockDrawCommand::DrawTextRun {
        coords: CellCoords { x: 1, y: 0 }, // Cursor at (0,1) over 'B'
        text: "B".to_string(),
        style: cursor_style,
    };
    assert!(
        commands.contains(&expected_cursor_draw),
        "Missing cursor draw. Commands: {:?}",
        commands
    );

    assert_eq!(commands.last().unwrap(), &MockDrawCommand::Present);
}
