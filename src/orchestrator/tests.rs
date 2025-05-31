// src/orchestrator/tests.rs
#![cfg(test)]

// Updated imports based on current crate structure
use crate::color::Color;
use crate::platform::backends::{CellCoords, CellRect, Driver, TextRunStyle, FocusState}; // Driver was RenderAdapter, TextRunStyle was ResolvedCellAttrs
use crate::platform::backends::x11::window::CursorVisibility; // Import for MockDriver
use crate::renderer::{Renderer, RENDERER_DEFAULT_BG, RENDERER_DEFAULT_FG}; // Rgb is now a variant Color::Rgb(...). Colors struct removed.
                                                                           // FontDesc is now FontConfig. Using ColorScheme for palette.
use crate::glyph::{AttrFlags, Attributes, Glyph}; // Cell -> Glyph, CellAttrs -> Attributes, Flags -> AttrFlags
use crate::term::{
    CursorRenderState, CursorShape, RenderSnapshot, SelectionRenderState, SnapshotLine,
}; // CursorStyle removed, SelectionRange -> SelectionRenderState

use std::sync::Mutex;
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
    commands: Mutex<Vec<DrawCommand>>,
    font_width: usize,
    font_height: usize,
    display_width_px: u16,
    display_height_px: u16,
    is_focused: bool,
    is_cursor_visible: bool,
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
            is_focused: true,
            is_cursor_visible: true,
        }
    }

    fn commands(&self) -> Vec<DrawCommand> {
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
    fn process_events(&mut self) -> anyhow::Result<Vec<crate::platform::backends::BackendEvent>> {
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
            .push(DrawCommand::ClearAll(bg));
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
            .push(DrawCommand::DrawTextRun {
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
            .push(DrawCommand::FillRect { rect, color });
        Ok(())
    }

    fn present(&mut self) -> anyhow::Result<()> {
        self.commands.lock().unwrap().push(DrawCommand::Present);
        Ok(())
    }

    fn set_title(&mut self, _title: &str) {}
    fn bell(&mut self) {}
    fn set_cursor_visibility(&mut self, visibility: CursorVisibility) {
        self.is_cursor_visible = match visibility {
            CursorVisibility::Shown => true,
            CursorVisibility::Hidden => false,
        };
    }
    fn set_focus(&mut self, focus_state: FocusState) {
        self.is_focused = match focus_state {
            FocusState::Focused => true,
            FocusState::Unfocused => false,
        };
    }
    fn cleanup(&mut self) -> anyhow::Result<()> {
        Ok(())
    }
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

    let renderer = Renderer::new(); // Renderer::new takes no args now
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
        shape: CursorShape::Block, // Using Block shape
        cell_char_underneath: ' ',
        cell_attributes_underneath: default_attrs(),
    });

    let snapshot = create_test_snapshot(lines, cursor_render_state, num_cols, num_rows, None);
    renderer
        .draw(snapshot, &mut adapter)
        .expect("Render draw failed");

    let commands = adapter.commands();
    assert!(!commands.is_empty(), "Should have drawing commands");

    // Renderer now draws dirty lines. If all are dirty and empty:
    // Expect FillRect for each cell's background, then cursor, then Present.
    // The new renderer is more granular. It will try to coalesce.
    // For an empty screen, it might issue FillRect for entire lines if attributes are consistent.

    let expected_bg_fill_for_line0 = DrawCommand::FillRect {
        rect: CellRect {
            x: 0,
            y: 0,
            width: num_cols,
            height: 1,
        },
        color: RENDERER_DEFAULT_BG,
    };
    let expected_bg_fill_for_line1 = DrawCommand::FillRect {
        rect: CellRect {
            x: 0,
            y: 1,
            width: num_cols,
            height: 1,
        },
        color: RENDERER_DEFAULT_BG,
    };

    assert!(
        commands.contains(&expected_bg_fill_for_line0),
        "Missing background fill for line 0"
    );
    assert!(
        commands.contains(&expected_bg_fill_for_line1),
        "Missing background fill for line 1"
    );

    // Cursor drawing: The renderer draws the character under the cursor with fg/bg swapped.
    let cursor_draw_style = TextRunStyle {
        fg: RENDERER_DEFAULT_BG, // Original BG becomes FG
        bg: RENDERER_DEFAULT_FG, // Original FG becomes BG
        flags: AttrFlags::empty(),
    };
    let expected_cursor_draw = DrawCommand::DrawTextRun {
        coords: CellCoords { x: 0, y: 0 },
        text: " ".to_string(), // Character under cursor
        style: cursor_draw_style,
    };

    assert!(
        commands.contains(&expected_cursor_draw),
        "Missing cursor draw command. Commands: {:?}",
        commands
    );
    assert_eq!(
        commands.last().unwrap(),
        &DrawCommand::Present,
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
        x: 2, // Cursor after "Hi"
        y: 0,
        shape: CursorShape::Block,
        cell_char_underneath: ' ',
        cell_attributes_underneath: default_attrs(),
    });

    let snapshot = create_test_snapshot(lines, cursor_render_state, num_cols, num_rows, None);
    renderer
        .draw(snapshot, &mut adapter)
        .expect("Render draw failed");

    let commands = adapter.commands();

    // Expected: DrawTextRun for "H", DrawTextRun for "i", FillRect for remaining spaces, Cursor, Present
    // Or, a coalesced "Hi"
    let text_style = TextRunStyle {
        fg: RENDERER_DEFAULT_FG,
        bg: RENDERER_DEFAULT_BG,
        flags: AttrFlags::empty(),
    };

    let expected_text_hi = DrawCommand::DrawTextRun {
        coords: CellCoords { x: 0, y: 0 },
        text: "Hi".to_string(), // Renderer should coalesce text with same attributes
        style: text_style,
    };
    assert!(
        commands.contains(&expected_text_hi),
        "Missing text 'Hi'. Commands: {:?}",
        commands
    );

    let expected_fill_spaces = DrawCommand::FillRect {
        rect: CellRect {
            x: 2,
            y: 0,
            width: num_cols - 2,
            height: 1,
        },
        color: RENDERER_DEFAULT_BG,
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
    let expected_cursor_draw = DrawCommand::DrawTextRun {
        coords: CellCoords { x: 2, y: 0 },
        text: " ".to_string(),
        style: cursor_draw_style,
    };
    assert!(
        commands.contains(&expected_cursor_draw),
        "Missing cursor draw. Commands: {:?}",
        commands
    );
    assert_eq!(commands.last().unwrap(), &DrawCommand::Present);
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

    // Frame 1: line 0 is dirty, line 1 is dirty (initial state)
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
        None,
    );
    renderer
        .draw(snapshot1, &mut adapter)
        .expect("Render draw failed");
    adapter.clear_commands();

    // Frame 2: Only line 1 is dirty (e.g. cursor moved to line 1, or line 1 content changed)
    // Content of line 0 is 'A', ' ', ' '
    // Content of line 1 is 'B', 'B', 'B'
    // Let's say cursor moved to line 1, making line 0 and line 1 dirty for cursor redraw.
    // To test *content* dirtiness, let's make line 0 not dirty.
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
    }; // Same content

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
    }; // Changed content

    let lines_frame2 = vec![
        SnapshotLine {
            is_dirty: false,
            cells: line0_cells_frame2,
        }, // Line 0 NOT dirty
        SnapshotLine {
            is_dirty: true,
            cells: line1_cells_frame2,
        }, // Line 1 IS dirty
    ];
    // Cursor is now on line 1, which is dirty. Line 0 (old cursor pos) should also be marked dirty by snapshot source if cursor moved.
    // For this test, assume snapshot correctly marks lines dirty based on content *and* cursor moves.
    // If cursor moves from (0,0) to (0,1), both lines become dirty in the snapshot.
    // Let's simulate only line 1 content changing, and cursor remains on line 1.
    let cursor_state_frame2 = Some(CursorRenderState {
        x: 0,
        y: 1,
        shape: CursorShape::Block,
        cell_char_underneath: 'C',
        cell_attributes_underneath: default_attrs(),
    });

    let snapshot2 =
        create_test_snapshot(lines_frame2, cursor_state_frame2, num_cols, num_rows, None);
    renderer
        .draw(snapshot2, &mut adapter)
        .expect("Render draw failed");
    let commands_frame2 = adapter.commands();

    let text_style = TextRunStyle {
        fg: RENDERER_DEFAULT_FG,
        bg: RENDERER_DEFAULT_BG,
        flags: AttrFlags::empty(),
    };

    // Check that 'A' from line 0 was NOT redrawn
    let draw_a_cmd = DrawCommand::DrawTextRun {
        coords: CellCoords { x: 0, y: 0 },
        text: "A".to_string(),
        style: text_style,
    };
    assert!(
        !commands_frame2.contains(&draw_a_cmd),
        "Text 'A' from non-dirty line 0 should not be redrawn. Commands: {:?}",
        commands_frame2
    );

    // Check that 'C' from line 1 WAS redrawn
    let draw_c_cmd = DrawCommand::DrawTextRun {
        coords: CellCoords { x: 0, y: 1 },
        text: "C".to_string(),
        style: text_style,
    };
    assert!(
        commands_frame2.contains(&draw_c_cmd),
        "Text 'C' from dirty line 1 should be redrawn. Commands: {:?}",
        commands_frame2
    );

    // Check cursor on line 1
    let cursor_draw_style = TextRunStyle {
        fg: RENDERER_DEFAULT_BG,
        bg: RENDERER_DEFAULT_FG,
        flags: AttrFlags::empty(),
    };
    let expected_cursor_draw = DrawCommand::DrawTextRun {
        coords: CellCoords { x: 0, y: 1 },
        text: "C".to_string(), // Character under cursor
        style: cursor_draw_style,
    };
    assert!(
        commands_frame2.contains(&expected_cursor_draw),
        "Missing cursor draw on line 1. Commands: {:?}",
        commands_frame2
    );

    assert_eq!(commands_frame2.last().unwrap(), &DrawCommand::Present);
}

// The concept of "full redraw on font change" is now handled by the AppOrchestrator
// invalidating the terminal state (marking lines dirty) and requesting a new snapshot.
// The Renderer itself is stateless regarding font changes affecting previous frames.
// So, a direct test for `renderer.update_font()` causing a full redraw isn't applicable
// in the same way. The test would need to be at the orchestrator level or simulate
// the orchestrator's behavior of dirtying lines.

// For now, this test is removed as `Renderer` no longer has `update_font`.
// #[test]
// fn test_full_redraw_on_font_change() { ... }
