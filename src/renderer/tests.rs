// src/renderer/tests.rs
#![cfg(test)]
use super::*; // Imports Renderer, RenderAdapter, Rgb etc.
use crate::color::{ColorPalette, Colors};
use crate::config::FontDesc;
use crate::glyph::{CellAttrs, Flags, ResolvedCellAttrs, ResolvedColor}; // Ensure these are pub and constructible/inspectable
use crate::term::cell::Cell;
use crate::term::cursor::{Cursor, CursorShape, CursorStyle};
use crate::term::snapshot::{RenderSnapshot, SelectionRange}; // RenderSnapshot is key
use std::rc::Rc;
use std::sync::Mutex;

// MockRenderAdapter captures drawing commands for assertion.
#[derive(Debug, Clone, PartialEq)]
enum DrawCommand {
    Clear(Rgb),
    Rect {
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        color: Rgb,
    },
    Text {
        text: String,
        x: f32,
        y: f32,
        fg: Rgb,
        bg: Rgb,
        attrs: ResolvedCellAttrs, // Assuming ResolvedCellAttrs is inspectable (e.g. pub fields or getters)
        is_wide: bool,
    },
    Present,
}

// Helper to compare f32s with a tolerance
fn float_eq(a: f32, b: f32) -> bool {
    (a - b).abs() < 0.001
}

// Custom match for DrawCommand::Rect to handle f32 comparisons
fn match_rect_command(cmd: &DrawCommand, ex: f32, ey: f32, ew: f32, eh: f32, ecolor: Rgb) -> bool {
    if let DrawCommand::Rect {
        x,
        y,
        width,
        height,
        color,
    } = cmd
    {
        float_eq(*x, ex)
            && float_eq(*y, ey)
            && float_eq(*width, ew)
            && float_eq(*height, eh)
            && *color == ecolor
    } else {
        false
    }
}

// Custom match for DrawCommand::Text
fn match_text_command(
    cmd: &DrawCommand,
    etext: &str,
    ex: f32,
    ey: f32,
    efg: Rgb,
    ebg: Rgb,
    eattrs: ResolvedCellAttrs,
    eis_wide: bool,
) -> bool {
    if let DrawCommand::Text {
        text,
        x,
        y,
        fg,
        bg,
        attrs,
        is_wide,
    } = cmd
    {
        text == etext
            && float_eq(*x, ex)
            && float_eq(*y, ey)
            && *fg == efg
            && *bg == ebg
            && *attrs == eattrs // ResolvedCellAttrs needs to be PartialEq
            && *is_wide == eis_wide
    } else {
        false
    }
}

struct MockRenderAdapter {
    commands: Mutex<Vec<DrawCommand>>,
    char_width: f32,
    char_height: f32,
    // Store expected dimensions to ensure renderer uses them
    expected_screen_width: u32,
    expected_screen_height: u32,
}

impl MockRenderAdapter {
    fn new(char_width: f32, char_height: f32, screen_width: u32, screen_height: u32) -> Self {
        Self {
            commands: Mutex::new(Vec::new()),
            char_width,
            char_height,
            expected_screen_width: screen_width,
            expected_screen_height: screen_height,
        }
    }

    fn commands(&self) -> Vec<DrawCommand> {
        self.commands.lock().unwrap().clone() // Clone for inspection
    }

    fn clear_commands(&self) {
        self.commands.lock().unwrap().clear();
    }
}

impl RenderAdapter for MockRenderAdapter {
    fn clear(&mut self, color: Rgb) {
        self.commands
            .lock()
            .unwrap()
            .push(DrawCommand::Clear(color));
    }

    fn draw_rect(&mut self, x: f32, y: f32, width: f32, height: f32, color: Rgb) {
        self.commands.lock().unwrap().push(DrawCommand::Rect {
            x,
            y,
            width,
            height,
            color,
        });
    }

    fn draw_text(
        &mut self,
        text: &str,
        x: f32,
        y: f32,
        fg_color: Rgb,
        bg_color: Rgb,
        attrs: ResolvedCellAttrs,
        is_wide: bool,
    ) {
        self.commands.lock().unwrap().push(DrawCommand::Text {
            text: text.to_string(),
            x,
            y,
            fg: fg_color,
            bg: bg_color,
            attrs,
            is_wide,
        });
    }

    fn present(&mut self) {
        self.commands.lock().unwrap().push(DrawCommand::Present);
    }

    fn get_char_size(&self) -> (f32, f32) {
        (self.char_width, self.char_height)
    }

    fn get_screen_dimensions_pixels(&self) -> (u32, u32) {
        (self.expected_screen_width, self.expected_screen_height)
    }

    fn set_font_size(&mut self, _size: f32) -> Result<(), String> {
        // In a real scenario, this might affect get_char_size.
        // For mock, we can assume it's handled or test font updates separately.
        Ok(())
    }
}

// Helper to create a Renderer instance and its associated Colors struct for testing.
fn create_test_renderer_and_colors(
    num_cols: usize,
    num_rows: usize,
    char_width: f32,
    char_height: f32,
) -> (Renderer, Rc<Colors>, u32, u32) {
    let colors = Rc::new(Colors::default_palleted()); // Using a default palette
    let font_desc = FontDesc::new("monospace".to_string(), 15.0); // A default font
    let screen_width_pixels = (num_cols as f32 * char_width) as u32;
    let screen_height_pixels = (num_rows as f32 * char_height) as u32;
    let dpr = 1.0;

    let renderer = Renderer::new(
        screen_width_pixels,
        screen_height_pixels,
        dpr,
        font_desc,
        colors.clone(),
    )
    .expect("Failed to create renderer for test");
    (renderer, colors, screen_width_pixels, screen_height_pixels)
}

// Helper to create a RenderSnapshot for tests.
// Note: `cells` and `cursor` liftime 'a must outlive the snapshot.
// In tests, these are typically stack-allocated and live for the duration of the test.
fn create_test_snapshot<'a>(
    cells: &'a [Cell],
    cursor: &'a Cursor,
    num_cols: usize,
    num_rows: usize,
    focused: bool,
    selection: Option<SelectionRange>,
) -> RenderSnapshot<'a> {
    RenderSnapshot::new(
        cells,
        cursor,
        "Test Window Title".to_string(), // window_title is String
        num_cols,
        num_rows,
        0,     // scrollback_offset
        false, // is_alt_screen
        selection,
        focused,
    )
}

#[test]
fn test_render_empty_screen() {
    let char_width = 10.0;
    let char_height = 20.0;
    let num_cols = 3;
    let num_rows = 2;

    let (mut renderer, test_colors, screen_w, screen_h) =
        create_test_renderer_and_colors(num_cols, num_rows, char_width, char_height);
    let mut adapter = MockRenderAdapter::new(char_width, char_height, screen_w, screen_h);

    let cells = vec![Cell::default(); num_cols * num_rows];
    let cursor = Cursor::default(); // Default cursor at (0,0), block style, visible

    let snapshot = create_test_snapshot(&cells, &cursor, num_cols, num_rows, true, None);
    renderer.render_frame(&snapshot, &mut adapter);

    let commands = adapter.commands();
    assert!(!commands.is_empty(), "Should have drawing commands");

    let expected_bg = test_colors.default_bg();
    let cursor_fg_color = test_colors.cursor_fg(); // Or however cursor color is determined by Colors

    // 1. Expect a clear command
    assert!(
        matches!(commands[0], DrawCommand::Clear(color) if color == test_colors.background()),
        "First command should be Clear with background color"
    );

    // 2. Expect background rects for all cells (renderer might optimize, this depends on its strategy)
    // This test assumes simple renderer draws bg for all cells if they are default.
    let mut bg_rects_found = 0;
    for r in 0..num_rows {
        for c in 0..num_cols {
            let x = c as f32 * char_width;
            let y = r as f32 * char_height;
            if commands
                .iter()
                .any(|cmd| match_rect_command(cmd, x, y, char_width, char_height, expected_bg))
            {
                bg_rects_found += 1;
            }
        }
    }
    // This assertion depends heavily on the renderer's optimization strategy for default cells.
    // If it optimizes by not drawing individual rects when clear covers it, this needs adjustment.
    // For now, let's assume it draws them.
    assert_eq!(
        bg_rects_found,
        num_cols * num_rows,
        "Should draw background rects for all cells"
    );

    // 3. Expect cursor draw command (default is Block at 0,0)
    // Cursor rendering depends on its style and if the window is focused.
    // Default cursor is often a block.
    let cursor_x = cursor.col as f32 * char_width;
    let cursor_y = cursor.row as f32 * char_height;

    // Check for a rect that could be the block cursor
    assert!(
        commands.iter().any(|cmd| {
            match_rect_command(
                cmd,
                cursor_x,
                cursor_y,
                char_width,
                char_height,
                cursor_fg_color,
            )
        }),
        "Missing block cursor draw command at (0,0)"
    );

    // 4. Expect a Present command at the end
    assert!(
        matches!(commands.last().unwrap(), DrawCommand::Present),
        "Last command should be Present"
    );
}

#[test]
fn test_render_simple_text() {
    let char_width = 10.0;
    let char_height = 20.0;
    let num_cols = 5;
    let num_rows = 1;

    let (mut renderer, test_colors, screen_w, screen_h) =
        create_test_renderer_and_colors(num_cols, num_rows, char_width, char_height);
    let mut adapter = MockRenderAdapter::new(char_width, char_height, screen_w, screen_h);

    let mut cells = vec![Cell::default(); num_cols * num_rows];
    cells[0] = Cell {
        char: 'H',
        attrs: CellAttrs::default(),
        flags: Flags::empty(),
        width: 1,
    };
    cells[1] = Cell {
        char: 'i',
        attrs: CellAttrs::default(),
        flags: Flags::empty(),
        width: 1,
    };

    let cursor = Cursor {
        row: 0,
        col: 2,
        style: CursorStyle::default(),
        ..Default::default()
    }; // Cursor after "Hi"

    let snapshot = create_test_snapshot(&cells, &cursor, num_cols, num_rows, true, None);
    renderer.render_frame(&snapshot, &mut adapter);

    let commands = adapter.commands();

    let default_fg = test_colors.default_fg();
    let default_bg = test_colors.default_bg();
    let expected_attrs = ResolvedCellAttrs {
        // Assuming default resolution
        fg: ResolvedColor::from(default_fg),
        bg: ResolvedColor::from(default_bg),
        flags: Flags::empty(),
        hyperlink: None, // Assuming no hyperlink by default
    };

    // Check for text "H"
    assert!(
        commands.iter().any(|cmd| match_text_command(
            cmd,
            "H",
            0.0 * char_width,
            0.0 * char_height,
            default_fg,
            default_bg,
            expected_attrs,
            false // is_wide
        )),
        "Missing text 'H'"
    );

    // Check for text "i"
    assert!(
        commands.iter().any(|cmd| match_text_command(
            cmd,
            "i",
            1.0 * char_width,
            0.0 * char_height,
            default_fg,
            default_bg,
            expected_attrs,
            false // is_wide
        )),
        "Missing text 'i'"
    );
}

#[test]
fn test_damage_tracking_no_change() {
    let char_width = 10.0;
    let char_height = 20.0;
    let num_cols = 3;
    let num_rows = 1;

    let (mut renderer, _test_colors, screen_w, screen_h) =
        create_test_renderer_and_colors(num_cols, num_rows, char_width, char_height);
    let mut adapter = MockRenderAdapter::new(char_width, char_height, screen_w, screen_h);

    let mut cells = vec![Cell::default(); num_cols];
    cells[0] = Cell {
        char: 'A',
        ..Default::default()
    };
    let cursor = Cursor::default();

    // Frame 1
    let snapshot1 = create_test_snapshot(&cells, &cursor, num_cols, num_rows, true, None);
    renderer.render_frame(&snapshot1, &mut adapter);
    let commands_frame1_count = adapter.commands().len();
    adapter.clear_commands();

    // Frame 2 (identical snapshot content, though a new snapshot instance)
    // Create a new cells vec that is identical for the second snapshot to ensure 'static lifetime if needed by to_owned
    let cells_clone = cells.clone();
    let snapshot2 = create_test_snapshot(&cells_clone, &cursor, num_cols, num_rows, true, None);
    renderer.render_frame(&snapshot2, &mut adapter);
    let commands_frame2 = adapter.commands();

    // With damage tracking, if nothing changed, only essential commands (like Present, maybe cursor if it blinks) should be issued.
    // A very basic check: fewer commands than full redraw.
    // A more robust check would verify that no cell redraw commands were issued for 'A'.
    // For this test, let's expect only Clear (if background optimization is not perfect), Cursor, and Present.
    // Or, if cursor also didn't change and no blink, potentially only Present.
    // This highly depends on the renderer's damage optimization logic.
    // A simple expectation: fewer commands than the initial full render.
    // And no text command for 'A' if it wasn't damaged.

    let default_fg = _test_colors.default_fg();
    let default_bg = _test_colors.default_bg();
    let expected_attrs = ResolvedCellAttrs {
        fg: ResolvedColor::from(default_fg),
        bg: ResolvedColor::from(default_bg),
        flags: Flags::empty(),
        hyperlink: None,
    };

    let found_text_A_redraw = commands_frame2.iter().any(|cmd| {
        match_text_command(
            cmd,
            "A",
            0.0,
            0.0,
            default_fg,
            default_bg,
            expected_attrs,
            false,
        )
    });
    assert!(
        !found_text_A_redraw,
        "Cell 'A' should not have been redrawn if not damaged."
    );

    // Allow for Clear, Cursor, Present as a baseline for a "no-change" frame.
    // This threshold (e.g. <= 3) is arbitrary and depends on optimizations.
    assert!(
        commands_frame2.len() < commands_frame1_count,
        "Frame 2 (no change) should have fewer or equal commands than Frame 1 (full draw)"
    );
    assert!(
        commands_frame2.len() <= 5,
        "Expected very few commands for a no-change frame, e.g. Clear, Cursor-related, Present"
    );
}

#[test]
fn test_damage_tracking_one_cell_change() {
    let char_width = 10.0;
    let char_height = 20.0;
    let num_cols = 3;
    let num_rows = 1;

    let (mut renderer, test_colors, screen_w, screen_h) =
        create_test_renderer_and_colors(num_cols, num_rows, char_width, char_height);
    let mut adapter = MockRenderAdapter::new(char_width, char_height, screen_w, screen_h);

    let mut cells_frame1 = vec![Cell::default(); num_cols];
    cells_frame1[0] = Cell {
        char: 'A',
        ..Default::default()
    };
    cells_frame1[1] = Cell {
        char: 'B',
        ..Default::default()
    };
    let cursor = Cursor {
        col: 2,
        ..Default::default()
    };

    let snapshot1 = create_test_snapshot(&cells_frame1, &cursor, num_cols, num_rows, true, None);
    renderer.render_frame(&snapshot1, &mut adapter); // Populate prev_snapshot
    adapter.clear_commands();

    let mut cells_frame2 = cells_frame1.clone();
    cells_frame2[1] = Cell {
        char: 'C',
        ..Default::default()
    }; // Changed 'B' to 'C'

    let snapshot2 = create_test_snapshot(&cells_frame2, &cursor, num_cols, num_rows, true, None);
    renderer.render_frame(&snapshot2, &mut adapter);
    let commands_frame2 = adapter.commands();

    let default_fg = test_colors.default_fg();
    let default_bg = test_colors.default_bg();
    let expected_attrs = ResolvedCellAttrs {
        fg: ResolvedColor::from(default_fg),
        bg: ResolvedColor::from(default_bg),
        flags: Flags::empty(),
        hyperlink: None,
    };

    // Cell 'A' (index 0) should NOT be redrawn
    let found_text_A_redraw = commands_frame2.iter().any(|cmd| {
        match_text_command(
            cmd,
            "A",
            0.0 * char_width,
            0.0,
            default_fg,
            default_bg,
            expected_attrs,
            false,
        )
    });
    assert!(
        !found_text_A_redraw,
        "Cell 'A' should not have been redrawn."
    );

    // Cell at index 1 (char 'C') SHOULD be redrawn
    // This includes its background rect and the text itself.
    let x_cell1 = 1.0 * char_width;
    let y_cell1 = 0.0 * char_height;

    assert!(
        commands_frame2.iter().any(|cmd| {
            match_rect_command(cmd, x_cell1, y_cell1, char_width, char_height, default_bg)
        }),
        "Missing background rect for changed cell 'C'"
    );
    assert!(
        commands_frame2.iter().any(|cmd| {
            match_text_command(
                cmd,
                "C",
                x_cell1,
                y_cell1,
                default_fg,
                default_bg,
                expected_attrs,
                false,
            )
        }),
        "Missing text 'C' for changed cell"
    );

    // Ensure Present is still there
    assert!(matches!(
        commands_frame2.last().unwrap(),
        DrawCommand::Present
    ));
}

#[test]
fn test_full_redraw_on_font_change() {
    let char_width = 10.0;
    let char_height = 20.0;
    let num_cols = 2;
    let num_rows = 1;

    let (mut renderer, test_colors, screen_w, screen_h) =
        create_test_renderer_and_colors(num_cols, num_rows, char_width, char_height);
    let mut adapter = MockRenderAdapter::new(char_width, char_height, screen_w, screen_h);

    let mut cells = vec![Cell::default(); num_cols];
    cells[0] = Cell {
        char: 'X',
        ..Default::default()
    };
    cells[1] = Cell {
        char: 'Y',
        ..Default::default()
    };
    let cursor = Cursor::default();

    // Frame 1: Initial render
    let snapshot1 = create_test_snapshot(&cells, &cursor, num_cols, num_rows, true, None);
    renderer.render_frame(&snapshot1, &mut adapter);
    adapter.clear_commands();

    // Update font: This should trigger full redraw on next frame
    let new_font_desc = FontDesc::new("serif".to_string(), 16.0);
    renderer
        .update_font(new_font_desc)
        .expect("Failed to update font");

    // Frame 2: Render again with the same cell content
    // The renderer should detect font change and perform a full redraw.
    let cells_clone = cells.clone(); // Ensure data for snapshot lives
    let snapshot2 = create_test_snapshot(&cells_clone, &cursor, num_cols, num_rows, true, None);
    renderer.render_frame(&snapshot2, &mut adapter);
    let commands_frame2 = adapter.commands();

    // Assert that both 'X' and 'Y' were redrawn, indicating a full redraw.
    let default_fg = test_colors.default_fg();
    let default_bg = test_colors.default_bg();
    let expected_attrs = ResolvedCellAttrs {
        fg: ResolvedColor::from(default_fg),
        bg: ResolvedColor::from(default_bg),
        flags: Flags::empty(),
        hyperlink: None,
    };

    assert!(
        commands_frame2.iter().any(|cmd| {
            match_text_command(
                cmd,
                "X",
                0.0 * char_width,
                0.0,
                default_fg,
                default_bg,
                expected_attrs,
                false,
            )
        }),
        "Cell 'X' should have been redrawn after font change."
    );
    assert!(
        commands_frame2.iter().any(|cmd| {
            match_text_command(
                cmd,
                "Y",
                1.0 * char_width,
                0.0,
                default_fg,
                default_bg,
                expected_attrs,
                false,
            )
        }),
        "Cell 'Y' should have been redrawn after font change."
    );

    // Also check for background rects for all cells as part of full redraw
    let mut bg_rects_found = 0;
    for c in 0..num_cols {
        let x = c as f32 * char_width;
        let y = 0.0 * char_height;
        if commands_frame2
            .iter()
            .any(|cmd| match_rect_command(cmd, x, y, char_width, char_height, default_bg))
        {
            bg_rects_found += 1;
        }
    }
    assert_eq!(
        bg_rects_found, num_cols,
        "All cell backgrounds should be redrawn after font change."
    );
}
