// src/renderer/tests.rs
#![cfg(test)]
// These types are expected to be pub in src/renderer.rs
use crate::renderer::{RenderAdapter, Renderer, ResolvedCellAttrs, ResolvedColor};
// These types are expected to be pub in their respective modules,
// and the modules (color, config, glyph, term) must be declared in src/main.rs or src/lib.rs
use crate::color::{Colors, Rgb};
use crate::config::FontDesc;
use crate::glyph::{Cell, CellAttrs, Flags};
use crate::term::cursor::{Cursor, CursorShape, CursorStyle};
use crate::term::snapshot::{RenderSnapshot, SelectionRange};
use std::rc::Rc;
use std::sync::Mutex;

#[derive(Debug, Clone, PartialEq)]
enum DrawCommand {
    Clear(Rgb), // crate::color::Rgb
    Rect {
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        color: Rgb, // crate::color::Rgb
    },
    Text {
        text: String,
        x: f32,
        y: f32,
        fg: Rgb,                  // crate::color::Rgb
        bg: Rgb,                  // crate::color::Rgb
        attrs: ResolvedCellAttrs, // crate::renderer::ResolvedCellAttrs
        is_wide: bool,
    },
    Present,
}

fn float_eq(a: f32, b: f32) -> bool {
    (a - b).abs() < 0.001
}

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
            && *attrs == eattrs
            && *is_wide == eis_wide
    } else {
        false
    }
}

struct MockRenderAdapter {
    commands: Mutex<Vec<DrawCommand>>,
    char_width: f32,
    char_height: f32,
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
        self.commands.lock().unwrap().clone()
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
        Ok(())
    }
}

fn create_test_renderer_and_colors(
    num_cols: usize,
    num_rows: usize,
    char_width: f32,
    char_height: f32,
) -> (Renderer, Rc<Colors>, u32, u32) {
    let colors = Rc::new(Colors::default_palleted());
    let font_desc = FontDesc::new("monospace".to_string(), 15.0);
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

fn create_test_snapshot(
    cells_slice: &[Cell],
    cursor_ref: &Cursor,
    num_cols: usize,
    num_rows: usize,
    focused: bool,
    selection: Option<SelectionRange>,
) -> RenderSnapshot {
    RenderSnapshot::new(
        cells_slice,
        cursor_ref,
        "Test Window Title".to_string(),
        num_cols,
        num_rows,
        0,
        false,
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

    let cells_data = vec![Cell::default(); num_cols * num_rows];
    let cursor_data = Cursor::default();

    let snapshot = create_test_snapshot(&cells_data, &cursor_data, num_cols, num_rows, true, None);
    renderer.render_frame(&snapshot, &mut adapter);

    let commands = adapter.commands();
    assert!(!commands.is_empty(), "Should have drawing commands");

    let expected_bg = test_colors.default_bg();
    let cursor_fg_color = test_colors.cursor_fg();

    assert!(
        matches!(commands[0], DrawCommand::Clear(color) if color == test_colors.background()),
        "First command should be Clear with background color"
    );

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
    assert_eq!(
        bg_rects_found,
        num_cols * num_rows,
        "Should draw background rects for all cells"
    );

    let cursor_x = cursor_data.col as f32 * char_width;
    let cursor_y = cursor_data.row as f32 * char_height;

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

    let mut cells_data = vec![Cell::default(); num_cols * num_rows];
    cells_data[0] = Cell {
        char: 'H',
        attrs: CellAttrs::default(),
        flags: Flags::empty(),
        width: 1,
    };
    cells_data[1] = Cell {
        char: 'i',
        attrs: CellAttrs::default(),
        flags: Flags::empty(),
        width: 1,
    };

    let cursor_data = Cursor {
        row: 0,
        col: 2,
        style: CursorStyle::default(),
        ..Default::default()
    };

    let snapshot = create_test_snapshot(&cells_data, &cursor_data, num_cols, num_rows, true, None);
    renderer.render_frame(&snapshot, &mut adapter);

    let commands = adapter.commands();

    let default_fg = test_colors.default_fg();
    let default_bg = test_colors.default_bg();
    let expected_attrs = ResolvedCellAttrs {
        fg: ResolvedColor::from_rgb(default_fg),
        bg: ResolvedColor::from_rgb(default_bg),
        flags: Flags::empty(),
        hyperlink: None,
    };

    assert!(
        commands.iter().any(|cmd| match_text_command(
            cmd,
            "H",
            0.0 * char_width,
            0.0 * char_height,
            default_fg,
            default_bg,
            expected_attrs.clone(),
            false
        )),
        "Missing text 'H'"
    );

    assert!(
        commands.iter().any(|cmd| match_text_command(
            cmd,
            "i",
            1.0 * char_width,
            0.0 * char_height,
            default_fg,
            default_bg,
            expected_attrs,
            false
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

    let (mut renderer, test_colors, screen_w, screen_h) =
        create_test_renderer_and_colors(num_cols, num_rows, char_width, char_height);
    let mut adapter = MockRenderAdapter::new(char_width, char_height, screen_w, screen_h);

    let mut cells_data = vec![Cell::default(); num_cols];
    cells_data[0] = Cell {
        char: 'A',
        ..Default::default()
    };
    let cursor_data = Cursor::default();

    let snapshot1 = create_test_snapshot(&cells_data, &cursor_data, num_cols, num_rows, true, None);
    renderer.render_frame(&snapshot1, &mut adapter);
    let commands_frame1_count = adapter.commands().len();
    adapter.clear_commands();

    let cells_clone = cells_data.clone();
    let snapshot2 =
        create_test_snapshot(&cells_clone, &cursor_data, num_cols, num_rows, true, None);
    renderer.render_frame(&snapshot2, &mut adapter);
    let commands_frame2 = adapter.commands();

    let default_fg = test_colors.default_fg();
    let default_bg = test_colors.default_bg();
    let expected_attrs = ResolvedCellAttrs {
        fg: ResolvedColor::from_rgb(default_fg),
        bg: ResolvedColor::from_rgb(default_bg),
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

    assert!(
        commands_frame2.len() < commands_frame1_count || commands_frame1_count <= 5,
        "Frame 2 (no change) should have fewer or equal commands than Frame 1, or Frame 1 was already minimal"
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

    let mut cells_frame1_data = vec![Cell::default(); num_cols];
    cells_frame1_data[0] = Cell {
        char: 'A',
        ..Default::default()
    };
    cells_frame1_data[1] = Cell {
        char: 'B',
        ..Default::default()
    };
    let cursor_data = Cursor {
        col: 2,
        ..Default::default()
    };

    let snapshot1 = create_test_snapshot(
        &cells_frame1_data,
        &cursor_data,
        num_cols,
        num_rows,
        true,
        None,
    );
    renderer.render_frame(&snapshot1, &mut adapter);
    adapter.clear_commands();

    let mut cells_frame2_data = cells_frame1_data.clone();
    cells_frame2_data[1] = Cell {
        char: 'C',
        ..Default::default()
    };

    let snapshot2 = create_test_snapshot(
        &cells_frame2_data,
        &cursor_data,
        num_cols,
        num_rows,
        true,
        None,
    );
    renderer.render_frame(&snapshot2, &mut adapter);
    let commands_frame2 = adapter.commands();

    let default_fg = test_colors.default_fg();
    let default_bg = test_colors.default_bg();
    let expected_attrs = ResolvedCellAttrs {
        fg: ResolvedColor::from_rgb(default_fg),
        bg: ResolvedColor::from_rgb(default_bg),
        flags: Flags::empty(),
        hyperlink: None,
    };

    let found_text_A_redraw = commands_frame2.iter().any(|cmd| {
        match_text_command(
            cmd,
            "A",
            0.0 * char_width,
            0.0,
            default_fg,
            default_bg,
            expected_attrs.clone(),
            false,
        )
    });
    assert!(
        !found_text_A_redraw,
        "Cell 'A' should not have been redrawn."
    );

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

    let mut cells_data = vec![Cell::default(); num_cols];
    cells_data[0] = Cell {
        char: 'X',
        ..Default::default()
    };
    cells_data[1] = Cell {
        char: 'Y',
        ..Default::default()
    };
    let cursor_data = Cursor::default();

    let snapshot1 = create_test_snapshot(&cells_data, &cursor_data, num_cols, num_rows, true, None);
    renderer.render_frame(&snapshot1, &mut adapter);
    adapter.clear_commands();

    let new_font_desc = FontDesc::new("serif".to_string(), 16.0);
    renderer
        .update_font(new_font_desc)
        .expect("Failed to update font");

    let cells_clone = cells_data.clone();
    let snapshot2 =
        create_test_snapshot(&cells_clone, &cursor_data, num_cols, num_rows, true, None);
    renderer.render_frame(&snapshot2, &mut adapter);
    let commands_frame2 = adapter.commands();

    let default_fg = test_colors.default_fg();
    let default_bg = test_colors.default_bg();
    let expected_attrs = ResolvedCellAttrs {
        fg: ResolvedColor::from_rgb(default_fg),
        bg: ResolvedColor::from_rgb(default_bg),
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
                expected_attrs.clone(),
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
