// src/renderer/renderer_tests.rs
//
// Comprehensive test suite for the Renderer module.
// Goal: 80%+ code coverage with SQLite3-level rigor.
// Strategy: Test only the public API (prepare_render_commands) per STYLE.md

use super::renderer::Renderer;
use crate::color::{Color, NamedColor};
use crate::config::Config;
use crate::glyph::{AttrFlags, Attributes, ContentCell, Glyph};
use crate::renderer::{PlatformState, RenderCommand};
use crate::term::snapshot::{
    CursorRenderState, CursorShape, Point, Selection, SelectionMode, SelectionRange, SnapshotLine,
    TerminalSnapshot,
};

const TEST_TERM_WIDTH: usize = 80;
const TEST_TERM_HEIGHT: usize = 24;
const TEST_CELL_WIDTH_PX: usize = 10;
const TEST_CELL_HEIGHT_PX: usize = 16;

fn create_test_config() -> Config {
    Config::default()
}

fn create_test_platform_state() -> PlatformState {
    PlatformState {
        event_fd: None,
        font_cell_width_px: TEST_CELL_WIDTH_PX,
        font_cell_height_px: TEST_CELL_HEIGHT_PX,
        scale_factor: 1.0,
        display_width_px: (TEST_TERM_WIDTH * TEST_CELL_WIDTH_PX) as u16,
        display_height_px: (TEST_TERM_HEIGHT * TEST_CELL_HEIGHT_PX) as u16,
    }
}

fn create_empty_snapshot(width: usize, height: usize) -> TerminalSnapshot {
    let empty_cell = Glyph::Single(ContentCell {
        c: ' ',
        attr: Attributes::default(),
    });

    let lines = (0..height)
        .map(|_| SnapshotLine {
            is_dirty: false,
            cells: vec![empty_cell.clone(); width],
        })
        .collect();

    TerminalSnapshot {
        dimensions: (width, height),
        lines,
        cursor_state: None,
        selection: Selection::default(),
        cell_width_px: TEST_CELL_WIDTH_PX,
        cell_height_px: TEST_CELL_HEIGHT_PX,
    }
}

fn create_snapshot_with_text(lines_text: Vec<&str>) -> TerminalSnapshot {
    let height = lines_text.len();
    let width = lines_text.iter().map(|s| s.len()).max().unwrap_or(0);

    let default_attr = Attributes::default();
    let empty_cell = Glyph::Single(ContentCell {
        c: ' ',
        attr: default_attr,
    });

    let lines = lines_text
        .iter()
        .map(|text| {
            let mut cells: Vec<Glyph> = text
                .chars()
                .map(|c| {
                    Glyph::Single(ContentCell {
                        c,
                        attr: default_attr,
                    })
                })
                .collect();

            cells.resize(width, empty_cell.clone());

            SnapshotLine {
                is_dirty: true,
                cells,
            }
        })
        .collect();

    TerminalSnapshot {
        dimensions: (width, height),
        lines,
        cursor_state: None,
        selection: Selection::default(),
        cell_width_px: TEST_CELL_WIDTH_PX,
        cell_height_px: TEST_CELL_HEIGHT_PX,
    }
}

fn make_glyph(c: char, fg: Color, bg: Color, flags: AttrFlags) -> Glyph {
    Glyph::Single(ContentCell {
        c,
        attr: Attributes { fg, bg, flags },
    })
}

fn make_wide_char_pair(c: char, attr: Attributes) -> (Glyph, Glyph) {
    (
        Glyph::WidePrimary(ContentCell { c, attr }),
        Glyph::WideSpacer {
            primary_column_on_line: 0,
        },
    )
}

fn count_command_type(commands: &[RenderCommand], type_name: &str) -> usize {
    commands
        .iter()
        .filter(|cmd| match (type_name, cmd) {
            ("ClearAll", RenderCommand::ClearAll { .. }) => true,
            ("DrawTextRun", RenderCommand::DrawTextRun { .. }) => true,
            ("FillRect", RenderCommand::FillRect { .. }) => true,
            _ => false,
        })
        .count()
}

// =============================================================================
// API Contract Tests
// =============================================================================

#[test]
fn renderer_instances_have_identical_size() {
    let renderer1 = Renderer::new();
    let renderer2 = Renderer::new();

    assert_eq!(
        std::mem::size_of_val(&renderer1),
        std::mem::size_of_val(&renderer2)
    );
}

#[test]
fn empty_snapshot_produces_no_commands() {
    let renderer = Renderer::new();
    let config = create_test_config();
    let platform_state = create_test_platform_state();
    let snapshot = create_empty_snapshot(0, 0);

    let commands = renderer.prepare_render_commands(&snapshot, &config, &platform_state);

    assert_eq!(commands.len(), 0);
}

#[test]
fn clean_lines_produce_no_commands() {
    let renderer = Renderer::new();
    let config = create_test_config();
    let platform_state = create_test_platform_state();
    let snapshot = create_empty_snapshot(TEST_TERM_WIDTH, TEST_TERM_HEIGHT);

    let commands = renderer.prepare_render_commands(&snapshot, &config, &platform_state);

    assert_eq!(commands.len(), 0);
}

#[test]
fn dirty_line_with_text_produces_draw_commands() {
    let renderer = Renderer::new();
    let config = create_test_config();
    let platform_state = create_test_platform_state();
    let snapshot = create_snapshot_with_text(vec!["Hello"]);

    let commands = renderer.prepare_render_commands(&snapshot, &config, &platform_state);

    assert!(commands.len() > 0);
    assert!(count_command_type(&commands, "DrawTextRun") > 0);
}

#[test]
fn cursor_renders_as_inverted_text_run() {
    let renderer = Renderer::new();
    let config = create_test_config();
    let platform_state = create_test_platform_state();
    let mut snapshot = create_snapshot_with_text(vec!["Test"]);

    snapshot.cursor_state = Some(CursorRenderState {
        x: 2,
        y: 0,
        shape: CursorShape::Block,
        cell_char_underneath: 's',
        cell_attributes_underneath: Attributes::default(),
    });

    let commands = renderer.prepare_render_commands(&snapshot, &config, &platform_state);

    let has_inverted_run = commands.iter().any(|cmd| {
        if let RenderCommand::DrawTextRun { x, y, fg, bg, .. } = cmd {
            *x == 2 && *y == 0 && *fg == config.colors.background && *bg == config.colors.foreground
        } else {
            false
        }
    });

    assert!(has_inverted_run, "Cursor should render as inverted colors");
}

#[test]
fn cursor_overlay_drawn_after_text_content() {
    let renderer = Renderer::new();
    let config = create_test_config();
    let platform_state = create_test_platform_state();
    let mut snapshot = create_snapshot_with_text(vec!["Test"]);

    snapshot.cursor_state = Some(CursorRenderState {
        x: 0,
        y: 0,
        shape: CursorShape::Block,
        cell_char_underneath: 'T',
        cell_attributes_underneath: Attributes::default(),
    });

    let commands = renderer.prepare_render_commands(&snapshot, &config, &platform_state);

    let cursor_pos = commands.iter().position(|cmd| {
        if let RenderCommand::DrawTextRun { x, y, fg, bg, .. } = cmd {
            *x == 0 && *y == 0 && *fg == config.colors.background && *bg == config.colors.foreground
        } else {
            false
        }
    });

    let first_text_pos = commands
        .iter()
        .position(|cmd| matches!(cmd, RenderCommand::DrawTextRun { .. }));

    if let (Some(cursor_idx), Some(text_idx)) = (cursor_pos, first_text_pos) {
        assert!(
            cursor_idx > text_idx,
            "Cursor overlay should be drawn after content"
        );
    }
}

// =============================================================================
// Color Resolution Tests (via DrawTextRun output)
// =============================================================================

#[test]
fn default_colors_resolve_to_config_foreground_and_background() {
    let renderer = Renderer::new();
    let config = create_test_config();
    let platform_state = create_test_platform_state();

    let mut snapshot = create_empty_snapshot(5, 1);
    snapshot.lines[0].is_dirty = true;
    snapshot.lines[0].cells =
        vec![make_glyph('A', Color::Default, Color::Default, AttrFlags::empty()); 5];

    let commands = renderer.prepare_render_commands(&snapshot, &config, &platform_state);

    let text_runs: Vec<_> = commands
        .iter()
        .filter_map(|cmd| {
            if let RenderCommand::DrawTextRun { fg, bg, .. } = cmd {
                Some((fg, bg))
            } else {
                None
            }
        })
        .collect();

    assert!(text_runs.len() > 0);
    for (fg, bg) in text_runs {
        assert_eq!(*fg, config.colors.foreground);
        assert_eq!(*bg, config.colors.background);
    }
}

#[test]
fn explicit_colors_preserved_in_output() {
    let renderer = Renderer::new();
    let config = create_test_config();
    let platform_state = create_test_platform_state();

    let red = Color::Named(NamedColor::Red);
    let green = Color::Named(NamedColor::Green);

    let mut snapshot = create_empty_snapshot(5, 1);
    snapshot.lines[0].is_dirty = true;
    snapshot.lines[0].cells = vec![make_glyph('A', red, green, AttrFlags::empty()); 5];

    let commands = renderer.prepare_render_commands(&snapshot, &config, &platform_state);

    let text_runs: Vec<_> = commands
        .iter()
        .filter_map(|cmd| {
            if let RenderCommand::DrawTextRun { fg, bg, .. } = cmd {
                Some((fg, bg))
            } else {
                None
            }
        })
        .collect();

    assert!(text_runs.len() > 0);
    for (fg_out, bg_out) in text_runs {
        assert_eq!(*fg_out, red);
        assert_eq!(*bg_out, green);
    }
}

#[test]
fn reverse_video_flag_swaps_foreground_and_background() {
    let renderer = Renderer::new();
    let config = create_test_config();
    let platform_state = create_test_platform_state();

    let red = Color::Named(NamedColor::Red);
    let green = Color::Named(NamedColor::Green);

    let mut snapshot = create_empty_snapshot(5, 1);
    snapshot.lines[0].is_dirty = true;
    snapshot.lines[0].cells = vec![make_glyph('A', red, green, AttrFlags::REVERSE); 5];

    let commands = renderer.prepare_render_commands(&snapshot, &config, &platform_state);

    let text_runs: Vec<_> = commands
        .iter()
        .filter_map(|cmd| {
            if let RenderCommand::DrawTextRun { fg, bg, flags, .. } = cmd {
                Some((fg, bg, flags))
            } else {
                None
            }
        })
        .collect();

    assert!(text_runs.len() > 0);
    for (fg_out, bg_out, flags_out) in text_runs {
        assert_eq!(*fg_out, green);
        assert_eq!(*bg_out, red);
        assert!(!flags_out.contains(AttrFlags::REVERSE));
    }
}

#[test]
fn bold_and_italic_flags_preserved_when_reverse_removed() {
    let renderer = Renderer::new();
    let config = create_test_config();
    let platform_state = create_test_platform_state();

    let input_flags = AttrFlags::BOLD | AttrFlags::ITALIC | AttrFlags::REVERSE;

    let mut snapshot = create_empty_snapshot(5, 1);
    snapshot.lines[0].is_dirty = true;
    snapshot.lines[0].cells = vec![make_glyph('A', Color::Default, Color::Default, input_flags); 5];

    let commands = renderer.prepare_render_commands(&snapshot, &config, &platform_state);

    let text_runs: Vec<_> = commands
        .iter()
        .filter_map(|cmd| {
            if let RenderCommand::DrawTextRun { flags, .. } = cmd {
                Some(flags)
            } else {
                None
            }
        })
        .collect();

    assert!(text_runs.len() > 0);
    for flags_out in text_runs {
        assert!(flags_out.contains(AttrFlags::BOLD));
        assert!(flags_out.contains(AttrFlags::ITALIC));
        assert!(!flags_out.contains(AttrFlags::REVERSE));
    }
}

// =============================================================================
// Selection Tests (via is_selected flag in commands)
// =============================================================================

#[test]
fn no_selection_marks_all_text_runs_unselected() {
    let renderer = Renderer::new();
    let config = create_test_config();
    let platform_state = create_test_platform_state();

    let snapshot = create_snapshot_with_text(vec!["Hello World"]);

    let commands = renderer.prepare_render_commands(&snapshot, &config, &platform_state);

    let text_runs: Vec<_> = commands
        .iter()
        .filter_map(|cmd| {
            if let RenderCommand::DrawTextRun { is_selected, .. } = cmd {
                Some(*is_selected)
            } else {
                None
            }
        })
        .collect();

    assert!(text_runs.len() > 0);
    assert!(text_runs.iter().all(|&selected| !selected));
}

#[test]
fn inactive_selection_marks_text_runs_unselected() {
    let renderer = Renderer::new();
    let config = create_test_config();
    let platform_state = create_test_platform_state();

    let mut snapshot = create_snapshot_with_text(vec!["Hello World"]);
    snapshot.selection = Selection {
        range: Some(SelectionRange {
            start: Point { x: 0, y: 0 },
            end: Point { x: 5, y: 0 },
        }),
        mode: SelectionMode::Cell,
        is_active: false,
    };

    let commands = renderer.prepare_render_commands(&snapshot, &config, &platform_state);

    let text_runs: Vec<_> = commands
        .iter()
        .filter_map(|cmd| {
            if let RenderCommand::DrawTextRun { is_selected, .. } = cmd {
                Some(*is_selected)
            } else {
                None
            }
        })
        .collect();

    assert!(text_runs.len() > 0);
    assert!(text_runs.iter().all(|&selected| !selected));
}

#[test]
fn active_selection_marks_text_runs_selected() {
    let renderer = Renderer::new();
    let config = create_test_config();
    let platform_state = create_test_platform_state();

    let mut snapshot = create_snapshot_with_text(vec!["Hello World"]);
    snapshot.selection = Selection {
        range: Some(SelectionRange {
            start: Point { x: 0, y: 0 },
            end: Point { x: 4, y: 0 },
        }),
        mode: SelectionMode::Cell,
        is_active: true,
    };

    let commands = renderer.prepare_render_commands(&snapshot, &config, &platform_state);

    let has_selected_run = commands.iter().any(|cmd| {
        if let RenderCommand::DrawTextRun { is_selected, .. } = cmd {
            *is_selected
        } else {
            false
        }
    });

    assert!(has_selected_run);
}

#[test]
fn multiline_selection_spans_all_affected_lines() {
    let renderer = Renderer::new();
    let config = create_test_config();
    let platform_state = create_test_platform_state();

    let mut snapshot = create_snapshot_with_text(vec!["AAAA", "BBBB", "CCCC"]);
    snapshot.selection = Selection {
        range: Some(SelectionRange {
            start: Point { x: 0, y: 0 },
            end: Point { x: 3, y: 2 },
        }),
        mode: SelectionMode::Cell,
        is_active: true,
    };

    let commands = renderer.prepare_render_commands(&snapshot, &config, &platform_state);

    let lines_with_selection: std::collections::HashSet<_> = commands
        .iter()
        .filter_map(|cmd| {
            if let RenderCommand::DrawTextRun { y, is_selected, .. } = cmd {
                if *is_selected {
                    Some(*y)
                } else {
                    None
                }
            } else {
                None
            }
        })
        .collect();

    assert!(
        lines_with_selection.contains(&0),
        "Line 0 should have selected text"
    );
    assert!(
        lines_with_selection.contains(&1),
        "Line 1 should have selected text"
    );
    assert!(
        lines_with_selection.contains(&2),
        "Line 2 should have selected text"
    );
}

// =============================================================================
// Wide Character Tests
// =============================================================================

#[test]
fn wide_characters_render_as_text_runs() {
    let renderer = Renderer::new();
    let config = create_test_config();
    let platform_state = create_test_platform_state();

    let attr = Attributes::default();
    let (wide_primary, wide_spacer) = make_wide_char_pair('中', attr);

    let mut snapshot = create_empty_snapshot(10, 1);
    snapshot.lines[0].is_dirty = true;
    snapshot.lines[0].cells = vec![
        wide_primary,
        wide_spacer,
        make_glyph('A', Color::Default, Color::Default, AttrFlags::empty()),
    ];
    snapshot.lines[0].cells.resize(
        10,
        make_glyph(' ', Color::Default, Color::Default, AttrFlags::empty()),
    );

    let commands = renderer.prepare_render_commands(&snapshot, &config, &platform_state);

    let text_runs: Vec<_> = commands
        .iter()
        .filter_map(|cmd| {
            if let RenderCommand::DrawTextRun { text, .. } = cmd {
                Some(text.as_str())
            } else {
                None
            }
        })
        .collect();

    assert!(text_runs.iter().any(|&text| text.contains('中')));
}

#[test]
fn wide_character_at_line_end_renders_correctly() {
    let renderer = Renderer::new();
    let config = create_test_config();
    let platform_state = create_test_platform_state();

    let attr = Attributes::default();
    let (wide_primary, wide_spacer) = make_wide_char_pair('中', attr);

    let mut snapshot = create_empty_snapshot(5, 1);
    snapshot.lines[0].is_dirty = true;
    snapshot.lines[0].cells = vec![
        make_glyph('A', Color::Default, Color::Default, AttrFlags::empty()),
        make_glyph('B', Color::Default, Color::Default, AttrFlags::empty()),
        make_glyph('C', Color::Default, Color::Default, AttrFlags::empty()),
        wide_primary,
        wide_spacer,
    ];

    let commands = renderer.prepare_render_commands(&snapshot, &config, &platform_state);

    assert!(commands.len() > 0);
    assert!(count_command_type(&commands, "DrawTextRun") > 0);
}

#[test]
fn orphaned_wide_spacer_at_column_zero_fills_with_background() {
    let renderer = Renderer::new();
    let config = create_test_config();
    let platform_state = create_test_platform_state();

    let mut snapshot = create_empty_snapshot(5, 1);
    snapshot.lines[0].is_dirty = true;
    snapshot.lines[0].cells = vec![
        Glyph::WideSpacer {
            primary_column_on_line: 0,
        },
        make_glyph('A', Color::Default, Color::Default, AttrFlags::empty()),
        make_glyph('B', Color::Default, Color::Default, AttrFlags::empty()),
        make_glyph('C', Color::Default, Color::Default, AttrFlags::empty()),
        make_glyph('D', Color::Default, Color::Default, AttrFlags::empty()),
    ];

    let commands = renderer.prepare_render_commands(&snapshot, &config, &platform_state);

    assert!(commands.len() > 0);
    let has_fill_rect = count_command_type(&commands, "FillRect") > 0;
    assert!(
        has_fill_rect,
        "Orphaned spacer should be filled with background color"
    );
}

// =============================================================================
// Cursor Rendering Tests
// =============================================================================

#[test]
fn out_of_bounds_cursor_skipped() {
    let renderer = Renderer::new();
    let config = create_test_config();
    let platform_state = create_test_platform_state();

    let mut snapshot = create_snapshot_with_text(vec!["Test"]);
    let text_cmd_count_without_cursor = {
        let cmds = renderer.prepare_render_commands(&snapshot, &config, &platform_state);
        count_command_type(&cmds, "DrawTextRun")
    };

    snapshot.cursor_state = Some(CursorRenderState {
        x: 100,
        y: 100,
        shape: CursorShape::Block,
        cell_char_underneath: ' ',
        cell_attributes_underneath: Attributes::default(),
    });

    let commands = renderer.prepare_render_commands(&snapshot, &config, &platform_state);
    let text_cmd_count_with_oob_cursor = count_command_type(&commands, "DrawTextRun");

    assert_eq!(
        text_cmd_count_with_oob_cursor, text_cmd_count_without_cursor,
        "Out of bounds cursor should not add extra text run"
    );
}

// =============================================================================
// Edge Cases and Error Handling
// =============================================================================

#[test]
fn zero_width_terminal_produces_no_commands() {
    let renderer = Renderer::new();
    let config = create_test_config();
    let platform_state = create_test_platform_state();

    let snapshot = create_empty_snapshot(0, 10);

    let commands = renderer.prepare_render_commands(&snapshot, &config, &platform_state);

    assert_eq!(commands.len(), 0);
}

#[test]
fn zero_height_terminal_produces_no_commands() {
    let renderer = Renderer::new();
    let config = create_test_config();
    let platform_state = create_test_platform_state();

    let snapshot = create_empty_snapshot(80, 0);

    let commands = renderer.prepare_render_commands(&snapshot, &config, &platform_state);

    assert_eq!(commands.len(), 0);
}

#[test]
fn line_shorter_than_width_fills_remainder_with_background() {
    let renderer = Renderer::new();
    let config = create_test_config();
    let platform_state = create_test_platform_state();

    let mut snapshot = create_empty_snapshot(80, 1);
    snapshot.lines[0].is_dirty = true;
    snapshot.lines[0].cells =
        vec![make_glyph('A', Color::Default, Color::Default, AttrFlags::empty()); 5];

    let commands = renderer.prepare_render_commands(&snapshot, &config, &platform_state);

    let fill_rects: Vec<_> = commands
        .iter()
        .filter_map(|cmd| {
            if let RenderCommand::FillRect { x, width, .. } = cmd {
                Some((*x, *width))
            } else {
                None
            }
        })
        .collect();

    assert!(fill_rects.len() > 0);
}

#[test]
fn multiple_dirty_lines_all_render() {
    let renderer = Renderer::new();
    let config = create_test_config();
    let platform_state = create_test_platform_state();

    let snapshot = create_snapshot_with_text(vec!["Line1", "Line2", "Line3"]);

    let commands = renderer.prepare_render_commands(&snapshot, &config, &platform_state);

    let unique_y_values: std::collections::HashSet<_> = commands
        .iter()
        .filter_map(|cmd| match cmd {
            RenderCommand::DrawTextRun { y, .. } => Some(*y),
            RenderCommand::FillRect { y, .. } => Some(*y),
            _ => None,
        })
        .collect();

    assert!(unique_y_values.len() >= 3);
}

#[test]
fn consecutive_spaces_coalesce_into_fill_rect() {
    let renderer = Renderer::new();
    let config = create_test_config();
    let platform_state = create_test_platform_state();

    let mut snapshot = create_empty_snapshot(20, 1);
    snapshot.lines[0].is_dirty = true;
    snapshot.lines[0].cells = vec![
        make_glyph('A', Color::Default, Color::Default, AttrFlags::empty()),
        make_glyph(' ', Color::Default, Color::Default, AttrFlags::empty()),
        make_glyph(' ', Color::Default, Color::Default, AttrFlags::empty()),
        make_glyph(' ', Color::Default, Color::Default, AttrFlags::empty()),
        make_glyph(' ', Color::Default, Color::Default, AttrFlags::empty()),
        make_glyph('B', Color::Default, Color::Default, AttrFlags::empty()),
    ];
    snapshot.lines[0].cells.resize(
        20,
        make_glyph(' ', Color::Default, Color::Default, AttrFlags::empty()),
    );

    let commands = renderer.prepare_render_commands(&snapshot, &config, &platform_state);

    let fill_rect_count = count_command_type(&commands, "FillRect");
    assert!(fill_rect_count > 0);
}

#[test]
fn consecutive_chars_with_same_attrs_coalesce_into_single_run() {
    let renderer = Renderer::new();
    let config = create_test_config();
    let platform_state = create_test_platform_state();

    let mut snapshot = create_empty_snapshot(20, 1);
    snapshot.lines[0].is_dirty = true;
    snapshot.lines[0].cells = vec![
        make_glyph('A', Color::Default, Color::Default, AttrFlags::empty()),
        make_glyph('B', Color::Default, Color::Default, AttrFlags::empty()),
        make_glyph('C', Color::Default, Color::Default, AttrFlags::empty()),
        make_glyph('D', Color::Default, Color::Default, AttrFlags::empty()),
        make_glyph('E', Color::Default, Color::Default, AttrFlags::empty()),
    ];
    snapshot.lines[0].cells.resize(
        20,
        make_glyph(' ', Color::Default, Color::Default, AttrFlags::empty()),
    );

    let commands = renderer.prepare_render_commands(&snapshot, &config, &platform_state);

    let text_runs: Vec<_> = commands
        .iter()
        .filter_map(|cmd| {
            if let RenderCommand::DrawTextRun { text, .. } = cmd {
                Some(text.len())
            } else {
                None
            }
        })
        .collect();

    assert!(text_runs.iter().any(|&len| len >= 5));
}

#[test]
fn attribute_change_breaks_text_run_coalescing() {
    let renderer = Renderer::new();
    let config = create_test_config();
    let platform_state = create_test_platform_state();

    let mut snapshot = create_empty_snapshot(20, 1);
    snapshot.lines[0].is_dirty = true;
    snapshot.lines[0].cells = vec![
        make_glyph('A', Color::Default, Color::Default, AttrFlags::empty()),
        make_glyph('B', Color::Default, Color::Default, AttrFlags::empty()),
        make_glyph(
            'C',
            Color::Named(NamedColor::Red),
            Color::Default,
            AttrFlags::empty(),
        ),
        make_glyph(
            'D',
            Color::Named(NamedColor::Red),
            Color::Default,
            AttrFlags::empty(),
        ),
        make_glyph('E', Color::Default, Color::Default, AttrFlags::empty()),
    ];
    snapshot.lines[0].cells.resize(
        20,
        make_glyph(' ', Color::Default, Color::Default, AttrFlags::empty()),
    );

    let commands = renderer.prepare_render_commands(&snapshot, &config, &platform_state);

    let text_run_count = count_command_type(&commands, "DrawTextRun");
    assert!(text_run_count >= 3);
}

// =============================================================================
// HiDPI Scaling Tests - Tests for logical vs physical coordinate handling
// =============================================================================

#[test]
fn scale_factor_does_not_affect_command_coordinates() {
    let renderer = Renderer::new();
    let config = create_test_config();

    let snapshot = create_snapshot_with_text(vec!["Test"]);

    let commands_1x = {
        let mut ps = create_test_platform_state();
        ps.scale_factor = 1.0;
        renderer.prepare_render_commands(&snapshot, &config, &ps)
    };

    let commands_2x = {
        let mut ps = create_test_platform_state();
        ps.scale_factor = 2.0;
        renderer.prepare_render_commands(&snapshot, &config, &ps)
    };

    assert_eq!(
        commands_1x.len(),
        commands_2x.len(),
        "Scale factor should not change number of commands"
    );

    for (cmd1, cmd2) in commands_1x.iter().zip(commands_2x.iter()) {
        match (cmd1, cmd2) {
            (
                RenderCommand::DrawTextRun {
                    x: x1,
                    y: y1,
                    text: t1,
                    ..
                },
                RenderCommand::DrawTextRun {
                    x: x2,
                    y: y2,
                    text: t2,
                    ..
                },
            ) => {
                assert_eq!(
                    x1, x2,
                    "X coordinates should be identical regardless of scale"
                );
                assert_eq!(
                    y1, y2,
                    "Y coordinates should be identical regardless of scale"
                );
                assert_eq!(t1, t2, "Text should be identical regardless of scale");
            }
            (
                RenderCommand::FillRect {
                    x: x1,
                    y: y1,
                    width: w1,
                    height: h1,
                    ..
                },
                RenderCommand::FillRect {
                    x: x2,
                    y: y2,
                    width: w2,
                    height: h2,
                    ..
                },
            ) => {
                assert_eq!(x1, x2, "FillRect X should be identical regardless of scale");
                assert_eq!(y1, y2, "FillRect Y should be identical regardless of scale");
                assert_eq!(
                    w1, w2,
                    "FillRect width should be identical regardless of scale"
                );
                assert_eq!(
                    h1, h2,
                    "FillRect height should be identical regardless of scale"
                );
            }
            _ => {}
        }
    }
}

#[test]
fn cell_coordinates_stay_within_terminal_bounds() {
    let renderer = Renderer::new();
    let config = create_test_config();
    let platform_state = create_test_platform_state();

    let snapshot = create_snapshot_with_text(vec!["A".repeat(TEST_TERM_WIDTH).as_str()]);
    let commands = renderer.prepare_render_commands(&snapshot, &config, &platform_state);

    for cmd in commands.iter() {
        match cmd {
            RenderCommand::DrawTextRun { x, y, .. } => {
                assert!(
                    *x < TEST_TERM_WIDTH,
                    "DrawTextRun X {} exceeds terminal width",
                    x
                );
                assert!(
                    *y < TEST_TERM_HEIGHT,
                    "DrawTextRun Y {} exceeds terminal height",
                    y
                );
            }
            RenderCommand::FillRect {
                x,
                y,
                width,
                height,
                ..
            } => {
                assert!(
                    *x < TEST_TERM_WIDTH,
                    "FillRect X {} exceeds terminal width",
                    x
                );
                assert!(
                    *y < TEST_TERM_HEIGHT,
                    "FillRect Y {} exceeds terminal height",
                    y
                );
                assert!(
                    *x + *width <= TEST_TERM_WIDTH,
                    "FillRect extends beyond terminal width"
                );
                assert!(
                    *y + *height <= TEST_TERM_HEIGHT,
                    "FillRect extends beyond terminal height"
                );
            }
            _ => {}
        }
    }
}

// =============================================================================
// Edge Cases for Coverage
// =============================================================================

#[test]
fn cursor_on_wide_character_primary_renders_correctly() {
    let renderer = Renderer::new();
    let config = create_test_config();
    let platform_state = create_test_platform_state();

    let attr = Attributes::default();
    let (wide_primary, wide_spacer) = make_wide_char_pair('中', attr);

    let mut snapshot = create_empty_snapshot(10, 1);
    snapshot.lines[0].is_dirty = true;
    snapshot.lines[0].cells = vec![wide_primary, wide_spacer];
    snapshot.lines[0].cells.resize(
        10,
        make_glyph(' ', Color::Default, Color::Default, AttrFlags::empty()),
    );

    snapshot.cursor_state = Some(CursorRenderState {
        x: 0,
        y: 0,
        shape: CursorShape::Block,
        cell_char_underneath: '中',
        cell_attributes_underneath: attr,
    });

    let commands = renderer.prepare_render_commands(&snapshot, &config, &platform_state);

    let has_cursor = commands.iter().any(|cmd| {
        if let RenderCommand::DrawTextRun {
            x, y, text, fg, bg, ..
        } = cmd
        {
            *x == 0
                && *y == 0
                && text.contains('中')
                && *fg == config.colors.background
                && *bg == config.colors.foreground
        } else {
            false
        }
    });

    assert!(
        has_cursor,
        "Cursor on wide character should render the wide character inverted"
    );
}

#[test]
fn partial_line_selection_affects_rendering() {
    let renderer = Renderer::new();
    let config = create_test_config();
    let platform_state = create_test_platform_state();

    let mut snapshot = create_snapshot_with_text(vec!["AAABBBCCC"]);
    snapshot.selection = Selection {
        range: Some(SelectionRange {
            start: Point { x: 3, y: 0 },
            end: Point { x: 5, y: 0 },
        }),
        mode: SelectionMode::Cell,
        is_active: true,
    };

    let commands = renderer.prepare_render_commands(&snapshot, &config, &platform_state);

    let text_run_count = count_command_type(&commands, "DrawTextRun");

    assert!(
        text_run_count > 0,
        "Should produce text run commands for line with selection"
    );
}

#[test]
fn line_with_only_spaces_produces_fill_rect() {
    let renderer = Renderer::new();
    let config = create_test_config();
    let platform_state = create_test_platform_state();

    let mut snapshot = create_empty_snapshot(20, 1);
    snapshot.lines[0].is_dirty = true;
    snapshot.lines[0].cells =
        vec![make_glyph(' ', Color::Default, Color::Default, AttrFlags::empty()); 20];

    let commands = renderer.prepare_render_commands(&snapshot, &config, &platform_state);

    let fill_rect_count = count_command_type(&commands, "FillRect");
    let text_run_count = count_command_type(&commands, "DrawTextRun");

    assert!(
        fill_rect_count > 0,
        "Line of spaces should produce at least one FillRect"
    );
    assert_eq!(
        text_run_count, 0,
        "Line of only spaces should not produce DrawTextRun commands"
    );
}

#[test]
fn empty_terminal_with_cursor_only_draws_cursor() {
    let renderer = Renderer::new();
    let config = create_test_config();
    let platform_state = create_test_platform_state();

    let mut snapshot = create_empty_snapshot(TEST_TERM_WIDTH, TEST_TERM_HEIGHT);
    for line in &mut snapshot.lines {
        line.is_dirty = true;
    }

    snapshot.cursor_state = Some(CursorRenderState {
        x: 0,
        y: 0,
        shape: CursorShape::Block,
        cell_char_underneath: ' ',
        cell_attributes_underneath: Attributes::default(),
    });

    let commands = renderer.prepare_render_commands(&snapshot, &config, &platform_state);

    let has_cursor = commands.iter().any(|cmd| {
        if let RenderCommand::DrawTextRun { x, y, fg, bg, .. } = cmd {
            *x == 0 && *y == 0 && *fg == config.colors.background && *bg == config.colors.foreground
        } else {
            false
        }
    });

    assert!(has_cursor, "Empty terminal should still render cursor");
}

#[test]
fn text_with_different_background_colors_not_coalesced() {
    let renderer = Renderer::new();
    let config = create_test_config();
    let platform_state = create_test_platform_state();

    let mut snapshot = create_empty_snapshot(10, 1);
    snapshot.lines[0].is_dirty = true;
    snapshot.lines[0].cells = vec![
        make_glyph(
            'A',
            Color::Default,
            Color::Named(NamedColor::Red),
            AttrFlags::empty(),
        ),
        make_glyph(
            'B',
            Color::Default,
            Color::Named(NamedColor::Red),
            AttrFlags::empty(),
        ),
        make_glyph(
            'C',
            Color::Default,
            Color::Named(NamedColor::Blue),
            AttrFlags::empty(),
        ),
        make_glyph(
            'D',
            Color::Default,
            Color::Named(NamedColor::Blue),
            AttrFlags::empty(),
        ),
    ];
    snapshot.lines[0].cells.resize(
        10,
        make_glyph(' ', Color::Default, Color::Default, AttrFlags::empty()),
    );

    let commands = renderer.prepare_render_commands(&snapshot, &config, &platform_state);

    let text_run_count = count_command_type(&commands, "DrawTextRun");
    assert!(
        text_run_count >= 2,
        "Different background colors should break coalescing"
    );
}

#[test]
fn single_character_with_bold_flag() {
    let renderer = Renderer::new();
    let config = create_test_config();
    let platform_state = create_test_platform_state();

    let mut snapshot = create_empty_snapshot(10, 1);
    snapshot.lines[0].is_dirty = true;
    snapshot.lines[0].cells = vec![make_glyph(
        'B',
        Color::Default,
        Color::Default,
        AttrFlags::BOLD,
    )];
    snapshot.lines[0].cells.resize(
        10,
        make_glyph(' ', Color::Default, Color::Default, AttrFlags::empty()),
    );

    let commands = renderer.prepare_render_commands(&snapshot, &config, &platform_state);

    let has_bold = commands.iter().any(|cmd| {
        if let RenderCommand::DrawTextRun { text, flags, .. } = cmd {
            text.contains('B') && flags.contains(AttrFlags::BOLD)
        } else {
            false
        }
    });

    assert!(has_bold, "Bold flag should be preserved in DrawTextRun");
}

#[test]
fn wide_character_followed_by_attribute_change() {
    let renderer = Renderer::new();
    let config = create_test_config();
    let platform_state = create_test_platform_state();

    let attr_default = Attributes::default();
    let _attr_red = Attributes {
        fg: Color::Named(NamedColor::Red),
        bg: Color::Default,
        flags: AttrFlags::empty(),
    };

    let (wide_primary, wide_spacer) = make_wide_char_pair('中', attr_default);

    let mut snapshot = create_empty_snapshot(10, 1);
    snapshot.lines[0].is_dirty = true;
    snapshot.lines[0].cells = vec![
        wide_primary,
        wide_spacer,
        make_glyph(
            'A',
            Color::Named(NamedColor::Red),
            Color::Default,
            AttrFlags::empty(),
        ),
    ];
    snapshot.lines[0].cells.resize(
        10,
        make_glyph(' ', Color::Default, Color::Default, AttrFlags::empty()),
    );

    let commands = renderer.prepare_render_commands(&snapshot, &config, &platform_state);

    let text_runs: Vec<_> = commands
        .iter()
        .filter_map(|cmd| {
            if let RenderCommand::DrawTextRun { text, .. } = cmd {
                Some(text.as_str())
            } else {
                None
            }
        })
        .collect();

    assert!(
        text_runs.len() >= 2,
        "Wide char and different-colored char should be separate runs"
    );
}
