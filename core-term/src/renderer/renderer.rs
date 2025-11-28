// myterm/src/renderer.rs

//! Defines the `Renderer`, responsible for translating a terminal's visual state
//! into a series of drawing commands for a platform-specific backend.
//!
//! The `Renderer` is a stateless component that takes a `TerminalSnapshot` and produces
//! a `Vec<RenderCommand>`. It is designed to be completely independent of the
//! underlying graphics backend, which receives and interprets the commands.
//! This design allows the core rendering logic to be shared across different
//! platforms (e.g., X11, Wayland, macOS).

use crate::color::Color;
use crate::config::Config;
use crate::glyph::{AttrFlags, Attributes, Glyph};
use crate::renderer::{PlatformState, RenderCommand};
use crate::term::snapshot::{Point, Selection, TerminalSnapshot};
use crate::term::unicode::get_char_display_width;
use log::{trace, warn};

/// A constant representing a single terminal cell consumed by a drawing operation.
const SINGLE_CELL_CONSUMED: usize = 1;

/// Translates a `TerminalSnapshot` into a list of `RenderCommand`s.
///
/// The `Renderer` optimizes the drawing process by:
/// 1. Only processing lines that are marked as "dirty" in the snapshot.
/// 2. Coalescing consecutive characters with identical attributes into a single draw command.
/// 3. Handling complex scenarios like wide characters and selections.
#[derive(Clone, Default)]
pub struct Renderer {}

impl Renderer {
    /// Creates a new, stateless `Renderer`.
    pub fn new() -> Self {
        Self {}
    }

    /// Resolves `Color::Default` for foreground and background, and handles the
    /// `AttrFlags::REVERSE` flag to determine the effective colors and flags for rendering.
    fn get_effective_colors_and_flags(
        &self,
        cell_fg: Color,
        cell_bg: Color,
        cell_flags: AttrFlags,
        config: &Config, // Added config parameter
    ) -> (Color, Color, AttrFlags) {
        let mut resolved_fg = match cell_fg {
            Color::Default => config.colors.foreground,
            c => c,
        };
        let mut resolved_bg = match cell_bg {
            Color::Default => config.colors.background,
            c => c,
        };
        let mut effective_flags = cell_flags;
        if cell_flags.contains(AttrFlags::REVERSE) {
            std::mem::swap(&mut resolved_fg, &mut resolved_bg);
            effective_flags.remove(AttrFlags::REVERSE);
        }
        (resolved_fg, resolved_bg, effective_flags)
    }

    /// Generates a list of `RenderCommand`s from a `TerminalSnapshot`.
    ///
    /// This is the primary method of the `Renderer`. It inspects the snapshot,
    /// identifies dirty lines, and generates an optimized sequence of drawing
    /// commands for the backend.
    ///
    /// # Arguments
    ///
    /// * `snapshot`: A `TerminalSnapshot` of the current terminal state.
    /// * `config`: The application configuration, used for resolving default colors.
    /// * `platform_state`: The current state of the platform, providing font and
    ///   display metrics.
    ///
    /// # Returns
    ///
    /// A `Vec<RenderCommand>` that the platform backend can execute to draw
    /// the terminal window.
    pub fn prepare_render_commands(
        &self,
        snapshot: &TerminalSnapshot,
        config: &Config,
        platform_state: &PlatformState,
    ) -> Vec<RenderCommand> {
        let mut commands: Vec<RenderCommand> = Vec::new();
        let (term_width, term_height) = snapshot.dimensions;

        trace!(
            "Renderer::prepare_render_commands called with platform_state: {:?}",
            platform_state
        );

        if term_width == 0 || term_height == 0 {
            trace!("Renderer::prepare_render_commands: Terminal dimensions zero, skipping draw.");
            return commands;
        }

        let mut lines_to_draw_indices: Vec<usize> = snapshot
            .lines
            .iter()
            .enumerate()
            .filter_map(|(i, line)| if line.is_dirty { Some(i) } else { None })
            .collect();

        if let Some(cursor_render_state) = &snapshot.cursor_state {
            if cursor_render_state.y < term_height {
                lines_to_draw_indices.push(cursor_render_state.y);
            }
        }

        lines_to_draw_indices.sort_unstable();
        lines_to_draw_indices.dedup();

        trace!(
            "Renderer::prepare_render_commands: Processing lines for content: {:?}",
            lines_to_draw_indices
        );

        for &y_abs in &lines_to_draw_indices {
            if y_abs >= term_height {
                warn!(
                    "Renderer::prepare_render_commands: Attempted to draw out-of-bounds line y={}",
                    y_abs
                );
                continue;
            }

            let Some(line_data) = snapshot.lines.get(y_abs) else {
                warn!(
                    "Renderer::prepare_render_commands: Snapshot missing line data for y={}",
                    y_abs
                );
                continue;
            };

            self.draw_line_content_from_slice(
                y_abs,
                term_width,
                &line_data.cells,
                &Some(snapshot.selection.clone()),
                config,
                &mut commands,
            );
        }

        self.draw_cursor_overlay(snapshot, config, &mut commands);

        commands
    }

    /// Helper to check if a cell (x,y) is within the selection range.
    fn is_cell_selected(
        &self,
        x: usize,
        y: usize,
        term_width: usize,
        selection: &Option<Selection>,
    ) -> bool {
        if term_width == 0 {
            return false;
        }
        if let Some(sel) = selection {
            if !sel.is_active {
                return false;
            }
            if let Some(range) = &sel.range {
                let mut start_offset = range.start.y * term_width + range.start.x;
                let mut end_offset = range.end.y * term_width + range.end.x;

                if start_offset > end_offset {
                    std::mem::swap(&mut start_offset, &mut end_offset);
                }

                let current_offset = y * term_width + x;
                return current_offset >= start_offset && current_offset <= end_offset;
            }
        }
        false
    }

    fn draw_line_content_from_slice(
        &self,
        y_abs: usize,
        term_width: usize,
        line_glyphs: &[Glyph],
        selection: &Option<Selection>,
        config: &Config,
        commands: &mut Vec<RenderCommand>,
    ) {
        let mut current_col: usize = 0;

        while current_col < term_width {
            let Some(start_glyph) = line_glyphs.get(current_col) else {
                warn!("draw_line_content_from_slice: Column {} reached end of glyph data for line {} (len {}). Filling rest of line.", current_col, y_abs, line_glyphs.len());
                if current_col < term_width {
                    commands.push(RenderCommand::FillRect {
                        x: current_col,
                        y: y_abs,
                        width: term_width - current_col,
                        height: 1,
                        color: config.colors.background,
                        is_selection_bg: false,
                    });
                }
                break;
            };

            let cells_consumed = match start_glyph {
                Glyph::WideSpacer { .. } => self.handle_wide_char_placeholder(
                    current_col,
                    y_abs,
                    line_glyphs,
                    config,
                    commands,
                ),
                Glyph::Single(cc) if cc.c == ' ' => self.draw_space_run_from_slice(
                    current_col,
                    y_abs,
                    term_width,
                    start_glyph,
                    line_glyphs,
                    selection,
                    config,
                    commands,
                ),
                _ => self.handle_text_segment(
                    current_col,
                    y_abs,
                    term_width,
                    start_glyph,
                    line_glyphs,
                    selection,
                    config,
                    commands,
                ),
            };

            let Ok(cells_consumed_val) = cells_consumed else {
                warn!(
                    "A draw segment failed at ({}, {}), char '{}'. Advancing by 1 to prevent loop.",
                    current_col,
                    y_abs,
                    start_glyph.display_char()
                );
                current_col += 1;
                continue;
            };

            if cells_consumed_val == 0 {
                warn!("A draw segment reported consuming 0 cells at ({}, {}), char '{}'. Advancing by 1 to prevent loop.", current_col, y_abs, start_glyph.display_char());
                current_col += 1;
            } else {
                current_col += cells_consumed_val;
            }
        }
    }

    fn handle_wide_char_placeholder(
        &self,
        current_col: usize,
        y_abs: usize,
        line_glyphs: &[Glyph],
        config: &Config,
        commands: &mut Vec<RenderCommand>,
    ) -> Result<usize, ()> {
        if current_col == 0 {
            warn!("Placeholder found at column 0 on line {}. This is unexpected. Using default background.", y_abs);
            self.draw_placeholder_cell(current_col, y_abs, config.colors.background, commands);
            return Ok(SINGLE_CELL_CONSUMED);
        }

        let Some(prev_glyph) = line_glyphs.get(current_col - 1) else {
            warn!("Placeholder at ({},{}) with current_col > 0 but no previous glyph. Using default BG.", current_col, y_abs);
            self.draw_placeholder_cell(current_col, y_abs, config.colors.background, commands);
            return Ok(SINGLE_CELL_CONSUMED);
        };

        let prev_char_for_width_check;
        let prev_attr_for_flags_check;

        match prev_glyph {
            Glyph::Single(cc) => {
                prev_char_for_width_check = cc.c;
                prev_attr_for_flags_check = cc.attr;
            }
            Glyph::WidePrimary(cc) => {
                prev_char_for_width_check = cc.c;
                prev_attr_for_flags_check = cc.attr;
            }
            Glyph::WideSpacer { .. } => {
                warn!("Placeholder at ({},{}) but previous char ('{}') is a WideSpacer. Using default BG.", current_col, y_abs, prev_glyph.display_char());
                self.draw_placeholder_cell(current_col, y_abs, config.colors.background, commands);
                return Ok(SINGLE_CELL_CONSUMED);
            }
        }

        if !(matches!(prev_glyph, Glyph::WidePrimary(_))
            || get_char_display_width(prev_char_for_width_check) == 2)
        {
            warn!("Placeholder at ({},{}) but previous char ('{}') is not WidePrimary or double-width. Using default BG.", current_col, y_abs, prev_glyph.display_char());
            self.draw_placeholder_cell(current_col, y_abs, config.colors.background, commands);
            return Ok(SINGLE_CELL_CONSUMED);
        }

        let (_, prev_eff_bg, _) = self.get_effective_colors_and_flags(
            prev_attr_for_flags_check.fg,
            prev_attr_for_flags_check.bg,
            prev_attr_for_flags_check.flags,
            config,
        );
        self.draw_placeholder_cell(current_col, y_abs, prev_eff_bg, commands);
        Ok(SINGLE_CELL_CONSUMED)
    }

    fn handle_text_segment(
        &self,
        current_col: usize,
        y_abs: usize,
        term_width: usize,
        start_glyph: &Glyph,
        line_glyphs: &[Glyph],
        selection: &Option<Selection>,
        config: &Config,
        commands: &mut Vec<RenderCommand>,
    ) -> Result<usize, ()> {
        let cells_consumed_by_text_run = self.draw_text_segment_from_slice(
            current_col,
            y_abs,
            term_width,
            start_glyph,
            line_glyphs,
            selection,
            config,
            commands,
        )?;

        let start_glyph_char = match start_glyph {
            Glyph::Single(cc) | Glyph::WidePrimary(cc) => cc.c,
            Glyph::WideSpacer { .. } => return Ok(cells_consumed_by_text_run),
        };
        let start_glyph_attr = match start_glyph {
            Glyph::Single(cc) | Glyph::WidePrimary(cc) => cc.attr,
            Glyph::WideSpacer { .. } => Attributes::default(),
        };

        if get_char_display_width(start_glyph_char) != 2 {
            return Ok(cells_consumed_by_text_run);
        }

        let placeholder_col = current_col + 1;

        if placeholder_col >= term_width {
            if cells_consumed_by_text_run != SINGLE_CELL_CONSUMED {
                warn!("    Line {}, Col {}: Wide char '{}' at end of line, but text_segment consumed {} cells (expected 1).", y_abs, current_col, start_glyph.display_char(), cells_consumed_by_text_run);
            }
            return Ok(cells_consumed_by_text_run);
        }

        let (_, placeholder_expected_bg, _) = self.get_effective_colors_and_flags(
            start_glyph_attr.fg,
            start_glyph_attr.bg,
            start_glyph_attr.flags,
            config,
        );

        let Some(glyph_at_placeholder) = line_glyphs.get(placeholder_col) else {
            warn!("    Line {}, Col {}: Could not get glyph at placeholder position for wide char '{}'. Forcing fill with primary's bg.", y_abs, placeholder_col, start_glyph.display_char());
            commands.push(RenderCommand::FillRect {
                x: placeholder_col,
                y: y_abs,
                width: 1,
                height: 1,
                color: placeholder_expected_bg,
                is_selection_bg: false,
            });
            return Ok(cells_consumed_by_text_run);
        };

        let is_correct_spacer = matches!(glyph_at_placeholder, Glyph::WideSpacer { .. });
        let placeholder_attrs = match glyph_at_placeholder {
            Glyph::Single(cc) | Glyph::WidePrimary(cc) => cc.attr,
            Glyph::WideSpacer { .. } => Attributes::default(),
        };

        let (_, current_placeholder_actual_bg, _) = self.get_effective_colors_and_flags(
            placeholder_attrs.fg,
            placeholder_attrs.bg,
            placeholder_attrs.flags,
            config,
        );

        let placeholder_is_fine =
            is_correct_spacer && current_placeholder_actual_bg == placeholder_expected_bg;

        if !placeholder_is_fine {
            trace!("    Line {}, Col {}: Filling placeholder for wide char '{}' (primary at col {}). Expected bg: {:?}, actual_char: '{}', actual_flags: {:?}, actual_bg: {:?}, is_correct_spacer: {}.", y_abs,placeholder_col,start_glyph.display_char(),current_col,placeholder_expected_bg,glyph_at_placeholder.display_char(),placeholder_attrs.flags,current_placeholder_actual_bg,is_correct_spacer);
            commands.push(RenderCommand::FillRect {
                x: placeholder_col,
                y: y_abs,
                width: 1,
                height: 1,
                color: placeholder_expected_bg,
                is_selection_bg: false,
            });
        } else {
            trace!("    Line {}, Col {}: WIDE_CHAR_SPACER for '{}' (primary at col {}) already has correct bg ({:?}). No fill needed.", y_abs, placeholder_col, start_glyph.display_char(), current_col, placeholder_expected_bg);
        }
        Ok(cells_consumed_by_text_run)
    }

    fn draw_placeholder_cell(
        &self,
        x: usize,
        y: usize,
        effective_bg: Color,
        commands: &mut Vec<RenderCommand>,
    ) {
        trace!(
            "    Line {}, Col {}: Placeholder. FillRect with bg={:?}",
            y,
            x,
            effective_bg
        );
        commands.push(RenderCommand::FillRect {
            x,
            y,
            width: 1,
            height: 1,
            color: effective_bg,
            is_selection_bg: false,
        });
    }

    fn draw_space_run_from_slice(
        &self,
        start_col: usize,
        y: usize,
        term_width: usize,
        start_glyph: &Glyph,
        line_glyphs: &[Glyph],
        selection: &Option<Selection>,
        config: &Config,
        commands: &mut Vec<RenderCommand>,
    ) -> Result<usize, ()> {
        let start_glyph_cc = match start_glyph {
            Glyph::Single(cc) => cc,
            _ => {
                debug_assert!(
                    false,
                    "draw_space_run_from_slice called with non-Single glyph"
                );
                commands.push(RenderCommand::FillRect {
                    x: start_col,
                    y,
                    width: 1,
                    height: 1,
                    color: config.colors.background,
                    is_selection_bg: false,
                });
                return Ok(SINGLE_CELL_CONSUMED);
            }
        };
        debug_assert!(
            start_glyph_cc.c == ' ',
            "draw_space_run_from_slice called with non-space char"
        );

        let (_, start_eff_bg, start_eff_flags) = self.get_effective_colors_and_flags(
            start_glyph_cc.attr.fg,
            start_glyph_cc.attr.bg,
            start_glyph_cc.attr.flags,
            config,
        );

        let mut space_run_len = 0;
        for x_offset in 0..(term_width - start_col) {
            let current_scan_col = start_col + x_offset;
            let Some(glyph_at_scan) = line_glyphs.get(current_scan_col) else {
                break;
            };

            match glyph_at_scan {
                Glyph::Single(cc) if cc.c == ' ' => {
                    let (_, current_scan_eff_bg, current_scan_flags) = self
                        .get_effective_colors_and_flags(
                            cc.attr.fg,
                            cc.attr.bg,
                            cc.attr.flags,
                            config,
                        );
                    if current_scan_eff_bg != start_eff_bg || current_scan_flags != start_eff_flags
                    {
                        break;
                    }
                    space_run_len += 1;
                }
                _ => break,
            }
        }

        if space_run_len == 0 {
            warn!("Renderer::draw_space_run_from_slice: space_run_len ended up as 0 at ({},{}). Drawing single space.", start_col, y);
            let is_selected = self.is_cell_selected(start_col, y, term_width, selection);
            commands.push(RenderCommand::FillRect {
                x: start_col,
                y,
                width: 1,
                height: 1,
                color: start_eff_bg,
                is_selection_bg: is_selected,
            });
            return Ok(SINGLE_CELL_CONSUMED);
        }

        let is_run_selected = self.is_cell_selected(start_col, y, term_width, selection);

        trace!("    Line {}, Col {}: Space run (len {}). FillRect with bg={:?}, flags={:?}, selected: {}", y, start_col, space_run_len, start_eff_bg, start_eff_flags, is_run_selected);
        commands.push(RenderCommand::FillRect {
            x: start_col,
            y,
            width: space_run_len,
            height: 1,
            color: start_eff_bg,
            is_selection_bg: is_run_selected,
        });
        Ok(space_run_len)
    }

    fn draw_text_segment_from_slice(
        &self,
        start_col: usize,
        y: usize,
        term_width: usize,
        start_glyph: &Glyph,
        line_glyphs: &[Glyph],
        selection: &Option<Selection>,
        config: &Config,
        commands: &mut Vec<RenderCommand>,
    ) -> Result<usize, ()> {
        let start_glyph_cc = match start_glyph {
            Glyph::Single(cc) | Glyph::WidePrimary(cc) => cc,
            Glyph::WideSpacer { .. } => {
                debug_assert!(false, "draw_text_segment_from_slice called with WideSpacer");
                return Ok(SINGLE_CELL_CONSUMED);
            }
        };
        debug_assert!(
            start_glyph_cc.c != ' ' && start_glyph_cc.c != '\0',
            "draw_text_segment_from_slice called with space or placeholder"
        );

        let (start_eff_fg, start_eff_bg, start_eff_flags) = self.get_effective_colors_and_flags(
            start_glyph_cc.attr.fg,
            start_glyph_cc.attr.bg,
            start_glyph_cc.attr.flags,
            config,
        );

        let mut run_text = String::new();
        let mut run_total_cell_width = 0;
        let mut current_scan_col = start_col;

        while current_scan_col < term_width {
            let Some(glyph_at_scan) = line_glyphs.get(current_scan_col) else {
                break;
            };

            let current_glyph_cc = match glyph_at_scan {
                Glyph::Single(cc) | Glyph::WidePrimary(cc) => cc,
                Glyph::WideSpacer { .. } => break,
            };

            if current_glyph_cc.c == ' ' {
                break;
            }

            let (current_glyph_eff_fg, current_glyph_eff_bg, current_glyph_flags) = self
                .get_effective_colors_and_flags(
                    current_glyph_cc.attr.fg,
                    current_glyph_cc.attr.bg,
                    current_glyph_cc.attr.flags,
                    config,
                );

            if !(current_glyph_eff_fg == start_eff_fg
                && current_glyph_eff_bg == start_eff_bg
                && current_glyph_flags == start_eff_flags)
            {
                break;
            }

            let char_display_width = get_char_display_width(current_glyph_cc.c);
            if char_display_width == 0 {
                trace!(
                    "    Line {}, Col {}: Appending zero-width char '{}'.",
                    y,
                    current_scan_col,
                    current_glyph_cc.c
                );
                run_text.push(current_glyph_cc.c);
                current_scan_col += SINGLE_CELL_CONSUMED;
                continue;
            }
            if (start_col + run_total_cell_width + char_display_width) > term_width {
                break;
            }

            run_text.push(current_glyph_cc.c);
            run_total_cell_width += char_display_width;
            current_scan_col += char_display_width;
        }

        if run_text.is_empty() {
            let initial_char_width = get_char_display_width(start_glyph_cc.c);
            warn!(
                "    Line {}, Col {}: Text run for char '{}' (width {}) was empty. Consumed: {}",
                y,
                start_col,
                start_glyph.display_char(),
                initial_char_width,
                initial_char_width.max(SINGLE_CELL_CONSUMED)
            );
            return Ok(initial_char_width.max(SINGLE_CELL_CONSUMED));
        }

        let is_run_selected = self.is_cell_selected(start_col, y, term_width, selection);

        trace!("    Line {}, Col {}: Text run: '{}' ({} cells). DrawTextRun with style=fg:{:?},bg:{:?},flags:{:?},selected:{}", y, start_col, run_text, run_total_cell_width, start_eff_fg, start_eff_bg, start_eff_flags, is_run_selected);
        commands.push(RenderCommand::DrawTextRun {
            x: start_col,
            y,
            text: run_text,
            fg: start_eff_fg,
            bg: start_eff_bg,
            flags: start_eff_flags,
            is_selected: is_run_selected,
        });
        Ok(run_total_cell_width)
    }

    fn draw_cursor_overlay(
        &self,
        snapshot: &TerminalSnapshot,
        config: &Config,
        commands: &mut Vec<RenderCommand>,
    ) {
        let Some(cursor_state) = &snapshot.cursor_state else {
            return;
        };
        let (term_width, term_height) = snapshot.dimensions;
        let cursor_abs_x = cursor_state.x;
        let cursor_abs_y = cursor_state.y;

        if !(cursor_abs_x < term_width && cursor_abs_y < term_height) {
            warn!(
                "Cursor at ({}, {}) is out of bounds ({}x{}). Not drawing.",
                cursor_abs_x, cursor_abs_y, term_width, term_height
            );
            return;
        }

        let Some(glyph_under_cursor) = snapshot.get_glyph(Point {
            x: cursor_abs_x,
            y: cursor_abs_y,
        }) else {
            warn!(
                "Could not get glyph under cursor at ({}, {}) from snapshot. Not drawing cursor.",
                cursor_abs_x, cursor_abs_y
            );
            return;
        };

        let char_to_draw_at_cursor: char;
        let original_attrs_at_cursor: Attributes;
        let mut physical_cursor_x_for_draw = cursor_abs_x;

        match glyph_under_cursor {
            Glyph::Single(cc) => {
                char_to_draw_at_cursor = cc.c;
                original_attrs_at_cursor = cc.attr;
            }
            Glyph::WidePrimary(cc) => {
                char_to_draw_at_cursor = cc.c;
                original_attrs_at_cursor = cc.attr;
            }
            Glyph::WideSpacer { .. } => {
                if cursor_abs_x == 0 {
                    warn!("Cursor on WideSpacer at column 0 ({}, {}). This is unexpected. Not drawing cursor.", cursor_abs_x, cursor_abs_y);
                    return;
                }
                physical_cursor_x_for_draw = cursor_abs_x - 1;

                let Some(primary_glyph_enum) = snapshot.get_glyph(Point {
                    x: physical_cursor_x_for_draw,
                    y: cursor_abs_y,
                }) else {
                    warn!("Could not get primary part of wide char at ({},{}) for cursor on spacer. Not drawing.", physical_cursor_x_for_draw, cursor_abs_y);
                    return;
                };

                match primary_glyph_enum {
                    Glyph::WidePrimary(cc) => {
                        char_to_draw_at_cursor = cc.c;
                        original_attrs_at_cursor = cc.attr;
                        trace!("    Cursor on placeholder at ({},{}), using primary char '{}' from col {}", cursor_abs_x, cursor_abs_y, cc.c, physical_cursor_x_for_draw);
                    }
                    _ => {
                        warn!("Expected WidePrimary at ({},{}) for cursor on spacer, but found {:?}. Not drawing.", physical_cursor_x_for_draw, cursor_abs_y, primary_glyph_enum);
                        return;
                    }
                }
            }
        }

        let (resolved_original_fg, resolved_original_bg, resolved_original_flags) = self
            .get_effective_colors_and_flags(
                original_attrs_at_cursor.fg,
                original_attrs_at_cursor.bg,
                original_attrs_at_cursor.flags,
                config,
            );
        trace!(
            "    Original cell effective attrs for cursor: fg={:?}, bg={:?}, flags={:?}",
            resolved_original_fg,
            resolved_original_bg,
            resolved_original_flags
        );

        let cursor_char_fg = resolved_original_bg;
        let cursor_cell_bg = resolved_original_fg;
        let cursor_display_flags = resolved_original_flags;

        let final_char_to_draw_for_cursor = if char_to_draw_at_cursor == '\0'
            || get_char_display_width(char_to_draw_at_cursor) == 0
        {
            ' '
        } else {
            char_to_draw_at_cursor
        };

        let is_cursor_pos_selected = self.is_cell_selected(
            physical_cursor_x_for_draw,
            cursor_abs_y,
            term_width,
            &Some(snapshot.selection.clone()),
        );

        trace!("    Drawing cursor overlay: char='{}' at physical ({},{}) with style: fg:{:?},bg:{:?},flags:{:?},selected:{}", final_char_to_draw_for_cursor, physical_cursor_x_for_draw, cursor_abs_y, cursor_char_fg, cursor_cell_bg, cursor_display_flags, is_cursor_pos_selected);
        commands.push(RenderCommand::DrawTextRun {
            x: physical_cursor_x_for_draw,
            y: cursor_abs_y,
            text: final_char_to_draw_for_cursor.to_string(),
            fg: cursor_char_fg,
            bg: cursor_cell_bg,
            flags: cursor_display_flags,
            is_selected: is_cursor_pos_selected,
        });
    }
}

// Tests are in src/renderer/renderer_tests.rs
