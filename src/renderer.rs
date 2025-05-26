// myterm/src/renderer.rs

//! This module defines the `Renderer`.
//!
//! The `Renderer`'s primary responsibility is to translate the visual state of the
//! `TerminalEmulator` (obtained via a `RenderSnapshot`) into a series of
//! abstract drawing commands that can be executed by a `Driver`. It is designed
//! to be backend-agnostic. It defines default foreground and background colors
//! for resolving `Color::Default` from glyph attributes when rendering.

use crate::backends::{CellCoords, CellRect, Driver, TextRunStyle};
use crate::color::{Color, NamedColor};
use crate::glyph::{AttrFlags, Attributes, Glyph};
use crate::term::unicode::get_char_display_width;
// Assuming RenderSnapshot, SnapshotLine, CursorRenderState, and Point are defined in crate::term
use crate::term::{Point, RenderSnapshot};

use anyhow::Result; // For error handling.
use log::{trace, warn}; // For logging.

/// Default foreground color used by the renderer when a glyph specifies `Color::Default`.
pub const RENDERER_DEFAULT_FG: Color = Color::Named(NamedColor::White);
/// Default background color used by the renderer when a glyph specifies `Color::Default`.
pub const RENDERER_DEFAULT_BG: Color = Color::Named(NamedColor::Black);

/// Constant representing a single terminal cell consumed by a drawing operation.
const SINGLE_CELL_CONSUMED: usize = 1;


/*
Assumed definitions from src/term/mod.rs or similar (based on NORTH_STAR.md):

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Point {
    pub x: usize,
    pub y: usize,
}

#[derive(Debug, Clone)]
pub struct SnapshotLine {
    pub is_dirty: bool,
    pub cells: Vec<Glyph>,
}

// CursorRenderState no longer needs `is_visible`. Its presence in `Option` signifies visibility.
#[derive(Debug, Clone, Copy)]
pub struct CursorRenderState {
    pub x: usize, // Physical x of the cell the cursor is on/starts at
    pub y: usize, // Physical y
    // pub shape: term::cursor::CursorShape, // If needed by renderer for visual style
}

#[derive(Debug, Clone)]
pub struct RenderSnapshot {
    pub dimensions: (usize, usize), // cols, rows
    pub lines: Vec<SnapshotLine>,
    pub cursor_state: Option<CursorRenderState>, // If None, cursor is not drawn
    // pub selection_state: Option<SelectionRenderState>, // Not used in current renderer logic
}

impl RenderSnapshot {
    // Helper to get a glyph from the snapshot's lines
    pub fn get_glyph(&self, point: Point) -> Option<&Glyph> {
        self.lines.get(point.y)?.cells.get(point.x)
    }
}
*/


/// The `Renderer` translates `RenderSnapshot` state into abstract drawing commands
/// for a `Driver`.
///
/// It optimizes drawing by only processing lines marked as dirty in the snapshot
/// and by coalescing character runs with identical attributes.
#[derive(Clone, Default)]
pub struct Renderer {}

impl Renderer {
    /// Creates a new `Renderer` instance.
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
    ) -> (Color, Color, AttrFlags) {
        let mut resolved_fg = match cell_fg {
            Color::Default => RENDERER_DEFAULT_FG,
            c => c,
        };
        let mut resolved_bg = match cell_bg {
            Color::Default => RENDERER_DEFAULT_BG,
            c => c,
        };
        let mut effective_flags = cell_flags;
        if cell_flags.contains(AttrFlags::REVERSE) {
            std::mem::swap(&mut resolved_fg, &mut resolved_bg);
            effective_flags.remove(AttrFlags::REVERSE);
        }
        (resolved_fg, resolved_bg, effective_flags)
    }

    /// Draws the current state of the terminal using a `RenderSnapshot`.
    ///
    /// This method iterates over lines marked as dirty in the snapshot,
    /// draws their content, and then draws the cursor if its state is present
    /// in the snapshot. Finally, it presents the changes via the `Driver`.
    ///
    /// # Arguments
    /// * `snapshot`: A `RenderSnapshot` of the current terminal state.
    /// * `driver`: A mutable reference to a `Driver` implementation.
    ///
    /// # Returns
    /// * `Result<()>`: Ok if drawing and presentation were successful, or an error.
    pub fn draw(
        &self,
        snapshot: RenderSnapshot,
        driver: &mut dyn Driver,
    ) -> Result<()> {
        let (term_width, term_height) = snapshot.dimensions;

        // Guard: Avoid drawing if terminal dimensions are invalid.
        if term_width == 0 || term_height == 0 {
            trace!("Renderer::draw: Terminal dimensions zero, skipping draw.");
            return Ok(());
        }

        // Collect indices of dirty lines from the snapshot.
        let mut lines_to_draw_indices: Vec<usize> = snapshot
            .lines
            .iter()
            .enumerate()
            .filter_map(|(i, line)| if line.is_dirty { Some(i) } else { None })
            .collect();

        // If the cursor is to be drawn, ensure its line is also processed.
        if let Some(cursor_render_state) = &snapshot.cursor_state {
            if cursor_render_state.y < term_height {
                lines_to_draw_indices.push(cursor_render_state.y);
            }
        }
        
        // Sort and deduplicate the line indices.
        lines_to_draw_indices.sort_unstable();
        lines_to_draw_indices.dedup();

        trace!(
            "Renderer::draw: Processing lines for content: {:?}",
            lines_to_draw_indices
        );

        for &y_abs in &lines_to_draw_indices {
            // Guard: Ensure line index is within bounds (already implicitly handled by .get later, but good for clarity).
            if y_abs >= term_height {
                warn!("Renderer::draw: Attempted to draw out-of-bounds line y={}", y_abs);
                continue;
            }
            
            let Some(line_data) = snapshot.lines.get(y_abs) else {
                warn!("Renderer::draw: Snapshot missing line data for y={}", y_abs);
                continue; // Should not happen if y_abs is from snapshot lines enumeration
            };

            self.draw_line_content_from_slice(
                y_abs,
                term_width,
                &line_data.cells,
                driver,
            )?;
        }

        // Overlay the cursor if its state is present in the snapshot.
        self.draw_cursor_overlay(&snapshot, driver)?;

        // Always present changes at the end of a draw cycle.
        trace!("Renderer::draw: Presenting changes.");
        driver.present()?;
        Ok(())
    }

    /// Draws the content of a single terminal line using a slice of its glyphs.
    fn draw_line_content_from_slice(
        &self,
        y_abs: usize,
        term_width: usize,
        line_glyphs: &[Glyph],
        driver: &mut dyn Driver,
    ) -> Result<()> {
        trace!("Renderer::draw_line_content_from_slice: Drawing line y={}", y_abs);
        let mut current_col: usize = 0;

        while current_col < term_width {
            let Some(start_glyph) = line_glyphs.get(current_col) else {
                // This implies line_glyphs.len() < term_width.
                // This situation should ideally be handled by ensuring snapshot lines are always full width.
                // If they can be shorter, this fill is a reasonable fallback.
                warn!("draw_line_content_from_slice: Column {} reached end of glyph data for line {} (len {}). Filling rest of line.",
                    current_col, y_abs, line_glyphs.len());
                if current_col < term_width { // If there's still part of the conceptual line to fill
                    let rect = CellRect { x: current_col, y: y_abs, width: term_width - current_col, height: 1};
                    driver.fill_rect(rect, RENDERER_DEFAULT_BG)?;
                }
                break; // Exit the loop as there are no more glyphs in this line's data.
            };

            let cells_consumed = match start_glyph.c {
                '\0' => self.handle_wide_char_placeholder(current_col, y_abs, line_glyphs, driver)?,
                ' ' => self.draw_space_run_from_slice(current_col, y_abs, term_width, start_glyph, line_glyphs, driver)?,
                _ => self.handle_text_segment(current_col, y_abs, term_width, start_glyph, line_glyphs, driver)?,
            };

            // Ensure progress is made in the loop.
            if cells_consumed == 0 {
                // This can happen if a helper function was called with a glyph it didn't expect
                // (e.g., draw_space_run with a non-space) and returned 0, or if a ZWC was encountered alone.
                warn!("A draw segment reported consuming 0 cells at ({}, {}), char '{}'. Advancing by 1 to prevent loop.",
                    current_col, y_abs, start_glyph.c);
                current_col += 1; // Force advancement
            } else {
                current_col += cells_consumed;
            }
        }
        Ok(())
    }


    /// Handles drawing a placeholder for the second cell of a wide character.
    ///
    /// The background color is determined from the primary (first) cell of the wide character.
    ///
    /// # Returns
    /// `Ok(usize)`: The number of terminal cells consumed by this operation (always 1 for a placeholder).
    fn handle_wide_char_placeholder(
        &self,
        current_col: usize,
        y_abs: usize,
        line_glyphs: &[Glyph],
        driver: &mut dyn Driver,
    ) -> Result<usize> {
        // Guard: If placeholder is at column 0, it's unexpected. Use default background.
        if current_col == 0 {
            warn!("Placeholder found at column 0 on line {}. This is unexpected. Using default background.", y_abs);
            self.draw_placeholder_cell(current_col, y_abs, RENDERER_DEFAULT_BG, driver)?;
            return Ok(SINGLE_CELL_CONSUMED);
        }

        // Get the preceding glyph, which should be the primary part of the wide character.
        let Some(prev_glyph) = line_glyphs.get(current_col - 1) else {
            warn!("Placeholder at ({},{}) with current_col > 0 but no previous glyph. Using default BG.", current_col, y_abs);
            self.draw_placeholder_cell(current_col, y_abs, RENDERER_DEFAULT_BG, driver)?;
            return Ok(SINGLE_CELL_CONSUMED);
        };

        // Guard: Check if the previous glyph is indeed the primary part of a wide character.
        // If not, this placeholder is orphaned or misplaced.
        if !(prev_glyph.attr.flags.contains(AttrFlags::WIDE_CHAR_PRIMARY) || get_char_display_width(prev_glyph.c) == 2) {
            warn!("Placeholder at ({},{}) but previous char ('{}') is not WIDE_CHAR_PRIMARY or double-width. Using default BG.", current_col, y_abs, prev_glyph.c);
            self.draw_placeholder_cell(current_col, y_abs, RENDERER_DEFAULT_BG, driver)?;
            return Ok(SINGLE_CELL_CONSUMED);
        }

        // Determine the background color from the primary wide character.
        let (_, prev_eff_bg, _) = self.get_effective_colors_and_flags(
            prev_glyph.attr.fg,
            prev_glyph.attr.bg,
            prev_glyph.attr.flags,
        );
        self.draw_placeholder_cell(current_col, y_abs, prev_eff_bg, driver)?;
        Ok(SINGLE_CELL_CONSUMED)
    }

    /// Handles drawing a text segment and its potential wide character placeholder.
    ///
    /// After drawing the main text run, this function checks if the `start_glyph`
    /// was a wide character. If so, it ensures the subsequent cell (the placeholder)
    /// is filled with the correct background color.
    ///
    /// # Returns
    /// `Ok(usize)`: The number of terminal cells consumed by the primary text run.
    fn handle_text_segment(
        &self,
        current_col: usize,
        y_abs: usize,
        term_width: usize,
        start_glyph: &Glyph,
        line_glyphs: &[Glyph],
        driver: &mut dyn Driver,
    ) -> Result<usize> {
        let cells_consumed_by_text_run = self.draw_text_segment_from_slice(
            current_col,
            y_abs,
            term_width,
            start_glyph,
            line_glyphs,
            driver,
        )?;

        // If the start_glyph was not a wide character, no placeholder handling is needed.
        if get_char_display_width(start_glyph.c) != 2 {
            return Ok(cells_consumed_by_text_run);
        }

        // --- Wide Character Placeholder Styling ---
        let placeholder_col = current_col + 1;

        // Guard: Ensure there's actually space on the line for the placeholder.
        if placeholder_col >= term_width {
            if cells_consumed_by_text_run != SINGLE_CELL_CONSUMED {
                 warn!(
                    "    Line {}, Col {}: Wide char '{}' at end of line, but text_segment consumed {} cells (expected 1).",
                    y_abs, current_col, start_glyph.c, cells_consumed_by_text_run
                );
            }
            return Ok(cells_consumed_by_text_run); 
        }

        let (_, placeholder_expected_bg, _) = self.get_effective_colors_and_flags(
            start_glyph.attr.fg,
            start_glyph.attr.bg,
            start_glyph.attr.flags,
        );

        let Some(glyph_at_placeholder) = line_glyphs.get(placeholder_col) else {
            warn!(
                "    Line {}, Col {}: Could not get glyph at placeholder position for wide char '{}'. Forcing fill with primary's bg.",
                y_abs, placeholder_col, start_glyph.c
            );
            driver.fill_rect(CellRect { x: placeholder_col, y: y_abs, width: 1, height: 1 }, placeholder_expected_bg)?;
            return Ok(cells_consumed_by_text_run);
        };

        let is_correct_spacer = glyph_at_placeholder.c == '\0' && glyph_at_placeholder.attr.flags.contains(AttrFlags::WIDE_CHAR_SPACER);
        let (_, current_placeholder_actual_bg, _) = self.get_effective_colors_and_flags(
            glyph_at_placeholder.attr.fg,
            glyph_at_placeholder.attr.bg,
            glyph_at_placeholder.attr.flags
        );

        let placeholder_is_fine = is_correct_spacer && current_placeholder_actual_bg == placeholder_expected_bg;

        if !placeholder_is_fine {
            // Placeholder needs to be filled. Log details at trace level.
            trace!(
                "    Line {}, Col {}: Filling placeholder for wide char '{}' (primary at col {}). Expected bg: {:?}, actual_char: '{}', actual_flags: {:?}, actual_bg: {:?}, is_correct_spacer: {}.",
                y_abs, placeholder_col, start_glyph.c, current_col, placeholder_expected_bg,
                glyph_at_placeholder.c, glyph_at_placeholder.attr.flags, current_placeholder_actual_bg,
                is_correct_spacer
            );
            driver.fill_rect(CellRect { x: placeholder_col, y: y_abs, width: 1, height: 1 }, placeholder_expected_bg)?;
        } else {
            trace!(
                "    Line {}, Col {}: WIDE_CHAR_SPACER for '{}' (primary at col {}) already has correct bg ({:?}). No fill needed.",
                y_abs, placeholder_col, start_glyph.c, current_col, placeholder_expected_bg
            );
        }
        
        Ok(cells_consumed_by_text_run)
    }


    /// Draws a placeholder cell (typically the second half of a wide character)
    /// by filling it with the effective background color.
    ///
    /// # Returns
    /// `Ok(usize)`: The number of terminal cells consumed by this operation (always 1).
    fn draw_placeholder_cell(
        &self,
        x: usize,
        y: usize,
        effective_bg: Color,
        driver: &mut dyn Driver,
    ) -> Result<usize> {
        let rect = CellRect { x, y, width: 1, height: 1 };
        trace!("    Line {}, Col {}: Placeholder. FillRect with bg={:?}", y, x, effective_bg);
        driver.fill_rect(rect, effective_bg)?;
        Ok(SINGLE_CELL_CONSUMED)
    }

    /// Identifies and draws a contiguous run of space characters from a slice.
    ///
    /// # Returns
    /// `Ok(usize)`: The number of terminal cells consumed by this run of spaces.
    fn draw_space_run_from_slice(
        &self,
        start_col: usize,
        y: usize,
        term_width: usize,
        start_glyph: &Glyph,
        line_glyphs: &[Glyph],
        driver: &mut dyn Driver,
    ) -> Result<usize> {
        debug_assert!(start_glyph.c == ' ', "draw_space_run_from_slice called with non-space start_glyph");

        let (_, start_eff_bg, start_eff_flags) = self.get_effective_colors_and_flags(
            start_glyph.attr.fg,
            start_glyph.attr.bg,
            start_glyph.attr.flags,
        );

        let mut space_run_len = 0;
        for x_offset in 0..(term_width - start_col) {
            let current_scan_col = start_col + x_offset;
            let Some(glyph_at_scan) = line_glyphs.get(current_scan_col) else { break; };

            if glyph_at_scan.c != ' ' {
                break;
            }

            let (_, current_scan_eff_bg, current_scan_flags) = self.get_effective_colors_and_flags(
                glyph_at_scan.attr.fg,
                glyph_at_scan.attr.bg,
                glyph_at_scan.attr.flags,
            );

            if current_scan_eff_bg != start_eff_bg || current_scan_flags != start_eff_flags {
                break;
            }
            space_run_len += 1;
        }

        if space_run_len == 0 {
             warn!("Renderer::draw_space_run_from_slice: space_run_len is 0 at ({},{}). Drawing single space.", start_col, y);
             let rect = CellRect { x: start_col, y, width: 1, height: 1 };
             driver.fill_rect(rect, start_eff_bg)?;
             return Ok(SINGLE_CELL_CONSUMED);
        }

        let rect = CellRect { x: start_col, y, width: space_run_len, height: 1 };
        trace!(
            "    Line {}, Col {}: Space run (len {}). FillRect with bg={:?}, flags={:?}",
            y, start_col, space_run_len, start_eff_bg, start_eff_flags
        );
        driver.fill_rect(rect, start_eff_bg)?;
        Ok(space_run_len)
    }

    /// Identifies and draws a contiguous run of non-space, non-placeholder text
    /// characters that share the same effective attributes.
    ///
    /// # Returns
    /// `Ok(usize)`: The number of terminal cells consumed by this text segment,
    ///              accounting for character widths (e.g., wide characters consume 2 cells).
    fn draw_text_segment_from_slice(
        &self,
        start_col: usize,
        y: usize,
        term_width: usize,
        start_glyph: &Glyph,
        line_glyphs: &[Glyph],
        driver: &mut dyn Driver,
    ) -> Result<usize> {
        debug_assert!(start_glyph.c != ' ' && start_glyph.c != '\0',
            "draw_text_segment_from_slice called with space or placeholder start_glyph");

        let (start_eff_fg, start_eff_bg, start_eff_flags) = self.get_effective_colors_and_flags(
            start_glyph.attr.fg,
            start_glyph.attr.bg,
            start_glyph.attr.flags,
        );

        let mut run_text = String::new();
        let mut run_total_cell_width = 0; 
        let mut current_scan_col = start_col; 

        while current_scan_col < term_width {
            let Some(glyph_at_scan) = line_glyphs.get(current_scan_col) else { break; };

            if glyph_at_scan.c == ' ' || glyph_at_scan.c == '\0' {
                break;
            }
            
            let (current_glyph_eff_fg, current_glyph_eff_bg, current_glyph_flags) = self
                .get_effective_colors_and_flags(
                    glyph_at_scan.attr.fg,
                    glyph_at_scan.attr.bg,
                    glyph_at_scan.attr.flags,
                );

            if !(current_glyph_eff_fg == start_eff_fg
                && current_glyph_eff_bg == start_eff_bg
                && current_glyph_flags == start_eff_flags)
            {
                break;
            }

            let char_display_width = get_char_display_width(glyph_at_scan.c);
            
            if char_display_width == 0 { 
                trace!("    Line {}, Col {}: Appending zero-width char '{}'.", y, current_scan_col, glyph_at_scan.c);
                run_text.push(glyph_at_scan.c);
                current_scan_col += SINGLE_CELL_CONSUMED; 
                continue;
            }

            if (start_col + run_total_cell_width + char_display_width) > term_width {
                break; 
            }

            run_text.push(glyph_at_scan.c);
            run_total_cell_width += char_display_width;
            current_scan_col += char_display_width; 
        }

        if run_text.is_empty() {
            let initial_char_width = get_char_display_width(start_glyph.c);
            warn!("    Line {}, Col {}: Text run for char '{}' (width {}) was empty. Consumed: {}",
                y, start_col, start_glyph.c, initial_char_width, initial_char_width.max(SINGLE_CELL_CONSUMED));
            return Ok(initial_char_width.max(SINGLE_CELL_CONSUMED));
        }

        let coords = CellCoords { x: start_col, y };
        let style = TextRunStyle {
            fg: start_eff_fg,
            bg: start_eff_bg,
            flags: start_eff_flags,
        };
        trace!(
            "    Line {}, Col {}: Text run: '{}' ({} cells). DrawTextRun with style={:?}",
            y, start_col, run_text, run_total_cell_width, style
        );
        driver.draw_text_run(coords, &run_text, style)?;
        Ok(run_total_cell_width)
    }

    /// Draws the terminal cursor overlay using data from the `RenderSnapshot`.
    fn draw_cursor_overlay(
        &self,
        snapshot: &RenderSnapshot,
        driver: &mut dyn Driver,
    ) -> Result<()> {
        let Some(cursor_state) = &snapshot.cursor_state else {
            return Ok(())
        };

        let (term_width, term_height) = snapshot.dimensions;
        let cursor_abs_x = cursor_state.x;
        let cursor_abs_y = cursor_state.y;

        if !(cursor_abs_x < term_width && cursor_abs_y < term_height) {
            warn!("Cursor at ({}, {}) is out of bounds ({}x{}). Not drawing.",
                cursor_abs_x, cursor_abs_y, term_width, term_height);
            return Ok(());
        }

        let Some(glyph_under_cursor) = snapshot.get_glyph(Point { x: cursor_abs_x, y: cursor_abs_y }) else {
            warn!("Could not get glyph under cursor at ({}, {}) from snapshot. Not drawing cursor.", cursor_abs_x, cursor_abs_y);
            return Ok(());
        };

        let physical_cursor_x_for_draw: usize;
        let char_to_draw_at_cursor: char;
        let original_attrs_at_cursor: Attributes;

        if glyph_under_cursor.c == '\0' && cursor_abs_x > 0 {
            let Some(first_half_glyph) = snapshot.get_glyph(Point {x: cursor_abs_x - 1, y: cursor_abs_y}) else {
                warn!("Could not get first half of wide char at ({},{}) for cursor. Not drawing.", cursor_abs_x -1, cursor_abs_y);
                return Ok(());
            };
            char_to_draw_at_cursor = first_half_glyph.c;
            original_attrs_at_cursor = first_half_glyph.attr;
            physical_cursor_x_for_draw = cursor_abs_x - 1;
            trace!("    Cursor on placeholder, using first half: char='{}' from col {}",
                if char_to_draw_at_cursor == '\0' {'?'} else {char_to_draw_at_cursor}, physical_cursor_x_for_draw);
        } else {
            char_to_draw_at_cursor = glyph_under_cursor.c;
            original_attrs_at_cursor = glyph_under_cursor.attr;
            physical_cursor_x_for_draw = cursor_abs_x;
        }

        let (resolved_original_fg, resolved_original_bg, resolved_original_flags) = self
            .get_effective_colors_and_flags(
                original_attrs_at_cursor.fg,
                original_attrs_at_cursor.bg,
                original_attrs_at_cursor.flags,
            );
        trace!("    Original cell effective attrs for cursor: fg={:?}, bg={:?}, flags={:?}",
            resolved_original_fg, resolved_original_bg, resolved_original_flags);

        let cursor_char_fg = resolved_original_bg;
        let cursor_cell_bg = resolved_original_fg;
        let cursor_display_flags = resolved_original_flags;

        let coords = CellCoords { x: physical_cursor_x_for_draw, y: cursor_abs_y };
        let style = TextRunStyle {
            fg: cursor_char_fg,
            bg: cursor_cell_bg,
            flags: cursor_display_flags,
        };

        let final_char_to_draw_for_cursor = if char_to_draw_at_cursor == '\0' { 
            ' ' 
        } else { 
            char_to_draw_at_cursor 
        };
        trace!("    Drawing cursor overlay: char='{}' at physical ({},{}) with style: {:?}",
            final_char_to_draw_for_cursor, physical_cursor_x_for_draw, cursor_abs_y, style);
        driver.draw_text_run(coords, &final_char_to_draw_for_cursor.to_string(), style)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests;
