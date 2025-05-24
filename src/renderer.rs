// myterm/src/renderer.rs

//! This module defines the `Renderer`, responsible for translating the terminal's
//! visual state into drawing commands for a backend driver.

use crate::backends::{CellCoords, CellRect, Driver, TextRunStyle};
use crate::color::{Color, NamedColor};
use crate::glyph::{AttrFlags, Attributes, Glyph};
use crate::term::unicode::get_char_display_width;
use crate::term::{SelectionRenderState, TerminalInterface}; // Added SelectionRenderState

use anyhow::Result; 
use log::{debug, trace, warn}; 
use std::collections::HashSet;

/// Default foreground color used by the renderer when a glyph specifies `Color::Default`.
pub const RENDERER_DEFAULT_FG: Color = Color::Named(NamedColor::White);
/// Default background color used by the renderer when a glyph specifies `Color::Default`.
pub const RENDERER_DEFAULT_BG: Color = Color::Named(NamedColor::Black);

/// The `Renderer` translates `TerminalEmulator` state into abstract drawing commands
/// for a `Driver`.
///
/// It optimizes drawing by:
/// - Only processing lines marked as dirty by the terminal or affected by cursor/selection changes.
/// - Coalescing character runs with identical attributes into single draw calls.
/// - Handling selection display by appropriately flagging calls to the driver.
#[derive(Clone)]
pub struct Renderer {
    /// Tracks if this is the very first draw call.
    /// If `true`, a full screen clear and redraw is typically performed.
    /// This is set to `false` after the first successful draw.
    pub first_draw: bool,
}

impl Renderer {
    /// Creates a new `Renderer` instance.
    pub fn new() -> Self {
        Self { first_draw: true }
    }

    /// Resolves `Color::Default` for foreground and background, and handles the
    /// `AttrFlags::REVERSE` flag to determine the effective colors and flags for rendering.
    /// The `AttrFlags::REVERSE` flag is consumed in this process.
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
    
    /// Determines if a given cell `(col, row)` is part of the current selection.
    ///
    /// # Arguments
    /// * `col`: The 0-based column index of the cell.
    /// * `row`: The 0-based row index of the cell.
    /// * `selection`: An `Option<&SelectionRenderState>` representing the active selection.
    ///   If `None`, no selection is active, and the cell cannot be selected.
    /// * `term_width`: The total width of the terminal in cells. Used for `Normal` mode
    ///   to determine full-line selections between start and end rows.
    ///
    /// # Returns
    /// `true` if the cell is selected, `false` otherwise.
    ///
    /// # Behavior
    /// - **Normalization:** The selection start and end coordinates are normalized to ensure
    ///   `start_row <= end_row` and, if on the same line, `start_col <= end_col`.
    /// - **Normal Mode:**
    ///   - A cell is selected if its row is between `start_row` and `end_row` (inclusive).
    ///   - If on `start_row`, `col` must be `>= start_col`.
    ///   - If on `end_row`, `col` must be `<= end_col`.
    ///   - If on a row strictly between `start_row` and `end_row`, all columns are selected.
    /// - **Block Mode (Conceptual):**
    ///   - A cell is selected if its `row` is between the normalized `start_row` and `end_row`,
    ///     AND its `col` is between the min and max column of the original selection points.
    ///     (Note: Current implementation for Block Mode is identical to Normal due to TODOs elsewhere).
    fn is_cell_selected(
        &self,
        col: usize,
        row: usize,
        selection: Option<&SelectionRenderState>,
        _term_width: usize, // Currently unused, but kept for future Block mode or complex Normal mode logic
    ) -> bool {
        if let Some(sel) = selection {
            // Normalize coordinates: start should be lexicographically before or at end.
            let (start_col, start_row, end_col, end_row) = {
                let (mut s_col, mut s_row) = sel.start_coords;
                let (mut e_col, mut e_row) = sel.end_coords;

                if s_row > e_row || (s_row == e_row && s_col > e_col) {
                    std::mem::swap(&mut s_row, &mut e_row);
                    std::mem::swap(&mut s_col, &mut e_col);
                }
                (s_col, s_row, e_col, e_row)
            };

            match sel.mode {
                crate::term::SelectionMode::Normal => {
                    if row < start_row || row > end_row { return false; }
                    if row == start_row && col < start_col { return false; }
                    if row == end_row && col > end_col { return false; } // Selection is inclusive of end_col
                    true // Cell is within the selected range.
                }
                crate::term::SelectionMode::Block => {
                    // For block selection, ensure col is within min/max of start_col/end_col
                    // and row is within min/max of start_row/end_row.
                    let block_start_col = std::cmp::min(sel.start_coords.0, sel.end_coords.0);
                    let block_end_col = std::cmp::max(sel.start_coords.0, sel.end_coords.0);
                    // Note: start_row and end_row are already normalized.
                    row >= start_row && row <= end_row && col >= block_start_col && col <= block_end_col
                }
            }
        } else {
            false // No selection active.
        }
    }

    /// Draws the current state of the terminal to the `Driver`.
    ///
    /// It uses a `RenderSnapshot` from the `TerminalInterface` to get a consistent
    /// view of the terminal state. It determines which lines to redraw based on
    /// dirty flags in the snapshot, cursor movement, and selection changes.
    ///
    /// # Optimizations:
    /// - Only clears the entire screen on the `first_draw`.
    /// - Redraws only lines marked as dirty in the snapshot, plus lines affected by
    ///   cursor movement or selection changes.
    /// - Delegates line content drawing to `draw_line_content`, which further
    ///   optimizes by coalescing character runs.
    pub fn draw(
        &mut self,
        term: &mut (impl TerminalInterface + ?Sized), 
        driver: &mut dyn Driver,
    ) -> Result<()> {
        let snapshot = term.get_render_snapshot(); 
        let (term_width, term_height) = snapshot.dimensions;

        if term_width == 0 || term_height == 0 {
            trace!("Renderer::draw: Terminal dimensions zero ({}x{}), skipping draw.", term_width, term_height);
            return Ok(());
        }

        let mut something_was_drawn = false;
        let mut lines_to_draw_content: HashSet<usize> = snapshot
            .lines.iter().enumerate()
            .filter_map(|(idx, line)| if line.is_dirty { Some(idx) } else { None })
            .collect();

        trace!("Renderer::draw: Dirty lines from snapshot: {:?}, term_dims: {}x{}, first_draw: {}", lines_to_draw_content, term_width, term_height, self.first_draw);
        
        let (cursor_abs_x, cursor_abs_y) = snapshot.cursor_state.map_or((0,0), |cs| (cs.col, cs.row));
        let is_cursor_visible = snapshot.cursor_state.map_or(false, |cs| cs.is_visible);

        let perform_clear_all = self.first_draw;
        if self.first_draw { self.first_draw = false; }

        if perform_clear_all {
            trace!("Renderer::draw: Full refresh (first_draw=true). Clearing all with RENDERER_DEFAULT_BG.");
            driver.clear_all(RENDERER_DEFAULT_BG)?;
            something_was_drawn = true;
            lines_to_draw_content = (0..term_height).collect();
        } else {
            // Ensure cursor line is redrawn if cursor is visible, as cell content under cursor needs refresh.
            if is_cursor_visible && cursor_abs_y < term_height {
                if lines_to_draw_content.insert(cursor_abs_y) {
                    trace!("Renderer::draw: Added cursor line y={} to draw set.", cursor_abs_y);
                }
            }
            // If selection exists, ensure all lines covered by the selection are redrawn.
            // This handles cases where selection changes without content change (e.g., mouse drag).
            if let Some(sel_state) = &snapshot.selection_state {
                // Use normalized coordinates for iterating through rows.
                let (start_row, end_row) = if sel_state.start_coords.1 <= sel_state.end_coords.1 {
                    (sel_state.start_coords.1, sel_state.end_coords.1)
                } else {
                    (sel_state.end_coords.1, sel_state.start_coords.1)
                };
                for r in start_row..=end_row {
                    if r < term_height { lines_to_draw_content.insert(r); }
                }
                trace!("Renderer::draw: Added lines from selection {:?} to draw set.", sel_state.start_coords.1..=sel_state.end_coords.1);
            }
        }

        let mut sorted_lines_to_draw: Vec<usize> = lines_to_draw_content.into_iter().collect();
        sorted_lines_to_draw.sort_unstable(); 
        trace!("Renderer::draw: Final lines to process for content: {:?}", sorted_lines_to_draw);

        if !sorted_lines_to_draw.is_empty() { something_was_drawn = true; }

        for &y_abs in &sorted_lines_to_draw {
            if y_abs >= term_height {
                warn!("Renderer::draw: Attempted to draw out-of-bounds line y={}", y_abs);
                continue;
            }
            // Pass snapshot.selection_state to draw_line_content
            self.draw_line_content(y_abs, term_width, &snapshot, driver)?;
        }

        if is_cursor_visible {
            trace!("Renderer::draw: Cursor is visible, calling draw_cursor_overlay.");
            self.draw_cursor_overlay(cursor_abs_x, cursor_abs_y, &snapshot, driver, term_width, term_height)?;
            something_was_drawn = true;
        }
        
        let _ = term.take_dirty_lines(); // Clear emulator's dirty flags post-snapshot use.

        if something_was_drawn {
            trace!("Renderer::draw: Presenting changes.");
            driver.present()?;
        } else {
            trace!("Renderer::draw: No changes to present.");
        }
        Ok(())
    }

    /// Draws the content of a single terminal line.
    /// It iterates through cells, determines if each cell is selected, and calls
    /// appropriate drawing sub-routines (`draw_placeholder_cell`, `draw_space_run`, `draw_text_segment`)
    /// passing the selection status to them.
    fn draw_line_content(
        &self,
        y_abs: usize,
        term_width: usize,
        snapshot: &crate::term::RenderSnapshot, 
        driver: &mut dyn Driver,
    ) -> Result<()> {
        trace!("Renderer::draw_line_content: Drawing line y={}", y_abs);
        let mut current_col: usize = 0;
        let line_glyphs = &snapshot.lines[y_abs].cells;

        while current_col < term_width {
            let start_glyph = &line_glyphs[current_col];
            // Determine if the current cell is selected. This status is passed to drawing functions.
            let is_current_cell_selected = self.is_cell_selected(current_col, y_abs, snapshot.selection_state.as_ref(), term_width);
            
            let (eff_fg, eff_bg, eff_flags) = self.get_effective_colors_and_flags(start_glyph.attr.fg, start_glyph.attr.bg, start_glyph.attr.flags);
            let char_for_log = if start_glyph.c == '\0' { ' ' } else { start_glyph.c };
            debug!("  Line {}, Col {}: Start glyph='{}' (attr:{:?}), EffectiveStyle(fg:{:?}, bg:{:?}, flags:{:?}), Selected: {}", y_abs, current_col, char_for_log, start_glyph.attr, eff_fg, eff_bg, eff_flags, is_current_cell_selected);

            let cells_consumed = if start_glyph.c == '\0' { // WIDE_CHAR_PLACEHOLDER
                let bg_color_for_placeholder = if current_col > 0 {
                    let wide_char_glyph = &line_glyphs[current_col - 1];
                    let (_, placeholder_eff_bg, _) = self.get_effective_colors_and_flags(wide_char_glyph.attr.fg, wide_char_glyph.attr.bg, wide_char_glyph.attr.flags);
                    placeholder_eff_bg
                } else { RENDERER_DEFAULT_BG };
                // Pass `is_current_cell_selected` to draw_placeholder_cell.
                self.draw_placeholder_cell(current_col, y_abs, bg_color_for_placeholder, is_current_cell_selected, driver)?
            } else if start_glyph.c == ' ' {
                // Pass `is_current_cell_selected` (for the start of the run) to draw_space_run.
                self.draw_space_run(current_col, y_abs, term_width, start_glyph, is_current_cell_selected, snapshot, driver)?
            } else {
                // Pass `is_current_cell_selected` (for the start of the segment) to draw_text_segment.
                let cells_consumed_by_text_segment = self.draw_text_segment(current_col, y_abs, term_width, start_glyph, is_current_cell_selected, snapshot, driver)?;
                let char_actual_display_width = get_char_display_width(start_glyph.c);
                if char_actual_display_width == 2 && current_col + 1 < term_width {
                    let (_, placeholder_eff_bg, _) = self.get_effective_colors_and_flags(start_glyph.attr.fg, start_glyph.attr.bg, start_glyph.attr.flags);
                    // Placeholder's selection status is the same as the primary character's.
                    let is_placeholder_selected = is_current_cell_selected; 
                    let placeholder_rect = CellRect { x: current_col + 1, y: y_abs, width: 1, height: 1 };
                    trace!("    Line {}, Col {}: Explicitly filling placeholder for wide char '{}' at col {} with bg={:?}, selected={}", y_abs, current_col + 1, start_glyph.c, current_col, placeholder_eff_bg, is_placeholder_selected);
                    driver.fill_rect(placeholder_rect, placeholder_eff_bg, is_placeholder_selected)?;
                }
                cells_consumed_by_text_segment
            };

            if cells_consumed == 0 {
                warn!("Renderer::draw_line_content: A draw segment reported consuming 0 cells at ({}, {}), char '{}'. Advancing by 1 to prevent loop.", current_col, y_abs, start_glyph.c);
                current_col += 1; 
            } else {
                current_col += cells_consumed;
            }
        }
        Ok(())
    }

    /// Draws a placeholder cell, passing `is_selected` to the driver's `fill_rect`.
    fn draw_placeholder_cell(&self, x: usize, y: usize, effective_bg: Color, is_selected: bool, driver: &mut dyn Driver) -> Result<usize> {
        let rect = CellRect { x, y, width: 1, height: 1 };
        trace!("    Line {}, Col {}: Placeholder. FillRect with bg={:?}, selected={}", y, x, effective_bg, is_selected);
        // Pass `is_selected` to driver.fill_rect.
        driver.fill_rect(rect, effective_bg, is_selected)?;
        Ok(1) 
    }

    /// Draws a run of space characters, passing `start_is_selected` to the driver's `fill_rect`.
    /// The `start_is_selected` flag determines if the entire run is treated as selected.
    fn draw_space_run( &self, start_col: usize, y: usize, term_width: usize, start_glyph: &Glyph, start_is_selected: bool, snapshot: &crate::term::RenderSnapshot, driver: &mut dyn Driver) -> Result<usize> {
        let line_glyphs = &snapshot.lines[y].cells;
        let (_, start_eff_bg, start_eff_flags) = self.get_effective_colors_and_flags(start_glyph.attr.fg, start_glyph.attr.bg, start_glyph.attr.flags);
        let mut space_run_len = 0;

        for x_offset in 0..(term_width - start_col) {
            let current_scan_col = start_col + x_offset;
            let glyph_at_scan = &line_glyphs[current_scan_col];
            // Selection state for the current cell in the potential run.
            let is_current_scan_cell_selected = self.is_cell_selected(current_scan_col, y, snapshot.selection_state.as_ref(), term_width);
            let (_, current_scan_eff_bg, current_scan_flags) = self.get_effective_colors_and_flags(glyph_at_scan.attr.fg, glyph_at_scan.attr.bg, glyph_at_scan.attr.flags);

            // Break if not a space, or if style/selection state differs from the start of the run.
            if glyph_at_scan.c != ' ' || current_scan_eff_bg != start_eff_bg || current_scan_flags != start_eff_flags || is_current_scan_cell_selected != start_is_selected {
                break;
            }
            space_run_len += 1;
        }

        if space_run_len == 0 { warn!("Renderer::draw_space_run: Detected 0-length space run at ({},{}).", start_col, y); return Ok(0); }

        let rect = CellRect { x: start_col, y, width: space_run_len, height: 1 };
        trace!("    Line {}, Col {}: Space run (len {}). FillRect with bg={:?}, flags={:?}, selected={}", y, start_col, space_run_len, start_eff_bg, start_eff_flags, start_is_selected);
        // Pass `start_is_selected` to driver.fill_rect, applying to the whole run.
        driver.fill_rect(rect, start_eff_bg, start_is_selected)?;
        Ok(space_run_len)
    }

    /// Draws a segment of text characters, passing `start_is_selected` to the driver's `draw_text_run`.
    /// The `start_is_selected` flag determines if the entire segment is treated as selected.
    fn draw_text_segment( &self, start_col: usize, y: usize, term_width: usize, start_glyph: &Glyph, start_is_selected: bool, snapshot: &crate::term::RenderSnapshot, driver: &mut dyn Driver) -> Result<usize> {
        let line_glyphs = &snapshot.lines[y].cells;
        let (start_eff_fg, start_eff_bg, start_eff_flags) = self.get_effective_colors_and_flags(start_glyph.attr.fg, start_glyph.attr.bg, start_glyph.attr.flags);
        let mut run_text = String::new();
        let mut run_total_cell_width = 0;
        let mut current_scan_col = start_col;

        while current_scan_col < term_width {
            let glyph_at_scan = &line_glyphs[current_scan_col];
            // Selection state for the current cell in the potential run.
            let is_current_scan_cell_selected = self.is_cell_selected(current_scan_col, y, snapshot.selection_state.as_ref(), term_width);
            
            // Break if non-text, or if style/selection state differs from the start of the run.
            if glyph_at_scan.c == ' ' || glyph_at_scan.c == '\0' || is_current_scan_cell_selected != start_is_selected { break; }
            let (current_glyph_eff_fg, current_glyph_eff_bg, current_glyph_flags) = self.get_effective_colors_and_flags(glyph_at_scan.attr.fg, glyph_at_scan.attr.bg, glyph_at_scan.attr.flags);
            if !(current_glyph_eff_fg == start_eff_fg && current_glyph_eff_bg == start_eff_bg && current_glyph_flags == start_eff_flags) { break; }

            let char_display_width = get_char_display_width(glyph_at_scan.c);
            if char_display_width == 0 { // Handle zero-width chars by appending and advancing scan col by 1
                run_text.push(glyph_at_scan.c); current_scan_col += 1; continue;
            }
            if start_col + run_total_cell_width + char_display_width > term_width { break; } // Ensure run fits
            run_text.push(glyph_at_scan.c);
            run_total_cell_width += char_display_width;
            current_scan_col += char_display_width; // Advance by cell width consumed by char
        }

        if run_text.is_empty() {
             let advance_by = get_char_display_width(start_glyph.c).max(1);
             if start_col + advance_by <= term_width { // If the single char fits
                 run_text.push(start_glyph.c);
                 run_total_cell_width = advance_by;
             } else { // Single char does not fit (e.g. wide char at last column)
                 // warn!( ... ); // Kept commented out as per previous resolution
                 return Ok(advance_by); // Return width it would have taken
             }
        }

        let coords = CellCoords { x: start_col, y };
        let style = TextRunStyle { fg: start_eff_fg, bg: start_eff_bg, flags: start_eff_flags };
        trace!("    Line {}, Col {}: Text run: '{}' ({} cells). DrawTextRun with style={:?}, selected={}", y, start_col, run_text, run_total_cell_width, style, start_is_selected);
        // Pass `start_is_selected` to driver.draw_text_run, applying to the whole segment.
        driver.draw_text_run(coords, &run_text, style, start_is_selected)?;
        Ok(run_total_cell_width)
    }

    /// Draws the terminal cursor as an overlay.
    /// The `is_selected` flag for the driver call drawing the cursor itself is `false`,
    /// as the renderer pre-calculates the cursor's appearance (often inverting cell colors).
    fn draw_cursor_overlay( &self, cursor_abs_x: usize, cursor_abs_y: usize, snapshot: &crate::term::RenderSnapshot, driver: &mut dyn Driver, term_width: usize, term_height: usize) -> Result<()> {
        trace!("Renderer::draw_cursor_overlay: Screen cursor pos ({}, {})", cursor_abs_x, cursor_abs_y);
        if !(cursor_abs_x < term_width && cursor_abs_y < term_height) {
            warn!("Renderer::draw_cursor_overlay: Cursor at ({}, {}) is out of bounds ({}x{}). Not drawing.", cursor_abs_x, cursor_abs_y, term_width, term_height);
            return Ok(());
        }

        let physical_cursor_x_for_draw: usize;
        let char_to_draw_at_cursor: char;
        let original_attrs_at_cursor: Attributes;
        
        let glyph_at_logical_cursor = &snapshot.lines[cursor_abs_y].cells[cursor_abs_x];
        if glyph_at_logical_cursor.c == '\0' && cursor_abs_x > 0 {
            let first_half_glyph = &snapshot.lines[cursor_abs_y].cells[cursor_abs_x - 1];
            char_to_draw_at_cursor = first_half_glyph.c; 
            original_attrs_at_cursor = first_half_glyph.attr;
            physical_cursor_x_for_draw = cursor_abs_x - 1; 
        } else {
            char_to_draw_at_cursor = glyph_at_logical_cursor.c;
            original_attrs_at_cursor = glyph_at_logical_cursor.attr;
            physical_cursor_x_for_draw = cursor_abs_x;
        }

        let (resolved_original_fg, resolved_original_bg, resolved_original_flags) = self.get_effective_colors_and_flags(original_attrs_at_cursor.fg, original_attrs_at_cursor.bg, original_attrs_at_cursor.flags);
        
        // Determine if the cell *under* the cursor is selected.
        let is_underlying_cell_selected = self.is_cell_selected(physical_cursor_x_for_draw, cursor_abs_y, snapshot.selection_state.as_ref(), term_width);

        // Cursor inverts the colors of the cell it's on.
        // If the cell is selected, its colors are already effectively swapped by selection rendering.
        // The cursor should then "un-invert" them back to original for the cursor glyph,
        // or apply a distinct cursor color. Here, we use the common behavior of inverting the *current* visual.
        let (cursor_char_fg, cursor_cell_bg) = if is_underlying_cell_selected {
            // Cell is selected: its visual FG is original BG, visual BG is original FG.
            // Cursor inverts this: cursor FG becomes original FG, cursor BG becomes original BG.
            (resolved_original_fg, resolved_original_bg)
        } else {
            // Cell not selected: visual FG/BG are original resolved FG/BG.
            // Cursor inverts this: cursor FG becomes original BG, cursor BG becomes original FG.
            (resolved_original_bg, resolved_original_fg)
        };

        let coords = CellCoords { x: physical_cursor_x_for_draw, y: cursor_abs_y };
        let style = TextRunStyle { fg: cursor_char_fg, bg: cursor_cell_bg, flags: resolved_original_flags };
        let final_char_to_draw_for_cursor = if char_to_draw_at_cursor == '\0' { ' ' } else { char_to_draw_at_cursor };
        
        trace!("    Drawing cursor overlay: char='{}' at physical ({},{}) with style: {:?}, underlying_cell_selected={}", final_char_to_draw_for_cursor, physical_cursor_x_for_draw, cursor_abs_y, style, is_underlying_cell_selected);
        // The cursor itself is not "selected"; its appearance is distinct. So `is_selected` is false for this call.
        driver.draw_text_run(coords, &final_char_to_draw_for_cursor.to_string(), style, false)?;
        Ok(())
    }
}

impl Default for Renderer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests;
