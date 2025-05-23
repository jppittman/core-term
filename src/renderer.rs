// myterm/src/renderer.rs

//! This module defines the `Renderer`.
use crate::term::SelectionRenderState; // Added for selection
//!
//! The `Renderer`'s primary responsibility is to translate the visual state of the
//! `TerminalEmulator` (obtained via the `TerminalInterface`) into a series of
//! abstract drawing commands that can be executed by a `Driver`. It is designed
//! to be backend-agnostic. It defines default foreground and background colors
//! for resolving `Color::Default` from glyph attributes when rendering.

use crate::backends::{CellCoords, CellRect, Driver, TextRunStyle};
// Corrected: Import Color and NamedColor directly from the color module.
use crate::color::{Color, NamedColor};
// AttrFlags, Attributes, and Glyph are correctly from the glyph module.
use crate::glyph::{AttrFlags, Attributes, Glyph};
use crate::term::unicode::get_char_display_width;
use crate::term::TerminalInterface; // Trait for interacting with the terminal state.

use anyhow::Result; // For error handling.
use log::{debug, trace, warn}; // For logging.
use std::collections::HashSet; // For managing dirty line indices.

/// Default foreground color used by the renderer when a glyph specifies `Color::Default`.
pub const RENDERER_DEFAULT_FG: Color = Color::Named(NamedColor::White);
/// Default background color used by the renderer when a glyph specifies `Color::Default`.
pub const RENDERER_DEFAULT_BG: Color = Color::Named(NamedColor::Black);

/// The `Renderer` translates `TerminalEmulator` state into abstract drawing commands
/// for a `Driver`.
///
/// It optimizes drawing by only processing lines marked as dirty by the terminal
/// and by coalescing character runs with identical attributes.
#[derive(Clone)]
pub struct Renderer {
    /// Tracks if this is the very first draw call.
    /// If `true`, a full screen clear and redraw is typically performed.
    /// This is set to `false` after the first successful draw.
    pub first_draw: bool,
}

impl Renderer {
    /// Creates a new `Renderer` instance.
    /// The `first_draw` flag is initialized to `true`.
    pub fn new() -> Self {
        Self { first_draw: true }
    }

    /// Resolves `Color::Default` for foreground and background, and handles the
    /// `AttrFlags::REVERSE` flag to determine the effective colors and flags for rendering.
    ///
    /// The `AttrFlags::REVERSE` flag is consumed in this process (i.e., removed from
    /// the returned `AttrFlags`) because its effect is applied by swapping the
    /// foreground and background colors.
    ///
    /// # Arguments
    /// * `cell_fg`: The foreground `Color` from the glyph's attributes.
    /// * `cell_bg`: The background `Color` from the glyph's attributes.
    /// * `cell_flags`: The `AttrFlags` from the glyph's attributes.
    ///
    /// # Returns
    /// A tuple `(effective_fg, effective_bg, effective_flags)` containing the
    /// concrete colors and modified flags to be used for rendering.
    fn get_effective_colors_and_flags(
        &self,
        cell_fg: Color,
        cell_bg: Color,
        cell_flags: AttrFlags,
    ) -> (Color, Color, AttrFlags) {
        // Resolve Color::Default to the renderer's defined defaults.
        let mut resolved_fg = match cell_fg {
            Color::Default => RENDERER_DEFAULT_FG,
            c => c,
        };
        let mut resolved_bg = match cell_bg {
            Color::Default => RENDERER_DEFAULT_BG,
            c => c,
        };

        let mut effective_flags = cell_flags;

        // Apply REVERSE video attribute by swapping foreground and background.
        if cell_flags.contains(AttrFlags::REVERSE) {
            std::mem::swap(&mut resolved_fg, &mut resolved_bg);
            effective_flags.remove(AttrFlags::REVERSE); // REVERSE is now applied.
        }
        (resolved_fg, resolved_bg, effective_flags)
    }

    /// Draws the current state of the terminal (provided via `TerminalInterface`)
    /// using the drawing primitives of the `Driver`.
    ///
    /// This method implements the core rendering logic:
    /// 1. Determines if a full screen clear is needed (on `first_draw`).
    /// 2. Gets dirty line information from the terminal.
    /// 3. Iterates over lines that need redrawing (dirty lines and the cursor's line).
    /// 4. For each such line, it calls `draw_line_content`.
    /// 5. If the cursor is visible, it calls `draw_cursor_overlay`.
    /// 6. Finally, it calls `driver.present()` to make the changes visible.
    ///
    /// # Arguments
    /// * `term`: A mutable reference to an object implementing `TerminalInterface`.
    ///           It's `mut` because `take_dirty_lines` modifies the terminal's state.
    ///           The `+ ?Sized` allows for trait objects.
    /// * `driver`: A mutable reference to a `Driver` implementation.
    ///
    /// # Returns
    /// * `Result<()>`: Ok if drawing and presentation were successful, or an error.
    fn is_cell_selected(
        &self,
        col: usize,
        row: usize,
        selection: Option<&SelectionRenderState>,
        term_width: usize,
    ) -> bool {
        if let Some(sel) = selection {
            // Normalize coordinates: start should be before or at end.
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
                    if row < start_row || row > end_row {
                        return false;
                    }
                    if row == start_row && col < start_col {
                        return false;
                    }
                    if row == end_row && col > end_col { // Use > for end_col as selection is inclusive
                        return false;
                    }
                    // Cells between start_row and end_row are fully selected if mode is Normal
                    // or if it's the start/end row and within column bounds.
                    true
                }
                crate::term::SelectionMode::Block => {
                    // For block selection, ensure col is within min/max of start_col/end_col
                    let block_start_col = std::cmp::min(sel.start_coords.0, sel.end_coords.0);
                    let block_end_col = std::cmp::max(sel.start_coords.0, sel.end_coords.0);
                     row >= start_row && row <= end_row && col >= block_start_col && col <= block_end_col
                }
            }
        } else {
            false
        }
    }

    pub fn draw(
        &mut self,
        term: &mut (impl TerminalInterface + ?Sized), // Added + ?Sized
        driver: &mut dyn Driver,
    ) -> Result<()> {
        let snapshot = term.get_render_snapshot(); // Get the full snapshot
        let (term_width, term_height) = snapshot.dimensions;

        // Avoid drawing if terminal dimensions are invalid.
        if term_width == 0 || term_height == 0 {
            trace!(
                "Renderer::draw: Terminal dimensions zero ({}x{}), skipping draw.",
                term_width,
                term_height
            );
            return Ok(());
        }

        let mut something_was_drawn = false;

        // Use dirty lines from the snapshot for rendering decisions.
        // The term.take_dirty_lines() is for the emulator's internal state management,
        // but the snapshot provides the consistent view for this render pass.
        let mut lines_to_draw_content: HashSet<usize> = snapshot
            .lines
            .iter()
            .enumerate()
            .filter_map(|(idx, line)| if line.is_dirty { Some(idx) } else { None })
            .collect();

        trace!(
            "Renderer::draw: Dirty lines from snapshot: {:?}, term_dims: {}x{}, first_draw: {}",
            lines_to_draw_content, term_width, term_height, self.first_draw
        );
        
        // Cursor position from snapshot
        let (cursor_abs_x, cursor_abs_y) = snapshot.cursor_state
            .map_or((0,0), |cs| (cs.col, cs.row)); // Default if no cursor state
        let is_cursor_visible = snapshot.cursor_state.map_or(false, |cs| cs.is_visible);

        // Perform a full clear only on the very first draw operation.
        let perform_clear_all = self.first_draw;
        if self.first_draw {
            self.first_draw = false; // Ensure this only happens once.
        }

        if perform_clear_all {
            trace!(
                "Renderer::draw: Full refresh (first_draw=true). Clearing all with RENDERER_DEFAULT_BG."
            );
            driver.clear_all(RENDERER_DEFAULT_BG)?;
            something_was_drawn = true;
            // After a full clear, all lines are considered dirty for content redrawing.
            lines_to_draw_content = (0..term_height).collect();
        } else {
            // For subsequent draws, only redraw explicitly dirty lines and the cursor line.
            // The cursor line needs to be redrawn to correctly render the cell underneath
            // before overlaying the cursor itself.
            if is_cursor_visible && cursor_abs_y < term_height {
                if lines_to_draw_content.insert(cursor_abs_y) {
                    trace!(
                        "Renderer::draw: Added cursor line y={} to draw set as it was not initially dirty.",
                        cursor_abs_y
                    );
                }
            }
        }
        
        // Add lines affected by selection changes to the draw set, if not already full clear
        if !perform_clear_all {
            if let Some(sel_state) = &snapshot.selection_state {
                let (start_col, start_row, end_col, end_row) = { // Normalized
                    let (mut s_col, mut s_row) = sel_state.start_coords;
                    let (mut e_col, mut e_row) = sel_state.end_coords;
                    if s_row > e_row || (s_row == e_row && s_col > e_col) {
                        std::mem::swap(&mut s_row, &mut e_row);
                        std::mem::swap(&mut s_col, &mut e_col);
                    }
                    (s_col, s_row, e_col, e_row)
                };
                for r in start_row..=end_row {
                    if r < term_height {
                        lines_to_draw_content.insert(r);
                    }
                }
            }
        }


        let mut sorted_lines_to_draw: Vec<usize> = lines_to_draw_content.into_iter().collect();
        sorted_lines_to_draw.sort_unstable(); // Draw in logical order.
        trace!(
            "Renderer::draw: Final lines to process for content: {:?}",
            sorted_lines_to_draw
        );

        if !sorted_lines_to_draw.is_empty() {
            something_was_drawn = true;
        }

        for &y_abs in &sorted_lines_to_draw {
            if y_abs >= term_height {
                warn!(
                    "Renderer::draw: Attempted to draw out-of-bounds line y={}",
                    y_abs
                );
                continue;
            }
            self.draw_line_content(y_abs, term_width, &snapshot, driver)?; // Pass snapshot
        }

        // Overlay the cursor if it's visible.
        if is_cursor_visible {
            trace!("Renderer::draw: Cursor is visible, calling draw_cursor_overlay.");
            self.draw_cursor_overlay(
                cursor_abs_x,
                cursor_abs_y,
                &snapshot, // Pass snapshot
                driver,
                term_width,
                term_height,
            )?;
            something_was_drawn = true; // Drawing cursor means something was drawn.
        }
        
        // After rendering, clear the dirty flags in the actual terminal emulator state
        // This is crucial because the snapshot's dirty flags are consumed by this render pass.
        // The emulator needs to know that these lines have been processed for rendering.
        // This assumes `take_dirty_lines` on the TerminalInterface clears its internal flags.
        // If `get_render_snapshot` doesn't clear flags, then this step is essential.
        // Based on typical design, `take_dirty_lines` (if called by emulator before snapshot)
        // or a similar mechanism should handle this. If snapshot itself is the source of truth
        // for dirty lines for this render pass, then the emulator's state needs updating.
        // For now, assuming `term.take_dirty_lines()` was the source and already cleared them.
        // If not, a `term.confirm_lines_rendered(sorted_lines_to_draw)` might be needed.
        // The previous code had `term.take_dirty_lines()` at the start, which modifies term state.
        // Let's ensure this is called to clear the emulator's dirty state.
        let _ = term.take_dirty_lines(); // Call it to ensure emulator state is updated.


        // Present the changes to the display if any drawing operations occurred.
        if something_was_drawn {
            trace!("Renderer::draw: Presenting changes.");
            driver.present()?;
        } else {
            trace!("Renderer::draw: No changes to present.");
        }
        Ok(())
    }

    /// Draws the content of a single terminal line by iterating through its cells
    /// and dispatching to specialized drawing functions for text, spaces, or placeholders.
    ///
    /// # Arguments
    /// * `y_abs`: The absolute 0-based row index of the line to draw.
    /// * `term_width`: The width of the terminal in cells.
    /// * `term`: A reference to an object implementing `TerminalInterface`.
    /// * `driver`: A mutable reference to a `Driver` implementation.
    ///
    /// # Returns
    /// * `Result<()>`: Ok if the line was drawn successfully, or an error.
    fn draw_line_content(
        &self,
        y_abs: usize,
        term_width: usize,
        snapshot: &crate::term::RenderSnapshot, // Use RenderSnapshot
        driver: &mut dyn Driver,
    ) -> Result<()> {
        trace!("Renderer::draw_line_content: Drawing line y={}", y_abs);
        let mut current_col: usize = 0;
        let line_glyphs = &snapshot.lines[y_abs].cells;

        while current_col < term_width {
            // let start_glyph = term.get_glyph(current_col, y_abs);
            let start_glyph = &line_glyphs[current_col];
            let is_current_cell_selected =
                self.is_cell_selected(current_col, y_abs, snapshot.selection_state.as_ref(), term_width);
            let (eff_fg, eff_bg, eff_flags) = self.get_effective_colors_and_flags(
                start_glyph.attr.fg,
                start_glyph.attr.bg,
                start_glyph.attr.flags,
            );

            // Use a space for logging if the character is null (wide char placeholder).
            let char_for_log = if start_glyph.c == '\0' {
                ' '
            } else {
                start_glyph.c
            };
            debug!(
                // Changed to debug for potentially verbose output, trace for finer details.
                "  Line {}, Col {}: Start glyph='{}' (attr:{:?}), EffectiveStyle(fg:{:?}, bg:{:?}, flags:{:?}), Selected: {}",
                y_abs, current_col, char_for_log, start_glyph.attr, eff_fg, eff_bg, eff_flags, is_current_cell_selected
            );

            // Dispatch to appropriate drawing function based on glyph content.
            let cells_consumed = if start_glyph.c == '\0' {
                // Placeholder for the second half of a wide character.
                 let bg_color_for_placeholder = if current_col > 0 {
                    let wide_char_glyph = &line_glyphs[current_col - 1];
                    let (_, placeholder_eff_bg, _) = self.get_effective_colors_and_flags(
                        wide_char_glyph.attr.fg,
                        wide_char_glyph.attr.bg,
                        wide_char_glyph.attr.flags,
                    );
                    placeholder_eff_bg
                } else {
                     warn!(
                        "Placeholder found at column 0 on line {}, this is unexpected. Using default background.",
                        y_abs
                    );
                    RENDERER_DEFAULT_BG
                };
                self.draw_placeholder_cell(current_col, y_abs, bg_color_for_placeholder, is_current_cell_selected, driver)?
            } else if start_glyph.c == ' ' {
                // Space character; attempt to draw a run of spaces for optimization.
                self.draw_space_run(
                    current_col,
                    y_abs,
                    term_width,
                    &start_glyph, // Pass the initial space glyph
                    is_current_cell_selected, // Pass selection state
                    snapshot, // Pass snapshot
                    driver,
                )?
            } else {
                // Regular text character; attempt to draw a run of text.
                let cells_consumed_by_text_segment = self.draw_text_segment(
                    current_col,
                    y_abs,
                    term_width,
                    &start_glyph,
                    is_current_cell_selected, // Pass selection state
                    snapshot, // Pass snapshot
                    driver,
                )?;

                // After drawing a text segment, check if it was a wide character.
                // If so, we need to explicitly fill its placeholder cell.
                let char_actual_display_width = get_char_display_width(start_glyph.c);
                if char_actual_display_width == 2 {
                    if current_col + 1 < term_width {
                        // The placeholder is within bounds.
                        let (_, placeholder_eff_bg, _) = self.get_effective_colors_and_flags(
                            start_glyph.attr.fg,
                            start_glyph.attr.bg,
                            start_glyph.attr.flags,
                        );
                        // Selection status for the placeholder should be same as the primary char
                        let is_placeholder_selected = is_current_cell_selected; 
                        let placeholder_rect = CellRect {
                            x: current_col + 1,
                            y: y_abs,
                            width: 1,
                            height: 1,
                        };
                        trace!(
                            "    Line {}, Col {}: Explicitly filling placeholder for wide char '{}' at col {} with bg={:?}, selected={}",
                            y_abs,
                            current_col + 1, // Log placeholder's column
                            start_glyph.c,
                            current_col, // Log wide char's column
                            placeholder_eff_bg,
                            is_placeholder_selected
                        );
                        driver.fill_rect(placeholder_rect, placeholder_eff_bg, is_placeholder_selected)?;
                    } else {
                        // This case should ideally be prevented by draw_text_segment not overflowing.
                        warn!(
                            "    Line {}, Col {}: Wide char '{}' at end of line, no space for placeholder fill.",
                            y_abs, current_col, start_glyph.c
                        );
                    }
                }
                cells_consumed_by_text_segment // Assign to outer scope's cells_consumed
            };

            // Ensure progress is made to avoid infinite loops.
            if cells_consumed == 0 {
                warn!(
                    "Renderer::draw_line_content: A draw segment reported consuming 0 cells at ({}, {}), char '{}'. Advancing by 1 to prevent loop.",
                    current_col, y_abs, start_glyph.c
                );
                current_col += 1; // Fallback: advance by at least one cell.
            } else {
                current_col += cells_consumed;
            }
        }
        Ok(())
    }

    /// Draws a placeholder cell (typically the second half of a wide character)
    /// by filling it with the effective background color.
    ///
    /// # Returns
    /// `Ok(1)`: Always consumes 1 cell.
    fn draw_placeholder_cell(
        &self,
        x: usize,
        y: usize,
        effective_bg: Color, // Background color determined by the wide char's attributes.
        is_selected: bool,   // Is this placeholder cell part of a selection?
        driver: &mut dyn Driver,
    ) -> Result<usize> {
        let rect = CellRect {
            x,
            y,
            width: 1,
            height: 1,
        };
        trace!(
            "    Line {}, Col {}: Placeholder. FillRect with bg={:?}, selected={}",
            y,
            x,
            effective_bg,
            is_selected
        );
        driver.fill_rect(rect, effective_bg, is_selected)?;
        Ok(1) // A placeholder consumes 1 cell.
    }

    /// Identifies and draws a contiguous run of space characters that share the
    /// same effective background color, attribute flags, and selection state.
    /// This optimizes drawing by using a single `fill_rect` call for the run.
    ///
    /// # Returns
    /// `Ok(usize)`: The number of space cells consumed in this run.
    fn draw_space_run(
        &self,
        start_col: usize,
        y: usize,
        term_width: usize,
        start_glyph: &Glyph, // The glyph at start_col, known to be a space.
        start_is_selected: bool, // Selection state of the start_glyph
        snapshot: &crate::term::RenderSnapshot, // Use RenderSnapshot
        driver: &mut dyn Driver,
    ) -> Result<usize> {
        let line_glyphs = &snapshot.lines[y].cells;
        // Determine the effective style of the first space in the potential run.
        let (_, start_eff_bg, start_eff_flags) = self.get_effective_colors_and_flags(
            start_glyph.attr.fg,
            start_glyph.attr.bg,
            start_glyph.attr.flags,
        );

        let mut space_run_len = 0;
        // Scan right from start_col to find contiguous spaces with matching style and selection state.
        for x_offset in 0..(term_width - start_col) {
            let current_scan_col = start_col + x_offset;
            // let glyph_at_scan = term.get_glyph(current_scan_col, y);
            let glyph_at_scan = &line_glyphs[current_scan_col];
            let is_current_scan_selected = self.is_cell_selected(
                current_scan_col,
                y,
                snapshot.selection_state.as_ref(),
                term_width
            );

            let (_, current_scan_eff_bg, current_scan_flags) = self.get_effective_colors_and_flags(
                glyph_at_scan.attr.fg,
                glyph_at_scan.attr.bg,
                glyph_at_scan.attr.flags,
            );

            // Break the run if not a space, style differs, or selection state differs.
            if glyph_at_scan.c != ' '
                || current_scan_eff_bg != start_eff_bg
                || current_scan_flags != start_eff_flags
                || is_current_scan_selected != start_is_selected
            {
                break;
            }
            space_run_len += 1;
        }

        if space_run_len == 0 {
            warn!(
                "Renderer::draw_space_run: Detected 0-length space run at ({},{}). This might indicate an issue.",
                start_col, y
            );
            return Ok(0);
        }

        let rect = CellRect {
            x: start_col,
            y,
            width: space_run_len,
            height: 1,
        };
        trace!(
            "    Line {}, Col {}: Space run (len {}). FillRect with bg={:?}, flags={:?}, selected={}",
            y,
            start_col,
            space_run_len,
            start_eff_bg,
            start_eff_flags,
            start_is_selected
        );
        driver.fill_rect(rect, start_eff_bg, start_is_selected)?;

        Ok(space_run_len)
    }

    /// Identifies and draws a contiguous run of non-space, non-placeholder text
    /// characters that share the same effective style and selection state.
    ///
    /// # Returns
    /// `Ok(usize)`: The total number of cells consumed by this text segment.
    fn draw_text_segment(
        &self,
        start_col: usize,
        y: usize,
        term_width: usize,
        start_glyph: &Glyph, // The glyph at start_col.
        start_is_selected: bool, // Selection state of the start_glyph.
        snapshot: &crate::term::RenderSnapshot, // Use RenderSnapshot
        driver: &mut dyn Driver,
    ) -> Result<usize> {
        let line_glyphs = &snapshot.lines[y].cells;
        let (start_eff_fg, start_eff_bg, start_eff_flags) = self.get_effective_colors_and_flags(
            start_glyph.attr.fg,
            start_glyph.attr.bg,
            start_glyph.attr.flags,
        );

        let mut run_text = String::new();
        let mut run_total_cell_width = 0;
        let mut current_scan_col = start_col;

        while current_scan_col < term_width {
            // let glyph_at_scan = term.get_glyph(current_scan_col, y);
            let glyph_at_scan = &line_glyphs[current_scan_col];
            let is_current_scan_selected = self.is_cell_selected(
                current_scan_col,
                y,
                snapshot.selection_state.as_ref(),
                term_width
            );

            if glyph_at_scan.c == ' ' || glyph_at_scan.c == '\0' || is_current_scan_selected != start_is_selected {
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
                run_text.push(glyph_at_scan.c);
                current_scan_col += 1;
                continue;
            }
            if start_col + run_total_cell_width + char_display_width > term_width {
                break;
            }
            run_text.push(glyph_at_scan.c);
            run_total_cell_width += char_display_width;
            current_scan_col += char_display_width;
        }

        if run_text.is_empty() {
             let advance_by = get_char_display_width(start_glyph.c).max(1);
             // If it's just a single char that couldn't form a run due to selection change or style change,
             // we still need to draw it. This case handles if the loop condition broke immediately.
             if start_col + advance_by <= term_width {
                 run_text.push(start_glyph.c);
                 run_total_cell_width = advance_by;
             } else {
                 warn!( /* ... */ );
                 return Ok(advance_by);
             }
        }


        let coords = CellCoords { x: start_col, y };
        let style = TextRunStyle {
            fg: start_eff_fg,
            bg: start_eff_bg,
            flags: start_eff_flags,
        };
        trace!(
            "    Line {}, Col {}: Text run: '{}' ({} cells). DrawTextRun with style={:?}, selected={}",
            y, start_col, run_text, run_total_cell_width, style, start_is_selected
        );
        driver.draw_text_run(coords, &run_text, style, start_is_selected)?;

        Ok(run_total_cell_width)
    }

    /// Draws the terminal cursor as an overlay on top of the existing cell content.
    ///
    /// The cursor is typically rendered by inverting the foreground and background
    /// colors of the cell it occupies. For wide characters, the cursor is drawn
    /// at the start of the character.
    ///
    /// # Arguments
    /// * `cursor_abs_x`: Absolute 0-based column of the cursor.
    /// * `cursor_abs_y`: Absolute 0-based row of the cursor.
    /// * `snapshot`: The `RenderSnapshot` containing terminal state.
    /// * `driver`: A mutable reference to a `Driver` implementation.
    /// * `term_width`: The width of the terminal in cells.
    /// * `term_height`: The height of the terminal in cells.
    ///
    /// # Returns
    /// * `Result<()>`: Ok if the cursor was drawn successfully, or an error.
    fn draw_cursor_overlay(
        &self,
        cursor_abs_x: usize,
        cursor_abs_y: usize,
        snapshot: &crate::term::RenderSnapshot, // Use RenderSnapshot
        driver: &mut dyn Driver,
        term_width: usize,
        term_height: usize,
    ) -> Result<()> {
        trace!(
            "Renderer::draw_cursor_overlay: Screen cursor pos ({}, {})",
            cursor_abs_x,
            cursor_abs_y
        );

        // Do not draw cursor if it's out of bounds.
        if !(cursor_abs_x < term_width && cursor_abs_y < term_height) {
            warn!(
                "Renderer::draw_cursor_overlay: Cursor at ({}, {}) is out of bounds ({}x{}). Not drawing.",
                cursor_abs_x, cursor_abs_y, term_width, term_height
            );
            return Ok(());
        }

        let physical_cursor_x_for_draw: usize;
        let char_to_draw_at_cursor: char;
        let original_attrs_at_cursor: Attributes;
        
        // Get glyph from snapshot
        let glyph_at_logical_cursor = &snapshot.lines[cursor_abs_y].cells[cursor_abs_x];
        let char_for_log1 = if glyph_at_logical_cursor.c == '\0' {
            ' ' // Log placeholder as space for readability
        } else {
            glyph_at_logical_cursor.c
        };
        trace!(
            "  Cursor overlay: Glyph at logical cursor pos ({},{}): char='{}', attr={:?}",
            cursor_abs_x,
            cursor_abs_y,
            char_for_log1,
            glyph_at_logical_cursor.attr
        );

        // If cursor is on the second half of a wide character (placeholder '\0'),
        // adjust to draw the cursor over the first half of that wide character.
        if glyph_at_logical_cursor.c == '\0' && cursor_abs_x > 0 {
            // Get glyph from snapshot
            let first_half_glyph = &snapshot.lines[cursor_abs_y].cells[cursor_abs_x - 1];
            let char_for_log2 = if first_half_glyph.c == '\0' {
                '?' // Should not be '\0' if logic is correct
            } else {
                first_half_glyph.c
            };
            trace!(
                "    Cursor on placeholder, using first half: char='{}' from col {}",
                char_for_log2,
                cursor_abs_x - 1
            );

            char_to_draw_at_cursor = first_half_glyph.c; // The actual wide character.
            original_attrs_at_cursor = first_half_glyph.attr;
            physical_cursor_x_for_draw = cursor_abs_x - 1; // Draw cursor at the start of the wide char.
        } else {
            // Cursor is on a regular character or the first cell of a wide character.
            char_to_draw_at_cursor = glyph_at_logical_cursor.c;
            original_attrs_at_cursor = glyph_at_logical_cursor.attr;
            physical_cursor_x_for_draw = cursor_abs_x;
        }

        // Determine the effective style of the cell *under* the cursor.
        let (resolved_original_fg, resolved_original_bg, resolved_original_flags) = self
            .get_effective_colors_and_flags(
                original_attrs_at_cursor.fg,
                original_attrs_at_cursor.bg,
                original_attrs_at_cursor.flags, // REVERSE already handled by get_effective_colors_and_flags
            );
        trace!(
            "    Original cell effective attrs for cursor: fg={:?}, bg={:?}, flags={:?}",
            resolved_original_fg,
            resolved_original_bg,
            resolved_original_flags
        );

        // For cursor rendering, typically swap effective FG and BG.
        // However, if the cell is selected, the selection already did this.
        // The cursor should "invert" the currently displayed colors.
        
        let is_cursor_cell_selected = self.is_cell_selected(
            physical_cursor_x_for_draw, // Use physical_cursor_x_for_draw as it's the actual cell being drawn
            cursor_abs_y,
            snapshot.selection_state.as_ref(),
            term_width
        );

        let (cursor_char_fg, cursor_cell_bg) = if is_cursor_cell_selected {
            // If selected, the driver will handle selection colors.
            // The cursor should invert the *original* colors, not the selected colors.
            // So, we pass the original resolved_original_fg/bg, and let driver invert them for cursor
            // AND apply selection. This might need driver to be smart.
            // Alternative: Renderer calculates "selected fg/bg" then cursor inverts that.
            // For now, assume driver's draw_text_run with is_selected=true for cursor will do the right thing.
            // This means cursor over selected text is "double inverted" back to normal, or distinct.
            // Let's keep it simple: cursor inverts the current visual state.
            // If selected, current visual is (inverted_fg, inverted_bg). Cursor inverts this back.
            // So, if selected, cursor_char_fg = resolved_original_fg, cursor_cell_bg = resolved_original_bg
            // This is a common behavior: cursor "un-selects" the cell it's on.
            (resolved_original_fg, resolved_original_bg)
        } else {
            // Not selected, simple inversion for cursor
            (resolved_original_bg, resolved_original_fg)
        };

        let cursor_display_flags = resolved_original_flags;

        let coords = CellCoords {
            x: physical_cursor_x_for_draw,
            y: cursor_abs_y,
        };
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
        trace!(
            "    Drawing cursor overlay: char='{}' at physical ({},{}) with style: {:?}, selected_status_for_cursor_calc={}",
            final_char_to_draw_for_cursor,
            physical_cursor_x_for_draw,
            cursor_abs_y,
            style,
            is_cursor_cell_selected 
        );
        
        // The `is_selected` flag for draw_text_run when drawing the cursor itself is tricky.
        // If true, driver might invert again. If false, it won't.
        // Standard behavior: cursor appearance takes precedence.
        // Let's pass `false` for `is_selected` for the cursor draw call itself,
        // as the Renderer has already determined the cursor's specific FG/BG.
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
