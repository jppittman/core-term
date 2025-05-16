// myterm/src/renderer.rs

//! This module defines the `Renderer`.
//!
//! The `Renderer`'s primary responsibility is to translate the visual state of the
//! `TerminalEmulator` into a series of abstract drawing commands that can be
//! executed by a `Driver`. It is designed to be backend-agnostic.
//! It defines default foreground and background colors for resolving
//! `Color::Default` from glyph attributes.

use crate::backends::{CellCoords, CellRect, Driver, TextRunStyle};
use crate::glyph::{AttrFlags, Attributes, Color, Glyph, NamedColor};
use crate::term::TerminalInterface;
use crate::term::unicode::get_char_display_width;

use anyhow::Result;
use log::{trace, warn};
use std::collections::HashSet;

/// Default foreground color used by the renderer when a glyph specifies `Color::Default`.
const RENDERER_DEFAULT_FG: Color = Color::Named(NamedColor::White);
/// Default background color used by the renderer when a glyph specifies `Color::Default`.
const RENDERER_DEFAULT_BG: Color = Color::Named(NamedColor::Black);

/// The `Renderer` translates `TerminalEmulator` state into abstract drawing commands.
pub struct Renderer {
    // Tracks if this is the very first draw call to ensure an initial full clear.
    // This helps in scenarios like initial startup or after a resize where a
    // full screen refresh is desirable, distinct from just dirty line updates.
    pub first_draw: bool, // Made pub for tests to modify
}

impl Renderer {
    /// Creates a new `Renderer` instance.
    pub fn new() -> Self {
        Self { first_draw: true }
    }

    /// Resolves `Color::Default` and handles the `AttrFlags::REVERSE` flag
    /// to determine the effective foreground, background, and flags for rendering.
    /// The `AttrFlags::REVERSE` is consumed in this process and removed from the returned flags.
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
            effective_flags.remove(AttrFlags::REVERSE); // REVERSE is consumed by swapping colors.
        }
        (resolved_fg, resolved_bg, effective_flags)
    }

    /// Draws the current state of the `TerminalEmulator` using the provided `Driver`.
    pub fn draw(
        &mut self,
        term: &mut impl TerminalInterface,
        driver: &mut dyn Driver,
    ) -> Result<()> {
        // In main.rs, before any call to renderer.draw()
        let (term_width, term_height) = term.dimensions();

        if term_width == 0 || term_height == 0 {
            trace!("Renderer::draw: Terminal dimensions zero, skipping draw.");
            return Ok(());
        }

        let initially_dirty_lines_from_term: HashSet<usize> =
            term.take_dirty_lines().into_iter().collect();

        trace!(
            "Renderer::draw: Initially dirty lines from term: {:?}, term_dims: {}x{}, first_draw: {}",
            initially_dirty_lines_from_term, term_width, term_height, self.first_draw
        );

        let mut lines_to_draw_content: HashSet<usize> = initially_dirty_lines_from_term;
        let (cursor_abs_x, cursor_abs_y) = term.get_screen_cursor_pos();
        let mut something_was_drawn = !lines_to_draw_content.is_empty();

        // Determine if a ClearAll is needed. Only if it's the very first draw.
        let perform_clear_all = self.first_draw;

        if self.first_draw {
            self.first_draw = false; // Reset for subsequent calls
        }

        if perform_clear_all {
            trace!(
                "Renderer::draw: Full refresh (first_draw=true). Clearing all with RENDERER_DEFAULT_BG."
            );
            driver.clear_all(RENDERER_DEFAULT_BG)?;
            something_was_drawn = true;
            // If we cleared all, all lines effectively need their content redrawn.
            lines_to_draw_content = (0..term_height).collect();
        } else {
            // Not the first draw, so no automatic ClearAll based on the first_draw flag.
            // lines_to_draw_content is already populated from initially_dirty_lines_from_term.
            // Add cursor line if visible and not already in the set to be drawn,
            // as drawing the cursor implies the cell under it needs to be redrawn first.
            if term.is_cursor_visible() && cursor_abs_y < term_height {
                if lines_to_draw_content.insert(cursor_abs_y) {
                    trace!(
                        "Renderer::draw: Added cursor line y={} to draw set as it was not initially dirty.",
                        cursor_abs_y
                    );
                    something_was_drawn = true; // Adding a line to draw means something will be drawn
                }
            }
        }

        if !lines_to_draw_content.is_empty() {
            something_was_drawn = true; // Ensure this flag is true if there are lines to draw
        }

        let mut sorted_lines_to_draw: Vec<usize> = lines_to_draw_content.into_iter().collect();
        sorted_lines_to_draw.sort_unstable();
        trace!(
            "Renderer::draw: Final lines to process for content: {:?}",
            sorted_lines_to_draw
        );

        for &y_abs in &sorted_lines_to_draw {
            if y_abs >= term_height {
                warn!(
                    "Renderer::draw: Attempted to draw out-of-bounds line y={}",
                    y_abs
                );
                continue;
            }
            self.draw_line_content(y_abs, term_width, term, driver)?;
        }

        if term.is_cursor_visible() {
            trace!("Renderer::draw: Cursor is visible, calling draw_cursor overlay.");
            self.draw_cursor_overlay(
                cursor_abs_x,
                cursor_abs_y,
                term,
                driver,
                term_width,
                term_height,
            )?;
            something_was_drawn = true;
        }

        if something_was_drawn {
            trace!("Renderer::draw: Presenting changes.");
            driver.present()?;
        } else {
            trace!("Renderer::draw: No changes to present.");
        }
        Ok(())
    }

    /// Draws the content of a single terminal line.
    fn draw_line_content(
        &self,
        y_abs: usize,
        term_width: usize,
        term: &impl TerminalInterface,
        driver: &mut dyn Driver,
    ) -> Result<()> {
        trace!("Renderer::draw_line_content: Drawing line y={}", y_abs);
        let mut current_col: usize = 0;

        while current_col < term_width {
            let start_glyph = term.get_glyph(current_col, y_abs);
            let (eff_fg, eff_bg, eff_flags) = self.get_effective_colors_and_flags(
                start_glyph.attr.fg,
                start_glyph.attr.bg,
                start_glyph.attr.flags,
            );

            let char_for_log = if start_glyph.c == '\0' {
                ' '
            } else {
                start_glyph.c
            };
            trace!(
                "  Line {}, Col {}: Start glyph='{}' (attr:{:?}), EffectiveStyle(fg:{:?}, bg:{:?}, flags:{:?})",
                y_abs, current_col, char_for_log, start_glyph.attr, eff_fg, eff_bg, eff_flags
            );

            let cells_consumed = if start_glyph.c == '\0' {
                // Placeholder for wide char
                self.draw_placeholder_cell(current_col, y_abs, eff_bg, driver)?
            } else if start_glyph.c == ' ' {
                // Space character
                // Attempt to draw a run of spaces
                let space_run_len = self.draw_space_run(
                    current_col,
                    y_abs,
                    term_width,
                    &start_glyph,
                    term,
                    driver,
                )?;
                if space_run_len > 0 {
                    space_run_len // Consumed a run of spaces
                } else {
                    // Should not happen if start_glyph.c is ' ', draw_space_run should return at least 1.
                    // But as a fallback, treat as a single text segment.
                    warn!(
                        "Renderer::draw_line_content: draw_space_run returned 0 for a space at ({},{}). Treating as single char.",
                        current_col, y_abs
                    );
                    self.draw_text_segment(
                        current_col,
                        y_abs,
                        term_width,
                        &start_glyph,
                        term,
                        driver,
                    )?
                }
            } else {
                // Regular text character
                self.draw_text_segment(current_col, y_abs, term_width, &start_glyph, term, driver)?
            };

            if cells_consumed == 0 {
                // This case should ideally be prevented by draw_placeholder_cell, draw_space_run, or draw_text_segment
                // always returning at least 1 if they process the start_glyph.
                warn!(
                    "Renderer::draw_line_content: A draw segment reported consuming 0 cells at ({}, {}), char '{}'. Advancing by 1 to prevent loop.",
                    current_col, y_abs, start_glyph.c
                );
                current_col += 1;
            } else {
                current_col += cells_consumed;
            }
        }
        Ok(())
    }

    /// Draws a placeholder cell (typically the second half of a wide character).
    /// Returns the number of cells consumed (always 1 for a placeholder).
    fn draw_placeholder_cell(
        &self,
        x: usize,
        y: usize,
        effective_bg: Color,
        driver: &mut dyn Driver,
    ) -> Result<usize> {
        let rect = CellRect {
            x,
            y,
            width: 1,
            height: 1,
        };
        trace!(
            "    Line {}, Col {}: Placeholder. FillRect with bg={:?}",
            y, x, effective_bg
        );
        driver.fill_rect(rect, effective_bg)?;
        Ok(1) // Consumes 1 cell
    }

    /// Identifies and draws a run of space characters with the same effective background and flags.
    /// Returns the number of cells (spaces) consumed in this run.
    fn draw_space_run(
        &self,
        start_col: usize,
        y: usize,
        term_width: usize,
        start_glyph: &Glyph, // The glyph at start_col, known to be a space
        term: &impl TerminalInterface,
        driver: &mut dyn Driver,
    ) -> Result<usize> {
        // Effective attributes of the first space in the potential run
        let (_, start_eff_bg, start_eff_flags) = self.get_effective_colors_and_flags(
            start_glyph.attr.fg, // FG of space is usually ignored for rendering background
            start_glyph.attr.bg,
            start_glyph.attr.flags,
        );

        let mut space_run_len = 0;
        for x_offset in 0..(term_width - start_col) {
            let current_scan_col = start_col + x_offset;
            let glyph_at_scan = term.get_glyph(current_scan_col, y);

            // Effective attributes of the currently scanned glyph
            let (_, current_scan_eff_bg, current_scan_flags) = self.get_effective_colors_and_flags(
                glyph_at_scan.attr.fg,
                glyph_at_scan.attr.bg,
                glyph_at_scan.attr.flags,
            );

            // Break if not a space or if effective background/flags change
            if glyph_at_scan.c != ' '
                || current_scan_eff_bg != start_eff_bg
                || current_scan_flags != start_eff_flags
            {
                break;
            }
            space_run_len += 1;
        }

        if space_run_len == 0 {
            // This should not happen if called with start_glyph.c == ' '
            return Ok(0);
        }

        // Draw the identified run of spaces
        let rect = CellRect {
            x: start_col,
            y,
            width: space_run_len,
            height: 1,
        };
        trace!(
            "    Line {}, Col {}: Space run (len {}). FillRect with bg={:?}, flags={:?}",
            y, start_col, space_run_len, start_eff_bg, start_eff_flags
        );
        driver.fill_rect(rect, start_eff_bg)?; // Use effective_bg for filling spaces

        Ok(space_run_len) // Return number of cells consumed
    }

    /// Identifies and draws a run of non-space, non-placeholder text characters
    /// that share the same effective foreground, background, and flags.
    /// Returns the total number of cells consumed by this text segment (considering wide chars).
    fn draw_text_segment(
        &self,
        start_col: usize,
        y: usize,
        term_width: usize,
        start_glyph: &Glyph, // The glyph at start_col, known not to be ' ' or '\0'
        term: &impl TerminalInterface,
        driver: &mut dyn Driver,
    ) -> Result<usize> {
        // Effective attributes of the first character in the potential run
        let (start_eff_fg, start_eff_bg, start_eff_flags) = self.get_effective_colors_and_flags(
            start_glyph.attr.fg,
            start_glyph.attr.bg,
            start_glyph.attr.flags,
        );

        let mut run_text = String::new();
        let mut run_total_cell_width = 0; // Accumulates cell width (1 for narrow, 2 for wide)
        let mut current_scan_col = start_col; // Tracks the current column being scanned

        while current_scan_col < term_width {
            let glyph_at_scan = term.get_glyph(current_scan_col, y);

            // Stop conditions for the current text run:
            // 1. Encounter a space or placeholder (handled by other functions).
            // 2. Effective attributes change.
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
                break; // Attributes changed, end of current run
            }

            let char_display_width = get_char_display_width(glyph_at_scan.c);

            if char_display_width == 0 {
                // True zero-width char (e.g., combining mark not handled by precomposition)
                // Append to text but don't increment run_total_cell_width or current_scan_col significantly.
                // The main character it modifies should provide the width.
                // This logic assumes ZWCs are rare and don't form runs by themselves.
                trace!(
                    "    Line {}, Col {}: Appending zero-width char '{}' to text run. Scan column remains {}.",
                    y, current_scan_col, glyph_at_scan.c, current_scan_col
                );
                run_text.push(glyph_at_scan.c);
                // We need to advance past this ZWC in the grid for the next iteration.
                // If it's the *only* char processed, draw_line_content's advance-by-1 will handle it.
                // If it's part of a run, the next char will be at current_scan_col + 1.
                // For simplicity here, we assume the next iteration of the outer loop in draw_line_content
                // will correctly pick up from current_scan_col + (what this segment returns).
                // If this segment only contains ZWCs, it will return 0, and the outer loop advances by 1.
                current_scan_col += 1; // Advance scan past the ZWC for the next glyph.
                continue; // Don't add to run_total_cell_width
            }

            // Check if adding this character would exceed terminal width for the run
            if start_col + run_total_cell_width + char_display_width > term_width {
                break;
            }

            run_text.push(glyph_at_scan.c);
            run_total_cell_width += char_display_width;
            current_scan_col += char_display_width; // Advance scan by the cells this char occupies
        }

        if run_text.is_empty() {
            // This can happen if the start_glyph itself was a ZWC or something non-drawable
            // that didn't form a run. The outer loop in draw_line_content expects cells_consumed > 0.
            // Return 1 to ensure the outer loop advances past this problematic single cell.
            let advance_by = get_char_display_width(start_glyph.c).max(1);
            warn!(
                "    Line {}, Col {}: Single char '{}' (width {}) did not form text run. Advancing by {}.",
                y,
                start_col,
                start_glyph.c,
                get_char_display_width(start_glyph.c),
                advance_by
            );
            return Ok(advance_by);
        }

        // Draw the accumulated text run
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

        // After drawing the text run, if it contained wide characters,
        // their placeholder cells ('\0') need to be explicitly filled with the run's background.
        // The main loop in draw_line_content will eventually hit these '\0' cells and call draw_placeholder_cell,
        // which uses the placeholder's own background (which should match the wide char's).
        // So, no explicit placeholder filling is needed here if placeholder cells correctly inherit attributes.

        Ok(run_total_cell_width) // Return number of cells consumed by this run
    }

    /// Draws the terminal cursor as an overlay.
    fn draw_cursor_overlay(
        &self,
        cursor_abs_x: usize,
        cursor_abs_y: usize,
        term: &impl TerminalInterface,
        driver: &mut dyn Driver,
        term_width: usize,
        term_height: usize,
    ) -> Result<()> {
        trace!(
            "Renderer::draw_cursor_overlay: Screen cursor pos ({}, {})",
            cursor_abs_x, cursor_abs_y
        );

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

        let glyph_at_logical_cursor = term.get_glyph(cursor_abs_x, cursor_abs_y);
        let char_for_log1 = if glyph_at_logical_cursor.c == '\0' {
            ' '
        } else {
            glyph_at_logical_cursor.c
        };
        trace!(
            "  Cursor overlay: Glyph at logical cursor pos ({},{}): char='{}', attr={:?}",
            cursor_abs_x, cursor_abs_y, char_for_log1, glyph_at_logical_cursor.attr
        );

        if glyph_at_logical_cursor.c == '\0' && cursor_abs_x > 0 {
            // Cursor is on the placeholder of a wide character.
            // Use the attributes and character of the first half of the wide char.
            let first_half_glyph = term.get_glyph(cursor_abs_x - 1, cursor_abs_y);
            let char_for_log2 = if first_half_glyph.c == '\0' {
                '?'
            } else {
                first_half_glyph.c
            }; // Should not be '\0'
            trace!(
                "    Cursor on placeholder, using first half: char='{}' from col {}",
                char_for_log2,
                cursor_abs_x - 1
            );

            char_to_draw_at_cursor = first_half_glyph.c; // This is the wide char itself
            original_attrs_at_cursor = first_half_glyph.attr;
            physical_cursor_x_for_draw = cursor_abs_x - 1; // Draw the cursor at the start of the wide char
        } else {
            // Cursor is on a regular character or the first cell of a wide character.
            char_to_draw_at_cursor = glyph_at_logical_cursor.c;
            original_attrs_at_cursor = glyph_at_logical_cursor.attr;
            physical_cursor_x_for_draw = cursor_abs_x;
        }

        // Get effective colors of the cell *under* the cursor, before inversion for cursor
        let (resolved_original_fg, resolved_original_bg, resolved_original_flags) = self
            .get_effective_colors_and_flags(
                original_attrs_at_cursor.fg,
                original_attrs_at_cursor.bg,
                original_attrs_at_cursor.flags, // Original flags, REVERSE already handled
            );
        trace!(
            "    Original cell effective attrs for cursor: fg={:?}, bg={:?}, flags={:?}",
            resolved_original_fg, resolved_original_bg, resolved_original_flags
        );

        // For cursor: swap effective FG and BG
        let cursor_char_fg = resolved_original_bg;
        let cursor_cell_bg = resolved_original_fg;
        // Flags for cursor rendering should be the effective flags of the underlying cell
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

        // If the character under cursor was a placeholder, draw a space for the cursor.
        // Otherwise, draw the actual character.
        let final_char_to_draw_for_cursor = if char_to_draw_at_cursor == '\0' {
            ' '
        } else {
            char_to_draw_at_cursor
        };
        trace!(
            "    Drawing cursor overlay: char='{}' at physical ({},{}) with style: {:?}",
            final_char_to_draw_for_cursor, physical_cursor_x_for_draw, cursor_abs_y, style
        );

        driver.draw_text_run(coords, &final_char_to_draw_for_cursor.to_string(), style)?;
        Ok(())
    }
}

impl Default for Renderer {
    fn default() -> Self {
        Self::new()
    }
}
