// myterm/src/renderer.rs

//! This module defines the `Renderer`.
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
use crate::term::TerminalInterface;
use crate::term::unicode::get_char_display_width; // Trait for interacting with the terminal state.

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
    pub fn draw(
        &mut self,
        term: &mut (impl TerminalInterface + ?Sized), // Added + ?Sized
        driver: &mut dyn Driver,
    ) -> Result<()> {
        let (term_width, term_height) = term.dimensions();

        // Avoid drawing if terminal dimensions are invalid.
        if term_width == 0 || term_height == 0 {
            trace!(
                "Renderer::draw: Terminal dimensions zero ({}x{}), skipping draw.",
                term_width, term_height
            );
            return Ok(());
        }

        // Retrieve and clear dirty line flags from the terminal.
        let initially_dirty_lines_from_term: HashSet<usize> =
            term.take_dirty_lines().into_iter().collect();

        trace!(
            "Renderer::draw: Initially dirty lines from term: {:?}, term_dims: {}x{}, first_draw: {}",
            initially_dirty_lines_from_term, term_width, term_height, self.first_draw
        );

        let mut lines_to_draw_content: HashSet<usize> = initially_dirty_lines_from_term;
        let (cursor_abs_x, cursor_abs_y) = term.get_screen_cursor_pos();
        let mut something_was_drawn = !lines_to_draw_content.is_empty();

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
            if term.is_cursor_visible() && cursor_abs_y < term_height {
                if lines_to_draw_content.insert(cursor_abs_y) {
                    trace!(
                        "Renderer::draw: Added cursor line y={} to draw set as it was not initially dirty.",
                        cursor_abs_y
                    );
                }
                // Ensuring something_was_drawn is true if we are drawing the cursor line.
                something_was_drawn = true;
            }
        }

        // If there are any lines to draw content for (either from initial dirty set or full clear).
        if !lines_to_draw_content.is_empty() {
            something_was_drawn = true;
        }

        let mut sorted_lines_to_draw: Vec<usize> = lines_to_draw_content.into_iter().collect();
        sorted_lines_to_draw.sort_unstable(); // Draw in logical order.
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

        // Overlay the cursor if it's visible.
        if term.is_cursor_visible() {
            trace!("Renderer::draw: Cursor is visible, calling draw_cursor_overlay.");
            self.draw_cursor_overlay(
                cursor_abs_x,
                cursor_abs_y,
                term,
                driver,
                term_width,
                term_height,
            )?;
            something_was_drawn = true; // Drawing cursor means something was drawn.
        }

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
        term: &(impl TerminalInterface + ?Sized), // Added + ?Sized
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

            // Use a space for logging if the character is null (wide char placeholder).
            let char_for_log = if start_glyph.c == '\0' {
                ' '
            } else {
                start_glyph.c
            };
            debug!(
                // Changed to debug for potentially verbose output, trace for finer details.
                "  Line {}, Col {}: Start glyph='{}' (attr:{:?}), EffectiveStyle(fg:{:?}, bg:{:?}, flags:{:?})",
                y_abs, current_col, char_for_log, start_glyph.attr, eff_fg, eff_bg, eff_flags
            );

            // Dispatch to appropriate drawing function based on glyph content.
            let cells_consumed = if start_glyph.c == '\0' {
                // Placeholder for the second half of a wide character.
                self.draw_placeholder_cell(current_col, y_abs, eff_bg, driver)?
            } else if start_glyph.c == ' ' {
                // Space character; attempt to draw a run of spaces for optimization.
                self.draw_space_run(
                    current_col,
                    y_abs,
                    term_width,
                    &start_glyph, // Pass the initial space glyph
                    term,
                    driver,
                )?
            } else {
                // Regular text character; attempt to draw a run of text.
                self.draw_text_segment(current_col, y_abs, term_width, &start_glyph, term, driver)?
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
        Ok(1) // A placeholder consumes 1 cell.
    }

    /// Identifies and draws a contiguous run of space characters that share the
    /// same effective background color and attribute flags.
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
        term: &(impl TerminalInterface + ?Sized), // Added + ?Sized
        driver: &mut dyn Driver,
    ) -> Result<usize> {
        // Determine the effective style of the first space in the potential run.
        // The foreground of a space is usually irrelevant for background filling.
        let (_, start_eff_bg, start_eff_flags) = self.get_effective_colors_and_flags(
            start_glyph.attr.fg,
            start_glyph.attr.bg,
            start_glyph.attr.flags,
        );

        let mut space_run_len = 0;
        // Scan right from start_col to find contiguous spaces with matching style.
        for x_offset in 0..(term_width - start_col) {
            let current_scan_col = start_col + x_offset;
            let glyph_at_scan = term.get_glyph(current_scan_col, y);

            let (_, current_scan_eff_bg, current_scan_flags) = self.get_effective_colors_and_flags(
                glyph_at_scan.attr.fg,
                glyph_at_scan.attr.bg,
                glyph_at_scan.attr.flags,
            );

            // Break the run if the character is not a space, or if its effective style differs.
            if glyph_at_scan.c != ' '
                || current_scan_eff_bg != start_eff_bg
                || current_scan_flags != start_eff_flags
            // Ensure flags like underline also match if relevant for spaces
            {
                break;
            }
            space_run_len += 1;
        }

        if space_run_len == 0 {
            // This should ideally not be reached if called because start_glyph.c was ' '.
            // If it does, it means the start_glyph itself didn't form a run of 1.
            warn!(
                "Renderer::draw_space_run: Detected 0-length space run at ({},{}). This might indicate an issue.",
                start_col, y
            );
            return Ok(0); // No cells consumed by this specific call.
        }

        // Draw the identified run of spaces using a single fill_rect.
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
        driver.fill_rect(rect, start_eff_bg)?;

        Ok(space_run_len)
    }

    /// Identifies and draws a contiguous run of non-space, non-placeholder text
    /// characters that share the same effective foreground, background, and attribute flags.
    ///
    /// # Returns
    /// `Ok(usize)`: The total number of cells consumed by this text segment,
    ///              accounting for wide characters.
    fn draw_text_segment(
        &self,
        start_col: usize,
        y: usize,
        term_width: usize,
        start_glyph: &Glyph, // The glyph at start_col, known not to be ' ' or '\0'.
        term: &(impl TerminalInterface + ?Sized), // Added + ?Sized
        driver: &mut dyn Driver,
    ) -> Result<usize> {
        let (start_eff_fg, start_eff_bg, start_eff_flags) = self.get_effective_colors_and_flags(
            start_glyph.attr.fg,
            start_glyph.attr.bg,
            start_glyph.attr.flags,
        );

        let mut run_text = String::new();
        let mut run_total_cell_width = 0; // Accumulates cell width (1 for narrow, 2 for wide).
        let mut current_scan_col = start_col;

        while current_scan_col < term_width {
            let glyph_at_scan = term.get_glyph(current_scan_col, y);

            // Stop conditions: character is a space or placeholder, or attributes change.
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
                break; // Effective style changed, end of current run.
            }

            let char_display_width = get_char_display_width(glyph_at_scan.c);

            // Handle true zero-width characters (e.g., combining marks not precomposed).
            // These are appended to the string but do not consume additional cell width
            // or advance the primary scan column for attribute checking.
            if char_display_width == 0 {
                trace!(
                    "    Line {}, Col {}: Appending zero-width char '{}' to text run. Scan column remains {}.",
                    y, current_scan_col, glyph_at_scan.c, current_scan_col
                );
                run_text.push(glyph_at_scan.c);
                // Advance past the ZWC for the *next glyph* in the grid for the *next iteration* of this loop.
                current_scan_col += 1;
                continue; // Do not add to run_total_cell_width for ZWCs.
            }

            // Ensure the character (and its potential wide counterpart) fits.
            if start_col + run_total_cell_width + char_display_width > term_width {
                break; // Character would overflow the line width for this run.
            }

            run_text.push(glyph_at_scan.c);
            run_total_cell_width += char_display_width;
            // Advance scan by the number of cells this character occupies.
            // For a wide char, this jumps past its placeholder cell for the next iteration.
            current_scan_col += char_display_width;
        }

        if run_text.is_empty() {
            // This might happen if the start_glyph itself was a ZWC or other non-advancing char.
            // Ensure the outer loop in draw_line_content advances.
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

        // Draw the accumulated text run.
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

    /// Draws the terminal cursor as an overlay on top of the existing cell content.
    ///
    /// The cursor is typically rendered by inverting the foreground and background
    /// colors of the cell it occupies. For wide characters, the cursor is drawn
    /// at the start of the character.
    ///
    /// # Arguments
    /// * `cursor_abs_x`: Absolute 0-based column of the cursor.
    /// * `cursor_abs_y`: Absolute 0-based row of the cursor.
    /// * `term`: A reference to an object implementing `TerminalInterface`.
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
        term: &(impl TerminalInterface + ?Sized), // Added + ?Sized
        driver: &mut dyn Driver,
        term_width: usize,
        term_height: usize,
    ) -> Result<()> {
        trace!(
            "Renderer::draw_cursor_overlay: Screen cursor pos ({}, {})",
            cursor_abs_x, cursor_abs_y
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

        let glyph_at_logical_cursor = term.get_glyph(cursor_abs_x, cursor_abs_y);
        let char_for_log1 = if glyph_at_logical_cursor.c == '\0' {
            ' ' // Log placeholder as space for readability
        } else {
            glyph_at_logical_cursor.c
        };
        trace!(
            "  Cursor overlay: Glyph at logical cursor pos ({},{}): char='{}', attr={:?}",
            cursor_abs_x, cursor_abs_y, char_for_log1, glyph_at_logical_cursor.attr
        );

        // If cursor is on the second half of a wide character (placeholder '\0'),
        // adjust to draw the cursor over the first half of that wide character.
        if glyph_at_logical_cursor.c == '\0' && cursor_abs_x > 0 {
            let first_half_glyph = term.get_glyph(cursor_abs_x - 1, cursor_abs_y);
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
            resolved_original_fg, resolved_original_bg, resolved_original_flags
        );

        // For cursor rendering, typically swap effective FG and BG.
        let cursor_char_fg = resolved_original_bg;
        let cursor_cell_bg = resolved_original_fg;
        // Flags for cursor rendering should be the effective flags of the underlying cell.
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

        // If the character under cursor was a placeholder (e.g., second half of wide char),
        // draw a space for the cursor block itself. Otherwise, draw the actual character inverted.
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

#[cfg(test)]
mod tests;
