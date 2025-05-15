// src/renderer.rs

//! This module defines the `Renderer`.
//!
//! The `Renderer`'s primary responsibility is to translate the visual state of the
//! `TerminalEmulator` into a series of abstract drawing commands that can be
//! executed by a `Driver`. It is designed to be backend-agnostic, meaning
//! it does not contain any platform-specific drawing code (e.g., X11/Xft calls
//! or direct ANSI escape sequence generation). Instead, it relies on the `Driver`
//! trait to provide the necessary drawing primitives.

// Updated to use new structs from backends::mod_rs
use crate::backends::{Driver, CellCoords, TextRunStyle, CellRect};
use crate::term::TerminalEmulator;
use crate::glyph::{Color, AttrFlags, Glyph};
use crate::term::unicode::get_char_display_width;

use anyhow::Result;
use log::{trace, warn};

/// The `Renderer` translates `TerminalEmulator` state into abstract drawing commands.
///
/// It processes the terminal's grid, identifies dirty lines (lines that have changed
/// and require redrawing), and batches sequences of characters with identical visual
/// attributes into efficient drawing calls to the provided `Driver`. It also handles
/// rendering the cursor based on the emulator's state.
///
/// The `Renderer` itself is currently stateless beyond the scope of a single `draw` call.
pub struct Renderer {
    // Future enhancements might include caching mechanisms or configuration options,
    // but for now, it's kept simple.
}

impl Renderer {
    /// Creates a new `Renderer` instance.
    pub fn new() -> Self {
        Self {}
    }

    /// Draws the current state of the `TerminalEmulator` using the provided `Driver`.
    ///
    /// This method performs the core rendering logic:
    /// 1. Retrieves the terminal's current dimensions.
    /// 2. Fetches the list of "dirty" lines from the `TerminalEmulator`. This call
    ///    also clears the dirty flags within the emulator.
    /// 3. For each dirty line, calls `self.draw_dirty_line()`.
    /// 4. Renders the cursor by calling `self.draw_cursor()` if it's marked visible.
    /// 5. Calls `driver.present()` to flush all drawing commands to the display if
    ///    any drawing operations (including cursor changes) occurred.
    ///
    /// # Arguments
    ///
    /// * `term`: A mutable reference to the `TerminalEmulator`. Its `take_dirty_lines()`
    ///           method will be called, which modifies its internal state by clearing
    ///           the dirty line flags.
    /// * `driver`: A mutable reference to a `Driver` implementation, which will
    ///             execute the low-level drawing commands.
    ///
    /// # Returns
    ///
    /// * `Result<()>`: `Ok(())` if drawing was successful, or an `Err` if the driver
    ///                 reported an error during a drawing operation.
    pub fn draw(&self, term: &mut TerminalEmulator, driver: &mut dyn Driver) -> Result<()> {
        let (term_width, term_height) = term.dimensions();

        if term_width == 0 || term_height == 0 {
            // Nothing to draw if terminal dimensions are zero.
            return Ok(());
        }

        let dirty_line_indices = term.take_dirty_lines();
        let mut something_was_drawn = !dirty_line_indices.is_empty();

        for y_abs in dirty_line_indices {
            if y_abs >= term_height {
                warn!(
                    "Renderer: Dirty line index {} is out of bounds (height {}). Skipping.",
                    y_abs, term_height
                );
                continue;
            }
            self.draw_dirty_line(y_abs, term_width, term, driver)?;
        }

        // Assumes TerminalEmulator will have a method `is_cursor_visible()`
        // This replaces direct access to `term.dec_modes.cursor_visible`
        if term.is_cursor_visible() {
            self.draw_cursor(term, driver, term_width, term_height)?;
            something_was_drawn = true; // Cursor drawing also counts as drawing.
        }

        if something_was_drawn {
            driver.present()?;
        }

        Ok(())
    }

    /// Draws a single dirty line of the terminal.
    ///
    /// Iterates through the columns of the specified line (`y_abs`), calling
    /// `draw_run_or_fallback_char` to handle rendering segments of text or individual
    /// characters.
    ///
    /// # Arguments
    /// * `y_abs`: The absolute row index of the dirty line to draw.
    /// * `term_width`: The total width of the terminal in columns.
    /// * `term`: A reference to the `TerminalEmulator` to get glyph data.
    /// * `driver`: A mutable reference to the `Driver` for drawing operations.
    fn draw_dirty_line(
        &self,
        y_abs: usize,
        term_width: usize,
        term: &TerminalEmulator, // Changed to immutable as get_glyph doesn't need mut
        driver: &mut dyn Driver,
    ) -> Result<()> {
        trace!("Renderer: Drawing dirty line {}", y_abs);
        let mut current_col = 0;
        while current_col < term_width {
            let cols_drawn =
                self.draw_run_or_fallback_char(y_abs, current_col, term_width, term, driver)?;
            if cols_drawn == 0 {
                // Defensive break to prevent infinite loops if no columns are consumed.
                // This might happen with zero-width characters if not handled correctly,
                // or if get_char_display_width returns 0 incorrectly.
                warn!("Renderer: draw_run_or_fallback_char consumed 0 columns at ({}, {}). Breaking line draw to prevent loop.", current_col, y_abs);
                break;
            }
            current_col += cols_drawn;
        }
        Ok(())
    }

    /// Draws a run of text with identical attributes, or a single fallback character.
    ///
    /// This function starts at `current_col` on line `y_abs`. It first checks if the
    /// character at this position is a wide character placeholder (`\0`). If so, it fills
    /// that single cell. Otherwise, it attempts to scan forward to find a sequence (run)
    /// of characters that share the same visual attributes. If a run is found, it's
    /// drawn with `driver.draw_text_run()`. If no run is formed (e.g., the character at
    /// `current_col` has unique attributes or is a space), it's drawn as a single
    /// character (or a filled rectangle for spaces).
    ///
    /// # Arguments
    /// * `y_abs`: The absolute row index.
    /// * `current_col`: The starting column index on the line to process.
    /// * `term_width`: The total width of the terminal.
    /// * `term`: A reference to the `TerminalEmulator`.
    /// * `driver`: A mutable reference to the `Driver`.
    ///
    /// # Returns
    /// * `Result<usize>`: The number of columns consumed by this drawing operation.
    fn draw_run_or_fallback_char(
        &self,
        y_abs: usize,
        current_col: usize,
        term_width: usize,
        term: &TerminalEmulator, // Changed to immutable
        driver: &mut dyn Driver,
    ) -> Result<usize> {
        let start_glyph = term.get_glyph(current_col, y_abs);

        // Handle wide character placeholders ('\0').
        // These signify the second cell of a wide character.
        // We fill this placeholder cell with the correct background color.
        if start_glyph.c == '\0' { // WIDE_CHAR_PLACEHOLDER
            let (_, eff_bg, _) = self.get_effective_colors_and_flags(
                start_glyph.attr.fg,
                start_glyph.attr.bg,
                start_glyph.attr.flags,
            );
            // Updated fill_rect call
            let rect = CellRect { x: current_col, y: y_abs, width: 1, height: 1 };
            driver.fill_rect(rect, eff_bg)?;
            return Ok(1); // Consumed one column.
        }

        let (eff_fg, eff_bg, run_flags) = self.get_effective_colors_and_flags(
            start_glyph.attr.fg,
            start_glyph.attr.bg,
            start_glyph.attr.flags,
        );

        let mut run_text = String::new();
        let run_start_col = current_col;
        let mut run_width_in_cells = 0;

        // Scan forward to build a run of characters with the same attributes.
        let mut scan_col = current_col;
        while scan_col < term_width {
            let glyph_at_scan = term.get_glyph(scan_col, y_abs);

            if glyph_at_scan.c == '\0' { // WIDE_CHAR_PLACEHOLDER
                if run_text.is_empty() && scan_col == current_col {
                    // This is the case where the loop started on a placeholder,
                    // which should have been caught by the initial check.
                    // If it happens here, it's an unexpected state.
                     warn!("Renderer: Encountered unexpected isolated '\\0' at ({}, {}) during run scan start.", scan_col, y_abs);
                     // Fill this single cell and advance.
                     let rect = CellRect { x: scan_col, y: y_abs, width: 1, height: 1 };
                     let (_, placeholder_bg, _) = self.get_effective_colors_and_flags(
                         glyph_at_scan.attr.fg, glyph_at_scan.attr.bg, glyph_at_scan.attr.flags
                     );
                     driver.fill_rect(rect, placeholder_bg)?;
                     return Ok(1);
                } else if !run_text.is_empty() && scan_col == (run_start_col + run_width_in_cells) {
                    // This is the placeholder for the *last* char added to run_text,
                    // which must have been wide. We've already accounted for its width.
                    // So, we just advance past it.
                    scan_col += 1;
                    continue;
                } else {
                    // Placeholder encountered mid-run or not belonging to the current char.
                    // This implies the run should end *before* this placeholder.
                    break;
                }
            }

            let (current_glyph_eff_fg, current_glyph_eff_bg, current_glyph_run_flags) =
                self.get_effective_colors_and_flags(
                    glyph_at_scan.attr.fg,
                    glyph_at_scan.attr.bg,
                    glyph_at_scan.attr.flags,
                );

            if current_glyph_eff_fg == eff_fg
                && current_glyph_eff_bg == eff_bg
                && current_glyph_run_flags == run_flags
            {
                let char_display_width = get_char_display_width(glyph_at_scan.c);
                if char_display_width == 0 { // Non-spacing character, skip or handle if needed
                    scan_col +=1; // Advance past it, effectively ignoring it for run building.
                    continue;
                }
                if run_start_col + run_width_in_cells + char_display_width > term_width {
                    break; // Character would overflow; end run.
                }
                run_text.push(glyph_at_scan.c);
                run_width_in_cells += char_display_width;
                scan_col += char_display_width; // Advance by actual display width
            } else {
                break; // Attribute mismatch; end of run.
            }
        }

        if !run_text.is_empty() {
            // Updated draw_text_run call
            let coords = CellCoords { x: run_start_col, y: y_abs };
            let style = TextRunStyle { fg: eff_fg, bg: eff_bg, flags: run_flags };
            driver.draw_text_run(coords, &run_text, style)?;
            Ok(run_width_in_cells) // Consumed columns equal to the run's width.
        } else {
            // No run formed; handle single character.
            // `get_char_display_width` returns 0 for some control chars, use .max(1) to ensure progress.
            let char_display_width = get_char_display_width(start_glyph.c).max(1);
            if start_glyph.c == ' ' {
                // Updated fill_rect call
                let rect = CellRect { x: current_col, y: y_abs, width: char_display_width, height: 1 };
                driver.fill_rect(rect, eff_bg)?;
            } else {
                // Updated draw_text_run call
                let coords = CellCoords { x: current_col, y: y_abs };
                let style = TextRunStyle { fg: eff_fg, bg: eff_bg, flags: run_flags };
                driver.draw_text_run(coords, &start_glyph.c.to_string(), style)?;
            }
            Ok(char_display_width) // Consumed columns for the single character.
        }
    }

    /// Draws the terminal cursor if it's visible.
    ///
    /// Implements a block cursor by swapping the foreground and background colors
    /// of the character cell under the cursor.
    ///
    /// # Arguments
    /// * `term`: A reference to the `TerminalEmulator` to get cursor and glyph data.
    /// * `driver`: A mutable reference to the `Driver` for drawing.
    /// * `term_width`: The total width of the terminal.
    /// * `term_height`: The total height of the terminal.
    fn draw_cursor(
        &self,
        term: &TerminalEmulator, // Changed to immutable
        driver: &mut dyn Driver,
        term_width: usize,
        term_height: usize,
    ) -> Result<()> {
        // Assumes TerminalEmulator will have a method `get_screen_cursor_pos()`
        // This replaces direct access to `term.screen.cursor.x` and `term.screen.cursor.y`
        let (cursor_abs_x, cursor_abs_y) = term.get_screen_cursor_pos();

        // Guard clause: If cursor is out of bounds, warn and do nothing further.
        if !(cursor_abs_x < term_width && cursor_abs_y < term_height) {
            warn!(
                "Renderer: Cursor at ({}, {}) is out of terminal bounds ({}x{}). Not drawing cursor.",
                cursor_abs_x, cursor_abs_y, term_width, term_height
            );
            return Ok(());
        }

        // Cursor is within bounds, proceed with drawing.
        let glyph_under_cursor: Glyph = term.get_glyph(cursor_abs_x, cursor_abs_y);
        
        // If the character under the cursor is a wide char placeholder, draw on the *actual* character
        // which is one cell to the left.
        let (char_to_draw_at_cursor, physical_cursor_x) = if glyph_under_cursor.c == '\0' && cursor_abs_x > 0 {
            // This is the second half of a wide char. Get the first half.
            (term.get_glyph(cursor_abs_x - 1, cursor_abs_y).c, cursor_abs_x -1)
        } else {
            (glyph_under_cursor.c, cursor_abs_x)
        };
        
        let original_attrs = term.get_glyph(physical_cursor_x, cursor_abs_y).attr;


        // For block cursor, swap FG and BG.
        // Use the original glyph's FG as new BG, and original BG as new FG.
        let cursor_char_fg = original_attrs.bg; // Swapped
        let cursor_cell_bg = original_attrs.fg; // Swapped
        
        // Ensure REVERSE is not applied again by the driver if it was part of original_attrs.
        // The swap above effectively handles REVERSE for the cursor block.
        let cursor_display_flags = original_attrs.flags.difference(AttrFlags::REVERSE);

        // Updated draw_text_run call
        let coords = CellCoords { x: physical_cursor_x, y: cursor_abs_y };
        let style = TextRunStyle { fg: cursor_char_fg, bg: cursor_cell_bg, flags: cursor_display_flags };
        driver.draw_text_run(coords, &char_to_draw_at_cursor.to_string(), style)?;

        Ok(())
    }

    /// Determines the effective foreground, background colors, and rendering flags for a glyph.
    /// This helper function is responsible for applying the `AttrFlags::REVERSE` logic:
    /// if `REVERSE` is set in the input `flags`, it swaps the foreground and background colors.
    /// The `REVERSE` flag itself is then removed from the returned flags, as its visual
    /// effect has been incorporated into the colors.
    ///
    /// # Arguments
    /// * `fg`: The original foreground `Color`.
    /// * `bg`: The original background `Color`.
    /// * `flags`: The original `AttrFlags`.
    ///
    /// # Returns
    /// A tuple `(effective_fg, effective_bg, flags_for_driver)`.
    /// `flags_for_driver` will be the input `flags` with `AttrFlags::REVERSE` removed
    /// if it was present and applied.
    fn get_effective_colors_and_flags(
        &self,
        fg: Color,
        bg: Color,
        flags: AttrFlags,
    ) -> (Color, Color, AttrFlags) {
        if flags.contains(AttrFlags::REVERSE) {
            // If REVERSE flag is set, swap foreground and background colors.
            // The REVERSE flag itself is not passed to the driver, as its effect
            // is now handled by this color swap.
            (bg, fg, flags.difference(AttrFlags::REVERSE))
        } else {
            // If REVERSE is not set, use colors and flags as they are.
            (fg, bg, flags)
        }
    }
}

// Provides a default constructor for `Renderer`.
impl Default for Renderer {
    fn default() -> Self {
        Self::new()
    }
}

