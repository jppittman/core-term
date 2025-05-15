// src/renderer.rs

//! This module defines the `Renderer`.
//!
//! The `Renderer`'s primary responsibility is to translate the visual state of the
//! `TerminalEmulator` into a series of abstract drawing commands that can be
//! executed by a `Driver`. It is designed to be backend-agnostic.
//! It defines default foreground and background colors for resolving
//! `Color::Default` from glyph attributes.

use crate::backends::{Driver, CellCoords, TextRunStyle, CellRect};
use crate::term::TerminalInterface;
// Glyph import was previously warned as unused, but it's used in tests and for DEFAULT_GLYPH.
// It's not directly used in Renderer methods but is part of the crate's glyph system.
// Let's keep it if it's used by the associated test module or other parts of the crate.
// If truly unused by this file's direct logic, it could be removed, but given its
// relevance to the overall rendering process via TerminalInterface, it's fine.
use crate::glyph::{Color, AttrFlags, NamedColor};
use crate::term::unicode::get_char_display_width;

use anyhow::Result;
// error import was previously warned as unused. If no error! calls are made, it can be removed.
// Let's keep it for now in case future error handling needs it.
use log::{trace, warn};


// Define default colors within the Renderer module.
// These are the concrete colors that `Color::Default` will resolve to.
const RENDERER_DEFAULT_FG: Color = Color::Named(NamedColor::White);
const RENDERER_DEFAULT_BG: Color = Color::Named(NamedColor::Black);


/// The `Renderer` translates `TerminalEmulator` state into abstract drawing commands.
pub struct Renderer {
    // No state needed for now, but could hold caching or config options.
}

impl Renderer {
    /// Creates a new `Renderer` instance.
    pub fn new() -> Self {
        Self {}
    }

    /// Draws the current state of the `TerminalEmulator` using the provided `Driver`.
    /// This method ensures that all `Color::Default` are resolved before calling driver methods.
    pub fn draw(&self, term: &mut impl TerminalInterface, driver: &mut dyn Driver) -> Result<()> {
        let (term_width, term_height) = term.dimensions();

        if term_width == 0 || term_height == 0 {
            trace!("Renderer::draw: Terminal dimensions are zero ({}x{}), nothing to draw.", term_width, term_height);
            return Ok(());
        }
        
        let dirty_line_indices = term.take_dirty_lines();
        let mut something_was_drawn = !dirty_line_indices.is_empty();
        trace!("Renderer::draw: Dirty lines reported: {:?}", dirty_line_indices);


        // Condition for a full screen clear:
        // True if all lines are marked dirty and form a contiguous block from line 0.
        let is_full_refresh = term_height > 0 && // Ensure term_height is not zero
                              dirty_line_indices.len() == term_height &&
                              dirty_line_indices.iter().enumerate().all(|(i, &dl_idx)| i == dl_idx);

        if is_full_refresh {
            trace!("Renderer::draw: All lines dirty for {}x{} terminal, performing full clear_all with RENDERER_DEFAULT_BG ({:?}).", term_width, term_height, RENDERER_DEFAULT_BG);
            driver.clear_all(RENDERER_DEFAULT_BG)?;
            something_was_drawn = true;
        }


        for y_abs_ref in &dirty_line_indices {
            let y_abs = *y_abs_ref;
            if y_abs >= term_height {
                warn!(
                    "Renderer::draw: Dirty line index {} is out of bounds (height {}). Skipping.",
                    y_abs, term_height
                );
                continue;
            }
            // Draw each dirty line.
            self.draw_dirty_line(y_abs, term_width, term, driver)?;
        }

        // Draw the cursor if it's visible.
        if term.is_cursor_visible() {
            trace!("Renderer::draw: Cursor is visible, calling draw_cursor.");
            self.draw_cursor(term, driver, term_width, term_height)?;
            something_was_drawn = true;
        } else {
            trace!("Renderer::draw: Cursor is not visible.");
        }

        // If any drawing operations occurred, present the changes to the display.
        if something_was_drawn {
            trace!("Renderer::draw: Something was drawn, calling driver.present().");
            driver.present()?;
        } else {
            trace!("Renderer::draw: Nothing was drawn, skipping driver.present().");
        }

        Ok(())
    }

    /// Draws a single dirty line of the terminal.
    /// Iterates through the line, forming runs of characters with identical attributes
    /// and drawing them. Prioritizes FillRect for spaces and placeholders.
    fn draw_dirty_line(
        &self,
        y_abs: usize,
        term_width: usize,
        term: &impl TerminalInterface,
        driver: &mut dyn Driver,
    ) -> Result<()> {
        trace!("Renderer::draw_dirty_line: Drawing line y={}", y_abs);
        let mut current_col = 0;
        while current_col < term_width {
            let start_glyph = term.get_glyph(current_col, y_abs);
            trace!("  Col {}: Start glyph='{}' ({:?})", current_col, start_glyph.c, start_glyph.attr);

            // Determine effective colors and flags for the starting glyph of the potential segment
            let (start_eff_fg, start_eff_bg, start_flags) =
                self.get_effective_colors_and_flags(
                    start_glyph.attr.fg,
                    start_glyph.attr.bg,
                    start_glyph.attr.flags,
                );
            trace!("    Effective attributes: fg={:?}, bg={:?}, flags={:?}", start_eff_fg, start_eff_bg, start_flags);


            // Case 1: Handle a run of spaces
            if start_glyph.c == ' ' {
                let mut space_run_width_cells = 0;
                let mut scan_col = current_col;
                trace!("    Start glyph is a space. Scanning for space run...");
                while scan_col < term_width {
                    let glyph_at_scan = term.get_glyph(scan_col, y_abs);
                    // For spaces, only effective background and flags need to match for a run.
                    let (_, current_glyph_eff_bg, current_glyph_flags) =
                        self.get_effective_colors_and_flags(
                            glyph_at_scan.attr.fg, // fg of space doesn't affect its background fill
                            glyph_at_scan.attr.bg,
                            glyph_at_scan.attr.flags,
                        );

                    if glyph_at_scan.c == ' ' &&
                       current_glyph_eff_bg == start_eff_bg && // Background must match start_glyph's effective BG
                       current_glyph_flags == start_flags     // Flags must match start_glyph's effective flags
                    {
                        // Each space glyph contributes 1 to the cell width of the run
                        space_run_width_cells += 1; // Spaces are always 1 cell wide
                        scan_col += 1; // Advance by one cell for a space
                    } else {
                        trace!("    Space run ended at col {}. Next char: '{}', eff_bg: {:?}, eff_flags: {:?}", scan_col, glyph_at_scan.c, current_glyph_eff_bg, current_glyph_flags);
                        break; // End of space run
                    }
                }

                if space_run_width_cells > 0 {
                    let rect = CellRect { x: current_col, y: y_abs, width: space_run_width_cells, height: 1 };
                    trace!("    Found space run: width={}, from_col={}, bg={:?}. Calling FillRect.", space_run_width_cells, current_col, start_eff_bg);
                    driver.fill_rect(rect, start_eff_bg)?; // Fill with the determined effective background
                    current_col += space_run_width_cells;
                    continue; // Process next segment of the line
                } else {
                    // This should not happen if start_glyph.c was ' ' initially,
                    // as space_run_width_cells would be at least 1.
                    // However, as a fallback, treat as a single non-run character.
                    trace!("    Space run scan resulted in zero width. This is unexpected. Treating as single char.");
                }
            }

            // Case 2: Handle a wide char placeholder ('\0')
            // These are single cells that need to be filled with their own background.
            if start_glyph.c == '\0' {
                let rect = CellRect { x: current_col, y: y_abs, width: 1, height: 1 };
                // The background color for the placeholder is start_eff_bg,
                // which was derived from the placeholder's own attributes.
                trace!("    Glyph is placeholder. Calling FillRect for col {} with bg={:?}", current_col, start_eff_bg);
                driver.fill_rect(rect, start_eff_bg)?;
                current_col += 1; // Placeholders are 1 cell wide
                continue; // Process next segment of the line
            }

            // Case 3: Handle a run of printable, non-space, non-placeholder characters
            // This will now only be entered if start_glyph.c is not ' ' and not '\0'.
            trace!("    Start glyph is printable char '{}'. Scanning for text run...", start_glyph.c);
            let mut run_text = String::new();
            let run_start_col = current_col;
            let mut run_width_in_cells = 0;
            let mut scan_col_for_text = current_col;

            while scan_col_for_text < term_width {
                let glyph_at_scan = term.get_glyph(scan_col_for_text, y_abs);

                // If we hit a space or placeholder, the text run ends.
                if glyph_at_scan.c == ' ' || glyph_at_scan.c == '\0' {
                    trace!("    Text run ended at col {}. Encountered space or placeholder.", scan_col_for_text);
                    break;
                }

                let (current_glyph_eff_fg, current_glyph_eff_bg, current_glyph_flags) =
                    self.get_effective_colors_and_flags(
                        glyph_at_scan.attr.fg,
                        glyph_at_scan.attr.bg,
                        glyph_at_scan.attr.flags,
                    );

                // Check if the current glyph can extend the run (attributes must match)
                if current_glyph_eff_fg == start_eff_fg &&
                   current_glyph_eff_bg == start_eff_bg &&
                   current_glyph_flags == start_flags
                {
                    let char_display_width = get_char_display_width(glyph_at_scan.c);
                    // Skip zero-width characters for drawing, but advance past them if they are part of a sequence.
                    if char_display_width == 0 {
                        trace!("    Skipping zero-width char '{}' at col {}", glyph_at_scan.c, scan_col_for_text);
                        scan_col_for_text += 1; // Assume zero-width chars occupy 1 logical cell for scanning
                        continue;
                    }

                    // Ensure the character (and its width) fits on the line before adding it to the run
                    if current_col + run_width_in_cells + char_display_width > term_width {
                        trace!("    Text run ended at col {}. Char '{}' (width {}) would overflow.", scan_col_for_text, glyph_at_scan.c, char_display_width);
                        break; // Character would overflow, end run here
                    }

                    run_text.push(glyph_at_scan.c);
                    run_width_in_cells += char_display_width;
                    scan_col_for_text += char_display_width; // Advance by actual display width
                } else {
                    trace!("    Text run ended at col {}. Attributes changed. Next char: '{}', eff_fg:{:?}, eff_bg:{:?}, eff_flags:{:?}", scan_col_for_text, glyph_at_scan.c, current_glyph_eff_fg, current_glyph_eff_bg, current_glyph_flags);
                    break; // Attributes changed, run ends
                }
            }

            if !run_text.is_empty() {
                let coords = CellCoords { x: run_start_col, y: y_abs };
                let style = TextRunStyle { fg: start_eff_fg, bg: start_eff_bg, flags: start_flags };
                trace!("    Found text run: text='{}', width_cells={}, from_col={}, style={:?}. Calling DrawTextRun.", run_text, run_width_in_cells, run_start_col, style);
                driver.draw_text_run(coords, &run_text, style)?;
                current_col = run_start_col + run_width_in_cells;
            } else {
                // This case handles a single character that is not a space, not a placeholder,
                // and did not form a run (e.g., attributes changed immediately, or it's a zero-width char).
                // Only draw if it's a printable character with non-zero width.
                let char_display_width = get_char_display_width(start_glyph.c);
                if char_display_width > 0 {
                     // Ensure the single character fits
                    if current_col + char_display_width <= term_width {
                        let coords = CellCoords { x: current_col, y: y_abs };
                        let style = TextRunStyle { fg: start_eff_fg, bg: start_eff_bg, flags: start_flags };
                        trace!("    Drawing single char: '{}', width_cells={}, at_col={}, style={:?}. Calling DrawTextRun.", start_glyph.c, char_display_width, current_col, style);
                        driver.draw_text_run(coords, &start_glyph.c.to_string(), style)?;
                        current_col += char_display_width;
                    } else {
                        trace!("    Single char '{}' (width {}) at col {} does not fit. Advancing past.", start_glyph.c, char_display_width, current_col);
                        current_col = term_width; // End processing for this line
                    }
                } else {
                    // Zero-width character that wasn't part of a run, just advance past it.
                    trace!("    Skipping single zero-width char '{}' at col {}.", start_glyph.c, current_col);
                    current_col += 1; // Advance by one logical cell
                }
            }
        }
        Ok(())
    }
    
    /// Draws the terminal cursor if it's visible.
    fn draw_cursor(
        &self,
        term: &impl TerminalInterface,
        driver: &mut dyn Driver,
        term_width: usize,
        term_height: usize,
    ) -> Result<()> {
        let (cursor_abs_x, cursor_abs_y) = term.get_screen_cursor_pos();
        trace!("Renderer::draw_cursor: Screen cursor pos ({}, {})", cursor_abs_x, cursor_abs_y);

        // Ensure cursor position is within drawable bounds.
        if !(cursor_abs_x < term_width && cursor_abs_y < term_height) {
            warn!("Renderer::draw_cursor: Cursor at ({}, {}) is out of bounds ({}x{}). Not drawing.",
                cursor_abs_x, cursor_abs_y, term_width, term_height);
            return Ok(());
        }

        let physical_cursor_x_for_draw; // The starting column for drawing the cursor character(s)
        let char_to_draw_at_cursor;
        let original_attrs_at_cursor;

        // Get the glyph at the logical cursor position.
        let glyph_at_logical_cursor = term.get_glyph(cursor_abs_x, cursor_abs_y);
        trace!("  Glyph at cursor ({},{}): '{}' ({:?})", cursor_abs_x, cursor_abs_y, glyph_at_logical_cursor.c, glyph_at_logical_cursor.attr);


        // If the cursor is on the second half of a wide character (placeholder '\0'),
        // the cursor should visually cover the first half of that wide character.
        if glyph_at_logical_cursor.c == '\0' && cursor_abs_x > 0 {
            // Get the first half of the wide character.
            let first_half_glyph = term.get_glyph(cursor_abs_x - 1, cursor_abs_y);
            trace!("  Cursor on placeholder, using first half: '{}' ({:?}) at col {}", first_half_glyph.c, first_half_glyph.attr, cursor_abs_x - 1);
            char_to_draw_at_cursor = first_half_glyph.c;
            original_attrs_at_cursor = first_half_glyph.attr;
            physical_cursor_x_for_draw = cursor_abs_x - 1; // Draw starts at the first half's column
        } else {
            char_to_draw_at_cursor = glyph_at_logical_cursor.c;
            original_attrs_at_cursor = glyph_at_logical_cursor.attr;
            physical_cursor_x_for_draw = cursor_abs_x;
        }
        
        // Resolve the original attributes of the cell the cursor is on.
        // Note: get_effective_colors_and_flags handles REVERSE internally.
        let (resolved_original_fg, resolved_original_bg, resolved_original_flags) =
            self.get_effective_colors_and_flags(
                original_attrs_at_cursor.fg,
                original_attrs_at_cursor.bg,
                original_attrs_at_cursor.flags
            );
        trace!("    Original cell effective attrs: fg={:?}, bg={:?}, flags={:?}", resolved_original_fg, resolved_original_bg, resolved_original_flags);


        // For cursor drawing, foreground and background are typically swapped from the cell's effective colors.
        let cursor_char_fg = resolved_original_bg; // Cursor text color is original effective background
        let cursor_cell_bg = resolved_original_fg; // Cursor cell background is original effective foreground
        
        // The cursor display flags should be the original cell's effective flags (REVERSE already handled).
        let cursor_display_flags = resolved_original_flags;

        let coords = CellCoords { x: physical_cursor_x_for_draw, y: cursor_abs_y };
        let style = TextRunStyle { fg: cursor_char_fg, bg: cursor_cell_bg, flags: cursor_display_flags };
        
        // If the character under the cursor was a placeholder, draw a space for the cursor.
        // Otherwise, draw the actual character that was there.
        let final_char_to_draw_for_cursor = if char_to_draw_at_cursor == '\0' {
            ' ' // Draw a space if cursor is on a placeholder (second half of wide char)
        } else {
            char_to_draw_at_cursor
        };
        trace!("    Drawing cursor char '{}' at ({},{}) with style: {:?}", final_char_to_draw_for_cursor, physical_cursor_x_for_draw, cursor_abs_y, style);

        driver.draw_text_run(coords, &final_char_to_draw_for_cursor.to_string(), style)?;

        Ok(())
    }

    /// Determines effective foreground, background, and flags by resolving `Color::Default`
    /// and handling the `AttrFlags::REVERSE` flag.
    /// The returned flags will not include REVERSE if it was processed.
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

        // If REVERSE flag is set, swap foreground and background.
        // The REVERSE flag is then removed from the flags passed to the driver,
        // as its effect (color swapping) has been applied.
        if cell_flags.contains(AttrFlags::REVERSE) {
            std::mem::swap(&mut resolved_fg, &mut resolved_bg);
            (resolved_fg, resolved_bg, cell_flags.difference(AttrFlags::REVERSE))
        } else {
            (resolved_fg, resolved_bg, cell_flags)
        }
    }
}

impl Default for Renderer {
    fn default() -> Self {
        Self::new()
    }
}


#[cfg(test)]
mod tests; // Assuming tests are in a submodule or separate file like renderer/tests.rs

