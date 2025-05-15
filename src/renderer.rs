// src/renderer.rs

//! This module defines the `Renderer`.
//!
//! The `Renderer`'s primary responsibility is to translate the visual state of the
//! `TerminalEmulator` into a series of abstract drawing commands that can be
//! executed by a `Driver`. It is designed to be backend-agnostic.
//! It now temporarily defines default foreground and background colors for resolving
//! `Color::Default` from glyph attributes.

use crate::backends::{Driver, CellCoords, TextRunStyle, CellRect};
use crate::term::TerminalEmulator;
use crate::glyph::{Color, AttrFlags, Glyph, NamedColor}; // Added NamedColor
use crate::term::unicode::get_char_display_width;

use anyhow::Result;
use log::{trace, warn, error}; // Added error log

// Temporary: Define default colors within the Renderer module.
// These will eventually come from a configuration module.
const RENDERER_DEFAULT_FG: Color = Color::Named(NamedColor::White);
const RENDERER_DEFAULT_BG: Color = Color::Named(NamedColor::Black);


/// The `Renderer` translates `TerminalEmulator` state into abstract drawing commands.
pub struct Renderer {
    // Future enhancements might include caching mechanisms or configuration options.
}

impl Renderer {
    /// Creates a new `Renderer` instance.
    pub fn new() -> Self {
        Self {}
    }

    /// Draws the current state of the `TerminalEmulator` using the provided `Driver`.
    pub fn draw(&self, term: &mut TerminalEmulator, driver: &mut dyn Driver) -> Result<()> {
        let (term_width, term_height) = term.dimensions();

        if term_width == 0 || term_height == 0 {
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

        if term.is_cursor_visible() {
            self.draw_cursor(term, driver, term_width, term_height)?;
            something_was_drawn = true;
        }

        if something_was_drawn {
            driver.present()?;
        }

        Ok(())
    }

    /// Draws a single dirty line of the terminal.
    fn draw_dirty_line(
        &self,
        y_abs: usize,
        term_width: usize,
        term: &TerminalEmulator,
        driver: &mut dyn Driver,
    ) -> Result<()> {
        trace!("Renderer: Drawing dirty line {}", y_abs);
        let mut current_col = 0;
        while current_col < term_width {
            let cols_drawn =
                self.draw_run_or_fallback_char(y_abs, current_col, term_width, term, driver)?;
            if cols_drawn == 0 {
                warn!("Renderer: draw_run_or_fallback_char consumed 0 columns at ({}, {}). Breaking line draw.", current_col, y_abs);
                break;
            }
            current_col += cols_drawn;
        }
        Ok(())
    }

    /// Draws a run of text with identical attributes, or a single fallback character.
    fn draw_run_or_fallback_char(
        &self,
        y_abs: usize,
        current_col: usize,
        term_width: usize,
        term: &TerminalEmulator,
        driver: &mut dyn Driver,
    ) -> Result<usize> {
        let start_glyph = term.get_glyph(current_col, y_abs);

        if start_glyph.c == '\0' { // WIDE_CHAR_PLACEHOLDER
            let (_, eff_bg, _) = self.get_effective_colors_and_flags(
                start_glyph.attr.fg,
                start_glyph.attr.bg,
                start_glyph.attr.flags,
            );
            let rect = CellRect { x: current_col, y: y_abs, width: 1, height: 1 };
            driver.fill_rect(rect, eff_bg)?;
            return Ok(1);
        }

        let (eff_fg, eff_bg, run_flags) = self.get_effective_colors_and_flags(
            start_glyph.attr.fg,
            start_glyph.attr.bg,
            start_glyph.attr.flags,
        );

        let mut run_text = String::new();
        let run_start_col = current_col;
        let mut run_width_in_cells = 0;

        let mut scan_col = current_col;
        while scan_col < term_width {
            let glyph_at_scan = term.get_glyph(scan_col, y_abs);

            if glyph_at_scan.c == '\0' {
                if run_text.is_empty() && scan_col == current_col {
                     warn!("Renderer: Unexpected isolated '\\0' at ({}, {}) during run scan start.", scan_col, y_abs);
                     let (_, placeholder_bg, _) = self.get_effective_colors_and_flags(
                         glyph_at_scan.attr.fg, glyph_at_scan.attr.bg, glyph_at_scan.attr.flags
                     );
                     driver.fill_rect(CellRect { x: scan_col, y: y_abs, width: 1, height: 1 }, placeholder_bg)?;
                     return Ok(1);
                } else if !run_text.is_empty() && scan_col == (run_start_col + run_width_in_cells) {
                    scan_col += 1;
                    continue;
                } else {
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
                if char_display_width == 0 {
                    scan_col +=1;
                    continue;
                }
                if run_start_col + run_width_in_cells + char_display_width > term_width {
                    break;
                }
                run_text.push(glyph_at_scan.c);
                run_width_in_cells += char_display_width;
                scan_col += char_display_width;
            } else {
                break;
            }
        }

        if !run_text.is_empty() {
            let coords = CellCoords { x: run_start_col, y: y_abs };
            let style = TextRunStyle { fg: eff_fg, bg: eff_bg, flags: run_flags };
            driver.draw_text_run(coords, &run_text, style)?;
            Ok(run_width_in_cells)
        } else {
            let char_display_width = get_char_display_width(start_glyph.c).max(1);
            if start_glyph.c == ' ' {
                let rect = CellRect { x: current_col, y: y_abs, width: char_display_width, height: 1 };
                driver.fill_rect(rect, eff_bg)?;
            } else {
                let coords = CellCoords { x: current_col, y: y_abs };
                let style = TextRunStyle { fg: eff_fg, bg: eff_bg, flags: run_flags };
                driver.draw_text_run(coords, &start_glyph.c.to_string(), style)?;
            }
            Ok(char_display_width)
        }
    }

    /// Draws the terminal cursor if it's visible.
    fn draw_cursor(
        &self,
        term: &TerminalEmulator,
        driver: &mut dyn Driver,
        term_width: usize,
        term_height: usize,
    ) -> Result<()> {
        let (cursor_abs_x, cursor_abs_y) = term.get_screen_cursor_pos();

        if !(cursor_abs_x < term_width && cursor_abs_y < term_height) {
            warn!("Renderer: Cursor at ({}, {}) is out of bounds ({}x{}). Not drawing.",
                cursor_abs_x, cursor_abs_y, term_width, term_height);
            return Ok(());
        }

        let physical_cursor_x;
        let char_to_draw_at_cursor;
        let original_attrs;

        let glyph_at_logical_cursor = term.get_glyph(cursor_abs_x, cursor_abs_y);
        if glyph_at_logical_cursor.c == '\0' && cursor_abs_x > 0 {
            // Cursor is on the placeholder of a wide char, get the actual char cell.
            let first_half_glyph = term.get_glyph(cursor_abs_x - 1, cursor_abs_y);
            char_to_draw_at_cursor = first_half_glyph.c;
            original_attrs = first_half_glyph.attr;
            physical_cursor_x = cursor_abs_x - 1;
        } else {
            char_to_draw_at_cursor = glyph_at_logical_cursor.c;
            original_attrs = glyph_at_logical_cursor.attr;
            physical_cursor_x = cursor_abs_x;
        }
        
        // Resolve Color::Default for the original cell's attributes
        let resolved_original_fg = match original_attrs.fg {
            Color::Default => RENDERER_DEFAULT_FG,
            c => c,
        };
        let resolved_original_bg = match original_attrs.bg {
            Color::Default => RENDERER_DEFAULT_BG,
            c => c,
        };

        // For block cursor, swap resolved FG and BG.
        let cursor_char_fg = resolved_original_bg; 
        let cursor_cell_bg = resolved_original_fg;
        
        // REVERSE flag on original cell is handled by the swap.
        // Other flags are preserved for the cursor block.
        let cursor_display_flags = original_attrs.flags.difference(AttrFlags::REVERSE);

        let coords = CellCoords { x: physical_cursor_x, y: cursor_abs_y };
        let style = TextRunStyle { fg: cursor_char_fg, bg: cursor_cell_bg, flags: cursor_display_flags };
        
        // Ensure char_to_draw_at_cursor is not a placeholder itself if logic got here.
        // If physical_cursor_x points to a cell that is itself a placeholder (e.g. cursor_abs_x was 0, glyph was '\0'),
        // then char_to_draw_at_cursor would be '\0'. Draw a space instead.
        let final_char_to_draw = if char_to_draw_at_cursor == '\0' { ' ' } else { char_to_draw_at_cursor };

        driver.draw_text_run(coords, &final_char_to_draw.to_string(), style)?;

        Ok(())
    }

    /// Determines effective foreground, background, and flags, resolving `Color::Default`.
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
mod tests;

