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
use crate::glyph::{Color, AttrFlags, NamedColor}; // Glyph is used by Renderer's public interface indirectly via TerminalInterface
use crate::term::unicode::get_char_display_width;

use anyhow::Result;
use log::{trace, warn}; // error is used in the main Renderer code

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
            return Ok(());
        }
        
        let dirty_line_indices = term.take_dirty_lines();
        let mut something_was_drawn = !dirty_line_indices.is_empty();

        // Refined: ClearAll only if it's likely an initial full draw or full refresh.
        // A more robust way would be an explicit flag from the TerminalEmulator.
        // For now, if all lines are reported as dirty AND the terminal has more than one line
        // (to avoid triggering on single-line terminals where one dirty line means all dirty),
        // or if it's a 1x1 terminal and its single line is dirty.
        let is_full_refresh = dirty_line_indices.len() == term_height &&
                              dirty_line_indices.iter().enumerate().all(|(i, &dl_idx)| i == dl_idx);

        if is_full_refresh {
            trace!("Renderer: All lines dirty, performing full clear_all with renderer's default background.");
            driver.clear_all(RENDERER_DEFAULT_BG)?; 
            something_was_drawn = true; 
        }

        for y_abs in &dirty_line_indices { // Iterate by reference if not consuming
            if *y_abs >= term_height {
                warn!(
                    "Renderer: Dirty line index {} is out of bounds (height {}). Skipping.",
                    y_abs, term_height
                );
                continue;
            }
            self.draw_dirty_line(*y_abs, term_width, term, driver)?;
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

    fn draw_dirty_line(
        &self,
        y_abs: usize,
        term_width: usize,
        term: &impl TerminalInterface, 
        driver: &mut dyn Driver,
    ) -> Result<()> {
        trace!("Renderer: Drawing dirty line {}", y_abs);
        let mut current_col = 0;
        while current_col < term_width {
            let start_glyph = term.get_glyph(current_col, y_abs);
            
            let (run_eff_fg, run_eff_bg, run_flags) = self.get_effective_colors_and_flags(
                start_glyph.attr.fg,
                start_glyph.attr.bg,
                start_glyph.attr.flags,
            );

            // Check for a run of spaces with the same attributes
            if start_glyph.c == ' ' {
                let mut space_run_width = 0;
                let mut scan_col_spaces = current_col;
                while scan_col_spaces < term_width {
                    let glyph_at_scan = term.get_glyph(scan_col_spaces, y_abs);
                    let (current_glyph_eff_fg, current_glyph_eff_bg, current_glyph_run_flags) =
                        self.get_effective_colors_and_flags(
                            glyph_at_scan.attr.fg,
                            glyph_at_scan.attr.bg,
                            glyph_at_scan.attr.flags,
                        );

                    if glyph_at_scan.c == ' ' && 
                       current_glyph_eff_fg == run_eff_fg &&
                       current_glyph_eff_bg == run_eff_bg &&
                       current_glyph_run_flags == run_flags {
                        space_run_width += 1;
                        scan_col_spaces += 1;
                    } else {
                        break;
                    }
                }
                if space_run_width > 0 {
                    let rect = CellRect { x: current_col, y: y_abs, width: space_run_width, height: 1 };
                    driver.fill_rect(rect, run_eff_bg)?;
                    current_col += space_run_width;
                    continue; // Continue to next segment of the line
                }
            }
            
            // Handle WIDE_CHAR_PLACEHOLDER if it's the start of a segment
            // This case is for isolated placeholders or if a run starts with one.
            if start_glyph.c == '\0' { 
                let rect = CellRect { x: current_col, y: y_abs, width: 1, height: 1 };
                driver.fill_rect(rect, run_eff_bg)?; // Use run_eff_bg derived from placeholder's attrs
                current_col += 1;
                continue;
            }
            
            // Regular character run processing
            let mut run_text = String::new();
            let run_start_col = current_col;
            let mut run_width_in_cells = 0; 

            let mut scan_col = current_col;
            while scan_col < term_width {
                let glyph_at_scan = term.get_glyph(scan_col, y_abs);
                
                // If we encounter a placeholder, the current run must end.
                if glyph_at_scan.c == '\0' {
                     break; 
                }

                let (current_glyph_eff_fg, current_glyph_eff_bg, current_glyph_run_flags) =
                    self.get_effective_colors_and_flags(
                        glyph_at_scan.attr.fg,
                        glyph_at_scan.attr.bg,
                        glyph_at_scan.attr.flags,
                    );

                if current_glyph_eff_fg == run_eff_fg
                    && current_glyph_eff_bg == run_eff_bg
                    && current_glyph_run_flags == run_flags
                {
                    let char_display_width = get_char_display_width(glyph_at_scan.c);
                    if char_display_width == 0 { // Skip zero-width chars for drawing runs
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
                let style = TextRunStyle { fg: run_eff_fg, bg: run_eff_bg, flags: run_flags };
                driver.draw_text_run(coords, &run_text, style)?;
                current_col = run_start_col + run_width_in_cells;
            } else { 
                // This case should be rare if space runs and placeholders are handled above.
                // It implies a single, non-space, non-placeholder char that doesn't form a run.
                let char_display_width = get_char_display_width(start_glyph.c).max(1); 
                let coords = CellCoords { x: current_col, y: y_abs };
                let style = TextRunStyle { fg: run_eff_fg, bg: run_eff_bg, flags: run_flags };
                driver.draw_text_run(coords, &start_glyph.c.to_string(), style)?;
                current_col += char_display_width;
            }
        }
        Ok(())
    }
    
    fn draw_cursor(
        &self,
        term: &impl TerminalInterface, 
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
            let first_half_glyph = term.get_glyph(cursor_abs_x - 1, cursor_abs_y);
            char_to_draw_at_cursor = first_half_glyph.c;
            original_attrs = first_half_glyph.attr;
            physical_cursor_x = cursor_abs_x - 1; 
        } else {
            char_to_draw_at_cursor = glyph_at_logical_cursor.c;
            original_attrs = glyph_at_logical_cursor.attr;
            physical_cursor_x = cursor_abs_x;
        }
        
        let (resolved_original_fg, resolved_original_bg, resolved_original_flags) = 
            self.get_effective_colors_and_flags(
                original_attrs.fg,
                original_attrs.bg,
                original_attrs.flags
            );

        let cursor_char_fg = resolved_original_bg; 
        let cursor_cell_bg = resolved_original_fg;
        
        let cursor_display_flags = resolved_original_flags;

        let coords = CellCoords { x: physical_cursor_x, y: cursor_abs_y };
        let style = TextRunStyle { fg: cursor_char_fg, bg: cursor_cell_bg, flags: cursor_display_flags };
        
        let final_char_to_draw = if char_to_draw_at_cursor == '\0' { ' ' } else { char_to_draw_at_cursor };

        driver.draw_text_run(coords, &final_char_to_draw.to_string(), style)?;

        Ok(())
    }

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

