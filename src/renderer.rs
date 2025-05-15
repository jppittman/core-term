// src/renderer.rs

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
    first_draw: bool,
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
    pub fn draw(&mut self, term: &mut impl TerminalInterface, driver: &mut dyn Driver) -> Result<()> {
        let (term_width, term_height) = term.dimensions();

        if term_width == 0 || term_height == 0 {
            trace!("Renderer::draw: Terminal dimensions zero, skipping draw.");
            return Ok(());
        }

        let initially_dirty_lines_from_term: HashSet<usize> =
            term.take_dirty_lines().into_iter().collect();
        let mut something_was_drawn = !initially_dirty_lines_from_term.is_empty();
        trace!(
            "Renderer::draw: Initially dirty lines from term: {:?}, term_dims: {}x{}",
            initially_dirty_lines_from_term, term_width, term_height
        );
        
        let mut is_full_refresh = self.first_draw;
        if self.first_draw {
            self.first_draw = false; 
        } else if term_height > 0 {
            // Heuristic: if all lines are reported dirty sequentially, assume a full refresh.
            let mut all_lines_dirty_sequentially = initially_dirty_lines_from_term.len() == term_height;
            if all_lines_dirty_sequentially {
                for i in 0..term_height {
                    if !initially_dirty_lines_from_term.contains(&i) {
                        all_lines_dirty_sequentially = false;
                        break;
                    }
                }
            }
            if all_lines_dirty_sequentially {
                 trace!(
                    "Renderer::draw: Heuristic: All lines 0..{} are dirty. Triggering full refresh.",
                    term_height.saturating_sub(1)
                );
                is_full_refresh = true;
            }
        }

        if is_full_refresh {
            trace!("Renderer::draw: Full refresh. Clearing all with RENDERER_DEFAULT_BG.");
            driver.clear_all(RENDERER_DEFAULT_BG)?;
            something_was_drawn = true;
        }

        let mut lines_to_draw_content: HashSet<usize> = initially_dirty_lines_from_term;
        let (cursor_abs_x, cursor_abs_y) = term.get_screen_cursor_pos();

        if is_full_refresh {
            lines_to_draw_content = (0..term_height).collect();
        } else if term.is_cursor_visible() && cursor_abs_y < term_height {
            lines_to_draw_content.insert(cursor_abs_y);
        }

        if !lines_to_draw_content.is_empty() {
            something_was_drawn = true;
        }

        let mut sorted_lines_to_draw: Vec<usize> = lines_to_draw_content.into_iter().collect();
        sorted_lines_to_draw.sort_unstable();
        trace!("Renderer::draw: Final lines to process for content: {:?}", sorted_lines_to_draw);

        for &y_abs in &sorted_lines_to_draw {
            if y_abs >= term_height {
                warn!("Renderer::draw: Attempted to draw out-of-bounds line y={}", y_abs);
                continue;
            }
            self.draw_line_content(y_abs, term_width, term, driver)?;
        }

        if term.is_cursor_visible() {
            trace!("Renderer::draw: Cursor is visible, calling draw_cursor overlay.");
            self.draw_cursor_overlay(cursor_abs_x, cursor_abs_y, term, driver, term_width, term_height)?;
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
        &self, // No longer &mut self if Renderer is stateless regarding the driver
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

            let char_for_log = if start_glyph.c == '\0' { ' ' } else { start_glyph.c };
            trace!(
                "  Line {}, Col {}: Start glyph='{}' (attr:{:?}), EffectiveStyle(fg:{:?}, bg:{:?}, flags:{:?})",
                y_abs, current_col, char_for_log, start_glyph.attr, eff_fg, eff_bg, eff_flags
            );

            let cells_consumed = if start_glyph.c == '\0' {
                self.draw_placeholder_cell(current_col, y_abs, eff_bg, driver)?
            } else if start_glyph.c == ' ' {
                let space_run_len = self.draw_space_run(current_col, y_abs, term_width, &start_glyph, term, driver)?;
                if space_run_len > 0 {
                    space_run_len
                } else {
                    self.draw_text_segment(current_col, y_abs, term_width, &start_glyph, term, driver)?
                }
            } else {
                self.draw_text_segment(current_col, y_abs, term_width, &start_glyph, term, driver)?
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

    /// Draws a placeholder cell.
    fn draw_placeholder_cell(&self, x: usize, y: usize, effective_bg: Color, driver: &mut dyn Driver) -> Result<usize> {
        let rect = CellRect { x, y, width: 1, height: 1 };
        trace!("    Line {}, Col {}: Placeholder. FillRect with bg={:?}", y, x, effective_bg);
        driver.fill_rect(rect, effective_bg)?;
        Ok(1)
    }

    /// Identifies and draws a run of space characters.
    fn draw_space_run(
        &self,
        start_col: usize,
        y: usize,
        term_width: usize,
        start_glyph: &Glyph,
        term: &impl TerminalInterface,
        driver: &mut dyn Driver,
    ) -> Result<usize> {
        let (_, start_eff_bg, start_eff_flags) = self.get_effective_colors_and_flags(
            start_glyph.attr.fg,
            start_glyph.attr.bg,
            start_glyph.attr.flags,
        );

        let mut space_run_len = 0;
        for x_offset in 0..(term_width - start_col) {
            let current_scan_col = start_col + x_offset;
            let glyph_at_scan = term.get_glyph(current_scan_col, y);
            let (_, current_scan_eff_bg, current_scan_flags) = self.get_effective_colors_and_flags(
                glyph_at_scan.attr.fg,
                glyph_at_scan.attr.bg,
                glyph_at_scan.attr.flags,
            );

            if glyph_at_scan.c != ' ' || current_scan_eff_bg != start_eff_bg || current_scan_flags != start_eff_flags {
                break;
            }
            space_run_len += 1;
        }

        if space_run_len == 0 { return Ok(0); }

        let rect = CellRect { x: start_col, y, width: space_run_len, height: 1 };
        trace!("    Line {}, Col {}: Space run (len {}). FillRect with bg={:?}", y, start_col, space_run_len, start_eff_bg);
        driver.fill_rect(rect, start_eff_bg)?;
        Ok(space_run_len)
    }

    /// Identifies and draws a run of non-space, non-placeholder text characters.
    fn draw_text_segment(
        &self,
        start_col: usize,
        y: usize,
        term_width: usize,
        start_glyph: &Glyph,
        term: &impl TerminalInterface,
        driver: &mut dyn Driver,
    ) -> Result<usize> {
        let (start_eff_fg, start_eff_bg, start_eff_flags) = self.get_effective_colors_and_flags(
            start_glyph.attr.fg,
            start_glyph.attr.bg,
            start_glyph.attr.flags,
        );

        let mut run_text = String::new();
        let mut run_total_cell_width = 0;
        let mut current_scan_col = start_col;

        while current_scan_col < term_width {
            let glyph_at_scan = term.get_glyph(current_scan_col, y);

            if glyph_at_scan.c == ' ' || glyph_at_scan.c == '\0' { break; }

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
                trace!("    Line {}, Col {}: Appending zero-width char '{}' to text run, advancing scan_col by 1.", y, current_scan_col, glyph_at_scan.c);
                run_text.push(glyph_at_scan.c);
                current_scan_col += 1; 
                // run_total_cell_width does not increase for true ZWCs, but the logical cell for scan does.
                // However, to ensure the main loop advances, we must count it as consuming a logical cell if it's the *only* thing.
                // This is tricky. If it's part of a sequence, the primary char width covers it.
                // If it's standalone, we need to advance.
                // For simplicity in run_total_cell_width, we only add display width.
                // The outer loop in draw_line_content will handle advancement if cells_consumed is 0.
                continue;
            }

            if start_col + run_total_cell_width + char_display_width > term_width { break; }

            run_text.push(glyph_at_scan.c);
            run_total_cell_width += char_display_width;
            current_scan_col += char_display_width;
        }

        if run_text.is_empty() {
            // If the start_glyph itself was a ZWC and was skipped, or other non-drawable.
            let advance = get_char_display_width(start_glyph.c).max(1);
             warn!(
                "    Line {}, Col {}: Single char '{}' (width {}) did not form text run. Advancing by {}.",
                y, start_col, start_glyph.c, get_char_display_width(start_glyph.c), advance
            );
            return Ok(advance);
        }

        let coords = CellCoords { x: start_col, y };
        let style = TextRunStyle { fg: start_eff_fg, bg: start_eff_bg, flags: start_eff_flags };
        trace!(
            "    Line {}, Col {}: Text run: '{}' ({} cells). DrawTextRun with style={:?}",
            y, start_col, run_text, run_total_cell_width, style
        );
        driver.draw_text_run(coords, &run_text, style)?;

        let mut col_offset_in_run_for_placeholders = 0;
        for ch_in_run in run_text.chars() {
            let char_cell_width = get_char_display_width(ch_in_run);
            if char_cell_width == 2 {
                let placeholder_x = start_col + col_offset_in_run_for_placeholders + 1;
                if placeholder_x < term_width {
                    let placeholder_rect = CellRect { x: placeholder_x, y, width: 1, height: 1 };
                    trace!(
                        "      Line {}, Col {}: Explicitly filling placeholder for wide char '{}' in run, with bg {:?}",
                        y, placeholder_x, ch_in_run, start_eff_bg
                    );
                    driver.fill_rect(placeholder_rect, start_eff_bg)?;
                }
            }
            col_offset_in_run_for_placeholders += char_cell_width;
        }
        Ok(run_total_cell_width)
    }

    /// Draws the terminal cursor as an overlay.
    fn draw_cursor_overlay(
        &self, // No longer &mut self
        cursor_abs_x: usize,
        cursor_abs_y: usize,
        term: &impl TerminalInterface,
        driver: &mut dyn Driver,
        term_width: usize,
        term_height: usize,
    ) -> Result<()> {
        trace!("Renderer::draw_cursor_overlay: Screen cursor pos ({}, {})", cursor_abs_x, cursor_abs_y);

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
        let char_for_log1 = if glyph_at_logical_cursor.c == '\0' { ' ' } else { glyph_at_logical_cursor.c };
        trace!(
            "  Cursor overlay: Glyph at logical cursor pos ({},{}): char='{}', attr={:?}",
            cursor_abs_x, cursor_abs_y, char_for_log1, glyph_at_logical_cursor.attr
        );

        if glyph_at_logical_cursor.c == '\0' && cursor_abs_x > 0 {
            let first_half_glyph = term.get_glyph(cursor_abs_x - 1, cursor_abs_y);
            let char_for_log2 = if first_half_glyph.c == '\0' { '?' } else { first_half_glyph.c };
            trace!(
                "    Cursor on placeholder, using first half: char='{}' from col {}",
                char_for_log2, cursor_abs_x - 1
            );

            char_to_draw_at_cursor = first_half_glyph.c;
            original_attrs_at_cursor = first_half_glyph.attr;
            physical_cursor_x_for_draw = cursor_abs_x - 1;
        } else {
            char_to_draw_at_cursor = glyph_at_logical_cursor.c;
            original_attrs_at_cursor = glyph_at_logical_cursor.attr;
            physical_cursor_x_for_draw = cursor_abs_x;
        }

        let (resolved_original_fg, resolved_original_bg, resolved_original_flags) = self
            .get_effective_colors_and_flags(
                original_attrs_at_cursor.fg,
                original_attrs_at_cursor.bg,
                original_attrs_at_cursor.flags,
            );
        trace!(
            "    Original cell effective attrs for cursor: fg={:?}, bg={:?}, flags={:?}",
            resolved_original_fg, resolved_original_bg, resolved_original_flags
        );

        let cursor_char_fg = resolved_original_bg;
        let cursor_cell_bg = resolved_original_fg;
        let cursor_display_flags = resolved_original_flags;

        let coords = CellCoords { x: physical_cursor_x_for_draw, y: cursor_abs_y };
        let style = TextRunStyle { fg: cursor_char_fg, bg: cursor_cell_bg, flags: cursor_display_flags };

        let final_char_to_draw_for_cursor = if char_to_draw_at_cursor == '\0' { ' ' } else { char_to_draw_at_cursor };
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

// Test module remains in src/renderer/tests.rs
#[cfg(test)]
mod tests;


