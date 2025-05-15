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
            // Nothing to draw if terminal dimensions are zero.
            return Ok(());
        }
        
        let dirty_line_indices = term.take_dirty_lines();
        let mut something_was_drawn = !dirty_line_indices.is_empty();

        // If all lines were initially dirty (e.g. after resize or first draw),
        // clear the whole screen with the renderer's default background.
        // This check identifies if the entire screen content is considered new/changed.
        if dirty_line_indices.len() == term_height && 
           dirty_line_indices.iter().enumerate().all(|(i, &dl_idx)| i == dl_idx) {
            trace!("Renderer: All lines dirty, performing full clear_all with renderer's default background.");
            driver.clear_all(RENDERER_DEFAULT_BG)?; // Use RENDERER_DEFAULT_BG
            something_was_drawn = true; 
        }


        for y_abs in dirty_line_indices {
            if y_abs >= term_height {
                warn!(
                    "Renderer: Dirty line index {} is out of bounds (height {}). Skipping.",
                    y_abs, term_height
                );
                continue;
            }
            // Draw each dirty line.
            self.draw_dirty_line(y_abs, term_width, term, driver)?;
            // something_was_drawn is already true if dirty_line_indices was not empty
        }

        // Draw the cursor if it's visible.
        if term.is_cursor_visible() {
            self.draw_cursor(term, driver, term_width, term_height)?;
            something_was_drawn = true;
        }

        // If any drawing operations occurred, present the changes to the display.
        if something_was_drawn {
            driver.present()?;
        }

        Ok(())
    }

    /// Draws a single dirty line of the terminal.
    /// Iterates through the line, forming runs of characters with identical attributes
    /// and drawing them.
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

            if start_glyph.c == '\0' { // Handle WIDE_CHAR_PLACEHOLDER
                let (_, eff_bg, _) = self.get_effective_colors_and_flags(
                    start_glyph.attr.fg,
                    start_glyph.attr.bg,
                    start_glyph.attr.flags,
                );
                let rect = CellRect { x: current_col, y: y_abs, width: 1, height: 1 };
                driver.fill_rect(rect, eff_bg)?;
                current_col += 1;
                continue;
            }
            
            let (run_eff_fg, run_eff_bg, run_flags) = self.get_effective_colors_and_flags(
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
                     if scan_col == (run_start_col + run_width_in_cells) && !run_text.is_empty() {
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

                if current_glyph_eff_fg == run_eff_fg
                    && current_glyph_eff_bg == run_eff_bg
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
                let style = TextRunStyle { fg: run_eff_fg, bg: run_eff_bg, flags: run_flags };
                driver.draw_text_run(coords, &run_text, style)?;
                current_col = run_start_col + run_width_in_cells;
            } else { // Single character or space
                let char_display_width = get_char_display_width(start_glyph.c).max(1); 
                if start_glyph.c == ' ' { 
                    let rect = CellRect { x: current_col, y: y_abs, width: char_display_width, height: 1 };
                    driver.fill_rect(rect, run_eff_bg)?; 
                } else {
                    let coords = CellCoords { x: current_col, y: y_abs };
                    let style = TextRunStyle { fg: run_eff_fg, bg: run_eff_bg, flags: run_flags };
                    driver.draw_text_run(coords, &start_glyph.c.to_string(), style)?;
                }
                current_col += char_display_width;
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

    /// Determines effective foreground, background, and flags by resolving `Color::Default`
    /// and handling the `AttrFlags::REVERSE` flag.
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
mod tests {
    use super::*; // Imports Renderer, RENDERER_DEFAULT_FG, RENDERER_DEFAULT_BG from parent module
    use crate::term::TerminalInterface; // The trait
    // Glyph is used for MockTerminal and test setups
    use crate::glyph::{Glyph, Attributes, Color, AttrFlags, NamedColor, DEFAULT_GLYPH}; 
    use crate::backends::{Driver, BackendEvent, CellCoords, TextRunStyle, CellRect};
    use anyhow::Result; 
    use std::os::unix::io::RawFd; 
    use std::collections::HashSet; 
    // No specific log imports needed here if tests don't log directly.

    // --- MockTerminal Definition ---
    #[derive(Clone)]
    struct MockTerminal {
        width: usize,
        height: usize,
        grid: Vec<Vec<Glyph>>,
        cursor_visible: bool,
        cursor_x: usize,
        cursor_y: usize,
        dirty_lines: HashSet<usize>,
    }

    impl MockTerminal {
        fn new(width: usize, height: usize) -> Self {
            let default_glyph = DEFAULT_GLYPH; // This comes from crate::glyph
            MockTerminal {
                width,
                height,
                grid: vec![vec![default_glyph; width]; height],
                cursor_visible: true,
                cursor_x: 0,
                cursor_y: 0,
                dirty_lines: (0..height).collect(), 
            }
        }

        fn set_glyph(&mut self, x: usize, y: usize, glyph: Glyph) {
            if y < self.height && x < self.width {
                self.grid[y][x] = glyph;
            }
        }

        fn set_cursor_pos(&mut self, x: usize, y: usize) {
            self.cursor_x = x;
            self.cursor_y = y;
        }
        
        fn set_cursor_visibility(&mut self, visible: bool) {
            self.cursor_visible = visible;
        }

        #[allow(dead_code)] // Potentially unused in some test setups but useful for a mock
        fn mark_line_dirty(&mut self, y: usize) {
            if y < self.height {
                self.dirty_lines.insert(y);
            }
        }
        
        #[allow(dead_code)] // Potentially unused
         fn clear_dirty_lines(&mut self) {
            self.dirty_lines.clear();
        }
    }

    impl TerminalInterface for MockTerminal {
        fn dimensions(&self) -> (usize, usize) { (self.width, self.height) }
        fn get_glyph(&self, x: usize, y: usize) -> Glyph {
            if y < self.height && x < self.width { self.grid[y][x].clone() } else { DEFAULT_GLYPH }
        }
        fn is_cursor_visible(&self) -> bool { self.cursor_visible }
        fn get_screen_cursor_pos(&self) -> (usize, usize) { (self.cursor_x, self.cursor_y) }
        fn take_dirty_lines(&mut self) -> Vec<usize> {
            let mut lines: Vec<usize> = self.dirty_lines.drain().collect();
            lines.sort_unstable();
            lines
        }
    }

    // --- MockDriver Definition ---
    // User will fix PartialEq for these types in their respective files.
    // For now, we keep PartialEq here to satisfy the user's request to not change this part.
    #[derive(Debug, Clone, PartialEq)]
    enum MockDriverCall {
        ClearAll { bg: Color },
        DrawTextRun { coords: CellCoords, text: String, style: TextRunStyle },
        FillRect { rect: CellRect, color: Color },
        Present,
    }

    struct MockDriver {
        calls: Vec<MockDriverCall>,
    }

    impl MockDriver {
        fn new() -> Self {
            MockDriver {
                calls: Vec::new(),
            }
        }
        
        fn has_call(&self, expected_call: &MockDriverCall) -> bool {
            self.calls.contains(expected_call)
        }
        
        fn count_draw_text_run_calls(&self) -> usize {
            self.calls.iter().filter(|call| matches!(call, MockDriverCall::DrawTextRun { .. })).count()
        }
    }

    impl Driver for MockDriver {
        fn new() -> Result<Self> { Ok(Self::new()) }
        fn get_event_fd(&self) -> Option<RawFd> { None }
        fn process_events(&mut self) -> Result<Vec<BackendEvent>> { Ok(Vec::new()) }
        fn get_font_dimensions(&self) -> (usize, usize) { (8, 16) } 
        fn get_display_dimensions_pixels(&self) -> (u16, u16) { (640, 480) }
        fn clear_all(&mut self, bg: Color) -> Result<()> {
            self.calls.push(MockDriverCall::ClearAll { bg }); Ok(())
        }
        fn draw_text_run(&mut self, coords: CellCoords, text: &str, style: TextRunStyle) -> Result<()> {
            self.calls.push(MockDriverCall::DrawTextRun { coords, text: text.to_string(), style }); Ok(())
        }
        fn fill_rect(&mut self, rect: CellRect, color: Color) -> Result<()> {
            self.calls.push(MockDriverCall::FillRect { rect, color }); Ok(())
        }
        fn present(&mut self) -> Result<()> { self.calls.push(MockDriverCall::Present); Ok(()) }
        fn cleanup(&mut self) -> Result<()> { Ok(()) }
    }

    // --- Renderer Tests ---

    #[test]
    fn test_draw_empty_terminal() {
        let renderer = Renderer::new();
        let mut mock_term = MockTerminal::new(0, 0); 
        let mut mock_driver = MockDriver::new();
        renderer.draw(&mut mock_term, &mut mock_driver).unwrap();
        assert!(mock_driver.calls.is_empty(), "No draw calls should be made for 0x0 terminal");
    }

    #[test]
    fn test_initial_draw_clears_all_with_default_bg() {
        let renderer = Renderer::new();
        let mut mock_term = MockTerminal::new(2, 1); 
        let mut mock_driver = MockDriver::new();

        renderer.draw(&mut mock_term, &mut mock_driver).unwrap();
        
        // Check if ClearAll was called with the correct background
        let clear_all_called_correctly = mock_driver.calls.iter().any(|call| {
            matches!(call, MockDriverCall::ClearAll { bg } if *bg == RENDERER_DEFAULT_BG)
        });
        assert!(clear_all_called_correctly, "ClearAll with RENDERER_DEFAULT_BG was not called. Calls: {:?}", mock_driver.calls);

        assert!(mock_driver.count_draw_text_run_calls() >= 1, "Expected at least one DrawTextRun for cursor. Calls: {:?}", mock_driver.calls);
        
        let present_called = mock_driver.calls.iter().any(|call| matches!(call, MockDriverCall::Present));
        assert!(present_called, "Present was not called. Calls: {:?}", mock_driver.calls);
    }
    
    #[test]
    fn test_draw_single_default_char() {
        let renderer = Renderer::new();
        let mut mock_term = MockTerminal::new(1, 1);
        let mut mock_driver = MockDriver::new();

        mock_term.set_glyph(0, 0, Glyph { c: 'A', attr: Attributes::default() });
        mock_term.mark_line_dirty(0); 

        renderer.draw(&mut mock_term, &mut mock_driver).unwrap();
        
        assert!(!mock_driver.calls.iter().any(|call| matches!(call, MockDriverCall::ClearAll { .. })), "ClearAll should not be called for a single dirty line. Calls: {:?}", mock_driver.calls);

        let expected_text_run = MockDriverCall::DrawTextRun {
            coords: CellCoords { x: 0, y: 0 },
            text: "A".to_string(),
            style: TextRunStyle { fg: RENDERER_DEFAULT_FG, bg: RENDERER_DEFAULT_BG, flags: AttrFlags::empty() }
        };
        assert!(mock_driver.has_call(&expected_text_run), "Expected text run for 'A' not found. Calls: {:?}", mock_driver.calls);
        
        let expected_cursor_draw = MockDriverCall::DrawTextRun {
            coords: CellCoords { x: 0, y: 0 }, 
            text: "A".to_string(), 
            style: TextRunStyle { fg: RENDERER_DEFAULT_BG, bg: RENDERER_DEFAULT_FG, flags: AttrFlags::empty() } 
        };
        assert!(mock_driver.has_call(&expected_cursor_draw), "Expected cursor draw for 'A' not found. Calls: {:?}", mock_driver.calls);
        
        assert!(mock_driver.has_call(&MockDriverCall::Present), "Present was not called. Calls: {:?}", mock_driver.calls);
    }

    #[test]
    fn test_draw_char_with_specific_colors() {
        let renderer = Renderer::new();
        let mut mock_term = MockTerminal::new(1, 1);
        let mut mock_driver = MockDriver::new();
        let fg_red = Color::Named(NamedColor::Red);
        let bg_blue = Color::Named(NamedColor::Blue);

        mock_term.set_glyph(0, 0, Glyph { c: 'R', attr: Attributes { fg: fg_red, bg: bg_blue, flags: AttrFlags::empty() } });
        mock_term.mark_line_dirty(0);

        renderer.draw(&mut mock_term, &mut mock_driver).unwrap();

        let expected_call = MockDriverCall::DrawTextRun {
            coords: CellCoords { x: 0, y: 0 },
            text: "R".to_string(),
            style: TextRunStyle { fg: fg_red, bg: bg_blue, flags: AttrFlags::empty() }
        };
        assert!(mock_driver.has_call(&expected_call), "Specific color text run not found. Calls: {:?}", mock_driver.calls);
    }

    #[test]
    fn test_draw_char_with_reverse_video() {
        let renderer = Renderer::new();
        let mut mock_term = MockTerminal::new(1, 1);
        let mut mock_driver = MockDriver::new();

        mock_term.set_glyph(0, 0, Glyph {
            c: 'X',
            attr: Attributes { fg: Color::Default, bg: Color::Default, flags: AttrFlags::REVERSE }
        });
        mock_term.mark_line_dirty(0);
        renderer.draw(&mut mock_term, &mut mock_driver).unwrap();

        let expected_call = MockDriverCall::DrawTextRun {
            coords: CellCoords { x: 0, y: 0 },
            text: "X".to_string(),
            style: TextRunStyle {
                fg: RENDERER_DEFAULT_BG, 
                bg: RENDERER_DEFAULT_FG, 
                flags: AttrFlags::empty() 
            }
        };
        assert!(mock_driver.has_call(&expected_call), "Reversed video text run not found. Calls: {:?}", mock_driver.calls);
    }

    #[test]
    fn test_draw_cursor_on_colored_background() {
        let renderer = Renderer::new();
        let mut mock_term = MockTerminal::new(1, 1);
        let mut mock_driver = MockDriver::new();
        let cell_fg = Color::Named(NamedColor::Yellow);
        let cell_bg = Color::Named(NamedColor::Magenta);

        mock_term.set_glyph(0, 0, Glyph { c: 'C', attr: Attributes { fg: cell_fg, bg: cell_bg, flags: AttrFlags::empty() } });
        mock_term.set_cursor_pos(0,0);
        mock_term.set_cursor_visibility(true);
        mock_term.mark_line_dirty(0);

        renderer.draw(&mut mock_term, &mut mock_driver).unwrap();

        let mut cursor_call_found = false;
        let mut cell_call_found = false;

        for call in &mock_driver.calls {
            if let MockDriverCall::DrawTextRun { coords, text, style } = call {
                if coords.x == 0 && coords.y == 0 && text == "C" {
                    if style.fg == cell_bg && style.bg == cell_fg { 
                        cursor_call_found = true;
                    } else if style.fg == cell_fg && style.bg == cell_bg { 
                        cell_call_found = true;
                    }
                }
            }
        }
        assert!(cell_call_found, "Cell draw call not found. Calls: {:?}", mock_driver.calls);
        assert!(cursor_call_found, "Cursor draw call with inverted specific colors not found. Calls: {:?}", mock_driver.calls);
    }

    #[test]
    fn test_draw_only_dirty_lines() {
        let renderer = Renderer::new();
        let mut mock_term = MockTerminal::new(2, 2);
        let mut mock_driver = MockDriver::new();

        mock_term.set_glyph(0, 0, Glyph {c: 'A', attr: Attributes::default()});
        mock_term.set_glyph(0, 1, Glyph {c: 'B', attr: Attributes::default()});
        
        mock_term.clear_dirty_lines(); 
        mock_term.mark_line_dirty(1);  

        renderer.draw(&mut mock_term, &mut mock_driver).unwrap();

        let mut line_0_drawn_as_text_run = false;
        let mut line_1_drawn_as_text_run = false;
        for call in &mock_driver.calls {
            if let MockDriverCall::DrawTextRun { coords, text, .. } = call {
                if coords.y == 0 && text == "A" { line_0_drawn_as_text_run = true; }
                if coords.y == 1 && text == "B" { line_1_drawn_as_text_run = true; }
            }
        }
        assert!(!line_0_drawn_as_text_run, "Line 0 should not have been drawn explicitly as text run. Calls: {:?}", mock_driver.calls);
        assert!(line_1_drawn_as_text_run, "Line 1 should have been drawn. Calls: {:?}", mock_driver.calls);
    }
    
    #[test]
    fn test_draw_wide_char_and_placeholder() {
        let renderer = Renderer::new();
        let mut mock_term = MockTerminal::new(3, 1); 
        let mut mock_driver = MockDriver::new();

        let wide_char = '世'; 
        let wide_char_attrs = Attributes { fg: Color::Named(NamedColor::Green), bg: Color::Default, flags: AttrFlags::empty() };
        
        mock_term.set_glyph(0, 0, Glyph { c: wide_char, attr: wide_char_attrs });
        mock_term.set_glyph(1, 0, Glyph { c: '\0', attr: wide_char_attrs }); 
        mock_term.set_glyph(2, 0, Glyph { c: 'N', attr: Attributes::default() }); 
        mock_term.mark_line_dirty(0);

        renderer.draw(&mut mock_term, &mut mock_driver).unwrap();

        let expected_wide_char_call = MockDriverCall::DrawTextRun {
            coords: CellCoords { x: 0, y: 0 },
            text: wide_char.to_string(),
            style: TextRunStyle { fg: Color::Named(NamedColor::Green), bg: RENDERER_DEFAULT_BG, flags: AttrFlags::empty() }
        };
        assert!(mock_driver.has_call(&expected_wide_char_call), "Wide char text run not found. Calls: {:?}", mock_driver.calls);
        
        let expected_placeholder_fill = MockDriverCall::FillRect {
            rect: CellRect { x: 1, y: 0, width: 1, height: 1 },
            color: RENDERER_DEFAULT_BG 
        };
        assert!(mock_driver.has_call(&expected_placeholder_fill), "Placeholder fill not found. Calls: {:?}", mock_driver.calls);
        
        let expected_normal_char_call = MockDriverCall::DrawTextRun {
            coords: CellCoords { x: 2, y: 0 },
            text: "N".to_string(),
            style: TextRunStyle { fg: RENDERER_DEFAULT_FG, bg: RENDERER_DEFAULT_BG, flags: AttrFlags::empty() }
        };
        assert!(mock_driver.has_call(&expected_normal_char_call), "Normal char text run not found. Calls: {:?}", mock_driver.calls);
    }

    #[test]
    fn test_draw_line_of_spaces() {
        let renderer = Renderer::new();
        let mut mock_term = MockTerminal::new(3, 1);
        let mut mock_driver = MockDriver::new();

        mock_term.set_glyph(0, 0, Glyph {c: ' ', attr: Attributes::default()});
        mock_term.set_glyph(1, 0, Glyph {c: ' ', attr: Attributes::default()});
        mock_term.set_glyph(2, 0, Glyph {c: ' ', attr: Attributes::default()});
        mock_term.mark_line_dirty(0);

        renderer.draw(&mut mock_term, &mut mock_driver).unwrap();

        let expected_fill = MockDriverCall::FillRect {
            rect: CellRect {x: 0, y: 0, width: 3, height: 1},
            color: RENDERER_DEFAULT_BG
        };
        assert!(mock_driver.has_call(&expected_fill), "FillRect for spaces not found. Calls: {:?}", mock_driver.calls);
        
        let draw_text_run_for_space_count = mock_driver.calls.iter().filter(|call| {
            matches!(call, MockDriverCall::DrawTextRun { text, .. } if text == " ")
        }).count();
        assert_eq!(draw_text_run_for_space_count, 0, "Spaces should be drawn with FillRect, not DrawTextRun. Calls: {:?}", mock_driver.calls);
    }

     #[test]
    fn test_cursor_on_wide_char_second_half() {
        let renderer = Renderer::new();
        let mut mock_term = MockTerminal::new(2, 1);
        let mut mock_driver = MockDriver::new();

        let wide_char = '世';
        let attrs = Attributes { fg: Color::Named(NamedColor::Cyan), bg: RENDERER_DEFAULT_BG, flags: AttrFlags::empty() };
        mock_term.set_glyph(0, 0, Glyph { c: wide_char, attr: attrs });
        mock_term.set_glyph(1, 0, Glyph { c: '\0', attr: attrs }); 

        mock_term.set_cursor_pos(1, 0); 
        mock_term.set_cursor_visibility(true);
        mock_term.mark_line_dirty(0);

        renderer.draw(&mut mock_term, &mut mock_driver).unwrap();

        let mut cursor_draw_found = false;
        for call in &mock_driver.calls {
            if let MockDriverCall::DrawTextRun { coords, text, style } = call {
                if coords.x == 0 && coords.y == 0 && *text == wide_char.to_string() && style.fg == RENDERER_DEFAULT_BG && style.bg == Color::Named(NamedColor::Cyan) {
                    cursor_draw_found = true;
                    break;
                }
            }
        }
        assert!(cursor_draw_found, "Cursor on wide char placeholder not drawn correctly. Calls: {:?}", mock_driver.calls);
    }
}
