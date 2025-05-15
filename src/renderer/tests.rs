#[cfg(test)]
mod render_tests {
    use crate::backends::{BackendEvent, CellCoords, CellRect, Driver, TextRunStyle};
    use crate::glyph::{AttrFlags, Attributes, Color, Glyph, NamedColor};
    use crate::renderer::*; // Imports Renderer, RENDERER_DEFAULT_FG, RENDERER_DEFAULT_BG from parent module
    use crate::term::TerminalInterface;
    use anyhow::Result;
    use std::collections::HashSet;
    use std::os::unix::io::RawFd;
    use test_log::test; // For logging within tests

    // --- MockTerminal Definition (from your existing tests.txt) ---
    #[derive(Clone)]
    struct MockTerminal {
        width: usize,
        height: usize,
        grid: Vec<Vec<Glyph>>,
        cursor_visible: bool,
        cursor_x: usize, // physical screen x for cursor
        cursor_y: usize, // physical screen y for cursor
        dirty_lines: HashSet<usize>,
        // Added to control default attributes for get_glyph if needed for a test
        current_default_attributes: Attributes,
    }

    impl MockTerminal {
        fn new(width: usize, height: usize) -> Self {
            let default_attributes = Attributes {
                fg: RENDERER_DEFAULT_FG, // Use renderer's defaults for mock consistency
                bg: RENDERER_DEFAULT_BG,
                flags: AttrFlags::empty(),
            };
            let default_glyph_for_mock = Glyph {
                c: ' ',
                attr: default_attributes,
            };
            MockTerminal {
                width,
                height,
                grid: vec![vec![default_glyph_for_mock; width]; height],
                cursor_visible: true,
                cursor_x: 0,
                cursor_y: 0,
                dirty_lines: (0..height).collect(),
                current_default_attributes: default_attributes,
            }
        }

        fn set_glyph(&mut self, x: usize, y: usize, glyph: Glyph) {
            if y < self.height && x < self.width {
                self.grid[y][x] = glyph;
            }
        }

        // Sets the *screen* cursor position (physical) for the mock
        fn set_cursor_screen_pos(&mut self, x: usize, y: usize) {
            self.cursor_x = x;
            self.cursor_y = y;
        }

        #[allow(dead_code)] // Keep for potential future use in tests
        fn set_cursor_visibility(&mut self, visible: bool) {
            self.cursor_visible = visible;
        }

        fn mark_line_dirty(&mut self, y: usize) {
            if y < self.height {
                self.dirty_lines.insert(y);
            }
        }

        fn clear_all_dirty_lines(&mut self) {
            self.dirty_lines.clear();
        }
    }

    impl TerminalInterface for MockTerminal {
        fn dimensions(&self) -> (usize, usize) {
            (self.width, self.height)
        }
        fn get_glyph(&self, x: usize, y: usize) -> Glyph {
            if y < self.height && x < self.width {
                self.grid[y][x].clone()
            } else {
                // Return a glyph with the mock's current default attributes
                Glyph {
                    c: ' ',
                    attr: self.current_default_attributes,
                }
            }
        }
        fn is_cursor_visible(&self) -> bool {
            self.cursor_visible
        }
        fn get_screen_cursor_pos(&self) -> (usize, usize) {
            (self.cursor_x, self.cursor_y)
        }
        fn take_dirty_lines(&mut self) -> Vec<usize> {
            let mut lines: Vec<usize> = self.dirty_lines.drain().collect();
            lines.sort_unstable();
            lines
        }
    }

    // --- MockDriver Definition (from your existing tests.txt) ---
    #[derive(Debug, Clone, PartialEq)]
    enum MockDriverCall {
        ClearAll {
            bg: Color,
        },
        DrawTextRun {
            coords: CellCoords,
            text: String,
            style: TextRunStyle,
        },
        FillRect {
            rect: CellRect,
            color: Color,
        },
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

        #[allow(dead_code)] // Keep for potential future use
        fn has_call(&self, expected_call: &MockDriverCall) -> bool {
            self.calls.contains(expected_call)
        }

        #[allow(dead_code)] // May not be used in all revised tests
        fn count_draw_text_run_calls(&self) -> usize {
            self.calls
                .iter()
                .filter(|call| matches!(call, MockDriverCall::DrawTextRun { .. }))
                .count()
        }
    }

    impl Driver for MockDriver {
        fn new() -> Result<Self> {
            Ok(Self::new())
        }
        fn get_event_fd(&self) -> Option<RawFd> {
            None
        }
        fn process_events(&mut self) -> Result<Vec<BackendEvent>> {
            Ok(Vec::new())
        }
        fn get_font_dimensions(&self) -> (usize, usize) {
            (8, 16) // Standard mock dimensions
        }
        fn get_display_dimensions_pixels(&self) -> (u16, u16) {
            (640, 480) // Standard mock dimensions
        }
        fn clear_all(&mut self, bg: Color) -> Result<()> {
            self.calls.push(MockDriverCall::ClearAll { bg });
            Ok(())
        }
        fn draw_text_run(
            &mut self,
            coords: CellCoords,
            text: &str,
            style: TextRunStyle,
        ) -> Result<()> {
            self.calls.push(MockDriverCall::DrawTextRun {
                coords,
                text: text.to_string(),
                style,
            });
            Ok(())
        }
        fn fill_rect(&mut self, rect: CellRect, color: Color) -> Result<()> {
            self.calls.push(MockDriverCall::FillRect { rect, color });
            Ok(())
        }
        fn present(&mut self) -> Result<()> {
            self.calls.push(MockDriverCall::Present);
            Ok(())
        }
        fn cleanup(&mut self) -> Result<()> {
            Ok(())
        }
    }

    // --- Renderer Tests ---

    #[test]
    fn test_draw_empty_terminal() {
        let mut renderer = Renderer::new();
        let mut mock_term = MockTerminal::new(0, 0); // Terminal with zero dimensions
        let mut mock_driver = MockDriver::new();
        renderer.draw(&mut mock_term, &mut mock_driver).unwrap();
        assert!(
            mock_driver.calls.is_empty(),
            "No draw calls should be made for 0x0 terminal. Calls: {:?}",
            mock_driver.calls
        );
    }

    #[test]
    fn test_initial_draw_clears_all_with_default_bg() {
        let mut renderer = Renderer::new(); // first_draw is true
        let mut mock_term = MockTerminal::new(2, 1); // All lines initially dirty in mock
        let mut mock_driver = MockDriver::new();

        renderer.draw(&mut mock_term, &mut mock_driver).unwrap();

        // 1. Check for ClearAll
        assert!(
            mock_driver.calls.contains(&MockDriverCall::ClearAll {
                bg: RENDERER_DEFAULT_BG // Renderer's default background
            }),
            "Expected ClearAll with renderer's default background. Calls: {:?}",
            mock_driver.calls
        );

        // 2. Check for drawing the space cells.
        // Renderer should optimize contiguous spaces with the same background into a single FillRect.
        let expected_combined_space_fill = MockDriverCall::FillRect {
            rect: CellRect { x: 0, y: 0, width: 2, height: 1 }, // Covers both cells (0,0) and (1,0)
            color: RENDERER_DEFAULT_BG, // Effective background of the spaces
        };
        assert!(
            mock_driver.calls.contains(&expected_combined_space_fill),
            "Cells (0,0) and (1,0) (spaces) should be drawn by a combined FillRect. Calls: {:?}",
            mock_driver.calls
        );

        // 3. Check for cursor draw (inverted, over the space at 0,0)
        let expected_cursor_draw = MockDriverCall::DrawTextRun {
            coords: CellCoords { x: 0, y: 0 },
            text: " ".to_string(), // Cursor is over a space
            style: TextRunStyle {
                fg: RENDERER_DEFAULT_BG, // Inverted FG (becomes effective BG of space)
                bg: RENDERER_DEFAULT_FG, // Inverted BG (becomes effective FG of space)
                flags: AttrFlags::empty(),
            },
        };
        assert!(
            mock_driver.calls.contains(&expected_cursor_draw),
            "Cursor draw not found or incorrect. Calls: {:?}",
            mock_driver.calls
        );

        // 4. Check for Present
        assert!(
            mock_driver.calls.contains(&MockDriverCall::Present),
            "Present was not called. Calls: {:?}",
            mock_driver.calls
        );
        assert!(!renderer.first_draw, "first_draw flag should be false after the initial draw.");
    }

    #[test]
    fn test_draw_single_default_char_no_clearall() {
        let mut renderer = Renderer::new();
        // Explicitly set first_draw to false to simulate a subsequent partial update.
        renderer.first_draw = false; 

        let mut mock_term = MockTerminal::new(1, 1);
        mock_term.clear_all_dirty_lines(); // Start with no lines marked dirty by the terminal itself
        let mut mock_driver = MockDriver::new();

        // Setup: A single character 'A' at (0,0) with default attributes.
        // Line 0 will be marked dirty by the terminal due to this change.
        mock_term.set_glyph(
            0,
            0,
            Glyph {
                c: 'A',
                attr: Attributes::default(), // fg: Default, bg: Default
            },
        );
        mock_term.mark_line_dirty(0); // Simulate terminal marking this line dirty
        mock_term.set_cursor_screen_pos(0, 0); // Cursor is on 'A'

        renderer.draw(&mut mock_term, &mut mock_driver).unwrap();

        // 1. Assert ClearAll was NOT called
        assert!(
            !mock_driver
                .calls
                .iter()
                .any(|call| matches!(call, MockDriverCall::ClearAll { .. })),
            "ClearAll should NOT be called for a single dirty line when not the first_draw. Calls: {:?}",
            mock_driver.calls
        );

        // 2. Assert 'A' is drawn with resolved default styles
        let expected_text_run_a = MockDriverCall::DrawTextRun {
            coords: CellCoords { x: 0, y: 0 },
            text: "A".to_string(),
            style: TextRunStyle {
                fg: RENDERER_DEFAULT_FG, // Resolved from Color::Default
                bg: RENDERER_DEFAULT_BG, // Resolved from Color::Default
                flags: AttrFlags::empty(),
            },
        };
        assert!(
            mock_driver.calls.contains(&expected_text_run_a),
            "Expected text run for 'A' with default style not found. Calls: {:?}",
            mock_driver.calls
        );
        
        // 3. Assert cursor is drawn over 'A' (inverted default styles)
        let expected_cursor_draw_over_a = MockDriverCall::DrawTextRun {
            coords: CellCoords { x: 0, y: 0 },
            text: "A".to_string(),
            style: TextRunStyle {
                fg: RENDERER_DEFAULT_BG, // Inverted FG (becomes default BG)
                bg: RENDERER_DEFAULT_FG, // Inverted BG (becomes default FG)
                flags: AttrFlags::empty(),
            },
        };
         assert!(
            mock_driver.calls.contains(&expected_cursor_draw_over_a),
            "Expected cursor draw for 'A' not found. Calls: {:?}",
            mock_driver.calls
        );

        // 4. Assert Present was called
        assert!(
            mock_driver.calls.contains(&MockDriverCall::Present),
            "Present was not called. Calls: {:?}",
            mock_driver.calls
        );
    }

    #[test]
    fn test_draw_char_with_specific_colors() {
        let mut renderer = Renderer::new();
        renderer.first_draw = false; 
        let mut mock_term = MockTerminal::new(1, 1);
        mock_term.clear_all_dirty_lines();
        let mut mock_driver = MockDriver::new();
        let fg_red = Color::Named(NamedColor::Red);
        let bg_blue = Color::Named(NamedColor::Blue);

        mock_term.set_glyph(
            0,
            0,
            Glyph {
                c: 'R',
                attr: Attributes {
                    fg: fg_red,
                    bg: bg_blue,
                    flags: AttrFlags::empty(),
                },
            },
        );
        mock_term.mark_line_dirty(0);
        mock_term.set_cursor_screen_pos(0, 0);

        renderer.draw(&mut mock_term, &mut mock_driver).unwrap();

        let expected_cell_draw = MockDriverCall::DrawTextRun {
            coords: CellCoords { x: 0, y: 0 },
            text: "R".to_string(),
            style: TextRunStyle {
                fg: fg_red,
                bg: bg_blue,
                flags: AttrFlags::empty(),
            },
        };
        let expected_cursor_draw = MockDriverCall::DrawTextRun {
            coords: CellCoords { x: 0, y: 0 },
            text: "R".to_string(),
            style: TextRunStyle {
                fg: bg_blue, 
                bg: fg_red,  
                flags: AttrFlags::empty(),
            },
        };

        assert!(mock_driver.calls.contains(&expected_cell_draw), "Specific color text run not found. Calls: {:?}", mock_driver.calls);
        assert!(mock_driver.calls.contains(&expected_cursor_draw), "Cursor over specific color text run not found. Calls: {:?}", mock_driver.calls);
    }

    #[test]
    fn test_draw_char_with_reverse_video() {
        let mut renderer = Renderer::new();
        renderer.first_draw = false;
        let mut mock_term = MockTerminal::new(1, 1);
        mock_term.clear_all_dirty_lines();
        let mut mock_driver = MockDriver::new();

        mock_term.set_glyph(
            0,
            0,
            Glyph {
                c: 'X',
                attr: Attributes {
                    fg: RENDERER_DEFAULT_FG, 
                    bg: RENDERER_DEFAULT_BG, 
                    flags: AttrFlags::REVERSE,
                },
            },
        );
        mock_term.mark_line_dirty(0);
        mock_term.set_cursor_screen_pos(0, 0);

        renderer.draw(&mut mock_term, &mut mock_driver).unwrap();

        let expected_cell_draw = MockDriverCall::DrawTextRun {
            coords: CellCoords { x: 0, y: 0 },
            text: "X".to_string(),
            style: TextRunStyle {
                fg: RENDERER_DEFAULT_BG,   
                bg: RENDERER_DEFAULT_FG,   
                flags: AttrFlags::empty(), 
            },
        };
        let expected_cursor_draw = MockDriverCall::DrawTextRun {
            coords: CellCoords { x: 0, y: 0 },
            text: "X".to_string(),
            style: TextRunStyle {
                fg: RENDERER_DEFAULT_FG, 
                bg: RENDERER_DEFAULT_BG, 
                flags: AttrFlags::empty(),
            },
        };
        assert!(mock_driver.calls.contains(&expected_cell_draw), "Reversed video text run for cell not found. Calls: {:?}", mock_driver.calls);
        assert!(mock_driver.calls.contains(&expected_cursor_draw), "Cursor over reversed video text run not found. Calls: {:?}", mock_driver.calls);
    }

    #[test]
    fn test_draw_only_dirty_lines_and_cursor_line() {
        let mut renderer = Renderer::new();
        renderer.first_draw = false;
        let mut mock_term = MockTerminal::new(2, 2);
        mock_term.clear_all_dirty_lines(); 
        let mut mock_driver = MockDriver::new();

        mock_term.set_glyph(0,0, Glyph { c: 'A', attr: Attributes::default() }); 
        mock_term.set_glyph(1,0, Glyph { c: ' ', attr: Attributes::default() }); // Cell (1,0) is a space
        mock_term.set_glyph(0,1, Glyph { c: 'B', attr: Attributes::default() }); 
        mock_term.set_glyph(1,1, Glyph { c: ' ', attr: Attributes::default() }); // Cell (1,1) is a space

        mock_term.mark_line_dirty(1); // Only line 1 is dirty from terminal's perspective
        mock_term.set_cursor_screen_pos(0, 0); // Cursor on line 0

        renderer.draw(&mut mock_term, &mut mock_driver).unwrap();

        let mut drawn_content_a_on_line_0 = false;
        let mut drawn_content_b_on_line_1 = false;
        let mut cursor_drawn_on_line_0 = false;

        for call in &mock_driver.calls {
            if let MockDriverCall::DrawTextRun { coords, text, style } = call {
                if coords.y == 0 && text == "A" && style.fg == RENDERER_DEFAULT_FG {
                    drawn_content_a_on_line_0 = true;
                } else if coords.y == 0 && text == "A" && style.fg == RENDERER_DEFAULT_BG {
                    cursor_drawn_on_line_0 = true;
                } else if coords.y == 1 && text == "B" && style.fg == RENDERER_DEFAULT_FG {
                    drawn_content_b_on_line_1 = true;
                }
            }
        }
        assert!(drawn_content_a_on_line_0, "Content 'A' of Line 0 (cursor line) should have been drawn. Calls: {:?}", mock_driver.calls);
        assert!(cursor_drawn_on_line_0, "Cursor on Line 0 should have been drawn. Calls: {:?}", mock_driver.calls);
        assert!(drawn_content_b_on_line_1, "Content 'B' of Line 1 (dirty) should have been drawn. Calls: {:?}", mock_driver.calls);
    }

    #[test]
    fn test_draw_wide_char_and_placeholder_correctly() {
        let mut renderer = Renderer::new();
        renderer.first_draw = false;
        let mut mock_term = MockTerminal::new(3, 1);
        mock_term.clear_all_dirty_lines();
        let mut mock_driver = MockDriver::new();

        let wide_char = '世';
        let wide_char_attrs = Attributes { fg: Color::Named(NamedColor::Green), bg: RENDERER_DEFAULT_BG, flags: AttrFlags::empty() };

        mock_term.set_glyph(0,0, Glyph { c: wide_char, attr: wide_char_attrs });
        mock_term.set_glyph(1,0, Glyph { c: '\0', attr: wide_char_attrs }); 
        mock_term.set_glyph(2,0, Glyph { c: 'N', attr: Attributes::default() });
        mock_term.mark_line_dirty(0);
        mock_term.set_cursor_screen_pos(0, 0); 

        renderer.draw(&mut mock_term, &mut mock_driver).unwrap();

        let mut wide_char_text_drawn = false;
        let mut wide_char_cursor_drawn = false;
        let mut placeholder_filled = false;
        let mut normal_char_drawn = false;

        for call in &mock_driver.calls {
            match call {
                MockDriverCall::DrawTextRun { coords, text, style } => {
                    if *coords == (CellCoords { x: 0, y: 0 }) && text == &wide_char.to_string() {
                        if style.fg == Color::Named(NamedColor::Green) && style.bg == RENDERER_DEFAULT_BG {
                            wide_char_text_drawn = true;
                        } else if style.fg == RENDERER_DEFAULT_BG && style.bg == Color::Named(NamedColor::Green) {
                            wide_char_cursor_drawn = true;
                        }
                    } else if *coords == (CellCoords { x: 2, y: 0 }) && text == "N" {
                        if style.fg == RENDERER_DEFAULT_FG && style.bg == RENDERER_DEFAULT_BG {
                             normal_char_drawn = true;
                        }
                    }
                }
                MockDriverCall::FillRect { rect, color } => {
                    if *rect == (CellRect { x: 1, y: 0, width: 1, height: 1 }) && *color == RENDERER_DEFAULT_BG {
                        placeholder_filled = true;
                    }
                }
                _ => {}
            }
        }
        assert!(wide_char_text_drawn, "Wide char text run not found. Calls: {:?}", mock_driver.calls);
        assert!(wide_char_cursor_drawn, "Wide char cursor run not found. Calls: {:?}", mock_driver.calls);
        assert!(placeholder_filled, "Placeholder fill not found. Calls: {:?}", mock_driver.calls);
        assert!(normal_char_drawn, "Normal char text run not found. Calls: {:?}", mock_driver.calls);
    }

    #[test]
    fn test_draw_line_of_spaces_optimised() {
        let mut renderer = Renderer::new();
        renderer.first_draw = false;
        let mut mock_term = MockTerminal::new(3, 1);
        mock_term.clear_all_dirty_lines();
        let mut mock_driver = MockDriver::new();

        let space_attributes = Attributes { fg: RENDERER_DEFAULT_FG, bg: RENDERER_DEFAULT_BG, flags: AttrFlags::empty() };
        mock_term.set_glyph(0,0, Glyph { c: ' ', attr: space_attributes });
        mock_term.set_glyph(1,0, Glyph { c: ' ', attr: space_attributes });
        mock_term.set_glyph(2,0, Glyph { c: ' ', attr: space_attributes });
        mock_term.mark_line_dirty(0);
        mock_term.set_cursor_screen_pos(0, 0);

        renderer.draw(&mut mock_term, &mut mock_driver).unwrap();

        let expected_fill = MockDriverCall::FillRect {
            rect: CellRect { x: 0, y: 0, width: 3, height: 1 },
            color: RENDERER_DEFAULT_BG,
        };
        assert!(mock_driver.calls.contains(&expected_fill), "Expected FillRect for the line of spaces. Calls: {:?}", mock_driver.calls);
        
        let text_run_for_spaces_exists = mock_driver.calls.iter().any(|call| {
             matches!(call, MockDriverCall::DrawTextRun { text, style, .. } if text.trim().is_empty() && style.fg == RENDERER_DEFAULT_FG && style.bg == RENDERER_DEFAULT_BG)
        });
        assert!(!text_run_for_spaces_exists, "Spaces should be drawn with FillRect, not DrawTextRun for content. Calls: {:?}", mock_driver.calls);

        let expected_cursor_draw = MockDriverCall::DrawTextRun {
            coords: CellCoords { x: 0, y: 0 },
            text: " ".to_string(),
            style: TextRunStyle { fg: RENDERER_DEFAULT_BG, bg: RENDERER_DEFAULT_FG, flags: AttrFlags::empty() },
        };
        assert!(mock_driver.calls.contains(&expected_cursor_draw), "Cursor draw not found over spaces. Calls: {:?}", mock_driver.calls);
    }

    #[test]
    fn test_cursor_movement_redraws_old_and_new_cell_correctly() {
        let mut renderer = Renderer::new();
        renderer.first_draw = false;
        let mut mock_term = MockTerminal::new(2, 1);
        mock_term.clear_all_dirty_lines();
        let mut mock_driver = MockDriver::new();

        let attr_a = Attributes { fg: Color::Named(NamedColor::Red), bg: RENDERER_DEFAULT_BG, flags: AttrFlags::empty() };
        let attr_b = Attributes { fg: Color::Named(NamedColor::Blue), bg: RENDERER_DEFAULT_BG, flags: AttrFlags::empty() };
        mock_term.set_glyph(0,0, Glyph { c: 'A', attr: attr_a });
        mock_term.set_glyph(1,0, Glyph { c: 'B', attr: attr_b });

        mock_term.set_cursor_screen_pos(0, 0);
        mock_term.mark_line_dirty(0); // Line 0 is dirty due to cursor being there
        renderer.draw(&mut mock_term, &mut mock_driver).unwrap();
        mock_driver.calls.clear(); // Clear calls from this initial setup draw

        // Action: Move cursor from (0,0) to (1,0)
        mock_term.clear_all_dirty_lines(); // Clear previous dirty state
        mock_term.set_cursor_screen_pos(1, 0);
        mock_term.mark_line_dirty(0); // Line 0 is dirty because cursor moved on/off it

        renderer.draw(&mut mock_term, &mut mock_driver).unwrap();

        let expected_redraw_a = MockDriverCall::DrawTextRun {
            coords: CellCoords { x: 0, y: 0 },
            text: "A".to_string(),
            style: TextRunStyle { fg: Color::Named(NamedColor::Red), bg: RENDERER_DEFAULT_BG, flags: AttrFlags::empty() },
        };
        assert!(mock_driver.calls.contains(&expected_redraw_a), "Old cursor cell 'A' not redrawn correctly. Calls: {:?}", mock_driver.calls);

        let expected_draw_b_content = MockDriverCall::DrawTextRun {
            coords: CellCoords { x: 1, y: 0 },
            text: "B".to_string(),
            style: TextRunStyle { fg: Color::Named(NamedColor::Blue), bg: RENDERER_DEFAULT_BG, flags: AttrFlags::empty() },
        };
        assert!(mock_driver.calls.contains(&expected_draw_b_content), "New cursor cell 'B' content not drawn. Calls: {:?}", mock_driver.calls);

        let expected_cursor_on_b = MockDriverCall::DrawTextRun {
            coords: CellCoords { x: 1, y: 0 },
            text: "B".to_string(),
            style: TextRunStyle { fg: RENDERER_DEFAULT_BG, bg: Color::Named(NamedColor::Blue), flags: AttrFlags::empty() },
        };
        assert!(mock_driver.calls.contains(&expected_cursor_on_b), "Cursor not drawn correctly at new position over 'B'. Calls: {:?}", mock_driver.calls);
        assert!(mock_driver.calls.contains(&MockDriverCall::Present));
    }

    #[test]
    fn test_visual_bug_cursor_leaves_white_tile_then_text_reappears() {
        let mut renderer = Renderer::new();
        let mut mock_term = MockTerminal::new(2, 1);
        let mut mock_driver = MockDriver::new();

        let initial_text_attr = Attributes { fg: RENDERER_DEFAULT_FG, bg: RENDERER_DEFAULT_BG, flags: AttrFlags::empty() };
        let space_attr = Attributes { fg: RENDERER_DEFAULT_FG, bg: RENDERER_DEFAULT_BG, flags: AttrFlags::empty() };

        mock_term.set_glyph(0,0, Glyph { c: 'T', attr: initial_text_attr });
        mock_term.set_glyph(1,0, Glyph { c: ' ', attr: space_attr });

        // Step 1: Cursor on 'T'. Ensure it's treated as the first draw.
        renderer.first_draw = true; 
        mock_term.clear_all_dirty_lines();
        mock_term.mark_line_dirty(0); // Line 0 is dirty
        mock_term.set_cursor_screen_pos(0, 0);
        renderer.draw(&mut mock_term, &mut mock_driver).unwrap();

        let expected_cursor_on_t_step1 = MockDriverCall::DrawTextRun {
            coords: CellCoords { x: 0, y: 0 },
            text: "T".to_string(),
            style: TextRunStyle { fg: RENDERER_DEFAULT_BG, bg: RENDERER_DEFAULT_FG, flags: AttrFlags::empty() },
        };
        assert!(mock_driver.calls.contains(&MockDriverCall::ClearAll { bg: RENDERER_DEFAULT_BG }), "Step 1 ClearAll missing");
        assert!(mock_driver.calls.contains(&expected_cursor_on_t_step1), "Step 1 FAILURE: Cursor on 'T' was not drawn inverted. Calls: {:?}", mock_driver.calls);
        mock_driver.calls.clear();
        // renderer.first_draw is now false due to the draw call.

        // Step 2: Cursor moves to space at (1,0)
        mock_term.clear_all_dirty_lines();
        mock_term.mark_line_dirty(0); // Line 0 is dirty because cursor moved
        mock_term.set_cursor_screen_pos(1, 0);
        renderer.draw(&mut mock_term, &mut mock_driver).unwrap();

        let expected_redraw_t_original_style_step2 = MockDriverCall::DrawTextRun {
            coords: CellCoords { x: 0, y: 0 },
            text: "T".to_string(),
            style: TextRunStyle { fg: RENDERER_DEFAULT_FG, bg: RENDERER_DEFAULT_BG, flags: AttrFlags::empty() },
        };
        assert!(mock_driver.calls.contains(&expected_redraw_t_original_style_step2), "Step 2 FAILURE: Original 'T' at (0,0) not redrawn. Calls: {:?}", mock_driver.calls);

        let expected_cursor_on_space_step2 = MockDriverCall::DrawTextRun {
            coords: CellCoords { x: 1, y: 0 },
            text: " ".to_string(),
            style: TextRunStyle { fg: RENDERER_DEFAULT_BG, bg: RENDERER_DEFAULT_FG, flags: AttrFlags::empty() },
        };
        assert!(mock_driver.calls.contains(&expected_cursor_on_space_step2), "Step 2 FAILURE: Cursor on space at (1,0) not drawn inverted. Calls: {:?}", mock_driver.calls);
        mock_driver.calls.clear();

        // Step 3: Cursor moves back to 'T' at (0,0)
        mock_term.clear_all_dirty_lines();
        mock_term.mark_line_dirty(0); // Line 0 is dirty
        mock_term.set_cursor_screen_pos(0, 0);
        renderer.draw(&mut mock_term, &mut mock_driver).unwrap();

        let expected_redraw_space_original_style_step3 = MockDriverCall::FillRect {
            rect: CellRect { x: 1, y: 0, width: 1, height: 1 },
            color: RENDERER_DEFAULT_BG,
        };
        assert!(mock_driver.calls.contains(&expected_redraw_space_original_style_step3), "Step 3 FAILURE: Original space at (1,0) not redrawn. Calls: {:?}", mock_driver.calls);

        let expected_cursor_on_t_again_step3 = MockDriverCall::DrawTextRun {
            coords: CellCoords { x: 0, y: 0 },
            text: "T".to_string(),
            style: TextRunStyle { fg: RENDERER_DEFAULT_BG, bg: RENDERER_DEFAULT_FG, flags: AttrFlags::empty() },
        };
        assert!(mock_driver.calls.contains(&expected_cursor_on_t_again_step3), "Step 3 FAILURE: Cursor moving back onto 'T' not drawn inverted. Calls: {:?}", mock_driver.calls);
        assert!(mock_driver.calls.contains(&MockDriverCall::Present), "Present was not called at the end of Step 3.");
    }
}
