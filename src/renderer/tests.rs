// src/renderer/tests.rs

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
        // Added to simulate driver knowing its own default background if needed
        // For most tests, renderer dictates colors.
        // default_driver_bg: Color,
    }

    impl MockDriver {
        fn new() -> Self {
            MockDriver {
                calls: Vec::new(),
                // default_driver_bg: RENDERER_DEFAULT_BG, // Align with renderer
            }
        }

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
            (8, 16)
        }
        fn get_display_dimensions_pixels(&self) -> (u16, u16) {
            (640, 480)
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
        let mut mock_term = MockTerminal::new(0, 0);
        let mut mock_driver = MockDriver::new();
        renderer.draw(&mut mock_term, &mut mock_driver).unwrap();
        assert!(
            mock_driver.calls.is_empty(),
            "No draw calls should be made for 0x0 terminal"
        );
    }

    #[test]
    fn test_initial_draw_clears_all_with_default_bg() {
        let mut renderer = Renderer::new();
        let mut mock_term = MockTerminal::new(2, 1); // All lines initially dirty in mock
        let mut mock_driver = MockDriver::new();

        renderer.draw(&mut mock_term, &mut mock_driver).unwrap();

        assert!(
            mock_driver.calls.contains(&MockDriverCall::ClearAll {
                bg: RENDERER_DEFAULT_BG
            }),
            "Expected ClearAll with renderer's default background. Calls: {:?}",
            mock_driver.calls
        );

        // Cells are ' ' by default. Cursor at (0,0).
        let expected_cell_draw_at_0_0_as_space_run = MockDriverCall::FillRect {
            rect: CellRect {
                x: 0,
                y: 0,
                width: 1,
                height: 1,
            }, // Assuming space run logic produces this for single space
            color: RENDERER_DEFAULT_BG,
        };
        let expected_cell_draw_at_1_0_as_space_run = MockDriverCall::FillRect {
            rect: CellRect {
                x: 1,
                y: 0,
                width: 1,
                height: 1,
            },
            color: RENDERER_DEFAULT_BG,
        };

        // This assertion depends on whether a single space is FillRect or DrawTextRun in your optimized path.
        // Given the new space handling, it's likely FillRect.
        assert!(
            mock_driver
                .calls
                .contains(&expected_cell_draw_at_0_0_as_space_run)
                || mock_driver.calls.contains(&MockDriverCall::DrawTextRun {
                    coords: CellCoords { x: 0, y: 0 },
                    text: " ".to_string(),
                    style: TextRunStyle {
                        fg: RENDERER_DEFAULT_FG,
                        bg: RENDERER_DEFAULT_BG,
                        flags: AttrFlags::empty()
                    }
                }),
            "Cell (0,0) should be drawn. Calls: {:?}",
            mock_driver.calls
        );
        assert!(
            mock_driver
                .calls
                .contains(&expected_cell_draw_at_1_0_as_space_run)
                || mock_driver.calls.contains(&MockDriverCall::DrawTextRun {
                    coords: CellCoords { x: 1, y: 0 },
                    text: " ".to_string(),
                    style: TextRunStyle {
                        fg: RENDERER_DEFAULT_FG,
                        bg: RENDERER_DEFAULT_BG,
                        flags: AttrFlags::empty()
                    }
                }),
            "Cell (1,0) should be drawn. Calls: {:?}",
            mock_driver.calls
        );

        let expected_cursor_draw = MockDriverCall::DrawTextRun {
            coords: CellCoords { x: 0, y: 0 },
            text: " ".to_string(), // Cursor over default space
            style: TextRunStyle {
                fg: RENDERER_DEFAULT_BG,
                bg: RENDERER_DEFAULT_FG,
                flags: AttrFlags::empty(),
            },
        };
        assert!(
            mock_driver.calls.contains(&expected_cursor_draw),
            "Cursor draw not found. Calls: {:?}",
            mock_driver.calls
        );
        assert!(
            mock_driver.calls.contains(&MockDriverCall::Present),
            "Present was not called. Calls: {:?}",
            mock_driver.calls
        );
    }

    #[test]
    fn test_draw_single_default_char_no_clearall() {
        // Renamed to reflect the "no ClearAll" expectation
        let mut renderer = Renderer::new();
        let mut mock_term = MockTerminal::new(1, 1);
        mock_term.clear_all_dirty_lines(); // Start with a clean slate
        let mut mock_driver = MockDriver::new();

        mock_term.set_glyph(
            0,
            0,
            Glyph {
                c: 'A',
                attr: Attributes::default(),
            },
        );
        mock_term.mark_line_dirty(0);
        mock_term.set_cursor_screen_pos(0, 0); // Cursor is on 'A'

        renderer.draw(&mut mock_term, &mut mock_driver).unwrap();

        assert!(
            !mock_driver
                .calls
                .iter()
                .any(|call| matches!(call, MockDriverCall::ClearAll { .. })),
            "ClearAll should NOT be called for a single dirty line. Calls: {:?}",
            mock_driver.calls
        );

        let expected_text_run_a = MockDriverCall::DrawTextRun {
            coords: CellCoords { x: 0, y: 0 },
            text: "A".to_string(),
            style: TextRunStyle {
                fg: RENDERER_DEFAULT_FG,
                bg: RENDERER_DEFAULT_BG,
                flags: AttrFlags::empty(),
            },
        };
        // The order matters: cell content, then cursor overlay
        let first_draw_text_run = mock_driver.calls.iter().find_map(|call| {
            if let MockDriverCall::DrawTextRun {
                coords,
                text,
                style,
            } = call
            {
                if coords.x == 0 && coords.y == 0 && text == "A" && style.fg == RENDERER_DEFAULT_FG
                {
                    return Some(call.clone());
                }
            }
            None
        });
        assert_eq!(
            first_draw_text_run,
            Some(expected_text_run_a),
            "Expected text run for 'A' with default style not found or not first. Calls: {:?}",
            mock_driver.calls
        );

        let expected_cursor_draw_over_a = MockDriverCall::DrawTextRun {
            coords: CellCoords { x: 0, y: 0 },
            text: "A".to_string(),
            style: TextRunStyle {
                fg: RENDERER_DEFAULT_BG,
                bg: RENDERER_DEFAULT_FG,
                flags: AttrFlags::empty(),
            },
        };
        let second_draw_text_run = mock_driver
            .calls
            .iter()
            .filter_map(|call| {
                if let MockDriverCall::DrawTextRun {
                    coords,
                    text,
                    style,
                } = call
                {
                    if coords.x == 0
                        && coords.y == 0
                        && text == "A"
                        && style.fg == RENDERER_DEFAULT_BG
                    {
                        return Some(call.clone());
                    }
                }
                None
            })
            .nth(0); // Get the first match for cursor draw
        assert_eq!(
            second_draw_text_run,
            Some(expected_cursor_draw_over_a),
            "Expected cursor draw for 'A' not found or not second. Calls: {:?}",
            mock_driver.calls
        );

        assert!(
            mock_driver.calls.contains(&MockDriverCall::Present),
            "Present was not called. Calls: {:?}",
            mock_driver.calls
        );
    }

    #[test]
    fn test_draw_char_with_specific_colors() {
        let mut renderer = Renderer::new();
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
            }, // Inverted
        };

        let found_cell_draw = mock_driver
            .calls
            .iter()
            .find(|&call| *call == expected_cell_draw)
            .is_some();
        let found_cursor_draw = mock_driver
            .calls
            .iter()
            .find(|&call| *call == expected_cursor_draw)
            .is_some();

        assert!(
            found_cell_draw,
            "Specific color text run not found. Calls: {:?}",
            mock_driver.calls
        );
        assert!(
            found_cursor_draw,
            "Cursor over specific color text run not found. Calls: {:?}",
            mock_driver.calls
        );
    }

    #[test]
    fn test_draw_char_with_reverse_video() {
        let mut renderer = Renderer::new();
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

        // Cell content should be drawn with REVERSE applied
        let expected_cell_draw = MockDriverCall::DrawTextRun {
            coords: CellCoords { x: 0, y: 0 },
            text: "X".to_string(),
            style: TextRunStyle {
                fg: RENDERER_DEFAULT_BG,   // Swapped due to REVERSE
                bg: RENDERER_DEFAULT_FG,   // Swapped due to REVERSE
                flags: AttrFlags::empty(), // REVERSE flag consumed by get_effective_colors_and_flags
            },
        };
        // Cursor should draw over this, inverting the *effective* (already reversed) colors
        let expected_cursor_draw = MockDriverCall::DrawTextRun {
            coords: CellCoords { x: 0, y: 0 },
            text: "X".to_string(),
            style: TextRunStyle {
                fg: RENDERER_DEFAULT_FG, // Swapped back
                bg: RENDERER_DEFAULT_BG, // Swapped back
                flags: AttrFlags::empty(),
            },
        };

        let found_cell_draw = mock_driver
            .calls
            .iter()
            .find(|&call| *call == expected_cell_draw)
            .is_some();
        let found_cursor_draw = mock_driver
            .calls
            .iter()
            .find(|&call| *call == expected_cursor_draw)
            .is_some();

        assert!(
            found_cell_draw,
            "Reversed video text run for cell not found. Calls: {:?}",
            mock_driver.calls
        );
        assert!(
            found_cursor_draw,
            "Cursor over reversed video text run not found. Calls: {:?}",
            mock_driver.calls
        );
    }

    #[test]
    fn test_draw_only_dirty_lines_and_cursor_line() {
        let mut renderer = Renderer::new();
        let mut mock_term = MockTerminal::new(2, 2);
        mock_term.clear_all_dirty_lines(); // Start clean
        let mut mock_driver = MockDriver::new();

        // Content for lines
        mock_term.set_glyph(
            0,
            0,
            Glyph {
                c: 'A',
                attr: Attributes::default(),
            },
        ); // Line 0
        mock_term.set_glyph(
            0,
            1,
            Glyph {
                c: 'B',
                attr: Attributes::default(),
            },
        ); // Line 1

        // Mark only line 1 dirty for content
        mock_term.mark_line_dirty(1);
        // Cursor is on line 0 (default (0,0))
        mock_term.set_cursor_screen_pos(0, 0);

        renderer.draw(&mut mock_term, &mut mock_driver).unwrap();

        let mut drawn_content_on_line_0 = false;
        let mut drawn_content_on_line_1 = false;
        let mut cursor_drawn_on_line_0 = false;

        for call in &mock_driver.calls {
            if let MockDriverCall::DrawTextRun {
                coords,
                text,
                style,
            } = call
            {
                if coords.y == 0 {
                    if text == "A" && style.fg == RENDERER_DEFAULT_FG {
                        // Original content of 'A'
                        drawn_content_on_line_0 = true;
                    } else if text == "A" && style.fg == RENDERER_DEFAULT_BG {
                        // Cursor over 'A'
                        cursor_drawn_on_line_0 = true;
                    }
                } else if coords.y == 1 && text == "B" && style.fg == RENDERER_DEFAULT_FG{
                        // Content of 'B'
                        drawn_content_on_line_1 = true;
                }
            }
        }
        // Line 0 content 'A' should be drawn because the cursor is there, making it effectively dirty for rendering.
        // The original test was too strict. The line of the cursor *will* be redrawn.
        assert!(
            drawn_content_on_line_0,
            "Content of Line 0 (cursor line) should have been drawn. Calls: {:?}",
            mock_driver.calls
        );
        assert!(
            cursor_drawn_on_line_0,
            "Cursor on Line 0 should have been drawn. Calls: {:?}",
            mock_driver.calls
        );
        assert!(
            drawn_content_on_line_1,
            "Content of Line 1 (dirty) should have been drawn. Calls: {:?}",
            mock_driver.calls
        );
    }

    #[test]
    fn test_draw_wide_char_and_placeholder_correctly() {
        // Renamed
        let mut renderer = Renderer::new();
        let mut mock_term = MockTerminal::new(3, 1);
        mock_term.clear_all_dirty_lines();
        let mut mock_driver = MockDriver::new();

        let wide_char = '世';
        let wide_char_attrs = Attributes {
            fg: Color::Named(NamedColor::Green),
            bg: RENDERER_DEFAULT_BG,
            flags: AttrFlags::empty(),
        };

        mock_term.set_glyph(
            0,
            0,
            Glyph {
                c: wide_char,
                attr: wide_char_attrs,
            },
        );
        mock_term.set_glyph(
            1,
            0,
            Glyph {
                c: '\0',
                attr: wide_char_attrs,
            },
        ); // Placeholder with same attributes
        mock_term.set_glyph(
            2,
            0,
            Glyph {
                c: 'N',
                attr: Attributes::default(),
            },
        );
        mock_term.mark_line_dirty(0);
        mock_term.set_cursor_screen_pos(0, 0); // Cursor on the wide char

        renderer.draw(&mut mock_term, &mut mock_driver).unwrap();

        let _expected_wide_char_text_call = MockDriverCall::DrawTextRun {
            coords: CellCoords { x: 0, y: 0 },
            text: wide_char.to_string(),
            style: TextRunStyle {
                fg: Color::Named(NamedColor::Green),
                bg: RENDERER_DEFAULT_BG,
                flags: AttrFlags::empty(),
            },
        };
        // Cursor will be drawn over the wide char.
        let _expected_wide_char_cursor_call = MockDriverCall::DrawTextRun {
            coords: CellCoords { x: 0, y: 0 },
            text: wide_char.to_string(),
            style: TextRunStyle {
                fg: RENDERER_DEFAULT_BG,
                bg: Color::Named(NamedColor::Green),
                flags: AttrFlags::empty(),
            },
        };

        let _expected_placeholder_fill = MockDriverCall::FillRect {
            rect: CellRect {
                x: 1,
                y: 0,
                width: 1,
                height: 1,
            },
            // Background should be that of the wide char cell
            color: RENDERER_DEFAULT_BG,
        };

        let _expected_normal_char_call = MockDriverCall::DrawTextRun {
            coords: CellCoords { x: 2, y: 0 },
            text: "N".to_string(),
            style: TextRunStyle {
                fg: RENDERER_DEFAULT_FG,
                bg: RENDERER_DEFAULT_BG,
                flags: AttrFlags::empty(),
            },
        };

        // Check for calls, order might be tricky due to how text runs are formed vs cursor
        let mut wide_char_text_drawn = false;
        let mut wide_char_cursor_drawn = false;
        let mut placeholder_filled = false;
        let mut normal_char_drawn = false;

        for call in &mock_driver.calls {
            match call {
                MockDriverCall::DrawTextRun {
                    coords,
                    text,
                    style,
                } => {
                    if *coords == (CellCoords { x: 0, y: 0 }) && text == &wide_char.to_string() {
                        if style.fg == Color::Named(NamedColor::Green) {
                            wide_char_text_drawn = true;
                        } else if style.fg == RENDERER_DEFAULT_BG {
                            wide_char_cursor_drawn = true;
                        }
                    } else if *coords == (CellCoords { x: 2, y: 0 }) && text == "N" {
                        normal_char_drawn = true;
                    }
                }
                MockDriverCall::FillRect { rect, color } => {
                    if *rect
                        == (CellRect {
                            x: 1,
                            y: 0,
                            width: 1,
                            height: 1,
                        })
                        && *color == RENDERER_DEFAULT_BG
                    {
                        placeholder_filled = true;
                    }
                }
                _ => {}
            }
        }
        assert!(
            wide_char_text_drawn,
            "Wide char text run not found as expected. Calls: {:?}",
            mock_driver.calls
        );
        assert!(
            wide_char_cursor_drawn,
            "Wide char cursor run not found as expected. Calls: {:?}",
            mock_driver.calls
        );
        assert!(
            placeholder_filled,
            "Placeholder fill not found as expected. Calls: {:?}",
            mock_driver.calls
        );
        assert!(
            normal_char_drawn,
            "Normal char text run not found as expected. Calls: {:?}",
            mock_driver.calls
        );
    }

    #[test]
    fn test_draw_line_of_spaces_optimised() {
        // Renamed
        let mut renderer = Renderer::new();
        let mut mock_term = MockTerminal::new(3, 1);
        mock_term.clear_all_dirty_lines();
        let mut mock_driver = MockDriver::new();

        // All cells are spaces with default background
        mock_term.set_glyph(
            0,
            0,
            Glyph {
                c: ' ',
                attr: Attributes::default(),
            },
        );
        mock_term.set_glyph(
            1,
            0,
            Glyph {
                c: ' ',
                attr: Attributes::default(),
            },
        );
        mock_term.set_glyph(
            2,
            0,
            Glyph {
                c: ' ',
                attr: Attributes::default(),
            },
        );
        mock_term.mark_line_dirty(0);
        mock_term.set_cursor_screen_pos(0, 0); // Cursor at start of spaces

        renderer.draw(&mut mock_term, &mut mock_driver).unwrap();

        // Expect a single FillRect for the entire line of spaces
        let expected_fill = MockDriverCall::FillRect {
            rect: CellRect {
                x: 0,
                y: 0,
                width: 3,
                height: 1,
            },
            color: RENDERER_DEFAULT_BG, // Based on effective background of spaces
        };
        assert!(
            mock_driver.calls.contains(&expected_fill),
            "Expected FillRect for the line of spaces. Calls: {:?}",
            mock_driver.calls
        );

        // Check that no DrawTextRun was made for the space characters themselves
        let text_run_for_spaces_exists = mock_driver.calls.iter().any(|call| {
            matches!(call, MockDriverCall::DrawTextRun { text, style, .. } if text.trim().is_empty() && style.fg == RENDERER_DEFAULT_FG)
        });
        assert!(
            !text_run_for_spaces_exists,
            "Spaces should be drawn with FillRect, not DrawTextRun for content. Calls: {:?}",
            mock_driver.calls
        );

        // Cursor should still be drawn
        let expected_cursor_draw = MockDriverCall::DrawTextRun {
            coords: CellCoords { x: 0, y: 0 },
            text: " ".to_string(),
            style: TextRunStyle {
                fg: RENDERER_DEFAULT_BG,
                bg: RENDERER_DEFAULT_FG,
                flags: AttrFlags::empty(),
            },
        };
        assert!(
            mock_driver.calls.contains(&expected_cursor_draw),
            "Cursor draw not found over spaces. Calls: {:?}",
            mock_driver.calls
        );
    }

    #[test]
    fn test_cursor_movement_redraws_old_and_new_cell_correctly() {
        let mut renderer = Renderer::new();
        let mut mock_term = MockTerminal::new(2, 1);
        mock_term.clear_all_dirty_lines();
        let mut mock_driver = MockDriver::new();

        let attr_a = Attributes {
            fg: Color::Named(NamedColor::Red),
            bg: RENDERER_DEFAULT_BG,
            flags: AttrFlags::empty(),
        };
        let attr_b = Attributes {
            fg: Color::Named(NamedColor::Blue),
            bg: RENDERER_DEFAULT_BG,
            flags: AttrFlags::empty(),
        };
        mock_term.set_glyph(
            0,
            0,
            Glyph {
                c: 'A',
                attr: attr_a,
            },
        );
        mock_term.set_glyph(
            1,
            0,
            Glyph {
                c: 'B',
                attr: attr_b,
            },
        );

        // Initial state: Cursor on 'A'
        mock_term.set_cursor_screen_pos(0, 0);
        mock_term.mark_line_dirty(0);
        renderer.draw(&mut mock_term, &mut mock_driver).unwrap();
        mock_driver.calls.clear(); // Clear calls from initial draw

        // Action: Move cursor from (0,0) to (1,0)
        // The terminal emulator logic would mark line 0 dirty because the cursor moved.
        mock_term.set_cursor_screen_pos(1, 0);
        mock_term.mark_line_dirty(0); // Simulate term logic marking old/new cursor line dirty

        renderer.draw(&mut mock_term, &mut mock_driver).unwrap();

        // Verify old cell (0,0) 'A' is redrawn with its original style
        let expected_redraw_a = MockDriverCall::DrawTextRun {
            coords: CellCoords { x: 0, y: 0 },
            text: "A".to_string(),
            style: TextRunStyle {
                fg: Color::Named(NamedColor::Red),
                bg: RENDERER_DEFAULT_BG,
                flags: AttrFlags::empty(),
            },
        };
        assert!(
            mock_driver.calls.contains(&expected_redraw_a),
            "Old cursor cell 'A' not redrawn correctly. Calls: {:?}",
            mock_driver.calls
        );

        // Verify new cell (1,0) 'B' is drawn (as part of line redraw)
        let expected_draw_b_content = MockDriverCall::DrawTextRun {
            coords: CellCoords { x: 1, y: 0 },
            text: "B".to_string(),
            style: TextRunStyle {
                fg: Color::Named(NamedColor::Blue),
                bg: RENDERER_DEFAULT_BG,
                flags: AttrFlags::empty(),
            },
        };
        assert!(
            mock_driver.calls.contains(&expected_draw_b_content),
            "New cursor cell 'B' content not drawn. Calls: {:?}",
            mock_driver.calls
        );

        // Verify cursor is drawn at new position (1,0) over 'B'
        let expected_cursor_on_b = MockDriverCall::DrawTextRun {
            coords: CellCoords { x: 1, y: 0 },
            text: "B".to_string(),
            style: TextRunStyle {
                fg: RENDERER_DEFAULT_BG,
                bg: Color::Named(NamedColor::Blue),
                flags: AttrFlags::empty(),
            },
        };
        assert!(
            mock_driver.calls.contains(&expected_cursor_on_b),
            "Cursor not drawn correctly at new position over 'B'. Calls: {:?}",
            mock_driver.calls
        );

        assert!(mock_driver.calls.contains(&MockDriverCall::Present));
    }

    // Located in myterm/src/renderer/tests.rs

    #[test]
    fn test_visual_bug_cursor_leaves_white_tile_then_text_reappears() {
        let mut renderer = Renderer::new();
        let mut mock_term = MockTerminal::new(2, 1); // A 2-cell wide, 1-row high terminal
        let mut mock_driver = MockDriver::new();

        // Initial cell attributes (e.g., White text on Black background)
        let initial_text_attr = Attributes {
            fg: RENDERER_DEFAULT_FG, // White
            bg: RENDERER_DEFAULT_BG, // Black
            flags: AttrFlags::empty(),
        };
        let space_attr = Attributes {
            fg: RENDERER_DEFAULT_FG,
            bg: RENDERER_DEFAULT_BG,
            flags: AttrFlags::empty(),
        };

        mock_term.set_glyph(
            0,
            0,
            Glyph {
                c: 'T',
                attr: initial_text_attr,
            },
        );
        mock_term.set_glyph(
            1,
            0,
            Glyph {
                c: ' ',
                attr: space_attr,
            },
        ); // A space for the cursor to move to

        // --- Step 1: Cursor at (0,0) on 'T'. Renderer draws it. ---
        // This draw should make cell (0,0) visually appear as Black 'T' on White background.
        mock_term.clear_all_dirty_lines();
        mock_term.mark_line_dirty(0);
        mock_term.set_cursor_screen_pos(0, 0);
        renderer.draw(&mut mock_term, &mut mock_driver).unwrap();

        // Verify cursor was drawn inverted over 'T'
        let expected_cursor_on_t_step1 = MockDriverCall::DrawTextRun {
            coords: CellCoords { x: 0, y: 0 },
            text: "T".to_string(),
            style: TextRunStyle {
                fg: RENDERER_DEFAULT_BG, // Black text (inverted)
                bg: RENDERER_DEFAULT_FG, // White background (inverted)
                flags: AttrFlags::empty(),
            },
        };
        assert!(
            mock_driver.calls.contains(&expected_cursor_on_t_step1),
            "Step 1 FAILURE: Cursor on 'T' was not drawn inverted. Calls: {:?}",
            mock_driver.calls
        );
        mock_driver.calls.clear(); // Clear calls for the next step

        // --- Step 2: Cursor moves to (1,0) (over the space). Renderer draws again. ---
        // This is where the "white wake" bug would manifest if not fixed.
        // We expect cell (0,0) to be redrawn with its *original* 'T' (White on Black).
        // We expect the cursor to be drawn over the space at (1,0) (Black ' ' on White).
        mock_term.clear_all_dirty_lines();
        mock_term.mark_line_dirty(0); // Line 0 is dirty because the cursor moved off it and onto it.
        mock_term.set_cursor_screen_pos(1, 0);
        renderer.draw(&mut mock_term, &mut mock_driver).unwrap();

        // Verify 'T' at (0,0) is redrawn with its original, non-inverted style.
        // This asserts that the "white wake" is painted over.
        let expected_redraw_t_original_style_step2 = MockDriverCall::DrawTextRun {
            coords: CellCoords { x: 0, y: 0 },
            text: "T".to_string(),
            style: TextRunStyle {
                fg: RENDERER_DEFAULT_FG, // White text
                bg: RENDERER_DEFAULT_BG, // Black background
                flags: AttrFlags::empty(),
            },
        };
        assert!(
            mock_driver
                .calls
                .contains(&expected_redraw_t_original_style_step2),
            "Step 2 FAILURE: Original 'T' at (0,0) not redrawn with its correct style (White on Black) after cursor moved away. This indicates the 'white wake' bug. Calls: {:?}",
            mock_driver.calls
        );

        // Verify cursor is drawn inverted over the space at (1,0)
        let expected_cursor_on_space_step2 = MockDriverCall::DrawTextRun {
            coords: CellCoords { x: 1, y: 0 },
            text: " ".to_string(),
            style: TextRunStyle {
                fg: RENDERER_DEFAULT_BG, // Black text (inverted space)
                bg: RENDERER_DEFAULT_FG, // White background (inverted)
                flags: AttrFlags::empty(),
            },
        };
        assert!(
            mock_driver.calls.contains(&expected_cursor_on_space_step2),
            "Step 2 FAILURE: Cursor on space at (1,0) not drawn inverted. Calls: {:?}",
            mock_driver.calls
        );
        mock_driver.calls.clear();

        // --- Step 3: Cursor moves back to (0,0) over 'T'. Renderer draws again. ---
        // Now, cell (0,0) should have been correctly repainted (White 'T' on Black background in Step 2).
        // The cursor moving back onto it should draw 'T' as Black text on White background.
        // Cell (1,0) (the space) should be redrawn with its original style (White on Black).
        mock_term.clear_all_dirty_lines();
        mock_term.mark_line_dirty(0); // Line 0 is dirty.
        mock_term.set_cursor_screen_pos(0, 0);
        renderer.draw(&mut mock_term, &mut mock_driver).unwrap();

        // Verify space at (1,0) is redrawn with its original style (e.g., FillRect with Black)
        let expected_redraw_space_original_style_step3 = MockDriverCall::FillRect {
            rect: CellRect {
                x: 1,
                y: 0,
                width: 1,
                height: 1,
            },
            color: RENDERER_DEFAULT_BG, // Black background for the space
        };
        // Fallback if single space is DrawTextRun (though FillRect is preferred by current renderer logic)
        let redraw_space_original_style_alt_step3 = MockDriverCall::DrawTextRun {
            coords: CellCoords { x: 1, y: 0 },
            text: " ".to_string(),
            style: TextRunStyle {
                fg: RENDERER_DEFAULT_FG,
                bg: RENDERER_DEFAULT_BG,
                flags: AttrFlags::empty(),
            },
        };
        assert!(
            mock_driver
                .calls
                .contains(&expected_redraw_space_original_style_step3)
                || mock_driver
                    .calls
                    .contains(&redraw_space_original_style_alt_step3),
            "Step 3 FAILURE: Original space at (1,0) not redrawn correctly after cursor moved away. Calls: {:?}",
            mock_driver.calls
        );

        // Verify cursor is drawn over 'T' at (0,0) correctly (Black text on White background)
        let expected_cursor_on_t_again_step3 = MockDriverCall::DrawTextRun {
            coords: CellCoords { x: 0, y: 0 },
            text: "T".to_string(),
            style: TextRunStyle {
                fg: RENDERER_DEFAULT_BG, // Black text
                bg: RENDERER_DEFAULT_FG, // White background
                flags: AttrFlags::empty(),
            },
        };
        assert!(
            mock_driver
                .calls
                .contains(&expected_cursor_on_t_again_step3),
            "Step 3 FAILURE: Cursor moving back onto 'T' did not draw it as Black on White. This would mean the text remains invisible if the 'white wake' persisted and wasn't painted over in step 2. Calls: {:?}",
            mock_driver.calls
        );

        assert!(
            mock_driver.calls.contains(&MockDriverCall::Present),
            "Present was not called at the end of Step 3."
        );
    }
}
