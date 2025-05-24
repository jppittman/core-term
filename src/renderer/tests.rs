// src/renderer/tests.rs

#[cfg(test)]
mod render_tests {
    use crate::backends::{BackendEvent, CellCoords, CellRect, Driver, TextRunStyle};
    use crate::color::{Color, NamedColor};
    use crate::glyph::{AttrFlags, Attributes, Glyph};
    use crate::renderer::*; 
    use crate::term::{
        CursorRenderState, EmulatorAction, EmulatorInput, RenderSnapshot, SelectionMode,
        SelectionRenderState, SnapshotLine, TerminalInterface, DEFAULT_CURSOR_SHAPE,
    }; // Added SelectionRenderState, RenderSnapshot, etc.
    use anyhow::Result;
    use std::collections::{HashSet, VecDeque}; 
    use std::os::unix::io::RawFd;
    use test_log::test; 

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
        current_default_attributes: Attributes,
        actions_to_return_on_next_call: VecDeque<Option<EmulatorAction>>,
        inputs_received: Vec<EmulatorInput>,
        selection_state: Option<SelectionRenderState>, // Added field
    }

    impl MockTerminal {
        fn new(width: usize, height: usize) -> Self {
            let default_attributes = Attributes {
                fg: RENDERER_DEFAULT_FG,
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
                actions_to_return_on_next_call: VecDeque::new(),
                inputs_received: Vec::new(),
                selection_state: None, // Initialize
            }
        }

        fn set_glyph(&mut self, x: usize, y: usize, glyph: Glyph) {
            if y < self.height && x < self.width {
                self.grid[y][x] = glyph;
            }
        }

        fn set_cursor_screen_pos(&mut self, x: usize, y: usize) {
            self.cursor_x = x;
            self.cursor_y = y;
        }

        #[allow(dead_code)]
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

        #[allow(dead_code)]
        fn expect_action_for_next_call(&mut self, action: Option<EmulatorAction>) {
            self.actions_to_return_on_next_call.push_back(action);
        }

        // Added method
        fn set_selection(&mut self, selection: Option<SelectionRenderState>) {
            self.selection_state = selection;
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
        fn interpret_input(&mut self, input: EmulatorInput) -> Option<EmulatorAction> {
            self.inputs_received.push(input); 
            self.actions_to_return_on_next_call
                .pop_front()
                .unwrap_or(None)
        }

        // Added method
        fn get_render_snapshot(&self) -> RenderSnapshot {
            let (width, height) = self.dimensions();
            let mut lines = Vec::with_capacity(height);
            for y_idx in 0..height {
                lines.push(SnapshotLine {
                    cells: self.grid[y_idx].clone(),
                    is_dirty: self.dirty_lines.contains(&y_idx),
                });
            }
            RenderSnapshot {
                dimensions: (width, height),
                lines,
                cursor_state: if self.cursor_visible {
                    Some(CursorRenderState {
                        col: self.cursor_x,
                        row: self.cursor_y,
                        shape: DEFAULT_CURSOR_SHAPE, 
                        is_visible: true,
                    })
                } else {
                    None
                },
                selection_state: self.selection_state,
            }
        }
    }

    // --- MockDriver Definition ---
    #[derive(Debug, Clone, PartialEq)]
    enum MockDriverCall {
        ClearAll {
            bg: Color,
        },
        DrawTextRun {
            coords: CellCoords,
            text: String,
            style: TextRunStyle,
            is_selected: bool, // Updated
        },
        FillRect {
            rect: CellRect,
            color: Color,
            is_selected: bool, // Updated
        },
        Present,
        SetTitle {
            title: String,
        },
        Bell,
        SetCursorVisibility {
            visible: bool,
        },
        SetFocus {
            focused: bool,
        },
        Cleanup, 
    }

    struct MockDriver {
        calls: Vec<MockDriverCall>,
        focus_state: bool,
    }

    impl MockDriver {
        fn new() -> Self {
            MockDriver {
                calls: Vec::new(),
                focus_state: true, 
            }
        }

        #[allow(dead_code)]
        fn has_call(&self, expected_call: &MockDriverCall) -> bool {
            self.calls.contains(expected_call)
        }

        #[allow(dead_code)]
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
        // Updated signature and implementation
        fn draw_text_run(
            &mut self,
            coords: CellCoords,
            text: &str,
            style: TextRunStyle,
            is_selected: bool,
        ) -> Result<()> {
            self.calls.push(MockDriverCall::DrawTextRun {
                coords,
                text: text.to_string(),
                style,
                is_selected,
            });
            Ok(())
        }
        // Updated signature and implementation
        fn fill_rect(&mut self, rect: CellRect, color: Color, is_selected: bool) -> Result<()> {
            self.calls.push(MockDriverCall::FillRect { rect, color, is_selected });
            Ok(())
        }
        fn present(&mut self) -> Result<()> {
            self.calls.push(MockDriverCall::Present);
            Ok(())
        }
        fn cleanup(&mut self) -> Result<()> {
            self.calls.push(MockDriverCall::Cleanup);
            Ok(())
        }
        fn set_title(&mut self, title: &str) {
            self.calls.push(MockDriverCall::SetTitle {
                title: title.to_string(),
            });
        }
        fn bell(&mut self) {
            self.calls.push(MockDriverCall::Bell);
        }
        fn set_cursor_visibility(&mut self, visible: bool) {
            self.calls
                .push(MockDriverCall::SetCursorVisibility { visible });
        }
        fn set_focus(&mut self, focused: bool) {
            self.focus_state = focused; 
            self.calls.push(MockDriverCall::SetFocus { focused });
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
            "No draw calls should be made for 0x0 terminal. Calls: {:?}",
            mock_driver.calls
        );
    }

    #[test]
    fn test_initial_draw_clears_all_with_default_bg() {
        let mut renderer = Renderer::new();
        let mut mock_term = MockTerminal::new(2, 1);
        let mut mock_driver = MockDriver::new();

        renderer.draw(&mut mock_term, &mut mock_driver).unwrap();

        assert!(
            mock_driver.calls.contains(&MockDriverCall::ClearAll {
                bg: RENDERER_DEFAULT_BG
            }),
            "Expected ClearAll with renderer's default background. Calls: {:?}",
            mock_driver.calls
        );

        let expected_combined_space_fill = MockDriverCall::FillRect {
            rect: CellRect {
                x: 0,
                y: 0,
                width: 2,
                height: 1,
            },
            color: RENDERER_DEFAULT_BG,
            is_selected: false, // Spaces are not selected by default
        };
        assert!(
            mock_driver.calls.contains(&expected_combined_space_fill),
            "Cells (0,0) and (1,0) (spaces) should be drawn by a combined FillRect. Calls: {:?}",
            mock_driver.calls
        );

        let expected_cursor_draw = MockDriverCall::DrawTextRun {
            coords: CellCoords { x: 0, y: 0 },
            text: " ".to_string(),
            style: TextRunStyle {
                fg: RENDERER_DEFAULT_BG, // Cursor inverts
                bg: RENDERER_DEFAULT_FG,
                flags: AttrFlags::empty(),
            },
            is_selected: false, // Cursor cell itself is not "selected" for this call's purpose
        };
        assert!(
            mock_driver.calls.contains(&expected_cursor_draw),
            "Cursor draw not found or incorrect. Calls: {:?}",
            mock_driver.calls
        );

        assert!(
            mock_driver.calls.contains(&MockDriverCall::Present),
            "Present was not called. Calls: {:?}",
            mock_driver.calls
        );
        assert!(
            !renderer.first_draw,
            "first_draw flag should be false after the initial draw."
        );
    }

    #[test]
    fn test_draw_single_default_char_no_clearall() {
        let mut renderer = Renderer::new();
        renderer.first_draw = false;

        let mut mock_term = MockTerminal::new(1, 1);
        mock_term.clear_all_dirty_lines();
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
        mock_term.set_cursor_screen_pos(0, 0);

        renderer.draw(&mut mock_term, &mut mock_driver).unwrap();

        assert!(
            !mock_driver
                .calls
                .iter()
                .any(|call| matches!(call, MockDriverCall::ClearAll { .. })),
            "ClearAll should NOT be called for a single dirty line when not the first_draw. Calls: {:?}",
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
            is_selected: false,
        };
        assert!(
            mock_driver.calls.contains(&expected_text_run_a),
            "Expected text run for 'A' with default style not found. Calls: {:?}",
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
            is_selected: false,
        };
        assert!(
            mock_driver.calls.contains(&expected_cursor_draw_over_a),
            "Expected cursor draw for 'A' not found. Calls: {:?}",
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
            is_selected: false,
        };
        let expected_cursor_draw = MockDriverCall::DrawTextRun {
            coords: CellCoords { x: 0, y: 0 },
            text: "R".to_string(),
            style: TextRunStyle {
                fg: bg_blue, // Inverted for cursor
                bg: fg_red,  // Inverted for cursor
                flags: AttrFlags::empty(),
            },
            is_selected: false,
        };

        assert!(
            mock_driver.calls.contains(&expected_cell_draw),
            "Specific color text run not found. Calls: {:?}",
            mock_driver.calls
        );
        assert!(
            mock_driver.calls.contains(&expected_cursor_draw),
            "Cursor over specific color text run not found. Calls: {:?}",
            mock_driver.calls
        );
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
                fg: RENDERER_DEFAULT_BG, // Swapped due to REVERSE
                bg: RENDERER_DEFAULT_FG,
                flags: AttrFlags::empty(), // REVERSE flag consumed by renderer
            },
            is_selected: false,
        };
        let expected_cursor_draw = MockDriverCall::DrawTextRun {
            coords: CellCoords { x: 0, y: 0 },
            text: "X".to_string(),
            style: TextRunStyle {
                fg: RENDERER_DEFAULT_FG, // Inverted from cell's effective bg
                bg: RENDERER_DEFAULT_BG, // Inverted from cell's effective fg
                flags: AttrFlags::empty(),
            },
            is_selected: false,
        };
        assert!(
            mock_driver.calls.contains(&expected_cell_draw),
            "Reversed video text run for cell not found. Calls: {:?}",
            mock_driver.calls
        );
        assert!(
            mock_driver.calls.contains(&expected_cursor_draw),
            "Cursor over reversed video text run not found. Calls: {:?}",
            mock_driver.calls
        );
    }

    #[test]
    fn test_draw_only_dirty_lines_and_cursor_line() {
        let mut renderer = Renderer::new();
        renderer.first_draw = false;
        let mut mock_term = MockTerminal::new(2, 2);
        mock_term.clear_all_dirty_lines();
        let mut mock_driver = MockDriver::new();

        mock_term.set_glyph(0,0, Glyph { c: 'A', attr: Attributes::default() });
        mock_term.set_glyph(1,0, Glyph { c: ' ', attr: Attributes::default() });
        mock_term.set_glyph(0,1, Glyph { c: 'B', attr: Attributes::default() });
        mock_term.set_glyph(1,1, Glyph { c: ' ', attr: Attributes::default() });

        mock_term.mark_line_dirty(1); // Line 1 is dirty
        mock_term.set_cursor_screen_pos(0, 0); // Cursor on line 0

        renderer.draw(&mut mock_term, &mut mock_driver).unwrap();

        let mut drawn_content_a_on_line_0 = false;
        let mut drawn_content_b_on_line_1 = false;
        let mut cursor_drawn_on_line_0 = false;

        for call in &mock_driver.calls {
            if let MockDriverCall::DrawTextRun { coords, text, style, is_selected } = call {
                if coords.y == 0 && text == "A" && style.fg == RENDERER_DEFAULT_FG && !*is_selected {
                    drawn_content_a_on_line_0 = true;
                } else if coords.y == 0 && text == "A" && style.fg == RENDERER_DEFAULT_BG && !*is_selected {
                    cursor_drawn_on_line_0 = true;
                } else if coords.y == 1 && text == "B" && style.fg == RENDERER_DEFAULT_FG && !*is_selected {
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
        mock_term.set_glyph(1,0, Glyph { c: '\0', attr: wide_char_attrs }); // Placeholder
        mock_term.set_glyph(2,0, Glyph { c: 'N', attr: Attributes::default() });
        mock_term.mark_line_dirty(0);
        mock_term.set_cursor_screen_pos(0, 0); // Cursor on the wide char

        renderer.draw(&mut mock_term, &mut mock_driver).unwrap();

        let mut wide_char_text_drawn = false;
        let mut wide_char_cursor_drawn = false;
        let mut placeholder_filled = false;
        let mut normal_char_drawn = false;

        for call in &mock_driver.calls {
            match call {
                MockDriverCall::DrawTextRun { coords, text, style, is_selected } => {
                    if *coords == (CellCoords { x: 0, y: 0 }) && text == &wide_char.to_string() {
                        if style.fg == Color::Named(NamedColor::Green) && style.bg == RENDERER_DEFAULT_BG && !*is_selected {
                            wide_char_text_drawn = true;
                        } else if style.fg == RENDERER_DEFAULT_BG && style.bg == Color::Named(NamedColor::Green) && !*is_selected {
                            wide_char_cursor_drawn = true;
                        }
                    } else if *coords == (CellCoords { x: 2, y: 0 }) && text == "N" && style.fg == RENDERER_DEFAULT_FG && style.bg == RENDERER_DEFAULT_BG && !*is_selected {
                        normal_char_drawn = true;
                    }
                }
                MockDriverCall::FillRect { rect, color, is_selected } => {
                    if *rect == (CellRect { x: 1, y: 0, width: 1, height: 1 }) && *color == RENDERER_DEFAULT_BG && !*is_selected {
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
    fn test_draw_selected_text_passes_is_selected_true_to_driver() {
        let mut renderer = Renderer::new();
        renderer.first_draw = false;
        let mut mock_term = MockTerminal::new(5, 1);
        mock_term.clear_all_dirty_lines();
        let mut mock_driver = MockDriver::new();

        mock_term.set_glyph(0,0, Glyph { c: 'H', attr: Attributes::default() });
        mock_term.set_glyph(1,0, Glyph { c: 'e', attr: Attributes::default() });
        mock_term.set_glyph(2,0, Glyph { c: 'l', attr: Attributes::default() });
        mock_term.set_glyph(3,0, Glyph { c: 'l', attr: Attributes::default() });
        mock_term.set_glyph(4,0, Glyph { c: 'o', attr: Attributes::default() });
        
        mock_term.set_selection(Some(SelectionRenderState {
            start_coords: (1,0), end_coords: (3,0), mode: SelectionMode::Normal
        }));
        mock_term.mark_line_dirty(0);
        mock_term.set_cursor_screen_pos(0,0); // Cursor not in selection

        renderer.draw(&mut mock_term, &mut mock_driver).unwrap();

        let mut h_drawn_unselected = false;
        let mut ell_drawn_selected = false;
        let mut o_drawn_unselected = false;

        for call in &mock_driver.calls {
            if let MockDriverCall::DrawTextRun { text, is_selected, coords, .. } = call {
                if coords.x == 0 && text == "H" { h_drawn_unselected = !*is_selected; }
                if coords.x == 1 && text == "ell" { ell_drawn_selected = *is_selected; }
                if coords.x == 4 && text == "o" { o_drawn_unselected = !*is_selected; }
            }
        }
        assert!(h_drawn_unselected, "H should be drawn unselected. Calls: {:?}", mock_driver.calls);
        assert!(ell_drawn_selected, "'ell' should be drawn selected. Calls: {:?}", mock_driver.calls);
        assert!(o_drawn_unselected, "'o' should be drawn unselected. Calls: {:?}", mock_driver.calls);
    }

    #[test]
    fn test_draw_selected_spaces_passes_is_selected_true_to_driver_fill_rect() {
        let mut renderer = Renderer::new();
        renderer.first_draw = false;
        let mut mock_term = MockTerminal::new(5, 1); // "A B C"
        mock_term.clear_all_dirty_lines();
        let mut mock_driver = MockDriver::new();

        mock_term.set_glyph(0,0, Glyph { c: 'A', attr: Attributes::default() });
        mock_term.set_glyph(1,0, Glyph { c: ' ', attr: Attributes::default() });
        mock_term.set_glyph(2,0, Glyph { c: 'B', attr: Attributes::default() });
        mock_term.set_glyph(3,0, Glyph { c: ' ', attr: Attributes::default() });
        mock_term.set_glyph(4,0, Glyph { c: 'C', attr: Attributes::default() });

        mock_term.set_selection(Some(SelectionRenderState {
            start_coords: (1,0), end_coords: (1,0), mode: SelectionMode::Normal // Select space at (1,0)
        }));
        mock_term.mark_line_dirty(0);
        mock_term.set_cursor_screen_pos(0,0); // Cursor on A

        renderer.draw(&mut mock_term, &mut mock_driver).unwrap();
        
        let mut space_at_1_selected = false;
        let mut space_at_3_unselected = false;

        for call in &mock_driver.calls {
            if let MockDriverCall::FillRect { rect, is_selected, .. } = call {
                if rect.x == 1 && rect.width == 1 { space_at_1_selected = *is_selected; }
                if rect.x == 3 && rect.width == 1 { space_at_3_unselected = !*is_selected; }
            }
        }
        assert!(space_at_1_selected, "Space at (1,0) should be filled as selected. Calls: {:?}", mock_driver.calls);
        assert!(space_at_3_unselected, "Space at (3,0) should be filled as unselected. Calls: {:?}", mock_driver.calls);
    }

    #[test]
    fn test_cursor_over_selected_text_uses_correct_selection_flags_for_driver() {
        let mut renderer = Renderer::new();
        renderer.first_draw = false;
        let mut mock_term = MockTerminal::new(3, 1); // "XYZ"
        mock_term.clear_all_dirty_lines();
        let mut mock_driver = MockDriver::new();

        mock_term.set_glyph(0,0, Glyph { c: 'X', attr: Attributes::default() });
        mock_term.set_glyph(1,0, Glyph { c: 'Y', attr: Attributes::default() });
        mock_term.set_glyph(2,0, Glyph { c: 'Z', attr: Attributes::default() });
        
        mock_term.set_selection(Some(SelectionRenderState {
            start_coords: (0,0), end_coords: (2,0), mode: SelectionMode::Normal // Select "XYZ"
        }));
        mock_term.mark_line_dirty(0);
        mock_term.set_cursor_screen_pos(1,0); // Cursor on 'Y'

        renderer.draw(&mut mock_term, &mut mock_driver).unwrap();

        let mut x_content_selected = false;
        let mut z_content_selected = false;
        let mut y_content_selected_under_cursor = false; // The cell content draw for 'Y'
        let mut y_cursor_itself_not_selected_flag = false; // The cursor overlay draw for 'Y'

        for call in &mock_driver.calls {
            if let MockDriverCall::DrawTextRun { coords, text, style, is_selected } = call {
                if coords.x == 0 && text == "X" { x_content_selected = *is_selected; }
                if coords.x == 2 && text == "Z" { z_content_selected = *is_selected; }
                if coords.x == 1 && text == "Y" {
                    // This is tricky: renderer draws content first, then cursor.
                    // Content 'Y' should be drawn selected.
                    // Cursor 'Y' should be drawn with is_selected=false (as its colors are pre-inverted).
                    if style.fg == RENDERER_DEFAULT_FG && style.bg == RENDERER_DEFAULT_BG { // Content style for selected
                        y_content_selected_under_cursor = *is_selected;
                    } else if style.fg == RENDERER_DEFAULT_BG && style.bg == RENDERER_DEFAULT_FG { // Cursor style
                        y_cursor_itself_not_selected_flag = !*is_selected;
                    }
                }
            }
        }
        assert!(x_content_selected, "Content 'X' should be drawn as selected. Calls: {:?}", mock_driver.calls);
        assert!(z_content_selected, "Content 'Z' should be drawn as selected. Calls: {:?}", mock_driver.calls);
        assert!(y_content_selected_under_cursor, "Content 'Y' (under cursor) should be drawn as selected. Calls: {:?}", mock_driver.calls);
        assert!(y_cursor_itself_not_selected_flag, "Cursor overlay for 'Y' should have is_selected=false. Calls: {:?}", mock_driver.calls);
    }
}
