// src/term/selection.rs

use crate::glyph::Glyph; // Ensure Glyph is in scope if not already via screen::Row
use crate::term::screen::Row; // Import Row (Vec<Glyph>)
use crate::term::snapshot::{SelectionMode, SelectionRenderState};
use std::cmp::{max, min};

/// Represents the state of text selection in the terminal.
#[derive(Debug, Clone, PartialEq, Default)]
pub(super) struct Selection {
    /// Start point of the selection (col, row). Set when selection begins.
    pub(super) start: Option<(usize, usize)>,
    /// End point of the selection (col, row). Updated as mouse moves during selection.
    pub(super) end: Option<(usize, usize)>,
    /// Mode of selection (Normal or Block).
    pub(super) mode: SelectionMode,
    /// Whether a selection is currently active (e.g., mouse button is pressed).
    pub(super) is_active: bool,
}

impl Selection {
    /// Creates a new, empty selection.
    pub fn new() -> Self {
        Self::default()
    }

    /// Starts a new selection.
    ///
    /// # Arguments
    /// * `col`: The starting column of the selection.
    /// * `row`: The starting row of the selection.
    /// * `mode`: The `SelectionMode` (Normal or Block).
    pub fn start_selection(&mut self, col: usize, row: usize, mode: SelectionMode) {
        self.start = Some((col, row));
        self.end = Some((col, row)); // Initially, start and end are the same
        self.mode = mode;
        self.is_active = true;
        log::debug!("Selection started at ({}, {}) mode {:?}", col, row, mode);
    }

    /// Updates the end point of the active selection.
    ///
    /// # Arguments
    /// * `col`: The current column of the cursor during selection.
    /// * `row`: The current row of the cursor during selection.
    pub fn update_selection(&mut self, col: usize, row: usize) {
        if self.is_active {
            self.end = Some((col, row));
        }
    }

    /// Ends the current selection process.
    pub fn end_selection(&mut self) {
        if self.is_active {
            self.is_active = false;
            log::debug!(
                "Selection ended. Start: {:?}, End: {:?}",
                self.start,
                self.end
            );
        }
    }

    /// Clears the selection.
    pub fn clear_selection(&mut self) {
        self.start = None;
        self.end = None;
        self.is_active = false;
        log::debug!("Selection cleared.");
    }

    /// Gets the current selection state for rendering.
    pub fn get_render_state(
        &self,
        term_cols: usize,
        term_rows: usize,
    ) -> Option<SelectionRenderState> {
        if let (Some(start_abs), Some(end_abs)) = (self.start, self.end) {
            if term_cols == 0 || term_rows == 0 {
                return None;
            }

            let start_col = min(start_abs.0, term_cols - 1);
            let start_row = min(start_abs.1, term_rows - 1);
            let end_col = min(end_abs.0, term_cols - 1);
            let end_row = min(end_abs.1, term_rows - 1);

            let norm_start_row = min(start_row, end_row);
            let norm_end_row = max(start_row, end_row);

            let (norm_start_col, norm_end_col) = if self.mode == SelectionMode::Block {
                (min(start_col, end_col), max(start_col, end_col))
            } else {
                // Normal mode
                if norm_start_row == norm_end_row {
                    (min(start_col, end_col), max(start_col, end_col))
                } else if start_row < end_row {
                    (start_col, end_col)
                } else {
                    (end_col, start_col)
                }
            };

            Some(SelectionRenderState {
                start_coords: (norm_start_col, norm_start_row),
                end_coords: (norm_end_col, norm_end_row),
                mode: self.mode,
            })
        } else {
            None
        }
    }

    /// Extracts the selected text from the terminal grid.
    ///
    /// # Arguments
    /// * `grid_lines`: A slice of `Row` (Vec<Glyph>) representing the terminal screen buffer.
    /// * `term_cols`: Total number of columns in the terminal.
    /// * `term_rows`: Total number of rows in the terminal.
    ///
    /// # Returns
    /// A string containing the selected text.
    pub fn get_selected_text(
        &self,
        grid_lines: &[Row],
        term_cols: usize,
        term_rows: usize,
    ) -> String {
        let render_state = self.get_render_state(term_cols, term_rows);
        if render_state.is_none() {
            return String::new();
        }
        let state = render_state.unwrap();

        let mut result = String::new();
        let (start_col_norm, start_row_norm) = state.start_coords;
        let (end_col_norm, end_row_norm) = state.end_coords;

        if self.mode == SelectionMode::Block {
            for r in start_row_norm..=end_row_norm {
                if r >= grid_lines.len() {
                    break;
                }
                let line: &Row = &grid_lines[r];
                for c in start_col_norm..=end_col_norm {
                    if c >= line.len() {
                        break;
                    }
                    result.push(line[c].c); // Access Glyph's 'c' field
                }
                if r < end_row_norm {
                    result.push('\n'); // Corrected: character literal
                }
            }
        } else {
            // Normal selection
            for r in start_row_norm..=end_row_norm {
                if r >= grid_lines.len() {
                    break;
                }
                let line: &Row = &grid_lines[r];
                let line_start_col = if r == start_row_norm {
                    start_col_norm
                } else {
                    0
                };
                let line_end_col = if r == end_row_norm {
                    end_col_norm
                } else {
                    term_cols - 1
                };

                for c in line_start_col..=min(line_end_col, line.len().saturating_sub(1)) {
                    if c < line.len() {
                        // Ensure we don't go out of bounds for the current line
                        result.push(line[c].c); // Access Glyph's 'c' field
                    }
                }
                if r < end_row_norm {
                    result.push('\n'); // Corrected: character literal
                }
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::glyph::{Attributes, Glyph};
    use crate::term::snapshot::SnapshotLine; // Keep for test helper if needed, or adapt helper

    // Helper to create a Row (Vec<Glyph>) for text extraction tests
    fn create_row_from_str(text: &str) -> Row {
        let attrs = Attributes::default();
        text.chars().map(|c| Glyph { c, attr: attrs }).collect()
    }

    // Helper to create SnapshotLine (if still used by other tests, or adapt tests)
    #[allow(dead_code)] // It might not be used anymore if all tests adapt to Row
    fn create_snapshot_line(text: &str) -> SnapshotLine {
        let attrs = Attributes::default();
        SnapshotLine {
            is_dirty: false,
            cells: text.chars().map(|c| Glyph { c, attr: attrs }).collect(),
        }
    }

    #[test]
    fn test_selection_new_is_empty() {
        let selection = Selection::new();
        assert_eq!(selection.start, None);
        assert_eq!(selection.end, None);
        assert!(!selection.is_active);
        assert_eq!(selection.mode, SelectionMode::Normal);
    }

    #[test]
    fn test_selection_start_update_end() {
        let mut selection = Selection::new();
        selection.start_selection(1, 2, SelectionMode::Normal);
        assert_eq!(selection.start, Some((1, 2)));
        assert_eq!(selection.end, Some((1, 2)));
        assert!(selection.is_active);
        assert_eq!(selection.mode, SelectionMode::Normal);

        selection.update_selection(5, 2);
        assert_eq!(selection.end, Some((5, 2)));
        assert!(selection.is_active);

        selection.end_selection();
        assert!(!selection.is_active);
        assert_eq!(selection.start, Some((1, 2)));
        assert_eq!(selection.end, Some((5, 2)));
    }

    #[test]
    fn test_selection_clear() {
        let mut selection = Selection::new();
        selection.start_selection(1, 2, SelectionMode::Normal);
        selection.update_selection(5, 2);
        selection.clear_selection();
        assert_eq!(selection.start, None);
        assert_eq!(selection.end, None);
        assert!(!selection.is_active);
    }

    #[test]
    fn test_get_render_state_empty() {
        let selection = Selection::new();
        assert_eq!(selection.get_render_state(80, 24), None);
    }

    #[test]
    fn test_get_render_state_normal_single_line() {
        let mut selection = Selection::new();
        selection.start_selection(5, 10, SelectionMode::Normal);
        selection.update_selection(15, 10);
        selection.end_selection();
        let render_state = selection.get_render_state(80, 24);
        assert_eq!(
            render_state,
            Some(SelectionRenderState {
                start_coords: (5, 10),
                end_coords: (15, 10),
                mode: SelectionMode::Normal,
            })
        );
    }

    #[test]
    fn test_get_render_state_normal_multi_line_forward() {
        let mut selection = Selection::new();
        selection.start_selection(20, 5, SelectionMode::Normal);
        selection.update_selection(10, 7);
        selection.end_selection();
        let render_state = selection.get_render_state(80, 24);
        assert_eq!(
            render_state,
            Some(SelectionRenderState {
                start_coords: (20, 5),
                end_coords: (10, 7),
                mode: SelectionMode::Normal,
            })
        );
    }

    #[test]
    fn test_get_render_state_normal_multi_line_backward() {
        let mut selection = Selection::new();
        selection.start_selection(10, 7, SelectionMode::Normal);
        selection.update_selection(20, 5);
        selection.end_selection();
        let render_state = selection.get_render_state(80, 24);
        assert_eq!(
            render_state,
            Some(SelectionRenderState {
                start_coords: (20, 5),
                end_coords: (10, 7),
                mode: SelectionMode::Normal,
            })
        );
    }

    #[test]
    fn test_get_render_state_block() {
        let mut selection = Selection::new();
        selection.start_selection(10, 5, SelectionMode::Block);
        selection.update_selection(20, 15);
        selection.end_selection();
        let render_state = selection.get_render_state(80, 24);
        assert_eq!(
            render_state,
            Some(SelectionRenderState {
                start_coords: (10, 5),
                end_coords: (20, 15),
                mode: SelectionMode::Block,
            })
        );

        selection.start_selection(20, 15, SelectionMode::Block);
        selection.update_selection(10, 5);
        selection.end_selection();
        let render_state_backward = selection.get_render_state(80, 24);
        assert_eq!(
            render_state_backward,
            Some(SelectionRenderState {
                start_coords: (10, 5),
                end_coords: (20, 15),
                mode: SelectionMode::Block,
            })
        );
    }

    #[test]
    fn test_get_selected_text_normal_single_line() {
        let mut selection = Selection::new();
        selection.start_selection(1, 0, SelectionMode::Normal);
        selection.update_selection(3, 0);
        selection.end_selection();

        let grid_lines = vec![create_row_from_str("Hello")];
        let selected_text = selection.get_selected_text(&grid_lines, 5, 1);
        assert_eq!(selected_text, "ell");
    }

    #[test]
    fn test_get_selected_text_normal_multi_line() {
        let mut selection = Selection::new();
        selection.start_selection(2, 0, SelectionMode::Normal);
        selection.update_selection(2, 2);
        selection.end_selection();

        let grid_lines = vec![
            create_row_from_str("Hello"),
            create_row_from_str("World"),
            create_row_from_str("There"),
        ];
        let selected_text = selection.get_selected_text(&grid_lines, 5, 3);
        assert_eq!(selected_text, "llo\nWorld\nTh");
    }

    #[test]
    fn test_get_selected_text_normal_multi_line_backward_selection() {
        let mut selection = Selection::new();
        selection.start_selection(2, 2, SelectionMode::Normal);
        selection.update_selection(2, 0);
        selection.end_selection();

        let grid_lines = vec![
            create_row_from_str("Hello"),
            create_row_from_str("World"),
            create_row_from_str("There"),
        ];
        let selected_text = selection.get_selected_text(&grid_lines, 5, 3);
        assert_eq!(selected_text, "llo\nWorld\nTh");
    }

    #[test]
    fn test_get_selected_text_block() {
        let mut selection = Selection::new();
        selection.start_selection(1, 0, SelectionMode::Block);
        selection.update_selection(3, 1);
        selection.end_selection();

        let grid_lines = vec![
            create_row_from_str("ABCDE"),
            create_row_from_str("FGHIJ"),
            create_row_from_str("KLMNO"),
        ];
        let selected_text = selection.get_selected_text(&grid_lines, 5, 3);
        assert_eq!(selected_text, "BCD\nGHI");
    }

    #[test]
    fn test_get_selected_text_block_backward() {
        let mut selection = Selection::new();
        selection.start_selection(3, 1, SelectionMode::Block);
        selection.update_selection(1, 0);
        selection.end_selection();

        let grid_lines = vec![
            create_row_from_str("ABCDE"),
            create_row_from_str("FGHIJ"),
            create_row_from_str("KLMNO"),
        ];
        let selected_text = selection.get_selected_text(&grid_lines, 5, 3);
        assert_eq!(selected_text, "BCD\nGHI");
    }

    #[test]
    fn test_get_render_state_out_of_bounds_clamp() {
        let mut selection = Selection::new();
        selection.start_selection(70, 20, SelectionMode::Normal);
        selection.update_selection(90, 30);
        selection.end_selection();

        let render_state = selection.get_render_state(80, 24);
        assert_eq!(
            render_state,
            Some(SelectionRenderState {
                start_coords: (70, 20),
                end_coords: (79, 23),
                mode: SelectionMode::Normal,
            })
        );
    }

    #[test]
    fn test_get_selected_text_empty_if_render_state_none() {
        let selection = Selection::new();
        let grid_lines = vec![create_row_from_str("Hello")];
        assert_eq!(selection.get_selected_text(&grid_lines, 5, 1), "");

        let mut sel_active_no_points = Selection::new();
        sel_active_no_points.is_active = true;
        assert_eq!(
            sel_active_no_points.get_selected_text(&grid_lines, 5, 1),
            ""
        );
    }

    #[test]
    fn test_get_selected_text_normal_selection_ends_beyond_line_length() {
        let mut selection = Selection::new();
        selection.start_selection(2, 0, SelectionMode::Normal);
        selection.update_selection(10, 0);
        selection.end_selection();

        let grid_lines = vec![create_row_from_str("Hello")];
        let selected_text = selection.get_selected_text(&grid_lines, 80, 1);
        assert_eq!(selected_text, "llo");
    }

    #[test]
    fn test_get_selected_text_block_selection_ends_beyond_line_length() {
        let mut selection = Selection::new();
        selection.start_selection(1, 0, SelectionMode::Block);
        selection.update_selection(8, 1);
        selection.end_selection();

        let grid_lines = vec![create_row_from_str("Short"), create_row_from_str("LineTwo")];
        let selected_text = selection.get_selected_text(&grid_lines, 80, 2);
        assert_eq!(selected_text, "hort\nineTwo");
    }
}
