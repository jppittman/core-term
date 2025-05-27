// src/term/selection.rs

use crate::term::snapshot::{SelectionMode, SelectionRenderState, SnapshotLine};
use std::cmp::{min, max}; // For ordering coordinates

/// Represents the state of text selection in the terminal.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Selection {
    /// Start point of the selection (col, row). Set when selection begins.
    pub start: Option<(usize, usize)>,
    /// End point of the selection (col, row). Updated as mouse moves during selection.
    pub end: Option<(usize, usize)>,
    /// Mode of selection (Normal or Block).
    pub mode: SelectionMode,
    /// Whether a selection is currently active (e.g., mouse button is pressed).
    pub is_active: bool,
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
            // log::trace!("Selection updated to ({}, {})", col, row); // Can be too verbose
        }
    }

    /// Ends the current selection process.
    /// The selection remains (start and end points are kept) but is no longer "active"
    /// (i.e., mouse button released).
    pub fn end_selection(&mut self) {
        if self.is_active {
            self.is_active = false;
            log::debug!("Selection ended. Start: {:?}, End: {:?}", self.start, self.end);
        }
    }

    /// Clears the selection, resetting all fields to default.
    pub fn clear_selection(&mut self) {
        self.start = None;
        self.end = None;
        self.is_active = false;
        // self.mode remains as is, or could be reset to a default like Normal
        log::debug!("Selection cleared.");
    }

    /// Gets the current selection state for rendering.
    ///
    /// Returns `None` if no selection is active or defined.
    /// Otherwise, returns a `SelectionRenderState` with normalized coordinates
    /// (start_coords is top-left, end_coords is bottom-right).
    ///
    /// # Arguments
    /// * `term_cols`: Total number of columns in the terminal.
    /// * `term_rows`: Total number of rows in the terminal.
    pub fn get_render_state(&self, term_cols: usize, term_rows: usize) -> Option<SelectionRenderState> {
        if let (Some(start_abs), Some(end_abs)) = (self.start, self.end) {
            if term_cols == 0 || term_rows == 0 {
                return None;
            }

            // Clamp coordinates to be within terminal boundaries.
            // This is important if selection was made and then terminal resized smaller.
            let start_col = min(start_abs.0, term_cols - 1);
            let start_row = min(start_abs.1, term_rows - 1);
            let end_col = min(end_abs.0, term_cols - 1);
            let end_row = min(end_abs.1, term_rows - 1);

            // Normalize coordinates: start_coords should be top-left, end_coords bottom-right.
            let norm_start_row = min(start_row, end_row);
            let norm_end_row = max(start_row, end_row);

            let (norm_start_col, norm_end_col) = if norm_start_row == norm_end_row {
                // Single line selection
                (min(start_col, end_col), max(start_col, end_col))
            } else if start_row < end_row {
                // Multi-line, forward selection (start is above end)
                (start_col, end_col)
            } else {
                // Multi-line, backward selection (end is above start)
                (end_col, start_col)
            };
            
            // For block selection, columns are simply min/max of the selection rectangle
            if self.mode == SelectionMode::Block {
                 let block_start_col = min(start_col, end_col);
                 let block_end_col = max(start_col, end_col);
                 return Some(SelectionRenderState {
                    start_coords: (block_start_col, norm_start_row),
                    end_coords: (block_end_col, norm_end_row),
                    mode: self.mode,
                });
            }


            Some(SelectionRenderState {
                start_coords: (norm_start_col, norm_start_row),
                end_coords: (norm_end_col, norm_end_row),
                mode: self.mode,
            })
        } else {
            None // No selection active
        }
    }

    /// Extracts the selected text from the terminal lines.
    /// Placeholder implementation: This will be complex and depends on how `SnapshotLine` and `Glyph` are structured.
    ///
    /// # Arguments
    /// * `lines`: A slice of `SnapshotLine` representing the terminal screen buffer.
    /// * `term_cols`: Total number of columns in the terminal.
    /// * `term_rows`: Total number of rows in the terminal.
    ///
    /// # Returns
    /// A string containing the selected text.
    pub fn get_selected_text(&self, lines: &[SnapshotLine], term_cols: usize, term_rows: usize) -> String {
        let render_state = self.get_render_state(term_cols, term_rows);
        if render_state.is_none() {
            return String::new();
        }
        let state = render_state.unwrap();

        let mut result = String::new();
        let (start_col, mut current_row) = state.start_coords;
        let (end_col, end_row) = state.end_coords;

        if current_row > end_row || (current_row == end_row && start_col > end_col && self.mode != SelectionMode::Block) {
             // Should not happen with normalized coordinates from get_render_state for Normal mode
            return String::new();
        }
        
        if current_row >= lines.len() {
            return String::new(); // Out of bounds
        }

        if self.mode == SelectionMode::Block {
            for r in current_row..=end_row {
                if r >= lines.len() { break; }
                let line = &lines[r];
                let line_start_col = state.start_coords.0; // Block selection uses fixed start_col for all lines
                let line_end_col = state.end_coords.0;   // Block selection uses fixed end_col for all lines

                for c in line_start_col..=line_end_col {
                    if c >= line.cells.len() { break; } // Past end of line
                    result.push(line.cells[c].char);
                }
                if r < end_row { // Add newline for all but the last line of the block
                    result.push('
');
                }
            }
        } else { // Normal selection
            // First line
            let line = &lines[current_row];
            let first_line_end_col = if current_row == end_row { end_col } else { term_cols - 1 };
            for c in start_col..=min(first_line_end_col, line.cells.len().saturating_sub(1)) {
                result.push(line.cells[c].char);
            }

            // Middle lines (if any)
            current_row += 1;
            while current_row < end_row {
                if current_row >= lines.len() { break; }
                result.push('
');
                let line = &lines[current_row];
                for c in 0..min(term_cols, line.cells.len()) {
                    result.push(line.cells[c].char);
                }
                current_row += 1;
            }

            // Last line (if different from first)
            if current_row == end_row && current_row > state.start_coords.1 {
                 if current_row >= lines.len() { /* out of bounds */ } else {
                    result.push('
');
                    let line = &lines[current_row];
                    for c in 0..=min(end_col, line.cells.len().saturating_sub(1)) {
                        result.push(line.cells[c].char);
                    }
                }
            }
            // TODO: Handle trailing spaces / full width characters appropriately.
            // This basic implementation might strip trailing spaces if not careful with line iteration.
            // For now, focus on character extraction.
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::glyph::{Attributes, Glyph}; // For text extraction tests
                                           // SnapshotLine and SelectionRenderState are already imported at the top level of selection.rs
                                           // SelectionMode is also available.

    #[test]
    fn test_selection_new_is_empty() {
        let selection = Selection::new();
        assert_eq!(selection.start, None);
        assert_eq!(selection.end, None);
        assert!(!selection.is_active);
        assert_eq!(selection.mode, SelectionMode::Normal); // Default mode is Normal
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
        assert!(selection.is_active); // Still active

        selection.end_selection();
        assert!(!selection.is_active); // No longer active
        assert_eq!(selection.start, Some((1, 2))); // Start remains
        assert_eq!(selection.end, Some((5, 2))); // End remains
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
                start_coords: (20, 5), // (start_col_for_start_row, start_row)
                end_coords: (10, 7),   // (end_col_for_end_row, end_row)
                mode: SelectionMode::Normal,
            })
        );
    }

    #[test]
    fn test_get_render_state_normal_multi_line_backward() {
        let mut selection = Selection::new();
        selection.start_selection(10, 7, SelectionMode::Normal); // Start at (10,7)
        selection.update_selection(20, 5); // End at (20,5)
        selection.end_selection();
        let render_state = selection.get_render_state(80, 24);
        // Normalization:
        // start_row = min(7,5) = 5
        // end_row = max(7,5) = 7
        // Since original start_row (7) > original end_row (5), it's a backward selection.
        // norm_start_col becomes end_col_abs (20)
        // norm_end_col becomes start_col_abs (10)
        assert_eq!(
            render_state,
            Some(SelectionRenderState {
                start_coords: (20, 5), // (col_for_norm_start_row, norm_start_row)
                end_coords: (10, 7),   // (col_for_norm_end_row, norm_end_row)
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
                start_coords: (10, 5), // min_col, min_row
                end_coords: (20, 15),  // max_col, max_row
                mode: SelectionMode::Block,
            })
        );

        // Test backward block selection
        selection.start_selection(20, 15, SelectionMode::Block);
        selection.update_selection(10, 5);
        selection.end_selection();
        let render_state_backward = selection.get_render_state(80, 24);
        assert_eq!(
            render_state_backward,
            Some(SelectionRenderState {
                start_coords: (10, 5), // min_col, min_row
                end_coords: (20, 15),  // max_col, max_row
                mode: SelectionMode::Block,
            })
        );
    }

    fn create_snapshot_line(text: &str) -> SnapshotLine {
        let attrs = Attributes::default();
        SnapshotLine {
            is_dirty: false,
            cells: text.chars().map(|c| Glyph::new(c, attrs)).collect(),
        }
    }

    #[test]
    fn test_get_selected_text_normal_single_line() {
        let mut selection = Selection::new();
        selection.start_selection(1, 0, SelectionMode::Normal);
        selection.update_selection(3, 0);
        selection.end_selection();

        let lines = vec![create_snapshot_line("Hello")];
        let selected_text = selection.get_selected_text(&lines, 5, 1);
        assert_eq!(selected_text, "ell");
    }

    #[test]
    fn test_get_selected_text_normal_multi_line() {
        let mut selection = Selection::new();
        selection.start_selection(2, 0, SelectionMode::Normal); // "llo" from "Hello"
        selection.update_selection(2, 2); // "Th" from "There"
        selection.end_selection();

        let lines = vec![
            create_snapshot_line("Hello"), // Line 0
            create_snapshot_line("World"), // Line 1
            create_snapshot_line("There"), // Line 2
        ];
        let selected_text = selection.get_selected_text(&lines, 5, 3);
        assert_eq!(selected_text, "llo\nWorld\nTh");
    }
    
    #[test]
    fn test_get_selected_text_normal_multi_line_backward_selection() {
        let mut selection = Selection::new();
        selection.start_selection(2, 2, SelectionMode::Normal); // Start at "Th|ere" on line 2
        selection.update_selection(2, 0);                   // End at "He|llo" on line 0
        selection.end_selection();

        let lines = vec![
            create_snapshot_line("Hello"), // Line 0
            create_snapshot_line("World"), // Line 1
            create_snapshot_line("There"), // Line 2
        ];
        // Expected: "llo" from line 0, "World" from line 1, "Th" from line 2
        // The get_render_state normalizes coordinates, so text extraction should be the same
        // as forward selection between these points.
        let selected_text = selection.get_selected_text(&lines, 5, 3);
        assert_eq!(selected_text, "llo\nWorld\nTh");
    }


    #[test]
    fn test_get_selected_text_block() {
        let mut selection = Selection::new();
        selection.start_selection(1, 0, SelectionMode::Block); // Column B, Line 0
        selection.update_selection(3, 1); // Column D, Line 1
        selection.end_selection();

        let lines = vec![
            create_snapshot_line("ABCDE"), // Line 0
            create_snapshot_line("FGHIJ"), // Line 1
            create_snapshot_line("KLMNO"), // Line 2
        ];
        // Expected:
        // Line 0: BCD (cols 1,2,3)
        // Line 1: GHI (cols 1,2,3)
        let selected_text = selection.get_selected_text(&lines, 5, 3);
        assert_eq!(selected_text, "BCD\nGHI");
    }

    #[test]
    fn test_get_selected_text_block_backward() {
        let mut selection = Selection::new();
        selection.start_selection(3, 1, SelectionMode::Block); // Start at D1
        selection.update_selection(1, 0);                   // End at B0
        selection.end_selection();

        let lines = vec![
            create_snapshot_line("ABCDE"),
            create_snapshot_line("FGHIJ"),
            create_snapshot_line("KLMNO"),
        ];
        // Render state normalizes to (1,0) and (3,1) for block.
        // Expected:
        // Line 0: BCD
        // Line 1: GHI
        let selected_text = selection.get_selected_text(&lines, 5, 3);
        assert_eq!(selected_text, "BCD\nGHI");
    }
    
    #[test]
    fn test_get_render_state_out_of_bounds_clamp() {
        let mut selection = Selection::new();
        selection.start_selection(70, 20, SelectionMode::Normal);
        selection.update_selection(90, 30); // Ends are out of 80x24 bounds
        selection.end_selection();

        let render_state = selection.get_render_state(80, 24); // term_cols=80, term_rows=24
        assert_eq!(
            render_state,
            Some(SelectionRenderState {
                start_coords: (70, 20), // Start (70,20) is within bounds
                end_coords: (79, 23),   // End (90,30) clamped to (79,23)
                mode: SelectionMode::Normal,
            })
        );
    }

    #[test]
    fn test_get_selected_text_empty_if_render_state_none() {
        let selection = Selection::new(); // No selection started
        let lines = vec![create_snapshot_line("Hello")];
        assert_eq!(selection.get_selected_text(&lines, 5,1), "");

        let mut sel_active_no_points = Selection::new();
        sel_active_no_points.is_active = true; // Active but no points
        assert_eq!(sel_active_no_points.get_selected_text(&lines, 5,1), "");
    }
     #[test]
    fn test_get_selected_text_normal_selection_ends_beyond_line_length() {
        let mut selection = Selection::new();
        selection.start_selection(2, 0, SelectionMode::Normal); // Start at 'l' in "Hello"
        selection.update_selection(10, 0); // End column 10, way past "Hello" (len 5)
        selection.end_selection();

        let lines = vec![create_snapshot_line("Hello")];
        // Expected: "llo" (cols 2,3,4). Selection should be clamped to line length.
        let selected_text = selection.get_selected_text(&lines, 80, 1); // term_cols is large
        assert_eq!(selected_text, "llo");
    }

    #[test]
    fn test_get_selected_text_block_selection_ends_beyond_line_length() {
        let mut selection = Selection::new();
        selection.start_selection(1, 0, SelectionMode::Block); // Start col 1
        selection.update_selection(8, 1); // End col 8 (line "Short" is len 5, "LineTwo" is len 7)
        selection.end_selection();

        let lines = vec![
            create_snapshot_line("Short"),   // Line 0, len 5
            create_snapshot_line("LineTwo"), // Line 1, len 7
        ];
        // Expected:
        // Line 0: "hort" (cols 1,2,3,4 of "Short")
        // Line 1: "ineTwo" (cols 1,2,3,4,5,6 of "LineTwo")
        let selected_text = selection.get_selected_text(&lines, 80, 2);
        assert_eq!(selected_text, "hort\nineTwo");
    }
}
