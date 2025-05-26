// src/term/snapshot.rs

use crate::glyph::{Attributes, Glyph};
use std::ops::Index;

/// Represents the visual shape of the cursor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CursorShape {
    Block,
    Underline,
    Bar,
}

/// Represents the mode of text selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectionMode {
    Normal, // Character-wise selection
    Block,  // Rectangular block selection
}

/// A snapshot of a single line in the terminal grid.
#[derive(Debug, Clone, PartialEq)]
pub struct SnapshotLine {
    pub is_dirty: bool,
    pub cells: Vec<Glyph>,
}

impl Index<usize> for SnapshotLine {
    type Output = Glyph;

    fn index(&self, column_index: usize) -> &Self::Output {
        &self.cells[column_index] // Delegates to Vec<Glyph>'s indexing
    }
}

/// Information needed by the Renderer to draw the cursor.
#[derive(Debug, Clone, PartialEq)]
pub struct CursorRenderState {
    pub x: usize, // Physical x of the cell the cursor is on/starts at
    pub y: usize, // Physical y
    pub shape: CursorShape,
    pub cell_char_underneath: char, // Character in the cell (could be space)
    pub cell_attributes_underneath: Attributes, // Attributes of the cell
}

/// Information needed by the Renderer to draw selection highlighting.
#[derive(Debug, Clone, PartialEq)]
pub struct SelectionRenderState {
    pub start_coords: (usize, usize), // 0-based (col, row)
    pub end_coords: (usize, usize),   // 0-based (col, row)
    pub mode: SelectionMode,
}

/// A complete snapshot of the terminal's visible state at a moment in time.
/// This structure is provided by the `TerminalEmulator` to the `Renderer`.
#[derive(Debug, Clone, PartialEq)]
pub struct RenderSnapshot {
    pub dimensions: (usize, usize), // cols, rows
    pub lines: Vec<SnapshotLine>,
    pub cursor_state: Option<CursorRenderState>,
    pub selection_state: Option<SelectionRenderState>,
}

#[derive(Debug, Clone)]
pub struct Point {
    pub x: usize,
    pub y: usize,
}

impl RenderSnapshot {
    pub fn get_glyph(&self, p: Point) -> Option<Glyph> {
        let (term_width, term_height) = self.dimensions;
        if p.x >= term_width || p.y >= term_height {
            return None;
        }
        Some(self.lines[p.y][p.x])
    }
}
