// src/term/snapshot.rs

//! Defines data structures representing snapshots of terminal state for rendering,
//! including individual glyphs, lines, cursor, and selection state.

use crate::glyph::{Attributes, Glyph};
use std::ops::Index;

/// Represents the visual shape of the cursor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CursorShape {
    Block,
    Underline,
    Bar,
}

/// Represents a 2D point in the terminal grid, typically (column, row).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Point {
    /// The column (x-coordinate), 0-based.
    pub x: usize,
    /// The row (y-coordinate), 0-based.
    pub y: usize,
}

/// Represents the start and end points of a selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SelectionRange {
    pub start: Point,
    pub end: Point,
}

/// Represents the mode of text selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SelectionMode {
    #[default]
    Cell, // Standard character-by-character selection
          // Block, // Future: for block selection
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

/// Represents the state of a text selection in the terminal.
/// This includes the range of the selection, the selection mode (e.g., cell-wise),
/// and whether the selection is currently active (e.g., being dragged).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Selection {
    pub range: Option<SelectionRange>, // Make it optional
    pub mode: SelectionMode,
    pub is_active: bool, // True if selection process is ongoing (e.g., mouse button held down)
}

/// A complete snapshot of the terminal's visible state at a moment in time.
/// This structure is provided by the `TerminalEmulator` to the `Renderer`.
#[derive(Debug, Clone, PartialEq)]
pub struct RenderSnapshot {
    pub dimensions: (usize, usize), // cols, rows
    pub lines: Vec<SnapshotLine>,
    pub cursor_state: Option<CursorRenderState>,
    pub selection: Selection, // Current selection state.
}

// Point struct was moved up

impl RenderSnapshot {
    /// Gets the glyph at the given `Point` (column, row) if it exists within the snapshot dimensions.
    ///
    /// Returns `None` if the coordinates are out of bounds.
    pub fn get_glyph(&self, p: Point) -> Option<Glyph> {
        let (term_width, term_height) = self.dimensions;
        if p.x >= term_width || p.y >= term_height {
            return None;
        }
        Some(self.lines[p.y][p.x])
    }

    /// Clears and resizes the snapshot to the specified dimensions.
    ///
    /// This method reuses existing Vec allocations where possible to minimize allocations.
    /// It's designed for snapshot buffer reuse in the actor architecture.
    pub fn clear_and_resize(&mut self, cols: usize, rows: usize) {
        self.dimensions = (cols, rows);

        // Resize the lines Vec to match the new row count, reusing existing lines
        self.lines.resize_with(rows, || SnapshotLine {
            is_dirty: true,
            cells: Vec::new(),
        });

        // Clear cursor and selection state
        self.cursor_state = None;
        self.selection = Selection::default();
    }
}
