//! Grid buffer for terminal cell storage.
//!
//! The GridBuffer stores cell data (character + attributes) in a flat array,
//! optimized for SIMD sampling by the TerminalSurface.
//!
//! Colors are stored as semantic `Color` values (from pixelflow-render),
//! allowing platform-specific pixel format conversion at render time.

use crate::color::Color;
use crate::glyph::{ContentCell, Glyph};
use crate::term::snapshot::TerminalSnapshot;
use pixelflow_graphics::render::NamedColor;

/// A single cell in the grid.
#[derive(Debug, Clone, Copy)]
pub struct Cell {
    /// The character to render (or '\0' for empty).
    pub ch: char,
    /// Foreground color (semantic, not raw pixel).
    pub fg: Color,
    /// Background color (semantic, not raw pixel).
    pub bg: Color,
    /// Style flags (bold, italic, etc.) - simplified for now.
    pub bold: bool,
    pub italic: bool,
}

impl Default for Cell {
    fn default() -> Self {
        Self {
            ch: ' ',
            fg: Color::Named(NamedColor::White),
            bg: Color::Named(NamedColor::Black),
            bold: false,
            italic: false,
        }
    }
}

impl Cell {
    /// Creates a cell from a Glyph.
    pub fn from_glyph(glyph: &Glyph, default_fg: Color, default_bg: Color) -> Self {
        match glyph {
            Glyph::Single(cc) | Glyph::WidePrimary(cc) => {
                Self::from_content_cell(cc, default_fg, default_bg)
            }
            Glyph::WideSpacer { .. } => Self::default(),
        }
    }

    /// Creates a cell from a ContentCell.
    fn from_content_cell(cc: &ContentCell, default_fg: Color, default_bg: Color) -> Self {
        // Resolve Color::Default to actual default colors
        let fg = match cc.attr.fg {
            Color::Default => default_fg,
            other => other,
        };
        let bg = match cc.attr.bg {
            Color::Default => default_bg,
            other => other,
        };

        Self {
            ch: cc.c,
            fg,
            bg,
            bold: cc.attr.flags.contains(crate::glyph::AttrFlags::BOLD),
            italic: cc.attr.flags.contains(crate::glyph::AttrFlags::ITALIC),
        }
    }
}

/// A buffer of cells representing the terminal grid.
///
/// Stored in row-major order for cache-friendly access.
#[derive(Clone)]
pub struct GridBuffer {
    /// The cell data.
    cells: Vec<Cell>,
    /// Number of columns.
    pub cols: usize,
    /// Number of rows.
    pub rows: usize,
}

impl GridBuffer {
    /// Creates a new grid buffer with the given dimensions.
    pub fn new(cols: usize, rows: usize) -> Self {
        Self {
            cells: vec![Cell::default(); cols * rows],
            cols,
            rows,
        }
    }

    /// Creates a grid buffer populated from a terminal snapshot.
    ///
    /// This is the bridge between the terminal emulator's state and the
    /// Surface-based rendering pipeline.
    ///
    /// Note: For incremental updates, use `update_from_snapshot` instead
    /// to only process dirty lines.
    pub fn from_snapshot(
        snapshot: &TerminalSnapshot,
        default_fg: Color,
        default_bg: Color,
    ) -> Self {
        let (cols, rows) = snapshot.dimensions;
        let mut grid = Self::new(cols, rows);

        for (row_idx, line) in snapshot.lines.iter().enumerate() {
            if row_idx >= rows {
                break;
            }
            Self::copy_line_to_grid(&mut grid.cells, cols, row_idx, line, default_fg, default_bg);
        }

        grid
    }

    /// Updates an existing grid buffer from a terminal snapshot.
    ///
    /// Only processes lines marked as dirty, skipping unchanged lines entirely.
    /// Returns the number of dirty lines that were updated.
    ///
    /// If dimensions changed, falls back to full reconstruction.
    pub fn update_from_snapshot(
        &mut self,
        snapshot: &TerminalSnapshot,
        default_fg: Color,
        default_bg: Color,
    ) -> usize {
        let (cols, rows) = snapshot.dimensions;

        // If dimensions changed, resize and do full update
        if cols != self.cols || rows != self.rows {
            *self = Self::from_snapshot(snapshot, default_fg, default_bg);
            return rows; // All lines updated
        }

        let mut dirty_count = 0;

        for (row_idx, line) in snapshot.lines.iter().enumerate() {
            if row_idx >= rows {
                break;
            }

            // Skip clean lines - this is the optimization!
            if !line.is_dirty {
                continue;
            }

            Self::copy_line_to_grid(&mut self.cells, cols, row_idx, line, default_fg, default_bg);
            dirty_count += 1;
        }

        dirty_count
    }

    /// Internal helper to copy a single line into the grid buffer.
    #[inline]
    fn copy_line_to_grid(
        cells: &mut [Cell],
        cols: usize,
        row_idx: usize,
        line: &crate::term::snapshot::SnapshotLine,
        default_fg: Color,
        default_bg: Color,
    ) {
        let row_start = row_idx * cols;
        for (col_idx, glyph) in line.cells.iter().enumerate() {
            if col_idx >= cols {
                break;
            }
            cells[row_start + col_idx] = Cell::from_glyph(glyph, default_fg, default_bg);
        }
    }

    /// Gets the cell at (col, row).
    #[inline(always)]
    pub fn get(&self, col: usize, row: usize) -> &Cell {
        let idx = row * self.cols + col;
        &self.cells[idx.min(self.cells.len() - 1)]
    }

    /// Gets a mutable reference to the cell at (col, row).
    #[inline(always)]
    pub fn get_mut(&mut self, col: usize, row: usize) -> &mut Cell {
        let idx = row * self.cols + col;
        let len = self.cells.len();
        &mut self.cells[idx.min(len - 1)]
    }

    /// Sets the cell at (col, row).
    #[inline(always)]
    pub fn set(&mut self, col: usize, row: usize, cell: Cell) {
        if col < self.cols && row < self.rows {
            let idx = row * self.cols + col;
            self.cells[idx] = cell;
        }
    }

    /// Clears all cells to the default.
    pub fn clear(&mut self) {
        for cell in &mut self.cells {
            *cell = Cell::default();
        }
    }

    /// Resizes the grid, preserving data where possible.
    pub fn resize(&mut self, new_cols: usize, new_rows: usize) {
        let mut new_cells = vec![Cell::default(); new_cols * new_rows];

        let copy_cols = self.cols.min(new_cols);
        let copy_rows = self.rows.min(new_rows);

        for row in 0..copy_rows {
            for col in 0..copy_cols {
                let old_idx = row * self.cols + col;
                let new_idx = row * new_cols + col;
                new_cells[new_idx] = self.cells[old_idx];
            }
        }

        self.cells = new_cells;
        self.cols = new_cols;
        self.rows = new_rows;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grid_new() {
        let grid = GridBuffer::new(80, 24);
        assert_eq!(grid.cols, 80);
        assert_eq!(grid.rows, 24);
    }

    #[test]
    fn test_grid_get_set() {
        let mut grid = GridBuffer::new(10, 5);

        let cell = Cell {
            ch: 'X',
            fg: Color::Rgb(255, 0, 0), // Red
            bg: Color::Rgb(0, 0, 255), // Blue
            bold: true,
            italic: false,
        };

        grid.set(3, 2, cell);

        let retrieved = grid.get(3, 2);
        assert_eq!(retrieved.ch, 'X');
        assert_eq!(retrieved.fg, Color::Rgb(255, 0, 0));
        assert!(retrieved.bold);
    }

    #[test]
    fn test_default_cell_colors() {
        let cell = Cell::default();
        assert_eq!(cell.fg, Color::Named(NamedColor::White));
        assert_eq!(cell.bg, Color::Named(NamedColor::Black));
    }

    #[test]
    fn test_update_from_snapshot_skips_clean_lines() {
        use crate::glyph::{AttrFlags, Attributes, ContentCell, Glyph};
        use crate::term::snapshot::{SnapshotLine, TerminalSnapshot};
        use std::sync::Arc;

        let default_fg = Color::Named(NamedColor::White);
        let default_bg = Color::Named(NamedColor::Black);

        // Create a simple 3x2 grid
        let make_glyph = |c: char| {
            Glyph::Single(ContentCell {
                c,
                attr: Attributes {
                    fg: default_fg,
                    bg: default_bg,
                    flags: AttrFlags::empty(),
                },
            })
        };

        // Initial snapshot - all lines dirty
        let initial_snapshot = TerminalSnapshot {
            dimensions: (3, 2),
            lines: vec![
                SnapshotLine {
                    is_dirty: true,
                    cells: Arc::new(vec![make_glyph('A'), make_glyph('B'), make_glyph('C')]),
                },
                SnapshotLine {
                    is_dirty: true,
                    cells: Arc::new(vec![make_glyph('D'), make_glyph('E'), make_glyph('F')]),
                },
            ],
            cursor_state: None,
            selection: Default::default(),
            cell_width_px: 8,
            cell_height_px: 16,
        };

        let mut grid = GridBuffer::from_snapshot(&initial_snapshot, default_fg, default_bg);
        assert_eq!(grid.get(0, 0).ch, 'A');
        assert_eq!(grid.get(0, 1).ch, 'D');

        // Second snapshot - only row 0 is dirty (changed to X, Y, Z)
        let update_snapshot = TerminalSnapshot {
            dimensions: (3, 2),
            lines: vec![
                SnapshotLine {
                    is_dirty: true, // This line changed
                    cells: Arc::new(vec![make_glyph('X'), make_glyph('Y'), make_glyph('Z')]),
                },
                SnapshotLine {
                    is_dirty: false, // This line is clean - should be skipped!
                    cells: Arc::new(vec![make_glyph('?'), make_glyph('?'), make_glyph('?')]),
                },
            ],
            cursor_state: None,
            selection: Default::default(),
            cell_width_px: 8,
            cell_height_px: 16,
        };

        let dirty_count = grid.update_from_snapshot(&update_snapshot, default_fg, default_bg);

        // Should have only updated 1 line
        assert_eq!(dirty_count, 1);

        // Row 0 should be updated
        assert_eq!(grid.get(0, 0).ch, 'X');
        assert_eq!(grid.get(1, 0).ch, 'Y');
        assert_eq!(grid.get(2, 0).ch, 'Z');

        // Row 1 should still have the OLD values (D, E, F), not the new ones (?, ?, ?)
        // because is_dirty was false
        assert_eq!(grid.get(0, 1).ch, 'D');
        assert_eq!(grid.get(1, 1).ch, 'E');
        assert_eq!(grid.get(2, 1).ch, 'F');
    }
}
