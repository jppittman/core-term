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
use pixelflow_render::NamedColor;

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
    pub fn from_snapshot(
        snapshot: &TerminalSnapshot,
        default_fg: Color,
        default_bg: Color,
    ) -> Self {
        let (cols, rows) = snapshot.dimensions;
        let mut grid = Self::new(cols, rows);

        for (row_idx, line) in snapshot.lines.iter().enumerate() {
            // Process ALL lines - Surface model evaluates entire screen each frame
            for (col_idx, glyph) in line.cells.iter().enumerate() {
                if col_idx < cols && row_idx < rows {
                    grid.set(
                        col_idx,
                        row_idx,
                        Cell::from_glyph(glyph, default_fg, default_bg),
                    );
                }
            }
        }

        grid
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
}
