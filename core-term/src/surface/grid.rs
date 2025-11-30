//! Grid buffer for terminal cell storage.
//!
//! The GridBuffer stores cell data (character + attributes) in a flat array,
//! optimized for SIMD sampling by the TerminalSurface.

use crate::color::Color;
use crate::glyph::{Attributes, ContentCell, Glyph};
use crate::term::snapshot::TerminalSnapshot;

/// A single cell in the grid, packed for efficient access.
#[derive(Debug, Clone, Copy)]
pub struct Cell {
    /// The character to render (or '\0' for empty).
    pub ch: char,
    /// Foreground color as packed u32 (ARGB).
    pub fg: u32,
    /// Background color as packed u32 (ARGB).
    pub bg: u32,
    /// Style flags (bold, italic, etc.) - simplified for now.
    pub bold: bool,
    pub italic: bool,
}

impl Default for Cell {
    fn default() -> Self {
        Self {
            ch: ' ',
            fg: 0xFF_FF_FF_FF, // White
            bg: 0xFF_00_00_00, // Black
            bold: false,
            italic: false,
        }
    }
}

impl Cell {
    /// Creates a cell from a Glyph.
    pub fn from_glyph(glyph: &Glyph, default_fg: u32, default_bg: u32) -> Self {
        match glyph {
            Glyph::Single(cc) | Glyph::WidePrimary(cc) => {
                Self::from_content_cell(cc, default_fg, default_bg)
            }
            Glyph::WideSpacer { .. } => Self::default(),
        }
    }

    /// Creates a cell from a ContentCell.
    fn from_content_cell(cc: &ContentCell, default_fg: u32, default_bg: u32) -> Self {
        let fg = color_to_u32(&cc.attr.fg, default_fg);
        let bg = color_to_u32(&cc.attr.bg, default_bg);

        Self {
            ch: cc.c,
            fg,
            bg,
            bold: cc.attr.flags.contains(crate::glyph::AttrFlags::BOLD),
            italic: cc.attr.flags.contains(crate::glyph::AttrFlags::ITALIC),
        }
    }
}

/// Convert a Color to u32 ARGB format.
fn color_to_u32(color: &Color, default: u32) -> u32 {
    match color {
        Color::Default => default,
        Color::Rgb(r, g, b) => {
            0xFF_00_00_00 | ((*r as u32) << 16) | ((*g as u32) << 8) | (*b as u32)
        }
        Color::Named(named) => {
            let (r, g, b) = named.to_rgb();
            0xFF_00_00_00 | ((r as u32) << 16) | ((g as u32) << 8) | (b as u32)
        }
        Color::Indexed(idx) => indexed_color_to_rgb(*idx),
    }
}

/// Convert indexed color (0-255) to RGB.
fn indexed_color_to_rgb(idx: u8) -> u32 {
    // Standard 16-color palette
    const BASIC: [u32; 16] = [
        0xFF_00_00_00, // 0: Black
        0xFF_80_00_00, // 1: Red
        0xFF_00_80_00, // 2: Green
        0xFF_80_80_00, // 3: Yellow
        0xFF_00_00_80, // 4: Blue
        0xFF_80_00_80, // 5: Magenta
        0xFF_00_80_80, // 6: Cyan
        0xFF_C0_C0_C0, // 7: White
        0xFF_80_80_80, // 8: Bright Black
        0xFF_FF_00_00, // 9: Bright Red
        0xFF_00_FF_00, // 10: Bright Green
        0xFF_FF_FF_00, // 11: Bright Yellow
        0xFF_00_00_FF, // 12: Bright Blue
        0xFF_FF_00_FF, // 13: Bright Magenta
        0xFF_00_FF_FF, // 14: Bright Cyan
        0xFF_FF_FF_FF, // 15: Bright White
    ];

    if idx < 16 {
        BASIC[idx as usize]
    } else if idx < 232 {
        // 6x6x6 color cube (indices 16-231)
        let idx = idx - 16;
        let r = (idx / 36) % 6;
        let g = (idx / 6) % 6;
        let b = idx % 6;
        let r = if r == 0 { 0 } else { 55 + r * 40 };
        let g = if g == 0 { 0 } else { 55 + g * 40 };
        let b = if b == 0 { 0 } else { 55 + b * 40 };
        0xFF_00_00_00 | ((r as u32) << 16) | ((g as u32) << 8) | (b as u32)
    } else {
        // Grayscale (indices 232-255)
        let gray = 8 + (idx - 232) * 10;
        0xFF_00_00_00 | ((gray as u32) << 16) | ((gray as u32) << 8) | (gray as u32)
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
    /// Surface-based rendering pipeline. Only processes dirty lines.
    pub fn from_snapshot(snapshot: &TerminalSnapshot, default_fg: u32, default_bg: u32) -> Self {
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
            fg: 0xFF_FF_00_00,
            bg: 0xFF_00_00_FF,
            bold: true,
            italic: false,
        };

        grid.set(3, 2, cell);

        let retrieved = grid.get(3, 2);
        assert_eq!(retrieved.ch, 'X');
        assert_eq!(retrieved.fg, 0xFF_FF_00_00);
        assert!(retrieved.bold);
    }

    #[test]
    fn test_indexed_color() {
        // Test basic colors
        assert_eq!(indexed_color_to_rgb(0), 0xFF_00_00_00); // Black
        assert_eq!(indexed_color_to_rgb(15), 0xFF_FF_FF_FF); // Bright White

        // Test grayscale
        let gray = indexed_color_to_rgb(232); // First grayscale
        assert_eq!(gray, 0xFF_08_08_08);
    }
}
