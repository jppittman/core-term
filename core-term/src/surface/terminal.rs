//! Terminal surface implementation.
//!
//! The TerminalSurface implements `Surface<u32>` to enable functional
//! composition with pixelflow-core. It samples the grid buffer and
//! renders glyphs using the font system.

use crate::surface::grid::GridBuffer;
use pixelflow_core::pipe::Surface;
use pixelflow_core::Batch;

/// A terminal rendered as a functional surface.
///
/// When sampled at pixel coordinates (x, y), it determines which cell
/// the pixel falls into, samples the glyph SDF, and returns the blended color.
pub struct TerminalSurface {
    /// The grid of cells.
    pub grid: GridBuffer,
    /// Cell width in pixels.
    pub cell_width: usize,
    /// Cell height in pixels.
    pub cell_height: usize,
}

impl TerminalSurface {
    /// Creates a new terminal surface.
    pub fn new(cols: usize, rows: usize, cell_width: usize, cell_height: usize) -> Self {
        Self {
            grid: GridBuffer::new(cols, rows),
            cell_width,
            cell_height,
        }
    }

    /// Evaluates the surface at a single pixel coordinate.
    ///
    /// This is the scalar version used for testing and fallback.
    #[inline]
    pub fn eval_scalar(&self, x: u32, y: u32) -> u32 {
        // Determine which cell this pixel is in
        let col = (x as usize) / self.cell_width;
        let row = (y as usize) / self.cell_height;

        // Bounds check
        if col >= self.grid.cols || row >= self.grid.rows {
            return 0xFF_00_00_00; // Black for out of bounds
        }

        // Get the cell
        let cell = self.grid.get(col, row);

        // For space or empty, just return background
        if cell.ch == ' ' || cell.ch == '\0' {
            return cell.bg;
        }

        // Local coordinates within the cell
        let lx = (x as usize) % self.cell_width;
        let ly = (y as usize) % self.cell_height;

        // Simple box rendering for now (placeholder until Loop-Blinn is wired)
        // This just renders a solid foreground for non-space characters
        // TODO: Wire up proper SDF glyph rendering
        let in_glyph = lx >= 1 && lx < self.cell_width - 1 && ly >= 2 && ly < self.cell_height - 2;

        if in_glyph {
            cell.fg
        } else {
            cell.bg
        }
    }
}

// Implement Surface trait for SIMD evaluation
impl Surface<u32> for TerminalSurface {
    fn eval(&self, x: Batch<u32>, y: Batch<u32>) -> Batch<u32> {
        // Extract scalar values and evaluate each lane
        // TODO: Optimize with proper SIMD gather operations
        let x_arr = x.to_array_usize();
        let y_arr = y.to_array_usize();

        Batch::new(
            self.eval_scalar(x_arr[0] as u32, y_arr[0] as u32),
            self.eval_scalar(x_arr[1] as u32, y_arr[1] as u32),
            self.eval_scalar(x_arr[2] as u32, y_arr[2] as u32),
            self.eval_scalar(x_arr[3] as u32, y_arr[3] as u32),
        )
    }
}

// TerminalSurface needs to be Copy for Surface trait, but it contains Vec
// We need a different approach - use a reference-based surface
impl Clone for TerminalSurface {
    fn clone(&self) -> Self {
        Self {
            grid: self.grid.clone(),
            cell_width: self.cell_width,
            cell_height: self.cell_height,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::surface::grid::Cell;

    #[test]
    fn test_terminal_surface_new() {
        let surface = TerminalSurface::new(80, 24, 10, 16);
        assert_eq!(surface.grid.cols, 80);
        assert_eq!(surface.grid.rows, 24);
        assert_eq!(surface.cell_width, 10);
        assert_eq!(surface.cell_height, 16);
    }

    #[test]
    fn test_eval_scalar_empty() {
        let surface = TerminalSurface::new(10, 5, 10, 16);

        // Empty cell should return background
        let color = surface.eval_scalar(5, 5);
        // Default bg is black
        assert_eq!(color, 0xFF_00_00_00);
    }

    #[test]
    fn test_eval_scalar_with_char() {
        let mut surface = TerminalSurface::new(10, 5, 10, 16);

        // Set a character in cell (0, 0)
        let cell = Cell {
            ch: 'A',
            fg: 0xFF_FF_00_00, // Red
            bg: 0xFF_00_00_FF, // Blue
            bold: false,
            italic: false,
        };
        surface.grid.set(0, 0, cell);

        // Sample inside the glyph area
        let color = surface.eval_scalar(5, 8);
        assert_eq!(color, 0xFF_FF_00_00); // Should be foreground (red)

        // Sample in the border area
        let color = surface.eval_scalar(0, 0);
        assert_eq!(color, 0xFF_00_00_FF); // Should be background (blue)
    }

    #[test]
    fn test_eval_batch() {
        let surface = TerminalSurface::new(10, 5, 10, 16);

        let x = Batch::new(0, 5, 10, 15);
        let y = Batch::new(0, 8, 16, 24);

        let result = surface.eval(x, y);

        // All should be background (black) for empty grid
        let mut arr = [0u32; 4];
        unsafe { result.store(arr.as_mut_ptr()) };

        for &c in &arr {
            assert_eq!(c, 0xFF_00_00_00);
        }
    }
}
