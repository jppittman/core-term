use crate::surface::grid::GridBuffer;
use pixelflow_core::pipe::Surface;
use pixelflow_core::Batch;
use pixelflow_render::GlyphCache;

pub struct TerminalSurface {
    pub grid: GridBuffer,
    pub cell_width: usize,
    pub cell_height: usize,
    pub glyph_cache: GlyphCache,
}

impl TerminalSurface {
    pub fn new(
        cols: usize,
        rows: usize,
        cell_width: usize,
        cell_height: usize,
        glyph_cache: GlyphCache,
    ) -> Self {
        Self {
            grid: GridBuffer::new(cols, rows),
            cell_width,
            cell_height,
            glyph_cache,
        }
    }

    #[inline]
    pub fn eval_scalar(&self, x: u32, y: u32) -> u32 {
        let col = (x as usize) / self.cell_width;
        let row = (y as usize) / self.cell_height;

        if col >= self.grid.cols || row >= self.grid.rows {
            return 0xFF_00_00_00;
        }

        let cell = self.grid.get(col, row);

        if cell.ch == ' ' || cell.ch == '\0' {
            return cell.bg;
        }

        let lx = (x as usize) % self.cell_width;
        let ly = (y as usize) % self.cell_height;

        let baked = self.glyph_cache.get(cell.ch);
        let val = baked.eval(Batch::splat(lx as u32), Batch::splat(ly as u32));
        let coverage = val.cast::<u32>().extract(0);

        if coverage > 100 {
            return cell.fg;
        } else {
            return cell.bg;
        }
    }
}

impl Surface<u32> for TerminalSurface {
    fn eval(&self, x: Batch<u32>, y: Batch<u32>) -> Batch<u32> {
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

impl Clone for TerminalSurface {
    fn clone(&self) -> Self {
        Self {
            grid: self.grid.clone(),
            cell_width: self.cell_width,
            cell_height: self.cell_height,
            glyph_cache: self.glyph_cache.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::surface::grid::Cell;
    use pixelflow_render::font;

    fn create_cache() -> GlyphCache {
        GlyphCache::new(font().clone(), 10, 16, 1.0)
    }

    #[test]
    fn test_terminal_surface_new() {
        let surface = TerminalSurface::new(80, 24, 10, 16, create_cache());
        assert_eq!(surface.grid.cols, 80);
        assert_eq!(surface.grid.rows, 24);
        assert_eq!(surface.cell_width, 10);
        assert_eq!(surface.cell_height, 16);
    }

    #[test]
    fn test_eval_scalar_empty() {
        let surface = TerminalSurface::new(10, 5, 10, 16, create_cache());
        let color = surface.eval_scalar(5, 5);
        assert_eq!(color, 0xFF_00_00_00);
    }

    #[test]
    fn test_eval_scalar_with_char() {
        let mut surface = TerminalSurface::new(10, 5, 10, 16, create_cache());
        let cell = Cell {
            ch: 'A',
            fg: 0xFF_FF_00_00,
            bg: 0xFF_00_00_FF,
            bold: false,
            italic: false,
        };
        surface.grid.set(0, 0, cell);
        let color = surface.eval_scalar(5, 8);
    }

    #[test]
    fn test_eval_batch() {
        let surface = TerminalSurface::new(10, 5, 10, 16, create_cache());
        let x = Batch::new(0, 5, 10, 15);
        let y = Batch::new(0, 8, 16, 24);
        let result = surface.eval(x, y);
        let mut arr = [0u32; 4];
        unsafe { result.store(arr.as_mut_ptr()) };
        for &c in &arr {
            assert_eq!(c, 0xFF_00_00_00);
        }
    }
}
