//! Terminal surface implementation.
//!
//! The TerminalSurface implements `Surface<P>` to enable functional
//! composition with pixelflow-core. It samples the grid buffer and
//! renders glyphs using the font system.
//!
//! The surface is generic over pixel format `P: Pixel`, allowing
//! platform-specific pixel formats (Rgba for Cocoa, Bgra for X11).

use crate::color::Color;
use crate::surface::grid::GridBuffer;
use core::marker::PhantomData;
use pixelflow_core::dsl::MaskExt;
use pixelflow_core::ops::Baked;
use pixelflow_core::pipe::Surface;
use pixelflow_core::Batch;
use pixelflow_fonts::{glyphs, Lazy};
use pixelflow_render::font;
use pixelflow_render::Pixel;
use std::sync::Arc;

/// Glyph factory type - closure that returns lazily-baked glyphs.
type GlyphFactory = Arc<dyn Fn(char) -> Lazy<'static, Baked<u8>> + Send + Sync>;

/// A terminal rendered as a functional surface.
///
/// When sampled at pixel coordinates (x, y), it determines which cell
/// the pixel falls into, samples the glyph surface, and returns the blended color.
///
/// Generic over pixel format `P` for platform-specific rendering:
/// - `Rgba` for Cocoa (macOS)
/// - `Bgra` for X11 (Linux)
///
/// The `P: Surface<P>` bound ensures colors can be used as constant surfaces
/// for the pixelflow DSL (e.g., `glyph.over(fg, bg)`).
pub struct TerminalSurface<P: Pixel + Surface<P>> {
    /// The grid of cells.
    pub grid: GridBuffer,
    /// Glyph factory - caching is automatic via Lazy.
    glyph: GlyphFactory,
    /// Cell width in pixels.
    pub cell_width: u32,
    /// Cell height in pixels.
    pub cell_height: u32,
    /// Phantom data for pixel format.
    _pixel: PhantomData<P>,
}

impl<P: Pixel + Surface<P>> TerminalSurface<P> {
    /// Creates a new terminal surface.
    pub fn new(cols: usize, rows: usize, cell_width: u32, cell_height: u32) -> Self {
        let f = font();
        let glyph_fn = glyphs(f.clone(), cell_width, cell_height);

        Self {
            grid: GridBuffer::new(cols, rows),
            glyph: Arc::new(glyph_fn),
            cell_width,
            cell_height,
            _pixel: PhantomData,
        }
    }

    /// Creates a terminal surface with an existing grid and glyph factory.
    pub fn with_grid(grid: GridBuffer, glyph: GlyphFactory, cell_width: u32, cell_height: u32) -> Self {
        Self {
            grid,
            glyph,
            cell_width,
            cell_height,
            _pixel: PhantomData,
        }
    }

    /// Converts a semantic Color to this surface's pixel format.
    #[inline]
    fn color_to_pixel(color: Color) -> P {
        // Convert via Rgba (canonical format), then to target pixel type
        // For Rgba -> Rgba: no-op. For Rgba -> Bgra: swizzles R and B.
        P::from_u32(color.to_rgba().0)
    }

    /// Evaluates the surface at a single pixel coordinate.
    ///
    /// Uses the pixelflow DSL: `glyph.over(fg, bg)` for correct alpha blending.
    #[inline]
    pub fn eval_scalar(&self, x: u32, y: u32) -> P {
        // Determine which cell this pixel is in
        let col = (x / self.cell_width) as usize;
        let row = (y / self.cell_height) as usize;

        // Bounds check - return black for out of bounds
        if col >= self.grid.cols || row >= self.grid.rows {
            return P::from_u32(0xFF_00_00_00);
        }

        // Get the cell
        let cell = self.grid.get(col, row);
        let bg_pixel = Self::color_to_pixel(cell.bg);

        // For space or empty, just return background
        if cell.ch == ' ' || cell.ch == '\0' {
            return bg_pixel;
        }

        // Local coordinates within the cell
        let lx = x % self.cell_width;
        let ly = y % self.cell_height;

        // Get the lazily-cached glyph (Surface<u8>)
        let glyph_lazy = (self.glyph)(cell.ch);
        let baked: &Baked<u8> = glyph_lazy.get();
        let fg_pixel = Self::color_to_pixel(cell.fg);

        // Use DSL: glyph.over(fg, bg) - evaluated at single point via batch
        // Create single-value batches for scalar evaluation
        let x_batch = Batch::splat(lx);
        let y_batch = Batch::splat(ly);

        // Compose using pixelflow DSL
        let composed = baked.over::<P, _, _>(fg_pixel, bg_pixel);
        let result: Batch<P> = composed.eval(x_batch, y_batch);

        // Extract first lane (all lanes are identical for splat inputs)
        P::from_u32(result.transmute::<u32>().to_array_usize()[0] as u32)
    }
}

// Implement Surface trait for SIMD evaluation
impl<P: Pixel + Surface<P>> Surface<P> for TerminalSurface<P> {
    fn eval(&self, x: Batch<u32>, y: Batch<u32>) -> Batch<P> {
        // Extract scalar values and evaluate each lane
        // TODO: Optimize with proper SIMD gather operations
        let x_arr = x.to_array_usize();
        let y_arr = y.to_array_usize();

        let p0 = self.eval_scalar(x_arr[0] as u32, y_arr[0] as u32);
        let p1 = self.eval_scalar(x_arr[1] as u32, y_arr[1] as u32);
        let p2 = self.eval_scalar(x_arr[2] as u32, y_arr[2] as u32);
        let p3 = self.eval_scalar(x_arr[3] as u32, y_arr[3] as u32);

        Batch::new(p0.to_u32(), p1.to_u32(), p2.to_u32(), p3.to_u32()).transmute()
    }
}

impl<P: Pixel + Surface<P>> Clone for TerminalSurface<P> {
    fn clone(&self) -> Self {
        Self {
            grid: self.grid.clone(),
            glyph: self.glyph.clone(), // Arc clone - shares the factory
            cell_width: self.cell_width,
            cell_height: self.cell_height,
            _pixel: PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::surface::grid::Cell;
    use pixelflow_render::{NamedColor, Rgba};

    #[test]
    fn test_terminal_surface_new() {
        let surface: TerminalSurface<Rgba> = TerminalSurface::new(80, 24, 10, 16);
        assert_eq!(surface.grid.cols, 80);
        assert_eq!(surface.grid.rows, 24);
        assert_eq!(surface.cell_width, 10);
        assert_eq!(surface.cell_height, 16);
    }

    #[test]
    fn test_eval_scalar_empty() {
        let surface: TerminalSurface<Rgba> = TerminalSurface::new(10, 5, 10, 16);

        // Empty cell should return background (default is black)
        let color = surface.eval_scalar(5, 5);
        // Default bg is NamedColor::Black = (0, 0, 0) -> Rgba(0xFF000000)
        // Check alpha is fully opaque
        assert_eq!(color.a(), 0xFF);
        // Check it's black
        assert_eq!(color.r(), 0);
        assert_eq!(color.g(), 0);
        assert_eq!(color.b(), 0);
    }

    #[test]
    fn test_eval_scalar_with_char() {
        let mut surface: TerminalSurface<Rgba> = TerminalSurface::new(10, 5, 10, 16);

        // Set a character in cell (0, 0)
        let cell = Cell {
            ch: 'A',
            fg: Color::Rgb(255, 0, 0), // Red
            bg: Color::Rgb(0, 0, 255), // Blue
            bold: false,
            italic: false,
        };
        surface.grid.set(0, 0, cell);

        // Sample somewhere in the cell - should get a blended color
        let color = surface.eval_scalar(5, 8);
        // The exact color depends on the glyph shape, but alpha should be fully opaque
        assert_eq!(color.a(), 0xFF);
    }

    #[test]
    fn test_eval_batch() {
        let surface: TerminalSurface<Rgba> = TerminalSurface::new(10, 5, 10, 16);

        let x = Batch::new(0, 5, 10, 15);
        let y = Batch::new(0, 8, 16, 24);

        let result: Batch<Rgba> = surface.eval(x, y);

        // All should be background (black) for empty grid
        let result_u32: Batch<u32> = result.transmute();
        let arr = result_u32.to_array_usize();

        for &c in &arr {
            let c = c as u32;
            let pixel = Rgba(c);
            // Black with full alpha
            assert_eq!(pixel.a(), 0xFF);
            assert_eq!(pixel.r(), 0);
            assert_eq!(pixel.g(), 0);
            assert_eq!(pixel.b(), 0);
        }
    }

    #[test]
    fn test_color_conversion() {
        // Test that Color converts correctly to Rgba
        let red = Color::Rgb(255, 0, 0);
        let rgba = red.to_rgba();
        assert_eq!(rgba.r(), 255);
        assert_eq!(rgba.g(), 0);
        assert_eq!(rgba.b(), 0);
        assert_eq!(rgba.a(), 255);

        // Test named colors
        let white = Color::Named(NamedColor::White);
        let rgba = white.to_rgba();
        assert_eq!(rgba.r(), 229); // NamedColor::White = (229, 229, 229)
        assert_eq!(rgba.g(), 229);
        assert_eq!(rgba.b(), 229);
    }
}
