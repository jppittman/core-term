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
use pixelflow_core::{Batch, SimdBatch};
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
    pub fn with_grid(
        grid: GridBuffer,
        glyph: GlyphFactory,
        cell_width: u32,
        cell_height: u32,
    ) -> Self {
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
        let x_batch = Batch::<u32>::splat(lx);
        let y_batch = Batch::<u32>::splat(ly);

        // Compose using pixelflow DSL
        let composed = baked.over::<P, _, _>(fg_pixel, bg_pixel);
        let result: Batch<P> = composed.eval(x_batch, y_batch);

        // Extract first lane (all lanes are identical for splat inputs)
        result.first()
    }
}

// Implement Surface trait for SIMD evaluation
impl<P: Pixel + Surface<P>> Surface<P> for TerminalSurface<P> {
    fn eval(&self, x: Batch<u32>, y: Batch<u32>) -> Batch<P> {
        use pixelflow_core::batch::LANES;
        use pixelflow_core::SimdBatch;

        // Extract coordinates to arrays
        let mut x_arr = [0u32; LANES];
        let mut y_arr = [0u32; LANES];
        SimdBatch::store(&x, &mut x_arr);
        SimdBatch::store(&y, &mut y_arr);

        // Evaluate each lane (TODO: optimize with proper SIMD gather)
        let mut results = [0u32; LANES];
        for i in 0..LANES {
            results[i] = self.eval_scalar(x_arr[i], y_arr[i]).to_u32();
        }

        // Convert to Batch<P>
        P::batch_from_u32(Batch::<u32>::load(&results))
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
    use pixelflow_core::backend::SimdBatch;
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

        // Test with splat - all lanes same value
        let x = Batch::<u32>::splat(5);
        let y = Batch::<u32>::splat(8);

        let result: Batch<Rgba> = surface.eval(x, y);

        // Should be background (black) for empty grid
        // SAFETY: transmute between same-sized SIMD register types
        let result_u32: Batch<u32> = unsafe { core::mem::transmute(result) };
        let c = result_u32.first();
        let pixel = Rgba(c);
        // Black with full alpha
        assert_eq!(pixel.a(), 0xFF);
        assert_eq!(pixel.r(), 0);
        assert_eq!(pixel.g(), 0);
        assert_eq!(pixel.b(), 0);
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

    // =========================================================================
    // Rendering correctness tests - these catch color format bugs
    // =========================================================================

    #[test]
    fn test_black_background_stays_black() {
        // CRITICAL: Empty cells with black bg must render as pure black.
        // If colors are corrupted/swizzled, this will fail.
        let mut surface: TerminalSurface<Rgba> = TerminalSurface::new(10, 5, 10, 16);

        // Set explicit black background
        let cell = Cell {
            ch: ' ', // Space - no glyph coverage
            fg: Color::Rgb(255, 255, 255),
            bg: Color::Rgb(0, 0, 0), // Pure black
            bold: false,
            italic: false,
        };
        surface.grid.set(0, 0, cell);

        // Sample multiple points in the cell - all should be black
        for x in 0..10 {
            for y in 0..16 {
                let pixel = surface.eval_scalar(x, y);
                assert_eq!(
                    (pixel.r(), pixel.g(), pixel.b(), pixel.a()),
                    (0, 0, 0, 255),
                    "Pixel at ({}, {}) should be black, got RGBA({}, {}, {}, {})",
                    x,
                    y,
                    pixel.r(),
                    pixel.g(),
                    pixel.b(),
                    pixel.a()
                );
            }
        }
    }

    #[test]
    fn test_white_background_stays_white() {
        // White bg must render as pure white
        let mut surface: TerminalSurface<Rgba> = TerminalSurface::new(10, 5, 10, 16);

        let cell = Cell {
            ch: ' ',
            fg: Color::Rgb(0, 0, 0),
            bg: Color::Rgb(255, 255, 255), // Pure white
            bold: false,
            italic: false,
        };
        surface.grid.set(0, 0, cell);

        for x in 0..10 {
            for y in 0..16 {
                let pixel = surface.eval_scalar(x, y);
                assert_eq!(
                    (pixel.r(), pixel.g(), pixel.b(), pixel.a()),
                    (255, 255, 255, 255),
                    "Pixel at ({}, {}) should be white, got RGBA({}, {}, {}, {})",
                    x,
                    y,
                    pixel.r(),
                    pixel.g(),
                    pixel.b(),
                    pixel.a()
                );
            }
        }
    }

    #[test]
    fn test_red_background_is_red_not_blue() {
        // This catches R/B channel swapping
        let mut surface: TerminalSurface<Rgba> = TerminalSurface::new(10, 5, 10, 16);

        let cell = Cell {
            ch: ' ',
            fg: Color::Rgb(0, 0, 0),
            bg: Color::Rgb(255, 0, 0), // Pure red
            bold: false,
            italic: false,
        };
        surface.grid.set(0, 0, cell);

        let pixel = surface.eval_scalar(5, 8);
        assert_eq!(pixel.r(), 255, "Red channel should be 255");
        assert_eq!(pixel.g(), 0, "Green channel should be 0");
        assert_eq!(
            pixel.b(),
            0,
            "Blue channel should be 0 (not swapped with red)"
        );
        assert_eq!(pixel.a(), 255, "Alpha should be 255");
    }

    #[test]
    fn test_blue_background_is_blue_not_red() {
        // This catches R/B channel swapping
        let mut surface: TerminalSurface<Rgba> = TerminalSurface::new(10, 5, 10, 16);

        let cell = Cell {
            ch: ' ',
            fg: Color::Rgb(0, 0, 0),
            bg: Color::Rgb(0, 0, 255), // Pure blue
            bold: false,
            italic: false,
        };
        surface.grid.set(0, 0, cell);

        let pixel = surface.eval_scalar(5, 8);
        assert_eq!(
            pixel.r(),
            0,
            "Red channel should be 0 (not swapped with blue)"
        );
        assert_eq!(pixel.g(), 0, "Green channel should be 0");
        assert_eq!(pixel.b(), 255, "Blue channel should be 255");
        assert_eq!(pixel.a(), 255, "Alpha should be 255");
    }

    #[test]
    fn test_glyph_coverage_and_blending() {
        // Debug test: check what the glyph coverage actually is and how blending works
        use pixelflow_core::ops::Baked;
        use pixelflow_fonts::{glyphs, Lazy};
        use pixelflow_render::font;

        // Get a glyph for 'A'
        let f = font();
        let glyph_fn = glyphs(f.clone(), 10, 16);
        let glyph_lazy: Lazy<'static, Baked<u8>> = glyph_fn('A');
        let baked: &Baked<u8> = glyph_lazy.get();

        // Check coverage at corners
        let corners = [(0u32, 0u32), (9, 0), (0, 15), (9, 15)];
        println!("Glyph 'A' coverage at corners (cell 10x16):");
        for (x, y) in corners {
            let x_batch = Batch::<u32>::splat(x);
            let y_batch = Batch::<u32>::splat(y);
            let coverage: Batch<u8> = baked.eval(x_batch, y_batch);
            let alpha = coverage.first();
            println!("  ({}, {}): coverage = {}", x, y, alpha);
        }

        // Now test that over() produces correct results with known alpha
        // Create a simple test: 50% alpha should give ~50/50 blend
        use pixelflow_core::dsl::MaskExt;

        // Create a constant alpha mask
        struct ConstAlpha(u8);
        impl Surface<u8> for ConstAlpha {
            fn eval(&self, _x: Batch<u32>, _y: Batch<u32>) -> Batch<u8> {
                Batch::<u8>::splat(self.0)
            }
        }

        let fg = Rgba::new(255, 0, 0, 255); // Red
        let bg = Rgba::new(0, 0, 255, 255); // Blue

        // Test with 0% alpha (should be pure bg)
        let composed = ConstAlpha(0).over::<Rgba, _, _>(fg, bg);
        let result: Batch<Rgba> = composed.eval(Batch::<u32>::splat(0), Batch::<u32>::splat(0));
        // SAFETY: transmute between same-sized SIMD register types
        let pixel = Rgba(unsafe { core::mem::transmute::<Batch<Rgba>, Batch<u32>>(result) }.first());
        println!(
            "\nAlpha=0 (expect pure blue bg): r={}, g={}, b={}",
            pixel.r(),
            pixel.g(),
            pixel.b()
        );
        assert_eq!(pixel.r(), 0, "With alpha=0, red should be 0");
        assert_eq!(pixel.b(), 255, "With alpha=0, blue should be 255");

        // Test with 255 alpha (should be pure fg)
        // Note: blend_math does (fg * 255 + bg * 1) / 256, so 255*255/256 = 254
        let composed = ConstAlpha(255).over::<Rgba, _, _>(fg, bg);
        let result: Batch<Rgba> = composed.eval(Batch::<u32>::splat(0), Batch::<u32>::splat(0));
        let pixel = Rgba(unsafe { core::mem::transmute::<Batch<Rgba>, Batch<u32>>(result) }.first());
        println!(
            "Alpha=255 (expect ~pure red fg): r={}, g={}, b={}",
            pixel.r(),
            pixel.g(),
            pixel.b()
        );
        assert!(
            pixel.r() >= 254,
            "With alpha=255, red should be ~255, got {}",
            pixel.r()
        );
        assert!(
            pixel.b() <= 1,
            "With alpha=255, blue should be ~0, got {}",
            pixel.b()
        );

        // Test with 128 alpha (should be ~50/50)
        let composed = ConstAlpha(128).over::<Rgba, _, _>(fg, bg);
        let result: Batch<Rgba> = composed.eval(Batch::<u32>::splat(0), Batch::<u32>::splat(0));
        let pixel = Rgba(unsafe { core::mem::transmute::<Batch<Rgba>, Batch<u32>>(result) }.first());
        println!(
            "Alpha=128 (expect ~50/50): r={}, g={}, b={}",
            pixel.r(),
            pixel.g(),
            pixel.b()
        );
        // Should be roughly 128 for both red and blue
        assert!(
            pixel.r() > 100 && pixel.r() < 150,
            "Red should be ~128, got {}",
            pixel.r()
        );
        assert!(
            pixel.b() > 100 && pixel.b() < 150,
            "Blue should be ~128, got {}",
            pixel.b()
        );
    }

    #[test]
    fn test_render_to_u32_buffer() {
        // Simulate the actual platform render path:
        // Surface -> render_pixel -> u32 buffer
        // This catches issues in the rasterizer or pixel format conversion
        use pixelflow_render::rasterizer::render_pixel;

        let mut surface: TerminalSurface<Rgba> = TerminalSurface::new(2, 1, 10, 16);

        // Red and blue cells
        surface.grid.set(
            0,
            0,
            Cell {
                ch: ' ',
                fg: Color::Rgb(0, 0, 0),
                bg: Color::Rgb(255, 0, 0), // Red
                bold: false,
                italic: false,
            },
        );
        surface.grid.set(
            1,
            0,
            Cell {
                ch: ' ',
                fg: Color::Rgb(0, 0, 0),
                bg: Color::Rgb(0, 0, 255), // Blue
                bold: false,
                italic: false,
            },
        );

        // Render to u32 buffer (20x16 pixels)
        let mut buffer = vec![0u32; 20 * 16];
        render_pixel::<Rgba, _>(&surface, &mut buffer, 20, 16);

        // Check a pixel in the red cell (center of cell 0)
        let red_pixel_idx = 8 * 20 + 5; // y=8, x=5
        let red_pixel = Rgba(buffer[red_pixel_idx]);
        println!(
            "Red cell pixel: r={}, g={}, b={}, a={}",
            red_pixel.r(),
            red_pixel.g(),
            red_pixel.b(),
            red_pixel.a()
        );
        assert_eq!(red_pixel.r(), 255, "Red cell should have r=255");
        assert_eq!(red_pixel.g(), 0, "Red cell should have g=0");
        assert_eq!(red_pixel.b(), 0, "Red cell should have b=0");

        // Check a pixel in the blue cell (center of cell 1)
        let blue_pixel_idx = 8 * 20 + 15; // y=8, x=15
        let blue_pixel = Rgba(buffer[blue_pixel_idx]);
        println!(
            "Blue cell pixel: r={}, g={}, b={}, a={}",
            blue_pixel.r(),
            blue_pixel.g(),
            blue_pixel.b(),
            blue_pixel.a()
        );
        assert_eq!(blue_pixel.r(), 0, "Blue cell should have r=0");
        assert_eq!(blue_pixel.g(), 0, "Blue cell should have g=0");
        assert_eq!(blue_pixel.b(), 255, "Blue cell should have b=255");

        // Also check the raw u32 values to see byte ordering
        println!("\nRaw u32 values:");
        println!("  Red cell:  0x{:08X}", buffer[red_pixel_idx]);
        println!("  Blue cell: 0x{:08X}", buffer[blue_pixel_idx]);

        // For RGBA little-endian: Red = 0xFF0000FF (A=FF, B=00, G=00, R=FF)
        // The low byte (R) should be 255 for red
        assert_eq!(
            buffer[red_pixel_idx] & 0xFF,
            255,
            "Red pixel low byte should be 255 (RGBA format)"
        );
    }

    #[test]
    fn test_bake_surface_preserves_colors() {
        // Bake the entire surface and verify colors are preserved
        use pixelflow_core::ops::Baked;

        let mut surface: TerminalSurface<Rgba> = TerminalSurface::new(2, 2, 10, 16);

        // Set up a 2x2 grid with different background colors
        surface.grid.set(
            0,
            0,
            Cell {
                ch: ' ',
                fg: Color::Rgb(0, 0, 0),
                bg: Color::Rgb(255, 0, 0), // Red
                bold: false,
                italic: false,
            },
        );
        surface.grid.set(
            1,
            0,
            Cell {
                ch: ' ',
                fg: Color::Rgb(0, 0, 0),
                bg: Color::Rgb(0, 255, 0), // Green
                bold: false,
                italic: false,
            },
        );
        surface.grid.set(
            0,
            1,
            Cell {
                ch: ' ',
                fg: Color::Rgb(0, 0, 0),
                bg: Color::Rgb(0, 0, 255), // Blue
                bold: false,
                italic: false,
            },
        );
        surface.grid.set(
            1,
            1,
            Cell {
                ch: ' ',
                fg: Color::Rgb(0, 0, 0),
                bg: Color::Rgb(255, 255, 255), // White
                bold: false,
                italic: false,
            },
        );

        // Bake the surface (20x32 pixels for 2x2 cells at 10x16)
        let baked: Baked<Rgba> = Baked::new(&surface, 20, 32);

        // Sample center of each cell
        let check =
            |cx: u32, cy: u32, expected_r: u8, expected_g: u8, expected_b: u8, name: &str| {
                let x_batch = Batch::<u32>::splat(cx);
                let y_batch = Batch::<u32>::splat(cy);
                let result: Batch<Rgba> = baked.eval(x_batch, y_batch);
                let pixel = Rgba(unsafe { core::mem::transmute::<Batch<Rgba>, Batch<u32>>(result) }.first());
                assert_eq!(
                    (pixel.r(), pixel.g(), pixel.b()),
                    (expected_r, expected_g, expected_b),
                    "{} cell at ({}, {}) - expected RGB({}, {}, {}), got RGB({}, {}, {})",
                    name,
                    cx,
                    cy,
                    expected_r,
                    expected_g,
                    expected_b,
                    pixel.r(),
                    pixel.g(),
                    pixel.b()
                );
            };

        check(5, 8, 255, 0, 0, "Red"); // Cell (0,0) center
        check(15, 8, 0, 255, 0, "Green"); // Cell (1,0) center
        check(5, 24, 0, 0, 255, "Blue"); // Cell (0,1) center
        check(15, 24, 255, 255, 255, "White"); // Cell (1,1) center
    }

    #[test]
    fn test_cgimage_byte_order_expectations() {
        // This test documents what the Cocoa CGImage driver expects.
        // CGImage with kCGImageAlphaPremultipliedLast (bitmap_info=1) expects:
        //   Memory bytes: [R, G, B, A] per pixel
        //
        // Our Rgba pixel format stores bytes as:
        //   Memory bytes: [R, G, B, A] (via u32::from_le_bytes/to_le_bytes)
        //   u32 value (little-endian): 0xAABBGGRR
        //
        // So if CGImage interprets our bytes correctly, colors should match.
        // If R/B are swapped visually, CGImage is interpreting as BGRA.

        use pixelflow_render::rasterizer::render_pixel;

        let mut surface: TerminalSurface<Rgba> = TerminalSurface::new(1, 1, 4, 4);
        surface.grid.set(
            0,
            0,
            Cell {
                ch: ' ',
                fg: Color::Rgb(0, 0, 0),
                bg: Color::Rgb(255, 0, 0), // Pure red
                bold: false,
                italic: false,
            },
        );

        let mut buffer = vec![0u32; 16];
        render_pixel::<Rgba, _>(&surface, &mut buffer, 4, 4);

        // Get the raw bytes as CGDataProvider would see them
        let bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(buffer.as_ptr() as *const u8, buffer.len() * 4) };

        // For a red pixel, we expect memory bytes [255, 0, 0, 255] = [R, G, B, A]
        let pixel_bytes = &bytes[0..4];
        println!(
            "Red pixel raw bytes: [{}, {}, {}, {}]",
            pixel_bytes[0], pixel_bytes[1], pixel_bytes[2], pixel_bytes[3]
        );
        println!("Expected for RGBA: [255, 0, 0, 255]");
        println!("If CGImage shows BLUE, it's interpreting as BGRA");

        assert_eq!(pixel_bytes[0], 255, "Byte 0 should be R=255");
        assert_eq!(pixel_bytes[1], 0, "Byte 1 should be G=0");
        assert_eq!(pixel_bytes[2], 0, "Byte 2 should be B=0");
        assert_eq!(pixel_bytes[3], 255, "Byte 3 should be A=255");

        // Document the u32 interpretation
        let raw_u32 = buffer[0];
        println!("Raw u32: 0x{:08X}", raw_u32);
        println!("For RGBA little-endian, red should be 0xFF0000FF");
        assert_eq!(raw_u32, 0xFF0000FF, "Red pixel u32 should be 0xFF0000FF");
    }
}
