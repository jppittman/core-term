//! Glyph surface representation.
//!
//! This module defines the [`Glyph`] struct, which implements the [`Manifold`](pixelflow_core::traits::Manifold)
//! trait. This allows glyphs to be directly evaluated (rasterized) into buffers using `pixelflow-core`'s
//! rendering pipeline.

use crate::curves::Segment;
use pixelflow_core::backend::{Backend, BatchArithmetic, FloatBatchOps};
use pixelflow_core::batch::{Batch, NativeBackend};
use pixelflow_core::traits::Manifold;
use pixelflow_core::SimdBatch;
use std::sync::Arc;

/// The bounding box of a glyph in pixel coordinates.
///
/// Use this to determine the size of the buffer needed to render the glyph.
#[derive(Clone, Copy, Debug, Default)]
pub struct GlyphBounds {
    /// Width of the glyph bounding box in pixels.
    pub width: u32,
    /// Height of the glyph bounding box in pixels.
    pub height: u32,
    /// Horizontal offset from the cursor position to the left edge of the bounding box.
    pub bearing_x: i32,
    /// Vertical offset from the baseline to the top edge of the bounding box.
    pub bearing_y: i32,
}

/// A glyph positioned within a cell using font metrics.
///
/// This wrapper handles the coordinate transformation from cell coordinates
/// to curve coordinates, properly positioning the glyph using the font's ascender.
#[derive(Clone)]
pub struct CellGlyph {
    pub(crate) segments: Arc<[Segment]>,
    pub(crate) bounds: GlyphBounds,
    /// The font ascender in scaled pixel units. This determines where
    /// the top of the cell maps to in curve space.
    pub(crate) ascender: i32,
}

impl CellGlyph {
    /// Create a cell glyph from a glyph and font ascender (both at the same scale).
    pub fn new(glyph: Glyph, ascender: i32) -> Self {
        Self {
            segments: glyph.segments,
            bounds: glyph.bounds,
            ascender,
        }
    }
}

impl Manifold<u32, f32> for CellGlyph {
    fn eval(&self, x: Batch<f32>, y: Batch<f32>, _z: Batch<f32>, _w: Batch<f32>) -> Batch<u32> {
        eval_curves_cell(
            &self.segments,
            self.bounds,
            self.ascender,
            x,
            y,
            Batch::<f32>::splat(0.0),
        )
    }
}

/// Evaluates curves at cell coordinates, using ascender for proper baseline positioning.
///
/// Cell coordinate (0, 0) is top-left of the cell.
/// Curve coordinate y = ascender is the top of the cell.
/// The baseline is at curve y = 0.
///
/// Returns pixels with coverage in alpha channel: R=G=B=255, A=coverage.
#[allow(clippy::too_many_arguments)]
pub fn eval_curves_cell(
    curves: &[Segment],
    bounds: GlyphBounds,
    ascender: i32,
    x: Batch<f32>,
    y: Batch<f32>,
    dilation: Batch<f32>,
) -> Batch<u32> {
    let bx = bounds.bearing_x as f32;
    let asc = ascender as f32;

    // Use f32 coordinates directly
    let px_pixel = x;
    let py_pixel = y;

    // Coordinate mapping for cell:
    // - Cell (0, 0) top-left maps to curve (bearing_x, ascender)
    // - Cell y increases downward, curve y increases upward
    let bx_batch = Batch::<f32>::splat(bx);
    let asc_batch = Batch::<f32>::splat(asc);

    let cx = px_pixel + bx_batch;
    let cy = asc_batch - py_pixel;

    // Initialize winding and min distance
    let mut winding = Batch::<u32>::splat(0u32);
    let mut min_dist = Batch::<f32>::splat(1000.0f32);

    // Evaluate all curve segments
    for segment in curves {
        winding = winding + segment.winding_batch(cx, cy);
        let d = segment.min_distance_batch(cx, cy);
        min_dist = min_dist.abs().min(d.abs());
    }

    // Determine inside/outside based on winding
    let zero_u = Batch::<u32>::splat(0);
    let inside = winding.cmp_ne(zero_u);

    // signed_dist = inside ? -min_dist : min_dist
    let neg_min_dist = Batch::<f32>::splat(0.0) - min_dist;
    let neg_dist_u32 = NativeBackend::transmute_f32_to_u32(neg_min_dist);
    let min_dist_u32 = NativeBackend::transmute_f32_to_u32(min_dist);
    let signed_dist =
        NativeBackend::transmute_u32_to_f32(inside.select(neg_dist_u32, min_dist_u32));

    // Apply dilation (bolding): dist' = dist - dilation
    let signed_dist = signed_dist - dilation;

    // Alpha = 0.5 - signed_dist
    let half = Batch::<f32>::splat(0.5);
    let alpha = half - signed_dist;

    // Clamp to [0, 1]
    let zero_f = Batch::<f32>::splat(0.0);
    let one_f = Batch::<f32>::splat(1.0);
    let clamped = alpha.max(zero_f).min(one_f);

    // Convert to alpha value (0-255)
    let scale = Batch::<f32>::splat(255.0);
    let alpha_val = clamped * scale;
    let alpha_u32 = NativeBackend::f32_to_u32(alpha_val);

    // Pack as white pixel with coverage in alpha: R=G=B=255, A=coverage
    let white_rgb = Batch::<u32>::splat(0x00FFFFFF);
    let alpha_shifted = alpha_u32 << 24;
    white_rgb | alpha_shifted
}

/// A surface that exposes its underlying curves and bounds.
///
/// CurveSurface outputs `Manifold<u32>` with coverage in the alpha channel.
pub trait CurveSurface: Manifold<u32, f32> {
    /// Returns the list of curve segments (Lines and Quadratics) that define the shape.
    fn curves(&self) -> &[Segment];
    /// Returns the pixel-space bounds of the shape.
    fn bounds(&self) -> GlyphBounds;
}

/// A vector-based representation of a single character.
///
/// `Glyph` holds the geometric description of a character (lines and curves).
/// It implements `Manifold<u32>`, meaning it can be evaluated at any (x, y) coordinate
/// to produce a pixel with coverage in the alpha channel (R=G=B=255, A=coverage).
///
/// The evaluation uses an analytic signed distance field (SDF) method, ensuring
/// crisp edges and anti-aliasing at any resolution.
#[derive(Clone)]
pub struct Glyph {
    pub(crate) segments: Arc<[Segment]>,
    pub(crate) bounds: GlyphBounds,
}

impl Glyph {
    /// Get the bounds of this glyph.
    #[inline]
    pub fn bounds(&self) -> GlyphBounds {
        self.bounds
    }
}

impl CurveSurface for Glyph {
    fn curves(&self) -> &[Segment] {
        &self.segments
    }
    fn bounds(&self) -> GlyphBounds {
        self.bounds
    }
}

/// Core rasterization logic: evaluates curves at pixel coordinates to produce coverage.
///
/// This function computes the Signed Distance Field (SDF) of the shape defined by `curves`
/// at the given `x`, `y` coordinates.
///
/// # Arguments
///
/// * `curves` - The segments defining the shape.
/// * `bounds` - The bounds used for coordinate system mapping.
/// * `x`, `y` - SIMD batches of pixel coordinates.
/// * `dilation` - A value to subtract from the distance. Positive values make the shape "bolder"
///   (expanding the inside), negative values make it thinner.
///
/// # Returns
///
/// Pixels with coverage in alpha channel (R=G=B=255, A=coverage).
pub fn eval_curves(
    curves: &[Segment],
    bounds: GlyphBounds,
    x: Batch<f32>,
    y: Batch<f32>,
    dilation: Batch<f32>,
) -> Batch<u32> {
    let bx = bounds.bearing_x as f32;
    let by_top = bounds.bearing_y as f32;

    // Use f32 coordinates directly
    let px_pixel = x;
    let py_pixel = y;

    // Coordinate mapping: pixel (0,0) -> (bx, by_top) in curve space (Y goes UP)
    // The font coordinate system usually has Y up, while screen is Y down.
    // We render relative to the top-left of the bounding box.
    let bx_batch = Batch::<f32>::splat(bx);
    let by_batch = Batch::<f32>::splat(by_top);

    let cx = px_pixel + bx_batch;
    let cy = by_batch - py_pixel;

    // Initialize winding and min distance
    let mut winding = Batch::<u32>::splat(0u32);
    let mut min_dist = Batch::<f32>::splat(1000.0f32);

    // Evaluate all curve segments
    for segment in curves {
        winding = winding + segment.winding_batch(cx, cy);
        let d = segment.min_distance_batch(cx, cy);
        min_dist = min_dist.abs().min(d.abs());
    }

    // Determine inside/outside based on winding
    let zero_u = Batch::<u32>::splat(0);
    let inside = winding.cmp_ne(zero_u);

    // signed_dist = inside ? -min_dist : min_dist
    let neg_min_dist = Batch::<f32>::splat(0.0) - min_dist;
    let neg_dist_u32 = NativeBackend::transmute_f32_to_u32(neg_min_dist);
    let min_dist_u32 = NativeBackend::transmute_f32_to_u32(min_dist);
    let signed_dist =
        NativeBackend::transmute_u32_to_f32(inside.select(neg_dist_u32, min_dist_u32));

    // Apply dilation (bolding): dist' = dist - dilation
    let signed_dist = signed_dist - dilation;

    // Alpha = 0.5 - signed_dist
    let half = Batch::<f32>::splat(0.5);
    let alpha = half - signed_dist;

    // Clamp to [0, 1]
    let zero_f = Batch::<f32>::splat(0.0);
    let one_f = Batch::<f32>::splat(1.0);
    let clamped = alpha.max(zero_f).min(one_f);

    // Convert to alpha value (0-255)
    let scale = Batch::<f32>::splat(255.0);
    let alpha_val = clamped * scale;
    let alpha_u32 = NativeBackend::f32_to_u32(alpha_val);

    // Pack as white pixel with coverage in alpha: R=G=B=255, A=coverage
    let white_rgb = Batch::<u32>::splat(0x00FFFFFF);
    let alpha_shifted = alpha_u32 << 24;
    white_rgb | alpha_shifted
}

impl Manifold<u32, f32> for Glyph {
    fn eval(&self, x: Batch<f32>, y: Batch<f32>, _z: Batch<f32>, _w: Batch<f32>) -> Batch<u32> {
        eval_curves(&self.segments, self.bounds, x, y, Batch::<f32>::splat(0.0))
    }
}

// Implement for &Glyph to allow easy sharing
impl Manifold<u32, f32> for &Glyph {
    #[inline(always)]
    fn eval(&self, x: Batch<f32>, y: Batch<f32>, z: Batch<f32>, w: Batch<f32>) -> Batch<u32> {
        (*self).eval(x, y, z, w)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::curves::{Line, Segment};
    use pixelflow_core::backend::SimdBatch;

    /// Creates a 10x10 pixel square glyph from (0,0) to (10,10) in curve space.
    /// The glyph covers pixels 0-9 in both x and y.
    fn square_glyph() -> (Vec<Segment>, GlyphBounds) {
        let segments = vec![
            Segment::Line(Line {
                p0: [0.0, 0.0],
                p1: [10.0, 0.0],
            }), // bottom
            Segment::Line(Line {
                p0: [10.0, 0.0],
                p1: [10.0, 10.0],
            }), // right
            Segment::Line(Line {
                p0: [10.0, 10.0],
                p1: [0.0, 10.0],
            }), // top
            Segment::Line(Line {
                p0: [0.0, 10.0],
                p1: [0.0, 0.0],
            }), // left
        ];
        let bounds = GlyphBounds {
            width: 10,
            height: 10,
            bearing_x: 0,
            bearing_y: 10, // Top of glyph in curve space
        };
        (segments, bounds)
    }

    fn eval_pixel(segments: &[Segment], bounds: GlyphBounds, px: u32, py: u32) -> u8 {
        let x = Batch::<f32>::splat(px as f32 + 0.5);
        let y = Batch::<f32>::splat(py as f32 + 0.5);
        let dilation = Batch::<f32>::splat(0.0);
        (eval_curves(segments, bounds, x, y, dilation).first() >> 24) as u8
    }

    #[test]
    fn produces_opaque_pixels_inside_square() {
        let (segments, bounds) = square_glyph();

        // Center pixel should be fully opaque
        let alpha = eval_pixel(&segments, bounds, 5, 5);
        assert!(
            alpha > 250,
            "Center pixel should be nearly opaque, got {}",
            alpha
        );

        // Pixel at (2, 2) should also be opaque
        let alpha2 = eval_pixel(&segments, bounds, 2, 2);
        assert!(
            alpha2 > 250,
            "Pixel (2,2) should be nearly opaque, got {}",
            alpha2
        );
    }

    #[test]
    fn produces_transparent_pixels_outside_square() {
        let (segments, bounds) = square_glyph();

        // Pixel way outside should be transparent
        // Note: We need to extend bounds to test outside
        let extended_bounds = GlyphBounds {
            width: 20,
            height: 20,
            bearing_x: -5,
            bearing_y: 15,
        };

        // Pixel at (15, 5) in pixel coords -> (10.5, 9.5) in curve coords -> outside
        let alpha = eval_pixel(&segments, extended_bounds, 15, 5);
        assert!(
            alpha < 5,
            "Pixel outside should be nearly transparent, got {}",
            alpha
        );
    }

    #[test]
    fn produces_antialiased_edge_pixels() {
        // Create a square where the edge passes through pixel centers.
        // Square from (0.5, 0.5) to (9.5, 9.5) in curve space.
        // Pixel 0's center is at (0.5, ...) which is ON the left edge.
        let segments = vec![
            Segment::Line(Line {
                p0: [0.5, 0.5],
                p1: [9.5, 0.5],
            }),
            Segment::Line(Line {
                p0: [9.5, 0.5],
                p1: [9.5, 9.5],
            }),
            Segment::Line(Line {
                p0: [9.5, 9.5],
                p1: [0.5, 9.5],
            }),
            Segment::Line(Line {
                p0: [0.5, 9.5],
                p1: [0.5, 0.5],
            }),
        ];
        let bounds = GlyphBounds {
            width: 10,
            height: 10,
            bearing_x: 0,
            bearing_y: 10,
        };

        // Pixel (0, 5): center at (0.5, 4.5) in curve space
        // Left edge is at x=0.5, so this pixel is ON the edge
        let alpha = eval_pixel(&segments, bounds, 0, 5);
        // Distance should be ~0, so alpha ~0.5 -> ~127
        assert!(
            alpha > 100 && alpha < 155,
            "Edge pixel should be ~50% coverage (~127), got {}",
            alpha
        );
    }

    #[test]
    fn handles_bearing_offset_correctly() {
        // Square from (5, 5) to (15, 15) in curve space
        let segments = vec![
            Segment::Line(Line {
                p0: [5.0, 5.0],
                p1: [15.0, 5.0],
            }),
            Segment::Line(Line {
                p0: [15.0, 5.0],
                p1: [15.0, 15.0],
            }),
            Segment::Line(Line {
                p0: [15.0, 15.0],
                p1: [5.0, 15.0],
            }),
            Segment::Line(Line {
                p0: [5.0, 15.0],
                p1: [5.0, 5.0],
            }),
        ];
        let bounds = GlyphBounds {
            width: 10,
            height: 10,
            bearing_x: 5,  // x_min in curve space
            bearing_y: 15, // y_max in curve space
        };

        // Center of glyph bitmap (5, 5) -> curve space (10.5, 9.5)
        // which should be inside [5,15] x [5,15]
        let alpha = eval_pixel(&segments, bounds, 5, 5);
        assert!(
            alpha > 250,
            "Center with bearing offset should be opaque, got {}",
            alpha
        );
    }

    #[test]
    fn returns_valid_alpha_no_nan_or_garbage() {
        let (segments, bounds) = square_glyph();

        // Test a grid of pixels
        for py in 0..bounds.height {
            for px in 0..bounds.width {
                let alpha = eval_pixel(&segments, bounds, px, py);
                // Alpha should be in valid range, not garbage
                assert!(
                    alpha <= 255,
                    "Alpha out of range at ({}, {}): {}",
                    px,
                    py,
                    alpha
                );
            }
        }
    }

    #[test]
    fn dilation_increases_coverage() {
        // Square where edge passes through pixel center
        let segments = vec![
            Segment::Line(Line {
                p0: [0.5, 0.5],
                p1: [9.5, 0.5],
            }),
            Segment::Line(Line {
                p0: [9.5, 0.5],
                p1: [9.5, 9.5],
            }),
            Segment::Line(Line {
                p0: [9.5, 9.5],
                p1: [0.5, 9.5],
            }),
            Segment::Line(Line {
                p0: [0.5, 9.5],
                p1: [0.5, 0.5],
            }),
        ];
        let bounds = GlyphBounds {
            width: 10,
            height: 10,
            bearing_x: 0,
            bearing_y: 10,
        };

        let x = Batch::<f32>::splat(0.5);
        let y = Batch::<f32>::splat(5.5);

        // Edge pixel without dilation - should be ~50%
        let alpha_normal =
            (eval_curves(&segments, bounds, x, y, Batch::<f32>::splat(0.0)).first() >> 24) as u8;

        // Edge pixel with dilation (bolding) - should be more opaque
        let alpha_bold =
            (eval_curves(&segments, bounds, x, y, Batch::<f32>::splat(0.3)).first() >> 24) as u8;

        assert!(
            alpha_bold > alpha_normal,
            "Dilation should increase coverage: normal={}, bold={}",
            alpha_normal,
            alpha_bold
        );
    }

    #[test]
    fn period_top_half_is_transparent() {
        // A period glyph sits at the baseline - its top half should be empty.
        // Simulate a period: small square at bottom of glyph bounds.
        // Period is typically in bottom ~20% of the em square.
        //
        // Glyph bounds: 10x20 pixels
        // Period dot: from (2, 2) to (8, 8) in curve space (bottom portion)
        // bearing_y = 20 (top of bounds in curve space)
        //
        // Pixel row 0 (top) -> curve y = 20 - 0.5 = 19.5 (far above the dot)
        // Pixel row 10 (middle) -> curve y = 20 - 10.5 = 9.5 (above the dot)
        // Pixel row 15 -> curve y = 20 - 15.5 = 4.5 (inside the dot)

        let segments = vec![
            Segment::Line(Line {
                p0: [2.0, 2.0],
                p1: [8.0, 2.0],
            }),
            Segment::Line(Line {
                p0: [8.0, 2.0],
                p1: [8.0, 8.0],
            }),
            Segment::Line(Line {
                p0: [8.0, 8.0],
                p1: [2.0, 8.0],
            }),
            Segment::Line(Line {
                p0: [2.0, 8.0],
                p1: [2.0, 2.0],
            }),
        ];
        let bounds = GlyphBounds {
            width: 10,
            height: 20,
            bearing_x: 0,
            bearing_y: 20,
        };

        // Top half of the glyph (rows 0-9) should be fully transparent
        for py in 0..10u32 {
            for px in 0..bounds.width {
                let alpha = eval_pixel(&segments, bounds, px, py);
                assert_eq!(
                    alpha, 0,
                    "Top half pixel ({}, {}) should be transparent, got {}",
                    px, py, alpha
                );
            }
        }

        // Bottom portion should have some opaque pixels (the dot)
        let center_alpha = eval_pixel(&segments, bounds, 5, 15);
        assert!(
            center_alpha > 200,
            "Center of period dot should be opaque, got {}",
            center_alpha
        );
    }

    #[test]
    fn real_font_period_top_half_is_transparent() {
        // The period glyph is a small dot near the baseline.
        // The top half of the glyph bounds should be completely transparent.
        use crate::font::Font;
        use crate::glyph::CurveSurface;
        use pixelflow_core::traits::Surface;

        let font_bytes = include_bytes!("../assets/NotoSansMono-Regular.ttf");
        let font = Font::from_bytes(font_bytes).expect("Failed to load font");

        let glyph = font.glyph('.', 16.0).expect("No period glyph");
        let bounds = glyph.bounds();

        eprintln!("Period bounds: {:?}", bounds);

        // Render the whole glyph for debugging
        eprintln!("Period bitmap (top=row0):");
        for py in 0..bounds.height {
            let mut row = String::new();
            for px in 0..bounds.width {
                let x = Batch::<f32>::splat(px as f32 + 0.5);
                let y = Batch::<f32>::splat(py as f32 + 0.5);
                let alpha = (glyph.eval(x, y).first() >> 24) as u8;
                let ch = if alpha > 200 {
                    '#'
                } else if alpha > 100 {
                    '+'
                } else if alpha > 50 {
                    '.'
                } else if alpha > 0 {
                    ','
                } else {
                    ' '
                };
                row.push(ch);
            }
            eprintln!("{:2}: |{}|", py, row);
        }

        // ... rest of test logic (doesn't use eval) ...

        // For now, verify the glyph renders SOMETHING (not all transparent)
        let mut any_opaque = false;
        for py in 0..bounds.height {
            for px in 0..bounds.width {
                let x = Batch::<f32>::splat(px as f32 + 0.5);
                let y = Batch::<f32>::splat(py as f32 + 0.5);
                let alpha = (glyph.eval(x, y).first() >> 24) as u8;
                if alpha > 200 {
                    any_opaque = true;
                }
            }
        }
        assert!(any_opaque, "Period glyph should have opaque pixels");
    }

    #[test]
    fn real_font_letter_a_top_is_transparent() {
        // Use 'A' - the top of the letter should be transparent (above the apex)
        use crate::font::Font;
        use crate::glyph::CurveSurface;
        use pixelflow_core::traits::Surface;

        let font_bytes = include_bytes!("../assets/NotoSansMono-Regular.ttf");
        let font = Font::from_bytes(font_bytes).expect("Failed to load font");

        let glyph = font.glyph('A', 16.0).expect("No A glyph");
        let bounds = glyph.bounds();

        eprintln!("'A' bounds: {:?}", bounds);

        // Render the whole glyph and print it
        eprintln!("Glyph bitmap (top=row0):");
        for py in 0..bounds.height {
            let mut row = String::new();
            for px in 0..bounds.width {
                let x = Batch::<f32>::splat(px as f32 + 0.5);
                let y = Batch::<f32>::splat(py as f32 + 0.5);
                let alpha = (glyph.eval(x, y).first() >> 24) as u8;
                let ch = if alpha > 200 {
                    '#'
                } else if alpha > 100 {
                    '+'
                } else if alpha > 50 {
                    '.'
                } else if alpha > 0 {
                    ','
                } else {
                    ' '
                };
                row.push(ch);
            }
            eprintln!("{:2}: |{}|", py, row);
        }

        // ...

        // Check left edge of row 0 (should be empty - apex is centered)
        let left_alpha = {
            let x = Batch::<f32>::splat(0.5);
            let y = Batch::<f32>::splat(0.5);
            (glyph.eval(x, y).first() >> 24) as u8
        };
        eprintln!("Row 0, col 0 alpha: {}", left_alpha);
        assert!(
            left_alpha < 50,
            "Top-left corner should be transparent, got {}",
            left_alpha
        );

        // Check center of row 0 (should have coverage - the apex)
        let center_alpha = {
            let x = Batch::<f32>::splat(bounds.width as f32 / 2.0 + 0.5);
            let y = Batch::<f32>::splat(0.5);
            (glyph.eval(x, y).first() >> 24) as u8
        };
        eprintln!("Row 0, center alpha: {}", center_alpha);
        assert!(
            center_alpha > 100,
            "Top-center (apex) should have coverage, got {}",
            center_alpha
        );
    }

    fn eval_cell_pixel(
        segments: &[Segment],
        bounds: GlyphBounds,
        ascender: i32,
        px: u32,
        py: u32,
    ) -> u8 {
        let x = Batch::<f32>::splat(px as f32 + 0.5);
        let y = Batch::<f32>::splat(py as f32 + 0.5);
        let dilation = Batch::<f32>::splat(0.0);
        (eval_curves_cell(segments, bounds, ascender, x, y, dilation).first() >> 24) as u8
    }

    #[test]
    fn cell_bottom_pixel_not_clipped() {
        // Simulate a glyph with a descender that extends to the bottom of a cell.
        // Cell: 16 pixels tall, ascender at y=12, descender at y=-4
        // The glyph is a rectangle from y=-4 to y=12 (full height of the cell).
        let cell_height = 16u32;
        let ascender = 12i32;
        let descender = -4i32;

        // Create a vertical bar that spans the full cell height
        let segments = vec![
            Segment::Line(Line { p0: [2.0, descender as f32], p1: [8.0, descender as f32] }),
            Segment::Line(Line { p0: [8.0, descender as f32], p1: [8.0, ascender as f32] }),
            Segment::Line(Line { p0: [8.0, ascender as f32], p1: [2.0, ascender as f32] }),
            Segment::Line(Line { p0: [2.0, ascender as f32], p1: [2.0, descender as f32] }),
        ];
        let bounds = GlyphBounds {
            width: 10,
            height: cell_height,
            bearing_x: 0,
            bearing_y: ascender,
        };

        // The center of the bar should be opaque at every row
        let center_x = 5u32;

        // Top pixel (row 0) - should be opaque since bar extends to y=12
        let top_alpha = eval_cell_pixel(&segments, bounds, ascender, center_x, 0);
        assert!(
            top_alpha > 200,
            "Top pixel should be opaque, got {}",
            top_alpha
        );

        // Middle pixel - definitely opaque
        let mid_alpha = eval_cell_pixel(&segments, bounds, ascender, center_x, 8);
        assert!(
            mid_alpha > 200,
            "Middle pixel should be opaque, got {}",
            mid_alpha
        );

        // Bottom pixel (row 15) - should be opaque since bar extends to y=-4
        let bottom_alpha = eval_cell_pixel(&segments, bounds, ascender, center_x, cell_height - 1);
        assert!(
            bottom_alpha > 200,
            "Bottom pixel should be opaque, got {} (this indicates descender clipping)",
            bottom_alpha
        );
    }

    #[test]
    fn real_font_descender_not_clipped() {
        // Test that 'g' (which has a descender) renders fully within a cell.
        use crate::combinators::glyphs;
        use crate::font::Font;
        use pixelflow_core::surfaces::Baked;
        use pixelflow_core::traits::Surface;

        let font_bytes = include_bytes!("../assets/NotoSansMono-Regular.ttf");
        let font = Font::from_bytes(font_bytes).expect("Failed to load font");

        // Print font metrics
        let metrics = font.metrics();
        eprintln!("Font metrics: ascent={}, descent={}, line_gap={}, units_per_em={}",
            metrics.ascent, metrics.descent, metrics.line_gap, metrics.units_per_em);

        let line_height = metrics.ascent as f32 - metrics.descent as f32;
        eprintln!("Line height (ascent - descent): {}", line_height);

        let cell_w = 10u32;
        let cell_h = 20u32;

        let ascender = (metrics.ascent as f32 * cell_h as f32 / line_height).round() as i32;
        eprintln!("Calculated ascender for {}px cell: {}", cell_h, ascender);
        eprintln!("Calculated descender (ascender - cell_h): {}", ascender - cell_h as i32);

        let glyph_fn = glyphs(font.clone(), cell_w, cell_h);
        let baked: Baked<u32> = glyph_fn('g').get().clone();

        eprintln!("\n'g' rendered in {}x{} cell:", cell_w, cell_h);
        for py in 0..cell_h {
            let mut row = String::new();
            for px in 0..cell_w {
                let x = Batch::<u32>::splat(px);
                let y = Batch::<u32>::splat(py);
                let alpha = (baked.eval(x, y).first() >> 24) as u8;
                let ch = if alpha > 200 {
                    '#'
                } else if alpha > 100 {
                    '+'
                } else if alpha > 50 {
                    '.'
                } else if alpha > 0 {
                    ','
                } else {
                    ' '
                };
                row.push(ch);
            }
            eprintln!("{:2}: |{}|", py, row);
        }

        // The descender of 'g' should have pixels in the bottom portion of the cell.
        // Check that there's some coverage in the bottom 25% of the cell.
        let mut bottom_coverage = 0u32;
        let bottom_start = cell_h * 3 / 4; // Bottom 25%
        for py in bottom_start..cell_h {
            for px in 0..cell_w {
                let x = Batch::<u32>::splat(px);
                let y = Batch::<u32>::splat(py);
                let alpha = (baked.eval(x, y).first() >> 24) as u8;
                if alpha > 50 {
                    bottom_coverage += 1;
                }
            }
        }

        eprintln!("Bottom 25% coverage count: {}", bottom_coverage);
        assert!(
            bottom_coverage > 0,
            "Descender of 'g' should have pixels in bottom 25% of cell"
        );

        // Also verify the very bottom row has some coverage (if the descender extends there)
        let mut last_row_coverage = 0u32;
        for px in 0..cell_w {
            let x = Batch::<u32>::splat(px);
            let y = Batch::<u32>::splat(cell_h - 1);
            let alpha = (baked.eval(x, y).first() >> 24) as u8;
            if alpha > 0 {
                last_row_coverage += 1;
            }
        }
        eprintln!("Last row coverage count: {}", last_row_coverage);
    }

    #[test]
    fn glyph_line_height_fits_cell() {
        // The glyph's line height (ascent - descent) should fit within the cell.
        // Glyphs must be scaled by cell_h * units_per_em / line_height, not just cell_h.
        use crate::font::Font;

        let font_bytes = include_bytes!("../assets/NotoSansMono-Regular.ttf");
        let font = Font::from_bytes(font_bytes).expect("Failed to load font");

        let metrics = font.metrics();
        let line_height = (metrics.ascent - metrics.descent) as f32;
        let units_per_em = metrics.units_per_em as f32;

        eprintln!("line_height={}, units_per_em={}", line_height, units_per_em);

        let cell_h = 20u32;

        // Calculate the correct glyph size (as done in combinators.rs)
        let glyph_size = cell_h as f32 * units_per_em / line_height;
        eprintln!("Correct glyph_size for {}px cell: {}", cell_h, glyph_size);

        let glyph = font.glyph('g', glyph_size).expect("No 'g' glyph");
        let bounds = glyph.bounds();

        eprintln!("'g' bounds at corrected size: {:?}", bounds);
        eprintln!("bearing_y (top of glyph in curve space): {}", bounds.bearing_y);

        // Calculate where the bottom of the glyph is in curve space
        let glyph_bottom = bounds.bearing_y as i32 - bounds.height as i32;
        eprintln!("Glyph bottom in curve space: {}", glyph_bottom);

        // Calculate the cell's sampling range (using floor to match combinators.rs)
        let ascender = (metrics.ascent as f32 * cell_h as f32 / line_height).floor() as i32;
        let cell_top_sample = ascender as f32 - 0.5; // pixel 0 center
        let cell_bottom_sample = ascender as f32 - (cell_h as f32 - 0.5); // pixel (h-1) center

        eprintln!("Cell ascender: {}", ascender);
        eprintln!("Cell samples from curve y={} to y={}", cell_top_sample, cell_bottom_sample);

        // The glyph bottom should be within the cell's sampling range
        assert!(
            glyph_bottom as f32 >= cell_bottom_sample,
            "Glyph bottom ({}) is below cell sampling range ({}), causing clipping!",
            glyph_bottom,
            cell_bottom_sample
        );
    }
}
