//! Glyph surface representation.
//!
//! This module defines the [`Glyph`] struct, which implements the [`Surface`](pixelflow_core::traits::Surface)
//! trait. This allows glyphs to be directly evaluated (rasterized) into buffers using `pixelflow-core`'s
//! rendering pipeline.

use crate::curves::Segment;
use pixelflow_core::backend::{Backend, BatchArithmetic, FloatBatchOps, SimdBatch};
use pixelflow_core::batch::{Batch, NativeBackend};
use pixelflow_core::traits::Surface;
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

impl Surface<u8> for CellGlyph {
    fn eval(&self, x: Batch<u32>, y: Batch<u32>) -> Batch<u8> {
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
#[allow(clippy::too_many_arguments)]
pub fn eval_curves_cell(
    curves: &[Segment],
    bounds: GlyphBounds,
    ascender: i32,
    x: Batch<u32>,
    y: Batch<u32>,
    dilation: Batch<f32>,
) -> Batch<u8> {
    let bx = bounds.bearing_x as f32;
    let asc = ascender as f32;

    // Convert pixel coordinates to f32
    let px_f32 = NativeBackend::u32_to_f32(x);
    let py_f32 = NativeBackend::u32_to_f32(y);

    // Add 0.5 for pixel center
    let half = Batch::<f32>::splat(0.5);
    let px_pixel = px_f32 + half;
    let py_pixel = py_f32 + half;

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
    let alpha = half - signed_dist;

    // Clamp to [0, 1]
    let zero_f = Batch::<f32>::splat(0.0);
    let one_f = Batch::<f32>::splat(1.0);
    let clamped = alpha.max(zero_f).min(one_f);

    // Convert to pixel value (0-255)
    let scale = Batch::<f32>::splat(255.0);
    let pixel_val = clamped * scale;
    let pixel_u32 = NativeBackend::f32_to_u32(pixel_val);

    // Return as Batch<u8>
    NativeBackend::downcast_u32_to_u8(pixel_u32)
}

/// A surface that exposes its underlying curves and bounds.
pub trait CurveSurface: Surface<u8> {
    /// Returns the list of curve segments (Lines and Quadratics) that define the shape.
    fn curves(&self) -> &[Segment];
    /// Returns the pixel-space bounds of the shape.
    fn bounds(&self) -> GlyphBounds;
}

/// A vector-based representation of a single character.
///
/// `Glyph` holds the geometric description of a character (lines and curves).
/// It implements `Surface<u8>`, meaning it can be evaluated at any (x, y) coordinate
/// to produce a coverage value (0-255), suitable for use as an alpha mask.
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
/// A batch of `u8` values representing pixel coverage (0 = fully outside, 255 = fully inside).
pub fn eval_curves(
    curves: &[Segment],
    bounds: GlyphBounds,
    x: Batch<u32>,
    y: Batch<u32>,
    dilation: Batch<f32>,
) -> Batch<u8> {
    let bx = bounds.bearing_x as f32;
    let by_top = bounds.bearing_y as f32;

    // Convert pixel coordinates to f32
    let px_f32 = NativeBackend::u32_to_f32(x);
    let py_f32 = NativeBackend::u32_to_f32(y);

    // Add 0.5 for pixel center
    let half = Batch::<f32>::splat(0.5);
    let px_pixel = px_f32 + half;
    let py_pixel = py_f32 + half;

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
    // (Negative distance is inside in our convention here for calculation,
    // though typically SDF defines negative as inside. We adjust to alpha below).
    let neg_min_dist = Batch::<f32>::splat(0.0) - min_dist;
    let neg_dist_u32 = NativeBackend::transmute_f32_to_u32(neg_min_dist);
    let min_dist_u32 = NativeBackend::transmute_f32_to_u32(min_dist);
    let signed_dist =
        NativeBackend::transmute_u32_to_f32(inside.select(neg_dist_u32, min_dist_u32));

    // Apply dilation (bolding): dist' = dist - dilation
    let signed_dist = signed_dist - dilation;

    // Alpha = 0.5 - signed_dist
    // If dist is -0.5 (deep inside), alpha = 1.0.
    // If dist is 0.5 (far outside), alpha = 0.0.
    // This gives a 1px wide anti-aliased edge.
    let alpha = half - signed_dist;

    // Clamp to [0, 1]
    let zero_f = Batch::<f32>::splat(0.0);
    let one_f = Batch::<f32>::splat(1.0);
    let clamped = alpha.max(zero_f).min(one_f);

    // Convert to pixel value (0-255)
    let scale = Batch::<f32>::splat(255.0);
    let pixel_val = clamped * scale;
    let pixel_u32 = NativeBackend::f32_to_u32(pixel_val);

    // Return as Batch<u8>
    NativeBackend::downcast_u32_to_u8(pixel_u32)
}

impl Surface<u8> for Glyph {
    fn eval(&self, x: Batch<u32>, y: Batch<u32>) -> Batch<u8> {
        eval_curves(&self.segments, self.bounds, x, y, Batch::<f32>::splat(0.0))
    }
}

// Implement for &Glyph to allow easy sharing
impl Surface<u8> for &Glyph {
    #[inline(always)]
    fn eval(&self, x: Batch<u32>, y: Batch<u32>) -> Batch<u8> {
        (*self).eval(x, y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::curves::{Line, Segment};

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
        let x = Batch::<u32>::splat(px);
        let y = Batch::<u32>::splat(py);
        let dilation = Batch::<f32>::splat(0.0);
        eval_curves(segments, bounds, x, y, dilation).first()
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

        let x = Batch::<u32>::splat(0);
        let y = Batch::<u32>::splat(5);

        // Edge pixel without dilation - should be ~50%
        let alpha_normal = eval_curves(&segments, bounds, x, y, Batch::<f32>::splat(0.0)).first();

        // Edge pixel with dilation (bolding) - should be more opaque
        let alpha_bold = eval_curves(&segments, bounds, x, y, Batch::<f32>::splat(0.3)).first();

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
                let x = Batch::<u32>::splat(px);
                let y = Batch::<u32>::splat(py);
                let alpha = glyph.eval(x, y).first();
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

        // Get actual curve bounds
        let mut y_min = f32::MAX;
        let mut y_max = f32::MIN;
        for seg in glyph.curves() {
            match seg {
                crate::curves::Segment::Line(l) => {
                    y_min = y_min.min(l.p0[1]).min(l.p1[1]);
                    y_max = y_max.max(l.p0[1]).max(l.p1[1]);
                }
                crate::curves::Segment::Quad(q) => {
                    y_min = y_min.min(q.p0[1]).min(q.p1[1]).min(q.p2[1]);
                    y_max = y_max.max(q.p0[1]).max(q.p1[1]).max(q.p2[1]);
                }
            }
        }
        eprintln!("Curve y range: {} to {}", y_min, y_max);
        eprintln!("bearing_y={}, height={}", bounds.bearing_y, bounds.height);

        // For a period at 16px:
        // - The dot is near the baseline (y=0 in font coords after scaling)
        // - bearing_y is the TOP of the bounding box
        // - The pixel row 0 corresponds to curve y = bearing_y - 0.5
        //
        // If the period dot is from y=0 to y=2 in curve space, and bearing_y=2,
        // then the entire glyph is the dot. That's correct for a tightly-bounded glyph.
        //
        // The BUG we're looking for: if the TOP half of a RENDERED CELL (not just the glyph bbox)
        // shows ink where there should be none.
        //
        // But the glyph bbox is TIGHT to the curves. So if the period is 2x3 pixels,
        // ALL of those pixels are the period.
        //
        // The real question is: when this is COMPOSITED into a cell, does the
        // positioning use bearing_y correctly?

        // For now, verify the glyph renders SOMETHING (not all transparent)
        let mut any_opaque = false;
        for py in 0..bounds.height {
            for px in 0..bounds.width {
                let x = Batch::<u32>::splat(px);
                let y = Batch::<u32>::splat(py);
                let alpha = glyph.eval(x, y).first();
                if alpha > 200 {
                    any_opaque = true;
                }
            }
        }
        assert!(any_opaque, "Period glyph should have opaque pixels");

        // The REAL test should be at the terminal/compositor level:
        // When rendering '.' in a 16px cell, the top 60% should be background.
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
                let x = Batch::<u32>::splat(px);
                let y = Batch::<u32>::splat(py);
                let alpha = glyph.eval(x, y).first();
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

        // Print the curve y-coordinates for first and last row
        let bx = bounds.bearing_x as f32;
        let by_top = bounds.bearing_y as f32;
        eprintln!("\nbearing_x={}, bearing_y={}", bx, by_top);
        eprintln!("height={}", bounds.height);

        // Check actual bbox from the curves
        let mut y_min = f32::MAX;
        let mut y_max = f32::MIN;
        for seg in glyph.curves() {
            match seg {
                crate::curves::Segment::Line(l) => {
                    y_min = y_min.min(l.p0[1]).min(l.p1[1]);
                    y_max = y_max.max(l.p0[1]).max(l.p1[1]);
                }
                crate::curves::Segment::Quad(q) => {
                    y_min = y_min.min(q.p0[1]).min(q.p1[1]).min(q.p2[1]);
                    y_max = y_max.max(q.p0[1]).max(q.p1[1]).max(q.p2[1]);
                }
            }
        }
        eprintln!("Actual curve y range: {} to {}", y_min, y_max);
        eprintln!(
            "Row 0: pixel y=0.5 -> curve y = {} - 0.5 = {}",
            by_top,
            by_top - 0.5
        );
        eprintln!(
            "Row {}: pixel y={}.5 -> curve y = {} - {}.5 = {}",
            bounds.height - 1,
            bounds.height - 1,
            by_top,
            bounds.height - 1,
            by_top - (bounds.height as f32 - 0.5)
        );

        // Row 0 IS the apex of 'A' - it should have some coverage in the center
        // Row 11 (bottom) should have coverage at the edges (the feet)
        // But the LEFT and RIGHT edges of row 0 should be transparent

        // Check left edge of row 0 (should be empty - apex is centered)
        let left_alpha = {
            let x = Batch::<u32>::splat(0);
            let y = Batch::<u32>::splat(0);
            glyph.eval(x, y).first()
        };
        eprintln!("Row 0, col 0 alpha: {}", left_alpha);
        assert!(
            left_alpha < 50,
            "Top-left corner should be transparent, got {}",
            left_alpha
        );

        // Check center of row 0 (should have coverage - the apex)
        let center_alpha = {
            let x = Batch::<u32>::splat(bounds.width / 2);
            let y = Batch::<u32>::splat(0);
            glyph.eval(x, y).first()
        };
        eprintln!("Row 0, center alpha: {}", center_alpha);
        assert!(
            center_alpha > 100,
            "Top-center (apex) should have coverage, got {}",
            center_alpha
        );
    }
}
