//! Glyph surface representation.
//!
//! This module defines the [`Glyph`] struct, which implements the [`Surface`](pixelflow_core::pipe::Surface)
//! trait. This allows glyphs to be directly evaluated (rasterized) into buffers using `pixelflow-core`'s
//! rendering pipeline.

use crate::curves::Segment;
use pixelflow_core::batch::{Batch, NativeBackend};
use pixelflow_core::backend::{Backend, BatchArithmetic, FloatBatchOps};
use pixelflow_core::pipe::Surface;
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

/// A trait for surfaces that are defined by a collection of vector curves.
///
/// This allows different implementations (like transformed glyphs) to expose
/// their underlying geometry for evaluation.
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
    let signed_dist = NativeBackend::transmute_u32_to_f32(inside.select(neg_dist_u32, min_dist_u32));

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
