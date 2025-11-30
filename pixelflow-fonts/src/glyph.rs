//! Glyph: A Surface<u8> that lazily evaluates SDF coverage.

use crate::curves::Segment;
use pixelflow_core::{pipe::Surface, Batch, SimdFloatOps, SimdOps};
use std::sync::Arc;

/// Glyph bounds in pixel coordinates.
#[derive(Clone, Copy, Debug, Default)]
pub struct GlyphBounds {
    pub width: u32,
    pub height: u32,
    pub bearing_x: i32,
    pub bearing_y: i32,
}

/// A surface that exposes its underlying curves and bounds.
pub trait CurveSurface: Surface<u8> {
    fn curves(&self) -> &[Segment];
    fn bounds(&self) -> GlyphBounds;
}

/// A glyph is a Surface<u8> (coverage/alpha mask).
///
/// Holds scaled outline segments, evaluates SDF coverage on demand.
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

/// Evaluates curves at pixel coordinates to produce coverage (SDF).
///
/// `dilation`: Amount to subtract from distance (positive = bold).
pub fn eval_curves(curves: &[Segment], bounds: GlyphBounds, x: Batch<u32>, y: Batch<u32>, dilation: Batch<f32>) -> Batch<u8> {
    let bx = bounds.bearing_x as f32;
    let by_top = bounds.bearing_y as f32;

    // Coordinate mapping: pixel (0,0) -> (bx, by_top) in curve space (if Y goes UP)
    // curve_x = pixel_x + bx
    // curve_y = by_top - pixel_y

    let py_pixel = y.to_f32() + Batch::splat(0.5);
    let px_pixel = x.to_f32() + Batch::splat(0.5);

    let cx = px_pixel + Batch::splat(bx);
    let cy = Batch::splat(by_top) - py_pixel;

    let mut winding = Batch::splat(0u32);
    let mut min_dist = Batch::splat(1000.0f32);

    for segment in curves {
        winding = winding + segment.winding_batch(cx, cy);
        let d = segment.min_distance_batch(cx, cy);
        min_dist = min_dist.abs().min(d.abs());
    }

    let inside = winding.cmp_ne(Batch::splat(0));
    let signed_dist = min_dist.select(
        Batch::splat(0.0) - min_dist, // -dist if inside
        inside.transmute::<f32>()
    );

    // Apply dilation (bolding)
    // dist' = dist - dilation.
    // If dilation > 0, dist becomes smaller (more negative if inside, less positive if outside).
    // This expands the shape.
    let signed_dist = signed_dist - dilation;

    // Alpha = 0.5 - signed_dist
    let alpha = Batch::splat(0.5) - signed_dist;
    let clamped = alpha.max(Batch::splat(0.0)).min(Batch::splat(1.0));

    let pixel_val = clamped * Batch::splat(255.0);
    let pixel_u32 = pixel_val.to_u32();

    // Return as Batch<u8> - values in u32 lanes (natural SIMD layout)
    pixel_u32.transmute()
}

impl Surface<u8> for Glyph {
    fn eval(&self, x: Batch<u32>, y: Batch<u32>) -> Batch<u8> {
        eval_curves(&self.segments, self.bounds, x, y, Batch::splat(0.0))
    }
}

// Implement for &Glyph
impl Surface<u8> for &Glyph {
    #[inline(always)]
    fn eval(&self, x: Batch<u32>, y: Batch<u32>) -> Batch<u8> {
        (*self).eval(x, y)
    }
}
