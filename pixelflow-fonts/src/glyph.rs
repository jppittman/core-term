//! Glyph: A Surface<u8> that lazily evaluates SDF coverage.

use crate::curves::Segment;
use pixelflow_core::batch::{Batch, NativeBackend};
use pixelflow_core::backend::{Backend, SimdBatch, BatchArithmetic, FloatBatchOps};
use pixelflow_core::pipe::Surface;
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
    let signed_dist = NativeBackend::transmute_u32_to_f32(inside.select(neg_dist_u32, min_dist_u32));

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

impl Surface<u8> for Glyph {
    fn eval(&self, x: Batch<u32>, y: Batch<u32>) -> Batch<u8> {
        eval_curves(&self.segments, self.bounds, x, y, Batch::<f32>::splat(0.0))
    }
}

// Implement for &Glyph
impl Surface<u8> for &Glyph {
    #[inline(always)]
    fn eval(&self, x: Batch<u32>, y: Batch<u32>) -> Batch<u8> {
        (*self).eval(x, y)
    }
}
