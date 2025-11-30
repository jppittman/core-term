//! Glyph: A Surface<u8> that lazily evaluates SDF coverage.
//!
//! The pixelflow way: Glyph IS the Surface, not a wrapper around one.

use crate::curves::Segment;
use pixelflow_core::{pipe::Surface, Batch};

/// Glyph bounds in pixel coordinates.
#[derive(Clone, Copy, Debug, Default)]
pub struct GlyphBounds {
    pub width: u32,
    pub height: u32,
    pub bearing_x: i32,
    pub bearing_y: i32,
}

/// A glyph is a Surface<u8> (coverage/alpha mask).
///
/// Holds scaled outline segments, evaluates SDF coverage on demand.
/// This is the pixelflow way: lazy evaluation, composes via combinators.
pub struct Glyph {
    pub(crate) segments: Vec<Segment>,
    pub(crate) bounds: GlyphBounds,
}

impl Glyph {
    /// Get the bounds of this glyph.
    #[inline]
    pub fn bounds(&self) -> GlyphBounds {
        self.bounds
    }
}

impl Surface<u8> for Glyph {
    fn eval(&self, x: Batch<u32>, y: Batch<u32>) -> Batch<u8> {
        let xs = x.to_array_usize();
        let ys = y.to_array_usize();
        let mut results = [0u32; 4];

        // Bounds for early-out optimization
        // Segments are normalized to [0, width] x [0, height] (relative to bbox origin)
        let w = self.bounds.width as f32;
        let h = self.bounds.height as f32;

        let min_x = -1.0;
        let max_x = w + 1.0;
        let min_y = -1.0;
        let max_y = h + 1.0;

        for i in 0..4 {
            let px = xs[i] as f32 + 0.5; // pixel center
            let py = ys[i] as f32 + 0.5;

            // Early-out: skip if outside bounds
            if px < min_x || px > max_x || py < min_y || py > max_y {
                results[i] = 0;
                continue;
            }

            let mut winding = 0;
            let mut min_signed_dist: f32 = 1000.0;

            for segment in &self.segments {
                winding += segment.winding(px, py);

                let dist = segment.signed_pseudo_distance(px, py);
                if dist.abs() < min_signed_dist.abs() {
                    min_signed_dist = dist;
                }
            }

            let inside = winding != 0;

            let signed_dist = if inside {
                -min_signed_dist.abs()
            } else {
                min_signed_dist.abs()
            };

            // Convert SDF to alpha: smooth step at distance 0.5
            let alpha = (0.5 - signed_dist).clamp(0.0, 1.0);
            results[i] = (alpha * 255.0) as u32;
        }

        Batch::new(results[0], results[1], results[2], results[3]).cast()
    }
}

// Also implement for &Glyph so it can be borrowed in combinators
impl Surface<u8> for &Glyph {
    #[inline]
    fn eval(&self, x: Batch<u32>, y: Batch<u32>) -> Batch<u8> {
        (*self).eval(x, y)
    }
}
