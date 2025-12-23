//! Glyph representation.
//!
//! A glyph is a collection of curve segments with bounds and advance info.

use super::curves::Segment;
use pixelflow_core::{Field, Manifold, Numeric};
use std::sync::Arc;

/// Bounds and metrics for a glyph.
#[derive(Clone, Copy, Debug, Default)]
pub struct GlyphBounds {
    pub width: u32,
    pub height: u32,
    pub bearing_x: i32,
    pub bearing_y: i32,
}

/// A glyph ready for rendering.
#[derive(Clone, Debug)]
pub struct Glyph {
    pub segments: Arc<[Segment]>,
    pub bounds: GlyphBounds,
    pub advance: f32,
}

impl Manifold for Glyph {
    type Output = Field;

    fn eval_raw(&self, x: Field, y: Field, z: Field, w: Field) -> Field {
        let winding = self
            .segments
            .iter()
            .map(|s| s.eval_raw(x, y, z, w))
            .fold(Field::from(0.0), |a, b| a + b);

        // Non-zero winding rule: inside if |winding| >= 0.5
        let winding_abs = Numeric::abs(winding);
        let inside = Numeric::ge(winding_abs, Field::from(0.5));
        Numeric::select(inside, Field::from(1.0), Field::from(0.0))
    }
}

/// Trait for types that provide curve geometry (for combinators).
pub trait CurveSurface: Send + Sync {
    fn curves(&self) -> &[Segment];
    fn bounds(&self) -> GlyphBounds;
}

impl CurveSurface for Glyph {
    fn curves(&self) -> &[Segment] {
        &self.segments
    }
    fn bounds(&self) -> GlyphBounds {
        self.bounds
    }
}

/// Evaluate curves with bold offset.
pub fn eval_curves(
    curves: &[Segment],
    _bounds: GlyphBounds,
    x: Field,
    y: Field,
    bold_offset: Field,
) -> Field {
    let winding = curves
        .iter()
        .map(|s| s.eval_raw(x, y, Field::from(0.0), Field::from(0.0)))
        .fold(Field::from(0.0), |a, b| a + b);

    let winding_abs = Numeric::abs(winding);
    let threshold = Field::from(0.5) - bold_offset;
    let inside = Numeric::ge(winding_abs, threshold);
    Numeric::select(inside, Field::from(1.0), Field::from(0.0))
}

/// Evaluate curves with cell positioning (for terminal grid alignment).
pub fn eval_curves_cell(
    curves: &[Segment],
    bounds: GlyphBounds,
    ascender: i32,
    x: Field,
    y: Field,
    bold_offset: Field,
) -> Field {
    let glyph_x = x - Field::from(bounds.bearing_x as f32);
    let glyph_y = y - Field::from((ascender - bounds.bearing_y) as f32);
    eval_curves(curves, bounds, glyph_x, glyph_y, bold_offset)
}
