//! Glyph representation and curve evaluation.
//!
//! This module provides the core types for representing glyphs as collections
//! of curve segments, along with the winding-number-based evaluation function.

use super::curves::{Line, Quadratic, Segment};
use pixelflow_core::{Field, Numeric};

/// Bounds and metrics for a glyph.
#[derive(Clone, Copy, Debug, Default)]
pub struct GlyphBounds {
    /// Width of the glyph bounding box in pixels.
    pub width: u32,
    /// Height of the glyph bounding box in pixels.
    pub height: u32,
    /// Horizontal bearing (offset from origin to left edge).
    pub bearing_x: i32,
    /// Vertical bearing (offset from baseline to top edge).
    pub bearing_y: i32,
}

/// Trait for types that provide curve geometry.
pub trait CurveSurface: Send + Sync {
    /// Get the curve segments that define this glyph's outline.
    fn curves(&self) -> &[Segment];

    /// Get the glyph bounds.
    fn bounds(&self) -> GlyphBounds;
}

// ============================================================================
// Curve Evaluation (Winding Number)
// ============================================================================

/// Evaluate winding number contribution from a line segment.
#[inline]
fn eval_line_winding(line: &Line, x: Field, y: Field) -> Field {
    let p0_y = Field::from(line.p0[1]);
    let p1_y = Field::from(line.p1[1]);
    let dy = p1_y - p0_y;

    // Skip near-horizontal lines
    let dy_abs = Numeric::abs(dy);
    let valid = Numeric::gt(dy_abs, Field::from(1e-6));

    let p0_x = Field::from(line.p0[0]);
    let p1_x = Field::from(line.p1[0]);
    let dx = p1_x - p0_x;

    // Y-range check: y >= y_min AND y <= y_max
    let y_min = Numeric::min(p0_y, p1_y);
    let y_max = Numeric::max(p0_y, p1_y);

    let geq_min = Numeric::ge(y, y_min);
    let leq_max = Numeric::le(y, y_max);
    let in_y = geq_min * leq_max; // AND via multiplication of masks

    // X-intersection: where does the line cross this y?
    let x_int = p0_x + (y - p0_y) * (dx / dy);
    let is_left = Numeric::lt(x, x_int);

    // Direction: +1 if going up, -1 if going down
    let going_up = Numeric::gt(dy, Field::from(0.0));
    let dir = Numeric::select(going_up, Field::from(1.0), Field::from(-1.0));

    // Combine: valid * in_y * is_left * dir
    let contrib = Numeric::select(is_left, dir, Field::from(0.0));
    let result = Numeric::select(in_y, contrib, Field::from(0.0));
    Numeric::select(valid, result, Field::from(0.0))
}

/// Evaluate winding number contribution from a quadratic curve.
///
/// For now, we approximate by treating it as a line from p0 to p2.
/// TODO: Implement proper quadratic curve winding.
#[inline]
fn eval_quad_winding(quad: &Quadratic, x: Field, y: Field) -> Field {
    // Approximate as line for winding (control point doesn't affect winding)
    let line = Line {
        p0: quad.p0,
        p1: quad.p2,
    };
    eval_line_winding(&line, x, y)
}

/// Evaluate the glyph at given coordinates using winding number rule.
///
/// Returns a coverage value in [0, 1] based on the winding number.
/// The `bold_offset` parameter can be used to thicken/thin the glyph.
pub fn eval_curves(
    curves: &[Segment],
    _bounds: GlyphBounds,
    x: Field,
    y: Field,
    bold_offset: Field,
) -> Field {
    let mut winding = Field::from(0.0);

    for seg in curves {
        match seg {
            Segment::Line(line) => {
                winding = winding + eval_line_winding(line, x, y);
            }
            Segment::Quad(quad) => {
                winding = winding + eval_quad_winding(quad, x, y);
            }
        }
    }

    // Non-zero winding rule: inside if |winding| >= 0.5
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
    // Transform from cell coordinates to glyph coordinates
    let glyph_x = x - Field::from(bounds.bearing_x as f32);
    let glyph_y = y - Field::from((ascender - bounds.bearing_y) as f32);

    eval_curves(curves, bounds, glyph_x, glyph_y, bold_offset)
}
