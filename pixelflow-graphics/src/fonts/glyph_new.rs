//! Glyph assembly and manifold composition.
//!
//! Glyphs are composed of analytical curves (lines and quadratics).
//! All curves are evaluated in parallel via SIMD.
//! Derivatives are precomputed polynomials - no Jets needed.

use super::curve::{LineKernel, QuadKernel};
use pixelflow_core::{Field, Manifold, ManifoldExt};
use std::sync::Arc;

/// The standard 4D Field domain type.
type Field4 = (Field, Field, Field, Field);

// ═══════════════════════════════════════════════════════════════════════════
// Glyph Geometry
// ═══════════════════════════════════════════════════════════════════════════

/// A glyph's geometry: collection of analytical curves.
///
/// Separates lines and quads to avoid enum dispatch overhead.
/// All curves are precomputed with Loop-Blinn coefficients.
#[derive(Clone)]
pub struct GlyphGeometry {
    pub lines: Arc<[LineKernel]>,
    pub quads: Arc<[QuadKernel]>,
}

impl GlyphGeometry {
    /// Create empty geometry.
    #[inline]
    pub fn empty() -> Self {
        Self {
            lines: Arc::new([]),
            quads: Arc::new([]),
        }
    }

    /// Create geometry from curves.
    #[inline]
    pub fn new(lines: Vec<LineKernel>, quads: Vec<QuadKernel>) -> Self {
        Self {
            lines: lines.into(),
            quads: quads.into(),
        }
    }
}

impl Manifold<Field4> for GlyphGeometry {
    type Output = Field;

    #[inline(always)]
    fn eval(&self, p: Field4) -> Field {
        let (x, y, z, w) = p;
        let fzero = Field::from(0.0);

        let line_sum = if self.lines.is_empty() {
            fzero
        } else {
            self.lines.iter().fold(fzero, |acc, line| {
                let val = line.eval_raw(x, y, z, w);
                (acc + val).eval_raw(fzero, fzero, fzero, fzero)
            })
        };

        let quad_sum = if self.quads.is_empty() {
            fzero
        } else {
            self.quads.iter().fold(fzero, |acc, quad| {
                let val = quad.eval_raw(x, y, z, w);
                (acc + val).eval_raw(fzero, fzero, fzero, fzero)
            })
        };

        (line_sum + quad_sum).eval_raw(fzero, fzero, fzero, fzero)
    }
}


// ═══════════════════════════════════════════════════════════════════════════
// Glyph with Transform
// ═══════════════════════════════════════════════════════════════════════════

/// A glyph: geometry + affine transform.
///
/// The transform maps from screen space to glyph's unit space.
#[derive(Clone)]
pub struct Glyph {
    pub geometry: GlyphGeometry,
    pub transform: [f32; 6], // [a, b, c, d, tx, ty]
}

impl Glyph {
    /// Create a glyph with identity transform.
    #[inline]
    pub fn new(geometry: GlyphGeometry) -> Self {
        Self {
            geometry,
            transform: [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        }
    }

    /// Create a glyph with custom transform.
    #[inline]
    pub fn with_transform(geometry: GlyphGeometry, transform: [f32; 6]) -> Self {
        Self {
            geometry,
            transform,
        }
    }

    /// Apply affine transform to coordinates, then evaluate geometry.
    #[inline]
    fn transform_point(&self, x: Field, y: Field) -> (Field, Field) {
        let [a, b, c, d, tx, ty] = self.transform;

        let det = a * d - b * c;
        let inv_det = if det.abs() < 1e-6 { 0.0 } else { 1.0 / det };

        let inv_a = d * inv_det;
        let inv_b = -b * inv_det;
        let inv_c = -c * inv_det;
        let inv_d = a * inv_det;

        let x_t = (x.constant() - tx) * inv_a + (y.constant() - ty) * inv_b;
        let y_t = (x.constant() - tx) * inv_c + (y.constant() - ty) * inv_d;

        (Field::from(x_t.to_f32()), Field::from(y_t.to_f32()))
    }
}

impl Manifold<Field4> for Glyph {
    type Output = Field;

    #[inline(always)]
    fn eval(&self, p: Field4) -> Field {
        let (x, y, z, w) = p;
        let (x_t, y_t) = self.transform_point(x, y);
        self.geometry.eval((x_t, y_t, z, w))
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use pixelflow_core::ManifoldCompat;

    #[test]
    fn test_empty_glyph() {
        let geo = GlyphGeometry::empty();
        let glyph = Glyph::new(geo);
        let p = (
            Field::from(0.5),
            Field::from(0.5),
            Field::from(0.0),
            Field::from(0.0),
        );
        let result = glyph.eval(p);
        assert!(result.to_f32().abs() < 1e-6);
    }
}
