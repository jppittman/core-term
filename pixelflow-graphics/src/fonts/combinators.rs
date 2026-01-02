//! Font combinators using affine transforms.

use super::ttf::{affine, Glyph, Sum};

/// Extension trait for glyph transforms.
pub trait GlyphExt<L, Q>: Sized {
    /// Apply uniform scale.
    fn scale(self, factor: f32) -> Glyph<L, Q>;

    /// Apply shear (italic/slant).
    fn slant(self, factor: f32) -> Glyph<L, Q>;

    /// Apply 2x2 matrix transform.
    fn transform(self, m: [f32; 4]) -> Glyph<L, Q>;

    /// Apply translation.
    fn translate(self, dx: f32, dy: f32) -> Glyph<L, Q>;
}

impl<L, Q> GlyphExt<L, Q> for Glyph<L, Q> {
    fn scale(self, factor: f32) -> Glyph<L, Q> {
        Glyph::Compound(Sum(
            [affine(self, [factor, 0.0, 0.0, factor, 0.0, 0.0])].into()
        ))
    }

    fn slant(self, factor: f32) -> Glyph<L, Q> {
        // Shear in x based on y: x' = x + y*factor
        Glyph::Compound(Sum([affine(self, [1.0, factor, 0.0, 1.0, 0.0, 0.0])].into()))
    }

    fn transform(self, [a, b, c, d]: [f32; 4]) -> Glyph<L, Q> {
        Glyph::Compound(Sum([affine(self, [a, b, c, d, 0.0, 0.0])].into()))
    }

    fn translate(self, dx: f32, dy: f32) -> Glyph<L, Q> {
        Glyph::Compound(Sum([affine(self, [1.0, 0.0, 0.0, 1.0, dx, dy])].into()))
    }
}
