//! Font combinators using affine transforms.

use super::ttf::{affine, Glyph, Sum};

/// Extension trait for glyph transforms.
pub trait GlyphExt: Sized {
    /// Apply uniform scale.
    fn scale(self, factor: f32) -> Glyph;

    /// Apply shear (italic/slant).
    fn slant(self, factor: f32) -> Glyph;

    /// Apply 2x2 matrix transform.
    fn transform(self, m: [f32; 4]) -> Glyph;

    /// Apply translation.
    fn translate(self, dx: f32, dy: f32) -> Glyph;
}

impl GlyphExt for Glyph {
    fn scale(self, factor: f32) -> Glyph {
        Glyph::Compound(Sum(
            [affine(self, [factor, 0.0, 0.0, factor, 0.0, 0.0])].into(),
        ))
    }

    fn slant(self, factor: f32) -> Glyph {
        // Shear in x based on y: x' = x + y*factor
        Glyph::Compound(Sum(
            [affine(self, [1.0, factor, 0.0, 1.0, 0.0, 0.0])].into(),
        ))
    }

    fn transform(self, [a, b, c, d]: [f32; 4]) -> Glyph {
        Glyph::Compound(Sum(
            [affine(self, [a, b, c, d, 0.0, 0.0])].into(),
        ))
    }

    fn translate(self, dx: f32, dy: f32) -> Glyph {
        Glyph::Compound(Sum(
            [affine(self, [1.0, 0.0, 0.0, 1.0, dx, dy])].into(),
        ))
    }
}
