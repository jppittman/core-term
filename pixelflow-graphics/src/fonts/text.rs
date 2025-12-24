//! Text rendering helpers.
//!
//! Provides the `Text` manifold for rendering strings.

use super::ttf::{Affine, Font, Glyph, Sum};
use pixelflow_core::{Manifold, Numeric};

/// A manifold representing a line of text.
#[derive(Clone)]
pub struct Text(pub Glyph);

impl Text {
    /// Create a new Text manifold from a string.
    pub fn new(font: &Font, text: &str, size: f32) -> Self {
        let mut glyphs = Vec::new();
        let mut cursor_x = 0.0f32;
        let mut prev_char = None;

        for ch in text.chars() {
            // Apply kerning
            if let Some(prev) = prev_char {
                cursor_x += font.kern(prev, ch, size);
            }

            // Retrieve glyph
            if let Some(g) = font.glyph_scaled(ch, size) {
                // Translate glyph to cursor position
                glyphs.push(Affine::new(g, [1.0, 0.0, 0.0, 1.0, cursor_x, 0.0]));
            }

            // Advance cursor
            if let Some(adv) = font.advance_scaled(ch, size) {
                cursor_x += adv;
            }

            prev_char = Some(ch);
        }

        Self(Glyph::Compound(Sum(glyphs.into())))
    }

    /// Get the width of the text in pixels.
    pub fn width(font: &Font, text: &str, size: f32) -> f32 {
        let mut w = 0.0f32;
        let mut prev = None;
        for ch in text.chars() {
            if let Some(p) = prev {
                w += font.kern(p, ch, size);
            }
            if let Some(adv) = font.advance_scaled(ch, size) {
                w += adv;
            }
            prev = Some(ch);
        }
        w
    }
}

impl<I: Numeric> Manifold<I> for Text {
    type Output = I;
    fn eval_raw(&self, x: I, y: I, z: I, w: I) -> I {
        self.0.eval_raw(x, y, z, w)
    }
}
