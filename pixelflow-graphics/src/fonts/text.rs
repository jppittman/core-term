//! Text rendering helpers.
//!
//! Provides the `Text` manifold for rendering strings.

use super::font::Font;
use super::loopblinn::Glyph;
use pixelflow_core::{Field, Manifold};

/// A manifold representing a line of text.
#[derive(Clone, Debug)]
pub struct Text {
    glyphs: Vec<(Glyph, f32)>, // Glyph and its X position
    pub width: f32,
}

impl Text {
    /// Create a new Text manifold from a string.
    pub fn new(font: &Font, text: &str, size: f32) -> Self {
        let mut glyphs = Vec::new();
        let mut cursor_x = 0.0;
        let mut prev_char = None;

        for ch in text.chars() {
            // Apply kerning
            if let Some(prev) = prev_char {
                cursor_x += font.kern(prev, ch, size);
            }

            // Retrieve glyph
            if let Some(glyph) = font.glyph(ch, size) {
                // Use the glyph's advance to update cursor
                let advance = glyph.advance;
                glyphs.push((glyph, cursor_x));
                cursor_x += advance;
            }

            prev_char = Some(ch);
        }

        Self {
            glyphs,
            width: cursor_x,
        }
    }
}

// DIAGNOSTIC: Commented out to investigate trait conflict
// impl Manifold for Text {
//     type Output = Field;
//
//     fn eval_raw(&self, x: Field, y: Field, z: Field, w: Field) -> Field {
//         let mut sum = Field::from(0.0);
//
//         // Naive iteration over all glyphs in the line.
//         // For production use, a spatial partition or checking bounding box overlap
//         // (x range) would be preferred.
//         for (glyph, pos_x) in &self.glyphs {
//             // Translate x into glyph's local space (glyph is at 0,0 locally)
//             // local_x = x - pos_x
//             let local_x = x - Field::from(*pos_x);
//
//             // Eval glyph
//             let val = glyph.eval_raw(local_x, y, z, w);
//             sum = sum + val;
//         }
//
//         // Clamp to 0..1
//         sum.min(Field::from(1.0)).max(Field::from(0.0))
//     }
// }
