//! Text rendering as a category of composition.
//!
//! We map a string into a Sum of Translated, Scaled Glyphs.

use super::ttf::{Font, Glyph, Sum};
use crate::transform::Translate;
use pixelflow_core::jet::Jet2;
use pixelflow_core::{Field, Manifold};
use std::sync::Arc;

/// A Monoid representing a line of text.
///
/// It is literally just the Sum of its parts.
/// Type: Sum<Translate<Glyph>>
#[derive(Clone, Debug)]
pub struct Text {
    // We Monomorphize the scene graph to a concrete type for maximum throughput.
    // No dynamic dispatch. No VTables. Just a massive inlineable expression.
    pub inner: Sum<Translate<Glyph>>,
    pub width: f32,
}

impl Text {
    /// Bind the string to the font geometry.
    ///
    /// This is a scan (prefix sum) operation over the character stream,
    /// lifting each character into the Manifold category.
    pub fn new(font: &Font, text: &str, size: f32) -> Self {
        // The Stream: Char -> (Scaled Glyph, Scaled Advance)
        // glyph_scaled handles: scaling, Y-axis flip, and ascent offset
        let stream = text.chars().map(|ch| {
            (
                font.glyph_scaled(ch, size).unwrap_or(Glyph::Empty),
                font.advance_scaled(ch, size).unwrap_or(0.0),
            )
        });

        // The Scan: Accumulate X position
        let mut cursor = 0.0;
        let terms: Vec<_> = stream
            .map(|(glyph, advance)| {
                let pos = cursor;
                cursor += advance;

                // The Morphism: Translate the pre-scaled glyph
                Translate {
                    manifold: glyph,
                    offset: [pos, 0.0],
                }
            })
            .collect();

        // The Monoid: Sum the terms
        Self {
            inner: Sum(Arc::from(terms)),
            width: cursor,
        }
    }
}

// The Text object is itself a Manifold.
// It simply delegates to the inner Sum.
//
// We use concrete impls because Line/Quad/Segment have different
// implementations for Field (hard edges) vs Jet2 (anti-aliased).

impl Manifold<Field> for Text {
    type Output = Field;

    #[inline(always)]
    fn eval_raw(&self, x: Field, y: Field, z: Field, w: Field) -> Field {
        self.inner.eval_raw(x, y, z, w)
    }
}

impl Manifold<Jet2> for Text {
    type Output = Jet2;

    #[inline(always)]
    fn eval_raw(&self, x: Jet2, y: Jet2, z: Jet2, w: Jet2) -> Jet2 {
        self.inner.eval_raw(x, y, z, w)
    }
}
