//! Text layout: a string laid out as one outline.
//!
//! A string is a scan (prefix sum) over character advances with each glyph's
//! outline placed at its pen position. **A laid-out string is one outline**,
//! and its coverage is the winding number of all those contours together —
//! the same non-zero rule TrueType already specifies *within* a glyph, which
//! is what fills a `B` drawn as two overlapping shapes.
//!
//! That one rule is load-bearing, because coverage is not additive. Summing
//! per-glyph coverages reaches 2 where two glyphs' ink overlaps, and 2 is
//! not a coverage; it also makes overlapping glyphs a different function
//! from overlapping contours, which are the same situation one level apart.
//!
//! [`text`] is ONE kernel over the whole frame: every pixel asks every
//! segment.

use super::loop_blinn::{self, Glyph};
use super::outline::Outline;
use super::ttf::Font;

/// Lay out uncached analytical text as a single coverage [`Glyph`], in raw
/// glyph space.
///
/// Advance-based (kerning-free) layout: each glyph is scaled to `size` and
/// placed at the accumulated advance, and the placed contours are **one
/// outline**. Antialiasing comes from the kernel's `Dwrt` ramps at bake.
///
/// This is the denotation — the function a laid-out string *is*. Like any
/// other glyph kernel, it carries no coordinate frame: a caller wanting
/// pixel-center sampling applies `.at(&(X + 0.5), &(Y + 0.5))` before
/// baking, same as a raw glyph kernel would — and must bind
/// [`Glyph::binding`] first, the winding sum's own piece table.
#[must_use]
pub fn text(font: &Font, text_str: &str, size: f32) -> Glyph {
    loop_blinn::glyph(&placed_outline(font, text_str, size))
}

/// Every glyph of the string, placed at its pen, as one outline.
fn placed_outline(font: &Font, text_str: &str, size: f32) -> Outline {
    let mut outline = Outline::default();
    for placed in layout(font, text_str, size) {
        outline.append(placed);
    }
    outline
}

/// The scan: every glyph's outline in the screen frame, translated to its
/// pen position, in order.
fn layout<'a>(font: &'a Font, text_str: &'a str, size: f32) -> impl Iterator<Item = Outline> + 'a {
    let mut cursor = 0.0f32;
    text_str.chars().map(move |ch| {
        // Single CMAP lookup per character.
        let id = font.cmap_lookup(ch).unwrap_or(0);
        let outline = font.outline_scaled_by_id(id, size);
        let advance = font.advance_scaled_by_id(id, size).unwrap_or(0.0);
        let pen = cursor;
        cursor += advance;
        outline
            .map(|o| o.translated([pen, 0.0]))
            .unwrap_or_default()
    })
}
