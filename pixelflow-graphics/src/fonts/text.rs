//! Text layout, in both encodings of the same function.
//!
//! A string is a scan (prefix sum) over character advances with each glyph's
//! outline placed at its pen position. What differs between [`text`] and
//! [`text_union`] is not the values *given the same sampling convention* —
//! they agree to within the compiler's rounding once one is applied — but
//! *who applies it and who visits what*: [`text_union`] carries the
//! pixel-center contramap itself (see its own doc), while [`text`] does not
//! and expects its caller to supply one, same as any other `Kernel`.
//!
//! - [`text`] sums the placed glyphs into ONE kernel. Each glyph is cut to
//!   its own support by a mask, so a guard can skip it where a batch is
//!   entirely outside — but every pixel still asks every glyph, and the sum
//!   is the range encoding of the glyphs' extents.
//! - [`text_union`] puts the extents on the *domain*: the string is one
//!   outline, and [`loop_blinn::cells`] cuts the frame into a grid of
//!   disjoint index ranges, each carrying only the segments that reach it.
//!   Per-pixel cost stops depending on the string's length, and on the
//!   glyph's segment count too: it depends on how many segments pass near
//!   the pixel's cell.

use super::loop_blinn;
use super::outline::Outline;
use super::ttf::Font;
use pixelflow_core::{IndexRange, Kernel, Lattice, Union};

/// The rectangle of samples [`text_union`] decomposes a frame into.
///
/// **Argued, not measured**, which is the honest state of it: wide enough
/// that a row of a cell is a whole SIMD batch at every ISA level (lanes ride
/// X), short enough that a cell holds few segments, and a shape L3's own
/// finding pushes toward — narrow summands cost per-row call overhead
/// (measured there at 1.46× for 15-px-wide cells). A benchmark should
/// replace this reasoning; see docs/plans/2026-09-08-loop-blinn-glyph.md §8.
pub(crate) const TEXT_CELL: [usize; 2] = [16, 8];

/// Lay out uncached analytical text as a single coverage [`Kernel`], in raw
/// glyph space.
///
/// Advance-based (kerning-free) layout: each glyph is scaled to `size` and
/// placed at the accumulated advance. Antialiasing comes from the glyph
/// kernels' `Dwrt` ramps at bake.
///
/// This is the denotation — the function a laid-out string *is*. Rendering it
/// over a whole frame is what [`text_union`] does faster. Like any other
/// `Kernel`, it carries no coordinate frame: a caller wanting pixel-center
/// sampling applies `.at(&(X + 0.5), &(Y + 0.5))` before baking, same as a
/// raw glyph kernel would.
#[must_use]
pub fn text(font: &Font, text_str: &str, size: f32) -> Kernel {
    let terms: Vec<Kernel> = layout(font, text_str, size)
        .map(|outline| loop_blinn::glyph(&outline).kernel)
        .collect();
    Kernel::sum(&terms)
}

/// Lay out the same text as a [`Union`] of disjoint index ranges over
/// `lattice` — the domain-side encoding of the glyphs' extents.
///
/// The string's outline is the outlines of its glyphs, placed; the frame is
/// cut into `TEXT_CELL`-sized rectangles and each is placed with only the
/// segments that reach it ([`loop_blinn::cells`]). Rectangles no segment
/// reaches are the constant coverage of their interior, and the exterior's
/// are not placed at all — which is what the sum gives there too.
///
/// Kerning, shaping and proportional layout are not modelled here (they are
/// not modelled by [`text`] either); nothing in the construction assumes a
/// constant advance.
///
/// Each cell is placed at pixel centers — the contramap is folded into the
/// cell geometry, not applied as a `Kernel::at`, because a `Union` summand
/// collapses by pure index and `lattice` carries no coordinate frame to lend
/// it. This is the one place [`text`] and `text_union` genuinely differ in
/// *how* they reach a number (`text`'s caller applies the same contramap over
/// the whole sum instead).
#[must_use]
pub fn text_union(font: &Font, lattice: Lattice, text_str: &str, size: f32) -> Union {
    let mut union = Union::over(lattice);
    for cell in text_cells(font, lattice, text_str, size) {
        union.place(cell.range, &cell.kernel);
    }
    union
}

/// One cell of [`text_cells`]' decomposition: the samples it owns, and the
/// kernel that answers for them, already in index space.
///
/// Hidden for the same reason [`text_cells`] is: it is the decomposition's
/// own shape, exposed so a test outside this crate can bake one cell on its
/// own, not a type a consumer builds against.
#[doc(hidden)]
pub struct TextCell {
    /// The samples this cell answers for.
    pub range: IndexRange,
    /// Its kernel, sampled by index (the pixel-center convention is folded
    /// in).
    pub kernel: Kernel,
}

/// The decomposition [`text_union`] places, before it is folded into a
/// [`Union`].
///
/// **Not public API** (`#[doc(hidden)]`): this exists so
/// `tests/text_union_identity.rs`, which lives outside this crate, can bake
/// each cell in isolation and compare it against the union it came from.
/// Use [`text_union`].
#[doc(hidden)]
#[must_use]
pub fn text_cells(font: &Font, lattice: Lattice, text_str: &str, size: f32) -> Vec<TextCell> {
    if lattice.is_empty() {
        return Vec::new();
    }
    let mut outline = Outline::default();
    for placed in layout(font, text_str, size) {
        outline.append(placed);
    }
    loop_blinn::cells(&outline, lattice, TEXT_CELL)
        .into_iter()
        .map(|(range, kernel)| TextCell { range, kernel })
        .collect()
}

/// The shared scan: every glyph's outline in the screen frame, translated to
/// its pen position, in order. One definition, so the sum and the union
/// cannot lay text out differently.
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
