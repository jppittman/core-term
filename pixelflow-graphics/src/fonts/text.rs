//! Text layout, in both encodings of the same function.
//!
//! A string is a scan (prefix sum) over character advances with each glyph's
//! coverage [`Kernel`] contramapped to its pen position. What differs between
//! [`text`] and [`text_union`] is not the values *given the same sampling
//! convention* — they agree bit for bit once one is applied — but *who
//! applies it and who visits what*: [`text_union`] carries the pixel-center
//! contramap itself (see its own doc), while [`text`] does not and expects
//! its caller to supply one, same as any other `Kernel`. Baked straight over
//! a lattice with no contramap at all, the two differ by exactly that half
//! pixel.
//!
//! - [`text`] sums the placed glyphs into ONE kernel. A sum has no mask, so
//!   nothing in it can be guarded and **every pixel evaluates every glyph**.
//!   Denotationally correct and structurally blind: the disjointness of the
//!   glyphs, the one fact that makes text cheap, is nowhere in the program.
//!   This is the range encoding of the glyphs' extents, and it is what
//!   [`text_union`] is checked against.
//! - [`text_union`] puts the same extents on the *domain*: a
//!   [`Union`] of disjoint column ranges, each carrying only the glyphs that
//!   can be nonzero there. Per-pixel cost stops depending on the string's
//!   length, because the loop does not go where a glyph is zero.
//!
//! ## Why a cell can hold more than one glyph
//!
//! A glyph's kernel is exactly zero outside its [`Support`] — the unit-square
//! mask every outline is cut to, warped to screen. That box is wider than the
//! ink: the antialiasing ramp reaches past the outline by `‖∇d‖` pixels, which
//! for a nearly horizontal segment is tens of them, so a tighter box would be
//! a guess and the union would stop being bit-identical. Cells are cut at the
//! pen positions, and a cell takes every glyph whose support reaches it —
//! usually its own, sometimes a neighbour whose support spills over. The
//! *values* are unaffected either way; only the cost is.
//!
//! [`Union`]: pixelflow_core::Union
//! [`Support`]: super::ttf::Support

use super::ttf::{Font, Support};
use super::PIXEL_CENTER;
use pixelflow_core::{IndexRange, Kernel, Lattice, Union};

// `PIXEL_CENTER` (this crate's shared rasterizer convention) is used by
// `text_cells`' own column-cut math to decide which columns a glyph's
// support reaches, and by `text_union` to place each cell's kernel where a
// caller sampling at pixel centers expects it.
//
// `layout` itself stays in raw glyph space and does **not** apply it: `text`
// and `text_union` are convention-agnostic composable `Kernel` values, same
// as any other kernel in the language (`text(..).at(&(X * 0.5), &Y)` scales,
// exactly as composing any other kernel would). A caller that wants pixel
// centers applies this contramap itself — `text_union` does so once per
// cell, since a `Union` summand collapses by pure index with no coordinate
// frame of its own to lend it; `text`'s callers do the same over the whole
// sum.

/// Lay out uncached analytical text as a single coverage [`Kernel`], in raw
/// glyph space.
///
/// Advance-based (kerning-free) layout: each glyph is scaled to `size` and
/// contramapped by the accumulated advance. Antialiasing comes from the glyph
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
        .map(|placed| placed.kernel)
        .collect();
    Kernel::sum(&terms)
}

/// Lay out the same text as a [`Union`] of disjoint column ranges over
/// `lattice` — the domain-side encoding of the glyphs' extents.
///
/// Columns are cut at the pen positions, so there is one range per character;
/// each range's kernel is the sum of the glyphs whose support reaches it, in
/// the same left-to-right order [`text`] sums them in. Ranges no glyph reaches
/// are not placed at all, and collapse to 0 — which is what the sum gives
/// there too.
///
/// Kerning, shaping and proportional layout are not modelled here (they are
/// not modelled by [`text`] either); the construction does not assume a
/// constant advance, but it pays off in proportion to how well the glyphs'
/// supports stay in their own cells, which is what a monospace face
/// guarantees.
///
/// Each cell's kernel is placed at pixel centers — a `.at(&(X + ½), &(Y +
/// ½))` contramap applied once per cell here, not in `layout`, because a
/// `Union` summand collapses by pure index and `lattice` carries no
/// coordinate frame to lend it. This is the one place [`text`] and
/// `text_union` genuinely differ in *how* they reach a number (`text`'s
/// caller applies the same contramap over the whole sum instead) — they
/// still agree bit for bit given the same convention, since summing before
/// or after one shared contramap is the same value.
#[must_use]
pub fn text_union(font: &Font, lattice: Lattice, text_str: &str, size: f32) -> Union {
    let mut union = Union::over(lattice);
    for cell in text_cells(font, lattice, text_str, size) {
        let centered = Kernel::sum(&cell.glyphs).at(
            &Kernel::x().add(&Kernel::constant(PIXEL_CENTER)),
            &Kernel::y().add(&Kernel::constant(PIXEL_CENTER)),
        );
        union.place(cell.range, &centered);
    }
    union
}

/// One cell of [`text_cells`]' decomposition: the columns it owns, and the
/// placed glyph kernels that can be nonzero anywhere in them, in the order
/// [`text`] sums them.
///
/// Hidden for the same reason [`text_cells`] is: it is the decomposition's
/// own shape, exposed so a test outside this crate can see a cell's arity,
/// not a type a consumer builds against.
#[doc(hidden)]
pub struct TextCell {
    /// The columns this cell answers for, over the whole height of the
    /// frame. Cut assuming pixel-center sampling (`text_union`'s own
    /// convention) — a consumer using a different convention to place
    /// `glyphs` may cut columns slightly differently than this range does.
    pub range: IndexRange,
    /// The glyphs that can be nonzero there, in raw glyph space (unshifted,
    /// same as `layout` produces) — never empty. Reading these directly
    /// rather than through [`text_union`] means applying the pixel-center
    /// contramap yourself to land where `range` was cut for.
    pub glyphs: Vec<Kernel>,
}

/// The decomposition [`text_union`] places, before it is folded into a
/// [`Union`] — one cell per character, minus the ones no glyph reaches.
///
/// **Not public API** (`#[doc(hidden)]`): this exists so
/// `tests/text_union_identity.rs`, which lives outside this crate, can see a
/// cell's **arity**. That is the difference between a rewrite and an
/// approximation — a cell holding one glyph carries that glyph's arena
/// unchanged, while a cell holding two or more is a different arena from
/// anything standalone and the compiler may schedule it differently (see the
/// [module documentation](self)) — and a test that cannot see it cannot tell
/// the two apart. Use [`text_union`].
#[doc(hidden)]
#[must_use]
pub fn text_cells(font: &Font, lattice: Lattice, text_str: &str, size: f32) -> Vec<TextCell> {
    let columns = lattice.extent[0] as usize;
    let rows = lattice.extent[1] as usize;
    if columns == 0 || rows == 0 {
        return Vec::new();
    }
    let placed: Vec<Placed> = layout(font, text_str, size).collect();
    if placed.is_empty() {
        return Vec::new();
    }

    // Cell i is [cut[i], cut[i+1]) — the columns whose sample centers fall at
    // or after glyph i's pen and before glyph i+1's. The first cell reaches
    // the left edge and the last the right, so the cells partition the frame
    // and every column has exactly one owner.
    let center = PIXEL_CENTER;
    let mut cut: Vec<usize> = Vec::with_capacity(placed.len() + 1);
    cut.push(0);
    for p in &placed[1..] {
        let at = (p.pen - center).ceil().clamp(0.0, columns as f32) as usize;
        cut.push(at.max(*cut.last().expect("cut is never empty")));
    }
    cut.push(columns);

    let reach: Vec<Option<[i64; 2]>> = placed.iter().map(|p| p.support.columns(center)).collect();
    (0..placed.len())
        .filter_map(|cell| {
            let (x0, x1) = (cut[cell], cut[cell + 1]);
            if x0 >= x1 {
                return None;
            }
            let glyphs: Vec<Kernel> = placed
                .iter()
                .zip(&reach)
                .filter(|(_, columns)| reaches(**columns, x0, x1))
                .map(|(p, _)| p.kernel.clone())
                .collect();
            if glyphs.is_empty() {
                return None;
            }
            Some(TextCell {
                range: IndexRange::new(x0, 0, x1 - x0, rows),
                glyphs,
            })
        })
        .collect()
}

/// One glyph placed at its pen position: the contramapped coverage kernel, and
/// the box outside which that kernel is exactly zero, shifted alike.
struct Placed {
    kernel: Kernel,
    pen: f32,
    support: Support,
}

/// Whether a glyph reaching the inclusive columns `columns` can be nonzero
/// anywhere in `[x0, x1)`.
fn reaches(columns: Option<[i64; 2]>, x0: usize, x1: usize) -> bool {
    let Some([first, last]) = columns else {
        return false;
    };
    first < x1 as i64 && x0 as i64 <= last
}

/// The shared scan: glyph kernels contramapped to their pen positions, in
/// order. One definition, so the sum and the union cannot lay text out
/// differently.
fn layout<'a>(
    font: &'a Font,
    text_str: &'a str,
    size: f32,
) -> impl Iterator<Item = Placed> + use<'a> {
    let mut cursor = 0.0f32;
    text_str.chars().map(move |ch| {
        // Single CMAP lookup per character.
        let id = font.cmap_lookup(ch).unwrap_or(0);
        let glyph = font.glyph_scaled_by_id(id, size);
        let advance = font.advance_scaled_by_id(id, size).unwrap_or(0.0);
        let pen = cursor;
        cursor += advance;

        let (kernel, support) = match glyph {
            Some(g) => (g.kernel, g.support.shifted_x(pen)),
            None => (Kernel::constant(0.0), Support::EMPTY),
        };
        Placed {
            // Translate: sample the glyph at (X - pen, Y). Raw glyph space —
            // no sampling convention here; see the module docs.
            kernel: kernel.at(&Kernel::x().sub(&Kernel::constant(pen)), &Kernel::y()),
            pen,
            support,
        }
    })
}
