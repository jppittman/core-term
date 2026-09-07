//! A committed hash of the production `GlyphAtlas` coverage buffer — the
//! buffer `core-term` actually gathers rendered text from
//! (`atlas.rs:168-184`, the same bake path
//! `tests/production_glyph_arena_dump.rs` documents).
//!
//! **This is a change-detector, not an oracle.** The hash below records what
//! this bake path produced when this test was last regenerated, not a value
//! independently derived from font metrics or a second rasterizer — a
//! sibling effort in this session established, with an independent
//! rasterizer, that a golden corpus silently encodes whatever the code did
//! when it was minted, and this is no different. What it *is* good for:
//! nothing else in this workspace compares rendered bytes across revisions.
//! Every other font test in this crate bounds a property (coverage stays in
//! `[0, 1]`, enough ink renders, a value matches within a tolerance) rather
//! than pinning the bytes, so a change that stays within every one of those
//! bounds but still moves every pixel ships silently. That is not
//! hypothetical: the change that added this test moved this exact buffer by
//! up to 3.6e-5 in coverage (below one 8-bit quantization step, `1/255 ≈
//! 3.9e-3`, so nothing visible changed) by moving a pixel-center `+0.5` from
//! a runtime origin into the kernel arena, where the e-graph reassociates it
//! — `(X + 0.5) − pen` at `X = i` is not the same float expression as
//! `X − pen` at `X = i + 0.5`. Two same-form checks in this same change
//! (JIT vs interpreter on one arena; `text` vs `text_union` sharing one
//! placement) could not see it, because both sides of each moved together.
//! Per CLAUDE.md, "a same-form check cannot see a shared-definition bug;
//! only an external bound can" — this hash is that external bound, at the
//! coarsest possible grain (did the bytes change at all), and it is the
//! check that was missing.
//!
//! **A known collision.** A concurrent change to glyph rasterization (fixing
//! a real rendering bug the FreeType comparison caught) will also move this
//! hash. That is not a conflict to resolve by preferring one value — both
//! changes are real and their effects compose — so whichever change lands
//! second regenerates the constant below and says why in its own commit.
//!
//! Regenerating: run this test, take the hash it reports as `left`, and
//! paste it in below with a note on why it moved.

use pixelflow_graphics::fonts::{Font, GlyphAtlas};

const FONT_DATA: &[u8] = include_bytes!("../assets/DejaVuSansMono-Fallback.ttf");

/// The production atlas shape (`tests/production_glyph_arena_dump.rs`'s
/// `GlyphAtlas::new(cell_height, density, ATLAS_CAPACITY)` at density 1.0):
/// `tile_px = 16`, 12 slots per row, 11 rows for capacity 128, so
/// `width = 12 * (16 + 2) = 216`, `height = 11 * (16 + 2) = 198`.
const CELL_HEIGHT_PT: f32 = 16.0;
const DENSITY: f32 = 1.0;
const ATLAS_CAPACITY: usize = 128;

/// FNV-1a 64, byte at a time.
fn fnv1a64(bytes: &[u8]) -> u64 {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut hash = OFFSET;
    for &b in bytes {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(PRIME);
    }
    hash
}

#[test]
fn glyph_atlas_coverage_is_unchanged() {
    let font = Font::parse(FONT_DATA).expect("parse font");
    let mut atlas = GlyphAtlas::new(CELL_HEIGHT_PT, DENSITY, ATLAS_CAPACITY);
    atlas.warm(&font, ' '..='~');
    let buffer = atlas.buffer();

    // The shape itself is committed structure, not just the hash: a shape
    // change would otherwise present as an opaque hash mismatch with no clue
    // which dimension moved.
    assert_eq!(
        atlas.width(),
        216,
        "atlas width drifted from the committed shape"
    );
    assert_eq!(
        atlas.height(),
        198,
        "atlas height drifted from the committed shape"
    );
    assert_eq!(buffer.len(), 216 * 198);

    let bytes: Vec<u8> = buffer.iter().flat_map(|f| f.to_le_bytes()).collect();
    let hash = fnv1a64(&bytes);
    assert_eq!(
        format!("{hash:016x}"),
        "deba52394189ba02",
        "GlyphAtlas coverage bytes changed — see this test's module doc for what \
         that means and how to regenerate this constant"
    );
}
