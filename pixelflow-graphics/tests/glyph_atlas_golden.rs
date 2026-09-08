//! A golden for the production `GlyphAtlas` coverage buffer — the buffer
//! `core-term` actually gathers rendered text from (`atlas.rs:168-184`, the
//! same bake path `tests/production_glyph_arena_dump.rs` documents).
//!
//! ## What this promises, and what it deliberately does not
//!
//! **It does not promise `f32` bit identity, and cannot**: CLAUDE.md's
//! platform-divergence table is explicit that `MulAdd` is one rounding with
//! hardware FMA and two without, `Recip`/`Rsqrt` are estimates that differ
//! *between ISA levels of the same machine*, and `Round` ties away from zero
//! on aarch64 where x86 ties to even. Glyph coverage is computed through
//! exactly that arithmetic (winding sums, `Dwrt` antialiasing ramps), so a
//! bit-exact hash of the raw `f32` buffer is architecture- *and*
//! build-configuration-dependent by construction, and asserting it is stable
//! asserts something false. Measured, not assumed: dumping this exact atlas
//! at the SSE2 baseline, AVX2+FMA and AVX512F+DQ levels on one x86_64 host
//! shows up to 2630 of 42768 texels differing in the raw `f32` bytes between
//! ISA levels, worst delta ~2.1e-4 — this is *before* considering aarch64,
//! which was not available to measure directly and uses different rounding
//! modes again.
//!
//! **It promises what reaches the screen stays the same**, within the
//! platform noise the language licenses. The buffer is quantized to the same
//! truncating 8-bit form `render/pixel.rs` and `pack_rgba` use
//! (`(v.clamp(0, 1) * 255.0) as u8` — truncation, not rounding, matching
//! `cvttps`/`cvttps2dq`) and compared via [`common::assert_golden`], the same
//! tolerance-plus-mismatch-budget mechanism the scene renderers' goldens
//! already use for exactly this reason (see `tests/common/mod.rs`'s own
//! module docs). One 8-bit step is `1/255 ≈ 3.9e-3`; the deltas this PR
//! measured are ≤3.6e-5, two orders of magnitude smaller, which is the actual
//! argument for why quantizing is sound here and not merely convenient.
//!
//! **Quantization alone is not quite enough, measured**: of 2669 texels with
//! genuine (non-trivial, i.e. not exactly `0.0` or `1.0`) computed coverage,
//! the closest to a quantization boundary sits 2.15e-6 away in coverage units
//! — comfortably within the platform divergence above. Diffing the truncated
//! 8-bit bytes across the same three x86 ISA levels: 0 texels differ between
//! SSE2 and AVX2+FMA; **12 of 42768 differ by exactly 1** between either of
//! those and AVX512F+DQ, all boundary-straddling texels flipping across a
//! single quantization step. `tolerance = 1` absorbs exactly that, measured;
//! `max_mismatched_fraction = 0.01` (matching every other golden in this
//! crate) is headroom for aarch64's unmeasured but plausibly comparable
//! divergence, not a number chosen to make this pass.
//!
//! A real regression is nothing like either number, checked rather than
//! asserted: temporarily reverting `atlas.rs`'s pixel-center contramap to no
//! shift at all (`kernel.at(&X, &Y)` instead of `&(X+½), &(Y+½)`) — the exact
//! bug this PR's own history includes — fails this test at **3729/42768
//! texels (8.7%) exceeding tolerance, worst delta 255**, three orders of
//! magnitude past the 1% budget. A real regression is loud; platform noise is
//! not.
//!
//! **What this means for comparing against `origin/main`, honestly**: an
//! earlier, `f32`-bit-hash version of this golden distinguished `main` from
//! this branch by construction — any bit-level float difference changes a
//! hash — which is precisely the false precision F1 corrected, not a property
//! worth preserving. Diffing this exact buffer's *quantized* bytes between
//! `main` (`919596e5`) and this branch at the same (SSE2 baseline) build
//! configuration shows **0 of 42768 differ**: this PR's own ≤3.6e-5 delta is,
//! correctly, invisible at the quantization granularity that reaches the
//! screen. That is not this check failing to do its job; it is the corrected
//! "moves no visible pixels" claim holding up under the same instrument that
//! would catch a real one.
//!
//! ## Regenerating
//!
//! `UPDATE_GOLDENS=1 cargo test -p pixelflow-graphics --test glyph_atlas_golden`
//! (see `tests/common/mod.rs`). Regenerate at the workspace's default build
//! configuration — no `RUSTFLAGS` target-feature override, i.e. whatever
//! `.cargo/config.toml`'s `[build] rustflags` alone produces (the SSE2
//! baseline on x86_64) — so the stored golden matches what `cargo test`
//! produces by default rather than whatever `-C target-cpu`/`target-feature`
//! happened to be set on the generating machine. A committed golden that
//! depends on the generator's own flags is a trap for whoever regenerates it
//! next.
//!
//! **A known, expected collision**: a concurrent change to glyph
//! rasterization (fixing a real rendering bug a FreeType comparison caught,
//! PR #1187) also touches this buffer and moves *visible* pixels — it will
//! move this golden past its tolerance budget, correctly, since that is
//! exactly what this check is for. Not a conflict to resolve by preferring
//! one value: whichever change lands second regenerates the golden with
//! `UPDATE_GOLDENS=1` and says why in its own commit.

mod common;

use pixelflow_graphics::fonts::{Font, GlyphAtlas};
use pixelflow_graphics::render::color::Rgba8;
use pixelflow_graphics::render::frame::Frame;

const FONT_DATA: &[u8] = include_bytes!("../assets/DejaVuSansMono-Fallback.ttf");

/// The production atlas shape (`tests/production_glyph_arena_dump.rs`'s
/// `GlyphAtlas::new(cell_height, density, ATLAS_CAPACITY)` at density 1.0):
/// `tile_px = 16`, 12 slots per row, 11 rows for capacity 128, so
/// `width = 12 * (16 + 2) = 216`, `height = 11 * (16 + 2) = 198`.
const CELL_HEIGHT_PT: f32 = 16.0;
const DENSITY: f32 = 1.0;
const ATLAS_CAPACITY: usize = 128;

/// Coverage steps a texel may differ by and still count as platform noise
/// rather than a mismatch — see the module doc for the measurement behind
/// this exact value.
const TOLERANCE: u8 = 1;

/// Fraction of texels allowed to exceed `TOLERANCE` before this is treated as
/// a real regression rather than platform edge noise — matches every other
/// golden in this crate (`tests/common/mod.rs`'s callers).
const MAX_MISMATCHED_FRACTION: f64 = 0.01;

#[test]
fn glyph_atlas_coverage_is_unchanged() {
    let font = Font::parse(FONT_DATA).expect("parse font");
    let mut atlas = GlyphAtlas::new(CELL_HEIGHT_PT, DENSITY, ATLAS_CAPACITY);
    atlas.warm(&font, ' '..='~');
    let buffer = atlas.buffer();

    // The shape itself is committed structure, not just the golden: a shape
    // change would otherwise present as a wall of mismatched texels with no
    // clue which dimension moved.
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

    // Truncating quantization, matching `pack_rgba`/`render/pixel.rs` — the
    // bytes that actually reach the screen, not the f32 bits behind them.
    let data: Vec<Rgba8> = buffer
        .iter()
        .map(|&v| {
            let byte = (v.clamp(0.0, 1.0) * 255.0) as u8;
            Rgba8::new(byte, byte, byte, 255)
        })
        .collect();
    let frame = Frame::from_data(data, atlas.width() as u32, atlas.height() as u32);

    common::assert_golden(
        "glyph_atlas_coverage",
        &frame,
        TOLERANCE,
        MAX_MISMATCHED_FRACTION,
    );
}
