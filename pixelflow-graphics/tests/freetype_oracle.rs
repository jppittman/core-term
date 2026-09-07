//! An **external** check: our glyph coverage against FreeType's.
//!
//! Every other guard in this crate compares us to ourselves — the JIT against
//! the interpreter on one arena, the optimized arena against the raw one, this
//! release against a corpus minted from the last one. CLAUDE.md names the
//! limit of that: *a same-form check cannot see a shared-definition bug; only
//! an external bound can.* A regression corpus is a change-detector, not an
//! oracle, and it silently encodes whatever the code did on the day it was
//! minted. This suite exists because that is not a hypothetical here — a
//! spurious half-covered smear four texels wide sat outside `'8'` at 19 px,
//! agreed on by our raw arena, our goldens and our corpus, and was found only
//! by asking a different rasterizer.
//!
//! **What it covers, exactly.** It evaluates `eval_scalar` on the *raw
//! lowered* arena: no optimizer, no JIT. So it is an external bound on the IR
//! interpreter's reading of the unoptimized arena, and it reaches real pixels
//! only transitively — `kernel_glyph_optimize` ties the optimized arena to the
//! raw one, and `kernel_glyph_golden` ties the JIT to the interpreter. A
//! miscompile that this suite cannot see is one of those two tests' business.
//!
//! **The assertion is topological, not photometric.** FreeType's antialiasing
//! convention is its own and need not equal ours; comparing coverage values
//! would fail for reasons that are not bugs. What must agree is *where the ink
//! is*: wherever we mark a texel covered, an independent rasterizer must find
//! ink within one texel of it. A spurious crossing puts ink where FreeType has
//! none at all (`0.000` across several texels); a difference of AA convention
//! never does.
//!
//! **This is a blocking check, and it depends on a vendored rasterizer.** It
//! needs no system library — `freetype-sys` compiles FreeType from vendored
//! sources with `cc`, unconditionally, with no `pkg_config` probe — so the
//! required `Test on ubuntu-latest` / `Test on macos-latest` jobs already run
//! it, via `cargo nextest run --workspace --all-features`. There is no
//! separate job and no `apt-get`; an earlier revision of this file added both
//! and justified them with a system-library dependency that does not exist.
//! The cost of that is worth stating plainly: a required merge gate now
//! depends on a third-party rasterizer, so a `freetype-rs` bump can break the
//! build for reasons that have nothing to do with this crate.
//!
//! **Being feature-gated, this file is invisible to the obvious local lint.**
//! `cargo clippy --workspace --all-targets` never compiles it — the same shape
//! of gap this suite exists to close, one level down: a target no job builds
//! is a target no check can see. CI runs
//! `cargo clippy --workspace --all-targets --all-features -- -D warnings`,
//! and that is the command to run before touching this file; the plain one
//! will pass on code that does not compile. Likewise the test itself runs only
//! under `--features freetype`.
//!
//! Known and deliberately not asserted: the reverse direction. There are two
//! texels where FreeType has ink and we do not, unchanged by any work here —
//! this rasterizer ramps coverage only in X, so a horizontal edge gets no
//! vertical antialiasing and lands a texel boundary hard. That is a separate,
//! pre-existing defect with its own fix, and gating on it would assert a bug.
#![cfg(feature = "freetype")]

use freetype as ft;
use pixelflow_graphics::fonts::Font;
use pixelflow_ir::{eval_scalar, passes::lower_dwrt_owned, BindingTable};

/// Device samples per texel edge when rasterizing the reference.
const SUPERSAMPLE: i64 = 16;
/// Where we call a texel inked.
const OURS_INKED: f32 = 0.5;
/// What counts as corroborating ink in the reference. Deliberately lower than
/// [`OURS_INKED`]: a texel whose true coverage sits near a half straddles the
/// threshold differently in two rasterizers, and that is not a finding. A
/// spurious crossing is not near a half — it is near zero.
const REFERENCE_INKED: f32 = 0.25;
/// How far the total ink we lay down may stray from the reference's. Measured
/// at 0.22% over the pairs below (1842.94 against 1847.03), so this is 9x
/// headroom — loose enough never to fire on an antialiasing difference, tight
/// enough that dropping or duplicating whole strokes cannot hide behind it.
const INK_RATIO_TOLERANCE: f64 = 0.02;
/// Texels FreeType inks and we do not, over the pairs below — **pinned**, not
/// capped. It is zero on this set, which reads like a claim that the defect is
/// absent; it is not. Widen the glyph set and it is 50: this rasterizer ramps
/// coverage only in X, so a horizontal edge gets no vertical antialiasing and
/// lands a texel boundary hard. Pinned rather than bounded because both
/// directions are news — upward is that defect spreading, downward is somebody
/// having fixed it, and either should be a deliberate edit here rather than a
/// silent drift.
const TEXELS_WE_MISS: u32 = 0;

fn font_path() -> String {
    format!(
        "{}/assets/DejaVuSansMono-Fallback.ttf",
        env!("CARGO_MANIFEST_DIR")
    )
}

#[test]
fn our_ink_is_never_more_than_a_texel_from_freetype_s() {
    let path = font_path();
    let bytes = std::fs::read(&path).expect("font bytes");
    let ours = Font::parse(&bytes).expect("parse");
    let lib = ft::Library::init().expect("freetype");
    let face = lib.new_face(&path, 0).expect("face");

    // Our screen frame: `scale = size / (ascender + |descender|)`, y flipped
    // about the baseline at `ascent_px`. Read the metrics from FreeType so the
    // mapping is not calibrated against the thing under test.
    let upem = face.raw().units_per_EM as f32;
    let ascender = face.raw().ascender as f32;
    let descender = face.raw().descender as f32;

    let mut orphans = Vec::new();
    // The reverse direction, recorded as a number that may not grow rather
    // than asserted to zero: FreeType inks texels we leave blank because this
    // rasterizer ramps coverage only in X, so a horizontal edge gets no
    // vertical antialiasing. That is a real, separate defect (see the module
    // docs); pinning today's count stops it spreading without asserting it
    // away.
    let mut we_miss = 0u32;
    let mut ink_ours = 0.0f64;
    let mut ink_reference = 0.0f64;

    for ch in ['8', 'O', 'S', 'g', 'e', 'A'] {
        for size in [7u32, 11, 13, 15, 17, 19, 21, 32] {
            let scale = size as f32 / (ascender + descender.abs());
            let ascent_px = ascender * scale;
            let extent = (size + size / 2) as i64;

            // Render the reference at SUPERSAMPLE device pixels per texel, in
            // our frame, so texel (i, j) is exactly one SUPERSAMPLE² block.
            let px = SUPERSAMPLE as f32 * scale * upem;
            face.set_char_size((px * 64.0) as isize, 0, 72, 72)
                .expect("char size");
            face.load_char(ch as usize, ft::face::LoadFlag::RENDER)
                .expect("load");
            let glyph = face.glyph();
            let bitmap = glyph.bitmap();
            let (bw, bh) = (bitmap.width() as i64, bitmap.rows() as i64);
            let (left, top) = (glyph.bitmap_left() as i64, glyph.bitmap_top() as i64);
            let pitch = bitmap.pitch() as i64;
            let buf = bitmap.buffer();
            let sample = |c: i64, r: i64| -> u8 {
                if c < 0 || r < 0 || c >= bw || r >= bh {
                    0
                } else {
                    buf[(r * pitch + c) as usize]
                }
            };
            let reference = |i: i64, j: i64| -> f32 {
                let mut acc = 0u32;
                for sy in 0..SUPERSAMPLE {
                    let screen_y = j as f32 + (sy as f32 + 0.5) / SUPERSAMPLE as f32;
                    let device_y = (SUPERSAMPLE as f32 * (ascent_px - screen_y)).floor() as i64;
                    let row = top - 1 - device_y;
                    for sx in 0..SUPERSAMPLE {
                        acc += u32::from(sample(i * SUPERSAMPLE + sx - left, row));
                    }
                }
                acc as f32 / (255.0 * (SUPERSAMPLE * SUPERSAMPLE) as f32)
            };

            let kernel = ours.glyph_kernel_scaled(ch, size as f32).expect("glyph");
            let (arena, root) = kernel.parts();
            let (lowered, r) = lower_dwrt_owned(arena, root).expect("lower");

            let inked: Vec<bool> = (0..extent * extent)
                .map(|n| reference(n % extent, n / extent) > REFERENCE_INKED)
                .collect();
            let corroborated = |i: i64, j: i64| {
                (-1..=1).any(|dj| {
                    (-1..=1).any(|di| {
                        let (a, b) = (i + di, j + dj);
                        a >= 0
                            && b >= 0
                            && a < extent
                            && b < extent
                            && inked[(b * extent + a) as usize]
                    })
                })
            };

            for j in 0..extent {
                for i in 0..extent {
                    let cov = eval_scalar(
                        &lowered,
                        r,
                        &[i as f32 + 0.5, j as f32 + 0.5],
                        &BindingTable::empty(),
                    );
                    assert!(
                        cov.is_finite(),
                        "{ch}@{size} texel ({i},{j}): coverage is {cov}"
                    );
                    ink_ours += f64::from(cov.clamp(0.0, 1.0));
                    ink_reference += f64::from(reference(i, j));
                    if reference(i, j) > OURS_INKED {
                        let ours_near = (-1..=1).any(|dj| {
                            (-1..=1).any(|di| {
                                let (a, b) = (i + di, j + dj);
                                a >= 0
                                    && b >= 0
                                    && a < extent
                                    && b < extent
                                    && eval_scalar(
                                        &lowered,
                                        r,
                                        &[a as f32 + 0.5, b as f32 + 0.5],
                                        &BindingTable::empty(),
                                    ) > OURS_INKED
                            })
                        });
                        if !ours_near {
                            we_miss += 1;
                        }
                    }
                    if cov > OURS_INKED && !corroborated(i, j) {
                        let mut best = 0.0f32;
                        for dj in -1..=1i64 {
                            for di in -1..=1i64 {
                                let (a, b) = (i + di, j + dj);
                                if a >= 0 && b >= 0 && a < extent && b < extent {
                                    best = best.max(reference(a, b));
                                }
                            }
                        }
                        orphans.push(format!(
                            "{ch}@{size} texel ({i},{j}): ours {cov:.3}, \
                             best FreeType coverage within one texel {best:.3}"
                        ));
                    }
                }
            }
        }
    }

    // Total ink. A predicate about individual texels is weak on its own — it
    // survives deleting every other row, or shifting the whole glyph by a
    // texel — and this closes that: two rasterizers drawing the same outlines
    // put down the same amount of ink. Measured 0.22% apart, bounded at 2%.
    let ratio = ink_ours / ink_reference;
    assert!(
        (ratio - 1.0).abs() < INK_RATIO_TOLERANCE,
        "we lay down {ratio:.4}x FreeType's ink ({ink_ours:.1} vs \
         {ink_reference:.1}) — the outlines are not being filled the same way"
    );
    assert_eq!(
        we_miss, TEXELS_WE_MISS,
        "FreeType inks {we_miss} texels we leave blank, pinned at \
         {TEXELS_WE_MISS} — up means the no-vertical-antialiasing defect has \
         spread, down means it has been fixed and this number wants lowering"
    );

    assert!(
        orphans.is_empty(),
        "we put ink where an independent rasterizer finds none ({} texels) — \
         a crossing is being counted that does not exist:\n{}",
        orphans.len(),
        orphans.join("\n")
    );
}
