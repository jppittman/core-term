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
//! **The assertion is topological, not photometric.** FreeType's antialiasing
//! convention is its own and need not equal ours; comparing coverage values
//! would fail for reasons that are not bugs. What must agree is *where the ink
//! is*: wherever we mark a texel covered, an independent rasterizer must find
//! ink within one texel of it. A spurious crossing puts ink where FreeType has
//! none at all (`0.000` across several texels); a difference of AA convention
//! never does.
//!
//! Not part of the blocking set: it needs a system `libfreetype`, so it is
//! feature-gated and its CI job is informational. An informational job that
//! would have caught this is worth more than no job.
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
    let upem = (*face.raw()).units_per_EM as f32;
    let ascender = (*face.raw()).ascender as f32;
    let descender = (*face.raw()).descender as f32;

    let mut orphans = Vec::new();

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

    assert!(
        orphans.is_empty(),
        "we put ink where an independent rasterizer finds none ({} texels) — \
         a crossing is being counted that does not exist:\n{}",
        orphans.len(),
        orphans.join("\n")
    );
}
