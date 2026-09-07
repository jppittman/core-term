//! The two encodings of a laid-out string, and what makes them agree.
//!
//! `text` sums the placed glyphs into one kernel, so every pixel evaluates
//! every glyph; `text_union` restricts the index instead, so a glyph is only
//! asked about the columns its support reaches. Four things have to hold for
//! that to be a rewrite rather than an approximation, and the first three are
//! pinned **bit-exactly**:
//!
//! 1. **The union machinery is exact.** A union whose one range is the whole
//!    frame, carrying the same kernel, bakes the same buffer as a plain bake —
//!    down to the bit. That pins the ranges, the destination offsets, the
//!    sampling origin, and the no-overhang collapse.
//! 2. **A glyph is *exactly* zero outside its support.** Not small: the
//!    literal `0.0`. That is the whole licence for dropping a glyph from a
//!    cell, and it is a property of how `ttf::compile` composes an outline (a
//!    unit-square mask whose false arm is the constant 0), so it is checked
//!    against the font rather than assumed.
//! 3. **Every cell bakes exactly its own kernel at its own shape.** Together
//!    with (2) that *is* the correctness argument: the union computes the
//!    right function, and drops only summands that were the literal zero.
//! 4. **What is left is the compiler's, not the union's.** E-graph extraction
//!    is a function of the arena *and* of the lattice shape — the same glyph
//!    kernel compiled at 160×24 and at 9×24 differs in 6 of 216 samples by
//!    2.4e-7 — so a decomposition necessarily reschedules its pieces and
//!    rounds differently at antialiasing edges. That is not a difference the
//!    union introduces: baking the pieces separately and adding them in f32,
//!    with no union anywhere, pays the same. Test (4) is the only one with a
//!    threshold, and it is checked over the **whole frame** — every column,
//!    whatever its cell's arity — so a reach bug that quietly dropped a glyph
//!    from a cell has nowhere to hide.
//!
//! ## What runs presubmit
//!
//! The four bodies below are parameterized by corpus and sizes. The `#[test]`s
//! run them over [`CORPUS_FAST`] at [`SIZES_FAST`]; `the_full_corpus` runs the
//! same bodies over every string at every size and is `#[ignore]`d, because it
//! bakes a few thousand kernels and belongs in postsubmit:
//!
//! ```text
//! cargo test -p pixelflow-graphics --test text_union_identity -- --ignored
//! ```

use pixelflow_core::{IndexRange, Kernel, Lattice, Union};
use pixelflow_graphics::fonts::{text, text_cells, text_union, Font, TextCell};

const FONT_DATA: &[u8] = include_bytes!("../assets/DejaVuSansMono-Fallback.ttf");

/// Presubmit: one string whose cells hold a single glyph each, one whose wide
/// glyphs make cells share, at the size the benchmarks use.
const CORPUS_FAST: [&str; 2] = ["ABCDEFGHIJKLMNOPQRSTUVWXYZ", "iiiiWWWWmmmm"];
const SIZES_FAST: [f32; 1] = [16.0];

const CORPUS_FULL: [&str; 10] = [
    "A",
    "AB",
    "HELLO",
    "ABCDEFGHIJ",
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    "The quick brown fox jumps over the lazy dog",
    "iiiiWWWWmmmm",
    "WWW",
    "{{{",
    " leading and trailing ",
];
const SIZES_FULL: [f32; 4] = [12.0, 16.0, 20.0, 32.0];

/// Sizes the support check sweeps the printable range at. One presubmit; the
/// large size is where the crossing ramp reaches furthest past the ink, so it
/// is the one that would catch an ink-box-plus-apron mistake.
const SUPPORT_SIZES_FAST: [f32; 1] = [20.0];
const SUPPORT_SIZES_FULL: [f32; 3] = [12.0, 20.0, 48.0];

/// A frame wide enough for the string. `text`/`text_union` bake the
/// sample-center convention into their own kernels now (`layout`), so the
/// frame is a plain index — no origin to carry it.
fn frame(text_str: &str, size: f32) -> Lattice {
    let columns = text_str.chars().count().max(1) as f32;
    Lattice::frame(
        (columns * size).ceil() as usize,
        (size * 1.5).ceil() as usize,
    )
}

/// One cell of the decomposition, baked on its own: the range's own extent,
/// collapsed starting at the range's own index — exactly the per-summand step
/// [`Union`] performs, exposed here so a cell can be compared against the
/// union it came from.
fn cell_baked(cell: &TextCell) -> Vec<f32> {
    cell.range.bake(&Kernel::sum(&cell.glyphs)).into_buffer()
}

/// Samples that differ, and by how much.
fn compare(a: &[f32], b: &[f32]) -> (usize, f32) {
    assert_eq!(a.len(), b.len(), "buffers of different size");
    let differing = a
        .iter()
        .zip(b)
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count();
    let worst = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    (differing, worst)
}

/// Every glyph of a string, placed at its pen, baked on its own over `lattice`
/// and added in f32 left to right — the sum's own association, with no union
/// anywhere. The reference for (4).
fn split_and_added(font: &Font, lattice: Lattice, text_str: &str, size: f32) -> Vec<f32> {
    let mut split = vec![0.0f32; lattice.len()];
    let mut cursor = 0.0f32;
    for ch in text_str.chars() {
        let id = font.cmap_lookup(ch).unwrap_or(0);
        let advance = font.advance_scaled_by_id(id, size).unwrap_or(0.0);
        let pen = cursor;
        cursor += advance;
        let Some(glyph) = font.glyph_scaled_by_id(id, size) else {
            continue;
        };
        // Same placement `layout` makes: pen shift plus the pixel-center ½,
        // since `lattice` is a plain index with no origin to carry it.
        let placed = glyph.kernel.at(
            &Kernel::x()
                .add(&Kernel::constant(0.5))
                .sub(&Kernel::constant(pen)),
            &Kernel::y().add(&Kernel::constant(0.5)),
        );
        for (dst, src) in split.iter_mut().zip(lattice.bake(&placed).buffer()) {
            *dst += src;
        }
    }
    split
}

// ─────────────────────────── the four bodies ───────────────────────────

/// 1. The union machinery, with the arena held fixed: one range covering the
///    whole frame, carrying exactly the kernel a plain bake would take.
fn the_union_machinery_is_exact(font: &Font, corpus: &[&str], sizes: &[f32]) {
    for text_str in corpus {
        for &size in sizes {
            let lattice = frame(text_str, size);
            let kernel = text(font, text_str, size);
            let mut union = Union::over(lattice);
            union.place(
                IndexRange::new(0, 0, lattice.extent[0] as usize, lattice.extent[1] as usize),
                &kernel,
            );
            let (differing, worst) = compare(lattice.bake(&kernel).buffer(), union.bake().buffer());
            assert_eq!(
                differing, 0,
                "{text_str:?} at {size}: a whole-frame union moved {differing} samples \
                 (worst {worst:e}) — the union's own machinery must be exact"
            );
        }
    }
}

/// 2. The licence for dropping a glyph: outside the box
///    `Font::glyph_scaled_by_id` reports, its kernel is the literal `0.0` at
///    every sample.
///
/// A tolerance here would defeat the purpose — "small enough to ignore" is
/// precisely the claim this is meant to refuse. It is also the assumption an
/// ink box plus a fixed apron would get *wrong*: the crossing ramp is `‖∇d‖`
/// pixels wide in X, so a nearly horizontal segment leaves a glyph nonzero
/// tens of pixels past its outline (`}` at 48 px reaches 21 px right of its
/// ink). The unit-square mask does not care, because its false arm is a
/// constant.
fn a_glyph_is_zero_outside_its_support(font: &Font, sizes: &[f32]) {
    const PEN: f32 = 48.0;
    const CENTER: f32 = 0.5;
    for &size in sizes {
        let (w, h) = (160usize, (size * 2.5) as usize);
        let lattice = Lattice::frame(w, h);
        for ch in ' '..='~' {
            let id = font.cmap_lookup(ch).unwrap_or(0);
            let Some(glyph) = font.glyph_scaled_by_id(id, size) else {
                continue;
            };
            // Placed, and shifted onto the rasterizer's pixel centers — the
            // same contramap `text`'s `layout` bakes in, applied by hand here
            // since this checks the raw glyph kernel directly.
            let placed = glyph.kernel.at(
                &Kernel::x().add(&Kernel::constant(CENTER - PEN)),
                &Kernel::y().add(&Kernel::constant(CENTER)),
            );
            let [x0, y0, x1, y1] = glyph.support.shifted_x(PEN).bounds();
            let baked = lattice.bake(&placed);
            for row in 0..h {
                let cy = row as f32 + CENTER;
                for col in 0..w {
                    let cx = col as f32 + CENTER;
                    if (x0..=x1).contains(&cx) && (y0..=y1).contains(&cy) {
                        continue;
                    }
                    let v = baked.buffer()[row * w + col];
                    assert_eq!(
                        v.to_bits(),
                        0f32.to_bits(),
                        "{ch:?} at {size}: sample ({cx}, {cy}) is {v:e} outside the \
                         support [{x0}, {y0}, {x1}, {y1}]"
                    );
                }
            }
        }
    }
}

/// 3. **Every cell bakes exactly its own kernel at its own shape** — bit for
///    bit, no tolerance. This is the complete statement of what the union
///    promises: the range, the destination offset, the sampling origin and the
///    no-overhang collapse, with nothing left for the compiler to vary,
///    because the reference is compiled from the same arena at the same extent
///    and sampled at the same coordinates.
fn every_cell_bakes_its_own_kernel(font: &Font, corpus: &[&str], sizes: &[f32]) {
    let mut checked = 0usize;
    for text_str in corpus {
        for &size in sizes {
            let lattice = frame(text_str, size);
            let width = lattice.extent[0] as usize;
            let united = text_union(font, lattice, text_str, size).bake();
            for cell in text_cells(font, lattice, text_str, size) {
                checked += 1;
                let alone = cell_baked(&cell);
                for row in 0..cell.range.rows() {
                    for col in 0..cell.range.width() {
                        let here = (cell.range.y0() + row) * width + cell.range.x0() + col;
                        let there = row * cell.range.width() + col;
                        assert_eq!(
                            united.buffer()[here].to_bits(),
                            alone[there].to_bits(),
                            "{text_str:?} at {size}: the cell at column {} is {:?} at \
                             (row {row}, col {col}) in the union but {:?} baked on its own",
                            cell.range.x0(),
                            united.buffer()[here],
                            alone[there],
                        );
                    }
                }
            }
        }
    }
    assert!(
        checked > 20,
        "only {checked} cells — this gate is not covering anything"
    );
}

/// One step of an 8-bit coverage ramp. **Nothing in this path quantizes to 8
/// bits** — it is a chosen display scale, named here because it is the
/// smallest difference a reader can picture and it sits an order of magnitude
/// above what is being bounded.
///
/// The line it draws is between two failure modes. A **wrongly dropped glyph**
/// is a coverage-scale difference: a whole antialiased edge pixel, tens to
/// hundreds of steps. **Scheduling noise** — the same expression extracted
/// differently because the arena or the lattice shape changed — is a
/// rounding-scale difference; worst observed over the full corpus is 1.785e-4,
/// 4.6% of one step.
const COVERAGE_STEP: f32 = 1.0 / 255.0;

/// 4. What is left once (1), (2) and (3) have pinned the union exactly: the
///    cells are a *different arena at a different extent* from the whole
///    string's, and e-graph extraction is a function of both, so the compiler
///    may schedule them differently and round differently at edges.
///
/// The gate is that this stays scheduling noise, over the **whole frame** —
/// single-glyph cells included, so a reach bug that dropped a glyph from a
/// shared cell would show up here even though (3) cannot see it. It is checked
/// against the same measurement made with **no union anywhere**, so the bound
/// is shown to be the compiler's rather than the union's. (Their *ordering* is
/// not a theorem: three schedules of one expression do not round in any
/// particular order, and asserting one is below the other fails on rounding
/// coincidences that mean nothing.)
fn the_union_only_moves_a_sample_by_scheduling_noise(font: &Font, corpus: &[&str], sizes: &[f32]) {
    for text_str in corpus {
        for &size in sizes {
            let lattice = frame(text_str, size);
            let whole = lattice.bake(&text(font, text_str, size));
            let united = text_union(font, lattice, text_str, size).bake();
            let split = split_and_added(font, lattice, text_str, size);

            let (union_n, union_worst) = compare(whole.buffer(), united.buffer());
            let (_, split_worst) = compare(whole.buffer(), &split);
            assert!(
                union_worst < COVERAGE_STEP,
                "{text_str:?} at {size}: the union moved {union_n} samples by up to \
                 {union_worst:e}, which is {:.1} coverage steps — that is a dropped \
                 glyph, not a schedule",
                union_worst / COVERAGE_STEP
            );
            assert!(
                split_worst < COVERAGE_STEP,
                "{text_str:?} at {size}: splitting the arena with no union in sight \
                 already moves a sample by {split_worst:e} ({:.1} coverage steps), so \
                 this gate is measuring something other than the union",
                split_worst / COVERAGE_STEP
            );
        }
    }
}

// ───────────────────────────── presubmit ─────────────────────────────

#[test]
fn a_whole_frame_union_bakes_the_plain_bake_bit_for_bit() {
    let font = Font::parse(FONT_DATA).expect("font");
    the_union_machinery_is_exact(&font, &CORPUS_FAST, &SIZES_FAST);
}

#[test]
fn a_glyph_is_exactly_zero_outside_its_support() {
    let font = Font::parse(FONT_DATA).expect("font");
    a_glyph_is_zero_outside_its_support(&font, &SUPPORT_SIZES_FAST);
}

#[test]
fn every_cell_bakes_exactly_its_own_kernel_at_its_own_shape() {
    let font = Font::parse(FONT_DATA).expect("font");
    every_cell_bakes_its_own_kernel(&font, &CORPUS_FAST, &SIZES_FAST);
}

#[test]
fn the_union_agrees_with_the_sum_to_within_the_compiler() {
    let font = Font::parse(FONT_DATA).expect("font");
    the_union_only_moves_a_sample_by_scheduling_noise(&font, &CORPUS_FAST, &SIZES_FAST);
}

/// A one-glyph string has one cell that takes every glyph there is, so the
/// union's kernel *is* the sum's and there is nothing left for the compiler to
/// schedule differently. The agreement is then exact, which is the sanity
/// check on (4)'s threshold: it is not hiding a constant offset.
#[test]
fn a_cell_that_takes_every_glyph_agrees_exactly() {
    let font = Font::parse(FONT_DATA).expect("font");
    for text_str in ["W", "@", "{"] {
        for size in [16.0f32, 48.0] {
            let lattice = frame(text_str, size);
            let (differing, worst) = compare(
                lattice.bake(&text(&font, text_str, size)).buffer(),
                text_union(&font, lattice, text_str, size).bake().buffer(),
            );
            assert_eq!(
                differing, 0,
                "{text_str:?} at {size}: {differing} samples differ by up to {worst:e} \
                 with only one glyph to schedule"
            );
        }
    }
}

/// The empty string places no summand, and an unclaimed frame is all zeros.
#[test]
fn text_that_reaches_nothing_collapses_to_zero() {
    let font = Font::parse(FONT_DATA).expect("font");
    let lattice = Lattice::frame(32, 32);
    let union = text_union(&font, lattice, "", 16.0);
    assert!(union.is_empty(), "the empty string places no summand");
    assert!(union.bake().buffer().iter().all(|&v| v == 0.0));
}

/// Two summands claiming one column is not a blend and not a painter's order.
/// The union refuses it when it is built, so a scene that would have produced
/// silently-wrong pixels never reaches a collapse.
#[test]
#[should_panic(expected = "overlaps the summand")]
fn overlapping_ranges_are_refused_at_build() {
    let lattice = Lattice::frame(64, 16);
    let mut union = Union::over(lattice);
    union.place(IndexRange::new(0, 0, 32, 16), &Kernel::constant(1.0));
    union.place(IndexRange::new(31, 0, 33, 16), &Kernel::constant(2.0));
}

// ───────────────────────────── postsubmit ─────────────────────────────

/// The same four gates over every string at every size, and the support sweep
/// at three. Bakes a few thousand kernels, so it is `#[ignore]`d:
///
/// ```text
/// cargo test -p pixelflow-graphics --test text_union_identity -- --ignored
/// ```
#[test]
#[ignore = "the full corpus: cargo test -p pixelflow-graphics --test text_union_identity -- --ignored"]
fn the_full_corpus() {
    let font = Font::parse(FONT_DATA).expect("font");
    the_union_machinery_is_exact(&font, &CORPUS_FULL, &SIZES_FULL);
    a_glyph_is_zero_outside_its_support(&font, &SUPPORT_SIZES_FULL);
    every_cell_bakes_its_own_kernel(&font, &CORPUS_FULL, &SIZES_FULL);
    the_union_only_moves_a_sample_by_scheduling_noise(&font, &CORPUS_FULL, &SIZES_FULL);
}
