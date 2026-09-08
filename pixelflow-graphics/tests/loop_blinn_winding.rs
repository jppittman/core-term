//! The Loop–Blinn glyph kernel against an independent winding-number oracle.
//!
//! `loop_blinn` computes the winding number of an outline as a sum of
//! per-segment terms relative to a reference point, with Loop–Blinn's
//! implicit for the curved slivers. The oracle here is the *other*
//! formulation — a horizontal ray cast in `f64`, intersecting each quadratic
//! by solving its `y(t)` — written from scratch and sharing no code or
//! constants with the kernel. Where both are defined and away from the
//! antialiasing ramp, they must agree exactly: coverage there is an exact
//! integer, 0 or 1, and a kernel that gets it wrong has the wrong winding,
//! not the wrong rounding.
//!
//! What this suite covers, and what it does not: it pins the *winding*
//! (interior/exterior classification, holes, self-intersections, contour
//! direction, compound placement) and the exactness of the support. The
//! antialiasing ramp's shape is pinned by `font_antialiasing.rs`, and where
//! the ink is against a second rasterizer by `freetype_oracle.rs`.

use pixelflow_core::{Kernel, Lattice, Union};
use pixelflow_graphics::fonts::{loop_blinn, Contour, Font, Outline, Segment};

const FONT_DATA: &[u8] = include_bytes!("../assets/DejaVuSansMono-Fallback.ttf");

/// The rasterizer's pixel-centre convention.
const CENTER: f32 = 0.5;

/// Samples closer than this to the outline are in the antialiasing ramp
/// (half a pixel wide on each side) and are not the oracle's to judge. A
/// little past the ramp's half-pixel, for the fast `sqrt`'s estimate error.
const RAMP_CLEARANCE: f64 = 0.75;

/// Pieces each curve is cut into when measuring distance to the outline.
const DISTANCE_PIECES: usize = 64;

// ────────────────────────────── the oracle ──────────────────────────────

fn wide([x, y]: [f32; 2]) -> [f64; 2] {
    [f64::from(x), f64::from(y)]
}

/// The winding number at `p` by a horizontal ray to `+x`, half-open in `y`
/// so a vertex on the ray counts once; quadratics are intersected by solving
/// `y(t) = p.y` exactly.
fn winding(outline: &Outline, [px, py]: [f64; 2]) -> i32 {
    let mut w = 0i32;
    for seg in outline.segments() {
        match seg {
            Segment::Line { from, to } => {
                let (a, b) = (wide(from), wide(to));
                w += crossing_sign(a, b, [px, py]);
            }
            Segment::Quad { from, control, to } => {
                let (p0, p1, p2) = (wide(from), wide(control), wide(to));
                // y(t) = ay t² + by t + cy − py = 0
                let ay = p0[1] - 2.0 * p1[1] + p2[1];
                let by = 2.0 * (p1[1] - p0[1]);
                let cy = p0[1] - py;
                let roots: Vec<f64> = if ay.abs() < 1e-12 {
                    if by.abs() < 1e-12 {
                        vec![]
                    } else {
                        vec![-cy / by]
                    }
                } else {
                    let disc = by * by - 4.0 * ay * cy;
                    if disc < 0.0 {
                        vec![]
                    } else {
                        let s = disc.sqrt();
                        vec![(-by - s) / (2.0 * ay), (-by + s) / (2.0 * ay)]
                    }
                };
                for t in roots {
                    // Half-open in t, matching the half-open y rule at the
                    // endpoints: a root at t = 1 is the next segment's t = 0.
                    if !(0.0..1.0).contains(&t) {
                        continue;
                    }
                    let x =
                        (1.0 - t) * (1.0 - t) * p0[0] + 2.0 * (1.0 - t) * t * p1[0] + t * t * p2[0];
                    let dy = 2.0 * ay * t + by;
                    if x > px && dy != 0.0 {
                        w += if dy > 0.0 { 1 } else { -1 };
                    }
                }
            }
        }
    }
    w
}

fn crossing_sign(a: [f64; 2], b: [f64; 2], [px, py]: [f64; 2]) -> i32 {
    if (a[1] <= py) == (b[1] <= py) {
        return 0;
    }
    let t = (py - a[1]) / (b[1] - a[1]);
    let x = a[0] + t * (b[0] - a[0]);
    if x > px {
        if b[1] > a[1] {
            1
        } else {
            -1
        }
    } else {
        0
    }
}

/// Distance from `p` to the outline, by the distance to each segment's
/// polyline.
fn distance(outline: &Outline, p: [f64; 2]) -> f64 {
    let mut best = f64::INFINITY;
    for seg in outline.segments() {
        let pts: Vec<[f64; 2]> = match seg {
            Segment::Line { from, to } => vec![wide(from), wide(to)],
            Segment::Quad { from, control, to } => {
                let (p0, p1, p2) = (wide(from), wide(control), wide(to));
                (0..=DISTANCE_PIECES)
                    .map(|i| {
                        let t = i as f64 / DISTANCE_PIECES as f64;
                        let (a, b, c) = ((1.0 - t) * (1.0 - t), 2.0 * (1.0 - t) * t, t * t);
                        [
                            a * p0[0] + b * p1[0] + c * p2[0],
                            a * p0[1] + b * p1[1] + c * p2[1],
                        ]
                    })
                    .collect()
            }
        };
        for w in pts.windows(2) {
            best = best.min(segment_distance(w[0], w[1], p));
        }
    }
    best
}

fn segment_distance(a: [f64; 2], b: [f64; 2], p: [f64; 2]) -> f64 {
    let (ex, ey) = (b[0] - a[0], b[1] - a[1]);
    let len2 = ex * ex + ey * ey;
    let t = if len2 == 0.0 {
        0.0
    } else {
        (((p[0] - a[0]) * ex + (p[1] - a[1]) * ey) / len2).clamp(0.0, 1.0)
    };
    let (qx, qy) = (a[0] + t * ex, a[1] + t * ey);
    ((p[0] - qx).powi(2) + (p[1] - qy).powi(2)).sqrt()
}

/// The oracle's coverage: `min(|w|, 1)`.
fn oracle(outline: &Outline, p: [f64; 2]) -> f32 {
    winding(outline, p).unsigned_abs().min(1) as f32
}

// ────────────────────────────── harness ──────────────────────────────

fn pixel_centered(k: &Kernel) -> Kernel {
    k.at(
        &Kernel::x().add(&Kernel::constant(CENTER)),
        &Kernel::y().add(&Kernel::constant(CENTER)),
    )
}

/// Bake the single-kernel form over `lattice` at pixel centres.
fn bake_single(outline: &Outline, lattice: Lattice) -> Vec<f32> {
    lattice
        .bake(&pixel_centered(&loop_blinn::glyph(outline).kernel))
        .into_buffer()
}

/// Bake the union-of-cells form over `lattice`.
fn bake_cells(outline: &Outline, lattice: Lattice, cell: [usize; 2]) -> Vec<f32> {
    let mut union = Union::over(lattice);
    for (range, kernel) in loop_blinn::cells(outline, lattice, cell) {
        union.place(range, &kernel);
    }
    union.bake().into_buffer()
}

/// Compare a baked buffer against the oracle at every sample clear of the
/// ramp; returns (samples judged, samples in the ramp). Panics on the first
/// disagreement with the offending sample.
fn judge(label: &str, outline: &Outline, lattice: Lattice, baked: &[f32]) -> (usize, usize) {
    let [w, h] = lattice.extent.map(|e| e as usize);
    let (mut judged, mut in_ramp) = (0usize, 0usize);
    let mut failures = Vec::new();
    for j in 0..h {
        for i in 0..w {
            let p = [i as f64 + f64::from(CENTER), j as f64 + f64::from(CENTER)];
            let got = baked[j * w + i];
            assert!(
                got.is_finite(),
                "{label}: non-finite coverage {got} at ({i},{j})"
            );
            assert!(
                (0.0..=1.0).contains(&got),
                "{label}: coverage {got} out of range at ({i},{j})"
            );
            if distance(outline, p) < RAMP_CLEARANCE {
                in_ramp += 1;
                continue;
            }
            judged += 1;
            let want = oracle(outline, p);
            // `!=` rather than bit comparison: a saturated sum is an exact
            // integer, and the optimizer is free to hand back `-0.0` for it.
            if got != want {
                failures.push(format!("({i},{j}): kernel {got} oracle {want}"));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{label}: {} of {judged} samples clear of the ramp disagree with the oracle:\n{}",
        failures.len(),
        failures.join("\n")
    );
    (judged, in_ramp)
}

fn square(x0: f32, y0: f32, x1: f32, y1: f32, clockwise: bool) -> Contour {
    let mut pts = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]];
    if clockwise {
        pts.reverse();
    }
    Contour {
        segments: (0..4)
            .map(|i| Segment::Line {
                from: pts[i],
                to: pts[(i + 1) % 4],
            })
            .collect(),
    }
}

/// A circle of radius `r` about `c` from four quadratics through the
/// diagonal control points — the curved case with every tangent direction.
fn ring(c: [f32; 2], r: f32, clockwise: bool) -> Contour {
    let k = r; // control points at the corners of the bounding square
    let mut on = [
        [c[0] + r, c[1]],
        [c[0], c[1] + r],
        [c[0] - r, c[1]],
        [c[0], c[1] - r],
    ];
    let mut off = [
        [c[0] + k, c[1] + k],
        [c[0] - k, c[1] + k],
        [c[0] - k, c[1] - k],
        [c[0] + k, c[1] - k],
    ];
    if clockwise {
        on.reverse();
        off.reverse();
        off.rotate_right(1);
    }
    Contour {
        segments: (0..4)
            .map(|i| Segment::Quad {
                from: on[i],
                control: off[i],
                to: on[(i + 1) % 4],
            })
            .collect(),
    }
}

fn outline_of(contours: Vec<Contour>) -> Outline {
    Outline { contours }
}

// ────────────────────────────── synthetic outlines ──────────────────────────────

#[test]
fn synthetic_outlines_wind_like_the_oracle() {
    let lattice = Lattice::frame(48, 48);
    let cases: Vec<(&str, Outline)> = vec![
        (
            "square",
            outline_of(vec![square(8.0, 8.0, 40.0, 40.0, false)]),
        ),
        (
            "square clockwise",
            outline_of(vec![square(8.0, 8.0, 40.0, 40.0, true)]),
        ),
        (
            "square with a hole",
            outline_of(vec![
                square(4.0, 4.0, 44.0, 44.0, false),
                square(16.0, 16.0, 32.0, 32.0, true),
            ]),
        ),
        (
            "nested same direction: winding 2 is still covered",
            outline_of(vec![
                square(4.0, 4.0, 44.0, 44.0, false),
                square(16.0, 16.0, 32.0, 32.0, false),
            ]),
        ),
        (
            "bow tie",
            outline_of(vec![Contour {
                segments: vec![
                    Segment::Line {
                        from: [4.0, 4.0],
                        to: [44.0, 44.0],
                    },
                    Segment::Line {
                        from: [44.0, 44.0],
                        to: [4.0, 44.0],
                    },
                    Segment::Line {
                        from: [4.0, 44.0],
                        to: [44.0, 4.0],
                    },
                    Segment::Line {
                        from: [44.0, 4.0],
                        to: [4.0, 4.0],
                    },
                ],
            }]),
        ),
        (
            "sharp triangle",
            outline_of(vec![Contour {
                segments: vec![
                    Segment::Line {
                        from: [4.0, 4.0],
                        to: [44.0, 22.0],
                    },
                    Segment::Line {
                        from: [44.0, 22.0],
                        to: [4.0, 30.0],
                    },
                    Segment::Line {
                        from: [4.0, 30.0],
                        to: [4.0, 4.0],
                    },
                ],
            }]),
        ),
        ("ring", outline_of(vec![ring([24.0, 24.0], 18.0, false)])),
        (
            "ring clockwise",
            outline_of(vec![ring([24.0, 24.0], 18.0, true)]),
        ),
        (
            "annulus",
            outline_of(vec![
                ring([24.0, 24.0], 20.0, false),
                ring([24.0, 24.0], 9.0, true),
            ]),
        ),
        (
            "concave bulges: a square pinched by curves",
            outline_of(vec![Contour {
                segments: vec![
                    Segment::Quad {
                        from: [4.0, 4.0],
                        control: [24.0, 20.0],
                        to: [44.0, 4.0],
                    },
                    Segment::Quad {
                        from: [44.0, 4.0],
                        control: [28.0, 24.0],
                        to: [44.0, 44.0],
                    },
                    Segment::Quad {
                        from: [44.0, 44.0],
                        control: [24.0, 28.0],
                        to: [4.0, 44.0],
                    },
                    Segment::Quad {
                        from: [4.0, 44.0],
                        control: [20.0, 24.0],
                        to: [4.0, 4.0],
                    },
                ],
            }]),
        ),
    ];
    for (name, outline) in &cases {
        let single = bake_single(outline, lattice);
        let (judged, _) = judge(&format!("{name} (single)"), outline, lattice, &single);
        assert!(judged > 1000, "{name}: only {judged} samples judged");
        // Two cell shapes, neither a divisor of the frame, so the last row
        // and column of cells are partial.
        for cell in [[20, 12], [48, 48]] {
            let cells = bake_cells(outline, lattice, cell);
            judge(
                &format!("{name} (cells {cell:?})"),
                outline,
                lattice,
                &cells,
            );
        }
    }
}

/// One-sample cells: every sample its own program, its own reference point
/// and its own pruning — the degenerate extent every argument in the
/// module has to survive.
#[test]
fn one_sample_cells_wind_like_the_oracle() {
    let lattice = Lattice::frame(12, 12);
    let outline = outline_of(vec![
        square(1.0, 1.0, 11.0, 11.0, false),
        square(4.0, 4.0, 8.0, 8.0, true),
    ]);
    let cells = bake_cells(&outline, lattice, [1, 1]);
    let (judged, _) = judge("holed square (cells [1, 1])", &outline, lattice, &cells);
    assert!(judged > 20, "only {judged} samples judged");
}

// ────────────────────────────── real glyphs ──────────────────────────────

const GLYPHS: [char; 12] = ['O', '8', 'A', 'g', 'W', '@', '%', '{', 'f', 'e', 'S', '&'];
const SIZES: [f32; 3] = [7.0, 13.0, 24.0];

/// The single-kernel form of real glyphs: every contour shape a font
/// produces — holes, counters, sharp joins, tangent joins, slivers.
#[test]
fn glyphs_wind_like_the_oracle() {
    let font = Font::parse(FONT_DATA).expect("font");
    let mut judged_total = 0usize;
    for size in SIZES {
        let n = (size * 1.5).ceil() as usize;
        let lattice = Lattice::frame(n, n);
        for ch in GLYPHS {
            let id = font.cmap_lookup(ch).expect("glyph");
            let outline = font.outline_scaled_by_id(id, size).expect("outline");
            let single = bake_single(&outline, lattice);
            let (judged, _) = judge(&format!("{ch}@{size} (single)"), &outline, lattice, &single);
            judged_total += judged;
        }
    }
    assert!(judged_total > 5_000, "only {judged_total} samples judged");
}

/// The domain-side form of real glyphs, at a cell size that leaves several
/// segments per boundary cell and whole interior cells to the constant.
#[test]
fn glyph_cells_wind_like_the_oracle() {
    let font = Font::parse(FONT_DATA).expect("font");
    for (size, cell) in [(13.0f32, [8usize, 8usize]), (24.0, [16, 8])] {
        let n = (size * 1.5).ceil() as usize;
        let lattice = Lattice::frame(n, n);
        for ch in ['O', '8', 'g', '@', '{', '&'] {
            let id = font.cmap_lookup(ch).expect("glyph");
            let outline = font.outline_scaled_by_id(id, size).expect("outline");
            let cells = bake_cells(&outline, lattice, cell);
            judge(
                &format!("{ch}@{size} (cells {cell:?})"),
                &outline,
                lattice,
                &cells,
            );
        }
    }
}

/// Every printable glyph, one size: the whole font's parser and every
/// contour shape it produces, against the oracle.
#[test]
fn every_printable_glyph_winds_like_the_oracle() {
    let font = Font::parse(FONT_DATA).expect("font");
    let size = 20.0f32;
    let n = 30usize;
    let lattice = Lattice::frame(n, n);
    for ch in ' '..='~' {
        let id = font.cmap_lookup(ch).expect("glyph");
        let outline = font.outline_scaled_by_id(id, size).expect("outline");
        let single = bake_single(&outline, lattice);
        judge(&format!("{ch:?}@{size}"), &outline, lattice, &single);
    }
}

// ────────────────────────────── support ──────────────────────────────

/// Outside the reported support the kernel is the literal `0.0` — sampled
/// on a two-pixel ring just outside the box, and far away.
#[test]
fn a_glyph_is_exactly_zero_outside_its_support() {
    let font = Font::parse(FONT_DATA).expect("font");
    let corpus: Vec<(f32, char)> = (' '..='~')
        .map(|ch| (48.0f32, ch))
        .chain(GLYPHS.iter().flat_map(|&ch| [(12.0, ch), (20.0, ch)]))
        .collect();
    for (size, ch) in corpus {
        {
            let id = font.cmap_lookup(ch).expect("glyph");
            let glyph = font.glyph_scaled_by_id(id, size).expect("glyph");
            if glyph.support.is_empty() {
                continue;
            }
            let [x0, y0, x1, y1] = glyph.support.bounds();
            let probes: Vec<[f32; 2]> = (0..40)
                .flat_map(|i| {
                    let t = i as f32 / 40.0;
                    let (w, h) = (x1 - x0 + 4.0, y1 - y0 + 4.0);
                    [
                        [x0 - 2.0 + t * w, y0 - 1.0],
                        [x0 - 2.0 + t * w, y1 + 1.0],
                        [x0 - 1.0, y0 - 2.0 + t * h],
                        [x1 + 1.0, y0 - 2.0 + t * h],
                        [x0 - 2.0 + t * w, y0 - 100.0],
                        [x1 + 1000.0, y0 - 2.0 + t * h],
                    ]
                })
                .collect();
            for [x, y] in probes {
                let v = Lattice::eval_at(&glyph.kernel, x, y);
                assert_eq!(
                    v.to_bits(),
                    0f32.to_bits(),
                    "{ch:?}@{size}: {v:e} at ({x}, {y}) outside the support {:?}",
                    glyph.support.bounds()
                );
            }
        }
    }
}

/// The bounding box of the *ink* plus the ramp's reach: the support is not
/// the loose box the scanline kernel's trailing ramps forced (up to 21 px past
/// the ink at 48 px) — it is the outline's bounding box plus one pixel.
#[test]
fn the_support_is_the_outline_bounds_plus_the_ramp_reach() {
    let font = Font::parse(FONT_DATA).expect("font");
    let id = font.cmap_lookup('}').expect("glyph");
    let outline = font.outline_scaled_by_id(id, 48.0).expect("outline");
    let [x0, y0, x1, y1] = outline.bounds().expect("bounds");
    let support = font.glyph_scaled_by_id(id, 48.0).expect("glyph").support;
    assert_eq!(
        support.bounds(),
        [
            x0 - loop_blinn::RAMP_REACH,
            y0 - loop_blinn::RAMP_REACH,
            x1 + loop_blinn::RAMP_REACH,
            y1 + loop_blinn::RAMP_REACH
        ]
    );
}

// ────────────────────────────── the two forms agree ──────────────────────────────

/// The single kernel and the union of cells compute the same function, and
/// not merely the same classification: each cell drops the pieces that
/// cannot reach it and picks its own reference point, and neither changes a
/// value. What is left is the compiler's — a cell is a different arena at a
/// different extent, so extraction may schedule it differently and round
/// differently — so the bound is one 8-bit coverage step, an order of
/// magnitude above the worst measured (1.1e-3 ≈ 0.3 of a step), and far
/// below the half a pixel a dropped piece would move.
#[test]
fn cells_and_the_single_kernel_agree_to_within_the_compiler() {
    let font = Font::parse(FONT_DATA).expect("font");
    let mut worst = 0.0f32;
    let mut worst_where = String::new();
    let mut histogram = [0usize; 11];
    for size in [24.0f32] {
        let n = (size * 1.5).ceil() as usize;
        let lattice = Lattice::frame(n, n);
        for ch in ['O', '8', 'A', 'g', 'W', '&'] {
            let id = font.cmap_lookup(ch).expect("glyph");
            let outline = font.outline_scaled_by_id(id, size).expect("outline");
            let single = bake_single(&outline, lattice);
            let cells = bake_cells(&outline, lattice, [16, 16]);
            for (k, (a, b)) in single.iter().zip(&cells).enumerate() {
                let d = (a - b).abs();
                histogram[(d * 10.0).round() as usize] += 1;
                if d > worst {
                    worst = d;
                    worst_where = format!(
                        "{ch}@{size} at ({}, {}): single {a} cells {b}",
                        k % n,
                        k / n
                    );
                }
            }
        }
    }
    eprintln!("|single − cells| histogram in tenths: {histogram:?}; worst {worst_where}");
    assert!(
        worst < 1.0 / 255.0,
        "the two forms differ by {worst} ({worst_where}), more than one 8-bit \
         coverage step — that is a dropped piece or a moved reference point, \
         not a schedule"
    );
}
