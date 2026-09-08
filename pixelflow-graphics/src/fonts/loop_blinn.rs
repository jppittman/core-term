//! Coverage of an outline as a [`Kernel`]: an exact winding number decides
//! inside from outside, a distance decides how much of the pixel, and
//! Loop–Blinn's implicit is what makes a *curved* edge as cheap to ask
//! about as a straight one.
//!
//! ## A glyph is a circle with a sum where the sign used to be
//!
//! A circle is easy because it arrives as a formula whose sign is the
//! answer: `x² + y² − 1`, negative inside. Sign says inside, magnitude says
//! how far to the edge, and both come from one polynomial.
//!
//! A glyph needs the same two things and cannot get them from one
//! polynomial, because its edge is many pieces rather than one:
//!
//! ```text
//! inside   = (Σ crossings) ≠ 0          // the sign
//! distance = min over boundary pieces   // the magnitude
//! coverage = inside ? ½ + distance : ½ − distance
//! ```
//!
//! A **crossing** is the schoolbook inside test: fire a ray from the sample
//! to `+X`, count the edges it meets on the way out — `+1` for an edge
//! running upward, `−1` downward. Nonzero means inside. That sum is the
//! winding number, and the non-zero rule is what TrueType specifies.
//!
//! For a straight edge a crossing is two comparisons and a multiply-add,
//! and all but the final comparison is a function of `Y` alone, so it
//! hoists into the row prologue. That is the *good* kind of Y-only work.
//! What made the scanline formulation slow was hoisting a **root solve** —
//! discriminant, square root, two roots, per segment per row — and there is
//! none here, because every crossing is against a straight chord.
//!
//! ## What Loop–Blinn contributes, and it is exactly one thing
//!
//! Replace each curve by its straight chord and the crossing count above is
//! wrong by precisely one region: the crescent between the chord and the
//! real curve. Loop and Blinn give the membership test for that crescent
//! without ever solving for where the curve is.
//!
//! The trick is that all parabolas are the same shape, the way all circles
//! are. Skew the plane — an affine map, so straight lines stay straight —
//! until your particular arc lands on the one parabola `v = u²`. Then
//! "which side of the curve" is the sign of `u² − v`, where `u` and `v` are
//! each just `a·X + b·Y + c`. A formula whose sign is the answer, exactly
//! like the circle. Root-finding becomes arithmetic, and the `'8'` waist
//! defect — two segments disagreeing about their own discriminant within an
//! ulp at a tangency — is not expressible.
//!
//! The skew is a **contramap**, six numbers per curve computed on the host.
//! It only lands *your arc* on the parabola; keep going and you are on the
//! rest of an infinite parabola that is not your curve, where `u² − v` is
//! not slow but **wrong**. So the test is fenced — and the fence is one
//! half-plane, the chord's: `crescent = {v ≥ u²} ∩ {v ≤ u}`, which is
//! already bounded, because `u² ≤ v ≤ u` holds nowhere but `u ∈ [0, 1]`.
//! A GPU spends a whole rasterized triangle on that fence; two comparisons
//! do it here. The domain-side extent this stage needed is for the
//! *string*, not the arc — see [`cells`].
//!
//! ## Only a boundary softens a pixel
//!
//! `distance` is a minimum over the pieces — but over the pieces that are
//! actually *boundaries*, not over every piece the font draws. A stroke
//! built from two overlapping contours, or two overlapping components of a
//! compound glyph, puts drawn edges inside the ink, where coverage should
//! stay saturated. Minimising over those too dims a line straight through
//! solid fill: measured at exactly ½ along the shared edges of two
//! overlapping squares, a seam one pixel wide.
//!
//! Whether a piece is a boundary is a question the winding already answers.
//! With `w₋` the winding minus that piece's own contribution, its two sides
//! are `w₋` and `w₋ + dir` — they differ by one, so never both zero — and
//! the piece separates ink from no-ink exactly when one of them is zero.
//!
//! ## The winding is exact; only the distance is soft
//!
//! Every winding term is a hard mask selecting a signed constant, so the sum
//! is an integer. Keeping that separate from the ramp is the point: an
//! earlier draft made a crossing's *existence* and its *coverage* one
//! number, so a comparison landing on the wrong side of an edge moved
//! coverage by half a unit instead of by a rounding. That coupling is the
//! `'8'` defect's whole family.
//!
//! Distances are in **pixels of the lattice being collapsed**, not of the
//! frame the outline lives in: each is divided by the gradient magnitude of
//! its own edge function, with `DX`/`DY` resolved symbolically at bake, so
//! the chain rule carries the scale through every enclosing `Kernel::at`.
//! For an affine edge function that normalisation folds to a compile-time
//! constant.
//!
//! Per piece the distance is the larger of two estimates, because neither
//! is usable alone: the **capsule distance to the chord** (exact for a
//! line; for a curve a lower bound once the curve's deviation is
//! subtracted) and the **implicit's own** `|f|/‖∇f‖` (accurate near the arc
//! and unsound away from it — it underestimates where the curve is sharp,
//! and an underestimate past the ramp is a saturation failure, not a
//! rounding). A lower bound never exceeds the truth, so the larger of the
//! two is the closer. That is also why a segment straying more than half a
//! pixel (`MAX_DEVIATION`) from its chord is halved until it does not.
//!
//! ## The domain-side form
//!
//! [`cells`] cuts a lattice into rectangles and gives each only what it
//! needs. The pruning follows from the ray: a chord spanning every row of a
//! rectangle and lying wholly to one side is crossed by every ray from it
//! at the same sign, so it folds into a host-side constant; a chord whose Y
//! range ends inside those rows must be evaluated; a chord missing those
//! rows is dropped. Slivers and ramps are regions rather than rays, and
//! prune by reach. Rectangles needing no term are a constant.
//!
//! ## What is exactly zero, and where
//!
//! Beyond half a pixel from every boundary the ramp is saturated and the
//! winding is exact, so coverage is exactly 0 or exactly 1 — and outside
//! the outline it is 0. [`glyph`] cuts its kernel to the outline's bounding
//! box dilated by [`RAMP_REACH`] and reports that box as its [`Support`]:
//! exact, whatever the glyph's shape.

use super::outline::{Outline, Point, Segment};
use super::PIXEL_CENTER;
use pixelflow_core::{DiscreteManifold, IndexRange, Kernel, Lattice};
use pixelflow_ir::arena::BufferIdentity;
use std::sync::Arc;

/// How far coverage can reach past the outline, in the frame the kernel is
/// built in. A ramp is one unit wide and centred on its edge, so half a unit
/// is the analytic reach; a whole unit keeps the bound a whole pixel when
/// that frame is the screen, at no cost — the mask sits where every ramp is
/// already saturated.
pub const RAMP_REACH: f32 = 1.0;

/// Gradient floor for every ramp, so a degenerate edge divides by something.
const MIN_GRADIENT: f32 = 1e-3;

/// A chord shorter than this crosses nothing and is dropped.
const ZERO_LENGTH: f64 = 1e-6;

/// How far a segment may stray from its chord, in the kernel's own units,
/// before it is split in two.
///
/// It is an **accuracy** bound on the antialiasing ramp and nothing else.
/// The ramp's distance is the implicit's, taken against a lower bound built
/// from the chord — and that bound is the chord's own distance less this
/// deviation, so a fatter curve makes the bound looser, and past half a
/// pixel it stops bounding anything useful (measured: a ring whose control
/// points sat at its bounding square's corners, deviation 5 px, softened
/// every sample within 5 px of a chord to exactly ½). Splitting is exact —
/// de Casteljau halves a quadratic into two quadratics — so the *winding*
/// never sees this constant, and no split is ever needed for correctness.
const MAX_DEVIATION: f64 = 0.5;

/// Splits a segment will take before it is accepted however fat it is. Each
/// halving quarters the deviation, so six cover a curve straying 2000× the
/// bound; past that the ramp is simply the implicit's, unbounded below.
const MAX_SPLITS: u32 = 6;

/// A curve straying less than this from its chord is drawn as its chord.
/// Dropping a sliver moves coverage by at most the sliver's own width, so
/// this is an order of magnitude below one 8-bit step — and fonts are full
/// of quadratics that are straight to within it, each of which would
/// otherwise carry a sliver and an implicit for nothing.
const FLAT_ENOUGH: f64 = 1.0 / 256.0;

// ═══════════════════════════════════════════════════════════════════════════
// The glyph, and the box outside which it is exactly zero
// ═══════════════════════════════════════════════════════════════════════════

/// A coverage kernel together with the box outside which it is exactly
/// zero. They travel together because they are derived together — a support
/// restated separately from the composition it describes is a future
/// divergence.
#[derive(Clone)]
pub struct Glyph {
    /// The coverage kernel.
    pub kernel: Kernel,
    /// Where it can be nonzero.
    pub support: Support,
    /// The piece coefficient table [`kernel`](Self::kernel)'s winding sum
    /// reads at its `Kernel::sum_over` binder ([`glyph`]'s S1a rewrite —
    /// docs/plans/2026-09-09-glyph-as-a-fold-execution.md).
    ///
    /// `None` for the empty glyph (a space: no outline, the kernel is the
    /// literal constant `0.0` and declares no buffer). Otherwise `Some` and
    /// **required**: `kernel` cannot be baked or collapsed without it bound
    /// (`Manifold::bind`) — a bare `Lattice::bake` panics on the declared,
    /// unbound slot. `binding` and `kernel` are minted together by the same
    /// call so their identity always agrees; a caller must not try to
    /// recompute one without the other.
    pub binding: Option<(BufferIdentity, Arc<Vec<f32>>)>,
}

/// **The box outside which a coverage [`Kernel`] is exactly zero.**
///
/// The outline's bounding box dilated by [`RAMP_REACH`], in the frame the
/// kernel was built in; [`glyph`] cuts its kernel to this box, so the claim
/// holds under any coordinate warp and not only where the ramps' own
/// saturation makes it true. Coordinates are `[x0, y0, x1, y1]`; a box with
/// no area is [`Support::EMPTY`] and meets nothing.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Support([f32; 4]);

impl Support {
    /// No support at all: a glyph with no outline is the constant 0 everywhere.
    pub const EMPTY: Self = Self([0.0, 0.0, 0.0, 0.0]);

    /// The bounding box `[x0, y0, x1, y1]` dilated by [`RAMP_REACH`].
    fn around([x0, y0, x1, y1]: [f32; 4]) -> Self {
        Self([
            x0 - RAMP_REACH,
            y0 - RAMP_REACH,
            x1 + RAMP_REACH,
            y1 + RAMP_REACH,
        ])
    }

    /// `[x0, y0, x1, y1]`.
    #[must_use]
    pub fn bounds(self) -> [f32; 4] {
        self.0
    }

    /// Whether the box encloses no samples.
    #[must_use]
    pub fn is_empty(self) -> bool {
        self.0[2] <= self.0[0] || self.0[3] <= self.0[1]
    }

    /// The box shifted `dx` along X.
    #[must_use]
    pub fn shifted_x(self, dx: f32) -> Self {
        if self.is_empty() {
            return Self::EMPTY;
        }
        Self([self.0[0] + dx, self.0[1], self.0[2] + dx, self.0[3]])
    }
}

/// The coverage of `outline` as ONE kernel over the whole plane, in the
/// outline's own frame, together with its [`Support`].
///
/// Every chord and every sliver contributes at every sample, so the cost is
/// linear in the outline's segment count; that is the price of a kernel with
/// no extent, and the reason [`cells`] exists. The kernel is cut to its
/// support by a mask whose false arm is the literal 0, so it composes into a
/// larger scene as a guardable arm.
#[must_use]
pub fn glyph(outline: &Outline) -> Glyph {
    let pieces = Pieces::of(outline);
    let (Some(bounds), false) = (outline.bounds(), pieces.is_empty()) else {
        return Glyph {
            kernel: constant(0.0),
            support: Support::EMPTY,
            binding: None,
        };
    };

    // The winding sum, as ONE `Kernel::sum_over` reading a bound table: row
    // `i` is piece `i`'s [`piece_row`], and the reduce binder supplies the
    // row index, so the arena holds one `Reduce` node rather than `n`
    // per-piece fragments folded in Rust (S1a of
    // docs/plans/2026-09-09-glyph-as-a-fold-execution.md — only the winding;
    // the distance/boundary fold below is unconverted and still builds a
    // per-piece constant fragment per [`prepare`]).
    let n = u32::try_from(pieces.pieces.len())
        .expect("a glyph outline has far fewer pieces than u32::MAX");
    let data: Vec<f32> = pieces
        .pieces
        .iter()
        .flat_map(|p| piece_row(*p))
        .collect();
    let id = BufferIdentity::mint();
    let table = DiscreteManifold::kernel_for(id, PIECE_ROW_COLS as u32, n);
    let winding = Kernel::sum_over(n, |row| {
        let coeff = |k: usize| table.at(&Kernel::constant(k as f32), row);
        crossing_term(&coeff).add(&sliver_term(&coeff))
    });

    // Every piece is present unconditionally: with the whole outline in the
    // sum above, the ray from any sample is fully accounted for.
    let included: Vec<Included> = pieces
        .pieces
        .iter()
        .map(|p| Included {
            piece: *p,
            crossing: true,
            bulge: p.bulge,
            ramp: true,
        })
        .collect();
    let prepared = prepare(&included);
    let coverage = coverage(winding, &prepared);

    let support = Support::around(bounds);
    let [x0, y0, x1, y1] = support.bounds();
    let inside = Kernel::x()
        .ge(&constant(x0))
        .and(&Kernel::x().le(&constant(x1)))
        .and(&Kernel::y().ge(&constant(y0)))
        .and(&Kernel::y().le(&constant(y1)));
    Glyph {
        kernel: inside.select(&coverage, &constant(0.0)),
        support,
        binding: Some((id, Arc::new(data))),
    }
}

/// The coverage of `outline` as a **union of index ranges** over `lattice`:
/// the domain-side form, one summand per `cell`-sized rectangle of samples
/// that the outline reaches.
///
/// `outline` is in the lattice's continuous frame, sampled at pixel centres
/// — index `(i, j)` reads coordinate `(i + ½, j + ½)`, this crate's shared
/// convention — because a `Union` summand collapses by pure index and has no
/// coordinate frame of its own to lend it. The contramap is folded into the
/// geometry here rather than applied as a `Kernel::at`, so each cell's
/// constants are exact.
///
/// Each summand holds only what its own rectangle needs, and what a
/// rectangle needs follows from the ray the winding is counted along. A
/// chord that spans *every* row of the rectangle and lies wholly to one
/// side of it is crossed by every ray from it, at the same sign — so it is
/// folded into the summand's constant on the host, and no term is emitted.
/// A chord whose Y range ends inside those rows, or which passes through
/// them, is crossed at a row-dependent place and must be evaluated. A chord
/// missing those rows entirely is crossed by nothing here and is dropped.
/// Slivers and ramps are pruned by reach instead, since both are regions
/// rather than rays.
///
/// Rectangles that need no term at all are the constant coverage of their
/// interior, and those where that constant is 0 are not placed. Place the
/// result with `Union::place`; the ranges are disjoint by construction.
///
/// # Panics
///
/// Panics on a zero cell extent.
#[must_use]
pub fn cells(outline: &Outline, lattice: Lattice, cell: [usize; 2]) -> Vec<(IndexRange, Kernel)> {
    assert!(
        cell[0] > 0 && cell[1] > 0,
        "loop_blinn::cells: a cell must have samples, got {cell:?}"
    );
    let indexed = outline.translated([-PIXEL_CENTER, -PIXEL_CENTER]);
    let pieces = Pieces::of(&indexed);
    let [ex, ey] = lattice.extent.map(|e| e as usize);
    let mut out = Vec::new();
    if pieces.is_empty() {
        return out;
    }
    for y0 in (0..ey).step_by(cell[1]) {
        for x0 in (0..ex).step_by(cell[0]) {
            let (w, h) = (cell[0].min(ex - x0), cell[1].min(ey - y0));
            let samples = Rect {
                x0: x0 as f64,
                y0: y0 as f64,
                x1: (x0 + w - 1) as f64,
                y1: (y0 + h - 1) as f64,
            };
            let reach = samples.dilated(f64::from(RAMP_REACH));
            let mut constant_winding = 0.0f64;
            let included: Vec<Included> = pieces
                .pieces
                .iter()
                .filter_map(|p| {
                    // A contour that does not reach this rectangle encloses
                    // none of its samples, so it contributes no winding
                    // here and no ramp: the whole character drops at once.
                    let [bx0, by0, bx1, by1] = p.contour_bounds;
                    if bx1 < reach.x0 || bx0 > reach.x1 || by1 < reach.y0 || by0 > reach.y1 {
                        return None;
                    }
                    let [cy0, cy1] = p.chord.y_range();
                    let [cx0, _] = p.chord.x_range();
                    // Half-open in Y, matching `crossing_term`'s own rule.
                    let spans_all_rows = cy0 <= samples.y0 && cy1 > samples.y1;
                    let misses_all_rows =
                        p.chord.is_horizontal() || cy1 <= samples.y0 || cy0 > samples.y1;
                    let crossing = if misses_all_rows {
                        false
                    } else if spans_all_rows && cx0 > samples.x1 {
                        // Crossed by every ray from this rectangle, always
                        // to the right: one number, the same at every
                        // sample, so the host adds it and the kernel does
                        // not carry a term.
                        constant_winding += p.chord.direction();
                        false
                    } else {
                        // Wholly left of the rectangle and spanning it
                        // contributes nothing to any ray; anything else has
                        // to be asked per sample.
                        !(spans_all_rows && p.chord.x_range()[1] <= samples.x0)
                    };
                    // The ramp's lower bound is the chord's distance less
                    // the curve's deviation, so a curve softens a sample
                    // only within the reach *plus* that deviation.
                    let ramp = samples
                        .dilated(f64::from(RAMP_REACH) + p.deviation())
                        .meets_segment(p.chord.a, p.chord.b);
                    // A curve's ramp is the implicit taken against that
                    // bound, so wherever the ramp is built the sliver is
                    // too — the bound alone is not an estimate, only a
                    // floor. Where the ramp is not built the sliver is
                    // still needed if its triangle reaches the cell: it
                    // carries winding, not just a distance.
                    let bulge = p
                        .bulge
                        .filter(|b| ramp || reach.meets_triangle(b.p0, b.p1, b.p2));
                    (crossing || bulge.is_some() || ramp).then_some(Included {
                        piece: *p,
                        crossing,
                        bulge,
                        ramp,
                    })
                })
                .collect();
            let range = IndexRange::new(x0, y0, w, h);
            if included.is_empty() {
                // Nothing varies here: the constant alone decides, and an
                // uncovered rectangle is not placed at all.
                if constant_winding != 0.0 {
                    out.push((range, constant(1.0)));
                }
                continue;
            }
            // Per-piece constants, unconverted (S1a is `glyph`'s table-read
            // winding only — see its doc comment): `prepare` builds each
            // included piece's own term from its own row, and this cell's
            // winding is their sum plus the pieces already folded into the
            // host constant.
            let prepared = prepare(&included);
            let terms: Vec<Kernel> = std::iter::once(constant_of(constant_winding))
                .chain(prepared.iter().map(|p| p.own.clone()))
                .collect();
            let winding = Kernel::sum(&terms);
            out.push((range, coverage(winding, &prepared)));
        }
    }
    out
}

// ═══════════════════════════════════════════════════════════════════════════
// Host geometry: the outline as chords and slivers, in f64
// ═══════════════════════════════════════════════════════════════════════════

type P = [f64; 2];

fn wide([x, y]: Point) -> P {
    [f64::from(x), f64::from(y)]
}

/// The signed area of `(o, a, b)`, doubled: positive where `b` is
/// counter-clockwise from `a` around `o` (in a y-up frame; the sign is only
/// ever compared with itself).
fn cross(o: P, a: P, b: P) -> f64 {
    (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
}

/// Distance from `p` to the closed segment `chord`.
fn point_to_segment(p: P, chord: Chord) -> f64 {
    let (ex, ey) = (chord.b[0] - chord.a[0], chord.b[1] - chord.a[1]);
    let len2 = ex * ex + ey * ey;
    let t = if len2 == 0.0 {
        0.0
    } else {
        (((p[0] - chord.a[0]) * ex + (p[1] - chord.a[1]) * ey) / len2).clamp(0.0, 1.0)
    };
    let (qx, qy) = (chord.a[0] + t * ex, chord.a[1] + t * ey);
    ((p[0] - qx).powi(2) + (p[1] - qy).powi(2)).sqrt()
}

/// A straight chord `a → b`, never of zero length.
#[derive(Clone, Copy, Debug)]
struct Chord {
    a: P,
    b: P,
}

impl Chord {
    fn length(self) -> f64 {
        ((self.b[0] - self.a[0]).powi(2) + (self.b[1] - self.a[1]).powi(2)).sqrt()
    }

    /// `[min, max]` of the chord's Y, and of its X.
    fn y_range(self) -> [f64; 2] {
        [self.a[1].min(self.b[1]), self.a[1].max(self.b[1])]
    }
    fn x_range(self) -> [f64; 2] {
        [self.a[0].min(self.b[0]), self.a[0].max(self.b[0])]
    }

    /// `+1` where the chord runs in the direction of increasing Y, `−1`
    /// where it runs the other way, and **`0` where it is horizontal** —
    /// the sign a ray crossing it picks up, and a horizontal chord is
    /// crossed by none.
    ///
    /// That third case is not a special case, it is the answer: a
    /// horizontal edge carries no winding jump at all. The jump across the
    /// top of a rectangle is carried by the *vertical* edges, whose
    /// half-open Y bands end there. Returning `−1` for it — which this did,
    /// because `b.y > a.y` is false when they are equal — fabricates a jump
    /// of one wherever a caller asks a horizontal chord which side of it
    /// you are on, and every caller had to remember to ask `is_horizontal`
    /// first. Two did; one did not.
    fn direction(self) -> f64 {
        match self.is_horizontal() {
            true => 0.0,
            false => match self.b[1] > self.a[1] {
                true => 1.0,
                false => -1.0,
            },
        }
    }

    /// Whether the chord is horizontal, and so is crossed by no horizontal
    /// ray: it spans no row, and contributes nothing to any winding.
    fn is_horizontal(self) -> bool {
        (self.b[1] - self.a[1]).abs() < ZERO_LENGTH
    }
}

/// The sliver a quadratic bulges past its chord: its control triangle, and
/// the sign the sliver's winding carries.
#[derive(Clone, Copy, Debug)]
struct Bulge {
    p0: P,
    p1: P,
    p2: P,
    /// `sign(cross(p0, p1, p2))`: the control triangle's orientation.
    sign: f64,
}

/// One outline segment, prepared: its chord, and its sliver if it curves
/// enough to have one.
#[derive(Clone, Copy, Debug)]
struct Piece {
    chord: Chord,
    bulge: Option<Bulge>,
    /// Whether this piece lies under *other* contours' ink, so that it may
    /// be inside the filled region rather than bound it. Only these pieces
    /// pay for the boundary test in [`coverage`], and paying is expensive:
    /// each test references the whole winding sum, and every `Kernel`
    /// combinator copies its arena, so gating every piece makes a
    /// whole-string kernel quadratic.
    ///
    /// Measured on this crate's own font: of 189 printable and Latin-1
    /// glyphs, exactly two need it — `Ç` and `ç`, where the cedilla
    /// overlaps the C. Bounding boxes are far too coarse a proxy (adjacent
    /// glyphs in a string overlap constantly without their ink ever
    /// touching), so this asks the winding directly.
    may_be_interior: bool,
    /// The box containing this piece's whole **contour**.
    ///
    /// A closed contour that does not enclose a sample contributes exactly
    /// zero to its winding — a ray from the sample crosses it an even
    /// number of times, netting nothing. So a sample outside this box need
    /// not evaluate *any* of the contour's pieces, which is what lets a
    /// cell drop a whole character rather than pruning its chords one at a
    /// time against a ray that runs to infinity.
    contour_bounds: [f64; 4],
}

impl Piece {
    /// A straight piece, or nothing when its ends coincide: a zero-length
    /// chord crosses nothing.
    fn line(a: P, b: P) -> Option<Self> {
        let chord = Chord { a, b };
        (chord.length() >= ZERO_LENGTH).then_some(Self {
            chord,
            bulge: None,
            may_be_interior: false,
            contour_bounds: [0.0; 4],
        })
    }

    /// A point on the segment, for asking what else covers it.
    fn midpoint(self) -> P {
        match self.bulge {
            // B(½) = (p0 + 2p1 + p2) / 4.
            Some(b) => [
                (b.p0[0] + 2.0 * b.p1[0] + b.p2[0]) / 4.0,
                (b.p0[1] + 2.0 * b.p1[1] + b.p2[1]) / 4.0,
            ],
            None => [
                (self.chord.a[0] + self.chord.b[0]) / 2.0,
                (self.chord.a[1] + self.chord.b[1]) / 2.0,
            ],
        }
    }

    /// The quadratic `p0 → p1 → p2` as pieces, each straying no more than
    /// [`MAX_DEVIATION`] from its own chord, appended to `out`. A curve
    /// flatter than that is one piece; a fatter one is halved at its
    /// midpoint (de Casteljau, exact) and each half asked again. A piece
    /// whose curve is flat enough to be a line is one — its sliver would be
    /// empty and its implicit degenerate.
    fn quad(p0: P, p1: P, p2: P, splits: u32, out: &mut Vec<Self>) {
        let chord = Chord { a: p0, b: p2 };
        if chord.length() < ZERO_LENGTH {
            // A closed loop: the curve leaves and returns, enclosing no
            // area on either side of a chord that does not exist.
            return;
        }
        let area2 = cross(p0, p1, p2);
        // The same measure [`Piece::deviation`] reports, because both
        // decisions below are about the quantity the ramp's bound
        // subtracts. Measuring against the chord's infinite *line* here —
        // `|area2| / length / 2`, which this used to do — is the very trap
        // that method's doc names, and it fails in both directions on the
        // hook it cites: `(0,0) → (−100, 0.1) → (1,0)` reads as 0.05 and is
        // never split though the curve strays 49.75, and flattening the
        // control point to `(−100, 0.005)` reads as 0.0025, under
        // `FLAT_ENOUGH`, so the whole hook is discarded as a straight line.
        // That one changes the *winding*, which is never approximated.
        let deviation = point_to_segment(p1, chord);
        if deviation < FLAT_ENOUGH {
            out.extend(Self::line(p0, p2));
            return;
        }
        if splits > 0 && deviation > MAX_DEVIATION {
            let mid = |a: P, b: P| [(a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0];
            let (l, r) = (mid(p0, p1), mid(p1, p2));
            let m = mid(l, r);
            Self::quad(p0, l, m, splits - 1, out);
            Self::quad(m, r, p2, splits - 1, out);
            return;
        }
        out.push(Self {
            chord,
            bulge: Some(Bulge {
                p0,
                p1,
                p2,
                sign: area2.signum(),
            }),
            may_be_interior: false,
            contour_bounds: [0.0; 4],
        });
    }

    /// How far the segment strays from its chord *segment*: 0 for a line,
    /// and for a curve the distance from its control point to the chord.
    ///
    /// The curve lies in the convex hull of its three control points, and
    /// distance to a convex set is itself convex, so its maximum over the
    /// hull is attained at a vertex. Two of those vertices are the chord's
    /// own ends, at distance 0. So the control point's distance bounds the
    /// whole curve's, which is what the ramp's lower bound and the cell's
    /// pruning both need.
    ///
    /// Not the perpendicular distance to the chord's infinite *line*: a
    /// control point can project far outside the chord's span — the curve
    /// `(0,0) → (−100, 0.1) → (1,0)` deviates 0.05 from the line and 50
    /// from the segment — and a bound that misses that is not a bound.
    fn deviation(self) -> f64 {
        self.bulge
            .map_or(0.0, |b| point_to_segment(b.p1, self.chord))
    }
}

/// The box containing every control point of a contour, and so the whole
/// contour.
fn contour_bounds(pieces: &[Piece]) -> [f64; 4] {
    let mut b = [
        f64::INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
    ];
    for piece in pieces {
        for p in [piece.chord.a, piece.chord.b]
            .into_iter()
            .chain(piece.bulge.map(|g| g.p1))
        {
            b[0] = b[0].min(p[0]);
            b[1] = b[1].min(p[1]);
            b[2] = b[2].max(p[0]);
            b[3] = b[3].max(p[1]);
        }
    }
    b
}

/// The winding of a contour's chords at `p`, by a horizontal ray. Chords
/// rather than curves: this decides only whether a piece needs the boundary
/// test, and a curve's crescent is at most [`MAX_DEVIATION`] wide, so a
/// point that close to the answer is already inside the ramp.
fn chord_winding(pieces: &[Piece], p: P) -> i32 {
    let mut w = 0;
    for piece in pieces {
        let (a, b) = (piece.chord.a, piece.chord.b);
        if (a[1] <= p[1]) == (b[1] <= p[1]) {
            continue;
        }
        let t = (p[1] - a[1]) / (b[1] - a[1]);
        if a[0] + t * (b[0] - a[0]) > p[0] {
            w += if b[1] > a[1] { 1 } else { -1 };
        }
    }
    w
}

/// An outline prepared for the kernel.
struct Pieces {
    pieces: Vec<Piece>,
}

impl Pieces {
    fn of(outline: &Outline) -> Self {
        // Per contour, so a piece can be asked whether any *other* contour
        // reaches it. Contours are what overlap in a font — a stroke drawn
        // as two shapes, a compound glyph's components — and an edge under
        // another contour's ink is not a boundary.
        let mut by_contour: Vec<Vec<Piece>> = Vec::with_capacity(outline.contours.len());
        for contour in &outline.contours {
            let mut pieces = Vec::new();
            for segment in &contour.segments {
                match *segment {
                    Segment::Line { from, to } => {
                        pieces.extend(Piece::line(wide(from), wide(to)));
                    }
                    Segment::Quad { from, control, to } => {
                        Piece::quad(wide(from), wide(control), wide(to), MAX_SPLITS, &mut pieces);
                    }
                }
            }
            by_contour.push(pieces);
        }
        // A piece is possibly-interior exactly when the *other* contours
        // already cover it: their winding at its midpoint is nonzero. Asked
        // directly rather than through bounding boxes, because boxes touch
        // whenever glyphs sit side by side and ink almost never does.
        let bounds: Vec<[f64; 4]> = by_contour.iter().map(|c| contour_bounds(c)).collect();
        let mut pieces = Vec::new();
        for (i, contour) in by_contour.iter().enumerate() {
            for piece in contour {
                let mid = piece.midpoint();
                let covered = by_contour
                    .iter()
                    .enumerate()
                    .filter(|(j, _)| *j != i)
                    .any(|(_, other)| chord_winding(other, mid) != 0);
                pieces.push(Piece {
                    may_be_interior: covered,
                    contour_bounds: bounds[i],
                    ..*piece
                });
            }
        }
        Self { pieces }
    }

    fn is_empty(&self) -> bool {
        self.pieces.is_empty()
    }
}

/// An axis-aligned rectangle `[x0, x1] × [y0, y1]`, closed.
#[derive(Clone, Copy, Debug)]
struct Rect {
    x0: f64,
    y0: f64,
    x1: f64,
    y1: f64,
}

impl Rect {
    fn dilated(self, r: f64) -> Self {
        Self {
            x0: self.x0 - r,
            y0: self.y0 - r,
            x1: self.x1 + r,
            y1: self.y1 + r,
        }
    }

    fn centre(self) -> P {
        [(self.x0 + self.x1) / 2.0, (self.y0 + self.y1) / 2.0]
    }

    /// Whether the closed segment `a → b` has a point in the rectangle
    /// (Liang–Barsky clipping of the parameter interval).
    fn meets_segment(self, a: P, b: P) -> bool {
        let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
        let (mut t0, mut t1) = (0.0f64, 1.0f64);
        for (p, q) in [
            (-dx, a[0] - self.x0),
            (dx, self.x1 - a[0]),
            (-dy, a[1] - self.y0),
            (dy, self.y1 - a[1]),
        ] {
            if p == 0.0 {
                if q < 0.0 {
                    return false;
                }
                continue;
            }
            let t = q / p;
            if p < 0.0 {
                t0 = t0.max(t);
            } else {
                t1 = t1.min(t);
            }
        }
        t0 <= t1
    }

    /// Whether the closed triangle has a point in the rectangle: an edge
    /// crosses it, or (the triangle contains the rectangle) its centre is
    /// inside the triangle.
    fn meets_triangle(self, p0: P, p1: P, p2: P) -> bool {
        if self.meets_segment(p0, p1) || self.meets_segment(p1, p2) || self.meets_segment(p2, p0) {
            return true;
        }
        let c = self.centre();
        let (s0, s1, s2) = (cross(p0, p1, c), cross(p1, p2, c), cross(p2, p0, c));
        (s0 >= 0.0 && s1 >= 0.0 && s2 >= 0.0) || (s0 <= 0.0 && s1 <= 0.0 && s2 <= 0.0)
    }
}

/// A piece as one cell sees it: which of its terms that cell needs.
#[derive(Clone, Copy, Debug)]
struct Included {
    piece: Piece,
    /// Whether a ray from a sample in the cell can cross this chord at a
    /// row-dependent place — i.e. the chord's Y range ends inside the
    /// cell's rows, or it passes through them. A chord that spans every row
    /// of the cell and lies wholly to one side contributes the same sign at
    /// every sample, and is folded into the cell's constant instead.
    crossing: bool,
    /// The sliver, if its control triangle reaches the cell.
    bulge: Option<Bulge>,
    /// Whether the segment is near enough to soften a sample in the cell.
    ramp: bool,
}

// ═══════════════════════════════════════════════════════════════════════════
// The kernel
// ═══════════════════════════════════════════════════════════════════════════

fn constant(v: f32) -> Kernel {
    Kernel::constant(v)
}

fn constant_of(v: f64) -> Kernel {
    Kernel::constant(v as f32)
}

// ─────────────────────────────────────────────────────────────────────────
// The piece row: one layout, two coefficient sources
// ─────────────────────────────────────────────────────────────────────────
//
// [`crossing_term`] and [`sliver_term`] read a piece's numbers by column
// rather than by field, so they have exactly one definition regardless of
// where the numbers come from: [`glyph`] reads column `k` of row `i` from a
// bound table at the `Kernel::sum_over` binder (`i` varies per sample-batch
// iteration, unrolled); [`prepare`] (feeding both `cells` and `glyph`'s
// distance/boundary fold) reads column `k` of one piece's own row, folded
// in as a `Kernel` constant. Same body, two [`Coeff`] sources — the
// `Scalar::{Const, Uniform}` distinction the codebase already has
// elsewhere, here a closure.
//
// A line is a quadratic whose sliver sign (column 12) is 0, so it carries
// no per-kind branch: [`sliver_term`]'s mask is always true where the u/v
// columns are all 0 (`0 ≥ 0` and `0·0 − 0 ≤ 0`), and it always selects the
// `0.0` sign either way — never a mask multiplied by a coefficient (see
// "Floating point at the edges" in CLAUDE.md: a mask is a bit pattern, and
// `mask * 0.0` is not `0.0` when the mask is all-ones-as-NaN).

/// `y_min` of the chord.
const COL_Y_MIN: usize = 0;
/// `y_max` of the chord.
const COL_Y_MAX: usize = 1;
/// `dx/dy` of the chord — `0.0` for a horizontal chord (H1: the crossing
/// band is empty for one regardless, so this is never read as a divisor,
/// but a finite placeholder keeps an `inf` from ever entering the table).
const COL_DX_OVER_DY: usize = 2;
/// `a.x`, the chord's start.
const COL_AX: usize = 3;
/// `a.y`, the chord's start.
const COL_AY: usize = 4;
/// The chord's `direction` (±1 running, `0.0` for a horizontal chord).
const COL_DIRECTION: usize = 5;
/// `u`'s affine coefficients `(a, b, c)` — `0.0` for a line.
const COL_U_A: usize = 6;
const COL_U_B: usize = 7;
const COL_U_C: usize = 8;
/// `v`'s affine coefficients `(a, b, c)` — `0.0` for a line.
const COL_V_A: usize = 9;
const COL_V_B: usize = 10;
const COL_V_C: usize = 11;
/// The sliver's sign (±1), `0.0` for a line — the sum's identity.
const COL_SLIVER_SIGN: usize = 12;
/// Columns in one piece's row: `y_min, y_max, dx/dy, a.x, a.y, direction,
/// u.a, u.b, u.c, v.a, v.b, v.c, sliver_sign`.
const PIECE_ROW_COLS: usize = 13;

/// One piece's row, column `k`, as a [`Kernel`] — a bound-table read at a
/// binder ([`glyph`]) or a literal per-piece constant ([`prepare`]). See the
/// module section above.
type Coeff<'a> = &'a dyn Fn(usize) -> Kernel;

/// The affine maps taking `(X, Y)` to Loop–Blinn's canonical `(u, v)`,
/// landing `bulge`'s arc on the one parabola `v = u²` — computed once, on
/// the host, for [`piece_row`]. [`implicit_distance`] keeps its own copy:
/// the distance/boundary fold is out of scope for S1a (only the winding sum
/// converts to a table read), so it is left untouched rather than routed
/// through this.
fn bulge_uv(bulge: Bulge) -> (Linear, Linear) {
    let Bulge { p0, p1, p2, .. } = bulge;
    let e1 = [p1[0] - p0[0], p1[1] - p0[1]];
    let e2 = [p2[0] - p0[0], p2[1] - p0[1]];
    let det = e1[0] * e2[1] - e1[1] * e2[0];
    // Barycentric coordinates as affine functions of the sample: λ₁ weights
    // the control point, λ₂ the end point, λ₀ = 1 − λ₁ − λ₂ the start.
    let lambda1 = Linear {
        a: e2[1] / det,
        b: -e2[0] / det,
        c: (p0[1] * e2[0] - p0[0] * e2[1]) / det,
    };
    let lambda2 = Linear {
        a: -e1[1] / det,
        b: e1[0] / det,
        c: (e1[1] * p0[0] - e1[0] * p0[1]) / det,
    };
    // (u, v) = (λ₁/2 + λ₂, λ₂), so λ₂ = v and λ₁ = 2(u − v).
    let u = Linear {
        a: lambda1.a / 2.0 + lambda2.a,
        b: lambda1.b / 2.0 + lambda2.b,
        c: lambda1.c / 2.0 + lambda2.c,
    };
    (u, lambda2)
}

/// One piece's row in the coefficient table (host-side, `f32`) — see the
/// column layout above. A line's sliver columns (6–12) stay `0.0`, the
/// sum's identity.
fn piece_row(piece: Piece) -> [f32; PIECE_ROW_COLS] {
    let chord = piece.chord;
    let [y_min, y_max] = chord.y_range();
    // H1: never store an infinite dx/dy — see the module note above. The
    // direction needs no such guard: `Chord::direction` is already `0` for
    // a horizontal chord, which is the honest winding jump rather than a
    // placeholder.
    let dx_over_dy = match chord.is_horizontal() {
        true => 0.0,
        false => (chord.b[0] - chord.a[0]) / (chord.b[1] - chord.a[1]),
    };
    let direction = chord.direction();
    let mut row = [0.0f32; PIECE_ROW_COLS];
    row[COL_Y_MIN] = y_min as f32;
    row[COL_Y_MAX] = y_max as f32;
    row[COL_DX_OVER_DY] = dx_over_dy as f32;
    row[COL_AX] = chord.a[0] as f32;
    row[COL_AY] = chord.a[1] as f32;
    row[COL_DIRECTION] = direction as f32;
    if let Some(bulge) = piece.bulge {
        let (u, v) = bulge_uv(bulge);
        row[COL_U_A] = u.a as f32;
        row[COL_U_B] = u.b as f32;
        row[COL_U_C] = u.c as f32;
        row[COL_V_A] = v.a as f32;
        row[COL_V_B] = v.b as f32;
        row[COL_V_C] = v.c as f32;
        row[COL_SLIVER_SIGN] = bulge.sign as f32;
    }
    row
}

/// `a·X + b·Y + c`.
#[derive(Clone, Copy, Debug)]
struct Linear {
    a: f64,
    b: f64,
    c: f64,
}

impl Linear {
    fn kernel(self) -> Kernel {
        Kernel::x()
            .mul(&constant_of(self.a))
            .add(&Kernel::y().mul(&constant_of(self.b)))
            .add(&constant_of(self.c))
    }

    /// `n · (P − a)`.
    fn projection(a: P, n: P) -> Self {
        Self {
            a: n[0],
            b: n[1],
            c: -(n[0] * a[0] + n[1] * a[1]),
        }
    }
}

/// The minimum of `terms`, folded as a **balanced tree** rather than a
/// running accumulator.
///
/// Every `Kernel` combinator copies its arena, so `acc = acc.min(&d)` in a
/// loop copies a growing arena `n` times: quadratic. `Kernel::sum` is
/// special-cased against exactly this and says so; `min` has no such
/// variadic form, and a balanced fold gets the same effect from the outside
/// — each level copies `O(n)` in total and there are `log n` levels.
///
/// Measured on a 26-character string (about 700 pieces) before this: 4.7
/// million arena nodes and seven seconds to *construct* the kernel.
fn min_of(mut terms: Vec<Kernel>) -> Kernel {
    assert!(!terms.is_empty(), "min_of: no terms");
    while terms.len() > 1 {
        terms = terms
            .chunks(2)
            .map(|pair| match pair {
                [a, b] => a.min(b),
                [a] => a.clone(),
                _ => unreachable!("chunks(2) yields one or two"),
            })
            .collect();
    }
    terms.pop().expect("non-empty")
}

/// `value / (‖∇scale‖ + ε)`: `value` in units of `scale`'s own gradient,
/// which the calculus resolves through every enclosing coordinate warp, so
/// a distance built from it is in pixels of whatever lattice the kernel is
/// baked on.
fn in_pixels(value: &Kernel, scale: &Kernel) -> Kernel {
    let gradient = scale.dx().hypot(&scale.dy());
    value.div(&gradient.add(&constant(MIN_GRADIENT)))
}

/// The signed distance from the sample to the chord's line, in the kernel's
/// own units, positive on the left of `a → b`.
fn across(chord: Chord) -> Kernel {
    let len = chord.length();
    let u = [
        (chord.b[0] - chord.a[0]) / len,
        (chord.b[1] - chord.a[1]) / len,
    ];
    Linear::projection(chord.a, [-u[1], u[0]]).kernel()
}

/// The distance from the sample to the chord *segment*, in pixels:
/// perpendicular within the segment's span, distance to the nearer endpoint
/// beyond it.
///
/// The capsule, not the distance to the segment's infinite line, and that
/// is load-bearing rather than tidy: a sample outside a convex corner is
/// far from both segments but close to the *line* of each, so a minimum
/// over line distances would soften a pixel a corner's width away — a dark
/// halo on every serif.
fn segment_distance(chord: Chord, across: &Kernel) -> Kernel {
    let len = chord.length();
    let u = [
        (chord.b[0] - chord.a[0]) / len,
        (chord.b[1] - chord.a[1]) / len,
    ];
    let along = Linear::projection(chord.a, u).kernel();
    let half = constant_of(len / 2.0);
    let overshoot = along.sub(&half).abs().sub(&half).max(&constant(0.0));
    let dn = in_pixels(across, across);
    let tn = in_pixels(&overshoot, &along);
    dn.mul(&dn).add(&tn.mul(&tn)).sqrt()
}

/// **The crossing term**: `±1` where a ray from the sample to `+X` crosses
/// this chord, `0` where it does not.
///
/// The schoolbook inside test, and the whole of the winding for a straight
/// edge: does the chord span this row, and is it to the right? The row test
/// is half-open (`y_min ≤ Y < y_max`) so a vertex shared by two chords is
/// counted once, and the sign is the direction the chord runs in.
///
/// Everything here but the final comparison is a function of `Y` alone —
/// one multiply-add per chord per row — so it hoists into the row prologue
/// and leaves one comparison per pixel. That is the *good* kind of Y-only
/// work; what made the scanline formulation slow was hoisting a **root
/// solve** per segment per row, and there is none here because a chord is a
/// line. A curve's own contribution is not a crossing at all — see
/// [`sliver_term`].
///
/// Reads columns 0–5 of `c` (`y_min, y_max, dx/dy, a.x, a.y, direction`) —
/// see the piece-row layout above [`piece_row`]. Whether this term is even
/// asked for is a decision its caller makes ([`glyph`] asks unconditionally
/// via a bound table; [`prepare`] prunes by [`Included::crossing`] and a
/// horizontal chord).
fn crossing_term(c: Coeff) -> Kernel {
    let spans = Kernel::y()
        .ge(&c(COL_Y_MIN))
        .and(&Kernel::y().lt(&c(COL_Y_MAX)));
    // x = a.x + (Y − a.y)·dx/dy, exact and linear: a chord is a line, so
    // there is no discriminant and no root to be on the wrong side of.
    let crossing_x = Kernel::y()
        .sub(&c(COL_AY))
        .mul(&c(COL_DX_OVER_DY))
        .add(&c(COL_AX));
    spans
        .and(&crossing_x.gt(&Kernel::x()))
        .select(&c(COL_DIRECTION), &constant(0.0))
}

/// **The sliver term**: `±1` inside the crescent between a curve and its
/// chord, `0` outside.
///
/// Replacing a curve by its chord changes the winding by exactly the closed
/// loop *chord → curve*, whose winding is `±1` on the crescent between them
/// and `0` elsewhere, with the sign of the control triangle's orientation.
/// So this is the only place a curve differs from a straight edge, and the
/// only place Loop and Blinn's construction is used.
///
/// The crescent is `{v ≥ u²} ∩ {v ≤ u}` under the affine map taking
/// `P0, P1, P2` to `(0,0), (½,0), (1,1)` — the change of coordinates that
/// lands *every* quadratic on the one parabola `v = u²`, turning "solve for
/// where the curve is" into "evaluate a formula and read its sign".
///
/// ## The fence is one half-plane, not a triangle
///
/// `u² − v` is *wrong* outside the arc, not merely slow, so the test has to
/// be fenced — but the control triangle is not the cheapest fence, because
/// two of its three edges are implied by the parabola itself. In canonical
/// coordinates the triangle is `v ≥ 0`, `v ≤ u` (the chord), and
/// `1 + v − 2u ≥ 0`; given `v ≥ u²`,
///
/// - `v ≥ u² ≥ 0` is the first edge, and
/// - `1 + v − 2u ≥ 1 + u² − 2u = (1 − u)² ≥ 0` is the third.
///
/// So only the chord's own half-plane is left, and `u² ≤ v ≤ u` is already
/// bounded on its own: it forces `u ∈ [0, 1]`, since `u² ≤ u` holds nowhere
/// else. An axis-aligned box would have been strictly worse — it is not
/// equivalent to the triangle, so it would be needed *in addition to* the
/// chord test rather than instead of it.
///
/// Affine maps preserve half-planes and intersections, so this is the
/// crescent in screen space too: two comparisons and one `and`.
///
/// Reads columns 6–12 of `c` (`u.a, u.b, u.c, v.a, v.b, v.c, sign` — see
/// [`piece_row`]/[`bulge_uv`]); the `(u, v)` affine map itself is computed
/// once on the host, not here. A line's row makes every one of those
/// columns `0.0`, which makes this term the identity unconditionally (see
/// the module note above [`COL_Y_MIN`]) — so unlike [`crossing_term`],
/// [`glyph`]'s table-read caller does not need to special-case a line, and
/// [`prepare`] still prunes it by [`Included::bulge`] as an arena-size
/// optimization only, never for correctness.
fn sliver_term(c: Coeff) -> Kernel {
    let u = Kernel::x()
        .mul(&c(COL_U_A))
        .add(&Kernel::y().mul(&c(COL_U_B)))
        .add(&c(COL_U_C));
    let v = Kernel::x()
        .mul(&c(COL_V_A))
        .add(&Kernel::y().mul(&c(COL_V_B)))
        .add(&c(COL_V_C));
    let under_the_chord = u.ge(&v);
    let over_the_parabola = u.mul(&u).sub(&v).le(&constant(0.0));
    under_the_chord
        .and(&over_the_parabola)
        .select(&c(COL_SLIVER_SIGN), &constant(0.0))
}

/// One included piece's contribution, computed once — from its own row,
/// via a per-piece constant [`Coeff`] — so the winding accumulation and the
/// boundary/ramp test below cannot drift apart: both read this same `own`.
struct Prepared {
    piece: Piece,
    /// This piece's own crossing+sliver term, pruned to what `included`
    /// says this call site needs ([`Included::crossing`],
    /// [`Included::bulge`]).
    own: Kernel,
    /// The ramp distance, when the piece is near enough to soften a sample
    /// ([`Included::ramp`]); `None` otherwise, so [`coverage`] skips it.
    d: Option<Kernel>,
}

/// Build each included piece's [`Prepared`] contribution: the pruned
/// crossing+sliver `own` term every caller needs (summed into the winding
/// by [`cells`], read at the boundary test by [`coverage`]), and — only
/// where [`Included::ramp`] asks for one — the antialiasing distance.
///
/// Reads every column through a per-piece constant [`Coeff`]
/// (`Kernel::constant(row[k])`): the row is a concrete piece here, unlike
/// [`glyph`]'s table read at the reduce binder. Both call [`crossing_term`]
/// and [`sliver_term`], so the term itself cannot drift between the two
/// coefficient sources.
fn prepare(included: &[Included]) -> Vec<Prepared> {
    included
        .iter()
        .map(|inc| {
            let piece = inc.piece;
            let row = piece_row(piece);
            let coeff = |k: usize| Kernel::constant(row[k]);
            let mut own_terms = Vec::new();
            if inc.crossing && !piece.chord.is_horizontal() {
                own_terms.push(crossing_term(&coeff));
            }
            if inc.bulge.is_some() {
                own_terms.push(sliver_term(&coeff));
            }
            let own = Kernel::sum(&own_terms);
            // The distance to the chord, less how far the segment strays
            // from it, is a lower bound on the distance to the segment —
            // exact for a line, where nothing strays. For a curve the
            // implicit is the better estimate near the arc and the bound is
            // the sound one away from it, so take whichever is larger: a
            // bound can never exceed the truth, so the larger is the
            // closer.
            let d = inc.ramp.then(|| {
                let across = across(piece.chord);
                let chord = segment_distance(piece.chord, &across);
                let bound = chord.sub(&constant_of(piece.deviation()));
                match piece.bulge {
                    Some(bulge) => implicit_distance(bulge).max(&bound),
                    None => bound,
                }
            });
            Prepared { piece, own, d }
        })
        .collect()
}

/// Coverage, given the winding: an exact integer decides inside from
/// outside, and a distance decides how much of the pixel.
///
/// `winding` is already resolved by the caller — [`glyph`]'s
/// `Kernel::sum_over` a bound table, or [`cells`]'s `Kernel::sum` of a host
/// constant plus each [`Prepared::own`] — because how it is built is the
/// one thing S1a changes; everything below (the boundary test, the ramp,
/// the clamp) is unconverted and reads it as a plain `Kernel` either way.
///
/// The two halves are separate on purpose. An earlier draft made each
/// winding term *soft*, so a crossing's existence and its coverage were one
/// number, and a comparison landing on the wrong side of an edge moved
/// coverage by half a unit rather than by a rounding. That coupling is what
/// made `'8'` render with a smear at its waist.
fn coverage(winding: Kernel, prepared: &[Prepared]) -> Kernel {
    // **Only a boundary softens a pixel.** A font may draw an edge that is
    // not on the outside of anything — a stroke built from two overlapping
    // contours, or two components of a compound glyph that overlap — and
    // ink continues across such an edge rather than stopping at it. Taking
    // the nearest *drawn* edge would dim a line straight through solid ink
    // (measured: two overlapping squares read 0.5 along their shared
    // interior edges, a half-dark seam one pixel wide).
    //
    // Whether a piece is a boundary is a local question the winding already
    // answers. Let `w₋` be the winding with this piece's own contribution
    // removed; the two sides of the piece are `w₋` and `w₋ + dir`, which
    // differ by one and so are never both zero. The piece separates ink
    // from no-ink exactly when one of them is zero.
    //
    // Only the pieces that other contours' ink actually covers pay for it.
    // The test references the whole winding sum, and every `Kernel`
    // combinator copies its arena, so asking it of every piece makes a
    // long string's kernel quadratic — measured at 6 GB and climbing before
    // this early-out existed.
    let mut distances = vec![constant(RAMP_REACH)];
    for p in prepared {
        let Some(d) = &p.d else { continue };
        if !p.piece.may_be_interior {
            // Nothing else covers it, so it bounds the ink it draws.
            distances.push(d.clone());
            continue;
        }
        let without = winding.sub(&p.own);
        let other_side = without.add(&constant_of(p.piece.chord.direction()));
        let separates = without.abs().min(&other_side.abs()).lt(&constant(0.5));
        distances.push(separates.select(d, &constant(RAMP_REACH)));
    }
    let distance = min_of(distances);
    let inside = winding.abs().ge(&constant(1.0));
    // A distance is not negative, and this is the one place that can be
    // told: the chord bound is the chord's own distance less the curve's
    // deviation, which goes below zero for a sample on the chord of a
    // curve. Unclamped it makes the outside arm `½ − d` exceed one — a
    // coverage of 1.17, which is not a rounding difference but a value
    // outside the function's range.
    let distance = distance.max(&constant(0.0));
    let half = constant(0.5);
    inside.select(
        &half.add(&distance).min(&constant(1.0)),
        &half.sub(&distance).max(&constant(0.0)),
    )
}

/// `|f|/‖∇f‖`, the implicit's own first-order distance to the parabola —
/// what a GPU Loop–Blinn shader antialiases with. Accurate near the curve
/// and **not sound away from it**: it underestimates where the curve is
/// sharp (measured: a texel a full pixel outside `O` at 7 px read as an
/// edge texel), and an underestimate past the ramp is not a rounding
/// difference but a saturation failure, which is also what a cell's pruning
/// relies on. [`coverage`] therefore takes it against a bound built from
/// the chord.
fn implicit_distance(bulge: Bulge) -> Kernel {
    let Bulge { p0, p1, p2, .. } = bulge;
    let e1 = [p1[0] - p0[0], p1[1] - p0[1]];
    let e2 = [p2[0] - p0[0], p2[1] - p0[1]];
    let det = e1[0] * e2[1] - e1[1] * e2[0];
    let lambda1 = Linear {
        a: e2[1] / det,
        b: -e2[0] / det,
        c: (p0[1] * e2[0] - p0[0] * e2[1]) / det,
    };
    let lambda2 = Linear {
        a: -e1[1] / det,
        b: e1[0] / det,
        c: (e1[1] * p0[0] - e1[0] * p0[1]) / det,
    };
    let u = Linear {
        a: lambda1.a / 2.0 + lambda2.a,
        b: lambda1.b / 2.0 + lambda2.b,
        c: lambda1.c / 2.0 + lambda2.c,
    }
    .kernel();
    let v = lambda2.kernel();
    let f = u.mul(&u).sub(&v);
    in_pixels(&f, &f).abs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_segment_meets_a_rectangle_it_crosses_and_misses_one_it_skirts() {
        let r = Rect {
            x0: 0.0,
            y0: 0.0,
            x1: 4.0,
            y1: 4.0,
        };
        assert!(r.meets_segment([-1.0, 2.0], [5.0, 2.0]));
        assert!(r.meets_segment([1.0, 1.0], [2.0, 2.0]));
        assert!(r.meets_segment([-1.0, -1.0], [5.0, 5.0]));
        assert!(!r.meets_segment([-1.0, 5.0], [5.0, 5.0]));
        assert!(!r.meets_segment([5.0, -1.0], [5.0, 5.0]));
        // Touching a corner counts: the rectangle is closed.
        assert!(r.meets_segment([-1.0, 5.0], [5.0, -1.0]));
        // A vertical segment beside the rectangle, and one through it.
        assert!(!r.meets_segment([4.5, -1.0], [4.5, 5.0]));
        assert!(r.meets_segment([4.0, -1.0], [4.0, 5.0]));
    }

    #[test]
    fn a_triangle_meets_a_rectangle_it_contains() {
        let r = Rect {
            x0: 1.0,
            y0: 1.0,
            x1: 2.0,
            y1: 2.0,
        };
        assert!(r.meets_triangle([-10.0, -10.0], [10.0, -10.0], [0.0, 10.0]));
        assert!(!r.meets_triangle([5.0, 5.0], [6.0, 5.0], [5.0, 6.0]));
        assert!(r.meets_triangle([0.0, 0.0], [3.0, 0.0], [0.0, 3.0]));
    }

    #[test]
    fn a_flat_quadratic_is_its_chord() {
        let mut o = Outline::default();
        o.contours.push(super::super::outline::Contour {
            segments: vec![Segment::Quad {
                from: [0.0, 0.0],
                control: [5.0, 0.001], // deviation 5e-4, under FLAT_ENOUGH
                to: [10.0, 0.0],
            }],
        });
        let pieces = Pieces::of(&o);
        assert_eq!(pieces.pieces.len(), 1);
        assert!(pieces.pieces[0].bulge.is_none());
    }

    /// A curve too fat for the ramp's chord bound is split until it is not,
    /// and every piece is a real quadratic — the winding never needs a
    /// split, so a split must not change it.
    #[test]
    fn a_fat_quadratic_is_split_until_its_chord_bounds_it() {
        let mut o = Outline::default();
        o.contours.push(super::super::outline::Contour {
            segments: vec![Segment::Quad {
                from: [0.0, 0.0],
                control: [10.0, 40.0],
                to: [20.0, 0.0],
            }],
        });
        let pieces = Pieces::of(&o);
        assert!(pieces.pieces.len() > 1, "a 20-px bulge was not split");
        for p in &pieces.pieces {
            assert!(
                p.deviation() <= MAX_DEVIATION,
                "a piece still strays {} from its chord",
                p.deviation()
            );
            assert!(p.bulge.is_some(), "a split piece lost its curve");
        }
    }

    /// A curve whose control point projects *outside* its chord's span is
    /// still measured against the segment, not the chord's infinite line.
    ///
    /// This is [`Piece::deviation`]'s own documented counterexample, and
    /// [`Piece::quad`] used to miss it in both directions: measuring
    /// `|area2| / length / 2` the hook below reads as 0.0025 — under
    /// [`FLAT_ENOUGH`] — so the whole curve was discarded as a straight
    /// line while it strays about 50 units from that line. Discarding a
    /// sliver changes the **winding**, which nothing is allowed to
    /// approximate, so this is a stronger requirement than the ramp's.
    ///
    /// Scaled down by 1e-3 from the doc's `(0, 0) → (−100, 0.1) → (1, 0)`
    /// only to put the deviation under `FLAT_ENOUGH` on the line measure —
    /// the failure is scale-free, and this is the half of it that is not
    /// merely a loose bound.
    #[test]
    fn a_hook_is_not_flattened_by_measuring_the_wrong_distance() {
        let mut o = Outline::default();
        o.contours.push(super::super::outline::Contour {
            segments: vec![Segment::Quad {
                from: [0.0, 0.0],
                control: [-100.0, 0.005],
                to: [1.0, 0.0],
            }],
        });
        let pieces = Pieces::of(&o);
        // Not "every piece is a curve": once split, a sub-arc of a hook
        // really can be straight to within `FLAT_ENOUGH`, and emitting it
        // as a line is right. What must survive is the *shape* — the
        // pieces have to go where the curve goes, so their endpoints reach
        // far from the chord the whole thing used to collapse onto.
        let chord = Chord {
            a: [0.0, 0.0],
            b: [1.0, 0.0],
        };
        let reach = pieces
            .pieces
            .iter()
            .map(|p| point_to_segment(p.chord.a, chord).max(point_to_segment(p.chord.b, chord)))
            .fold(0.0f64, f64::max);
        assert!(
            reach > 10.0,
            "a hook straying ~50 units collapsed onto its chord: reach {reach}"
        );
        for p in &pieces.pieces {
            assert!(
                p.deviation() <= MAX_DEVIATION,
                "a piece still strays {} from its chord",
                p.deviation()
            );
        }
    }
}
