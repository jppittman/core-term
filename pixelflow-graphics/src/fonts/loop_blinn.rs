//! Coverage of an outline as a [`Kernel`]: the winding number, computed
//! relative to a reference point, with Loop–Blinn's implicit test for the
//! part of each quadratic that bulges past its chord.
//!
//! ## The denotation
//!
//! Coverage is `min(|w|, 1)` for the winding number `w` of the outline —
//! the non-zero rule TrueType specifies. For any reference point `O` off
//! every chord line, `w` decomposes exactly into per-segment terms:
//!
//! ```text
//! w(P) = w_poly(O) − Σᵢ σ(O, Aᵢ, Bᵢ)·[P ∈ shadowᵢ(O)] + Σⱼ σ(P0ⱼ, P1ⱼ, P2ⱼ)·[P ∈ Sⱼ]
//! ```
//!
//! - `i` ranges over the **chords**: every line, and every quadratic
//!   `P0 → P1 → P2` replaced by its straight chord `P0 → P2`. Together the
//!   chords form the *chord polygon*, and `w_poly(O)` is its winding number
//!   at `O`, a host-side constant. `σ(O, A, B)` is the sign of the triangle
//!   `(O, A, B)`'s signed area, and `shadowᵢ(O)` is the region the chord
//!   shadows as seen from `O`: the cone spanned by the rays `O→A`, `O→B`,
//!   cut to the far side of the chord's line. `P ∈ shadowᵢ(O)` is exactly
//!   "the segment `O→P` crosses chord `i`".
//! - `j` ranges over the quadratics that actually curve. Loop and Blinn
//!   assign `(u, v) = (0,0), (½,0), (1,1)` to `P0, P1, P2` and interpolate
//!   affinely; inside the control triangle `T` the curve is exactly the zero
//!   set of `f = u² − v`, and `Sⱼ = T ∩ {f ≤ 0}` is the sliver between the
//!   chord and the curve. Replacing a curve by its chord plus the closed
//!   loop *chord → curve* changes the winding by the loop's own, which is
//!   `±1` on `Sⱼ` with the sign of the control triangle's orientation.
//!
//! Every term is affine tests of `(X, Y)` plus one quadratic form. Nothing
//! is a function of `Y` alone, so nothing hoists into a per-row prologue
//! and runs there unconditionally for every segment — which is where the
//! scanline formulation spent 33 ns/px on a 40-px lattice. And no term
//! intersects a ray with a curve, so there is no discriminant and no
//! `disc ≥ 0` knife edge at a tangency: the `'8'` waist defect that
//! `tests/freetype_oracle.rs` pinned is not expressible here.
//!
//! ## The scanline was this with the reference point at infinity
//!
//! Send `O` to `(−∞, Y)`. The cone from `O` through a chord becomes the
//! horizontal band `y_min ≤ Y < y_max`, the far side of the chord becomes
//! `X > x_int(Y)`, and the formula above becomes the leftward ray-crossing
//! count `ttf_curve_analytical.rs` used to compute — with one difference:
//! for a quadratic the ray had to be intersected with the curve itself,
//! which is a quadratic in `Y`, whose two roots are functions of `Y` alone,
//! and whose existence is decided by a discriminant. Moving `O` to a finite
//! point makes every test oblique, so it is per-pixel, and the curve is
//! never intersected at all: its chord is crossed like any line, and the
//! sliver it bulges past the chord is a region test, `f ≤ 0`.
//!
//! ## Antialiasing: the winding is exact, the ramp is a distance
//!
//! Every term above is a **hard** indicator — a mask selecting a signed
//! constant — so the sum is the exact winding number at every sample, an
//! integer. What is soft is one number: the distance from the sample to the
//! nearest piece of the outline. Coverage is then
//!
//! ```text
//! coverage = inside ? min(1, ½ + d) : max(0, ½ − d)
//! ```
//!
//! which is the exact area of a pixel cut by a straight edge `d` away, and
//! within a pixel of a vertex is the corner approximation every
//! one-ramp-per-edge rasterizer makes. A horizontal edge is antialiased like
//! any other — the scanline formulation could not, having dropped horizontal
//! segments as contributing no crossing, which is why FreeType found ink at
//! texels this rasterizer left blank.
//!
//! Splitting them is not a detail. An earlier draft softened each *winding*
//! term with its own ramp, which made antialiasing depend on the reference
//! point: a chord's ramp lived inside its cone, and a reference point near
//! that chord's line makes the cone thinner than the ramp, so a sample half
//! a pixel from the edge could fall outside the cone and take the hard
//! value. Here no ramp is inside a cone, so no reference point can bias one.
//!
//! `d` is measured in **pixels of the lattice being collapsed**, not of the
//! frame the outline lives in: every distance is divided by the gradient
//! magnitude of its own edge function, `DX`/`DY` resolved symbolically at
//! bake, so the chain rule carries the scale through every enclosing
//! `Kernel::at`. For an affine edge function that whole normalisation is a
//! compile-time constant, which is what `tests/kernel_glyph_optimize.rs`
//! pins.
//!
//! Per piece, `d` is the larger of two estimates, because each is unusable
//! alone:
//!
//! - The **capsule distance to the chord** — perpendicular within the
//!   segment's span, distance to the nearer endpoint beyond it. Exact for a
//!   line. For a curve it is a *lower* bound once the curve's deviation from
//!   its chord is subtracted, and only a bound: too pessimistic to use by
//!   itself. (The capsule, not the distance to the chord's infinite line:
//!   a sample outside a convex corner is far from both segments but close to
//!   each one's line, so a minimum over line distances would soften a pixel
//!   a corner away — a dark halo on every serif.)
//! - The **implicit's own distance** `|f|/‖∇f‖`, what a GPU Loop–Blinn
//!   shader antialiases with. Accurate near the arc and *not sound* away
//!   from it: it underestimates where the curve is sharp — measured, a texel
//!   a full pixel outside `O` at 7 px read as an edge texel — and an
//!   underestimate past the ramp is not a rounding difference, it is a
//!   saturation failure, which is also what a cell's pruning relies on.
//!
//! A lower bound can never exceed the truth, so the larger of the two is the
//! closer of the two. That leaves one requirement on the geometry: the bound
//! has to be tight enough to rescue the implicit, which is why a segment
//! straying more than [`MAX_DEVIATION`] from its chord is halved (exactly,
//! de Casteljau) until it does not. The winding never needs a split.
//!
//! **Half-open cones.** A sample exactly on a ray through a vertex must be
//! counted by exactly one of the two chords meeting there, and by neither
//! where the outline turns back and both cones lie on the same side. Each
//! cone includes its clockwise boundary ray and excludes its
//! counter-clockwise one, *decided by the chord's own sign* — the rule that
//! always includes the first ray is correct at a pass-through vertex and
//! wrong at every turn-back one. Both chords compute the ray's edge function
//! from one shared node, so the two decisions cannot disagree by a rounding.
//!
//! ## The domain-side form
//!
//! For a rectangle `C` of samples with `O ∈ C`: `O→P` lies in `C` for every
//! `P ∈ C`, so a chord that does not meet `C` shadows nothing in it, and a
//! sliver whose control triangle does not meet `C` is empty there. [`cells`]
//! cuts a lattice into rectangles, keeps for each only the chords and
//! control triangles meeting the rectangle dilated by the ramp's reach, and
//! places the result on an [`IndexRange`] of a `Union` — the plan's
//! "domain-side triangle extent": outside its rectangle a cell's kernel is
//! never asked, and the Loop–Blinn implicit, which is meaningless outside
//! its triangle, is only ever evaluated where a triangle can reach.
//! Rectangles nothing reaches are the constant `min(|w_poly(O)|, 1)`.
//!
//! ## What is exactly zero, and where
//!
//! Beyond half a pixel from every segment the ramp is saturated and the
//! winding is exact, so coverage is exactly 0 or exactly 1 — and outside
//! the outline it is 0. [`glyph`] cuts its kernel to the outline's
//! bounding box dilated by [`RAMP_REACH`] and reports that box as its
//! [`Support`]: exact, whatever the glyph's shape — the scanline's support
//! could only ever be the bounding box of its *mask*, because its ramps
//! trailed tens of pixels past the ink.

use std::collections::HashMap;

use super::outline::{Outline, Point, Segment};
use super::PIXEL_CENTER;
use pixelflow_core::{IndexRange, Kernel, Lattice};

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

/// A reference point must sit at least this far from the line of every
/// chord it casts a shadow from: on the line, the cone is degenerate and
/// the chord's sign is meaningless.
const REFERENCE_CLEARANCE: f64 = 1e-3;

/// Where a reference point is tried, in order, relative to the centre it
/// was asked for: the centre itself, then nudges no two of which are
/// collinear with it, so that no single chord line can spoil more than two
/// of them. Every nudge is well under half a pixel, which keeps the point
/// inside the dilated rectangle [`cells`] prunes against.
const NUDGES: [[f64; 2]; 8] = [
    [0.0, 0.0],
    [0.137, 0.061],
    [-0.083, 0.151],
    [0.191, -0.109],
    [-0.163, -0.127],
    [0.059, 0.197],
    [-0.203, 0.043],
    [0.113, -0.181],
];

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
        };
    };
    let centre = [
        f64::from(bounds[0] + bounds[2]) / 2.0,
        f64::from(bounds[1] + bounds[3]) / 2.0,
    ];
    let all_chords: Vec<Chord> = pieces.chords().collect();
    let reference = Reference::clear_of(centre, &all_chords, &pieces);
    let included: Vec<Included> = pieces
        .pieces
        .iter()
        .map(|p| Included {
            piece: *p,
            chord_near: true,
            bulge: p.bulge,
            ramp_near: true,
        })
        .collect();
    let coverage = coverage(&reference, &included);

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
/// Each summand's kernel holds only the chords whose segments, and the
/// quadratics whose control triangles, meet its rectangle dilated by
/// [`RAMP_REACH`]; rectangles nothing reaches are the constant coverage of
/// their interior, and those where that constant is 0 are not placed at all.
/// Place the result with `Union::place`; the ranges are disjoint by
/// construction.
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
            let included: Vec<Included> = pieces
                .pieces
                .iter()
                .filter_map(|p| {
                    let chord_near = reach.meets_segment(p.chord.a, p.chord.b);
                    // The ramp's lower bound is the chord's distance less
                    // the curve's deviation, so a curve softens a sample
                    // only within the reach *plus* that deviation.
                    let ramp_near = samples
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
                        .filter(|b| ramp_near || reach.meets_triangle(b.p0, b.p1, b.p2));
                    (chord_near || bulge.is_some() || ramp_near).then_some(Included {
                        piece: *p,
                        chord_near,
                        bulge,
                        ramp_near,
                    })
                })
                .collect();
            let range = IndexRange::new(x0, y0, w, h);
            if included.is_empty() {
                // Nothing reaches this rectangle: a constant, which is 0
                // (not placed) or 1 (the interior). The centre is clear of
                // every chord by more than the reach, so its winding is
                // well defined without a nudge.
                if winding_of_chord_polygon(samples.centre(), pieces.chords()) != 0 {
                    out.push((range, constant(1.0)));
                }
                continue;
            }
            let shadowing: Vec<Chord> = included.iter().map(|i| i.piece.chord).collect();
            let reference = Reference::clear_of(samples.centre(), &shadowing, &pieces);
            out.push((range, coverage(&reference, &included)));
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

    /// Distance from `p` to the chord's infinite line.
    fn line_distance(self, p: P) -> f64 {
        cross(self.a, self.b, p).abs() / self.length()
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
}

impl Piece {
    /// A straight piece, or nothing when its ends coincide: a zero-length
    /// chord crosses nothing.
    fn line(a: P, b: P) -> Option<Self> {
        let chord = Chord { a, b };
        (chord.length() >= ZERO_LENGTH).then_some(Self { chord, bulge: None })
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
        let deviation = area2.abs() / chord.length() / 2.0;
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
        });
    }

    /// How far the segment strays from its chord: 0 for a line, the
    /// curve's maximum deviation for a quadratic. It is what separates the
    /// chord from the segment everywhere — the distance to the segment is
    /// within this of the distance to the chord — so it is the slack the
    /// ramp's lower bound and the cell's pruning both carry.
    fn deviation(self) -> f64 {
        self.bulge.map_or(0.0, |b| {
            cross(b.p0, b.p1, b.p2).abs() / self.chord.length() / 2.0
        })
    }
}

/// An outline prepared for the kernel.
struct Pieces {
    pieces: Vec<Piece>,
}

impl Pieces {
    fn of(outline: &Outline) -> Self {
        let mut pieces = Vec::new();
        for segment in outline.segments() {
            match segment {
                Segment::Line { from, to } => {
                    pieces.extend(Piece::line(wide(from), wide(to)));
                }
                Segment::Quad { from, control, to } => {
                    Piece::quad(wide(from), wide(control), wide(to), MAX_SPLITS, &mut pieces);
                }
            }
        }
        Self { pieces }
    }

    fn is_empty(&self) -> bool {
        self.pieces.is_empty()
    }

    fn chords(&self) -> impl Iterator<Item = Chord> + '_ {
        self.pieces.iter().map(|p| p.chord)
    }
}

/// A piece as one cell sees it: which of its terms the cell needs.
#[derive(Clone, Copy, Debug)]
struct Included {
    piece: Piece,
    /// Whether the chord's shadow can reach the cell.
    chord_near: bool,
    /// The sliver, if it can reach the cell.
    bulge: Option<Bulge>,
    /// Whether the segment is near enough to soften a sample in the cell.
    /// Its ramp is skipped when not: past the reach every ramp has
    /// saturated, and the cell's samples are already on the right side of
    /// the outline by the winding alone.
    ramp_near: bool,
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

/// The winding number of the chord polygon at `o`: signed crossings of the
/// ray `+x` from `o`, with the half-open `y_min ≤ y < y_max` rule so a
/// vertex on the ray is counted once. `o` must not lie on a chord.
fn winding_of_chord_polygon(o: P, chords: impl Iterator<Item = Chord>) -> i32 {
    let mut winding = 0;
    for Chord { a, b } in chords {
        if (a[1] <= o[1]) == (b[1] <= o[1]) {
            continue;
        }
        let t = (o[1] - a[1]) / (b[1] - a[1]);
        let x = a[0] + t * (b[0] - a[0]);
        if x > o[0] {
            winding += if b[1] > a[1] { 1 } else { -1 };
        }
    }
    winding
}

/// A reference point and the chord polygon's winding number there.
struct Reference {
    o: P,
    winding: i32,
}

impl Reference {
    /// The first of [`NUDGES`] from `centre` that keeps [`REFERENCE_CLEARANCE`]
    /// from the line of every chord in `shadowing` — the chords whose cones
    /// this point will be the apex of.
    ///
    /// # Panics
    ///
    /// Panics if every nudge lands on some chord line: eight distinct lines
    /// through eight chosen points is not a shape a font produces, and a
    /// silent fallback here would be a kernel with a meaningless sign.
    fn clear_of(centre: P, shadowing: &[Chord], all: &Pieces) -> Self {
        for [dx, dy] in NUDGES {
            let o = [centre[0] + dx, centre[1] + dy];
            if shadowing
                .iter()
                .all(|c| c.line_distance(o) >= REFERENCE_CLEARANCE)
            {
                return Self {
                    o,
                    winding: winding_of_chord_polygon(o, all.chords()),
                };
            }
        }
        panic!("loop_blinn: every candidate reference point near {centre:?} lies on a chord line");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// The kernel: winding terms from f64 geometry
// ═══════════════════════════════════════════════════════════════════════════

fn constant(v: f32) -> Kernel {
    Kernel::constant(v)
}

fn constant_of(v: f64) -> Kernel {
    Kernel::constant(v as f32)
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

    /// `cross(o, v, P)` as a function of `P`: the edge function of the ray
    /// from `o` through `v`.
    fn spoke(o: P, v: P) -> Self {
        let (ux, uy) = (v[0] - o[0], v[1] - o[1]);
        Self {
            a: -uy,
            b: ux,
            c: uy * o[0] - ux * o[1],
        }
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

/// The edge functions of the rays from one reference point, one node per
/// vertex: two chords meeting at a vertex test the *same* node, which is
/// what lets their half-open cones partition directions exactly.
struct Spokes {
    o: P,
    by_vertex: HashMap<[u64; 2], Kernel>,
}

impl Spokes {
    fn from(o: P) -> Self {
        Self {
            o,
            by_vertex: HashMap::new(),
        }
    }

    fn through(&mut self, v: P) -> Kernel {
        self.by_vertex
            .entry([v[0].to_bits(), v[1].to_bits()])
            .or_insert_with(|| Linear::spoke(self.o, v).kernel())
            .clone()
    }
}

/// `value / (‖∇scale‖ + ε)`: `value` in units of `scale`'s own gradient,
/// which the calculus resolves through every enclosing coordinate warp, so
/// a distance built from it is in pixels of whatever lattice the kernel is
/// baked on.
fn in_pixels(value: &Kernel, scale: &Kernel) -> Kernel {
    let gradient = scale.dx().hypot(&scale.dy());
    value.div(&gradient.add(&constant(MIN_GRADIENT)))
}

/// The signed distance from the sample to the chord's line, positive on the
/// far side from `o`, in the kernel's own units.
fn across(o: P, chord: Chord) -> Kernel {
    let len = chord.length();
    let u = [
        (chord.b[0] - chord.a[0]) / len,
        (chord.b[1] - chord.a[1]) / len,
    ];
    let mut n = [-u[1], u[0]];
    if n[0] * (o[0] - chord.a[0]) + n[1] * (o[1] - chord.a[1]) > 0.0 {
        n = [-n[0], -n[1]];
    }
    Linear::projection(chord.a, n).kernel()
}

/// The distance from the sample to a segment, in pixels: perpendicular
/// within the segment's span, distance to the nearer endpoint beyond it.
///
/// The capsule, not the distance to the segment's infinite line, and that
/// is load-bearing rather than tidy: a sample outside a convex corner is
/// far from both segments but close to the *line* of each, so a min over
/// line distances would soften a pixel a corner's width away — a dark halo
/// on every serif.
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

/// The half-open cone of the chord as seen from the spokes' reference
/// point — clockwise ray included, counter-clockwise excluded — and the
/// chord's sign there.
fn cone(spokes: &mut Spokes, chord: Chord) -> (Kernel, f64) {
    let zero = constant(0.0);
    let sigma = cross(spokes.o, chord.a, chord.b);
    let (ha, hb) = (spokes.through(chord.a), spokes.through(chord.b));
    let mask = if sigma > 0.0 {
        ha.ge(&zero).and(&hb.lt(&zero))
    } else {
        ha.lt(&zero).and(&hb.ge(&zero))
    };
    (mask, sigma)
}

/// The chord's shadow term, `−σ(o, a, b)·[P ∈ shadow]`: inside the cone and
/// strictly past the chord's line.
fn shadow_term(cone: &Kernel, sigma: f64, across: &Kernel) -> Kernel {
    let zero = constant(0.0);
    let shadow = cone.and(&across.gt(&zero));
    shadow.select(&constant_of(-sigma.signum()), &zero)
}

/// The sliver term `sign·[P ∈ S]`, hard.
///
/// `S = T ∩ {f ≤ 0}`: the two outer edges of the control triangle, the
/// curve `f ≤ 0`, and the chord edge — the latter written with the same
/// `across` the shadow term tests, with the complementary half-open rule
/// when the sliver is on the reference point's side, so that a sample
/// exactly on the chord line lands on one side of it in both terms at once.
///
/// Returns the winding term and `|f|/‖∇f‖`, the implicit's own first-order
/// distance to the parabola — what a GPU Loop–Blinn shader antialiases
/// with. Accurate near the curve and **not sound far from it**: it
/// underestimates where the curve is sharp (measured: a texel a full pixel
/// outside `O` at 7 px read as an edge texel), and an underestimate past
/// the ramp is not a rounding difference but a saturation failure, which is
/// also what a cell's pruning relies on. [`coverage`] therefore takes it
/// against a sound lower bound built from the chord.
fn sliver(o: P, bulge: Bulge, across: &Kernel) -> (Kernel, Kernel) {
    let zero = constant(0.0);
    let Bulge { p0, p1, p2, sign } = bulge;
    let e1 = [p1[0] - p0[0], p1[1] - p0[1]];
    let e2 = [p2[0] - p0[0], p2[1] - p0[1]];
    let det = e1[0] * e2[1] - e1[1] * e2[0];
    // Barycentric coordinates as affine functions of the sample: λ₁ is the
    // weight of the control point, λ₂ of the end point.
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
    // Loop–Blinn: (u, v) = (0,0), (½,0), (1,1) at p0, p1, p2.
    let u = Linear {
        a: lambda1.a / 2.0 + lambda2.a,
        b: lambda1.b / 2.0 + lambda2.b,
        c: lambda1.c / 2.0 + lambda2.c,
    }
    .kernel();
    let v = lambda2.kernel();
    let f = u.mul(&u).sub(&v);
    // λ₂ > 0 and λ₀ = 1 − 2u + v > 0: the two edges the sliver never crosses.
    let outer_edges = v
        .gt(&zero)
        .and(&constant(1.0).add(&v).sub(&u.mul(&constant(2.0))).gt(&zero));
    // The sliver lies on the control point's side of the chord; `across`
    // is positive on the far side from `o`.
    let opposite = cross(p0, p2, o) * cross(p0, p2, p1) < 0.0;
    let chord_side = if opposite {
        across.gt(&zero)
    } else {
        across.le(&zero)
    };
    let inside = outer_edges.and(&f.le(&zero)).and(&chord_side);
    (
        inside.select(&constant_of(sign), &zero),
        in_pixels(&f, &f).abs(),
    )
}

/// Coverage of the included pieces relative to `reference`: the exact
/// winding number, hard, decides inside from outside; the distance to the
/// nearest piece, soft, decides how much of the pixel.
///
/// The two are separate on purpose. An earlier draft softened each winding
/// term with its own ramp, which made the antialiasing depend on the
/// reference point: a chord's ramp lived inside its cone, and a reference
/// point near a chord's line makes the cone thinner than the ramp, so a
/// sample half a pixel from the edge could fall outside it and get the hard
/// value. Here every term is exact whatever the reference point, and the
/// ramp is the one signed-distance ramp every edge of the outline shares.
fn coverage(reference: &Reference, included: &[Included]) -> Kernel {
    let mut spokes = Spokes::from(reference.o);
    let mut terms = Vec::with_capacity(1 + 2 * included.len());
    terms.push(constant_of(f64::from(reference.winding)));
    let mut distance = constant(RAMP_REACH);
    for included in included {
        let piece = included.piece;
        let across = across(reference.o, piece.chord);
        if included.chord_near {
            let (cone, sigma) = cone(&mut spokes, piece.chord);
            terms.push(shadow_term(&cone, sigma, &across));
        }
        let implicit = included.bulge.map(|bulge| {
            let (term, implicit) = sliver(reference.o, bulge, &across);
            terms.push(term);
            implicit
        });
        if !included.ramp_near {
            continue;
        }
        // The distance to the chord, less how far the segment strays from
        // it, is a lower bound on the distance to the segment — and it is
        // the exact distance for a line, where nothing strays. For a curve
        // the implicit is the better estimate near the arc and the bound is
        // the sound one away from it, so take whichever is larger: the
        // bound can never exceed the truth, so the larger is the closer.
        let chord = segment_distance(piece.chord, &across);
        let bound = chord.sub(&constant_of(piece.deviation()));
        distance = distance.min(&match implicit {
            Some(implicit) => implicit.max(&bound),
            None => bound,
        });
    }
    let winding = Kernel::sum(&terms);
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
    fn the_chord_polygon_winding_counts_a_square_once() {
        let sq = |o: P, ccw: bool| {
            let v = [[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]];
            let order: Vec<usize> = if ccw {
                vec![0, 1, 2, 3]
            } else {
                vec![0, 3, 2, 1]
            };
            let chords: Vec<Chord> = (0..4)
                .map(|i| Chord {
                    a: v[order[i]],
                    b: v[order[(i + 1) % 4]],
                })
                .collect();
            winding_of_chord_polygon(o, chords.into_iter())
        };
        assert_eq!(sq([2.0, 2.0], true).abs(), 1);
        assert_eq!(sq([2.0, 2.0], true), -sq([2.0, 2.0], false));
        assert_eq!(sq([5.0, 2.0], true), 0);
        assert_eq!(sq([-1.0, 2.0], true), 0);
        // Level with a vertex: counted once, never twice.
        assert_eq!(sq([2.0, 0.0], true).abs(), 1);
        assert_eq!(sq([2.0, 4.0], true), 0);
    }

    #[test]
    fn a_reference_point_is_nudged_off_a_chord_line() {
        let chord = Chord {
            a: [0.0, 2.0],
            b: [4.0, 2.0],
        };
        let pieces = Pieces {
            pieces: vec![Piece { chord, bulge: None }],
        };
        let r = Reference::clear_of([2.0, 2.0], &[chord], &pieces);
        assert!(chord.line_distance(r.o) >= REFERENCE_CLEARANCE);
        assert_ne!(r.o, [2.0, 2.0]);
    }

    /// A curve straight to within [`FLAT_ENOUGH`] is a line: no sliver, no
    /// implicit, no triangle.
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
}
