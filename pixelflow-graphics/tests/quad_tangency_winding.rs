//! A tangential touch must contribute zero winding — and must keep doing so
//! when the two segments that form it round their discriminants differently.
//!
//! Where an outline reaches a local Y-extremum at an on-curve point, the two
//! quadratics meeting there each have that point as an endpoint *and* as their
//! own extremum (`t_vertex` exactly 0 for the outgoing one, exactly 1 for the
//! incoming one — 12 of `'8'`'s 32 quadratics are this shape). A horizontal
//! ray through that row grazes the outline: it enters and leaves at the same
//! point, so the two segments contribute equal coverage with opposite winding
//! signs and the sum is zero.
//!
//! That cancellation is load-bearing and it is not robust. Each segment
//! decides whether it reaches the row with its own `disc >= 0`, and on the
//! extremum row `disc = Y*4ay + (by^2 - 4ay*cy)` sits within an ulp of zero,
//! so the two decisions are made by two different roundings of two different
//! expressions. When they disagree, one segment contributes its single valid
//! root and the other contributes nothing, and a whole crossing survives.
//!
//! This test is deliberately free of the e-graph: it evaluates the lowered
//! arena directly, so it pins the numerics rather than an extraction choice.
//! A guard that only holds for the extraction the optimizer happens to pick
//! is not guarding anything.

use pixelflow_graphics::fonts::ttf_curve_analytical::AnalyticalQuad;
use pixelflow_ir::{eval_scalar, passes::lower_dwrt_owned, BindingTable};

/// The largest winding a grazing ray may pick up. A crossing is 1.0 and a
/// half-covered crossing 0.5, so anything at that scale is a lost or doubled
/// intersection rather than rounding.
const GRAZE_TOLERANCE: f32 = 1.0e-2;

/// Found by search over shared-extremum segment pairs: coefficients whose
/// discriminants straddle zero differently at the extremum row.
/// How far either side of the extremum row to walk, in ulps. The band the fix
/// installs is ~2200 ulps wide and the residual peaks around 100 ulps out, so
/// a narrow window reports a number from the wrong part of the curve: at ±8
/// ulps this test measured 0.0076 where the true saturated peak is 8.76/K.
const WALK_ULPS: i32 = 512;

/// Worst grazing winding within [`WALK_ULPS`] of `extremum`, for a pair of
/// quadratics meeting there, sampled at `x`.
fn worst_grazing(
    incoming: AnalyticalQuad,
    outgoing: AnalyticalQuad,
    extremum: f32,
    x: f32,
) -> (f32, f32) {
    let sum = incoming.kernel().add(&outgoing.kernel());
    let (arena, root) = sum.parts();
    let (lowered, r) = lower_dwrt_owned(arena, root).expect("lower");

    // `next_down`/`next_up`, not bit arithmetic: the extremum here can be
    // exactly 0.0, where `to_bits() - 1` underflows (a debug panic, and in
    // release a wrap to a NaN bit pattern that silently walks nowhere near the
    // row under test).
    let mut y = extremum;
    for _ in 0..WALK_ULPS {
        y = y.next_down();
    }
    let (mut worst, mut worst_y) = (0.0f32, y);
    for _ in 0..=(2 * WALK_ULPS) {
        let v = eval_scalar(&lowered, r, &[x, y], &BindingTable::empty());
        assert!(
            v.is_finite(),
            "winding is {v} at y = {y:?} ({:#x}) — a non-finite coverage is a \
             division by a vanished quantity, not an inaccurate one",
            y.to_bits()
        );
        if v.abs() > worst.abs() {
            worst = v;
            worst_y = y;
        }
        y = y.next_up();
    }
    (worst, worst_y)
}

/// Found by search over shared-extremum segment pairs: coefficients whose
/// discriminants straddle zero differently at the extremum row.
#[test]
fn a_grazing_ray_picks_up_no_winding_at_a_shared_extremum() {
    let shared = [-0.966_354_37f32, 8.683_796];
    let incoming = AnalyticalQuad::new(
        [-4.499_054, 0.079_550_94],
        [-1.554_296_3, 8.683_796],
        shared,
    );
    let outgoing =
        AnalyticalQuad::new(shared, [1.835_096_6, 8.683_796], [4.617_066_4, 3.184_099_4]);
    let (worst, worst_y) = worst_grazing(incoming, outgoing, shared[1], 39.033_646);
    assert!(
        worst.abs() < GRAZE_TOLERANCE,
        "a grazing ray picked up {worst} of winding at y = {worst_y:?} \
         ({:#x}) — the two segments' reach into this row disagreed and one \
         crossing survived uncancelled",
        worst_y.to_bits()
    );
}

/// The same pair, translated so the shared extremum sits at **exactly zero**.
///
/// This is not a contrived coordinate. `Font::compile` normalizes every
/// outline with the bbox y-minimum at exactly 0.0, so any glyph whose lowest
/// point is an on-curve extremum — `'8'`, `'O'`, `'e'`, most round letters —
/// puts this pair at `y == 0.0` in the frame the kernel is built in.
///
/// A band scaled by the discriminant's distance from the coordinate origin
/// vanishes here, and `0 * (1/0)` is NaN. The band must be built from
/// differences of the control points, which translation cannot touch.
#[test]
fn a_shared_extremum_at_the_origin_still_cancels() {
    let dy = -8.683_796f32; // put the shared extremum on y == 0.0
    let shared = [-0.966_354_37f32, 8.683_796 + dy];
    assert_eq!(shared[1], 0.0, "the extremum must land exactly on zero");
    let incoming = AnalyticalQuad::new(
        [-4.499_054, 0.079_550_94 + dy],
        [-1.554_296_3, 8.683_796 + dy],
        shared,
    );
    let outgoing = AnalyticalQuad::new(
        shared,
        [1.835_096_6, 8.683_796 + dy],
        [4.617_066_4, 3.184_099_4 + dy],
    );
    let (worst, worst_y) = worst_grazing(incoming, outgoing, shared[1], 39.033_646);
    assert!(
        worst.abs() < GRAZE_TOLERANCE,
        "a grazing ray picked up {worst} of winding at y = {worst_y:?} at the \
         origin — the band is being measured from the origin rather than from \
         the segment's own geometry"
    );
}
