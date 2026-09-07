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
#[test]
fn a_grazing_ray_picks_up_no_winding_at_a_shared_extremum() {
    // Incoming segment rises to the shared point at its END (y1 == y2, so
    // t_vertex == 1); outgoing leaves it at its START (y0 == y1, t_vertex == 0).
    let shared = [-0.966_354_37f32, 8.683_796];
    let incoming = AnalyticalQuad::new(
        [-4.499_054, 0.079_550_94],
        [-1.554_296_3, 8.683_796],
        shared,
    );
    let outgoing =
        AnalyticalQuad::new(shared, [1.835_096_6, 8.683_796], [4.617_066_4, 3.184_099_4]);

    let sum = incoming.kernel().add(&outgoing.kernel());
    let (arena, root) = sum.parts();
    let (lowered, r) = lower_dwrt_owned(arena, root).expect("lower");

    // A ray origin well to the right of both segments, so both crossings are
    // fully covered and any surviving one shows at its full weight.
    let x = 39.033_646f32;

    // Walk the extremum row in single ulps: the disagreement band is a few
    // ulps wide, so sampling only the exact row would miss it.
    let mut y = shared[1];
    for _ in 0..8 {
        y = f32::from_bits(y.to_bits() - 1);
    }
    let mut worst = 0.0f32;
    let mut worst_y = y;
    for _ in 0..17 {
        let v = eval_scalar(&lowered, r, &[x, y, 0.0, 0.0], &BindingTable::empty());
        if v.abs() > worst.abs() {
            worst = v;
            worst_y = y;
        }
        y = f32::from_bits(y.to_bits() + 1);
    }

    assert!(
        worst.abs() < GRAZE_TOLERANCE,
        "a grazing ray picked up {worst} of winding at y = {worst_y:?} \
         ({:#x}) — the two segments' `disc >= 0` decisions disagreed and one \
         crossing survived uncancelled",
        worst_y.to_bits()
    );
}
