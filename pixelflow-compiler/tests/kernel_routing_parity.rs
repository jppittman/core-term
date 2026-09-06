//! Tier parity: the LLVM tier (`kernel!`) and the arena tier (`kernel_jit!`)
//! must agree with scalar f32 ground truth on the same body.
//!
//! The two tiers no longer share a value: `kernel!` emits a combinator ZST
//! that a caller evaluates, and `kernel_jit!` produces a `Kernel` that a
//! caller *bakes* over a lattice. So parity is checked the only way it can
//! be — against ground truth, from both sides, at the same tolerance:
//!
//! - the arena tier through `Lattice::point(..).bake(..)`, the one evaluation
//!   entry there is;
//! - the LLVM tier through this test's own loop over `Manifold::eval` (a test
//!   owns its loop; that is not an API).
//!
//! Transcendentals are deliberately absent: the combinator backend's `sin` is
//! known-inaccurate (see the header of `jit_parity.rs`), which is a parity bug
//! in the to-be-retired backend, tracked until that backend dies.

use pixelflow_compiler::{kernel, kernel_jit};
use pixelflow_core::combinator::Manifold;
use pixelflow_core::{Field, Kernel, Lattice};

type F4 = (Field, Field, Field, Field);

fn lane0(f: Field) -> f32 {
    unsafe { core::mem::transmute_copy(&f) }
}

/// The LLVM tier, evaluated by this test's own loop.
fn eval_template(m: &impl Manifold<F4, Output = Field>, p: (f32, f32, f32, f32)) -> f32 {
    lane0(m.eval((
        Field::from(p.0),
        Field::from(p.1),
        Field::from(p.2),
        Field::from(p.3),
    )))
}

/// The arena tier, tabulated over a one-point lattice.
fn bake(k: &Kernel, p: (f32, f32, f32, f32)) -> f32 {
    Lattice::point(p.0, p.1, p.2, p.3).bake(k).into_buffer()[0]
}

const SAMPLES: &[(f32, f32, f32, f32)] = &[
    (0.0, 0.0, 0.0, 0.0),
    (1.0, 2.0, 3.0, 4.0),
    (-1.5, 0.5, 2.25, -3.0),
    (3.0, 4.0, 0.0, 1.0),
    (0.7, 1.3, -0.9, 2.1),
];

// Wide enough for the combinator backend's approximate-rsqrt `sqrt`
// (~5e-4 relative; the JIT emits exact hardware sqrt — its numerics are the
// canonical ones per the plan). Tightens when the combinator emitter dies.
fn check(name: &str, got: f32, want: f32) {
    let diff = (got - want).abs();
    let tol = 1e-3 + 1e-3 * want.abs();
    assert!(
        diff <= tol,
        "{name}: got={got} want={want} diff={diff} tol={tol}"
    );
}

#[test]
fn arithmetic_matches_truth_on_both_tiers() {
    let template = kernel!(|| (X - Y) * Z + X / (W + 10.0))();
    let arena = kernel_jit!(|| (X - Y) * Z + X / (W + 10.0));
    for &p in SAMPLES {
        let want = (p.0 - p.1) * p.2 + p.0 / (p.3 + 10.0);
        check("arith/template", eval_template(&template, p), want);
        check("arith/arena", bake(&arena, p), want);
    }
}

#[test]
fn params_match_truth_on_both_tiers() {
    let (cx, cy, r) = (0.25_f32, -0.75_f32, 1.5_f32);
    let template = kernel!(|cx: f32, cy: f32, r: f32| {
        let dx = X - cx;
        let dy = Y - cy;
        (dx * dx + dy * dy).sqrt() - r
    })(cx, cy, r);
    let arena = kernel_jit!(|cx: f32, cy: f32, r: f32| {
        let dx = X - cx;
        let dy = Y - cy;
        (dx * dx + dy * dy).sqrt() - r
    })(cx, cy, r);
    for &p in SAMPLES {
        let want = ((p.0 - cx).powi(2) + (p.1 - cy).powi(2)).sqrt() - r;
        check("circle_sdf/template", eval_template(&template, p), want);
        check("circle_sdf/arena", bake(&arena, p), want);
    }
}

#[test]
fn piecewise_matches_truth_on_both_tiers() {
    let template = kernel!(|| (X * Y).max(Z).min(10.0))();
    let arena = kernel_jit!(|| (X * Y).max(Z).min(10.0));
    for &p in SAMPLES {
        let want = (p.0 * p.1).max(p.2).min(10.0);
        check("minmax/template", eval_template(&template, p), want);
        check("minmax/arena", bake(&arena, p), want);
    }

    let template = kernel!(|| (X.lt(Y)).select(Z, W))();
    let arena = kernel_jit!(|| (X.lt(Y)).select(Z, W));
    for &p in SAMPLES {
        let want = if p.0 < p.1 { p.2 } else { p.3 };
        check("select/template", eval_template(&template, p), want);
        check("select/arena", bake(&arena, p), want);
    }
}

#[test]
fn abs_recip_matches_truth_on_both_tiers() {
    let template = kernel!(|| (X - Y).abs() + (Z + 5.0).recip())();
    let arena = kernel_jit!(|| (X - Y).abs() + (Z + 5.0).recip());
    for &p in SAMPLES {
        let want = (p.0 - p.1).abs() + 1.0 / (p.2 + 5.0);
        check("abs_recip/template", eval_template(&template, p), want);
        check("abs_recip/arena", bake(&arena, p), want);
    }
}

/// Derivative projections are where the tiers *intentionally* disagree, and
/// that disagreement is the reason they are two macros rather than one
/// routed surface. Over a `Field` domain the combinator backend's `DX` is 0
/// (no derivative information — the fonts' load-bearing "hard step" case);
/// the arena tier differentiates symbolically. A kernel that wants the
/// symbolic reading asks for it by spelling `kernel_jit!`.
#[test]
fn projections_differ_by_tier_and_that_is_the_contract() {
    let template = kernel!(|| DX(X * X))();
    let arena = kernel_jit!(|| DX(X * X));
    for &p in SAMPLES {
        check("dx/template_is_zero", eval_template(&template, p), 0.0);
        check("dx/arena_is_symbolic", bake(&arena, p), 2.0 * p.0);
    }
}
