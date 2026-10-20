//! `exp` and `exp2` stay finite and monotone as they run through the exponent
//! clamp.
//!
//! CLAUDE.md, "Floating point at the edges": `exp` and `exp2` saturate past
//! ±126 exponents rather than overflowing to `inf`. The exact saturated values
//! (`2^126` above the clamp, `2^-126` below it) are pinned bit for bit in
//! `transcendental_jit.rs`. What that file does not pin is the sweep: as the
//! input climbs through the clamp, the result must never be `inf` or NaN, and
//! it must never fall. A kernel that overflows, wraps its exponent field, or
//! saturates downward breaks one of those two laws somewhere on the sweep.
//!
//! The sweeps are ascending and straddle the clamp on both sides, so the
//! transition from an in-range value to a saturated one is inside the check.
//! `exp(x)` is `exp2(x·log2 e)`, which reaches the clamp at `x ≈ ±87.34`.

use pixelflow_codegen::jit_cache;
use pixelflow_ir::{Kernel, LatticeShape};

/// `exp2` clamps its argument at ±126; this sweep runs through both boundaries.
const EXP2_SWEEP: [f32; 12] = [
    -1e6, -200.0, -127.0, -126.0, -1.0, 0.0, 1.0, 125.0, 126.0, 127.0, 200.0, 1e6,
];

/// `exp` clamps `x·log2 e` at ±126, which is `x ≈ ±87.34`. The sweep takes
/// in-range values just inside that point, and ±90 just past it.
const EXP_SWEEP: [f32; 12] = [
    -1e6, -200.0, -127.0, -90.0, -88.0, 0.0, 87.0, 88.0, 90.0, 127.0, 200.0, 1e6,
];

/// One point of a kernel compiled at [`LatticeShape::POINT`], read back through
/// the origin block.
fn eval_point(jit: &pixelflow_codegen::emit::CompiledKernel, x: f32) -> f32 {
    let mut out = [0.0f32; 1];
    let origin = [x, 0.0];
    // SAFETY: the kernels here declare no buffer and no uniform, so `ctx[0]`,
    // the uniform-block slot, is unread; `ctx[1]` is the origin block, and
    // `out` holds the one sample a single-point lattice writes.
    let ctx: [*const f32; 2] = [core::ptr::null(), origin.as_ptr()];
    unsafe {
        jit.call(ctx.as_ptr(), out.as_mut_ptr(), 1);
    }
    out[0]
}

fn check_finite_and_monotone(name: &str, k: &Kernel, sweep: &[f32]) {
    let jit = jit_cache::compile(k, LatticeShape::POINT)
        .unwrap_or_else(|e| panic!("{name}: kernel failed to compile on this backend: {e}"))
        .kernel;
    let mut previous = f32::NEG_INFINITY;
    for &x in sweep {
        let got = eval_point(&jit, x);
        assert!(
            got.is_finite(),
            "{name}({x}): got {got} ({:#010x}); past the clamp the result must saturate, not be inf or NaN",
            got.to_bits()
        );
        assert!(
            got >= previous,
            "{name} is not monotone: the result fell from {previous:e} to {got:e} at x = {x}"
        );
        previous = got;
    }
}

#[test]
fn exp2_stays_finite_and_monotone_through_the_clamp() {
    check_finite_and_monotone("exp2", &Kernel::x().exp2(), &EXP2_SWEEP);
}

#[test]
fn exp_stays_finite_and_monotone_through_the_clamp() {
    check_finite_and_monotone("exp", &Kernel::x().exp(), &EXP_SWEEP);
}
