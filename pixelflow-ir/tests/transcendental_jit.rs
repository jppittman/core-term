//! exp/log kernels through the JIT, checked numerically at this build's width.
//!
//! These lower (`expand_transcendentals`) to the integer-domain primitives
//! TruncToInt/IntToFloat/IAdd/Shl/Shr — ops a backend can silently lack while
//! every other test stays green, because `Field`'s combinator tier computes
//! its own log2/exp via native intrinsics and never consults the JIT. That is
//! exactly how Avx512Backend shipped refusing `ShiftImm`: `test_log2` passed
//! (combinator tier), the full workspace passed under `+avx512f`, and a
//! `Kernel::log2` still failed to compile. The completeness sweep catches the
//! emission gap; this catches the semantics — the bytes must also be right.

use pixelflow_ir::{Kernel, jit_cache};

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
fn call_x(jit: &pixelflow_ir::JitManifold, x: f32) -> f32 {
    use core::arch::x86_64::*;
    unsafe {
        let v = _mm512_set1_ps(x);
        let z = _mm512_setzero_ps();
        _mm512_cvtss_f32(jit.call(v, z, z, z))
    }
}
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    not(target_feature = "avx512f")
))]
fn call_x(jit: &pixelflow_ir::JitManifold, x: f32) -> f32 {
    use core::arch::x86_64::*;
    unsafe {
        let v = _mm256_set1_ps(x);
        let z = _mm256_setzero_ps();
        _mm256_cvtss_f32(jit.call(v, z, z, z))
    }
}
#[cfg(all(
    target_arch = "x86_64",
    not(target_feature = "avx2"),
    not(target_feature = "avx512f")
))]
fn call_x(jit: &pixelflow_ir::JitManifold, x: f32) -> f32 {
    use core::arch::x86_64::*;
    unsafe {
        let v = _mm_set1_ps(x);
        let z = _mm_setzero_ps();
        _mm_cvtss_f32(jit.call(v, z, z, z))
    }
}
#[cfg(target_arch = "aarch64")]
fn call_x(jit: &pixelflow_ir::JitManifold, x: f32) -> f32 {
    use core::arch::aarch64::*;
    unsafe {
        let v = vdupq_n_f32(x);
        let z = vdupq_n_f32(0.0);
        vgetq_lane_f32::<0>(jit.call(v, z, z, z))
    }
}

/// Two-operand form of [`call_x`]: X = `x`, Y = `y`, lane 0 of the result.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
fn call_xy(jit: &pixelflow_ir::JitManifold, x: f32, y: f32) -> f32 {
    use core::arch::x86_64::*;
    unsafe {
        let z = _mm512_setzero_ps();
        _mm512_cvtss_f32(jit.call(_mm512_set1_ps(x), _mm512_set1_ps(y), z, z))
    }
}
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    not(target_feature = "avx512f")
))]
fn call_xy(jit: &pixelflow_ir::JitManifold, x: f32, y: f32) -> f32 {
    use core::arch::x86_64::*;
    unsafe {
        let z = _mm256_setzero_ps();
        _mm256_cvtss_f32(jit.call(_mm256_set1_ps(x), _mm256_set1_ps(y), z, z))
    }
}
#[cfg(all(
    target_arch = "x86_64",
    not(target_feature = "avx2"),
    not(target_feature = "avx512f")
))]
fn call_xy(jit: &pixelflow_ir::JitManifold, x: f32, y: f32) -> f32 {
    use core::arch::x86_64::*;
    unsafe {
        let z = _mm_setzero_ps();
        _mm_cvtss_f32(jit.call(_mm_set1_ps(x), _mm_set1_ps(y), z, z))
    }
}
#[cfg(target_arch = "aarch64")]
fn call_xy(jit: &pixelflow_ir::JitManifold, x: f32, y: f32) -> f32 {
    use core::arch::aarch64::*;
    unsafe {
        let z = vdupq_n_f32(0.0);
        vgetq_lane_f32::<0>(jit.call(vdupq_n_f32(x), vdupq_n_f32(y), z, z))
    }
}

fn check(name: &str, k: &Kernel, inputs: &[f32], reference: impl Fn(f32) -> f32) {
    let (arena, root) = k.parts();
    let jit = jit_cache::compile_cached(arena, root)
        .unwrap_or_else(|e| panic!("{name}: kernel failed to compile on this backend: {e}"));
    for &x in inputs {
        let got = call_x(&jit, x);
        let want = reference(x);
        let rel = ((got - want) / want.abs().max(1e-6)).abs();
        assert!(
            rel <= 1e-3 || (got - want).abs() <= 1e-4,
            "{name}({x}): got {got}, want {want} (rel {rel:.2e})"
        );
    }
}

#[test]
fn exp_log_kernels_compile_and_agree_with_std() {
    // Log domains can range wide; exp inputs stay where e^x is a finite f32.
    // The expansion clamps its exponent to ±126 by design (expand_exp2,
    // lowering.rs) — the JIT saturates where std overflows to inf, so inputs
    // past ~88.7 (ln(f32::MAX)) compare a clamp against an inf, which is a
    // contract difference, not an encoding bug.
    let logs = [0.5f32, 1.0, 2.0, 8.0, 100.0, 4096.0];
    let exps = [-10.0f32, -1.0, 0.0, 0.5, 1.0, 8.0, 80.0];
    check("log2", &Kernel::x().log2(), &logs, f32::log2);
    check("ln", &Kernel::x().ln(), &logs, f32::ln);
    check("exp", &Kernel::x().exp(), &exps, f32::exp);
    check("exp2", &Kernel::x().exp2(), &exps, f32::exp2);
}

// ── Floating-point edge cases: the documented contract, not IEEE ─────────────
//
// The language's FP contract (CLAUDE.md, "Floating point at the edges") is
// *hardware semantics*, deliberately not scalar Rust's: NaN comparisons,
// Min/Max NaN handling and Round's tie-breaking follow whatever the SIMD
// instruction does, because matching `f32::min`/`f32::round` would cost extra
// instructions in the hot path for cases the language does not promise.
//
// What the contract DOES require is that the two tiers agree: the JIT and
// `OpKind::eval_*` (the differential oracle, and the e-graph's constant
// folder) must produce the same answer, or a folded expression differs from
// the same expression computed at runtime. These tests pin that agreement at
// whatever width the build selects — they are the reason the oracle was
// changed to match the hardware rather than the encoders changed to match
// scalar Rust.

/// Every lane of a binary kernel, JIT vs oracle, on edge-case inputs.
fn assert_tiers_agree_binary(name: &str, k: &Kernel, op: pixelflow_ir::OpKind) {
    let (arena, root) = k.parts();
    let jit = jit_cache::compile_cached(arena, root).unwrap_or_else(|e| panic!("{name}: {e}"));
    let nan = f32::NAN;
    for (x, y) in [
        (1.0f32, nan),
        (nan, 1.0f32),
        (nan, nan),
        (2.5, 2.5),
        (-0.0, 0.0),
        (1.0, 2.0),
        (2.0, 1.0),
        (f32::INFINITY, 1.0),
    ] {
        let got = call_xy(&jit, x, y);
        let want = op.eval_binary(x, y).expect("oracle covers this op");
        assert!(
            got == want || (got.is_nan() && want.is_nan()),
            "{name}({x}, {y}): JIT gave {got}, oracle gave {want} — tiers must agree"
        );
    }
}

#[test]
fn min_max_nan_handling_agrees_between_tiers() {
    use pixelflow_ir::OpKind;
    assert_tiers_agree_binary("min", &Kernel::x().min(&Kernel::y()), OpKind::Min);
    assert_tiers_agree_binary("max", &Kernel::x().max(&Kernel::y()), OpKind::Max);
}

#[test]
fn nan_comparisons_agree_between_tiers() {
    use pixelflow_ir::OpKind;
    // Gt/Ge are the unordered predicates (true for NaN); Lt/Le are ordered.
    // The asymmetry is the hardware's, and the oracle now records it.
    let one = Kernel::constant(1.0);
    let zero = Kernel::constant(0.0);
    for (name, op, k) in [
        ("gt", OpKind::Gt, Kernel::x().gt(&Kernel::y())),
        ("ge", OpKind::Ge, Kernel::x().ge(&Kernel::y())),
        ("lt", OpKind::Lt, Kernel::x().lt(&Kernel::y())),
        ("le", OpKind::Le, Kernel::x().le(&Kernel::y())),
    ] {
        // Masks are not numbers; select turns one back into 1.0/0.0 so the
        // comparison against the oracle's 1.0/0.0 is meaningful.
        assert_tiers_agree_binary(name, &k.select(&one, &zero), op);
    }
}

#[test]
fn round_breaks_ties_to_even_in_both_tiers() {
    use pixelflow_ir::OpKind;
    let rounded = Kernel::x().round();
    let (arena, root) = rounded.parts();
    let jit = jit_cache::compile_cached(arena, root).expect("round compiles");
    for x in [0.5f32, 1.5, 2.5, 3.5, -0.5, -1.5, -2.5, 2.4, 2.6] {
        let got = call_x(&jit, x);
        let want = OpKind::Round.eval_unary(x).expect("oracle covers Round");
        assert_eq!(got, want, "round({x}): JIT {got} vs oracle {want}");
    }
    // And the contract itself: nearest-even, so 2.5 -> 2, not 3.
    assert_eq!(call_x(&jit, 2.5), 2.0);
    assert_eq!(call_x(&jit, 3.5), 4.0);
}
