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
