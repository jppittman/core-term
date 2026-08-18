//! The one `run_scalar`: broadcast-and-read-lane-0 execution of a JIT-compiled
//! [`ExecutableCode`] against the oracle's scalar calling convention.
//!
//! Every harness that cross-checks the JIT against
//! [`pixelflow_ir::eval_scalar`] needs to call compiled code with a single
//! `(x, y, z, w)` tuple and read back a single `f32` — but the JIT's ABI is
//! whatever SIMD width the build's target features selected (`__m128`,
//! `__m256`, `__m512`, or NEON's `float32x4_t`), so "call it with a scalar"
//! means "broadcast to every lane, then read lane 0." This used to be three
//! byte-identical copies (`training::quarantine`, `bin::bench_extraction_3way`,
//! `pixelflow-codegen`'s `tests/prod_kernel_jit.rs`) that a fix to one could
//! silently fail to reach — a NO-SILENT-FAILURES violation in waiting
//! (docs/plans/2026-08-17-cost-model-domain.md, J14). One definition here,
//! imported everywhere, closes that.
use pixelflow_codegen::emit::executable::{ExecutableCode, KernelFn};

/// Broadcast `(x, y, z, w)` to every SIMD lane, call `code`, and read back
/// lane 0. The lane width is whichever ABI the JIT emitted for the build's
/// target features, so this branches on the same `target_feature`/
/// `target_arch` cfgs the JIT's codegen does.
#[cfg(all(
    target_arch = "x86_64",
    not(target_feature = "avx512f"),
    not(target_feature = "avx2")
))]
pub fn run_scalar(code: &ExecutableCode, x: f32, y: f32, z: f32, w: f32) -> f32 {
    use core::arch::x86_64::{_mm_cvtss_f32, _mm_set1_ps};
    // SAFETY: SSE2 is the baseline on x86-64; the JIT emitted the `__m128` ABI.
    unsafe {
        let f: KernelFn = code.as_fn();
        let r = f(
            _mm_set1_ps(x),
            _mm_set1_ps(y),
            _mm_set1_ps(z),
            _mm_set1_ps(w),
        );
        _mm_cvtss_f32(r)
    }
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    not(target_feature = "avx512f")
))]
pub fn run_scalar(code: &ExecutableCode, x: f32, y: f32, z: f32, w: f32) -> f32 {
    use core::arch::x86_64::{_mm256_cvtss_f32, _mm256_set1_ps};
    // SAFETY: built with +avx2 (not +avx512f), so the JIT emitted the
    // `__m256` ABI.
    unsafe {
        let f: KernelFn = code.as_fn();
        let r = f(
            _mm256_set1_ps(x),
            _mm256_set1_ps(y),
            _mm256_set1_ps(z),
            _mm256_set1_ps(w),
        );
        _mm256_cvtss_f32(r)
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
pub fn run_scalar(code: &ExecutableCode, x: f32, y: f32, z: f32, w: f32) -> f32 {
    use core::arch::x86_64::{_mm512_cvtss_f32, _mm512_set1_ps};
    // SAFETY: built with +avx512f, so the JIT emitted the `__m512` ABI.
    unsafe {
        let f: KernelFn = code.as_fn();
        let r = f(
            _mm512_set1_ps(x),
            _mm512_set1_ps(y),
            _mm512_set1_ps(z),
            _mm512_set1_ps(w),
        );
        _mm512_cvtss_f32(r)
    }
}

#[cfg(target_arch = "aarch64")]
pub fn run_scalar(code: &ExecutableCode, x: f32, y: f32, z: f32, w: f32) -> f32 {
    use core::arch::aarch64::{vdupq_n_f32, vgetq_lane_f32};
    // SAFETY: NEON is mandatory on aarch64; the JIT emitted the `float32x4_t` ABI.
    unsafe {
        let f: KernelFn = code.as_fn();
        let r = f(
            vdupq_n_f32(x),
            vdupq_n_f32(y),
            vdupq_n_f32(z),
            vdupq_n_f32(w),
        );
        vgetq_lane_f32(r, 0)
    }
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
pub fn run_scalar(_code: &ExecutableCode, _x: f32, _y: f32, _z: f32, _w: f32) -> f32 {
    panic!("oracle_compare::run_scalar requires a JIT-capable architecture")
}
