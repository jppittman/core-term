//! Integration tests for the kernel_jit! macro.
//!
//! These tests verify the full pipeline from macro input to executable JIT code.

use pixelflow_compiler::kernel_jit;

#[test]
#[cfg(target_arch = "aarch64")]
fn test_jit_macro_return_x() {
    // Simplest: return X
    let func = kernel_jit!(|| X);

    unsafe {
        use core::arch::aarch64::*;
        let x = vdupq_n_f32(42.0);
        let y = vdupq_n_f32(0.0);
        let z = vdupq_n_f32(0.0);
        let w = vdupq_n_f32(0.0);

        let result = func(x, y, z, w);
        assert_eq!(vgetq_lane_f32(result, 0), 42.0);
    }
}

#[test]
#[cfg(target_arch = "aarch64")]
fn test_jit_macro_add_xy() {
    // X + Y
    let func = kernel_jit!(|| X + Y);

    unsafe {
        use core::arch::aarch64::*;
        let x = vdupq_n_f32(10.0);
        let y = vdupq_n_f32(32.0);
        let z = vdupq_n_f32(0.0);
        let w = vdupq_n_f32(0.0);

        let result = func(x, y, z, w);
        assert_eq!(vgetq_lane_f32(result, 0), 42.0);
    }
}

#[test]
#[cfg(target_arch = "aarch64")]
fn test_jit_macro_complex_expr() {
    // (X + Y) * Z
    let func = kernel_jit!(|| (X + Y) * Z);

    unsafe {
        use core::arch::aarch64::*;
        let x = vdupq_n_f32(2.0);
        let y = vdupq_n_f32(5.0);
        let z = vdupq_n_f32(6.0);
        let w = vdupq_n_f32(0.0);

        let result = func(x, y, z, w);
        assert_eq!(vgetq_lane_f32(result, 0), 42.0);  // (2+5)*6 = 42
    }
}

#[test]
#[cfg(target_arch = "aarch64")]
fn test_jit_macro_subtraction() {
    // X - Y
    let func = kernel_jit!(|| X - Y);

    unsafe {
        use core::arch::aarch64::*;
        let x = vdupq_n_f32(100.0);
        let y = vdupq_n_f32(58.0);
        let z = vdupq_n_f32(0.0);
        let w = vdupq_n_f32(0.0);

        let result = func(x, y, z, w);
        assert_eq!(vgetq_lane_f32(result, 0), 42.0);
    }
}

#[test]
#[cfg(target_arch = "aarch64")]
fn test_jit_macro_division() {
    // X / Y
    let func = kernel_jit!(|| X / Y);

    unsafe {
        use core::arch::aarch64::*;
        let x = vdupq_n_f32(84.0);
        let y = vdupq_n_f32(2.0);
        let z = vdupq_n_f32(0.0);
        let w = vdupq_n_f32(0.0);

        let result = func(x, y, z, w);
        assert_eq!(vgetq_lane_f32(result, 0), 42.0);
    }
}

#[test]
#[cfg(target_arch = "aarch64")]
fn test_jit_macro_negation() {
    // -X
    let func = kernel_jit!(|| -X);

    unsafe {
        use core::arch::aarch64::*;
        let x = vdupq_n_f32(-42.0);
        let y = vdupq_n_f32(0.0);
        let z = vdupq_n_f32(0.0);
        let w = vdupq_n_f32(0.0);

        let result = func(x, y, z, w);
        assert_eq!(vgetq_lane_f32(result, 0), 42.0);
    }
}

// ============================================================================
// Transcendentals (lowered via polynomial)
// ============================================================================

#[test]
#[cfg(target_arch = "aarch64")]
fn test_jit_macro_sin() {
    // sin(X) - test that sin(0) ≈ 0
    let func = kernel_jit!(|| X.sin());

    unsafe {
        use core::arch::aarch64::*;
        let x = vdupq_n_f32(0.0);  // sin(0) = 0
        let y = vdupq_n_f32(0.0);
        let z = vdupq_n_f32(0.0);
        let w = vdupq_n_f32(0.0);

        let result = func(x, y, z, w);
        let val = vgetq_lane_f32(result, 0);
        assert!((val - 0.0).abs() < 0.001, "sin(0) = {}, expected ~0", val);
    }
}

#[test]
#[cfg(target_arch = "aarch64")]
fn test_jit_macro_sin_pi_half() {
    // sin(π/2) ≈ 1
    let func = kernel_jit!(|| X.sin());

    unsafe {
        use core::arch::aarch64::*;
        let pi_half = core::f32::consts::FRAC_PI_2;
        let x = vdupq_n_f32(pi_half);
        let y = vdupq_n_f32(0.0);
        let z = vdupq_n_f32(0.0);
        let w = vdupq_n_f32(0.0);

        let result = func(x, y, z, w);
        let val = vgetq_lane_f32(result, 0);
        assert!((val - 1.0).abs() < 0.01, "sin(π/2) = {}, expected ~1", val);
    }
}

#[test]
#[cfg(target_arch = "aarch64")]
fn test_jit_macro_cos() {
    // cos(0) = 1
    let func = kernel_jit!(|| X.cos());

    unsafe {
        use core::arch::aarch64::*;
        let x = vdupq_n_f32(0.0);
        let y = vdupq_n_f32(0.0);
        let z = vdupq_n_f32(0.0);
        let w = vdupq_n_f32(0.0);

        let result = func(x, y, z, w);
        let val = vgetq_lane_f32(result, 0);
        assert!((val - 1.0).abs() < 0.01, "cos(0) = {}, expected ~1", val);
    }
}

#[test]
#[cfg(target_arch = "aarch64")]
fn test_jit_macro_sqrt() {
    // sqrt(X)
    let func = kernel_jit!(|| X.sqrt());

    unsafe {
        use core::arch::aarch64::*;
        let x = vdupq_n_f32(1764.0);  // sqrt(1764) = 42
        let y = vdupq_n_f32(0.0);
        let z = vdupq_n_f32(0.0);
        let w = vdupq_n_f32(0.0);

        let result = func(x, y, z, w);
        assert_eq!(vgetq_lane_f32(result, 0), 42.0);
    }
}

#[test]
#[cfg(target_arch = "aarch64")]
fn test_jit_macro_abs() {
    // abs(X)
    let func = kernel_jit!(|| X.abs());

    unsafe {
        use core::arch::aarch64::*;
        let x = vdupq_n_f32(-42.0);
        let y = vdupq_n_f32(0.0);
        let z = vdupq_n_f32(0.0);
        let w = vdupq_n_f32(0.0);

        let result = func(x, y, z, w);
        assert_eq!(vgetq_lane_f32(result, 0), 42.0);
    }
}

#[test]
#[cfg(target_arch = "aarch64")]
fn test_jit_macro_min_max() {
    // min(X, Y)
    let func_min = kernel_jit!(|| X.min(Y));
    let func_max = kernel_jit!(|| X.max(Y));

    unsafe {
        use core::arch::aarch64::*;
        let x = vdupq_n_f32(10.0);
        let y = vdupq_n_f32(42.0);
        let z = vdupq_n_f32(0.0);
        let w = vdupq_n_f32(0.0);

        let result_min = func_min(x, y, z, w);
        let result_max = func_max(x, y, z, w);

        assert_eq!(vgetq_lane_f32(result_min, 0), 10.0);
        assert_eq!(vgetq_lane_f32(result_max, 0), 42.0);
    }
}

// ============================================================================
// x86-64 versions
// ============================================================================

#[test]
#[cfg(target_arch = "x86_64")]
fn test_jit_macro_return_x_x86() {
    let func = kernel_jit!(|| X);

    unsafe {
        use core::arch::x86_64::*;
        let x = _mm_set1_ps(42.0);
        let y = _mm_setzero_ps();
        let z = _mm_setzero_ps();
        let w = _mm_setzero_ps();

        let result = func(x, y, z, w);
        assert_eq!(_mm_cvtss_f32(result), 42.0);
    }
}

#[test]
#[cfg(target_arch = "x86_64")]
fn test_jit_macro_add_xy_x86() {
    let func = kernel_jit!(|| X + Y);

    unsafe {
        use core::arch::x86_64::*;
        let x = _mm_set1_ps(10.0);
        let y = _mm_set1_ps(32.0);
        let z = _mm_setzero_ps();
        let w = _mm_setzero_ps();

        let result = func(x, y, z, w);
        assert_eq!(_mm_cvtss_f32(result), 42.0);
    }
}
