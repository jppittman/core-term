//! Integration tests for the `kernel!` macro.
//!
//! These verify the full pipeline from macro input to numbers: parse, sema,
//! e-graph, arena, and then the one evaluation entry there is — compiled at a
//! lattice's shape and collapsed.

use pixelflow_compiler::kernel;
use pixelflow_core::{Kernel, Lattice};

// ============================================================================
// Helpers
// ============================================================================

/// Tabulate a kernel over a one-point lattice and read the value back.
///
/// This is the whole evaluation surface: a kernel plus a lattice, baked. There
/// is no per-batch entry to call instead.
fn bake1(k: &Kernel, x: f32, y: f32, z: f32) -> f32 {
    Lattice::point(x, y, z, 0.0).bake(k).into_buffer()[0]
}

fn eval1(k: &Kernel, x: f32) -> f32 {
    bake1(k, x, 0.0, 0.0)
}

fn eval2(k: &Kernel, x: f32, y: f32) -> f32 {
    bake1(k, x, y, 0.0)
}

fn eval3(k: &Kernel, x: f32, y: f32, z: f32) -> f32 {
    bake1(k, x, y, z)
}

// ============================================================================
// Basic arithmetic
// ============================================================================

#[test]
fn macro_return_x() {
    let m = kernel!(|| X);
    assert_eq!(eval1(&m, 42.0), 42.0);
}

#[test]
fn macro_add_xy() {
    let m = kernel!(|| X + Y);
    assert_eq!(eval2(&m, 10.0, 32.0), 42.0);
}

#[test]
fn macro_complex_expr() {
    // (X + Y) * Z
    let m = kernel!(|| (X + Y) * Z);
    assert_eq!(eval3(&m, 2.0, 5.0, 6.0), 42.0); // (2+5)*6 = 42
}

#[test]
fn macro_subtraction() {
    let m = kernel!(|| X - Y);
    assert_eq!(eval2(&m, 100.0, 58.0), 42.0);
}

#[test]
fn macro_division() {
    let m = kernel!(|| X / Y);
    assert_eq!(eval2(&m, 84.0, 2.0), 42.0);
}

#[test]
fn macro_negation() {
    let m = kernel!(|| -X);
    assert_eq!(eval1(&m, -42.0), 42.0);
}

// ============================================================================
// Transcendentals (lowered via polynomial)
// ============================================================================

#[test]
#[cfg(not(target_feature = "avx512f"))] // transcendentals: not in AVX-512 Stage-1 op set
fn macro_sin() {
    // sin(0) = 0
    let m = kernel!(|| X.sin());
    let val = eval1(&m, 0.0);
    assert!((val - 0.0).abs() < 0.001, "sin(0) = {val}, expected ~0");
}

#[test]
#[cfg(not(target_feature = "avx512f"))] // transcendentals: not in AVX-512 Stage-1 op set
fn macro_sin_pi_half() {
    // sin(π/2) ≈ 1
    let m = kernel!(|| X.sin());
    let val = eval1(&m, core::f32::consts::FRAC_PI_2);
    assert!((val - 1.0).abs() < 0.01, "sin(π/2) = {val}, expected ~1");
}

#[test]
#[cfg(not(target_feature = "avx512f"))] // transcendentals: not in AVX-512 Stage-1 op set
fn macro_cos() {
    // cos(0) = 1
    let m = kernel!(|| X.cos());
    let val = eval1(&m, 0.0);
    assert!((val - 1.0).abs() < 0.01, "cos(0) = {val}, expected ~1");
}

#[test]
fn macro_sqrt() {
    // sqrt(1764) = 42
    let m = kernel!(|| X.sqrt());
    assert_eq!(eval1(&m, 1764.0), 42.0);
}

#[test]
fn macro_abs() {
    let m = kernel!(|| X.abs());
    assert_eq!(eval1(&m, -42.0), 42.0);
}

#[test]
fn macro_min_returns_smaller_and_max_returns_larger() {
    let m_min = kernel!(|| X.min(Y));
    let m_max = kernel!(|| X.max(Y));
    assert_eq!(eval2(&m_min, 10.0, 42.0), 10.0);
    assert_eq!(eval2(&m_max, 10.0, 42.0), 42.0);
}

// ============================================================================
// Parameter tests — builder closure API
// ============================================================================

#[test]
fn no_params_is_a_kernel() {
    // Zero-param case: the macro evaluates to a `Kernel` value directly.
    let k = kernel!(|| X + Y);
    assert!((eval2(&k, 10.0, 32.0) - 42.0).abs() < 1e-5);
}

#[test]
fn one_param_is_a_builder() {
    // A single scalar param returns a builder closure |offset: f32| -> Kernel.
    let builder = kernel!(|offset: f32| X + offset);
    let k = builder(32.0_f32);
    assert!((eval1(&k, 10.0) - 42.0).abs() < 1e-5);
}

#[test]
fn two_params_is_a_builder() {
    let builder = kernel!(|cx: f32, r: f32| (X - cx) * r);
    let k = builder(1.0_f32, 2.0_f32);
    // X=5.0: (5.0 - 1.0) * 2.0 = 8.0
    assert!((eval1(&k, 5.0) - 8.0).abs() < 1e-5);
}

// ============================================================================
// Inverse trigonometric functions
// ============================================================================

#[test]
#[cfg(not(target_feature = "avx512f"))] // transcendentals: not in AVX-512 Stage-1 op set
fn macro_atan2_matches_reference_at_boundary_and_interior_points() {
    let m = kernel!(|| Y.atan2(X));
    // atan2(1, 1) = π/4 — polynomial has ~0.06 error at t=1 boundary
    let val = eval2(&m, 1.0, 1.0);
    assert!(
        (val - std::f32::consts::FRAC_PI_4).abs() < 0.07,
        "atan2(1, 1) = {val}, expected ~{}",
        std::f32::consts::FRAC_PI_4
    );

    // atan2(1, 2) = atan(0.5) ≈ 0.4636 — well inside polynomial range
    let val2 = eval2(&m, 2.0, 1.0);
    let expected2 = 1.0_f32.atan2(2.0);
    assert!(
        (val2 - expected2).abs() < 0.02,
        "atan2(1, 2) = {val2}, expected ~{expected2}"
    );
}

#[test]
#[cfg(not(target_feature = "avx512f"))] // transcendentals: not in AVX-512 Stage-1 op set
fn macro_atan2_quadrants() {
    let m = kernel!(|| Y.atan2(X));

    // Use ratio = 0.5 (well inside polynomial range) for quadrant tests
    // atan2(1, 2): Q1 — atan(0.5) ≈ 0.4636
    let q1 = eval2(&m, 2.0, 1.0);
    let expected_q1 = 1.0_f32.atan2(2.0);
    assert!(
        (q1 - expected_q1).abs() < 0.02,
        "Q1: atan2(1, 2) = {q1}, expected ~{expected_q1}"
    );

    // atan2(1, -2): Q2 — π - atan(0.5) ≈ 2.678
    let q2 = eval2(&m, -2.0, 1.0);
    let expected_q2 = 1.0_f32.atan2(-2.0);
    assert!(
        (q2 - expected_q2).abs() < 0.07,
        "Q2: atan2(1, -2) = {q2}, expected ~{expected_q2}"
    );

    // atan2(-1, -2): Q3 — -(π - atan(0.5)) ≈ -2.678
    let q3 = eval2(&m, -2.0, -1.0);
    let expected_q3 = (-1.0_f32).atan2(-2.0);
    assert!(
        (q3 - expected_q3).abs() < 0.07,
        "Q3: atan2(-1, -2) = {q3}, expected ~{expected_q3}"
    );

    // atan2(-1, 2): Q4 — -atan(0.5) ≈ -0.4636
    let q4 = eval2(&m, 2.0, -1.0);
    let expected_q4 = (-1.0_f32).atan2(2.0);
    assert!(
        (q4 - expected_q4).abs() < 0.02,
        "Q4: atan2(-1, 2) = {q4}, expected ~{expected_q4}"
    );
}

#[test]
#[cfg(not(target_feature = "avx512f"))] // transcendentals: not in AVX-512 Stage-1 op set
fn macro_atan() {
    let m = kernel!(|| X.atan());
    // atan(0.5) ≈ 0.4636 — well within polynomial range
    let val = eval1(&m, 0.5);
    let expected = 0.5_f32.atan();
    assert!(
        (val - expected).abs() < 0.02,
        "atan(0.5) = {val}, expected ~{expected}"
    );
    // atan(0) = 0
    let val0 = eval1(&m, 0.0);
    assert!(val0.abs() < 0.01, "atan(0) = {val0}, expected ~0");
}

#[test]
#[cfg(not(target_feature = "avx512f"))] // transcendentals: not in AVX-512 Stage-1 op set
fn macro_asin() {
    let m = kernel!(|| X.asin());
    // asin(0) = 0
    let val0 = eval1(&m, 0.0);
    assert!(val0.abs() < 0.01, "asin(0) = {val0}, expected ~0");
    // asin(0.5) = π/6 ≈ 0.5236 — ratio < 1, polynomial is accurate
    let val_half = eval1(&m, 0.5);
    let expected = 0.5_f32.asin();
    assert!(
        (val_half - expected).abs() < 0.02,
        "asin(0.5) = {val_half}, expected ~{expected}"
    );
}

#[test]
#[cfg(not(target_feature = "avx512f"))] // transcendentals: not in AVX-512 Stage-1 op set
fn macro_acos() {
    let m = kernel!(|| X.acos());
    // acos(0.5) = π/3 ≈ 1.047 — exercises large-ratio path (ratio ≈ 1.73)
    let val_half = eval1(&m, 0.5);
    let expected = 0.5_f32.acos();
    assert!(
        (val_half - expected).abs() < 0.05,
        "acos(0.5) = {val_half}, expected ~{expected}"
    );
    // acos(0) = π/2
    let val0 = eval1(&m, 0.0);
    assert!(
        (val0 - std::f32::consts::FRAC_PI_2).abs() < 0.07,
        "acos(0) = {val0}, expected ~π/2"
    );
}
