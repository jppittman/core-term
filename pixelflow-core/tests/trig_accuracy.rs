//! Accuracy tests for Chebyshev trig functions
//!
//! Verifies that Chebyshev approximations produce correct results
//! through the public Manifold API using materialize_scalar.

use pixelflow_core::{ManifoldExt, X, Y};
use std::f32::consts::PI;

#[test]
fn test_sin_basic() {
    // Basic test: sin(0) should be ~0, sin(π/2) should be ~1
    let sin_x = X.sin();

    // sin(0) ≈ 0
    let mut zero_out = [0.0f32; 16];
    pixelflow_core::materialize_scalar(&sin_x, 0.0, 0.0, &mut zero_out);
    println!("sin(0) = {}", zero_out[0]);
    assert!(zero_out[0].abs() < 0.001, "sin(0) should be ~0, got {}", zero_out[0]);

    // sin(π/2) ≈ 1
    let mut half_pi_out = [0.0f32; 16];
    pixelflow_core::materialize_scalar(&sin_x, PI / 2.0, 0.0, &mut half_pi_out);
    println!("sin(π/2) = {}", half_pi_out[0]);
    assert!(
        (half_pi_out[0] - 1.0).abs() < 0.001,
        "sin(π/2) should be ~1, got {}",
        half_pi_out[0]
    );
}

#[test]
fn test_cos_basic() {
    // Basic test: cos(0) should be ~1, cos(π) should be ~-1
    let cos_x = X.cos();

    // cos(0) ≈ 1
    let mut zero_out = [0.0f32; 16];
    pixelflow_core::materialize_scalar(&cos_x, 0.0, 0.0, &mut zero_out);
    assert!(
        (zero_out[0] - 1.0).abs() < 0.001,
        "cos(0) should be ~1"
    );

    // cos(π) ≈ -1
    let mut pi_out = [0.0f32; 16];
    pixelflow_core::materialize_scalar(&cos_x, PI, 0.0, &mut pi_out);
    assert!(
        (pi_out[0] + 1.0).abs() < 0.001,
        "cos(π) should be ~-1"
    );
}

#[test]
fn test_atan2_basic() {
    // Basic test: atan2(1, 1) should be ~π/4
    let atan_yx = Y.atan2(X);
    let mut out = [0.0f32; 16];
    pixelflow_core::materialize_scalar(&atan_yx, 1.0, 1.0, &mut out);
    assert!(
        (out[0] - PI / 4.0).abs() < 0.01,
        "atan2(1, 1) should be ~π/4, got {}",
        out[0]
    );
}
