use pixelflow_core::{Field, Manifold};
use pixelflow_core::spherical::{sh2_basis_at, cosine_lobe_sh2, sh2_multiply, Sh2};
use pixelflow_core::materialize;
use pixelflow_core::ops::Vector;
use pixelflow_core::variables::Axis;

const EPSILON: f32 = 1e-3;

#[derive(Clone, Copy)]
struct Wrapper(Field);

// Manifold needs to be implemented for (Field, Field, Field, Field)
impl Manifold<(Field, Field, Field, Field)> for Wrapper {
    type Output = Wrapper;
    fn eval(&self, _: (Field, Field, Field, Field)) -> Self::Output {
        *self
    }
}

impl Vector for Wrapper {
    type Component = Field;
    fn get(&self, _axis: Axis) -> Field {
        self.0
    }
}

/// Helper to extract the first lane of a Field using the public API.
fn get_first(f: Field) -> f32 {
    let w = Wrapper(f);
    // Buffer for 4 components * PARALLELISM lanes
    // materialize writes interleaved RGBA
    let mut buf = vec![0.0f32; pixelflow_core::PARALLELISM * 4];
    // Evaluate at origin (coordinates don't matter for Wrapper)
    materialize(&w, 0.0, 0.0, &mut buf);
    buf[0]
}

#[test]
fn sh2_basis_should_match_analytical_values_at_axes() {
    // Check Y_l^m values at specific points on the unit sphere.
    // Based on standard Real Spherical Harmonics definitions.
    // Note: The code uses a specific normalization and phase convention.
    // We reverse engineer the expected values from the SH_NORM table
    // and standard formulas to verify consistency.

    // Basis order in sh2_basis_at:
    // 0: Y00
    // 1: Y1-1 (y)
    // 2: Y10  (z)
    // 3: Y11  (x)
    // 4: Y2-2 (xy)
    // 5: Y2-1 (yz)
    // 6: Y20  (3z^2-1)
    // 7: Y21  (xz)
    // 8: Y22  (x^2-y^2)

    // Direction: +Z (0, 0, 1)
    // x=0, y=0, z=1
    // Y00 = 0.282
    // Y1-1 = 0
    // Y10 = 0.488 * 1 = 0.488
    // Y11 = 0
    // Y2-2 = 0
    // Y2-1 = 0
    // Y20 = 0.315 * (3*1 - 1) = 0.630
    // Y21 = 0
    // Y22 = 0.546 * (0 - 0) = 0

    let dir_z = (Field::from(0.0), Field::from(0.0), Field::from(1.0));
    let basis_z = sh2_basis_at(dir_z);

    let vals_z: Vec<f32> = basis_z.iter().map(|f| get_first(*f)).collect();

    assert!((vals_z[0] - 0.28209).abs() < EPSILON, "Y00 at Z. Got {}", vals_z[0]);
    assert!((vals_z[2] - 0.48860).abs() < EPSILON, "Y10 at Z. Got {}", vals_z[2]);
    assert!((vals_z[6] - 0.63078).abs() < EPSILON, "Y20 at Z. Got {}", vals_z[6]); // 0.31539 * 2

    // Direction: +X (1, 0, 0)
    // x=1, y=0, z=0
    // Y11 = 0.488 * 1 = 0.488
    // Y22 = 0.546 * (1 - 0) = 0.546
    // Y20 = 0.315 * (0 - 1) = -0.315
    let dir_x = (Field::from(1.0), Field::from(0.0), Field::from(0.0));
    let basis_x = sh2_basis_at(dir_x);
    let vals_x: Vec<f32> = basis_x.iter().map(|f| get_first(*f)).collect();

    assert!((vals_x[3] - 0.48860).abs() < EPSILON, "Y11 at X. Got {}", vals_x[3]);
    assert!((vals_x[8] - 0.54627).abs() < EPSILON, "Y22 at X. Got {}", vals_x[8]);
    assert!((vals_x[6] - (-0.31539)).abs() < EPSILON, "Y20 at X. Got {}", vals_x[6]);

    // Direction: Diagonal (1/sqrt(2), 1/sqrt(2), 0)
    // x = y = 0.7071
    // xy = 0.5
    // Y2-2 (xy term) is the one we suspect is buggy.
    // Expected: 1.0925 * xy = 1.0925 * 0.5 = 0.5462
    // If bug (0.546 coefficient): 0.546 * 0.5 = 0.273

    let inv_sqrt_2 = 1.0 / 2.0f32.sqrt();
    let dir_xy = (Field::from(inv_sqrt_2), Field::from(inv_sqrt_2), Field::from(0.0));
    let basis_xy = sh2_basis_at(dir_xy);
    let vals_xy: Vec<f32> = basis_xy.iter().map(|f| get_first(*f)).collect();

    // Index 4 is Y2-2
    assert!((vals_xy[4] - 0.54627).abs() < EPSILON, "Y2-2 (xy) at diagonal. Got {}, expected ~0.54627", vals_xy[4]);
}

#[test]
fn cosine_lobe_coefficients_should_match_expected_values() {
    // Normal +Z
    let n = (Field::from(0.0), Field::from(0.0), Field::from(1.0));
    let sh = cosine_lobe_sh2(n);

    // L0: 0.886
    // L1: z term -> 1.023 * 1 = 1.023 (Y10)
    // Others 0
    let l0 = get_first(sh.coeffs[0]);
    let y10 = get_first(sh.coeffs[2]); // Y10 is index 2

    assert!((l0 - 0.8862).abs() < EPSILON, "L0 mismatch");
    assert!((y10 - 1.0233).abs() < EPSILON, "L1 (z) mismatch");
}

#[test]
fn sh2_multiply_should_preserve_identity_when_multiplying_by_one() {
    // Multiply DC * DC
    // Y00 = 0.282
    // Y00 * Y00 = 0.282 * 0.282 = 0.0795
    // Result should be pure DC component of 0.0795?
    // No, product of functions.
    // Constant function 1.0 represented as Y00 * sqrt(4pi) = Y00 * 3.54
    // Actually, let's just test that multiplying by Identity (all 1s) works?
    // Or simpler: Square a function.

    // Let's use the code's own multiplication table to check consistency.
    // If we have a function f = 1 (constant everywhere).
    // Coeffs: c0 = 2*sqrt(pi) = 3.5449. No, Y00 is 1/2sqrt(pi).
    // To get 1.0 everywhere: 1.0 = c0 * Y00 -> c0 = 1.0 / 0.28209 = 3.5449.

    let c0 = 3.5449077;
    let one = Sh2 { coeffs: [c0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] };

    // 1 * 1 = 1
    let res = sh2_multiply(&one, &one);

    assert!((res.coeffs[0] - c0).abs() < 0.01, "1*1 should be 1. Got {}", res.coeffs[0]);

    // Check that other coeffs are near zero
    for i in 1..9 {
        assert!(res.coeffs[i].abs() < 0.01, "Coeff {} should be 0, got {}", i, res.coeffs[i]);
    }
}
