use pixelflow_core::{
    combinators::spherical::{
        Sh2, SphericalHarmonic, sh2_multiply,
    },
    Field, Manifold, materialize,
    variables::Axis,
    PARALLELISM,
};

/// Helper to extract scalar value from Field using materialize.
fn extract_scalar(field: Field) -> f32 {
    #[derive(Clone, Copy)]
    struct Wrapper(Field);

    impl pixelflow_core::ops::Vector for Wrapper {
        type Component = Field;
        fn get(&self, _axis: Axis) -> Field { self.0 }
    }

    // We need to implement Manifold for Wrapper to pass it to materialize
    impl Manifold for Wrapper {
        type Output = Wrapper;
        fn eval(&self, _: (Field, Field, Field, Field)) -> Wrapper { *self }
    }

    let m = Wrapper(field);
    // output buffer size must be >= PARALLELISM * 4 (since materialize outputs 4 channels)
    let mut out = vec![0.0f32; PARALLELISM * 4];
    materialize(&m, 0.0, 0.0, &mut out);

    // Return the first value (lane 0, channel 0)
    out[0]
}

/// Evaluate a Spherical Harmonic basis function Y_l^m at a direction (x, y, z).
fn eval_sh_at<const L: usize, const M: i32>(x: f32, y: f32, z: f32) -> f32 {
    let sh = SphericalHarmonic::<L, M>;
    let val = sh.eval((
        Field::from(x),
        Field::from(y),
        Field::from(z),
        Field::from(0.0),
    ));
    extract_scalar(val)
}

/// Generate N points on the sphere using Fibonacci lattice.
/// Returns a vector of (x, y, z) tuples.
fn fibonacci_sphere(n: usize) -> Vec<(f32, f32, f32)> {
    let mut points = Vec::with_capacity(n);
    let phi = (1.0 + 5.0f32.sqrt()) / 2.0; // Golden ratio

    for i in 0..n {
        let i_f = i as f32;
        let n_f = n as f32;

        let z = 1.0 - (2.0 * i_f + 1.0) / n_f; // z goes from 1 to -1
        let radius = (1.0 - z * z).sqrt();

        let theta = 2.0 * std::f32::consts::PI * i_f / phi;

        let x = radius * theta.cos();
        let y = radius * theta.sin();

        points.push((x, y, z));
    }
    points
}

#[test]
fn test_sh_basis_values_at_cardinal_points() {
    // Y_0^0 is constant 1/(2sqrt(pi)) approx 0.28209
    let expected_y00 = 0.28209479;

    assert!((eval_sh_at::<0, 0>(1.0, 0.0, 0.0) - expected_y00).abs() < 1e-4, "Y_0^0 at X");
    assert!((eval_sh_at::<0, 0>(0.0, 1.0, 0.0) - expected_y00).abs() < 1e-4, "Y_0^0 at Y");
    assert!((eval_sh_at::<0, 0>(0.0, 0.0, 1.0) - expected_y00).abs() < 1e-4, "Y_0^0 at Z");

    // Y_1^0 (z) is proportional to z. Should be max at Z, 0 at X, Y.
    // K_1^0 = sqrt(3/(4pi)) approx 0.4886
    let k10 = 0.4886025;

    let val_z = eval_sh_at::<1, 0>(0.0, 0.0, 1.0);
    assert!((val_z - k10).abs() < 1e-3, "Y_1^0 at Z (1): expected {}, got {}", k10, val_z);

    let val_neg_z = eval_sh_at::<1, 0>(0.0, 0.0, -1.0);
    assert!((val_neg_z - (-k10)).abs() < 1e-3, "Y_1^0 at -Z (-1): expected {}, got {}", -k10, val_neg_z);
    assert!((eval_sh_at::<1, 0>(1.0, 0.0, 0.0)).abs() < 1e-4, "Y_1^0 at X (0)");
    assert!((eval_sh_at::<1, 0>(0.0, 1.0, 0.0)).abs() < 1e-4, "Y_1^0 at Y (0)");

    // Y_1^-1 (y) is proportional to y.
    let v_y = eval_sh_at::<1, -1>(0.0, 1.0, 0.0);
    assert!((v_y - k10).abs() < 1e-3, "Y_1^-1 at Y (1): expected {}, got {}", k10, v_y);
    let v_x = eval_sh_at::<1, -1>(1.0, 0.0, 0.0);
    assert!(v_x.abs() < 1e-3, "Y_1^-1 at X (0): expected 0, got {}", v_x);

    // Y_1^1 (x) is proportional to x.
    let v_x1 = eval_sh_at::<1, 1>(1.0, 0.0, 0.0);
    assert!((v_x1 - k10).abs() < 1e-3, "Y_1^1 at X (1): expected {}, got {}", k10, v_x1);
    let v_y1 = eval_sh_at::<1, 1>(0.0, 1.0, 0.0);
    assert!(v_y1.abs() < 1e-3, "Y_1^1 at Y (0): expected 0, got {}", v_y1);
}

#[test]
fn test_sh_orthonormality_fibonacci() {
    let n_samples = 1000;
    let points = fibonacci_sphere(n_samples);
    let sphere_area = 4.0 * std::f32::consts::PI;
    let weight = sphere_area / (n_samples as f32);

    // Test orthogonality between Y_0^0 and Y_1^0
    let mut sum = 0.0;
    for (x, y, z) in &points {
        let v0 = eval_sh_at::<0, 0>(*x, *y, *z);
        let v1 = eval_sh_at::<1, 0>(*x, *y, *z);
        sum += v0 * v1 * weight;
    }
    assert!(sum.abs() < 1e-2, "Orthogonality failed: <Y00, Y10> = {}", sum);

    // Test normality of Y_0^0
    let mut sum_sq = 0.0;
    for (x, y, z) in &points {
        let v0 = eval_sh_at::<0, 0>(*x, *y, *z);
        sum_sq += v0 * v0 * weight;
    }
    assert!((sum_sq - 1.0).abs() < 1e-2, "Normality failed: <Y00, Y00> = {}", sum_sq);

    // Test normality of Y_1^1
    let mut sum_sq = 0.0;
    for (x, y, z) in &points {
        let v = eval_sh_at::<1, 1>(*x, *y, *z);
        sum_sq += v * v * weight;
    }
    assert!((sum_sq - 1.0).abs() < 1e-2, "Normality failed: <Y11, Y11> = {}", sum_sq);
}

#[test]
fn test_sh_parity() {
    // Y_l^m(-r) = (-1)^l Y_l^m(r)
    // l=0: even
    // l=1: odd
    // l=2: even

    let points = fibonacci_sphere(10); // few points are enough

    for (x, y, z) in points {
        // L=0
        let v0_pos = eval_sh_at::<0, 0>(x, y, z);
        let v0_neg = eval_sh_at::<0, 0>(-x, -y, -z);
        assert!((v0_pos - v0_neg).abs() < 1e-5, "Parity failed for L=0");

        // L=1
        let v1_pos = eval_sh_at::<1, 0>(x, y, z);
        let v1_neg = eval_sh_at::<1, 0>(-x, -y, -z);
        assert!((v1_pos + v1_neg).abs() < 1e-5, "Parity failed for L=1");

        // L=2
        let v2_pos = eval_sh_at::<2, 0>(x, y, z);
        let v2_neg = eval_sh_at::<2, 0>(-x, -y, -z);
        assert!((v2_pos - v2_neg).abs() < 1e-5, "Parity failed for L=2");
    }
}

#[test]
fn test_clebsch_gordan_multiplication() {
    // Validate sh2_multiply against direct multiplication of evaluations.
    // (A * B).eval(dir) should approx equal A.eval(dir) * B.eval(dir)

    let points = fibonacci_sphere(50);

    // Create two test SH vectors
    // A: purely Zonal (L=0 + L=1 Z)
    let a = Sh2 {
        coeffs: [
            1.0, // L0
            0.0, 0.5, 0.0, // L1 (y, z, x) -> 0.5 * Y10
            0.0, 0.0, 0.0, 0.0, 0.0 // L2
        ]
    };

    // B: purely X (L=1 X)
    let b = Sh2 {
        coeffs: [
            0.5, // L0
            0.0, 0.0, 1.0, // L1 (y, z, x) -> 1.0 * Y11
            0.0, 0.0, 0.0, 0.0, 0.0 // L2
        ]
    };

    let prod = sh2_multiply(&a, &b);

    for (x, y, z) in points {
        let dir = (Field::from(x), Field::from(y), Field::from(z));

        let val_a = extract_scalar(a.eval_const(dir));
        let val_b = extract_scalar(b.eval_const(dir));
        let val_prod = extract_scalar(prod.eval_const(dir));

        let expected = val_a * val_b;

        // Tolerance is higher because product is truncated to band 2
        // Clebsch-Gordan multiplication is exact only if the result fits in the band.
        // But here we multiply L1 * L1 which results in L0 + L2, so it fits in Sh2 (band 2).
        // So expected error should be small.
        assert!((val_prod - expected).abs() < 1e-3,
            "Product failed at ({:.2},{:.2},{:.2}): expected {:.4}, got {:.4}",
            x, y, z, expected, val_prod);
    }
}
