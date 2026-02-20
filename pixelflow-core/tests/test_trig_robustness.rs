use pixelflow_core::{Field, Manifold};
use pixelflow_core::combinators::spherical::SphericalHarmonic;

/// Helper to evaluate a manifold and extract the first f32 value
fn eval_to_f32<M: Manifold<(Field, Field, Field, Field), Output = Field>>(m: M, x: f32, y: f32, z: f32) -> f32 {
    let fx = Field::from(x);
    let fy = Field::from(y);
    let fz = Field::from(z);
    let fw = Field::from(0.0);
    let coords = (fx, fy, fz, fw);
    let result = m.eval(coords);
    // Field is a SIMD type - extract first lane
    unsafe { *(&result as *const Field as *const f32) }
}

#[test]
fn spherical_harmonic_should_be_zero_at_north_pole() {
    // Y_1^1 at (0, 0, 1) should be 0.0 (or very close, or at least not NaN)
    let sh = SphericalHarmonic::<1, 1>;
    let val = eval_to_f32(sh, 0.0, 0.0, 1.0);
    assert!(!val.is_nan(), "SphericalHarmonic<1, 1> at North Pole is NaN");
    // Tolerance relaxed to 1e-3 because sin(acos(x)) near x=1 amplifies precision errors.
    // Error ~ sqrt(epsilon) -> sqrt(1e-7) ~ 3e-4.
    assert!((val - 0.0).abs() < 1e-3, "SphericalHarmonic<1, 1> at North Pole should be 0.0, got {}", val);
}

#[test]
fn spherical_harmonic_should_be_zero_at_south_pole() {
    // Y_1^1 at (0, 0, -1) should be 0.0
    let sh = SphericalHarmonic::<1, 1>;
    let val = eval_to_f32(sh, 0.0, 0.0, -1.0);
    assert!(!val.is_nan(), "SphericalHarmonic<1, 1> at South Pole is NaN");
    // Tolerance relaxed to 1e-3.
    assert!((val - 0.0).abs() < 1e-3, "SphericalHarmonic<1, 1> at South Pole should be 0.0, got {}", val);
}

#[test]
fn spherical_harmonic_should_be_valid_at_equator() {
    // Y_1^0 at (1, 0, 0) -> theta=PI/2, cos(theta)=0. P_1^0(0)=0. Result 0.
    let sh = SphericalHarmonic::<1, 0>;
    let val = eval_to_f32(sh, 1.0, 0.0, 0.0);
    assert!(!val.is_nan(), "SphericalHarmonic<1, 0> at Equator (1,0,0) is NaN");
    assert!((val - 0.0).abs() < 1e-5, "SphericalHarmonic<1, 0> at Equator (1,0,0) should be 0.0, got {}", val);

    // Y_1^1 at (1, 0, 0) -> theta=PI/2. P_1^1(0)=-1. phi=0. cos(phi)=1. Result K * -1 * 1 != 0.
    // K_1^1 = sqrt(3/(8pi)) * -1 approx -0.345
    // SH_NORM[1][1] is used.
    // Let's just check it's not NaN.
    let sh11 = SphericalHarmonic::<1, 1>;
    let val11 = eval_to_f32(sh11, 1.0, 0.0, 0.0);
    assert!(!val11.is_nan(), "SphericalHarmonic<1, 1> at Equator (1,0,0) is NaN");
    assert!(val11.abs() > 0.1, "SphericalHarmonic<1, 1> at Equator (1,0,0) should be non-zero");
}

#[test]
fn spherical_harmonic_should_be_nan_at_origin() {
    // At (0,0,0), normalization fails (0/0), so we expect NaN.
    // This confirms that bad inputs do produce NaN, distinguishing from the pole case which is valid.
    let sh = SphericalHarmonic::<1, 1>;
    let val = eval_to_f32(sh, 0.0, 0.0, 0.0);
    assert!(val.is_nan(), "SphericalHarmonic<1, 1> at (0,0,0) should be NaN due to normalization failure");
}
