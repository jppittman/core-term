use pixelflow_core::{Field, Manifold, ManifoldExt, ManifoldExpr, materialize, PARALLELISM, variables::Axis};
use pixelflow_core::combinators::spherical::SphericalHarmonic;

// Helper to extract the first lane
fn eval_scalar<M>(m: M, x: f32, y: f32, z: f32, w: f32) -> f32
where M: Manifold<(Field, Field, Field, Field), Output = Field>
{
    let fx = Field::from(x);
    let fy = Field::from(y);
    let fz = Field::from(z);
    let fw = Field::from(w);

    let res = m.eval((fx, fy, fz, fw));

    #[derive(Clone, Copy)]
    struct Wrapper(Field);
    impl pixelflow_core::ops::Vector for Wrapper {
        type Component = Field;
        fn get(&self, _axis: Axis) -> Field { self.0 }
    }

    struct ConstManifold(Wrapper);
    impl Manifold<(Field, Field, Field, Field)> for ConstManifold {
        type Output = Wrapper;
        fn eval(&self, _: (Field, Field, Field, Field)) -> Wrapper { self.0 }
    }

    let wrapper = Wrapper(res);
    let cm = ConstManifold(wrapper);

    let mut out = vec![0.0f32; PARALLELISM * 4];
    materialize(&cm, 0.0, 0.0, &mut out);

    out[0]
}

#[test]
fn log2_should_return_inf_or_nan_for_non_positive() {
    // log2(0) -> -inf
    #[derive(Clone, Copy)]
    struct ConstField(f32);
    impl Manifold<(Field, Field, Field, Field)> for ConstField {
        type Output = Field;
        fn eval(&self, _: (Field, Field, Field, Field)) -> Field { Field::from(self.0) }
    }
    impl ManifoldExpr for ConstField {}

    let zero = ConstField(0.0).log2();
    let val_zero = eval_scalar(zero, 0.0, 0.0, 0.0, 0.0);
    assert_eq!(val_zero, f32::NEG_INFINITY, "log2(0) should be -inf, got {}", val_zero);

    // log2(-1) -> NaN
    let neg = ConstField(-1.0).log2();
    let val_neg = eval_scalar(neg, 0.0, 0.0, 0.0, 0.0);
    assert!(val_neg.is_nan(), "log2(-1) should be NaN, got {}", val_neg);
}

#[test]
fn spherical_harmonic_should_handle_north_pole() {
    // Y_0^0 at (0, 0, 1) - North Pole
    // This triggers atan2(0, 0) inside
    // (x,y,z) = (0, 0, 1)
    let sh = SphericalHarmonic::<0, 0>;
    let val = eval_scalar(sh, 0.0, 0.0, 1.0, 0.0);

    println!("SH at (0,0,1) = {}", val);
    assert!(!val.is_nan(), "SphericalHarmonic at North Pole returned NaN");
    // Value should be approx 0.282
    assert!((val - 0.2820948).abs() < 1e-4, "Expected ~0.282, got {}", val);
}
