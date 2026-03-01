//! Test that kernel! macro compiles with 5+ parameters using WithContext.
//!
//! These tests verify that the macro correctly handles large parameter counts
//! by evaluating the resulting kernels.

use pixelflow_core::{Field, Manifold};
use pixelflow_macros::kernel;
use pixelflow_core::{PARALLELISM, materialize, variables::Axis};

fn assert_val(field: Field, expected: f32) {
    #[derive(Clone, Copy)]
    struct Wrapper(Field);
    impl pixelflow_core::ops::Vector for Wrapper {
        type Component = Field;
        fn get(&self, _axis: Axis) -> Field { self.0 }
    }
    impl Manifold for Wrapper {
        type Output = Wrapper;
        fn eval(&self, _: (Field, Field, Field, Field)) -> Wrapper { *self }
    }

    let m = Wrapper(field);
    let mut out = vec![0.0f32; PARALLELISM * 4];
    materialize(&m, 0.0, 0.0, &mut out);

    // Check first lane
    let actual = out[0];
    assert!((actual - expected).abs() < 1e-4, "Expected {}, got {}", expected, actual);
}

#[test]
fn kernel_should_sum_5_params_correctly() {
    let k = kernel!(|a: f32, b: f32, c: f32, d: f32, e: f32| { a + b + c + d + e });
    let instance = k(1.0, 2.0, 3.0, 4.0, 5.0);

    let zero = Field::from(0.0);
    let res = instance.eval((zero, zero, zero, zero));
    assert_val(res, 15.0);
}

#[test]
fn kernel_should_sum_6_params_correctly() {
    let k = kernel!(|a: f32, b: f32, c: f32, d: f32, e: f32, f: f32| { a + b + c + d + e + f });
    let instance = k(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);

    let zero = Field::from(0.0);
    let res = instance.eval((zero, zero, zero, zero));
    assert_val(res, 21.0);
}

#[test]
fn kernel_should_sum_7_params_correctly() {
    let k = kernel!(|a: f32, b: f32, c: f32, d: f32, e: f32, f: f32, g: f32| {
        a + b + c + d + e + f + g
    });
    let instance = k(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0);

    let zero = Field::from(0.0);
    let res = instance.eval((zero, zero, zero, zero));
    assert_val(res, 28.0);
}

#[test]
fn kernel_should_sum_8_params_correctly() {
    let k = kernel!(
        |a: f32, b: f32, c: f32, d: f32, e: f32, f: f32, g: f32, h: f32| {
            a + b + c + d + e + f + g + h
        }
    );
    let instance = k(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);

    let zero = Field::from(0.0);
    let res = instance.eval((zero, zero, zero, zero));
    assert_val(res, 36.0);
}

#[test]
fn jet_kernel_should_compute_derivative_correctly() {
    use pixelflow_core::{Field, Manifold, jet::Jet3};
    use pixelflow_macros::kernel;

    type Jet3_4 = (Jet3, Jet3, Jet3, Jet3);

    // k(h) = h / Y
    let k = kernel!(|h: f32| -> Jet3 { h / Y });
    let f = k(6.0);

    // Evaluate at Y=2.0 (with dy=1.0)
    let y_jet = Jet3::y(Field::from(2.0)); // val=2, dy=1
    let zero = Jet3::constant(Field::from(0.0));

    let p: Jet3_4 = (zero, y_jet, zero, zero);

    let result = f.eval(p);

    // Result = 6.0 / Y
    // val = 6 / 2 = 3
    // d/dy (6/y) = -6/y^2 = -6/4 = -1.5

    assert_val(result.val, 3.0);
    assert_val(result.dy, -1.5);
}
