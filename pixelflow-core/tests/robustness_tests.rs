//! Robustness and regression tests for pixelflow-core.
//!
//! These tests target specific edge cases and optimization logic to ensure
//! correctness and prevent regressions (mutant killing).

extern crate std;
use std::prelude::v1::*;

use pixelflow_core::{
    Field,
    Manifold,
    ManifoldCompat,
    ManifoldExt,
    Select,
    PARALLELISM,
};

// ============================================================================
// Field Robustness Tests
// ============================================================================

#[test]
fn field_sqrt_zero_should_be_zero() {
    // Regression test for sqrt(0) -> NaN if not handled carefully
    // when using rsqrt * self.
    // .sqrt() constructs an AST node Sqrt<Field>
    let expr = Field::from(0.0).sqrt();

    // Evaluate to get a Field result
    let zero = Field::from(0.0);
    let val = expr.eval_raw(zero, zero, zero, zero);

    // Check values
    let epsilon = Field::from(1e-6);
    let diff = (val - Field::from(0.0)).abs();

    // Evaluate comparison
    let is_small = diff.lt(epsilon);
    let mask = is_small.eval_raw(zero, zero, zero, zero);

    assert!(mask.all(), "sqrt(0.0) must be 0.0, got {:?}", val);
}

#[test]
fn field_sqrt_negative_should_be_zero() {
    // Based on implementation, sqrt(negative) returns 0 via select.
    let expr = Field::from(-1.0).sqrt();

    let zero = Field::from(0.0);
    let val = expr.eval_raw(zero, zero, zero, zero);

    let epsilon = Field::from(1e-6);
    let diff = (val - Field::from(0.0)).abs();

    let is_small = diff.lt(epsilon);
    let mask = is_small.eval_raw(zero, zero, zero, zero);

    assert!(mask.all(), "sqrt(-1.0) must be 0.0, got {:?}", val);
}

// ============================================================================
// Select Short-Circuit Tests
// ============================================================================

#[derive(Clone, Copy)]
struct Panics;

type Field4 = (Field, Field, Field, Field);

impl Manifold<Field4> for Panics {
    type Output = Field;
    fn eval(&self, _p: Field4) -> Field {
        panic!("Manifold evaluated when it should have been short-circuited!");
    }
}

#[derive(Clone, Copy)]
struct Safe(f32);

impl Manifold<Field4> for Safe {
    type Output = Field;
    fn eval(&self, _p: Field4) -> Field {
        Field::from(self.0)
    }
}

#[test]
fn select_should_short_circuit_true_branch() {
    let select = Select {
        cond: pixelflow_core::X.gt(pixelflow_core::X), // Always false (x > x is false)
        if_true: Panics,
        if_false: Safe(42.0),
    };

    let zero = Field::from(0.0);
    // Should return 42.0 and NOT panic
    let result = select.eval_raw(zero, zero, zero, zero);

    let epsilon = Field::from(1e-6);
    let diff = (result - Field::from(42.0)).abs();
    let is_small = diff.lt(epsilon);
    let mask = is_small.eval_raw(zero, zero, zero, zero);

    assert!(mask.all());
}

#[test]
fn select_should_short_circuit_false_branch() {
    let select = Select {
        cond: pixelflow_core::X.ge(pixelflow_core::X), // Always true (x >= x is true)
        if_true: Safe(42.0),
        if_false: Panics,
    };

    let zero = Field::from(0.0);
    // Should return 42.0 and NOT panic
    let result = select.eval_raw(zero, zero, zero, zero);

    let epsilon = Field::from(1e-6);
    let diff = (result - Field::from(42.0)).abs();
    let is_small = diff.lt(epsilon);
    let mask = is_small.eval_raw(zero, zero, zero, zero);

    assert!(mask.all());
}

#[test]
fn select_should_blend_mixed_mask() {
    // Construct a mixed mask if PARALLELISM > 1
    if PARALLELISM < 2 {
        return; // Can't test mixed mask on scalar backend
    }

    // We can't eval select directly into components.
    // Use materialize trick.

    use pixelflow_core::materialize;
    use pixelflow_core::ops::Vector;
    use pixelflow_core::variables::Axis;

    #[derive(Clone, Copy)]
    struct Vec1(Field);
    impl Vector for Vec1 {
        type Component = Field;
        fn get(&self, _: Axis) -> Field { self.0 }
    }

    impl Manifold<Field4> for Vec1 {
        type Output = Vec1;
        fn eval(&self, p: Field4) -> Vec1 {
            // Evaluate the select
            let s = Select {
                cond: pixelflow_core::X.lt(1.0f32),
                if_true: Safe(10.0),
                if_false: Safe(20.0),
            };
            Vec1(s.eval(p))
        }
    }

    let m = Vec1(Field::from(0.0)); // Dummy, eval creates fresh
    let mut out = vec![0.0f32; PARALLELISM * 4];
    materialize(&m, 0.0, 0.0, &mut out);

    // Out is interleaved RGBA (XXXX).
    let lane0 = out[0];
    let lane1 = out[4]; // Next pixel starts at index 4 (R channel)

    assert!((lane0 - 10.0).abs() < 1e-5, "Lane 0 (0.0 < 1.0) should be 10.0, got {}", lane0);
    assert!((lane1 - 20.0).abs() < 1e-5, "Lane 1 (1.0 < 1.0 false) should be 20.0, got {}", lane1);
}
