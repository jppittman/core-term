//! Comprehensive tests for Field and Manifold operations.

extern crate std;

use std::prelude::v1::*;

use pixelflow_core::jet::Jet2;
use pixelflow_core::{
    // Operations
    // Core types
    Field,
    Manifold,
    ManifoldExt,
    PARALLELISM,
    W,
    // Variables
    X,
    Y,
    Z,
    // Combinators
    combinators::{Fix, Map, Select},
    // Materialize
    materialize,
    scale,
    variables::Axis,
};

// ============================================================================
// Test Helpers
// ============================================================================

/// Asserts that a Field is approximately equal to a float value across all lanes.
fn assert_field_approx_eq(field: Field, expected: f32) {
    let expected_field = Field::from(expected);
    let diff = (field - expected_field).abs();
    // Loosened epsilon to 1e-2 to account for rcp/rsqrt approximations (approx 12-bit precision)
    // especially when scaled by larger inputs (e.g. scale combinator).
    let is_small = diff.lt(Field::from(1e-2));

    // Evaluate at dummy coordinates (Field ignores them)
    let zero = Field::from(0.0);
    let coords = (zero, zero, zero, zero);
    let mask = is_small.eval(coords);

    if !mask.all() {
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

        let mut actual_values = Vec::new();
        for i in 0..PARALLELISM {
            actual_values.push(out[i * 4]);
        }

        panic!("Field assertion failed.\nExpected: approx {}\nActual (lanes): {:?}", expected, actual_values);
    }
}

// ============================================================================
// Field Tests
// ============================================================================

#[test]
fn field_from_f32_broadcasts_value() {
    let f: Field = 42.0f32.into();
    assert_field_approx_eq(f, 42.0);
}

#[test]
fn field_from_i32_broadcasts_value() {
    let f: Field = 42i32.into();
    assert_field_approx_eq(f, 42.0);
}

#[test]
fn field_arithmetic_computes_basic_ops() {
    let a: Field = 2.0f32.into();
    let b: Field = 3.0f32.into();
    let zero = Field::from(0.0);
    let coords = (zero, zero, zero, zero);

    assert_field_approx_eq((a + b).eval(coords), 5.0);
    assert_field_approx_eq((a - b).eval(coords), -1.0);
    assert_field_approx_eq((a * b).eval(coords), 6.0);
    assert_field_approx_eq((a / b).eval(coords), 2.0 / 3.0);
}

#[test]
fn field_bitwise_computes_correctly() {
    let a: Field = 1.0f32.into(); // 0x3f800000
    let b: Field = 2.0f32.into(); // 0x40000000

    // 1.0 & 2.0 -> 0.0 (bitwise representation mismatch)
    assert_field_approx_eq(a & b, 0.0);
}

#[test]
fn field_min_max_selects_extremes() {
    let a: Field = 5.0f32.into();
    let b: Field = 3.0f32.into();
    let zero = Field::from(0.0);
    let coords = (zero, zero, zero, zero);
    assert_field_approx_eq(a.min(b).eval(coords), 3.0);
    assert_field_approx_eq(a.max(b).eval(coords), 5.0);
}

#[test]
fn field_sqrt_returns_zero_for_zero() {
    let expr = Field::from(0.0).sqrt();
    let zero = Field::from(0.0);
    let coords = (zero, zero, zero, zero);

    assert_field_approx_eq(expr.eval(coords), 0.0);
}

#[test]
fn field_sqrt_returns_zero_for_negative() {
    // Current implementation defines sqrt(x < 0) as 0.0
    let expr = Field::from(-1.0).sqrt();
    let zero = Field::from(0.0);
    let coords = (zero, zero, zero, zero);

    assert_field_approx_eq(expr.eval(coords), 0.0);
}

#[test]
fn field_rsqrt_should_approximate_inverse_sqrt_with_high_precision() {
    // rsqrt(4.0) = 0.5
    // rsqrt(2.0) = ~0.70710678
    // With just SIMD rsqrt, precision is ~12 bits (~2e-4 error).
    // With Newton-Raphson, precision should be near f32 epsilon (~1e-7).
    // We check for < 1e-6 error to kill the mutant (missing refinement).

    let x_vals = [4.0f32, 2.0, 0.5, 100.0];

    for x in x_vals {
        let field_x = Field::from(x);
        let zero = Field::from(0.0);
        let coords = (zero, zero, zero, zero);

        // ManifoldExt::rsqrt returns an AST (Rsqrt<Field>), so we evaluate it.
        let rsqrt_ast = field_x.rsqrt();
        let result = rsqrt_ast.eval(coords);

        let expected = 1.0 / x.sqrt();
        let diff = (result - Field::from(expected)).abs();

        let is_accurate = diff.lt(Field::from(1e-5)); // 1e-5 requires refinement

        let mask = is_accurate.eval(coords);

        if !mask.all() {
             // Materialize to see the value
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
            let m = Wrapper(result);
            let mut out = vec![0.0f32; PARALLELISM * 4];
            materialize(&m, 0.0, 0.0, &mut out);

            panic!("rsqrt({}) failed high precision check.\nExpected: {}\nActual: {}\nDiff: {}",
                x, expected, out[0], (out[0] - expected).abs());
        }
    }
}


// ============================================================================
// Variable Tests
// ============================================================================

#[test]
fn coordinate_variables_evaluate_to_inputs() {
    let coords = (
        Field::from(5.0),
        Field::from(3.0),
        Field::from(1.0),
        Field::from(7.0),
    );

    let x_res = X.eval(coords);
    let y_res = Y.eval(coords);
    let z_res = Z.eval(coords);
    let w_res = W.eval(coords);

    assert_field_approx_eq(x_res, 5.0);
    assert_field_approx_eq(y_res, 3.0);
    assert_field_approx_eq(z_res, 1.0);
    assert_field_approx_eq(w_res, 7.0);
}

// ============================================================================
// Combinator Tests
// ============================================================================

#[test]
fn scale_combinator_scales_coordinates() {
    // scale(X, 2.0) evals X at x/2.0
    let scaled = scale(X, 2.0);
    let x = Field::from(10.0);
    let zero = Field::from(0.0);
    let coords = (x, zero, zero, zero);

    assert_field_approx_eq(scaled.eval(coords), 5.0);
}

#[test]
fn select_chooses_branch_based_on_condition() {
    let coords_pos = (Field::from(5.0), Field::from(10.0), Field::from(20.0), Field::from(0.0));
    let coords_neg = (Field::from(-1.0), Field::from(10.0), Field::from(20.0), Field::from(0.0));

    let sel_pos = Select {
        cond: X.gt(0.0f32),
        if_true: Y,
        if_false: Z,
    };

    // 5.0 > 0 -> True -> Y (10.0)
    assert_field_approx_eq(sel_pos.eval(coords_pos), 10.0);

    // -1.0 > 0 -> False -> Z (20.0)
    assert_field_approx_eq(sel_pos.eval(coords_neg), 20.0);
}

#[test]
fn select_comparisons_respect_equality_boundary() {
    let zero = Field::from(0.0);
    let coords = (zero, zero, zero, zero);

    // Gt: 0 > 0 -> False
    let sel_gt = Select {
        cond: X.gt(0.0f32),
        if_true: Field::from(1.0),
        if_false: Field::from(2.0),
    };
    assert_field_approx_eq(sel_gt.eval(coords), 2.0);

    // Ge: 0 >= 0 -> True
    let sel_ge = Select {
        cond: X.ge(0.0f32),
        if_true: Field::from(1.0),
        if_false: Field::from(2.0),
    };
    assert_field_approx_eq(sel_ge.eval(coords), 1.0);
}

#[test]
fn map_transforms_coordinates() {
    // Substitute X with X+X
    let doubled = Map::new(X, X + X);
    let x = Field::from(5.0);
    let zero = Field::from(0.0);
    let coords = (x, zero, zero, zero);

    // Input x=5, Map transforms coords to x=10, then evals X (which returns current x)
    assert_field_approx_eq(doubled.eval(coords), 10.0);
}

#[test]
fn fix_combinator_iterates_until_done() {
    // Iterate: start at 0, add 1 each step, stop at 5
    let fix = Fix {
        seed: 0.0f32,
        step: W + 1.0f32,
        done: W.ge(5.0f32),
    };
    let zero = Field::from(0.0);
    let coords = (zero, zero, zero, zero);
    assert_field_approx_eq(fix.eval(coords), 5.0);
}

// ============================================================================
// Jet2 Tests
// ============================================================================

#[test]
fn jet2_computes_correct_derivatives() {
    // Test x^2 + y
    let expr = X * X + Y;

    let x_jet = Jet2::x(Field::from(5.0));
    let y_jet = Jet2::y(Field::from(3.0));
    let zero = Jet2::constant(Field::from(0.0));
    let coords = (x_jet, y_jet, zero, zero);

    let result = expr.eval(coords);

    assert_field_approx_eq(result.val, 28.0);
    assert_field_approx_eq(result.dx, 10.0); // 2x
    assert_field_approx_eq(result.dy, 1.0);  // 1
}

// ============================================================================
// Logic Tests
// ============================================================================

#[test]
fn bnot_inverts_logic() {
    use pixelflow_core::ops::logic::BNot;

    let not_x = BNot(X.gt(0.0f32));

    let x = Field::from(5.0);
    let zero = Field::from(0.0);
    let coords = (x, zero, zero, zero);

    // 5.0 > 0.0 is True. Not True is False (0.0).
    let result = not_x.eval(coords);
    assert_field_approx_eq(result, 0.0);

    // Try false case
    let x_neg = Field::from(-5.0);
    let coords_neg = (x_neg, zero, zero, zero);

    // -5 > 0 is False. Not False is True.
    // Use select to verify mask logic: Select checks for truthiness (non-zero)
    let sel = Select { cond: not_x, if_true: Field::from(1.0), if_false: Field::from(0.0) };
    assert_field_approx_eq(sel.eval(coords_neg), 1.0);
}
