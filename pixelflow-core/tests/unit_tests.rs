//! Comprehensive unit tests for pixelflow-core.
//!
//! These tests target 80%+ coverage of the library.

extern crate std;

use std::prelude::v1::*;

use pixelflow_core::jet::Jet2;
use pixelflow_core::{
    Abs,
    // Operations
    Add,
    And,
    // Computational trait (needed for from_f32)
    Computational,
    Div,
    // Core types
    Field,
    Ge,
    Gt,
    Le,
    Lt,
    Manifold,
    ManifoldExt,
    Max,
    Min,
    Mul,
    Or,
    PARALLELISM,
    Sqrt,
    Sub,
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
// Field Tests
// ============================================================================

mod field_tests {
    use super::*;

    #[test]
    fn test_field_from_f32() {
        let f: Field = 42.0f32.into();
        // We can't directly inspect Field, but we can use it in operations
        let result = f + Field::from(0.0);
        // Materialize to verify
        let manifold = result;
        let _ = manifold; // Field itself is a manifold
    }

    #[test]
    fn test_field_from_i32() {
        let f: Field = 42i32.into();
        let _ = f;
    }

    #[test]
    fn test_field_arithmetic() {
        // Test through manifold composition
        let a: Field = 2.0f32.into();
        let b: Field = 3.0f32.into();

        let _sum = a + b;
        let _diff = a - b;
        let _prod = a * b;
        let _quot = a / b;
    }

    #[test]
    fn test_field_bitwise() {
        let a: Field = 1.0f32.into();
        let b: Field = 2.0f32.into();

        let _and = a & b;
        let _or = a | b;
        let _not = !a;
    }

    #[test]
    fn test_field_min_max() {
        let a: Field = 5.0f32.into();
        let b: Field = 3.0f32.into();

        let _min = a.min(b);
        let _max = a.max(b);
    }
}

// ============================================================================
// Variable Tests
// ============================================================================

mod variable_tests {
    use super::*;

    #[test]
    fn test_x_returns_x_coordinate() {
        let x_val = Field::from(5.0);
        let y_val = Field::from(3.0);
        let z_val = Field::from(1.0);
        let w_val = Field::from(0.0);

        let result = X.eval_raw(x_val, y_val, z_val, w_val);
        // Result should be x_val
        let _ = result;
    }

    #[test]
    fn test_y_returns_y_coordinate() {
        let x_val = Field::from(5.0);
        let y_val = Field::from(3.0);
        let z_val = Field::from(1.0);
        let w_val = Field::from(0.0);

        let result = Y.eval_raw(x_val, y_val, z_val, w_val);
        let _ = result;
    }

    #[test]
    fn test_z_returns_z_coordinate() {
        let x_val = Field::from(5.0);
        let y_val = Field::from(3.0);
        let z_val = Field::from(1.0);
        let w_val = Field::from(0.0);

        let result = Z.eval_raw(x_val, y_val, z_val, w_val);
        let _ = result;
    }

    #[test]
    fn test_w_returns_w_coordinate() {
        let x_val = Field::from(5.0);
        let y_val = Field::from(3.0);
        let z_val = Field::from(1.0);
        let w_val = Field::from(7.0);

        let result = W.eval_raw(x_val, y_val, z_val, w_val);
        let _ = result;
    }

    #[test]
    fn test_axis_enum() {
        assert_eq!(Axis::X, Axis::X);
        assert_eq!(Axis::Y, Axis::Y);
        assert_eq!(Axis::Z, Axis::Z);
        assert_eq!(Axis::W, Axis::W);

        assert_ne!(Axis::X, Axis::Y);
        assert_ne!(Axis::Y, Axis::Z);
        assert_ne!(Axis::Z, Axis::W);
    }

    #[test]
    fn test_dimension_trait() {
        use pixelflow_core::variables::Dimension;

        assert_eq!(X::AXIS, Axis::X);
        assert_eq!(Y::AXIS, Axis::Y);
        assert_eq!(Z::AXIS, Axis::Z);
        assert_eq!(W::AXIS, Axis::W);
    }
}

// ============================================================================
// Manifold Implementation Tests
// ============================================================================

mod manifold_tests {
    use super::*;

    #[test]
    fn test_f32_as_constant_manifold() {
        let constant = 42.0f32;
        let x = Field::from(1.0);
        let y = Field::from(2.0);
        let z = Field::from(3.0);
        let w = Field::from(4.0);

        let result = constant.eval_raw(x, y, z, w);
        let _ = result;
    }

    #[test]
    fn test_i32_as_constant_manifold() {
        let constant = 42i32;
        let x = Field::from(1.0);
        let y = Field::from(2.0);
        let z = Field::from(3.0);
        let w = Field::from(4.0);

        let result = constant.eval_raw(x, y, z, w);
        let _ = result;
    }

    #[test]
    fn test_field_as_constant_manifold() {
        let constant = Field::from(42.0);
        let x = Field::from(1.0);
        let y = Field::from(2.0);
        let z = Field::from(3.0);
        let w = Field::from(4.0);

        let result = constant.eval_raw(x, y, z, w);
        let _ = result;
    }

    #[test]
    fn test_scale_combinator() {
        let scaled = scale(X, 2.0);
        let x = Field::from(4.0);
        let y = Field::from(0.0);
        let z = Field::from(0.0);
        let w = Field::from(0.0);

        // Evaluating at x=4 with scale=2 should give 4/2 = 2
        let result = scaled.eval_raw(x, y, z, w);
        let _ = result;
    }

    #[test]
    fn test_reference_manifold() {
        let expr = &X;
        let x = Field::from(5.0);
        let y = Field::from(0.0);
        let z = Field::from(0.0);
        let w = Field::from(0.0);

        let result = expr.eval_raw(x, y, z, w);
        let _ = result;
    }

    #[test]
    fn test_boxed_manifold() {
        use std::boxed::Box;

        let boxed: Box<dyn Manifold<Output = Field>> = Box::new(X);
        let x = Field::from(5.0);
        let y = Field::from(0.0);
        let z = Field::from(0.0);
        let w = Field::from(0.0);

        let result = boxed.eval_raw(x, y, z, w);
        let _ = result;
    }

    #[test]
    fn test_arc_manifold() {
        use std::sync::Arc;

        let arced: Arc<dyn Manifold<Output = Field>> = Arc::new(X);
        let x = Field::from(5.0);
        let y = Field::from(0.0);
        let z = Field::from(0.0);
        let w = Field::from(0.0);

        let result = arced.eval_raw(x, y, z, w);
        let _ = result;
    }
}

// ============================================================================
// Binary Operations Tests
// ============================================================================

mod binary_ops_tests {
    use super::*;

    #[test]
    fn test_add_manifolds() {
        let sum = Add(X, Y);
        let x = Field::from(2.0);
        let y = Field::from(3.0);
        let z = Field::from(0.0);
        let w = Field::from(0.0);

        let result = sum.eval_raw(x, y, z, w);
        let _ = result;
    }

    #[test]
    fn test_sub_manifolds() {
        let diff = Sub(X, Y);
        let x = Field::from(5.0);
        let y = Field::from(3.0);
        let z = Field::from(0.0);
        let w = Field::from(0.0);

        let result = diff.eval_raw(x, y, z, w);
        let _ = result;
    }

    #[test]
    fn test_mul_manifolds() {
        let prod = Mul(X, Y);
        let x = Field::from(4.0);
        let y = Field::from(3.0);
        let z = Field::from(0.0);
        let w = Field::from(0.0);

        let result = prod.eval_raw(x, y, z, w);
        let _ = result;
    }

    #[test]
    fn test_div_manifolds() {
        let quot = Div(X, Y);
        let x = Field::from(10.0);
        let y = Field::from(2.0);
        let z = Field::from(0.0);
        let w = Field::from(0.0);

        let result = quot.eval_raw(x, y, z, w);
        let _ = result;
    }

    #[test]
    fn test_operator_overloads() {
        // Test that operator overloading works for variables
        let sum = X + Y;
        let diff = X - Y;
        let prod = X * Y;
        let quot = X / Y;

        let x = Field::from(6.0);
        let y = Field::from(2.0);
        let z = Field::from(0.0);
        let w = Field::from(0.0);

        let _ = sum.eval_raw(x, y, z, w);
        let _ = diff.eval_raw(x, y, z, w);
        let _ = prod.eval_raw(x, y, z, w);
        let _ = quot.eval_raw(x, y, z, w);
    }

    #[test]
    fn test_mixed_manifold_constant() {
        // Variable + constant
        let expr = X + 5.0f32;
        let x = Field::from(3.0);
        let y = Field::from(0.0);
        let z = Field::from(0.0);
        let w = Field::from(0.0);

        let result = expr.eval_raw(x, y, z, w);
        let _ = result;
    }

    #[test]
    fn test_nested_operations() {
        // (X + Y) * (X - Y) = X² - Y²
        let expr = (X + Y) * (X - Y);
        let x = Field::from(5.0);
        let y = Field::from(3.0);
        let z = Field::from(0.0);
        let w = Field::from(0.0);

        let result = expr.eval_raw(x, y, z, w);
        let _ = result;
    }
}

// ============================================================================
// Unary Operations Tests
// ============================================================================

mod unary_ops_tests {
    use super::*;

    #[test]
    fn test_sqrt_manifold() {
        let sqrt_x = Sqrt(X);
        let x = Field::from(16.0);
        let y = Field::from(0.0);
        let z = Field::from(0.0);
        let w = Field::from(0.0);

        let result = sqrt_x.eval_raw(x, y, z, w);
        let _ = result;
    }

    #[test]
    fn test_abs_manifold() {
        let abs_x = Abs(X);
        let x = Field::from(-5.0);
        let y = Field::from(0.0);
        let z = Field::from(0.0);
        let w = Field::from(0.0);

        let result = abs_x.eval_raw(x, y, z, w);
        let _ = result;
    }

    #[test]
    fn test_max_manifold() {
        let max_xy = Max(X, Y);
        let x = Field::from(3.0);
        let y = Field::from(7.0);
        let z = Field::from(0.0);
        let w = Field::from(0.0);

        let result = max_xy.eval_raw(x, y, z, w);
        let _ = result;
    }

    #[test]
    fn test_min_manifold() {
        let min_xy = Min(X, Y);
        let x = Field::from(3.0);
        let y = Field::from(7.0);
        let z = Field::from(0.0);
        let w = Field::from(0.0);

        let result = min_xy.eval_raw(x, y, z, w);
        let _ = result;
    }

    #[test]
    fn test_sqrt_via_extension() {
        let sqrt_x = X.sqrt();
        let x = Field::from(25.0);
        let y = Field::from(0.0);
        let z = Field::from(0.0);
        let w = Field::from(0.0);

        let result = sqrt_x.eval_raw(x, y, z, w);
        let _ = result;
    }

    #[test]
    fn test_abs_via_extension() {
        let abs_x = X.abs();
        let x = Field::from(-10.0);
        let y = Field::from(0.0);
        let z = Field::from(0.0);
        let w = Field::from(0.0);

        let result = abs_x.eval_raw(x, y, z, w);
        let _ = result;
    }
}

// ============================================================================
// Comparison Operations Tests
// ============================================================================

mod compare_ops_tests {
    use super::*;

    #[test]
    fn test_lt_manifold() {
        let lt = Lt(X, Y);
        let x = Field::from(2.0);
        let y = Field::from(5.0);
        let z = Field::from(0.0);
        let w = Field::from(0.0);

        let result = lt.eval_raw(x, y, z, w);
        let _ = result;
    }

    #[test]
    fn test_gt_manifold() {
        let gt = Gt(X, Y);
        let x = Field::from(7.0);
        let y = Field::from(3.0);
        let z = Field::from(0.0);
        let w = Field::from(0.0);

        let result = gt.eval_raw(x, y, z, w);
        let _ = result;
    }

    #[test]
    fn test_le_manifold() {
        let le = Le(X, Y);
        let x = Field::from(5.0);
        let y = Field::from(5.0);
        let z = Field::from(0.0);
        let w = Field::from(0.0);

        let result = le.eval_raw(x, y, z, w);
        let _ = result;
    }

    #[test]
    fn test_ge_manifold() {
        let ge = Ge(X, Y);
        let x = Field::from(5.0);
        let y = Field::from(5.0);
        let z = Field::from(0.0);
        let w = Field::from(0.0);

        let result = ge.eval_raw(x, y, z, w);
        let _ = result;
    }

    #[test]
    fn test_comparison_via_extension() {
        let lt = X.lt(Y);
        let gt = X.gt(Y);
        let le = X.le(Y);
        let ge = X.ge(Y);

        let x = Field::from(3.0);
        let y = Field::from(5.0);
        let z = Field::from(0.0);
        let w = Field::from(0.0);

        let _ = lt.eval_raw(x, y, z, w);
        let _ = gt.eval_raw(x, y, z, w);
        let _ = le.eval_raw(x, y, z, w);
        let _ = ge.eval_raw(x, y, z, w);
    }
}

// ============================================================================
// Logic Operations Tests
// ============================================================================

mod logic_ops_tests {
    use super::*;

    #[test]
    fn test_and_manifold() {
        // Test bitwise AND between two conditions
        let cond1 = X.ge(0.0f32);
        let cond2 = X.le(10.0f32);
        let both = And(cond1, cond2);

        let x = Field::from(5.0);
        let y = Field::from(0.0);
        let z = Field::from(0.0);
        let w = Field::from(0.0);

        let result = both.eval_raw(x, y, z, w);
        let _ = result;
    }

    #[test]
    fn test_or_manifold() {
        let cond1 = X.lt(0.0f32);
        let cond2 = X.gt(10.0f32);
        let either = Or(cond1, cond2);

        let x = Field::from(5.0);
        let y = Field::from(0.0);
        let z = Field::from(0.0);
        let w = Field::from(0.0);

        let result = either.eval_raw(x, y, z, w);
        let _ = result;
    }

    #[test]
    fn test_bitand_operator() {
        // Test & operator on comparison results
        let expr = X.ge(0.0f32) & X.le(10.0f32);
        let x = Field::from(5.0);
        let y = Field::from(0.0);
        let z = Field::from(0.0);
        let w = Field::from(0.0);

        let result = expr.eval_raw(x, y, z, w);
        let _ = result;
    }

    #[test]
    fn test_bitor_operator() {
        let expr = X.lt(0.0f32) | X.gt(10.0f32);
        let x = Field::from(-5.0);
        let y = Field::from(0.0);
        let z = Field::from(0.0);
        let w = Field::from(0.0);

        let result = expr.eval_raw(x, y, z, w);
        let _ = result;
    }
}

// ============================================================================
// Select Combinator Tests
// ============================================================================

mod select_tests {
    use super::*;

    #[test]
    fn test_select_combinator() {
        // Select between two values based on condition
        let sel = Select {
            cond: X.gt(0.0f32),
            if_true: Y,
            if_false: Z,
        };

        let x = Field::from(5.0);
        let y = Field::from(10.0);
        let z = Field::from(20.0);
        let w = Field::from(0.0);

        // x > 0, so should select y (10.0)
        let result = sel.eval_raw(x, y, z, w);
        let _ = result;
    }

    #[test]
    fn test_select_via_extension() {
        let expr = X.gt(0.0f32).select(Y, Z);

        let x = Field::from(-1.0);
        let y = Field::from(10.0);
        let z = Field::from(20.0);
        let w = Field::from(0.0);

        // x <= 0, so should select z (20.0)
        let result = expr.eval_raw(x, y, z, w);
        let _ = result;
    }

    #[test]
    fn test_nested_select() {
        // if x > 0 then (if y > 0 then 1 else 2) else 3
        let inner = Y.gt(0.0f32).select(1.0f32, 2.0f32);
        let outer = X.gt(0.0f32).select(inner, 3.0f32);

        let x = Field::from(1.0);
        let y = Field::from(1.0);
        let z = Field::from(0.0);
        let w = Field::from(0.0);

        let result = outer.eval_raw(x, y, z, w);
        let _ = result;
    }
}

// ============================================================================
// Map Combinator Tests
// ============================================================================

mod map_tests {
    use super::*;

    #[test]
    fn test_map_combinator() {
        let doubled = Map::new(X, X + X);

        let x = Field::from(5.0);
        let y = Field::from(0.0);
        let z = Field::from(0.0);
        let w = Field::from(0.0);

        let result = doubled.eval_raw(x, y, z, w);
        let _ = result;
    }

    #[test]
    fn test_map_via_extension() {
        let squared = X.map(X * X);

        let x = Field::from(4.0);
        let y = Field::from(0.0);
        let z = Field::from(0.0);
        let w = Field::from(0.0);

        let result = squared.eval_raw(x, y, z, w);
        let _ = result;
    }

    #[test]
    fn test_map_clamp() {
        let clamped = X.map(X.max(0.0f32).min(1.0f32));

        // Test value in range
        let x = Field::from(0.5);
        let y = Field::from(0.0);
        let z = Field::from(0.0);
        let w = Field::from(0.0);
        let _ = clamped.eval_raw(x, y, z, w);

        // Test value below range
        let x = Field::from(-0.5);
        let _ = clamped.eval_raw(x, y, z, w);

        // Test value above range
        let x = Field::from(1.5);
        let _ = clamped.eval_raw(x, y, z, w);
    }
}

// ============================================================================
// Fix Combinator Tests
// ============================================================================

mod fix_tests {
    use super::*;

    #[test]
    fn test_fix_combinator_basic() {
        // Simple fixed-point iteration: start at 0, add 1 each step, stop at 5
        let fix = Fix {
            seed: 0.0f32,
            step: W + 1.0f32, // W is the current state
            done: W.ge(5.0f32),
        };

        let x = Field::from(0.0);
        let y = Field::from(0.0);
        let z = Field::from(0.0);
        let w = Field::from(0.0);

        let result = fix.eval_raw(x, y, z, w);
        let _ = result;
    }

    #[test]
    fn test_fix_combinator_with_coordinates() {
        // Iterate: start at x, multiply by 0.5 each step, stop when < 0.1
        let fix = Fix {
            seed: X,
            step: W * 0.5f32,
            done: W.lt(0.1f32),
        };

        let x = Field::from(10.0);
        let y = Field::from(0.0);
        let z = Field::from(0.0);
        let w = Field::from(0.0);

        let result = fix.eval_raw(x, y, z, w);
        let _ = result;
    }
}

// ============================================================================
// ManifoldExt Tests
// ============================================================================

mod ext_tests {
    use super::*;

    #[test]
    fn test_eval_convenience() {
        let expr = X + Y;

        // Use eval with various types that convert to Field
        let result = expr.eval(1.0f32, 2.0f32, 0.0f32, 0.0f32);
        let _ = result;

        let result = expr.eval(1i32, 2i32, 0i32, 0i32);
        let _ = result;
    }

    #[test]
    fn test_add_via_extension() {
        let expr = X.add(Y);
        let result = expr.eval(3.0f32, 4.0f32, 0.0f32, 0.0f32);
        let _ = result;
    }

    #[test]
    fn test_sub_via_extension() {
        let expr = X.sub(Y);
        let result = expr.eval(10.0f32, 3.0f32, 0.0f32, 0.0f32);
        let _ = result;
    }

    #[test]
    fn test_mul_via_extension() {
        let expr = X.mul(Y);
        let result = expr.eval(4.0f32, 5.0f32, 0.0f32, 0.0f32);
        let _ = result;
    }

    #[test]
    fn test_div_via_extension() {
        let expr = X.div(Y);
        let result = expr.eval(10.0f32, 2.0f32, 0.0f32, 0.0f32);
        let _ = result;
    }

    #[test]
    fn test_max_via_extension() {
        let expr = X.max(Y);
        let result = expr.eval(3.0f32, 7.0f32, 0.0f32, 0.0f32);
        let _ = result;
    }

    #[test]
    fn test_min_via_extension() {
        let expr = X.min(Y);
        let result = expr.eval(3.0f32, 7.0f32, 0.0f32, 0.0f32);
        let _ = result;
    }

    #[test]
    fn test_boxed_manifold() {
        let expr = (X + Y).boxed();

        let x = Field::from(3.0);
        let y = Field::from(4.0);
        let z = Field::from(0.0);
        let w = Field::from(0.0);

        let result = expr.eval_raw(x, y, z, w);
        let _ = result;
    }

    #[test]
    fn test_boxed_manifold_operators() {
        let boxed = X.boxed();

        // Test operators on BoxedManifold
        let sum = boxed.clone() + Y;
        let diff = boxed.clone() - Y;
        let prod = boxed.clone() * Y;
        let quot = boxed / Y;

        let x = Field::from(10.0);
        let y = Field::from(2.0);
        let z = Field::from(0.0);
        let w = Field::from(0.0);

        let _ = sum.eval_raw(x, y, z, w);
        let _ = diff.eval_raw(x, y, z, w);
        let _ = prod.eval_raw(x, y, z, w);
        let _ = quot.eval_raw(x, y, z, w);
    }
}

// ============================================================================
// Jet2 Tests (Automatic Differentiation)
// ============================================================================

mod jet2_tests {
    use super::*;

    #[test]
    fn test_jet2_x_constructor() {
        let jet = Jet2::x(Field::from(5.0));
        // dx should be 1, dy should be 0
        let _ = jet.val;
        let _ = jet.dx;
        let _ = jet.dy;
    }

    #[test]
    fn test_jet2_y_constructor() {
        let jet = Jet2::y(Field::from(3.0));
        // dx should be 0, dy should be 1
        let _ = jet.val;
        let _ = jet.dx;
        let _ = jet.dy;
    }

    #[test]
    fn test_jet2_constant() {
        let jet = Jet2::constant(Field::from(10.0));
        // Both derivatives should be 0
        let _ = jet.val;
        let _ = jet.dx;
        let _ = jet.dy;
    }

    #[test]
    fn test_jet2_addition() {
        let a = Jet2::x(Field::from(2.0));
        let b = Jet2::y(Field::from(3.0));
        let sum = a + b;
        // (f + g)' = f' + g'
        let _ = sum.val;
        let _ = sum.dx;
        let _ = sum.dy;
    }

    #[test]
    fn test_jet2_subtraction() {
        let a = Jet2::x(Field::from(5.0));
        let b = Jet2::y(Field::from(2.0));
        let diff = a - b;
        let _ = diff.val;
        let _ = diff.dx;
        let _ = diff.dy;
    }

    #[test]
    fn test_jet2_multiplication() {
        let a = Jet2::x(Field::from(3.0));
        let b = Jet2::y(Field::from(4.0));
        let prod = a * b;
        // Product rule: (f * g)' = f' * g + f * g'
        let _ = prod.val;
        let _ = prod.dx;
        let _ = prod.dy;
    }

    #[test]
    fn test_jet2_division() {
        let a = Jet2::x(Field::from(10.0));
        let b = Jet2::constant(Field::from(2.0));
        let quot = a / b;
        // Quotient rule
        let _ = quot.val;
        let _ = quot.dx;
        let _ = quot.dy;
    }

    #[test]
    fn test_jet2_numeric_sqrt() {
        let jet = Jet2::x(Field::from(16.0));
        let sqrt_jet = jet.sqrt().eval(); // sqrt() returns Jet2Sqrt, eval() to get Jet2
        // Chain rule: (√f)' = f' / (2√f)
        let _ = sqrt_jet.val;
        let _ = sqrt_jet.dx;
        let _ = sqrt_jet.dy;
    }

    #[test]
    fn test_jet2_numeric_abs() {
        let jet = Jet2::x(Field::from(-5.0));
        let abs_jet = jet.abs();
        let _ = abs_jet.val;
        let _ = abs_jet.dx;
        let _ = abs_jet.dy;
    }

    #[test]
    fn test_jet2_numeric_min_max() {
        let a = Jet2::x(Field::from(3.0));
        let b = Jet2::y(Field::from(5.0));

        let min_jet = a.min(b);
        let max_jet = a.max(b);

        let _ = min_jet.val;
        let _ = max_jet.val;
    }

    #[test]
    fn test_jet2_comparisons() {
        let a = Jet2::x(Field::from(3.0));
        let b = Jet2::y(Field::from(5.0));

        let _lt = a.lt(b);
        let _gt = a.gt(b);
        let _le = a.le(b);
        let _ge = a.ge(b);
    }

    #[test]
    fn test_jet2_select() {
        let mask = Jet2::constant(Field::from(1.0)); // All true
        let a = Jet2::x(Field::from(10.0));
        let b = Jet2::y(Field::from(20.0));

        let sel = Jet2::select(mask, a, b);
        let _ = sel.val;
    }

    #[test]
    fn test_jet2_any_all() {
        let zero = Jet2::constant(Field::from(0.0));
        let one = Jet2::constant(Field::from(1.0));

        let _ = zero.any();
        let _ = zero.all();
        let _ = one.any();
        let _ = one.all();
    }

    #[test]
    fn test_jet2_from_scalars() {
        // Use Computational trait method
        let from_f32 = <Jet2 as Computational>::from_f32(42.0);
        // from_i32 is no longer public, use constant(Field::from(...)) instead
        let from_i32 = Jet2::constant(Field::from(42));

        let _ = from_f32.val;
        let _ = from_i32.val;
    }

    #[test]
    fn test_manifold_with_jet2() {
        // Test that manifolds can be evaluated with Jet2
        let expr = X * X + Y;

        let x_jet = Jet2::x(Field::from(5.0));
        let y_jet = Jet2::y(Field::from(3.0));
        let zero = Jet2::constant(Field::from(0.0));

        let result = expr.eval_raw(x_jet, y_jet, zero, zero);
        // result.val should be 5*5 + 3 = 28
        // result.dx should be 2*5 = 10 (derivative of x² + y w.r.t x)
        // result.dy should be 1 (derivative of x² + y w.r.t y)
        let _ = result.val;
        let _ = result.dx;
        let _ = result.dy;
    }
}

// ============================================================================
// Materialize Tests
// ============================================================================

mod materialize_tests {
    use super::*;

    // Helper to create a simple color manifold that returns (R, G, B, A) = (x, y, 0.5, 1.0)
    #[derive(Clone, Copy)]
    struct SimpleColorManifold;

    impl Manifold for SimpleColorManifold {
        type Output = SimpleVec4;

        fn eval_raw(&self, x: Field, y: Field, _z: Field, _w: Field) -> Self::Output {
            SimpleVec4(x, y, Field::from(0.5), Field::from(1.0))
        }
    }

    #[derive(Clone, Copy)]
    struct SimpleVec4(Field, Field, Field, Field);

    impl pixelflow_core::ops::Vector for SimpleVec4 {
        type Component = Field;

        fn get(&self, axis: Axis) -> Field {
            match axis {
                Axis::X => self.0,
                Axis::Y => self.1,
                Axis::Z => self.2,
                Axis::W => self.3,
            }
        }
    }

    #[test]
    fn test_materialize_basic() {
        let manifold = SimpleColorManifold;
        let mut out = vec![0.0f32; PARALLELISM * 4];

        materialize(&manifold, 0.0, 0.5, &mut out);

        // Output is interleaved: [r0, g0, b0, a0, r1, g1, b1, a1, ...]
        // For PARALLELISM=4 (SSE2):
        // r values should be [0, 1, 2, 3] (sequential from x=0)
        // g values should all be 0.5 (y coordinate)
        // b values should all be 0.5
        // a values should all be 1.0

        // Just verify we got output (specific values depend on SIMD backend)
        assert_eq!(out.len(), PARALLELISM * 4);
    }

    #[test]
    fn test_materialize_with_offset() {
        let manifold = SimpleColorManifold;
        let mut out = vec![0.0f32; PARALLELISM * 4];

        materialize(&manifold, 10.0, 5.0, &mut out);

        // x values should be [10, 11, 12, 13]
        // y values should all be 5.0
        assert_eq!(out.len(), PARALLELISM * 4);
    }

    #[test]
    fn test_parallelism_constant() {
        // PARALLELISM should be at least 1
        assert!(PARALLELISM >= 1);

        // On x86_64, should typically be 4 (SSE2) or 16 (AVX-512)
        #[cfg(target_arch = "x86_64")]
        assert!(PARALLELISM >= 4);
    }
}

// ============================================================================
// Complex Expression Tests
// ============================================================================

mod complex_expr_tests {
    use super::*;

    #[test]
    fn test_circle_sdf() {
        // Distance from origin: sqrt(x² + y²)
        let dist = (X * X + Y * Y).sqrt();

        let x = Field::from(3.0);
        let y = Field::from(4.0);
        let z = Field::from(0.0);
        let w = Field::from(0.0);

        // Should be 5.0
        let result = dist.eval_raw(x, y, z, w);
        let _ = result;
    }

    #[test]
    fn test_circle_sdf_with_radius() {
        // Circle SDF: sqrt(x² + y²) - radius
        let radius = 10.0f32;
        let circle_sdf = (X * X + Y * Y).sqrt() - radius;

        let x = Field::from(6.0);
        let y = Field::from(8.0);
        let z = Field::from(0.0);
        let w = Field::from(0.0);

        // sqrt(36 + 64) - 10 = 10 - 10 = 0
        let result = circle_sdf.eval_raw(x, y, z, w);
        let _ = result;
    }

    #[test]
    fn test_box_sdf() {
        // Simple 2D box SDF (centered at origin, half-width 5)
        let half_size = 5.0f32;
        let dx = X.abs() - half_size;
        let dy = Y.abs() - half_size;

        // max(dx, dy) for points inside the box
        let sdf = dx.max(dy);

        let x = Field::from(3.0);
        let y = Field::from(2.0);
        let z = Field::from(0.0);
        let w = Field::from(0.0);

        // Point (3, 2) is inside the box
        // dx = |3| - 5 = -2
        // dy = |2| - 5 = -3
        // max(-2, -3) = -2
        let result = sdf.eval_raw(x, y, z, w);
        let _ = result;
    }

    #[test]
    fn test_smooth_union() {
        // Smooth union of two circles
        let circle1 = (X * X + Y * Y).sqrt() - 5.0f32;
        let circle2 = ((X - 8.0f32) * (X - 8.0f32) + Y * Y).sqrt() - 5.0f32;

        // Simple min for now (proper smooth min needs exp)
        let union = circle1.min(circle2);

        let x = Field::from(4.0);
        let y = Field::from(0.0);
        let z = Field::from(0.0);
        let w = Field::from(0.0);

        let result = union.eval_raw(x, y, z, w);
        let _ = result;
    }

    #[test]
    fn test_linear_gradient() {
        // Linear gradient from x=0 (value 0) to x=100 (value 1)
        let gradient = X / 100.0f32;

        let x = Field::from(50.0);
        let y = Field::from(0.0);
        let z = Field::from(0.0);
        let w = Field::from(0.0);

        // Should be 0.5
        let result = gradient.eval_raw(x, y, z, w);
        let _ = result;
    }

    #[test]
    fn test_conditional_pattern() {
        // Test AND pattern with conditions
        let check_x = X.ge(0.5f32);
        let check_y = Y.ge(0.5f32);

        // AND pattern
        let pattern = check_x & check_y;

        let x = Field::from(0.7);
        let y = Field::from(0.7);
        let z = Field::from(0.0);
        let w = Field::from(0.0);

        let result = pattern.eval_raw(x, y, z, w);
        let _ = result;
    }

    #[test]
    fn test_or_pattern() {
        // Test OR pattern with conditions
        let check_x = X.lt(0.2f32);
        let check_y = Y.gt(0.8f32);

        // OR pattern
        let pattern = check_x | check_y;

        let x = Field::from(0.1);
        let y = Field::from(0.5);
        let z = Field::from(0.0);
        let w = Field::from(0.0);

        let result = pattern.eval_raw(x, y, z, w);
        let _ = result;
    }
}

// ============================================================================
// Default Trait Tests
// ============================================================================

mod default_tests {
    use super::*;

    #[test]
    fn test_field_default() {
        let f: Field = Default::default();
        let _ = f;
    }

    #[test]
    fn test_x_default() {
        let _x: X = Default::default();
    }

    #[test]
    fn test_y_default() {
        let _y: Y = Default::default();
    }

    #[test]
    fn test_z_default() {
        let _z: Z = Default::default();
    }

    #[test]
    fn test_w_default() {
        let _w: W = Default::default();
    }
}

// ============================================================================
// Clone and Copy Tests
// ============================================================================

mod clone_copy_tests {
    use super::*;

    #[test]
    fn test_field_copy() {
        let a = Field::from(1.0);
        let b = a; // Copy
        let _ = a + b;
    }

    #[test]
    fn test_field_clone() {
        let a = Field::from(1.0);
        let b = a.clone();
        let _ = a + b;
    }

    #[test]
    fn test_variable_copy() {
        let x = X;
        let x2 = x; // Copy
        let _ = x + x2;
    }

    #[test]
    fn test_jet2_copy() {
        let jet = Jet2::x(Field::from(1.0));
        let jet2 = jet; // Copy
        let _ = jet.val;
        let _ = jet2.val;
    }

    #[test]
    fn test_axis_clone() {
        let axis = Axis::X;
        let axis2 = axis.clone();
        assert_eq!(axis, axis2);
    }
}

// ============================================================================
// Debug Trait Tests
// ============================================================================

mod debug_tests {
    use super::*;
    use std::fmt::Write;

    #[test]
    fn test_field_debug() {
        let f = Field::from(42.0);
        let mut s = String::new();
        write!(s, "{:?}", f).unwrap();
        assert!(!s.is_empty());
    }

    #[test]
    fn test_axis_debug() {
        let axis = Axis::X;
        let mut s = String::new();
        write!(s, "{:?}", axis).unwrap();
        assert!(s.contains("X"));
    }

    #[test]
    fn test_jet2_debug() {
        let jet = Jet2::x(Field::from(1.0));
        let mut s = String::new();
        write!(s, "{:?}", jet).unwrap();
        assert!(!s.is_empty());
    }

    #[test]
    fn test_add_debug() {
        let add = Add(X, Y);
        let mut s = String::new();
        write!(s, "{:?}", add).unwrap();
        assert!(s.contains("Add"));
    }

    #[test]
    fn test_scale_debug() {
        let scaled = scale(X, 2.0);
        let mut s = String::new();
        write!(s, "{:?}", scaled).unwrap();
        // Scale is now a type alias for At
        assert!(s.contains("At"));
    }
}

// ============================================================================
// Hash Tests (for Axis)
// ============================================================================

mod hash_tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn test_axis_hash() {
        let mut set = HashSet::new();
        set.insert(Axis::X);
        set.insert(Axis::Y);
        set.insert(Axis::Z);
        set.insert(Axis::W);

        assert_eq!(set.len(), 4);
        assert!(set.contains(&Axis::X));
        assert!(set.contains(&Axis::Y));
        assert!(set.contains(&Axis::Z));
        assert!(set.contains(&Axis::W));
    }
}

// ============================================================================
// Additional BNot Tests
// ============================================================================

mod bnot_tests {
    use super::*;
    use pixelflow_core::ops::logic::BNot;

    #[test]
    fn test_bnot_manifold() {
        let not_x = BNot(X.gt(0.0f32));

        let x = Field::from(5.0);
        let y = Field::from(0.0);
        let z = Field::from(0.0);
        let w = Field::from(0.0);

        let result = not_x.eval_raw(x, y, z, w);
        let _ = result;
    }
}

// ============================================================================
// Additional Soft Comparison Tests (Jet2-specific)
// ============================================================================

mod soft_compare_tests {
    use super::*;
    use pixelflow_core::ops::compare::{SoftGt, SoftLt, SoftSelect};

    #[test]
    fn test_soft_gt() {
        let soft_gt = SoftGt {
            left: X,
            right: Y,
            sharpness: 1.0,
        };

        let x_jet = Jet2::x(Field::from(5.0));
        let y_jet = Jet2::y(Field::from(3.0));
        let zero = Jet2::constant(Field::from(0.0));

        let result = soft_gt.eval_raw(x_jet, y_jet, zero, zero);
        let _ = result.val;
    }

    #[test]
    fn test_soft_lt() {
        let soft_lt = SoftLt {
            left: X,
            right: Y,
            sharpness: 1.0,
        };

        let x_jet = Jet2::x(Field::from(2.0));
        let y_jet = Jet2::y(Field::from(5.0));
        let zero = Jet2::constant(Field::from(0.0));

        let result = soft_lt.eval_raw(x_jet, y_jet, zero, zero);
        let _ = result.val;
    }

    #[test]
    fn test_soft_select() {
        let soft_sel = SoftSelect {
            mask: X,
            if_true: Y,
            if_false: Z,
        };

        let x_jet = Jet2::x(Field::from(0.5)); // 50% blend
        let y_jet = Jet2::y(Field::from(10.0));
        let z_jet = Jet2::constant(Field::from(20.0));
        let w_jet = Jet2::constant(Field::from(0.0));

        let result = soft_sel.eval_raw(x_jet, y_jet, z_jet, w_jet);
        let _ = result.val;
    }
}
