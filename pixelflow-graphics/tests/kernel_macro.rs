//! Integration tests for the kernel! macro.
//!
//! These tests verify that the macro-generated kernels work correctly with
//! the full PixelFlow stack: ManifoldExt, rasterization, and SIMD evaluation.
//!
//! ## Architecture
//!
//! The kernel! macro generates:
//! 1. A struct holding captured parameters (the environment)
//! 2. A ZST expression tree using coordinate variables (the code)
//! 3. An `.at()` binding that threads parameters into coordinate slots
//!
//! Parameters map to coordinate slots: first param → Z, second param → W.
//!
//! ## Testing Strategy
//!
//! Since Field is a SIMD type (multiple f32 lanes), we can't extract single
//! values. Instead we test using Field-level comparisons and check that all
//! lanes satisfy the expected condition.

use pixelflow_core::{Field, Manifold, ManifoldExt};
use pixelflow_macros::kernel;

type Field4 = (Field, Field, Field, Field);

/// Helper: Convert f32 tuple to Field4
fn field4(x: f32, y: f32, z: f32, w: f32) -> Field4 {
    (Field::from(x), Field::from(y), Field::from(z), Field::from(w))
}

/// Helper: Check if two Fields are approximately equal across all lanes.
///
/// Since Field operators create AST nodes (not immediate values), we:
/// 1. Build the comparison expression as an AST
/// 2. Evaluate it using `.constant()` to get a concrete Field
/// 3. Use Field's native `.all()` to check all lanes
fn fields_close(a: Field, b: Field, epsilon: f32) -> bool {
    // Build AST: Abs<Sub<Field, Field>>
    let diff_ast = (a - b).abs();
    // Evaluate at origin to collapse AST → Field
    let diff_field = diff_ast.constant();
    let eps = Field::from(epsilon);
    // Now use Field's native lt (returns mask-as-Field)
    Field::lt(diff_field, eps).all()
}

/// Test a kernel with one parameter.
#[test]
fn test_one_param_kernel() {
    // X offset by a parameter
    let offset_x = kernel!(|dx: f32| X + dx);
    let k = offset_x(10.0);

    // At x=5: result should be 5 + 10 = 15
    let result = k.eval(field4(5.0, 0.0, 0.0, 0.0));
    let expected = Field::from(15.0);
    assert!(
        fields_close(result, expected, 0.001),
        "expected 15, got different value"
    );
}

/// Test a kernel with two parameters.
#[test]
fn test_two_param_kernel() {
    // Offset both X and Y
    let offset_xy = kernel!(|dx: f32, dy: f32| (X + dx) + (Y + dy));
    let k = offset_xy(10.0, 20.0);

    // At (5, 3): result should be (5+10) + (3+20) = 38
    let result = k.eval(field4(5.0, 3.0, 0.0, 0.0));
    let expected = Field::from(38.0);
    assert!(
        fields_close(result, expected, 0.001),
        "expected 38, got different value"
    );
}

/// Test a kernel with no parameters.
#[test]
fn test_zero_param_kernel() {
    // Simple distance from origin (no params)
    let dist = kernel!(|| (X * X + Y * Y).sqrt());
    let k = dist();

    // At (3, 4): distance = 5
    let result = k.eval(field4(3.0, 4.0, 0.0, 0.0));
    let expected = Field::from(5.0);
    assert!(
        fields_close(result, expected, 0.001),
        "expected 5, got different value"
    );
}

/// Test method chaining (ManifoldExt integration).
#[test]
fn test_method_chaining() {
    // Clamped value using .max().min()
    let clamp = kernel!(|lo: f32, hi: f32| X.max(lo).min(hi));
    let k = clamp(0.0, 1.0);

    // Below range: clamp(-5) should be 0
    let below = k.eval(field4(-5.0, 0.0, 0.0, 0.0));
    assert!(
        fields_close(below, Field::from(0.0), 0.001),
        "clamp below failed"
    );

    // In range: clamp(0.5) should be 0.5
    let middle = k.eval(field4(0.5, 0.0, 0.0, 0.0));
    assert!(
        fields_close(middle, Field::from(0.5), 0.001),
        "clamp middle failed"
    );

    // Above range: clamp(5) should be 1
    let above = k.eval(field4(5.0, 0.0, 0.0, 0.0));
    assert!(
        fields_close(above, Field::from(1.0), 0.001),
        "clamp above failed"
    );
}

/// Test that kernels are Clone (not Copy, since they hold data).
#[test]
fn test_kernel_is_clone() {
    let scale = kernel!(|factor: f32| X * factor);
    let k1 = scale(2.0);
    let k2 = k1.clone();

    let r1 = k1.eval(field4(5.0, 0.0, 0.0, 0.0));
    let r2 = k2.eval(field4(5.0, 0.0, 0.0, 0.0));

    assert!(fields_close(r1, Field::from(10.0), 0.001));
    assert!(fields_close(r2, Field::from(10.0), 0.001));
}

/// Test that different instantiations are independent.
#[test]
fn test_independent_instantiations() {
    let scale = kernel!(|factor: f32| X * factor);

    let double = scale(2.0);
    let triple = scale(3.0);

    let r_double = double.eval(field4(5.0, 0.0, 0.0, 0.0));
    let r_triple = triple.eval(field4(5.0, 0.0, 0.0, 0.0));

    assert!(
        fields_close(r_double, Field::from(10.0), 0.001),
        "5 * 2 = 10"
    );
    assert!(
        fields_close(r_triple, Field::from(15.0), 0.001),
        "5 * 3 = 15"
    );
}

/// Test sqrt method.
#[test]
fn test_sqrt() {
    let root = kernel!(|val: f32| (X + val).sqrt());
    let k = root(7.0);

    // sqrt(9 + 7) = sqrt(16) = 4
    let result = k.eval(field4(9.0, 0.0, 0.0, 0.0));
    assert!(fields_close(result, Field::from(4.0), 0.001));
}

/// Test floor method.
#[test]
fn test_floor() {
    let floored = kernel!(|| X.floor());
    let k = floored();

    let result = k.eval(field4(3.7, 0.0, 0.0, 0.0));
    assert!(fields_close(result, Field::from(3.0), 0.001));

    let negative = k.eval(field4(-1.3, 0.0, 0.0, 0.0));
    assert!(fields_close(negative, Field::from(-2.0), 0.001));
}

/// Test abs method.
#[test]
fn test_abs() {
    let absolute = kernel!(|offset: f32| (X - offset).abs());
    let k = absolute(5.0);

    // |3 - 5| = 2
    let result = k.eval(field4(3.0, 0.0, 0.0, 0.0));
    assert!(fields_close(result, Field::from(2.0), 0.001));

    // |7 - 5| = 2
    let result2 = k.eval(field4(7.0, 0.0, 0.0, 0.0));
    assert!(fields_close(result2, Field::from(2.0), 0.001));
}

/// Test that the generated expression tree is ZST-based (Copy).
/// This is verified by the fact that the kernel compiles at all -
/// if parameters were injected directly, the expression wouldn't be Copy.
#[test]
fn test_zst_expression_is_copy() {
    // This kernel uses the parameter twice in the expression.
    // If the expression weren't Copy (ZST-based), this wouldn't compile
    // because the parameter would be moved on first use.
    let square_offset = kernel!(|d: f32| (X - d) * (X - d) + (Y - d) * (Y - d));
    let k = square_offset(1.0);

    // (3-1)² + (4-1)² = 4 + 9 = 13
    let result = k.eval(field4(3.0, 4.0, 0.0, 0.0));
    assert!(fields_close(result, Field::from(13.0), 0.001));
}
