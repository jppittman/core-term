//! Integration tests for the kernel! macro.
//!
//! These tests verify that the macro-generated kernels work correctly with
//! the full PixelFlow stack: ManifoldExt, rasterization, and SIMD evaluation.
//!
//! ## Architecture
//!
//! The kernel! macro generates:
//! 1. A struct holding captured parameters (the environment)
//! 2. A ZST expression tree using Var<N> for parameter references (the code)
//! 3. Nested Let::new() bindings that extend the domain with parameter values
//!
//! Parameters use Peano-encoded stack indices:
//! - First param (index 0) → Var<N{n-1}> (deepest in stack)
//! - Last param (index n-1) → Var<N0> (head of stack)
//!
//! This enables **unlimited parameters** (up to 8), compared to the old
//! Z/W coordinate slot approach which was limited to 2 parameters.
//!
//! ## Testing Strategy
//!
//! Since Field is a SIMD type (multiple f32 lanes), we can't extract single
//! values. Instead we test using Field-level comparisons and check that all
//! lanes satisfy the expected condition.

use pixelflow_core::jet::Jet3;
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

// ============================================================================
// Tests for >2 parameters (using Let/Var binding system)
// ============================================================================

/// Test a kernel with three parameters.
/// This demonstrates the new Let/Var binding system since Z/W slots only support 2.
#[test]
fn test_three_param_kernel() {
    // Translate point by (dx, dy) and add z offset
    let translate_3d = kernel!(|dx: f32, dy: f32, dz: f32| (X + dx) + (Y + dy) + dz);
    let k = translate_3d(10.0, 20.0, 30.0);

    // At (5, 3): result should be (5+10) + (3+20) + 30 = 68
    let result = k.eval(field4(5.0, 3.0, 0.0, 0.0));
    let expected = Field::from(68.0);
    assert!(
        fields_close(result, expected, 0.001),
        "expected 68, got different value"
    );
}

/// Test a kernel with four parameters.
/// Demonstrates full 4-parameter support.
#[test]
fn test_four_param_kernel() {
    // Combine all four parameters with coordinates
    let quad_combine = kernel!(|a: f32, b: f32, c: f32, d: f32| a + b + c + d + X + Y);
    let k = quad_combine(1.0, 2.0, 3.0, 4.0);

    // At (5, 6): result should be 1 + 2 + 3 + 4 + 5 + 6 = 21
    let result = k.eval(field4(5.0, 6.0, 0.0, 0.0));
    let expected = Field::from(21.0);
    assert!(
        fields_close(result, expected, 0.001),
        "expected 21, got different value"
    );
}

/// Test a 4-parameter sphere SDF kernel.
/// This is a practical example: signed distance from a sphere at (cx, cy, cz) with radius r.
#[test]
fn test_sphere_sdf_kernel() {
    // Sphere SDF: distance from center minus radius
    let sphere_sdf = kernel!(|cx: f32, cy: f32, cz: f32, r: f32| {
        let dx = X - cx;
        let dy = Y - cy;
        let dz = Z - cz;
        (dx * dx + dy * dy + dz * dz).sqrt() - r
    });

    // Sphere at (2, 3, 4) with radius 5
    let k = sphere_sdf(2.0, 3.0, 4.0, 5.0);

    // Point at origin (0, 0, 0): distance = sqrt(4 + 9 + 16) - 5 = sqrt(29) - 5 ≈ 0.385
    let result = k.eval(field4(0.0, 0.0, 0.0, 0.0));
    let expected = Field::from((29.0f32).sqrt() - 5.0);
    assert!(
        fields_close(result, expected, 0.01),
        "sphere SDF at origin should be ~0.385"
    );

    // Point at (2, 3, 4) (center): distance = 0 - 5 = -5
    let result_center = k.eval(field4(2.0, 3.0, 4.0, 0.0));
    let expected_center = Field::from(-5.0);
    assert!(
        fields_close(result_center, expected_center, 0.001),
        "sphere SDF at center should be -5"
    );
}

/// Test parameter ordering with 3 parameters.
/// Verifies that parameters are correctly bound (first param deepest in stack).
#[test]
fn test_parameter_ordering_three() {
    // Each parameter has a different multiplier to verify correct binding
    let order_test = kernel!(|a: f32, b: f32, c: f32| a * 100.0 + b * 10.0 + c);
    let k = order_test(1.0, 2.0, 3.0);

    // Result should be 1*100 + 2*10 + 3 = 123
    let result = k.eval(field4(0.0, 0.0, 0.0, 0.0));
    let expected = Field::from(123.0);
    assert!(
        fields_close(result, expected, 0.001),
        "expected 123 (a=1, b=2, c=3)"
    );
}

/// Test that parameters can be used multiple times in 3+ param kernels.
#[test]
fn test_param_reuse_three() {
    // Use each parameter twice
    let reuse = kernel!(|a: f32, b: f32, c: f32| (a + a) + (b + b) + (c + c));
    let k = reuse(1.0, 2.0, 3.0);

    // Result should be 2 + 4 + 6 = 12
    let result = k.eval(field4(0.0, 0.0, 0.0, 0.0));
    let expected = Field::from(12.0);
    assert!(
        fields_close(result, expected, 0.001),
        "expected 12 (each param doubled)"
    );
}

// ============================================================================
// Tests for Jet3 output (automatic differentiation)
// ============================================================================

type Jet3_4 = (Jet3, Jet3, Jet3, Jet3);

/// Helper: Convert f32 tuple to Jet3_4 (constant jets)
fn jet3_4(x: f32, y: f32, z: f32, w: f32) -> Jet3_4 {
    (
        Jet3::constant(Field::from(x)),
        Jet3::constant(Field::from(y)),
        Jet3::constant(Field::from(z)),
        Jet3::constant(Field::from(w)),
    )
}

/// Test a simple Jet3 kernel (domain inferred from output type).
#[test]
fn test_jet3_simple() {
    // X + Y with Jet3 output → domain is Jet3_4
    let add_xy = kernel!(|| -> Jet3 X + Y);
    let k = add_xy();

    // At (3, 4, 0, 0): result should be 7
    let result = k.eval(jet3_4(3.0, 4.0, 0.0, 0.0));
    let expected = Jet3::constant(Field::from(7.0));

    // Compare the val component
    let diff = (result.val - expected.val).abs();
    let eps = Field::from(0.001);
    assert!(
        Field::lt(diff.constant(), eps).all(),
        "Jet3 X + Y at (3,4) should be 7"
    );
}

/// Test Jet3 kernel with parameters (sphere SDF).
#[test]
fn test_jet3_sphere_sdf() {
    // Sphere SDF: distance from center minus radius
    // Using Jet3 for automatic differentiation (normals)
    let sphere_sdf = kernel!(|cx: f32, cy: f32, cz: f32, r: f32| -> Jet3 {
        let dx = X - cx;
        let dy = Y - cy;
        let dz = Z - cz;
        (dx * dx + dy * dy + dz * dz).sqrt() - r
    });

    // Sphere at (2, 3, 4) with radius 5
    let k = sphere_sdf(2.0, 3.0, 4.0, 5.0);

    // Point at origin (0, 0, 0): distance = sqrt(4 + 9 + 16) - 5 = sqrt(29) - 5 ≈ 0.385
    let result = k.eval(jet3_4(0.0, 0.0, 0.0, 0.0));
    let expected_val = (29.0f32).sqrt() - 5.0;
    let expected = Jet3::constant(Field::from(expected_val));

    let diff = (result.val - expected.val).abs();
    let eps = Field::from(0.01);
    assert!(
        Field::lt(diff.constant(), eps).all(),
        "Jet3 sphere SDF at origin should be ~0.385"
    );
}

// ============================================================================
// Tests for kernel composition (manifold parameters)
// ============================================================================

/// Test basic kernel composition with a single manifold parameter.
/// This is the core use case: composing a distance function with a circle SDF.
#[test]
fn test_simple_kernel_composition() {
    // Distance from a point (parametric)
    let dist = kernel!(|cx: f32, cy: f32| {
        let dx = X - cx;
        let dy = Y - cy;
        (dx * dx + dy * dy).sqrt()
    });

    // Circle SDF: takes a distance manifold and subtracts radius
    let circle = kernel!(|inner: kernel, r: f32| inner - r);

    // Compose: circle centered at origin with radius 1
    let c = circle(dist(0.0, 0.0), 1.0);

    // At (2, 0): distance from origin is 2, minus radius 1 = 1
    let result = c.eval(field4(2.0, 0.0, 0.0, 0.0));
    let expected = Field::from(1.0);
    assert!(
        fields_close(result, expected, 0.001),
        "circle SDF at (2,0) should be 1.0"
    );

    // At (0, 0): distance from origin is 0, minus radius 1 = -1 (inside)
    let result_center = c.eval(field4(0.0, 0.0, 0.0, 0.0));
    let expected_center = Field::from(-1.0);
    assert!(
        fields_close(result_center, expected_center, 0.001),
        "circle SDF at center should be -1.0"
    );
}

/// Test kernel composition with offset centers.
#[test]
fn test_kernel_composition_with_offset() {
    let dist = kernel!(|cx: f32, cy: f32| {
        let dx = X - cx;
        let dy = Y - cy;
        (dx * dx + dy * dy).sqrt()
    });

    let circle = kernel!(|inner: kernel, r: f32| inner - r);

    // Circle at (3, 4) with radius 5
    let c = circle(dist(3.0, 4.0), 5.0);

    // At origin (0, 0): distance from (3,4) is 5, minus radius 5 = 0 (on surface)
    let result = c.eval(field4(0.0, 0.0, 0.0, 0.0));
    let expected = Field::from(0.0);
    assert!(
        fields_close(result, expected, 0.001),
        "circle SDF at origin should be 0 (on surface)"
    );

    // At (3, 4): center, SDF = -5
    let result_center = c.eval(field4(3.0, 4.0, 0.0, 0.0));
    let expected_center = Field::from(-5.0);
    assert!(
        fields_close(result_center, expected_center, 0.001),
        "circle SDF at center should be -5"
    );
}

/// Test multiple manifold parameters (SDF union).
#[test]
fn test_two_manifold_params() {
    // Basic circle SDF
    let circle_sdf = kernel!(|cx: f32, cy: f32, r: f32| {
        let dx = X - cx;
        let dy = Y - cy;
        (dx * dx + dy * dy).sqrt() - r
    });

    // SDF union: min of two SDFs
    let sdf_union = kernel!(|a: kernel, b: kernel| a.min(b));

    // Union of two circles: one at origin r=1, one at (3,0) r=1
    let c1 = circle_sdf(0.0, 0.0, 1.0);
    let c2 = circle_sdf(3.0, 0.0, 1.0);
    let union = sdf_union(c1, c2);

    // At (0, 0): inside first circle, SDF = -1
    let result_c1 = union.eval(field4(0.0, 0.0, 0.0, 0.0));
    assert!(
        fields_close(result_c1, Field::from(-1.0), 0.001),
        "union at first circle center should be -1"
    );

    // At (3, 0): inside second circle, SDF = -1
    let result_c2 = union.eval(field4(3.0, 0.0, 0.0, 0.0));
    assert!(
        fields_close(result_c2, Field::from(-1.0), 0.001),
        "union at second circle center should be -1"
    );

    // At (1.5, 0): midpoint between circles, both circles contribute
    // Distance to c1 center = 1.5, minus r=1 → 0.5
    // Distance to c2 center = 1.5, minus r=1 → 0.5
    // min(0.5, 0.5) = 0.5
    let result_mid = union.eval(field4(1.5, 0.0, 0.0, 0.0));
    assert!(
        fields_close(result_mid, Field::from(0.5), 0.001),
        "union at midpoint should be 0.5"
    );
}

/// Test mixed manifold and scalar parameters.
#[test]
fn test_mixed_manifold_scalar_params() {
    // Scale an SDF by a factor
    let scale_sdf = kernel!(|inner: kernel, factor: f32| inner * factor);

    // Simple distance from origin
    let dist = kernel!(|| (X * X + Y * Y).sqrt());

    // Scale the distance by 2
    let scaled = scale_sdf(dist(), 2.0);

    // At (3, 4): distance = 5, scaled = 10
    let result = scaled.eval(field4(3.0, 4.0, 0.0, 0.0));
    assert!(
        fields_close(result, Field::from(10.0), 0.001),
        "scaled distance at (3,4) should be 10"
    );
}

/// Test chained kernel composition (three levels deep).
#[test]
fn test_chained_composition() {
    // Basic X coordinate
    let get_x = kernel!(|| X);

    // Add a constant to a manifold
    let add_const = kernel!(|inner: kernel, val: f32| inner + val);

    // Multiply a manifold by a constant
    let mul_const = kernel!(|inner: kernel, val: f32| inner * val);

    // Chain: (X + 5) * 2
    let composed = mul_const(add_const(get_x(), 5.0), 2.0);

    // At x=3: (3 + 5) * 2 = 16
    let result = composed.eval(field4(3.0, 0.0, 0.0, 0.0));
    assert!(
        fields_close(result, Field::from(16.0), 0.001),
        "(3 + 5) * 2 should be 16"
    );
}

/// Test that composed kernels can be cloned (the inner kernel is owned).
#[test]
fn test_composed_kernel_ownership() {
    let dist = kernel!(|cx: f32, cy: f32| {
        let dx = X - cx;
        let dy = Y - cy;
        (dx * dx + dy * dy).sqrt()
    });

    let circle = kernel!(|inner: kernel, r: f32| inner - r);

    // Create a composed kernel
    let c = circle(dist(0.0, 0.0), 1.0);

    // Evaluate multiple times (the kernel is borrowed, not moved)
    let r1 = c.eval(field4(2.0, 0.0, 0.0, 0.0));
    let r2 = c.eval(field4(0.0, 2.0, 0.0, 0.0));

    assert!(fields_close(r1, Field::from(1.0), 0.001));
    assert!(fields_close(r2, Field::from(1.0), 0.001));
}
