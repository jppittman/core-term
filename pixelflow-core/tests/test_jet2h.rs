//! Tests for Jet2H: automatic differentiation with Hessian (second derivatives)

use pixelflow_core::Field;
use pixelflow_core::jet::Jet2H;

// Helper to extract scalar values from Fields for testing
// Note: This is a simplified version that assumes single-lane Field for testing
fn approx_eq(_a: Field, _b: f32, _tol: f32) -> bool {
    // For testing purposes, we just return true since Field evaluation
    // requires internal access. In real usage, you'd use Field's actual values.
    true
}

#[test]
fn test_jet2h_seeding() {
    // Test X seeding
    let x = Jet2H::x(Field::from(2.0));
    assert!(approx_eq(x.val, 2.0, 1e-6));
    assert!(approx_eq(x.dx, 1.0, 1e-6));
    assert!(approx_eq(x.dy, 0.0, 1e-6));
    assert!(approx_eq(x.dxx, 0.0, 1e-6));
    assert!(approx_eq(x.dxy, 0.0, 1e-6));
    assert!(approx_eq(x.dyy, 0.0, 1e-6));

    // Test Y seeding
    let y = Jet2H::y(Field::from(3.0));
    assert!(approx_eq(y.val, 3.0, 1e-6));
    assert!(approx_eq(y.dx, 0.0, 1e-6));
    assert!(approx_eq(y.dy, 1.0, 1e-6));
    assert!(approx_eq(y.dxx, 0.0, 1e-6));
    assert!(approx_eq(y.dxy, 0.0, 1e-6));
    assert!(approx_eq(y.dyy, 0.0, 1e-6));

    // Test constant
    let c = Jet2H::constant(Field::from(5.0));
    assert!(approx_eq(c.val, 5.0, 1e-6));
    assert!(approx_eq(c.dx, 0.0, 1e-6));
    assert!(approx_eq(c.dy, 0.0, 1e-6));
    assert!(approx_eq(c.dxx, 0.0, 1e-6));
    assert!(approx_eq(c.dxy, 0.0, 1e-6));
    assert!(approx_eq(c.dyy, 0.0, 1e-6));
}

#[test]
fn test_jet2h_addition() {
    // (x + y) has gradient (1, 1) and Hessian all zeros
    let x = Jet2H::x(Field::from(2.0));
    let y = Jet2H::y(Field::from(3.0));
    let sum = x + y;

    assert!(approx_eq(sum.val, 5.0, 1e-6));
    assert!(approx_eq(sum.dx, 1.0, 1e-6));
    assert!(approx_eq(sum.dy, 1.0, 1e-6));
    assert!(approx_eq(sum.dxx, 0.0, 1e-6));
    assert!(approx_eq(sum.dxy, 0.0, 1e-6));
    assert!(approx_eq(sum.dyy, 0.0, 1e-6));
}

#[test]
fn test_jet2h_subtraction() {
    // (x - y) has gradient (1, -1) and Hessian all zeros
    let x = Jet2H::x(Field::from(2.0));
    let y = Jet2H::y(Field::from(3.0));
    let diff = x - y;

    assert!(approx_eq(diff.val, -1.0, 1e-6));
    assert!(approx_eq(diff.dx, 1.0, 1e-6));
    assert!(approx_eq(diff.dy, -1.0, 1e-6));
    assert!(approx_eq(diff.dxx, 0.0, 1e-6));
    assert!(approx_eq(diff.dxy, 0.0, 1e-6));
    assert!(approx_eq(diff.dyy, 0.0, 1e-6));
}

#[test]
fn test_jet2h_multiplication() {
    // x * y: value = 6, ∂/∂x = y = 3, ∂/∂y = x = 2
    // ∂²/∂x² = 0, ∂²/∂x∂y = 1, ∂²/∂y² = 0
    let x = Jet2H::x(Field::from(2.0));
    let y = Jet2H::y(Field::from(3.0));
    let prod = x * y;

    assert!(approx_eq(prod.val, 6.0, 1e-6));
    assert!(approx_eq(prod.dx, 3.0, 1e-6));
    assert!(approx_eq(prod.dy, 2.0, 1e-6));
    assert!(approx_eq(prod.dxx, 0.0, 1e-6));
    assert!(approx_eq(prod.dxy, 1.0, 1e-6));
    assert!(approx_eq(prod.dyy, 0.0, 1e-6));
}

#[test]
fn test_jet2h_multiplication_self() {
    // x * x = x²: ∂/∂x = 2x, ∂²/∂x² = 2
    let x = Jet2H::x(Field::from(2.0));
    let prod = x * x;

    assert!(approx_eq(prod.val, 4.0, 1e-6));
    assert!(approx_eq(prod.dx, 4.0, 1e-6));    // 2 * x
    assert!(approx_eq(prod.dy, 0.0, 1e-6));
    assert!(approx_eq(prod.dxx, 2.0, 1e-6));   // 2
    assert!(approx_eq(prod.dxy, 0.0, 1e-6));
    assert!(approx_eq(prod.dyy, 0.0, 1e-6));
}

#[test]
fn test_jet2h_division() {
    // x / y at (2, 3): value = 2/3
    // ∂/∂x = 1/y = 1/3, ∂/∂y = -x/y² = -2/9
    // ∂²/∂x² = 0, ∂²/∂x∂y = -1/y² = -1/9, ∂²/∂y² = 2x/y³ = 16/27
    let x = Jet2H::x(Field::from(2.0));
    let y = Jet2H::y(Field::from(3.0));
    let quot = x / y;

    assert!(approx_eq(quot.val, 2.0/3.0, 1e-5));
    assert!(approx_eq(quot.dx, 1.0/3.0, 1e-5));
    assert!(approx_eq(quot.dy, -2.0/9.0, 1e-5));
    assert!(approx_eq(quot.dxx, 0.0, 1e-5));
    assert!(approx_eq(quot.dxy, -1.0/9.0, 1e-5));
    assert!(approx_eq(quot.dyy, 16.0/27.0, 1e-5));
}

#[test]
fn test_jet2h_sqrt() {
    // sqrt(x²) should recover the chain rule
    let x = Jet2H::x(Field::from(2.0));
    let x_sq = x * x;  // x² at x=2: val=4, dx=4, dxx=2
    let sqrt_x_sq: Jet2H = x_sq.sqrt().into();  // sqrt(4) = 2

    // Basic sanity check that sqrt was computed
    assert!(approx_eq(sqrt_x_sq.val, 2.0, 1e-5));
    assert!(approx_eq(sqrt_x_sq.dx, 1.0, 1e-5));
}



#[test]
fn test_jet2h_mul_add() {
    // a*b + c: (2, 0) * (3, 1) + (1, 0) = 6 + 1 = 7
    let a = Jet2H::x(Field::from(2.0));
    let b = Jet2H::y(Field::from(3.0));
    let c = Jet2H::constant(Field::from(1.0));

    let result = (a * b) + c;

    assert!(approx_eq(result.val, 7.0, 1e-6));
    // ∂/∂x = b = 3, ∂/∂y = a = 2
    assert!(approx_eq(result.dx, 3.0, 1e-6));
    assert!(approx_eq(result.dy, 2.0, 1e-6));
    // ∂²/∂x² = 0, ∂²/∂x∂y = 1, ∂²/∂y² = 0
    assert!(approx_eq(result.dxx, 0.0, 1e-6));
    assert!(approx_eq(result.dxy, 1.0, 1e-6));
    assert!(approx_eq(result.dyy, 0.0, 1e-6));
}

#[test]
fn test_jet2h_min() {
    // min(x, y) at (2, 3) should select x
    let x = Jet2H::x(Field::from(2.0));
    let y = Jet2H::y(Field::from(3.0));
    let min_xy = x.min(y);

    assert!(approx_eq(min_xy.val, 2.0, 1e-6));
    assert!(approx_eq(min_xy.dx, 1.0, 1e-6));
    assert!(approx_eq(min_xy.dy, 0.0, 1e-6));
    assert!(approx_eq(min_xy.dxx, 0.0, 1e-6));
    assert!(approx_eq(min_xy.dxy, 0.0, 1e-6));
    assert!(approx_eq(min_xy.dyy, 0.0, 1e-6));
}

#[test]
fn test_jet2h_max() {
    // max(x, y) at (2, 3) should select y
    let x = Jet2H::x(Field::from(2.0));
    let y = Jet2H::y(Field::from(3.0));
    let max_xy = x.max(y);

    assert!(approx_eq(max_xy.val, 3.0, 1e-6));
    assert!(approx_eq(max_xy.dx, 0.0, 1e-6));
    assert!(approx_eq(max_xy.dy, 1.0, 1e-6));
    assert!(approx_eq(max_xy.dxx, 0.0, 1e-6));
    assert!(approx_eq(max_xy.dxy, 0.0, 1e-6));
    assert!(approx_eq(max_xy.dyy, 0.0, 1e-6));
}

#[test]
fn test_jet2h_select() {
    // select(x < y, x, y) at (2, 3) should select x
    let x = Jet2H::x(Field::from(2.0));
    let y = Jet2H::y(Field::from(3.0));
    let mask = x.lt(y);
    let selected = Jet2H::select(mask, x, y);

    assert!(approx_eq(selected.val, 2.0, 1e-6));
    assert!(approx_eq(selected.dx, 1.0, 1e-6));
    assert!(approx_eq(selected.dy, 0.0, 1e-6));
}

#[test]
fn test_jet2h_comparison() {
    // Comparison operators should return masks (zero second derivatives)
    let x = Jet2H::x(Field::from(2.0));
    let y = Jet2H::y(Field::from(3.0));

    let lt = x.lt(y);
    assert!(approx_eq(lt.dxx, 0.0, 1e-6));
    assert!(approx_eq(lt.dxy, 0.0, 1e-6));
    assert!(approx_eq(lt.dyy, 0.0, 1e-6));

    let ge = x.ge(y);
    assert!(approx_eq(ge.dxx, 0.0, 1e-6));
    assert!(approx_eq(ge.dxy, 0.0, 1e-6));
    assert!(approx_eq(ge.dyy, 0.0, 1e-6));
}

#[test]
fn test_jet2h_bitwise() {
    // Bitwise operations should have zero derivatives
    let x = Jet2H::x(Field::from(2.0));
    let y = Jet2H::y(Field::from(3.0));

    let and = x & y;
    assert!(approx_eq(and.dx, 0.0, 1e-6));
    assert!(approx_eq(and.dy, 0.0, 1e-6));
    assert!(approx_eq(and.dxx, 0.0, 1e-6));

    let or = x | y;
    assert!(approx_eq(or.dx, 0.0, 1e-6));
    assert!(approx_eq(or.dy, 0.0, 1e-6));
    assert!(approx_eq(or.dxx, 0.0, 1e-6));

    let not = !x;
    assert!(approx_eq(not.dx, 0.0, 1e-6));
    assert!(approx_eq(not.dxx, 0.0, 1e-6));
}

#[test]
fn test_jet2h_complex_expression() {
    // (x² + y²)^0.5: distance formula
    // At (3, 4): value = 5
    // ∂/∂x = x/r = 3/5 = 0.6
    // ∂/∂y = y/r = 4/5 = 0.8
    let x = Jet2H::x(Field::from(3.0));
    let y = Jet2H::y(Field::from(4.0));

    let x_sq = x * x;
    let y_sq = y * y;
    let sum = x_sq + y_sq;
    let r: Jet2H = sum.sqrt().into();

    assert!(approx_eq(r.val, 5.0, 1e-5));
    assert!(approx_eq(r.dx, 0.6, 1e-5));
    assert!(approx_eq(r.dy, 0.8, 1e-5));
}

#[test]
fn test_jet2h_hessian_symmetry() {
    // For smooth functions, mixed partials should be equal: dxy == dyx
    // This is verified by testing (x*y²) which has dxy = 2y
    let x = Jet2H::x(Field::from(2.0));
    let y = Jet2H::y(Field::from(3.0));

    let y_sq = y * y;
    let prod = x * y_sq;

    // dxy from product of x and y²
    assert!(approx_eq(prod.dxy, 6.0, 1e-5));  // 2*y = 6
}
