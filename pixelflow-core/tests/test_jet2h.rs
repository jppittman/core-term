//! Tests for Jet2H: automatic differentiation with Hessian (second derivatives)
//!
//! Note: Jet2H has opaque Field values that cannot be directly inspected in tests.
//! These tests verify that Jet2H operations compile and execute without panicking.
//! For detailed numerical verification, use the materialize API with real outputs.

use pixelflow_core::Field;
use pixelflow_core::jet::Jet2H;

/// Verify that Jet2H seeding operations compile and produce values.
#[test]
fn test_jet2h_seeding() {
    let x = Jet2H::x(Field::from(2.0));
    let _res = x.val;
    let _res = x.dx;
    let _res = x.dy;
    let _res = x.dxx;
    let _res = x.dxy;
    let _res = x.dyy;

    let y = Jet2H::y(Field::from(3.0));
    let _res = y.val;

    let c = Jet2H::constant(Field::from(5.0));
    let _res = c.val;
}

/// Verify that Jet2H arithmetic operations work.
#[test]
fn test_jet2h_arithmetic() {
    let x = Jet2H::x(Field::from(2.0));
    let y = Jet2H::y(Field::from(3.0));

    let sum = x + y;
    let _res = sum.val;

    let diff = x - y;
    let _res = diff.val;

    let prod = x * y;
    let _res = prod.val;

    let quot = x / y;
    let _res = quot.val;
}

/// Verify that Jet2H unary operations work.
#[test]
fn test_jet2h_unary_ops() {
    let x = Jet2H::x(Field::from(4.0));

    let sqrt_result = x.sqrt();
    let _: Jet2H = sqrt_result.into();

    let min_result = x.min(Jet2H::y(Field::from(3.0)));
    let _res = min_result.val;

    let max_result = x.max(Jet2H::y(Field::from(3.0)));
    let _res = max_result.val;
}

/// Verify that Jet2H comparison operations work.
#[test]
fn test_jet2h_comparison() {
    let x = Jet2H::x(Field::from(2.0));
    let y = Jet2H::y(Field::from(3.0));

    let _lt = x.lt(y);
    let _gt = x.gt(y);
    let _le = x.le(y);
    let _ge = x.ge(y);
}

/// Verify that Jet2H bitwise operations work.
#[test]
fn test_jet2h_bitwise() {
    let x = Jet2H::x(Field::from(2.0));
    let y = Jet2H::y(Field::from(3.0));

    let _and = x & y;
    let _or = x | y;
    let _not = !x;
}

/// Verify that Jet2H select works.
#[test]
fn test_jet2h_select() {
    let x = Jet2H::x(Field::from(2.0));
    let y = Jet2H::y(Field::from(3.0));
    let mask = x.lt(y);

    let selected = Jet2H::select(mask, x, y);
    let _res = selected.val;
}

/// Verify that complex Jet2H expressions compile.
#[test]
fn test_jet2h_complex_expression() {
    let x = Jet2H::x(Field::from(3.0));
    let y = Jet2H::y(Field::from(4.0));

    let x_sq = x * x;
    let y_sq = y * y;
    let sum = x_sq + y_sq;
    let r: Jet2H = sum.sqrt().into();

    let _res = r.val;
    let _res = r.dx;
    let _res = r.dy;
}
