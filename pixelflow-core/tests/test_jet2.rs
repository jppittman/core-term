use pixelflow_core::jet::Jet2;
use pixelflow_core::{Manifold, ManifoldExt, X, Y};

#[test]
#[ignore = "Needs internal Field access for lane extraction"]
fn test_jet2_automatic_gradient() {
    // Expression: x² + y
    let expr = X * X + Y;

    // Evaluate at (5, 3) with jets
    let x_jet = Jet2::x(5.0.into());
    let y_jet = Jet2::y(3.0.into());
    let zero = Jet2::constant(0.0.into());

    let _result = expr.eval_raw(x_jet, y_jet, zero, zero);

    // Fields are accessed directly: result.val, result.dx, result.dy
    // But we can't extract scalar values without internal store()
    // This test needs to be restructured to work with the new API
}

#[test]
#[ignore = "Needs internal Field access for lane extraction"]
fn test_jet2_product_rule() {
    let expr = X * Y;

    let x_jet = Jet2::x(3.0.into());
    let y_jet = Jet2::y(4.0.into());
    let zero = Jet2::constant(0.0.into());

    let _result = expr.eval_raw(x_jet, y_jet, zero, zero);
    // result.val, result.dx, result.dy are Fields
}

#[test]
#[ignore = "Needs internal Field access for lane extraction"]
fn test_jet2_chain_rule_sqrt() {
    let expr = X.sqrt();

    let x_jet = Jet2::x(16.0.into());
    let zero = Jet2::constant(0.0.into());

    let _result = expr.eval_raw(x_jet, zero, zero, zero);
}

#[test]
#[ignore = "Needs internal Field access for lane extraction"]
fn test_jet2_circle_normal() {
    let circle = (X * X + Y * Y).sqrt() - 100.0;

    let x_jet = Jet2::x(50.0.into());
    let y_jet = Jet2::y(50.0.into());
    let zero = Jet2::constant(0.0.into());

    let _result = circle.eval_raw(x_jet, y_jet, zero, zero);
}
