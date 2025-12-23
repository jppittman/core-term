use pixelflow_core::{Jet2, Manifold, ManifoldExt, X, Y};

#[test]
fn test_jet2_automatic_gradient() {
    // Expression: x² + y
    let expr = X * X + Y;

    // Evaluate at (5, 3) with jets
    let x_jet = Jet2::x(5.0.into());
    let y_jet = Jet2::y(3.0.into());
    let zero = Jet2::constant(0.0.into());

    let result = expr.eval_raw(x_jet, y_jet, zero, zero);
    let (val, dx, dy) = result.extract_scalar();

    // Value: 5² + 3 = 28
    assert!(
        (val - 28.0).abs() < 0.001,
        "Value should be 28, got {}",
        val
    );

    // ∂(x² + y)/∂x = 2x = 10
    assert!((dx - 10.0).abs() < 0.001, "dx should be 10, got {}", dx);

    // ∂(x² + y)/∂y = 1
    assert!((dy - 1.0).abs() < 0.001, "dy should be 1, got {}", dy);
}

#[test]
fn test_jet2_product_rule() {
    // Expression: x * y
    let expr = X * Y;

    let x_jet = Jet2::x(3.0.into());
    let y_jet = Jet2::y(4.0.into());
    let zero = Jet2::constant(0.0.into());

    let result = expr.eval_raw(x_jet, y_jet, zero, zero);
    let (val, dx, dy) = result.extract_scalar();

    // Value: 3 * 4 = 12
    assert!((val - 12.0).abs() < 0.001);

    // ∂(x*y)/∂x = y = 4
    assert!((dx - 4.0).abs() < 0.001, "dx should be 4, got {}", dx);

    // ∂(x*y)/∂y = x = 3
    assert!((dy - 3.0).abs() < 0.001, "dy should be 3, got {}", dy);
}

#[test]
fn test_jet2_chain_rule_sqrt() {
    // Expression: sqrt(x)
    let expr = X.sqrt();

    let x_jet = Jet2::x(16.0.into());
    let zero = Jet2::constant(0.0.into());

    let result = expr.eval_raw(x_jet, zero, zero, zero);
    let (val, dx, _dy) = result.extract_scalar();

    // Value: sqrt(16) = 4
    assert!((val - 4.0).abs() < 0.001);

    // ∂(√x)/∂x = 1/(2√x) = 1/8 = 0.125
    assert!((dx - 0.125).abs() < 0.001, "dx should be 0.125, got {}", dx);
}

#[test]
fn test_jet2_circle_normal() {
    // Circle SDF: sqrt(x² + y²) - radius
    // The gradient IS the normal!
    let circle = (X * X + Y * Y).sqrt() - 100.0;

    let x_jet = Jet2::x(50.0.into());
    let y_jet = Jet2::y(50.0.into());
    let zero = Jet2::constant(0.0.into());

    let result = circle.eval_raw(x_jet, y_jet, zero, zero);
    let (val, dx, dy) = result.extract_scalar();

    // At (50, 50): distance = sqrt(5000) - 100 ≈ 70.71 - 100 = -29.29 (inside)
    let expected_dist = (5000.0f32).sqrt() - 100.0;
    assert!((val - expected_dist).abs() < 0.1, "Distance mismatch");

    // Gradient = normal = (x, y) / ||(x, y)|| = (50, 50) / 70.71
    let expected_normal = 50.0 / (5000.0f32).sqrt();
    assert!(
        (dx - expected_normal).abs() < 0.001,
        "Normal x component should be {}, got {}",
        expected_normal,
        dx
    );
    assert!(
        (dy - expected_normal).abs() < 0.001,
        "Normal y component should be {}, got {}",
        expected_normal,
        dy
    );
}
