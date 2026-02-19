use pixelflow_core::jet::Jet2;
use pixelflow_core::{ManifoldCompat, ManifoldExt, X, Y, Field, Manifold, materialize, PARALLELISM, Log2};
use pixelflow_core::ops::Vector;
use pixelflow_core::variables::Axis;

#[derive(Clone, Copy)]
struct Jet2Wrapper(Jet2);

impl Manifold for Jet2Wrapper {
    type Output = Jet2Wrapper;
    fn eval(&self, _: (Field, Field, Field, Field)) -> Jet2Wrapper {
        *self
    }
}

impl Vector for Jet2Wrapper {
    type Component = Field;
    fn get(&self, axis: Axis) -> Field {
        match axis {
            Axis::X => self.0.val,
            Axis::Y => self.0.dx,
            Axis::Z => self.0.dy,
            Axis::W => Field::from(0.0),
        }
    }
}

fn extract_jet2(jet: Jet2) -> (f32, f32, f32) {
    let wrapper = Jet2Wrapper(jet);
    let mut out = [0.0; PARALLELISM * 4];
    materialize(&wrapper, 0.0, 0.0, &mut out);
    (out[0], out[1], out[2])
}

fn eval_jet(m: impl Manifold<(Field, Field, Field, Field), Output = Jet2>) -> Jet2 {
    let zero = Field::from(0.0);
    m.eval((zero, zero, zero, zero))
}

#[test]
fn automatic_gradient_should_match_analytical_for_polynomial() {
    // Expression: x² + y
    let expr = X * X + Y;

    // Evaluate at (5, 3) with jets
    let x_jet = Jet2::x(5.0.into());
    let y_jet = Jet2::y(3.0.into());
    let zero = Jet2::constant(0.0.into());

    let result = expr.eval_raw(x_jet, y_jet, zero, zero);
    let (val, dx, dy) = extract_jet2(result);

    // f = x^2 + y
    // f(5, 3) = 25 + 3 = 28
    // df/dx = 2x = 10
    // df/dy = 1

    assert!((val - 28.0).abs() < 1e-4, "Expected val ~ 28.0, got {}", val);
    assert!((dx - 10.0).abs() < 1e-4, "Expected dx ~ 10.0, got {}", dx);
    assert!((dy - 1.0).abs() < 1e-4, "Expected dy ~ 1.0, got {}", dy);
}

#[test]
fn product_rule_should_be_applied_correctly() {
    let expr = X * Y;

    let x_jet = Jet2::x(3.0.into());
    let y_jet = Jet2::y(4.0.into());
    let zero = Jet2::constant(0.0.into());

    let result = expr.eval_raw(x_jet, y_jet, zero, zero);
    let (val, dx, dy) = extract_jet2(result);

    // f = x * y
    // f(3, 4) = 12
    // df/dx = y = 4
    // df/dy = x = 3

    assert!((val - 12.0).abs() < 1e-4, "Expected val ~ 12.0, got {}", val);
    assert!((dx - 4.0).abs() < 1e-4, "Expected dx ~ 4.0, got {}", dx);
    assert!((dy - 3.0).abs() < 1e-4, "Expected dy ~ 3.0, got {}", dy);
}

#[test]
fn chain_rule_sqrt_should_work() {
    let expr = X.sqrt();

    let x_jet = Jet2::x(16.0.into());
    let zero = Jet2::constant(0.0.into());

    let result = expr.eval_raw(x_jet, zero, zero, zero);
    let (val, dx, dy) = extract_jet2(result);

    // f = sqrt(x)
    // f(16) = 4
    // df/dx = 1/(2*sqrt(x)) = 1/(2*4) = 1/8 = 0.125
    // df/dy = 0

    // Relaxed tolerance for rsqrt approximation
    assert!((val - 4.0).abs() < 2e-3, "Expected val ~ 4.0, got {}", val);
    assert!((dx - 0.125).abs() < 1e-3, "Expected dx ~ 0.125, got {}", dx);
    assert!((dy - 0.0).abs() < 1e-3, "Expected dy ~ 0.0, got {}", dy);
}

#[test]
fn circle_normal_should_be_normalized_vector() {
    // Compute distance from origin - the SDF of a circle centered at origin
    // We use just the sqrt(x² + y²) part to get the autodiff gradients.
    // The constant radius subtraction happens after evaluation to keep Jet2 compatibility.
    let dist = (X * X + Y * Y).sqrt();

    // At (50, 50), dist is sqrt(5000) approx 70.71
    // Normal should be (50/70.71, 50/70.71) = (0.7071, 0.7071)

    let x_jet = Jet2::x(50.0.into());
    let y_jet = Jet2::y(50.0.into());
    let zero = Jet2::constant(0.0.into());

    // Evaluate with jets to get automatic gradients (distance and partial derivatives)
    let dist_result = dist.eval_raw(x_jet, y_jet, zero, zero);
    let (val, dx, dy) = extract_jet2(dist_result);

    let expected_dist = (50.0f32 * 50.0 + 50.0 * 50.0).sqrt();
    let expected_grad = 50.0 / expected_dist; // 1/sqrt(2)

    // Relaxed tolerance for rsqrt approximation accumulated error
    assert!((val - expected_dist).abs() < 0.05, "Expected val ~ {}, got {}", expected_dist, val);
    assert!((dx - expected_grad).abs() < 1e-3, "Expected dx ~ {}, got {}", expected_grad, dx);
    assert!((dy - expected_grad).abs() < 1e-3, "Expected dy ~ {}, got {}", expected_grad, dy);
}

#[test]
fn min_should_return_rhs_derivative_when_values_equal() {
    // min(x, y) at x=5, y=5
    // If equal, our implementation returns rhs derivative.
    // lhs: x, dx=1, dy=0
    // rhs: y, dx=0, dy=1
    // expect dx=0, dy=1

    let lhs = Jet2::x(5.0.into());
    let rhs = Jet2::y(5.0.into());

    let result = lhs.min(rhs);
    let (val, dx, dy) = extract_jet2(result);

    assert_eq!(val, 5.0);
    // Since 5.0 == 5.0, check implementation detail: it uses Select with mask <
    // mask = 5.0 < 5.0 is false.
    // select(mask, lhs, rhs) returns rhs.
    // So we expect rhs derivatives.
    assert_eq!(dx, 0.0);
    assert_eq!(dy, 1.0);
}

#[test]
fn max_should_return_rhs_derivative_when_values_equal() {
    // max(x, y) at x=5, y=5
    // mask = lhs > rhs -> 5 > 5 -> false.
    // select(mask, lhs, rhs) -> rhs.

    let lhs = Jet2::x(5.0.into());
    let rhs = Jet2::y(5.0.into());

    let result = lhs.max(rhs);
    let (val, dx, dy) = extract_jet2(result);

    assert_eq!(val, 5.0);
    assert_eq!(dx, 0.0);
    assert_eq!(dy, 1.0);
}

#[test]
fn log2_should_return_nan_when_input_is_zero() {
    let x = Jet2::x(0.0.into());
    let result = eval_jet(Log2(x));
    let (val, dx, _dy) = extract_jet2(result);

    // log2(0) = -Inf
    // d/dx log2(x) = 1/(x ln 2). At 0, 1/0 = Inf.

    // Our implementation:
    // val.log2() -> -Inf (presumably, or specialized)
    // inv_val = 1/0 = Inf
    // deriv = Inf * log2_e = Inf

    // Print for debugging if it fails
    if !(val == f32::NEG_INFINITY || val.is_nan()) {
        println!("log2(0) returned: {}", val);
    }

    assert!(val == f32::NEG_INFINITY || val.is_nan() || val < -100.0, "Expected -Inf or very small, got {}", val);
    assert!(dx.is_infinite() || dx.is_nan(), "Expected dx Inf/NaN, got {}", dx);
}

#[test]
fn sqrt_should_return_zero_when_input_is_zero() {
    let x = Jet2::x(0.0.into());
    let result = x.sqrt().into();
    let (val, dx, _dy) = extract_jet2(result);

    // sqrt(0) = 0
    // d/dx sqrt(x) = 1/(2 sqrt(x)) = 1/0 = Inf

    // Implementation:
    // rsqrt(0) -> Inf
    // sqrt = 0 * Inf = NaN -> fixed to 0 by select(x<=0, 0, ...)
    // dx = 0.5 * rsqrt(0) * rsqrt(0) ... wait.
    // In Jet2::sqrt:
    // rsqrt_val = val.rsqrt(). If val=0, rsqrt=Inf.
    // sqrt_val = val * rsqrt_val = 0 * Inf = NaN.
    // half_rsqrt = rsqrt_val * 0.5 = Inf.
    // dx = self.dx * half_rsqrt = 1 * Inf = Inf.

    // But wait, `Field::sqrt_fast` fixes `val` to 0. `Jet2` uses `Field::rsqrt` directly?
    // Jet2::sqrt implementation:
    // let rsqrt_val = self.val.rsqrt();
    // let sqrt_val = self.val * rsqrt_val;
    // let half_rsqrt = rsqrt_val * Field::from(0.5);
    // Self::new(sqrt_val, ...)

    // It does NOT fix `sqrt_val`. So `val` will be NaN for 0 input!
    // This is a BUG if `Jet2::sqrt` doesn't match `Field::sqrt` behavior (which handles 0).
    // `Field::sqrt` does `select_raw(is_zero_or_neg, zero, result)`.
    // `Jet2::sqrt` does not.

    // So `val` will be NaN.

    // Let's verify this failure.
    // If it fails, I found a mutant/bug!

    // If it fails, I should fix the code in `Jet2::sqrt`.

    // But for now, I'll write the test to expect what I think is correct (val=0).
    // If it fails, I will fix the implementation.

    assert_eq!(val, 0.0, "sqrt(0) should be 0, but got {}", val);
    // After fix, we clamp x<=0 to 0, so derivative becomes 0.
    // This is safer for graphics than returning Inf.
    assert_eq!(dx, 0.0, "Derivative at 0 should be 0 (clamped)");
}
