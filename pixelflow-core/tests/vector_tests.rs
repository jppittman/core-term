use pixelflow_core::batch::Batch;
use pixelflow_core::diff::{DiffCoord, DiffSurface};
use pixelflow_core::vector::QuadraticCurve;

#[test]
fn test_batch_f32_ops() {
    let a = Batch::<f32>::splat(4.0);
    let b = Batch::<f32>::splat(2.0);

    let sum = a + b;
    let diff = a - b;
    let prod = a * b;
    let quot = a / b;
    let root = a.sqrt();

    assert_eq!(sum.extract(0), 6.0);
    assert_eq!(diff.extract(0), 2.0);
    assert_eq!(prod.extract(0), 8.0);
    assert_eq!(quot.extract(0), 2.0);
    assert_eq!(root.extract(0), 2.0);
}

#[test]
fn test_quadratic_curve_canonical() {
    // Control points for simple parabola y = x^2 mapping to u^2 - v = 0.
    // Canonical points: P0=(0,0), P1=(0.5,0), P2=(1,1).
    let curve = QuadraticCurve::new(
        [0.0, 0.0],
        [0.5, 0.0],
        [1.0, 1.0]
    );

    // Check matrix is roughly identity (floating point tolerance)
    assert!((curve.matrix.m00 - 1.0).abs() < 1e-5); // u = x
    assert!((curve.matrix.m01).abs() < 1e-5);
    assert!((curve.matrix.m02).abs() < 1e-5);
    assert!((curve.matrix.m10).abs() < 1e-5);
    assert!((curve.matrix.m11 - 1.0).abs() < 1e-5); // v = y
    assert!((curve.matrix.m12).abs() < 1e-5);

    // Sample at (0.5, 0.25) -> on curve (0.25 - 0.25 = 0)
    // We use constant coordinates (no derivatives) for simple value check
    // But we need to handle the fact that gradient will be 0, causing div by zero (NaN/Inf).
    // If we want to check the sign (inside/outside), Inf or -Inf preserves sign.

    // Since gradient is 0, result is NaN or Inf.
    // But logically we want to test implicit value u^2 - v.
    // The DiffSurface implementation divides by grad_len.
    // To test implicit value, we should provide non-zero derivatives even if dummy.

    let x_dummy = DiffCoord { val: Batch::splat(0.5), dx: Batch::splat(1.0), dy: Batch::splat(0.0) };
    let y_dummy = DiffCoord { val: Batch::splat(0.25), dx: Batch::splat(0.0), dy: Batch::splat(1.0) };

    let dist = curve.sample_diff(x_dummy, y_dummy);
    // f = 0.25 - 0.25 = 0.
    assert!(dist.extract(0).abs() < 1e-4);

    // Sample at (0, 1) -> u=0, v=1 -> f = -1 (inside)
    let x_in = DiffCoord { val: Batch::splat(0.0), dx: Batch::splat(1.0), dy: Batch::splat(0.0) };
    let y_in = DiffCoord { val: Batch::splat(1.0), dx: Batch::splat(0.0), dy: Batch::splat(1.0) };
    let dist_in = curve.sample_diff(x_in, y_in);

    // f = 0 - 1 = -1.
    // gradient? u=0 -> du/dx?
    // u = x -> du/dx = 1, du/dy = 0.
    // v = y -> dv/dx = 0, dv/dy = 1.
    // f = u^2 - v.
    // df/dx = 2u * du/dx - dv/dx = 0 - 0 = 0.
    // df/dy = 2u * du/dy - dv/dy = 0 - 1 = -1.
    // |grad| = 1.
    // result = -1 / 1 = -1.

    assert!((dist_in.extract(0) - (-1.0)).abs() < 1e-4);
}

#[test]
fn test_quadratic_curve_aa() {
    let curve = QuadraticCurve::new(
        [0.0, 0.0],
        [0.5, 0.0],
        [1.0, 1.0]
    );

    let x_coord = DiffCoord {
        val: Batch::splat(1.1),
        dx: Batch::splat(1.0),
        dy: Batch::splat(0.0),
    };

    let y_coord = DiffCoord {
        val: Batch::splat(1.0),
        dx: Batch::splat(0.0),
        dy: Batch::splat(1.0),
    };

    let dist = curve.sample_diff(x_coord, y_coord);
    let d = dist.extract(0);

    // f = x^2 - y = 1.21 - 1 = 0.21
    // df/dx = 2x = 2.2
    // df/dy = -1
    // |grad| = sqrt(2.2^2 + 1) = sqrt(4.84 + 1) = sqrt(5.84) ~ 2.4166
    // Expected dist = 0.21 / 2.4166 ~ 0.0869

    assert!((d - 0.0869).abs() < 0.001, "Expected ~0.0869, got {}", d);
}
