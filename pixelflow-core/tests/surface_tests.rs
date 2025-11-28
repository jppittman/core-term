use pixelflow_core::batch::Batch;
use pixelflow_core::surface::{Surface, X, Y};

#[test]
fn test_surface_basics() {
    let u = Batch::new(0.0, 1.0, 2.0, 3.0);
    let v = Batch::splat(10.0);

    // Test X
    let x_out = X.sample(u, v);
    assert_eq!(x_out.to_array(), [0.0, 1.0, 2.0, 3.0]);

    // Test Y
    let y_out = Y.sample(u, v);
    assert_eq!(y_out.to_array(), [10.0, 10.0, 10.0, 10.0]);

    // Test Constant
    let c = Batch::splat(42.0);
    let c_out = c.sample(u, v);
    assert_eq!(c_out.to_array(), [42.0, 42.0, 42.0, 42.0]);
}
