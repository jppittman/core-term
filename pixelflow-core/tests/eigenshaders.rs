use pixelflow_core::dsl::SurfaceExt;
use pixelflow_core::field::Field;
use pixelflow_core::surfaces::{W, X, Y, Z};
// use pixelflow_core::batch::Batch; // Unused
use pixelflow_core::traits::Manifold; // For eval

#[test]
fn test_grade_surface() {
    let source = Field::splat(10.0);

    // Test basic constant slope/bias
    // 10 * 2 + 5 = 25
    let graded = source.grade(Field::splat(2.0), Field::splat(5.0));

    let x = Field::zero();
    let res = graded.eval(x, x, x, x);
    assert_eq!(res, 25.0);

    // Test with functional slope/bias
    // Slope = 0.5, Bias = 1.0 => 6.0
    let graded_fn = source.grade(Field::splat(0.5), Field::splat(1.0));

    let res_fn = graded_fn.eval(x, x, x, x);
    assert_eq!(res_fn, 6.0);
}

#[test]
fn test_lerp_surface() {
    let a = Field::splat(0.0);
    let b = Field::splat(100.0);

    // t = 0.5
    let lerped = a.lerp(Field::splat(0.5), b);

    let x = Field::zero();
    let res = lerped.eval(x, x, x, x);
    assert_eq!(res, 50.0);
}

#[test]
fn test_warp_surface() {
    // Source: f(x,y) = x
    let source = X;

    // Warp: shift x by +10
    // new_x = x + 10
    // Warp coordinate: (X+10, Y, Z, W)

    // Note: X + 10.0 promotes 10.0 to constant field.
    let warped = source.warp((X + 10.0, Y, Z, W));

    // Evaluate at 0,0
    let x = Field::zero();
    let y = Field::zero();
    let z = Field::zero();
    let w = Field::zero();

    // warp takes coordinate function and evaluates source at (warp(x,y,z,w))
    let res = warped.eval(x, y, z, w);

    assert_eq!(res, 10.0);
}
