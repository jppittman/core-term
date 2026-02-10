use pixelflow_core::jet::{Jet3, PathJet};
use pixelflow_core::{Field, Manifold, ManifoldExt};
use pixelflow_graphics::scene3d::PathJetSphere;

// PathJet4 is defined in scene3d.rs as:
// type PathJet4 = (PathJet<Jet3>, PathJet<Jet3>, PathJet<Jet3>, PathJet<Jet3>);

#[test]
fn path_jet_sphere_should_return_intersection_time_when_ray_hits() {
    // Sphere at (0, 0, 5) with radius 1
    let sphere = PathJetSphere::new((0.0, 0.0, 5.0), 1.0);

    // Ray from origin (0, 0, 0) towards +Z (0, 0, 1)
    let zero = Jet3::constant(Field::from(0.0));
    let one = Jet3::constant(Field::from(1.0));

    let px = PathJet { val: zero, dir: zero }; // x=0, dx=0
    let py = PathJet { val: zero, dir: zero }; // y=0, dy=0
    let pz = PathJet { val: zero, dir: one };  // z=0, dz=1
    let pw = PathJet { val: one, dir: zero };  // w=1, dw=0

    // Evaluate
    // PathJetSphere implements Manifold<PathJet4>, so eval() works with tuple
    let result_jet: Jet3 = sphere.eval((px, py, pz, pw));
    let t = result_jet.val;

    // Expected intersection at t=4.0
    // Note: PathJetSphere adds epsilon=0.0001 inside sqrt, so t will be slightly less than 4.0
    // t = 5 - sqrt(1 + 0.0001) ≈ 5 - 1.00005 = 3.99995
    let expected = Field::from(4.0);
    let tolerance = Field::from(1e-3);

    // Construct AST: (|t - expected|)
    // (t - expected) creates a Manifold (Sub<Field, Field>).
    // .abs() creates a Manifold (Abs<Sub...>) via ManifoldExt.
    let check_manifold = (t - expected).abs();

    // Evaluate the check manifold to get concrete Field
    // Since Field (t, expected) ignores input, we can pass dummy inputs.
    let fzero = Field::from(0.0);
    let diff_val = check_manifold.eval((fzero, fzero, fzero, fzero));

    // Check diff < tolerance using inherent Field::lt
    assert!(diff_val.lt(tolerance).all(), "Expected t=4.0, got deviation");
}

#[test]
fn path_jet_sphere_should_return_nan_when_ray_misses() {
    // Sphere at (0, 0, 5) with radius 1
    let sphere = PathJetSphere::new((0.0, 0.0, 5.0), 1.0);

    // Ray from origin (0, 0, 0) towards +X (1, 0, 0) - Misses completely
    let zero = Jet3::constant(Field::from(0.0));
    let one = Jet3::constant(Field::from(1.0));

    let px = PathJet { val: zero, dir: one };  // x=0, dx=1
    let py = PathJet { val: zero, dir: zero }; // y=0, dy=0
    let pz = PathJet { val: zero, dir: zero }; // z=0, dz=0
    let pw = PathJet { val: one, dir: zero };  // w=1, dw=0

    // Evaluate
    let result_jet: Jet3 = sphere.eval((px, py, pz, pw));
    let t = result_jet.val;

    // Expected Result: NaN
    let fzero = Field::from(0.0);
    // Comparison with NaN returns false
    let is_lt_zero = t.lt(fzero);
    let is_ge_zero = t.ge(fzero);

    assert!(!is_lt_zero.any() && !is_ge_zero.any(), "Expected NaN");
}
