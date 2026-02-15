//! Rigorous unit tests for Scene3D components.
//! These tests verify individual logic components like PathJetSphere and Reflect
//! with precise inputs, ensuring robustness against mutants.

use pixelflow_core::jet::{Jet3, PathJet};
use pixelflow_core::{Field, HasDerivatives, Manifold};
use pixelflow_graphics::scene3d::{PathJetSphere, Reflect};
type PathJet4 = (PathJet<Jet3>, PathJet<Jet3>, PathJet<Jet3>, PathJet<Jet3>);

/// Helper to read a scalar value from a Field.
/// Uses unsafe casting to extract the first SIMD lane, same as in pixelflow-core tests.
fn read_field(f: Field) -> f32 {
    unsafe { *(&f as *const Field as *const f32) }
}

/// Helper to construct a constant Jet3 with zero derivatives.
fn constant_jet3(val: f32) -> Jet3 {
    Jet3::constant(Field::from(val))
}

/// Helper to construct a Jet3 with specific value and derivatives.
fn jet3(val: f32, dx: f32, dy: f32, dz: f32) -> Jet3 {
    Jet3 {
        val: Field::from(val),
        dx: Field::from(dx),
        dy: Field::from(dy),
        dz: Field::from(dz),
    }
}

#[test]
fn path_jet_sphere_should_return_correct_t_when_ray_hits_center() {
    // Sphere at (0, 0, 0) radius 1.0
    let sphere = PathJetSphere {
        center: (0.0, 0.0, 0.0),
        radius: 1.0,
    };

    // Ray: Origin (0, 0, 5), Direction (0, 0, -1)
    // Should hit at (0, 0, 1). Distance t = 4.

    // Construct PathJet inputs
    // x: val=0, dir=0
    // y: val=0, dir=0
    // z: val=5, dir=-1
    // w: val=1, dir=0 (homogeneous coordinate, usually 1 for pos, 0 for dir)

    let zero = constant_jet3(0.0);
    let five = constant_jet3(5.0);
    let neg_one = constant_jet3(-1.0);
    let one = constant_jet3(1.0);

    let px = PathJet { val: zero, dir: zero };
    let py = PathJet { val: zero, dir: zero };
    let pz = PathJet { val: five, dir: neg_one };
    let pw = PathJet { val: one, dir: zero };

    let t_jet = sphere.eval((px, py, pz, pw));
    let t = read_field(t_jet.val());

    // Expected t = 4.0
    // Relax tolerance for rsqrt approximation
    assert!((t - 4.0).abs() < 2e-3, "Expected t=4.0, got {}", t);
}

#[test]
fn path_jet_sphere_should_return_miss_when_ray_misses() {
    // Sphere at (0, 0, 0) radius 1.0
    let sphere = PathJetSphere {
        center: (0.0, 0.0, 0.0),
        radius: 1.0,
    };

    // Ray: Origin (2, 0, 5), Direction (0, 0, -1)
    // Misses sphere (x=2 > radius=1)

    let zero = constant_jet3(0.0);
    let two = constant_jet3(2.0);
    let five = constant_jet3(5.0);
    let neg_one = constant_jet3(-1.0);
    let one = constant_jet3(1.0);

    let px = PathJet { val: two, dir: zero };
    let py = PathJet { val: zero, dir: zero };
    let pz = PathJet { val: five, dir: neg_one };
    let pw = PathJet { val: one, dir: zero };

    let t_jet = sphere.eval((px, py, pz, pw));
    let t = read_field(t_jet.val());

    // We expect t to be invalid (negative or NaN) to indicate a miss.
    // Surface kernel checks (t > 0) & (t < max), which handles NaN as false.
    assert!(t < 0.0 || t.is_nan(), "Expected negative or NaN t for miss, got {}", t);
}

// Separate test for each component to workaround Output=Field limitation
#[test]
fn reflect_should_compute_correct_reflection_vector_x() {
    let inv_sqrt_2 = 1.0 / 2.0f32.sqrt();
    check_reflect_component(inv_sqrt_2, -inv_sqrt_2, 0.0, 0, inv_sqrt_2);
}

#[test]
fn reflect_should_compute_correct_reflection_vector_y() {
    let inv_sqrt_2 = 1.0 / 2.0f32.sqrt();
    check_reflect_component(inv_sqrt_2, -inv_sqrt_2, 0.0, 1, inv_sqrt_2);
}

#[test]
fn reflect_should_compute_correct_reflection_vector_z() {
    let inv_sqrt_2 = 1.0 / 2.0f32.sqrt();
    check_reflect_component(inv_sqrt_2, -inv_sqrt_2, 0.0, 2, 0.0);
}

fn check_reflect_component(dx: f32, dy: f32, dz: f32, component: usize, expected: f32) {
    struct SpySurface { component: usize }
    impl Manifold<PathJet4> for SpySurface {
        type Output = Field;
        fn eval(&self, p: PathJet4) -> Self::Output {
            let (x, y, z, _w) = p;
            match self.component {
                0 => x.dir.val,
                1 => y.dir.val,
                _ => z.dir.val,
            }
        }
    }

    let reflect = Reflect { inner: SpySurface { component } };

    // Setup for horizontal plane N=(0,1,0)
    let px = PathJet {
        val: jet3(0.0, 1.0, 0.0, 0.0),
        dir: jet3(dx, 0.0, 0.0, 0.0)
    };
    let py = PathJet {
        val: jet3(0.0, 0.0, 0.0, 0.0),
        dir: jet3(dy, 0.0, 0.0, 0.0)
    };
    let pz = PathJet {
        val: jet3(0.0, 0.0, 1.0, 0.0),
        dir: jet3(dz, 0.0, 0.0, 0.0)
    };
    let pw = PathJet {
        val: constant_jet3(1.0),
        dir: constant_jet3(0.0)
    };

    let res_jet = reflect.eval((px, py, pz, pw)); // Returns Field directly (no Jet wrapper from Reflect)

    let res = read_field(res_jet);
    assert!((res - expected).abs() < 2e-3, "Component {}: expected {}, got {}", component, expected, res);
}
