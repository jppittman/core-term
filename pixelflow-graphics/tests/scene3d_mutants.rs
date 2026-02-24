use pixelflow_core::jet::{Jet3, PathJet};
use pixelflow_core::{Field, Manifold, ManifoldExpr};
use pixelflow_graphics::scene3d::{
    PathJetSphere, Reflect, Surface, UnitSphere,
};

type Jet3_4 = (Jet3, Jet3, Jet3, Jet3);

fn zero_jet() -> Jet3 {
    Jet3::constant(Field::from(0.0))
}

fn dummy_p() -> (Field, Field, Field, Field) {
    let z = Field::from(0.0);
    (z, z, z, z)
}

fn assert_field_approx_eq(val: Field, expected: Field, epsilon: f32, msg: &str) {
    // val and expected are Fields. Subtraction creates an AST.
    // We must evaluate the AST to get a Field result.
    let diff_expr = val - expected;
    let diff = diff_expr.eval(dummy_p());
    let eps = Field::from(epsilon);

    // Check if diff < epsilon
    let ok = diff.abs().lt(eps).all();

    // For debug printing, we can't easily extract f32 from Field without store() which is crate-private.
    // So we rely on Debug impl of Field if available, or just the assertion failure.
    assert!(ok, "{}: val={:?}, expected={:?}", msg, val, expected);
}

// ============================================================================
// UnitSphere Tests
// ============================================================================

#[test]
fn unit_sphere_should_intersect_ray_at_distance_one_when_ray_origin_is_center() {
    // UnitSphere assumes ray origin is (0,0,0).
    // Ray direction: along Z axis (0, 0, 1)
    let rx = Jet3::constant(Field::from(0.0));
    let ry = Jet3::constant(Field::from(0.0));
    let rz = Jet3::constant(Field::from(1.0));
    let rw = Jet3::constant(Field::from(1.0));

    let sphere = UnitSphere;
    let t = sphere.eval((rx, ry, rz, rw));

    // t = 1 / |ray| = 1 / 1 = 1
    // Use relaxed epsilon due to fast rsqrt approximation
    assert_field_approx_eq(t.val, Field::from(1.0), 1e-3, "t should be 1.0");
}

#[test]
fn unit_sphere_should_handle_non_normalized_rays() {
    // Ray direction: (0, 0, 2)
    let rx = Jet3::constant(Field::from(0.0));
    let ry = Jet3::constant(Field::from(0.0));
    let rz = Jet3::constant(Field::from(2.0));
    let rw = Jet3::constant(Field::from(1.0));

    let sphere = UnitSphere;
    let t = sphere.eval((rx, ry, rz, rw));

    // t = 1 / |ray| = 1 / 2 = 0.5
    assert_field_approx_eq(t.val, Field::from(0.5), 1e-3, "t should be 0.5");
}

// ============================================================================
// PathJetSphere Tests
// ============================================================================

#[test]
fn path_jet_sphere_should_intersect_off_center_ray_correctly() {
    // Sphere at (0, 0, 5) radius 1
    let sphere = PathJetSphere::new((0.0, 0.0, 5.0), 1.0);

    // Ray origin (0, 0, 0), direction (0, 0, 1)
    let origin = Jet3::constant(Field::from(0.0));
    let dir_z = Jet3::constant(Field::from(1.0));

    // PathJet inputs
    let px = PathJet { val: origin.clone(), dir: origin.clone() }; // x=0, dx=0
    let py = PathJet { val: origin.clone(), dir: origin.clone() }; // y=0, dy=0
    let pz = PathJet { val: origin.clone(), dir: dir_z.clone() };  // z=0, dz=1
    let pw = PathJet { val: origin.clone(), dir: origin.clone() };

    let t = sphere.eval((px, py, pz, pw));

    // Analytical solution:
    // Ray P(t) = (0, 0, t)
    // Sphere |P - C|^2 = r^2 => (t - 5)^2 = 1^2 => t - 5 = +/- 1 => t = 4 or 6
    // Expected t = 4 (closest intersection)

    assert_field_approx_eq(t.val, Field::from(4.0), 1e-3, "Expected t=4");
}

// ============================================================================
// Reflect Tests
// ============================================================================

// Mock material that returns one component of the ray direction (rx, ry, or rz)
#[derive(Clone, Copy)]
struct CoordinateInspector {
    component: usize, // 0=x, 1=y, 2=z
}

impl ManifoldExpr for CoordinateInspector {}

impl Manifold<Jet3_4> for CoordinateInspector {
    type Output = Field;
    fn eval(&self, p: Jet3_4) -> Field {
        let (x, y, z, _w) = p;
        match self.component {
            0 => x.val,
            1 => y.val,
            _ => z.val,
        }
    }
}

#[test]
fn reflect_should_compute_correct_reflection_vector_for_flat_surface() {
    // Scenario:
    // Hit point P on a surface.
    // Incident ray D coming from origin to P.
    // Normal N at P.
    // Reflect should compute R = D - 2(D.N)N.

    let f0 = Field::from(0.0);
    let f1 = Field::from(1.0);
    let fm1 = Field::from(-1.0);

    // Construct Jet3 inputs
    // x: val=1, dx=1 (Tu.x)
    // y: val=-1, dy=0
    // z: val=0, dy=-1 (Tv.z)

    let x = Jet3 { val: f1.clone(), dx: f1.clone(), dy: f0.clone(), dz: f0.clone() };
    let y = Jet3 { val: fm1.clone(), dx: f0.clone(), dy: f0.clone(), dz: f0.clone() };
    let z = Jet3 { val: f0.clone(), dx: f0.clone(), dy: fm1.clone(), dz: f0.clone() };
    let w = Jet3::constant(f1.clone());

    // Check X component
    let reflector_x = Reflect { inner: CoordinateInspector { component: 0 } };
    let rx = reflector_x.eval((x.clone(), y.clone(), z.clone(), w.clone()));

    // Check Y component
    let reflector_y = Reflect { inner: CoordinateInspector { component: 1 } };
    let ry = reflector_y.eval((x.clone(), y.clone(), z.clone(), w.clone()));

    // Check Z component
    let reflector_z = Reflect { inner: CoordinateInspector { component: 2 } };
    let rz = reflector_z.eval((x.clone(), y.clone(), z.clone(), w.clone()));

    let val_inv = (2.0f32).sqrt().recip();

    // Expect R = (val_inv, val_inv, 0)
    assert_field_approx_eq(rx, Field::from(val_inv), 1e-3, "Rx mismatch");
    assert_field_approx_eq(ry, Field::from(val_inv), 1e-3, "Ry mismatch");
    assert_field_approx_eq(rz, Field::from(0.0), 1e-3, "Rz mismatch");
}

// ============================================================================
// Surface Tests
// ============================================================================

// Mock geometry that returns specific t and derivatives
#[derive(Clone, Copy)]
struct MockGeometry {
    t: f32,
    dx: f32,
}

impl ManifoldExpr for MockGeometry {}

impl Manifold<Jet3_4> for MockGeometry {
    type Output = Jet3;
    fn eval(&self, _p: Jet3_4) -> Jet3 {
        Jet3 {
            val: Field::from(self.t),
            dx: Field::from(self.dx),
            dy: Field::from(0.0),
            dz: Field::from(0.0),
        }
    }
}

// Mock material returns 1.0, background returns 0.0
#[derive(Clone, Copy)]
struct WhiteMaterial;
impl ManifoldExpr for WhiteMaterial {}

impl Manifold<Jet3_4> for WhiteMaterial {
    type Output = Field;
    fn eval(&self, _p: Jet3_4) -> Field { Field::from(1.0) }
}

#[derive(Clone, Copy)]
struct BlackBackground;
impl ManifoldExpr for BlackBackground {}

impl Manifold<Jet3_4> for BlackBackground {
    type Output = Field;
    fn eval(&self, _p: Jet3_4) -> Field { Field::from(0.0) }
}

#[test]
fn surface_should_reject_hits_beyond_t_max() {
    let surface = Surface {
        geometry: MockGeometry { t: 2_000_000.0, dx: 1.0 },
        material: WhiteMaterial,
        background: BlackBackground,
    };

    let p = (zero_jet(), zero_jet(), zero_jet(), zero_jet());
    let res = surface.eval(p);

    assert_field_approx_eq(res, Field::from(0.0), 1e-6, "Should return background (0.0) for t > t_max");
}

#[test]
fn surface_should_reject_hits_with_excessive_derivatives() {
    let surface = Surface {
        geometry: MockGeometry { t: 5.0, dx: 20_000.0 }, // deriv > 10000
        material: WhiteMaterial,
        background: BlackBackground,
    };

    let p = (zero_jet(), zero_jet(), zero_jet(), zero_jet());
    let res = surface.eval(p);

    assert_field_approx_eq(res, Field::from(0.0), 1e-6, "Should return background (0.0) for deriv > deriv_max");
}

#[test]
fn surface_should_accept_valid_hits() {
    let surface = Surface {
        geometry: MockGeometry { t: 5.0, dx: 1.0 },
        material: WhiteMaterial,
        background: BlackBackground,
    };

    let p = (zero_jet(), zero_jet(), zero_jet(), zero_jet());
    let res = surface.eval(p);

    assert_field_approx_eq(res, Field::from(1.0), 1e-6, "Should return material (1.0) for valid hit");
}
