//! Unit tests for Surface kernel logic in scene3d.
//! Targets boundary conditions and logic validation.

use pixelflow_core::jet::Jet3;
use pixelflow_core::{Field, Manifold, ManifoldCompat, ManifoldExt};
use pixelflow_graphics::scene3d::Surface;
use pixelflow_macros::ManifoldExpr;

type Jet3_4 = (Jet3, Jet3, Jet3, Jet3);
type Field4 = (Field, Field, Field, Field);

// ============================================================================
// Mocks
// ============================================================================

#[derive(Clone, Copy, ManifoldExpr)]
struct MockGeometry {
    t: f32,
    dx: f32,
    dy: f32,
    dz: f32,
}

impl Manifold<Jet3_4> for MockGeometry {
    type Output = Jet3;
    fn eval(&self, _: Jet3_4) -> Jet3 {
        Jet3::new(
            Field::from(self.t),
            Field::from(self.dx),
            Field::from(self.dy),
            Field::from(self.dz),
        )
    }
}

#[derive(Clone, Copy, ManifoldExpr)]
struct MockMaterial {
    val: f32,
}

impl Manifold<Jet3_4> for MockMaterial {
    type Output = Field;
    fn eval(&self, _: Jet3_4) -> Field {
        Field::from(self.val)
    }
}
// Also implement Field4 for flexibility if needed, but Jet3_4 is what Surface calls
impl Manifold<Field4> for MockMaterial {
    type Output = Field;
    fn eval(&self, _: Field4) -> Field {
        Field::from(self.val)
    }
}

#[derive(Clone, Copy, ManifoldExpr)]
struct MockBackground {
    val: f32,
}

impl Manifold<Jet3_4> for MockBackground {
    type Output = Field;
    fn eval(&self, _: Jet3_4) -> Field {
        Field::from(self.val)
    }
}

// ============================================================================
// Helpers
// ============================================================================

fn assert_field_eq(f: Field, expected: f32) {
    let diff = (f - Field::from(expected)).abs();
    // Use a small epsilon for float comparison
    let mask = diff.lt(Field::from(1e-5));
    // Evaluate at dummy coords
    let zero = Field::from(0.0);
    let coords = (zero, zero, zero, zero);
    let result_mask = mask.eval(coords);

    assert!(result_mask.all(), "Field value mismatch. Expected: {}, Got logic check failed", expected);
}

// ============================================================================
// Tests
// ============================================================================

#[test]
fn surface_should_hit_when_t_is_positive_and_valid() {
    let surface = Surface {
        geometry: MockGeometry { t: 5.0, dx: 0.0, dy: 0.0, dz: 0.0 },
        material: MockMaterial { val: 100.0 },
        background: MockBackground { val: 0.0 },
    };

    let zero = Jet3::constant(Field::from(0.0));
    let coords = (zero, zero, zero, zero);

    // Should hit material (100.0)
    assert_field_eq(surface.eval(coords), 100.0);
}

#[test]
fn surface_should_miss_when_t_is_zero() {
    let surface = Surface {
        geometry: MockGeometry { t: 0.0, dx: 0.0, dy: 0.0, dz: 0.0 },
        material: MockMaterial { val: 100.0 },
        background: MockBackground { val: 0.0 },
    };

    let zero = Jet3::constant(Field::from(0.0));
    let coords = (zero, zero, zero, zero);

    // Should miss and hit background (0.0)
    // If logic is >= 0.0, this will fail (it would return 100.0)
    assert_field_eq(surface.eval(coords), 0.0);
}

#[test]
fn surface_should_miss_when_t_is_negative() {
    let surface = Surface {
        geometry: MockGeometry { t: -1.0, dx: 0.0, dy: 0.0, dz: 0.0 },
        material: MockMaterial { val: 100.0 },
        background: MockBackground { val: 0.0 },
    };

    let zero = Jet3::constant(Field::from(0.0));
    let coords = (zero, zero, zero, zero);

    // Should miss and hit background (0.0)
    assert_field_eq(surface.eval(coords), 0.0);
}

#[test]
fn surface_should_miss_when_t_exceeds_max() {
    // t_max is 1,000,000.0 in scene3d.rs
    let surface = Surface {
        geometry: MockGeometry { t: 1_000_000.1, dx: 0.0, dy: 0.0, dz: 0.0 },
        material: MockMaterial { val: 100.0 },
        background: MockBackground { val: 0.0 },
    };

    let zero = Jet3::constant(Field::from(0.0));
    let coords = (zero, zero, zero, zero);

    // Should miss and hit background (0.0)
    assert_field_eq(surface.eval(coords), 0.0);
}

#[test]
fn surface_should_hit_when_t_is_just_below_max() {
    let surface = Surface {
        geometry: MockGeometry { t: 999_999.0, dx: 0.0, dy: 0.0, dz: 0.0 },
        material: MockMaterial { val: 100.0 },
        background: MockBackground { val: 0.0 },
    };

    let zero = Jet3::constant(Field::from(0.0));
    let coords = (zero, zero, zero, zero);

    // Should hit material (100.0)
    assert_field_eq(surface.eval(coords), 100.0);
}

#[test]
fn surface_should_miss_when_derivatives_exceed_limit() {
    // deriv_max is 10,000.0. deriv_mag_sq < deriv_max^2.
    // If dx = 10000.0, dx^2 = 10000^2.
    // If dy=0, dz=0. mag_sq = 10000^2.
    // Condition is < (not <=). So 10000.0 should miss.

    let surface = Surface {
        geometry: MockGeometry { t: 5.0, dx: 10000.0, dy: 0.0, dz: 0.0 },
        material: MockMaterial { val: 100.0 },
        background: MockBackground { val: 0.0 },
    };

    let zero = Jet3::constant(Field::from(0.0));
    let coords = (zero, zero, zero, zero);

    // Should miss and hit background (0.0)
    assert_field_eq(surface.eval(coords), 0.0);
}

#[test]
fn surface_should_hit_when_derivatives_are_within_limit() {
    let surface = Surface {
        geometry: MockGeometry { t: 5.0, dx: 9999.0, dy: 0.0, dz: 0.0 },
        material: MockMaterial { val: 100.0 },
        background: MockBackground { val: 0.0 },
    };

    let zero = Jet3::constant(Field::from(0.0));
    let coords = (zero, zero, zero, zero);

    // Should hit material (100.0)
    assert_field_eq(surface.eval(coords), 100.0);
}

#[test]
fn surface_should_eval_material_at_hit_point() {
    // We want to verify that material is evaluated at P = ray * t
    // Ray comes from inputs (rx, ry, rz). t comes from geometry.
    // Let's set ray to (1, 0, 0) and t to 2.0. P should be (2, 0, 0).
    // Material can evaluate X+Y+Z. If result is 2.0, then it's correct.

    #[derive(Clone, Copy, ManifoldExpr)]
    struct CheckPosMaterial;
    impl Manifold<Jet3_4> for CheckPosMaterial {
        type Output = Field;
        fn eval(&self, p: Jet3_4) -> Field {
            let (x, y, z, _w) = p;
            let sum = x.val + y.val + z.val;
            let zero = Field::from(0.0);
            let dummy = (zero, zero, zero, zero);
            sum.eval(dummy)
        }
    }

    let surface = Surface {
        geometry: MockGeometry { t: 2.0, dx: 0.0, dy: 0.0, dz: 0.0 },
        material: CheckPosMaterial,
        background: MockBackground { val: -1.0 },
    };

    // Input Ray: X=1, Y=0, Z=0.
    let one = Jet3::constant(Field::from(1.0));
    let zero = Jet3::constant(Field::from(0.0));
    let coords = (one, zero, zero, zero); // (rx, ry, rz, w)

    // Expected: P = (1*2, 0*2, 0*2) = (2, 0, 0).
    // Material eval: 2 + 0 + 0 = 2.0.

    assert_field_eq(surface.eval(coords), 2.0);
}
