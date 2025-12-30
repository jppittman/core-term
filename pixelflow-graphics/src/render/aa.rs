//! Automatic antialiasing using gradient-based coverage.
//!
//! Uses Jet2 automatic differentiation to compute exact pixel coverage.
//! The gradient tells us how fast the SDF changes per pixel, giving us
//! the exact coverage without any smoothstep hacks.

use pixelflow_core::jet::Jet2;
use pixelflow_core::{BoxedManifold, Field, Manifold, ManifoldExt};

const MIN_GRADIENT: f32 = 0.001;

/// Computes antialiased coverage using automatic gradient calculation.
///
/// For a signed distance field:
/// - `sdf / grad_mag` gives distance in pixels to the edge
/// - Coverage is linear: 0.5 at edge, 1.0 at +0.5px inside, 0.0 at -0.5px outside
///
/// No smoothstep. The gradient IS the antialiasing.
pub fn aa_coverage<M>(manifold: M) -> BoxedManifold
where
    M: Manifold<Jet2, Output = Jet2> + Manifold<Field, Output = Field> + Clone + 'static,
{
    AACoverage { manifold }.boxed()
}

/// Antialiased coverage manifold using automatic gradients.
#[derive(Clone, Debug)]
pub struct AACoverage<M> {
    pub manifold: M,
}

impl<M> Manifold for AACoverage<M>
where
    M: Manifold<Jet2, Output = Jet2> + Manifold<Field, Output = Field>,
{
    type Output = Field;

    fn eval_raw(&self, x: Field, y: Field, z: Field, w: Field) -> Field {
        // Evaluate SDF with Jet2 to get value AND gradient in one pass
        let result =
            self.manifold
                .eval_raw(Jet2::x(x), Jet2::y(y), Jet2::constant(z), Jet2::constant(w));

        // Gradient magnitude = how many SDF units per pixel
        let grad_mag = (result.dx * result.dx + result.dy * result.dy).sqrt();

        // Distance to edge in pixels = sdf / grad_mag
        // Coverage: 0.5 at edge, +0.5 adds 1.0, -0.5 subtracts 1.0
        // Linear ramp over 1 pixel centered on edge
        let coverage = result.val / (grad_mag + Field::from(MIN_GRADIENT)) + Field::from(0.5);

        // Clamp to [0, 1] - collapse AST to Field
        coverage.max(0.0f32).min(1.0f32).at(x, y, z, w).eval()
    }
}
