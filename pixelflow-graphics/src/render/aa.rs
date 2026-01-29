//! Automatic antialiasing using gradient-based coverage.
//!
//! Uses Jet2 automatic differentiation to compute exact pixel coverage.
//! The gradient tells us how fast the SDF changes per pixel, giving us
//! the exact coverage without any smoothstep hacks.

use pixelflow_core::jet::Jet2;
use pixelflow_core::{Computational, Field, Manifold, ManifoldCompat, ManifoldExt};

/// The standard 4D Field domain type.
type Field4 = (Field, Field, Field, Field);

const MIN_GRADIENT: f32 = 0.001;

/// Computes antialiased coverage using gradient information from Jet2.
///
/// Generic over input coordinate types - works with any manifold that returns Jet2:
/// - `sdf / grad_mag` gives distance in pixels to the edge
/// - Coverage is linear: 0.5 at edge, 1.0 at +0.5px inside, 0.0 at -0.5px outside
///
/// No smoothstep. The gradient IS the antialiasing.
pub fn aa_coverage<C, M>(manifold: M) -> AACoverage<M, C>
where
    M: ManifoldCompat<C, Output = Jet2> + Send + Sync,
    C: Computational + From<Field> + Send + Sync + 'static,
{
    AACoverage {
        manifold,
        _phantom: std::marker::PhantomData,
    }
}

/// Antialiased coverage manifold using gradients from Jet2.
#[derive(Clone, Debug)]
pub struct AACoverage<M, C = Field> {
    pub manifold: M,
    _phantom: std::marker::PhantomData<C>,
}

impl<C, M> Manifold<Field4> for AACoverage<M, C>
where
    M: ManifoldCompat<C, Output = Jet2> + Send + Sync,
    C: Computational + From<Field> + Send + Sync + 'static,
{
    type Output = Field;

    fn eval(&self, p: Field4) -> Field {
        let (x, y, z, w) = p;
        // Convert Field coordinates to the manifold's coordinate type and evaluate
        let result = self
            .manifold
            .eval_raw(C::from(x), C::from(y), C::from(z), C::from(w));

        // Gradient magnitude = how many SDF units per pixel
        let grad_mag = (result.dx * result.dx + result.dy * result.dy).sqrt();

        // Distance to edge in pixels = sdf / grad_mag
        // Coverage: 0.5 at edge, +0.5 adds 1.0, -0.5 subtracts 1.0
        // Linear ramp over 1 pixel centered on edge
        let coverage = result.val / (grad_mag + Field::from(MIN_GRADIENT)) + Field::from(0.5);

        // Clamp to [0, 1] - collapse AST to Field
        coverage.max(0.0f32).min(1.0f32).at(x, y, z, w).collapse()
    }
}
