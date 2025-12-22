//! Automatic antialiasing using gradient-based smoothing.
//!
//! Uses Jet2 automatic differentiation to compute gradient magnitude,
//! which determines the proper antialiasing width per pixel.

use pixelflow_core::{BoxedManifold, Field, Jet2, Manifold, ManifoldExt};

const MIN_GRADIENT: f32 = 0.001;

/// Computes antialiased coverage using automatic gradient calculation.
///
/// For a signed distance field, the gradient magnitude ‖∇f‖ tells us how fast
/// the field is changing. To smooth over exactly 1 pixel, we use AA width = 1 / ‖∇f‖.
///
/// **No magic numbers** - the derivative determines the exact AA width.
pub fn aa_coverage<M>(manifold: M) -> BoxedManifold
where
    M: Manifold<Jet2, Output = Jet2> + Manifold<Field, Output = Field> + Copy + 'static,
{
    AACoverage { manifold }.boxed()
}

/// Antialiased coverage manifold using automatic gradients.
#[derive(Copy, Clone)]
struct AACoverage<M> {
    manifold: M,
}

impl<M> Manifold for AACoverage<M>
where
    M: Manifold<Jet2, Output = Jet2> + Manifold<Field, Output = Field>,
{
    type Output = Field;

    fn eval_raw(&self, x: Field, y: Field, z: Field, w: Field) -> Field {
        // Evaluate with Jet2 to extract gradient
        let x_jet = Jet2::x(x);
        let y_jet = Jet2::y(y);
        let z_jet = Jet2::constant(z);
        let w_jet = Jet2::constant(w);

        let result = self.manifold.eval_raw(x_jet, y_jet, z_jet, w_jet);

        // Gradient magnitude: ‖∇f‖ = sqrt(dx² + dy²)
        let grad_sq = (result.dx * result.dx + result.dy * result.dy).eval_raw(x, y, z, w);
        let grad_mag = grad_sq.sqrt().eval_raw(x, y, z, w);

        // AA width = 1 pixel / gradient magnitude (cache to avoid recomputation)
        let aa_width =
            (Field::from(1.0) / (grad_mag + Field::from(MIN_GRADIENT))).eval_raw(x, y, z, w);

        // Smooth step from -width to +width
        let edge0 = (aa_width * Field::from(-1.0)).eval_raw(x, y, z, w);
        let edge1 = aa_width;

        // Clamp t to [0, 1]
        let range = (edge1 - edge0).eval_raw(x, y, z, w);
        let t_unclamped = ((result.val - edge0) / range).eval_raw(x, y, z, w);
        let t = (t_unclamped.max(Field::from(0.0)).min(Field::from(1.0))).eval_raw(x, y, z, w);

        // Hermite interpolation: 3t² - 2t³
        let t2 = (t * t).eval_raw(x, y, z, w);
        let t3 = (t2 * t).eval_raw(x, y, z, w);
        (t3 * Field::from(-2.0) + t2 * Field::from(3.0)).eval_raw(x, y, z, w)
    }
}
