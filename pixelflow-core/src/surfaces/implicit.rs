use crate::batch::Batch;
use crate::geometry::affine::Mat3;
use crate::traits::Surface;

/// Trait for defining the implicit equation of a curve.
pub trait ImplicitEvaluator {
    /// Evaluates the implicit function f(u, v) = 0.
    fn implicit_at(&self, u: Batch<f32>, v: Batch<f32>) -> Batch<f32>;
}

/// The implicit form: (x, y) → f where f=0 is the curve
pub struct Implicit<C> {
    /// The curve evaluator (e.g. quadratic or cubic).
    pub curve: C,
    /// Matrix projecting screen space to parameter space.
    pub projection: Mat3,
}

impl<C: ImplicitEvaluator + Send + Sync + 'static> Surface<f32> for Implicit<C> {
    fn eval(&self, x: Batch<u32>, y: Batch<u32>) -> Batch<f32> {
        // Warp: screen → (u, v)
        let (u, v) = self.projection.transform(x, y);
        // Evaluate implicit equation
        self.curve.implicit_at(u, v)
    }
}

// Common Implicit Evaluators

/// Quadratic Bézier implicit: u^2 - v
pub struct QuadraticImplicit;

impl ImplicitEvaluator for QuadraticImplicit {
    fn implicit_at(&self, u: Batch<f32>, v: Batch<f32>) -> Batch<f32> {
        u * u - v
    }
}

/// Cubic Bézier implicit approximation: u^3 - v
pub struct CubicImplicit;

impl ImplicitEvaluator for CubicImplicit {
    fn implicit_at(&self, u: Batch<f32>, v: Batch<f32>) -> Batch<f32> {
        u * u * u - v
    }
}
