use crate::backend::{Backend, SimdBatch};
use crate::batch::{Batch, NativeBackend};
use crate::pipe::Surface;

/// A 3x3 matrix for 2D affine transformations.
///
/// Layout:
/// [ m00 m01 m02 ]
/// [ m10 m11 m12 ]
/// [ m20 m21 m22 ]
#[derive(Clone, Copy, Debug)]
pub struct Mat3 {
    /// Row-major matrix elements.
    pub m: [[f32; 3]; 3],
}

impl Mat3 {
    /// Identity matrix.
    pub fn identity() -> Self {
        Self {
            m: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        }
    }

    /// Creates a transformation matrix that maps unit triangle to the given points.
    ///
    /// Maps:
    /// (0, 0) -> p0
    /// (0.5, 0) -> p1
    /// (1, 1) -> p2
    pub fn from_affine_points(p0: [f32; 2], p1: [f32; 2], p2: [f32; 2]) -> Option<Self> {
        let c = p0[0];
        let f = p0[1];

        let a = (p1[0] - c) * 2.0;
        let d = (p1[1] - f) * 2.0;

        let b = p2[0] - a - c;
        let e = p2[1] - d - f;

        let forward = Self {
            m: [[a, b, c], [d, e, f], [0.0, 0.0, 1.0]],
        };

        forward.inverse()
    }

    /// Computes the inverse of the matrix.
    pub fn inverse(&self) -> Option<Self> {
        NativeBackend::inverse_mat3(self.m).map(|m| Self { m })
    }

    /// Transforms screen coordinates (x, y) into parameter space (u, v).
    /// Note: Assumes homogenous coordinate w=1.
    pub fn transform(&self, x: Batch<u32>, y: Batch<u32>) -> (Batch<f32>, Batch<f32>) {
        let x_f = NativeBackend::u32_to_f32(x);
        let y_f = NativeBackend::u32_to_f32(y);

        let m00 = Batch::<f32>::splat(self.m[0][0]);
        let m01 = Batch::<f32>::splat(self.m[0][1]);
        let m02 = Batch::<f32>::splat(self.m[0][2]);

        let m10 = Batch::<f32>::splat(self.m[1][0]);
        let m11 = Batch::<f32>::splat(self.m[1][1]);
        let m12 = Batch::<f32>::splat(self.m[1][2]);

        // u = m00*x + m01*y + m02
        let u = m00 * x_f + m01 * y_f + m02;

        // v = m10*x + m11*y + m12
        let v = m10 * x_f + m11 * y_f + m12;

        (u, v)
    }
}

/// A polynomial curve. Degree is coefficients.len() - 1
pub struct Poly<C: AsRef<[f32]>> {
    /// Coefficients of the polynomial, from lowest to highest degree.
    pub coefficients: C,
}

impl<C: AsRef<[f32]>> Poly<C> {
    /// Evaluate at t using Horner's method
    pub fn eval(&self, t: Batch<f32>) -> Batch<f32> {
        let c = self.coefficients.as_ref();
        if c.is_empty() {
            return Batch::<f32>::splat(0.0);
        }
        let mut result = Batch::<f32>::splat(c[c.len() - 1]);
        for i in (0..c.len() - 1).rev() {
            result = result * t + Batch::<f32>::splat(c[i]);
        }
        result
    }
}

/// A 2D parametric curve: t → (x, y)
pub struct Curve2D<X, Y> {
    /// Polynomial for x(t)
    pub x: X,
    /// Polynomial for y(t)
    pub y: Y,
}

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
