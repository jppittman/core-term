use crate::backend::SimdBatch;
use crate::batch::Batch;

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
