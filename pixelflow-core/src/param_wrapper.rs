//! Parameter wrapper: convert f32 scalar parameters to any algebra via Manifold.
//!
//! When kernels have scalar parameters (e.g., `|tx: f32, ty: f32|`), they need to
//! participate in generic domain computations. `F32Param` wraps an f32 value and
//! implements `Manifold<P>` for any domain P where `P::Coord: Computational`.
//!
//! This enables seamless conversion of f32 parameters to the target algebra at
//! evaluation time, without requiring domain-specific code or explicit type annotations.
//!
//! # Example
//!
//! ```ignore
//! // Kernel with f32 parameters stored as F32Param wrapper
//! struct MyKernel {
//!     radius: F32Param,
//! }
//!
//! impl<P> Manifold<P> for MyKernel
//! where
//!     P: Spatial,
//!     P::Coord: Computational,
//! {
//!     type Output = P::Coord;
//!     fn eval(&self, p: P) -> P::Coord {
//!         let radius = self.radius.eval(p);  // Automatically converts f32 to P::Coord
//!         (X * X + Y * Y).sqrt() - radius
//!     }
//! }
//! ```

use crate::{Manifold, Spatial, Computational, Numeric, ext::ManifoldExpr};

/// Wraps an f32 scalar parameter for use in generic domain kernels.
///
/// `F32Param` is a zero-sized wrapper (due to `Copy`) that stores an f32 value and
/// implements `Manifold<P>` for any domain P. When evaluated, it converts the f32
/// to `P::Coord` using `Computational::from_f32()`.
///
/// This enables kernels to work with any algebra without explicit type annotations
/// or domain-specific code.
#[derive(Copy, Clone, Debug)]
pub struct F32Param(pub f32);

impl F32Param {
    /// Create a new f32 parameter wrapper.
    #[inline(always)]
    pub const fn new(value: f32) -> Self {
        Self(value)
    }
}

impl ManifoldExpr for F32Param {}

impl<P> Manifold<P> for F32Param
where
    P: Spatial + Copy + Send + Sync,
    P::Coord: Computational + Numeric + Copy + Send + Sync,
{
    type Output = P::Coord;

    #[inline(always)]
    fn eval(&self, _: P) -> P::Coord {
        P::Coord::from_f32(self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Field;

    #[test]
    fn test_f32_param_with_field() {
        let param = F32Param::new(5.0);
        let domain = (Field::from(1.0), Field::from(2.0));
        let result = param.eval(domain);

        // Result should be a Field with all lanes = 5.0
        let mut buf = [0.0f32; crate::PARALLELISM];
        result.store(&mut buf);
        assert_eq!(buf[0], 5.0);
    }
}
