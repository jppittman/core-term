//! # Coordinate Variables
//!
//! The base coordinate manifolds: X, Y, Z, W.

use crate::{Field, Manifold};

/// The X coordinate.
#[derive(Clone, Copy, Debug, Default)]
pub struct X;

/// The Y coordinate.
#[derive(Clone, Copy, Debug, Default)]
pub struct Y;

/// The Z coordinate.
#[derive(Clone, Copy, Debug, Default)]
pub struct Z;

/// The W coordinate (time/extra dimension).
#[derive(Clone, Copy, Debug, Default)]
pub struct W;

/// Generate a scalar Manifold impl that returns Field.
macro_rules! impl_scalar_manifold {
    ($ty:ty, |$x:ident, $y:ident, $z:ident, $w:ident| $body:expr) => {
        impl Manifold for $ty {
            type Output = Field;
            #[inline(always)]
            fn eval(&self, $x: Field, $y: Field, $z: Field, $w: Field) -> Field {
                $body
            }
        }
    };
}

impl_scalar_manifold!(X, |x, _y, _z, _w| x);
impl_scalar_manifold!(Y, |_x, y, _z, _w| y);
impl_scalar_manifold!(Z, |_x, _y, z, _w| z);
impl_scalar_manifold!(W, |_x, _y, _z, w| w);
