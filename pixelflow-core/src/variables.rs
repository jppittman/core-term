//! # Coordinate Variables
//!
//! The base coordinate manifolds: X, Y, Z, W.
//! Also defines the `Axis` enum for 4D topology.

use crate::{Field, Manifold};

/// The explicit 4D axes of the manifold topology.
/// Used for indexing `Vector` outputs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Axis {
    /// The first dimension (X / Red).
    X,
    /// The second dimension (Y / Green).
    Y,
    /// The third dimension (Z / Blue).
    Z,
    /// The fourth dimension (W / Alpha).
    W,
}

// Coordinate Manifolds (Variables)

/// The X coordinate (Index 0).
#[derive(Clone, Copy, Debug, Default)]
pub struct X;

/// The Y coordinate (Index 1).
#[derive(Clone, Copy, Debug, Default)]
pub struct Y;

/// The Z coordinate (Index 2).
#[derive(Clone, Copy, Debug, Default)]
pub struct Z;

/// The W coordinate (Index 3).
#[derive(Clone, Copy, Debug, Default)]
pub struct W;

/// A marker trait for types that represent a static Axis.
pub trait Dimension {
    /// The axis this type represents.
    const AXIS: Axis;
}

impl Dimension for X {
    const AXIS: Axis = Axis::X;
}
impl Dimension for Y {
    const AXIS: Axis = Axis::Y;
}
impl Dimension for Z {
    const AXIS: Axis = Axis::Z;
}
impl Dimension for W {
    const AXIS: Axis = Axis::W;
}

/// Generate a scalar Manifold impl that returns Field.
macro_rules! impl_scalar_manifold {
    ($ty:ty, |$x:ident, $y:ident, $z:ident, $w:ident| $body:expr) => {
        impl Manifold for $ty {
            type Output = Field;
            #[inline(always)]
            fn eval_raw(&self, $x: Field, $y: Field, $z: Field, $w: Field) -> Field {
                $body
            }
        }
    };
}

impl_scalar_manifold!(X, |x, _y, _z, _w| x);
impl_scalar_manifold!(Y, |_x, y, _z, _w| y);
impl_scalar_manifold!(Z, |_x, _y, z, _w| z);
impl_scalar_manifold!(W, |_x, _y, _z, w| w);
