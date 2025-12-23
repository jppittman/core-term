//! # Coordinate Variables
//!
//! The base coordinate manifolds: X, Y, Z, W.
//! Also defines the `Axis` enum for 4D topology.

use crate::Manifold;

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

// Variables are polymorphic - they return their coordinate unchanged
impl<I: crate::numeric::Numeric> Manifold<I> for X {
    type Output = I;
    #[inline(always)]
    fn eval_raw(&self, x: I, _y: I, _z: I, _w: I) -> I {
        x
    }
}

impl<I: crate::numeric::Numeric> Manifold<I> for Y {
    type Output = I;
    #[inline(always)]
    fn eval_raw(&self, _x: I, y: I, _z: I, _w: I) -> I {
        y
    }
}

impl<I: crate::numeric::Numeric> Manifold<I> for Z {
    type Output = I;
    #[inline(always)]
    fn eval_raw(&self, _x: I, _y: I, z: I, _w: I) -> I {
        z
    }
}

impl<I: crate::numeric::Numeric> Manifold<I> for W {
    type Output = I;
    #[inline(always)]
    fn eval_raw(&self, _x: I, _y: I, _z: I, w: I) -> I {
        w
    }
}
