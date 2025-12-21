//! # Vector Operations
//!
//! Traits for types that behave as vectors (compound values).

use crate::variables::Axis;

/// A trait for types that have indexable components via the standard 4D Axis topology.
///
/// This allows a Manifold to produce a "Vector" (like a Color or Position)
/// which can then be decomposed by the `Project` combinator using `Axis`.
pub trait Vector {
    /// The type of the components (usually `Field`).
    type Component;

    /// Extract the component at the given Axis.
    fn get(&self, axis: Axis) -> Self::Component;
}
