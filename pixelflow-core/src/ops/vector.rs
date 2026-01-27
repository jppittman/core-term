//! # Vector Operations
//!
//! Traits for projectable types.

use crate::variables::Axis;

/// A marker trait for types that have indexable components via the 4D Axis topology.
///
/// Types implementing this trait can be used with the `Project` combinator
/// to extract individual components.
///
/// The actual component access is done via the `Projectable` trait.
pub trait Vector: Send + Sync + Copy {
    /// The type of the components (usually `Field`).
    type Component;

    /// Extract the component at the given Axis.
    fn get(&self, axis: Axis) -> Self::Component;
}
