use crate::batch::Batch;

/// A pure functional surface: coordinates → values.
///
/// This is a "Lazy Array"—it doesn't own memory; it describes how to
/// compute a value at any coordinate. Surfaces are the universal
/// abstraction for anything that varies over 2D space.
pub trait Surface: Copy + Send + Sync {
    /// The output type of this surface.
    type Output: Copy;

    /// Sample the surface at the given coordinates.
    ///
    /// # Parameters
    /// * `u` - Horizontal coordinates (batch of floats)
    /// * `v` - Vertical coordinates (batch of floats)
    ///
    /// # Returns
    /// A batch of output values.
    fn sample(&self, u: Batch<f32>, v: Batch<f32>) -> Self::Output;
}

/// Primitive surface implementations (X, Y, etc.).
pub mod primitives;

pub use primitives::{X, Y};
