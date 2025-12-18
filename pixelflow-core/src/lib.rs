//! # PixelFlow Core
//!
//! A zero-cost, type-driven SIMD abstraction for pixel operations.

#![no_std]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

extern crate alloc;

/// SIMD backend abstractions.
pub mod backend;
/// SIMD batch types and operations.
pub mod batch;
pub mod bitwise;
/// Domain Specific Language for surface composition.
pub mod dsl;
/// Geometric primitives and transformations.
pub mod geometry;
/// Functional surface implementations and combinators.
pub mod surfaces;
/// Core rendering traits (Surface, Volume, Manifold).
pub mod traits;

// Re-exports
pub use backend::FloatBatchOps;
pub use batch::{Batch, BatchOps, SHUFFLE_RGBA_BGRA};
pub use bitwise::Bitwise;
pub use geometry::{Curve2D, Mat3, Poly, Rect};
pub use surfaces::{
    Baked, Compute, Grade, Implicit, Lerp, Map, Max, Offset, Partition, Scale, Select, Skew, W,
    Warp, X, Y, Z,
};
pub use traits::{Manifold, Surface, Volume};
