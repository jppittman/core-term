//! # PixelFlow Core
//!
//! A zero-cost, type-driven SIMD abstraction for pixel operations.

#![no_std]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

extern crate alloc;

pub mod backend;
pub mod batch;
pub mod dsl;
pub mod geometry;
pub mod platform;
pub mod raster;
pub mod surfaces;
pub mod traits;

// Re-exports
pub use backend::{FloatBatchOps, SimdBatch};
pub use batch::{Batch, BatchOps, SHUFFLE_RGBA_BGRA};
pub use geometry::{Curve2D, Mat3, Poly, Rect};
pub use platform::{PixelFormat, Platform};
pub use raster::{TensorView, TensorViewMut, execute, execute_stripe};
pub use surfaces::{
    Baked, Compute, FnSurface, Grade, Implicit, Lerp, Map, Offset, Partition, SampleAtlas,
    Scale, Select, Skew, Warp,
};
pub use traits::{Manifold, Surface, Volume};
