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
pub mod pixel;
pub mod platform;
pub mod raster;
pub mod surfaces;
pub mod traits;

// Re-exports
pub use backend::{FloatBatchOps, SimdBatch};
pub use batch::{Batch, BatchOps, SHUFFLE_RGBA_BGRA};
pub use geometry::{Mat3, Poly, Curve2D, Rect};
pub use pixel::Pixel;
pub use platform::{PixelFormat, Platform};
pub use raster::{Tensor2x2, Tensor2x1, Tensor1x2, Tensor1x1, TensorView, TensorViewMut, execute, execute_stripe};
pub use surfaces::{Baked, FnSurface, Implicit, Map, Offset, Over, Partition, SampleAtlas, Scale, Select, Skew};
pub use traits::{Surface, Volume};
