//! # PixelFlow Core
//!
//! A zero-cost, type-driven SIMD abstraction for pixel operations.

#![no_std]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

extern crate alloc;

pub mod backend;
pub mod batch;
/// Domain Specific Language extensions.
pub mod dsl;
/// Geometric primitives and math.
pub mod geometry;
/// Pixel formats and traits.
pub mod pixel;
/// Platform abstraction layer.
pub mod platform;
/// Rasterization and execution logic.
pub mod raster;
/// Surface implementations and combinators.
pub mod surfaces;
/// Core traits.
pub mod traits;

// Re-exports
pub use backend::{FloatBatchOps, SimdBatch};
pub use batch::{Batch, BatchOps, SHUFFLE_RGBA_BGRA};
pub use geometry::{Curve2D, Mat3, Poly, Rect};
pub use pixel::Pixel;
pub use platform::{PixelFormat, Platform};
pub use raster::{Tensor1x1, Tensor1x2, Tensor2x1, Tensor2x2, TensorView, TensorViewMut, execute};
pub use surfaces::{
    Baked, FnSurface, Implicit, Map, Offset, Over, Partition, SampleAtlas, Scale, Select, Skew,
};
pub use traits::{Surface, Volume};
