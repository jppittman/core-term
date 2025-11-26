//! Functional pixel pipeline system.
//!
//! This module defines the core abstraction for zero-copy, composable image processing.
//! A `Surface` is a pure function `(x, y) -> Batch<T>` that can be composed into
//! complex compute graphs. The Rust compiler fuses these compositions via monomorphization
//! into a single tight SIMD loop with no function call overhead.

use crate::Batch;

/// A functional pixel source: `(x, y) -> Batch<T>`.
///
/// This is the fundamental building block of our compile-time compute graph.
/// All image operations (sampling, transformation, blending) implement this trait.
///
/// # Performance
///
/// - `Surface` is `Copy`, so graph nodes are passed by value on the stack
/// - Generic composition via `impl Surface` enables monomorphization
/// - The compiler inlines entire pipelines into a single fused loop
///
/// # Examples
///
/// ```ignore
/// // A constant white surface
/// let white = |_x, _y| Batch::splat(255u8);
///
/// // Compose surfaces
/// let blended = Over {
///     mask: sampler,
///     fg: 0xFFFFFFFF,
///     bg: 0x00000000,
/// };
/// ```
pub trait Surface<T: Copy>: Copy {
    /// Evaluate the surface at the given coordinates.
    ///
    /// Returns a batch of 4 pixel values (one per SIMD lane).
    fn eval(&self, x: Batch<u32>, y: Batch<u32>) -> Batch<T>;
}

/// Blanket implementation for closures.
///
/// This allows lambda functions to be used as surfaces:
///
/// ```ignore
/// let checker = |x: Batch<u32>, y: Batch<u32>| (x ^ y) & Batch::splat(1);
/// execute(checker, &mut target);
/// ```
impl<F, T: Copy> Surface<T> for F
where
    F: Fn(Batch<u32>, Batch<u32>) -> Batch<T> + Copy,
{
    #[inline(always)]
    fn eval(&self, x: Batch<u32>, y: Batch<u32>) -> Batch<T> {
        self(x, y)
    }
}
