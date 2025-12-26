//! # Software Rasterizer
//!
//! Bridges Manifolds (continuous) to Framebuffers (discrete).
//! Color manifolds output `Discrete` (packed u32 pixels) directly.

use crate::render::color::Pixel;
use pixelflow_core::{
    materialize_discrete, materialize_discrete_fields, Discrete, Field, Manifold, PARALLELISM,
};

pub mod parallel;
pub mod pool;
pub use parallel::{render_parallel, render_parallel_pooled, RenderOptions};
pub use pool::ThreadPool;

/// A wrapper that adapts a continuous manifold for rasterization.
#[derive(Clone, Copy, Debug)]
pub struct Rasterize<M>(pub M);

impl<M: Manifold> Manifold for Rasterize<M> {
    type Output = M::Output;

    #[inline(always)]
    fn eval_raw(&self, x: Field, y: Field, z: Field, w: Field) -> Self::Output {
        self.0.eval_raw(x, y, z, w)
    }
}

/// Dimensions and memory layout of a 2D tensor.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct TensorShape {
    /// Width in pixels.
    pub width: usize,
    /// Height in pixels.
    pub height: usize,
}

impl TensorShape {
    /// Create a new TensorShape.
    pub const fn new(width: usize, height: usize) -> Self {
        Self { width, height }
    }
}

/// A horizontal stripe of rows for parallel rendering.
#[derive(Copy, Clone, Debug)]
pub struct Stripe {
    /// Starting Y coordinate.
    pub start_y: usize,
    /// Ending Y coordinate (exclusive).
    pub end_y: usize,
}

/// Software rasterizer entry point.
///
/// Takes a color manifold that outputs Discrete (packed u32 pixels)
/// and writes them to the target buffer.
pub fn execute<P, M>(manifold: &M, target: &mut [P], shape: TensorShape)
where
    P: Pixel,
    M: Manifold<Output = Discrete> + ?Sized,
{
    if shape.width == 0 || shape.height == 0 {
        return;
    }
    execute_stripe(
        manifold,
        target,
        shape.width,
        Stripe {
            start_y: 0,
            end_y: shape.height,
        },
    );
}

/// Render a specific row range into the target buffer.
pub fn execute_stripe<P, M>(manifold: &M, target: &mut [P], width: usize, stripe: Stripe)
where
    P: Pixel,
    M: Manifold<Output = Discrete> + ?Sized,
{
    let mut packed = [0u32; PARALLELISM];

    for (row_idx, y) in (stripe.start_y..stripe.end_y).enumerate() {
        let row_offset = row_idx * width;
        let mut x = 0;

        let ys = Field::from(y as f32 + 0.5);

        // SIMD Hot Path - process PARALLELISM pixels at a time
        while x + PARALLELISM <= width {
            let xs = Field::sequential(x as f32 + 0.5);

            // Use materialize_discrete_fields to evaluate and store
            materialize_discrete_fields(manifold, xs, ys, &mut packed);

            // Copy to target
            for i in 0..PARALLELISM {
                target[row_offset + x + i] = P::from_u32(packed[i]);
            }

            x += PARALLELISM;
        }

        // Scalar Fallback (Tail) - handle remaining pixels one at a time
        while x < width {
            let fx = x as f32 + 0.5;
            let fy = y as f32 + 0.5;

            // For single pixels, use materialize_discrete with a 1-element buffer
            materialize_discrete(manifold, fx, fy, &mut packed);
            target[row_offset + x] = P::from_u32(packed[0]);

            x += 1;
        }
    }
}
