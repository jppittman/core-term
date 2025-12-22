//! # Software Rasterizer
//!
//! Bridges Manifolds (continuous) to Framebuffers (discrete).
//! Uses `pixelflow_core::materialize_vector` for the SoA → AoS transpose.

use crate::render::color::{ColorVector, Pixel};
use pixelflow_core::{materialize_vector, Field, Manifold, PARALLELISM};

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
pub fn execute<P, M>(manifold: &M, target: &mut [P], shape: TensorShape)
where
    P: Pixel,
    M: Manifold<Output = ColorVector> + ?Sized,
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
    M: Manifold<Output = ColorVector> + ?Sized,
{
    // Buffer for interleaved f32 RGBA values from materialize_vector
    let mut interleaved = [0.0f32; PARALLELISM * 4];

    for (row_idx, y) in (stripe.start_y..stripe.end_y).enumerate() {
        let row_offset = row_idx * width;
        let mut x = 0;

        // SIMD Hot Path - process PARALLELISM pixels at a time
        while x + PARALLELISM <= width {
            let fx = x as f32 + 0.5;
            let fy = y as f32 + 0.5;

            // Use core's materialize_vector which does SoA → AoS transpose
            materialize_vector(manifold, fx, fy, &mut interleaved);

            // Convert interleaved f32 RGBA to pixel format
            for i in 0..PARALLELISM {
                let base = i * 4;
                target[row_offset + x + i] = P::from_rgba(
                    interleaved[base],
                    interleaved[base + 1],
                    interleaved[base + 2],
                    interleaved[base + 3],
                );
            }

            x += PARALLELISM;
        }

        // Scalar Fallback (Tail) - handle remaining pixels one at a time
        while x < width {
            let fx = x as f32 + 0.5;
            let fy = y as f32 + 0.5;

            // Evaluate single pixel
            let color = manifold.eval_raw(
                Field::from(fx),
                Field::from(fy),
                Field::from(0.0),
                Field::from(0.0),
            );

            // For a single pixel, just read first lane (all lanes have same value)
            materialize_vector(manifold, fx, fy, &mut interleaved);
            target[row_offset + x] = P::from_rgba(
                interleaved[0],
                interleaved[1],
                interleaved[2],
                interleaved[3],
            );

            x += 1;
        }
    }
}
