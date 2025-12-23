//! Parallel rasterization with thread spawning.

use pixelflow_core::Manifold;
use pixelflow_graphics::render::color::Pixel;
use pixelflow_graphics::render::rasterizer::{execute, execute_stripe, Stripe, TensorShape};
use pixelflow_graphics::Discrete;

/// Parallel rendering options.
#[derive(Copy, Clone, Debug)]
pub struct RenderOptions {
    /// Number of threads to use for rasterization.
    pub num_threads: usize,
}

impl Default for RenderOptions {
    fn default() -> Self {
        Self { num_threads: 1 }
    }
}

/// Render with parallel rasterization.
pub fn render_parallel<P, M>(
    manifold: &M,
    buffer: &mut [P],
    shape: TensorShape,
    options: RenderOptions,
) where
    P: Pixel + Send,
    M: Manifold<Output = Discrete> + Sync,
{
    if options.num_threads <= 1 {
        execute(manifold, buffer, shape);
        return;
    }

    // Partitioning
    let rows_per_thread = shape.height / options.num_threads;
    let remainder = shape.height % options.num_threads;

    // Split buffer into disjoint slices - Rust can verify safety!
    let mut buffer_chunks = Vec::with_capacity(options.num_threads);
    let mut remaining = buffer;
    let mut start_y = 0;

    for i in 0..options.num_threads {
        let extra = if i < remainder { 1 } else { 0 };
        let rows = rows_per_thread + extra;
        let end_y = start_y + rows;
        let stripe_len = rows * shape.width;

        let (chunk, rest) = remaining.split_at_mut(stripe_len);
        buffer_chunks.push((chunk, start_y, end_y));
        remaining = rest;
        start_y = end_y;
    }

    // Use scope to spawn threads with borrowed data
    std::thread::scope(|s| {
        for (chunk, start_y, end_y) in buffer_chunks {
            s.spawn(move || {
                execute_stripe(manifold, chunk, shape.width, Stripe { start_y, end_y });
            });
        }
    });
}
