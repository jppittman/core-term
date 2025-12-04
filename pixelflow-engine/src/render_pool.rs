//! Parallel rasterization with thread spawning.

use pixelflow_core::traits::Surface;
use pixelflow_render::Pixel;
use std::sync::Arc;

/// Render with parallel rasterization using the specified number of threads.
/// Falls back to single-threaded if num_threads <= 1.
pub fn render_parallel<P, S>(
    surface: &S,
    buffer: &mut [P],
    width: usize,
    height: usize,
    num_threads: usize,
) where
    P: Pixel + Send,
    S: Surface<P> + Sync,
{
    if num_threads <= 1 {
        pixelflow_core::execute(surface, buffer, width, height);
        return;
    }

    // Partitioning
    let rows_per_thread = height / num_threads;
    let remainder = height % num_threads;

    // Split buffer into disjoint slices - Rust can verify safety!
    let mut buffer_chunks = Vec::with_capacity(num_threads);
    let mut remaining = buffer;
    let mut start_y = 0;

    for i in 0..num_threads {
        let extra = if i < remainder { 1 } else { 0 };
        let rows = rows_per_thread + extra;
        let end_y = start_y + rows;
        let stripe_len = rows * width;

        let (chunk, rest) = remaining.split_at_mut(stripe_len);
        buffer_chunks.push((chunk, start_y, end_y));
        remaining = rest;
        start_y = end_y;
    }

    // Use scope to spawn threads with borrowed data
    std::thread::scope(|s| {
        for (chunk, start_y, end_y) in buffer_chunks {
            s.spawn(move || {
                pixelflow_core::execute_stripe(surface, chunk, width, start_y, end_y);
            });
        }
    });
}
