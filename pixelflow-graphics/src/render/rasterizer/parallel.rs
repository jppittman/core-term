//! Parallel rasterization support.

use crate::render::color::Pixel;
use crate::render::rasterizer::{execute, execute_stripe, Stripe, TensorShape, ThreadPool};
use pixelflow_core::{Discrete, Manifold};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

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

/// Raw pointer wrapper to allow sending buffer slices to threads.
/// Safety: Logic must ensure threads do not access this after owner scope ends.
#[derive(Copy, Clone)]
struct SendPtr<T>(*mut T);
unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}

/// Render with parallel rasterization using a persistent thread pool.
///
/// Workers compete for rows from a shared atomic counter - same dynamic
/// load balancing as render_work_stealing but using persistent threads.
///
/// # Safety
/// This function uses `unsafe` to share stack buffers with worker threads.
/// It uses atomic counting to ensure all rows complete before returning.
pub fn render_parallel_pooled<P, M>(
    pool: &ThreadPool,
    manifold: &M,
    buffer: &mut [P],
    shape: TensorShape,
) where
    P: Pixel + Send,
    M: Manifold<Output = Discrete> + Clone + Sync + 'static,
{
    let num_threads = pool.size();
    if num_threads <= 1 {
        execute(manifold, buffer, shape);
        return;
    }

    let buffer_ptr = buffer.as_mut_ptr();
    let next_row = Arc::new(AtomicUsize::new(0));
    let rows_done = Arc::new(AtomicUsize::new(0));
    let height = shape.height;
    let width = shape.width;

    // Dispatch N workers that COMPETE for rows
    for _ in 0..num_threads {
        let m_clone = manifold.clone();
        let next = Arc::clone(&next_row);
        let done = Arc::clone(&rows_done);
        let send_ptr = SendPtr(buffer_ptr);

        let job = Box::new(move || {
            let ptr = send_ptr;
            loop {
                // Compete for next row
                let row = next.fetch_add(1, Ordering::Relaxed);
                if row >= height {
                    break;
                }

                // Process this row
                let offset = row * width;
                let row_slice = unsafe {
                    std::slice::from_raw_parts_mut(ptr.0.add(offset), width)
                };
                execute_stripe(&m_clone, row_slice, width, Stripe { start_y: row, end_y: row + 1 });

                // Mark done
                done.fetch_add(1, Ordering::Release);
            }
        });

        pool.submit(job);
    }

    // Wait for all rows to complete
    while rows_done.load(Ordering::Acquire) < height {
        std::hint::spin_loop();
    }
}

/// Render with parallel rasterization (spawns new threads).
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

/// Render with work-stealing parallelism (GCD-style).
///
/// Threads race to grab rows via atomic fetch_add. No channels, no barriers,
/// no allocations. Just raw atomics and scoped threads.
pub fn render_work_stealing<P, M>(
    manifold: &M,
    buffer: &mut [P],
    shape: TensorShape,
    num_threads: usize,
) where
    P: Pixel + Send,
    M: Manifold<Output = Discrete> + Sync,
{
    if num_threads <= 1 || shape.height <= 1 {
        execute(manifold, buffer, shape);
        return;
    }

    // Atomic counter - threads race to grab rows
    let next_row = AtomicUsize::new(0);
    let width = shape.width;
    let height = shape.height;

    // Wrap raw pointer for Send
    let buffer_ptr = SendPtr(buffer.as_mut_ptr());

    // 64KB stack per thread (default is 2-8MB)
    const STACK_SIZE: usize = 64 * 1024;

    std::thread::scope(|s| {
        for _ in 0..num_threads {
            std::thread::Builder::new()
                .stack_size(STACK_SIZE)
                .spawn_scoped(s, || {
                    let ptr = buffer_ptr; // capture the wrapper
                    loop {
                        // Grab next row atomically
                        let row = next_row.fetch_add(1, Ordering::Relaxed);
                        if row >= height {
                            break;
                        }

                        // Calculate row slice
                        let offset = row * width;
                        let row_slice = unsafe {
                            std::slice::from_raw_parts_mut(ptr.0.add(offset), width)
                        };

                        // Render single row
                        execute_stripe(manifold, row_slice, width, Stripe { start_y: row, end_y: row + 1 });
                    }
                })
                .expect("Failed to spawn render thread");
        }
    });
}