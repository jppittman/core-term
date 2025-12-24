//! Parallel rasterization support.

use crate::render::color::Pixel;
use crate::render::rasterizer::{execute, execute_stripe, Stripe, TensorShape, ThreadPool};
use pixelflow_core::{Discrete, Manifold};
use std::sync::{Arc, Barrier};

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
/// This avoids thread creation overhead.
///
/// # Safety
/// This function uses `unsafe` to share stack buffers with worker threads.
/// It uses a `Barrier` to ensure all workers complete before returning,
/// preserving memory safety.
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

    // Partitioning
    let rows_per_thread = shape.height / num_threads;
    let remainder = shape.height % num_threads;

    // We can't use `split_at_mut` easily with the pool dispatch logic 
    // because we need to construct all jobs first.
    // Instead, we calculate offsets and use raw pointers.
    
    let buffer_ptr = buffer.as_mut_ptr();
    let barrier = Arc::new(Barrier::new(num_threads + 1));
    // Explicit type annotation to ensure Box<dyn FnOnce...> trait object
    let mut jobs: Vec<Box<dyn FnOnce() + Send>> = Vec::with_capacity(num_threads);

    let mut start_y = 0;

    for i in 0..num_threads {
        let extra = if i < remainder { 1 } else { 0 };
        let rows = rows_per_thread + extra;
        let end_y = start_y + rows;
        let stripe_len = rows * shape.width;
        
        // Calculate pointer offset for this chunk
        let offset = start_y * shape.width;
        let chunk_ptr = unsafe { buffer_ptr.add(offset) };
        let send_ptr = SendPtr(chunk_ptr);
        
        let m_clone = manifold.clone();
        let b_clone = barrier.clone();
        let width = shape.width;
        let stripe = Stripe { start_y, end_y };

        // Construct Job closure
        let job = Box::new(move || {
            // Force capture of the wrapper, not the inner field (disjoint capture fix)
            let ptr_wrapper = send_ptr; 
            // Reconstruct slice
            let slice = unsafe { std::slice::from_raw_parts_mut(ptr_wrapper.0, stripe_len) };
            execute_stripe(&m_clone, slice, width, stripe);
            b_clone.wait();
        });
        
        jobs.push(job);
        start_y = end_y;
    }

    // Dispatch
    pool.dispatch(jobs);

    // Wait for workers
    barrier.wait();
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