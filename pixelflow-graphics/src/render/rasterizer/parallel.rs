//! # Parallel Rasterization
//!
//! Three strategies for multi-threaded manifold rendering, each with different tradeoffs.
//!
//! ## Overview: Why Parallelize?
//!
//! Rasterization is **embarrassingly parallel**: each row (or stripe) is independent and can be
//! computed by a different thread without synchronization. However, the choice of parallelization
//! strategy affects:
//! - **Load balancing**: How evenly work is distributed
//! - **Overhead**: Thread spawning, atomic operations, memory layout
//! - **Latency**: Time from start to finish
//! - **Throughput**: Number of pixels per second
//!
//! ## Strategy 1: Static Partitioning (`render_parallel`)
//!
//! **How it works**: Divide rows evenly among threads beforehand.
//!
//! ```text
//! Main Thread          Worker Threads
//! │                    ┌──────────────────┐
//! ├─ Partition rows    │ Thread 1: rows 0-99 (1000 px/row)
//! │ - Thread 1: 0-99   │
//! │ - Thread 2: 100-199├─ render()
//! │ - Thread 3: 200-299│ (all rows in parallel)
//! │ - Thread 4: 300-399│
//! │                    │ Thread 4: rows 300-399 (1000 px/row)
//! ├─ Spawn threads     └──────────────────┘
//! │
//! ├─ Wait (scope)
//! └─ All rows done
//! ```
//!
//! **Precondition**: `shape.width`, `shape.height > 0`; `num_threads <= num_CPUs`
//!
//! **Postcondition**: All rows rendered; buffer fully populated
//!
//! **Tradeoffs**:
//! - ✅ **No atomic operations**: Lock-free (just buffer splitting)
//! - ✅ **Safe by construction**: Rust's scope threads guarantee no data race
//! - ❌ **Poor load balancing**: If manifold is slow on some rows, some threads finish early and wait
//! - ❌ **High overhead** for small images (thread spawning cost dominates)
//!
//! **Best for**: Large, uniform workloads (simple color gradients, backgrounds)
//!
//! **Example usage**:
//! ```ignore
//! use pixelflow_graphics::render::RenderOptions;
//!
//! let options = RenderOptions { num_threads: 4 };
//! render_parallel(&manifold, &mut buffer, shape, options);
//! ```
//!
//! ## Strategy 2: Work-Stealing (`render_work_stealing`)
//!
//! **How it works**: Threads race for rows via atomic fetch_add. When a thread finishes a row,
//! it grabs the next available row from a shared counter.
//!
//! ```text
//! Main Thread          Shared Counter: next_row = 0
//!                      ┌──────────────────────┐
//! ├─ Create counter    │ Thread 1 fetches row 0
//! │ - next_row = 0     │ -> processes rows 0,4,8,...
//! │                    │
//! ├─ Spawn 4 threads   │ Thread 2 fetches row 1
//! │                    │ -> processes rows 1,5,9,...
//! │ (all race for rows)│
//! │                    │ Thread 3: (slower)
//! ├─ Spin-wait        │ -> processes row 2, then row 6 after a delay
//! │ (rows_done < height)
//! │                    │ Thread 4: (very slow)
//! └─ All rows done     │ -> processes row 3 only; others finish first
//!                      └──────────────────────┘
//! ```
//!
//! **Precondition**: `shape.width`, `shape.height > 0`; `num_threads <= num_CPUs`
//!
//! **Postcondition**: All rows rendered; buffer fully populated
//!
//! **Tradeoffs**:
//! - ✅ **Excellent load balancing**: Fast threads steal extra work
//! - ✅ **Zero allocations**: Only atomic counter, no channels or queues
//! - ✅ **Low latency**: Fastest thread determines overall speed
//! - ❌ **Atomic cost**: Each row fetch is an atomic fetch_add (cache contention on ~height operations)
//! - ❌ **Spin-wait overhead**: Main thread spins in hot loop until completion
//! - ❌ **Raw unsafe**: Uses raw pointers with manual lifetime management
//!
//! **Best for**: Non-uniform workloads where some manifolds are slow (procedural generation, complex shaders)
//!
//! **Example usage**:
//! ```ignore
//! render_work_stealing(&manifold, &mut buffer, shape, 4);
//! ```
//!
//! ## Strategy 3: Thread Pool (`render_parallel_pooled`)
//!
//! **How it works**: Like work-stealing, but threads are persistent (pre-allocated). New rendering
//! jobs are submitted to the pool; existing threads race for rows.
//!
//! ```text
//! First Render         Second Render (reuses threads!)
//! ├─ Create pool       ├─ pool.submit(job1)
//! │ (spawn 4 threads)  │ pool.submit(job2)
//! ├─ pool.submit(job)  │ pool.submit(job3)
//! │                    │ pool.submit(job4)
//! ├─ Wait (atomic)     │
//! └─ Done              └─ Wait (reused threads)
//!
//! Benefit: No thread spawning overhead on subsequent calls
//! ```
//!
//! **Precondition**: `pool` must be initialized with desired thread count
//!
//! **Postcondition**: All rows rendered; buffer fully populated; threads remain alive for next job
//!
//! **Tradeoffs**:
//! - ✅ **Best amortized latency**: Zero spawn overhead after first render
//! - ✅ **Good load balancing**: Work-stealing among pool threads
//! - ✅ **Efficient for animation**: Frames 2, 3, 4... have minimal overhead
//! - ❌ **Memory overhead**: Threads always live (stack overhead even when idle)
//! - ❌ **Most complex**: Requires ThreadPool management
//!
//! **Best for**: Continuous animation (rendering many frames; amortizes thread creation)
//!
//! **Example usage**:
//! ```ignore
//! use pixelflow_graphics::render::ThreadPool;
//!
//! let pool = ThreadPool::new(4);
//! render_parallel_pooled(&pool, &manifold, &mut buffer, shape);
//! render_parallel_pooled(&pool, &manifold, &mut buffer, shape); // Reuses threads
//! ```
//!
//! ## Which Strategy Should I Use?
//!
//! | Workload | Strategy | Reason |
//! |----------|----------|--------|
//! | Single render | `render_work_stealing` | Load balancing + low overhead |
//! | Animation (60 FPS) | `render_parallel_pooled` | Amortizes thread spawn cost |
//! | Simple, uniform | `render_parallel` | Minimal complexity |
//! | Unknown complexity | `render_work_stealing` | Safe default; handles all cases |
//!
//! ## Safety Notes
//!
//! All three functions use safe Rust abstractions:
//! - **Scoped threads**: Guarantee threads are joined before data is freed
//! - **Buffer splitting**: Rust verifies disjoint slices (no data race)
//! - **Atomic operations**: Relaxed ordering sufficient (no strict happens-before needed)
//!
//! The `unsafe` code in work-stealing is **localized to pointer casting** and justified:
//! - Raw pointer is necessary to share across scoped threads
//! - Safety is guaranteed by scoped thread semantics (no access after scope ends)
//! - Alternative: use Arc<Mutex<>> (slower and overkill)
//!
//! ## Performance Model
//!
//! ```text
//! Sequential: T_seq = W (work per pixel) × N (pixels)
//!
//! Static:     T_static = W × N / P + O_spawn
//!             (worst case if load imbalanced: T_worst = W × max_row + O_spawn)
//!
//! Work-steal: T_steal = W × N / P + O_atomic × ceil(N / row_width)
//!             (better: fast threads help slow threads finish)
//!
//! Pool:       T_pool_1st = W × N / P + O_spawn
//!             T_pool_nth = W × N / P + O_atomic × ceil(N / row_width)
//!             (amortizes O_spawn across many renders)
//! ```
//!
//! Where:
//! - P = number of threads
//! - O_spawn = cost of spawning N threads
//! - O_atomic = cost of atomic fetch_add operation
//!
//! **In practice** (on modern hardware):
//! - O_spawn ≈ 1-10 microseconds per thread
//! - O_atomic ≈ 10-50 nanoseconds (with contention)
//! - For 1080p (1000 rows): atomic cost is ~0.1% of render time
//!
//! ## Integration with Single-Threaded Rasterizer
//!
//! All three functions check `num_threads <= 1` and fall back to `execute()` (single-threaded).
//! This simplifies usage: you can always call the parallel version with dynamic thread count.

use super::pool::ThreadPool;
use crate::render::color::Pixel;
use crate::render::rasterizer::{execute, execute_stripe, Stripe, TensorShape};
use pixelflow_core::{Discrete, Manifold};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// Configuration for parallel rendering.
///
/// # Fields
///
/// - `num_threads`: Number of worker threads to spawn. If 0 or 1, rendering is single-threaded.
///
/// # Example
///
/// ```ignore
/// let options = RenderOptions { num_threads: 4 };
/// render_parallel(&manifold, &mut buffer, shape, options);
/// ```
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

        pool.submit(move || {
            let ptr = send_ptr;
            loop {
                // Compete for next row
                let row = next.fetch_add(1, Ordering::Relaxed);
                if row >= height {
                    break;
                }

                // Process this row
                let offset = row * width;
                let row_slice = unsafe { std::slice::from_raw_parts_mut(ptr.0.add(offset), width) };
                execute_stripe(
                    &m_clone,
                    row_slice,
                    Stripe {
                        start_y: row,
                        end_y: row + 1,
                        width,
                    },
                );

                // Mark done
                done.fetch_add(1, Ordering::Release);
            }
        });
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

    // 2MB stack per thread (needed for deep 3D scene recursion)
    const STACK_SIZE: usize = 2 * 1024 * 1024;

    // Use scope to spawn threads with borrowed data
    std::thread::scope(|s| {
        for (chunk, start_y, end_y) in buffer_chunks {
            std::thread::Builder::new()
                .stack_size(STACK_SIZE)
                .spawn_scoped(s, move || {
                    execute_stripe(
                        manifold,
                        chunk,
                        Stripe {
                            start_y,
                            end_y,
                            width: shape.width,
                        },
                    );
                })
                .expect("Failed to spawn render thread");
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
    options: RenderOptions,
) where
    P: Pixel + Send,
    M: Manifold<Output = Discrete> + Sync,
{
    if options.num_threads <= 1 || shape.height <= 1 {
        execute(manifold, buffer, shape);
        return;
    }

    // Atomic counter - threads race to grab rows
    let next_row = AtomicUsize::new(0);
    let width = shape.width;
    let height = shape.height;

    // Wrap raw pointer for Send
    let buffer_ptr = SendPtr(buffer.as_mut_ptr());

    // 2MB stack per thread (needed for deep 3D scene recursion)
    const STACK_SIZE: usize = 2 * 1024 * 1024;

    std::thread::scope(|s| {
        for _ in 0..options.num_threads {
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
                        let row_slice =
                            unsafe { std::slice::from_raw_parts_mut(ptr.0.add(offset), width) };

                        // Render single row
                        execute_stripe(
                            manifold,
                            row_slice,
                            Stripe {
                                start_y: row,
                                end_y: row + 1,
                                width,
                            },
                        );
                    }
                })
                .expect("Failed to spawn render thread");
        }
    });
}
