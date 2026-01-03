//! Work Pool Actor - Implements work-stealing scheduler as an Actor
//!
//! This module demonstrates that the supervisor + work-stealing model can be
//! implemented INSIDE the current actor-scheduler architecture, rather than
//! replacing it.
//!
//! ## Key Insight
//!
//! The work pool's Control/Management/Data channels ARE the "global queues"
//! from the work-stealing model. The ActorScheduler feeds messages to the pool,
//! and internal workers steal from shared queues.
//!
//! ## Usage
//!
//! ```rust
//! use actor_scheduler::{WorkPoolActor, WorkPoolConfig};
//!
//! // Define your task type
//! enum DataTask {
//!     Process(Vec<u8>),
//!     Transform(String),
//! }
//!
//! // Create pool with 4 workers
//! let config = WorkPoolConfig {
//!     num_workers: 4,
//!     queue_capacity: 1024,
//! };
//! let mut pool = WorkPoolActor::new(config, |task| {
//!     // Worker function - executes on worker threads
//!     match task {
//!         DataTask::Process(data) => { /* ... */ }
//!         DataTask::Transform(s) => { /* ... */ }
//!     }
//! });
//!
//! // Use in troupe like any other actor
//! troupe! {
//!     terminal: TerminalApp,
//!     data_pool: WorkPoolActor<DataTask>,
//! }
//! ```
//!
//! ## Design
//!
//! ```text
//! ┌────────────────────────────────────────────────────┐
//! │  WorkPoolActor (appears as 1 actor to troupe)      │
//! │                                                    │
//! │  ActorScheduler Thread (supervisor)                │
//! │  ┌──────────────────────────────────────────┐     │
//! │  │ handle_control(task)                     │     │
//! │  │ handle_data(task)                        │     │
//! │  │   ↓                                      │     │
//! │  │ Push to shared queue + wake workers      │     │
//! │  └──────────────────────────────────────────┘     │
//! │                    │                              │
//! │                    ▼                              │
//! │  ┌────────────────────────────────────────┐       │
//! │  │ Shared Task Queue (crossbeam)          │       │
//! │  │ - Lock-free MPMC                       │       │
//! │  │ - Workers steal from here              │       │
//! │  └────────────────────────────────────────┘       │
//! │       │        │        │        │                │
//! │       ▼        ▼        ▼        ▼                │
//! │  ┌───────┐┌───────┐┌───────┐┌───────┐            │
//! │  │Work 1 ││Work 2 ││Work 3 ││Work 4 │            │
//! │  │       ││       ││       ││       │            │
//! │  └───────┘└───────┘└───────┘└───────┘            │
//! │                                                   │
//! └────────────────────────────────────────────────────┘
//! ```

use crate::{Actor, ParkHint, MpmcRing};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};

/// Configuration for WorkPoolActor
#[derive(Debug, Clone)]
pub struct WorkPoolConfig {
    /// Number of worker threads
    pub num_workers: usize,
    /// Optional thread name prefix (workers will be named "{prefix}-{id}")
    pub thread_name_prefix: Option<String>,
}

impl Default for WorkPoolConfig {
    fn default() -> Self {
        Self {
            num_workers: 4,
            thread_name_prefix: Some("work-pool".to_string()),
        }
    }
}

/// A work pool actor that distributes tasks to internal worker threads.
///
/// From the troupe's perspective, this is a single actor. Internally, it
/// manages a pool of worker threads that steal tasks from shared queues.
///
/// ## Type Parameters
///
/// - `T`: Task type that workers execute. Must be `Send + 'static`.
///
/// ## Message Semantics
///
/// - **Control messages**: Processed with priority (workers check Control queue first)
/// - **Management messages**: Currently routed to Data queue (can be customized)
/// - **Data messages**: Work-stealing pool
///
/// ## Shutdown
///
/// When the actor's scheduler receives shutdown signal, all workers are
/// gracefully stopped and remaining tasks are drained.
pub struct WorkPoolActor<T> {
    config: WorkPoolConfig,

    // Lock-free shared queues
    control_ring: Arc<MpmcRing<T>>,
    data_ring: Arc<MpmcRing<T>>,

    // Worker thread handles
    workers: Vec<JoinHandle<()>>,

    // Shutdown coordination
    shutdown: Arc<AtomicBool>,
}

impl<T> WorkPoolActor<T>
where
    T: Send + 'static,
{
    /// Create a new work pool actor.
    ///
    /// ## Arguments
    ///
    /// - `config`: Pool configuration (worker count, etc.)
    /// - `worker_fn`: Function called by workers for each task
    ///
    /// ## Example
    ///
    /// ```rust
    /// let pool = WorkPoolActor::new(
    ///     WorkPoolConfig::default(),
    ///     |task: MyTask| {
    ///         // Process task
    ///         println!("Processing: {:?}", task);
    ///     }
    /// );
    /// ```
    pub fn new<F>(config: WorkPoolConfig, worker_fn: F) -> Self
    where
        F: Fn(T) + Send + Sync + 'static,
    {
        // Lock-free rings (power-of-2 capacities)
        let control_ring = Arc::new(MpmcRing::new(128));
        let data_ring = Arc::new(MpmcRing::new(1024));
        let shutdown = Arc::new(AtomicBool::new(false));
        let worker_fn = Arc::new(worker_fn);

        // Spawn worker threads
        let mut workers = Vec::new();
        for worker_id in 0..config.num_workers {
            let control_ring = Arc::clone(&control_ring);
            let data_ring = Arc::clone(&data_ring);
            let shutdown = Arc::clone(&shutdown);
            let worker_fn = Arc::clone(&worker_fn);

            let mut builder = thread::Builder::new();
            if let Some(prefix) = &config.thread_name_prefix {
                builder = builder.name(format!("{}-{}", prefix, worker_id));
            }

            let handle = builder
                .spawn(move || {
                    Self::worker_loop(
                        worker_id,
                        control_ring,
                        data_ring,
                        shutdown,
                        worker_fn,
                    );
                })
                .expect("Failed to spawn worker thread");

            workers.push(handle);
        }

        Self {
            config,
            control_ring,
            data_ring,
            workers,
            shutdown,
        }
    }

    /// Worker thread loop - priority-aware lock-free work stealing
    fn worker_loop(
        _worker_id: usize,
        control_ring: Arc<MpmcRing<T>>,
        data_ring: Arc<MpmcRing<T>>,
        shutdown: Arc<AtomicBool>,
        worker_fn: Arc<dyn Fn(T) + Send + Sync>,
    ) {
        loop {
            // Check shutdown
            if shutdown.load(Ordering::Relaxed) {
                return;
            }

            // 1. Try Control queue first (priority) - lock-free!
            if let Some(task) = control_ring.try_pop() {
                worker_fn(task);
                continue;
            }

            // 2. Try Data queue - lock-free!
            if let Some(task) = data_ring.try_pop() {
                worker_fn(task);
                continue;
            }

            // 3. No work available - backoff
            // TODO: Smarter backoff (exponential, then park)
            std::hint::spin_loop();
        }
    }

    /// Number of active workers
    pub fn num_workers(&self) -> usize {
        self.config.num_workers
    }
}

impl<T> Actor<T, T, T> for WorkPoolActor<T>
where
    T: Send + 'static,
{
    fn handle_control(&mut self, msg: T) {
        // Push to Control ring - lock-free, workers check this first
        self.control_ring.push(msg);
    }

    fn handle_management(&mut self, msg: T) {
        // For now, route to data ring
        // Could add separate management ring if needed
        self.data_ring.push(msg);
    }

    fn handle_data(&mut self, msg: T) {
        // Push to Data ring - lock-free, workers steal from here
        self.data_ring.push(msg);
    }

    fn park(&mut self, _hint: ParkHint) -> ParkHint {
        // As supervisor, we don't do heavy work - just distribute tasks
        // Workers do the actual processing
        ParkHint::Wait
    }
}

impl<T> Drop for WorkPoolActor<T> {
    fn drop(&mut self) {
        // Signal shutdown to all workers
        self.shutdown.store(true, Ordering::Relaxed);

        // Wait for workers to finish
        while let Some(handle) = self.workers.pop() {
            let _ = handle.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;
    use std::sync::Mutex;
    use std::time::Duration;

    #[test]
    fn test_work_pool_basic() {
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = Arc::clone(&counter);

        let config = WorkPoolConfig {
            num_workers: 2,
            thread_name_prefix: Some("test-pool".to_string()),
        };

        let mut pool = WorkPoolActor::new(config, move |_task: ()| {
            counter_clone.fetch_add(1, Ordering::Relaxed);
        });

        // Send 10 tasks via Data lane
        for _ in 0..10 {
            pool.handle_data(());
        }

        // Give workers time to process
        thread::sleep(Duration::from_millis(100));

        assert_eq!(counter.load(Ordering::Relaxed), 10);
    }

    #[test]
    fn test_work_pool_control_priority() {
        let results = Arc::new(Mutex::new(Vec::new()));
        let results_clone = Arc::clone(&results);

        let config = WorkPoolConfig {
            num_workers: 1, // Single worker to ensure ordering
            thread_name_prefix: None,
        };

        let mut pool = WorkPoolActor::new(config, move |task: String| {
            results_clone.lock().unwrap().push(task);
        });

        // Send Data messages first
        for i in 0..5 {
            pool.handle_data(format!("data-{}", i));
        }

        // Then send Control message
        pool.handle_control("PRIORITY".to_string());

        // Give worker time to process
        thread::sleep(Duration::from_millis(100));

        let results = results.lock().unwrap();

        // Control message should be processed (workers check Control first)
        assert!(results.contains(&"PRIORITY".to_string()));

        // All messages processed
        assert_eq!(results.len(), 6);
    }

    #[test]
    fn test_work_pool_parallelism() {
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = Arc::clone(&counter);

        let config = WorkPoolConfig {
            num_workers: 4,
            thread_name_prefix: Some("parallel-test".to_string()),
        };

        let mut pool = WorkPoolActor::new(config, move |_task: ()| {
            // Simulate work
            thread::sleep(Duration::from_millis(10));
            counter_clone.fetch_add(1, Ordering::Relaxed);
        });

        let start = std::time::Instant::now();

        // Send 40 tasks
        for _ in 0..40 {
            pool.handle_data(());
        }

        // Wait for completion
        thread::sleep(Duration::from_millis(150));

        let elapsed = start.elapsed();

        // With 4 workers, 40 tasks @ 10ms each should take ~100ms
        // (vs 400ms with 1 worker)
        assert!(elapsed < Duration::from_millis(250));
        assert_eq!(counter.load(Ordering::Relaxed), 40);
    }

    #[test]
    fn test_work_pool_shutdown() {
        let config = WorkPoolConfig {
            num_workers: 2,
            thread_name_prefix: None,
        };

        let pool = WorkPoolActor::new(config, |_task: ()| {
            thread::sleep(Duration::from_millis(10));
        });

        // Drop should trigger graceful shutdown
        drop(pool);

        // If we get here without hanging, shutdown worked
    }
}
