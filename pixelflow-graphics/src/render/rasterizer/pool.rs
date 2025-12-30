//! Thread pool with spin-before-block workers.
//! No external dependencies - uses std atomics and Mutex.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex};
use std::thread::{self, JoinHandle};

/// A boxed closure that can be executed by a thread pool worker.
///
/// This is the fundamental unit of work for the thread pool. Each job is a
/// one-shot closure that will be executed exactly once by a worker thread.
pub type Job = Box<dyn FnOnce() + Send + 'static>;

/// Number of spin iterations before blocking.
const SPIN_COUNT: u32 = 1000;

/// Shared job queue with condition variable for wake/sleep.
struct JobQueue {
    jobs: Mutex<VecDeque<Job>>,
    available: Condvar,
    pending: AtomicUsize,
    shutdown: AtomicBool,
}

impl JobQueue {
    fn new() -> Self {
        Self {
            jobs: Mutex::new(VecDeque::new()),
            available: Condvar::new(),
            pending: AtomicUsize::new(0),
            shutdown: AtomicBool::new(false),
        }
    }

    fn push(&self, job: Job) {
        {
            let mut queue = self.jobs.lock().unwrap();
            queue.push_back(job);
        }
        self.pending.fetch_add(1, Ordering::Release);
        self.available.notify_one();
    }

    fn try_pop(&self) -> Option<Job> {
        // Fast path: check pending count first (no lock)
        if self.pending.load(Ordering::Acquire) == 0 {
            return None;
        }

        let mut queue = self.jobs.lock().unwrap();
        if let Some(job) = queue.pop_front() {
            self.pending.fetch_sub(1, Ordering::Release);
            Some(job)
        } else {
            None
        }
    }

    fn pop_blocking(&self) -> Option<Job> {
        let mut queue = self.jobs.lock().unwrap();
        loop {
            if self.shutdown.load(Ordering::Acquire) {
                return None;
            }
            if let Some(job) = queue.pop_front() {
                self.pending.fetch_sub(1, Ordering::Release);
                return Some(job);
            }
            queue = self.available.wait(queue).unwrap();
        }
    }

    fn shutdown(&self) {
        self.shutdown.store(true, Ordering::Release);
        self.available.notify_all();
    }
}

/// A fixed-size thread pool with shared job queue.
/// Workers spin briefly before blocking, reducing latency for bursty workloads.
pub struct ThreadPool {
    queue: std::sync::Arc<JobQueue>,
    _workers: Vec<JoinHandle<()>>,
}

impl ThreadPool {
    /// Create a new thread pool with the specified number of threads.
    pub fn new(size: usize) -> Self {
        let queue = std::sync::Arc::new(JobQueue::new());
        let mut workers = Vec::with_capacity(size);

        for _ in 0..size {
            let q = std::sync::Arc::clone(&queue);
            let handle = thread::spawn(move || worker_loop(&q));
            workers.push(handle);
        }

        Self {
            queue,
            _workers: workers,
        }
    }

    /// Submit a closure to be executed by a worker thread.
    ///
    /// The closure is boxed and placed in the job queue, where it will be
    /// picked up by the next available worker.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let pool = ThreadPool::new(4);
    /// pool.submit(|| {
    ///     println!("Hello from worker thread!");
    /// });
    /// ```
    pub fn submit<F>(&self, f: F)
    where
        F: FnOnce() + Send + 'static,
    {
        self.queue.push(Box::new(f));
    }

    /// Submit a pre-boxed job to the queue.
    ///
    /// Use this when you already have a [`Job`] (boxed closure), or when
    /// dispatching heterogeneous closures that cannot share a common type.
    pub fn submit_boxed(&self, job: Job) {
        self.queue.push(job);
    }

    /// Dispatch multiple jobs to the queue.
    ///
    /// Jobs are added in order, but execution order depends on worker availability.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let pool = ThreadPool::new(4);
    /// let jobs: Vec<Job> = (0..10)
    ///     .map(|i| Box::new(move || println!("Job {}", i)) as Job)
    ///     .collect();
    /// pool.dispatch(jobs);
    /// ```
    pub fn dispatch(&self, jobs: impl IntoIterator<Item = Job>) {
        for job in jobs {
            self.queue.push(job);
        }
    }

    /// Get the number of threads in the pool.
    pub fn size(&self) -> usize {
        self._workers.len()
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        self.queue.shutdown();
    }
}

/// Worker loop: spin before blocking.
fn worker_loop(queue: &JobQueue) {
    loop {
        // Phase 1: Spin - try_pop is fast (just atomic check + possible lock)
        for _ in 0..SPIN_COUNT {
            if let Some(job) = queue.try_pop() {
                job();
                continue;
            }
            if queue.shutdown.load(Ordering::Acquire) {
                return;
            }
            std::hint::spin_loop();
        }

        // Phase 2: Block - no work after spinning, wait on condvar
        match queue.pop_blocking() {
            Some(job) => job(),
            None => return, // Shutdown
        }
    }
}
