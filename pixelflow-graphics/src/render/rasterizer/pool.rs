//! Simple thread pool for persistent worker threads.

use std::sync::mpsc::{channel, Sender};
use std::thread;

type Job = Box<dyn FnOnce() + Send + 'static>;

/// A fixed-size thread pool.
pub struct ThreadPool {
    workers: Vec<Sender<Job>>,
}

impl ThreadPool {
    /// Create a new thread pool with the specified number of threads.
    pub fn new(size: usize) -> Self {
        let mut workers = Vec::with_capacity(size);
        for _ in 0..size {
            let (tx, rx) = channel::<Job>();
            thread::spawn(move || {
                while let Ok(job) = rx.recv() {
                    job();
                }
            });
            workers.push(tx);
        }
        Self { workers }
    }

    /// Dispatch a list of jobs to workers 1:1.
    ///
    /// # Panics
    /// Panics if `jobs.len()` > number of workers.
    pub fn dispatch(&self, jobs: Vec<Job>) {
        if jobs.len() > self.workers.len() {
            panic!("ThreadPool: more jobs than workers");
        }
        for (i, job) in jobs.into_iter().enumerate() {
            self.workers[i].send(job).expect("Worker thread died");
        }
    }

    /// Get the number of threads in the pool.
    pub fn size(&self) -> usize {
        self.workers.len()
    }
}
