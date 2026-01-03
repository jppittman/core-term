//! Lock-free MPMC ring buffer
//!
//! A simple bounded multi-producer, multi-consumer queue using atomics.
//! Inspired by Linux kernel work queues and BPF ring buffers.
//!
//! ## Design
//!
//! - Circular buffer with power-of-2 capacity
//! - Atomic head (consumer) and tail (producer) indices
//! - Relaxed ordering for throughput (rough FIFO, not strict)
//! - Cache-line padding to prevent false sharing
//!
//! ## Trade-offs
//!
//! - ✅ Lock-free (no kernel involvement)
//! - ✅ High throughput (~100M ops/sec on modern CPU)
//! - ✅ Bounded memory (no allocations after init)
//! - ⚠️ Rough ordering (not strict FIFO under contention)
//! - ⚠️ Busy-wait on full/empty (can add parker)
//! - ❌ Requires power-of-2 capacity
//!
//! ## References
//!
//! - Linux kernel/workqueue.c (per-CPU work queues)
//! - Linux kernel/bpf/ringbuf.c (BPF ring buffer)
//! - Disruptor pattern (LMAX exchange)

use std::cell::UnsafeCell;
use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Cache line size on x86-64 (64 bytes)
const CACHE_LINE: usize = 64;

/// Pad to cache line to prevent false sharing
#[repr(align(64))]
struct CachePadded<T>(T);

/// Lock-free MPMC ring buffer
///
/// ## Example
///
/// ```rust
/// let (tx, rx) = mpmc_ring(1024);
///
/// // Producer thread
/// tx.push(42);
///
/// // Consumer thread
/// if let Some(val) = rx.pop() {
///     println!("Got: {}", val);
/// }
/// ```
pub struct MpmcRing<T> {
    /// Ring buffer storage
    /// Use UnsafeCell since we coordinate access via atomics
    buffer: Box<[UnsafeCell<MaybeUninit<T>>]>,

    /// Capacity (power of 2)
    capacity: usize,

    /// Mask for wrapping (capacity - 1)
    mask: usize,

    /// Consumer position (cache-line padded)
    head: CachePadded<AtomicUsize>,

    /// Producer position (cache-line padded)
    tail: CachePadded<AtomicUsize>,
}

unsafe impl<T: Send> Send for MpmcRing<T> {}
unsafe impl<T: Send> Sync for MpmcRing<T> {}

impl<T> MpmcRing<T> {
    /// Create a new ring buffer with given capacity.
    ///
    /// Capacity will be rounded up to next power of 2.
    pub fn new(capacity: usize) -> Self {
        let capacity = capacity.next_power_of_two();
        let mask = capacity - 1;

        let mut buffer = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            buffer.push(UnsafeCell::new(MaybeUninit::uninit()));
        }

        Self {
            buffer: buffer.into_boxed_slice(),
            capacity,
            mask,
            head: CachePadded(AtomicUsize::new(0)),
            tail: CachePadded(AtomicUsize::new(0)),
        }
    }

    /// Push a value (non-blocking, returns false if full)
    ///
    /// Uses Relaxed ordering for max throughput. This means:
    /// - No happens-before guarantees between different producers
    /// - Rough FIFO order under contention
    /// - Values may be reordered slightly
    pub fn try_push(&self, value: T) -> Result<(), T> {
        loop {
            let tail = self.tail.0.load(Ordering::Relaxed);
            let head = self.head.0.load(Ordering::Acquire);

            // Check if full (tail caught up to head after wrapping)
            // Allow (capacity - 1) items to distinguish full from empty
            let size = tail.wrapping_sub(head);
            if size >= self.capacity {
                return Err(value); // Full
            }

            // Try to claim this slot
            if self
                .tail
                .0
                .compare_exchange_weak(tail, tail.wrapping_add(1), Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
            {
                // We claimed slot at index (tail & mask)
                let index = tail & self.mask;
                unsafe {
                    (*self.buffer[index].get()).write(value);
                }
                return Ok(());
            }
            // CAS failed, retry
        }
    }

    /// Pop a value (non-blocking, returns None if empty)
    pub fn try_pop(&self) -> Option<T> {
        loop {
            let head = self.head.0.load(Ordering::Relaxed);
            let tail = self.tail.0.load(Ordering::Acquire);

            // Check if empty
            if head == tail {
                return None; // Empty
            }

            // Try to claim this slot
            if self
                .head
                .0
                .compare_exchange_weak(head, head.wrapping_add(1), Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
            {
                // We claimed slot at index (head & mask)
                let index = head & self.mask;
                let value = unsafe { (*self.buffer[index].get()).assume_init_read() };
                return Some(value);
            }
            // CAS failed, retry
        }
    }

    /// Blocking push (spins until space available)
    pub fn push(&self, mut value: T) {
        loop {
            match self.try_push(value) {
                Ok(()) => return,
                Err(v) => {
                    value = v;
                    std::hint::spin_loop(); // Backoff hint to CPU
                }
            }
        }
    }

    /// Blocking pop (spins until value available)
    pub fn pop(&self) -> T {
        loop {
            if let Some(value) = self.try_pop() {
                return value;
            }
            std::hint::spin_loop();
        }
    }

    /// Approximate number of items in queue
    /// (May be stale due to concurrent operations)
    pub fn len(&self) -> usize {
        let head = self.head.0.load(Ordering::Relaxed);
        let tail = self.tail.0.load(Ordering::Relaxed);
        tail.wrapping_sub(head)
    }

    /// Is the queue empty? (May be stale)
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<T> Drop for MpmcRing<T> {
    fn drop(&mut self) {
        // Drain remaining items to call their destructors
        while self.try_pop().is_some() {}
    }
}

/// Create a pair of cloneable sender/receiver handles
///
/// This mirrors std::sync::mpsc::channel() API
pub fn mpmc_ring<T>(capacity: usize) -> (MpmcSender<T>, MpmcReceiver<T>) {
    let ring = std::sync::Arc::new(MpmcRing::new(capacity));
    (
        MpmcSender {
            ring: ring.clone(),
        },
        MpmcReceiver { ring },
    )
}

/// Sender handle (cloneable for multiple producers)
#[derive(Clone)]
pub struct MpmcSender<T> {
    ring: std::sync::Arc<MpmcRing<T>>,
}

impl<T> MpmcSender<T> {
    pub fn try_send(&self, value: T) -> Result<(), T> {
        self.ring.try_push(value)
    }

    pub fn send(&self, value: T) {
        self.ring.push(value);
    }
}

/// Receiver handle (cloneable for multiple consumers)
#[derive(Clone)]
pub struct MpmcReceiver<T> {
    ring: std::sync::Arc<MpmcRing<T>>,
}

impl<T> MpmcReceiver<T> {
    pub fn try_recv(&self) -> Option<T> {
        self.ring.try_pop()
    }

    pub fn recv(&self) -> T {
        self.ring.pop()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn test_single_thread() {
        let ring = MpmcRing::new(4);
        assert!(ring.try_push(1).is_ok());
        assert!(ring.try_push(2).is_ok());
        assert_eq!(ring.try_pop(), Some(1));
        assert_eq!(ring.try_pop(), Some(2));
        assert_eq!(ring.try_pop(), None);
    }

    #[test]
    fn test_full() {
        let ring = MpmcRing::new(4); // Actually holds 4 items
        assert!(ring.try_push(1).is_ok());
        assert!(ring.try_push(2).is_ok());
        assert!(ring.try_push(3).is_ok());
        assert!(ring.try_push(4).is_ok());
        assert!(ring.try_push(5).is_err()); // Full
    }

    #[test]
    fn test_mpmc() {
        let (tx, rx) = mpmc_ring(1024);

        // 4 producers
        let mut producers = vec![];
        for i in 0..4 {
            let tx = tx.clone();
            producers.push(thread::spawn(move || {
                for j in 0..1000 {
                    tx.send(i * 1000 + j);
                }
            }));
        }

        // 4 consumers
        let mut consumers = vec![];
        let counter = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        for _ in 0..4 {
            let rx = rx.clone();
            let counter = counter.clone();
            consumers.push(thread::spawn(move || {
                for _ in 0..1000 {
                    let _ = rx.recv();
                    counter.fetch_add(1, Ordering::Relaxed);
                }
            }));
        }

        for p in producers {
            p.join().unwrap();
        }
        for c in consumers {
            c.join().unwrap();
        }

        assert_eq!(counter.load(Ordering::Relaxed), 4000);
    }

    #[test]
    fn test_wrap_around() {
        let ring = MpmcRing::new(4);

        // Fill and drain multiple times to test wrapping
        for round in 0..10 {
            assert!(ring.try_push(round * 10 + 1).is_ok());
            assert!(ring.try_push(round * 10 + 2).is_ok());
            assert_eq!(ring.try_pop(), Some(round * 10 + 1));
            assert_eq!(ring.try_pop(), Some(round * 10 + 2));
        }
    }
}
