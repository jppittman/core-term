//! Stress tests for actor-scheduler concurrency edge cases
//!
//! These tests verify the scheduler behaves correctly under heavy load,
//! concurrent access, and various edge conditions.

use actor_scheduler::{Actor, ActorScheduler, Message, ActorStatus, SystemStatus};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::thread;
use std::time::Duration;

// ============================================================================
// Test Handler Implementations
// ============================================================================

struct CountingHandler {
    data_count: Arc<AtomicUsize>,
    ctrl_count: Arc<AtomicUsize>,
    mgmt_count: Arc<AtomicUsize>,
}

impl Actor<u64, u64, u64> for CountingHandler {
    fn handle_data(&mut self, _msg: u64) {
        self.data_count.fetch_add(1, Ordering::SeqCst);
    }
    fn handle_control(&mut self, _msg: u64) {
        self.ctrl_count.fetch_add(1, Ordering::SeqCst);
    }
    fn handle_management(&mut self, _msg: u64) {
        self.mgmt_count.fetch_add(1, Ordering::SeqCst);
    }
    fn park(&mut self, _hint: SystemStatus) -> ActorStatus {
        ActorStatus::Idle
    }
}

struct SlowHandler {
    delay: Duration,
    processed: Arc<AtomicUsize>,
}

impl Actor<u64, u64, u64> for SlowHandler {
    fn handle_data(&mut self, _msg: u64) {
        thread::sleep(self.delay);
        self.processed.fetch_add(1, Ordering::SeqCst);
    }
    fn handle_control(&mut self, _msg: u64) {}
    fn handle_management(&mut self, _msg: u64) {}
    fn park(&mut self, _hint: SystemStatus) -> ActorStatus {
        ActorStatus::Idle
    }
}

struct NoOpHandler;

impl Actor<(), (), ()> for NoOpHandler {
    fn handle_data(&mut self, _: ()) {}
    fn handle_control(&mut self, _: ()) {}
    fn handle_management(&mut self, _: ()) {}
    fn park(&mut self, _hint: SystemStatus) -> ActorStatus {
        ActorStatus::Idle
    }
}

// ============================================================================
// High Contention Tests
// ============================================================================

#[test]
fn high_contention_all_messages_delivered() {
    const NUM_SENDERS: usize = 10;
    const MESSAGES_PER_SENDER: usize = 100;

    let (tx, mut rx) = ActorScheduler::new(1024, 1024);
    let data_count = Arc::new(AtomicUsize::new(0));
    let ctrl_count = Arc::new(AtomicUsize::new(0));
    let mgmt_count = Arc::new(AtomicUsize::new(0));

    let handler = CountingHandler {
        data_count: data_count.clone(),
        ctrl_count: ctrl_count.clone(),
        mgmt_count: mgmt_count.clone(),
    };

    // Spawn receiver
    let receiver_handle = thread::spawn(move || {
        let mut h = handler;
        rx.run(&mut h);
    });

    // Spawn multiple senders
    let mut sender_handles = Vec::new();
    for _ in 0..NUM_SENDERS {
        let tx_clone = tx.clone();
        let handle = thread::spawn(move || {
            for i in 0..MESSAGES_PER_SENDER {
                match i % 3 {
                    0 => tx_clone.send(Message::Data(i as u64)).unwrap(),
                    1 => tx_clone.send(Message::Control(i as u64)).unwrap(),
                    _ => tx_clone.send(Message::Management(i as u64)).unwrap(),
                }
            }
        });
        sender_handles.push(handle);
    }

    // Wait for all senders
    for handle in sender_handles {
        handle.join().unwrap();
    }

    // Give time for processing
    thread::sleep(Duration::from_millis(100));

    // Drop sender to terminate receiver
    drop(tx);
    receiver_handle.join().unwrap();

    // Verify all messages were delivered
    let total = data_count.load(Ordering::SeqCst)
        + ctrl_count.load(Ordering::SeqCst)
        + mgmt_count.load(Ordering::SeqCst);

    assert_eq!(
        total,
        NUM_SENDERS * MESSAGES_PER_SENDER,
        "All messages should be delivered under high contention"
    );
}

#[test]
fn high_contention_fairness() {
    // Multiple senders competing for control lane should all eventually succeed
    const NUM_SENDERS: usize = 5;
    const MESSAGES_PER_SENDER: usize = 50;

    let (tx, mut rx) = ActorScheduler::new(10, 100);
    let ctrl_count = Arc::new(AtomicUsize::new(0));

    let handler = CountingHandler {
        data_count: Arc::new(AtomicUsize::new(0)),
        ctrl_count: ctrl_count.clone(),
        mgmt_count: Arc::new(AtomicUsize::new(0)),
    };

    let receiver_handle = thread::spawn(move || {
        let mut h = handler;
        rx.run(&mut h);
    });

    let mut sender_handles = Vec::new();
    for _ in 0..NUM_SENDERS {
        let tx_clone = tx.clone();
        let handle = thread::spawn(move || {
            for i in 0..MESSAGES_PER_SENDER {
                tx_clone.send(Message::Control(i as u64)).unwrap();
            }
        });
        sender_handles.push(handle);
    }

    for handle in sender_handles {
        handle.join().unwrap();
    }

    thread::sleep(Duration::from_millis(100));
    drop(tx);
    receiver_handle.join().unwrap();

    assert_eq!(
        ctrl_count.load(Ordering::SeqCst),
        NUM_SENDERS * MESSAGES_PER_SENDER,
        "All control messages should be delivered"
    );
}

// ============================================================================
// Rapid Channel Creation/Destruction Tests
// ============================================================================

#[test]
fn rapid_channel_creation_does_not_leak() {
    // Create and drop many channels rapidly
    for _ in 0..1000 {
        let (tx, _rx) = ActorScheduler::<u64, u64, u64>::new(10, 100);
        // Send a few messages
        let _ = tx.send(Message::Data(1));
        let _ = tx.send(Message::Control(2));
        // Let it drop
    }
    // If we get here without running out of memory, the test passes
}

#[test]
fn rapid_sender_clone_drop() {
    // Use a large enough buffer to avoid blocking during this test
    let (tx, _rx) = ActorScheduler::<u64, u64, u64>::new(10, 2000);

    // Clone and drop handles rapidly - messages will queue but not block
    for _ in 0..1000 {
        let cloned = tx.clone();
        let _ = cloned.send(Message::Data(42));
        drop(cloned);
    }
}

// ============================================================================
// Backpressure Tests
// ============================================================================

#[test]
fn backpressure_with_slow_consumer() {
    let (tx, mut rx) = ActorScheduler::new(10, 10); // Small buffer
    let processed = Arc::new(AtomicUsize::new(0));
    let processed_clone = processed.clone();

    let handler = SlowHandler {
        delay: Duration::from_millis(1),
        processed: processed_clone,
    };

    let receiver_handle = thread::spawn(move || {
        let mut h = handler;
        rx.run(&mut h);
    });

    // Send many messages - should block when buffer is full
    let sender_handle = {
        let tx_clone = tx.clone();
        thread::spawn(move || {
            for i in 0..50 {
                tx_clone.send(Message::Data(i)).unwrap();
            }
        })
    };

    sender_handle.join().unwrap();
    thread::sleep(Duration::from_millis(100));
    drop(tx);
    receiver_handle.join().unwrap();

    assert_eq!(
        processed.load(Ordering::SeqCst),
        50,
        "All messages should eventually be processed"
    );
}

// ============================================================================
// Wake Handler Tests
// ============================================================================

#[test]
fn custom_wake_handler_is_called() {
    use actor_scheduler::WakeHandler;

    struct TestWakeHandler {
        called: Arc<AtomicBool>,
    }

    impl WakeHandler for TestWakeHandler {
        fn wake(&self) {
            self.called.store(true, Ordering::SeqCst);
        }
    }

    let called = Arc::new(AtomicBool::new(false));
    let wake_handler = Arc::new(TestWakeHandler {
        called: called.clone(),
    });

    let (tx, _rx) =
        ActorScheduler::<u64, u64, u64>::new_with_wake_handler(10, 100, Some(wake_handler));

    tx.send(Message::Data(42)).unwrap();

    assert!(
        called.load(Ordering::SeqCst),
        "Wake handler should be called on message send"
    );
}

// ============================================================================
// Burst Limit Tests
// ============================================================================

#[test]
fn burst_limit_prevents_data_starvation() {
    let (tx, mut rx) = ActorScheduler::new(2, 1000); // Burst limit of 2
    let data_processed = Arc::new(AtomicUsize::new(0));
    let mgmt_processed = Arc::new(AtomicUsize::new(0));

    struct BurstHandler {
        data_processed: Arc<AtomicUsize>,
        mgmt_processed: Arc<AtomicUsize>,
    }

    impl Actor<u64, u64, u64> for BurstHandler {
        fn handle_data(&mut self, _: u64) {
            self.data_processed.fetch_add(1, Ordering::SeqCst);
        }
        fn handle_control(&mut self, _: u64) {}
        fn handle_management(&mut self, _: u64) {
            self.mgmt_processed.fetch_add(1, Ordering::SeqCst);
        }
        fn park(&mut self, _: SystemStatus) -> ActorStatus {
            ActorStatus::Idle
        }
    }

    let handler = BurstHandler {
        data_processed: data_processed.clone(),
        mgmt_processed: mgmt_processed.clone(),
    };

    let receiver_handle = thread::spawn(move || {
        let mut h = handler;
        rx.run(&mut h);
    });

    // Send many data messages and some management
    for i in 0..100 {
        tx.send(Message::Data(i)).unwrap();
    }
    for i in 0..10 {
        tx.send(Message::Management(i)).unwrap();
    }

    thread::sleep(Duration::from_millis(100));
    drop(tx);
    receiver_handle.join().unwrap();

    // Both should be fully processed
    assert_eq!(data_processed.load(Ordering::SeqCst), 100);
    assert_eq!(mgmt_processed.load(Ordering::SeqCst), 10);
}

// ============================================================================
// Thread Safety Tests
// ============================================================================

#[test]
fn handle_is_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<actor_scheduler::ActorHandle<u64, u64, u64>>();
}

#[test]
fn concurrent_clone_and_send() {
    let (tx, mut rx) = ActorScheduler::new(10, 1000);
    let count = Arc::new(AtomicUsize::new(0));
    let count_clone = count.clone();

    struct SimpleHandler {
        count: Arc<AtomicUsize>,
    }

    impl Actor<u64, u64, u64> for SimpleHandler {
        fn handle_data(&mut self, _: u64) {
            self.count.fetch_add(1, Ordering::SeqCst);
        }
        fn handle_control(&mut self, _: u64) {}
        fn handle_management(&mut self, _: u64) {}
        fn park(&mut self, _: SystemStatus) -> ActorStatus {
            ActorStatus::Idle
        }
    }

    let receiver_handle = thread::spawn(move || {
        let mut h = SimpleHandler { count: count_clone };
        rx.run(&mut h);
    });

    // Many threads clone the handle and send concurrently
    let mut handles = Vec::new();
    for _ in 0..10 {
        let tx_clone = tx.clone();
        handles.push(thread::spawn(move || {
            for i in 0..100 {
                tx_clone.send(Message::Data(i)).unwrap();
            }
        }));
    }

    for h in handles {
        h.join().unwrap();
    }

    thread::sleep(Duration::from_millis(100));
    drop(tx);
    receiver_handle.join().unwrap();

    assert_eq!(count.load(Ordering::SeqCst), 1000);
}

// ============================================================================
// Empty Message Type Tests
// ============================================================================

#[test]
fn empty_message_types_work_under_load() {
    let (tx, mut rx) = ActorScheduler::<(), (), ()>::new(10, 100);

    let receiver_handle = thread::spawn(move || {
        let mut h = NoOpHandler;
        rx.run(&mut h);
    });

    for _ in 0..1000 {
        tx.send(Message::Data(())).unwrap();
        tx.send(Message::Control(())).unwrap();
        tx.send(Message::Management(())).unwrap();
    }

    thread::sleep(Duration::from_millis(50));
    drop(tx);
    receiver_handle.join().unwrap();
}

// ============================================================================
// Large Message Tests
// ============================================================================

#[test]
fn large_messages_work() {
    struct LargeMessage {
        data: [u8; 4096],
    }

    struct LargeHandler {
        received: Arc<AtomicUsize>,
    }

    impl Actor<LargeMessage, (), ()> for LargeHandler {
        fn handle_data(&mut self, msg: LargeMessage) {
            // Verify data integrity
            assert!(msg.data.iter().all(|&b| b == 42));
            self.received.fetch_add(1, Ordering::SeqCst);
        }
        fn handle_control(&mut self, _: ()) {}
        fn handle_management(&mut self, _: ()) {}
        fn park(&mut self, _: SystemStatus) -> ActorStatus {
            ActorStatus::Idle
        }
    }

    let (tx, mut rx) = ActorScheduler::new(10, 100);
    let received = Arc::new(AtomicUsize::new(0));

    let handler = LargeHandler {
        received: received.clone(),
    };

    let receiver_handle = thread::spawn(move || {
        let mut h = handler;
        rx.run(&mut h);
    });

    for _ in 0..10 {
        tx.send(Message::Data(LargeMessage { data: [42; 4096] }))
            .unwrap();
    }

    thread::sleep(Duration::from_millis(50));
    drop(tx);
    receiver_handle.join().unwrap();

    assert_eq!(received.load(Ordering::SeqCst), 10);
}
