//! Stress tests for actor-scheduler concurrency edge cases
//!
//! These tests verify the scheduler behaves correctly under heavy load,
//! concurrent access, and various edge conditions.
//!
//! With SPSC-sharded channels, each producer has its own dedicated ring buffer.
//! Multi-producer tests use ActorBuilder to create separate handles.

use actor_scheduler::{
    Actor, ActorBuilder, ActorScheduler, ActorStatus, HandlerError, HandlerResult, Message,
    ShutdownMode, SystemStatus, WakeHandler,
};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Duration;

struct NoOpHandler;

impl Actor<(), (), ()> for NoOpHandler {
    fn handle_data(&mut self, _: ()) -> HandlerResult {
        Ok(())
    }
    fn handle_control(&mut self, _: ()) -> HandlerResult {
        Ok(())
    }
    fn handle_management(&mut self, _: ()) -> HandlerResult {
        Ok(())
    }
    fn park(&mut self, _hint: SystemStatus) -> Result<ActorStatus, HandlerError> {
        Ok(ActorStatus::Idle)
    }
}

// ============================================================================
// Rapid Channel Creation/Destruction Tests
// ============================================================================

#[test]
fn rapid_channel_creation_does_not_leak() {
    // Create and drop many channels rapidly
    for _ in 0..1000 {
        let (tx, _rx) = ActorScheduler::<u64, u64, u64>::new(10, 100);
        // Send a few messages (ignore errors - receiver not running)
        tx.send(Message::Data(1)).ok();
        tx.send(Message::Control(2)).ok();
        // Let it drop
    }
    // If we get here without running out of memory, the test passes
}

#[test]
fn rapid_producer_creation() {
    // Create and drop many producers rapidly
    let mut builder = ActorBuilder::<u64, u64, u64>::new(100, None);
    for _ in 0..100 {
        let tx = builder.add_producer();
        tx.send(Message::Data(42)).ok();
        drop(tx);
    }
    let _rx = builder.build();
}

// ============================================================================
// Wake Handler Tests
// ============================================================================

#[test]
fn custom_wake_handler_is_called() {
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
// Thread Safety Tests
// ============================================================================

#[test]
fn handle_is_send() {
    // ActorHandle is Send (can be moved to another thread) but NOT Sync
    // (Cell<usize> in SpscSender prevents shared references across threads).
    // This is correct: SPSC = one producer per channel.
    fn assert_send<T: Send>() {}
    assert_send::<actor_scheduler::ActorHandle<u64, u64, u64>>();
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
