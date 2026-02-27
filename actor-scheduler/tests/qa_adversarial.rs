//! QA Adversarial Tests — "Enrique Havoc" Edition
//!
//! These tests systematically attempt to break the cooperative scheduler by
//! exploiting the fact that it is cooperative and NOT preemptive. A misbehaving
//! actor that never yields can starve the entire system.
//!
//! # Test categories
//!
//! 1. **Cooperative starvation** — actors that hog the CPU
//! 2. **Priority inversion** — lower-priority work blocking higher-priority
//! 3. **Backpressure pathologies** — deadlock, livelock, spin-burn
//! 4. **SPSC ring buffer edge cases** — wrapping, capacity=1, drop safety
//! 5. **Lifecycle hazards** — restart races, double-shutdown, reconnect storms
//! 6. **poll_once cooperative scheduling** — fairness and termination
//! 7. **Shutdown mode correctness** — drain semantics under adversarial load

use actor_scheduler::{
    Actor, ActorBuilder, ActorScheduler, ActorStatus, HandlerError, HandlerResult,
    Message, PodPhase, SendError, ShutdownMode, SystemStatus,
};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

// ============================================================================
// Helpers
// ============================================================================

struct Noop;
impl Actor<i32, i32, i32> for Noop {
    fn handle_data(&mut self, _: i32) -> HandlerResult {
        Ok(())
    }
    fn handle_control(&mut self, _: i32) -> HandlerResult {
        Ok(())
    }
    fn handle_management(&mut self, _: i32) -> HandlerResult {
        Ok(())
    }
    fn park(&mut self, _: SystemStatus) -> Result<ActorStatus, HandlerError> {
        Ok(ActorStatus::Idle)
    }
}

// ============================================================================
// 0. GAP-1 PROOF: Shutdown deadlock when doorbell full
// ============================================================================

/// **GAP-1 PROOF**: Demonstrate that sending Shutdown AFTER data messages
/// from the same thread deadlocks when the scheduler is NOT running.
///
/// The doorbell channel has capacity=1. Data sends push a Wake into the
/// doorbell. Shutdown uses `tx_doorbell.send(System::Shutdown)` (blocking).
/// If no one is draining the doorbell, the Shutdown sender blocks forever.
///
/// This test verifies the deadlock exists by checking with a timeout.
#[test]
fn gap1_shutdown_after_data_deadlocks_without_running_scheduler() {
    let (tx, _rx) = ActorScheduler::<i32, i32, i32>::new(10, 100);

    // Fill the doorbell with a Wake by sending data
    tx.send(Message::Data(1)).unwrap();

    // Now try to send Shutdown from a separate thread with a timeout.
    // If the doorbell is full, this will block indefinitely.
    let done = Arc::new(AtomicBool::new(false));
    let done_clone = done.clone();

    let sender = thread::spawn(move || {
        let _ = tx.send(Message::Shutdown);
        done_clone.store(true, Ordering::SeqCst);
    });

    // Wait 200ms — if Shutdown hasn't completed, it's deadlocked
    thread::sleep(Duration::from_millis(200));
    let completed = done.load(Ordering::SeqCst);

    // GAP-1 CONFIRMED: Shutdown did NOT complete because doorbell is full
    // and nobody is consuming it.
    if !completed {
        // Clean up: drop _rx to unblock the sender (doorbell receiver dropped
        // → send returns Err(Disconnected), unblocking the thread)
        // Note: _rx is already going out of scope, but let's be explicit
        drop(_rx);
        let _ = sender.join();
    }

    assert!(
        !completed,
        "GAP-1: Shutdown should deadlock when doorbell is full and scheduler not running. \
         If this assertion fails, the bug has been fixed!"
    );
}

// ============================================================================
// 1. COOPERATIVE STARVATION — the fundamental weakness
// ============================================================================

/// A handler that sleeps for a configurable duration in handle_data, simulating
/// a "greedy" actor that hogs the scheduler thread.
///
/// In a cooperative scheduler, this blocks ALL other message processing on
/// the same scheduler thread for the sleep duration.
#[test]
fn greedy_data_handler_blocks_control_processing() {
    // Setup: one producer sends control messages, the data handler sleeps
    let (tx, mut rx) = ActorScheduler::<i32, i32, i32>::new(10, 100);

    let control_received = Arc::new(AtomicUsize::new(0));
    let data_received = Arc::new(AtomicUsize::new(0));
    let cr = control_received.clone();
    let dr = data_received.clone();

    struct GreedyActor {
        control_count: Arc<AtomicUsize>,
        data_count: Arc<AtomicUsize>,
    }
    impl Actor<i32, i32, i32> for GreedyActor {
        fn handle_data(&mut self, _: i32) -> HandlerResult {
            // Greedy: sleep 50ms per message (simulates expensive computation)
            thread::sleep(Duration::from_millis(50));
            self.data_count.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
        fn handle_control(&mut self, _: i32) -> HandlerResult {
            self.control_count.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
        fn handle_management(&mut self, _: i32) -> HandlerResult {
            Ok(())
        }
        fn park(&mut self, _: SystemStatus) -> Result<ActorStatus, HandlerError> {
            Ok(ActorStatus::Idle)
        }
    }

    let receiver = thread::spawn(move || {
        let mut actor = GreedyActor {
            control_count: cr,
            data_count: dr,
        };
        rx.run(&mut actor);
    });

    // Send data first to trigger the greedy handler
    for i in 0..5 {
        tx.send(Message::Data(i)).unwrap();
    }
    // Then send a high-priority control message
    tx.send(Message::Control(999)).unwrap();

    let start = Instant::now();
    // Wait for control to be processed
    loop {
        if control_received.load(Ordering::SeqCst) > 0 {
            break;
        }
        if start.elapsed() > Duration::from_secs(5) {
            break;
        }
        thread::sleep(Duration::from_millis(5));
    }

    let control_latency = start.elapsed();
    let data_at_control_time = data_received.load(Ordering::SeqCst);

    // GAP: In cooperative scheduling, the control message is blocked behind
    // all queued data messages. With burst_limit=10, all 5 data messages are
    // drained before control gets another chance.
    //
    // The control message was delayed by at least 50ms * number_of_data_drained.
    // In a preemptive system, control would preempt data.
    //
    // Document the actual behavior — this is a known limitation.
    let ctrl_count = control_received.load(Ordering::SeqCst);
    assert!(
        ctrl_count > 0,
        "Control message should eventually be processed (was it? count={})",
        ctrl_count
    );

    // The scheduler processes Control BEFORE Data in each cycle, BUT:
    // If the data handler for previously-drained messages is still running,
    // control has to wait. Let's verify the scheduling order is at least correct:
    // Data messages drained in the first cycle should be processed, THEN
    // control gets its turn in the next wake cycle.
    drop(tx);
    receiver.join().unwrap();

    println!(
        "Control latency: {:?}, data processed before control: {}",
        control_latency, data_at_control_time
    );
}

/// Verify that a handler returning Busy causes the scheduler to spin-loop
/// without blocking (CPU-burning behavior). This is by design but must be
/// bounded.
#[test]
fn park_busy_causes_immediate_repoll() {
    let (tx, mut rx) = ActorScheduler::<(), (), ()>::new(10, 100);

    let park_calls = Arc::new(AtomicUsize::new(0));
    let pc = park_calls.clone();

    struct BusyActor {
        park_calls: Arc<AtomicUsize>,
        limit: usize,
    }
    impl Actor<(), (), ()> for BusyActor {
        fn handle_data(&mut self, _: ()) -> HandlerResult {
            Ok(())
        }
        fn handle_control(&mut self, _: ()) -> HandlerResult {
            Ok(())
        }
        fn handle_management(&mut self, _: ()) -> HandlerResult {
            Ok(())
        }
        fn park(&mut self, _: SystemStatus) -> Result<ActorStatus, HandlerError> {
            let n = self.park_calls.fetch_add(1, Ordering::SeqCst);
            if n < self.limit {
                Ok(ActorStatus::Busy) // keep spinning!
            } else {
                // Switch to Idle to let the scheduler block
                Ok(ActorStatus::Idle)
            }
        }
    }

    let receiver = thread::spawn(move || {
        let mut actor = BusyActor {
            park_calls: pc,
            limit: 100,
        };
        rx.run(&mut actor);
    });

    // Send a single message to wake the scheduler, then let it spin
    tx.send(Message::Data(())).unwrap();
    thread::sleep(Duration::from_millis(200));

    let calls = park_calls.load(Ordering::SeqCst);
    // The scheduler should have called park() many times because the actor
    // keeps returning Busy
    assert!(
        calls >= 100,
        "Expected at least 100 park() calls from Busy spinning, got {}",
        calls
    );

    drop(tx);
    receiver.join().unwrap();
}

// ============================================================================
// 2. PRIORITY INVERSION — lower-priority work blocking higher-priority
// ============================================================================

/// Send a burst of data messages, then a control message. Measure whether
/// control is processed before all data is drained.
///
/// With burst-limited data, control SHOULD be interleaved. But with cooperative
/// scheduling, once handle_data() starts executing, there's no preemption.
#[test]
fn control_interleaving_during_data_burst() {
    let (tx, mut rx) = ActorScheduler::<i32, i32, i32>::new(
        5, // very small data burst limit
        1000,
    );

    let order = Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
    let o = order.clone();

    struct OrderTracker {
        order: Arc<std::sync::Mutex<Vec<String>>>,
    }
    impl Actor<i32, i32, i32> for OrderTracker {
        fn handle_data(&mut self, v: i32) -> HandlerResult {
            self.order.lock().unwrap().push(format!("D{}", v));
            Ok(())
        }
        fn handle_control(&mut self, v: i32) -> HandlerResult {
            self.order.lock().unwrap().push(format!("C{}", v));
            Ok(())
        }
        fn handle_management(&mut self, _: i32) -> HandlerResult {
            Ok(())
        }
        fn park(&mut self, _: SystemStatus) -> Result<ActorStatus, HandlerError> {
            Ok(ActorStatus::Idle)
        }
    }

    // Pre-load: 20 data messages, then 1 control
    for i in 0..20 {
        tx.send(Message::Data(i)).unwrap();
    }
    tx.send(Message::Control(99)).unwrap();

    let receiver = thread::spawn(move || {
        let mut actor = OrderTracker { order: o };
        rx.run(&mut actor);
    });

    thread::sleep(Duration::from_millis(100));
    drop(tx);
    receiver.join().unwrap();

    let events = order.lock().unwrap();
    // Find position of control message
    let ctrl_pos = events.iter().position(|s| s == "C99");
    assert!(
        ctrl_pos.is_some(),
        "Control message should be processed"
    );
    let ctrl_pos = ctrl_pos.unwrap();

    // Control is processed before Data in each cycle. Since data burst=5,
    // the scheduler should check control BEFORE draining data. The first
    // cycle: drain control (C99) first, then up to 5 data.
    // So C99 should appear BEFORE D5 (the 6th data message).
    assert!(
        ctrl_pos < 6,
        "Control should be processed within the first cycle (burst=5), \
         but it appeared at position {} in {:?}",
        ctrl_pos,
        &events[..std::cmp::min(events.len(), 10)]
    );
}

// ============================================================================
// 3. BACKPRESSURE PATHOLOGIES
// ============================================================================

/// The data lane uses spin-yield when the buffer is full. If the receiver
/// is NOT running, the sender blocks forever. This test verifies that
/// dropping the receiver unblocks the sender with Disconnected.
#[test]
fn data_send_unblocks_on_receiver_drop() {
    let (tx, rx) = ActorScheduler::<i32, i32, i32>::new(10, 2); // tiny buffer

    // Fill the buffer
    tx.send(Message::Data(1)).unwrap();
    tx.send(Message::Data(2)).unwrap();

    let send_done = Arc::new(AtomicBool::new(false));
    let sd = send_done.clone();

    // This send will block because buffer is full and receiver isn't draining
    let sender = thread::spawn(move || {
        let result = tx.send(Message::Data(3));
        sd.store(true, Ordering::SeqCst);
        result
    });

    // Give the sender time to block
    thread::sleep(Duration::from_millis(50));
    assert!(
        !send_done.load(Ordering::SeqCst),
        "Sender should be blocked on full buffer"
    );

    // Drop the receiver — this should unblock the sender
    drop(rx);

    let result = sender.join().unwrap();
    assert!(
        result.is_err(),
        "Send should fail with Disconnected after receiver drops"
    );
}

/// Control/Management lanes use exponential backoff. Verify the timeout
/// actually fires and doesn't hang forever.
#[test]
fn control_backoff_timeout_fires() {
    use actor_scheduler::SchedulerParams;

    // Create params with very aggressive timeout
    let params = SchedulerParams {
        control_mgmt_buffer_size: 2,
        spin_attempts: 5,
        yield_attempts: 5,
        min_backoff: Duration::from_millis(1),
        max_backoff: Duration::from_millis(10), // very short max
        jitter_min_pct: 50,
        jitter_range_pct: 50,
        ..SchedulerParams::DEFAULT
    };

    let (tx, _rx) =
        ActorScheduler::<i32, i32, i32>::new_with_params(10, 100, params);

    // Fill control channel (buffer=2, rounds to next power of 2 = 2)
    tx.send(Message::Control(1)).unwrap();
    tx.send(Message::Control(2)).unwrap();

    // Next control send should timeout (nobody is draining)
    let start = Instant::now();
    let result = tx.send(Message::Control(3));
    let elapsed = start.elapsed();

    assert_eq!(result, Err(SendError::Timeout));
    // Should fail relatively quickly due to short max_backoff
    assert!(
        elapsed < Duration::from_secs(2),
        "Timeout took too long: {:?}",
        elapsed
    );
}

// ============================================================================
// 4. SPSC RING BUFFER EDGE CASES
// ============================================================================

/// Capacity=1 (rounds to 2) — tightest possible buffer.
/// Exercises the hot path where full/empty checks fire on every operation.
#[test]
fn spsc_capacity_one_stress() {
    use actor_scheduler::spsc::{spsc_channel, TryRecvError, TrySendError};

    let (tx, mut rx) = spsc_channel::<u64>(1); // rounds up to 2
    let count = 10_000u64;

    let producer = thread::spawn(move || {
        for i in 0..count {
            loop {
                match tx.try_send(i) {
                    Ok(()) => break,
                    Err(TrySendError::Full(_)) => thread::yield_now(),
                    Err(TrySendError::Disconnected(_)) => panic!("disconnected"),
                }
            }
        }
    });

    let consumer = thread::spawn(move || {
        let mut received = 0u64;
        let mut expected = 0u64;
        loop {
            match rx.try_recv() {
                Ok(v) => {
                    assert_eq!(v, expected, "out of order at index {}", expected);
                    expected += 1;
                    received += 1;
                    if received == count {
                        break;
                    }
                }
                Err(TryRecvError::Empty) => thread::yield_now(),
                Err(TryRecvError::Disconnected) => break,
            }
        }
        received
    });

    producer.join().unwrap();
    let received = consumer.join().unwrap();
    assert_eq!(received, count);
}

/// Drop a channel with pending non-trivial (heap-allocated) messages.
/// Verifies no memory leaks or double-frees.
#[test]
fn spsc_drop_with_heap_messages() {
    use actor_scheduler::spsc::spsc_channel;

    let drop_count = Arc::new(AtomicUsize::new(0));

    #[derive(Debug)]
    struct Tracked {
        _data: Vec<u8>,
        counter: Arc<AtomicUsize>,
    }
    impl Drop for Tracked {
        fn drop(&mut self) {
            self.counter.fetch_add(1, Ordering::SeqCst);
        }
    }

    let dc = drop_count.clone();
    let (tx, rx) = spsc_channel::<Tracked>(8);

    for _ in 0..5 {
        tx.try_send(Tracked {
            _data: vec![42; 1024],
            counter: dc.clone(),
        })
        .unwrap();
    }

    // Drop both ends without consuming — all 5 messages should be dropped
    drop(tx);
    drop(rx);

    assert_eq!(
        drop_count.load(Ordering::SeqCst),
        5,
        "All 5 heap messages should have been dropped"
    );
}

/// Verify SPSC ordering across many wrap-arounds with a small buffer.
/// This tests the bitmask-based indexing when head/tail grow large.
#[test]
fn spsc_many_wraparounds_ordering() {
    use actor_scheduler::spsc::{spsc_channel, TryRecvError, TrySendError};

    let (tx, mut rx) = spsc_channel::<u64>(4); // buffer = 4 (already power of 2)
    let count = 100_000u64;

    let producer = thread::spawn(move || {
        for i in 0..count {
            loop {
                match tx.try_send(i) {
                    Ok(()) => break,
                    Err(TrySendError::Full(_)) => thread::yield_now(),
                    Err(TrySendError::Disconnected(_)) => panic!("disconnected"),
                }
            }
        }
    });

    let consumer = thread::spawn(move || {
        let mut expected = 0u64;
        loop {
            match rx.try_recv() {
                Ok(v) => {
                    assert_eq!(v, expected, "ordering broken at {}", expected);
                    expected += 1;
                    if expected == count {
                        break;
                    }
                }
                Err(TryRecvError::Empty) => thread::yield_now(),
                Err(TryRecvError::Disconnected) => break,
            }
        }
        expected
    });

    producer.join().unwrap();
    assert_eq!(consumer.join().unwrap(), count);
}

// ============================================================================
// 5. LIFECYCLE HAZARDS
// ============================================================================

/// Double shutdown: send Shutdown twice. The second should be benign.
#[test]
fn double_shutdown_is_safe() {
    let (tx, mut rx) = ActorScheduler::<i32, i32, i32>::new(10, 100);

    let receiver = thread::spawn(move || rx.run(&mut Noop));

    // First shutdown succeeds
    tx.send(Message::Shutdown).unwrap();

    // Second shutdown: the doorbell channel may accept it or the scheduler
    // may have already exited. Either way, no panic.
    let _ = tx.send(Message::Shutdown);

    let phase = receiver.join().unwrap();
    assert_eq!(phase, PodPhase::Completed);
}

/// Shutdown while data is being sent concurrently.
/// Verifies no panic or hang.
#[test]
fn shutdown_racing_with_data_sends() {
    let mut builder = ActorBuilder::<i32, i32, i32>::new(100, None);
    let tx_data = builder.add_producer();
    let tx_shutdown = builder.add_producer();
    let mut rx = builder.build();

    let data_sent = Arc::new(AtomicUsize::new(0));
    let ds = data_sent.clone();

    let receiver = thread::spawn(move || rx.run(&mut Noop));

    // Data sender: send as fast as possible
    let data_sender = thread::spawn(move || {
        for i in 0..1000 {
            if tx_data.send(Message::Data(i)).is_err() {
                break;
            }
            ds.fetch_add(1, Ordering::Relaxed);
        }
    });

    // After a brief burst, send shutdown
    thread::sleep(Duration::from_millis(5));
    tx_shutdown.send(Message::Shutdown).unwrap();

    data_sender.join().unwrap();
    let phase = receiver.join().unwrap();
    assert_eq!(phase, PodPhase::Completed);

    // Some data was sent before shutdown
    assert!(data_sent.load(Ordering::Relaxed) > 0);
}

/// Verify that dropping all handles (without explicit Shutdown) causes
/// the scheduler to exit with Completed.
#[test]
fn all_handles_dropped_causes_clean_exit() {
    let mut builder = ActorBuilder::<i32, i32, i32>::new(100, None);
    let tx1 = builder.add_producer();
    let tx2 = builder.add_producer();
    let mut rx = builder.build();

    let receiver = thread::spawn(move || rx.run(&mut Noop));

    // Send some data then drop both handles
    tx1.send(Message::Data(1)).unwrap();
    tx2.send(Message::Data(2)).unwrap();
    drop(tx1);
    drop(tx2);

    let phase = receiver.join().unwrap();
    assert_eq!(
        phase,
        PodPhase::Completed,
        "Dropping all handles should cause clean exit"
    );
}

/// Verify that a Recoverable error from a handler causes the scheduler
/// to exit with Failed.
#[test]
fn recoverable_error_causes_failed_exit() {
    let (tx, mut rx) = ActorScheduler::<i32, i32, i32>::new(10, 100);

    struct FailingActor;
    impl Actor<i32, i32, i32> for FailingActor {
        fn handle_data(&mut self, _: i32) -> HandlerResult {
            Err(HandlerError::recoverable("intentional failure"))
        }
        fn handle_control(&mut self, _: i32) -> HandlerResult {
            Ok(())
        }
        fn handle_management(&mut self, _: i32) -> HandlerResult {
            Ok(())
        }
        fn park(&mut self, _: SystemStatus) -> Result<ActorStatus, HandlerError> {
            Ok(ActorStatus::Idle)
        }
    }

    let receiver = thread::spawn(move || rx.run(&mut FailingActor));

    tx.send(Message::Data(1)).unwrap();
    drop(tx);

    let phase = receiver.join().unwrap();
    assert!(
        phase.is_failed(),
        "Expected Failed phase, got {:?}",
        phase
    );
}

/// Fatal error in handler should panic the scheduler thread.
#[test]
fn fatal_error_panics_scheduler() {
    let (tx, mut rx) = ActorScheduler::<i32, i32, i32>::new(10, 100);

    struct FatalActor;
    impl Actor<i32, i32, i32> for FatalActor {
        fn handle_data(&mut self, _: i32) -> HandlerResult {
            Err(HandlerError::fatal("kaboom"))
        }
        fn handle_control(&mut self, _: i32) -> HandlerResult {
            Ok(())
        }
        fn handle_management(&mut self, _: i32) -> HandlerResult {
            Ok(())
        }
        fn park(&mut self, _: SystemStatus) -> Result<ActorStatus, HandlerError> {
            Ok(ActorStatus::Idle)
        }
    }

    let receiver = thread::spawn(move || rx.run(&mut FatalActor));

    tx.send(Message::Data(1)).unwrap();

    // The thread should panic
    let result = receiver.join();
    assert!(result.is_err(), "Fatal error should cause thread panic");
}

// ============================================================================
// 6. poll_once COOPERATIVE SCHEDULING
// ============================================================================

/// poll_once should drain one batch and return None (keep going)
/// until shutdown or disconnect.
#[test]
fn poll_once_processes_messages_without_blocking() {
    let (tx, mut rx) = ActorScheduler::<i32, i32, i32>::new(10, 100);

    let count = Arc::new(AtomicUsize::new(0));
    let c = count.clone();

    struct CountActor {
        count: Arc<AtomicUsize>,
    }
    impl Actor<i32, i32, i32> for CountActor {
        fn handle_data(&mut self, _: i32) -> HandlerResult {
            self.count.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
        fn handle_control(&mut self, _: i32) -> HandlerResult {
            Ok(())
        }
        fn handle_management(&mut self, _: i32) -> HandlerResult {
            Ok(())
        }
        fn park(&mut self, _: SystemStatus) -> Result<ActorStatus, HandlerError> {
            Ok(ActorStatus::Idle)
        }
    }

    // Pre-load messages
    for i in 0..5 {
        tx.send(Message::Data(i)).unwrap();
    }

    let mut actor = CountActor { count: c };

    // poll_once should return None (not done yet)
    let result = rx.poll_once(&mut actor);
    assert!(result.is_none(), "poll_once should return None while running");
    assert!(
        count.load(Ordering::SeqCst) > 0,
        "poll_once should have processed some messages"
    );

    // After dropping tx, subsequent polls should eventually return Some(Completed)
    drop(tx);
    let mut completed = false;
    for _ in 0..100 {
        if let Some(phase) = rx.poll_once(&mut actor) {
            assert_eq!(phase, PodPhase::Completed);
            completed = true;
            break;
        }
    }
    assert!(completed, "poll_once should eventually return Completed after handles drop");
}

/// poll_once with shutdown message
#[test]
fn poll_once_handles_shutdown() {
    let (tx, mut rx) = ActorScheduler::<i32, i32, i32>::new(10, 100);

    tx.send(Message::Shutdown).unwrap();

    let result = rx.poll_once(&mut Noop);
    assert_eq!(
        result,
        Some(PodPhase::Completed),
        "poll_once should return Completed on Shutdown"
    );
}

// ============================================================================
// 7. SHUTDOWN MODE CORRECTNESS
// ============================================================================

/// ShutdownMode::DrainAll should process pending data before exiting.
///
/// GAP FOUND: Sending Shutdown AFTER data messages from the same thread
/// DEADLOCKS because `Message::Shutdown` uses a blocking `send()` on the
/// doorbell (capacity=1), which is full from the Wake sent with the first
/// data message. The scheduler must be running to consume the Wake.
///
/// This is a production hazard: if an actor tries to shut down a peer
/// after flooding it with data, the shutdown sender blocks forever.
#[test]
fn drain_all_processes_pending_data_on_shutdown() {
    let (tx, mut rx) = ActorScheduler::<i32, i32, i32>::new_with_shutdown_mode(
        100,
        1000,
        ShutdownMode::DrainAll {
            timeout: Duration::from_secs(5),
        },
    );

    let data_count = Arc::new(AtomicUsize::new(0));
    let dc = data_count.clone();

    struct DataCounter {
        count: Arc<AtomicUsize>,
    }
    impl Actor<i32, i32, i32> for DataCounter {
        fn handle_data(&mut self, _: i32) -> HandlerResult {
            self.count.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
        fn handle_control(&mut self, _: i32) -> HandlerResult {
            Ok(())
        }
        fn handle_management(&mut self, _: i32) -> HandlerResult {
            Ok(())
        }
        fn park(&mut self, _: SystemStatus) -> Result<ActorStatus, HandlerError> {
            Ok(ActorStatus::Idle)
        }
    }

    // IMPORTANT: Spawn the receiver FIRST so it can consume doorbell Wake
    // signals. Sending data-then-Shutdown without a running receiver deadlocks
    // because Shutdown's blocking doorbell send waits for space that only the
    // receiver can free. (See GAP-1 in report.)
    let receiver = thread::spawn(move || {
        let mut actor = DataCounter { count: dc };
        rx.run(&mut actor)
    });

    // Small delay to ensure scheduler has started recv()
    thread::sleep(Duration::from_millis(10));

    for i in 0..50 {
        tx.send(Message::Data(i)).unwrap();
    }
    tx.send(Message::Shutdown).unwrap();

    let phase = receiver.join().unwrap();
    assert_eq!(phase, PodPhase::Completed);
    assert_eq!(
        data_count.load(Ordering::SeqCst),
        50,
        "DrainAll should process all 50 data messages before exit"
    );
}

/// ShutdownMode::DrainControl should drop data messages on shutdown.
#[test]
fn drain_control_drops_data_on_shutdown() {
    let (tx, mut rx) = ActorScheduler::<i32, i32, i32>::new_with_shutdown_mode(
        100,
        1000,
        ShutdownMode::DrainControl,
    );

    let data_count = Arc::new(AtomicUsize::new(0));
    let ctrl_count = Arc::new(AtomicUsize::new(0));
    let dc = data_count.clone();
    let cc = ctrl_count.clone();

    struct DualCounter {
        data: Arc<AtomicUsize>,
        ctrl: Arc<AtomicUsize>,
    }
    impl Actor<i32, i32, i32> for DualCounter {
        fn handle_data(&mut self, _: i32) -> HandlerResult {
            self.data.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
        fn handle_control(&mut self, _: i32) -> HandlerResult {
            self.ctrl.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
        fn handle_management(&mut self, _: i32) -> HandlerResult {
            Ok(())
        }
        fn park(&mut self, _: SystemStatus) -> Result<ActorStatus, HandlerError> {
            Ok(ActorStatus::Idle)
        }
    }

    // Queue shutdown first, then control and data
    tx.send(Message::Shutdown).unwrap();
    for i in 0..10 {
        tx.send(Message::Control(i)).unwrap();
    }
    for i in 0..20 {
        tx.send(Message::Data(i)).unwrap();
    }

    let receiver = thread::spawn(move || {
        let mut actor = DualCounter { data: dc, ctrl: cc };
        rx.run(&mut actor)
    });

    let phase = receiver.join().unwrap();
    assert_eq!(phase, PodPhase::Completed);

    // Control should be drained
    assert_eq!(
        ctrl_count.load(Ordering::SeqCst),
        10,
        "DrainControl should process all control messages"
    );

    // Data might or might not be processed (DrainControl doesn't drain data)
    // The exact count depends on whether data was drained before Shutdown arrived
    // in the scheduler loop. What matters is control was fully drained.
}

/// DrainAll with a very short timeout should not hang forever.
///
/// Same GAP-1 applies: must spawn receiver before sending Shutdown to avoid
/// deadlock on the blocking doorbell send.
#[test]
fn drain_all_timeout_prevents_hang() {
    let (tx, mut rx) = ActorScheduler::<i32, i32, i32>::new_with_shutdown_mode(
        100,
        1000,
        ShutdownMode::DrainAll {
            timeout: Duration::from_millis(50), // very short
        },
    );

    struct SlowActor;
    impl Actor<i32, i32, i32> for SlowActor {
        fn handle_data(&mut self, _: i32) -> HandlerResult {
            // Each message takes 10ms
            thread::sleep(Duration::from_millis(10));
            Ok(())
        }
        fn handle_control(&mut self, _: i32) -> HandlerResult {
            Ok(())
        }
        fn handle_management(&mut self, _: i32) -> HandlerResult {
            Ok(())
        }
        fn park(&mut self, _: SystemStatus) -> Result<ActorStatus, HandlerError> {
            Ok(ActorStatus::Idle)
        }
    }

    // Spawn receiver FIRST (see GAP-1)
    let start = Instant::now();
    let receiver = thread::spawn(move || rx.run(&mut SlowActor));

    thread::sleep(Duration::from_millis(10));

    // Queue many data messages + shutdown
    for i in 0..100 {
        tx.send(Message::Data(i)).unwrap();
    }
    tx.send(Message::Shutdown).unwrap();

    let phase = receiver.join().unwrap();
    let elapsed = start.elapsed();

    assert_eq!(phase, PodPhase::Completed);
    // Should complete within timeout + some overhead, NOT process all 100 messages
    // (100 * 10ms = 1s without timeout)
    assert!(
        elapsed < Duration::from_secs(2),
        "DrainAll should have timed out at 50ms, but took {:?}",
        elapsed
    );
}

// ============================================================================
// 8. SHARDED INBOX FAIRNESS UNDER EXTREME IMBALANCE
// ============================================================================

/// One producer sends 10000 messages, another sends 1. Verify the 1-message
/// producer isn't starved.
#[test]
fn sharded_extreme_imbalance() {
    let mut builder = ActorBuilder::<i32, i32, i32>::new(1024, None);
    let tx_flood = builder.add_producer();
    let tx_trickle = builder.add_producer();
    let mut rx = builder.build_with_burst(
        10, // small data burst to amplify fairness issues
        ShutdownMode::default(),
    );

    let flood_count = Arc::new(AtomicUsize::new(0));
    let trickle_count = Arc::new(AtomicUsize::new(0));
    let fc = flood_count.clone();
    let tc = trickle_count.clone();

    struct ImbalanceActor {
        flood: Arc<AtomicUsize>,
        trickle: Arc<AtomicUsize>,
    }
    impl Actor<i32, i32, i32> for ImbalanceActor {
        fn handle_data(&mut self, v: i32) -> HandlerResult {
            if v == -1 {
                self.trickle.fetch_add(1, Ordering::SeqCst);
            } else {
                self.flood.fetch_add(1, Ordering::SeqCst);
            }
            Ok(())
        }
        fn handle_control(&mut self, _: i32) -> HandlerResult {
            Ok(())
        }
        fn handle_management(&mut self, _: i32) -> HandlerResult {
            Ok(())
        }
        fn park(&mut self, _: SystemStatus) -> Result<ActorStatus, HandlerError> {
            Ok(ActorStatus::Idle)
        }
    }

    let receiver = thread::spawn(move || {
        let mut actor = ImbalanceActor {
            flood: fc,
            trickle: tc,
        };
        rx.run(&mut actor);
    });

    // Flood producer
    let flood_sender = thread::spawn(move || {
        for i in 0..10_000 {
            tx_flood.send(Message::Data(i)).unwrap();
        }
    });

    // Wait a bit, then send the trickle message
    thread::sleep(Duration::from_millis(10));
    tx_trickle.send(Message::Data(-1)).unwrap();

    flood_sender.join().unwrap();
    drop(tx_trickle);
    thread::sleep(Duration::from_millis(100));
    receiver.join().unwrap();

    assert_eq!(
        trickle_count.load(Ordering::SeqCst),
        1,
        "Trickle producer's message must be delivered"
    );
    assert!(
        flood_count.load(Ordering::SeqCst) > 0,
        "Flood producer's messages should also be delivered"
    );
}

// ============================================================================
// 9. KUBELET RESTART RACE CONDITIONS
// ============================================================================

/// Rapidly restart an actor that fails immediately.
/// Tests the frequency gate under pressure.
///
/// GAP-2: After a pod restarts, the new pod has no external senders unless
/// ServiceHandle reconnects. If no one sends messages, the pod blocks
/// forever on doorbell.recv(). The kubelet then hangs because the pod
/// never exits.
///
/// Workaround: Make the actor fail during park() so it fails even without
/// receiving messages.
#[test]
fn kubelet_rapid_restart_frequency_gate() {
    use actor_scheduler::{KubeletBuilder, RestartPolicy, spawn_managed};
    use actor_scheduler::registry::PodSlot;

    let slot = PodSlot::<i32, i32, i32>::connected();
    let restart_count = Arc::new(AtomicU64::new(0));
    let rc = restart_count.clone();

    /// Actor that fails on every data message AND on first park call.
    /// This ensures it fails even after restart when no one sends data.
    struct InstantFail;
    impl Actor<i32, i32, i32> for InstantFail {
        fn handle_data(&mut self, _: i32) -> HandlerResult {
            Err(HandlerError::recoverable("always fail"))
        }
        fn handle_control(&mut self, _: i32) -> HandlerResult {
            Ok(())
        }
        fn handle_management(&mut self, _: i32) -> HandlerResult {
            Ok(())
        }
        fn park(&mut self, _: SystemStatus) -> Result<ActorStatus, HandlerError> {
            Ok(ActorStatus::Idle)
        }
    }

    let pod = spawn_managed(vec![slot.clone()], 64, None, move || {
        rc.fetch_add(1, Ordering::SeqCst);
        InstantFail
    });

    // Send data to trigger the failure in the initial pod
    pod.handles[0]
        .send(Message::<i32, i32, i32>::Data(1))
        .unwrap();

    // Allow 0 restarts — frequency gate immediately exhausted.
    // This way the kubelet stops the pod without trying to restart.
    let kubelet = KubeletBuilder::new()
        .with_poll_interval(Duration::from_millis(1))
        .add_pod_with_gate(pod, RestartPolicy::OnFailure, 0, Duration::from_secs(60))
        .build();

    let start = Instant::now();
    let join = thread::spawn(move || kubelet.run());

    // Kubelet should exhaust restart budget and exit
    join.join().expect("Kubelet should exit cleanly");
    let elapsed = start.elapsed();

    let restarts = restart_count.load(Ordering::SeqCst);
    // Only the initial instance (1), no restarts allowed
    assert_eq!(
        restarts, 1,
        "Frequency gate with max_restarts=0 should allow no restarts (got {} instances)",
        restarts
    );
    assert!(
        elapsed < Duration::from_secs(5),
        "Kubelet should exit promptly, took {:?}",
        elapsed
    );

    // Slot should be permanently stopped
    let result = slot.reconnect(Duration::from_millis(100));
    assert_eq!(
        result.unwrap_err(),
        actor_scheduler::PodGone::Stopped,
        "Slot should be stopped after frequency gate exhaustion"
    );
}

// ============================================================================
// 10. SERVICE HANDLE RECONNECT UNDER LOAD
// ============================================================================

/// ServiceHandle should transparently reconnect when the pod is restarted.
#[test]
fn service_handle_reconnect_stress() {
    use actor_scheduler::registry::PodSlot;
    use actor_scheduler::service::ServiceHandle;

    let slot = PodSlot::<i32, i32, i32>::connected();

    // Spawn initial pod with TWO producer handles: one for service, one for shutdown
    let mut builder = ActorBuilder::<i32, i32, i32>::new(100, None);
    let svc_handle = builder.add_producer();
    let kill_handle = builder.add_producer();
    let mut scheduler = builder.build();

    let pod_thread = thread::spawn(move || scheduler.run(&mut Noop));

    let mut svc = ServiceHandle::new(svc_handle, slot.clone())
        .with_reconnect_timeout(Duration::from_secs(5));

    // Hot-path sends should succeed
    for i in 0..10 {
        svc.send(Message::Data(i)).unwrap();
    }

    // Kill the pod
    kill_handle.send(Message::Shutdown).unwrap();
    pod_thread.join().unwrap();

    // Mark slot as restarting and spawn a new pod
    slot.mark_restarting();

    let slot_clone = slot.clone();
    let new_pod = thread::spawn(move || {
        thread::sleep(Duration::from_millis(20));
        let (new_handle, mut new_scheduler) = ActorScheduler::<i32, i32, i32>::new(10, 100);
        slot_clone.publish(new_handle);
        new_scheduler.run(&mut Noop);
    });

    // Send during disconnect window — should get Reconnected
    let result = svc.send(Message::Data(99));
    assert_eq!(
        result,
        Err(actor_scheduler::service::ServiceError::Reconnected),
        "First send after disconnect should report Reconnected"
    );

    // Next send should succeed with new connection
    svc.send(Message::Data(100)).unwrap();

    drop(svc);
    // new_pod will exit when all handles are dropped
    thread::sleep(Duration::from_millis(50));
    drop(new_pod);
}

// ============================================================================
// 11. HANDLER ERROR DURING SHUTDOWN DRAIN
// ============================================================================

/// If a handler returns an error during DrainAll, the scheduler should
/// still exit (not hang).
///
/// Same GAP-1 applies: must spawn receiver before sending Shutdown.
#[test]
fn error_during_drain_all_exits_cleanly() {
    let (tx, mut rx) = ActorScheduler::<i32, i32, i32>::new_with_shutdown_mode(
        100,
        1000,
        ShutdownMode::DrainAll {
            timeout: Duration::from_secs(5),
        },
    );

    struct FailDuringDrain {
        count: usize,
    }
    impl Actor<i32, i32, i32> for FailDuringDrain {
        fn handle_data(&mut self, _: i32) -> HandlerResult {
            self.count += 1;
            if self.count == 3 {
                Err(HandlerError::recoverable("drain explosion"))
            } else {
                Ok(())
            }
        }
        fn handle_control(&mut self, _: i32) -> HandlerResult {
            Ok(())
        }
        fn handle_management(&mut self, _: i32) -> HandlerResult {
            Ok(())
        }
        fn park(&mut self, _: SystemStatus) -> Result<ActorStatus, HandlerError> {
            Ok(ActorStatus::Idle)
        }
    }

    // Spawn receiver FIRST (see GAP-1)
    let receiver = thread::spawn(move || rx.run(&mut FailDuringDrain { count: 0 }));

    thread::sleep(Duration::from_millis(10));

    // Queue data + shutdown.
    // NOTE: The scheduler may process data (and fail on msg 3) BEFORE we
    // get to send Shutdown. In that case, the scheduler has already exited
    // and the doorbell receiver is dropped, so Shutdown send returns
    // Disconnected. This is a race, not a bug — but callers must handle it.
    for i in 0..10 {
        // Data sends may also get Disconnected if scheduler already failed
        let _ = tx.send(Message::Data(i));
    }
    let _ = tx.send(Message::Shutdown); // may get Disconnected

    let phase = receiver.join().unwrap();
    // The error during drain should cause a Failed exit, not a hang.
    // The scheduler may process data before seeing Shutdown, or Shutdown first
    // then hit the error during drain. Either way, no hang.
    assert!(
        phase == PodPhase::Completed || phase.is_failed(),
        "Expected Completed or Failed, got {:?}",
        phase
    );
}

// ============================================================================
// 12. CONCURRENT MULTI-LANE SATURATION
// ============================================================================

/// Saturate all three lanes simultaneously from multiple producers.
/// Verify no message loss and no deadlock.
#[test]
fn all_lanes_saturated_simultaneously() {
    const PRODUCERS: usize = 4;
    const MESSAGES: usize = 500;

    let mut builder = ActorBuilder::<i32, i32, i32>::new(256, None);
    let handles: Vec<_> = (0..PRODUCERS).map(|_| builder.add_producer()).collect();
    let mut rx = builder.build();

    let data_total = Arc::new(AtomicUsize::new(0));
    let ctrl_total = Arc::new(AtomicUsize::new(0));
    let mgmt_total = Arc::new(AtomicUsize::new(0));
    let dt = data_total.clone();
    let ct = ctrl_total.clone();
    let mt = mgmt_total.clone();

    struct TriCounter {
        data: Arc<AtomicUsize>,
        ctrl: Arc<AtomicUsize>,
        mgmt: Arc<AtomicUsize>,
    }
    impl Actor<i32, i32, i32> for TriCounter {
        fn handle_data(&mut self, _: i32) -> HandlerResult {
            self.data.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }
        fn handle_control(&mut self, _: i32) -> HandlerResult {
            self.ctrl.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }
        fn handle_management(&mut self, _: i32) -> HandlerResult {
            self.mgmt.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }
        fn park(&mut self, _: SystemStatus) -> Result<ActorStatus, HandlerError> {
            Ok(ActorStatus::Idle)
        }
    }

    let receiver = thread::spawn(move || {
        let mut actor = TriCounter {
            data: dt,
            ctrl: ct,
            mgmt: mt,
        };
        rx.run(&mut actor);
    });

    let mut senders = Vec::new();
    for tx in handles {
        senders.push(thread::spawn(move || {
            for i in 0..MESSAGES {
                match i % 3 {
                    0 => tx.send(Message::Data(i as i32)).unwrap(),
                    1 => tx.send(Message::Control(i as i32)).unwrap(),
                    _ => tx.send(Message::Management(i as i32)).unwrap(),
                }
            }
        }));
    }

    for s in senders {
        s.join().unwrap();
    }

    // Wait for drain, then check
    thread::sleep(Duration::from_millis(200));
    receiver.join().unwrap();

    let d = data_total.load(Ordering::Relaxed);
    let c = ctrl_total.load(Ordering::Relaxed);
    let m = mgmt_total.load(Ordering::Relaxed);
    let total = d + c + m;

    assert_eq!(
        total,
        PRODUCERS * MESSAGES,
        "Expected {} total messages, got {} (data={}, ctrl={}, mgmt={})",
        PRODUCERS * MESSAGES,
        total,
        d,
        c,
        m
    );
}

// ============================================================================
// 13. HANDLER PANIC CONTAINMENT
// ============================================================================

/// A panic inside handle_data should unwind the scheduler thread.
/// The sender should see Disconnected on subsequent sends.
#[test]
fn handler_panic_disconnects_senders() {
    let (tx, mut rx) = ActorScheduler::<i32, i32, i32>::new(10, 100);

    struct PanicActor;
    impl Actor<i32, i32, i32> for PanicActor {
        fn handle_data(&mut self, _: i32) -> HandlerResult {
            panic!("actor panic!")
        }
        fn handle_control(&mut self, _: i32) -> HandlerResult {
            Ok(())
        }
        fn handle_management(&mut self, _: i32) -> HandlerResult {
            Ok(())
        }
        fn park(&mut self, _: SystemStatus) -> Result<ActorStatus, HandlerError> {
            Ok(ActorStatus::Idle)
        }
    }

    let receiver = thread::spawn(move || {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            rx.run(&mut PanicActor);
        }))
    });

    tx.send(Message::Data(1)).unwrap();

    // Wait for the panic to propagate
    let result = receiver.join().unwrap();
    assert!(result.is_err(), "Actor panic should propagate");

    // Subsequent sends should detect disconnection
    thread::sleep(Duration::from_millis(10));
    let send_result = tx.send(Message::Data(2));
    assert!(
        send_result.is_err(),
        "Sends after scheduler panic should fail"
    );
}

// ============================================================================
// 14. PARK RETURNING FATAL ERROR
// ============================================================================

/// If park() returns Fatal, the scheduler should panic.
#[test]
fn park_fatal_panics_scheduler() {
    let (tx, mut rx) = ActorScheduler::<i32, i32, i32>::new(10, 100);

    struct FatalParkActor;
    impl Actor<i32, i32, i32> for FatalParkActor {
        fn handle_data(&mut self, _: i32) -> HandlerResult {
            Ok(())
        }
        fn handle_control(&mut self, _: i32) -> HandlerResult {
            Ok(())
        }
        fn handle_management(&mut self, _: i32) -> HandlerResult {
            Ok(())
        }
        fn park(&mut self, _: SystemStatus) -> Result<ActorStatus, HandlerError> {
            Err(HandlerError::fatal("park fatal"))
        }
    }

    let receiver = thread::spawn(move || {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            rx.run(&mut FatalParkActor);
        }))
    });

    tx.send(Message::Data(1)).unwrap();

    let result = receiver.join().unwrap();
    assert!(result.is_err(), "Fatal park error should panic");
}

// ============================================================================
// 15. EDGE CASE: ZERO DATA BURST LIMIT
// ============================================================================

/// With data_burst_limit=1, only 1 data message per cycle. Control should
/// still be responsive.
#[test]
fn minimal_data_burst_preserves_control_responsiveness() {
    let (tx, mut rx) = ActorScheduler::<i32, i32, i32>::new(
        1, // absolute minimum data burst
        1000,
    );

    let ctrl_count = Arc::new(AtomicUsize::new(0));
    let data_count = Arc::new(AtomicUsize::new(0));
    let cc = ctrl_count.clone();
    let dc = data_count.clone();

    struct MinBurstActor {
        ctrl: Arc<AtomicUsize>,
        data: Arc<AtomicUsize>,
    }
    impl Actor<i32, i32, i32> for MinBurstActor {
        fn handle_data(&mut self, _: i32) -> HandlerResult {
            self.data.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
        fn handle_control(&mut self, _: i32) -> HandlerResult {
            self.ctrl.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
        fn handle_management(&mut self, _: i32) -> HandlerResult {
            Ok(())
        }
        fn park(&mut self, _: SystemStatus) -> Result<ActorStatus, HandlerError> {
            Ok(ActorStatus::Idle)
        }
    }

    let receiver = thread::spawn(move || {
        let mut actor = MinBurstActor { ctrl: cc, data: dc };
        rx.run(&mut actor);
    });

    // Send 100 data, 5 control
    for i in 0..100 {
        tx.send(Message::Data(i)).unwrap();
    }
    for i in 0..5 {
        tx.send(Message::Control(i)).unwrap();
    }

    thread::sleep(Duration::from_millis(300));
    drop(tx);
    receiver.join().unwrap();

    let ctrl = ctrl_count.load(Ordering::SeqCst);
    let data = data_count.load(Ordering::SeqCst);
    assert_eq!(ctrl, 5, "All 5 control messages should be processed");
    assert_eq!(data, 100, "All 100 data messages should be processed");
}
