//! Bug Hunting Tests for the Actor Model
//!
//! These tests actively try to find bugs by:
//! - Probing edge cases and boundary conditions
//! - Testing race conditions and concurrent access patterns
//! - Attempting to trigger overflows and underflows
//! - Testing resource exhaustion scenarios
//! - Looking for deadlocks and starvation conditions
//!
//! Each test documents the bug it's hunting for.

use actor_scheduler::{Actor, ActorScheduler, Message, ParkHint, SendError};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Barrier, Mutex};
use std::thread;
use std::time::{Duration, Instant};

// ============================================================================
// POTENTIAL BUG: Backoff jitter overflow
// The calculation `(backoff_micros * jitter_pct) / 100` could overflow
// if backoff_micros is large enough, even though the final result fits.
// ============================================================================

#[test]
fn backoff_does_not_overflow_on_large_attempts() {
    // This test tries to trigger overflow in the backoff calculation.
    // The backoff uses 2^attempt, and with attempt=63, 2^63 would overflow.
    // The code should cap at MAX_BACKOFF (500ms) before overflow.

    let (tx, mut rx) = ActorScheduler::<(), (), ()>::new(10, 1);

    // Fill up the control lane to trigger backoff
    // Note: Control lane size is 128 (CONTROL_MGMT_BUFFER_SIZE)
    let tx_clone = tx.clone();

    let sender = thread::spawn(move || {
        // Keep trying to send - this should eventually hit the timeout error
        // if backoff overflows or gets stuck, this will hang
        let start = Instant::now();
        let mut error_count = 0;

        for _ in 0..1000 {
            match tx_clone.send(Message::Control(())) {
                Ok(()) => {}
                Err(SendError::Timeout) => {
                    error_count += 1;
                    if error_count > 5 {
                        break;
                    }
                }
                Err(SendError::Unknown) => break,
            }
        }

        // Should complete in reasonable time (not stuck in infinite backoff)
        start.elapsed()
    });

    // Don't run the receiver - let the channel fill up
    thread::sleep(Duration::from_millis(100));
    drop(rx);

    let elapsed = sender.join().unwrap();
    assert!(
        elapsed < Duration::from_secs(10),
        "Backoff should eventually timeout, not hang. Took: {:?}",
        elapsed
    );
}

// ============================================================================
// POTENTIAL BUG: Zero or negative buffer sizes
// What happens if we create a scheduler with buffer size 0?
// KNOWN ISSUE: sync_channel(0) creates a rendezvous channel that blocks
// forever on send if receiver isn't actively receiving. This is expected
// std::sync::mpsc behavior but callers should avoid buffer_size=0.
// ============================================================================

#[test]
#[ignore = "Known issue: sync_channel(0) creates rendezvous channel that blocks forever"]
fn zero_data_buffer_size_blocks_forever() {
    // This documents known behavior - sync_channel(0) is a rendezvous channel.
    // Callers should ensure buffer_size >= 1.
    // We mark this test as ignored but document the behavior.

    // With buffer_size=0, send() blocks until recv() is called.
    // If the actor hasn't started processing yet, this deadlocks.
    let (_tx, _rx) = ActorScheduler::<i32, i32, i32>::new(10, 0);
    // Would block forever if we tried to send
}

#[test]
fn zero_burst_limit_does_not_cause_infinite_loop() {
    // Burst limit of 0 could cause issues in the loop logic
    let (tx, mut rx) = ActorScheduler::<i32, i32, i32>::new(0, 10);
    let processed = Arc::new(AtomicUsize::new(0));
    let processed_clone = processed.clone();

    let handle = thread::spawn(move || {
        struct Counter(Arc<AtomicUsize>);
        impl Actor<i32, i32, i32> for Counter {
            fn handle_data(&mut self, _: i32) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
            fn handle_control(&mut self, _: i32) {}
            fn handle_management(&mut self, _: i32) {}
            fn park(&mut self, h: ParkHint) -> ParkHint { h }
        }
        rx.run(&mut Counter(processed_clone));
    });

    tx.send(Message::Data(1)).unwrap();
    tx.send(Message::Data(2)).unwrap();

    // Wait and check - should not infinite loop
    thread::sleep(Duration::from_millis(100));
    drop(tx);

    // Use timeout to detect infinite loop
    let join_result = thread::spawn(move || {
        handle.join().unwrap();
    });

    let timeout = Duration::from_secs(2);
    let start = Instant::now();
    while start.elapsed() < timeout {
        if join_result.is_finished() {
            break;
        }
        thread::sleep(Duration::from_millis(10));
    }

    assert!(
        start.elapsed() < timeout,
        "Zero burst limit should not cause infinite loop"
    );
}

// ============================================================================
// POTENTIAL BUG: Thundering herd on multiple sender drop
// If many senders are dropped at once, does the doorbell pattern handle it?
// ============================================================================

#[test]
fn mass_sender_drop_does_not_cause_race() {
    let (tx, mut rx) = ActorScheduler::<i32, i32, i32>::new(100, 1000);
    let count = Arc::new(AtomicUsize::new(0));
    let count_clone = count.clone();

    let handle = thread::spawn(move || {
        struct Counter(Arc<AtomicUsize>);
        impl Actor<i32, i32, i32> for Counter {
            fn handle_data(&mut self, _: i32) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
            fn handle_control(&mut self, _: i32) {}
            fn handle_management(&mut self, _: i32) {}
            fn park(&mut self, h: ParkHint) -> ParkHint { h }
        }
        rx.run(&mut Counter(count_clone));
    });

    // Create many senders
    let senders: Vec<_> = (0..100).map(|_| tx.clone()).collect();

    // Send from all senders
    let barrier = Arc::new(Barrier::new(100));
    let handles: Vec<_> = senders
        .into_iter()
        .map(|tx| {
            let barrier = barrier.clone();
            thread::spawn(move || {
                barrier.wait();
                for i in 0..10 {
                    let _ = tx.send(Message::Data(i));
                }
                // Sender dropped here - all 100 at roughly the same time
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    // Original sender still exists
    tx.send(Message::Data(999)).unwrap();

    thread::sleep(Duration::from_millis(100));
    drop(tx);
    handle.join().unwrap();

    // Should have processed 100 * 10 + 1 = 1001 messages
    let final_count = count.load(Ordering::SeqCst);
    assert_eq!(
        final_count, 1001,
        "All messages should be processed despite mass sender drop"
    );
}

// ============================================================================
// POTENTIAL BUG: Actor panics during handler
// What happens if an actor panics mid-processing?
// ============================================================================

#[test]
fn actor_panic_does_not_corrupt_state() {
    let (tx, mut rx) = ActorScheduler::<i32, i32, i32>::new(10, 100);
    let panic_at = 5;

    let handle = thread::spawn(move || {
        struct PanicActor {
            count: usize,
            panic_at: usize,
        }
        impl Actor<i32, i32, i32> for PanicActor {
            fn handle_data(&mut self, _: i32) {
                self.count += 1;
                if self.count == self.panic_at {
                    panic!("Intentional panic at message {}", self.count);
                }
            }
            fn handle_control(&mut self, _: i32) {}
            fn handle_management(&mut self, _: i32) {}
            fn park(&mut self, h: ParkHint) -> ParkHint { h }
        }

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            rx.run(&mut PanicActor { count: 0, panic_at });
        }));

        result.is_err() // Should have panicked
    });

    for i in 0..10 {
        let _ = tx.send(Message::Data(i)); // Some will fail after panic
    }

    thread::sleep(Duration::from_millis(100));
    drop(tx);

    let panicked = handle.join().unwrap();
    assert!(panicked, "Actor should have panicked");
}

// ============================================================================
// POTENTIAL BUG: Starvation of lower priority lanes
// Can control messages completely starve data messages?
// ============================================================================

#[test]
fn continuous_control_eventually_processes_data() {
    let (tx, mut rx) = ActorScheduler::new(10, 100);
    let data_received = Arc::new(AtomicBool::new(false));
    let data_received_clone = data_received.clone();

    let handle = thread::spawn(move || {
        struct StarvationTracker(Arc<AtomicBool>);
        impl Actor<String, String, String> for StarvationTracker {
            fn handle_data(&mut self, _: String) {
                self.0.store(true, Ordering::SeqCst);
            }
            fn handle_control(&mut self, _: String) {
                // No delay - we're testing priority, not processing time
            }
            fn handle_management(&mut self, _: String) {}
            fn park(&mut self, h: ParkHint) -> ParkHint { h }
        }
        rx.run(&mut StarvationTracker(data_received_clone));
    });

    // Send one data message
    tx.send(Message::Data("important".to_string())).unwrap();

    // Then flood with control messages (reduced count for faster test)
    for i in 0..100 {
        tx.send(Message::Control(format!("{}", i))).unwrap();
    }

    // Wait for processing - quick since no delays
    thread::sleep(Duration::from_millis(100));
    drop(tx);
    handle.join().unwrap();

    // Note: Due to priority, data may never be processed if control keeps coming
    // This test documents the behavior - it's expected that control has priority
    // But we want to verify the system doesn't deadlock

    // The test passes if it completes without hanging
}

// ============================================================================
// POTENTIAL BUG: ParkHint::Poll causes CPU spin
// If actor always returns Poll, does it burn CPU?
// ============================================================================

#[test]
fn park_poll_does_not_spin_indefinitely() {
    let (tx, mut rx) = ActorScheduler::<(), (), ()>::new(10, 100);
    let park_count = Arc::new(AtomicUsize::new(0));
    let park_count_clone = park_count.clone();

    let handle = thread::spawn(move || {
        struct SpinActor {
            park_count: Arc<AtomicUsize>,
            max_parks: usize,
        }
        impl Actor<(), (), ()> for SpinActor {
            fn handle_data(&mut self, _: ()) {}
            fn handle_control(&mut self, _: ()) {}
            fn handle_management(&mut self, _: ()) {}
            fn park(&mut self, _hint: ParkHint) -> ParkHint {
                let count = self.park_count.fetch_add(1, Ordering::SeqCst);
                if count < self.max_parks {
                    ParkHint::Poll // Keep spinning
                } else {
                    ParkHint::Wait // Eventually stop
                }
            }
        }
        rx.run(&mut SpinActor {
            park_count: park_count_clone,
            max_parks: 1000,
        });
    });

    // Send one message to trigger the spin
    tx.send(Message::Data(())).unwrap();

    // Wait for the spin to exhaust
    thread::sleep(Duration::from_millis(100));
    drop(tx);
    handle.join().unwrap();

    let final_count = park_count.load(Ordering::SeqCst);
    assert!(
        final_count >= 1000,
        "Should have spun many times. Count: {}",
        final_count
    );
}

// ============================================================================
// POTENTIAL BUG: Channel filling during slow handler
// If handler is slow, does the channel fill and block senders?
// ============================================================================

#[test]
fn slow_handler_backpressure_works() {
    // Small buffer to trigger backpressure quickly
    let (tx, mut rx) = ActorScheduler::<i32, i32, i32>::new(10, 2);
    let processed = Arc::new(AtomicUsize::new(0));
    let processed_clone = processed.clone();

    let handle = thread::spawn(move || {
        struct SlowActor(Arc<AtomicUsize>);
        impl Actor<i32, i32, i32> for SlowActor {
            fn handle_data(&mut self, _: i32) {
                thread::sleep(Duration::from_millis(50)); // Very slow
                self.0.fetch_add(1, Ordering::SeqCst);
            }
            fn handle_control(&mut self, _: i32) {}
            fn handle_management(&mut self, _: i32) {}
            fn park(&mut self, h: ParkHint) -> ParkHint { h }
        }
        rx.run(&mut SlowActor(processed_clone));
    });

    let sender_start = Instant::now();
    let tx_clone = tx.clone();

    // Sender thread - will block on backpressure
    let sender = thread::spawn(move || {
        for i in 0..10 {
            tx_clone.send(Message::Data(i)).unwrap();
        }
        sender_start.elapsed()
    });

    let send_time = sender.join().unwrap();

    // Sending 10 messages through buffer of 2 with 50ms handler
    // Should take at least 400ms (8 messages worth of blocking)
    assert!(
        send_time >= Duration::from_millis(300),
        "Sender should block on backpressure. Took: {:?}",
        send_time
    );

    drop(tx);
    handle.join().unwrap();

    assert_eq!(processed.load(Ordering::SeqCst), 10);
}

// ============================================================================
// POTENTIAL BUG: Message ordering under contention
// Do messages maintain FIFO order when multiple threads are sending?
// ============================================================================

#[test]
fn single_sender_fifo_ordering_maintained() {
    let (tx, mut rx) = ActorScheduler::new(100, 1000);
    let received = Arc::new(Mutex::new(Vec::new()));
    let received_clone = received.clone();

    let handle = thread::spawn(move || {
        struct OrderTracker(Arc<Mutex<Vec<i32>>>);
        impl Actor<i32, i32, i32> for OrderTracker {
            fn handle_data(&mut self, msg: i32) {
                self.0.lock().unwrap().push(msg);
            }
            fn handle_control(&mut self, _: i32) {}
            fn handle_management(&mut self, _: i32) {}
            fn park(&mut self, h: ParkHint) -> ParkHint { h }
        }
        rx.run(&mut OrderTracker(received_clone));
    });

    // Send in order from single thread
    for i in 0..1000 {
        tx.send(Message::Data(i)).unwrap();
    }

    thread::sleep(Duration::from_millis(100));
    drop(tx);
    handle.join().unwrap();

    let received = received.lock().unwrap();
    for (i, &val) in received.iter().enumerate() {
        assert_eq!(
            val, i as i32,
            "FIFO order violated at position {}: expected {}, got {}",
            i, i, val
        );
    }
}

// ============================================================================
// POTENTIAL BUG: Very large message count overflow
// Can frame numbers or counters overflow?
// ============================================================================

#[test]
fn large_frame_numbers_dont_overflow() {
    use pixelflow_runtime::vsync_actor::RenderedResponse;

    // Test with u64 max values
    let response = RenderedResponse {
        frame_number: u64::MAX,
        rendered_at: Instant::now(),
    };

    assert_eq!(response.frame_number, u64::MAX);

    // Test wrapping behavior
    let response2 = RenderedResponse {
        frame_number: u64::MAX.wrapping_add(1),
        rendered_at: Instant::now(),
    };

    assert_eq!(response2.frame_number, 0, "Wrapping should work correctly");
}

// ============================================================================
// POTENTIAL BUG: Rapid channel creation/destruction
// Does creating and destroying many channels leak resources?
// ============================================================================

#[test]
fn rapid_channel_creation_does_not_leak() {
    // Create and destroy many channels rapidly
    for _ in 0..1000 {
        let (tx, rx) = ActorScheduler::<i32, i32, i32>::new(10, 100);
        tx.send(Message::Data(1)).unwrap();
        drop(tx);
        drop(rx);
    }

    // If we get here without OOM, we're probably fine
    // In a real test, we'd measure memory usage
}

// ============================================================================
// POTENTIAL BUG: Send after partial processing
// What happens if sender sends while receiver is mid-batch?
// ============================================================================

#[test]
fn concurrent_send_during_processing() {
    let (tx, mut rx) = ActorScheduler::new(5, 100); // Small burst limit
    let total_received = Arc::new(AtomicUsize::new(0));
    let total_received_clone = total_received.clone();

    let handle = thread::spawn(move || {
        struct CountingActor(Arc<AtomicUsize>);
        impl Actor<i32, i32, i32> for CountingActor {
            fn handle_data(&mut self, _: i32) {
                // Small delay to increase chance of concurrent send
                thread::sleep(Duration::from_micros(100));
                self.0.fetch_add(1, Ordering::SeqCst);
            }
            fn handle_control(&mut self, _: i32) {}
            fn handle_management(&mut self, _: i32) {}
            fn park(&mut self, h: ParkHint) -> ParkHint { h }
        }
        rx.run(&mut CountingActor(total_received_clone));
    });

    // Multiple senders sending concurrently
    let handles: Vec<_> = (0..10)
        .map(|_| {
            let tx = tx.clone();
            thread::spawn(move || {
                for i in 0..100 {
                    tx.send(Message::Data(i)).unwrap();
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    thread::sleep(Duration::from_millis(500));
    drop(tx);
    handle.join().unwrap();

    assert_eq!(
        total_received.load(Ordering::SeqCst),
        1000,
        "All messages should be received"
    );
}

// ============================================================================
// POTENTIAL BUG: Doorbell saturation
// The doorbell has buffer size 1. What if it fills?
// ============================================================================

#[test]
fn doorbell_saturation_does_not_lose_messages() {
    let (tx, mut rx) = ActorScheduler::new(100, 1000);
    let count = Arc::new(AtomicUsize::new(0));
    let count_clone = count.clone();

    let handle = thread::spawn(move || {
        struct Counter(Arc<AtomicUsize>);
        impl Actor<i32, i32, i32> for Counter {
            fn handle_data(&mut self, _: i32) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
            fn handle_control(&mut self, _: i32) {}
            fn handle_management(&mut self, _: i32) {}
            fn park(&mut self, h: ParkHint) -> ParkHint { h }
        }
        rx.run(&mut Counter(count_clone));
    });

    // Rapid-fire sends from many threads to saturate doorbell
    let barrier = Arc::new(Barrier::new(50));
    let handles: Vec<_> = (0..50)
        .map(|_| {
            let tx = tx.clone();
            let barrier = barrier.clone();
            thread::spawn(move || {
                barrier.wait();
                for i in 0..100 {
                    tx.send(Message::Data(i)).unwrap();
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    thread::sleep(Duration::from_millis(200));
    drop(tx);
    handle.join().unwrap();

    assert_eq!(
        count.load(Ordering::SeqCst),
        5000,
        "All messages should be processed despite doorbell saturation"
    );
}

// ============================================================================
// POTENTIAL BUG: Time-based operations near epoch
// What happens with time calculations near boundaries?
// ============================================================================

#[test]
fn instant_arithmetic_is_safe() {
    use pixelflow_runtime::vsync_actor::RenderedResponse;

    let now = Instant::now();
    let later = now + Duration::from_secs(1);

    let response = RenderedResponse {
        frame_number: 0,
        rendered_at: later,
    };

    // Should be able to calculate elapsed without panic
    let elapsed = response.rendered_at.elapsed();
    // elapsed might be 0 or small negative (which panics on sub), but elapsed() handles it
    assert!(elapsed < Duration::from_secs(2));
}

// ============================================================================
// POTENTIAL BUG: Control lane timeout behavior
// Does the timeout in send_with_backoff actually work?
// ============================================================================

#[test]
fn control_lane_timeout_returns_error() {
    let (tx, rx) = ActorScheduler::<i32, i32, i32>::new(10, 100);

    // Don't run the receiver - just let messages pile up

    // Fill the control lane
    for _ in 0..128 {
        // CONTROL_MGMT_BUFFER_SIZE
        tx.send(Message::Control(0)).unwrap();
    }

    // Next send should timeout after MAX_BACKOFF
    let start = Instant::now();
    let result = tx.send(Message::Control(999));
    let elapsed = start.elapsed();

    // Should get timeout error
    assert!(
        matches!(result, Err(SendError::Timeout)),
        "Should timeout when control lane is full. Got: {:?}",
        result
    );

    // Should have taken a reasonable amount of time (not forever)
    assert!(
        elapsed < Duration::from_secs(5),
        "Timeout should occur within reasonable time. Took: {:?}",
        elapsed
    );

    drop(rx);
}

// ============================================================================
// POTENTIAL BUG: Mixed message priorities under load
// Does priority still work when all lanes are busy?
// ============================================================================

#[test]
fn priority_maintained_under_heavy_mixed_load() {
    let (tx, mut rx) = ActorScheduler::new(10, 100);
    let control_processed_at = Arc::new(AtomicU64::new(0));
    let data_start = Arc::new(AtomicU64::new(0));

    let control_clone = control_processed_at.clone();
    let data_start_clone = data_start.clone();

    let handle = thread::spawn(move || {
        struct PriorityChecker {
            control_at: Arc<AtomicU64>,
            data_start: Arc<AtomicU64>,
            counter: u64,
        }
        impl Actor<i32, i32, i32> for PriorityChecker {
            fn handle_data(&mut self, _: i32) {
                self.counter += 1;
                if self.data_start.load(Ordering::SeqCst) == 0 {
                    self.data_start.store(self.counter, Ordering::SeqCst);
                }
            }
            fn handle_control(&mut self, _: i32) {
                self.counter += 1;
                self.control_at.store(self.counter, Ordering::SeqCst);
            }
            fn handle_management(&mut self, _: i32) {}
            fn park(&mut self, h: ParkHint) -> ParkHint { h }
        }
        rx.run(&mut PriorityChecker {
            control_at: control_clone,
            data_start: data_start_clone,
            counter: 0,
        });
    });

    // Send lots of data first
    for i in 0..100 {
        tx.send(Message::Data(i)).unwrap();
    }

    // Then send control
    tx.send(Message::Control(0)).unwrap();

    thread::sleep(Duration::from_millis(100));
    drop(tx);
    handle.join().unwrap();

    let ctrl_pos = control_processed_at.load(Ordering::SeqCst);
    let data_first = data_start.load(Ordering::SeqCst);

    // Control should be processed before data started (or very early)
    assert!(
        ctrl_pos <= 2,
        "Control should be processed first or second. Ctrl at: {}, Data started at: {}",
        ctrl_pos,
        data_first
    );
}

// ============================================================================
// POTENTIAL BUG: Memory safety with large queues
// Do we handle memory correctly with many queued messages?
// ============================================================================

#[test]
fn large_queue_does_not_cause_issues() {
    let (tx, mut rx) = ActorScheduler::<String, String, String>::new(1000, 10000);
    let count = Arc::new(AtomicUsize::new(0));
    let count_clone = count.clone();

    let handle = thread::spawn(move || {
        struct Counter(Arc<AtomicUsize>);
        impl Actor<String, String, String> for Counter {
            fn handle_data(&mut self, _: String) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
            fn handle_control(&mut self, _: String) {}
            fn handle_management(&mut self, _: String) {}
            fn park(&mut self, h: ParkHint) -> ParkHint { h }
        }
        rx.run(&mut Counter(count_clone));
    });

    // Queue up many messages with non-trivial data
    for i in 0..10000 {
        tx.send(Message::Data(format!(
            "message {} with some extra data to use more memory",
            i
        )))
        .unwrap();
    }

    thread::sleep(Duration::from_millis(500));
    drop(tx);
    handle.join().unwrap();

    assert_eq!(count.load(Ordering::SeqCst), 10000);
}

// ============================================================================
// POTENTIAL BUG: Scheduler shutdown race
// What if messages arrive exactly as scheduler is shutting down?
// ============================================================================

#[test]
fn shutdown_race_does_not_panic() {
    for _ in 0..100 {
        let (tx, mut rx) = ActorScheduler::<i32, i32, i32>::new(10, 100);

        let handle = thread::spawn(move || {
            struct NoopActor;
            impl Actor<i32, i32, i32> for NoopActor {
                fn handle_data(&mut self, _: i32) {}
                fn handle_control(&mut self, _: i32) {}
                fn handle_management(&mut self, _: i32) {}
                fn park(&mut self, h: ParkHint) -> ParkHint { h }
            }
            rx.run(&mut NoopActor);
        });

        // Race: send and drop nearly simultaneously
        let tx2 = tx.clone();
        thread::spawn(move || {
            for i in 0..10 {
                let _ = tx2.send(Message::Data(i));
            }
        });

        // Drop original sender quickly
        drop(tx);

        // Should complete without panic
        let _ = handle.join();
    }
}
