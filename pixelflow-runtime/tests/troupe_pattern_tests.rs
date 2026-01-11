//! Tests for the Troupe Pattern
//!
//! The troupe system provides lifecycle management for groups of actors.
//! These tests verify:
//! - Directory pattern (actors accessing each other's handles)
//! - Two-phase initialization (new() -> exposed() -> play())
//! - Exposed handles lifetime
//! - Cross-actor messaging
//! - Thread spawning and cleanup
//! - Error handling in actor threads
//!
//! Note: We test the patterns manually here since the troupe! macro
//! generates code that's hard to test in isolation.

use actor_scheduler::{
    Actor, ActorHandle, ActorScheduler, ActorTypes, Message, ActorStatus, TroupeActor,
};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Barrier, Mutex};
use std::thread;
use std::time::Duration;

// ============================================================================
// Test Message Types
// ============================================================================

#[derive(Debug)]
struct AlphaData(String);

#[derive(Debug, Default)]
enum AlphaControl {
    Ping,
    #[default]
    Shutdown,
}

#[derive(Debug)]
struct AlphaManagement;

#[derive(Debug)]
struct BetaData(i32);

#[derive(Debug, Default)]
enum BetaControl {
    Pong,
    #[default]
    Shutdown,
}

#[derive(Debug)]
struct BetaManagement;

// ============================================================================
// Manual Directory (what troupe! generates)
// ============================================================================

struct TestDirectory {
    alpha: ActorHandle<AlphaData, AlphaControl, AlphaManagement>,
    beta: ActorHandle<BetaData, BetaControl, BetaManagement>,
}

// ============================================================================
// Test Actors
// ============================================================================

struct AlphaActor<'a> {
    dir: &'a TestDirectory,
    log: Arc<Mutex<Vec<String>>>,
}

impl ActorTypes for AlphaActor<'_> {
    type Data = AlphaData;
    type Control = AlphaControl;
    type Management = AlphaManagement;
}

impl<'a, Dir: 'a> TroupeActor<'a, Dir> for AlphaActor<'a> {
    fn new(_dir: &'a Dir) -> Self {
        panic!("use new_with_log instead")
    }
}

impl Actor<AlphaData, AlphaControl, AlphaManagement> for AlphaActor<'_> {
    fn handle_data(&mut self, msg: AlphaData) -> Result<(), actor_scheduler::ActorError> {

        self.log
            .lock()
            .unwrap()
            .push(format!("Alpha:Data:{}", msg.0));
        Ok(())
    }

    fn handle_control(&mut self, cmd: AlphaControl) -> Result<(), actor_scheduler::ActorError> {

        match cmd {
            AlphaControl::Ping => {
                self.log.lock().unwrap().push("Alpha:Ping".to_string());
                // Send pong to beta
                let _ = self.dir.beta.send(Message::Control(BetaControl::Pong));
            }
            AlphaControl::Shutdown => {
                self.log.lock().unwrap().push("Alpha:Shutdown".to_string());
            }
        }
        Ok(())
    }

    fn handle_management(&mut self, _: AlphaManagement) -> Result<(), actor_scheduler::ActorError> { Ok(()) }

    fn park(&mut self, hint: ActorStatus) -> ActorStatus {
        hint
    }
    Ok(())
}

struct BetaActor<'a> {
    dir: &'a TestDirectory,
    log: Arc<Mutex<Vec<String>>>,
}

impl ActorTypes for BetaActor<'_> {
    type Data = BetaData;
    type Control = BetaControl;
    type Management = BetaManagement;
}

impl<'a, Dir: 'a> TroupeActor<'a, Dir> for BetaActor<'a> {
    fn new(_dir: &'a Dir) -> Self {
        panic!("use new_with_log instead")
    }
}

impl Actor<BetaData, BetaControl, BetaManagement> for BetaActor<'_> {
    fn handle_data(&mut self, msg: BetaData) -> Result<(), actor_scheduler::ActorError> {

        self.log
            .lock()
            .unwrap()
            .push(format!("Beta:Data:{}", msg.0));
        Ok(())
    }

    fn handle_control(&mut self, cmd: BetaControl) -> Result<(), actor_scheduler::ActorError> {

        match cmd {
            BetaControl::Pong => {
                self.log.lock().unwrap().push("Beta:Pong".to_string());
                // Send back to alpha
                let _ = self
                    .dir
                    .alpha
                    .send(Message::Data(AlphaData("pong-response".to_string())));
            }
            BetaControl::Shutdown => {
                self.log.lock().unwrap().push("Beta:Shutdown".to_string());
            }
        }
        Ok(())
    }

    fn handle_management(&mut self, _: BetaManagement) -> Result<(), actor_scheduler::ActorError> { Ok(()) }

    fn park(&mut self, hint: ActorStatus) -> ActorStatus {
        hint
    }
    Ok(())
}

// ============================================================================
// Directory Pattern Tests
// ============================================================================

/// Tests cross-actor messaging through a shared directory.
///
/// This test demonstrates a key design consideration: when actors hold handles
/// to each other via a shared directory, clean shutdown requires careful ordering.
/// The scheduler's run() only exits when all senders are dropped, but if actors
/// hold circular handle references, a deadlock occurs.
///
/// Solution: Use timeout-based verification and don't wait for thread join.
/// Real applications should use a dedicated shutdown coordinator or have actors
/// drop their directory references when they receive shutdown.
#[test]
fn directory_allows_cross_actor_messaging() {
    let log = Arc::new(Mutex::new(Vec::new()));

    // Phase 1: Create handles and schedulers
    let (alpha_h, mut alpha_s) =
        ActorScheduler::<AlphaData, AlphaControl, AlphaManagement>::new(100, 1024);
    let (beta_h, mut beta_s) =
        ActorScheduler::<BetaData, BetaControl, BetaManagement>::new(100, 1024);

    // Phase 2: Build directory
    let dir = Arc::new(TestDirectory {
        alpha: alpha_h.clone(),
        beta: beta_h.clone(),
    });

    // Phase 3: Spawn actors with directory reference
    let log_alpha = log.clone();
    let dir_alpha = dir.clone();
    thread::spawn(move || {
        // Safety: dir lives in Arc, accessible throughout thread lifetime
        let dir_ref: &TestDirectory =
            unsafe { &*(Arc::as_ptr(&dir_alpha) as *const TestDirectory) };
        let mut actor = AlphaActor {
            dir: dir_ref,
            log: log_alpha,
        };
        alpha_s.run(&mut actor);
    });

    let log_beta = log.clone();
    let dir_beta = dir.clone();
    thread::spawn(move || {
        let dir_ref: &TestDirectory = unsafe { &*(Arc::as_ptr(&dir_beta) as *const TestDirectory) };
        let mut actor = BetaActor {
            dir: dir_ref,
            log: log_beta,
        };
        beta_s.run(&mut actor);
    });

    // Phase 4: Test cross-actor messaging
    // Send Ping to Alpha -> Alpha sends Pong to Beta -> Beta sends Data to Alpha
    alpha_h.send(Message::Control(AlphaControl::Ping)).unwrap();

    // Wait for message chain to complete (with timeout)
    let start = std::time::Instant::now();
    loop {
        thread::sleep(Duration::from_millis(10));
        let log_snapshot = log.lock().unwrap();
        let has_ping = log_snapshot.contains(&"Alpha:Ping".to_string());
        let has_pong = log_snapshot.contains(&"Beta:Pong".to_string());
        let has_response = log_snapshot.contains(&"Alpha:Data:pong-response".to_string());

        if has_ping && has_pong && has_response {
            break; // Success!
        }

        if start.elapsed() > Duration::from_secs(2) {
            panic!("Cross-actor messaging timed out. Log: {:?}", *log_snapshot);
        }
    }

    // Note: We don't join threads here because of circular handle references.
    // The actors hold handles to each other via the directory, preventing clean shutdown.
    // This is acceptable for testing - threads will be cleaned up when the test exits.
}

// ============================================================================
// Two-Phase Initialization Tests
// ============================================================================

/// Simulates the Troupe struct that troupe! would generate
struct TestTroupe {
    directory: TestDirectory,
    alpha_scheduler: ActorScheduler<AlphaData, AlphaControl, AlphaManagement>,
    beta_scheduler: ActorScheduler<BetaData, BetaControl, BetaManagement>,
}

struct TestExposedHandles {
    alpha: ActorHandle<AlphaData, AlphaControl, AlphaManagement>,
}

impl TestTroupe {
    fn new() -> Self {
        let (alpha_h, alpha_s) =
            ActorScheduler::<AlphaData, AlphaControl, AlphaManagement>::new(100, 1024);
        let (beta_h, beta_s) =
            ActorScheduler::<BetaData, BetaControl, BetaManagement>::new(100, 1024);

        Self {
            directory: TestDirectory {
                alpha: alpha_h,
                beta: beta_h,
            },
            alpha_scheduler: alpha_s,
            beta_scheduler: beta_s,
        }
    }

    fn exposed(&self) -> TestExposedHandles {
        TestExposedHandles {
            alpha: self.directory.alpha.clone(),
        }
    }
}

#[test]
fn two_phase_initialization_queues_messages_before_play() {
    // Phase 1: Create troupe
    let troupe = TestTroupe::new();

    // Phase 2: Get exposed handles
    let exposed = troupe.exposed();

    // Phase 3: Send messages BEFORE play() - they should queue
    exposed
        .alpha
        .send(Message::Data(AlphaData("early-bird".to_string())))
        .unwrap();
    exposed
        .alpha
        .send(Message::Data(AlphaData("gets-the-worm".to_string())))
        .unwrap();

    // Verify messages are queued (not processed yet - no threads running)
    // We can't directly check, but the sends should succeed

    // Drop troupe to clean up
    drop(troupe);
    drop(exposed);
}

#[test]
fn exposed_handles_can_outlive_troupe_new() {
    let exposed = {
        let troupe = TestTroupe::new();
        troupe.exposed() // Handle escapes
    };
    // Troupe dropped here

    // Handles still valid for sending
    // (though messages will never be processed since scheduler dropped)
    let result = exposed
        .alpha
        .send(Message::Data(AlphaData("orphan".to_string())));

    // Send might succeed (channel still open) or fail (receiver dropped)
    // Either is acceptable - the important thing is no panic
    drop(result);
}

// ============================================================================
// Thread Lifecycle Tests
// ============================================================================

#[test]
fn all_actor_threads_exit_on_channel_close() {
    let (alpha_h, mut alpha_s) =
        ActorScheduler::<AlphaData, AlphaControl, AlphaManagement>::new(100, 1024);
    let (beta_h, mut beta_s) =
        ActorScheduler::<BetaData, BetaControl, BetaManagement>::new(100, 1024);

    let alpha_exited = Arc::new(AtomicBool::new(false));
    let beta_exited = Arc::new(AtomicBool::new(false));

    let alpha_exit = alpha_exited.clone();
    let beta_exit = beta_exited.clone();

    let alpha_thread = thread::spawn(move || {
        struct NoopActor;
        impl Actor<AlphaData, AlphaControl, AlphaManagement> for NoopActor {
            fn handle_data(&mut self, _: AlphaData) -> Result<(), actor_scheduler::ActorError> { Ok(()) }
            fn handle_control(&mut self, _: AlphaControl) -> Result<(), actor_scheduler::ActorError> { Ok(()) }
            fn handle_management(&mut self, _: AlphaManagement) -> Result<(), actor_scheduler::ActorError> { Ok(()) }
            fn park(&mut self, h: ActorStatus) -> ActorStatus {
                h
            }
        }
        alpha_s.run(&mut NoopActor);
        alpha_exit.store(true, Ordering::SeqCst);
    });

    let beta_thread = thread::spawn(move || {
        struct NoopActor;
        impl Actor<BetaData, BetaControl, BetaManagement> for NoopActor {
            fn handle_data(&mut self, _: BetaData) -> Result<(), actor_scheduler::ActorError> { Ok(()) }
            fn handle_control(&mut self, _: BetaControl) -> Result<(), actor_scheduler::ActorError> { Ok(()) }
            fn handle_management(&mut self, _: BetaManagement) -> Result<(), actor_scheduler::ActorError> { Ok(()) }
            fn park(&mut self, h: ActorStatus) -> ActorStatus {
                h
            }
        }
        beta_s.run(&mut NoopActor);
        beta_exit.store(true, Ordering::SeqCst);
    });

    // Verify threads are running
    thread::sleep(Duration::from_millis(20));
    assert!(!alpha_exited.load(Ordering::SeqCst));
    assert!(!beta_exited.load(Ordering::SeqCst));

    // Drop handles - should trigger exit
    drop(alpha_h);
    drop(beta_h);

    // Wait for threads
    alpha_thread.join().unwrap();
    beta_thread.join().unwrap();

    assert!(alpha_exited.load(Ordering::SeqCst));
    assert!(beta_exited.load(Ordering::SeqCst));
}

#[test]
fn actor_thread_panic_isolated() {
    let (alpha_h, mut alpha_s) =
        ActorScheduler::<AlphaData, AlphaControl, AlphaManagement>::new(100, 1024);
    let (beta_h, mut beta_s) =
        ActorScheduler::<BetaData, BetaControl, BetaManagement>::new(100, 1024);

    let beta_count = Arc::new(AtomicUsize::new(0));
    let beta_count_clone = beta_count.clone();

    // Alpha will panic
    let alpha_thread = thread::spawn(move || {
        struct PanicActor;
        impl Actor<AlphaData, AlphaControl, AlphaManagement> for PanicActor {
            fn handle_data(&mut self, _: AlphaData) -> Result<(), actor_scheduler::ActorError> {

                panic!("Alpha panics!");
                Ok(())
            }
            fn handle_control(&mut self, _: AlphaControl) -> Result<(), actor_scheduler::ActorError> { Ok(()) }
            fn handle_management(&mut self, _: AlphaManagement) -> Result<(), actor_scheduler::ActorError> { Ok(()) }
            fn park(&mut self, h: ActorStatus) -> ActorStatus {
                h
            }
        }
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            alpha_s.run(&mut PanicActor);
        }))
    });

    // Beta should continue working
    let beta_thread = thread::spawn(move || {
        struct CountActor(Arc<AtomicUsize>);
        impl Actor<BetaData, BetaControl, BetaManagement> for CountActor {
            fn handle_data(&mut self, _: BetaData) -> Result<(), actor_scheduler::ActorError> {

                self.0.fetch_add(1, Ordering::SeqCst);
                Ok(())
            }
            fn handle_control(&mut self, _: BetaControl) -> Result<(), actor_scheduler::ActorError> { Ok(()) }
            fn handle_management(&mut self, _: BetaManagement) -> Result<(), actor_scheduler::ActorError> { Ok(()) }
            fn park(&mut self, h: ActorStatus) -> ActorStatus {
                h
            }
        }
        beta_s.run(&mut CountActor(beta_count_clone));
    });

    // Trigger alpha panic
    alpha_h
        .send(Message::Data(AlphaData("boom".to_string())))
        .unwrap();

    thread::sleep(Duration::from_millis(50));

    // Beta should still work
    beta_h.send(Message::Data(BetaData(1))).unwrap();
    beta_h.send(Message::Data(BetaData(2))).unwrap();

    thread::sleep(Duration::from_millis(50));

    drop(alpha_h);
    drop(beta_h);

    let alpha_result = alpha_thread.join();
    beta_thread.join().unwrap();

    // Alpha should have panicked
    assert!(alpha_result.is_ok()); // Thread itself didn't panic (we caught it)
    assert!(alpha_result.unwrap().is_err()); // But the closure panicked

    // Beta should have processed messages
    assert_eq!(beta_count.load(Ordering::SeqCst), 2);
}

// ============================================================================
// Circular Messaging Tests
// ============================================================================

/// Tests that circular messaging between actors does not deadlock.
///
/// This verifies that the actor scheduler handles ping-pong message patterns
/// without blocking. Note that clean shutdown is not tested here due to
/// circular handle references (see directory_allows_cross_actor_messaging).
#[test]
fn circular_messaging_does_not_deadlock() {
    let (alpha_h, mut alpha_s) =
        ActorScheduler::<AlphaData, AlphaControl, AlphaManagement>::new(100, 1000);
    let (beta_h, mut beta_s) =
        ActorScheduler::<BetaData, BetaControl, BetaManagement>::new(100, 1000);

    let ping_count = Arc::new(AtomicUsize::new(0));
    let pong_count = Arc::new(AtomicUsize::new(0));

    let dir = Arc::new((alpha_h.clone(), beta_h.clone()));

    let ping_clone = ping_count.clone();
    let dir_alpha = dir.clone();
    thread::spawn(move || {
        struct PingActor {
            beta_h: ActorHandle<BetaData, BetaControl, BetaManagement>,
            count: Arc<AtomicUsize>,
            max: usize,
        }
        impl Actor<AlphaData, AlphaControl, AlphaManagement> for PingActor {
            fn handle_data(&mut self, _: AlphaData) -> Result<(), actor_scheduler::ActorError> { Ok(()) }
            fn handle_control(&mut self, cmd: AlphaControl) -> Result<(), actor_scheduler::ActorError> {

                if matches!(cmd, AlphaControl::Ping) {
                    let c = self.count.fetch_add(1, Ordering::SeqCst);
                    if c < self.max {
                        let _ = self.beta_h.send(Message::Control(BetaControl::Pong));
                    }
                }
            }
            fn handle_management(&mut self, _: AlphaManagement) -> Result<(), actor_scheduler::ActorError> { Ok(()) }
            fn park(&mut self, h: ActorStatus) -> ActorStatus {
                h
            }
        }
        alpha_s.run(&mut PingActor {
            beta_h: dir_alpha.1.clone(),
            count: ping_clone,
            max: 100,
        });
    });

    let pong_clone = pong_count.clone();
    let dir_beta = dir.clone();
    thread::spawn(move || {
        struct PongActor {
            alpha_h: ActorHandle<AlphaData, AlphaControl, AlphaManagement>,
            count: Arc<AtomicUsize>,
            max: usize,
        }
        impl Actor<BetaData, BetaControl, BetaManagement> for PongActor {
            fn handle_data(&mut self, _: BetaData) -> Result<(), actor_scheduler::ActorError> { Ok(()) }
            fn handle_control(&mut self, cmd: BetaControl) -> Result<(), actor_scheduler::ActorError> {

                if matches!(cmd, BetaControl::Pong) {
                    let c = self.count.fetch_add(1, Ordering::SeqCst);
                    if c < self.max {
                        let _ = self.alpha_h.send(Message::Control(AlphaControl::Ping));
                    }
                }
            }
            fn handle_management(&mut self, _: BetaManagement) -> Result<(), actor_scheduler::ActorError> { Ok(()) }
            fn park(&mut self, h: ActorStatus) -> ActorStatus {
                h
            }
        }
        beta_s.run(&mut PongActor {
            alpha_h: dir_beta.0.clone(),
            count: pong_clone,
            max: 100,
        });
    });

    // Start the ping-pong
    alpha_h.send(Message::Control(AlphaControl::Ping)).unwrap();

    // Wait for it to complete (with timeout to detect deadlock)
    let start = std::time::Instant::now();
    while ping_count.load(Ordering::SeqCst) < 100 || pong_count.load(Ordering::SeqCst) < 100 {
        if start.elapsed() > Duration::from_secs(5) {
            panic!(
                "Deadlock detected in circular messaging! ping={}, pong={}",
                ping_count.load(Ordering::SeqCst),
                pong_count.load(Ordering::SeqCst)
            );
        }
        thread::sleep(Duration::from_millis(10));
    }

    // Success! Don't try to join threads due to circular handle references.
    // Note: counts may exceed 100 slightly due to race conditions in the ping-pong.
    // One actor may increment before the other checks its own count.
    assert!(ping_count.load(Ordering::SeqCst) >= 100);
    assert!(pong_count.load(Ordering::SeqCst) >= 100);
}

// ============================================================================
// Handle Cloning Tests
// ============================================================================

#[test]
fn cloned_directory_handles_work_independently() {
    let (alpha_h, mut alpha_s) =
        ActorScheduler::<AlphaData, AlphaControl, AlphaManagement>::new(100, 1024);

    let count = Arc::new(AtomicUsize::new(0));
    let count_clone = count.clone();

    let handle = thread::spawn(move || {
        struct CountActor(Arc<AtomicUsize>);
        impl Actor<AlphaData, AlphaControl, AlphaManagement> for CountActor {
            fn handle_data(&mut self, _: AlphaData) -> Result<(), actor_scheduler::ActorError> {

                self.0.fetch_add(1, Ordering::SeqCst);
                Ok(())
            }
            fn handle_control(&mut self, _: AlphaControl) -> Result<(), actor_scheduler::ActorError> { Ok(()) }
            fn handle_management(&mut self, _: AlphaManagement) -> Result<(), actor_scheduler::ActorError> { Ok(()) }
            fn park(&mut self, h: ActorStatus) -> ActorStatus {
                h
            }
        }
        alpha_s.run(&mut CountActor(count_clone));
    });

    // Clone handle multiple times (simulating multiple actors holding it)
    let h1 = alpha_h.clone();
    let h2 = alpha_h.clone();
    let h3 = alpha_h.clone();

    // Send from all
    h1.send(Message::Data(AlphaData("1".to_string()))).unwrap();
    h2.send(Message::Data(AlphaData("2".to_string()))).unwrap();
    h3.send(Message::Data(AlphaData("3".to_string()))).unwrap();

    // Drop some clones
    drop(h1);
    drop(h2);

    // Original still works
    alpha_h
        .send(Message::Data(AlphaData("4".to_string())))
        .unwrap();

    thread::sleep(Duration::from_millis(50));

    // Drop all
    drop(h3);
    drop(alpha_h);
    handle.join().unwrap();

    assert_eq!(count.load(Ordering::SeqCst), 4);
}

// ============================================================================
// Barrier Pattern Tests (coordinated startup)
// ============================================================================

#[test]
fn actors_can_coordinate_startup_with_barrier() {
    let (alpha_h, mut alpha_s) = ActorScheduler::<(), (), ()>::new(100, 1024);
    let (beta_h, mut beta_s) = ActorScheduler::<(), (), ()>::new(100, 1024);

    let barrier = Arc::new(Barrier::new(3)); // 2 actors + 1 main

    let alpha_started = Arc::new(AtomicBool::new(false));
    let beta_started = Arc::new(AtomicBool::new(false));

    let barrier_a = barrier.clone();
    let started_a = alpha_started.clone();
    let alpha_thread = thread::spawn(move || {
        // Wait for everyone before processing
        barrier_a.wait();
        started_a.store(true, Ordering::SeqCst);

        struct NoopActor;
        impl Actor<(), (), ()> for NoopActor {
            fn handle_data(&mut self, _: ()) -> Result<(), actor_scheduler::ActorError> { Ok(()) }
            fn handle_control(&mut self, _: ()) -> Result<(), actor_scheduler::ActorError> { Ok(()) }
            fn handle_management(&mut self, _: ()) -> Result<(), actor_scheduler::ActorError> { Ok(()) }
            fn park(&mut self, h: ActorStatus) -> ActorStatus {
                h
            }
        }
        alpha_s.run(&mut NoopActor);
    });

    let barrier_b = barrier.clone();
    let started_b = beta_started.clone();
    let beta_thread = thread::spawn(move || {
        barrier_b.wait();
        started_b.store(true, Ordering::SeqCst);

        struct NoopActor;
        impl Actor<(), (), ()> for NoopActor {
            fn handle_data(&mut self, _: ()) -> Result<(), actor_scheduler::ActorError> { Ok(()) }
            fn handle_control(&mut self, _: ()) -> Result<(), actor_scheduler::ActorError> { Ok(()) }
            fn handle_management(&mut self, _: ()) -> Result<(), actor_scheduler::ActorError> { Ok(()) }
            fn park(&mut self, h: ActorStatus) -> ActorStatus {
                h
            }
        }
        beta_s.run(&mut NoopActor);
    });

    // Neither should have started yet
    thread::sleep(Duration::from_millis(10));
    assert!(!alpha_started.load(Ordering::SeqCst));
    assert!(!beta_started.load(Ordering::SeqCst));

    // Release the barrier
    barrier.wait();

    // Now both should start
    thread::sleep(Duration::from_millis(50));
    assert!(alpha_started.load(Ordering::SeqCst));
    assert!(beta_started.load(Ordering::SeqCst));

    drop(alpha_h);
    drop(beta_h);

    alpha_thread.join().unwrap();
    beta_thread.join().unwrap();
}

// ============================================================================
// Message::Shutdown Tests
// ============================================================================

#[test]
fn shutdown_message_causes_actor_exit() {
    let (alpha_h, mut alpha_s) =
        ActorScheduler::<AlphaData, AlphaControl, AlphaManagement>::new(100, 1024);

    let exited = Arc::new(AtomicBool::new(false));
    let exited_clone = exited.clone();

    let handle = thread::spawn(move || {
        struct NoopActor;
        impl Actor<AlphaData, AlphaControl, AlphaManagement> for NoopActor {
            fn handle_data(&mut self, _: AlphaData) -> Result<(), actor_scheduler::ActorError> { Ok(()) }
            fn handle_control(&mut self, _: AlphaControl) -> Result<(), actor_scheduler::ActorError> { Ok(()) }
            fn handle_management(&mut self, _: AlphaManagement) -> Result<(), actor_scheduler::ActorError> { Ok(()) }
            fn park(&mut self, h: ActorStatus) -> ActorStatus {
                h
            }
        }
        alpha_s.run(&mut NoopActor);
        exited_clone.store(true, Ordering::SeqCst);
    });

    // Verify running
    thread::sleep(Duration::from_millis(20));
    assert!(!exited.load(Ordering::SeqCst));

    // Send shutdown
    alpha_h.send(Message::Shutdown).unwrap();

    // Should exit
    handle.join().unwrap();
    assert!(exited.load(Ordering::SeqCst));
}

#[test]
fn shutdown_works_with_multiple_actors() {
    let (alpha_h, mut alpha_s) =
        ActorScheduler::<AlphaData, AlphaControl, AlphaManagement>::new(100, 1024);
    let (beta_h, mut beta_s) =
        ActorScheduler::<BetaData, BetaControl, BetaManagement>::new(100, 1024);

    let alpha_exited = Arc::new(AtomicBool::new(false));
    let beta_exited = Arc::new(AtomicBool::new(false));

    let alpha_exit = alpha_exited.clone();
    let beta_exit = beta_exited.clone();

    let alpha_thread = thread::spawn(move || {
        struct NoopActor;
        impl Actor<AlphaData, AlphaControl, AlphaManagement> for NoopActor {
            fn handle_data(&mut self, _: AlphaData) -> Result<(), actor_scheduler::ActorError> { Ok(()) }
            fn handle_control(&mut self, _: AlphaControl) -> Result<(), actor_scheduler::ActorError> { Ok(()) }
            fn handle_management(&mut self, _: AlphaManagement) -> Result<(), actor_scheduler::ActorError> { Ok(()) }
            fn park(&mut self, h: ActorStatus) -> ActorStatus {
                h
            }
        }
        alpha_s.run(&mut NoopActor);
        alpha_exit.store(true, Ordering::SeqCst);
    });

    let beta_thread = thread::spawn(move || {
        struct NoopActor;
        impl Actor<BetaData, BetaControl, BetaManagement> for NoopActor {
            fn handle_data(&mut self, _: BetaData) -> Result<(), actor_scheduler::ActorError> { Ok(()) }
            fn handle_control(&mut self, _: BetaControl) -> Result<(), actor_scheduler::ActorError> { Ok(()) }
            fn handle_management(&mut self, _: BetaManagement) -> Result<(), actor_scheduler::ActorError> { Ok(()) }
            fn park(&mut self, h: ActorStatus) -> ActorStatus {
                h
            }
        }
        beta_s.run(&mut NoopActor);
        beta_exit.store(true, Ordering::SeqCst);
    });

    // Verify both running
    thread::sleep(Duration::from_millis(20));
    assert!(!alpha_exited.load(Ordering::SeqCst));
    assert!(!beta_exited.load(Ordering::SeqCst));

    // Shutdown both (simulating directory.shutdown())
    beta_h.send(Message::Shutdown).unwrap();
    alpha_h.send(Message::Shutdown).unwrap();

    // Both should exit
    alpha_thread.join().unwrap();
    beta_thread.join().unwrap();

    assert!(alpha_exited.load(Ordering::SeqCst));
    assert!(beta_exited.load(Ordering::SeqCst));
}
