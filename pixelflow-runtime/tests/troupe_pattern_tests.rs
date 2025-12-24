//! Tests for the Troupe Pattern
//!
//! The troupe system provides lifecycle management for groups of actors.
//! These tests verify:
//! - Directory pattern (actors access each other via directory)
//! - Shutdown as message propagation
//! - Two-phase initialization (new() -> exposed() -> play())
//! - Cross-actor messaging
//!
//! Key design principles:
//! - Actors borrow &'a Directory via TroupeActor trait (in scoped threads)
//! - Shutdown is an explicit Control message with Default impl
//! - run() only exits when all channel senders are dropped
//! - [main] actor runs on calling thread, others in scoped threads

use actor_scheduler::{Actor, ActorHandle, ActorScheduler, Message, ParkHint};
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
#[allow(dead_code)]
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

impl TestDirectory {
    /// Send shutdown to all actors. Each Control type has Default = Shutdown.
    fn shutdown_all(&self) {
        let _ = self.alpha.send(Message::Control(AlphaControl::default()));
        let _ = self.beta.send(Message::Control(BetaControl::default()));
    }
}

// ============================================================================
// Shutdown Message Propagation Tests
// ============================================================================

/// Tests that shutdown is a message that propagates between actors.
///
/// The shutdown pattern:
/// 1. Send Shutdown to initiating actor
/// 2. Actor processes Shutdown and propagates to dependent actors
/// 3. All actors receive and process Shutdown messages
#[test]
fn shutdown_propagates_as_message() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let alpha_shutdown = Arc::new(AtomicBool::new(false));
    let beta_shutdown = Arc::new(AtomicBool::new(false));

    let (alpha_h, mut alpha_s) = ActorScheduler::<AlphaData, AlphaControl, AlphaManagement>::new(100, 1024);
    let (beta_h, mut beta_s) = ActorScheduler::<BetaData, BetaControl, BetaManagement>::new(100, 1024);

    // For testing, use Arc to share directory
    let dir = Arc::new(TestDirectory {
        alpha: alpha_h.clone(),
        beta: beta_h.clone(),
    });

    let dir_a = dir.clone();
    let log_a = log.clone();
    let shutdown_a = alpha_shutdown.clone();
    thread::spawn(move || {
        struct AlphaActorArc {
            dir: Arc<TestDirectory>,
            log: Arc<Mutex<Vec<String>>>,
            shutdown_received: Arc<AtomicBool>,
        }
        impl Actor<AlphaData, AlphaControl, AlphaManagement> for AlphaActorArc {
            fn handle_data(&mut self, msg: AlphaData) {
                self.log.lock().unwrap().push(format!("Alpha:Data:{}", msg.0));
            }
            fn handle_control(&mut self, cmd: AlphaControl) {
                match cmd {
                    AlphaControl::Ping => {
                        self.log.lock().unwrap().push("Alpha:Ping".to_string());
                        let _ = self.dir.beta.send(Message::Control(BetaControl::Pong));
                    }
                    AlphaControl::Shutdown => {
                        self.log.lock().unwrap().push("Alpha:Shutdown".to_string());
                        self.shutdown_received.store(true, Ordering::SeqCst);
                        // Propagate shutdown
                        let _ = self.dir.beta.send(Message::Control(BetaControl::Shutdown));
                    }
                }
            }
            fn handle_management(&mut self, _: AlphaManagement) {}
            fn park(&mut self, h: ParkHint) -> ParkHint { h }
        }
        let mut actor = AlphaActorArc { dir: dir_a, log: log_a, shutdown_received: shutdown_a };
        alpha_s.run(&mut actor);
    });

    let dir_b = dir.clone();
    let log_b = log.clone();
    let shutdown_b = beta_shutdown.clone();
    thread::spawn(move || {
        struct BetaActorArc {
            log: Arc<Mutex<Vec<String>>>,
            shutdown_received: Arc<AtomicBool>,
        }
        impl Actor<BetaData, BetaControl, BetaManagement> for BetaActorArc {
            fn handle_data(&mut self, _: BetaData) {}
            fn handle_control(&mut self, cmd: BetaControl) {
                match cmd {
                    BetaControl::Pong => {
                        self.log.lock().unwrap().push("Beta:Pong".to_string());
                    }
                    BetaControl::Shutdown => {
                        self.log.lock().unwrap().push("Beta:Shutdown".to_string());
                        self.shutdown_received.store(true, Ordering::SeqCst);
                    }
                }
            }
            fn handle_management(&mut self, _: BetaManagement) {}
            fn park(&mut self, h: ParkHint) -> ParkHint { h }
        }
        let mut actor = BetaActorArc { log: log_b, shutdown_received: shutdown_b };
        let _ = dir_b; // Keep dir_b alive
        beta_s.run(&mut actor);
    });

    // Send Shutdown to Alpha - it should propagate to Beta
    alpha_h.send(Message::Control(AlphaControl::Shutdown)).unwrap();

    // Wait for shutdown messages to be processed
    let start = std::time::Instant::now();
    while !alpha_shutdown.load(Ordering::SeqCst) || !beta_shutdown.load(Ordering::SeqCst) {
        if start.elapsed() > Duration::from_secs(2) {
            panic!("Timeout waiting for shutdown propagation");
        }
        thread::sleep(Duration::from_millis(10));
    }

    // Verify shutdown was received by both
    assert!(alpha_shutdown.load(Ordering::SeqCst));
    assert!(beta_shutdown.load(Ordering::SeqCst));

    let log = log.lock().unwrap();
    assert!(log.contains(&"Alpha:Shutdown".to_string()));
    assert!(log.contains(&"Beta:Shutdown".to_string()));
}

/// Tests that shutdown_all() sends shutdown to all actors.
#[test]
fn shutdown_all_sends_to_all_actors() {
    let alpha_shutdown = Arc::new(AtomicBool::new(false));
    let beta_shutdown = Arc::new(AtomicBool::new(false));

    let (alpha_h, mut alpha_s) = ActorScheduler::<AlphaData, AlphaControl, AlphaManagement>::new(100, 1024);
    let (beta_h, mut beta_s) = ActorScheduler::<BetaData, BetaControl, BetaManagement>::new(100, 1024);

    let dir = TestDirectory {
        alpha: alpha_h.clone(),
        beta: beta_h.clone(),
    };

    let shutdown_a = alpha_shutdown.clone();
    thread::spawn(move || {
        struct ShutdownTracker(Arc<AtomicBool>);
        impl Actor<AlphaData, AlphaControl, AlphaManagement> for ShutdownTracker {
            fn handle_data(&mut self, _: AlphaData) {}
            fn handle_control(&mut self, cmd: AlphaControl) {
                if matches!(cmd, AlphaControl::Shutdown) {
                    self.0.store(true, Ordering::SeqCst);
                }
            }
            fn handle_management(&mut self, _: AlphaManagement) {}
            fn park(&mut self, h: ParkHint) -> ParkHint { h }
        }
        alpha_s.run(&mut ShutdownTracker(shutdown_a));
    });

    let shutdown_b = beta_shutdown.clone();
    thread::spawn(move || {
        struct ShutdownTracker(Arc<AtomicBool>);
        impl Actor<BetaData, BetaControl, BetaManagement> for ShutdownTracker {
            fn handle_data(&mut self, _: BetaData) {}
            fn handle_control(&mut self, cmd: BetaControl) {
                if matches!(cmd, BetaControl::Shutdown) {
                    self.0.store(true, Ordering::SeqCst);
                }
            }
            fn handle_management(&mut self, _: BetaManagement) {}
            fn park(&mut self, h: ParkHint) -> ParkHint { h }
        }
        beta_s.run(&mut ShutdownTracker(shutdown_b));
    });

    // shutdown_all() sends Default::default() to all actors
    dir.shutdown_all();

    // Wait for both to receive shutdown
    let start = std::time::Instant::now();
    while !alpha_shutdown.load(Ordering::SeqCst) || !beta_shutdown.load(Ordering::SeqCst) {
        if start.elapsed() > Duration::from_secs(2) {
            panic!("Timeout waiting for shutdown_all");
        }
        thread::sleep(Duration::from_millis(10));
    }

    assert!(alpha_shutdown.load(Ordering::SeqCst));
    assert!(beta_shutdown.load(Ordering::SeqCst));
}

// ============================================================================
// Cross-Actor Messaging Tests
// ============================================================================

/// Tests full message chain: Alpha -> Beta -> Alpha
#[test]
fn cross_actor_message_chain_completes() {
    let log = Arc::new(Mutex::new(Vec::new()));

    let (alpha_h, mut alpha_s) = ActorScheduler::<AlphaData, AlphaControl, AlphaManagement>::new(100, 1024);
    let (beta_h, mut beta_s) = ActorScheduler::<BetaData, BetaControl, BetaManagement>::new(100, 1024);

    let dir = Arc::new(TestDirectory {
        alpha: alpha_h.clone(),
        beta: beta_h.clone(),
    });

    let dir_a = dir.clone();
    let log_a = log.clone();
    thread::spawn(move || {
        struct AlphaActor {
            dir: Arc<TestDirectory>,
            log: Arc<Mutex<Vec<String>>>,
        }
        impl Actor<AlphaData, AlphaControl, AlphaManagement> for AlphaActor {
            fn handle_data(&mut self, msg: AlphaData) {
                self.log.lock().unwrap().push(format!("Alpha:Data:{}", msg.0));
            }
            fn handle_control(&mut self, cmd: AlphaControl) {
                if matches!(cmd, AlphaControl::Ping) {
                    self.log.lock().unwrap().push("Alpha:Ping".to_string());
                    let _ = self.dir.beta.send(Message::Control(BetaControl::Pong));
                }
            }
            fn handle_management(&mut self, _: AlphaManagement) {}
            fn park(&mut self, h: ParkHint) -> ParkHint { h }
        }
        alpha_s.run(&mut AlphaActor { dir: dir_a, log: log_a });
    });

    let dir_b = dir.clone();
    let log_b = log.clone();
    thread::spawn(move || {
        struct BetaActor {
            dir: Arc<TestDirectory>,
            log: Arc<Mutex<Vec<String>>>,
        }
        impl Actor<BetaData, BetaControl, BetaManagement> for BetaActor {
            fn handle_data(&mut self, _: BetaData) {}
            fn handle_control(&mut self, cmd: BetaControl) {
                if matches!(cmd, BetaControl::Pong) {
                    self.log.lock().unwrap().push("Beta:Pong".to_string());
                    let _ = self.dir.alpha.send(Message::Data(AlphaData("pong-response".to_string())));
                }
            }
            fn handle_management(&mut self, _: BetaManagement) {}
            fn park(&mut self, h: ParkHint) -> ParkHint { h }
        }
        beta_s.run(&mut BetaActor { dir: dir_b, log: log_b });
    });

    // Alpha:Ping -> Beta:Pong -> Alpha:Data
    alpha_h.send(Message::Control(AlphaControl::Ping)).unwrap();

    // Wait for full chain
    let start = std::time::Instant::now();
    loop {
        thread::sleep(Duration::from_millis(10));
        let snapshot = log.lock().unwrap();
        if snapshot.contains(&"Alpha:Data:pong-response".to_string()) {
            break;
        }
        if start.elapsed() > Duration::from_secs(2) {
            panic!("Timeout waiting for message chain. Log: {:?}", *snapshot);
        }
    }

    let log = log.lock().unwrap();
    assert!(log.contains(&"Alpha:Ping".to_string()));
    assert!(log.contains(&"Beta:Pong".to_string()));
    assert!(log.contains(&"Alpha:Data:pong-response".to_string()));
}

/// Tests circular ping-pong messaging completes without deadlock.
#[test]
fn circular_messaging_completes_without_deadlock() {
    let (alpha_h, mut alpha_s) = ActorScheduler::<AlphaData, AlphaControl, AlphaManagement>::new(100, 1000);
    let (beta_h, mut beta_s) = ActorScheduler::<BetaData, BetaControl, BetaManagement>::new(100, 1000);

    let dir = Arc::new(TestDirectory {
        alpha: alpha_h.clone(),
        beta: beta_h.clone(),
    });

    let ping_count = Arc::new(AtomicUsize::new(0));
    let pong_count = Arc::new(AtomicUsize::new(0));

    let dir_a = dir.clone();
    let ping_clone = ping_count.clone();
    thread::spawn(move || {
        struct PingActor {
            dir: Arc<TestDirectory>,
            count: Arc<AtomicUsize>,
            max: usize,
        }
        impl Actor<AlphaData, AlphaControl, AlphaManagement> for PingActor {
            fn handle_data(&mut self, _: AlphaData) {}
            fn handle_control(&mut self, cmd: AlphaControl) {
                if matches!(cmd, AlphaControl::Ping) {
                    let c = self.count.fetch_add(1, Ordering::SeqCst);
                    if c < self.max {
                        let _ = self.dir.beta.send(Message::Control(BetaControl::Pong));
                    }
                }
            }
            fn handle_management(&mut self, _: AlphaManagement) {}
            fn park(&mut self, h: ParkHint) -> ParkHint { h }
        }
        alpha_s.run(&mut PingActor { dir: dir_a, count: ping_clone, max: 100 });
    });

    let dir_b = dir.clone();
    let pong_clone = pong_count.clone();
    thread::spawn(move || {
        struct PongActor {
            dir: Arc<TestDirectory>,
            count: Arc<AtomicUsize>,
            max: usize,
        }
        impl Actor<BetaData, BetaControl, BetaManagement> for PongActor {
            fn handle_data(&mut self, _: BetaData) {}
            fn handle_control(&mut self, cmd: BetaControl) {
                if matches!(cmd, BetaControl::Pong) {
                    let c = self.count.fetch_add(1, Ordering::SeqCst);
                    if c < self.max {
                        let _ = self.dir.alpha.send(Message::Control(AlphaControl::Ping));
                    }
                }
            }
            fn handle_management(&mut self, _: BetaManagement) {}
            fn park(&mut self, h: ParkHint) -> ParkHint { h }
        }
        beta_s.run(&mut PongActor { dir: dir_b, count: pong_clone, max: 100 });
    });

    // Start ping-pong
    alpha_h.send(Message::Control(AlphaControl::Ping)).unwrap();

    // Wait for completion
    let start = std::time::Instant::now();
    while ping_count.load(Ordering::SeqCst) < 100 || pong_count.load(Ordering::SeqCst) < 100 {
        if start.elapsed() > Duration::from_secs(5) {
            panic!(
                "Deadlock detected! ping={}, pong={}",
                ping_count.load(Ordering::SeqCst),
                pong_count.load(Ordering::SeqCst)
            );
        }
        thread::sleep(Duration::from_millis(10));
    }

    assert!(ping_count.load(Ordering::SeqCst) >= 100);
    assert!(pong_count.load(Ordering::SeqCst) >= 100);
}

// ============================================================================
// Two-Phase Initialization Tests
// ============================================================================

/// Simulates the Troupe struct that troupe! would generate
struct TestTroupe {
    directory: TestDirectory,
    #[allow(dead_code)]
    alpha_scheduler: ActorScheduler<AlphaData, AlphaControl, AlphaManagement>,
    #[allow(dead_code)]
    beta_scheduler: ActorScheduler<BetaData, BetaControl, BetaManagement>,
}

struct TestExposedHandles {
    alpha: ActorHandle<AlphaData, AlphaControl, AlphaManagement>,
}

impl TestTroupe {
    fn new() -> Self {
        let (alpha_h, alpha_s) = ActorScheduler::<AlphaData, AlphaControl, AlphaManagement>::new(100, 1024);
        let (beta_h, beta_s) = ActorScheduler::<BetaData, BetaControl, BetaManagement>::new(100, 1024);

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
    let troupe = TestTroupe::new();

    // Phase 1: Get exposed handles and send messages before actors run
    let exposed = troupe.exposed();

    // Messages queue before play() is called
    exposed.alpha.send(Message::Control(AlphaControl::Ping)).unwrap();
    exposed.alpha.send(Message::Data(AlphaData("pre-play".to_string()))).unwrap();

    // Verify handles work
    let alpha_h2 = exposed.alpha.clone();
    alpha_h2.send(Message::Control(AlphaControl::Ping)).unwrap();
}

#[test]
fn exposed_handles_can_outlive_troupe_reference() {
    let exposed = {
        let troupe = TestTroupe::new();
        troupe.exposed()
    };
    // Troupe dropped, but handles reference still valid
    let _: TestExposedHandles = exposed;
}

// ============================================================================
// Thread Lifecycle Tests
// ============================================================================

#[test]
fn all_actor_threads_exit_on_channel_close() {
    let (alpha_h, mut alpha_s) = ActorScheduler::<AlphaData, AlphaControl, AlphaManagement>::new(100, 1024);
    let (beta_h, mut beta_s) = ActorScheduler::<BetaData, BetaControl, BetaManagement>::new(100, 1024);

    let alpha_exited = Arc::new(AtomicBool::new(false));
    let beta_exited = Arc::new(AtomicBool::new(false));

    let alpha_exit = alpha_exited.clone();
    let beta_exit = beta_exited.clone();

    let alpha_thread = thread::spawn(move || {
        struct NoopActor;
        impl Actor<AlphaData, AlphaControl, AlphaManagement> for NoopActor {
            fn handle_data(&mut self, _: AlphaData) {}
            fn handle_control(&mut self, _: AlphaControl) {}
            fn handle_management(&mut self, _: AlphaManagement) {}
            fn park(&mut self, h: ParkHint) -> ParkHint { h }
        }
        alpha_s.run(&mut NoopActor);
        alpha_exit.store(true, Ordering::SeqCst);
    });

    let beta_thread = thread::spawn(move || {
        struct NoopActor;
        impl Actor<BetaData, BetaControl, BetaManagement> for NoopActor {
            fn handle_data(&mut self, _: BetaData) {}
            fn handle_control(&mut self, _: BetaControl) {}
            fn handle_management(&mut self, _: BetaManagement) {}
            fn park(&mut self, h: ParkHint) -> ParkHint { h }
        }
        beta_s.run(&mut NoopActor);
        beta_exit.store(true, Ordering::SeqCst);
    });

    thread::sleep(Duration::from_millis(20));
    assert!(!alpha_exited.load(Ordering::SeqCst));
    assert!(!beta_exited.load(Ordering::SeqCst));

    // Drop handles - channels close, run() exits
    drop(alpha_h);
    drop(beta_h);

    alpha_thread.join().unwrap();
    beta_thread.join().unwrap();

    assert!(alpha_exited.load(Ordering::SeqCst));
    assert!(beta_exited.load(Ordering::SeqCst));
}

#[test]
fn actor_thread_panic_is_isolated() {
    let (alpha_h, mut alpha_s) = ActorScheduler::<AlphaData, AlphaControl, AlphaManagement>::new(100, 1024);
    let (beta_h, mut beta_s) = ActorScheduler::<BetaData, BetaControl, BetaManagement>::new(100, 1024);

    let beta_count = Arc::new(AtomicUsize::new(0));
    let beta_count_clone = beta_count.clone();

    let alpha_thread = thread::spawn(move || {
        struct PanicActor;
        impl Actor<AlphaData, AlphaControl, AlphaManagement> for PanicActor {
            fn handle_data(&mut self, _: AlphaData) {
                panic!("Alpha panics!");
            }
            fn handle_control(&mut self, _: AlphaControl) {}
            fn handle_management(&mut self, _: AlphaManagement) {}
            fn park(&mut self, h: ParkHint) -> ParkHint { h }
        }
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            alpha_s.run(&mut PanicActor);
        }))
    });

    let beta_thread = thread::spawn(move || {
        struct CountActor(Arc<AtomicUsize>);
        impl Actor<BetaData, BetaControl, BetaManagement> for CountActor {
            fn handle_data(&mut self, _: BetaData) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
            fn handle_control(&mut self, _: BetaControl) {}
            fn handle_management(&mut self, _: BetaManagement) {}
            fn park(&mut self, h: ParkHint) -> ParkHint { h }
        }
        beta_s.run(&mut CountActor(beta_count_clone));
    });

    alpha_h.send(Message::Data(AlphaData("boom".to_string()))).unwrap();
    thread::sleep(Duration::from_millis(50));

    beta_h.send(Message::Data(BetaData(1))).unwrap();
    beta_h.send(Message::Data(BetaData(2))).unwrap();
    thread::sleep(Duration::from_millis(50));

    drop(alpha_h);
    drop(beta_h);

    let alpha_result = alpha_thread.join();
    beta_thread.join().unwrap();

    assert!(alpha_result.is_ok());
    assert!(alpha_result.unwrap().is_err());
    assert_eq!(beta_count.load(Ordering::SeqCst), 2);
}

// ============================================================================
// Startup Coordination Tests
// ============================================================================

#[test]
fn actors_can_coordinate_startup_with_barrier() {
    let barrier = Arc::new(Barrier::new(3));

    let (alpha_h, mut alpha_s) = ActorScheduler::<AlphaData, AlphaControl, AlphaManagement>::new(100, 1024);
    let (beta_h, mut beta_s) = ActorScheduler::<BetaData, BetaControl, BetaManagement>::new(100, 1024);

    let alpha_started = Arc::new(AtomicBool::new(false));
    let beta_started = Arc::new(AtomicBool::new(false));

    let barrier_a = barrier.clone();
    let started_a = alpha_started.clone();
    let alpha_thread = thread::spawn(move || {
        started_a.store(true, Ordering::SeqCst);
        barrier_a.wait();
        struct NoopActor;
        impl Actor<AlphaData, AlphaControl, AlphaManagement> for NoopActor {
            fn handle_data(&mut self, _: AlphaData) {}
            fn handle_control(&mut self, _: AlphaControl) {}
            fn handle_management(&mut self, _: AlphaManagement) {}
            fn park(&mut self, h: ParkHint) -> ParkHint { h }
        }
        alpha_s.run(&mut NoopActor);
    });

    let barrier_b = barrier.clone();
    let started_b = beta_started.clone();
    let beta_thread = thread::spawn(move || {
        started_b.store(true, Ordering::SeqCst);
        barrier_b.wait();
        struct NoopActor;
        impl Actor<BetaData, BetaControl, BetaManagement> for NoopActor {
            fn handle_data(&mut self, _: BetaData) {}
            fn handle_control(&mut self, _: BetaControl) {}
            fn handle_management(&mut self, _: BetaManagement) {}
            fn park(&mut self, h: ParkHint) -> ParkHint { h }
        }
        beta_s.run(&mut NoopActor);
    });

    barrier.wait();

    assert!(alpha_started.load(Ordering::SeqCst));
    assert!(beta_started.load(Ordering::SeqCst));

    drop(alpha_h);
    drop(beta_h);

    alpha_thread.join().unwrap();
    beta_thread.join().unwrap();
}

// ============================================================================
// Default Control Message Tests
// ============================================================================

#[test]
fn default_control_is_shutdown() {
    let alpha_default = AlphaControl::default();
    let beta_default = BetaControl::default();

    assert!(matches!(alpha_default, AlphaControl::Shutdown));
    assert!(matches!(beta_default, BetaControl::Shutdown));
}

#[test]
fn sending_default_control_via_directory() {
    let (alpha_h, _alpha_s) = ActorScheduler::<AlphaData, AlphaControl, AlphaManagement>::new(100, 1024);
    let (beta_h, _beta_s) = ActorScheduler::<BetaData, BetaControl, BetaManagement>::new(100, 1024);

    let dir = TestDirectory {
        alpha: alpha_h,
        beta: beta_h,
    };

    // shutdown_all sends Default::default() which is Shutdown
    dir.shutdown_all();
}
