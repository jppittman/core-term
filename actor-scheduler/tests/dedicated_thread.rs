//! Proof that placement is a driver choice (design doc §5, §7): the same `Transducer` runs
//! unmodified on a dedicated OS thread (`DedicatedThread`) or hosted inside a `Host`, and an
//! OS-bridge `Transducer` (one that overrides `step_os`) only makes sense on the former.

use std::convert::Infallible;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use actor_scheduler::mealy::{Flush, Node, Transducer, Wiring, send_port};
use actor_scheduler::spsc::{SpscSender, spsc_channel};
use actor_scheduler::{ActorBuilder, ActorStatus, HandlerError, Host, Message, SystemStatus};

// ────────────────────────────────────────────────────────────────────────────
// A pure Transducer: same shape as mealy.rs's own `LaneLog` fixture, reused
// unmodified across the dedicated-thread and hosted tests below.
// ────────────────────────────────────────────────────────────────────────────

/// Records which lane each step came from, and echoes the tag out its one port.
struct LaneLog;

#[derive(Default)]
struct LaneLogOut {
    seen: Option<&'static str>,
}

struct LaneLogWiring {
    tx: SpscSender<&'static str>,
}

impl Wiring for LaneLogWiring {
    type Out = LaneLogOut;

    fn flush(&mut self, out: &mut LaneLogOut) -> Flush {
        send_port(&mut out.seen, &self.tx)
    }
}

impl Transducer for LaneLog {
    type Control = ();
    type Management = ();
    type Data = ();
    type Out = LaneLogOut;

    fn step_data(&mut self, (): ()) -> Result<LaneLogOut, HandlerError> {
        Ok(LaneLogOut { seen: Some("D") })
    }

    fn step_control(&mut self, (): ()) -> Result<LaneLogOut, HandlerError> {
        Ok(LaneLogOut { seen: Some("C") })
    }

    fn step_management(&mut self, (): ()) -> Result<LaneLogOut, HandlerError> {
        Ok(LaneLogOut { seen: Some("M") })
    }
}

/// Preload a burst across all three lanes through ordinary `ActorHandle::send`, before the
/// dedicated thread ever runs, then confirm Control > Management > Data held for the whole
/// burst — the same priority `ActorScheduler::handle_wake` and `Node::poll`'s lane cycle give,
/// ported rather than redesigned (mealy.rs module doc).
#[test]
fn priority_holds_on_a_dedicated_thread_fed_by_ordinary_handles() {
    let mut builder = ActorBuilder::<(), (), ()>::new(64, None);
    let handle = builder.add_producer();
    let (tx_seen, mut rx_seen) = spsc_channel::<&'static str>(32);
    let mut thread = builder.build_dedicated_thread(LaneLog, LaneLogWiring { tx: tx_seen });

    // The whole burst is queued before the thread exists to drain any of it.
    for _ in 0..3 {
        handle.send(Message::Data(())).unwrap();
    }
    for _ in 0..3 {
        handle.send(Message::Management(())).unwrap();
    }
    for _ in 0..3 {
        handle.send(Message::Control(())).unwrap();
    }

    let worker = std::thread::spawn(move || thread.run());

    let deadline = Instant::now() + Duration::from_secs(5);
    let mut seen = Vec::new();
    while seen.len() < 9 {
        if let Ok(tag) = rx_seen.try_recv() {
            seen.push(tag);
        }
        assert!(
            Instant::now() < deadline,
            "dedicated thread never drained the preloaded burst, got {seen:?}"
        );
    }

    let last_control = seen.iter().rposition(|&t| t == "C").unwrap();
    let first_management = seen.iter().position(|&t| t == "M").unwrap();
    let last_management = seen.iter().rposition(|&t| t == "M").unwrap();
    let first_data = seen.iter().position(|&t| t == "D").unwrap();
    assert!(
        last_control < first_management,
        "all control before any management: {seen:?}"
    );
    assert!(
        last_management < first_data,
        "all management before any data: {seen:?}"
    );

    handle.send(Message::Shutdown).unwrap();
    worker.join().expect("dedicated thread must not panic");
}

// ────────────────────────────────────────────────────────────────────────────
// An OS-bridge Transducer: `step_os` is the only way its data ever moves.
// ────────────────────────────────────────────────────────────────────────────

/// Reads from an external `mpsc::Receiver` — standing in for an epoll set, a PTY, or any
/// other OS source `Host` cannot touch (a green actor may not block).
struct BridgeSource {
    external: mpsc::Receiver<u32>,
}

#[derive(Default)]
struct BridgeOut {
    word: Option<u32>,
}

struct BridgeWiring {
    tx: SpscSender<u32>,
}

impl Wiring for BridgeWiring {
    type Out = BridgeOut;

    fn flush(&mut self, out: &mut BridgeOut) -> Flush {
        send_port(&mut out.word, &self.tx)
    }
}

impl Transducer for BridgeSource {
    type Control = Infallible;
    type Management = Infallible;
    type Data = Infallible;
    type Out = BridgeOut;

    fn step_data(&mut self, msg: Infallible) -> Result<BridgeOut, HandlerError> {
        match msg {}
    }

    fn step_os(&mut self, _status: SystemStatus) -> Result<(BridgeOut, ActorStatus), HandlerError> {
        match self.external.try_recv() {
            // Busy: re-poll without blocking. Whether this was actually the last item is
            // unknown until the next attempt says otherwise — costing one extra idle round
            // trip is cheaper than a second channel just to peek.
            Ok(v) => Ok((BridgeOut { word: Some(v) }, ActorStatus::Busy)),
            Err(mpsc::TryRecvError::Empty | mpsc::TryRecvError::Disconnected) => {
                Ok((BridgeOut::default(), ActorStatus::Idle))
            }
        }
    }
}

/// Bridged data flows out the wiring purely through `step_os`, and the thread returns to
/// blocking on its doorbell once the source runs dry — observed the only way that is really
/// observable in a unit test: everything arrives, and shutdown still works cleanly afterward.
#[test]
fn an_os_bridge_transducer_drains_through_step_os_and_parks_when_idle() {
    let (tx_ext, rx_ext) = mpsc::channel::<u32>();
    for i in 0..5u32 {
        tx_ext.send(i).unwrap();
    }
    drop(tx_ext); // the source is finite: once drained, `step_os` must see it as quiet, not stalled

    let mut builder = ActorBuilder::<Infallible, Infallible, Infallible>::new(4, None);
    let handle = builder.add_producer();
    let (tx_out, mut rx_out) = spsc_channel::<u32>(16);
    let mut thread = builder.build_dedicated_thread(
        BridgeSource { external: rx_ext },
        BridgeWiring { tx: tx_out },
    );

    let worker = std::thread::spawn(move || thread.run());

    // Nothing rings this actor's doorbell on its own — there is no lane traffic, by design (its
    // only input is the OS source). One external wake is the realistic bootstrap: in
    // production this is whatever notices the OS source has data (an epoll thread, a signal
    // handler) and is a driver-integration concern `step_os` itself deliberately has no
    // opinion about.
    handle.waker().wake();

    let deadline = Instant::now() + Duration::from_secs(5);
    let mut got = Vec::new();
    while got.len() < 5 {
        if let Ok(v) = rx_out.try_recv() {
            got.push(v);
        }
        assert!(
            Instant::now() < deadline,
            "bridged items never arrived through step_os, got {got:?}"
        );
    }
    assert_eq!(got, vec![0, 1, 2, 3, 4]);

    handle.send(Message::Shutdown).unwrap();
    worker
        .join()
        .expect("dedicated thread must exit cleanly, not hang or panic");
}

// ────────────────────────────────────────────────────────────────────────────
// The unification's whole claim: the same Transducer, either placement.
// ────────────────────────────────────────────────────────────────────────────

/// `LaneLog` — unchanged from the dedicated-thread test above — adopted into a `Host` and swept
/// directly, with no thread and no `ActorBuilder` involved. Same actor, same `Transducer` impl;
/// only the driver differs, exactly as design doc §5 claims.
#[test]
fn the_same_transducer_runs_dedicated_or_hosted() {
    let (tx_d, rx_d) = spsc_channel::<()>(8);
    let (tx_seen, mut rx_seen) = spsc_channel::<&'static str>(8);

    let mut host = Host::new();
    host.adopt(Node::new(LaneLog, rx_d, LaneLogWiring { tx: tx_seen }));

    tx_d.try_send(()).unwrap();
    assert_eq!(
        host.sweep(),
        ActorStatus::Busy,
        "the hosted node had a data-lane message to run"
    );
    assert_eq!(
        rx_seen.try_recv().unwrap(),
        "D",
        "same Transducer, same step_data, now driven by Host::sweep instead of a doorbell thread"
    );

    // Nothing left: a quiet host reports Idle, exactly like the dedicated thread reports Idle
    // to its own doorbell loop when there is nothing to do.
    assert_eq!(host.sweep(), ActorStatus::Idle);
}

// ────────────────────────────────────────────────────────────────────────────
// Driver-loop liveness: the review findings on #967, pinned.
// ────────────────────────────────────────────────────────────────────────────

/// Consumes and discards everything — for tests about the driver loop, not delivery.
struct DiscardWiring;

impl Wiring for DiscardWiring {
    type Out = LaneLogOut;

    fn flush(&mut self, out: &mut LaneLogOut) -> Flush {
        out.seen = None;
        Flush::Done
    }
}

/// A producer that never stops sending must not be able to starve shutdown: the sweep's lane
/// drain is bounded (`sweep_burst`), so the driver revisits its doorbell no matter how full
/// the lanes stay. Without the bound, this test hangs on `join`.
#[test]
fn a_flooded_lane_does_not_starve_shutdown() {
    let mut builder = ActorBuilder::<(), (), ()>::new(64, None);
    let flood_handle = builder.add_producer();
    let shutdown_handle = builder.add_producer();
    let mut thread = builder.build_dedicated_thread(LaneLog, DiscardWiring);

    // The driver hands its node back after exiting, so the lanes and doorbell stay alive
    // until the flooder is stopped — `ActorHandle::wake` panics on a disconnected doorbell by
    // contract, so the flood must be stopped by flag, not by racing the teardown.
    let worker = std::thread::spawn(move || {
        thread.run();
        thread
    });
    let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let stop_flag = stop.clone();
    let flooder = std::thread::spawn(move || {
        while !stop_flag.load(std::sync::atomic::Ordering::Relaxed) {
            // send_port, the one public non-blocking path to this handle: a full ring must not
            // stall the flood — pressure, not delivery, is the point. `Blocked` is expected
            // mid-flood; keep pressing regardless of the outcome.
            let mut port = Some(Message::Data(()));
            send_port(&mut port, &flood_handle);
        }
    });

    // Let the flood establish itself before asking for shutdown.
    std::thread::sleep(Duration::from_millis(50));
    shutdown_handle
        .send(Message::Shutdown)
        .expect("the doorbell must remain reachable during a flood");

    let thread = worker
        .join()
        .expect("driver must not panic, and must exit despite the flood");

    stop.store(true, std::sync::atomic::Ordering::Relaxed);
    flooder.join().expect("flooder exits on the stop flag");
    drop(thread);
}

/// Dropping the last `ActorHandle` disconnects every lane, but an OS bridge's external source
/// outlives the handles by design — `step_os` gets its turn each sweep until it reports idle,
/// so queued external work drains instead of being discarded on lane completion.
#[test]
fn step_os_gets_a_final_turn_after_the_last_handle_drops() {
    let (tx_ext, rx_ext) = mpsc::channel::<u32>();
    for i in 0..5u32 {
        tx_ext.send(i).unwrap();
    }
    drop(tx_ext);

    let mut builder = ActorBuilder::<Infallible, Infallible, Infallible>::new(4, None);
    let handle = builder.add_producer();
    let (tx_out, mut rx_out) = spsc_channel::<u32>(16);
    let mut thread = builder.build_dedicated_thread(
        BridgeSource { external: rx_ext },
        BridgeWiring { tx: tx_out },
    );

    // Every handle is gone before the driver takes a single step.
    drop(handle);

    thread.run();

    let mut got = Vec::new();
    while let Ok(v) = rx_out.try_recv() {
        got.push(v);
    }
    assert_eq!(
        got,
        vec![0, 1, 2, 3, 4],
        "external work queued behind dead lanes still drains before completion"
    );
}

/// The blocking contract, end to end: a `step_os` that genuinely blocks (the hook's intended
/// use — epoll, PTY reads) must be paired with a `WakeHandler` whose signal persists until
/// consumed, and with one wired, `Shutdown` interrupts the wait and completes the thread. The
/// fake OS wait blocks on an `mpsc::Receiver`; the handler's "self-pipe byte" is a sentinel
/// pushed into that same channel.
#[test]
fn a_blocking_step_os_is_interrupted_for_shutdown() {
    use actor_scheduler::WakeHandler;
    use std::sync::Mutex;

    const WAKE_SENTINEL: u32 = u32::MAX;

    struct SentinelWaker {
        tx: Mutex<mpsc::Sender<u32>>,
    }

    impl WakeHandler for SentinelWaker {
        fn wake(&self) {
            // A dead receiver just means the bridge is gone; a wake has nobody to interrupt.
            let _unused = self.tx.lock().unwrap().send(WAKE_SENTINEL);
        }
    }

    struct BlockingBridge {
        external: mpsc::Receiver<u32>,
    }

    impl Transducer for BlockingBridge {
        type Control = Infallible;
        type Management = Infallible;
        type Data = Infallible;
        type Out = BridgeOut;

        fn step_data(&mut self, msg: Infallible) -> Result<BridgeOut, HandlerError> {
            match msg {}
        }

        fn step_os(
            &mut self,
            _status: SystemStatus,
        ) -> Result<(BridgeOut, ActorStatus), HandlerError> {
            // A real OS wait: blocks until an event or a wake arrives. Always Busy — a
            // blocking bridge's doorbell is its poll set, so the driver must come back
            // through step_os rather than sleep on the channel doorbell (PtyReader's exact
            // posture, and why the driver's try_recv between Busy sweeps is what observes
            // Shutdown).
            match self.external.recv() {
                Ok(WAKE_SENTINEL) => Ok((BridgeOut::default(), ActorStatus::Busy)),
                Ok(v) => Ok((BridgeOut { word: Some(v) }, ActorStatus::Busy)),
                // Source gone: nothing will ever arrive again; let the thread sleep/complete.
                Err(_) => Ok((BridgeOut::default(), ActorStatus::Idle)),
            }
        }
    }

    let (tx_ext, rx_ext) = mpsc::channel::<u32>();
    let handler = std::sync::Arc::new(SentinelWaker {
        tx: Mutex::new(tx_ext.clone()),
    });

    let mut builder = ActorBuilder::<Infallible, Infallible, Infallible>::new(4, Some(handler));
    let handle = builder.add_producer();
    let (tx_out, mut rx_out) = spsc_channel::<u32>(8);
    let mut thread = builder.build_dedicated_thread(
        BlockingBridge { external: rx_ext },
        BridgeWiring { tx: tx_out },
    );

    let worker = std::thread::spawn(move || thread.run());
    handle.waker().wake(); // first entry into the blocking wait

    // A real event flows while the bridge lives.
    tx_ext.send(21).unwrap();
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        if let Ok(v) = rx_out.try_recv() {
            assert_eq!(v, 21);
            break;
        }
        assert!(
            Instant::now() < deadline,
            "the blocking bridge never emitted"
        );
    }

    // The bridge is now blocked in recv() with nothing coming. Shutdown must interrupt it via
    // the wake handler — without one, this join would hang forever.
    handle.send(Message::Shutdown).unwrap();
    worker.join().expect("driver must not panic");
}

/// A gone wiring target can never heal, and no lane can progress past the parked word (the
/// outbox-first contract) — so the driver panics loudly instead of spinning a core on retries
/// that cannot succeed. There is no supervisor to hand the problem to instead; fail fast, fail
/// loudly.
#[test]
#[should_panic(expected = "wiring target disconnected")]
fn a_disconnected_wiring_target_panics_instead_of_spinning() {
    let mut builder = ActorBuilder::<(), (), ()>::new(8, None);
    let handle = builder.add_producer();
    let (tx_seen, rx_seen) = spsc_channel::<&'static str>(8);
    let mut thread = builder.build_dedicated_thread(LaneLog, LaneLogWiring { tx: tx_seen });

    drop(rx_seen); // the downstream consumer dies before the first delivery

    handle.send(Message::Data(())).unwrap();
    thread.run();
}

/// Dead lanes are not completion while a `Waker` is still held somewhere: the waker is
/// someone's declared intent to come back, so an idle OS bridge sleeps instead of exiting —
/// and wakes up for the event when it arrives. Terminality belongs to the doorbell.
#[test]
fn an_idle_os_bridge_survives_while_a_waker_exists() {
    let (tx_ext, rx_ext) = mpsc::channel::<u32>();

    let mut builder = ActorBuilder::<Infallible, Infallible, Infallible>::new(4, None);
    let handle = builder.add_producer();
    let waker = handle.waker();
    let (tx_out, mut rx_out) = spsc_channel::<u32>(8);
    let mut thread = builder.build_dedicated_thread(
        BridgeSource { external: rx_ext },
        BridgeWiring { tx: tx_out },
    );

    // Every lane dies before the driver starts; only the waker keeps the doorbell alive.
    drop(handle);
    let worker = std::thread::spawn(move || thread.run());

    // The bridge is asleep on its doorbell. A late event arrives: push, then wake.
    std::thread::sleep(Duration::from_millis(30));
    tx_ext.send(9).unwrap();
    waker.wake();

    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        if let Ok(v) = rx_out.try_recv() {
            assert_eq!(v, 9, "the late event flowed through the surviving bridge");
            break;
        }
        assert!(
            Instant::now() < deadline,
            "the bridge exited on dead lanes instead of sleeping on its live waker"
        );
    }

    // Now nothing can ever wake it again — the doorbell disconnects, and that is completion.
    drop(tx_ext);
    drop(waker);
    worker.join().expect("driver must not panic");
}

/// The retention guarantee across lane completion: a `step_os` output that parks on a full
/// ring, with every handle already dropped, must hold the driver open until it delivers —
/// exiting `Completed` there would silently drop the word the actor just produced.
#[test]
fn a_parked_final_word_survives_lane_completion() {
    let (tx_ext, rx_ext) = mpsc::channel::<u32>();
    tx_ext.send(5).unwrap();
    drop(tx_ext);

    let mut builder = ActorBuilder::<Infallible, Infallible, Infallible>::new(4, None);
    let handle = builder.add_producer();
    let (tx_out, mut rx_out) = spsc_channel::<u32>(2);
    while tx_out.try_send(99).is_ok() {}

    let mut thread = builder.build_dedicated_thread(
        BridgeSource { external: rx_ext },
        BridgeWiring { tx: tx_out },
    );
    drop(handle);

    let worker = std::thread::spawn(move || thread.run());

    // The driver is spinning on the parked word; free the ring and it must deliver, then
    // complete.
    std::thread::sleep(Duration::from_millis(50));
    let mut got = Vec::new();
    let deadline = Instant::now() + Duration::from_secs(5);
    while got.iter().filter(|&&v| v == 5).count() == 0 {
        if let Ok(v) = rx_out.try_recv() {
            got.push(v);
        }
        assert!(
            Instant::now() < deadline,
            "the parked word never arrived, got {got:?}"
        );
    }

    worker.join().expect("driver must not panic");
}

/// A `step_os` that yields a continuation and honestly reports itself `Idle` must still make
/// progress: the stored continuation is runtime-owned work, so the driver keeps sweeping
/// instead of blocking on a doorbell nothing will ring.
#[test]
fn a_step_os_continuation_resumes_without_an_external_wake() {
    struct OsKick {
        kicked: bool,
    }

    #[derive(Default)]
    struct OsKickOut {
        word: Option<u32>,
        again: Option<u32>,
    }

    struct OsKickWiring {
        tx: SpscSender<u32>,
    }

    impl Wiring for OsKickWiring {
        type Out = OsKickOut;

        fn flush(&mut self, out: &mut OsKickOut) -> Flush {
            send_port(&mut out.word, &self.tx)
        }
    }

    impl Transducer for OsKick {
        type Control = Infallible;
        type Management = Infallible;
        type Data = u32;
        type Out = OsKickOut;

        fn step_data(&mut self, n: u32) -> Result<OsKickOut, HandlerError> {
            Ok(OsKickOut {
                word: Some(n * 10),
                again: None,
            })
        }

        fn step_os(&mut self, _: SystemStatus) -> Result<(OsKickOut, ActorStatus), HandlerError> {
            if self.kicked {
                return Ok((OsKickOut::default(), ActorStatus::Idle));
            }
            self.kicked = true;
            // Idle is the trap: the continuation alone must keep the driver going.
            Ok((
                OsKickOut {
                    word: None,
                    again: Some(7),
                },
                ActorStatus::Idle,
            ))
        }

        fn take_continuation(out: &mut OsKickOut) -> Option<u32> {
            out.again.take()
        }
    }

    let mut builder = ActorBuilder::<u32, Infallible, Infallible>::new(4, None);
    let handle = builder.add_producer();
    let (tx_out, mut rx_out) = spsc_channel::<u32>(8);
    let mut thread = builder.build_dedicated_thread(OsKick { kicked: false }, OsKickWiring { tx: tx_out });
    drop(handle);

    thread.run();
    assert_eq!(
        rx_out.try_recv().unwrap(),
        70,
        "the continuation resumed through step_data with no external wake"
    );
}
