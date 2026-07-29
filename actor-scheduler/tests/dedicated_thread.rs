//! Proof that placement is a driver choice (design doc §5, §7): the same `Transducer` runs
//! unmodified on a dedicated OS thread (`DedicatedThread`) or hosted inside a `Host`, and an
//! OS-bridge `Transducer` (one that overrides `step_os`) only makes sense on the former.

use std::convert::Infallible;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use actor_scheduler::mealy::{Delivery, Flush, Node, Transducer, Wiring, send_port};
use actor_scheduler::spsc::{SpscSender, spsc_channel};
use actor_scheduler::{ActorBuilder, ActorStatus, Exit, HandlerError, Host, Message, SystemStatus};

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
        send_port(&mut out.seen, &self.tx, Delivery::Blocking)
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
    let mut thread = builder.build_node(LaneLog, LaneLogWiring { tx: tx_seen });

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
    let exit = worker.join().expect("dedicated thread must not panic");
    assert_eq!(exit, Exit::Completed);
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
        send_port(&mut out.word, &self.tx, Delivery::Blocking)
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
    let mut thread = builder.build_node(
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
    let exit = worker
        .join()
        .expect("dedicated thread must exit cleanly, not hang or panic");
    assert_eq!(exit, Exit::Completed);
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
        host.sweep().status,
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
    assert_eq!(host.sweep().status, ActorStatus::Idle);
}
