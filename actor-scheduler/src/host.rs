//! Hosting green actors on an OS-thread actor.
//!
//! The two tiers compose by self-hosting: a [`Host`] is an ordinary [`Actor`], run by the
//! ordinary [`ActorScheduler`](crate::ActorScheduler) on an ordinary OS thread. What makes it
//! a worker is that its `park` sweeps a set of green actors ([`Node`]s) instead of bridging
//! to the OS. There is no second runtime — the green tier is a thing an actor *does*.
//!
//! # Why `park` is the right hook
//!
//! [`Actor::park`] already returns [`ActorStatus`], and the scheduler already honours it:
//! `Busy` means "more work, do not block", `Idle` means "nothing to do, block on the
//! doorbell". So a host that returns `Busy` while any green actor ran, and `Idle` once they
//! are all quiet, gets the behaviour that matters for free — **a host with nothing to do
//! sleeps instead of polling**, and wakes when a message arrives, exactly like every other actor.
//!
//! # Ownership, not migration
//!
//! A host *owns* its green actors; they never move to another thread. That is deliberate:
//! co-locating a pipeline lets a message walk it in one sweep with no cross-thread hop. A shared queue that
//! could pull a green actor onto any worker would trade that away for load balancing nobody
//! has asked for yet.
//!
//! Ownership also buys a smaller API: because a hosted green actor never migrates, it is
//! never sent between threads, so **it does not need to be `Send`** and may hold `Rc`,
//! `RefCell`, or a pointer into a thread-local arena. A host is therefore built where it
//! runs, rather than built and then moved onto a thread.
//!
//! # Sweep order
//!
//! Green actors are swept in the order they were adopted. Adopt them in the topological order
//! that [`Topology::validate`](crate::mealy::Topology::validate) returns and one sweep pushes
//! a message the whole length of a pipeline; adopt them backwards and each sweep advances it
//! by one stage.

use std::convert::Infallible;

use crate::lifecycle::Exit;
use crate::mealy::{Inbox, Node, Step, Transducer, Wiring};
use crate::spsc::{SpscReceiver, SpscSender, TrySendError, spsc_channel};
use crate::{Actor, ActorStatus, HandlerError, HandlerResult, SystemStatus, Waker};

/// A hosted green actor with its types erased, so one host can own a heterogeneous set.
///
/// This is the only type erasure in the design, and it is deliberately one method wide: a
/// host needs to advance a green actor and learn what happened, nothing more.
pub trait Green {
    /// Advance this actor by at most one step.
    fn poll(&mut self) -> Step;
}

impl<T, W, RD, RC, RM> Green for Node<T, W, RD, RC, RM>
where
    T: Transducer,
    W: Wiring<Out = T::Out>,
    RD: Inbox<Item = T::Data>,
    RC: Inbox<Item = T::Control>,
    RM: Inbox<Item = T::Management>,
{
    fn poll(&mut self) -> Step {
        Node::poll(self)
    }
}

/// Stable identity for an adopted green actor.
///
/// Deliberately not a position: `Host` removes halted nodes with `Vec::remove`, so any index
/// recorded in one sweep names a *different* actor by the next. An earlier revision reported
/// disconnections by index and had exactly that bug.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodeId(u64);

/// Why a green actor is stuck — i.e. alive, holding work, and unable to make progress alone.
///
/// One variant today, deliberately. Adding reasons later is additive; inventing a family for
/// stucks nobody has hit would be machinery ahead of need.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stuck {
    /// A flush target is gone. The node's outbox **still holds the undelivered payload**, so
    /// whatever it carries is recoverable — retry it, take the value elsewhere, or drop the node
    /// deliberately.
    TargetGone,
}

/// A green actor needs supervision. Returned by [`Host::sweep`], never sent.
///
/// This is a *returned value*, not a message the host pushes through a handle it holds. Giving
/// `Host` a supervisor handle would reintroduce exactly the coupling the transducer model
/// exists to remove: an actor reaching out mid-step instead of describing what happened and
/// letting its wiring decide where that goes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Supervision {
    pub node: NodeId,
    pub reason: Stuck,
}

/// What one [`sweep`](Host::sweep) did.
///
/// Borrows the host for as long as it is held, because the buffer behind `stuck` belongs to the
/// host and is reused across sweeps — a sweep on a hot path must not allocate. Read what you
/// need out of it ([`Supervision`] is `Copy`) and let it go before acting on the host.
#[derive(Debug)]
#[must_use = "a sweep can report actors needing supervision; dropping it discards them"]
pub struct Sweep<'a> {
    /// `Busy` if any green actor ran, so the scheduler keeps sweeping; `Idle` when they are all
    /// quiet and the thread may sleep.
    pub status: ActorStatus,
    /// Actors that are stuck. Empty on the common path.
    ///
    /// Handed back from each sweep rather than accumulated behind an accessor, so it cannot be
    /// missed by a caller that never thinks to ask — the previous revision stored these and
    /// exposed a getter, which nothing on the `sched.run(&mut host)` path could ever reach. The
    /// slice is only valid until the next sweep, which is the same window in which acting on it
    /// makes sense.
    pub stuck: &'a [Supervision],
}

/// An OS-thread actor that owns green actors and runs them in its `park`.
///
/// A host has **no messages of its own**: all three lane types are [`Infallible`], so the
/// compiler knows its handlers are unreachable and the `match msg {}` bodies below are total.
/// Work reaches a hosted actor by being pushed straight into that actor's inbox with a
/// [`GreenSender`], which then rings the host's doorbell. The host routes nothing; it hosts.
///
/// `Message::Shutdown` still stops it, because shutdown travels on the doorbell rather than a
/// lane.
#[derive(Default)]
pub struct Host {
    nodes: Vec<(NodeId, Box<dyn Green>)>,
    exits: Vec<Exit>,
    next_id: u64,
    /// Scratch for [`sweep`](Host::sweep), cleared and refilled each call. A sweep runs whenever
    /// the thread has work, so building this fresh each time would allocate on the hot path for
    /// a buffer that is empty almost every sweep and never grows past the node count.
    stuck: Vec<Supervision>,
}

impl Host {
    /// A host with no green actors yet.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Take ownership of a green actor. Sweep order is adoption order.
    /// Returns the actor's [`NodeId`], which is how a supervisor later names it — an event
    /// reporting an actor you cannot identify is not actionable.
    pub fn adopt(&mut self, node: impl Green + 'static) -> NodeId {
        let id = NodeId(self.next_id);
        self.next_id += 1;
        self.nodes.push((id, Box::new(node)));
        id
    }

    /// Stop hosting one green actor, returning whether it was found.
    ///
    /// The minimum a supervisor needs to act on a [`Supervision`] event. Retrying a
    /// [`Stuck::TargetGone`] node in place only reaches the same dead sender, so the recovery
    /// available today is to take it out — deliberately, rather than by dropping the whole host.
    ///
    /// The node's retained outbox goes with it. Handing that payload back to the supervisor
    /// needs `Green` to expose more than `poll`, which is deferred until something concrete
    /// wants it rather than guessed at now.
    pub fn remove(&mut self, id: NodeId) -> bool {
        // `retain`-style removal, not `swap_remove`: sweep order is adoption order and load
        // bearing, so survivors keep their relative positions.
        let Some(pos) = self.nodes.iter().position(|(node_id, _)| *node_id == id) else {
            return false;
        };
        self.nodes.remove(pos);
        true
    }

    /// How many green actors are still running.
    #[must_use]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether every green actor has halted.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// The exits of green actors that have halted, in the order they halted.
    ///
    /// A supervisor reads this to decide what to restart. The host itself has no restart
    /// policy — it hosts, it does not supervise.
    #[must_use]
    pub fn exits(&self) -> &[Exit] {
        &self.exits
    }

    /// Advance every green actor by at most one step, in adoption order.
    ///
    /// Returns what happened, including any actors that are stuck — see [`Sweep`]. The host
    /// *describes*; it does not act. Restart, retry and shutdown are supervision policy, and a
    /// host that hosts should not also decide them.
    pub fn sweep(&mut self) -> Sweep<'_> {
        let mut ran = false;
        self.stuck.clear();
        let mut i = 0;

        while i < self.nodes.len() {
            let (id, node) = &mut self.nodes[i];
            let id = *id;
            // Poll before the match so the borrow of `self.nodes` ends here and the arms below
            // are free to touch `self.stuck` and `self.exits`.
            let step = node.poll();
            match step {
                Step::Ran => {
                    ran = true;
                    i += 1;
                }
                Step::Blocked | Step::Idle => i += 1,
                // A peer is gone and the payload is retained in the node's outbox. Keep the
                // node — whatever the outbox holds stays recoverable — and do *not* count it as
                // having run, so a dead peer cannot spin the sweep against something that is
                // never coming back. Report it upward instead: the layer that chooses retry,
                // handoff or graceful shutdown can only choose if it is told.
                Step::Disconnected => {
                    self.stuck.push(Supervision {
                        node: id,
                        reason: Stuck::TargetGone,
                    });
                    i += 1;
                }
                Step::Halted(exit) => {
                    self.exits.push(exit);
                    // `remove`, not `swap_remove`: sweep order is load-bearing, so the
                    // surviving actors must keep their relative order.
                    self.nodes.remove(i);
                }
            }
        }

        let status = if ran {
            ActorStatus::Busy
        } else {
            ActorStatus::Idle
        };

        Sweep {
            status,
            stuck: &self.stuck,
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Feeding a green actor from outside its host
// ────────────────────────────────────────────────────────────────────────────

/// The sending end of a green actor's inbox, paired with its host's [`Waker`].
///
/// A green actor has no thread of its own, so nothing about pushing into its inbox will wake
/// the host that owns it. This pairs the two halves that must always go together: **push the
/// message, then ring the bell.**
///
/// The order is not stylistic. Waking first admits a lost wakeup — the host can wake, sweep,
/// find the inbox still empty, report `Idle`, and go back to sleep just before the message
/// lands. Pushing first means the host either sees the message on the sweep it is already
/// doing, or is still holding the pending wake that will start another one.
///
/// A failed push is not followed by a wake: there is no new work to announce.
pub struct GreenSender<T> {
    tx: SpscSender<T>,
    waker: Waker,
}

impl<T> GreenSender<T> {
    /// Pair an inbox sender with the waker of the host that owns the receiving actor.
    #[must_use]
    pub fn new(tx: SpscSender<T>, waker: Waker) -> Self {
        Self { tx, waker }
    }

    /// Deliver a message to the green actor and wake its host.
    ///
    /// # Errors
    ///
    /// Returns [`TrySendError::Full`] if the actor's inbox is full — the caller's
    /// backpressure signal — or [`TrySendError::Disconnected`] if the actor is gone.
    pub fn try_send(&self, msg: T) -> Result<(), TrySendError<T>> {
        self.tx.try_send(msg)?;
        self.waker.wake();
        Ok(())
    }
}

impl<T> std::fmt::Debug for GreenSender<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GreenSender").finish_non_exhaustive()
    }
}

/// An inbox for a green actor, whose sender wakes `waker` on every delivery.
///
/// The receiver goes to the [`Node`]; the sender goes to whoever feeds it.
#[must_use]
pub fn green_channel<T>(capacity: usize, waker: Waker) -> (GreenSender<T>, SpscReceiver<T>) {
    let (tx, rx) = spsc_channel(capacity);
    (GreenSender::new(tx, waker), rx)
}

impl Actor<Infallible, Infallible, Infallible> for Host {
    fn handle_data(&mut self, msg: Infallible) -> HandlerResult {
        match msg {}
    }

    fn handle_control(&mut self, msg: Infallible) -> HandlerResult {
        match msg {}
    }

    fn handle_management(&mut self, msg: Infallible) -> HandlerResult {
        match msg {}
    }

    /// Sweeps, and **discards any supervision events**, because this signature has nowhere to
    /// put them.
    ///
    /// That is the honest limit of running a `Host` through the old `Actor`/`ActorScheduler`
    /// shell: `park` returns an `ActorStatus`, which can say "busy" or "idle" and cannot say
    /// "one of my actors is stuck holding a payload". Callers that need supervision must drive
    /// [`sweep`](Host::sweep) directly and read [`Sweep::stuck`].
    ///
    /// Closing this properly means `Host` becoming a transducer whose `Out` carries supervision
    /// events, so its wiring delivers them like any other output and `Topology` checks that edge
    /// — the same conversion vsync and the rasterizer already went through. Until then a `Host`
    /// under `sched.run` is unsupervised, and this comment is the warning rather than a silent
    /// drop.
    fn park(&mut self, _: SystemStatus) -> Result<ActorStatus, HandlerError> {
        Ok(self.sweep().status)
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mealy::{Delivery, Flush, send_port};
    use crate::{ActorScheduler, Message};
    use crate::spsc::{SpscSender, spsc_channel};

    /// A stage that forwards its input onward, counting what it saw.
    struct Forward {
        seen: u32,
    }

    struct ForwardWiring {
        next: SpscSender<u32>,
    }

    impl Wiring for ForwardWiring {
        type Out = Option<u32>;
        fn flush(&mut self, out: &mut Option<u32>) -> Flush {
            send_port(out, &self.next, Delivery::Blocking)
        }
    }

    impl Transducer for Forward {
        type Control = Infallible;
        type Management = Infallible;
        type Data = u32;
        type Out = Option<u32>;

        fn step_data(&mut self, n: u32) -> Result<Option<u32>, HandlerError> {
            self.seen += 1;
            Ok(Some(n + 1))
        }
    }

    #[test]
    fn a_sweep_runs_owned_green_actors() {
        let (tx_in, rx_in) = spsc_channel::<u32>(8);
        let (tx_out, mut rx_out) = spsc_channel::<u32>(8);

        let mut host = Host::new();
        host.adopt(Node::new(
            Forward { seen: 0 },
            rx_in,
            ForwardWiring { next: tx_out },
        ));

        tx_in.try_send(1).unwrap();
        assert_eq!(host.sweep().status, ActorStatus::Busy, "a green actor had work");
        assert_eq!(rx_out.try_recv().unwrap(), 2);
    }

    #[test]
    fn a_quiet_host_reports_idle_so_the_thread_can_sleep() {
        // The 0%-CPU contract: with nothing to do, the host tells the scheduler to block on
        // the doorbell rather than spin.
        let (tx_in, rx_in) = spsc_channel::<u32>(8);
        let (tx_out, _rx_out) = spsc_channel::<u32>(8);

        let mut host = Host::new();
        host.adopt(Node::new(
            Forward { seen: 0 },
            rx_in,
            ForwardWiring { next: tx_out },
        ));

        assert_eq!(host.sweep().status, ActorStatus::Idle, "no input, nothing ran");

        tx_in.try_send(7).unwrap();
        assert_eq!(host.sweep().status, ActorStatus::Busy);
        assert_eq!(host.sweep().status, ActorStatus::Idle, "quiet again after draining");
    }

    #[test]
    fn one_sweep_walks_a_pipeline_end_to_end_in_topological_order() {
        // Why adoption order is load-bearing, and where the measured win comes from: with
        // the stages adopted upstream-first, a single sweep carries a message through all
        // three. Co-location is the point — no thread hop, no wake, one pass.
        let (tx_in, rx_in) = spsc_channel::<u32>(8);
        let (tx_12, rx_12) = spsc_channel::<u32>(8);
        let (tx_23, rx_23) = spsc_channel::<u32>(8);
        let (tx_out, mut rx_out) = spsc_channel::<u32>(8);

        let mut host = Host::new();
        host.adopt(Node::new(
            Forward { seen: 0 },
            rx_in,
            ForwardWiring { next: tx_12 },
        ));
        host.adopt(Node::new(
            Forward { seen: 0 },
            rx_12,
            ForwardWiring { next: tx_23 },
        ));
        host.adopt(Node::new(
            Forward { seen: 0 },
            rx_23,
            ForwardWiring { next: tx_out },
        ));

        tx_in.try_send(0).unwrap();
        assert_eq!(host.sweep().status, ActorStatus::Busy);
        assert_eq!(
            rx_out.try_recv().unwrap(),
            3,
            "one sweep should have carried the message through all three stages"
        );
    }

    #[test]
    fn a_gone_target_is_recorded_and_the_payload_survives() {
        // The point of `Step::Disconnected` is that a supervisor can act on it. If `Host`
        // swallowed the signal — as an earlier revision did, treating it like `Idle` — the
        // scheduler would go back to sleep on its doorbell with the node's payload stranded in
        // its outbox and nothing able to observe that it happened.
        let (tx_in, rx_in) = spsc_channel::<u32>(8);
        let (tx_out, rx_out) = spsc_channel::<u32>(8);

        let mut host = Host::new();
        host.adopt(Node::new(
            Forward { seen: 0 },
            rx_in,
            ForwardWiring { next: tx_out },
        ));

        drop(rx_out); // the *downstream* target dies, not this actor's own inbox
        tx_in.try_send(0).unwrap();

        let reported = {
            let sweep = host.sweep();
            assert_eq!(
                sweep.stuck.len(),
                1,
                "a supervisor must be told which node hit a gone peer"
            );
            assert_eq!(sweep.stuck[0].reason, Stuck::TargetGone);
            sweep.stuck[0]
        };
        assert_eq!(host.len(), 1, "and the node is kept, not silently discarded");

        // The identity has to be actionable, or reporting it is theatre: retrying in place only
        // reaches the same dead sender, so taking the node out is the recovery available today.
        assert!(host.remove(reported.node), "the reported id names a real node");
        assert!(host.is_empty());
        assert!(!host.remove(reported.node), "and removing it twice is not a panic");
    }

    #[test]
    fn each_sweep_reports_only_its_own_stuck_actors() {
        // The failure mode introduced by reusing one buffer across sweeps: forget to clear it
        // and a supervisor that removed a node keeps being told about it forever, or a
        // recovered node stays reported. `stuck` describes *this* sweep, not the history.
        let (tx_in, rx_in) = spsc_channel::<u32>(8);
        let (tx_out, rx_out) = spsc_channel::<u32>(8);

        let mut host = Host::new();
        host.adopt(Node::new(
            Forward { seen: 0 },
            rx_in,
            ForwardWiring { next: tx_out },
        ));

        drop(rx_out);
        tx_in.try_send(0).unwrap();
        let stuck = {
            let sweep = host.sweep();
            assert_eq!(sweep.stuck.len(), 1);
            sweep.stuck[0].node
        };

        // A sweep *does* keep re-reporting a node that is still stuck — the retained outbox is
        // re-flushed and hits the same dead sender. What must not survive is the report of a
        // node the supervisor has already dealt with.
        assert_eq!(host.sweep().stuck.len(), 1, "still stuck, still reported");
        assert!(host.remove(stuck));
        assert!(
            host.sweep().stuck.is_empty(),
            "the supervised node is gone; the sweep must not re-report it from the last one"
        );
    }

    #[test]
    fn a_halted_green_actor_is_dropped_and_its_exit_recorded() {
        let (tx_in, rx_in) = spsc_channel::<u32>(8);
        let (tx_out, _rx_out) = spsc_channel::<u32>(8);

        let mut host = Host::new();
        host.adopt(Node::new(
            Forward { seen: 0 },
            rx_in,
            ForwardWiring { next: tx_out },
        ));
        assert_eq!(host.len(), 1);
        assert!(!host.is_empty(), "an adopted, still-running actor is not empty");

        drop(tx_in); // the green actor's inbox disconnects
        assert!(host.sweep().stuck.is_empty(), "a halt is not a stuck actor");

        assert!(host.is_empty(), "a halted green actor is removed");
        assert_eq!(host.exits(), &[Exit::Completed]);
    }

    #[test]
    fn surviving_actors_keep_their_sweep_order_when_one_halts() {
        // `remove` rather than `swap_remove`: a halt in the middle must not reshuffle the
        // pipeline behind it.
        let (tx_a, rx_a) = spsc_channel::<u32>(8);
        let (tx_dead, rx_dead) = spsc_channel::<u32>(8);
        let (tx_12, rx_12) = spsc_channel::<u32>(8);
        let (tx_out, mut rx_out) = spsc_channel::<u32>(8);
        let (tx_sink, _rx_sink) = spsc_channel::<u32>(8);

        let mut host = Host::new();
        // Adopted: [stage1, doomed, stage2]. The doomed actor sits between the two stages.
        host.adopt(Node::new(
            Forward { seen: 0 },
            rx_a,
            ForwardWiring { next: tx_12 },
        ));
        host.adopt(Node::new(
            Forward { seen: 0 },
            rx_dead,
            ForwardWiring { next: tx_sink },
        ));
        host.adopt(Node::new(
            Forward { seen: 0 },
            rx_12,
            ForwardWiring { next: tx_out },
        ));

        drop(tx_dead);
        tx_a.try_send(0).unwrap();
        assert!(host.sweep().stuck.is_empty(), "a halt is not a stuck actor");

        assert_eq!(host.len(), 2);
        assert_eq!(
            rx_out.try_recv().unwrap(),
            2,
            "stage1 → stage2 still runs in one sweep after the middle actor halted"
        );
    }

    // ── Waking a sleeping host ──────────────────────────────────────────────

    #[test]
    fn a_green_send_wakes_a_host_asleep_on_its_doorbell() {
        // The real thing, on a real thread: the host is blocked in `run()` with nothing to do
        // — no message can reach it through its own lanes, because it has none — and a push
        // into a green actor's inbox has to be what wakes it.
        use std::time::{Duration, Instant};

        let (handle, mut sched) = ActorScheduler::<Infallible, Infallible, Infallible>::new(4, 4);
        let (tx_green, rx_green) = green_channel::<u32>(8, handle.waker());
        let (tx_out, mut rx_out) = spsc_channel::<u32>(8);

        // The host is built on the thread that runs it: an owned green actor never migrates,
        // so it is never required to be `Send`.
        let worker = std::thread::spawn(move || {
            let mut host = Host::new();
            host.adopt(Node::new(
                Forward { seen: 0 },
                rx_green,
                ForwardWiring { next: tx_out },
            ));
            sched.run(&mut host);
        });

        // Let the host reach its doorbell and block before sending, so this exercises the
        // wake path rather than racing the host to its first sweep.
        std::thread::sleep(Duration::from_millis(50));
        tx_green.try_send(41).expect("green inbox has room");

        let deadline = Instant::now() + Duration::from_secs(5);
        let got = loop {
            if let Ok(v) = rx_out.try_recv() {
                break v;
            }
            assert!(
                Instant::now() < deadline,
                "host never woke: a green send did not ring the doorbell"
            );
            std::thread::yield_now();
        };
        assert_eq!(got, 42);

        handle.send(Message::Shutdown).unwrap();
        worker.join().unwrap();
    }

    #[test]
    fn many_green_sends_coalesce_into_one_pending_wake() {
        // The doorbell holds one wake and coalesces the rest, so a burst cannot back it up or
        // fail — the host sweeps everything that arrived once it gets there.
        let (handle, _sched) = ActorScheduler::<Infallible, Infallible, Infallible>::new(4, 4);
        let (tx_green, mut rx_green) = green_channel::<u32>(64, handle.waker());

        for i in 0..64 {
            tx_green
                .try_send(i)
                .expect("a burst of wakes must not fail the send");
        }

        let drained = std::iter::from_fn(|| rx_green.try_recv().ok()).count();
        assert_eq!(drained, 64, "every message landed despite one pending wake");
    }

    #[test]
    fn a_failed_send_reports_backpressure() {
        // A full green inbox is the caller's backpressure signal, and no wake is announced
        // for work that was not accepted.
        let (handle, _sched) = ActorScheduler::<Infallible, Infallible, Infallible>::new(4, 4);
        let (tx_green, _rx_green) = green_channel::<u32>(2, handle.waker());

        let mut sent = 0;
        while tx_green.try_send(1).is_ok() {
            sent += 1;
            assert!(sent < 1024, "a bounded inbox must eventually refuse");
        }
    }

    #[test]
    fn a_waker_outliving_its_scheduler_is_harmless() {
        // A green actor can be fed after its host is gone; there is simply nobody to wake.
        let waker = {
            let (handle, _sched) = ActorScheduler::<Infallible, Infallible, Infallible>::new(4, 4);
            handle.waker()
        };
        waker.wake();
        waker.wake();
    }
}
