//! Hosting green actors on an OS-thread actor.
//!
//! The two tiers compose by self-hosting: a [`Host`] owns a set of green actors ([`Node`]s) and
//! advances them a step at a time. There is no second runtime — the green tier is a thing an
//! actor *does*.
//!
//! # Three ways to drive one, and when supervision needs the wired one
//!
//! A host is an ordinary [`Actor`], so an ordinary [`ActorScheduler`](crate::ActorScheduler) runs
//! one on a thread and the tiers nest with no second runtime. It is *also* a [`Transducer`], so a
//! host can be a green actor inside another host. Same sweep either way; what differs is where
//! its findings can go.
//!
//! - [`Actor`] impl — `handle_os` sweeps, and reports only `Busy`/`Idle`. Simplest, and enough
//!   whenever the green actors have nowhere to escalate to.
//! - [`Transducer`] impl — [`RunSweep`] on the data lane, [`HostOut`] carrying supervision out a
//!   port that its wiring delivers and [`Topology`](crate::mealy::Topology) can check.
//!   [`GreenThread`] runs this one on a thread.
//! - [`sweep`](Host::sweep) — called directly, findings as a return value.
//!
//! Supervision needs one of the last two, because [`ActorStatus`] is `Busy | Idle` and has no
//! room for "node 3 is stuck holding a framebuffer". The sweep behind `handle_os` computes those
//! events and then drops them, and no amount of improving how the sweep *reports* them changes
//! that — the discard is in the signature. That is a limit of the hook, not a defect in it:
//! `handle_os` exists to invert control flow around an external event source you do not own — an
//! epoll set, a PTY, `XNextEvent`, a Cocoa event queue — and `Busy`/`Idle` is exactly the
//! vocabulary that job needs. Reach for `handle_os` when you *have* to give up the loop; reach for a
//! port when you have something to say.
//!
//! The sleep behaviour survives the move: on the wired path it is the self-addressed
//! continuation, the transducer model's own way of saying "more work". A sweep in which
//! something ran yields [`HostOut::again`] and is stepped straight back; one where nothing ran
//! yields `None`, the node goes [`Idle`](Step::Idle), and the driver may block. **A host with
//! nothing to do sleeps instead of polling either way.**
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
use crate::{
    Actor, ActorStatus, HandlerError, HandlerResult, SchedulerParams, SendError, SystemStatus,
    Waker,
};

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
    /// missed by a caller that never thinks to ask. The slice is only valid until the next
    /// sweep, which is the same window in which acting on it makes sense.
    ///
    /// This is the direct path, for a caller holding the host itself. The [`Transducer`] impl is
    /// the wired one, and emits the same findings a step at a time over a port.
    pub stuck: &'a [Supervision],
}

/// Advance the green actors one step. The [`Host`]'s only input.
///
/// Carries nothing: a sweep is an opportunity to run, not a description of work. It arrives on
/// the data lane either from outside (a doorbell woke the thread) or from the host's own
/// continuation, and those are deliberately the same message — "there may be work" is the same
/// request whoever asks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RunSweep;

/// What a [`Host`] emits from one step.
///
/// `Default` — no supervision, no continuation — is the quiet sweep, which is the common case
/// and the one that lets the driving thread sleep.
#[derive(Debug, Default)]
pub struct HostOut {
    /// A green actor that needs supervision, if one was found.
    ///
    /// One per step rather than a batch: an output word has a single slot per port, and a
    /// sweep that finds several drains them across successive steps under its own continuation.
    /// The alternative — a `Vec` in the slot — would allocate on the hot path for a port that
    /// is empty on almost every sweep.
    pub supervision: Option<Supervision>,
    /// The self-addressed continuation: set when this step left work behind.
    ///
    /// Either a green actor ran (so another sweep may find more) or findings remain to report.
    /// Absent means genuinely quiet, which is what tells the driver it may block.
    pub again: Option<RunSweep>,
}

/// An OS-thread actor that owns green actors and advances them a step at a time.
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
    /// How much of `stuck` the transducer has emitted, since a step emits at most one event.
    ///
    /// Only the [`Transducer`] path uses this; [`sweep`](Host::sweep) hands the whole slice back
    /// at once and has nothing to track. It is what stops a re-sweep from clearing findings that
    /// were never reported.
    reported: usize,
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
    params: SchedulerParams,
}

impl<T> GreenSender<T> {
    /// Pair an inbox sender with the waker of the host that owns the receiving actor.
    #[must_use]
    pub fn new(tx: SpscSender<T>, waker: Waker) -> Self {
        Self::new_with_params(tx, waker, SchedulerParams::DEFAULT)
    }

    /// [`new`](Self::new) with explicit backoff tuning for [`send`](Self::send). Tests use
    /// this to make the timeout short; production senders have no reason to deviate from
    /// [`SchedulerParams::DEFAULT`].
    #[must_use]
    pub fn new_with_params(tx: SpscSender<T>, waker: Waker, params: SchedulerParams) -> Self {
        params.validate();
        Self { tx, waker, params }
    }

    /// Deliver a message, blocking with bounded patience when the inbox is full — the same
    /// spin → yield → exponential-backoff discipline as [`ActorHandle::send`]
    /// (`crate::ActorHandle`), because a full green inbox means the same thing a full lane
    /// ring does: the consumer is behind, and the sender's job is to wait, not to invent a
    /// shedding policy of its own. Delivery policy belongs to the scheduler.
    ///
    /// The patience is bounded: once backoff exceeds `params.max_backoff`, this returns
    /// [`SendError::Timeout`] — a host that unresponsive is wedged, and the mantra is fail
    /// fast, fail loudly, not wait forever or drop silently.
    ///
    /// # Errors
    ///
    /// [`SendError::Timeout`] if the inbox stayed full past the backoff window;
    /// [`SendError::Disconnected`] if the actor is gone.
    pub fn send(&self, msg: T) -> Result<(), SendError> {
        crate::send_with_backoff(&self.tx, msg, &self.params)?;
        self.waker.wake();
        Ok(())
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

/// Runs a [`Host`] on an OS thread under the ordinary [`ActorScheduler`](crate::ActorScheduler).
///
/// The green tier needs one thing from the OS tier that it cannot supply itself: something has
/// to notice that the doorbell rang and turn that into an input. That is `handle_os`'s actual job —
/// bridging the outside world to a message — and it is all this does. It feeds one [`RunSweep`],
/// advances the host until quiet, and reports whether the thread may sleep.
///
/// What it deliberately does **not** do is carry supervision. By the time `handle_os` returns, any
/// event has already left over the host's own wiring, so there is nothing for an
/// `ActorStatus` to fail to express. That is the whole difference from the `impl Actor for Host`
/// this replaces, which computed events inside `handle_os` and dropped them for want of a channel.
pub struct GreenThread<W: Wiring<Out = HostOut>> {
    node: Node<Host, W, SpscReceiver<RunSweep>>,
    /// Feeds the sweep that a doorbell wake implies. Capacity one: a sweep already queued is as
    /// good as a fresh one, since [`RunSweep`] carries nothing to distinguish them.
    tick: SpscSender<RunSweep>,
}

impl<W: Wiring<Out = HostOut>> GreenThread<W> {
    /// Wire a host to its supervision output and make it runnable on a thread.
    ///
    /// Takes the host already populated: green actors are adopted before this point, on the
    /// thread that will run them, which is the same "built where it runs" constraint that lets a
    /// hosted actor be non-`Send`.
    #[must_use]
    pub fn new(host: Host, wiring: W) -> Self {
        let (tick, rx) = spsc_channel(1);
        Self {
            node: Node::new(host, rx, wiring),
            tick,
        }
    }
}

impl<W: Wiring<Out = HostOut>> Actor<Infallible, Infallible, Infallible> for GreenThread<W> {
    fn handle_data(&mut self, msg: Infallible) -> HandlerResult {
        match msg {}
    }

    fn handle_control(&mut self, msg: Infallible) -> HandlerResult {
        match msg {}
    }

    fn handle_management(&mut self, msg: Infallible) -> HandlerResult {
        match msg {}
    }

    fn handle_os(&mut self, _: SystemStatus) -> Result<ActorStatus, HandlerError> {
        match self.tick.try_send(RunSweep) {
            Ok(()) => {}
            // Full: a sweep is already queued, which is what this was going to ask for —
            // `RunSweep` carries nothing to distinguish a second one from the first.
            // Disconnected: unreachable, since this owns both ends of `tick`.
            // Neither is worth failing a handle_os over, and the poll below runs either way.
            Err(TrySendError::Full(_) | TrySendError::Disconnected(_)) => {}
        }

        loop {
            match self.node.poll() {
                // More to do; the host's continuation is driving.
                Step::Ran => continue,
                // Quiet: nothing ran and nothing is queued, so the thread may block.
                Step::Idle | Step::Halted(_) => return Ok(ActorStatus::Idle),
                // The supervision target is full or gone. Neither resolves by sleeping — a
                // blocked port drains when its consumer runs, and a gone one needs the
                // supervisor that is not reading. Stay awake rather than risk a lost wakeup
                // with an event still in the outbox.
                Step::Blocked | Step::Disconnected => return Ok(ActorStatus::Busy),
            }
        }
    }
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

    /// Sweep, and report only whether the thread may sleep.
    ///
    /// This is the plain hierarchical composition: a host is an ordinary [`Actor`], so an
    /// ordinary [`ActorScheduler`](crate::ActorScheduler) can run one on a thread and the two
    /// tiers nest without a second runtime. Keep reaching for it when a host's green actors
    /// have nowhere to escalate to — most do not.
    ///
    /// It cannot carry supervision, and that is a property of the signature rather than an
    /// oversight: `ActorStatus` is `Busy | Idle`, which answers "may this thread sleep?" and has
    /// no room for "node 3 is stuck holding a payload". Findings are therefore dropped here.
    /// When they matter, use the [`Transducer`] impl — the same host, wired, emitting
    /// [`HostOut::supervision`] over a port — via [`GreenThread`] to run it on a thread, or
    /// [`sweep`](Host::sweep) to read them directly. All three are the same sweep.
    fn handle_os(&mut self, _: SystemStatus) -> Result<ActorStatus, HandlerError> {
        Ok(self.sweep().status)
    }
}

impl Transducer for Host {
    /// No control lane: a host has no time-critical input of its own. Shutdown reaches it by
    /// dropping it, and its green actors by their own lanes.
    type Control = Infallible;
    /// No management lane. Adoption and removal are direct calls on the host, made by whoever
    /// built it, not messages — a host is constructed where it runs.
    type Management = Infallible;
    type Data = RunSweep;
    type Out = HostOut;

    /// Report one outstanding finding, or sweep for more.
    ///
    /// Findings come first. A stuck node is stuck until a supervisor acts, so sweeping again
    /// before its event has been delivered cannot improve matters — and [`sweep`](Host::sweep)
    /// clears the buffer, so it would destroy the very report this exists to deliver.
    fn step_data(&mut self, _: RunSweep) -> Result<HostOut, HandlerError> {
        if let Some(&event) = self.stuck.get(self.reported) {
            self.reported += 1;
            return Ok(HostOut {
                supervision: Some(event),
                // More findings, or a sweep still owed once they are drained.
                again: Some(RunSweep),
            });
        }

        let status = self.sweep().status;
        self.reported = 0;

        let supervision = self.stuck.first().copied();
        if supervision.is_some() {
            self.reported = 1;
        }

        // Yield again if this step left work behind: something ran, or findings remain. A node
        // that is merely *stuck* is neither — reporting it once must not spin the host against a
        // peer that is never coming back, which is the same reason `sweep` does not count
        // `Disconnected` as having run.
        let more_findings = self.reported < self.stuck.len();
        let again = (more_findings || status == ActorStatus::Busy).then_some(RunSweep);

        Ok(HostOut { supervision, again })
    }

    fn take_continuation(out: &mut HostOut) -> Option<RunSweep> {
        out.again.take()
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mealy::{Delivery, Flush, send_port};
    use crate::spsc::{SpscSender, spsc_channel};
    use crate::{ActorScheduler, Message};

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
        assert_eq!(
            host.sweep().status,
            ActorStatus::Busy,
            "a green actor had work"
        );
        assert_eq!(rx_out.try_recv().unwrap(), 2);
    }

    #[test]
    fn a_sweep_advances_at_most_one_step_per_actor() {
        // The doc contract on `Host::sweep`: "advance every green actor by at most one step".
        // Queue two messages for a single actor and take one sweep — only the first may be
        // forwarded, leaving the second for the next sweep rather than draining the actor in
        // one call.
        let (tx_in, rx_in) = spsc_channel::<u32>(8);
        let (tx_out, mut rx_out) = spsc_channel::<u32>(8);

        let mut host = Host::new();
        host.adopt(Node::new(
            Forward { seen: 0 },
            rx_in,
            ForwardWiring { next: tx_out },
        ));
        assert!(
            !host.is_empty(),
            "a host with an adopted actor is not empty"
        );

        tx_in.try_send(1).unwrap();
        tx_in.try_send(2).unwrap();

        assert_eq!(host.sweep().status, ActorStatus::Busy);
        assert_eq!(
            rx_out.try_recv().unwrap(),
            2,
            "only the first message is stepped"
        );
        assert!(
            rx_out.try_recv().is_err(),
            "the second message must wait for the next sweep"
        );

        assert_eq!(host.sweep().status, ActorStatus::Busy);
        assert_eq!(
            rx_out.try_recv().unwrap(),
            3,
            "the second message steps on the next sweep"
        );
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

        assert_eq!(
            host.sweep().status,
            ActorStatus::Idle,
            "no input, nothing ran"
        );

        tx_in.try_send(7).unwrap();
        assert_eq!(host.sweep().status, ActorStatus::Busy);
        assert_eq!(
            host.sweep().status,
            ActorStatus::Idle,
            "quiet again after draining"
        );
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
        assert_eq!(
            host.len(),
            1,
            "and the node is kept, not silently discarded"
        );

        // The identity has to be actionable, or reporting it is theatre: retrying in place only
        // reaches the same dead sender, so taking the node out is the recovery available today.
        assert!(
            host.remove(reported.node),
            "the reported id names a real node"
        );
        assert!(host.is_empty());
        assert!(
            !host.remove(reported.node),
            "and removing it twice is not a panic"
        );
    }

    // ────────────────────────────────────────────────────────────────────────
    // The host as a transducer: supervision over an edge, not through `handle_os`
    // ────────────────────────────────────────────────────────────────────────

    /// Delivers a host's supervision events onward, which is the whole point of the conversion:
    /// under the old `handle_os` they were computed and dropped.
    struct SupervisionWiring {
        supervisor: SpscSender<Supervision>,
    }

    impl Wiring for SupervisionWiring {
        type Out = HostOut;
        fn flush(&mut self, out: &mut HostOut) -> Flush {
            send_port(&mut out.supervision, &self.supervisor, Delivery::Blocking)
        }
    }

    /// A host driven as a green node, wired to a supervision port.
    type HostedNode = Node<Host, SupervisionWiring, SpscReceiver<RunSweep>>;

    /// A host wired as a green node, plus the ends a test drives it by: the sweep input and the
    /// supervision output.
    fn hosted(host: Host) -> (HostedNode, SpscSender<RunSweep>, SpscReceiver<Supervision>) {
        let (tx_sweep, rx_sweep) = spsc_channel::<RunSweep>(8);
        let (tx_sup, rx_sup) = spsc_channel::<Supervision>(8);
        let node = Node::new(host, rx_sweep, SupervisionWiring { supervisor: tx_sup });
        (node, tx_sweep, rx_sup)
    }

    /// The conversion's reason for existing. Under `handle_os` this event was computed and discarded
    /// because `Result<ActorStatus, _>` had nowhere to put it; here it arrives at a supervisor
    /// as an ordinary message.
    #[test]
    fn a_stuck_actor_reaches_a_supervisor_over_a_port() {
        let (tx_in, rx_in) = spsc_channel::<u32>(8);
        let (tx_out, rx_out) = spsc_channel::<u32>(8);

        let mut host = Host::new();
        let stuck_node = host.adopt(Node::new(
            Forward { seen: 0 },
            rx_in,
            ForwardWiring { next: tx_out },
        ));

        drop(rx_out); // the downstream target dies
        tx_in.try_send(0).unwrap();

        let (mut node, tx_sweep, mut supervisor) = hosted(host);
        tx_sweep.try_send(RunSweep).unwrap();

        assert_eq!(node.poll(), Step::Ran);
        let event = supervisor
            .try_recv()
            .expect("the supervision event must be delivered, not discarded");
        assert_eq!(event.node, stuck_node);
        assert_eq!(event.reason, Stuck::TargetGone);
    }

    /// The sleep contract `handle_os` used to provide, in the transducer's own vocabulary: a sweep
    /// that ran something yields a continuation and is stepped straight back; a quiet one does
    /// not, so the node reports `Idle` and the driving thread may block.
    #[test]
    fn a_busy_sweep_yields_a_continuation_and_a_quiet_one_does_not() {
        let (tx_in, rx_in) = spsc_channel::<u32>(8);
        let (tx_out, mut rx_out) = spsc_channel::<u32>(8);

        let mut host = Host::new();
        host.adopt(Node::new(
            Forward { seen: 0 },
            rx_in,
            ForwardWiring { next: tx_out },
        ));

        let (mut node, tx_sweep, _supervisor) = hosted(host);
        tx_in.try_send(1).unwrap();
        tx_sweep.try_send(RunSweep).unwrap();

        // The green actor ran, so the host asks to be stepped again without a new message.
        assert_eq!(node.poll(), Step::Ran);
        assert_eq!(rx_out.try_recv().unwrap(), 2);

        // That continuation is consumed here, finds nothing left to do, and stops.
        assert_eq!(
            node.poll(),
            Step::Ran,
            "the continuation is a step of its own"
        );
        assert_eq!(
            node.poll(),
            Step::Idle,
            "nothing ran and nothing is queued, so the thread may sleep"
        );
    }

    /// A stuck node must not spin the host. Reporting it is not "work done" — the same reason
    /// `sweep` does not count `Disconnected` as having run — so once the event is out, the host
    /// goes quiet even though the node is still stuck.
    #[test]
    fn reporting_a_stuck_actor_does_not_spin_the_host() {
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

        let (mut node, tx_sweep, mut supervisor) = hosted(host);
        tx_sweep.try_send(RunSweep).unwrap();

        assert_eq!(node.poll(), Step::Ran);
        assert!(supervisor.try_recv().is_ok(), "reported once");

        // The continuation from that step re-sweeps, finds the same stuck node, and reports it
        // again — but then stops, rather than yielding forever against a peer that is not
        // coming back.
        node.poll();
        let mut further = 0;
        for _ in 0..8 {
            if node.poll() == Step::Idle {
                break;
            }
            further += 1;
        }
        assert!(
            further < 8,
            "a permanently stuck node must not keep the host awake forever"
        );
    }

    /// Several stuck actors exceed the one supervision slot an output word has, so they drain
    /// across successive steps under the host's own continuation rather than being batched into
    /// an allocation on the hot path.
    #[test]
    fn multiple_stuck_actors_drain_one_per_step() {
        let mut host = Host::new();
        let mut ids = Vec::new();
        let mut senders = Vec::new();
        for _ in 0..3 {
            let (tx_in, rx_in) = spsc_channel::<u32>(8);
            let (tx_out, rx_out) = spsc_channel::<u32>(8);
            ids.push(host.adopt(Node::new(
                Forward { seen: 0 },
                rx_in,
                ForwardWiring { next: tx_out },
            )));
            drop(rx_out);
            tx_in.try_send(0).unwrap();
            senders.push(tx_in);
        }

        let (mut node, tx_sweep, mut supervisor) = hosted(host);
        tx_sweep.try_send(RunSweep).unwrap();

        let mut seen = Vec::new();
        for _ in 0..6 {
            node.poll();
            while let Ok(event) = supervisor.try_recv() {
                if !seen.contains(&event.node) {
                    seen.push(event.node);
                }
            }
            if seen.len() == ids.len() {
                break;
            }
        }

        seen.sort();
        let mut expected = ids.clone();
        expected.sort();
        assert_eq!(
            seen, expected,
            "every stuck actor must be reported, not just the first one the sweep found"
        );
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

    /// The same wake, on the wired path. `GreenThread` is what turns a doorbell wake into the
    /// `RunSweep` the transducer needs, so a host that reports supervision over a port keeps the
    /// same 0%-CPU behaviour as the plain `Actor` one.
    #[test]
    fn a_green_send_wakes_a_wired_host_too() {
        use std::time::{Duration, Instant};

        let (handle, mut sched) = ActorScheduler::<Infallible, Infallible, Infallible>::new(4, 4);
        let (tx_green, rx_green) = green_channel::<u32>(8, handle.waker());
        let (tx_out, mut rx_out) = spsc_channel::<u32>(8);
        let (tx_sup, _rx_sup) = spsc_channel::<Supervision>(8);

        let worker = std::thread::spawn(move || {
            let mut host = Host::new();
            host.adopt(Node::new(
                Forward { seen: 0 },
                rx_green,
                ForwardWiring { next: tx_out },
            ));
            let mut thread = GreenThread::new(host, SupervisionWiring { supervisor: tx_sup });
            sched.run(&mut thread);
        });

        std::thread::sleep(Duration::from_millis(50));
        tx_green.try_send(41).expect("green inbox has room");

        let deadline = Instant::now() + Duration::from_secs(5);
        let got = loop {
            if let Ok(v) = rx_out.try_recv() {
                break v;
            }
            assert!(
                Instant::now() < deadline,
                "wired host never woke: a green send did not reach a sweep"
            );
            std::thread::yield_now();
        };
        assert_eq!(got, 42);

        handle.send(Message::Shutdown).unwrap();
        worker.join().unwrap();
    }

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

    #[test]
    fn green_sender_debug_names_its_type() {
        let (handle, _sched) = ActorScheduler::<Infallible, Infallible, Infallible>::new(4, 4);
        let (tx_green, _rx_green) = green_channel::<u32>(2, handle.waker());
        assert!(format!("{:?}", tx_green).contains("GreenSender"));
    }
}
