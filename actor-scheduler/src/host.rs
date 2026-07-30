//! Hosting green actors on an OS-thread actor.
//!
//! The two tiers compose by self-hosting: a [`Host`] owns a set of green actors ([`Node`]s) and
//! advances them a step at a time. There is no second runtime — the green tier is a thing an
//! actor *does*.
//!
//! # Two ways to drive one
//!
//! A host is an ordinary [`Actor`], so an ordinary [`ActorScheduler`](crate::ActorScheduler) runs
//! one on a thread and the tiers nest with no second runtime — `handle_os` sweeps, and reports
//! only `Busy`/`Idle`. It is *also* a [`Transducer`], so a host can be a green actor inside
//! another host, driven by [`GreenThread`] on a thread of its own. Same sweep either way; a
//! sweep that finds a green actor's flush target gone does not report it upward for someone else
//! to decide what to do — it panics, right there in [`Node`] (`mealy.rs`), the instant the flush
//! discovers it. There is no supervisor to hand the finding to, so failing fast at the point of
//! discovery is the whole policy.
//!
//! The sleep behaviour is what both driving styles share: a sweep in which something ran keeps
//! being stepped without blocking; one where nothing ran reports [`Idle`](Step::Idle) and the
//! driver may block. **A host with nothing to do sleeps instead of polling either way.**
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

use crate::mealy::{Flush, Inbox, Node, Step, Transducer, Wiring};
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
/// `Default` — no continuation — is the quiet sweep, which is the common case and the one that
/// lets the driving thread sleep.
#[derive(Debug, Default)]
pub struct HostOut {
    /// The self-addressed continuation: set when this step left work behind (a green actor
    /// ran, so another sweep may find more). Absent means genuinely quiet, which is what tells
    /// the driver it may block.
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
///
/// A step that cannot deliver panics inside `Node` (`mealy.rs`) the instant its flush discovers
/// a gone target — a `Host` has no policy for that beyond hosting, and nowhere to escalate it.
#[derive(Default)]
pub struct Host {
    nodes: Vec<Box<dyn Green>>,
}

impl Host {
    /// A host with no green actors yet.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Take ownership of a green actor. Sweep order is adoption order.
    pub fn adopt(&mut self, node: impl Green + 'static) {
        self.nodes.push(Box::new(node));
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

    /// Advance every green actor by at most one step, in adoption order.
    ///
    /// Returns `Busy` if any green actor ran, so the caller keeps sweeping; `Idle` when they
    /// are all quiet and the thread may sleep. A step that cannot deliver panics inside `Node`
    /// (`mealy.rs`) the instant its flush discovers a gone target — there is nothing for this
    /// sweep to report upward, because there is no policy beyond hosting.
    pub fn sweep(&mut self) -> ActorStatus {
        let mut ran = false;
        let mut i = 0;

        while i < self.nodes.len() {
            let node = &mut self.nodes[i];
            match node.poll() {
                Step::Ran => {
                    ran = true;
                    i += 1;
                }
                Step::Blocked | Step::Idle => i += 1,
                Step::Halted => {
                    // `remove`, not `swap_remove`: sweep order is load-bearing, so the
                    // surviving actors must keep their relative order.
                    self.nodes.remove(i);
                }
            }
        }

        if ran {
            ActorStatus::Busy
        } else {
            ActorStatus::Idle
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
    /// `pub(crate)`, not public: **the scheduler owns send policy, a caller names a target.**
    /// `mealy::send_port` (via [`mealy::PortTarget`](crate::mealy::PortTarget)) is the one public
    /// path to a `GreenSender` from a `Wiring::flush` — it is what parks the payload on refusal
    /// rather than leaving the choice to whoever called `try_send` directly. Flush runs inside a
    /// green actor's own step, which may not block, so this try-once-and-put-back behavior is
    /// exactly what a flush needs and nothing else should reach for: a handler pushing into
    /// another green actor's inbox from outside a flush should use [`Self::send`] instead, whose
    /// bounded backoff and loud timeout are the wanted posture everywhere else.
    ///
    /// # Errors
    ///
    /// Returns [`TrySendError::Full`] if the actor's inbox is full — the caller's
    /// backpressure signal — or [`TrySendError::Disconnected`] if the actor is gone.
    pub(crate) fn try_send(&self, msg: T) -> Result<(), TrySendError<T>> {
        self.tx.try_send(msg)?;
        self.waker.wake();
        Ok(())
    }
}

/// The port sender's view of a `GreenSender`: `mealy::send_port` calls this to try once and
/// hand the payload back on refusal, exactly as it does for an `SpscSender` or `ActorHandle`.
impl<T> crate::mealy::PortTarget<T> for GreenSender<T> {
    fn try_deliver(&self, msg: T) -> Result<(), TrySendError<T>> {
        self.try_send(msg)
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

/// The host's own wiring. `HostOut` carries only the self-addressed continuation, which
/// `Node` always lifts out before a flush runs (see [`Node::poll`]), so there is nothing left
/// for this to ever actually deliver — it exists only because [`Node`] needs a [`Wiring`] to be
/// generic over.
struct NoDelivery;

impl Wiring for NoDelivery {
    type Out = HostOut;

    fn flush(&mut self, out: &mut HostOut) -> Flush {
        debug_assert!(
            out.again.is_none(),
            "the continuation must never reach wiring"
        );
        Flush::Done
    }
}

/// Runs a [`Host`] on an OS thread under the ordinary [`ActorScheduler`](crate::ActorScheduler).
///
/// The green tier needs one thing from the OS tier that it cannot supply itself: something has
/// to notice that the doorbell rang and turn that into an input. That is `handle_os`'s actual job —
/// bridging the outside world to a message — and it is all this does. It feeds one [`RunSweep`],
/// advances the host until quiet, and reports whether the thread may sleep.
pub struct GreenThread {
    node: Node<Host, NoDelivery, SpscReceiver<RunSweep>>,
    /// Feeds the sweep that a doorbell wake implies. Capacity one: a sweep already queued is as
    /// good as a fresh one, since [`RunSweep`] carries nothing to distinguish them.
    tick: SpscSender<RunSweep>,
}

impl GreenThread {
    /// Make a populated host runnable on a thread.
    ///
    /// Takes the host already populated: green actors are adopted before this point, on the
    /// thread that will run them, which is the same "built where it runs" constraint that lets a
    /// hosted actor be non-`Send`.
    #[must_use]
    pub fn new(host: Host) -> Self {
        let (tick, rx) = spsc_channel(1);
        Self {
            node: Node::new(host, rx, NoDelivery),
            tick,
        }
    }
}

impl Actor<Infallible, Infallible, Infallible> for GreenThread {
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
                Step::Idle | Step::Halted => return Ok(ActorStatus::Idle),
                // Unreachable in practice — this wiring never blocks or panics on its own — but
                // Step is shared with the lane-driven path, so the match stays exhaustive.
                Step::Blocked => return Ok(ActorStatus::Busy),
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
    /// tiers nest without a second runtime.
    fn handle_os(&mut self, _: SystemStatus) -> Result<ActorStatus, HandlerError> {
        Ok(self.sweep())
    }
}

impl Transducer for Host {
    /// No control lane: a host has no time-critical input of its own. Shutdown reaches it by
    /// dropping it, and its green actors by their own lanes.
    type Control = Infallible;
    /// No management lane. Adoption is a direct call on the host, made by whoever built it, not
    /// a message — a host is constructed where it runs.
    type Management = Infallible;
    type Data = RunSweep;
    type Out = HostOut;

    /// Sweep, and ask to be swept again if anything ran.
    fn step_data(&mut self, _: RunSweep) -> Result<HostOut, HandlerError> {
        let status = self.sweep();
        let again = (status == ActorStatus::Busy).then_some(RunSweep);
        Ok(HostOut { again })
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
    use crate::mealy::{Flush, send_port};
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
            send_port(out, &self.next)
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
        assert_eq!(host.sweep(), ActorStatus::Busy, "a green actor had work");
        assert_eq!(rx_out.try_recv().unwrap(), 2);
    }

    /// A step whose output parks still counts as having run — `Node::finish_step` reports
    /// `Ran` for it — so the host stays `Busy` and keeps sweeping until the retry can flush.
    /// If the park were reported as `Blocked` on the same sweep that consumed the input, this
    /// host would go `Idle` over an actor holding an undelivered word, and nothing external
    /// would wake it to retry.
    #[test]
    fn a_parked_first_output_keeps_the_host_busy() {
        let (tx_in, rx_in) = spsc_channel::<u32>(8);
        let (tx_out, mut rx_out) = spsc_channel::<u32>(2);
        while tx_out.try_send(99).is_ok() {}

        let mut host = Host::new();
        host.adopt(Node::new(
            Forward { seen: 0 },
            rx_in,
            ForwardWiring { next: tx_out },
        ));

        tx_in.try_send(1).unwrap();
        assert_eq!(
            host.sweep(),
            ActorStatus::Busy,
            "the input was consumed; the park must not read as an idle host"
        );

        // Once the ring drains, the very next sweep retries the parked flush.
        while rx_out.try_recv().is_ok() {}
        host.sweep();
        assert_eq!(
            rx_out.try_recv().unwrap(),
            2,
            "the parked word was delivered"
        );
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

        assert_eq!(host.sweep(), ActorStatus::Busy);
        assert_eq!(
            rx_out.try_recv().unwrap(),
            2,
            "only the first message is stepped"
        );
        assert!(
            rx_out.try_recv().is_err(),
            "the second message must wait for the next sweep"
        );

        assert_eq!(host.sweep(), ActorStatus::Busy);
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

        assert_eq!(host.sweep(), ActorStatus::Idle, "no input, nothing ran");

        tx_in.try_send(7).unwrap();
        assert_eq!(host.sweep(), ActorStatus::Busy);
        assert_eq!(
            host.sweep(),
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
        assert_eq!(host.sweep(), ActorStatus::Busy);
        assert_eq!(
            rx_out.try_recv().unwrap(),
            3,
            "one sweep should have carried the message through all three stages"
        );
    }

    /// A gone flush target has nobody left to report it to — `Host` hosts, it does not
    /// supervise — so the node panics the moment its flush discovers the target.
    #[test]
    #[should_panic(expected = "wiring target disconnected")]
    fn a_gone_target_panics_the_sweep() {
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

        host.sweep();
    }

    // ────────────────────────────────────────────────────────────────────────
    // The host as a transducer: same sweep, driven as a green node
    // ────────────────────────────────────────────────────────────────────────

    /// A host driven as a green node. `NoDelivery` is the host's own production wiring — its
    /// `Out` has nothing left to deliver — so tests exercise the real thing rather than a stand-in.
    type HostedNode = Node<Host, NoDelivery, SpscReceiver<RunSweep>>;

    /// A host wired as a green node, plus the sweep input a test drives it by.
    fn hosted(host: Host) -> (HostedNode, SpscSender<RunSweep>) {
        let (tx_sweep, rx_sweep) = spsc_channel::<RunSweep>(8);
        let node = Node::new(host, rx_sweep, NoDelivery);
        (node, tx_sweep)
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

        let (mut node, tx_sweep) = hosted(host);
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

    #[test]
    fn a_halted_green_actor_is_dropped() {
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
        host.sweep();

        assert!(host.is_empty(), "a halted green actor is removed");
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
        host.sweep();

        assert_eq!(host.len(), 2);
        assert_eq!(
            rx_out.try_recv().unwrap(),
            2,
            "stage1 → stage2 still runs in one sweep after the middle actor halted"
        );
    }

    // ── Waking a sleeping host ──────────────────────────────────────────────

    /// The same wake, on the wired path. `GreenThread` is what turns a doorbell wake into the
    /// `RunSweep` the transducer needs, keeping the same 0%-CPU behaviour as the plain `Actor`
    /// one.
    #[test]
    fn a_green_send_wakes_a_wired_host_too() {
        use std::time::{Duration, Instant};

        let (handle, mut sched) = ActorScheduler::<Infallible, Infallible, Infallible>::new(4, 4);
        let (tx_green, rx_green) = green_channel::<u32>(8, handle.waker());
        let (tx_out, mut rx_out) = spsc_channel::<u32>(8);

        let worker = std::thread::spawn(move || {
            let mut host = Host::new();
            host.adopt(Node::new(
                Forward { seen: 0 },
                rx_green,
                ForwardWiring { next: tx_out },
            ));
            let mut thread = GreenThread::new(host);
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
