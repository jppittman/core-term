//! Mealy-transducer actors: the handler returns its output instead of sending it.
//!
//! An actor is a Mealy machine — `δ : State × Input → State × Output`. A step takes
//! `&mut self` (the state transition) and one input message, and **returns** the messages it
//! wants to emit. It never calls `send`, never blocks, and always runs to completion, so
//! there is no live stack to suspend: everything that survives between inputs is already in
//! `self`.
//!
//! The scheduler owns all sending. It takes the returned output, delivers each set port to
//! its target, and — when a target is full — keeps the undelivered remainder and declines to
//! step that actor again until the remainder drains. Backpressure is therefore a property of
//! the runtime, invisible to the handler author, and an actor can never outrun its slowest
//! consumer.
//!
//! Design: `docs/designs/actor-scheduler-mealy-transducer.md`.
//!
//! # Status
//!
//! Prototype. This module proves the runtime semantics (stage → partial flush → park →
//! resume, self-port yield, and Control/Management/Data priority) by hand, before the
//! `troupe!`/typed-handle macros are retargeted to generate the port structs. It is additive:
//! the existing [`Actor`](crate::Actor) / [`ActorScheduler`](crate::ActorScheduler) path is
//! untouched.
//!
//! # The three pieces
//!
//! | Piece | Role |
//! |-------|------|
//! | [`Transducer`] | The actor: one pure step per lane, `(state, input) → (state, output)` |
//! | [`Wiring`] | Where the output ports go. Delivers set ports; leaves blocked ones in place |
//! | [`Node`] | Scheduler-side: actor + three lanes + wiring + the parked outbox |
//!
//! Splitting `Transducer` from `Wiring` is what makes the step pure: the actor produces
//! *values* and never holds a sender, so it is table-testable with no scheduler in the loop.
//!
//! # Priority
//!
//! `Node` drains its three lanes in the same Control > Management > Data order, with the same
//! burst-limit ratio, as [`ActorScheduler`](crate::ActorScheduler)'s `handle_wake` — ported,
//! not redesigned, because a green actor that needs input responsiveness under load has
//! exactly the same starvation hazard a dedicated-thread one does. The only difference is
//! granularity: the OS-thread scheduler drains a whole burst per wake; `Node::poll` drains one
//! message per call, so a green actor sharing a [`Host`](crate::Host) with others never holds
//! the worker for longer than one step.
//!
//! # Yielding
//!
//! A port addressed to the actor's own inbox is the yield. Emitting a continuation to
//! yourself and returning lets the scheduler run everyone else before feeding you your own
//! message back. No coroutine, no `async`, no program counter — just a self-edge.

use std::marker::PhantomData;

use crate::HandlerError;
use crate::SchedulerParams;
use crate::lifecycle::Exit;
use crate::spsc::{SpscSender, TryRecvError, TrySendError};
use crate::{ActorStatus, SystemStatus};

// ────────────────────────────────────────────────────────────────────────────
// The actor
// ────────────────────────────────────────────────────────────────────────────

/// A Mealy machine on three priority lanes: one input symbol in, one output word out, state
/// mutated in place.
///
/// It is Mealy rather than Moore because the output is a function of state *and* input —
/// you echo the byte you just received. Moore would only see state.
///
/// The step is pure in the sense that matters: it performs no effects. `Ok(out)` describes
/// what should be emitted; `Err(e)` is a failed transition and emits nothing. "Emit *and*
/// warn" is deliberately not expressible — that is a message variant, not an error.
///
/// # Three lanes, not one
///
/// A step is triggered by whichever lane [`Node::poll`] drains from — Control, Management, or
/// Data — the same three lanes and the same priority ordering (Control > Management > Data)
/// as [`Actor`](crate::Actor)/[`ActorScheduler`](crate::ActorScheduler). Only `step_data` is
/// required; `step_control` and `step_management` default to the silent transition, so a
/// single-lane actor writes `type Control = Infallible; type Management = Infallible;` and one
/// method, exactly as [`Host`](crate::Host) does.
pub trait Transducer {
    /// The control-lane input. Highest priority — set to `Infallible` if unused.
    type Control;
    /// The management-lane input. Middle priority — set to `Infallible` if unused.
    type Management;
    /// The data-lane input. Lowest priority, and the only lane every actor has.
    type Data;

    /// The output word: a struct with one optional, typed slot per downstream port.
    ///
    /// `Default` is the silent transition (all ports unset), which is the common case.
    type Out: Default;

    /// Advance one step from the data lane.
    fn step_data(&mut self, msg: Self::Data) -> Result<Self::Out, HandlerError>;

    /// Advance one step from the control lane. Defaults to the silent transition.
    fn step_control(&mut self, _msg: Self::Control) -> Result<Self::Out, HandlerError> {
        Ok(Self::Out::default())
    }

    /// Advance one step from the management lane. Defaults to the silent transition.
    fn step_management(&mut self, _msg: Self::Management) -> Result<Self::Out, HandlerError> {
        Ok(Self::Out::default())
    }

    /// Take the self-addressed continuation out of an output word, if this machine yields.
    ///
    /// The default is `None`: no self-port, no yielding. A machine that yields overrides this
    /// to hand back its self-port, and the [`Node`] routes it to a dedicated single slot
    /// rather than through [`Wiring`] — see [`Node::poll`] for why that is the only way a
    /// self-edge can be deadlock-free. A continuation always resumes on the data lane: it is
    /// more work, not a control or management signal.
    fn take_continuation(_out: &mut Self::Out) -> Option<Self::Data> {
        None
    }

    /// The OS-bridge hook, in effects-as-values form: called only by a dedicated-thread driver
    /// when the lanes are quiet, never by a [`Host`](crate::Host) — a green actor may not
    /// block, and placement (design doc §5) is exactly the choice of which driver runs you.
    ///
    /// `SystemStatus` reports whether the lanes still hold work (same contract as
    /// [`Actor::handle_os`](crate::Actor::handle_os)). Return the output word to flush plus
    /// [`ActorStatus::Busy`] to be re-polled without blocking, or `Idle` to let the thread
    /// sleep on its doorbell.
    fn step_os(&mut self, _status: SystemStatus) -> Result<(Self::Out, ActorStatus), HandlerError> {
        Ok((Self::Out::default(), ActorStatus::Idle))
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Delivery
// ────────────────────────────────────────────────────────────────────────────

/// Whether an output word was fully delivered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Flush {
    /// Every set port was delivered; nothing remains.
    Done,
    /// At least one port could not be delivered (target full). The undelivered ports are
    /// still set in the output word, which the [`Node`] keeps as its parked outbox.
    Blocked,
    /// At least one target is **gone**. Like [`Blocked`](Flush::Blocked), the undelivered ports
    /// are **retained** in the output word — that is the load-bearing half of this variant.
    ///
    /// Previously a disconnected target was reported as [`Done`](Flush::Done), which consumed
    /// the message. For a port carrying a moved resource — the render pipeline's sole frame
    /// buffer, say — that silently destroyed it, and left the sender believing it had been
    /// delivered. Retaining the payload keeps every recovery open: retry, hand off, escalate,
    /// or shut down gracefully with the resource still in hand.
    ///
    /// This variant deliberately carries no policy. What to do about a dead peer is the
    /// supervisor's call, not the port's; see [`Step::Disconnected`].
    Disconnected,
}

/// Where an actor's output ports go.
///
/// Generated by the macro in the final design; hand-written here. An implementation takes
/// each set port out of the output word and tries to deliver it, **putting it back if the
/// target is full**. Ports are independent, so a partial flush is the natural behavior: the
/// ports that fit go now, the rest wait.
pub trait Wiring {
    /// The output word this wiring knows how to deliver.
    type Out;

    /// Deliver what fits. Clears delivered ports; leaves blocked ports set.
    fn flush(&mut self, out: &mut Self::Out) -> Flush;
}

/// Whether a port's delivery may park its producer.
///
/// The two edge kinds [`Topology`] knows about (§3.1): a cycle among blocking edges is a
/// bootstrap error, but a droppable edge cannot deadlock and so may legally close one. This
/// is the runtime counterpart, and it is also what the macro's `[drop]` port attribute
/// compiles to — one enum, not an accreting family of `send_port`/`send_port_*` functions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Delivery {
    /// Park the producer when the target is full. Propagates backpressure.
    Blocking,
    /// Discard when the target is full. Can never return [`Flush::Blocked`], which is what
    /// makes it legal as the closing edge of a cycle.
    ///
    /// Use it for data that is only worth delivering if still current — "drop this frame if
    /// the display is busy" — never for anything a consumer must not miss. Silence on a full
    /// ring is the whole point, and also the whole risk.
    Droppable,
}

/// Deliver one port according to `delivery`.
///
/// The building block every generated `Wiring::flush` is made of.
///
/// Outcomes:
/// - **Delivered**, or the port was empty → [`Flush::Done`].
/// - **Target full**, [`Delivery::Droppable`] → the message is discarded, [`Flush::Done`].
///   Silence on a full ring is the whole point of that delivery kind, and also its whole risk.
/// - **Target full**, [`Delivery::Blocking`] → the message is put back in `port`,
///   [`Flush::Blocked`]. Retry when the ring drains.
/// - **Target gone** (either delivery kind) → the message is **put back in `port`**,
///   [`Flush::Disconnected`]. It is deliberately *not* dropped: for a port carrying a moved
///   resource, dropping here would destroy it with nobody having decided to, while the sender
///   believed it was delivered. Blocking forever on a dead consumer would deadlock, so this
///   reports rather than parks — the caller decides whether to retry, hand off, or shut down,
///   and can do any of them because the value is still in hand.
pub fn send_port<T>(port: &mut Option<T>, tx: &SpscSender<T>, delivery: Delivery) -> Flush {
    let Some(msg) = port.take() else {
        return Flush::Done;
    };
    match (tx.try_send(msg), delivery) {
        (Ok(()), _) => Flush::Done,
        // The target is gone. Put the message back rather than dropping it: for a port carrying
        // a moved resource, dropping here destroys it with nobody having decided to.
        (Err(TrySendError::Disconnected(msg)), _) => {
            *port = Some(msg);
            Flush::Disconnected
        }
        (Err(TrySendError::Full(_)), Delivery::Droppable) => Flush::Done,
        (Err(TrySendError::Full(msg)), Delivery::Blocking) => {
            *port = Some(msg);
            Flush::Blocked
        }
    }
}

/// Combine port outcomes. `Disconnected` outranks `Blocked` outranks `Done` — a dead peer is
/// news the caller needs even if another port merely filled up, and it does not resolve by
/// waiting the way backpressure does.
///
/// **Consumes the iterator fully; never short-circuits.** A hand-written `Wiring::flush` may
/// pass a lazy iterator of [`send_port`] calls, so returning early would skip the sends that
/// hadn't been evaluated yet. Since a disconnected port stays set in the outbox and is retried
/// from the front every time, those later ports would be starved permanently — which is exactly
/// the independent-port partial flush this module promises.
#[must_use]
pub fn all(outcomes: impl IntoIterator<Item = Flush>) -> Flush {
    let mut worst = Flush::Done;
    for outcome in outcomes {
        worst = match (worst, outcome) {
            (Flush::Disconnected, _) | (_, Flush::Disconnected) => Flush::Disconnected,
            (Flush::Blocked, _) | (_, Flush::Blocked) => Flush::Blocked,
            (Flush::Done, Flush::Done) => Flush::Done,
        };
    }
    worst
}

// ────────────────────────────────────────────────────────────────────────────
// The scheduler's view of an actor
// ────────────────────────────────────────────────────────────────────────────

/// What one [`Node::poll`] accomplished.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Step {
    /// Consumed an input and advanced the machine.
    Ran,
    /// Parked on backpressure: a target is full and the outbox still holds ports. The actor
    /// will not be stepped again until the outbox drains.
    Blocked,
    /// No input available. Nothing to do until a message arrives.
    Idle,
    /// Terminal. The actor is done and should be removed (and possibly restarted).
    Halted(Exit),
    /// A target is gone. The outbox **still holds the undelivered ports**, so whatever they
    /// carry is recoverable.
    ///
    /// Distinct from [`Blocked`](Step::Blocked) because backpressure resolves by waiting and
    /// this does not — retrying a dead peer forever is a hang, and dropping the payload is data
    /// loss. Surfaced rather than handled here on purpose: retry, escalate, or shut down
    /// gracefully is a supervision decision, and the caller can make any of them because the
    /// resource is still in hand.
    Disconnected,
}

/// Pulling one input symbol. Implemented for the SPSC receiver; a trait so tests can drive a
/// node from a plain queue, and so an in-process worker can later use a non-atomic inbox.
pub trait Inbox {
    /// The symbol type.
    type Item;
    /// Take the next symbol, if any.
    fn take(&mut self) -> Result<Self::Item, TryRecvError>;
}

impl<T> Inbox for crate::spsc::SpscReceiver<T> {
    type Item = T;

    fn take(&mut self) -> Result<T, TryRecvError> {
        self.try_recv()
    }
}

/// An inbox with no producer: always reports [`TryRecvError::Disconnected`].
///
/// Stands in for a lane a [`Transducer`] does not use, so [`Node::new`] can wire a data-only
/// actor without a real control or management channel. Disconnected, not merely empty, is the
/// honest state — there is not, and will never be, a sender for this lane — and it is what
/// makes [`Node`]'s halt condition ("every lane disconnected") correct for a single-lane actor:
/// it reduces to exactly "the data lane disconnected," since control/management already report
/// disconnected unconditionally.
pub struct NoLane<T>(PhantomData<T>);

impl<T> NoLane<T> {
    /// A permanently empty lane.
    #[must_use]
    pub fn new() -> Self {
        Self(PhantomData)
    }
}

impl<T> Default for NoLane<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Inbox for NoLane<T> {
    type Item = T;

    fn take(&mut self) -> Result<T, TryRecvError> {
        Err(TryRecvError::Disconnected)
    }
}

/// Which of the three lanes [`Node::poll`] is currently favoring, and for how long, before it
/// rotates to the next.
///
/// Mirrors `ActorScheduler::handle_wake`'s exact cycle — control (half budget) → management
/// (full budget) → control (half budget again) → data (full budget) — ported to run one
/// message at a time across many `poll()` calls instead of batched into one call. Control gets
/// two turns per cycle so that anything arriving while management is favored is still seen
/// promptly, without giving control the whole cycle and starving data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Slot {
    Control1,
    Management,
    Control2,
    Data,
}

impl Slot {
    fn next(self) -> Self {
        match self {
            Slot::Control1 => Slot::Management,
            Slot::Management => Slot::Control2,
            Slot::Control2 => Slot::Data,
            Slot::Data => Slot::Control1,
        }
    }
}

/// The three inbox lanes a lane-aware [`Node`] reads from, bundled so
/// [`Node::new_with_lanes`] takes one argument instead of three.
pub struct Lanes<RC, RM, RD> {
    /// Highest priority.
    pub control: RC,
    /// Middle priority.
    pub management: RM,
    /// Lowest priority, and the only lane every actor has.
    pub data: RD,
}

/// The scheduler side of one actor: the machine, its three lanes, its wiring, and its outbox.
///
/// The outbox is the entire suspension state. Because a step runs to completion, a parked
/// actor's `self` is always consistent — there is no half-finished handler to resume, only
/// undelivered messages to retry. That is why this design needs no coroutine machinery.
///
/// `RC`/`RM` default to [`NoLane`] — a data-only actor is `Node<T, W, RD>`, exactly the call
/// shape every single-lane actor already uses. A lane-aware actor spells out all five
/// parameters (or, in practice, just calls [`Node::new_with_lanes`] and lets inference fill
/// them in).
pub struct Node<
    T: Transducer,
    W: Wiring<Out = T::Out>,
    RD,
    RC = NoLane<<T as Transducer>::Control>,
    RM = NoLane<<T as Transducer>::Management>,
> {
    actor: T,
    control: RC,
    management: RM,
    data: RD,
    wiring: W,
    /// Undelivered ports from the last step. `Some` means parked.
    outbox: Option<T::Out>,
    /// The self-addressed continuation: capacity exactly one, drained before any lane.
    ///
    /// This is the whole of the deadlock argument. A step consumes at most one continuation
    /// and produces at most one (the output word has a single slot per port), and the slot is
    /// emptied immediately before the step that could refill it — so it can never be full
    /// when written, can never block, and can never park. A self-edge routed through a
    /// *queue* instead would deadlock the moment that queue filled: the only actor that can
    /// drain it is the one parked on writing to it.
    continuation: Option<T::Data>,
    /// Steps taken. Lets tests assert that a parked actor is genuinely not stepped.
    steps: u64,
    /// Current position in the control/management/control/data cycle.
    slot: Slot,
    /// Messages taken in the current `slot` visit, toward that slot's limit.
    slot_progress: usize,
    control_half_limit: usize,
    management_limit: usize,
    data_limit: usize,
}

impl<T, W, RD, RC, RM> Node<T, W, RD, RC, RM>
where
    T: Transducer,
    W: Wiring<Out = T::Out>,
    RD: Inbox<Item = T::Data>,
    RC: Inbox<Item = T::Control>,
    RM: Inbox<Item = T::Management>,
{
    /// Wire an actor to all three lanes, its output ports, and burst-limit config.
    ///
    /// Reuses [`SchedulerParams`] verbatim, including its `default_data_burst_limit` — the
    /// same config type, the same `control_burst_limit()`/`management_burst_limit()` the
    /// dedicated-thread scheduler uses — so a lane-aware green actor is tuned exactly like an
    /// OS-thread one, rather than by a second, parallel set of knobs. Override the data limit
    /// the same way any `SchedulerParams` field is overridden: `SchedulerParams { default_data_
    /// burst_limit: n, ..SchedulerParams::DEFAULT }`.
    pub fn new_with_lanes(
        actor: T,
        lanes: Lanes<RC, RM, RD>,
        wiring: W,
        params: SchedulerParams,
    ) -> Self {
        Self {
            actor,
            control: lanes.control,
            management: lanes.management,
            data: lanes.data,
            wiring,
            outbox: None,
            continuation: None,
            steps: 0,
            slot: Slot::Control1,
            slot_progress: 0,
            control_half_limit: (params.control_burst_limit() / 2).max(1),
            management_limit: params.management_burst_limit(),
            data_limit: params.default_data_burst_limit.max(1),
        }
    }

    /// Steps taken so far.
    #[must_use]
    pub fn steps(&self) -> u64 {
        self.steps
    }

    /// The actor, for inspection in tests and for supervisors rebuilding state.
    #[must_use]
    pub fn actor(&self) -> &T {
        &self.actor
    }

    /// Drain the outbox, then take at most one input — from the continuation slot, or from
    /// whichever lane the priority cycle currently favors — and step.
    ///
    /// Ordering is the whole contract:
    ///
    /// 1. **Flush the outbox before consuming an input**, and never step an actor whose
    ///    outbox is non-empty. That is what makes backpressure propagate — a fast producer
    ///    stops advancing until its slow consumer catches up, without spinning and without
    ///    losing a message.
    /// 2. **Take the continuation before any lane**, so a machine finishes the work unit it
    ///    started before accepting new work. That bounds the number of half-finished
    ///    computations in flight at one.
    /// 3. **Consult lanes in Control > Management > Data order**, budget-limited exactly like
    ///    `ActorScheduler::handle_wake`, so control cannot starve data forever but does not
    ///    wait behind it either.
    /// 4. **Lift the continuation out of the output word before flushing**, so the self-edge
    ///    never reaches [`Wiring`] and so the slot is empty at the moment it is written.
    pub fn poll(&mut self) -> Step {
        if let Some(pending) = &mut self.outbox {
            match self.wiring.flush(pending) {
                Flush::Blocked => return Step::Blocked,
                Flush::Disconnected => return Step::Disconnected,
                Flush::Done => {}
            }
            self.outbox = None;
        }

        if let Some(resumed) = self.continuation.take() {
            return self.dispatch(|actor| actor.step_data(resumed));
        }

        // One full lap visits Control1, Management, Control2, and Data exactly once,
        // regardless of which slot the cycle happens to be sitting on when this call starts —
        // Slot::next() is a 4-cycle, so four steps always covers all four positions. This
        // relies on `advance_if_exhausted` never leaving `self.slot` pointing at an
        // already-spent slot between calls; without it, the first iteration of the *next*
        // call would be spent discovering that instead of checking a lane, and one of the
        // four real positions would go unchecked.
        let mut control_disconnected = false;
        let mut management_disconnected = false;
        let mut data_disconnected = false;

        for _ in 0..4 {
            let limit = match self.slot {
                Slot::Control1 | Slot::Control2 => self.control_half_limit,
                Slot::Management => self.management_limit,
                Slot::Data => self.data_limit,
            };

            if self.slot_progress >= limit {
                self.slot = self.slot.next();
                self.slot_progress = 0;
                continue;
            }

            match self.slot {
                Slot::Control1 | Slot::Control2 => match self.control.take() {
                    Ok(msg) => {
                        self.slot_progress += 1;
                        self.advance_if_exhausted(limit);
                        return self.dispatch(|actor| actor.step_control(msg));
                    }
                    Err(TryRecvError::Empty) => {
                        self.slot = self.slot.next();
                        self.slot_progress = 0;
                    }
                    Err(TryRecvError::Disconnected) => {
                        control_disconnected = true;
                        self.slot = self.slot.next();
                        self.slot_progress = 0;
                    }
                },
                Slot::Management => match self.management.take() {
                    Ok(msg) => {
                        self.slot_progress += 1;
                        self.advance_if_exhausted(limit);
                        return self.dispatch(|actor| actor.step_management(msg));
                    }
                    Err(TryRecvError::Empty) => {
                        self.slot = self.slot.next();
                        self.slot_progress = 0;
                    }
                    Err(TryRecvError::Disconnected) => {
                        management_disconnected = true;
                        self.slot = self.slot.next();
                        self.slot_progress = 0;
                    }
                },
                Slot::Data => match self.data.take() {
                    Ok(msg) => {
                        self.slot_progress += 1;
                        self.advance_if_exhausted(limit);
                        return self.dispatch(|actor| actor.step_data(msg));
                    }
                    Err(TryRecvError::Empty) => {
                        self.slot = self.slot.next();
                        self.slot_progress = 0;
                    }
                    Err(TryRecvError::Disconnected) => {
                        data_disconnected = true;
                        self.slot = self.slot.next();
                        self.slot_progress = 0;
                    }
                },
            }
        }

        if control_disconnected && management_disconnected && data_disconnected {
            Step::Halted(Exit::Completed)
        } else {
            Step::Idle
        }
    }

    /// Roll over to the next slot the instant the current one's budget is spent.
    ///
    /// Load-bearing for the loop in `poll`: it visits at most 4 slots per call on the
    /// assumption that whichever slot `self.slot` names when a call *begins* still has room
    /// (so checking it costs one iteration, not one to discover it's exhausted and a second
    /// to actually check the slot after it). A successful take is the only place progress
    /// grows, so it is the only place this can be left undone — advancing lazily, only when
    /// the *next* call finds the old slot still sitting at its limit, costs the extra
    /// iteration that visiting all 4 slots doesn't have room for.
    fn advance_if_exhausted(&mut self, limit: usize) {
        if self.slot_progress >= limit {
            self.slot = self.slot.next();
            self.slot_progress = 0;
        }
    }

    /// Run one step, advance the step counter, extract the continuation, and flush.
    ///
    /// The one path every lane and the continuation resume through, so "step, then flush,
    /// then maybe park" is written once rather than four times. [`Self::poll_os`] shares this
    /// same finish — the only thing that differs between a lane step and an OS-bridge step is
    /// how the output word was produced, not what happens to it afterward.
    fn dispatch(&mut self, step: impl FnOnce(&mut T) -> Result<T::Out, HandlerError>) -> Step {
        match step(&mut self.actor) {
            Ok(out) => self.finish_step(out),
            Err(HandlerError::Recoverable(msg)) => Step::Halted(Exit::Failed(msg)),
            Err(HandlerError::Fatal(msg)) => panic!("Actor fatal error: {msg}"),
        }
    }

    /// The shared tail of every step: count it, lift the continuation, flush, park on
    /// backpressure or a gone peer. Never called with a non-empty outbox — the caller (lane
    /// dispatch or [`Self::poll_os`]) is the one that guarantees that by flushing first.
    fn finish_step(&mut self, mut out: T::Out) -> Step {
        self.steps += 1;

        // The slot was emptied before this step ran, so this write can never overflow.
        self.continuation = T::take_continuation(&mut out);

        match self.wiring.flush(&mut out) {
            Flush::Blocked => {
                self.outbox = Some(out);
                Step::Blocked
            }
            Flush::Disconnected => {
                // Retained, not dropped — the payload is still in `out`.
                self.outbox = Some(out);
                Step::Disconnected
            }
            Flush::Done => Step::Ran,
        }
    }

    /// Run one [`Transducer::step_os`] through the same dispatch discipline as a lane: outbox
    /// empty first, then step, lift continuation, flush, park on backpressure or a gone peer —
    /// see [`Self::dispatch`]/[`Self::finish_step`], which this shares rather than duplicates.
    ///
    /// Only a dedicated-thread driver calls this, and only when the lanes are quiet — a
    /// [`Host`](crate::Host) never does, because a green actor may not block and `step_os` is
    /// exactly the hook that is allowed to (design doc §5).
    ///
    /// Returns the delivery outcome (same vocabulary as [`Node::poll`]) alongside the actor's
    /// own busy hint. The hint is [`ActorStatus::Idle`] whenever this poll could not actually
    /// run `step_os` because the outbox was still parked — a parked actor has nothing new to
    /// say, so there is no fresher answer than "no".
    ///
    /// Ring-not-full waking is not wired yet (consumers don't ring producers' doorbells today),
    /// so a driver that finds this parked has no way to be woken the instant the target drains;
    /// it retries on its next inbound message, same as the green tier's next sweep.
    pub fn poll_os(&mut self, status: SystemStatus) -> (Step, ActorStatus) {
        if let Some(pending) = &mut self.outbox {
            match self.wiring.flush(pending) {
                Flush::Blocked => return (Step::Blocked, ActorStatus::Idle),
                Flush::Disconnected => return (Step::Disconnected, ActorStatus::Idle),
                Flush::Done => {}
            }
            self.outbox = None;
        }

        match self.actor.step_os(status) {
            Ok((out, actor_status)) => (self.finish_step(out), actor_status),
            Err(HandlerError::Recoverable(msg)) => {
                (Step::Halted(Exit::Failed(msg)), ActorStatus::Idle)
            }
            Err(HandlerError::Fatal(msg)) => panic!("Actor fatal error: {msg}"),
        }
    }
}

impl<T, W, RD> Node<T, W, RD>
where
    T: Transducer,
    W: Wiring<Out = T::Out>,
    RD: Inbox<Item = T::Data>,
{
    /// A data-only node: no control or management lane.
    ///
    /// The common case, and the same three-argument call every single-lane actor already
    /// uses — `RC`/`RM` default to [`NoLane`], so this is `new_with_lanes` with both extra
    /// lanes disconnected and default burst parameters.
    pub fn new(actor: T, data: RD, wiring: W) -> Self {
        Self::new_with_lanes(
            actor,
            Lanes {
                control: NoLane::new(),
                management: NoLane::new(),
                data,
            },
            wiring,
            SchedulerParams::DEFAULT,
        )
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Credit: bounding a request without a new port kind
// ────────────────────────────────────────────────────────────────────────────

/// Bounds how many messages a request edge may have outstanding, without needing a third
/// [`Delivery`] kind.
///
/// A request/response pair — send a request, later receive its reply — is a cycle. [`Topology`]
/// only accepts a cycle closed by an edge that structurally cannot block: a continuation, or a
/// [`Delivery::Droppable`] port. A credit-bounded request *is* a droppable port; `Credit` is the
/// sender-side discipline that keeps the drop from ever actually happening.
///
/// The requester holds one `Credit` per edge, in its own `self` — never shared, never atomic,
/// touched only during that actor's own step:
///
/// - `try_consume` before deciding whether to set the request port. Refuse to emit rather than
///   emit past budget.
/// - `release` when the step that handles the corresponding reply runs — an ordinary input, on
///   whatever lane the reply arrives on.
///
/// As long as the constructor's `max` does not exceed the reply ring's capacity, and every
/// request is gated by `try_consume`, the physical ring can never fill from this edge — the
/// `[drop]` port is a backstop for a bug, not a path the well-behaved system ever takes. This
/// replaces a hand-rolled global atomic token bucket (the kind a request/response actor pair
/// reaches for today) with per-edge, non-atomic, typed state — cheaper, because nothing here is
/// shared across threads, and checked, because a `Credit` that runs dry stops the sender from
/// even trying rather than trusting a convention.
#[derive(Debug, Clone, Copy)]
pub struct Credit {
    available: u32,
    max: u32,
}

impl Credit {
    /// A fresh budget of `max` outstanding requests.
    #[must_use]
    pub fn new(max: u32) -> Self {
        Self {
            available: max,
            max,
        }
    }

    /// Consume one unit of budget. `false` means: do not emit this request.
    #[must_use]
    pub fn try_consume(&mut self) -> bool {
        if self.available > 0 {
            self.available -= 1;
            true
        } else {
            false
        }
    }

    /// Return one unit, on receiving the reply that corresponds to an earlier `try_consume`.
    ///
    /// Saturates at `max` rather than panicking on a spurious extra release — a reply that
    /// somehow arrives twice should not poison every later request on this edge.
    pub fn release(&mut self) {
        self.available = (self.available + 1).min(self.max);
    }

    /// Requests currently in flight.
    #[must_use]
    pub fn outstanding(&self) -> u32 {
        self.max - self.available
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Topology: blocking edges must form a DAG
// ────────────────────────────────────────────────────────────────────────────

/// A handle to an actor in a [`Topology`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActorId(usize);

/// A cycle among blocking edges — the deadlock, caught at bootstrap.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Cycle {
    /// The actors in the cycle, in order. The last one closes back to the first.
    pub actors: Vec<&'static str>,
}

impl std::fmt::Display for Cycle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "blocking cycle: ")?;
        for name in &self.actors {
            write!(f, "{name} → ")?;
        }
        write!(f, "{}", self.actors.first().copied().unwrap_or("?"))
    }
}

impl std::error::Error for Cycle {}

/// The declared actor graph, validated once at bootstrap.
///
/// # Why a DAG
///
/// Two actors that can each block sending to the other deadlock as surely as a self-edge
/// does: A is parked because B's inbox is full, B is parked because A's inbox is full, and
/// the only actor that can drain either inbox is the one parked on the other. Bounded
/// dataflow networks call this *artificial deadlock*, and the classical fix (Parks' 1995
/// algorithm: detect global deadlock at runtime, then grow the smallest blocked buffer) buys
/// generality at the price of a runtime deadlock detector and buffers that silently grow.
///
/// A static topology does not need to pay that. The graph is known before anything runs, so
/// the cycle can be rejected at bootstrap by a topological sort — no runtime cost, no
/// detector, no growth. The rule:
///
/// > **Blocking edges must form a DAG. A cycle may only be closed by an edge that cannot
/// > block.**
///
/// Two such edges exist:
/// - the **continuation** (a self-edge), which has a dedicated one-message slot and so is
///   structurally incapable of blocking — see [`Node::poll`];
/// - a **droppable** edge, which discards on a full target instead of parking (declared with
///   [`Topology::droppable_edge`]) — the "drop this frame if the display is busy" policy.
///
/// Request/response between two actors is therefore expressible: the reply edge is either
/// droppable or credit-bounded (the requester never has more requests outstanding than the
/// reply ring holds, so the reply can never find it full).
#[derive(Debug, Default)]
pub struct Topology {
    names: Vec<&'static str>,
    /// Blocking edges only. Droppable edges are recorded for diagnostics but do not
    /// constrain the order, because they cannot park a producer.
    blocking: Vec<(usize, usize)>,
    droppable: Vec<(usize, usize)>,
}

impl Topology {
    /// A new, empty topology.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Declare an actor.
    pub fn actor(&mut self, name: &'static str) -> ActorId {
        self.names.push(name);
        ActorId(self.names.len() - 1)
    }

    /// Declare an edge that parks its producer when the target is full.
    ///
    /// # Panics
    ///
    /// Panics on a self-edge: a blocking self-edge is an immediate deadlock, and a
    /// self-addressed port must instead be a continuation
    /// ([`Transducer::take_continuation`]), which cannot block.
    pub fn blocking_edge(&mut self, from: ActorId, to: ActorId) {
        assert!(
            from != to,
            "blocking self-edge on {}: a self-addressed port must be a continuation \
             (Transducer::take_continuation), which has a dedicated slot and cannot block",
            self.names[from.0]
        );
        self.blocking.push((from.0, to.0));
    }

    /// Declare an edge that discards on a full target instead of parking.
    ///
    /// Droppable edges are exempt from the DAG rule: they cannot park a producer, so they
    /// cannot participate in a deadlock. This is how a cycle is legally closed.
    pub fn droppable_edge(&mut self, from: ActorId, to: ActorId) {
        self.droppable.push((from.0, to.0));
    }

    /// Check the blocking edges for cycles, and return a safe polling order.
    ///
    /// The order is topological: upstream before downstream, which drains a pipeline in one
    /// sweep instead of trickling one message per pass.
    ///
    /// # Errors
    ///
    /// Returns the offending [`Cycle`] if the blocking edges are not a DAG.
    pub fn validate(&self) -> Result<Vec<ActorId>, Cycle> {
        if let Some(cycle) = self.find_cycle() {
            return Err(Cycle {
                actors: cycle.into_iter().map(|i| self.names[i]).collect(),
            });
        }

        // Kahn's algorithm: repeatedly emit an actor with no unemitted predecessor.
        let n = self.names.len();
        let mut indegree = vec![0usize; n];
        for &(_, to) in &self.blocking {
            indegree[to] += 1;
        }
        let mut ready: Vec<usize> = (0..n).filter(|&i| indegree[i] == 0).collect();
        let mut order = Vec::with_capacity(n);
        while let Some(node) = ready.pop() {
            order.push(ActorId(node));
            for &(from, to) in &self.blocking {
                if from != node {
                    continue;
                }
                indegree[to] -= 1;
                if indegree[to] == 0 {
                    ready.push(to);
                }
            }
        }

        debug_assert_eq!(order.len(), n, "acyclic graph must order every actor");
        Ok(order)
    }

    /// Depth-first search for a cycle, returning the actors on it.
    ///
    /// Iterative rather than recursive so a pathological topology cannot overflow the stack.
    fn find_cycle(&self) -> Option<Vec<usize>> {
        #[derive(Clone, Copy, PartialEq)]
        enum Mark {
            Unseen,
            OnPath,
            Done,
        }

        let n = self.names.len();
        let mut mark = vec![Mark::Unseen; n];

        for start in 0..n {
            if mark[start] != Mark::Unseen {
                continue;
            }
            let mut path = vec![start];
            let mut stack = vec![(start, 0usize)];
            mark[start] = Mark::OnPath;

            while let Some(&mut (node, ref mut next)) = stack.last_mut() {
                let successor = self
                    .blocking
                    .iter()
                    .filter(|&&(from, _)| from == node)
                    .map(|&(_, to)| to)
                    .nth(*next);
                *next += 1;

                match successor {
                    Some(to) if mark[to] == Mark::OnPath => {
                        // `to` is on the current path, so the path from it to here is a cycle.
                        let at = path.iter().position(|&p| p == to)?;
                        return Some(path[at..].to_vec());
                    }
                    Some(to) if mark[to] == Mark::Unseen => {
                        mark[to] = Mark::OnPath;
                        path.push(to);
                        stack.push((to, 0));
                    }
                    Some(_) => {} // already fully explored
                    None => {
                        mark[node] = Mark::Done;
                        path.pop();
                        stack.pop();
                    }
                }
            }
        }
        None
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spsc::spsc_channel;
    use std::convert::Infallible;

    // ── An actor that fans out to two differently-typed targets ─────────────
    //
    // The terminal's shape: one input byte both echoes to the PTY writer and updates the
    // screen. A `Vec<Msg>` output would erase the distinction between the two target types;
    // a struct of typed ports keeps both statically checked.

    #[derive(Debug, PartialEq, Eq)]
    struct Render(u8);
    #[derive(Debug, PartialEq, Eq)]
    struct Echo(u8);

    #[derive(Default)]
    struct AppOut {
        engine: Option<Render>,
        write: Option<Echo>,
    }

    struct AppWiring {
        engine: SpscSender<Render>,
        write: SpscSender<Echo>,
    }

    impl Wiring for AppWiring {
        type Out = AppOut;

        fn flush(&mut self, out: &mut AppOut) -> Flush {
            all([
                send_port(&mut out.engine, &self.engine, Delivery::Blocking),
                send_port(&mut out.write, &self.write, Delivery::Blocking),
            ])
        }
    }

    /// Echoes printable bytes to both ports; swallows NUL (the silent transition).
    struct App {
        seen: u32,
    }

    impl Transducer for App {
        type Control = Infallible;
        type Management = Infallible;
        type Data = u8;
        type Out = AppOut;

        fn step_data(&mut self, byte: u8) -> Result<AppOut, HandlerError> {
            self.seen += 1;
            if byte == 0 {
                return Ok(AppOut::default());
            }
            Ok(AppOut {
                engine: Some(Render(byte)),
                write: Some(Echo(byte)),
            })
        }
    }

    #[test]
    fn step_is_a_pure_function_of_state_and_input() {
        // The payoff of returning output instead of sending it: an actor is table-testable
        // with no scheduler, no wiring, and no mocked senders in the loop.
        let mut app = App { seen: 0 };

        let out = app.step_data(b'x').unwrap();
        assert_eq!(out.engine, Some(Render(b'x')));
        assert_eq!(out.write, Some(Echo(b'x')));

        let quiet = app.step_data(0).unwrap();
        assert!(quiet.engine.is_none() && quiet.write.is_none());

        assert_eq!(app.seen, 2, "state advanced on both inputs");
    }

    #[test]
    fn one_step_fans_out_to_both_typed_ports() {
        let (tx_in, rx_in) = spsc_channel::<u8>(4);
        let (tx_engine, mut rx_engine) = spsc_channel::<Render>(4);
        let (tx_write, mut rx_write) = spsc_channel::<Echo>(4);

        let mut node = Node::new(
            App { seen: 0 },
            rx_in,
            AppWiring {
                engine: tx_engine,
                write: tx_write,
            },
        );

        tx_in.try_send(b'q').unwrap();
        assert_eq!(node.poll(), Step::Ran);

        assert_eq!(rx_engine.try_recv().unwrap(), Render(b'q'));
        assert_eq!(rx_write.try_recv().unwrap(), Echo(b'q'));
    }

    #[test]
    fn empty_inbox_is_idle_and_does_not_step() {
        let (tx_in, rx_in) = spsc_channel::<u8>(4);
        let (tx_engine, _rx_engine) = spsc_channel::<Render>(4);
        let (tx_write, _rx_write) = spsc_channel::<Echo>(4);

        let mut node = Node::new(
            App { seen: 0 },
            rx_in,
            AppWiring {
                engine: tx_engine,
                write: tx_write,
            },
        );

        assert_eq!(node.poll(), Step::Idle);
        assert_eq!(node.steps(), 0);
        drop(tx_in);
    }

    // ── Backpressure: park on a full target, resume when it drains ──────────

    #[test]
    fn parks_on_full_target_and_does_not_step_again_until_it_drains() {
        // The core proof. A slow consumer must stop the producing actor mid-stream without
        // spinning, without dropping a message, and without leaving the actor's state
        // half-updated.
        let (tx_in, rx_in) = spsc_channel::<u8>(64);
        let (tx_engine, mut rx_engine) = spsc_channel::<Render>(2);
        let (tx_write, mut rx_write) = spsc_channel::<Echo>(64);

        let mut node = Node::new(
            App { seen: 0 },
            rx_in,
            AppWiring {
                engine: tx_engine,
                write: tx_write,
            },
        );

        for byte in b'a'..=b'h' {
            tx_in.try_send(byte).unwrap();
        }

        // Poll until the small engine ring backs up. Capacity is a power-of-two rounding of
        // the request, so drive it by behavior rather than assuming an exact count.
        let mut polls = 0;
        while node.poll() == Step::Ran {
            polls += 1;
            assert!(polls < 64, "engine ring should have filled by now");
        }
        assert_eq!(node.poll(), Step::Blocked, "parked on the full engine port");

        // While parked the actor must not advance: no input consumed, no state mutated.
        let steps_when_parked = node.steps();
        let seen_when_parked = node.actor().seen;
        for _ in 0..8 {
            assert_eq!(node.poll(), Step::Blocked);
        }
        assert_eq!(node.steps(), steps_when_parked, "parked actor was stepped");
        assert_eq!(node.actor().seen, seen_when_parked, "parked actor mutated");

        // Drain the target: the parked outbox flushes and the actor advances again.
        while rx_engine.try_recv().is_ok() {}
        assert_eq!(node.poll(), Step::Ran, "resumed once the target drained");
        assert!(node.steps() > steps_when_parked);

        // Nothing was lost: every byte accepted so far reached the write port in order.
        let mut echoed = Vec::new();
        while let Ok(Echo(b)) = rx_write.try_recv() {
            echoed.push(b);
        }
        let expected: Vec<u8> = (b'a'..=b'h').take(echoed.len()).collect();
        assert_eq!(
            echoed, expected,
            "messages delivered in order, none dropped"
        );
    }

    #[test]
    fn a_blocked_port_does_not_hold_up_the_ports_that_fit() {
        // Ports are independent, so a partial flush delivers what it can. The write port
        // keeps flowing while the engine port is backed up.
        let (tx_in, rx_in) = spsc_channel::<u8>(64);
        let (tx_engine, _rx_engine) = spsc_channel::<Render>(2);
        let (tx_write, mut rx_write) = spsc_channel::<Echo>(64);

        let mut node = Node::new(
            App { seen: 0 },
            rx_in,
            AppWiring {
                engine: tx_engine,
                write: tx_write,
            },
        );

        for byte in b'a'..=b'f' {
            tx_in.try_send(byte).unwrap();
        }
        while node.poll() == Step::Ran {}

        let delivered = std::iter::from_fn(|| rx_write.try_recv().ok()).count();
        assert!(
            delivered >= 3,
            "write port should have kept flowing past the engine ring's capacity, got {delivered}"
        );
    }

    #[test]
    fn a_droppable_port_never_blocks_and_never_keeps_a_message() {
        // The runtime half of `Topology::droppable_edge`: because it cannot park its
        // producer, it cannot take part in a deadlock, which is what lets it close a cycle.
        let (tx, mut rx) = spsc_channel::<Render>(2);

        let mut delivered = 0;
        for i in 0..32u8 {
            let mut port = Some(Render(i));
            assert_eq!(send_port(&mut port, &tx, Delivery::Droppable), Flush::Done);
            assert!(port.is_none(), "a droppable port always clears");
            if rx.try_recv().is_ok() {
                delivered += 1;
            }
        }
        assert!(delivered > 0, "some messages must actually get through");
    }

    #[test]
    fn a_port_to_a_dead_target_reports_it_and_keeps_the_payload() {
        // This previously asserted `Flush::Done` with the payload consumed — a dead peer was
        // indistinguishable from a successful send. For a port carrying a moved resource that
        // silently destroyed it while the sender believed it had been delivered. Retention is
        // the point of the variant; the caller can still retry, hand off, or shut down with the
        // value in hand.
        for delivery in [Delivery::Droppable, Delivery::Blocking] {
            let (tx, rx) = spsc_channel::<Render>(4);
            drop(rx);

            let mut port = Some(Render(1));
            assert_eq!(
                send_port(&mut port, &tx, delivery),
                Flush::Disconnected,
                "a gone target is reported, not silently swallowed ({delivery:?})"
            );
            assert!(
                port.is_some(),
                "and the payload stays recoverable ({delivery:?})"
            );
        }
    }

    #[test]
    fn disconnected_outranks_blocked_when_combining_outcomes() {
        // Backpressure resolves by waiting; a dead peer does not. If both happen in one flush
        // the caller needs to hear the one that won't fix itself.
        assert_eq!(
            all([Flush::Done, Flush::Blocked, Flush::Disconnected]),
            Flush::Disconnected
        );
        assert_eq!(all([Flush::Done, Flush::Blocked]), Flush::Blocked);
        assert_eq!(all([Flush::Done, Flush::Done]), Flush::Done);
    }

    // ── Credit: a request/response edge bounded without a new port kind ─────

    #[test]
    fn credit_exhausts_and_refuses_further_requests() {
        let mut credit = Credit::new(2);
        assert!(credit.try_consume());
        assert!(credit.try_consume());
        assert!(!credit.try_consume(), "budget of 2 must refuse the third");
        assert_eq!(credit.outstanding(), 2);
    }

    #[test]
    fn releasing_credit_restores_budget() {
        let mut credit = Credit::new(1);
        assert!(credit.try_consume());
        assert!(!credit.try_consume());

        credit.release();
        assert!(credit.try_consume(), "released budget must be usable again");
    }

    #[test]
    fn credit_saturates_at_max_rather_than_overflowing_on_a_spurious_release() {
        let mut credit = Credit::new(3);
        credit.release(); // no matching consume — must not push available above max
        credit.release();
        assert_eq!(credit.outstanding(), 0);
        assert!(credit.try_consume());
        assert!(credit.try_consume());
        assert!(credit.try_consume());
        assert!(
            !credit.try_consume(),
            "a spurious release must not grant extra budget"
        );
    }

    #[test]
    fn credit_gating_keeps_a_droppable_reply_edge_from_ever_dropping() {
        // The property that makes a droppable port the right closer for a credit-bounded
        // cycle: as long as every send is gated by `try_consume` and max <= ring capacity,
        // the ring can never fill, so the drop path this test's sibling exercises is never
        // actually taken.
        const RING_CAPACITY: u32 = 4;
        let (tx, mut rx) = spsc_channel::<Render>(RING_CAPACITY as usize);
        let mut credit = Credit::new(RING_CAPACITY);

        let mut sent = 0;
        for i in 0..64u8 {
            if !credit.try_consume() {
                break; // well-behaved: stop rather than send past budget
            }
            let mut port = Some(Render(i));
            assert_eq!(
                send_port(&mut port, &tx, Delivery::Droppable),
                Flush::Done,
                "gated by credit, so the ring never fills and nothing is ever dropped"
            );
            assert!(port.is_none());
            sent += 1;
        }

        assert_eq!(
            sent, RING_CAPACITY as usize,
            "stopped exactly at the ring's capacity"
        );
        assert_eq!(
            std::iter::from_fn(|| rx.try_recv().ok()).count(),
            RING_CAPACITY as usize,
            "every gated message actually arrived — the backstop was never needed"
        );
    }

    #[test]
    fn without_credit_gating_the_same_droppable_port_silently_drops_instead_of_deadlocking() {
        // The contrast case: an ungated sender that ignores the ring's real capacity does not
        // hang the way a Blocking port would — it drops. That is the backstop `Credit` exists
        // to make unreachable in the well-behaved case above.
        let (tx, mut rx) = spsc_channel::<Render>(4);

        for i in 0..64u8 {
            let mut port = Some(Render(i));
            assert_eq!(send_port(&mut port, &tx, Delivery::Droppable), Flush::Done);
        }

        let delivered = std::iter::from_fn(|| rx.try_recv().ok()).count();
        assert!(
            delivered < 64,
            "an ungated sender must have overrun the ring and dropped some messages"
        );
    }

    #[test]
    fn disconnected_inbox_halts_the_node() {
        let (tx_in, rx_in) = spsc_channel::<u8>(4);
        let (tx_engine, _rx_engine) = spsc_channel::<Render>(4);
        let (tx_write, _rx_write) = spsc_channel::<Echo>(4);

        let mut node = Node::new(
            App { seen: 0 },
            rx_in,
            AppWiring {
                engine: tx_engine,
                write: tx_write,
            },
        );

        drop(tx_in);
        assert_eq!(node.poll(), Step::Halted(Exit::Completed));
    }

    #[test]
    fn a_failed_step_halts_without_emitting() {
        struct Boom;
        struct BoomWiring {
            out: SpscSender<u8>,
        }
        impl Wiring for BoomWiring {
            type Out = Option<u8>;
            fn flush(&mut self, out: &mut Option<u8>) -> Flush {
                send_port(out, &self.out, Delivery::Blocking)
            }
        }
        impl Transducer for Boom {
            type Control = Infallible;
            type Management = Infallible;
            type Data = u8;
            type Out = Option<u8>;
            fn step_data(&mut self, _: u8) -> Result<Option<u8>, HandlerError> {
                Err(HandlerError::Recoverable("boom".into()))
            }
        }

        let (tx_in, rx_in) = spsc_channel::<u8>(4);
        let (tx_out, mut rx_out) = spsc_channel::<u8>(4);
        let mut node = Node::new(Boom, rx_in, BoomWiring { out: tx_out });

        tx_in.try_send(1).unwrap();
        assert_eq!(
            node.poll(),
            Step::Halted(Exit::Failed("boom".into())),
            "a failed transition reports the failure"
        );
        assert!(
            rx_out.try_recv().is_err(),
            "Err carries no output — a failed step emits nothing"
        );
    }

    // ── step_os / poll_os: the OS-bridge hook ────────────────────────────────
    //
    // `Bridge` stands in for an OS-bridge transducer (§5): its data lane is ordinary, but it
    // also has an internal queue it can only drain from `step_os`, played by a dedicated-thread
    // driver rather than a `Host`.

    #[derive(Default)]
    struct BridgeOut {
        word: Option<u32>,
        again: Option<u32>,
    }

    struct BridgeWiring {
        out: SpscSender<u32>,
    }

    impl Wiring for BridgeWiring {
        type Out = BridgeOut;

        fn flush(&mut self, out: &mut BridgeOut) -> Flush {
            send_port(&mut out.word, &self.out, Delivery::Blocking)
        }
    }

    struct Bridge {
        queue: Vec<u32>,
        /// Counts `step_os` calls, so a test can prove it was *not* invoked while parked.
        step_os_calls: u32,
    }

    impl Transducer for Bridge {
        type Control = Infallible;
        type Management = Infallible;
        type Data = u32;
        type Out = BridgeOut;

        fn step_data(&mut self, msg: u32) -> Result<BridgeOut, HandlerError> {
            Ok(BridgeOut {
                word: Some(msg),
                again: None,
            })
        }

        fn step_os(
            &mut self,
            _status: SystemStatus,
        ) -> Result<(BridgeOut, ActorStatus), HandlerError> {
            self.step_os_calls += 1;
            let Some(next) = self.queue.pop() else {
                return Ok((BridgeOut::default(), ActorStatus::Idle));
            };
            let more = if self.queue.is_empty() {
                ActorStatus::Idle
            } else {
                ActorStatus::Busy
            };
            Ok((
                BridgeOut {
                    word: Some(next),
                    again: None,
                },
                more,
            ))
        }

        fn take_continuation(out: &mut BridgeOut) -> Option<u32> {
            out.again.take()
        }
    }

    #[test]
    fn step_os_output_flushes_through_wiring() {
        let (_tx_in, rx_in) = spsc_channel::<u32>(4);
        let (tx_out, mut rx_out) = spsc_channel::<u32>(4);

        let mut node = Node::new(
            Bridge {
                queue: vec![7],
                step_os_calls: 0,
            },
            rx_in,
            BridgeWiring { out: tx_out },
        );

        assert_eq!(
            node.poll_os(SystemStatus::Idle),
            (Step::Ran, ActorStatus::Idle),
            "one queued item, nothing left after it: Idle"
        );
        assert_eq!(
            rx_out.try_recv().unwrap(),
            7,
            "the OS-bridge output reached the wiring"
        );
    }

    #[test]
    fn busy_propagates_from_step_os() {
        let (_tx_in, rx_in) = spsc_channel::<u32>(4);
        let (tx_out, mut rx_out) = spsc_channel::<u32>(4);

        let mut node = Node::new(
            Bridge {
                queue: vec![2, 1],
                step_os_calls: 0,
            },
            rx_in,
            BridgeWiring { out: tx_out },
        );

        // Two items queued: popping the first leaves one behind, so the actor reports Busy —
        // "re-poll me without blocking" — exactly what a dedicated-thread driver needs to keep
        // draining a bursty OS source without going back to the doorbell.
        assert_eq!(
            node.poll_os(SystemStatus::Idle),
            (Step::Ran, ActorStatus::Busy),
            "one item left after this pop"
        );
        assert_eq!(rx_out.try_recv().unwrap(), 1, "queue is popped LIFO");

        assert_eq!(
            node.poll_os(SystemStatus::Idle),
            (Step::Ran, ActorStatus::Idle),
            "queue now empty: Idle"
        );
        assert_eq!(rx_out.try_recv().unwrap(), 2);
    }

    /// An OS-bridge actor whose `step_os` yields to itself, exactly the way a data-lane step
    /// can (mealy.rs: "Yielding"). Isolates the continuation claim from `Bridge`'s output-word
    /// claim above.
    struct OsYield {
        last_resumed: Option<u32>,
    }

    #[derive(Default)]
    struct OsYieldOut {
        again: Option<u32>,
    }

    struct OsYieldWiring;

    impl Wiring for OsYieldWiring {
        type Out = OsYieldOut;

        fn flush(&mut self, out: &mut OsYieldOut) -> Flush {
            debug_assert!(
                out.again.is_none(),
                "continuation must not reach the wiring"
            );
            Flush::Done
        }
    }

    impl Transducer for OsYield {
        type Control = Infallible;
        type Management = Infallible;
        type Data = u32;
        type Out = OsYieldOut;

        fn step_data(&mut self, n: u32) -> Result<OsYieldOut, HandlerError> {
            self.last_resumed = Some(n);
            Ok(OsYieldOut::default())
        }

        fn step_os(
            &mut self,
            _status: SystemStatus,
        ) -> Result<(OsYieldOut, ActorStatus), HandlerError> {
            Ok((OsYieldOut { again: Some(42) }, ActorStatus::Idle))
        }

        fn take_continuation(out: &mut OsYieldOut) -> Option<u32> {
            out.again.take()
        }
    }

    #[test]
    fn step_os_continuation_lands_in_the_slot() {
        let (_tx_in, rx_in) = spsc_channel::<u32>(4);
        let mut node = Node::new(OsYield { last_resumed: None }, rx_in, OsYieldWiring);

        assert_eq!(
            node.poll_os(SystemStatus::Idle),
            (Step::Ran, ActorStatus::Idle)
        );
        assert_eq!(
            node.continuation,
            Some(42),
            "step_os's continuation lands in the same slot a lane's does"
        );

        // It resumes before any lane is even consulted, exactly like a lane-produced one.
        assert_eq!(node.poll(), Step::Ran);
        assert_eq!(
            node.actor().last_resumed,
            Some(42),
            "the continuation actually reached step_data"
        );
        assert!(node.continuation.is_none());
    }

    #[test]
    fn a_parked_outbox_defers_step_os() {
        let (tx_in, rx_in) = spsc_channel::<u32>(64);
        let (tx_out, mut rx_out) = spsc_channel::<u32>(2);

        let mut node = Node::new(
            Bridge {
                queue: vec![99],
                step_os_calls: 0,
            },
            rx_in,
            BridgeWiring { out: tx_out },
        );

        // Drive data steps until the small target ring backs up and the outbox parks.
        // Capacity is a power-of-two rounding of the request (see `parks_on_full_target_...`
        // above), so drive it by behavior rather than assuming an exact count.
        for i in 0..8 {
            tx_in.try_send(i).unwrap();
        }
        let mut polls = 0;
        while node.poll() == Step::Ran {
            polls += 1;
            assert!(polls < 32, "target ring should have filled by now");
        }
        assert_eq!(
            node.poll(),
            Step::Blocked,
            "outbox is parked going into poll_os"
        );

        assert_eq!(
            node.poll_os(SystemStatus::Idle),
            (Step::Blocked, ActorStatus::Idle),
            "a parked outbox must be retried before step_os runs at all"
        );
        assert_eq!(
            node.actor().step_os_calls,
            0,
            "step_os must not run while parked — it has nothing new to say"
        );

        // Drain the target ring (not the node's outbox — that message is still parked, waiting
        // for room): this only frees up space for the flush `poll_os` retries next.
        while rx_out.try_recv().is_ok() {}
        assert_eq!(
            node.poll_os(SystemStatus::Idle),
            (Step::Ran, ActorStatus::Idle),
            "outbox cleared, so this call reaches step_os and drains the queued item"
        );
        assert_eq!(node.actor().step_os_calls, 1);
        let unparked = rx_out
            .try_recv()
            .expect("the previously parked word flushes first, in the same poll_os call");
        assert!(
            (0..8).contains(&unparked),
            "that is the value that was blocked"
        );
        assert_eq!(
            rx_out.try_recv().unwrap(),
            99,
            "then step_os's own output, flushed right after"
        );
    }

    // ── The self-port is the yield ──────────────────────────────────────────

    /// Counts down, handing itself the next value each step. The long computation is chopped
    /// into steps by a self-addressed port — which the [`Node`] routes to its continuation
    /// slot, never to a queue.
    struct Countdown {
        finished: Option<u32>,
    }

    #[derive(Default)]
    struct CountdownOut {
        again: Option<u32>,
        done: Option<u32>,
    }

    struct CountdownWiring {
        done: SpscSender<u32>,
    }

    impl Wiring for CountdownWiring {
        type Out = CountdownOut;

        fn flush(&mut self, out: &mut CountdownOut) -> Flush {
            // The self-port never reaches the wiring: `take_continuation` lifts it out first.
            debug_assert!(
                out.again.is_none(),
                "continuation must not reach the wiring"
            );
            send_port(&mut out.done, &self.done, Delivery::Blocking)
        }
    }

    impl Transducer for Countdown {
        type Control = Infallible;
        type Management = Infallible;
        type Data = u32;
        type Out = CountdownOut;

        fn step_data(&mut self, n: u32) -> Result<CountdownOut, HandlerError> {
            if n == 0 {
                self.finished = Some(0);
                return Ok(CountdownOut {
                    done: Some(0),
                    ..Default::default()
                });
            }
            // Yield: hand the continuation to ourselves and return.
            Ok(CountdownOut {
                again: Some(n - 1),
                ..Default::default()
            })
        }

        fn take_continuation(out: &mut CountdownOut) -> Option<u32> {
            out.again.take()
        }
    }

    #[test]
    fn a_self_addressed_port_yields_between_steps() {
        let (tx_in, rx_in) = spsc_channel::<u32>(4);
        let (tx_done, mut rx_done) = spsc_channel::<u32>(4);

        let mut node = Node::new(
            Countdown { finished: None },
            rx_in,
            CountdownWiring { done: tx_done },
        );

        tx_in.try_send(5).unwrap();

        // Each poll runs exactly one step: the loop is driven by the scheduler, not by the
        // handler. Between any two steps the worker is free to run other actors.
        let mut steps = 0;
        while node.actor().finished.is_none() {
            assert_eq!(node.poll(), Step::Ran);
            steps += 1;
            assert!(steps <= 8, "countdown should finish in 6 steps");
        }

        assert_eq!(steps, 6, "5,4,3,2,1,0 — one step each, yielding between");
        assert_eq!(rx_done.try_recv().unwrap(), 0);
    }

    #[test]
    fn the_continuation_slot_holds_at_most_one_message() {
        // The deadlock argument, asserted: the slot is emptied before the step that refills
        // it, so a self-edge has a hard bound of one and can never find its target full.
        let (tx_in, rx_in) = spsc_channel::<u32>(8);
        let (tx_done, _rx_done) = spsc_channel::<u32>(8);
        let mut node = Node::new(
            Countdown { finished: None },
            rx_in,
            CountdownWiring { done: tx_done },
        );

        tx_in.try_send(3).unwrap();
        while node.poll() == Step::Ran {
            assert!(
                node.continuation.is_none() || node.continuation.is_some(),
                "slot is a single Option — it cannot hold two"
            );
        }
        assert!(
            node.continuation.is_none(),
            "the slot is empty once the machine finishes"
        );
    }

    #[test]
    fn a_continuation_resumes_before_new_inbox_work() {
        // Continuation-first: finish the work unit in flight before accepting new work, so
        // the number of half-finished computations stays at one.
        let (tx_in, rx_in) = spsc_channel::<u32>(8);
        let (tx_done, _rx_done) = spsc_channel::<u32>(8);
        let mut node = Node::new(
            Countdown { finished: None },
            rx_in,
            CountdownWiring { done: tx_done },
        );

        tx_in.try_send(2).unwrap();
        tx_in.try_send(9).unwrap(); // queued behind the in-flight countdown

        assert_eq!(node.poll(), Step::Ran); // 2 → continuation 1
        assert_eq!(node.continuation, Some(1));
        assert_eq!(node.poll(), Step::Ran); // resumes 1, not the queued 9
        assert_eq!(node.continuation, Some(0));
        assert_eq!(node.poll(), Step::Ran); // 0 → done
        assert_eq!(node.actor().finished, Some(0), "first unit finished first");
        assert!(node.continuation.is_none());
    }

    #[test]
    fn interleaves_a_yielding_actor_with_another_actor() {
        // The point of yielding: a long computation does not monopolise the worker. Round
        // robin here stands in for the real run-queue; the semantics being proven are that
        // one poll == one step, so progress interleaves.
        let (tx_count, rx_count) = spsc_channel::<u32>(8);
        let (tx_done, _rx_done) = spsc_channel::<u32>(4);
        let mut counter = Node::new(
            Countdown { finished: None },
            rx_count,
            CountdownWiring { done: tx_done },
        );

        let (tx_in, rx_in) = spsc_channel::<u8>(8);
        let (tx_engine, _rx_engine) = spsc_channel::<Render>(8);
        let (tx_write, _rx_write) = spsc_channel::<Echo>(8);
        let mut app = Node::new(
            App { seen: 0 },
            rx_in,
            AppWiring {
                engine: tx_engine,
                write: tx_write,
            },
        );

        tx_count.try_send(4).unwrap();
        for byte in b'a'..=b'c' {
            tx_in.try_send(byte).unwrap();
        }

        // One poll each, round robin, until both are quiet.
        for _ in 0..8 {
            let a = counter.poll();
            let b = app.poll();
            if a == Step::Idle && b == Step::Idle {
                break;
            }
        }

        assert_eq!(
            counter.actor().finished,
            Some(0),
            "the yielding actor ran to completion"
        );
        assert_eq!(
            app.actor().seen,
            3,
            "the other actor made progress in between — no monopolisation"
        );
    }

    // ── Priority lanes: ported from ActorScheduler::handle_wake ─────────────

    /// Records which lane each step came from. `Out = ()`: these tests are about input
    /// ordering, not output, so there is nothing to wire.
    struct LaneLog {
        seen: Vec<&'static str>,
    }

    struct NoWiring;
    impl Wiring for NoWiring {
        type Out = ();
        fn flush(&mut self, (): &mut ()) -> Flush {
            Flush::Done
        }
    }

    impl Transducer for LaneLog {
        type Control = ();
        type Management = ();
        type Data = ();
        type Out = ();

        fn step_data(&mut self, (): ()) -> Result<(), HandlerError> {
            self.seen.push("D");
            Ok(())
        }

        fn step_control(&mut self, (): ()) -> Result<(), HandlerError> {
            self.seen.push("C");
            Ok(())
        }

        fn step_management(&mut self, (): ()) -> Result<(), HandlerError> {
            self.seen.push("M");
            Ok(())
        }
    }

    #[test]
    fn one_message_per_lane_drains_control_then_management_then_data() {
        let (tx_c, rx_c) = spsc_channel::<()>(4);
        let (tx_m, rx_m) = spsc_channel::<()>(4);
        let (tx_d, rx_d) = spsc_channel::<()>(4);

        let mut node = Node::new_with_lanes(
            LaneLog { seen: Vec::new() },
            Lanes {
                control: rx_c,
                management: rx_m,
                data: rx_d,
            },
            NoWiring,
            SchedulerParams::DEFAULT,
        );

        tx_c.try_send(()).unwrap();
        tx_m.try_send(()).unwrap();
        tx_d.try_send(()).unwrap();

        for _ in 0..3 {
            assert_eq!(node.poll(), Step::Ran);
        }
        assert_eq!(
            node.actor().seen,
            vec!["C", "M", "D"],
            "one message per lane must drain in priority order"
        );
    }

    #[test]
    fn control_backlog_does_not_starve_data_forever() {
        // The property `ActorScheduler::handle_wake`'s half-control/mgmt/half-control/data
        // split exists for: with control_burst_limit=2 (half=1) and no management traffic,
        // control gets two turns per cycle (Control1, Control2) before data gets one — so
        // data is guaranteed a turn every cycle, however deep the control backlog runs.
        let params = SchedulerParams {
            control_mgmt_buffer_size: 1,
            control_burst_multiplier: 2, // control_burst_limit = 2, half = 1
            management_burst_multiplier: 1,
            default_data_burst_limit: 1,
            ..SchedulerParams::DEFAULT
        };

        let (tx_c, rx_c) = spsc_channel::<()>(64);
        let (_tx_m, rx_m) = spsc_channel::<()>(4);
        let (tx_d, rx_d) = spsc_channel::<()>(4);

        let mut node = Node::new_with_lanes(
            LaneLog { seen: Vec::new() },
            Lanes {
                control: rx_c,
                management: rx_m,
                data: rx_d,
            },
            NoWiring,
            params,
        );

        for _ in 0..5 {
            tx_c.try_send(()).unwrap();
        }
        tx_d.try_send(()).unwrap();

        for _ in 0..3 {
            assert_eq!(node.poll(), Step::Ran);
        }
        assert_eq!(
            node.actor().seen,
            vec!["C", "C", "D"],
            "data must be served after one cycle (2 control), not after the whole backlog"
        );
    }

    // Regression: half_control = control_burst_limit / 2 used to truncate to 0 when the
    // burst limit was 1, and a zero-limit slot could never make progress (mirrors
    // control_lane_progresses_with_burst_limit_one in lib.rs for the OS-thread scheduler).
    #[test]
    fn control_lane_progresses_with_burst_limit_one() {
        let params = SchedulerParams {
            control_mgmt_buffer_size: 1,
            control_burst_multiplier: 1, // control_burst_limit == 1
            ..SchedulerParams::DEFAULT
        };
        assert_eq!(params.control_burst_limit(), 1);

        let (tx_c, rx_c) = spsc_channel::<()>(4);
        let (_tx_m, rx_m) = spsc_channel::<()>(4);
        let (_tx_d, rx_d) = spsc_channel::<()>(4);

        let mut node = Node::new_with_lanes(
            LaneLog { seen: Vec::new() },
            Lanes {
                control: rx_c,
                management: rx_m,
                data: rx_d,
            },
            NoWiring,
            params,
        );

        tx_c.try_send(()).unwrap();
        assert_eq!(node.poll(), Step::Ran);
        assert_eq!(
            node.actor().seen,
            vec!["C"],
            "control message was never processed with control_burst_limit == 1"
        );
    }

    #[test]
    fn halts_only_once_every_lane_is_disconnected() {
        let (tx_c, rx_c) = spsc_channel::<()>(4);
        let (tx_m, rx_m) = spsc_channel::<()>(4);
        let (tx_d, rx_d) = spsc_channel::<()>(4);

        let mut node = Node::new_with_lanes(
            LaneLog { seen: Vec::new() },
            Lanes {
                control: rx_c,
                management: rx_m,
                data: rx_d,
            },
            NoWiring,
            SchedulerParams::DEFAULT,
        );

        drop(tx_c);
        drop(tx_m);
        // data's sender is still alive but has sent nothing: Empty, not Disconnected, so the
        // node must stay Idle rather than halt on a partial disconnect.
        assert_eq!(node.poll(), Step::Idle);

        drop(tx_d);
        assert_eq!(node.poll(), Step::Halted(Exit::Completed));
    }

    // Regression: a slot that finishes a poll() sitting exactly at its limit (e.g. Data,
    // after delivering the one message its budget allowed) used to advance lazily — only
    // when the *next* call found it still there. With small limits and idle control/
    // management lanes, that costs the wraparound its one spare iteration: the call that
    // should re-check Data instead spends its four iterations on Control1 (advance off the
    // exhausted slot), Management, Control2, and lands back on a *fresh* Data one iteration
    // too late to actually check it — reporting Idle with a message still queued. Caught by
    // `bench_priority.rs`'s drain loop, which (unlike the small fixed poll counts above)
    // keeps calling `poll` until every message is gone and so was long enough to wrap.
    #[test]
    fn many_data_messages_drain_steadily_with_idle_control_and_management() {
        let params = SchedulerParams {
            control_mgmt_buffer_size: 1,
            control_burst_multiplier: 2,
            management_burst_multiplier: 1,
            default_data_burst_limit: 1,
            ..SchedulerParams::DEFAULT
        };

        let (_tx_c, rx_c) = spsc_channel::<()>(4);
        let (_tx_m, rx_m) = spsc_channel::<()>(4);
        let (tx_d, rx_d) = spsc_channel::<()>(64);

        let mut node = Node::new_with_lanes(
            LaneLog { seen: Vec::new() },
            Lanes {
                control: rx_c,
                management: rx_m,
                data: rx_d,
            },
            NoWiring,
            params,
        );

        const MESSAGES: usize = 50;
        for _ in 0..MESSAGES {
            tx_d.try_send(()).unwrap();
        }

        for i in 0..MESSAGES {
            assert_eq!(
                node.poll(),
                Step::Ran,
                "message {i} of {MESSAGES}: control/management are merely empty, not \
                 disconnected, so this must never report Idle with data still queued"
            );
        }
        assert_eq!(node.actor().seen.len(), MESSAGES);
    }

    // ── Topology: cycles are a bootstrap error, not a runtime deadlock ──────

    #[test]
    fn a_pipeline_validates_and_orders_upstream_first() {
        let mut topo = Topology::new();
        let read = topo.actor("read");
        let parse = topo.actor("parse");
        let app = topo.actor("app");
        topo.blocking_edge(read, parse);
        topo.blocking_edge(parse, app);

        let order = topo.validate().expect("a pipeline is a DAG");
        let position = |id: ActorId| order.iter().position(|&o| o == id).unwrap();
        assert!(position(read) < position(parse), "upstream polls first");
        assert!(position(parse) < position(app));
    }

    #[test]
    fn a_blocking_cycle_is_rejected_with_the_cycle_named() {
        let mut topo = Topology::new();
        let app = topo.actor("app");
        let compiler = topo.actor("compiler");
        topo.blocking_edge(app, compiler); // request
        topo.blocking_edge(compiler, app); // reply — closes the cycle

        let cycle = topo
            .validate()
            .expect_err("mutual blocking must be rejected");
        assert_eq!(cycle.actors.len(), 2);
        assert!(cycle.actors.contains(&"app") && cycle.actors.contains(&"compiler"));
        assert!(
            cycle.to_string().contains("app"),
            "the message must name the cycle, got {cycle}"
        );
    }

    #[test]
    fn a_droppable_reply_edge_makes_request_response_legal() {
        // The escape hatch: a cycle may be closed by an edge that cannot park its producer.
        let mut topo = Topology::new();
        let app = topo.actor("app");
        let compiler = topo.actor("compiler");
        topo.blocking_edge(app, compiler);
        topo.droppable_edge(compiler, app);

        assert!(
            topo.validate().is_ok(),
            "a non-blocking reply edge cannot deadlock, so it does not constrain the order"
        );
    }

    #[test]
    fn a_longer_cycle_is_found() {
        let mut topo = Topology::new();
        let a = topo.actor("a");
        let b = topo.actor("b");
        let c = topo.actor("c");
        let d = topo.actor("d");
        topo.blocking_edge(a, b);
        topo.blocking_edge(b, c);
        topo.blocking_edge(c, d);
        topo.blocking_edge(d, b); // b → c → d → b

        let cycle = topo
            .validate()
            .expect_err("three-actor cycle must be caught");
        assert_eq!(cycle.actors.len(), 3, "got {cycle}");
        assert!(
            !cycle.actors.contains(&"a"),
            "a is upstream, not on the cycle"
        );
    }

    #[test]
    fn a_diamond_is_acyclic() {
        let mut topo = Topology::new();
        let src = topo.actor("src");
        let left = topo.actor("left");
        let right = topo.actor("right");
        let sink = topo.actor("sink");
        topo.blocking_edge(src, left);
        topo.blocking_edge(src, right);
        topo.blocking_edge(left, sink);
        topo.blocking_edge(right, sink);

        let order = topo.validate().expect("a diamond is a DAG");
        assert_eq!(order.len(), 4);
        let position = |id: ActorId| order.iter().position(|&o| o == id).unwrap();
        assert!(position(src) < position(sink));
    }

    #[test]
    #[should_panic(expected = "must be a continuation")]
    fn a_blocking_self_edge_is_rejected_at_declaration() {
        let mut topo = Topology::new();
        let solo = topo.actor("solo");
        topo.blocking_edge(solo, solo);
    }
}
