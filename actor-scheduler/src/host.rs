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
//! sleeps at 0% CPU**, and wakes when a message arrives, exactly like every other actor.
//!
//! # Ownership, not migration
//!
//! A host *owns* its green actors; they never move to another thread. That is deliberate:
//! the measured win (`docs/results/2026-07-24-mealy-vs-actor.md`) comes from co-locating a
//! pipeline so a message walks it in one sweep with no cross-thread hop. A shared queue that
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
use std::marker::PhantomData;

use crate::lifecycle::Exit;
use crate::mealy::{Inbox, Node, Step, Transducer, Wiring};
use crate::{Actor, ActorStatus, HandlerError, HandlerResult, SystemStatus};

/// A hosted green actor with its types erased, so one host can own a heterogeneous set.
///
/// This is the only type erasure in the design, and it is deliberately one method wide: a
/// host needs to advance a green actor and learn what happened, nothing more.
pub trait Green {
    /// Advance this actor by at most one step.
    fn poll(&mut self) -> Step;
}

impl<T, W, R> Green for Node<T, W, R>
where
    T: Transducer,
    W: Wiring<Out = T::Out>,
    R: Inbox<Item = T::In>,
{
    fn poll(&mut self) -> Step {
        Node::poll(self)
    }
}

/// An OS-thread actor that owns green actors and runs them in its `park`.
///
/// The host has no control or management messages of its own — those type parameters are
/// [`Infallible`], so the compiler knows the handlers are unreachable and the `match msg {}`
/// bodies below are total. Its data lane carries whatever the hosted actors are fed with:
/// `handle_data` hands each inbound message to the `deliver` closure, which pushes it into
/// the right green actor's inbox.
pub struct Host<D, F> {
    nodes: Vec<Box<dyn Green>>,
    deliver: F,
    exits: Vec<Exit>,
    _pd: PhantomData<D>,
}

impl<D, F> Host<D, F>
where
    F: FnMut(D),
{
    /// A host that routes its inbound messages with `deliver`.
    pub fn new(deliver: F) -> Self {
        Self {
            nodes: Vec::new(),
            deliver,
            exits: Vec::new(),
            _pd: PhantomData,
        }
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
    /// Returns `Busy` if any of them ran, so the scheduler keeps sweeping without blocking;
    /// `Idle` when they are all quiet — parked on backpressure or out of input — so the
    /// scheduler blocks on the doorbell and the thread sleeps.
    pub fn sweep(&mut self) -> ActorStatus {
        let mut ran = false;
        let mut i = 0;

        while i < self.nodes.len() {
            match self.nodes[i].poll() {
                Step::Ran => {
                    ran = true;
                    i += 1;
                }
                Step::Blocked | Step::Idle => i += 1,
                Step::Halted(exit) => {
                    self.exits.push(exit);
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

impl<D, F> Actor<D, Infallible, Infallible> for Host<D, F>
where
    F: FnMut(D),
{
    fn handle_data(&mut self, msg: D) -> HandlerResult {
        (self.deliver)(msg);
        Ok(())
    }

    fn handle_control(&mut self, msg: Infallible) -> HandlerResult {
        match msg {}
    }

    fn handle_management(&mut self, msg: Infallible) -> HandlerResult {
        match msg {}
    }

    fn park(&mut self, _: SystemStatus) -> Result<ActorStatus, HandlerError> {
        Ok(self.sweep())
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
        type In = u32;
        type Out = Option<u32>;

        fn step(&mut self, n: u32) -> Result<Option<u32>, HandlerError> {
            self.seen += 1;
            Ok(Some(n + 1))
        }
    }

    #[test]
    fn a_sweep_runs_owned_green_actors() {
        let (tx_in, rx_in) = spsc_channel::<u32>(8);
        let (tx_out, mut rx_out) = spsc_channel::<u32>(8);

        let mut host = Host::new(|_: u32| {});
        host.adopt(Node::new(
            Forward { seen: 0 },
            rx_in,
            ForwardWiring { next: tx_out },
        ));

        tx_in.try_send(1).unwrap();
        assert_eq!(host.sweep(), ActorStatus::Busy, "a green actor had work");
        assert_eq!(rx_out.try_recv().unwrap(), 2);
    }

    #[test]
    fn a_quiet_host_reports_idle_so_the_thread_can_sleep() {
        // The 0%-CPU contract: with nothing to do, the host tells the scheduler to block on
        // the doorbell rather than spin.
        let (tx_in, rx_in) = spsc_channel::<u32>(8);
        let (tx_out, _rx_out) = spsc_channel::<u32>(8);

        let mut host = Host::new(|_: u32| {});
        host.adopt(Node::new(
            Forward { seen: 0 },
            rx_in,
            ForwardWiring { next: tx_out },
        ));

        assert_eq!(host.sweep(), ActorStatus::Idle, "no input, nothing ran");

        tx_in.try_send(7).unwrap();
        assert_eq!(host.sweep(), ActorStatus::Busy);
        assert_eq!(host.sweep(), ActorStatus::Idle, "quiet again after draining");
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

        let mut host = Host::new(|_: u32| {});
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

    #[test]
    fn a_halted_green_actor_is_dropped_and_its_exit_recorded() {
        let (tx_in, rx_in) = spsc_channel::<u32>(8);
        let (tx_out, _rx_out) = spsc_channel::<u32>(8);

        let mut host = Host::new(|_: u32| {});
        host.adopt(Node::new(
            Forward { seen: 0 },
            rx_in,
            ForwardWiring { next: tx_out },
        ));
        assert_eq!(host.len(), 1);

        drop(tx_in); // the green actor's inbox disconnects
        host.sweep();

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

        let mut host = Host::new(|_: u32| {});
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

    #[test]
    fn the_host_routes_its_own_inbound_messages_into_a_green_inbox() {
        // End to end through the real scheduler: a message on the host's data lane is routed
        // by `handle_data` into a green actor's inbox, and the following `park` sweep runs it.
        use crate::{ActorScheduler, Message};

        let (tx_in, rx_in) = spsc_channel::<u32>(8);
        let (tx_out, mut rx_out) = spsc_channel::<u32>(8);

        let mut host = Host::new(move |n: u32| {
            tx_in.try_send(n).expect("green inbox sized for the test");
        });
        host.adopt(Node::new(
            Forward { seen: 0 },
            rx_in,
            ForwardWiring { next: tx_out },
        ));

        let (handle, mut sched) = ActorScheduler::<u32, Infallible, Infallible>::new(16, 16);
        handle.send(Message::Data(41)).unwrap();

        // One scheduler wake: drains the data lane (routing 41 into the green inbox), then
        // parks — and the park is the sweep that runs the green actor.
        assert!(sched.poll_once(&mut host).is_none(), "still running");
        assert_eq!(rx_out.try_recv().unwrap(), 42);
    }
}
