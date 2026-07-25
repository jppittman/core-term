//! Demo + benchmark for the priority lanes ported onto `Node`/`Transducer`.
//!
//! `bench_mealy.rs` only exercises the data lane — it cannot show the thing this port actually
//! bought: that a green actor's data lane keeps making progress under a continuous control
//! flood, at a bounded, predictable ratio, instead of starving until the flood stops.
//!
//! Each `bench_function` prints a human-readable before/after comparison once (outside the
//! timed loop) and then lets criterion measure the same scenario for statistical confidence.

use actor_scheduler::mealy::{Flush, Lanes, Node, Step, Transducer, Wiring};
use actor_scheduler::spsc::spsc_channel;
use actor_scheduler::{HandlerError, SchedulerParams};
use criterion::{Criterion, black_box, criterion_group, criterion_main};

/// Counts messages by lane; no output ports, so `Out = ()` and `NoWiring` is a no-op.
struct Counter {
    control_seen: u64,
    data_seen: u64,
}

struct NoWiring;
impl Wiring for NoWiring {
    type Out = ();
    fn flush(&mut self, (): &mut ()) -> Flush {
        Flush::Done
    }
}

impl Transducer for Counter {
    type Control = ();
    type Management = ();
    type Data = ();
    type Out = ();

    fn step_data(&mut self, (): ()) -> Result<(), HandlerError> {
        self.data_seen += 1;
        Ok(())
    }

    fn step_control(&mut self, (): ()) -> Result<(), HandlerError> {
        self.control_seen += 1;
        Ok(())
    }
}

/// A control:data burst ratio realistic enough to make contention visible without requiring
/// an unreasonable number of polls: two control turns per one data turn, per full cycle.
fn contention_params() -> SchedulerParams {
    SchedulerParams {
        control_mgmt_buffer_size: 1,
        control_burst_multiplier: 2, // control_burst_limit = 2, half = 1
        management_burst_multiplier: 1,
        default_data_burst_limit: 1,
        ..SchedulerParams::DEFAULT
    }
}

/// Poll until every data message queued up front has been delivered, feeding a fresh control
/// message back in after each one is consumed — an unbounded flood, not a fixed backlog.
/// Returns the number of polls it took.
fn drain_data_under_control_flood(data_messages: u64) -> u64 {
    let (tx_c, rx_c) = spsc_channel::<()>(8);
    let (tx_m, rx_m) = spsc_channel::<()>(4);
    let (tx_d, rx_d) = spsc_channel::<()>(1024);

    for _ in 0..data_messages {
        tx_d.try_send(()).unwrap();
    }
    // Prime the flood: as long as this stays topped up, control never runs dry.
    for _ in 0..4 {
        tx_c.try_send(()).unwrap();
    }
    drop(tx_m); // no management traffic in this demo

    let mut node = Node::new_with_lanes(
        Counter {
            control_seen: 0,
            data_seen: 0,
        },
        Lanes {
            control: rx_c,
            management: rx_m,
            data: rx_d,
        },
        NoWiring,
        contention_params(),
    );

    let mut polls = 0u64;
    while node.actor().data_seen < data_messages {
        // Keep the flood alive: whatever control took last poll, replace it. A real flooder
        // (e.g. continuous vsync ticks) never runs out either. A full ring just means the
        // flood is already saturated — nothing to do about that here.
        if tx_c.try_send(()).is_err() {
            // Receiver gone would be a bug in this demo (control is never dropped); a full
            // ring is fine and expected.
        }
        assert_eq!(node.poll(), Step::Ran, "must always make progress, never park or idle");
        polls += 1;
        assert!(
            polls < data_messages * 20,
            "data should not need more than a small constant factor of polls"
        );
    }
    polls
}

/// The same data volume with no control traffic at all — the baseline this demo compares
/// against.
fn drain_data_uncontended(data_messages: u64) -> u64 {
    let (_tx_c, rx_c) = spsc_channel::<()>(4);
    let (_tx_m, rx_m) = spsc_channel::<()>(4);
    let (tx_d, rx_d) = spsc_channel::<()>(1024);

    for _ in 0..data_messages {
        tx_d.try_send(()).unwrap();
    }

    let mut node = Node::new_with_lanes(
        Counter {
            control_seen: 0,
            data_seen: 0,
        },
        Lanes {
            control: rx_c,
            management: rx_m,
            data: rx_d,
        },
        NoWiring,
        contention_params(),
    );

    let mut polls = 0u64;
    while node.actor().data_seen < data_messages {
        assert_eq!(node.poll(), Step::Ran);
        polls += 1;
    }
    polls
}

fn bench_data_under_control_flood(c: &mut Criterion) {
    const DATA_MESSAGES: u64 = 1_000;

    // Printed once, outside the timed loop: the human-readable comparison.
    let uncontended_polls = drain_data_uncontended(DATA_MESSAGES);
    let contended_polls = drain_data_under_control_flood(DATA_MESSAGES);
    println!(
        "\n[bench_priority] delivering {DATA_MESSAGES} data messages:\n\
         \x20 no control traffic:        {uncontended_polls} polls ({:.2} polls/message)\n\
         \x20 continuous control flood:  {contended_polls} polls ({:.2} polls/message)\n\
         \x20 => every data message still arrived; the flood cost a bounded, predictable\n\
         \x20    ~{:.1}x more polls per message, not starvation.\n",
        uncontended_polls as f64 / DATA_MESSAGES as f64,
        contended_polls as f64 / DATA_MESSAGES as f64,
        contended_polls as f64 / uncontended_polls as f64,
    );

    let mut group = c.benchmark_group("priority");

    group.bench_function("data_uncontended", |b| {
        b.iter(|| black_box(drain_data_uncontended(black_box(DATA_MESSAGES))));
    });

    group.bench_function("data_under_control_flood", |b| {
        b.iter(|| black_box(drain_data_under_control_flood(black_box(DATA_MESSAGES))));
    });

    group.finish();
}

criterion_group!(benches, bench_data_under_control_flood);
criterion_main!(benches);
