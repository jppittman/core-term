//! `ports!` codegen: the output word and its wiring.
//!
//! These run from outside the crate because the generated code uses `::actor_scheduler::`
//! paths, which only resolve for a dependent crate.

use actor_scheduler::mealy::{Flush, Node, Step, Transducer, Wiring};
use actor_scheduler::spsc::spsc_channel;
use actor_scheduler::{HandlerError, ports};
use std::convert::Infallible;

// ────────────────────────────────────────────────────────────────────────────
// Fan-out to two differently-typed targets — the terminal's echo-and-draw shape
// ────────────────────────────────────────────────────────────────────────────

#[derive(Debug, PartialEq, Eq)]
struct Render(u8);
#[derive(Debug, PartialEq, Eq)]
struct Echo(u8);

ports! {
    App {
        engine: Render,
        write: Echo,
    }
}

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
            return Ok(AppOut::default()); // the silent transition
        }
        Ok(AppOut {
            engine: Some(Render(byte)),
            write: Some(Echo(byte)),
        })
    }
}

#[test]
fn generated_ports_carry_their_own_types() {
    let (tx_in, rx_in) = spsc_channel::<u8>(8);
    let (tx_engine, mut rx_engine) = spsc_channel::<Render>(8);
    let (tx_write, mut rx_write) = spsc_channel::<Echo>(8);

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
fn a_default_output_word_delivers_nothing() {
    let mut out = AppOut::default();
    assert!(out.engine.is_none() && out.write.is_none());

    let (tx_engine, mut rx_engine) = spsc_channel::<Render>(4);
    let (tx_write, mut rx_write) = spsc_channel::<Echo>(4);
    let mut wiring = AppWiring {
        engine: tx_engine,
        write: tx_write,
    };

    assert_eq!(wiring.flush(&mut out), Flush::Done);
    assert!(rx_engine.try_recv().is_err() && rx_write.try_recv().is_err());
}

#[test]
fn generated_flush_parks_on_a_full_target_and_keeps_the_message() {
    // The property that is easy to get wrong by hand: a blocked port must be left set, not
    // dropped, so the node can retry it.
    let (tx_engine, mut rx_engine) = spsc_channel::<Render>(1);
    let (tx_write, _rx_write) = spsc_channel::<Echo>(64);
    let mut wiring = AppWiring {
        engine: tx_engine,
        write: tx_write,
    };

    // Fill the engine ring.
    let mut blocked_at = None;
    for byte in 0..64u8 {
        let mut out = AppOut {
            engine: Some(Render(byte)),
            write: Some(Echo(byte)),
        };
        if wiring.flush(&mut out) == Flush::Blocked {
            assert_eq!(
                out.engine,
                Some(Render(byte)),
                "a blocked port must keep its message for the retry"
            );
            assert!(out.write.is_none(), "the port that fit was still delivered");
            blocked_at = Some(byte);
            break;
        }
    }
    assert!(blocked_at.is_some(), "the engine ring should have filled");

    // Draining the target unblocks the same wiring.
    while rx_engine.try_recv().is_ok() {}
    let mut out = AppOut {
        engine: Some(Render(9)),
        write: Some(Echo(9)),
    };
    assert_eq!(wiring.flush(&mut out), Flush::Done);
}

// ────────────────────────────────────────────────────────────────────────────
// `[drop]` is gone — see actor-scheduler-macros' `parse_port_kind`, which now panics at
// expansion time naming the removal. Not exercised here: a macro-expansion panic fails the
// whole test crate's compilation, so there is no runtime assertion to write for it; the panic
// message itself is the only artifact worth pinning, and it lives in the macro's own source.
// ────────────────────────────────────────────────────────────────────────────

// ────────────────────────────────────────────────────────────────────────────
// A `[self]` port is a continuation, not a wiring field
// ────────────────────────────────────────────────────────────────────────────

ports! {
    Countdown {
        again: u32 [self],
        done: u32,
    }
}

struct Countdown {
    finished: Option<u32>,
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
        Ok(CountdownOut {
            again: Some(n - 1),
            ..Default::default()
        })
    }

    fn take_continuation(out: &mut CountdownOut) -> Option<u32> {
        out.take_continuation()
    }
}

#[test]
fn a_self_port_yields_through_the_continuation_slot() {
    let (tx_in, rx_in) = spsc_channel::<u32>(8);
    let (tx_done, mut rx_done) = spsc_channel::<u32>(8);

    let mut node = Node::new(
        Countdown { finished: None },
        rx_in,
        CountdownWiring { done: tx_done },
    );

    tx_in.try_send(5).unwrap();

    let mut steps = 0;
    while node.actor().finished.is_none() {
        assert_eq!(node.poll(), Step::Ran);
        steps += 1;
        assert!(steps <= 8, "should finish in 6 steps");
    }

    assert_eq!(steps, 6, "5,4,3,2,1,0 — one step each, yielding between");
    assert_eq!(rx_done.try_recv().unwrap(), 0);
}

#[test]
fn a_self_port_has_no_wiring_field() {
    // `CountdownWiring` is constructed with only `done`. If the macro had emitted a field for
    // the `[self]` port, this would not compile — which is the point: a continuation cannot
    // be wired to a queue.
    let (tx_done, _rx_done) = spsc_channel::<u32>(4);
    let wiring = CountdownWiring { done: tx_done };
    drop(wiring);
}

// ────────────────────────────────────────────────────────────────────────────
// Generic port types pass through
// ────────────────────────────────────────────────────────────────────────────

ports! {
    Batcher {
        out: Vec<u8>,
    }
}

struct Batcher;

impl Transducer for Batcher {
    type Control = Infallible;
    type Management = Infallible;
    type Data = u8;
    type Out = BatcherOut;

    fn step_data(&mut self, byte: u8) -> Result<BatcherOut, HandlerError> {
        Ok(BatcherOut {
            out: Some(vec![byte; 3]),
        })
    }
}

#[test]
fn a_generic_port_type_survives_codegen() {
    let (tx_in, rx_in) = spsc_channel::<u8>(4);
    let (tx_out, mut rx_out) = spsc_channel::<Vec<u8>>(4);

    let mut node = Node::new(Batcher, rx_in, BatcherWiring { out: tx_out });
    tx_in.try_send(7).unwrap();
    assert_eq!(node.poll(), Step::Ran);
    assert_eq!(rx_out.try_recv().unwrap(), vec![7, 7, 7]);
}

#[test]
fn a_disconnected_inbox_halts_a_generated_node() {
    let (tx_in, rx_in) = spsc_channel::<u8>(4);
    let (tx_out, _rx_out) = spsc_channel::<Vec<u8>>(4);

    let mut node = Node::new(Batcher, rx_in, BatcherWiring { out: tx_out });
    drop(tx_in);
    assert_eq!(node.poll(), Step::Halted);
}

// ────────────────────────────────────────────────────────────────────────────
// Custom target types pass through (`-> TargetType`)
// ────────────────────────────────────────────────────────────────────────────

use actor_scheduler::actors::{Schedule, Timer};
use std::time::Duration;

ports! {
    CustomTarget {
        tick: u8 -> actor_scheduler::spsc::SpscSender<u8>,
        interval: Duration -> Timer,
    }
}

#[test]
fn a_custom_target_type_is_used_in_generated_wiring() {
    let (tx_tick, mut rx_tick) = spsc_channel::<u8>(4);
    let timer = Timer::spawn(
        "test-ports-custom-target",
        Schedule::Every(Duration::from_secs(3600)),
        || std::ops::ControlFlow::Continue(()),
    );

    let mut wiring = CustomTargetWiring {
        tick: tx_tick,
        interval: timer,
    };

    let mut out = CustomTargetOut {
        tick: Some(42),
        interval: Some(Duration::from_millis(100)),
    };

    assert_eq!(wiring.flush(&mut out), Flush::Done);
    assert!(out.tick.is_none());
    assert!(out.interval.is_none());

    assert_eq!(rx_tick.try_recv().unwrap(), 42);

    wiring.interval.stop();
}
