//! Mealy transducer vs the existing `Actor` path.
//!
//! Two comparisons, both moving the same messages through the same logic:
//!
//! 1. **dispatch** — one actor, single thread, drained to completion. Isolates the cost of a
//!    step: return-a-port-struct + scheduler-side flush, against handler-calls-send.
//! 2. **pipeline** — three actors chained. The existing path needs one OS thread per actor,
//!    so every message pays a cross-thread ring hop and a doorbell wake. The transducer path
//!    runs all three on one thread in topological order, so a message walks the whole
//!    pipeline in one sweep with no wake and no cross-core traffic.
//!
//! The second is the honest measure of the design: the dispatch delta is small either way,
//! but the thread-hop that disappears is not.

use actor_scheduler::mealy::{Delivery, Flush, Node, Step, Transducer, Wiring, all, send_port};
use actor_scheduler::spsc::{SpscSender, spsc_channel};
use actor_scheduler::{
    Actor, ActorScheduler, ActorStatus, HandlerError, HandlerResult, Message, SystemStatus,
};
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use std::convert::Infallible;

const MESSAGES: [usize; 3] = [1_000, 10_000, 100_000];

/// Ring capacity large enough that nothing backs up: both paths measure work, not parking.
fn capacity(messages: usize) -> usize {
    (messages * 2).next_power_of_two()
}

// ────────────────────────────────────────────────────────────────────────────
// Shared payloads
// ────────────────────────────────────────────────────────────────────────────

/// Deliberately not `Copy`: a real payload moves, and a `Copy` payload would let the
/// optimiser elide work that a real one cannot.
#[derive(Debug)]
struct Render(#[allow(dead_code)] u8);
#[derive(Debug)]
struct Echo(#[allow(dead_code)] u8);

// ────────────────────────────────────────────────────────────────────────────
// 1. Dispatch: one actor, one thread
// ────────────────────────────────────────────────────────────────────────────

/// Existing path: the handler holds the senders and calls them itself.
struct SendingActor {
    engine: SpscSender<Render>,
    write: SpscSender<Echo>,
    seen: u64,
}

impl Actor<u8, (), ()> for SendingActor {
    fn handle_data(&mut self, byte: u8) -> HandlerResult {
        self.seen += 1;
        // Rings are sized so neither can fill; a failure would invalidate the measurement.
        self.engine
            .try_send(Render(byte))
            .expect("engine ring sized for the run");
        self.write
            .try_send(Echo(byte))
            .expect("write ring sized for the run");
        Ok(())
    }
    fn handle_control(&mut self, (): ()) -> HandlerResult {
        Ok(())
    }
    fn handle_management(&mut self, (): ()) -> HandlerResult {
        Ok(())
    }
    fn handle_os(&mut self, _: SystemStatus) -> Result<ActorStatus, HandlerError> {
        Ok(ActorStatus::Idle)
    }
}

/// Transducer path: the step returns the ports; the scheduler sends them.
struct SteppingActor {
    seen: u64,
}

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

impl Transducer for SteppingActor {
    type Control = Infallible;
    type Management = Infallible;
    type Data = u8;
    type Out = AppOut;

    fn step_data(&mut self, byte: u8) -> Result<AppOut, HandlerError> {
        self.seen += 1;
        Ok(AppOut {
            engine: Some(Render(byte)),
            write: Some(Echo(byte)),
        })
    }
}

fn bench_dispatch(c: &mut Criterion) {
    let mut group = c.benchmark_group("dispatch");

    for count in MESSAGES {
        group.throughput(criterion::Throughput::Elements(count as u64));
        let cap = capacity(count);

        group.bench_with_input(BenchmarkId::new("actor_sends", count), &count, |b, &n| {
            b.iter(|| {
                let (tx_engine, mut rx_engine) = spsc_channel::<Render>(cap);
                let (tx_write, mut rx_write) = spsc_channel::<Echo>(cap);
                let (handle, mut sched) = ActorScheduler::<u8, (), ()>::new(n.max(1), cap);
                let mut actor = SendingActor {
                    engine: tx_engine,
                    write: tx_write,
                    seen: 0,
                };

                for i in 0..n {
                    handle.send(Message::Data(i as u8)).unwrap();
                }
                while actor.seen < n as u64 {
                    if sched.poll_once(&mut actor) {
                        break;
                    }
                }

                while rx_engine.try_recv().is_ok() {}
                while rx_write.try_recv().is_ok() {}
                black_box(actor.seen)
            });
        });

        group.bench_with_input(
            BenchmarkId::new("transducer_returns", count),
            &count,
            |b, &n| {
                b.iter(|| {
                    let (tx_engine, mut rx_engine) = spsc_channel::<Render>(cap);
                    let (tx_write, mut rx_write) = spsc_channel::<Echo>(cap);
                    let (tx_in, rx_in) = spsc_channel::<u8>(cap);
                    let mut node = Node::new(
                        SteppingActor { seen: 0 },
                        rx_in,
                        AppWiring {
                            engine: tx_engine,
                            write: tx_write,
                        },
                    );

                    for i in 0..n {
                        tx_in.try_send(i as u8).unwrap();
                    }
                    while node.poll() == Step::Ran {}

                    while rx_engine.try_recv().is_ok() {}
                    while rx_write.try_recv().is_ok() {}
                    black_box(node.steps())
                });
            },
        );
    }

    group.finish();
}

// ────────────────────────────────────────────────────────────────────────────
// 2. Pipeline: three actors chained
// ────────────────────────────────────────────────────────────────────────────

/// Existing path: a stage owning the handle of the next stage, run on its own thread.
struct ForwardActor {
    next: Option<actor_scheduler::ActorHandle<u8, (), ()>>,
    seen: u64,
}

impl Actor<u8, (), ()> for ForwardActor {
    fn handle_data(&mut self, byte: u8) -> HandlerResult {
        self.seen += 1;
        if let Some(next) = &self.next {
            next.send(Message::Data(byte.wrapping_add(1)))
                .map_err(|_| HandlerError::new("downstream gone"))?;
        }
        Ok(())
    }
    fn handle_control(&mut self, (): ()) -> HandlerResult {
        Ok(())
    }
    fn handle_management(&mut self, (): ()) -> HandlerResult {
        Ok(())
    }
    fn handle_os(&mut self, _: SystemStatus) -> Result<ActorStatus, HandlerError> {
        Ok(ActorStatus::Idle)
    }
}

/// Transducer path: a stage that returns its one downstream port.
struct ForwardStep {
    seen: u64,
}

struct ForwardWiring {
    next: SpscSender<u8>,
}

impl Wiring for ForwardWiring {
    type Out = Option<u8>;
    fn flush(&mut self, out: &mut Option<u8>) -> Flush {
        send_port(out, &self.next, Delivery::Blocking)
    }
}

impl Transducer for ForwardStep {
    type Control = Infallible;
    type Management = Infallible;
    type Data = u8;
    type Out = Option<u8>;

    fn step_data(&mut self, byte: u8) -> Result<Option<u8>, HandlerError> {
        self.seen += 1;
        Ok(Some(byte.wrapping_add(1)))
    }
}

fn bench_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("pipeline_3_stage");

    for count in MESSAGES {
        group.throughput(criterion::Throughput::Elements(count as u64));
        let cap = capacity(count);

        // Existing: three schedulers, three OS threads, two cross-thread hops per message.
        group.bench_with_input(BenchmarkId::new("three_threads", count), &count, |b, &n| {
            b.iter(|| {
                let (h3, mut s3) = ActorScheduler::<u8, (), ()>::new(n.max(1), cap);
                let (h2, mut s2) = ActorScheduler::<u8, (), ()>::new(n.max(1), cap);
                let (h1, mut s1) = ActorScheduler::<u8, (), ()>::new(n.max(1), cap);

                let mut a3 = ForwardActor {
                    next: None,
                    seen: 0,
                };
                let mut a2 = ForwardActor {
                    next: Some(h3),
                    seen: 0,
                };
                let mut a1 = ForwardActor {
                    next: Some(h2),
                    seen: 0,
                };

                let t3 = std::thread::spawn(move || {
                    s3.run(&mut a3);
                    a3.seen
                });
                let t2 = std::thread::spawn(move || {
                    s2.run(&mut a2);
                    a2.seen
                });
                let t1 = std::thread::spawn(move || {
                    s1.run(&mut a1);
                    a1.seen
                });

                for i in 0..n {
                    h1.send(Message::Data(i as u8)).unwrap();
                }
                // Dropping the head handle cascades shutdown down the chain.
                drop(h1);
                let seen = t1.join().unwrap();
                let mid = t2.join().unwrap();
                let tail = t3.join().unwrap();
                black_box((mid, tail));
                black_box(seen)
            });
        });

        // Transducer: three nodes, one thread, polled upstream-first.
        group.bench_with_input(BenchmarkId::new("one_thread", count), &count, |b, &n| {
            b.iter(|| {
                let (tx_in, rx_in) = spsc_channel::<u8>(cap);
                let (tx_12, rx_12) = spsc_channel::<u8>(cap);
                let (tx_23, rx_23) = spsc_channel::<u8>(cap);
                let (tx_out, mut rx_out) = spsc_channel::<u8>(cap);

                let mut n1 = Node::new(
                    ForwardStep { seen: 0 },
                    rx_in,
                    ForwardWiring { next: tx_12 },
                );
                let mut n2 = Node::new(
                    ForwardStep { seen: 0 },
                    rx_12,
                    ForwardWiring { next: tx_23 },
                );
                let mut n3 = Node::new(
                    ForwardStep { seen: 0 },
                    rx_23,
                    ForwardWiring { next: tx_out },
                );

                for i in 0..n {
                    tx_in.try_send(i as u8).unwrap();
                }

                // Topological order: each sweep pushes work all the way down the pipeline.
                loop {
                    let a = n1.poll();
                    let b = n2.poll();
                    let c = n3.poll();
                    if a != Step::Ran && b != Step::Ran && c != Step::Ran {
                        break;
                    }
                }

                while rx_out.try_recv().is_ok() {}
                black_box(n3.steps())
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_dispatch, bench_pipeline);
criterion_main!(benches);
