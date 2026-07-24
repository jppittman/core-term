# Mealy transducer vs the existing `Actor` path

- **Date**: 2026-07-24
- **Bench**: `actor-scheduler/benches/bench_mealy.rs`
- **Command**: `cargo bench -p actor-scheduler --bench bench_mealy`
- **Design**: `docs/designs/actor-scheduler-mealy-transducer.md`

Criterion, 20 samples, 3 s measurement. Medians reported.

## 1. Dispatch — one actor, one thread

Same logic both ways: consume a byte, emit to two differently-typed targets. The existing
path calls `send` inside the handler; the transducer returns a port struct that the scheduler
flushes.

| Messages | `actor_sends` | `transducer_returns` | Speedup |
|---------:|--------------:|---------------------:|--------:|
| 1 000 | 24.3 µs | 12.5 µs | **1.94×** |
| 10 000 | 244.1 µs | 141.5 µs | **1.73×** |
| 100 000 | 2.428 ms | 1.321 ms | **1.84×** |

Per message: **~24 ns → ~13 ns**. Throughput ~41 Melem/s → ~76 Melem/s.

## 2. Pipeline — three stages chained

Three forwarding stages. The existing path needs one OS thread per actor, so each message
pays two cross-thread ring hops plus doorbell wakes. The transducer path runs all three nodes
on one thread, polled upstream-first, so a message walks the whole pipeline in one sweep.

| Messages | `three_threads` | `one_thread` | Speedup |
|---------:|----------------:|-------------:|--------:|
| 1 000 | 332.0 µs | 17.2 µs | 19.3× † |
| 10 000 | 1.507 ms | 184.8 µs | 8.2× † |
| 100 000 | 11.181 ms | 1.714 ms | **6.5×** |

† Inflated: the three-thread arm spawns and joins three OS threads inside the measured
iteration, which dominates at small N. **The 100 000 row is the fair one** — spawn cost is
amortised there. Steady state: **~112 ns → ~17 ns per message**, i.e. the two cross-thread
hops cost roughly **95 ns per message**, ~47 ns per hop.

Single-thread throughput is flat at ~58 Melem/s across all sizes; the three-thread arm climbs
(3.0 → 6.6 → 8.9 Melem/s) as spawn amortises.

## What this does and does not show

**Does:**
- Returning output instead of sending it is *not* a tax — it is ~1.8× cheaper per dispatch.
  The existing path pays for `Message` enum wrapping, the doorbell, the burst-limited drain
  loop, the sharded inbox, and a `park()` call per wake; a step pays for none of it.
- Coordination, not computation, dominates a cheap-stage pipeline. Collapsing three threads
  onto one removes ~95 ns/message of pure overhead.

**Does not:**
- The transducer arm has **no waker**, because a same-thread worker does not need one. That
  is a real saving for co-located actors, but a multi-worker runtime will add some of it back
  when it has to wake a sleeping worker. These numbers are the co-located best case.
- Stages here are trivial (increment and forward). With CPU-heavy stages the three-thread arm
  wins back ground through genuine parallelism — this bench measures coordination overhead,
  not parallel speedup. The result argues for *co-locating cheap actors*, not for abolishing
  threads.
- Rings are sized so nothing backs up, so this measures the unblocked path. Backpressure
  behavior is covered by the unit tests, not here.
