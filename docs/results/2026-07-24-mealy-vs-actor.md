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

## Update (2026-07-25): after priority lanes + Credit

Re-ran the dispatch/pipeline benchmarks above after porting Control/Management/Data priority
lanes onto `Node` (`docs/designs/actor-scheduler-mealy-transducer.md` §9.5). Both arms —
including `actor_sends`, the pre-existing `ActorScheduler` path this session did not touch —
came back 15–70% faster than the 2026-07-24 run in this same container. That is environment
variance between runs on a shared box, not a real speedup from either session's changes; the
comparable, noise-resistant number is the *ratio* between arms within one run, which held at
roughly the same order of magnitude as before (transducer path several× faster than the
send-based path). Absolute cross-run timings in the table above should not be read as
before/after for the lane work — no regression was found, but claim only that, not a 3–5×
improvement from adding lane-checking overhead nobody would expect.

## New: priority-lane contention demo (`bench_priority.rs`)

`bench_mealy.rs` only exercises the data lane, so it can't show what porting priority lanes
actually bought. `bench_priority.rs` drains 1,000 data messages on a lane-aware `Node`
(`control_burst_limit = 2`, i.e. two control turns per cycle before one data turn), with no
control traffic vs. a continuous control flood that is topped up after every poll:

```
delivering 1000 data messages:
  no control traffic:        1000 polls (1.00 polls/message)
  continuous control flood:  3000 polls (3.00 polls/message)
  => every data message still arrived; the flood cost a bounded, predictable
     ~3.0x more polls per message, not starvation.

priority/data_uncontended          ~18.1 µs / 1000 msgs  (~18 ns/msg)
priority/data_under_control_flood  ~26.1 µs / 1000 msgs  (~26 ns/msg)
```

The 3.0x figure is not a measurement — it's `2 control-slot-visits + 1 data-slot-visit` per
cycle, exactly the ratio `control_burst_limit=2` declares, reproduced deterministically every
run. All 1,000 data messages arrive either way; the flood costs a bounded, predictable multiple
of polls, never an unbounded one. Wall-clock cost (~1.44×) is lower than the 3× poll-count
ratio because a poll that only advances past an empty/exhausted slot does less work than one
that dispatches a full step.

### A real bug this caught

Writing this benchmark's drain-to-completion loop (assert `Step::Ran` every call, no `Idle`
allowed until every message lands) found a genuine bug the unit tests had not: a slot that
finished a poll sitting exactly at its budget limit (e.g. `Data`, having just delivered the one
message its limit allowed) advanced lazily — only when the *next* call discovered it was still
there. With small custom limits and idle control/management lanes, the wraparound needed to
recheck that slot is exactly 4 iterations, and the lazy advance consumes the first one just
moving off the stale position, leaving only 3 to reach a lane with anything in it — one
iteration short. The existing small-number tests (`control_backlog_does_not_starve_data_
forever`, `control_lane_progresses_with_burst_limit_one`) never wrapped a full cycle within
their few assertions and didn't hit it; a tight drain loop over many messages did immediately.

Fixed by `Node::advance_if_exhausted`: roll a slot over to the next one the instant its budget
is spent, in the same call that spent it, rather than waiting for the next call to discover it.
Added `many_data_messages_drain_steadily_with_idle_control_and_management` as a unit-test
regression (confirmed it fails without the fix, by reverting it locally and observing the
assertion fire, before restoring it) so this doesn't require running the benchmark to catch
again.
