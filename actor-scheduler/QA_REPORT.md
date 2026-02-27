# Actor-Scheduler QA Report — "Enrique Havoc" Edition

**Date:** 2026-02-27
**Updated:** 2026-02-27
**Scope:** Cooperative green-thread scheduler, SPSC channels, Kubelet lifecycle, ServiceHandle
**Methodology:** Adversarial testing — 27 tests targeting breakpoints, not happy paths
**Result:** 3 production-severity gaps found, 2 fixed (GAP-1, GAP-2), 1 documented (GAP-3). Custom associated error types added. Idle timeout API added.

---

## Executive Summary

The actor-scheduler is **well-engineered for its intended use case** (3-thread terminal architecture at 155 FPS). The SPSC-per-producer design eliminates send-side contention, priority scheduling is correct, and the Kubelet restart machinery works.

**Fixes applied in this revision:**
- **GAP-1 FIXED:** Shutdown deadlock eliminated via `AtomicBool` + non-blocking wake
- **GAP-2 FIXED:** Idle timeout added (`SchedulerParams::idle_timeout`) for pod liveness
- **Associated Error types:** `Actor` trait now has `type Error: Display`, enabling domain-specific errors
- **Type-safe errors:** `HandlerError<E>` is generic — actors no longer forced to use `String`

**Verdict:** Production-ready for core-term's 3-actor topology. Cooperative scheduling is an intentional design choice for the target workload.

---

## Gaps Found (Production Severity)

### GAP-1: Shutdown Deadlock When Doorbell Full [FIXED]

**Test:** `gap1_shutdown_never_deadlocks_with_full_doorbell` (was `gap1_shutdown_after_data_deadlocks_without_running_scheduler`)

**What was broken:** `Message::Shutdown` used `tx_doorbell.send(System::Shutdown)` — a **blocking** send on a capacity-1 channel. If the doorbell already contained a Wake, the Shutdown sender blocked forever.

**Fix:** Shutdown now uses a shared `AtomicBool` flag (`shutdown_requested`) instead of the doorbell channel:
1. `ActorHandle::send(Message::Shutdown)` sets `shutdown_requested.store(true, Release)` and calls `wake()` (non-blocking `try_send`)
2. The scheduler checks the flag at the top of every loop iteration and after waking from `recv()`
3. `System::Shutdown` variant removed — the doorbell now carries only `Wake` signals

**Why this is correct:** `AtomicBool::store()` never blocks. The `wake()` call uses `try_send` which drops the Wake if the doorbell is already full (safe — one pending wake is sufficient). The scheduler will see the flag on its next iteration.

---

### GAP-2: Restarted Pods With No External Senders Block Forever [FIXED]

**Test:** `kubelet_rapid_restart_frequency_gate`

**What was broken:** When the Kubelet restarts a pod and no `ServiceHandle` reconnects, the new pod's scheduler blocks forever on `rx_doorbell.recv()`.

**Fix:** Added `idle_timeout: Option<Duration>` to `SchedulerParams`. When set, the scheduler uses `recv_timeout()` instead of blocking `recv()`. If no messages arrive within the timeout, the scheduler exits with `PodPhase::Completed`.

```rust
// Usage:
let params = SchedulerParams {
    idle_timeout: Some(Duration::from_secs(30)),
    ..SchedulerParams::DEFAULT
};
let mut builder = ActorBuilder::new_with_params(1024, None, params);
```

**Default:** `None` (no timeout — original behavior preserved for existing code).

---

### GAP-3: Race Between Handler Error and Shutdown Send [MEDIUM]

**Test:** `error_during_drain_all_exits_cleanly`

**What:** When the scheduler exits due to a handler error (Recoverable or Fatal), it drops the doorbell receiver. Any concurrent Shutdown send in flight gets `SendError::Disconnected`. Since the existing code uses `tx_doorbell.send(System::Shutdown)?` with `From<mpsc::SendError> for SendError`, this returns `SendError::Disconnected` rather than panicking. But callers using `unwrap()` on Shutdown sends will panic.

**Impact:** Teardown code that assumes Shutdown send always succeeds can panic if the target actor has already failed.

**Severity:** MEDIUM — error handling in teardown paths.

---

## API Improvements

### Associated Error Types

The `Actor` trait now requires an associated `type Error: Display`:

```rust
impl Actor<MyData, MyCtrl, MyMgmt> for MyActor {
    type Error = String;  // or any type implementing Display
    fn handle_data(&mut self, msg: MyData) -> HandlerResult<Self::Error> { ... }
    // ...
}
```

- `HandlerError<E = String>` is generic with a default — existing code using `HandlerError` bare continues to work
- `HandlerResult<E = String>` likewise defaults to `String`
- The scheduler converts `E` to `String` (via `Display`) at the `PodPhase` boundary
- Actors can use domain-specific error enums internally for type-safe error handling

### Idle Timeout

`SchedulerParams::idle_timeout: Option<Duration>` controls how long the scheduler blocks when idle:

- `None` (default): blocks indefinitely on `recv()` — original behavior
- `Some(duration)`: uses `recv_timeout(duration)` — exits with `PodPhase::Completed` if no messages arrive

---

## Documented Limitations (By Design, But Dangerous)

### LIM-1: Cooperative Scheduling Cannot Preempt Long Handlers

**Test:** `greedy_data_handler_blocks_control_processing`

A handler that sleeps/computes for 50ms blocks ALL message processing on that scheduler thread. Control messages queued during the handler execution must wait until the handler returns and the next scheduling cycle begins. With burst_limit=10, 5 data messages at 50ms each = 250ms of control latency.

**Mitigation in core-term:** Handlers are fast (microseconds). This is safe for the 3-actor terminal architecture.

**Risk for general use:** Any handler that calls blocking I/O, sleeps, or does heavy computation will degrade the entire actor's responsiveness.

### LIM-2: `park(Busy)` Burns CPU at 100%

**Test:** `park_busy_causes_immediate_repoll`

When an actor returns `ActorStatus::Busy` from `park()`, the scheduler immediately repolls without blocking. If the actor keeps returning Busy, this is a hot spin loop at 100% CPU utilization.

**Mitigation:** The actor must eventually return Idle. No built-in circuit breaker.

### LIM-3: Data Lane Backpressure is Unbounded Spin-Yield

The data lane uses `thread::yield_now()` in a tight loop when the buffer is full. There is no exponential backoff (unlike Control/Management). If the consumer is slow, the producer burns CPU yielding.

**Mitigation:** Data buffer sizes are tuned for the expected throughput. The SPSC architecture means only one thread is affected.

### LIM-4: Message Loss Window During Pod Restart

`ServiceHandle::send()` during the disconnect window loses the message. The caller receives `Err(ServiceError::Reconnected)` and must decide whether to retry. There is no built-in retry, acknowledgment, or NACK mechanism.

**Mitigation:** Callers treat this like a TCP connection reset — retry if critical.

### LIM-5: No Handler Timeout / Watchdog

There is no mechanism to detect a handler that has hung (deadlock, infinite loop, blocked on I/O). The scheduler trusts the actor to return promptly. A hung handler is indistinguishable from a slow handler.

---

## What Passed (Green)

| Category | Tests | Status |
|----------|-------|--------|
| Priority scheduling order | `control_interleaving_during_data_burst` | Correct: C before D in each cycle |
| Burst limiting | `minimal_data_burst_preserves_control_responsiveness` | Correct: small burst doesn't starve |
| SPSC correctness | `spsc_capacity_one_stress`, `spsc_many_wraparounds_ordering` | 100K messages, ordering preserved |
| SPSC drop safety | `spsc_drop_with_heap_messages` | All 5 heap allocs freed |
| Sharded fairness | `sharded_extreme_imbalance` | Trickle producer not starved |
| Backoff timeout | `control_backoff_timeout_fires` | Timeout fires within bounds |
| Lifecycle: clean exit | `all_handles_dropped_causes_clean_exit` | Correct |
| Lifecycle: error exit | `recoverable_error_causes_failed_exit` | Correct |
| Lifecycle: fatal panic | `fatal_error_panics_scheduler` | Correct |
| Lifecycle: double shutdown | `double_shutdown_is_safe` | No panic |
| poll_once | `poll_once_processes_messages_without_blocking` | Non-blocking, correct |
| DrainAll | `drain_all_processes_pending_data_on_shutdown` | 50/50 messages drained |
| DrainAll timeout | `drain_all_timeout_prevents_hang` | Timeout fires, doesn't hang |
| DrainControl | `drain_control_drops_data_on_shutdown` | Control drained, data dropped |
| Multi-lane saturation | `all_lanes_saturated_simultaneously` | 2000/2000 messages, no loss |
| Panic containment | `handler_panic_disconnects_senders` | Sender sees Disconnected |
| ServiceHandle reconnect | `service_handle_reconnect_stress` | Transparent reconnect works |
| Kubelet frequency gate | `kubelet_rapid_restart_frequency_gate` | Budget enforced |

---

## Get-Well Plan

### P0: Fix GAP-1 — Shutdown Doorbell Deadlock [DONE]

Implemented Option B: `AtomicBool shutdown_requested` shared between `ActorHandle` and `ActorScheduler`. The doorbell now carries only `Wake` signals. `System::Shutdown` variant removed entirely.

### P1: Fix GAP-2 — Restarted Pod Liveness [DONE]

Added `idle_timeout: Option<Duration>` to `SchedulerParams`. Uses `recv_timeout()` when idle. Default: `None` (preserves existing behavior).

### P2: Fix GAP-3 — Shutdown Send Error Handling

**Effort:** Small (documentation + API guidance)

Document that `send(Message::Shutdown)` can return `Err(SendError::Disconnected)` if the target has already exited. Callers should use `let _ = handle.send(Message::Shutdown);` in teardown paths.

### P3: Add Handler Watchdog (LIM-5)

**Effort:** Medium-Large

Add an optional watchdog timer that detects handlers exceeding a configured deadline. The `poll_once` + Kubelet cooperative scheduling path already provides natural yielding between actors.

---

## Priority Matrix

| ID | Gap | Severity | Status | Fix |
|----|-----|----------|--------|-----|
| P0 | GAP-1: Shutdown deadlock | CRITICAL | **FIXED** | AtomicBool shutdown flag, non-blocking wake |
| P1 | GAP-2: Pod liveness timeout | HIGH | **FIXED** | `idle_timeout` in SchedulerParams |
| P2 | GAP-3: Shutdown error handling | MEDIUM | Open | Document + convenience method |
| P3 | LIM-5: Handler watchdog | LOW | Open | Debug-mode timing + warnings |

---

## Test Inventory

27 adversarial tests added in `tests/qa_adversarial.rs`:

```
 0. gap1_shutdown_never_deadlocks_with_full_doorbell              [GAP-1 FIX VERIFIED]
 1. greedy_data_handler_blocks_control_processing                 [LIM-1]
 2. park_busy_causes_immediate_repoll                             [LIM-2]
 3. control_interleaving_during_data_burst                        [PASS]
 4. data_send_unblocks_on_receiver_drop                           [PASS]
 5. control_backoff_timeout_fires                                 [PASS]
 6. spsc_capacity_one_stress                                      [PASS]
 7. spsc_drop_with_heap_messages                                  [PASS]
 8. spsc_many_wraparounds_ordering                                [PASS]
 9. double_shutdown_is_safe                                       [PASS]
10. shutdown_racing_with_data_sends                               [PASS]
11. all_handles_dropped_causes_clean_exit                         [PASS]
12. recoverable_error_causes_failed_exit                          [PASS]
13. fatal_error_panics_scheduler                                  [PASS]
14. poll_once_processes_messages_without_blocking                  [PASS]
15. poll_once_handles_shutdown                                    [PASS]
16. drain_all_processes_pending_data_on_shutdown                   [GAP-1 workaround]
17. drain_control_drops_data_on_shutdown                           [PASS]
18. drain_all_timeout_prevents_hang                                [GAP-1 workaround]
19. sharded_extreme_imbalance                                     [PASS]
20. kubelet_rapid_restart_frequency_gate                           [GAP-2 workaround]
21. service_handle_reconnect_stress                                [PASS]
22. error_during_drain_all_exits_cleanly                           [GAP-3]
23. all_lanes_saturated_simultaneously                             [PASS]
24. handler_panic_disconnects_senders                              [PASS]
25. park_fatal_panics_scheduler                                    [PASS]
26. minimal_data_burst_preserves_control_responsiveness            [PASS]
```
