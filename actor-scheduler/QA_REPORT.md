# Actor-Scheduler QA Report — "Enrique Havoc" Edition

**Date:** 2026-02-27
**Scope:** Cooperative green-thread scheduler, SPSC channels, Kubelet lifecycle, ServiceHandle
**Methodology:** Adversarial testing — 27 tests targeting breakpoints, not happy paths
**Result:** 3 production-severity gaps found, 5 documented limitations, 1 roadmap item (NACKs)

---

## Executive Summary

The actor-scheduler is **well-engineered for its intended use case** (3-thread terminal architecture at 155 FPS). The SPSC-per-producer design eliminates send-side contention, priority scheduling is correct, and the Kubelet restart machinery works. However, the cooperative nature creates **real hazards** that callers must know about, and one bug (GAP-1) is a latent deadlock that can hit production code.

**Verdict: Not production-ready as a general-purpose scheduler.**
Production-ready for core-term's constrained 3-actor topology, IF the caller follows strict ordering rules. The gaps below must be addressed for wider adoption.

---

## Gaps Found (Production Severity)

### GAP-1: Shutdown Deadlock When Doorbell Full [CRITICAL]

**Test:** `gap1_shutdown_after_data_deadlocks_without_running_scheduler`

**What:** `Message::Shutdown` uses `tx_doorbell.send(System::Shutdown)` — a **blocking** send on a capacity-1 channel. If the doorbell already contains a Wake (from any prior Data/Control/Management send), the Shutdown sender blocks forever until the scheduler consumes the Wake.

**Reproduction:**
```rust
let (tx, _rx) = ActorScheduler::new(10, 100);
tx.send(Message::Data(1)).unwrap();  // fills doorbell with Wake
tx.send(Message::Shutdown).unwrap(); // DEADLOCKS HERE
```

**Impact:** Any actor that sends data to a peer and then immediately sends Shutdown (without the peer's scheduler running to drain the doorbell) will deadlock. This is a realistic pattern in cleanup/teardown paths.

**Root cause:** Shutdown send assumes the scheduler is actively draining, but the doorbell is a rendezvous point with capacity 1. The Wake and Shutdown signals share the same channel.

**Severity:** CRITICAL — latent deadlock in teardown paths.

---

### GAP-2: Restarted Pods With No External Senders Block Forever [HIGH]

**Test:** `kubelet_rapid_restart_frequency_gate` (original version hung)

**What:** When the Kubelet restarts a pod, `TypedPodHandler::restart()` creates fresh `ActorHandle`s and publishes them to `PodSlot`s. But if no `ServiceHandle` calls `reconnect()` (or reconnects slowly), the new pod's scheduler blocks forever on `rx_doorbell.recv()` because no one sends any messages to it.

**Impact:** Kubelet hangs waiting for the pod to exit, blocking all other pod management on that Kubelet thread. With `RestartPolicy::Always`, this is an infinite hang.

**Root cause:** The scheduler has no concept of a "liveness timeout" — it blocks indefinitely waiting for work. Combined with cooperative scheduling (no preemption), there's no way for the Kubelet to interrupt a blocking pod.

**Severity:** HIGH — Kubelet thread starvation in restart scenarios.

---

### GAP-3: Race Between Handler Error and Shutdown Send [MEDIUM]

**Test:** `error_during_drain_all_exits_cleanly`

**What:** When the scheduler exits due to a handler error (Recoverable or Fatal), it drops the doorbell receiver. Any concurrent Shutdown send in flight gets `SendError::Disconnected`. Since the existing code uses `tx_doorbell.send(System::Shutdown)?` with `From<mpsc::SendError> for SendError`, this returns `SendError::Disconnected` rather than panicking. But callers using `unwrap()` on Shutdown sends will panic.

**Impact:** Teardown code that assumes Shutdown send always succeeds can panic if the target actor has already failed.

**Severity:** MEDIUM — error handling in teardown paths.

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

### P0: Fix GAP-1 — Shutdown Doorbell Deadlock

**Effort:** Small (1 function change)

**Option A (Recommended):** Change Shutdown to use `try_send` + retry with timeout, falling back to a secondary mechanism if the doorbell is full.

```rust
// Instead of:
self.tx_doorbell.send(System::Shutdown)?;

// Use:
match self.tx_doorbell.try_send(System::Shutdown) {
    Ok(()) => {}
    Err(TrySendError::Full(_)) => {
        // Doorbell has a pending Wake. The scheduler will process it
        // and re-check the doorbell. Queue shutdown and wake again.
        // Option: use a separate AtomicBool for shutdown signaling.
    }
    Err(TrySendError::Disconnected(_)) => return Err(SendError::Disconnected),
}
```

**Option B:** Use a dedicated `AtomicBool` for shutdown signaling (checked on every doorbell drain), keeping the doorbell exclusively for Wake. This separates the concerns cleanly.

**Option C:** Increase doorbell capacity to 2 (one Wake + one Shutdown). Simplest fix but masks the root cause.

### P1: Fix GAP-2 — Restarted Pod Liveness

**Effort:** Medium (new mechanism)

Add a **liveness timeout** to `ActorScheduler::run()`:

```rust
// In the main loop, replace blocking recv with recv_timeout:
match self.rx_doorbell.recv_timeout(self.liveness_timeout) {
    Ok(signal) => { /* process */ }
    Err(RecvTimeoutError::Timeout) => {
        // No messages for liveness_timeout period.
        // Return PodPhase::Completed (clean idle exit).
    }
    Err(RecvTimeoutError::Disconnected) => { /* all handles dropped */ }
}
```

This allows the Kubelet to detect idle pods and reclaim resources. The timeout should be configurable (default: 30s matching ServiceHandle reconnect timeout).

### P2: Fix GAP-3 — Shutdown Send Error Handling

**Effort:** Small (documentation + API guidance)

Document that `send(Message::Shutdown)` can return `Err(SendError::Disconnected)` if the target has already exited. Callers should use `let _ = handle.send(Message::Shutdown);` in teardown paths, or handle both Ok and Disconnected.

Consider adding a convenience method:
```rust
impl ActorHandle {
    /// Send shutdown, ignoring Disconnected (target already exited).
    pub fn shutdown(&self) { let _ = self.send(Message::Shutdown); }
}
```

### P3: Add Handler Watchdog (LIM-5)

**Effort:** Medium-Large

Add an optional watchdog timer that detects handlers exceeding a configured deadline. Options:

1. **Cooperative check:** Add a `check_deadline()` method actors can call periodically in long handlers. Cheapest but requires actor cooperation.
2. **Separate watchdog thread:** Spawns a monitor that checks if the scheduler has made progress (heartbeat). Can log warnings or trigger recovery.
3. **`poll_once` with Kubelet timeout:** Use poll_once from the Kubelet thread with a per-poll budget. Already partially supported.

### P4 (Roadmap): NACK-Based Reliability for Message Loss (LIM-4)

Per the user's direction: acknowledgment protocols are heavy. Instead of ACKs, consider a **NACK + forward error correction** approach inspired by Reed-Solomon over UDP:

1. **Sequence numbers per lane:** Each message gets a monotonic sequence number.
2. **Gap detection at receiver:** If the receiver sees seq 5 then seq 7, it knows seq 6 was lost. It sends a NACK for seq 6.
3. **Sender-side retransmit buffer:** Keep the last N messages in a ring buffer. On NACK, retransmit from the buffer.
4. **FEC parity:** For every K data messages, generate 1 parity message (XOR-based, lightweight). The receiver can reconstruct 1 lost message per K without a NACK roundtrip.

This gives at-most-once delivery for normal operation (zero overhead) with optional at-least-once recovery via NACKs when loss is detected. No ack overhead on the hot path.

**Trade-offs vs the current "TCP reset" model:**
- Pro: No message loss during pod restarts
- Pro: No need for caller retry logic
- Con: Requires retransmit buffer memory (bounded)
- Con: FEC adds ~1/K overhead per message
- Con: Adds complexity to the SPSC hot path

**Recommendation:** Implement as an opt-in wrapper (`ReliableServiceHandle`) rather than baking it into the core SPSC path. The current zero-overhead hot path is a strength.

### P5: Cooperative Scheduling Guardrails

**Effort:** Small (documentation + optional runtime check)

Since the scheduler is cooperative by design:

1. **Document the contract:** Handlers MUST return within X microseconds (recommend: < 100us for 155 FPS target).
2. **Optional timing check:** In debug builds, measure handler duration and log warnings if exceeded:
   ```rust
   #[cfg(debug_assertions)]
   let start = Instant::now();
   actor.handle_data(msg)?;
   #[cfg(debug_assertions)]
   if start.elapsed() > HANDLER_WARNING_THRESHOLD {
       eprintln!("[WARN] handle_data took {:?}", start.elapsed());
   }
   ```
3. **Consider `poll_once` as default:** For the Kubelet cooperative scheduling path, `poll_once` already provides natural yielding between actors. Document this as the recommended pattern for multi-actor single-thread scenarios.

---

## Priority Matrix

| ID | Gap | Severity | Effort | Fix |
|----|-----|----------|--------|-----|
| P0 | GAP-1: Shutdown deadlock | CRITICAL | Small | Separate shutdown signal from doorbell |
| P1 | GAP-2: Pod liveness timeout | HIGH | Medium | recv_timeout in scheduler loop |
| P2 | GAP-3: Shutdown error handling | MEDIUM | Small | Document + convenience method |
| P3 | LIM-5: Handler watchdog | LOW | Medium | Debug-mode timing + warnings |
| P4 | LIM-4: NACK reliability | ROADMAP | Large | ReliableServiceHandle wrapper |
| P5 | Cooperative guardrails | LOW | Small | Documentation + debug warnings |

---

## Test Inventory

27 adversarial tests added in `tests/qa_adversarial.rs`:

```
 0. gap1_shutdown_after_data_deadlocks_without_running_scheduler  [GAP-1 PROOF]
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
