# Design Doc: Actor Scheduler — Kubernetes/TCP/Erlang Supervision Model

## Metadata
- **Author**: jppittman
- **Status**: Draft
- **Created**: 2026-02-20
- **Reviewers**: —

---

## 1. Overview

### 1.1 Problem Statement

The current `actor-scheduler` / `troupe!` system conflates three distinct concerns:

1. **Lifecycle** — how actors start, transition through states, and stop
2. **Discovery** — how actors find each other (currently: static `Directory` of raw `ActorHandle`s)
3. **Restarts** — what happens when an actor exits unexpectedly (currently: nothing)

Because discovery is implemented as direct `ActorHandle` ownership, the SPSC channel _is_ the address. When an actor dies, every peer's handle becomes a dangling `SendError::Disconnected` with no recovery path. There is no supervision layer — `HandlerError::Recoverable` exits the scheduler and the thread silently terminates.

### 1.2 Goals

- Decouple lifecycle, discovery, and restarts into distinct, composable abstractions
- Preserve all existing SPSC/priority-lane performance on the hot send path (zero overhead)
- Support per-pod restart policies (`Always`, `OnFailure`, `Never`)
- Front-load all allocation and wiring cost at bootstrap — zero runtime overhead after `play()`
- Introduce a **Kubelet** — a single thread (optionally core-pinned) that manages pod lifecycle
  and can also run lightweight cooperative pods, avoiding one-OS-thread-per-actor overhead
- Static topology only: all pods and services declared at bootstrap. No dynamic deployment.

### 1.3 Non-Goals

- Dynamic pod/service registration at runtime
- Cross-process or distributed actors
- Preemptive scheduling within the kubelet (cooperative only)
- Changing the three-lane priority model (Control > Management > Data)
- Changing SPSC ring buffer implementation

---

## 2. Background

### 2.1 Current State

```
troupe! {
    engine: EngineActor [expose],
    vsync:  VsyncActor  [expose],
    display: DisplayActor [main],
}
```

Generates a `Directory` of raw `ActorHandle<D,C,M>` per actor. Each actor receives a
`Directory` at construction pointing directly to its peers' SPSC endpoints.

**Lifecycle**: three phases — `new()`, `exposed()`, `play()`. After `play()`, topology is frozen.

**Discovery**: static. If a peer's thread exits, the `ActorHandle` goes dead. No path back.

**Restarts**: none. `HandlerError::Recoverable` causes `scheduler.run()` to return. The thread
exits. Nobody is told.

### 2.2 Concept Mapping

| Kubernetes | Actor Scheduler | Notes |
|------------|----------------|-------|
| Node | Physical CPU core | The hardware unit of execution |
| Kubelet | Thread pinned to a core | Manages all pods on that core |
| Pod | Group of actors on one OS thread | Unit of scheduling and restart |
| Container | Individual actor (message handler) | Runs inside a pod |
| Service / ClusterIP | `ServiceHandle<D,C,M>` | Stable address across pod restarts |
| Deployment | `PodSpec` + restart policy | Desired state declaration |
| DNS / kube-proxy | `PodRegistry` | Name → current endpoint resolution |

**Intra-pod communication** (actors within the same pod / OS thread): synchronous, direct
function call or shared mutable state. No channel overhead. Actors sharing a pod share L1/L2
cache naturally.

**Inter-pod communication** (across OS threads): SPSC priority channels, exactly as today.

### 2.3 Prior Art

| System | Lifecycle | Discovery | Restarts |
|--------|-----------|-----------|----------|
| Erlang/OTP | Supervisor tree, `gen_server` | pid / registered name | `one_for_one`, `one_for_all`, `rest_for_one` |
| Kubernetes | Pod phases, Deployment controller | Service / ClusterIP / DNS | Restart policy per pod, kubelet watches |
| TCP/IP | Connection state machine (SYN→ESTABLISHED→CLOSED) | DNS → IP:port (stable address) | Client reconnects to same address after server restart |
| DPDK/SPDK | — | — | Core-pinned polling threads, cooperative task queues |

The key insight synthesizing all three:

> **SPSC channels are TCP connections. `ServiceHandle` is the ClusterIP. Pod restart = server
> restart. Sender reconnects to the same stable address, not the same socket.**

DHCP provides the bootstrap model: all addresses are assigned up-front at initialization time.
No dynamic discovery needed at runtime because the full topology is declared statically. Callers
pay the allocation cost once; after that everything is pre-wired.

---

## 3. Design

### 3.1 Architecture

```
Physical layout
─────────────────────────────────────────────────────────────────

  Core 0                     Core 1                  Core N
  ──────────────────         ──────────────────       ──────────
  Kubelet thread             OS thread                OS thread
  │                          │                        │
  ├─ Pod: RenderPod           ├─ Pod: EnginePod         ├─ ...
  │   ├─ Actor: Rasterizer    │   ├─ Actor: Engine
  │   └─ Actor: Config        │   └─ Actor: Vsync
  │  [cooperative, shared     │  [dedicated thread,
  │   L1/L2 cache]            │   SPSC to other pods]
  │
  └─ Pod controller loop
      exit_rx × all pods

  ← intra-pod: direct call / shared state (same thread, L1 cache) →
  ←──────── inter-pod: SPSC priority channels ────────────────────→

Bootstrap (DHCP phase)
─────────────────────────────────────────────────────────────────
  manifest! { ... }
    │
    ├── Allocates all SPSC ring buffers (inter-pod channels only)
    ├── Registers all PodIds in PodRegistry
    ├── Pre-connects all ServiceHandles  ← "DHCP lease assignment"
    └── Wires all exit channels to Kubelet

Runtime: inter-pod discovery
─────────────────────────────────────────────────────────────────

  Actor A (Pod X)             PodRegistry           Actor B (Pod Y)
  ───────────────             ───────────           ───────────────
  svc_b.send(msg)             PodId::Y →            [running]
      │  (hot path)           current handle
      └──────────────────────────────────────────────┘
                              (pre-connected at bootstrap)

  svc_b.send(msg)             PodId::Y →            [restarting]
      │  Disconnected
      └── registry.connect(PodId::Y)  ← blocks until Kubelet
          then retries send               calls publish()
```

### 3.2 New Types

#### `PodPhase` — lifecycle state machine

```rust
pub enum PodPhase {
    Pending,          // Declared, not yet started
    Running,          // Thread executing, accepting messages
    Terminating,      // Received Shutdown, draining
    Completed,        // Normal exit (Shutdown handled)
    Failed(String),   // HandlerError::Recoverable or Fatal
}
```

Emitted by `ActorScheduler` via an `exit_tx: Sender<PodPhase>` on every transition.
The scheduler already knows all transitions — this just makes them observable.

#### `RestartPolicy`

```rust
pub enum RestartPolicy {
    Always,      // Restart on any exit including Completed (OTP Permanent)
    OnFailure,   // Restart only on Failed phase (OTP Transient)
    Never,       // No restart (OTP Temporary)
}
```

#### `PodSpec` — the deployment descriptor

A pod is a **group of actors** that share an OS thread. Actors within a pod communicate
synchronously (direct calls or shared `&mut` state passed between handlers). Only
inter-pod communication uses SPSC channels.

```rust
/// A pod groups one or more actors onto a single OS thread.
/// Actors within the pod can communicate directly — no channels.
pub struct PodSpec {
    pub actors:       Vec<Box<dyn AnyActorSpec>>,  // The containers in this pod
    pub restart:      RestartPolicy,
    pub thread_model: ThreadModel,
}

pub enum ThreadModel {
    Dedicated,    // Spawn new OS thread — for latency-critical pods
    Cooperative,  // Run on Kubelet thread via poll_once() — for lightweight pods
    Main,         // Must run on calling thread (Cocoa/X11 constraint)
}
```

#### `ServiceHandle<D, C, M>` — stable discovery address

The ClusterIP equivalent. Stateful (like a TCP socket), `&mut self` on send.
Each holder of a `ServiceHandle` has its own independent SPSC connection to the pod
(N holders = N SPSC channels into `ShardedInbox`, same as N TCP clients to one server).

```rust
pub struct ServiceHandle<D, C, M> {
    // The live "socket" — pre-connected at bootstrap
    connection: ActorHandle<D, C, M>,
    // How to get a new connection after pod restart
    registry:   Arc<PodRegistry>,
    pod_id:     PodId,
}

impl<D, C, M> ServiceHandle<D, C, M> {
    /// Hot path: one SPSC send, no indirection.
    /// Cold path (pod restarted): blocks in registry.connect() until
    /// Kubelet publishes the new endpoint, then retries.
    pub fn send(&mut self, msg: Message<D, C, M>) -> Result<(), SendError> {
        match self.connection.send(msg) {
            Err(SendError::Disconnected(m)) => {
                self.connection = self.registry.connect(self.pod_id)?;
                self.connection.send(m)
            }
            r => r,
        }
    }
}
```

#### `PodRegistry` — the DNS/DHCP server

```rust
pub struct PodRegistry {
    // PodId → (Sender endpoint factory, readiness condvar)
    // Written by Kubelet after restart, read by reconnecting ServiceHandles
    inner: Vec<PodSlot>,
}

impl PodRegistry {
    /// Called by Kubelet after spawning fresh pod.
    /// Publishes new ActorHandle, unblocks any waiting connect() calls.
    pub fn publish<D,C,M>(&self, id: PodId, handle: ActorHandle<D,C,M>);

    /// Called by ServiceHandle on Disconnected.
    /// Blocks until Kubelet calls publish() for this PodId.
    pub fn connect<D,C,M>(&self, id: PodId) -> Result<ActorHandle<D,C,M>, PodGone>;
}
```

At bootstrap (DHCP phase): all slots are pre-populated. `connect()` never blocks during
normal operation — only during a pod restart window.

#### `ActorScheduler::poll_once()` — cooperative pod support

New non-blocking step for `ThreadModel::Cooperative` pods running on the Kubelet thread:

```rust
impl<D, C, M> ActorScheduler<D, C, M> {
    /// Existing blocking loop — unchanged for dedicated-thread pods.
    pub fn run(&mut self, actor: &mut impl Actor<D, C, M>) -> PodPhase { ... }

    /// Single non-blocking drain cycle. Returns immediately.
    /// Used by Kubelet to cooperatively run multiple lightweight actors.
    pub fn poll_once(&mut self, actor: &mut impl Actor<D, C, M>) -> PodPhase { ... }
}
```

`poll_once()` does one priority-ordered drain cycle (Control → Management → Data up to
burst limit), calls `actor.park()`, and returns. The Kubelet round-robins across cooperative
pods between controller duties.

#### `Kubelet` — the node agent

```rust
pub struct Kubelet {
    pods:          Vec<ManagedPod>,
    cooperative:   Vec<CooperativePod>,  // ThreadModel::Cooperative pods
    registry:      Arc<PodRegistry>,
    max_restarts:  u32,
    within:        Duration,
}

impl Kubelet {
    /// Optionally pin this thread to a specific CPU core.
    pub fn with_core_affinity(self, core: CoreId) -> Self;

    /// Main Kubelet loop. Runs on dedicated thread (optionally pinned).
    pub fn run(self);
}
```

The Kubelet loop:

```
loop {
    // 1. Poll exit channels (non-blocking)
    for pod in &mut self.pods {
        if let Some(phase) = pod.exit_rx.try_recv() {
            self.handle_transition(pod, phase);  // may restart
        }
    }

    // 2. Run cooperative pods (one poll_once() each, round-robin)
    for pod in &mut self.cooperative {
        pod.scheduler.poll_once(&mut pod.actor);
    }

    // 3. Park if nothing to do
    if nothing_active {
        self.park_until_doorbell();
    }
}
```

Core pinning uses `libc::sched_setaffinity` (Linux) / `pthread_setaffinity_np` (macOS).
A pinned Kubelet gets deterministic lifecycle event latency and shares an L1/L2 cache
region with its cooperative pods. For high-restart-frequency scenarios, this eliminates
scheduling jitter from the OS.

### 3.3 Macro Interface

`troupe!` is extended to express pods (groups of actors on a thread) with lifecycle
annotations. Single-actor pods use existing syntax unchanged. Multi-actor pods use a
`pod { }` block.

```rust
troupe! {
    // Single-actor pod (existing syntax — backward compatible)
    // One actor = one pod = one dedicated OS thread
    vsync: VsyncActor [expose, restart = Always],

    // Multi-actor pod: Engine + Config share one OS thread
    // Intra-pod communication is direct (no SPSC overhead)
    engine_pod: pod {
        thread = Dedicated,
        restart = OnFailure,
        actors {
            engine: EngineActor [expose],
            config: ConfigActor,             // not exposed cross-pod
        }
    },

    // Main-thread pod: Cocoa/X11 event loop constraint
    display: DisplayActor [main, restart = Never],

    // Cooperative pod: runs on Kubelet thread (core 0)
    telemetry_pod: pod {
        thread = Cooperative,
        restart = OnFailure,
        actors {
            metrics: MetricsActor,
            logger:  LogActor,
        }
    },
}
```

The macro generates:
- `ServiceHandle<...>` in `Directory` for inter-pod actors (instead of raw `ActorHandle`)
- Intra-pod actors get direct `&mut` references, not channels
- Per-pod `PodSpec` with factory closure
- `PodRegistry` pre-populated at `new()`
- `Kubelet` wired with all exit channels and `PodSpec`s
- `play()` boots the Kubelet thread (optionally core-pinned), then proceeds as today

### 3.4 Bootstrap Sequence (DHCP Phase)

```
Troupe::new()
  │
  ├── Allocate all SPSC ring buffers (all pods, all priority lanes)
  ├── Construct all ActorHandle endpoints
  ├── Populate PodRegistry with all endpoints       ← "DHCP lease assignment"
  ├── Build all ServiceHandles (pre-connected)       ← zero connect() latency at runtime
  ├── Construct Directories (now ServiceHandle-based)
  └── Wire all exit_tx channels to Kubelet

Troupe::play()
  │
  ├── Spawn Kubelet thread (optionally pinned to core)
  ├── Spawn dedicated-thread pods (existing behavior)
  ├── Kubelet takes ownership of cooperative pods
  └── Main-thread pod runs on calling thread (existing behavior)
```

After `play()` returns, all handles are live. Discovery never blocks. Restarts are handled
transparently by the Kubelet.

### 3.5 Restart Sequence

```
Pod failure
  │
  ├── ActorScheduler emits PodPhase::Failed via exit_tx
  ├── Kubelet receives on exit_rx (non-blocking poll)
  ├── Check restart policy + frequency gate (maxR / maxT)
  │     ├── Policy allows restart:
  │     │     ├── Call PodSpec::factory() → fresh actor instance
  │     │     ├── Allocate fresh SPSC channels (ActorBuilder)
  │     │     ├── Spawn new thread (or enqueue in cooperative list)
  │     │     ├── registry.publish(pod_id, new_handle)
  │     │     │     └── Unblocks any ServiceHandle::send() waiting in connect()
  │     │     └── Pod transitions to PodPhase::Running
  │     └── Policy denies or frequency exceeded:
  │           └── Escalate to parent Kubelet (if nested) or panic
  └── Senders using ServiceHandle:
        ├── Currently blocked in registry.connect() → unblock, get new handle, retry
        └── Not yet hit Disconnected → will hit it on next send, then reconnect lazily
```

### 3.6 Error Handling

| Scenario | Behavior |
|----------|----------|
| Pod exits normally, `RestartPolicy::Never` | `PodPhase::Completed`, no restart |
| Pod exits normally, `RestartPolicy::Always` | Restart immediately |
| Pod exits with `Failed`, `RestartPolicy::OnFailure` | Restart |
| Pod exits with `Failed`, `RestartPolicy::Never` | No restart, registry marks pod as `PodGone` |
| Restart frequency exceeded (maxR in maxT) | Escalate to parent Kubelet or panic |
| `ServiceHandle::send()` while pod restarting | Block in `registry.connect()` until Running |
| `ServiceHandle::send()` to permanently dead pod | `Err(SendError::Disconnected)` immediately |
| `ThreadModel::Main` pod fails | Panic (cannot restart, Cocoa constraint) |

---

## 4. Implementation Plan

### 4.1 Task Breakdown

| Task | File(s) | Deps | Size | Notes |
|------|---------|------|------|-------|
| T1: `PodPhase` + exit channel on `ActorScheduler` | `actor-scheduler/src/lib.rs` | — | S | `run()` returns `PodPhase`; add `exit_tx` field |
| T2: `RestartPolicy` enum | `actor-scheduler/src/lib.rs` | — | S | Three variants |
| T3: `PodId` + `PodRegistry` | `actor-scheduler/src/registry.rs` | T1 | M | Pre-populated at bootstrap; condvar for `connect()` |
| T4: `ServiceHandle<D,C,M>` | `actor-scheduler/src/service.rs` | T3 | M | `&mut self send()`, reconnect on Disconnected |
| T5: `poll_once()` on `ActorScheduler` | `actor-scheduler/src/lib.rs` | T1 | S | Non-blocking single drain cycle |
| T6: `ThreadModel` enum + `PodSpec` | `actor-scheduler/src/pod.rs` | T2, T5 | M | Factory closure, buffer config |
| T7: `Kubelet` struct + run loop | `actor-scheduler/src/kubelet.rs` | T3, T4, T6 | L | Controller loop, cooperative pod runner, core affinity |
| T8: Update `troupe!` macro | `actor-scheduler-macros/src/lib.rs` | T4, T6, T7 | L | New annotations, generate Kubelet + ServiceHandles |
| T9: Update `Directory` to use `ServiceHandle` | `actor-scheduler-macros/src/lib.rs` | T8 | M | Replace ActorHandle fields |
| T10: Migrate `engine_troupe.rs` | `pixelflow-runtime/src/engine_troupe.rs` | T9 | M | First consumer |
| T11: Migrate `core-term` actors | `core-term/src/terminal_app.rs`, `io/` | T9 | M | Second consumer |

### 4.2 Parallelization

```
T1 ──────────────────────────────────────────────┐
                                                  │
T2 ──────────────────────────────────────┐        │
                                          │        │
T1+T2 done → T3 ──────┬──▶ T4 ──────────┤        │
                        │                 │        │
              T5 ───────┘                 │        │
                                          │        │
T6 (needs T2, T5) ───────────────────────┤        │
                                          ▼        ▼
                              T7 ──────▶ T8 ──▶ T9 ──▶ T10, T11
```

T1 and T2 are independent and can start immediately. T3–T6 can proceed in parallel after
their deps. T7 is the integration point; T8 onward is sequential macro + consumer migration.

### 4.3 Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| `registry.connect()` deadlock if Kubelet stalls | Low | High | Timeout in `connect()`, Kubelet health check |
| Core affinity API not portable | Medium | Low | Feature-flag; graceful fallback to unpinned |
| `poll_once()` cooperative pod starvation | Medium | Medium | Kubelet enforces max iterations per pod per loop |
| `ThreadModel::Main` actor failure unrecoverable | Known | High | Document, panic with clear message |
| Macro complexity explosion | Medium | Medium | Incremental: annotations optional, defaults preserved |

---

## 5. Testing Strategy

### 5.1 Unit Tests

- `PodRegistry`: publish/connect ordering, concurrent connect() unblocking
- `ServiceHandle`: hot-path send, reconnect-on-Disconnected, `PodGone` propagation
- `Kubelet`: restart policy enforcement, frequency gate (maxR/maxT), cooperative pod round-robin
- `poll_once()`: burst limit respected, priority order preserved, returns immediately

### 5.2 Integration Tests

- Full troupe restart: kill pod, verify ServiceHandle senders block then succeed
- `RestartPolicy::Always` loop: pod fails repeatedly, verify restart up to frequency limit
- Core-pinned Kubelet: verify affinity is set (Linux: `/proc/self/status` cpus_allowed)
- `ThreadModel::Cooperative`: multiple lightweight pods share Kubelet, no starvation

---

## 6. Alternatives Considered

| Alternative | Pros | Cons | Why Not |
|-------------|------|------|---------|
| Seqlock `ActorRef` (indirection on every send) | True `OneForOne` within troupe | ~15–30ns per send on hot path, breaks wait-free guarantee | Overhead unacceptable; lazy reconnect is free |
| Supervisor as an Actor (OTP style) | Familiar Erlang model | Circular handle problem; all supervision over priority lanes | Priority lanes are for application messages, not system signals |
| Per-pod controller thread | Simpler per-pod logic | One OS thread per pod just for lifecycle watching — expensive | Kubelet amortizes cost across all pods |
| Dynamic topology at runtime | More flexible | Impossible with static SPSC pre-allocation; breaks DHCP model | Explicitly non-goal |

---

## 7. Open Questions

- [ ] **Core affinity granularity**: One Kubelet per NUMA node? Or one per machine? For now: one
  per troupe. Revisit if NUMA topology matters.
- [ ] **Kubelet cooperative pod starvation policy**: Fixed round-robin or weighted by lane pressure?
- [ ] **`PodGone` vs panic**: When a `Never`-restart pod dies and a sender hits it, should
  `ServiceHandle` return `Err(PodGone)` or propagate as panic? Prefer error — let caller decide.
- [ ] **Nested Kube lets / troupe trees**: If a troupe contains a sub-troupe, does each have its
  own Kubelet? Yes — each troupe is a namespace with its own Kubelet. Cross-troupe handles
  are already `exposed()` `ServiceHandle`s; restart propagation stops at the troupe boundary
  unless the parent Kubelet is explicitly watching.
