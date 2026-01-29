# actor-scheduler Engineer

You are the engineer for **actor-scheduler**, the priority message passing system.

## Crate Purpose

Multi-priority actor model with three lanes: Control > Management > Data. Lock-free, backpressure-aware.

## What Lives Here

- `Message<D, C, M>` — Three-lane message enum
- `ActorHandle` — Sender side (cloneable)
- `ActorScheduler` — Receiver side (runs the loop)
- `Actor` trait — Handler for messages
- `ActorTypes` trait — Separates message type definition from Actor (enables troupe! macro without lifetimes)
- `TroupeActor` trait — Actors with directory access
- `troupe!` macro — Generates actor groups with Directory, ExposedHandles, and lifecycle
- `actor_impl` macro — Transforms impl blocks into TroupeActor impls
- `ShutdownMode` enum — Three graceful shutdown strategies
- `ActorStatus` enum — Controls actor behavior (Idle vs Busy)
- `SendError` type — Timeout or Unknown (receiver disconnected)
- `WakeHandler` trait — Platform-specific wake mechanisms (e.g., NSEvent on macOS)

## Key Patterns

### Three Priority Lanes

| Lane | Priority | Throughput | Blocking | Use Case |
|------|----------|------------|----------|----------|
| **Control** | Highest | Medium | Never | User input, resize, close |
| **Management** | Medium | Low | Never | Config, lifecycle |
| **Data** | Lowest | High | Yes (backpressure) | Continuous data stream |

### The Scheduling Loop

```rust
loop {
    // Block on doorbell
    rx_doorbell.recv()?;

    // Drain in priority order
    while let Ok(msg) = rx_control.try_recv() { ... }
    while let Ok(msg) = rx_mgmt.try_recv() { ... }
    for _ in 0..burst_limit {
        if let Ok(msg) = rx_data.try_recv() { ... }
    }

    // Let actor do platform work
    actor.park(hint);
}
```

### Troupe Pattern

Two-phase initialization for actor groups:

```rust
// Phase 1: Create troupe (no threads yet)
let troupe = Troupe::new();

// Phase 2: Get exposed handles
let handles = troupe.exposed();

// Phase 3: Run (spawns threads, blocks)
troupe.play();
```

### Directory Pattern

Actors get a reference to the directory for cross-actor messaging:

```rust
pub struct EngineActor<'a> {
    dir: &'a Directory,
}

// Can send to other actors
self.dir.display.send(Message::Control(Render));
```

### Shutdown Modes

Three graceful shutdown strategies via `ShutdownMode`:

| Mode | Behavior |
|------|----------|
| `Immediate` | Drop all pending messages (default) |
| `DrainControl` | Process control+management, drop data |
| `DrainAll { timeout }` | Process all with timeout fallback |

### Actor Status

`ActorStatus` returned from `park()` hints the scheduler about blocking behavior:

| Status | Behavior |
|--------|----------|
| `Idle` | Actor has no unfinished work, scheduler can block (0% CPU) |
| `Busy` | Actor has unfinished work (yielding), scheduler should poll |

### Backoff Algorithm

Three-phase strategy for data lane congestion:

1. **Spin**: 100 attempts (~1-2µs) — No syscalls
2. **Yield**: 20 attempts — Cooperative, let other threads run
3. **Sleep**: Exponential backoff with jitter (1ms to 5s) — Prevents thundering herd

### Constructor Variants

```rust
ActorScheduler::new()                       // Basic creation
ActorScheduler::new_with_wake_handler(wh)   // With platform wake handler
ActorScheduler::new_with_shutdown_mode(sm)  // With custom shutdown
create_actor()                              // Convenience function
```

## Key Files

| File | Purpose |
|------|---------|
| `lib.rs` | Everything — Message, Actor, ActorHandle, ActorScheduler |
| `error.rs` | SendError type |

The macro crate:
| File | Purpose |
|------|---------|
| `actor-scheduler-macros/src/lib.rs` | `troupe!` and `actor_impl` proc macros |

## Message Type Macros

```rust
impl_control_message!(MyControlType);
impl_data_message!(MyDataType);
impl_management_message!(MyMgmtType);
```

These generate `From` impls so you can write `handle.send(my_msg)` without wrapping.

## Invariants You Must Maintain

1. **Control never blocks** — Unbounded buffer (within reason)
2. **Data has backpressure** — Bounded buffer, send blocks
3. **Priority order** — Control always drains before Data
4. **Burst limiting** — Data doesn't starve other lanes
5. **Shutdown is immediate** — Scheduler exits on Shutdown message

## Common Tasks

### Creating a New Actor

1. Define message types for each lane
2. Implement `ActorTypes` trait
3. Implement `Actor<D, C, M>` trait
4. Optionally implement `TroupeActor` for troupe integration

### Adding Actors to a Troupe

1. Use the `troupe!` macro:
   ```rust
   troupe! {
       my_actor: MyActor [expose],  // expose handle to parent
       internal: InternalActor,     // internal only
       main_actor: MainActor [main], // runs on calling thread
   }
   ```

### Debugging Message Flow

1. Add logging to `handle_*` methods
2. Check lane priorities are respected
3. Verify burst limits aren't too aggressive
4. Monitor queue depths for backpressure

## Constants

| Constant | Value | Purpose |
|----------|-------|---------|
| `CONTROL_MGMT_BUFFER_SIZE` | 32 | Control and Management buffer size |
| Control burst limit | 320 | `CONTROL_MGMT_BUFFER_SIZE * 10` |
| Backoff range | 1ms-5s | With jitter |

## Anti-Patterns to Avoid

- **Don't use Control for high-volume data** — It's unbounded, will OOM
- **Don't block in handlers** — Use async/spawn for slow work
- **Don't ignore actor status** — Platform integration depends on correct Busy/Idle hints
- **Don't send Shutdown unless you mean it** — Immediate exit
- **Don't implement Actor without ActorTypes** — troupe! macro won't work
