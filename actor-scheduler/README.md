# actor-scheduler

A blazingly fast, priority-aware actor scheduler built on wait-free SPSC channels. Zero external dependencies beyond `std`.

## Why

Most actor frameworks treat message passing as a solved problem: throw messages into an mpsc channel, drain them in order, done. This breaks down when you need **priority**. A terminal emulator receiving a million bytes of `ls -R` output must still process your keystroke instantly. A render loop must never miss a vsync signal because the data pipe is full.

`actor-scheduler` solves this with three priority lanes, sharded SPSC channels, and Bayesian-optimized scheduling parameters — all in ~1,500 lines of `std`-only Rust.

## Benchmarks

Run `cargo bench -p actor-scheduler` to measure the scheduler on the current
codebase and hardware. The benchmark suites cover channel throughput, send and
roundtrip latency, priority-lane behavior, and adversarial workloads.

## Architecture

```
Producer A ──[SPSC]──┐
Producer B ──[SPSC]──┤  ShardedInbox (Control)
Producer C ──[SPSC]──┘       │
                             ├──→ Scheduler ──→ Actor
Producer A ──[SPSC]──┐      │
Producer B ──[SPSC]──┤  ShardedInbox (Data)
Producer C ──[SPSC]──┘       │
                             │
       Doorbell (mpsc, cap=1) ───┘  Wake/Shutdown signals
```

### Three priority lanes

| Lane | Priority | Backpressure | Use case |
|------|----------|--------------|----------|
| **Control** | Highest | Exponential backoff + jitter | Keystrokes, resize, close |
| **Management** | Medium | Exponential backoff + jitter | Config changes, lifecycle |
| **Data** | Lowest | Spin-yield (bounded buffer) | PTY output, frame data |

### Scheduling loop

```
loop {
    1. Drain Control    (half burst budget)
    2. Drain Management (burst budget)
    3. Drain Control    (remaining budget)
    4. Drain Data       (burst budget)
    5. park()           — actor yields to OS / event loop
}
```

Control gets two passes per cycle for priority, but all lanes are burst-limited to prevent monopolization.

### Sharded SPSC

Instead of N producers contending on one mpsc lock, each producer gets a dedicated wait-free SPSC ring buffer. The consumer drains all shards round-robin. Like shuffle-sharding in a load balancer: a noisy producer fills its own shard but cannot affect others.

```rust
let mut builder = ActorBuilder::<Data, Control, Mgmt>::new(1024, None);
let handle_a = builder.add_producer();  // dedicated SPSC channels
let handle_b = builder.add_producer();  // independent, no contention
let mut scheduler = builder.build();    // seals — no more producers
```

### Bayesian-optimized parameters

The benchmark tooling can use Bayesian optimization (Gaussian Process surrogate
and Expected Improvement acquisition) to explore `SchedulerParams`. Re-run the
optimizer when scheduler behavior or target workloads change rather than relying
on previously recorded results.

## Usage

### Single-producer actor

```rust
use actor_scheduler::{
    ActorScheduler, Message, Actor,
    ActorStatus, SystemStatus, HandlerResult, HandlerError,
};

struct MyActor;

impl Actor<String, String, String> for MyActor {
    fn handle_data(&mut self, msg: String) -> HandlerResult {
        println!("data: {msg}");
        Ok(())
    }
    fn handle_control(&mut self, msg: String) -> HandlerResult {
        println!("control: {msg}");
        Ok(())
    }
    fn handle_management(&mut self, msg: String) -> HandlerResult {
        println!("mgmt: {msg}");
        Ok(())
    }
    fn park(&mut self, _: SystemStatus) -> Result<ActorStatus, HandlerError> {
        Ok(ActorStatus::Idle)
    }
}

let (tx, mut rx) = ActorScheduler::<String, String, String>::new(100, 1024);

std::thread::spawn(move || {
    let mut actor = MyActor;
    rx.run(&mut actor);
});

tx.send(Message::Control("resize".into())).unwrap();
tx.send(Message::Data("bytes".into())).unwrap();
```

### Multi-producer (troupe pattern)

```rust
use actor_scheduler::{ActorBuilder, Message, ShutdownMode};

let mut builder = ActorBuilder::<Vec<u8>, (), ()>::new(4096, None);

let pty_handle = builder.add_producer();     // PTY reader thread
let input_handle = builder.add_producer();   // input thread
let timer_handle = builder.add_producer();   // vsync timer

let mut scheduler = builder.build();
// Each handle has zero-contention SPSC channels to the actor
```

### Troupe macro (actor groups)

```rust
use actor_scheduler::troupe;

troupe! {
    engine: EngineActor [expose],    // handle exposed to parent
    vsync: VsyncActor,               // internal only
    display: DisplayActor [main],    // runs on calling thread
}

run().expect("troupe failed");
```

### Shutdown modes

```rust
use actor_scheduler::ShutdownMode;
use std::time::Duration;

// Drop everything immediately (default)
ShutdownMode::Immediate;

// Process remaining control + management, drop data
ShutdownMode::DrainControl;

// Process all pending messages, with timeout fallback
ShutdownMode::DrainAll { timeout: Duration::from_secs(1) };
```

## Design decisions

**Why SPSC over mpsc?** Lock-free sends avoid contention between producers. The tradeoff is that producers must be registered at init time.

**Why not crossbeam/flume/tokio?** Zero dependencies. The SPSC ring buffer is ~200 lines. The entire crate is ~1,500 lines. We need a priority scheduler, not a general-purpose channel — building it lets us fuse priority scheduling directly into the drain loop.

**Why Bayesian optimization?** The parameters interact through non-linear constraints, making exhaustive grid search impractical. Bayesian optimization provides a way to explore the configuration space for the current workload.

**Why burst limiting?** Without it, a control flood starves data completely. With it, the scheduler guarantees forward progress on all lanes every cycle. The burst budget is the fundamental fairness knob.

## Benchmarks

```bash
# All benchmarks
cargo bench -p actor-scheduler

# Individual suites
cargo bench -p actor-scheduler --bench bench_throughput
cargo bench -p actor-scheduler --bench bench_latency
cargo bench -p actor-scheduler --bench bench_adversarial
cargo bench -p actor-scheduler --bench bench_spsc_vs_mpsc

# Bayesian parameter optimization (slow, ~minutes)
cargo bench -p actor-scheduler --bench bench_optimize
```

## License

Apache-2.0
