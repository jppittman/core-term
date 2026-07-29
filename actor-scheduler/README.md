# actor-scheduler

A blazingly fast, priority-aware actor scheduler built on wait-free SPSC channels. Zero external dependencies beyond `std`.

## Why

Most actor frameworks treat message passing as a solved problem: throw messages into an mpsc channel, drain them in order, done. This breaks down when you need **priority**. A terminal emulator receiving a million bytes of `ls -R` output must still process your keystroke instantly. A render loop must never miss a vsync signal because the data pipe is full.

`actor-scheduler` solves this with three priority lanes, sharded SPSC channels, and Bayesian-optimized scheduling parameters — all in ~1,500 lines of `std`-only Rust.

## Performance

Measured on CI hardware (Linux 4.4.0). Run `cargo bench -p actor-scheduler` to reproduce.

### Channel throughput

| Benchmark | SPSC | std mpsc | Speedup |
|-----------|------|----------|---------|
| 1M messages (single producer) | 95.2M msg/s | 49.0M msg/s | **1.9x** |
| 200K messages (4 producers) | 16.4M msg/s | 7.8M msg/s | **2.1x** |
| 400K messages (8 producers) | 13.5M msg/s | 4.1M msg/s | **3.3x** |

Sharded SPSC scales linearly with producer count. MPSC degrades under contention.

### Send latency

| Scenario | SPSC | std mpsc | Speedup |
|----------|------|----------|---------|
| Uncontended | **12ns** | 34ns | 2.8x |
| 2 contenders (sharded) | **1.5ns** | 10ns | 7x |
| 4 contenders (sharded) | **1.4ns** | 10ns | 7x |

Sharded SPSC sends are wait-free: a single `AtomicUsize::store(Release)`. No CAS, no retry loop, no contention between producers.

### Full scheduler (3-lane priority)

| Metric | Result |
|--------|--------|
| Data throughput (100K msgs) | 15.7M msg/s |
| Control throughput (10K msgs) | 7.0M msg/s |
| Mixed workload (30K msgs) | 12.0M msg/s |
| Roundtrip latency (SPSC) | 9.6us |
| Control latency under data flood | 9.9us |
| Management latency under control flood | 473ns |

All measured through the full priority scheduler with burst limiting, starvation protection, and doorbell wake signaling.

### Adversarial resilience

Under continuous control flood from 4 concurrent attackers, the data lane maintains **100% delivery** with no starvation. Burst limiting caps each lane per wake cycle, guaranteeing forward progress on all lanes.

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

The default `SchedulerParams` were found by Bayesian optimization (Gaussian Process surrogate + Expected Improvement acquisition) over 8 weighted metrics with hard domain constraints:

- **Frame budget**: min backoff under 6.45ms (one frame at 155 FPS)
- **Degradation window**: total backoff cascade under 12s
- **Jitter effectiveness**: spread >= 20% for thundering herd prevention
- **Backpressure delay**: buffer <= 128 for fast overload detection

Results vs hand-tuned baseline: 73% latency reduction under load, 89% control throughput improvement, 100% fairness maintained, all constraint penalties zero.

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

**Why SPSC over mpsc?** Lock-free sends (12ns vs 34ns), linear scaling with producers, no contention on the hot path. The tradeoff is that producers must be registered at init time.

**Why not crossbeam/flume/tokio?** Zero dependencies. The SPSC ring buffer is ~200 lines. The entire crate is ~1,500 lines. We need a priority scheduler, not a general-purpose channel — building it lets us fuse priority scheduling directly into the drain loop.

**Why Bayesian optimization?** 10 interacting parameters with non-linear constraints (frame budget, degradation window, jitter spread). Grid search is exponential. Manual tuning found local optima. BO with domain constraint penalties found configurations that are strictly better on every metric.

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
