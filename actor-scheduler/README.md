# actor-scheduler

`actor-scheduler` is a priority-aware actor runtime for workloads where control traffic must
remain responsive while data traffic is busy. It provides two layers that share the same
three-lane scheduling model:

1. Dedicated OS-thread actors backed by producer-sharded SPSC queues.
2. Hosted Mealy transducers (“green actors”) that run one transition at a time inside an
   ordinary actor.

The runtime crate uses `std` plus the workspace `actor-scheduler-macros` proc-macro crate.
Criterion is used by its benchmark targets.

## Dedicated-thread scheduler

Each producer registered with `ActorBuilder` receives a dedicated SPSC queue for every lane.
The consumer drains those shards round-robin, avoiding a shared producer-side queue lock and
isolating a full shard to its producer.

| Lane | Priority | Typical use |
|---|---|---|
| Control | Highest | Keystrokes, resize, close |
| Management | Middle | Configuration and lifecycle |
| Data | Lowest | PTY bytes and frame data |

One scheduling cycle visits Control, Management, Control again, then Data. Every visit is
burst-limited: Control gets lower latency without being allowed to prevent Data from making
forward progress.

```text
producer A ── SPSC shards ──┐
producer B ── SPSC shards ──┼── ShardedInbox ── ActorScheduler ── Actor
producer C ── SPSC shards ──┘                         ▲
                                                      │
                                             bounded doorbell
```

Producers are registered before `build()` seals the scheduler. Topology is static after that
point.

### Single producer

```rust
use actor_scheduler::{
    Actor, ActorScheduler, ActorStatus, HandlerError, HandlerResult, Message, SystemStatus,
};

struct Counter(u64);

impl Actor<u64, (), ()> for Counter {
    fn handle_data(&mut self, value: u64) -> HandlerResult {
        self.0 += value;
        Ok(())
    }

    fn handle_control(&mut self, _msg: ()) -> HandlerResult {
        Ok(())
    }

    fn handle_management(&mut self, _msg: ()) -> HandlerResult {
        Ok(())
    }

    fn park(&mut self, _status: SystemStatus) -> Result<ActorStatus, HandlerError> {
        Ok(ActorStatus::Idle)
    }
}

let (handle, mut scheduler) = ActorScheduler::<u64, (), ()>::new(100, 1024);

std::thread::spawn(move || {
    let mut counter = Counter(0);
    let exit = scheduler.run(&mut counter);
    assert!(!exit.is_failed(), "counter stopped: {exit}");
});

handle.send(Message::Data(1)).unwrap();
```

For multiple producers, register each one explicitly:

```rust
use actor_scheduler::ActorBuilder;

let mut builder = ActorBuilder::<Vec<u8>, (), ()>::new(4096, None);
let pty = builder.add_producer();
let input = builder.add_producer();
let scheduler = builder.build();
```

`troupe!` builds statically declared actor groups on top of this layer. It creates the actor
directory and dedicated SPSC handles at startup; it is not a dynamic actor registry.

## Hosted Mealy transducers

The green layer represents an actor as a typed state machine:

```rust
pub trait Transducer {
    type Control;
    type Management;
    type Data;
    type Out: Default;

    fn step_data(&mut self, msg: Self::Data) -> Result<Self::Out, HandlerError>;
}
```

A `Node` owns a `Transducer`, its inbox lanes, typed output `Wiring`, and at most one pending
output word. `Node::poll()` performs at most one transition. It always retries a blocked
outbox before consuming another input, so downstream backpressure stops the producer without
suspending a partially executed handler.

The `ports!` macro generates an output word and matching wiring. Ports are blocking by
default; `[drop]` ports deliberately discard on a full target, and one `[self]` port may carry
a continuation without entering a queue.

```rust
use actor_scheduler::ports;

struct RenderCommand;
struct ParserInput;

ports! {
    Parser {
        render: RenderCommand,
        write: Vec<u8> [drop],
        continue_parse: ParserInput [self],
    }
}
```

`Topology` validates the static graph before it runs. A cycle made entirely of blocking edges
is rejected; a cycle closed by a droppable edge or continuation cannot park every participant
on a full queue.

## Host: green actors on the existing runtime

`Host` is an ordinary dedicated-thread `Actor` whose `park` method sweeps owned `Node`s. There
is no second thread pool or coroutine runtime:

```text
ActorScheduler on one OS thread
              │
             Host
       ┌──────┼──────┐
       ▼      ▼      ▼
     Node A Node B Node C
```

The host owns a heterogeneous set of green actors. They do not migrate between threads and
therefore do not need to be `Send`; sweep order is adoption order. Adopting a pipeline in
topological order lets one host sweep carry a message through several local stages.

External producers use `green_channel`/`GreenSender`, which first pushes the message and then
wakes the host. A quiet host reports `Idle`, allowing its OS thread to sleep on the ordinary
scheduler doorbell.

Because a `Host` is itself an actor, a green actor may conceptually contain another host. That
recursive placement is preserved as a deferred capability for future larger static systems.
The current scope does not add dynamic spawning, migration, work stealing, or automatic
hierarchical topology construction.

The full design and its non-goals are recorded in
[`docs/designs/actor-scheduler-mealy-transducer.md`](../docs/designs/actor-scheduler-mealy-transducer.md).

## Lifecycle and shutdown

Green actors return `Exit` values when they halt. A host records those exits, but restart
policy belongs to a supervisor rather than the host itself.

Dedicated-thread schedulers support:

- `ShutdownMode::Immediate`
- `ShutdownMode::DrainControl`
- `ShutdownMode::DrainAll { timeout }`

## Tuning and benchmarks

`SchedulerParams` is shared by both scheduling tiers. The repository contains benchmark
tooling, including a Bayesian parameter search, but no recorded parameter set or throughput
number is treated as portable across machines and workloads.

```bash
cargo bench -p actor-scheduler
cargo bench -p actor-scheduler --bench bench_throughput
cargo bench -p actor-scheduler --bench bench_latency
cargo bench -p actor-scheduler --bench bench_adversarial
cargo bench -p actor-scheduler --bench bench_mealy
cargo bench -p actor-scheduler --bench bench_optimize
```

## Design constraints

- Register dedicated-thread producers during initialization.
- Keep hosted topology static while a host is running.
- Treat a droppable port as a semantic declaration that stale output may be lost.
- Keep transition handlers run-to-completion; suspension state belongs in the outbox.
- Validate cyclic backpressure at topology construction rather than discovering deadlock at
  runtime.

## License

Apache-2.0
