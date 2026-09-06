# PixelFlow — Pull-Based Graphics on CPU SIMD

PixelFlow is an experimental CPU graphics and kernel-language stack built around
two ideas: pixels pull values from functions over coordinates, and SIMD is part of the
algebra rather than an optimization added afterward.

`core-term` is its first application: a terminal emulator whose font, compositing, runtime,
and concurrency requirements exercise the libraries as a system.

## What is current

PixelFlow is under active architectural development. The current direction is JIT-first:
a [`Kernel`](pixelflow-ir/src/kernel.rs) is an immutable handle to an `ExprArena` fragment,
and composition (`add`, `select`, `at`, bounded reductions, derivatives) splices fragments
into a larger arena. A root is lowered and emitted for the host CPU.

```text
kernel_value! source ── parse / sema / e-graph ──┐
                                                 ▼
direct Kernel construction ── arena splicing ── ExprArena
                                                 │
                                      lowering and emission
                                                 │
                                                 ▼
                                              CPU JIT
```

This migration is not finished. `kernel_value!` builds arena-backed `Kernel` values today,
and the font pipeline composes and bakes them end to end. The older `kernel!` surface still
emits type-level combinators; it is the legacy tier that
[A Kernel with a Lattice](docs/plans/2026-09-06-kernel-with-a-lattice.md) retires. The
earlier plan of record is
[One Kernel Language](docs/plans/2026-07-20-kernel-unification.md); the current language
axiom and intended cost boundary are described in
[Totality and the Cost Model](docs/designs/2026-07-24-totality-and-the-cost-model.md).

## Why pull-based rendering

In an immediate raster pipeline, primitives push contributions into a framebuffer. In
PixelFlow, a pixel samples a function at its coordinate. This makes coordinate transforms,
composition, differentiation, and sampling explicit algebraic operations. Depending on the
scene and backend, it can avoid work such as off-sample primitive evaluation and intermediate
buffers; it does not promise that all scenes are free of overdraw or branches.

## Workspace

| Crate | Purpose |
|---|---|
| `pixelflow-core` | `no_std` SIMD fields, the `Manifold` substrate, coordinates, combinators, and compatibility layer |
| `pixelflow-ir` | `Kernel`, `ExprArena`, operations, lowering, and CPU emitters |
| `pixelflow-compiler` | `kernel!`, `kernel_value!`, and related parser/sema/optimization front ends |
| `pixelflow-search` | E-graphs, rewrite rules, extraction, provenance, and guided-search experiments |
| `pixelflow-pipeline` | Benchmark, corpus, and cost-model research tooling |
| `pixelflow-graphics` | Colors, font kernels and caches, scene composition, and framebuffer materialization |
| `pixelflow-ml` | Experimental ML and spherical-harmonic consumers of the kernel language |
| `pixelflow-runtime` | Cocoa/X11/Web display integration, input, and render orchestration |
| `actor-scheduler` | Priority OS-thread actors plus hosted Mealy transducers (“green actors”) |
| `actor-scheduler-macros` | Procedural macros for actor groups and typed transducer ports |
| `core-term` | ANSI parsing, PTY management, terminal state, and the application UI |
| `xtask` | Repository build and macOS bundle tasks |

Terminal-specific behavior remains in `core-term`; PixelFlow crates are intended to stay
general-purpose.

## A Kernel value

`kernel_value!` runs parsing, semantic analysis, and e-graph optimization at macro expansion,
then returns an uncompiled arena fragment. Scalar parameters are folded into the fragment;
larger programs compose as `Kernel` values and compile at a materialization boundary.

```rust
use pixelflow_compiler::kernel_value;
use pixelflow_core::Kernel;

let circle = kernel_value!(|cx: f32, cy: f32, radius: f32| {
    let dx = X - cx;
    let dy = Y - cy;
    (dx * dx + dy * dy).sqrt() - radius
});

let left = circle(-0.5, 0.0, 1.0);
let right = circle(0.5, 0.0, 1.0);
let pair = left.min(&right);
```

`Kernel` currently supports arithmetic, comparisons and selection, coordinate substitution
with `at`, symbolic `Dwrt` derivatives, and bounded monoid reductions. It intentionally does
not expose unbounded loops. Typed discrete fields and a closed-form cost interpretation are
plan-of-record work, not completed features.

## Graphics

The graphics crate turns kernels and manifolds into pixels. Its current font path parses TTF
outlines directly into fused coverage `Kernel`s, resolves antialiasing through symbolic
derivatives, and bakes reusable glyph lattices through the JIT. The ray-tracing modules remain
a useful application of the older polymorphic `Manifold` layer, including derivative-derived
surface normals.

See [pixelflow-graphics](pixelflow-graphics/README.md) for the current boundary between these
paths.

## Concurrency

`actor-scheduler` has two composable tiers:

- Dedicated OS-thread actors use three priority lanes (Control, Management, Data) over
  producer-sharded SPSC queues.
- `Transducer`/`Node` green actors run one transition at a time inside a `Host`, propagate
  backpressure through typed ports, and share the same priority semantics.

A `Host` is itself an ordinary actor, so hosting can be nested in principle. Dynamic topology,
migration, and a separate green-thread runtime are deliberately outside the current design.
See [actor-scheduler](actor-scheduler/README.md) and the
[Mealy-transducer design](docs/designs/actor-scheduler-mealy-transducer.md).

## Build and run

Use the stable toolchain selected by `rust-toolchain.toml`.

```bash
cargo build --workspace
cargo test --workspace
cargo clippy --workspace --all-targets --all-features -- -D warnings
```

Run the terminal directly:

```bash
cargo run --release -p core-term
```

On macOS, build and launch the application bundle:

```bash
cargo bundle-run
```

Focused benchmarks are available without treating any recorded result as permanent:

```bash
cargo bench -p pixelflow-core
cargo bench -p pixelflow-graphics
cargo bench -p actor-scheduler
```

Platform packages for Linux builds include X11, Xft, Fontconfig, FreeType, and xkbcommon
development headers.

## Documentation status

[docs/README.md](docs/README.md) classifies documents as current architecture, plan of
record, experiment/result, or historical/superseded. Read status metadata before treating a
design document as an implementation contract. In particular, the learned guided-saturation
work is research with explicit decision gates; the deterministic e-graph and static extraction
path do not depend on that research succeeding.

Additional repository guidance:

- [AGENTS.md](AGENTS.md) — repository boundaries, commands, and review conventions
- [docs/STYLE.md](docs/STYLE.md) — code style and design principles
- [docs/designs/](docs/designs/) — architecture and design records
- [docs/plans/](docs/plans/) — active and superseded implementation plans
- [docs/results/](docs/results/) — point-in-time measurements and experiment reports

## Research context

PixelFlow draws from denotational functional graphics, Halide-style separation of algorithm
and schedule, e-graphs, SIMD code generation, and automatic differentiation. The repository
records both successful designs and discarded approaches; historical documents are retained
to preserve the evidence behind current choices.

## License

[Apache License 2.0](LICENSE.md)
