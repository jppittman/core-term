# PixelFlow — Pull-Based Functional Graphics on CPU SIMD

**A GPU-free graphics engine proving that elegant algebraic abstractions can achieve 155 FPS at 1080p on pure CPU.**

PixelFlow is a research project demonstrating a novel paradigm for real-time graphics: **pull-based rendering** with **SIMD as algebra**. Nothing computes until a coordinate arrives. Pixels are sampled, not pushed. The type system builds compute graphs. The compiler emits optimal vector assembly.

**`core-term`** is the first consumer application—a high-performance, correct terminal emulator built entirely on PixelFlow.

## Vision

PixelFlow answers three questions:

1. **What if we stopped pushing pixels and started pulling them?** In traditional rasterization, every primitive computes its contribution to every pixel. In PixelFlow, pixels ask "what color am I?" and only that computation happens.

2. **What if SIMD was algebra, not an optimization?** Instead of SIMD as a lower-level concern, PixelFlow treats SIMD vectors as the natural representation of continuous fields over coordinates.

3. **What if the type system compiled graphics?** The entire rendering pipeline—composed from six primitives (Warp, Grade, Lerp, Select, Fix, Compute)—monomorphizes into fused kernels with zero runtime dispatch.

## The Stack

```
┌─────────────────────────────────────────┐
│          core-term (App)                │  First consumer: Terminal emulator
├─────────────────────────────────────────┤
│ pixelflow-runtime (Platform)            │  Cocoa/X11/Web display drivers,
│ actor-scheduler (Concurrency)           │  input handling, render loop
├─────────────────────────────────────────┤
│ pixelflow-graphics (Materialization)    │  Colors, fonts, compositing,
│                                         │  rasterization to pixels
├─────────────────────────────────────────┤
│ pixelflow-core (Algebra)                │  Field, Manifold, coordinates,
│                                         │  no_std, SIMD abstraction
└─────────────────────────────────────────┘
```

## Crates at a Glance

| Crate | Edition | Purpose |
|-------|---------|---------|
| `pixelflow-core` | 2024 | Pure algebra. `Field`, `Manifold`, coordinate variables. No I/O, no colors. |
| `pixelflow-graphics` | 2021 | Colors (`Rgba8`), fonts, rasterization, antialiasing via automatic differentiation. |
| `pixelflow-runtime` | 2021 | Display drivers (Cocoa/X11/Web), input handling, render orchestration. |
| `actor-scheduler` | 2024 | Priority message passing with `troupe!` macro for lock-free concurrent actors. |
| `core-term` | 2021 | Terminal emulator. ANSI parsing, PTY management, state machine. The first PixelFlow consumer. |

## The Manifold Abstraction

Everything in PixelFlow is a `Manifold`—a function from 4D coordinates to a value:

```rust
trait Manifold<Input = Field> {
    type Output;
    fn eval_raw(&self, x: Input, y: Input, z: Input, w: Input) -> Self::Output;
}
```

This single abstraction enables:
- **Coordinate warping** (camera movement, distortion)
- **Rendering** (colors, textures, signed distance fields)
- **Simulation** (fractals, iterative systems via the `Fix` combinator)
- **Automatic differentiation** (gradients for antialiasing, normals, ray marching)

## The Six Eigenshaders

All graphics in PixelFlow compose from six primitives:

1. **Warp** — Remap coordinates before sampling
2. **Grade** — Linear transform on values (matrix + bias)
3. **Lerp** — Smooth interpolation between two manifolds
4. **Select** — Branchless conditional (discrete choice)
5. **Fix** — Iteration as a dimension (fractals, feedback systems)
6. **Compute** — Escape hatch (any closure is a Manifold)

## Performance

- **Target:** 155 FPS at 1080p (~5 nanoseconds per pixel)
- **Backend:** Pure CPU, no GPU required. SIMD: AVX-512, SSE2, NEON
- **Memory:** Zero allocation per frame (ping-pong buffer strategy)
- **Compilation:** Entire scene monomorphizes into fused kernels

## Getting Started with PixelFlow

### Documentation

The complete PixelFlow architecture is documented in `/docs/`:

- **[NORTH_STAR.md](docs/NORTH_STAR.md)** — PixelFlow vision, philosophy, and high-level design
- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** — Detailed crate architecture and dependencies
- **[ACTOR_ARCHITECTURE.md](docs/ACTOR_ARCHITECTURE.md)** — Zero-latency input via actor model
- **[STYLE.md](docs/STYLE.md)** — Code style guide and design principles

For core-term specifics:
- **[THREADING_DESIGN.md](docs/THREADING_DESIGN.md)** — Platform-specific threading and I/O
- **[PERFORMANCE_ANALYSIS.md](docs/PERFORMANCE_ANALYSIS.md)** — Profiling and optimization notes

### Prerequisites

- **Rust:** Nightly (see `rust-toolchain.toml`)
- **Platform dependencies:**
  - **macOS:** Native Cocoa support
  - **Linux:** X11 development headers
    ```bash
    sudo apt-get install libx11-dev libxext-dev libxft-dev libfontconfig1-dev libfreetype6-dev libxkbcommon-dev
    ```

### Building

Standard Rust build:
```bash
cargo build --release
```

Run tests:
```bash
cargo test
```

Run benchmarks:
```bash
cargo bench -p pixelflow-core
cargo bench -p pixelflow-graphics
```

### Running core-term

#### Standard
```bash
cargo run --release -p core-term
```

#### macOS (bundled app)
```bash
cargo xtask bundle-run
```
Builds and launches `CoreTerm.app` with native macOS integration.

#### With Profiling
```bash
cargo xtask bundle-run --features profiling
```
Writes flamegraph on exit.

## Architecture Overview

### Pull-Based Rendering

Traditional GPU pipeline: **push** every primitive to every pixel.

PixelFlow: **pull** each pixel samples what it needs.

```rust
// A pixel asks: "What color am I?"
// The manifold computes only what's necessary.
let color = manifold.eval_raw(x, y, 0.0, 0.0);
```

This eliminates:
- Overdraw
- Primitive list parsing
- Conditional branching in the hot loop

### Actor Model for Zero-Latency Input

Three-thread architecture:

```
Main Thread (Display)          Orchestrator Thread          PTY I/O Thread
├─ Cocoa/X11 event loop       ├─ Terminal state machine   ├─ kqueue/epoll
├─ Platform events            ├─ ANSI parser              ├─ PTY read/write
└─ Render commands            └─ Scene generation         └─ I/O events
    (BackendEvent)                (Render command)            (IOEvent)
         ↓                              ↓                          ↓
    Three-lane priority channel (Control > Management > Data)
```

Input latency is decoupled from render latency.

### Crate Separation Philosophy

PixelFlow is extracted from core-term because:

1. **No terminal logic in PixelFlow.** Graphics library stays general-purpose.
2. **Gradual extraction:** Each crate is independently useful.
3. **Future applications:** PixelFlow can power other renderers (UI toolkits, games, simulations).

## Extending PixelFlow

### Creating a New Manifold

```rust
use pixelflow_core::{Manifold, X, Y};

// A circle signed distance field
let circle = (X * X + Y * Y).sqrt() - 100.0;

// Compose with warp (zoom by 2x)
let zoomed = circle.warp(|x, y, z, w| (x * 2.0, y * 2.0, z, w));

// Render to pixels (handled by pixelflow-graphics)
```

### Composing Graphics

```rust
use pixelflow_graphics::{Color, NamedColor};

let background = Color::Named(NamedColor::Black);
let foreground = circle.select(
    Color::Named(NamedColor::White),
    background,
);
```

## Contributing

See [CLAUDE.md](CLAUDE.md) for architectural constraints and development guidelines.

Key points:
- **Code style:** Follow Rust idioms. See [STYLE.md](docs/STYLE.md).
- **No magic in PixelFlow:** Keep the algebra pure and portable.
- **Tests:** Public API changes require test updates.

## Performance Targets

- **Throughput:** 155 FPS at 1080p (full terminal)
- **Per-pixel cost:** ~5 nanoseconds
- **Memory:** Zero allocations per frame (ping-pong buffers)
- **Latency:** <5ms input-to-render (actor model)

## Research Context

PixelFlow is inspired by:
- [Conal Elliott's denotational design](http://conal.net/papers/icfp97/)
- [Halide](https://halide-lang.org/) (pull-based, algebraic composition)
- [Elm](https://elm-lang.org/) and pure functional graphics
- [Seamless.js](https://github.com/scttnlsn/seamless) (algebraic surfaces)

The goal: prove that **pure algebra** scales to real-time graphics without GPU compromise.

## License

[MIT License](LICENSE.md)
