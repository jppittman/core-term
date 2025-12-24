# CLAUDE.md - AI Assistant Guide for core-term

## Critical Constraints

- **NO TERMINAL LOGIC GOES IN PIXELFLOW.** PixelFlow is a general-purpose graphics library being extracted to its own crate/repo. Keep it terminal-agnostic.

## Project Overview

**core-term** is a GPU-free terminal emulator built on PixelFlow, a pull-based functional graphics engine using CPU SIMD. The project demonstrates that elegant algebraic abstractions can achieve 155 FPS at 1080p on pure CPU.

### Philosophy

- **Pull-based rendering**: Pixels are sampled, not pushed. Nothing computes until coordinates arrive.
- **SIMD as algebra**: `Field` wraps SIMD vectors (AVX-512/NEON/SSE2) transparently. Users write equations, compiler emits assembly.
- **The Fixed Observer**: Camera is at origin. Movement is achieved by warping coordinate space.
- **Types are shaders**: Combinator trees monomorphize into fused kernels with no runtime dispatch.

## Crate Architecture

```
pixelflow-core        Pure algebra. Field, Manifold, no IO/colors.
      ↓               The "lambda calculus" of the system.
      ↓
pixelflow-graphics    Colors, fonts, compositing, materialization.
      ↓               Where algebra becomes pixels.
      ↓
pixelflow-runtime     Windowing, input, render loop, display drivers.
      ↓               Platform abstraction (Cocoa/X11/Web).
      ↓
actor-scheduler       Priority message passing, troupe macro.
      ↓               Three-lane scheduling (Control > Mgmt > Data).
      ↓
core-term             Terminal emulator application.
                      ANSI parsing, PTY I/O, terminal state.
```

### Key Crate Details

| Crate | Edition | Purpose |
|-------|---------|---------|
| `pixelflow-core` | 2024 | `no_std` SIMD algebra. `Field`, `Manifold`, coordinate variables. |
| `pixelflow-graphics` | 2021 | Font loading, colors (`Rgba8`, `Color`), rasterization. |
| `pixelflow-runtime` | 2021 | Display drivers (`display_cocoa`, `display_x11`, `display_web`). |
| `actor-scheduler` | 2024 | Priority channels with `troupe!` macro for actor groups. |
| `core-term` | 2021 | Terminal application, PTY management, ANSI processing. |

## Core Concepts

### The Manifold Abstraction

Everything is a `Manifold<Output = T>` - a function from coordinates to values:

```rust
// Manifold hierarchy (dimensional collapse):
// Manifold (x,y,z,w) → Volume (x,y,z) → Surface (x,y) → scalar
// All higher dimensions satisfy lower dimension contracts via blanket impls.

trait Manifold {
    type Output;
    fn eval_raw(&self, x: Field, y: Field, z: Field, w: Field) -> Self::Output;
}
```

### Six Eigenshaders

All shaders compose from six primitives:
1. **Warp** - Remap coordinates before sampling
2. **Grade** - Linear transform on values (matrix + bias)
3. **Lerp** - Continuous interpolation: `a + t*(b-a)`
4. **Select** - Branchless conditional (discrete)
5. **Fix** - Iteration as a dimension (fractals, simulation)
6. **Compute** - Escape hatch (any closure is a Manifold)

### Actor Model

Three-thread architecture for zero-latency input:

```
Main Thread (Display)     Orchestrator Thread      PTY I/O Thread
├─ Cocoa/X11 event loop   ├─ Terminal state        ├─ kqueue/epoll
├─ BackendEvent → channel ├─ ANSI parsing          ├─ PTY read/write
└─ Render commands        └─ Render generation     └─ IOEvent → channel
```

Priority lanes: **Control > Management > Data**

## Development Workflow

### Build Commands

```bash
# Standard build (auto-detects display driver)
cargo build

# Run tests
cargo test

# Run the terminal (macOS bundled app)
cargo xtask bundle-run

# With profiling (writes flamegraph on exit)
cargo xtask bundle-run --features profiling

# Run benchmarks
cargo bench -p pixelflow-core
cargo bench -p pixelflow-graphics
```

### Toolchain

- **Rust Nightly** required (see `rust-toolchain.toml`)
- Platform features: `display_cocoa` (macOS), `display_x11` (Linux), `display_web` (WASM)

### CI

GitHub Actions runs `cargo test` on `ubuntu-latest` and `macos-latest` for every PR.

## Code Style Guide

### Comments

1. **Clarity over comments** - Refactor unclear code rather than explaining it
2. **Rustdoc (`///`)** - Document public API contract (WHAT and HOW)
3. **Regular comments (`//`)** - Explain WHY, not what. Design rationale, workarounds, non-obvious logic.
4. **No historical notes** - Use git commit messages for change history

### Code Structure

1. **Avoid deep nesting** - Use guard clauses and early returns
2. **Prefer `match` over `else if`** - Especially for enums
3. **Functions < 4 arguments** - Group related args into structs
4. **No boolean arguments** - Use enums or separate functions
5. **Prefer idempotent APIs** - Same result on repeated calls

### Magic Numbers

Define constants with clear names. Prefer enums for related sets:
```rust
const STATUS_PROCESSING_COMPLETE: u8 = 4;  // Good
if status_code == 4 { ... }                 // Bad
```

### Testing

- Test public API against documented contract
- Avoid testing internal implementation details

## Key Files to Know

| Path | Purpose |
|------|---------|
| `pixelflow-core/src/lib.rs` | Field, Manifold, SIMD type selection |
| `pixelflow-core/src/manifold.rs` | Core Manifold trait definition |
| `pixelflow-graphics/src/render/` | Rasterization, color spaces, AA |
| `pixelflow-graphics/src/fonts/` | Glyph loading and caching |
| `pixelflow-runtime/src/platform/` | Cocoa/X11 platform code |
| `actor-scheduler/src/lib.rs` | Priority channel implementation |
| `core-term/src/term/` | Terminal emulator state |
| `core-term/src/ansi/` | ANSI escape sequence parser |
| `core-term/src/io/` | PTY and event monitor actors |

## Architecture Documentation

Detailed design docs in `docs/`:
- `NORTH_STAR.md` - PixelFlow vision and philosophy
- `ARCHITECTURE.md` - Crate architecture overview
- `ACTOR_ARCHITECTURE.md` - Threading model deep dive
- `THREADING_DESIGN.md` - Platform-specific threading
- `STYLE.md` - Full coding style guide

## Common Patterns

### Creating a Color Manifold

```rust
use pixelflow_graphics::{Color, NamedColor, Manifold};

let red = Color::Named(NamedColor::Red);
// Color implements Manifold<Output = Discrete>
```

### Composing Manifolds

```rust
// Warp coordinates, then evaluate
let warped = manifold.warp(|x, y, z, w| (x * 2.0, y * 2.0, z, w));

// Select between two manifolds based on condition
let selected = mask.select(if_true, if_false);
```

### Actor Message Sending

```rust
use actor_scheduler::{Message, ActorHandle};

// Send with priority
handle.send(Message::Control(MyControlMsg))?;  // Highest
handle.send(Message::Management(MyMgmtMsg))?;  // Medium
handle.send(Message::Data(MyDataMsg))?;        // Lowest (backpressure)
```

## Performance Notes

- Target: 155 FPS at 1080p, ~5ns per pixel
- Use `#[inline(always)]` on hot path eval methods
- Glyph caching via categorical morphisms (see `fonts/combinators.rs`)
- Automatic differentiation via `Jet2` for antialiasing

## Platform Notes

### macOS
- Cocoa MUST run on main thread (Apple requirement)
- Display driver runs NSApp.run() blocking event loop
- Bundle created via `cargo xtask bundle-run`

### Linux
- X11 driver feature: `display_x11`
- Uses epoll for PTY I/O

### Web (WASM)
- Feature: `display_web`
- SharedArrayBuffer for IPC with main thread
