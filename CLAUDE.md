# CLAUDE.md - AI Assistant Guide for core-term


## Project Overview

**core-term** is a GPU-free terminal emulator built on PixelFlow, a pull-based functional graphics engine using CPU SIMD. The project demonstrates that elegant algebraic abstractions can achieve 155 FPS at 1080p on pure CPU.
**pixelflow** is an eDSL built on rust isomorphic to the typed lambda calculus.
**pixelflow-graphic** is graphic library built using the aforementioned eDSL.
**pixelflow-runtime** offers a platform agnostic runtime for applications using pixelflow rendering.
**actor-scheduler** offers a user space cooperative scheduler for actor model based libraries/applications

## Critical Constraints

- **NO TERMINAL LOGIC GOES IN PIXELFLOW.** PixelFlow is a general-purpose graphics library being extracted to its own crate/repo. Keep it terminal-agnostic.
- Exporting direct manipulation of fields from pixelflow-core is strictly forbidden. Construct compute kernels at load time and render them.
- **NO PUBLIC raw_mul, raw_select, raw_add ETC USAGE** NONE. ZERO. Do not perform raw operations on fields/jets without explicit direction. ALWAYS construct the ast, then uses the nested contramap pattern to evaluate it.

### Philosophy

- **Pull-based rendering**: Pixels are sampled, not pushed. Nothing computes until coordinates arrive.
- **SIMD as algebra**: `Field` wraps SIMD vectors (AVX-512/NEON/SSE2) transparently. Users write equations, compiler emits assembly.
- **The Fixed Observer**: Camera is at origin. Movement is achieved by warping coordinate space.
- **Types are shaders**: Combinator trees monomorphize into fused kernels with no runtime dispatch.
- **Suckless Dependencies**: Keep dependencies to the bare, bare minimum. We probably need 1% of what they offer. This is something the suckless people got right. (no crossbeam)

## Workspace Structure

The repository is organized as a Cargo workspace with 11 member crates:

```
core-term/                  # Repository root
├── pixelflow-core/         # SIMD algebra (no_std)
├── pixelflow-graphics/     # Colors, fonts, rendering
├── pixelflow-ir/           # Shared IR and backend abstraction
├── pixelflow-macros/       # Proc-macro compiler frontend
├── pixelflow-ml/           # Experimental: ML as graphics
├── pixelflow-nnue/         # NNUE neural network for instruction selection
├── pixelflow-runtime/      # Platform drivers and runtime
├── pixelflow-search/       # E-graph optimization and rewrite search
├── actor-scheduler/        # Priority channels
├── actor-scheduler-macros/ # Proc macros for actors
├── core-term/              # Terminal application
├── xtask/                  # Build automation
├── docs/                   # Architecture documentation
├── assets/                 # Fonts, resources
├── scripts/                # Development scripts
├── .github/workflows/      # CI/CD automation
├── .claude/                # Claude Code configuration + table of contents
├── .agent/                 # Agent rules and configuration
└── .jules/                 # Jules AI configuration
```

### Special Directories

- **`.claude/`** - Claude Code (Anthropic) configuration and documentation index
- **`.agent/`** - AI agent rules (core.md, splats.md) for automated development
- **`.jules/`** - Jules AI assistant configuration
- **`.githooks/`** - Git hooks for pre-commit checks and automation

These directories configure AI assistants to understand project conventions and constraints.

### Key Crate Details

| Crate | Purpose |
|-------|---------|
| `pixelflow-core` | `no_std` SIMD algebra. `Field`, `Manifold`, coordinate variables. Multi-backend (AVX-512/SSE2/NEON/scalar). |
| `pixelflow-ir` | Shared IR (intermediate representation) and backend abstraction. Op traits, OpKind enum, backend execution traits. |
| `pixelflow-macros` | Proc-macro compiler frontend: `kernel!` macro, lexer, parser, semantic analysis, code generation. |
| `pixelflow-graphics` | Font loading, colors (`Rgba8`, `Color`), rasterization, antialiasing. |
| `pixelflow-ml` | Experimental: Linear attention as spherical harmonics. Research on neural rendering. |
| `pixelflow-nnue` | NNUE neural network for instruction selection, inspired by Stockfish. HalfEP feature encoding. |
| `pixelflow-runtime` | Display drivers (macOS Cocoa, X11, headless, Metal, Web WASM), input handling, vsync. |
| `pixelflow-search` | E-graph optimization framework. Rewrite rules, saturation, cost extraction, NNUE-guided search. |
| `actor-scheduler` | Priority channels with `troupe!` macro for actor groups. Lock-free concurrency. |
| `actor-scheduler-macros` | Procedural macros for actor system. |
| `core-term` | Terminal application, PTY management, ANSI processing. First PixelFlow consumer. |
| `xtask` | Build tooling for bundling macOS app and running development tasks. |

## Core Concepts

### The Manifold Abstraction

Everything is a `kernel` - the pixelflow-macros compiler uses this to generate profunctors from coordinates to values or a morphism on manifolds:
dimap is broken up into covariant `map` and contramap `at`
conditionals are performed using Select or postfix (ManifoldExt) `.select`

### Actor Model

Three-thread architecture for zero-latency input:

Priority lanes: **Control > Management > Data**

Control/Management prioritize latency over throughput.
Control creates backpressure by timing out senders who are too aggressive. If the timeout exceeds a threshold, an error is returned, likely causing a crash.

## Development Workflow

### Build Commands

```bash
# Standard build (auto-detects display driver based on platform)
cargo build

# Release build with full optimization
cargo build --release

# Distribution build (LTO, strip symbols, abort on panic)
cargo build --profile dist

# Run tests for all workspace crates
cargo test --workspace

# Run benchmarks
cargo bench -p pixelflow-core
cargo bench -p pixelflow-graphics

# Run core-term directly
cargo run --release -p core-term

# Run the terminal (macOS bundled app with native integration)
cargo xtask bundle-run

# With profiling (writes flamegraph on exit)
cargo xtask bundle-run --features profiling
```

### Build Profiles

The workspace defines four build profiles:

0. **dev** - Fastest compiles. opt-level=1 because deep Manifold recursion causes stack overflow without inlining. panic=abort.
1. **release** - Fast compile, good performance (opt-level=3, panic=abort)
2. **bench** - For benchmarking (LTO, codegen-units=1)
3. **dist** - Maximum optimization for distribution (LTO, strip, codegen-units=1, panic=abort)

### Workspace Lints

The workspace enforces strict error handling:

```toml
[workspace.lints.rust]
unused_must_use = "deny"  # Can't ignore Results with `let _ =`

[workspace.lints.clippy]
let_underscore_must_use = "deny"  # Catches `let _ = expr` on #[must_use]
must_use_candidate = "warn"       # Suggests adding #[must_use]
```

This prevents silent failures - all errors must be explicitly handled.

### Toolchain

- SIMD backend auto-detected at compile time via `build.rs` and target features
- Platform features automatically selected based on OS (macOS Cocoa, Linux X11, Web WASM)

### CI & Automation

The repository includes several GitHub Actions workflows:

1. **rust.yaml** - Presubmit tests
   - Runs `cargo test --workspace` on `ubuntu-latest` and `macos-latest`
   - Caches Cargo dependencies for faster builds
   - Required for PR merge

2. **Gemini AI Integration** - Automated code review and triage
   - `gemini-dispatch.yml` - Routes tasks to appropriate workflows
   - `gemini-review.yml` - AI-powered code review on PRs
   - `gemini-triage.yml` - Issue triage and labeling
   - `gemini-scheduled-triage.yml` - Periodic maintenance tasks

3. **Quality Assurance**
   - `benchmark_regression.yaml` - Performance regression detection
   - `postsubmit-flake-detection.yaml` - Identifies flaky tests
   - `automatic-revert.yaml` - Automated rollback on critical failures

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

### Core Infrastructure

| Path | Purpose |
|------|---------|
| `pixelflow-core/src/lib.rs` | Field, Manifold, SIMD type selection, re-exports |
| `pixelflow-core/src/manifold.rs` | Core Manifold trait and dimensional hierarchy |
| `pixelflow-core/src/ext.rs` | Manifold methods for ergonomic postfix notation |
| `pixelflow-core/src/combinators/at.rs` | Struct version of contramap |
| `pixelflow-core/src/combinators/select.rs` | Struct version of conditional |
| `pixelflow-core/src/backend/` | SIMD backend implementations (AVX-512, SSE2, NEON, scalar) |
| `pixelflow-core/src/combinators/` | Six eigenshaders: Warp, Grade, Lerp, Select, Fix, Compute |
| `pixelflow-core/src/jet/` | Automatic differentiation for antialiasing |

### Compiler Stack (New)

| Path | Purpose |
|------|---------|
| `pixelflow-macros/src/lib.rs` | `kernel!` and `kernel_raw!` proc-macros, compiler entry points |
| `pixelflow-macros/src/lexer.rs` | Token stream processing (delegated to syn) |
| `pixelflow-macros/src/parser.rs` | AST construction from closure syntax |
| `pixelflow-macros/src/sema.rs` | Semantic analysis, symbol resolution, type validation |
| `pixelflow-macros/src/optimize.rs` | AST optimization (constant folding, FMA fusion, algebraic simplification) |
| `pixelflow-macros/src/codegen/` | Code generation: struct + Manifold impl emission |
| `pixelflow-ir/src/lib.rs` | Shared IR: Op trait, OpKind enum, Expr tree |
| `pixelflow-ir/src/backend/` | Backend-specific lowering (x86, ARM, WASM, scalar) |
| `pixelflow-search/src/egraph/` | E-graph data structure, saturation, rewrite rules |
| `pixelflow-search/src/egraph/extract.rs` | Cost-based extraction from saturated e-graphs |
| `pixelflow-nnue/src/lib.rs` | NNUE network for cost estimation, HalfEP features |

### Graphics & Rendering

| Path | Purpose |
|------|---------|
| `pixelflow-graphics/src/render/` | Rasterization, color spaces, antialiasing |
| `pixelflow-graphics/src/fonts/` | Glyph loading, caching, and SDF generation |
| `pixelflow-graphics/src/scene3d.rs` | 3D scene graph and transformations |

### Runtime & Platform

| Path | Purpose |
|------|---------|
| `pixelflow-runtime/src/display/drivers/` | Display driver implementations (X11, headless, Metal, Web) |
| `pixelflow-runtime/src/platform/macos/` | macOS Cocoa integration (window, events, objc bindings) |
| `pixelflow-runtime/src/platform/linux.rs` | Linux platform abstractions |
| `pixelflow-runtime/src/engine_troupe.rs` | Actor-based render orchestration |
| `pixelflow-runtime/src/vsync_actor.rs` | Vsync coordination actor |

### Actor System

| Path | Purpose |
|------|---------|
| `actor-scheduler/src/lib.rs` | Priority channel implementation (Control/Management/Data lanes) |
| `actor-scheduler-macros/src/lib.rs` | Troupe macro for actor group management |

### Terminal Application

| Path | Purpose |
|------|---------|
| `core-term/src/term/` | Terminal emulator state machine and grid |
| `core-term/src/ansi/` | ANSI escape sequence parser |
| `core-term/src/io/` | PTY actors and event monitoring (kqueue/epoll) |
| `core-term/src/terminal_app.rs` | Top-level terminal application actor |
| `core-term/src/surface/` | Terminal rendering surface |

### Build & Development

| Path | Purpose |
|------|---------|
| `xtask/src/main.rs` | Build tasks (bundle-run for macOS app packaging) |
| `Cargo.toml` | Workspace definition and build profiles |
| `rust-toolchain.toml` | Rust nightly pinning |

## Architecture Documentation

### Available Documentation

Detailed design docs in `docs/`:
- `STYLE.md` - Coding style guide and conventions
- `.claude/`  - Includes Claude's auto-generated repo TOC
- `AUTODIFF_RENDERING.md` - Automatic differentiation for antialiasing
- `MESSAGE_CUJ_COVERAGE.md` - Message passing critical user journeys
- `gemini/` - Gemini AI integration documentation

### Top-Level Documentation

- `README.md` - Project overview, philosophy, getting started
- `CLAUDE.md` - This file: AI assistant development guide
- `LICENSE.md` - Apache license

### Agent Context Files

Specialized context files for AI agents live in `.claude/agents/`. Each file provides deep domain knowledge for working on specific parts of the codebase.

#### Crate Engineers

| Agent | Domain | Key Expertise |
|-------|--------|---------------|
| [`actor-scheduler.md`](.claude/agents/actor-scheduler.md) | Priority message passing | Three-lane messaging (Control/Management/Data), troupe pattern, backpressure |
| [`pixelflow-core.md`](.claude/agents/pixelflow-core.md) | SIMD algebra | Manifold trait, operator AST, chained.rs impls, SIMD backends |
| [`pixelflow-graphics.md`](.claude/agents/pixelflow-graphics.md) | Rendering | Colors as coordinates, glyph caching, rasterization pipeline |
| [`pixelflow-runtime.md`](.claude/agents/pixelflow-runtime.md) | Platform layer | Display drivers, three-thread architecture, frame recycling |
| [`pixelflow-ml.md`](.claude/agents/pixelflow-ml.md) | Neural/graphics research | Linear attention ≈ spherical harmonics, feature maps |
| [`core-term.md`](.claude/agents/core-term.md) | Terminal emulator | ANSI parsing, PTY I/O, terminal state machine |

#### Specialists

| Agent | Role | When to Consult |
|-------|------|-----------------|
| [`algebraist.md`](.claude/agents/algebraist.md) | Category theory | Designing combinators, composition laws, variance |
| [`numerics.md`](.claude/agents/numerics.md) | SIMD/performance | Optimization, automatic differentiation, precision |
| [`language-mechanic.md`](.claude/agents/language-mechanic.md) | Rust type system | Trait bounds, impl conflicts, macro issues |

#### Process Agents

| Agent | Role | When to Use |
|-------|------|-------------|
| [`reviewer.md`](.claude/agents/reviewer.md) | Code review | PR review, updating agent files with new patterns |
| [`refactor.md`](.claude/agents/refactor.md) | Incremental cleanup | Dead code removal, warning fixes, consistency |

These agents provide targeted context for specific domains. Consult the appropriate agent when working in their area of expertise.

## Common Patterns

### Using the `kernel!` Macro

The `kernel!` macro provides closure-like syntax for defining SIMD manifold kernels:

```rust
use pixelflow_macros::kernel;
use pixelflow_core::{X, Y, Manifold, ManifoldExt};

// Define a parameterized circle SDF
let circle = kernel!(|cx: f32, cy: f32, r: f32| {
    let dx = X - cx;
    let dy = Y - cy;
    (dx * dx + dy * dy).sqrt() - r
});

// Instantiate with concrete parameters
let unit_circle = circle(0.0, 0.0, 1.0);
```

The compiler pipeline: Lexer → Parser → Semantic Analysis → Optimization → Codegen

Use `kernel_raw!` to skip optimization (for benchmarking exact expression forms).

### Composing Manifolds

```rust
// Warp coordinates, then evaluate
let warped = manifold.warp(|x, y, z, w| (x * 2.0, y * 2.0, z, w));

// Select between two manifolds based on condition
let selected = mask.select(if_true, if_false);

// Variables as the symbol table
let circle = (X * X + Y * Y + Z * Z).sqrt();
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

- **Target:** 155 FPS at 1080p, ~5ns per pixel
- **Hot path optimization:** Use `#[inline(always)]` on eval methods in Manifold implementations
- **Glyph caching:** Categorical morphisms ensure glyphs are computed once (see `fonts/combinators.rs`)
- **Antialiasing:** Automatic differentiation via `Jet2` dual numbers for gradient-based antialiasing
- **Memory:** Zero allocations per frame (ping-pong buffer strategy)
- **Monomorphization:** Type-driven specialization - entire scene compiles to fused SIMD kernels

### SIMD Backend Selection

PixelFlow automatically selects the best SIMD backend at compile time:

1. **AVX-512** (x86_64 with AVX-512F) - 16-wide f32 vectors, highest performance
2. **SSE2** (x86/x86_64) - 4-wide f32 vectors, universal x86 baseline
3. **NEON** (ARM/AArch64) - 4-wide f32 vectors, ARM mobile/Apple Silicon
4. **Scalar** (fallback) - Software fallback, no SIMD

Backend detection uses both:
- `build.rs` CPU feature detection (compile-time)
- `target_feature` flags (ensures correct instruction set)

See `pixelflow-core/src/backend/` for implementation details.

## Platform Notes

### macOS
- **Threading constraint:** Cocoa MUST run on main thread (Apple requirement)
- **Display driver:** Uses native Cocoa APIs via manual objc bindings (`platform/macos/cocoa.rs`)
- **Event loop:** Blocking `NSApp.run()` event loop on main thread
- **Bundling:** `cargo xtask bundle-run` creates `CoreTerm.app` with proper Info.plist
- **PTY I/O:** kqueue-based async I/O on dedicated thread
- **Platform code:** `pixelflow-runtime/src/platform/macos/`

### Linux
- **Display driver:** X11 via `pixelflow-runtime/src/display/drivers/x11.rs`
- **PTY I/O:** epoll-based async I/O
- **Dependencies:** Requires X11 development headers (libx11-dev, libxft-dev, etc.)
- **Platform code:** `pixelflow-runtime/src/platform/linux.rs`

### Web (WASM)
- **Driver:** `pixelflow-runtime/src/display/drivers/web/`
- **Threading:** SharedArrayBuffer for cross-thread communication
- **Rendering:** Canvas API integration
- **SIMD:** WebAssembly SIMD when available, scalar fallback otherwise

### Headless (Testing)
- **Driver:** `pixelflow-runtime/src/display/drivers/headless.rs`
- **Purpose:** CI/testing, benchmarking without display
- **Usage:** Automatic in test environments

## Compiler Architecture

The PixelFlow compiler transforms DSL expressions into optimized SIMD code at compile time.

### Pipeline

```
Source → Lexer → Parser → Sema → Optimize → Codegen → Rust TokenStream
                   ↓           ↓
               Symbol Table  E-graph + NNUE
```

### Key Crates

| Crate | Role |
|-------|------|
| `pixelflow-macros` | Compiler frontend: proc-macros, parser, semantic analysis |
| `pixelflow-ir` | Intermediate representation: Op trait, OpKind, backend traits |
| `pixelflow-search` | E-graph optimization: saturation, rewrite rules, cost extraction |
| `pixelflow-nnue` | Neural cost model: NNUE-style network for instruction selection |

### E-graph Optimization

The compiler uses e-graphs (equality graphs) to find optimal instruction sequences:

1. **Build e-graph** from expression AST
2. **Saturate** by applying rewrite rules (associativity, FMA fusion, etc.)
3. **Extract** minimum-cost implementation using NNUE-guided search

### NNUE Cost Model

Inspired by Stockfish's NNUE, uses HalfEP (Half-Expression-Position) features:

| Chess (Stockfish) | Compiler (PixelFlow) |
|-------------------|---------------------|
| Position | Expression AST |
| Legal move | Valid rewrite rule |
| Evaluation (centipawns) | Cost (cycles) |
| HalfKP features | HalfEP features |

Incremental updates: only features for modified subtrees change, making evaluation O(rewrite_size).

## Experimental: pixelflow-ml

The `pixelflow-ml` crate explores a deep mathematical connection:

**Linear Attention IS Harmonic Global Illumination**

### The Insight

Both linear attention (transformers) and spherical harmonic lighting solve the same problem:
- Compress infinite/quadratic interactions into finite/linear operations
- Project onto orthogonal basis functions (φ for attention, Y_lm for SH)
- Compute via efficient inner products instead of exhaustive summation

### Current Status

- **Experimental research crate** - API unstable
- Depends only on `pixelflow-core` (pure algebra)
- Uses `libm` for `no_std` compatibility
- Edition 2024 (latest Rust features)

### Use Cases

- Neural rendering with SH-inspired feature maps
- Fast approximation of global illumination in PixelFlow scenes
- Exploration of attention mechanisms as graphics primitives

See `pixelflow-ml/src/lib.rs` for detailed mathematical exposition.

## Debugging & Common Pitfalls

### SIMD Backend Issues

**Problem:** Code works on one machine but crashes/produces incorrect results on another
- **Cause:** SIMD backend mismatch (e.g., using AVX-512 on CPU without support)
- **Solution:** Check `build.rs` output, verify target features match CPU capabilities
- **Debug:** Set `RUSTFLAGS="-C target-cpu=native"` to match your CPU exactly

**Problem:** Performance unexpectedly slow
- **Cause:** Falling back to scalar backend instead of SIMD
- **Check:** Look for "using scalar backend" in build output
- **Fix:** Ensure CPU features are detected correctly, check `target_feature` flags

### Platform-Specific Issues

**macOS:** "Cocoa API must run on main thread" panic
- **Cause:** Display driver not initialized on main thread
- **Fix:** Ensure `pixelflow_runtime::run()` is called from `fn main()`, not a spawned thread

**Linux:** X11 connection errors
- **Cause:** Missing X11 development headers or DISPLAY not set
- **Fix:** Install libx11-dev packages, ensure `DISPLAY` environment variable is set

**Headless CI:** Rendering failures in GitHub Actions
- **Solution:** Use headless driver automatically selected in CI environments

### Manifold Composition

**Problem:** Compiler error about trait bounds on complex Manifold compositions
- **Cause:** Type inference struggles with deeply nested generic types
- **Solution:** Add explicit type annotations, break composition into named intermediate manifolds

**Problem:** "method not found" on Manifold
- **Cause:** Missing trait import or wrong associated type
- **Solution:** Import all Manifold extension traits: `use pixelflow_core::Manifold;`

### Performance Debugging

**Problem:** Frame drops or slow rendering
1. Enable profiling: `cargo xtask bundle-run --features profiling`
2. Check flamegraph for hot paths
3. Verify `#[inline(always)]` on critical eval methods
4. Profile with `cargo bench` to isolate performance regression

**Problem:** High memory usage
- **Cause:** Allocations during frame rendering
- **Debug:** Use `cargo instruments` (macOS) or `valgrind --tool=massif` (Linux)
- **Fix:** Ensure ping-pong buffers are reused, no per-frame allocations

## API Visibility Rules

**CRITICAL:** Do NOT change visibility of internal APIs without explicit permission:
- Keep `pub(crate)` and private items encapsulated
- Public API surface should be minimal and intentional
- Use Manifold composition instead of exposing internals
- Changes to public API require architectural discussion

## Quick Reference

### Most Common Commands

```bash
# Development
cargo build --release              # Build with optimizations
cargo test --workspace            # Run all tests
cargo run --release -p core-term  # Run terminal directly
cargo xtask bundle-run            # Build and run macOS app

# Performance
cargo bench -p pixelflow-core     # Benchmark core algebra
cargo xtask bundle-run --features profiling  # Profile with flamegraph

# Platform-specific
cargo build --features display_x11     # Force X11 (Linux)
cargo build --features display_cocoa   # Force Cocoa (macOS)
```

### Key Concepts Cheat Sheet

| Concept | Summary |
|---------|---------|
| **Manifold** | Function from 4D coords (x,y,z,w) to value. Everything is a Manifold. |
| **Field** | SIMD vector wrapper (AVX-512/SSE2/NEON). Transparent algebra. |
| **Pull-based** | Pixels ask "what color?" instead of primitives pushing. |
| **Six Eigenshaders** | Warp, Grade, Lerp, Select, Fix, Compute - all graphics compose from these. |
| **Actor Model** | 3 threads, 3 priority lanes (Control > Management > Data). |
| **No allocations** | Zero per-frame allocations via ping-pong buffers. |
| **Monomorphization** | Type system compiles graphics to fused SIMD kernels. |

### File Path Patterns

```
pixelflow-core/src/
  lib.rs              - Re-exports, Field definition
  manifold.rs         - Core trait
  backend/            - SIMD implementations (avx512, sse2, neon, scalar)
  combinators/        - Six eigenshaders
  jet/                - Automatic differentiation

pixelflow-macros/src/
  lib.rs              - kernel! macro entry point
  parser.rs           - AST construction
  sema.rs             - Semantic analysis
  optimize.rs         - AST optimization
  codegen/            - Rust code emission

pixelflow-ir/src/
  lib.rs              - IR types and re-exports
  ops.rs              - Operation unit structs
  kind.rs             - OpKind enum
  backend/            - Target-specific lowering

pixelflow-search/src/
  egraph/             - E-graph optimization
    graph.rs          - Core e-graph structure
    saturate.rs       - Rewrite saturation
    extract.rs        - Cost-based extraction

pixelflow-nnue/src/
  lib.rs              - NNUE network, HalfEP features

pixelflow-graphics/src/
  render/             - Rasterization engine
  fonts/              - Glyph loading & SDF
  scene3d.rs          - 3D scene graph

pixelflow-runtime/src/
  display/drivers/    - X11, headless, Metal, Web
  platform/macos/     - Cocoa integration
  engine_troupe.rs    - Render actor orchestration

core-term/src/
  term/               - Terminal state machine
  ansi/               - Escape sequence parser
  io/                 - PTY actors
```

### When to Use Which Tool

| Task | Tool/Approach |
|------|---------------|
| Add new rendering primitive | Implement `Manifold` trait or compose from eigenshaders |
| Define parameterized shader | Use `kernel!` macro with closure syntax |
| Change colors | Use `Color` enum, never modify PixelFlow internals |
| Add terminal feature | Modify `core-term`, keep PixelFlow clean |
| Optimize performance | Profile first, check `#[inline(always)]`, verify SIMD backend |
| Add rewrite rule | Add to `pixelflow-search/src/egraph/rules.rs` |
| Add new IR operation | Add to `pixelflow-ir/src/ops.rs`, implement `Op` trait |
| Platform-specific code | Put in `pixelflow-runtime/src/platform/`, not in core |
| New graphics abstraction | Add to `pixelflow-graphics`, use `pixelflow-core` types |

### Architecture Principles (Never Violate)

1. **NO terminal logic in PixelFlow** - Keep it general-purpose
2. **Pull, not push** - Pixels sample, don't receive
3. **Types are shaders** - Use type system for optimization
    3a. Types are the AST
    3b. Fields/Jets are the IR.
    3c. variables.rs is the symbol table
4. **Minimal public API** - Composition over exposure
5. **Zero allocations** - No per-frame heap allocation
6. **No copies of unknown sized types** pixelflow language types are copy iff they are provably zero sized 
6. **Platform on main thread** - Especially macOS Cocoa
