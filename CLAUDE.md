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

The repository is organized as a Cargo workspace with 11 member crates (plus `pixelflow-ir` as a non-workspace path dependency):

```
core-term/                  # Repository root
├── pixelflow-core/         # SIMD algebra (edition 2024)
├── pixelflow-graphics/     # Colors, fonts, rendering
├── pixelflow-ir/           # Shared IR (NOT a workspace member, path dependency only)
├── pixelflow-macros/       # Proc-macro compiler frontend (edition 2024)
├── pixelflow-ml/           # ML for graphics & compiler optimization (edition 2024)
├── pixelflow-nnue/         # NNUE neural network for instruction selection (edition 2024)
├── pixelflow-runtime/      # Platform drivers and runtime
├── pixelflow-search/       # E-graph optimization and rewrite search (edition 2024)
├── actor-scheduler/        # Priority channels
├── actor-scheduler-macros/ # Proc macros for actors
├── core-term/              # Terminal application
├── xtask/                  # Build automation + codegen
├── docs/                   # Architecture documentation + design docs
├── assets/                 # Fonts (Noto Sans Mono), icons
├── scripts/                # Development scripts (Python + shell)
├── .github/workflows/      # CI/CD automation (9 workflows)
├── .claude/                # Claude Code configuration + agent context files
├── .agent/                 # Agent rules (core.md, splats.md)
├── .jules/                 # Jules AI configuration (bolt.md)
└── .githooks/              # Git hooks (pre-commit, pre-push, post-commit, etc.)
```

### Special Directories

- **`.claude/`** - Claude Code (Anthropic) configuration, `DOCS_TOC.md`, and `agents/` with 11 domain-specific context files
- **`.agent/`** - AI agent rules (`rules/core.md`, `rules/splats.md`) for automated development
- **`.jules/`** - Jules AI assistant configuration (`bolt.md`)
- **`.githooks/`** - Git hooks: `pre-commit`, `pre-push`, `post-commit`, `post-checkout`, `post-merge`

These directories configure AI assistants to understand project conventions and constraints.

### Non-Workspace Crate

`pixelflow-ir` exists as a directory with its own `Cargo.toml` but is **not** listed in the workspace `members`. It is consumed as a path dependency by `pixelflow-macros`, `pixelflow-ml`, `pixelflow-nnue`, and `pixelflow-search`. See `PIXELFLOW_IR_STATUS.md` for details.

### Key Crate Details

| Crate | Purpose | Edition |
|-------|---------|---------|
| `pixelflow-core` | SIMD algebra. `Field`, `Manifold`, coordinate variables, ops. Multi-backend (AVX-512/SSE2/NEON/scalar). | 2024 |
| `pixelflow-ir` | Shared IR. Op traits, OpKind enum, Expr tree, backend execution traits. **Not a workspace member.** | 2021 |
| `pixelflow-macros` | Proc-macro compiler: `kernel!` macro, lexer, parser, sema, AST optimization, codegen. | 2024 |
| `pixelflow-graphics` | Font loading (TTF, SDF), colors (`Rgba8`, `Color`), rasterization, antialiasing, shapes, animation. | 2021 |
| `pixelflow-ml` | Neural networks for graphics & compiler optimization. NNUE training, HCE extraction, e-graph training. | 2024 |
| `pixelflow-nnue` | NNUE neural network for instruction selection, inspired by Stockfish. HalfEP features, factored networks. | 2024 |
| `pixelflow-runtime` | Display drivers (macOS Cocoa, headless, Metal, Web WASM), input handling, vsync, render pool. | 2021 |
| `pixelflow-search` | E-graph optimization framework. Rewrite rules, saturation, cost extraction, NNUE-guided search. | 2024 |
| `actor-scheduler` | Priority channels with `troupe!` macro. Kubelet, lifecycle, SPSC queues, sharded dispatch. | 2021 |
| `actor-scheduler-macros` | Procedural macros for actor system. | 2021 |
| `core-term` | Terminal application: PTY management, ANSI processing, terminal emulator, key translation. | 2021 |
| `xtask` | Build tooling: macOS app bundling (`bundle-run`), codegen tasks. | 2021 |

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

- **Rust stable** toolchain (configured in `rust-toolchain.toml`)
- SIMD backend auto-detected at compile time via `build.rs` and target features
- Platform features automatically selected based on OS (macOS Cocoa, Linux X11, Web WASM)
- Build scripts (`build.rs`) exist in: `pixelflow-core`, `pixelflow-runtime`, `core-term`

### CI & Automation

The repository includes several GitHub Actions workflows:

1. **rust.yaml** - Presubmit tests
   - Runs `cargo test --workspace` on `ubuntu-latest` and `macos-latest`
   - Caches Cargo dependencies for faster builds
   - Required for PR merge

2. **Gemini AI Integration** - Automated code review and triage
   - `gemini-dispatch.yml` - Routes tasks to appropriate workflows
   - `gemini-invoke.yml` - Invokes Gemini for specific tasks
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
| `pixelflow-core/src/variables.rs` | Coordinate variables (X, Y, Z, W) - the symbol table |
| `pixelflow-core/src/algebra.rs` | Algebraic structures and laws |
| `pixelflow-core/src/ops/` | Operation types: `base`, `binary`, `unary`, `compare`, `trig`, `logic`, `derivative`, `vector` |
| `pixelflow-core/src/combinators/at.rs` | Struct version of contramap |
| `pixelflow-core/src/combinators/select.rs` | Struct version of conditional |
| `pixelflow-core/src/combinators/` | Combinators: at, map, select, fix, computed, binding, block, shift, pack, project, context, spherical, texture, with_gradient |
| `pixelflow-core/src/backend/` | SIMD backend implementations: `x86.rs` (AVX-512/SSE2), `arm.rs` (NEON), `scalar.rs`, `wasm.rs`, `fastmath.rs` |
| `pixelflow-core/src/jet/` | Automatic differentiation: `jet2.rs`, `jet2h.rs`, `jet3.rs`, `path_jet.rs` |
| `pixelflow-core/src/storage.rs` | Storage abstractions for SIMD data |
| `pixelflow-core/src/numeric.rs` | Numeric trait definitions |
| `pixelflow-core/src/mask.rs` | SIMD mask operations |
| `pixelflow-core/src/zst.rs` | Zero-sized type utilities |
| `pixelflow-core/src/domain.rs` | Domain abstractions |
| `pixelflow-core/src/dual.rs` | Dual number support |
| `pixelflow-core/src/generated_kernels.rs` | Auto-generated kernel implementations |

### Compiler Stack

| Path | Purpose |
|------|---------|
| `pixelflow-macros/src/lib.rs` | `kernel!` and `kernel_raw!` proc-macros, compiler entry points |
| `pixelflow-macros/src/lexer.rs` | Token stream processing (delegated to syn) |
| `pixelflow-macros/src/parser.rs` | AST construction from closure syntax |
| `pixelflow-macros/src/ast.rs` | AST node definitions |
| `pixelflow-macros/src/sema.rs` | Semantic analysis, symbol resolution, type validation |
| `pixelflow-macros/src/symbol.rs` | Symbol table management |
| `pixelflow-macros/src/optimize.rs` | AST optimization (constant folding, FMA fusion, algebraic simplification) |
| `pixelflow-macros/src/fold.rs` | AST fold/visitor infrastructure |
| `pixelflow-macros/src/ir_bridge.rs` | Bridge between macro AST and shared IR |
| `pixelflow-macros/src/rewrite_rules.rs` | Rewrite rule definitions for optimization |
| `pixelflow-macros/src/cost_builder.rs` | Cost model construction |
| `pixelflow-macros/src/manifold_expr.rs` | Manifold expression types |
| `pixelflow-macros/src/codegen/` | Code generation: `emitter.rs`, `struct_emitter.rs`, `binding.rs`, `leveled.rs`, `util.rs` |
| `pixelflow-ir/src/lib.rs` | Shared IR: Op trait, OpKind enum, Expr tree |
| `pixelflow-ir/src/ops.rs` | Operation unit structs |
| `pixelflow-ir/src/kind.rs` | OpKind enum |
| `pixelflow-ir/src/expr.rs` | Expression tree |
| `pixelflow-ir/src/traits.rs` | Backend trait definitions |
| `pixelflow-ir/src/features.rs` | Feature encoding for cost model |
| `pixelflow-ir/src/math.rs` | Math utilities |
| `pixelflow-ir/src/backend/` | Backend-specific lowering: `x86.rs`, `arm.rs`, `wasm.rs`, `scalar.rs`, `fastmath.rs` |
| `pixelflow-search/src/egraph/` | E-graph: `graph.rs`, `node.rs`, `ops.rs`, `saturate.rs`, `rewrite.rs`, `rules.rs`, `extract.rs`, `cost.rs`, `deps.rs`, `codegen.rs`, `algebra.rs`, `best_first.rs`, `guided.rs`, `nnue_adapter.rs` |
| `pixelflow-search/src/domain/` | Domain-specific algebra: `algebra.rs` |
| `pixelflow-search/src/model.rs` | Cost model |
| `pixelflow-nnue/src/lib.rs` | NNUE network for cost estimation, HalfEP features |
| `pixelflow-nnue/src/factored.rs` | Factored NNUE network variant |

### Graphics & Rendering

| Path | Purpose |
|------|---------|
| `pixelflow-graphics/src/lib.rs` | Re-exports and module declarations |
| `pixelflow-graphics/src/render/` | Rasterization engine: `aa.rs`, `color.rs`, `discrete.rs`, `frame.rs`, `pixel.rs`, `rasterizer/` (actor, parallel, messages) |
| `pixelflow-graphics/src/fonts/` | Font system: `loader.rs`, `cache.rs`, `combinators.rs`, `text.rs`, `ttf.rs`, `ttf_curve_analytical.rs` |
| `pixelflow-graphics/src/scene3d.rs` | 3D scene graph and transformations |
| `pixelflow-graphics/src/shapes.rs` | Shape primitives |
| `pixelflow-graphics/src/animation.rs` | Animation support |
| `pixelflow-graphics/src/mesh.rs` | Mesh data structures |
| `pixelflow-graphics/src/patch.rs` | Patch-based rendering |
| `pixelflow-graphics/src/image.rs` | Image handling |
| `pixelflow-graphics/src/transform.rs` | Geometric transformations |
| `pixelflow-graphics/src/spatial_bsp.rs` | BSP tree for spatial queries |
| `pixelflow-graphics/src/subdiv/` | Subdivision surfaces: `coeffs.rs` |
| `pixelflow-graphics/src/baked.rs` | Pre-baked rendering data |

### Runtime & Platform

| Path | Purpose |
|------|---------|
| `pixelflow-runtime/src/lib.rs` | Runtime entry point, module declarations |
| `pixelflow-runtime/src/display/drivers/` | Display drivers: `headless.rs`, `metal.rs`, `web/` (no standalone X11 driver - X11 handled via platform layer) |
| `pixelflow-runtime/src/display/` | Display abstractions: `driver.rs`, `messages.rs`, `ops.rs`, `platform.rs` |
| `pixelflow-runtime/src/platform/macos/` | macOS Cocoa: `cocoa.rs`, `events.rs`, `objc.rs`, `platform.rs`, `sys.rs`, `window.rs` |
| `pixelflow-runtime/src/platform/linux/` | Linux platform: `platform.rs`, `events.rs`, `window.rs` |
| `pixelflow-runtime/src/platform/waker.rs` | Cross-platform thread waking |
| `pixelflow-runtime/src/api/` | Public and private API surfaces: `public.rs`, `private.rs` |
| `pixelflow-runtime/src/engine_troupe.rs` | Actor-based render orchestration |
| `pixelflow-runtime/src/vsync_actor.rs` | Vsync coordination actor |
| `pixelflow-runtime/src/render_pool.rs` | Render thread pool management |
| `pixelflow-runtime/src/frame.rs` | Frame buffer management |
| `pixelflow-runtime/src/config.rs` | Runtime configuration |
| `pixelflow-runtime/src/input.rs` | Input event handling |
| `pixelflow-runtime/src/testing/` | Test infrastructure: `mock_engine.rs` |

### Actor System

| Path | Purpose |
|------|---------|
| `actor-scheduler/src/lib.rs` | Priority channel implementation (Control/Management/Data lanes) |
| `actor-scheduler/src/kubelet.rs` | Actor lifecycle management (kubelet pattern) |
| `actor-scheduler/src/lifecycle.rs` | Actor lifecycle states and transitions |
| `actor-scheduler/src/service.rs` | Service abstractions |
| `actor-scheduler/src/registry.rs` | Actor registry |
| `actor-scheduler/src/spsc.rs` | Single-producer single-consumer queue |
| `actor-scheduler/src/sharded.rs` | Sharded dispatch |
| `actor-scheduler/src/params.rs` | Configuration parameters |
| `actor-scheduler/src/error.rs` | Error types |
| `actor-scheduler-macros/src/lib.rs` | Troupe macro for actor group management |

### Terminal Application

| Path | Purpose |
|------|---------|
| `core-term/src/main.rs` | Application entry point |
| `core-term/src/terminal_app.rs` | Top-level terminal application actor |
| `core-term/src/config.rs` | Terminal configuration |
| `core-term/src/keys.rs` | Key binding definitions |
| `core-term/src/messages.rs` | Inter-actor message types |
| `core-term/src/color.rs` | Terminal color handling |
| `core-term/src/glyph.rs` | Glyph rendering bridge |
| `core-term/src/term/` | Terminal emulator: `screen.rs`, `cursor.rs`, `modes.rs`, `layout.rs`, `action.rs`, `charset.rs`, `snapshot.rs`, `unicode.rs` |
| `core-term/src/term/emulator/` | Emulator handlers: `ansi_handler.rs`, `char_processor.rs`, `cursor_handler.rs`, `input_handler.rs`, `key_translator.rs`, `mode_handler.rs`, `osc_handler.rs`, `screen_ops.rs`, `methods.rs` |
| `core-term/src/ansi/` | ANSI parser: `lexer.rs`, `parser.rs`, `commands.rs` |
| `core-term/src/io/` | PTY I/O: `pty.rs`, `traits.rs`, `kqueue.rs` (macOS), `epoll.rs` (Linux) |
| `core-term/src/io/event_monitor_actor/` | Event monitor: `read_thread/`, `write_thread/`, `parser_thread/` |
| `core-term/src/surface/` | Terminal rendering surface: `manifold.rs`, `terminal.rs` |

### Build & Development

| Path | Purpose |
|------|---------|
| `xtask/src/main.rs` | Build tasks (bundle-run for macOS app packaging) |
| `xtask/src/codegen.rs` | Code generation tasks |
| `Cargo.toml` | Workspace definition and build profiles |
| `rust-toolchain.toml` | Rust stable toolchain pinning |
| `pixelflow-core/build.rs` | SIMD backend detection (CPU feature probing) |
| `pixelflow-runtime/build.rs` | Platform feature detection (X11 pkg-config) |
| `core-term/build.rs` | Terminal build configuration |

## Architecture Documentation

### Available Documentation

Detailed design docs in `docs/`:
- `STYLE.md` - Coding style guide and conventions
- `AUTODIFF_RENDERING.md` - Automatic differentiation for antialiasing
- `MESSAGE_CUJ_COVERAGE.md` - Message passing critical user journeys
- `COMPILER_ANALYSIS.md` - Compiler pipeline analysis
- `COMPILER_OPPORTUNITIES.md` - Optimization opportunities
- `EGRAPH_SEARCH_INTEGRATION.md` - E-graph search integration details
- `FLAT_CONTEXT_TUPLE_PROTOTYPE.md` - Context representation prototype
- `GNN_REWRITE_GUIDANCE_VISION.md` - GNN-guided rewrite vision
- `KERNEL_PARAM_LIMIT_INVESTIGATION.md` - Kernel parameter limit analysis
- `NNUE_INTEGRATION_STATUS.md` - NNUE neural network integration status
- `lample_charton_2019_symbolic_math.md` - Symbolic math reference notes
- `gemini/` - Gemini AI integration documentation

Design documents in `docs/designs/`:
- `compiler-architecture-2026.md` - Compiler architecture roadmap
- `nnue-training-pipeline.md` - NNUE training pipeline design
- `actor-scheduler-supervisor-migration.md` - Supervisor migration plan

Templates in `docs/templates/`:
- `DESIGN_DOC.md` - Template for new design documents

Reference PDFs in `docs/`:
- `fop-conal.pdf`, `type-class-morphisms-long.pdf`, `2425_deep_learning_for_symbolic_mat.pdf`

### Top-Level Documentation

- `README.md` - Project overview, philosophy, getting started
- `CLAUDE.md` - This file: AI assistant development guide
- `GEMINI.md` - Gemini AI assistant guide
- `PERFORMANCE.md` - Performance benchmarks and targets
- `PERFORMANCE_ANALYSIS.md` - Detailed performance analysis
- `PIXELFLOW_IR_STATUS.md` - Status of pixelflow-ir crate (non-workspace member)
- `LICENSE.md` - Apache license
- `.claude/DOCS_TOC.md` - Auto-generated documentation table of contents

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
- **Display driver:** X11 via the `x11` crate (feature-gated with `display_x11`), integrated through platform layer
- **PTY I/O:** epoll-based async I/O (`core-term/src/io/epoll.rs`)
- **Dependencies:** Requires X11 development headers (libx11-dev, libxft-dev, etc.), detected via `pkg-config` in `build.rs`
- **Platform code:** `pixelflow-runtime/src/platform/linux/` (directory: `platform.rs`, `events.rs`, `window.rs`)

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

## pixelflow-ml: Neural Networks for Graphics & Compiler Optimization

The `pixelflow-ml` crate has evolved from a research crate into a practical training and evaluation framework.

### Core Insight

**Linear Attention IS Harmonic Global Illumination** - both solve the same problem:
- Compress infinite/quadratic interactions into finite/linear operations
- Project onto orthogonal basis functions (φ for attention, Y_lm for SH)
- Compute via efficient inner products instead of exhaustive summation

### Architecture

| Module | Purpose |
|--------|---------|
| `lib.rs` | Core types, mathematical exposition |
| `layer.rs` | Neural network layer definitions |
| `nnue.rs` | NNUE network implementation |
| `nnue_trainer.rs` | NNUE training loop |
| `evaluator.rs` | Expression evaluation |
| `hce_extractor.rs` | Hand-crafted evaluation feature extraction |
| `nonlinear_eval.rs` | Non-linear evaluation functions |
| `graphics.rs` | Graphics integration (SH features) |
| `benchmark.rs` | Benchmarking utilities |
| `training/` | Training infrastructure: `backprop.rs`, `data_gen.rs`, `egraph.rs`, `factored.rs`, `features.rs` |

### Feature Flags

- `graphics` (default) - Enable graphics integration, requires `pixelflow-core`
- `std` - Enable std for benchmarking and training data generation
- `training` - Enable training data generation (implies `std`)
- `egraph-training` - Enable e-graph training examples (requires `pixelflow-search`, `pixelflow-nnue`)

### Current Status

- Edition 2024, API stabilizing
- Depends on `pixelflow-ir` (always), `pixelflow-core` (optional via `graphics` feature)
- Uses `libm` for `no_std` compatibility
- Extensive benchmarks: `hce_bench`, `expr_perf`, `manifold_hce_validation`, `nnue_training_suite`, `generated_kernels`

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

## Feature Flags

Key feature flags across the workspace:

| Crate | Feature | Purpose |
|-------|---------|---------|
| `pixelflow-runtime` | `display_cocoa` | macOS Cocoa display driver |
| `pixelflow-runtime` | `display_x11` | Linux X11 display driver (pulls in `x11` crate) |
| `pixelflow-runtime` | `display_headless` | Headless driver for CI/testing |
| `pixelflow-runtime` | `display_web` | WebAssembly display driver |
| `pixelflow-ir` | `std`, `alloc` | Standard library and allocator support (both default) |
| `pixelflow-ml` | `graphics` (default) | Graphics integration with `pixelflow-core` |
| `pixelflow-ml` | `training` | Training data generation (implies `std`) |
| `pixelflow-ml` | `egraph-training` | E-graph training examples (implies `std`, requires `pixelflow-search`, `pixelflow-nnue`) |
| `pixelflow-nnue` | `std` (default) | Standard library support |
| `pixelflow-nnue` | `training` | NNUE training support |
| `pixelflow-search` | `std` (default) | Standard library support |
| `pixelflow-graphics` | `serde` | Serialization support |
| `core-term` | `profiling` | Flamegraph profiling via `pprof` |
| `core-term` | `display_cocoa` | Force Cocoa driver |
| `core-term` | `display_x11` | Force X11 driver |
| `core-term` | `display_headless` | Force headless driver |

Platform auto-detection: `core-term/Cargo.toml` uses `[target.'cfg(...)'.dependencies]` to automatically select the right display driver per platform. Feature flags are only needed for manual override.

## Scripts

Development scripts in `scripts/`:
- `compute_log2_coefficients.py` - Compute polynomial coefficients for fast log2 approximation
- `compute_log2_simple.py` - Simplified log2 coefficient computation
- `fit_log2_polynomial.py` - Polynomial fitting for log2
- `gen-docs-toc.sh` - Generate documentation table of contents

## Quick Reference

### Most Common Commands

```bash
# Development
cargo build --release              # Build with optimizations
cargo test --workspace            # Run all tests
cargo run --release -p core-term  # Run terminal directly
cargo xtask bundle-run            # Build and run macOS app

# Performance
cargo bench -p pixelflow-core     # Benchmark core algebra (core_benches)
cargo bench -p pixelflow-macros   # Benchmark macro compilation (macro_bench)
cargo bench -p pixelflow-ml       # ML benchmarks (hce_bench, expr_perf, manifold_hce_validation, nnue_training_suite, generated_kernels)
cargo bench -p core-term          # Terminal benchmarks (ansi_parser, keybinding_benchmark)
cargo xtask bundle-run --features profiling  # Profile with flamegraph

# Platform-specific
cargo build -p core-term --features display_x11       # Force X11 (Linux)
cargo build -p core-term --features display_cocoa      # Force Cocoa (macOS)
cargo build -p core-term --features display_headless   # Force headless (CI)
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
  variables.rs        - Coordinate variables (symbol table)
  algebra.rs          - Algebraic structures
  ops/                - Operation types (base, binary, unary, compare, trig, logic, derivative, vector)
  backend/            - SIMD implementations (x86, arm, scalar, wasm, fastmath)
  combinators/        - at, map, select, fix, computed, binding, block, shift, pack, project, etc.
  jet/                - Automatic differentiation (jet2, jet2h, jet3, path_jet)
  storage.rs          - SIMD storage abstractions
  numeric.rs          - Numeric traits
  mask.rs             - SIMD mask operations
  zst.rs              - Zero-sized type utilities

pixelflow-macros/src/
  lib.rs              - kernel! macro entry point
  ast.rs              - AST node definitions
  parser.rs           - AST construction
  sema.rs             - Semantic analysis
  symbol.rs           - Symbol table
  optimize.rs         - AST optimization
  fold.rs             - AST visitor/fold
  ir_bridge.rs        - Bridge to shared IR
  rewrite_rules.rs    - Optimization rewrite rules
  cost_builder.rs     - Cost model construction
  codegen/            - Code emission (emitter, struct_emitter, binding, leveled, util)

pixelflow-ir/src/      (NOT a workspace member)
  lib.rs              - IR types and re-exports
  ops.rs              - Operation unit structs
  kind.rs             - OpKind enum
  expr.rs             - Expression tree
  traits.rs           - Backend traits
  features.rs         - Feature encoding
  backend/            - Target-specific lowering (x86, arm, wasm, scalar, fastmath)

pixelflow-search/src/
  egraph/             - E-graph optimization
    graph.rs          - Core e-graph structure
    node.rs           - E-graph nodes
    saturate.rs       - Rewrite saturation
    extract.rs        - Cost-based extraction
    rules.rs          - Rewrite rule definitions
    guided.rs         - NNUE-guided search
    best_first.rs     - Best-first search strategy
    nnue_adapter.rs   - NNUE adapter
  domain/             - Domain-specific algebra

pixelflow-nnue/src/
  lib.rs              - NNUE network, HalfEP features
  factored.rs         - Factored NNUE variant

pixelflow-graphics/src/
  render/             - Rasterization engine (aa, color, discrete, frame, pixel, rasterizer/)
  fonts/              - Font system (loader, cache, combinators, text, ttf, ttf_curve_analytical)
  scene3d.rs          - 3D scene graph
  shapes.rs           - Shape primitives
  animation.rs        - Animation support
  mesh.rs             - Mesh structures
  spatial_bsp.rs      - BSP spatial queries

pixelflow-runtime/src/
  display/drivers/    - headless, Metal, Web (no standalone X11 driver)
  display/            - driver, messages, ops, platform abstractions
  platform/macos/     - Cocoa integration (cocoa, events, objc, platform, sys, window)
  platform/linux/     - Linux platform (platform, events, window)
  api/                - Public and private API surfaces
  engine_troupe.rs    - Render actor orchestration
  vsync_actor.rs      - Vsync coordination
  render_pool.rs      - Render thread pool
  testing/            - Mock engine for tests

pixelflow-ml/src/
  lib.rs              - Core types, math exposition
  training/           - Training infrastructure (backprop, data_gen, egraph, factored, features)
  nnue.rs             - NNUE implementation
  hce_extractor.rs    - Hand-crafted evaluation extraction

core-term/src/
  main.rs             - Entry point
  terminal_app.rs     - Top-level actor
  term/               - Terminal state machine (screen, cursor, modes, layout, emulator/)
  term/emulator/      - Handlers (ansi, char, cursor, input, key_translator, mode, osc, screen_ops)
  ansi/               - Escape sequence parser (lexer, parser, commands)
  io/                 - PTY actors (pty, kqueue, epoll, event_monitor_actor/)
  surface/            - Rendering surface (manifold, terminal)
  config.rs           - Configuration
  keys.rs             - Key bindings
  messages.rs         - Message types
  color.rs            - Color handling
  glyph.rs            - Glyph bridge
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
7. **Platform on main thread** - Especially macOS Cocoa
