# CoreTerm & PixelFlow: Agent Context

**Status**: Active Transition (v11.0 → v1.0)
**Primary Goal**: Refactor the codebase to align with the **PixelFlow 1.0 North Star** architecture.

---

## 1. PixelFlow 1.0: North Star (The Target Architecture)

**Crucial**: This is the architectural standard for all new code and refactoring.

### The Thesis
SIMD is not an optimization. It is the algebraic realization of the Field of Real Numbers. PixelFlow 1.0 resolves the false dichotomy between mathematical abstraction and hardware intrinsics.
**Write equations. Get assembly.**

### The Inversion (Pull-Based)
*   **Pull, don't push.** Surfaces are functions `F(x,y) -> T`. Nothing computes until sampled.
*   **Laziness is the contract.** Masks are just `Surface<bool>`. Bounds are Surfaces.
*   **No pre-pass.** Optimization happens via composition (e.g., `clip` combinator).
*   **The Observer is Fixed.** The camera is at `(0,0,0)`. The world warps around it. "It is always now."

### The Algebra
*   **Field**: The computational atom (SIMD vector, lane-agnostic).
*   **Surface**: `Fn(x, y) -> T`
*   **Volume**: `Fn(x, y, z) -> T`
*   **Manifold**: `Fn(x, y, z, w) -> T`
*   **Dimensional Collapse**: `Manifold` implies `Volume` (w=0). `Volume` implies `Surface` (z=0).

### The Six Eigenshaders
All shaders are compositions of these six primitives:
1.  **Warp**: `(S, ω) → S` (Coordinate remapping)
2.  **Grade**: `(S, M, b) → S` (Linear transform of values)
3.  **Lerp**: `(t, a, b) → S` (Interpolation)
4.  **Select**: `(cond, t, f) → S` (Branchless conditional, used for clipping)
5.  **Fix**: `(seed, step) → V` (Iteration as a dimension)
6.  **Compute**: `Fn(x,y) → T` (Escape hatch)

---

## 2. Current Project State (v11.0)

The codebase is currently in **v11.0**, which uses a slightly different terminology (Monolith, Ping-Pong buffers). We are migrating to v1.0.

### Directory Structure & Migration Map

| Current Crate | Purpose (v11.0) | Target State (v1.0) |
| :--- | :--- | :--- |
| `core-term/` | Main Application (Orchestrator) | **Application**. Depends on `pixelflow-engine`. |
| `pixelflow-core/` | Math & Traits | **Core**. Pure algebra. Needs heavy refactoring to implement Eigenshaders. |
| `pixelflow-render/` | Software Rendering Primitives | **Merge into `pixelflow-graphics`**. |
| `pixelflow-fonts/` | Font handling | **Merge into `pixelflow-graphics`**. |
| `pixelflow-engine/` | Execution Runtime | **Engine**. Scene graph, input, loop. |
| `actor-scheduler/` | Actor Runtime | Keep/Integrate into Engine. |
| `xtask/` | Build scripts | Keep. |

### Key Files
*   `core-term/src/main.rs`: Entry point.
*   `pixelflow-core/src/lib.rs`: The heart of the math library.
*   `GEMINI.md`: This file.
*   `docs/NORTH_STAR.md`: (Note: May contain v11.0 info, trust this file's "North Star" section over it).

---

## 3. Operational Guide

### Build & Run
*   **Build Release**: `cargo build --release` (Critical for SIMD performance)
*   **Run**: `cargo run --release`
*   **Bundle (macOS)**: `cargo xtask bundle-run` (Builds `.app` and launches)
*   **Test**: `cargo test`

### Dependencies (Linux)
`libx11-dev`, `libxext-dev`, `libxft-dev`, `libfontconfig1-dev`, `libfreetype6-dev`, `libxkbcommon-dev`.

---

## 4. Development Conventions

*   **Architecture First**: When modifying `pixelflow-core`, always verify alignment with the "Six Eigenshaders" model.
*   **No Allocation**: The render loop (`pixelflow-engine`) must **never** allocate.
*   **Documentation**: Public items require doc comments.
*   **Tests**: Add unit tests for new algebra primitives.
*   **Refactoring**: Do not be afraid to delete v11.0 concepts (e.g., "Rasterizer" traits) if they conflict with the v1.0 "Surface" model.
