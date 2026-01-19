# PixelFlow (core-term) Project Context

## Project Overview

**PixelFlow** is a research project demonstrating a novel paradigm for real-time graphics: **pull-based rendering** with **SIMD as algebra**. It achieves high performance (155 FPS at 1080p) on pure CPU without a GPU.

**`core-term`** is the primary consumer application: a high-performance, correct terminal emulator built entirely on the PixelFlow engine.

### Core Philosophy
1.  **Pull-based Rendering:** Pixels are sampled, not pushed. The system asks "what color is this pixel?", eliminating overdraw and complex rasterization state.
2.  **SIMD as Algebra:** The `Field` type wraps SIMD vectors (AVX-512, SSE2, NEON) transparently. Users write algebraic equations, and the compiler emits optimal vectorized assembly.
3.  **Manifold Abstraction:** Everything is a `Manifold` (a functor from 4D coordinates to a value). Composing manifolds creates complex scenes that are compiled into fused kernels.
4.  **Zero Allocations:** The rendering loop is designed to have zero heap allocations per frame.

## Workspace Structure

The project is a Rust workspace with the following key members:

*   **`core-term`**: The terminal emulator application. (First consumer)
*   **`pixelflow-core`**: Pure algebra, `Field`, `Manifold` traits. `no_std`, SIMD backend implementations.
*   **`pixelflow-graphics`**: Rendering logic, colors, fonts, rasterization.
*   **`pixelflow-runtime`**: Platform abstraction (Cocoa, X11, Web), input handling, render orchestration.
*   **`actor-scheduler`**: Lock-free, priority-based actor concurrency model (`Control > Management > Data` lanes).
*   **`xtask`**: Build automation (bundling macOS apps, etc.).

## Building and Running

### Prerequisites
*   **Rust Nightly:** (See `rust-toolchain.toml`)
*   **macOS:** Native Cocoa support.
*   **Linux:** X11 development headers (`libx11-dev`, `libxft-dev`, etc.).

### Key Commands

*   **Build Release:** `cargo build --release`
*   **Run Terminal:** `cargo run --release -p core-term`
*   **Run macOS App:** `cargo xtask bundle-run` (Bundles and runs `CoreTerm.app`)
*   **Run Tests:** `cargo test --workspace`
*   **Benchmarks:** `cargo bench -p pixelflow-core`

### Build Profiles
*   **`dev`**: `opt-level = 2` (Required to prevent stack overflows from deep Manifold recursion).
*   **`release`**: `opt-level = 3`, `lto = true`, `codegen-units = 1`.

## Development Conventions

### Architectural Constraints
*   **No Terminal Logic in PixelFlow:** Keep `pixelflow-*` crates general-purpose. Terminal specific logic belongs in `core-term`.
*   **Pull, Don't Push:** Rendering logic must adhere to the pull-based paradigm.
*   **Types are Shaders:** Use the type system to build compute graphs.
*   **Platform Isolation:** Platform-specific code (macOS/Linux/Web) goes in `pixelflow-runtime`.

### Coding Style (See `docs/STYLE.md`)
*   **Comments:**
    *   **Public API (`///`):** Document **WHAT** and **HOW**.
    *   **Implementation (`//`):** Document **WHY**. Explain design rationale, not obvious logic.
    *   **No History:** Do not put changelogs or "old code" in comments.
*   **Structure:**
    *   Avoid deep nesting; use guard clauses.
    *   Prefer `match` over `else if`.
*   **Functions:**
    *   Keep argument count low (< 4). Group related args into structs.
    *   **No Boolean Args:** Use enums for clarity (e.g., `Persistence::Permanent` vs `true`).
*   **Magic Numbers:** Use named constants or enums.

### Git & Workflow
*   **Atomic Commits:** Focus on one logical change per commit.
*   **Commit Messages:** Explain *why* a change was made.
*   **Tests:** Public API changes require test updates.
