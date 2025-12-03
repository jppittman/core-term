# CoreTerm (PixelFlow v11.0) Context

## Project Overview

**CoreTerm** is a high-performance, correct terminal emulator built on the **PixelFlow v11.0** architecture. It distinguishes itself by using a **Zero-Copy Functional Kernel** for rendering, avoiding GPU dependencies in favor of highly optimized CPU-based SIMD operations (AVX-512).

### Core Architecture
The project is a synthesis of Functional Programming and Actor Concurrency:
*   **Rendering (PixelFlow):** The entire screen is defined by a single function `F(u, v) -> Color` (the "Surface"). The engine acts as a compiler, monomorphizing the scene graph into a single executable kernel.
*   **Concurrency:** A **Three-Thread Actor Model** ensures zero-latency input and clean separation of concerns:
    1.  **Display Driver (Main Thread):** Handles platform-specific UI events (Cocoa/X11).
    2.  **Orchestrator (Logic Thread):** Platform-agnostic application logic, state management.
    3.  **PTY I/O (Background Thread):** Asynchronous pseudo-terminal I/O.
*   **Memory:** Uses a "Ping-Pong" buffer strategy to recycle memory between the Logic and Render threads, ensuring zero allocation per frame.

### Workspace Structure
The project is a Cargo workspace:
*   `core-term/`: The main application (Logic/Orchestrator).
    *   `src/main.rs`: Entry point. Initializes actors and platform.
    *   `src/config.rs`: Configuration logic (currently defaults-only).
*   `pixelflow-engine/`: The execution core and runtime environment (Render Loop).
*   `pixelflow-core/`: Zero-cost SIMD math abstractions and trait definitions.
*   `pixelflow-render/`: Software rendering primitives (Surfaces).
*   `pixelflow-fonts/`: Vector font handling (Loop-Blinn).
*   `xtask/`: Build automation scripts.

## Building and Running

### Prerequisites
*   **Rust:** Stable channel.
*   **Linux Dependencies:** `libx11-dev`, `libxext-dev`, `libxft-dev`, `libfontconfig1-dev`, `libfreetype6-dev`, `libxkbcommon-dev`.
*   **macOS:** Standard Xcode command line tools.

### Commands

| Action | Command | Notes |
| :--- | :--- | :--- |
| **Build** | `cargo build --release` | Release mode is strongly recommended for SIMD performance. |
| **Run** | `cargo run --release` | Runs the raw binary. |
| **Bundle & Run (macOS)** | `cargo xtask bundle-run` | Builds, creates `CoreTerm.app`, and launches it. |
| **Test** | `cargo test` | Runs standard unit tests. |

**Note:** When running via `xtask bundle-run` on macOS, logs are redirected to `/tmp/core-term.log`.

## Development Conventions

*   **Documentation:**
    *   **Reference:** Use `docs/STYLE.md` for coding standards.
    *   **Outdated:** Ignore `docs/NORTH_STAR.md`. It contains obsolete architectural details.
    *   **Sources:** Trust the `README.md` files in sub-crates (`pixelflow-core`, etc.) and this document.
*   **Performance:**
    *   The Render Thread (`pixelflow-engine`) must **never** allocate memory during the frame loop.
    *   The `sample` function in `pixelflow-core` surfaces must be `#[inline(always)]`.
*   **Code Style:** Standard Rust formatting (`cargo fmt`). Public items require documentation (`///`).
*   **Platform Handling:** Platform-specific code (Cocoa/X11) belongs in the Display Driver actor; logic belongs in the Orchestrator.

## Configuration
Configuration is currently defined in `core-term/src/config.rs` with a `Lazy` static `CONFIG`.
*   File loading is a placeholder (defaults are hardcoded).
*   Keybindings, appearance, and behavior are all struct-defined.
