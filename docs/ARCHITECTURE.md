# PixelFlow 1.0 Architecture

This document outlines the high-level architecture of PixelFlow 1.0, based on the [North Star](./NORTH_STAR.md) vision.

## 1. Architectural Principles

PixelFlow defines a **Zero-Copy Functional Kernel** for graphics. It rejects the traditional push-based rasterization pipeline (vertices → triangles → fragments) in favor of a pull-based algebraic model.

*   **Pull-Based Evaluation**: Pixels are not "drawn". Instead, the engine iterates over the screen coordinates and *samples* the scene description (the Surface) to determine the color at that point.
*   **SIMD as Algebra**: The core abstraction is the `Field` (a SIMD vector). Operations are defined algebraically, and the compiler generates efficient vectorized assembly (AVX-512, NEON, etc.).
*   **The Fixed Observer**: The camera is fixed at the origin. Movement and animation are achieved by warping the coordinate space of the world around the observer.
*   **Laziness**: Nothing computes until sampled. Scene descriptions are cheap and infinite; only the pixels actually seen are computed.

## 2. Crate Architecture

The project is organized into layers of increasing abstraction:

```
pixelflow-core        Pure algebra. Field, Surface, Volume, Manifold.
      ↓               Warp, Grade, Lerp, Select, Fix, Compute.
      ↓               No IO, no platform, no pixels, no colors.
      ↓
pixelflow-graphics    Materialization & Resources.
      ↓               (Currently pixelflow-render, pixelflow-fonts)
      ↓               Color spaces, Font loading, Compositing (Over).
      ↓
pixelflow-engine      The Runtime.
      ↓               Scene graph, Input handling, Windowing, Render Loop.
      ↓
  application         (e.g., core-term)
                      The end-user application logic.
```

### 2.1 `pixelflow-core` (The Math)
This is the pure algebraic kernel. It depends only on `std` (or `core` in no_std).
*   **Primitives**: `Field`, `Surface<T>`, `Volume<T>`, `Manifold<T>`.
*   **Eigenshaders**: `Warp`, `Grade`, `Lerp`, `Select`, `Fix`, `Compute`.
*   It knows nothing about pixels, colors, or windows. It just maps coordinates to values.

### 2.2 `pixelflow-graphics` (The Bridge)
*Note: Currently implemented across `pixelflow-render` and `pixelflow-fonts`.*
*   **Colors**: Definitions of `Color` (RGBA, etc.).
*   **Compositing**: Porter-Duff `Over` operator (which is `Lerp` + alpha blending).
*   **Fonts**: Loading fonts and exposing them as `Surface`s.
*   **Materialization**: The `materialize` function that drives the render loop, evaluating the surface for every pixel in a target buffer.

### 2.3 `pixelflow-engine` (The Runtime)
*   **Windowing**: Manages the OS window (using Winit, Wayland, or Cocoa directly).
*   **Input**: Captures keyboard/mouse events and translates them to algebraic changes (e.g., updating a `Warp`).
*   **The Loop**: Runs the game loop, calls `materialize` to fill the framebuffer, and presents it to the OS.

### 2.4 Application (`core-term`)
*   **Logic**: Defines the scene (Surfaces) and how they change over time.
*   **State**: Manages application state (e.g., terminal grid, cursor position).
*   **Architecture**: Uses an Actor model to handle I/O and Logic concurrently, submitting Surface descriptions to the Engine for rendering.

## 3. The Render Flow

1.  **Construction**: The Application constructs a `Surface` describing the current frame. This is a tree of combinators (e.g., `Background.over(Text.warp(Scroll))`).
2.  **Submission**: The Application hands this Surface to the Engine.
3.  **Materialization**: The Engine's render loop iterates over the window's pixel buffer.
    *   For each tile/scanline, it generates SIMD coordinate vectors (`Field`s).
    *   It calls `eval(x, y)` on the root Surface.
    *   The call propagates down the combinator tree.
    *   The resulting color values are written to the framebuffer.
4.  **Presentation**: The Engine swaps buffers and presents the frame to the OS.

## 4. Optimization Strategy

Optimization is not a separate phase; it is **Composition**.

*   **Bounds**: There is no culling pass. Bounds are just `Mask` surfaces (return 0 outside).
*   **Clipping**: The `clip(mask)` combinator uses `Select` to skip evaluation of the expensive branch if the mask is 0.
*   **Monomorphization**: The entire Surface tree is a single type. The Rust compiler inlines all `eval` calls into a single fused kernel, eliminating virtual function overhead.

## 5. Threading Model

*   **Logic Thread**: Runs the Application (Actors). Updates state, constructs the Scene Surface.
*   **Render Thread**: Owned by the Engine. Runs the Materialization loop.
*   **Input Handling**: Platform-dependent (often Main Thread). Forwards events to Logic.

Note: `core-term` specifically uses a 3-thread actor model (Display/Main, Orchestrator/Logic, PTY/IO) as detailed in `ACTOR_ARCHITECTURE.md`.
