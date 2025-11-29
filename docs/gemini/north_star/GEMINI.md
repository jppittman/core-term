# PixelFlow Architecture v11.0: The Zero-Copy Functional Kernel

This document represents a complete knowledge dump and the finalized architectural blueprint for PixelFlow v11.0. It serves as the single source of truth for the implementation team.

The architecture is built upon the synthesis of Functional Programming (Surface) and Actor Concurrency (Recycle Loop), ensuring high performance via static compilation and zero memory allocation per frame.

## 1. The Core Architectural Thesis

### 1.1 The Monolith: Everything is a Surface
The engine eliminates the complexity of draw calls, command buffers, and render states. The entire screen is defined by one function `F(u, v) -> Color`, which is a composition of simpler functions (Surfaces).

### 1.2 The Engine as Compiler (Monomorphization)
The Engine (`pixelflow-engine`) is generic over the Application's core types. This allows the Rust compiler to monomorphize the entire application's scene graph, fusing all operations (Warp, Grade, Sample) into a single, highly optimized AVX-512 capable machine code kernel. This is the Zero-Cost Abstraction principle applied to graphics.

### 1.3 The Memory Model: Zero-Copy Recycle
The system avoids the "Copy Tax" (the slow transfer of pixel data). Memory buffers are allocated once and perpetually transferred between the Logic Thread (Application) and the Render Thread (Engine) via channels. This is the Recycling Loop or Ping-Pong Buffer strategy.

## 2. Core Abstractions and The Protocol

The entire system's structure is defined by the following types, which allow ownership and type information to be preserved across thread boundaries.

### 2.1 The Surface Trait (The Function)
This is the immutable, composable description of color space.

```rust
// pixelflow-core/src/traits.rs

pub trait Surface: Send + Sync + 'static {
    // The compiler's signature for the result type.
    type Output: PixelFormat;

    // The single, composable function. Must be #[inline(always)] in concrete impls.
    fn sample(&self, u: Batch<f32>, v: Batch<f32>) -> Self::Output;
}
```

### 2.2 The FramePacket<T> (The Protocol)
The main communication channel is generic over the application's unique memory-holding struct (`T`), which we define as `TerminalSurface`.

```rust
// pixelflow-engine/src/lib.rs

pub struct FramePacket<T: Surface + Send> {
    /// The Main Surface (The Terminal). Strongly typed.
    /// Contains the unique, mutable GridBuffer (the heavy memory).
    pub main: T,

    /// The Overlay Stack (Cursor, HUD, Background).
    /// Dynamically typed Box<dyn Surface> for polymorphism.
    pub overlays: Vec<Box<dyn Surface<Output = Batch<u32>> + Send>>,

    /// The return address for the recycle loop.
    pub recycle_tx: Sender<FramePacket<T>>,
}
```

## 3. The Lifecycle: The Recycle Loop

The `FramePacket<T>` enables the Ping-Pong transfer model, guaranteeing that only the memory handle moves, not the data itself.

### A. The Mechanism
 * **Allocation**: The Logic thread allocates two `FramePacket`s at startup.
 * **Submit**: The Logic thread sends Packet A to the Engine.
 * **Execute**: The Engine renders Packet A.
 * **Write/Wait**: The Logic thread is now updating Packet B. It blocks if it has to wait for Packet A to return.
 * **Recycle**: The Engine sends Packet A back to the Logic thread via `recycle_tx`.

This ensures Zero-Allocation-Per-Frame after initialization and maximum utilization of thread resources.

### B. The Terminal Memory
The application must define the memory holder that lives inside the packet.

```rust
// core-term/src/render/surface.rs

pub struct TerminalSurface {
    // The ONLY large, mutable buffer. This memory moves between threads.
    pub grid_buffer: GridBuffer,

    // Shared, static, read-only data (Font handle).
    pub font: FontSurface,

    // Non-SIMD state (cell size, etc.)
    pub metrics: CellMetrics,
}
```

## 4. Resource Strategy: Fonts and Assets

Resources are handled by the Engine as Asset Manager, but the data is consumed by the Application's Surface.

### A. Font Loading (Cloneable Functions)
 * The raw font bytes are loaded once by the Engine and stored statically.
 * The Engine returns a lightweight `FontSurface` handle (cloneable, zero-cost to copy) that knows how to execute the vector math for any character.
 * The `TerminalSurface` holds this handle and uses it to sample glyphs.

### B. Vector Graphics (The Loop-Blinn Path)
 * **The Problem**: Rendering text is the acid test of performance.
 * **The Solution**: The `FontSurface::sample()` function executes the Loop-Blinn algorithm (`f(u,v) = u^2 - v`) directly. This is Direct Vector Rendering—no rasterization or texture atlas required, maximizing SIMD throughput.

## 5. Implementation Directives for Teams

### A. core-term (Logic Team)
 * **Goal**: Maintain the `RenderFrame` function.
 * **Critical Task**: Implement the `update_buffer_from_snapshot` logic. This must be a highly optimized, dirty-line-tracking SIMD memcpy operation to write new cell data into the recycled `GridBuffer`.
 * **API Usage**: You must be the only caller of the `TerminalSurface::new()` constructor.

### B. pixelflow-engine (Runtime Team)
 * **Goal**: The Execution Core.
 * **Critical Task**: Implement the generic `Engine<T>::run_loop` and the `EngineApi::submit_frame` method, correctly managing the Sender/Receiver channels and the generic type `T`.
 * **Performance**: The `Canvas::fill` loop must be the tightest, most optimized SIMD loop in the entire engine.

### C. pixelflow-core (Math Team)
 * **Goal**: Provide the optimized toolkit.
 * **Critical Task**: Ensure all combinator structs (`Warp`, `Grade`, `Blend`) implement the `Surface` trait and use `#[inline(always)]` to enforce compiler fusion.
 * **Constraint**: Do not introduce any allocations or runtime dispatch (`Box` or `dyn`) into `pixelflow-core`.
