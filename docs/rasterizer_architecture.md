# Materialization Architecture

*Note: This document replaces the legacy "Rasterizer Compiler" architecture.*

## Overview

In PixelFlow 1.0, "Rasterization" is redefined as **Materialization**. It is the process of sampling a continuous `Surface` at discrete pixel coordinates to produce a frame in a memory buffer.

## The Process

Materialization is the bridge between the algebraic world (`pixelflow-core`) and the physical world (pixels on a screen).

1.  **The Source**: A `Surface<Color>` describing the entire scene.
2.  **The Target**: A mutable byte buffer (the framebuffer).
3.  **The Loop**: A tight, optimized loop that iterates over the target buffer's coordinates.

```rust
// Simplified Mental Model
for y in 0..height {
    for x in (0..width).step_by(LANES) {
        // 1. Generate Coordinates (Fields)
        let u = Field::sequential_from(x);
        let v = Field::splat(y);

        // 2. Sample the Surface
        let color_batch = surface.eval(u, v);

        // 3. Write to Buffer
        color_batch.store(&mut buffer[index..]);
    }
}
```

## Zero-Copy & Pull-Based

*   **No Intermediate Buffers**: The Surface is sampled directly into the final framebuffer. There are no "draw calls" that rasterize to temporary textures.
*   **Pull-Based**: We don't push geometry. We pull colors. This allows for infinite detail, procedural generation, and resolution independence.

## Driver Interface

The `Driver` (in `pixelflow-engine` or platform-specific backends) exposes a minimal interface:

*   `get_framebuffer_mut()`: Provides access to the raw pixel memory.
*   `present()`: Swaps buffers / informs the OS to display the content.

The Engine's main loop calls `materialize(surface, driver.get_framebuffer_mut())` every frame.

## Optimization

All optimization happens **inside the Surface evaluation**, via composition:

*   **Clipping**: Using `Select` to skip expensive computations for pixels outside a mask.
*   **Monomorphization**: The compiler fuses the entire combinator tree into a single function, allowing register-level optimizations across the entire pipeline.
