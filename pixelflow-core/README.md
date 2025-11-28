# PixelFlow Core

**PixelFlow Core** is a zero-cost, type-driven SIMD abstraction library designed for high-performance pixel operations. It serves as the foundation for the PixelFlow rendering ecosystem, providing efficient primitives for image processing, tensor manipulation, and pipeline composition.

## Key Features

*   **Type-Driven SIMD**: The `Batch<T>` type automatically selects the optimal SIMD instruction set and lane width based on the data type:
    *   `Batch<u32>`: 32-bit operations (4 lanes)
    *   `Batch<u16>`: 16-bit operations (8 lanes)
    *   `Batch<u8>`: 8-bit operations (16 lanes)
*   **Zero-Cost Abstractions**: Compiles down to efficient hardware intrinsics (SSE2/NEON) without runtime overhead.
*   **Tensor Operations**: Provides `TensorView` and `TensorViewMut` for efficient 2D grid access, including optimized strided access and sub-view creation.
*   **Pipeline DSL**: Includes a Domain-Specific Language (DSL) for composing rendering pipelines with operations like `offset`, `skew`, and `over`.
*   **`no_std` Compatible**: Designed for embedded and bare-metal environments.

## New Shader Algebra (Proposed)

We are transitioning to a new "Shader Algebra" design based on functional combinators. This new approach offers:

*   **Coordinate-Based Surfaces**: Treating coordinates as surfaces (`X`, `Y`) enables arbitrary warping and distortion.
*   **Automatic Format Elision**: The type system optimizes away unnecessary color format conversions (packed vs. planar).
*   **Unified Abstraction**: Everything is a `Surface` (texture, noise, constant color), allowing seamless composition.

See [docs/SHADER_ALGEBRA.md](docs/SHADER_ALGEBRA.md) for the full design specification.

## Example Usage

The following example demonstrates how `pixelflow-core` uses Rust's type system to select the appropriate SIMD instructions:

```rust
use pixelflow_core::Batch;

fn main() {
    // 1. 32-bit addition (4 lanes)
    // Uses paddd (x86) or vaddq_u32 (ARM)
    let a = Batch::<u32>::splat(100);
    let b = Batch::<u32>::splat(50);
    let c = a + b;

    let mut result = [0u32; 4];
    unsafe { c.store(result.as_mut_ptr()) };
    println!("Result: {:?}", result); // [150, 150, 150, 150]

    // 2. 16-bit multiplication (8 lanes)
    // Uses pmullw (x86) or vmulq_u16 (ARM)
    let pixels = Batch::<u32>::new(0x01020304, 0x05060708, 0x090A0B0C, 0x0D0E0F10);
    let weights = Batch::<u32>::splat(256); // Fixed-point 1.0

    // Cast to u16 to exploit 16-bit SIMD multiply
    let pixels_16 = pixels.cast::<u16>();
    let weights_16 = weights.cast::<u16>();

    // The type system ensures the correct SIMD instruction is used
    let scaled = pixels_16 * weights_16;
}
```

## Architecture

*   **`batch`**: Core SIMD types (`Batch<T>`) and trait definitions.
*   **`ops`**: Implementations of fundamental operations (e.g., `Fill`, `Sample`).
*   **`pipe`**: Traits and structures for building composable rendering pipelines.
*   **`dsl`**: Helper traits for fluent pipeline construction.
