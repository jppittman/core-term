# PixelFlow Core

**PixelFlow Core** is the algebraic heart of the PixelFlow ecosystem. It provides a **Zero-Copy Functional** abstraction for defining continuous fields over coordinate spaces.

## The Thesis

SIMD is not an optimization. It is the algebraic realization of the Field of Real Numbers. `pixelflow-core` allows you to write equations that compile directly to optimal, lane-agnostic vector assembly.

## Core Primitives

### `Field`
The computational atom. A `Field` represents a value that exists in parallel across multiple SIMD lanes. It satisfies field axioms and supports standard math operations (`+`, `*`, `sin`, `cos`, etc.).
*(Note: Currently implemented via `Batch<T>` types).*

### `Surface<T>`
A function `(x: Field, y: Field) -> T`.
This is the fundamental 2D primitive. Surfaces are infinite and continuous.

### `Volume<T>` and `Manifold<T>`
Higher-dimensional generalizations:
*   `Volume`: `(x, y, z) -> T`
*   `Manifold`: `(x, y, z, w) -> T`

### Dimensional Collapse
Higher dimensions automatically satisfy lower-dimensional traits by binding extra coordinates to zero.
*   `Manifold` implies `Volume` (w=0).
*   `Volume` implies `Surface` (z=0).

## The Six Eigenshaders

All complex behavior is built from six orthogonal combinators:

1.  **Warp**: `(S, ω) -> S` — Remap coordinates (move/distort space).
2.  **Grade**: `(S, M, b) -> S` — Linear transform on values (color correction).
3.  **Lerp**: `(t, a, b) -> S` — Linear interpolation (blending).
4.  **Select**: `(cond, t, f) -> S` — Discrete choice (masking/clipping).
5.  **Fix**: `(seed, step) -> V` — Iteration as a dimension (fractals, simulation).
6.  **Compute**: `Fn(x,y) -> T` — The escape hatch.

## Usage

```rust
use pixelflow_core::prelude::*;

// Define a surface
let circle = Circle::new(100.0);

// Warp it (move space, not the object)
let moved = circle.warp(|x, y| (x - 50.0, y - 50.0));

// Evaluate it (get a value at a coordinate)
let value = moved.eval(Field::splat(10.0), Field::splat(10.0));
```

## Implementation

Under the hood, `pixelflow-core` uses generic `Batch<T>` types to abstract over SIMD widths (Scalar, SSE2, AVX2, AVX-512). The compiler monomorphizes your Surface combinator tree into a single, efficient kernel.
