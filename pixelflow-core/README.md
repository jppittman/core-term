# PixelFlow Core

**Pure Algebra for SIMD Graphics**

Write math. Get SIMD. No compromise.

```rust
use pixelflow_core::{Manifold, X, Y};

// Circle signed distance field
let circle = (X * X + Y * Y).sqrt() - 100.0;

// Evaluate at any coordinate
let distance = circle.eval_raw(x, y, 0.0, 0.0);
```

## Philosophy

Inspired by [Conal Elliott's denotational design](http://conal.net/papers/icfp97/) and [Halide](https://halide-lang.org/), `pixelflow-core` treats SIMD not as an optimization but as the **algebraic realization of continuous fields**.

**Everything is a manifold.** You compose functions, not values. No intermediate results. No pre-computation. Just expressions that fuse into optimal vector assembly.

The idiomatic style:
- ✅ **Write composable manifold expressions**: `X * scale + offset`, `circle.select(inner, outer)`
- ✅ **Use combinators**: `At`, `Select`, `Fix` for control flow
- ✅ **Stay generic**: Manifolds work with `Field`, `Jet2`, `Jet3` automatically
- ❌ **Avoid pre-computed values**: Don't materialize to `Field` then recombine

## Core Concepts

### Field

The computational atom—a SIMD vector of `f32` values. All operations work lane-wise.

```rust
// Fields are created implicitly when you evaluate manifolds
let result: Field = manifold.eval_raw(x, y, z, w);
```

### Manifold

The one trait. Everything is a function over 4D coordinates, generic over the numeric type:

```rust
pub trait Manifold<I: Numeric = Field>: Send + Sync {
    type Output;
    fn eval_raw(&self, x: I, y: I, z: I, w: I) -> Self::Output;
}
```

The `I` type parameter enables **automatic differentiation**: evaluate with `Field` for
concrete values, or `Jet2` to compute gradients automatically.

### Variables

Built-in coordinate accessors:

```rust
use pixelflow_core::{X, Y, Z, W};

// X just returns the x coordinate
// Y just returns the y coordinate
// etc.
```

### Operator Overloads

Write math naturally. Operators build an AST that evaluates to SIMD:

```rust
// All these "just work"
let sum = X + Y;
let product = X * 2.0;
let complex = (X + 2.0) * Y - 0.5;
let ratio = X / (Y + 1.0);
```

### Combinators: Composition Over Values

The idiomatic PixelFlow pattern is to **compose manifolds as manifolds**, not materialize to values.

**The Right Way**: Use combinators

```rust
use pixelflow_core::{Manifold, ManifoldExt, Select, At, X, Y};

// Blend two manifolds based on a condition
let sdf = (X * X + Y * Y).sqrt() - 10.0;
let condition = sdf.lt(0.0);
let result = condition.select(inner_color, outer_color);  // Both stay as manifolds!
```

**The Wrong Way**: Pre-compute to values

```rust
// ❌ Don't do this - loses compositional structure
let sdf_value = sdf.eval_raw(x, y, z, w);        // Materialized!
let condition_value = sdf_value < 0.0;           // Materialized!
let result = if condition_value { inner } else { outer };  // Lost genericity!
```

#### Select: Branchless Conditional

Choose between two manifolds without materializing intermediate values:

```rust
let mask = X.lt(50.0);
let result = mask.select(white, black);  // Fuses into single kernel
```

#### At: Pin Manifolds to Computed Coordinates

Evaluate manifolds at different coordinate systems, then blend:

```rust
use pixelflow_core::{At, Jet3};

// material_at_hit and background_at_ray are independent evaluations
// No pre-computed hit points or ray directions
let material_at_hit = At {
    inner: &material,
    x: hit_x,  // Jet3 constants are manifolds too
    y: hit_y,
    z: hit_z,
    w: Jet3::constant(0.0),
};

let background_at_ray = At {
    inner: &background,
    x: ray_x,
    y: ray_y,
    z: ray_z,
    w,
};

// Compose: all parts stay as manifolds
let scene = hit_mask.select(material_at_hit, background_at_ray);
```

The key insight: **Jet3 values themselves implement `Manifold`**, so they're not "pre-computed"—they're constant manifold expressions that integrate into the computation graph.

## Examples

### Circle SDF

```rust
use pixelflow_core::{Manifold, ManifoldExt, X, Y};

// Signed distance to a circle at origin, radius 100
let circle = (X * X + Y * Y).sqrt() - 100.0;

// Inside: negative. Outside: positive.
```

### Centered Circle

```rust
use pixelflow_core::{Manifold, ManifoldExt, X, Y};

// Circle centered at (50, 50)
let cx = 50.0;
let cy = 50.0;
let radius = 30.0;

let dx = X - cx;
let dy = Y - cy;
let circle = (dx * dx + dy * dy).sqrt() - radius;
```

### Smooth Gradient

```rust
use pixelflow_core::{Manifold, ManifoldExt, X, Y};

// Horizontal gradient from 0 to 1 across 100 pixels
let gradient = X / 100.0;

// Clamp to [0, 1]
let clamped = gradient.max(0.0).min(1.0);
```

### Checkerboard

```rust
use pixelflow_core::{Manifold, ManifoldExt, X, Y};

// 10x10 pixel checkerboard
let cell_size = 10.0;

// XOR of x and y cell parity
let x_cell = (X / cell_size).floor();
let y_cell = (Y / cell_size).floor();

// Use comparison and select for the pattern
let x_odd = (x_cell % 2.0).ge(1.0);
let y_odd = (y_cell % 2.0).ge(1.0);

// XOR via: (x && !y) || (!x && y)
// Or simpler: different parity = white
let checker = x_odd.select(
    y_odd.select(0.0, 1.0),  // x odd: white if y even
    y_odd.select(1.0, 0.0),  // x even: white if y odd
);
```

### Mandelbrot (using Fix)

```rust
use pixelflow_core::{Manifold, ManifoldExt, Fix, X, Y, W};

// W is the iteration state (escape radius squared)
// Simplified: just check if |z|² > 4 after N iterations

let mandelbrot = Fix {
    seed: 0.0,                        // z starts at 0
    step: W * W + X,                  // z = z² + c (simplified to 1D)
    done: W.abs().gt(2.0),            // escape when |z| > 2
};

// Evaluate: returns the final z value (or escape value)
```

## Ergonomics

### Seamless Scalar Promotion

Scalars auto-promote to `Field`:

```rust
// No splat() needed - 2.0 and 0.5 just work
let expr = X * 2.0 + 0.5;
```

### Select with Scalars

The `select` method accepts scalars directly:

```rust
let mask = X.lt(100.0);
let result = mask.select(1.0, 0.0);  // No Field::splat needed
```

### ManifoldExt Methods

Chaining methods for fluent APIs:

```rust
use pixelflow_core::ManifoldExt;

let result = X
    .add(10.0)        // X + 10
    .mul(Y)           // (X + 10) * Y
    .sqrt()           // sqrt((X + 10) * Y)
    .max(0.0)         // clamp to non-negative
    .min(1.0);        // clamp to max 1
```

### Materializing to Pixels

Render a manifold to a byte buffer:

```rust
use pixelflow_core::materialize;

let mut buffer = [0u8; WIDTH];
materialize(&manifold, x_start, y, &mut buffer);
// buffer now contains u8 values (0-255) from the manifold
```

## Automatic Differentiation with Jet2

All expressions built with `ManifoldExt` are generic over the `Numeric` type. This means
the same expression can be evaluated with `Field` (for values) or `Jet2` (for gradients):

```rust
use pixelflow_core::{ManifoldExt, X, Y, Jet2, Manifold, Numeric};

// Build expression using ManifoldExt
let circle = (X * X + Y * Y).sqrt() - 10.0;

// Evaluate with Field (concrete values)
let distance = circle.eval(3.0, 4.0, 0.0, 0.0);  // Returns 5.0 - 10.0 = -5.0

// Evaluate with Jet2 (automatic differentiation)
let x_jet = Jet2::x(3.0.into());  // x = 3, ∂x/∂x = 1, ∂x/∂y = 0
let y_jet = Jet2::y(4.0.into());  // y = 4, ∂y/∂x = 0, ∂y/∂y = 1
let zero = Jet2::constant(0.0.into());

let result = circle.eval_raw(x_jet, y_jet, zero, zero);
// result.val = -5.0 (the distance value)
// result.dx = 0.6   (∂distance/∂x = x/√(x²+y²) = 3/5)
// result.dy = 0.8   (∂distance/∂y = y/√(x²+y²) = 4/5)
```

This is useful for computing normals, gradients for optimization, and ray marching.

## Architecture

Under the hood:
- `Field` = `SimdVec<f32>` (4 lanes on SSE2, 16 on AVX-512)
- Expressions build an AST of `Add`, `Mul`, `Sqrt`, `Select`, etc.
- Evaluation inlines to tight SIMD loops
- Zero runtime dispatch—the compiler monomorphizes everything
- Generic `Numeric` trait enables both `Field` and `Jet2` evaluation

## License

MIT
