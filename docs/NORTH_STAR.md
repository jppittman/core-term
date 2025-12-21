# PixelFlow 1.0: North Star

**Status**: Draft v6  
**Audience**: Contributors, future-self, LLM agents

---

## Table of Contents

1. [The Thesis](#the-thesis) — SIMD as algebra
2. [The Inversion](#the-inversion) — Pull-based, laziness as contract
3. [The Problem](#the-problem) — What we're solving
4. [The Algebra](#the-algebra) — Field, Surface, Volume, Manifold, dimensional collapse
5. [The Fixed Observer](#the-fixed-observer) — Warp the world, not the camera; it is always now
6. [The Six Eigenshaders](#the-six-eigenshaders) — Warp, Grade, Lerp, Select, Fix, Compute
7. [Optimization is Composition](#optimization-is-composition) — Bounds are masks, clip is Select
8. [Type-Level Compilation](#type-level-compilation) — Types are shader programs
9. [Crate Architecture](#crate-architecture) — Core, Graphics, Engine
10. [What PixelFlow Is Not](#what-pixelflow-is-not) — Non-goals
11. [Performance Targets](#performance-targets) — The terminal benchmark
12. [Summary](#summary)

---

## The Thesis

SIMD is not an optimization. It is the algebraic realization of the Field of Real Numbers.

PixelFlow 1.0 resolves the false dichotomy between mathematical abstraction (slow, elegant) and hardware intrinsics (fast, brittle). Users compose pure functional manifolds over coordinate spaces. The compiler emits optimal, lane-agnostic vector assembly. The register width is a compiler secret.

**Write equations. Get assembly.**

---

## The Inversion

PixelFlow is **pull-based**. This is the root assumption from which everything else grows.

Every Surface is a function waiting to be queried. Nothing computes until coordinates arrive. There are no push pipelines, no invalidation signals, no identity-keyed caches. You do not "render" a Surface; you *sample* it.

**Laziness is not an optimization—it is the semantic contract.**

This inverts the gravity of traditional graphics. Most systems push pixels forward through a DAG: "here's geometry, transform it, rasterize it, composite it, write it." PixelFlow pulls values backward from the leaves: "I need the color at (x, y); what Surfaces must I query to get it?"

### Consequences

**Masks are just Surfaces.** No special primitives, no metadata channels. A mask is `Surface<bool>` or `Surface<Field>`. Infinite, composable, referentially transparent. Bounds are not structs—they're Surfaces that happen to be cheap.

**There is no pre-pass.** No tile culling phase, no spatial index, no "gather bounds then cull" step. Optimization happens inside `eval`, via composition. The loop is dumb; the Surface is smart.

**Baking is memoization, not mutation.** A baked Surface is backed by a buffer, but it's still a Surface—infinite domain, deterministic eval. The buffer is constructed (eagerly or lazily), then the Surface reads from it. Construction is when mutation happens; eval is pure.

Baking serves two purposes:
- *Collapse type depth*: prevent the compiler from exploding on deeply nested combinator trees
- *Materialize output*: the final rasterization writes to the framebuffer via an eager bake

Both produce a Surface. After baking, you have a function that happens to sample from memory.

**Early-exit is semantically invisible.** Mask-driven lane retirement, convergence checks, SIMD tricks—all within the semantics of evaluating a pure function. Users reason about Surfaces, not loops or lanes.

**Finite resources do not leak into the model.** SIMD width, tile padding, buffer extents—backend hygiene, not conceptual primitives. The meaning of a Surface is its value at every coordinate, not its organization in memory.

**Transparency is absolute.** The moment a combinator exposes stateful behavior, referential transparency fractures and the algebra collapses. State exists only in baked buffers, only as an implementation detail, never observable through the Surface interface.

---

## The Problem

Graphics programming forces developers to manage three concerns simultaneously:

1. **Topology**: Is this a 2D surface or 3D volume? Can I pass one where the other is expected?
2. **Vectorization**: How wide is a SIMD lane? How do I batch coordinates? What about the remainder?
3. **Materialization**: When do lazy descriptions become actual pixels? Who owns the buffer?

Existing solutions leak hardware topology into user code. Developers think in "lanes," "masks," and "batches" rather than mathematical functions. Dimension mismatches cause type errors or silent bugs. The abstraction gap between `f(x,y) = ...` and `_mm256_fmadd_ps` is vast.

PixelFlow closes this gap.

---

## The Algebra

### The Atom: Field

The computational substrate is `Field`—a type satisfying the field axioms (associativity, commutativity, distributivity, additive/multiplicative identity and inverse).

```rust
// The user sees this:
use pixelflow::prelude::*;

fn twist(x: Field, y: Field, z: Field) -> (Field, Field, Field) {
    let theta = z * 0.1;  // scalar promotion is implicit
    let (s, c) = (theta.sin(), theta.cos());
    (x * c - y * s, x * s + y * c, z)
}
```

Under the hood, `Field` wraps a SIMD vector. The width—8 lanes (AVX2), 16 lanes (AVX-512), 4 lanes (NEON)—is invisible. Scalars promote implicitly; `z * 0.1` broadcasts `0.1` across all lanes without user intervention.

**Why "Field" and not "Real"?** The abstraction is algebraic, not numeric. A `Field` could carry complex components, dual numbers for automatic differentiation, or interval arithmetic for verified computation. We're not the cops.

### Surfaces and Volumes

A **Surface** is a function `(x, y) → T`. A **Volume** is a function `(x, y, z) → T`. A **Manifold** is a function `(x, y, z, w) → T`.

```rust
pub trait Surface<T>: Send + Sync {
    fn eval(&self, x: Field, y: Field) -> T;
}

pub trait Volume<T>: Send + Sync {
    fn eval(&self, x: Field, y: Field, z: Field) -> T;
}

pub trait Manifold<T>: Send + Sync {
    fn eval(&self, x: Field, y: Field, z: Field, w: Field) -> T;
}
```

These are the spatial primitives. Curves are 1D slices of Surfaces. Higher dimensions are future work if needed, but 4D covers animation and simulation.

### The Dimensional Collapse

**All Manifolds are Volumes. All Volumes are Surfaces.** This is a compiler truth, not a convention:

```rust
impl<T, M: Manifold<T>> Volume<T> for M {
    fn eval(&self, x: Field, y: Field, z: Field) -> T {
        self.eval_raw(x, y, z, Field::ZERO)  // bind w = 0
    }
}

impl<T, V: Volume<T>> Surface<T> for V {
    fn eval(&self, x: Field, y: Field) -> T {
        self.eval_raw(x, y, Field::ZERO)  // bind z = 0
    }
}
```

A renderer expecting `impl Surface` accepts any Volume or Manifold unchanged. The higher-dimensional object is sampled at its origin cross-section. No adapter, no conversion, no runtime cost.

**The inverse requires explicit construction.** A Surface lacks z-information; a Volume lacks w-information. To promote:

```rust
pub struct Extrude<S>(pub S);

impl<T, S: Surface<T>> Volume<T> for Extrude<S> {
    fn eval(&self, x: Field, y: Field, _z: Field) -> T {
        self.0.eval_raw(x, y)  // z-invariant: infinite prism
    }
}

impl<T, V: Volume<T>> Manifold<T> for Extrude<V> {
    fn eval(&self, x: Field, y: Field, z: Field, _w: Field) -> T {
        self.0.eval_raw(x, y, z)  // w-invariant: infinite hyperprism
    }
}
```

This asymmetry is the "recursive topology": R⁴ → R³ → R² → R¹, each dimension implicitly satisfying the contracts below it. Scalars are 0-dimensional Surfaces (ignore all coordinates, return constant). **All constants are Manifolds.**

---

## The Fixed Observer

Traditional graphics transforms geometry to align with a moving camera. PixelFlow inverts this.

**The Observer is the origin. The Observer does not move. It is always now.**

This isn't a technical trick—it's modeling how you actually experience reality. You've never been anywhere else. You've never experienced another moment. "There" and "then" are stories you tell from here, now. The world comes to you; you don't go to it.

PixelFlow takes this seriously:

- Surfaces are evaluated at `(x, y)` — here
- Volumes are evaluated at `(x, y, z)` — here, at this depth
- Manifolds are evaluated at `(x, y, z, w)` — here, at this depth, now
- "Camera movement" is the world rearranging itself around you
- "Time passing" is the timeline sliding through the present

```rust
// To view a volume at z = k:
let shifted = volume.warp(|x, y, z| (x, y, z - k));
// You stay at z = 0. The world moves to meet you.

// To view a manifold at time t:
let at_time_t = manifold.warp(|x, y, z, w| (x, y, z, w - t));
// It is always now. The past moves to meet the present.
```

The dimensional collapse binds higher dimensions to zero: Volume→Surface binds z=0, Manifold→Volume binds w=0. This isn't arbitrary—**zero is where you are, in every dimension**. Here. Now. Always.

To animate, you don't "play" a Manifold. The Manifold already contains all of time—past, present, future coexist. You continuously warp it so that "now" aligns with your clock:

```rust
fn render_frame(manifold: impl Manifold<Color>, clock: f32) {
    let now = manifold.warp(|x, y, z, w| (x, y, z, w - clock));
    materialize(now);  // sample at w=0, which is clock-time
}
```

The Manifold doesn't change. It *is*. You just keep asking "what's here, now?" as the world flows through you.

This simplifies composition. Everything shares the same coordinate system—yours. No matrix stack, no coordinate confusion, no frame counters. Just: here, now, what do you see?

---

## The Six Eigenshaders

Any shader is expressible as composition of six orthogonal primitives:

| Combinator | Signature | Purpose |
|------------|-----------|---------|
| **Warp** | `(S, ω) → S` | Remap coordinates before sampling |
| **Grade** | `(S, M, b) → S` | Linear transform on values (matrix + bias) |
| **Lerp** | `(t, a, b) → S` | Continuous interpolation: `a + t*(b-a)` |
| **Select** | `(cond, t, f) → S` | Branchless conditional (discrete) |
| **Fix** | `(seed, step) → V` | Iteration as a dimension (see below) |
| **Compute** | `Fn(x,y) → T` | Escape hatch (any closure is a Surface) |

Warp, Grade, Lerp, Select, Compute work in any dimension. Fix is special: it *constructs* a Volume.

### Fix: Iteration is a Dimension

Iteration count isn't a loop—it's a coordinate. Fix constructs a Volume where z is the iteration axis:

```rust
let mandelbrot = Fix::new(
    Complex::ZERO,                    // seed: value at z=0
    |state, x, y| {                   // recurrence: z → z+1
        let c = Complex::new(x, y);
        state * state + c
    }
);

// mandelbrot.eval_raw(x, y, z=64) → 64 iterations from seed
// mandelbrot.eval_raw(x, y, z=0)  → seed (via dimensional collapse)
```

This unifies:
- **Mandelbrot/Julia**: iteration depth (z)
- **Raymarching**: step count along ray (z)
- **Simulation**: time axis (w) — each w-slice is a frame
- **Animation**: time axis (w) — interpolate or step through states

The recurrence relation defines adjacency along the iteration axis. You never write a loop; you sample the Volume (or Manifold) at the coordinate you want.

For time-based animation, Fix can construct a Manifold where w is time:

```rust
let physics_sim = Fix::new(
    initial_state,                    // state at w=0
    |state, x, y, z| step(state),     // w → w+1 transition
);

// physics_sim.eval_raw(x, y, z, w=60) → state after 60 frames
```

**Completeness**: Compute remains the escape hatch for truly opaque computations. But now iteration is structural—analyzable, serializable, differentiable.

### Lerp vs Select

Select is discrete (boolean condition, no blending). Lerp is continuous (Field-valued interpolation). This matters for antialiasing:

```rust
// Hard edge (Select)
let hard = sdf.is_negative().select(inside, outside);

// Soft edge (Lerp) — 1px AA band
let t = (sdf + 0.5).clamp(0.0, 1.0);
let soft = t.lerp(outside, inside);
```

### What about Over?

Porter-Duff compositing (`Over`) lives in **pixelflow-graphics**, not core. It's Lerp plus color semantics: premultiplied alpha, channel clamping, quantization to u8. Core stays pure Field algebra.

---

## Optimization is Composition

There is no pre-pass. There is no tile culling. There is no metadata.

The runtime is blind. It iterates pixels. It calls `eval`. Optimization happens *inside* the Surface, through composition.

### Bounds Are Masks

There is no `AABB` struct used for culling. "Bounds" are Surfaces—specifically, Masks—that are:

- **Cheap**: arithmetic only, no heavy trig or texture lookups
- **Conservative**: strictly contain the interesting region
- **Evaluated first**: placed at the root of the composition

A Mask is `Surface<bool>` (or `Surface<Field>` returning 0.0/1.0). Primitives like `Rect` and `Circle` *are* Masks. They don't have bounds; they *are* bounds.

### The Clip Combinator

To optimize an expensive Surface (fractal, noise, raymarching), compose it with a cheap Mask:

```rust
impl<T, S: Surface<T>> SurfaceExt<T> for S {
    fn clip<M: Surface<bool>>(self, mask: M) -> Select<M, Self, Empty<T>> {
        Select {
            condition: mask,      // evaluated FIRST
            if_true: self,        // evaluated only if mask passes
            if_false: Empty,      // immediate return (zero-cost)
        }
    }
}

// Usage: expensive surface clipped to cheap bounds
let optimized = perlin_noise.clip(Rect::new(0, 0, 100, 100));
```

`clip` is just `Select` with `Empty` as the false branch. There's no special optimization machinery—composition *is* the optimization.

### Select: The Hardware Truth

Select is where early-exit happens. The implementation exploits SIMD lane masks:

```rust
impl<C, T, F, V> Surface<V> for Select<C, T, F>
where
    C: Surface<bool>,
    T: Surface<V>,
    F: Surface<V>,
{
    fn eval(&self, x: Field, y: Field) -> V {
        let mask = self.condition.eval_raw(x, y);
        
        // All lanes false → skip true branch entirely
        if mask.none() { 
            return self.if_false.eval_raw(x, y); 
        }
        
        // All lanes true → skip false branch entirely
        if mask.all() { 
            return self.if_true.eval_raw(x, y); 
        }
        
        // Straddle: some lanes true, some false
        // Must evaluate both and blend
        blend(
            mask,
            self.if_true.eval_raw(x, y),
            self.if_false.eval_raw(x, y)
        )
    }
}
```

The `none()` and `all()` checks are single instructions on SIMD hardware. This is the *only* place culling happens—inside the kernel, not before it.

### Implicit vs Explicit Bounds

**Primitives are self-bounding.** `Circle::eval` naturally returns 0.0 outside the radius. There's nothing to clip; the math *is* the bound.

**Complex surfaces need explicit clipping.** Perlin noise, Voronoi, fractals—these are infinite. The user imposes bounds via composition:

```rust
// Infinite noise, clipped to a region
let bounded_noise = Perlin::new().clip(Rect::new(0, 0, 256, 256));

// Expensive fractal, clipped to where we'll actually sample
let bounded_fractal = mandelbrot.clip(viewport);
```

The engine does not auto-detect bounds. It cannot—infinite surfaces have no intrinsic bounds. The user decides where to stop, and expresses that decision as a Mask.

### The Loop is Dumb; The Surface is Smart

The bake/materialize loop knows nothing:

```rust
for y in 0..height {
    for x in (0..width).step_by(LANES) {
        let coords = (Field::sequential_from(x), Field::splat(y));
        let value = surface.eval_raw(coords.0, coords.1);
        value.store(&mut buffer[y * width + x..]);
    }
}
```

All intelligence—culling, early-exit, blending—lives in the Surface's `eval`. The loop just asks questions; the Surface decides how hard to work.

---

## Type-Level Compilation

The combinator types form a compile-time shader graph:

```rust
let scene = circle
    .extrude()           // Surface → Volume
    .warp(twist)         // Volume → Volume  
    .grade(sepia)        // Volume → Volume
    .lerp(fg, bg);       // Volume → Volume (blend by coverage)

// Concrete type (elided for sanity):
// Lerp<Grade<Warp<Extrude<Circle>, TwistFn>, SepiaMatrix>, Fg, Bg>
```

Rust monomorphizes this into a single fused kernel. LLVM sees through `#[inline(always)]` on every `eval`. No vtables, no heap allocation, no runtime dispatch inside the hot loop.

**The type is the shader program. The struct fields are the uniforms.**

---

## Crate Architecture

```
pixelflow-core        Pure algebra. Field, Surface, Volume, Manifold.
      ↓               Warp, Grade, Lerp, Select, Fix, Compute.
      ↓               No IO, no platform, no pixels, no colors.
      ↓
pixelflow-graphics    Fonts, colors, compositing, materialization.
      ↓               Over = Lerp + color semantics.
      ↓               (absorbs pixelflow-fonts, pixelflow-render)
      ↓
pixelflow-engine      The engine. Scene graph, physics, animation,
      ↓               input handling, the render loop.
      ↓
  application         Terminal, game, visualization.
```

**Core** exports `Field`, `Surface`, `Volume`, `Manifold`, the six eigenshaders, and nothing else. It is useful standalone for compute workloads (physics simulation, image processing, ML inference on continuous fields). No color types, no pixel formats.

**Graphics** handles the transition from algebra to pixels: color spaces, Porter-Duff compositing (`Over`), font loading, glyph rasterization, framebuffer management, platform display. This is where `materialize()` lives.

**Engine** is the engine. It coordinates scene composition, animation, physics, input handling, and the render loop. It's where applications live.

---

## What PixelFlow Is Not

- **Not a rasterization pipeline.** No triangles, no vertices, no index buffers. Geometry is implicit in the coordinate functions.
- **Not a GPU abstraction.** This is CPU SIMD. GPU backends are possible future work, but the algebra doesn't require them.
- **Not a text layout engine.** Fonts produce Surfaces. Shaping, bidirectional text (Arabic/Hebrew reordering), and line breaking are out of scope—use a layout library, feed us glyphs.

---

## Performance Targets

Measured on the terminal emulator (1920×1080, 155 FPS sustained):

| Metric | Target | Achieved |
|--------|--------|----------|
| Per-pixel time | < 10ns | ~5ns |
| Frame time | < 8ms | ~6.5ms |
| Memory (idle) | < 100MB | ~70MB |

The terminal is the validation workload. If PixelFlow can render a 1080p terminal at 155 FPS with font rendering, cursor blinking, selection highlighting, and scrollback—purely on CPU—the abstraction is not just elegant but *fast enough*.

---

## Summary

1. **Pull-based**: nothing computes until coordinates arrive; laziness is the contract
2. **Field** is the SIMD-width-agnostic computational atom
3. **Manifold ⊂ Volume ⊂ Surface** via blanket impl (dimensional collapse)
4. **Extrude** is invariant promotion; **Fix** is iterative promotion
5. **The Observer is fixed; it is always now**; movement is domain warping
6. **Six eigenshaders**: Warp, Grade, Lerp, Select, Fix, Compute
7. **Bounds are Masks**; optimization is composition via Select
8. **Baking is memoization**: buffers are Surfaces that read from memory
9. **Types compile to fused kernels**; no runtime dispatch
10. **Core is pure algebra**; Graphics adds colors; Engine runs the show

**All scalars are Manifolds. All geometry is function. Iteration is a dimension. Time is a dimension. It is always now. Eval is the only verb.**
