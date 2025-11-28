# PixelFlow Shader Algebra

**Version:** 2.0
**Status:** Proposed
**Author:** JP
**Date:** November 2025

## Executive Summary

This document describes a redesign of PixelFlow’s rendering abstraction around a **functional combinator pattern** we call the “Shader Algebra.” The core insight is that any pixel shader can be decomposed into compositions of three orthogonal operations—geometry transforms, color transforms, and signal modulation—plus branchless control flow. This yields a system that is simultaneously more expressive, more composable, and zero-cost.

## Table of Contents

1. [Motivation](#1-motivation)
1. [Design Philosophy](#2-design-philosophy)
1. [Core Abstraction: The Surface Trait](#3-core-abstraction-the-surface-trait)
1. [Primitive Surfaces](#4-primitive-surfaces)
1. [The Three Eigenshaders](#5-the-three-eigenshaders)
1. [Control Flow: Select](#6-control-flow-select)
1. [Escape Hatch: FnSurface](#7-escape-hatch-fnsurface)
1. [Automatic Format Elision](#8-automatic-format-elision)
1. [Automatic Grade Composition](#9-automatic-grade-composition)
1. [Extension Traits and Ergonomics](#10-extension-traits-and-ergonomics)
1. [Alternatives Considered](#11-alternatives-considered)
1. [Performance Characteristics](#12-performance-characteristics)
1. [Differential Surfaces](#13-differential-surfaces-trait-now-impls-tomorrow)
1. [Future Work](#14-future-work)

-----

## 1. Motivation

### The Problem

The previous API exposed `Batch<T>` directly to users:

```rust
trait Surface<T: Copy>: Copy {
    fn eval(&self, x: Batch<u32>, y: Batch<u32>) -> Batch<T>;
}
```

This had several problems:

1. **Leaky Abstraction**: Users needed to understand SIMD batching to write custom shaders. The mental model of “4 pixels at once” leaked into the API.
1. **Coordinate Rigidity**: Coordinates were concrete `Batch<u32>` values, not composable surfaces themselves. This made coordinate transforms (rotation, warping) feel bolted-on rather than first-class.
1. **Format Coupling**: Every stage operated on the same packed `u32` RGBA format. Color-manipulating stages (brightness, saturation) had to unpack, process, and repack—even when chained together.
1. **Limited Composability**: Without a unified abstraction for “things that vary over space,” users couldn’t easily express concepts like “warp coordinates by a noise function.”

### The Goal

A shader system where:

- Users think in terms of **pixels and coordinates**, not SIMD widths
- **Coordinates are themselves surfaces**, enabling arbitrary warping
- **Format conversions are automatic** and elided when unnecessary
- The **full power of SIMD is preserved** under the hood
- **Any shader is expressible** through composition of primitives

-----

## 2. Design Philosophy

### Functional Combinators

The system follows the **functional combinator pattern**, familiar from Rust’s iterator adapters. Each shader is a struct that owns its inputs. Composition creates nested structs. At sample time, the entire graph is inlined and optimized by LLVM.

```rust
// Iterator pattern
let result = vec.iter().map(f).filter(g).collect();

// Surface pattern
let result = texture.warp(distortion).grade(sepia).mask(scanlines);
```

### Lazy Evaluation

Surfaces are **lazy**. Constructing a pipeline allocates no memory and performs no computation. Work happens only when `sample()` is called with concrete coordinates. This enables:

- Zero-cost abstraction (unused branches are eliminated)
- Compile-time graph fusion
- No intermediate buffers between stages

### Algebraic Closure

The system is **algebraically closed**: composing surfaces yields surfaces. This means:

- Any surface can be used anywhere a surface is expected
- Complex effects are built from simple primitives
- The type system enforces correctness

-----

## 3. Core Abstraction: The Surface Trait

### Definition

```rust
/// A pure functional surface: coordinates → values.
///
/// This is a "Lazy Array"—it doesn't own memory; it describes how to
/// compute a value at any coordinate. Surfaces are the universal
/// abstraction for anything that varies over 2D space.
pub trait Surface: Copy + Send + Sync {
    /// The output type of this surface.
    type Output: Copy;

    /// Sample the surface at the given coordinates.
    ///
    /// # Parameters
    /// * `u` - Horizontal coordinates (batch of floats)
    /// * `v` - Vertical coordinates (batch of floats)
    ///
    /// # Returns
    /// A batch of output values.
    fn sample(&self, u: Batch<f32>, v: Batch<f32>) -> Self::Output;
}
```

### Key Design Decisions

**Associated Type for Output**: Rather than a generic parameter `Surface<T>`, we use an associated type `Surface::Output`. This:

- Simplifies type inference at use sites
- Makes each surface’s output type unambiguous
- Enables format elision (see §8)

**Float Coordinates**: Coordinates are `Batch<f32>`, not `Batch<u32>`. This:

- Enables sub-pixel precision for anti-aliasing
- Makes coordinate math (rotation, scaling) natural
- Matches GPU shader conventions

**Copy + Send + Sync Bounds**: Surfaces must be trivially copyable and thread-safe. This:

- Enables parallel rendering (stripe the framebuffer across cores)
- Ensures surfaces can be passed by value into closures
- Prevents accidental shared mutable state

-----

## 4. Primitive Surfaces

### Identity Coordinates

The fundamental building blocks are surfaces that return the input coordinates unchanged:

```rust
/// The X coordinate as a surface.
#[derive(Copy, Clone)]
pub struct X;

impl Surface for X {
    type Output = Batch<f32>;

    #[inline(always)]
    fn sample(&self, u: Batch<f32>, _v: Batch<f32>) -> Batch<f32> {
        u
    }
}

/// The Y coordinate as a surface.
#[derive(Copy, Clone)]
pub struct Y;

impl Surface for Y {
    type Output = Batch<f32>;

    #[inline(always)]
    fn sample(&self, _u: Batch<f32>, v: Batch<f32>) -> Batch<f32> {
        v
    }
}
```

These enable coordinate transforms to be expressed as surface composition:

```rust
// Translation: sample at (x + 10, y + 20)
let translated = Warp { source: texture, x: X + 10.0, y: Y + 20.0 };

// Rotation: sample at (x*cos - y*sin, x*sin + y*cos)
let rotated = Warp {
    source: texture,
    x: X * cos_theta - Y * sin_theta,
    y: X * sin_theta + Y * cos_theta,
};

// Arbitrary distortion: sample at (x + noise(x,y), y)
let distorted = Warp { source: texture, x: X + noise, y: Y };
```

### Constants

Any `Batch<T>` is itself a surface (ignoring coordinates):

```rust
impl<T: Copy> Surface for Batch<T> {
    type Output = Batch<T>;

    #[inline(always)]
    fn sample(&self, _u: Batch<f32>, _v: Batch<f32>) -> Batch<T> {
        *self
    }
}
```

This unifies constants and varying values under the same abstraction:

```rust
// These are both valid "color" arguments to blend:
let solid = Batch::splat(0xFF0000FF);  // Constant red
let gradient = /* computed surface */; // Varying color
```

### Textures

```rust
/// A source surface that samples from a texture atlas.
#[derive(Copy, Clone)]
pub struct Texture<'a> {
    pub data: &'a [u32],
    pub width: usize,
    pub height: usize,
}

impl<'a> Surface for Texture<'a> {
    type Output = Batch<u32>;

    #[inline(always)]
    fn sample(&self, u: Batch<f32>, v: Batch<f32>) -> Batch<u32> {
        // Bilinear interpolation with bounds checking
        // ... implementation details ...
    }
}
```

-----

## 5. The Three Eigenshaders

The central claim of this design is that any pixel shader can be decomposed into compositions of three orthogonal operations. We call these the **eigenshaders** (by analogy to eigenvectors—they are the “basis” of the shader space).

### 5.1 Warp: Geometry Transform

**Purpose**: Manipulate where pixels are sampled from, without touching their values.

```rust
/// A coordinate remapping transform.
///
/// This is the universal geometry primitive. Translation, rotation,
/// scaling, skewing, and arbitrary distortion are all expressible
/// as Warp with appropriate coordinate surfaces.
#[derive(Copy, Clone)]
pub struct Warp<S, X, Y> {
    /// The source surface to sample.
    pub source: S,
    /// Surface defining the new U coordinate.
    pub map_u: X,
    /// Surface defining the new V coordinate.
    pub map_v: Y,
}

impl<S, X, Y> Surface for Warp<S, X, Y>
where
    S: Surface,
    X: Surface<Output = Batch<f32>>,
    Y: Surface<Output = Batch<f32>>,
{
    type Output = S::Output;

    #[inline(always)]
    fn sample(&self, u: Batch<f32>, v: Batch<f32>) -> Self::Output {
        let new_u = self.map_u.sample(u, v);
        let new_v = self.map_v.sample(u, v);
        self.source.sample(new_u, new_v)
    }
}
```

**Use Cases**:

|Effect         |Implementation                                       |
|---------------|-----------------------------------------------------|
|Translation    |`Warp { source, x: X + dx, y: Y + dy }`              |
|Scaling        |`Warp { source, x: X * sx, y: Y * sy }`              |
|Rotation       |`Warp { source, x: X*cos - Y*sin, y: X*sin + Y*cos }`|
|Skew/Shear     |`Warp { source, x: X - Y * shear, y: Y }`            |
|Wave/Ripple    |`Warp { source, x: X + sin(Y * freq) * amp, y: Y }`  |
|Lens Distortion|`Warp { source, x: X * radial_fn, y: Y * radial_fn }`|

**Why Separate from Color?** Geometry transforms operate on coordinates, not pixel values. Keeping them separate means:

- Clear mental model (“where” vs “what color”)
- Type safety (coordinate surfaces vs color surfaces)
- Optimization opportunities (coordinate math uses different SIMD ops than color math)

### 5.2 Grade: Color Transform

**Purpose**: Apply linear transforms to color values via a 4×4 matrix plus bias.

```rust
/// A color grading transform using a 4×4 matrix.
///
/// This is the universal color primitive. Brightness, contrast,
/// saturation, hue rotation, sepia, grayscale, and color inversion
/// are all expressible as Grade with appropriate matrices.
#[derive(Copy, Clone)]
pub struct Grade<S> {
    /// The source surface providing colors.
    pub source: S,
    /// The 4×4 transformation matrix.
    pub matrix: Mat4,
    /// The bias vector (added after matrix multiply).
    pub bias: Vec4,
}

impl<S: Surface<Output = PlanarRGBA>> Surface for Grade<S> {
    type Output = PlanarRGBA;

    #[inline(always)]
    fn sample(&self, u: Batch<f32>, v: Batch<f32>) -> PlanarRGBA {
        let color = self.source.sample(u, v);

        // Matrix multiply: [r', g', b', a'] = M × [r, g, b, a] + bias
        let r = color.r * self.matrix.m00 + color.g * self.matrix.m01
              + color.b * self.matrix.m02 + color.a * self.matrix.m03
              + self.bias.x;
        let g = color.r * self.matrix.m10 + color.g * self.matrix.m11
              + color.b * self.matrix.m12 + color.a * self.matrix.m13
              + self.bias.y;
        let b = color.r * self.matrix.m20 + color.g * self.matrix.m21
              + color.b * self.matrix.m22 + color.a * self.matrix.m23
              + self.bias.z;
        let a = color.r * self.matrix.m30 + color.g * self.matrix.m31
              + color.b * self.matrix.m32 + color.a * self.matrix.m33
              + self.bias.w;

        PlanarRGBA { r, g, b, a }
    }
}
```

**Common Matrices**:

```rust
impl Mat4 {
    /// Identity (no change).
    pub const IDENTITY: Self = /* ... */;

    /// Grayscale using perceptual luminance weights.
    pub const GRAYSCALE: Self = Mat4 {
        m00: 0.299, m01: 0.587, m02: 0.114, m03: 0.0,
        m10: 0.299, m11: 0.587, m12: 0.114, m13: 0.0,
        m20: 0.299, m21: 0.587, m22: 0.114, m23: 0.0,
        m30: 0.0,   m31: 0.0,   m32: 0.0,   m33: 1.0,
    };

    /// Sepia tone.
    pub const SEPIA: Self = /* ... */;

    /// Invert colors.
    pub const INVERT: Self = /* ... */;

    /// Saturation adjustment (parameterized).
    pub fn saturation(amount: f32) -> Self { /* ... */ }

    /// Hue rotation (parameterized).
    pub fn hue_rotate(radians: f32) -> Self { /* ... */ }
}
```

**Why a Full 4×4 Matrix?**

A 4×4 matrix can express any linear color transform, including those that mix channels (sepia, hue rotation) or use alpha in calculations. The alternative—separate primitives for brightness, contrast, saturation, etc.—would require more code and wouldn’t compose as cleanly.

The optimizer handles identity rows/columns well (see §9).

### 5.3 Mask: Signal Modulation

**Purpose**: Multiply color values by a scalar intensity field.

```rust
/// A multiplicative modulation transform.
///
/// This applies a scalar "signal" to modulate color intensity.
/// Use for scanlines, vignettes, lighting, and texture overlays.
#[derive(Copy, Clone)]
pub struct Mask<S, M> {
    /// The source surface providing colors.
    pub source: S,
    /// The mask surface providing intensity values (0.0 to 1.0).
    pub mask: M,
}

impl<S, M> Surface for Mask<S, M>
where
    S: Surface<Output = PlanarRGBA>,
    M: Surface<Output = Batch<f32>>,
{
    type Output = PlanarRGBA;

    #[inline(always)]
    fn sample(&self, u: Batch<f32>, v: Batch<f32>) -> PlanarRGBA {
        let color = self.source.sample(u, v);
        let intensity = self.mask.sample(u, v);

        PlanarRGBA {
            r: color.r * intensity,
            g: color.g * intensity,
            b: color.b * intensity,
            a: color.a,  // Alpha typically preserved
        }
    }
}
```

**Use Cases**:

|Effect         |Mask Surface                                    |
|---------------|------------------------------------------------|
|CRT Scanlines  |`Step::new(Fract::new(Y * 100.0), 0.5)`         |
|Vignette       |`1.0 - length(X - center, Y - center) * falloff`|
|Spotlight      |`max(0, 1 - distance_from_light / radius)`      |
|Shadow Map     |Sampled from pre-rendered depth buffer          |
|Texture Overlay|Grayscale texture as mask                       |

**Why Separate from Grade?**

While masking could technically be expressed as a Grade matrix with the mask on the diagonal, separating it provides:

- Clearer intent (modulation vs transformation)
- Simpler API (scalar mask vs matrix construction)
- Potentially better optimization (multiply is cheaper than full matrix)

-----

## 6. Control Flow: Select

**Purpose**: Branchless conditional selection between two surfaces.

```rust
/// Branchless conditional surface selection.
///
/// This is the universal control flow primitive. Hard edges,
/// clipping, palette lookup, and region-based effects are all
/// expressible as Select with appropriate conditions.
#[derive(Copy, Clone)]
pub struct Select<C, T, F> {
    /// The condition surface (outputs boolean mask).
    pub condition: C,
    /// The surface to sample where condition is true.
    pub if_true: T,
    /// The surface to sample where condition is false.
    pub if_false: F,
}

impl<C, T, F> Surface for Select<C, T, F>
where
    C: Surface<Output = BatchMask>,
    T: Surface,
    F: Surface<Output = T::Output>,
{
    type Output = T::Output;

    #[inline(always)]
    fn sample(&self, u: Batch<f32>, v: Batch<f32>) -> Self::Output {
        let mask = self.condition.sample(u, v);
        let t_val = self.if_true.sample(u, v);
        let f_val = self.if_false.sample(u, v);

        // Maps to vblendvps (AVX) or bsl (NEON)
        mask.select(t_val, f_val)
    }
}
```

**Critical Property**: Both branches are always evaluated. This is **branchless SIMD**—no divergence, no pipeline stalls. The mask selects which result to keep per-lane.

**Example: Checkerboard**

```rust
// XOR of quantized coordinates
let check = Xor::new(
    Step::new(Fract::new(X * 10.0), 0.5),
    Step::new(Fract::new(Y * 10.0), 0.5),
);

let board = Select {
    condition: check,
    if_true: Constant(WHITE),
    if_false: Constant(BLACK),
};
```

**Example: Clipping Region**

```rust
let in_bounds = And::new(
    And::new(X >= 0.0, X < width),
    And::new(Y >= 0.0, Y < height),
);

let clipped = Select {
    condition: in_bounds,
    if_true: source,
    if_false: Constant(TRANSPARENT),
};
```

-----

## 7. Escape Hatch: FnSurface

**Purpose**: Allow arbitrary user-defined shaders via scalar closures.

```rust
/// A surface defined by a scalar closure.
///
/// This is the escape hatch for effects that cannot be expressed
/// with the combinator primitives. It sacrifices SIMD efficiency
/// for total flexibility.
#[derive(Copy, Clone)]
pub struct FnSurface<F> {
    pub func: F,
}

impl<F, T> Surface for FnSurface<F>
where
    F: Fn(f32, f32) -> T + Copy + Send + Sync,
    T: Copy,
{
    type Output = Batch<T>;

    #[inline(always)]
    fn sample(&self, u: Batch<f32>, v: Batch<f32>) -> Batch<T> {
        let u_arr = u.to_array();
        let v_arr = v.to_array();
        let mut out = [unsafe { std::mem::zeroed() }; LANES];

        for i in 0..LANES {
            out[i] = (self.func)(u_arr[i], v_arr[i]);
        }

        Batch::from_array(out)
    }
}
```

**When to Use**:

- Complex branching logic that would require deeply nested `Select`
- Algorithms that don’t vectorize well (recursive functions, lookup tables with data-dependent indexing)
- Rapid prototyping before optimizing into primitives
- Interfacing with external libraries that provide scalar APIs

**Performance Characteristics**:

- **No auto-vectorization**: The closure is called `LANES` times sequentially
- **Still batched**: Work is grouped, reducing call overhead
- **Inlines well**: LLVM can often optimize simple closures despite the loop

**Usage**:

```rust
// Mandelbrot set (complex iteration doesn't vectorize cleanly)
let mandelbrot = FnSurface::new(|x, y| {
    let c = Complex::new(x * 3.5 - 2.5, y * 2.0 - 1.0);
    let mut z = Complex::ZERO;
    let mut i = 0;
    while z.norm_sqr() < 4.0 && i < MAX_ITER {
        z = z * z + c;
        i += 1;
    }
    palette[i % palette.len()]
});
```

-----

## 8. Automatic Format Elision

### The Problem

Color manipulation stages (Grade, advanced blending) work best with **planar float** representation:

```rust
struct PlanarRGBA {
    r: Batch<f32>,  // Red channel for all pixels in batch
    g: Batch<f32>,  // Green channel
    b: Batch<f32>,  // Blue channel
    a: Batch<f32>,  // Alpha channel
}
```

But textures and framebuffers use **packed integer** representation:

```rust
type PackedRGBA = Batch<u32>;  // 0xAABBGGRR per pixel
```

Naively, every stage boundary would unpack and repack:

```rust
texture                    // Batch<u32>
    .unpack()              // -> PlanarRGBA
    .grade(brightness)     // -> PlanarRGBA
    .pack()                // -> Batch<u32>
    .unpack()              // -> PlanarRGBA  (wasteful!)
    .grade(saturation)     // -> PlanarRGBA
    .pack()                // -> Batch<u32>
```

### The Solution

Use a trait to abstract over color formats, with identity conversions that compile away:

```rust
/// Trait for color formats that can convert to/from planar representation.
pub trait ColorFormat: Copy {
    fn to_planar(self) -> PlanarRGBA;
    fn from_planar(p: PlanarRGBA) -> Self;
}

// Identity: already planar, no conversion needed
impl ColorFormat for PlanarRGBA {
    #[inline(always)]
    fn to_planar(self) -> PlanarRGBA { self }

    #[inline(always)]
    fn from_planar(p: PlanarRGBA) -> Self { p }
}

// Conversion: packed <-> planar
impl ColorFormat for Batch<u32> {
    #[inline(always)]
    fn to_planar(self) -> PlanarRGBA {
        unpack_rgba(self)  // Actual SIMD work
    }

    #[inline(always)]
    fn from_planar(p: PlanarRGBA) -> Self {
        pack_rgba(p)  // Actual SIMD work
    }
}
```

Grade (and other color operators) accept any `ColorFormat` input:

```rust
impl<S: Surface> Surface for Grade<S>
where
    S::Output: ColorFormat,
{
    type Output = PlanarRGBA;  // Always outputs planar

    fn sample(&self, u: Batch<f32>, v: Batch<f32>) -> PlanarRGBA {
        let input = self.source.sample(u, v).to_planar();  // Maybe no-op!
        apply_matrix(input, self.matrix)
    }
}
```

Chained grades now elide intermediate conversions:

```rust
texture                    // Batch<u32>
    .grade(brightness)     // .to_planar() unpacks once -> PlanarRGBA
    .grade(saturation)     // .to_planar() is identity -> PlanarRGBA
    .grade(hue)            // .to_planar() is identity -> PlanarRGBA
    .pack()                // packs once -> Batch<u32>
```

LLVM sees `fn to_planar(self) -> PlanarRGBA { self }` and eliminates it entirely.

### Pack/Unpack Implementation

```rust
#[inline(always)]
fn unpack_rgba(pixels: Batch<u32>) -> PlanarRGBA {
    let mask = Batch::splat(0xFF);
    let norm = Batch::splat(1.0 / 255.0);

    // Extract channels via bit manipulation
    let r = (pixels & mask).cast::<f32>() * norm;
    let g = ((pixels >> 8) & mask).cast::<f32>() * norm;
    let b = ((pixels >> 16) & mask).cast::<f32>() * norm;
    let a = ((pixels >> 24) & mask).cast::<f32>() * norm;

    PlanarRGBA { r, g, b, a }
}

#[inline(always)]
fn pack_rgba(p: PlanarRGBA) -> Batch<u32> {
    let scale = Batch::splat(255.0);
    let zero = Batch::splat(0.0);
    let max = Batch::splat(255.0);

    // Clamp to prevent overflow, then convert
    let r = (p.r * scale).clamp(zero, max).cast::<u32>();
    let g = (p.g * scale).clamp(zero, max).cast::<u32>();
    let b = (p.b * scale).clamp(zero, max).cast::<u32>();
    let a = (p.a * scale).clamp(zero, max).cast::<u32>();

    // Recombine via bit manipulation
    r | (g << 8) | (b << 16) | (a << 24)
}
```

-----

## 9. Automatic Grade Composition

### The Insight

Matrix multiplication is associative: `(A × B) × C = A × (B × C)`.

This means chained Grade operations can be **fused at construction time**, reducing N matrix multiplications to one:

```rust
impl<S: Surface<Output = PlanarRGBA>> Grade<S> {
    /// Chain another grade, fusing the matrices.
    pub fn grade(self, next_matrix: Mat4, next_bias: Vec4) -> Grade<S> {
        Grade {
            source: self.source,  // Unwrap, don't nest
            matrix: next_matrix * self.matrix,
            bias: next_matrix * self.bias + next_bias,
        }
    }
}
```

### Before (Nested)

```rust
// User writes:
texture.grade(A).grade(B).grade(C)

// Without fusion, this creates:
Grade { source: Grade { source: Grade { source: texture, matrix: A }, matrix: B }, matrix: C }

// At sample time: 3 matrix multiplications per pixel batch
```

### After (Fused)

```rust
// With fusion, this creates:
Grade { source: texture, matrix: C * B * A, bias: combined }

// At sample time: 1 matrix multiplication per pixel batch
```

### Why the Optimizer Might Not Do This

LLVM *could* theoretically fuse these, but:

1. It would need to prove the matrices are constant (hard if loaded from config)
1. Matrix multiplication involves many FMAs that don’t obviously simplify
1. The nested struct access patterns obscure the mathematical relationship

By fusing at construction time, we guarantee the optimization regardless of build settings.

### Identity Elimination

When a matrix is identity, multiplication is a no-op:

```rust
impl Mat4 {
    pub fn is_identity(&self) -> bool {
        *self == Self::IDENTITY
    }
}

impl<S: Surface<Output = PlanarRGBA>> Grade<S> {
    pub fn grade(self, next_matrix: Mat4, next_bias: Vec4) -> Self {
        if next_matrix.is_identity() && next_bias == Vec4::ZERO {
            self  // No-op, return unchanged
        } else if self.matrix.is_identity() && self.bias == Vec4::ZERO {
            Grade { source: self.source, matrix: next_matrix, bias: next_bias }
        } else {
            Grade {
                source: self.source,
                matrix: next_matrix * self.matrix,
                bias: next_matrix * self.bias + next_bias,
            }
        }
    }
}
```

-----

## 10. Extension Traits and Ergonomics

### SurfaceExt

Provides convenient methods on all surfaces:

```rust
pub trait SurfaceExt: Surface + Sized {
    /// Apply a coordinate warp.
    fn warp<X, Y>(self, x: X, y: Y) -> Warp<Self, X, Y>
    where
        X: Surface<Output = Batch<f32>>,
        Y: Surface<Output = Batch<f32>>,
    {
        Warp { source: self, map_u: x, map_v: y }
    }

    /// Translate by a fixed offset.
    fn offset(self, dx: f32, dy: f32) -> Warp<Self, impl Surface, impl Surface> {
        self.warp(X + dx, Y + dy)
    }

    /// Scale by fixed factors.
    fn scale(self, sx: f32, sy: f32) -> Warp<Self, impl Surface, impl Surface> {
        self.warp(X * sx, Y * sy)
    }

    /// Rotate around the origin.
    fn rotate(self, radians: f32) -> Warp<Self, impl Surface, impl Surface> {
        let (sin, cos) = radians.sin_cos();
        self.warp(X * cos - Y * sin, X * sin + Y * cos)
    }

    /// Apply multiplicative masking.
    fn mask<M>(self, mask: M) -> Mask<Self, M>
    where
        M: Surface<Output = Batch<f32>>,
    {
        Mask { source: self, mask }
    }

    /// Conditional selection.
    fn select<C, F>(self, condition: C, if_false: F) -> Select<C, Self, F>
    where
        C: Surface<Output = BatchMask>,
        F: Surface<Output = Self::Output>,
    {
        Select { condition, if_true: self, if_false }
    }
}

impl<T: Surface> SurfaceExt for T {}
```

### ColorExt

Additional methods for color-outputting surfaces:

```rust
pub trait ColorExt: Surface<Output = PlanarRGBA> + Sized {
    /// Apply color grading.
    fn grade(self, matrix: Mat4, bias: Vec4) -> Grade<Self> {
        Grade { source: self, matrix, bias }
    }

    /// Adjust brightness.
    fn brightness(self, amount: f32) -> Grade<Self> {
        self.grade(Mat4::IDENTITY, Vec4::splat(amount))
    }

    /// Adjust saturation.
    fn saturate(self, amount: f32) -> Grade<Self> {
        self.grade(Mat4::saturation(amount), Vec4::ZERO)
    }

    /// Convert to grayscale.
    fn grayscale(self) -> Grade<Self> {
        self.grade(Mat4::GRAYSCALE, Vec4::ZERO)
    }

    /// Apply sepia tone.
    fn sepia(self) -> Grade<Self> {
        self.grade(Mat4::SEPIA, Vec4::ZERO)
    }
}

impl<T: Surface<Output = PlanarRGBA>> ColorExt for T {}
```

### Arithmetic Operators

Surfaces support arithmetic for coordinate manipulation:

```rust
impl<A: Surface<Output = Batch<f32>>> Add<f32> for A {
    type Output = SurfaceAdd<A, Constant<f32>>;

    fn add(self, rhs: f32) -> Self::Output {
        SurfaceAdd(self, Constant(rhs))
    }
}

impl<A, B> Add<B> for A
where
    A: Surface<Output = Batch<f32>>,
    B: Surface<Output = Batch<f32>>,
{
    type Output = SurfaceAdd<A, B>;

    fn add(self, rhs: B) -> Self::Output {
        SurfaceAdd(self, rhs)
    }
}

// Similarly for Sub, Mul, Div, Neg, etc.
```

This enables natural expressions:

```rust
let wave = X + (Y * 0.1).sin() * 5.0;
let radial = (X * X + Y * Y).sqrt();
```

-----

## 11. Alternatives Considered

### 11.1 Runtime Shader Graphs (Rejected)

**Approach**: Build shader graphs as runtime data structures with virtual dispatch.

```rust
enum ShaderNode {
    Texture(TextureId),
    Grade(Box<ShaderNode>, Matrix4),
    Warp(Box<ShaderNode>, Box<ShaderNode>, Box<ShaderNode>),
    // ...
}
```

**Why Rejected**:

- Virtual dispatch prevents inlining
- Heap allocation for every composition
- Graph traversal overhead at sample time
- Cannot benefit from compile-time optimization

**When It Would Be Appropriate**: If shaders need to be defined at runtime (user-provided shader scripts, hot-reloading), a runtime graph would be necessary. Our use case (terminal effects, games) has static shaders known at compile time.

### 11.2 Macro-Based DSL (Rejected)

**Approach**: Define a domain-specific language that macros expand to SIMD code.

```rust
shader! {
    let warped = sample(texture, u + sin(v * 10.0), v);
    let graded = warped * brightness;
    output(graded)
}
```

**Why Rejected**:

- Macros are hard to debug (errors in generated code)
- IDE support (autocomplete, type checking) is poor
- Users must learn a new syntax
- Less flexible than native Rust

**When It Would Be Appropriate**: If we needed to target multiple backends (CPU SIMD, GPU shaders, etc.) from a single source, a DSL would enable that. For CPU-only rendering, native Rust is simpler.

### 11.3 Scalar-First with Auto-Vectorization (Partially Adopted)

**Approach**: Users write scalar code; the compiler vectorizes.

```rust
trait Surface {
    fn sample_one(&self, u: f32, v: f32) -> u32;

    // Default implementation batches scalar calls
    fn sample(&self, u: Batch<f32>, v: Batch<f32>) -> Batch<u32> {
        Batch::from_fn(|i| self.sample_one(u[i], v[i]))
    }
}
```

**Why Partially Adopted**:

- Clean user API (scalar thinking)
- Works for simple cases
- But: LLVM often fails to vectorize non-trivial code
- But: Hand-written SIMD is still faster for primitives

**Our Compromise**: `FnSurface` provides the scalar escape hatch. Built-in primitives use hand-written SIMD. Users get both options.

### 11.4 Separate Geometry/Color/Signal Pipelines (Rejected)

**Approach**: Three distinct pipeline types that don’t interleave.

```rust
let geometry = TranslateWarp::new(RotateWarp::new(...));
let color = SepiaGrade::new(BrightnessGrade::new(...));
let signal = ScanlinesSignal::new(VignetteSignal::new(...));

let final = Pipeline::new(geometry, color, signal);
```

**Why Rejected**:

- Prevents interleaving (e.g., warp → grade → warp)
- Less expressive than arbitrary composition
- No mathematical basis for the restriction

-----

## 12. Performance Characteristics

### Compile Time

The combinator pattern creates deeply nested generic types:

```rust
Warp<Grade<Mask<Texture, SineWave>, Mat4>, SurfaceAdd<X, ...>, Y>
```

This increases compile times due to:

- Monomorphization of each unique type
- LLVM optimization of large inlined functions

**Mitigation**: Use `#[inline(always)]` judiciously. Consider `#[inline(never)]` on debug builds.

### Runtime

|Operation       |Cycles (approx)|Notes                         |
|----------------|---------------|------------------------------|
|Batch arithmetic|1-3            |Pure ALU                      |
|Pack/Unpack     |5-10           |Format conversion             |
|Matrix multiply |20-30          |16 FMAs + adds                |
|Texture sample  |10-50          |Memory-bound, depends on cache|
|FnSurface scalar|N × scalar cost|No vectorization              |

**Memory Access Patterns**:

- Sequential framebuffer writes (good cache behavior)
- Texture reads may be scattered (depends on warp)
- No intermediate buffers between stages

### Benchmarks (Expected)

|Shader            |Pixels/sec (single core)|Notes                   |
|------------------|------------------------|------------------------|
|Solid color       |~2B                     |Memory bandwidth limited|
|Texture copy      |~500M                   |Cache dependent         |
|Texture + Grade   |~300M                   |One unpack + matrix     |
|Full CRT effect   |~100M                   |Warp + Grade + Mask     |
|FnSurface (simple)|~50M                    |Scalar fallback         |

-----

## 13. Differential Surfaces (Trait Now, Impls Tomorrow)

### Mathematical Foundation

By modeling our system as `f(u, v)`, we’ve entered the domain of **differential geometry**. If we track not just values but their *rates of change*, we unlock three capabilities that standard sprite-pushing engines struggle to implement:

|Capability                   |Problem Solved |Mechanism                       |
|-----------------------------|---------------|--------------------------------|
|Automatic Anti-aliasing      |Jagged edges   |Gradient reveals edge coverage  |
|Perfect Mipmapping           |Texture shimmer|Jacobian gives texture footprint|
|Resolution-Independent Shapes|Blurry zooms   |SDFs remain crisp at any scale  |

### Dual Numbers and Automatic Differentiation

Instead of tracking just the value `v` at a point, we track the value and its rate of change (gradient) relative to screen coordinates:

```
(value, ∂value/∂x, ∂value/∂y)
```

This is **forward-mode automatic differentiation**. The derivatives propagate through the computation graph via the chain rule—automatically.

**Example**: When rendering a circle edge, if `sample()` returns 0.5 (edge) but the gradient is steep, we know we’re crossing an edge *within* this pixel. The gradient magnitude tells us how much of the pixel is covered.

### The Jacobian

The **Jacobian matrix** describes how coordinate space stretches or compresses at any point:

```
J = | ∂u/∂x  ∂u/∂y |
    | ∂v/∂x  ∂v/∂y |
```

The **determinant** of this matrix gives the ratio of texture area to screen pixel area:

- `det(J) ≈ 1.0`: 1:1 mapping, use mip level 0
- `det(J) ≈ 4.0`: Each pixel covers 4 texels, use mip level 2
- `det(J) ≈ 0.25`: Magnification, use mip level 0 + filtering

Since coordinates are surfaces, and `Warp` transforms coordinates, the Jacobian computation falls out naturally from the derivative propagation.

### Signed Distance Fields (SDFs)

SDFs store the **distance to the nearest edge** rather than the shape itself:

- Negative → Inside
- Positive → Outside
- Zero → Edge

SDFs are simply `Surface<Output = Batch<f32>>`:

```rust
// Circle SDF
fn sample(&self, u: DiffCoord, v: DiffCoord) -> Batch<f32> {
    (u.val * u.val + v.val * v.val).sqrt() - self.radius
}

// Composition via min (union) and max (intersection)
let rounded_box = Max(rect_sdf, Neg(corner_circles));
```

Combined with derivatives, SDFs enable perfectly crisp edges at any zoom level and rotation.

### The DiffCoord Type

```rust
/// A differential coordinate: value plus screen-space derivatives.
///
/// This represents a coordinate and how it changes as we move across
/// the screen. Enables automatic mipmapping, anti-aliasing, and
/// resolution-independent rendering.
#[derive(Copy, Clone)]
pub struct DiffCoord {
    /// The coordinate value.
    pub val: Batch<f32>,
    /// Rate of change moving one pixel right (∂/∂screen_x).
    pub dx: Batch<f32>,
    /// Rate of change moving one pixel down (∂/∂screen_y).
    pub dy: Batch<f32>,
}

impl DiffCoord {
    /// Create a differential coordinate with no variation (constant).
    #[inline(always)]
    pub fn constant(val: Batch<f32>) -> Self {
        Self {
            val,
            dx: Batch::splat(0.0),
            dy: Batch::splat(0.0),
        }
    }

    /// The screen-space X coordinate (identity: dx=1, dy=0).
    #[inline(always)]
    pub fn screen_x(x: Batch<f32>) -> Self {
        Self {
            val: x,
            dx: Batch::splat(1.0),
            dy: Batch::splat(0.0),
        }
    }

    /// The screen-space Y coordinate (identity: dx=0, dy=1).
    #[inline(always)]
    pub fn screen_y(y: Batch<f32>) -> Self {
        Self {
            val: y,
            dx: Batch::splat(0.0),
            dy: Batch::splat(1.0),
        }
    }

    /// Compute the magnitude of the gradient (for LOD selection).
    #[inline(always)]
    pub fn gradient_magnitude(&self) -> Batch<f32> {
        (self.dx * self.dx + self.dy * self.dy).sqrt()
    }

    /// Compute log2 of gradient magnitude (mipmap level).
    #[inline(always)]
    pub fn lod(&self) -> Batch<f32> {
        self.gradient_magnitude().log2()
    }
}
```

### Derivative Propagation (Chain Rule)

Arithmetic operations propagate derivatives automatically:

```rust
impl Add<DiffCoord> for DiffCoord {
    type Output = DiffCoord;

    #[inline(always)]
    fn add(self, rhs: DiffCoord) -> DiffCoord {
        DiffCoord {
            val: self.val + rhs.val,
            dx: self.dx + rhs.dx,   // d(a+b)/dx = da/dx + db/dx
            dy: self.dy + rhs.dy,
        }
    }
}

impl Mul<DiffCoord> for DiffCoord {
    type Output = DiffCoord;

    #[inline(always)]
    fn mul(self, rhs: DiffCoord) -> DiffCoord {
        DiffCoord {
            val: self.val * rhs.val,
            // Product rule: d(ab)/dx = a*db/dx + b*da/dx
            dx: self.val * rhs.dx + rhs.val * self.dx,
            dy: self.val * rhs.dy + rhs.val * self.dy,
        }
    }
}

impl DiffCoord {
    /// Sine with derivative propagation.
    #[inline(always)]
    pub fn sin(self) -> DiffCoord {
        let cos_val = self.val.cos();
        DiffCoord {
            val: self.val.sin(),
            dx: cos_val * self.dx,  // d(sin(u))/dx = cos(u) * du/dx
            dy: cos_val * self.dy,
        }
    }

    /// Square root with derivative propagation.
    #[inline(always)]
    pub fn sqrt(self) -> DiffCoord {
        let sqrt_val = self.val.sqrt();
        let inv_2sqrt = Batch::splat(0.5) / sqrt_val;
        DiffCoord {
            val: sqrt_val,
            dx: inv_2sqrt * self.dx,  // d(√u)/dx = du/dx / (2√u)
            dy: inv_2sqrt * self.dy,
        }
    }
}
```

### The DiffSurface Trait

```rust
/// A surface that accepts differential coordinates.
///
/// This enables automatic computation of texture LOD, anti-aliased edges,
/// and resolution-independent rendering. Surfaces implementing this trait
/// receive screen-space derivative information and can use it for filtering.
///
/// # Implementation Status
///
/// **The trait is defined now; implementations are future work.**
///
/// Current `Surface` implementations will continue to work. When we need
/// differential features (mipmapping, AA, SDFs), we add `DiffSurface`
/// impls to the primitives that benefit from them.
pub trait DiffSurface: Copy + Send + Sync {
    /// The output type of this surface.
    type Output: Copy;

    /// Sample the surface with differential coordinates.
    ///
    /// The derivatives in `u` and `v` describe how texture coordinates
    /// change across the screen, enabling:
    /// - Mipmap level selection (via `u.lod()`, `v.lod()`)
    /// - Anti-aliased edges (via gradient magnitude)
    /// - Anisotropic filtering (via full Jacobian)
    fn sample_diff(&self, u: DiffCoord, v: DiffCoord) -> Self::Output;
}
```

### Bridging: Surface ↔ DiffSurface

Any `DiffSurface` can act as a `Surface` by constructing differential coordinates at the call site:

```rust
/// Wrapper that adapts a DiffSurface to the Surface trait.
#[derive(Copy, Clone)]
pub struct AsSurface<S>(pub S);

impl<S: DiffSurface> Surface for AsSurface<S> {
    type Output = S::Output;

    #[inline(always)]
    fn sample(&self, u: Batch<f32>, v: Batch<f32>) -> Self::Output {
        // At the screen boundary, we know derivatives are (1,0) and (0,1)
        self.0.sample_diff(DiffCoord::screen_x(u), DiffCoord::screen_y(v))
    }
}
```

Conversely, any `Surface` can act as a `DiffSurface` by ignoring derivatives:

```rust
/// Wrapper that adapts a Surface to the DiffSurface trait (drops derivatives).
#[derive(Copy, Clone)]
pub struct AsDiffSurface<S>(pub S);

impl<S: Surface> DiffSurface for AsDiffSurface<S> {
    type Output = S::Output;

    #[inline(always)]
    fn sample_diff(&self, u: DiffCoord, v: DiffCoord) -> Self::Output {
        // Ignore derivatives, just use values
        self.0.sample(u.val, v.val)
    }
}
```

### Example: Differential Warp

When `Warp` is upgraded to `DiffSurface`, it automatically computes the Jacobian:

```rust
impl<S, X, Y> DiffSurface for Warp<S, X, Y>
where
    S: DiffSurface,
    X: DiffSurface<Output = Batch<f32>>,
    Y: DiffSurface<Output = Batch<f32>>,
{
    type Output = S::Output;

    #[inline(always)]
    fn sample_diff(&self, u: DiffCoord, v: DiffCoord) -> Self::Output {
        // Transform coordinates (this is where derivatives propagate)
        let new_u = self.map_u.sample_diff(u, v);  // Returns DiffCoord
        let new_v = self.map_v.sample_diff(u, v);  // Returns DiffCoord

        // The new_u and new_v now contain the Jacobian of the warp!
        // new_u.dx = ∂(map_u)/∂x, new_u.dy = ∂(map_u)/∂y
        // new_v.dx = ∂(map_v)/∂x, new_v.dy = ∂(map_v)/∂y

        self.source.sample_diff(new_u, new_v)
    }
}
```

### Example: Differential Texture Sampling

```rust
impl<'a> DiffSurface for Texture<'a> {
    type Output = Batch<u32>;

    #[inline(always)]
    fn sample_diff(&self, u: DiffCoord, v: DiffCoord) -> Batch<u32> {
        // Compute LOD from derivatives
        let dudx = u.dx.abs();
        let dudy = u.dy.abs();
        let dvdx = v.dx.abs();
        let dvdy = v.dy.abs();

        // Maximum rate of change determines LOD
        let max_deriv = dudx.max(dudy).max(dvdx).max(dvdy);
        let lod = max_deriv.log2().max(Batch::splat(0.0));

        // Sample from appropriate mipmap level
        self.sample_mipmap(u.val, v.val, lod)
    }
}
```

### Example: Anti-Aliased SDF Circle

```rust
#[derive(Copy, Clone)]
pub struct Circle {
    pub center_x: f32,
    pub center_y: f32,
    pub radius: f32,
}

impl DiffSurface for Circle {
    type Output = Batch<f32>;  // Returns signed distance

    #[inline(always)]
    fn sample_diff(&self, u: DiffCoord, v: DiffCoord) -> Batch<f32> {
        let dx = u.val - Batch::splat(self.center_x);
        let dy = v.val - Batch::splat(self.center_y);

        // Signed distance to edge
        (dx * dx + dy * dy).sqrt() - Batch::splat(self.radius)
    }
}

/// Convert SDF to anti-aliased alpha using derivatives.
#[derive(Copy, Clone)]
pub struct SdfToAlpha<S> {
    pub sdf: S,
}

impl<S: DiffSurface<Output = Batch<f32>>> DiffSurface for SdfToAlpha<S> {
    type Output = Batch<f32>;  // Returns alpha (0.0 to 1.0)

    #[inline(always)]
    fn sample_diff(&self, u: DiffCoord, v: DiffCoord) -> Batch<f32> {
        let distance = self.sdf.sample_diff(u, v);

        // Gradient magnitude tells us pixel size in SDF space
        let gradient = (u.dx * u.dx + u.dy * u.dy + v.dx * v.dx + v.dy * v.dy).sqrt();

        // Smooth step based on pixel size
        // Inside (negative distance) → 1.0, Outside → 0.0
        // Transition width = one pixel in SDF space
        let edge_width = gradient;
        smoothstep(edge_width, -edge_width, distance)
    }
}
```

### When to Use DiffSurface

|Use Case                         |Surface|DiffSurface|
|---------------------------------|-------|-----------|
|Terminal text (fixed size)       |✓      |           |
|Pixel art (intentionally aliased)|✓      |           |
|Axis-aligned sprites             |✓      |           |
|Rotated/scaled sprites           |       |✓          |
|Resolution-independent UI        |       |✓          |
|Vector font rendering            |       |✓          |
|Ray tracing smooth surfaces      |       |✓          |

### Implementation Roadmap

**Phase 1 (Current)**: Define traits and types. Ship `Surface`-based rendering.

**Phase 2 (When Needed)**: Add `DiffSurface` impls to:

- `Texture` (mipmapping)
- `Warp` (Jacobian propagation)
- SDF primitives (circles, boxes, etc.)
- `SdfToAlpha` (anti-aliased edges)

**Phase 3 (Optimization)**: SIMD-optimized derivative propagation for hot paths.

-----

## 14. Future Work

### 14.1 Multi-Pass Effects

Some effects require render-to-texture:

- Bloom (downscale → blur → composite)
- Large-kernel blur (ping-pong passes)
- Shadow mapping
- Temporal anti-aliasing

This requires orchestration above the Surface abstraction—a `RenderGraph` that manages intermediate textures and pass ordering.

### 14.2 WASM Target

The shader algebra should port cleanly to WebAssembly with `wasm32-simd128`:

- 128-bit SIMD matches current Batch width
- Web Workers enable multi-core
- Browser demo would showcase the engine

### 14.3 GPU Backend

The functional structure maps naturally to GPU compute shaders:

- Each Surface becomes a shader function
- Composition becomes function calls
- Warp is texture sampling with computed UVs
- Grade is matrix uniform × input

A transpiler could emit WGSL/GLSL from the Rust surface graph.

### 14.4 Hot-Reloading

For development iteration, shader parameters (matrices, frequencies, colors) could be hot-reloadable without recompilation. This would require:

- Separating shader structure from parameters
- Runtime parameter binding
- File watcher integration

-----

## Appendix A: Complete Example

```rust
use pixelflow::prelude::*;

fn main() {
    // Load assets
    let font_atlas = Texture::from_file("font.png");
    let noise = Texture::from_file("perlin.png");

    // Build the shader pipeline
    let screen = TerminalBuffer::new(80, 24)
        // Geometry: subtle wave distortion
        .warp(
            X + noise.sample_gray(X * 0.01, Y * 0.01) * 2.0,
            Y,
        )
        // Color: warm tint
        .unpack()
        .grade(Mat4::saturation(1.1), Vec4::new(0.02, 0.01, 0.0, 0.0))
        // Signal: CRT scanlines
        .mask(Scanlines::new(frequency: 2.0, intensity: 0.1))
        // Signal: vignette
        .mask(Vignette::new(radius: 0.8, softness: 0.3))
        // Back to packed for framebuffer
        .pack();

    // Render loop
    loop {
        for y in 0..HEIGHT {
            for x in (0..WIDTH).step_by(LANES) {
                let u = Batch::from_fn(|i| (x + i) as f32);
                let v = Batch::splat(y as f32);
                let pixels = screen.sample(u, v);
                framebuffer.write(x, y, pixels);
            }
        }
        present();
    }
}
```

-----

## Appendix B: Glossary

|Term           |Definition                                                           |
|---------------|---------------------------------------------------------------------|
|**Batch**      |A SIMD vector of N pixels processed in parallel                      |
|**DiffCoord**  |A coordinate value plus its screen-space derivatives                 |
|**Dual Number**|A value paired with its derivative; enables automatic differentiation|
|**Eigenshader**|One of the three fundamental shader operations (Warp, Grade, Mask)   |
|**Jacobian**   |Matrix of partial derivatives describing coordinate space distortion |
|**LOD**        |Level of Detail; mipmap level for texture sampling                   |
|**Planar**     |Color format with separate vectors per channel (SoA)                 |
|**Packed**     |Color format with channels interleaved per pixel (AoS)               |
|**SDF**        |Signed Distance Field; stores distance to nearest edge               |
|**Surface**    |A lazy function from coordinates to values                           |
|**Warp**       |Coordinate remapping transform                                       |
|**Grade**      |Linear color transform via matrix                                    |
|**Mask**       |Multiplicative intensity modulation                                  |

-----

*End of Document*
