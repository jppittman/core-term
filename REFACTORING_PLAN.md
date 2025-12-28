# PixelFlow Examples Refactoring Plan

## Overview

Factor out bespoke example code into reusable, idiomatic PixelFlow library code. The goal is to extract general patterns from `scene3d`, `animated_sphere`, and font examples and compose them using manifold combinators rather than raw field operations.

## Current State Analysis

### Code Duplication Issues

1. **Surface Duplication**: `scene3d.rs` has two near-identical implementations:
   - `Surface<G, M, B>` outputs `Field` (grayscale)
   - `ColorSurface<G, M, B>` outputs `Discrete` (color)
   - These should be unified into a single generic combinator

2. **ScreenToDir Duplication**:
   - `ScreenToDir<M>` for Field output
   - `ColorScreenToDir<M>` for Discrete output
   - Duplicates the ray normalization logic

3. **Reflect Duplication**:
   - `Reflect<M>` and `ColorReflect<M>` with identical math, different output types

### Patterns to Extract

#### 1. **3D Scene Combinators** (Scene3D Library Module)
Currently in `scene3d.rs` but scattered:
- `ScreenToDir` → Generic ray direction converter
- `Surface` → Generic geometry/material/background compositor
- `Reflect` → Householder reflection for any manifold type
- Materials: `Checker`, `Sky` → Generalize to trait-based approach

#### 2. **Animation Combinators** (New Animation Module)
From `animated_sphere.rs` example:
- `TimeShift<M>` → Translate W (time) dimension (reusable!)
- `ScreenRemap<M>` → Normalize screen coordinates (reusable!)
- `OscillatingSphere` → Generalize to `Oscillate<G, F>` where G is geometry and F is oscillation function
- Animation pattern: wrap geometry with Sin(W * freq) * amplitude

#### 3. **Shape Combinators** (Enhance shapes.rs)
Current `shapes.rs` shows idiomatic pattern:
- Add more primitives: `sphere`, `ellipse`, `polygon`, `annulus`
- Generalize patterns: `bounded<M>`, `annular<M>` for composable bounds
- Maintain manifold composition style (no raw field operations)

#### 4. **Material Trait Pattern** (scene3d.rs Enhancement)
Currently materials are individual structs (`Checker`, `Sky`).
- Create `Material<Input, Output>` trait for composable materials
- Implement as combinators: `Gradient`, `Grid`, `Noise`, etc.
- Allows composition like `Gradient.compose(Transform::scale(2.0))`

#### 5. **Generic Transform Compositors** (New transforms module extension)
Extract from font `Affine<M>`:
- `Scale<M>` - uniform and non-uniform scaling
- `Rotate<M>` - rotation matrices
- `Shear<M>` - shear transforms
- `Translate<M>` - translation wrapper
- All already exist in fonts via `Affine<M>`, should be genericized

## Implementation Strategy

### Phase 1: Unify Duplicated Code

**1a. Create unified `Surface` combinator**
```rust
// pixelflow-graphics/src/scene3d.rs (refactored)
pub struct Surface<G, M, B> {
    pub geometry: G,
    pub material: M,
    pub background: B,
}

// Works with any Jet3→Output type via blanket impl
impl<G, M, B, O> Manifold<Jet3> for Surface<G, M, B>
where
    G: Manifold<Jet3, Output = Jet3>,
    M: Manifold<Jet3, Output = O>,
    B: Manifold<Jet3, Output = O>,
    O: Manifold<Output = O>, // Need MaskOps trait
```

**1b. Create unified `ScreenToDir` combinator**
- Single implementation that works with any output type
- Follows same pattern as generic `Surface`

**1c. Create unified `Reflect` combinator**
- Generic over output type

**Action**: Remove `Color*` variants, rely on unified generics

### Phase 2: Extract Animation Module

**Create**: `pixelflow-graphics/src/animation.rs`

```rust
// Time shifting - translates W (time dimension)
pub struct TimeShift<M> { inner: M, offset: f32 }

// Screen coordinate normalization
pub struct ScreenRemap<M> { inner: M, width: f32, height: f32 }

// Compositional oscillation for any geometry
pub struct Oscillate<G, F> {
    geometry: G,
    oscillation: F,  // Jet3 → Jet3 function
    amplitude: f32,
    frequency: f32,
}

// Generic wave functions (Sin, Cos, etc)
// Use pixelflow_core::ops::Sin directly via composition
```

**Manifest as**: Time-varying geometry composition pattern
- Animated sphere becomes: `Surface { geometry: Oscillate::new(sphere, sin_wave), ... }`
- TimeShift sets current time by wrapping the scene

### Phase 3: Enhance Shapes Module

**Add to**: `pixelflow-graphics/src/shapes.rs`

Geometric primitives following manifold composition:
```rust
// Existing (good patterns):
pub fn circle<F, B>(fg: F, bg: B) -> impl Manifold<Output = Field>
pub fn square<F, B>(fg: F, bg: B) -> Select<...>

// New primitives:
pub fn rectangle<F, B>(width: f32, height: f32, fg: F, bg: B)
pub fn ellipse<F, B>(rx: f32, ry: f32, fg: F, bg: B)
pub fn annulus<F, B>(r_inner: f32, r_outer: f32, fg: F, bg: B)
pub fn polygon<F, B>(vertices: &[[f32; 2]], fg: F, bg: B)

// Bounding combinators:
pub fn bounded<M, C>(cond: C, fg: M, bg: f32) -> Select<C, M, f32>
```

### Phase 4: Material Trait Pattern

**Create**: `pixelflow-graphics/src/materials.rs`

```rust
pub trait Material<T: Manifold>: Manifold {
    fn compose(self, other: T) -> impl Manifold;
}

// Concrete implementations:
pub struct Checker { color_a: (f32,f32,f32), color_b: (f32,f32,f32) }
pub struct Gradient { start: Color, end: Color }
pub struct Noise { scale: f32, amplitude: f32 }

// All follow manifold composition pattern
```

### Phase 5: Generalize Transform Combinators

**Enhance**: `pixelflow-graphics/src/fonts/combinators.rs` or new `combinators.rs`

Already good patterns in fonts:
- `Affine<M>` is already generic
- `Sum<M>` is already generic
- Extract to top-level if fonts-specific

Consider generalizing for any manifold:
```rust
pub struct Scale<M> { inner: M, factor: f32 }
pub struct Translate<M> { inner: M, dx: f32, dy: f32 }
pub struct Rotate<M> { inner: M, theta: f32 }
```

## Code Organization

### New Module Structure

```
pixelflow-graphics/
├── src/
│   ├── lib.rs (updated exports)
│   ├── shapes.rs (enhanced with more primitives)
│   ├── scene3d.rs (refactored, unified generics, no duplication)
│   ├── animation.rs (NEW - TimeShift, ScreenRemap, Oscillate)
│   ├── materials.rs (NEW - Material trait and implementations)
│   ├── combinators.rs (NEW - generic transforms: Scale, Translate, Rotate)
│   └── [existing modules: fonts, render, transform, ...]
```

### Public API Changes

**Removals**:
- `ColorScreenToDir` (use generic `ScreenToDir`)
- `ColorSurface` (use generic `Surface`)
- `ColorReflect` (use generic `Reflect`)
- `ColorSky`, `ColorChecker` (use generic materials)

**Additions**:
- `pub mod animation` with `TimeShift`, `ScreenRemap`, `Oscillate`
- `pub mod materials` with `Material` trait and implementations
- `pub mod combinators` with generic transforms
- Enhanced `shapes` with new primitives

## Style Compliance Checklist

- [x] **Manifold Composition**: Use combinators, not raw field operations
- [x] **Generic Over Concrete**: Single generic Surface instead of Field/Discrete variants
- [x] **No Deep Nesting**: Guard clauses in Surface mask logic
- [x] **Clear Type Names**: `ScreenToDir`, `TimeShift`, `Oscillate` are descriptive
- [x] **Rust Doc**: Document public combinator contracts
- [x] **Comments**: Only explain WHY (e.g., why soft sqrt for grazing angles)
- [x] **No Boolean Arguments**: Use enums where needed (e.g., `Material` trait)
- [x] **Function Count**: Keep combinators < 4 arguments by design
- [x] **Idempotent APIs**: Scene construction is idempotent

## Testing Strategy

1. **Unit Tests**: Each combinator tested against documented contract
2. **Example Updates**: Update all examples to use new composable API
   - `animated_sphere.rs` → Uses `Oscillate<SphereAt>` composition
   - `chrome_sphere.rs` → Uses generic `Surface` and materials
3. **Behavior Tests**: Render output identical before/after refactoring
4. **Performance**: Verify no regression in compilation or runtime

## Migration Path

1. Keep old code during transition
2. Add new combinators with clear names
3. Update examples to demonstrate new patterns
4. Deprecate old code
5. Remove in next major version (or when all examples migrated)

---

## Key Principles

**Idiomatic PixelFlow**: Write combinators, not code.
- A sphere is `SphereAt`, not raw math
- Animation is `TimeShift(scene)`, not imperative time handling
- Transforms are `Scale(manifold)`, not raw matrix ops
- Composition is _nesting_, not manual threading

**Manifold Algebra**: Everything composes via Manifold trait.
- Types are shaders (monomorphize at compile time)
- No runtime dispatch
- Short-circuit evaluation via Select's all/any checks

**Pull-Based**: Nothing computes until coordinates arrive.
- Manifolds are functions, not data structures
- Lazy evaluation built-in
