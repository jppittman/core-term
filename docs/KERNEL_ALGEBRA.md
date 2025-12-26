# PixelFlow Kernel Algebra

**Status**: Implemented
**Location**: `pixelflow-core/src/combinators/kernel.rs`

---

## The Problem

Non-local operations are expensive. Blur, shadows, ambient occlusion, global illumination, attention—all have the same shape:

```
Output(query) = ∫ K(query, source) · Value(source) d(source)
```

This is an integral over all sources. Naively: O(N) per query.

## The Insight

If the kernel K has **symmetry**, there exists a **basis** where K is diagonal (or sparse). In that basis, the integral becomes a dot product: O(basis size).

The kernel's eigenfunctions ARE the natural basis. You don't choose it—the symmetry tells you.

| Symmetry | Basis | Why |
|----------|-------|-----|
| Translation-invariant | Fourier | Eigenfunctions of translation group |
| Rotation-invariant (3D) | Spherical Harmonics | Eigenfunctions of SO(3) |
| Scale-invariant | Wavelets | Eigenfunctions of scale group |
| Isotropic (translation + rotation) | Radial | Collapses to 1D |

This is representation theory. Mathematicians solved it in the 1800s.

## The Architecture

### Symmetry as Trait

```rust
trait Symmetry {
    type Basis: Basis;
    type ProductTable;

    const PRODUCT: Self::ProductTable;  // precomputed, static
}

struct RotationInvariant3D;
impl Symmetry for RotationInvariant3D {
    type Basis = SphericalHarmonics<9>;
    type ProductTable = &'static [(usize, usize, usize, f32)];

    // Clebsch-Gordan coefficients - constants of the universe
    const PRODUCT: Self::ProductTable = CG_ORDER_2;
}
```

### Basis Operations

```rust
trait Basis: Sized {
    type Coeffs: Dot;

    /// Project a field onto this basis (bake step)
    fn project<M: Manifold>(field: M) -> Self::Coeffs;

    /// Evaluate basis functions at a point
    fn eval(&self, at: Coord) -> Self::Coeffs;
}
```

### Compressed Fields

```rust
/// A field projected onto its symmetry-appropriate basis.
/// Sampling is O(basis size), not O(field complexity).
struct Compressed<S: Symmetry> {
    coeffs: <S::Basis as Basis>::Coeffs,
    frame: Frame,
}

impl<S: Symmetry> Compressed<S> {
    pub fn bake<M: Manifold>(field: M, frame: Frame) -> Self {
        let coeffs = S::Basis::project(field.warp(frame.inverse()));
        Compressed { coeffs, frame }
    }

    pub fn sample(&self, query: Coord) -> Jet2 {
        let local = self.frame.apply(query);
        let weights = S::Basis::eval(local);
        weights.dot(&self.coeffs)
    }
}
```

### Multiplication Preserves Basis

```rust
impl<S: Symmetry> Mul for Compressed<S> {
    type Output = Compressed<S>;

    fn mul(self, other: Self) -> Self::Output {
        // Use precomputed product table
        Compressed {
            coeffs: sh_multiply(&self.coeffs, &other.coeffs, S::PRODUCT),
            frame: self.frame,
        }
    }
}
```

## Frame: Gauge Freedom

The same kernel can be easier or harder depending on where you "stand."

```rust
struct Frame {
    origin: (f32, f32, f32),
    // Optional rotation alignment
}
```

Light transport from a point: put frame at the light (isotropic from there).
Irradiance at a surface: align frame with the normal.

The frame is metadata. The kernel knows its natural frame.

## What We Don't Have: Separate Shadow/AO Kernels

This is crucial. Shadows are NOT a kernel.

**Shadows = Light × Visibility**

Both live in the same basis (SH for rotation-invariant). Multiply the coefficients. Done.

**Penumbra = Gradient of Visibility**

Jet2 gives us gradients for free. High gradient at visibility boundary = soft edge. That's penumbra. Not computed separately—emergent.

**Ambient Occlusion = DC term of Visibility**

The 0th SH coefficient of the visibility field is "how much of the hemisphere is blocked." That's AO. It's just `visibility_coeffs[0]`.

## Practical Example: Irradiance

```rust
// Environment light, baked once
let env: Compressed<RotationInvariant3D> = Compressed::bake(sky, Frame::identity());

// At each shading point:
let vis = Compressed::bake(visibility_field, Frame::aligned_to(point, normal));
let cosine = cosine_lobe_sh(normal);  // analytic

// All three are SH. Multiplication = coefficient ops.
let irradiance = (env * vis * cosine).sample(normal.into());
// Shadows included. Penumbra from Jet2. AO is vis.coeffs[0].
```

## Cost Analysis

### Staying in Basis (Clebsch-Gordan multiplication)

| Order | Coefficients | Ops | Use Case |
|-------|-------------|-----|----------|
| 2 | 9 | ~200 | Diffuse irradiance, soft shadows |
| 3 | 16 | ~1000 | Glossy, area lights |
| 10+ | 100+ | Millions | Don't. Wrong basis. |

For low-frequency (irradiance, ambient, soft shadows): ~200 FMA ops. Fits in L1. Nanoseconds.

### When to Switch Basis Entirely

If you need high frequency (sharp reflections, hard shadows): SH is wrong.

Switch to:
- Wavelets (sharp edges, LOD)
- Spatial hashing (point queries, InstantNGP style)
- Direct evaluation (when you only need one direction)

## Hierarchy

| Level | What | When |
|-------|------|------|
| Representation theory | Which symmetry → which basis | 1800s |
| Clebsch-Gordan tables | Precomputed multiplication tensors | Compile time |
| Basis projection | Bake field into coefficients | Load / scene change |
| Sampling | Dot product | Per pixel, nanoseconds |

## Relationship to Jet2

| System | Handles |
|--------|---------|
| Kernel Algebra | Choosing representation (which basis) |
| Jet2 | Differentiation (gradients for free) |

They're orthogonal. Kernel algebra picks where fields live. Jet2 makes differentiation implicit throughout.

Soft shadows aren't a blur pass—they're what you get when visibility has spatial variation (Jet2 gradient) and you're working in a bandwidth-limited basis (SH with finite bands).

## Connection to Attention / ML

The same integral structure appears in attention:

```
Attention(Q,K,V) = softmax(Q·K^T) · V
```

This is `∫ Kernel(query, source) · Value(source)`.

The embedding matrices (Q, K projections) are **learned change-of-basis** that make the kernel factorizable. Neural nets do numerically what we do analytically—find the representation where the operation is cheap.

See `pixelflow-ml` for the implementation of `HarmonicAttention` which directly uses SH coefficients as the feature map for linear attention.

## Implementation Files

- `pixelflow-core/src/combinators/kernel.rs` - Core kernel algebra types
- `pixelflow-core/src/combinators/spherical.rs` - Spherical harmonics implementation
- `pixelflow-ml/src/lib.rs` - Harmonic attention (SH + ML unification)

## Summary

1. **Symmetry → Basis**: A finite catalog. Translation = Fourier. Rotation = SH. Look it up.
2. **Multiplication in basis**: Precomputed sparse tensor. ~200 ops for diffuse lighting.
3. **Frame = gauge choice**: Where to put the observer. Kernel knows its natural frame.
4. **Jet2 for gradients**: Penumbra and soft edges are gradient magnitude, not blur.
5. **Effects emerge**: Shadows = light × visibility. AO = DC term of visibility. Nothing is separate.

Declare your fields. Declare their symmetry. Multiply. Sample. Done.
