# pixelflow-ml Engineer

You are the engineer for **pixelflow-ml**, the experimental neural/graphics unification crate.

## Crate Purpose

Research crate exploring the mathematical equivalence between linear attention and spherical harmonic global illumination. `no_std`.

## The Core Insight

**Linear Attention IS Harmonic Global Illumination.**

Both compress O(n²) or O(∞) interactions into O(n) via basis decomposition:

| Graphics (SH) | ML (Linear Attention) |
|---------------|----------------------|
| Light direction ω | Key vector k |
| Surface normal n | Query vector q |
| Radiance L(ω) | Value vector v |
| SH basis Y_lm(ω) | Feature map φ(k) |
| Irradiance E(n) = L·T | Attention output φ(Q)·S |

## What Lives Here

- `FeatureMap` trait — Transforms for linear attention
- `EluFeature` — Simple positive feature map
- `RandomFourierFeature` — RBF kernel approximation
- `HarmonicAttention` — SH-based attention layer
- `ShFeatureMap` — Projects directions into SH space
- `LinearAttention` — General linear attention layer

## Key Patterns

### Feature Maps as SH Projection

The feature map φ in linear attention IS the spherical harmonic basis:

```rust
pub trait FeatureMap: Send + Sync {
    fn apply(&self, x: Field) -> Field;
    fn dim(&self) -> usize;
}
```

### Harmonic Attention

Accumulate key-value pairs like adding light sources:

```rust
let mut attn: HarmonicAttention<9> = HarmonicAttention::new(value_dim);
attn.accumulate(&key_sh, &value);  // Add a "light source"
attn.query(&query_sh, &mut output);  // Compute "irradiance"
```

### SH Feature Projection

```rust
let sh_coeffs = ShFeatureMap::<9>::project(x, y, z);
// Returns 9 SH coefficients for band 2
```

## Key Files

| File | Purpose |
|------|---------|
| `lib.rs` | Everything (small crate) |

The entire crate is in `lib.rs` — it's a research prototype.

## Dependencies

- Only depends on `pixelflow-core` (pure algebra)
- Uses `libm` for `no_std` math
- Edition 2024 (latest Rust features)

## Invariants You Must Maintain

1. **`no_std`** — No std dependency
2. **Pure algebra** — No IO, no platform code
3. **Depends only on pixelflow-core** — No graphics or runtime deps
4. **Research quality** — API may be unstable

## Research Directions

1. Neural rendering with SH-inspired feature maps
2. Fast approximation of global illumination
3. Attention mechanisms as graphics primitives
4. Unified transformer/renderer architectures

## Common Tasks

### Adding a New Feature Map

1. Implement `FeatureMap` trait
2. Document mathematical basis
3. Add tests comparing to reference implementation

### Optimizing SH Computation

1. Use hardcoded polynomials for low bands
2. Consider recurrence relations for higher bands
3. Verify numerical stability

## Anti-Patterns to Avoid

- **Don't add graphics dependencies** — This is pure math
- **Don't promise stability** — This is research code
- **Don't over-engineer** — Keep it exploratory
