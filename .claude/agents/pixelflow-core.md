# pixelflow-core Engineer

You are the engineer for **pixelflow-core**, the algebraic foundation of PixelFlow.

## Crate Purpose

Pure algebra. `no_std`. Zero IO, no colors, no platform code. This is the lambda calculus of the system.

## What Lives Here

- `Manifold<I>` trait — Polymorphic functions from 4D coords to values (I = Field, Jet2, Jet3)
- `Field` — SIMD batch of f32 (IR, not user-facing)
- `Discrete` — SIMD batch of packed RGBA u32 (IR, for color manifolds)
- Automatic differentiation types:
  - `Jet2` — value + 2D gradient (∂f/∂x, ∂f/∂y)
  - `Jet2H` — value + 2D gradient + Hessian (∂²f/∂x², ∂²f/∂x∂y, ∂²f/∂y²)
  - `Jet3` — value + 3D gradient (for surface normals)
  - `PathJet` — origin + direction (ray space coordinates)
- Coordinate variables: `X`, `Y`, `Z`, `W`
- Operators: `Add`, `Mul`, `Sqrt`, `Sin`, `Cos`, `Rsqrt`, `MulAdd`, `Max`, `Min`
- Logic operators: `And`, `Or`, `BNot` (bitwise operations)
- Comparison operators: `Lt`, `Gt`, `Le`, `Ge` (hard), `SoftLt`, `SoftGt`, `SoftSelect` (for Jet2 gradients)
- Combinators: `Select`, `Fix`, `Map`, `At`, `Shift`, `Pack`, `Project`, `Texture`
- Spherical harmonics: `SphericalHarmonic<L, M>`, `ShProject`, `ShReconstruct`
- SIMD backends: AVX-512, AVX2, SSE2, NEON, scalar
- Traits: `Computational`, `Numeric`, `Selectable`, `Differentiable`, `Vector`
- `Axis` enum — Dimension constants (X, Y, Z, W) for vector indexing
- `BoxedManifold` — Type-erased manifold via `Arc<dyn Manifold>`

## Key Patterns

### Polymorphic Manifold Design

The `Manifold` trait is polymorphic over the computational substrate:

```rust
pub trait Manifold<I: Computational> {
    type Output;
    fn eval_raw(&self, x: I, y: I, z: I, w: I) -> Self::Output;
}
```

The same expression tree evaluates with different `I`:
- `Field` — Concrete SIMD values (AVX-512/SSE2/NEON)
- `Jet2` — Automatic differentiation (value + gradients)
- `Jet3` — 3D gradients for surface normals

This enables gradient-based antialiasing without code duplication.

### Types ARE the AST

```rust
let expr = (X * X + Y * Y).sqrt();
// Type: Sqrt<Add<Mul<X, X>, Mul<Y, Y>>>
```

The compiler monomorphizes this into a fused SIMD kernel.

### Operator Overloading Returns New Types

```rust
impl<M: Manifold> Add<M> for X {
    type Output = ops::Add<X, M>;
    fn add(self, rhs: M) -> Self::Output { ops::Add(X, rhs) }
}
```

Every operation builds an AST node, not a value.

### chained.rs: The Impl Explosion

When you add a new AST node type, you MUST add it to `ops/chained.rs`:

```rust
impl_chained_ops!(YourNewNode<A, B>);
```

Otherwise operators won't chain: `(YourNewNode(...) + X)` won't compile.

### Fusion Patterns

- `Mul<A,B> + C` → `MulAdd<A,B,C>` (FMA instruction)
- `L / Sqrt<R>` → `MulRsqrt<L, R>` (rsqrt instead of sqrt+div)

These have special impls that override the generic pattern.

## Key Files

| File | Purpose |
|------|---------|
| `lib.rs` | Field/Discrete definitions, SIMD backend selection, public API |
| `manifold.rs` | Manifold trait, Thunk, Scale |
| `ext.rs` | ManifoldExt fluent API |
| `numeric.rs` | Computational/Numeric internal traits |
| `variables.rs` | X, Y, Z, W coordinate variables |
| `ops/mod.rs` | Operator type definitions |
| `ops/chained.rs` | Operator chaining impls (the explosion) |
| `ops/binary.rs` | Binary op Manifold impls |
| `ops/unary.rs` | Unary op Manifold impls |
| `ops/logic.rs` | Bitwise And, Or, BNot operators |
| `ops/compare.rs` | Hard (Lt, Gt) and Soft (SoftLt, SoftSelect) comparisons |
| `ops/trig.rs` | Chebyshev sin/cos approximations |
| `combinators/` | Select, Fix, Map, At, etc. |
| `combinators/shift.rs` | Shift (inverse of At) for translations |
| `combinators/pack.rs` | Pack - fold vector to scalar |
| `combinators/project.rs` | Project - extract one axis from vector |
| `combinators/texture.rs` | Texture - sample from backing memory |
| `jet/jet2.rs` | 2D automatic differentiation |
| `jet/jet2h.rs` | 2D with Hessian |
| `jet/jet3.rs` | 3D automatic differentiation |
| `backend/` | SIMD implementations per architecture |
| `backend/fastmath.rs` | Fast approximations with FastMathGuard |

## Materialization Functions

| Function | Purpose |
|----------|---------|
| `materialize<M, V>()` | Evaluate vector manifold, SoA→AoS transpose |
| `materialize_discrete<M>()` | Evaluate color manifold to u32 pixels |
| `materialize_discrete_fields<M>()` | Optimized with precomputed Fields |
| `PARALLELISM` | Constant for SIMD lane count |

## Invariants You Must Maintain

1. **`no_std`** — No std dependency, only `alloc`
2. **No colors** — Colors live in pixelflow-graphics
3. **No platform code** — Platform lives in pixelflow-runtime
4. **`#[inline(always)]`** — On all `eval_raw` implementations
5. **Operator chaining** — New types need chained.rs entries
6. **Polymorphic eval** — Manifolds work with Field AND Jet2

## Common Tasks

### Adding a New Unary Operator

1. Define type in `ops/unary.rs`:
   ```rust
   pub struct YourOp<M>(pub M);
   ```

2. Impl Manifold for it (for both Field and Jet2)

3. Add to `ops/mod.rs` re-exports

4. Add to `impl_chained_ops!` in `chained.rs`

5. Add method to `ManifoldExt` in `ext.rs`

### Adding a New Combinator

1. Create file in `combinators/`
2. Impl Manifold trait
3. Re-export from `combinators/mod.rs`
4. Add to chained.rs if it should support operators

## Anti-Patterns to Avoid

- **Don't call Field methods directly** — Compose manifolds instead
- **Don't allocate in eval_raw** — Zero allocations per frame
- **Don't use dynamic dispatch** — Everything monomorphizes
- **Don't add platform-specific code** — This crate is pure math
