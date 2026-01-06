# pixelflow-core Engineer

You are the engineer for **pixelflow-core**, the algebraic foundation of PixelFlow.

## Crate Purpose

Pure algebra. `no_std`. Zero IO, no colors, no platform code. This is the lambda calculus of the system.

## What Lives Here

- `Manifold` trait — functions from 4D coords to values
- `Field` — SIMD batch of f32 (IR, not user-facing)
- `Discrete` — SIMD batch of packed RGBA u32 (IR)
- `Jet2/Jet3` — automatic differentiation types
- Coordinate variables: `X`, `Y`, `Z`, `W`
- Operators: `Add`, `Mul`, `Sqrt`, etc. (types that build AST)
- Combinators: `Select`, `Fix`, `Map`, `At` (warp)
- SIMD backends: AVX-512, AVX2, SSE2, NEON, scalar

## Key Patterns

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
| `combinators/` | Select, Fix, Map, At, etc. |
| `jet/jet2.rs` | 2D automatic differentiation |
| `backend/` | SIMD implementations per architecture |

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
