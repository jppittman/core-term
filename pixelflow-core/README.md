# PixelFlow Core

`pixelflow-core` is the `no_std` algebraic substrate for PixelFlow. It provides SIMD fields,
coordinate domains, the `Manifold` evaluation trait, automatic-differentiation and composition
utilities, and the lattice boundary that turns functions into finite data.

PixelFlow's current programming direction is arena-backed `Kernel` values. The older
type-level manifold combinators remain implemented as a compatibility and fallback layer, but
they are not the intended public authoring model for new kernels.

## The current split

There are two related representations in the workspace:

| Representation | Role today |
|---|---|
| `Kernel` (`pixelflow-ir`, re-exported here) | JIT-first program value: an `ExprArena` fragment composed before compilation |
| `Manifold<P>` | Evaluation substrate for SIMD fields, cached data, opaque Rust values, and code not yet migrated from combinators |

The distinction matters. A `Kernel` is PixelFlow-owned syntax with operations such as `at`,
`select`, symbolic derivatives, and bounded reductions. A `Manifold` is a Rust trait that can
evaluate a value over a domain. Baking a `Kernel` produces a discrete manifold; not every
manifold needs or has an arena representation.

## `Kernel`: the JIT-first value

Use `kernel_value!` for new arena-native expressions:

```rust
use pixelflow_compiler::kernel_value;
use pixelflow_core::{Kernel, Lattice};

let circle = kernel_value!(|cx: f32, cy: f32, radius: f32| {
    let dx = X - cx;
    let dy = Y - cy;
    (dx * dx + dy * dy).sqrt() - radius
});

let sdf: Kernel = circle(32.0, 32.0, 20.0);
let coverage = sdf
    .neg()
    .add(&Kernel::constant(0.5))
    .clamp(&Kernel::constant(0.0), &Kernel::constant(1.0));

let pixels = Lattice::frame(64, 64, 0.0).bake(&coverage);
assert_eq!(pixels.buffer().len(), 64 * 64);
```

`kernel_value!` parses and checks the DSL, runs e-graph optimization, and returns an uncompiled
arena fragment. Scalar parameters become constants in the fragment. Composition remains at
construction time; `Lattice::bake` JIT-compiles the fused root through the global compile cache
and tabulates it.

The `Kernel` surface currently includes:

- arithmetic, transcendental functions, comparisons, masks, and `select`;
- coordinate substitution with `at`;
- symbolic derivatives (`dwrt`, `dx`, and `dy`), resolved during lowering;
- variadic sums and bounded monoid reductions (`sum_over`, `product_over`, `min_over`, and
  related operations).

Bounded reductions have static extents. General unbounded recursion is intentionally outside
the language. Typed discrete fields and a cost-semiring interpretation are described by the
current design documents but are not all implemented yet.

## `Field`

`Field` is the SIMD computational atom. Arithmetic and comparisons operate lane-wise, and the
selected backend determines its lane count. Callers normally obtain fields by evaluating a
`Manifold` or materializing a kernel rather than depending on a particular SIMD width.

```rust
use pixelflow_core::{Field, Manifold};

fn sample<M>(m: &M, x: Field, y: Field) -> Field
where
    M: Manifold<(Field, Field, Field, Field), Output = Field>,
{
    m.eval((x, y, Field::from(0.0), Field::from(0.0)))
}
```

## `Manifold`

The current trait is generic over the entire input point, not over four positional arguments:

```rust
pub trait Manifold<P = (Field, Field, Field, Field)>: Send + Sync {
    type Output;
    fn eval(&self, p: P) -> Self::Output;
}
```

Different domains can carry plain fields, jets, bound context, or other coordinate structures.
`ManifoldCompat::eval_raw(x, y, z, w)` remains as an adapter for older four-argument callers.

The existing coordinate values (`X`, `Y`, `Z`, `W`), operators, `At`, `Select`, and related
combinators still build Rust type trees which monomorphize into SIMD evaluation. They are used
inside the repository and remain useful for opaque or cached Rust-backed manifolds. New
language-level work should prefer `Kernel` so the program is available to PixelFlow's own
optimizer and backends rather than encoded in Rust's type system.

## Lattices and materialization

A `Lattice` is a finite box over the four coordinate axes. It is the explicit boundary where
a pure function becomes stored samples.

```rust
use pixelflow_core::Lattice;

let frame = Lattice::frame(800, 600, 0.0);
let scanline = Lattice::scanline(800, 20.0, 0.0, 0.0);
let features = Lattice::index(128);
```

- `collapse(&manifold)` evaluates an existing `Manifold` over the domain.
- `bake(&kernel)` compiles an arena-backed `Kernel` and tabulates it.
- The resulting `DiscreteManifold` owns an `f32` buffer and is itself sampleable as a
  manifold.

The longer-term plan replaces the present lattice terminology with typed discrete fields while
preserving this explicit reification boundary. See
[`KERNELS_AND_LATTICES.md`](../docs/designs/KERNELS_AND_LATTICES.md) and
[`2026-07-24-totality-and-the-cost-model.md`](../docs/designs/2026-07-24-totality-and-the-cost-model.md).

## Automatic differentiation

Two mechanisms coexist during the migration:

- The combinator layer can evaluate compatible expressions over `Jet2` and `Jet3` domains.
- Arena-native kernels contain `Dwrt` nodes that are differentiated symbolically before
  backend emission.

New JIT-first consumers use symbolic derivatives. The font pipeline, for example, constructs
antialiased coverage ramps from `DX`/`DY` in the DSL and resolves those derivatives when the
glyph kernel is baked. Jets remain relevant to older manifold consumers, including parts of
the 3D graphics code.

## Compiler transition

The related macros currently have different output boundaries:

- `kernel_value!` returns an uncompiled arena-backed `Kernel` and is the preferred JIT-first
  composition surface.
- `kernel_jit!` returns a `Kernel` too, optimized through the e-graph; it is evaluated the
  same way, by baking it over a lattice.
- `kernel!` still emits type-level combinators. It is the legacy tier; the parity suite
  (`pixelflow-compiler/tests/kernel_routing_parity.rs`) checks both tiers against scalar
  ground truth until it is retired.

The plan is to converge these paths on one arena language and retire the combinator emitter
after remaining parity and consumer migrations are complete. Progress and known gaps are
tracked in [`2026-07-20-kernel-unification.md`](../docs/plans/2026-07-20-kernel-unification.md).

## Constraints

- `pixelflow-core` is `no_std`; allocation-dependent facilities use `alloc` internally.
- Consumers should not manipulate `ExprArena` directly. Compose `Kernel` values and compile at
  a materialization boundary.
- Do not silently fall back when an arena kernel cannot compile. `Lattice::bake` fails loudly.
- Do not assume a fixed SIMD lane count in consumer code.

## Test and benchmark

```bash
cargo test -p pixelflow-core
cargo bench -p pixelflow-core
```

## License

[Apache License 2.0](../LICENSE.md)
