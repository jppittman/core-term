# The graph differentiates; legalize is the fallback

**Date:** 2026-09-09
**Status:** Findings. No code. Three defects in the runtime pipeline's
ordering, and a criterion for when `Gather` is a leaf and when it is a
scheduling decision wearing one.
**Author:** JP (direction and both corrections), Claude (draft)
**Bears on:** S1b of
[2026-09-09-glyph-as-a-fold-execution.md](2026-09-09-glyph-as-a-fold-execution.md),
which cannot compile as planned until §3 lands.

---

## The intent

The e-graph differentiates. `Dwrt` is a first-class node in it
(`egraph/ops.rs:208` resolves it, `egraph/derivative.rs` holds the rules,
`egraph/rewrite.rs:153` expands it one chain-rule step per rewrite), and
`egraph/cost.rs` prices a *surviving* `Dwrt` at `usize::MAX/4` precisely so
extraction can never select an unlowered derivative. That pricing is only
coherent if the node is expected to live in the graph and be rewritten away
there.

`passes::legalize`'s `lower_dwrt` is the **fallback** — for arenas that never
reach the e-graph, and for whatever the rules do not cover.

The runtime pipeline has it the other way round.

## 1. The fallback is promoted to primary

`pixelflow-search/src/runtime.rs:171`:

```rust
pipeline![LowerDwrt, ExpandReduce, Saturate::runtime(shape)]
```

`LowerDwrt` runs *before* saturation, so no `Dwrt` node ever reaches the
rules that exist to rewrite it. The comment justifies it: differentiation
manufactures constants, and `ConstantFold` can only cascade over constants
that exist by the time saturation runs.

That argument holds against lowering *after* the e-graph. It does not hold
against differentiating *inside* it, which is what the rules are for — the
constants are then manufactured mid-saturation, where folding cascades over
them anyway.

## 2. `ExpandReduce` runs early because it is forced, not because it is right

`egraph/ops.rs:232`:

```rust
OpKind::Buffer | OpKind::Gather | OpKind::RawGather | OpKind::Reduce | OpKind::Uniform => None
```

`op_from_kind` returns `None` for `Reduce`. Unlike `Gather` — which
`Vocabulary::resolve` special-cases to `Some(&Gather)`, opaque to every rule
but present as structure, so hash-consing still sees it — `Reduce` has no
such case. Saturate before unrolling and the e-graph declines the whole
arena, which for a post-S1a glyph means compiling with no CSE and no fusion.

So the order is forced. The stated benefit — "what saturation sees is
binder-free arithmetic it can CSE and fold across the unrolled terms" — is
real but small, since the unrolled copies differ only in a gather's row
index, and it is a rationalisation of a constraint rather than a statement
of it.

**The cost is the one measured today.** Unrolling first means saturation
sees `N` copies of one body. A glyph carries one piece per row, so a
200-piece glyph pays ~200× the e-graph work on structurally identical terms.
`loop_blinn_winding` went 277s → 750s and `kernel_glyph_optimize` sits at
907s once S1a introduced a reduce; that is the mechanism, and it was
mis-attributed to "saturation cost" at the time.

The fix is to give `Reduce` what `Gather` has: representable as structure,
nameable by no template. The body is then one e-class, optimized once, and
`ExpandReduce` unrolls code that is already optimized. No rule can match the
node, so nothing can move across the binder — the same argument that makes
`Gather` sound today.

## 3. `Dwrt` over a `Gather` has no rule on either side

`passes.rs:626` and `:815`:

```rust
ExprNode::Buffer(_) => Err("lower_dwrt: cannot differentiate a bound-memory read"),
OpKind::RawGather  => Err("lower_dwrt: cannot differentiate a bound-memory read"),
```

The fallback **refuses categorically** — bound-memory reads, integer/bit
ops, reductions. And the rules do not cover it either: `derivative.rs` names
no `Gather` or `Buffer`, so the rewrite never fires, the `Dwrt` survives,
extraction cannot afford it, and `arena_to_schedule` panics on one that
reaches codegen. An error on one path, a panic on the other.

**This blocks S1b outright.** The distance fold is
`in_pixels(value, scale) = value / ‖∇scale‖`, and S1b's whole content is
that `scale` is built from coefficients read out of the piece table. That
puts a `Gather` under a `Dwrt`, and nothing can lower it.

The answer is trivially zero: the index is `(constant_column, binder)` and
mentions no coordinate, so the read is constant in X and Y. Nothing checks.
The rule to add is *a gather whose index does not mention a coordinate
differentiates to zero*, and per §1 it belongs in the graph, not as a fourth
special case in `lower_dwrt`.

## 4. What `Gather` is, and the criterion for when it is cruft

Not cruft in principle: `Lattice`/`DiscreteManifold` are a representable
functor whose law is `index(collapse(f)) = f`, `collapse` is the introducer,
and `Gather` is `index`. A buffer that can be written and never read is not
a manifold.

Two things have accreted onto it.

**Its semantics are the hardware's.** `Gather` means floor-then-clamp-to-edge
(`passes.rs:261`, `eval.rs:1758`). The law says nothing about what happens
off the lattice; the instruction does, so every caller inherits the
instruction's answer. `fonts/cache.rs` masks the result to the glyph's
extent for exactly this reason — a clamped read would smear a descender's
boundary coverage out to infinity — which is a caller paying for an
addressing mode it did not choose.

**It is doing two jobs.** The test that separates them: *could the program
have computed this content itself?*

| | example | verdict |
|---|---|---|
| **No** | terminal cell contents, an image, the bytes of a font file | a genuine leaf; binding is real and `Gather` is the right eliminator |
| **Yes** | the glyph cache, the atlas, S1a's piece table | a **materialization decision** written into the expression |

The second kind is the same move as `Union`. `Union` is the host encoding
*demand* in the domain; `Gather`-as-memoization is the host encoding
*materialization* in the expression. Both are scheduling decisions made by
hand because the compiler does not make them, and both cost identically:
opaque to the e-graph, no rewrite may name them, no derivative rule fires,
`MAX_BOUND_BUFFERS = 4`, indices exact only below 2²⁴ because they travel
through `f32`, plus binding tables, identities, and the API surface each one
forces on its callers.

That reading sharpens §3. `Dwrt` over a *genuine* leaf needs the rule.
`Dwrt` over a *materialized* `.at()` needed nothing — the derivative of the
kernel that got tabulated was available all along, and materializing before
differentiating destroyed it.

**The way out is not a vector of uniforms.** It is that *materialize here*
should be a schedule annotation the compiler derives, exactly as demand
should be, leaving `Gather` in the language only for content that came from
outside the program.

## Order

3 → 2 → 1. Each is independently landable with a test that proves it.

- **§3** is the smallest and is what S1b is currently blocked on.
- **§2** unblocks moving `ExpandReduce` after saturation, and should be
  measured by glyph compile time, which is the number §2 predicts falls by
  roughly the piece count.
- **§1** is then deleting `LowerDwrt` from the runtime pipeline and letting
  the rules do the job they were built for, with `legalize` still present as
  the fallback it was meant to be.

§4 is not scheduled and is not a refactor to start; it is the criterion to
apply when the next buffer is proposed.
