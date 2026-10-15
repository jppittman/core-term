# A schedule is a factorization of the domain

**Status:** `Proposed`. This is the "next note" that
`2026-10-15-kernels-evaluate-kernels.md` §3 promised: the premises there are
rulings; the framing here is offered for one. Written against
`docs/DENOTATION.md` §7, which states the claim; this note works it out.
**Verified against:** `2a99d933594230a6d9cf94c21d78e5dd134b4785`.

## 0. The claim

A lattice is a product. A backend chooses how to factor it, and that choice
is the schedule:

```
L ≅ Rows × Batches × Lanes          CPU SIMD
L ≅ Threads                         GPU, one pixel per thread
L ≅ Tiles × Vector                  TPU
```

`apply(f, s)` (DENOTATION §5) evaluates each sub-arrow of `f` at the
coarsest projection it factors through. Everything the schedule does is a
case of that one sentence:

| schedule concept | the factorization fact |
|---|---|
| hoisting a value out of the lane loop | it factors through `Rows × Batches → 1` on `Lanes` |
| hoisting out of the row loop | it factors through the projection onto `Rows` |
| computing a value once per frame | it factors through `L → 1`: it is a constant, or a uniform |
| broadcast vs contiguous load vs gather for a discrete kernel | its index factors through `1`, through the lattice position plus a translation, or through nothing |
| the loop nest | the order the factors are nested in |
| a stripe | a sub-range of the `Rows` factor |
| the tail | the remainder factor of `Batches × Lanes` when the width is not a multiple |
| tiling | refactoring `Rows × Cols` as `(Rows/t × Cols/t) × (t × t)` |
| materialization | tabulating a sub-arrow over the sub-product it factors through |

Variance is the compiler's name for "which projection a term factors
through" (DENOTATION §7, `b ∉ var(v) ⟹ ⟦v⟧ is constant along b`), so the
schedule reads a fact the compiler already computes and adds no analysis of
its own. That is why JP's "variance is factorization" and "I'd literally
insert folds to iterate over the lattice" are the same thought: the folds
are the factors, written as loops.

## 1. What is algorithm, what is schedule, what is backend

The three columns are Halide's, with one difference: the schedule is not a
language the author writes. The author writes the algorithm; each backend
*is* a schedule, chosen once per target.

| concept | algorithm (the kernel's) | schedule (per backend) | backend mechanics | lives today |
|---|---|---|---|---|
| the lattice's extent | yes: part of the program and the compile key | | | `pixelflow_core::Lattice`, `LatticeShape` |
| rows, batches, lanes | | yes: the CPU's factorization of `L` | | `pixelflow-ir/src/passes/lattice.rs` (`collapse`, `pack`) — **wrong crate**; moves to the CPU module's `Nest` |
| a fold (`sum_over`) | yes: per point on every target | | | `ExprNode::Reduce`; a surviving fold is a loop (`program/scopes.rs`) |
| the fold's loop vs copies | | | chosen by extraction's cost, target-agnostic | `fold_rules.rs` (peel, halve) |
| `If`'s one shape | yes: two guarded own-blocks and a select | | the guard's instructions | `program/ownership.rs`, `program/layout.rs`, `emit/select.rs` |
| hoisting | | yes: the apply rule | | `program/scopes.rs` (`Nest`) |
| a discrete kernel's realization (broadcast / load / gather) | | yes: read off its index's variance | the instruction | `emit/select.rs` |
| materialization (which sub-kernels to tabulate) | | yes, per backend | | nothing chooses it today; `bake` is sugar |
| `Write`/`Seq` | | | CPU mechanics for performing an application | `pixelflow-ir/src/passes/lattice.rs` — **wrong crate**, same move |
| register allocation, rematerialization, carries | | | CPU only; a GPU's vendor compiler allocates | `emit/regalloc/` |
| the uniform block and the `ctx` pointers | | | the ABI | `emit/`, `jit_cache.rs` |
| the ISA tier | | | a property of the process | `pixelflow-codegen/src/isa/` |
| extraction's cost table | | | the latency prior is a CPU table; a second backend brings its own | `CostModel::latency_prior()` |

Two rows are marked **wrong crate**. `collapse` and `pack` fold the CPU's
factors into the kernel inside `pixelflow-ir`, which is target-agnostic;
they are the CPU schedule and belong in the CPU module's first pass. That
move is the one code consequence this note asks for, and CLAUDE.md already
names it.

## 2. Each backend is a functor and a factorization

**CPU SIMD.** `L ≅ Rows × Batches × Lanes`, nested in that order, with
`Lanes` the ISA's width and `Batches × Lanes` the row's width plus a
remainder. The module is one function, `cpu::compile(kernel, lattice) →
code`, whose chain is Nest → Own → Layout → Select → Allocate → Assemble.
`Nest` is where `collapse` and `pack` go: it factors `L` and places every
term at its coarsest projection. The uniform block is the `L → 1` factor's
storage; a stripe is a sub-range of `Rows`.

**GPU.** `L ≅ Threads`; the lattice is the grid, and there is no factor the
backend owns below a thread. Every term uniform in `L` is a kernel argument,
every other term is per thread, and the module's chain stops at emitting the
vendor's input. `If`'s one shape stands: the guards are the hardware's
divergence handling, read per warp, and the select is the same select. The
backend does not estimate how often `m` is true any more than the CPU's
does.

**TPU** (speculative). `L ≅ Tiles × Vector`; terms uniform in `Vector`
hoist to the scalar unit, and a contraction over a shared index is the
matrix unit's work (§3).

The three agree on what a kernel means and differ only in the factorization
and the mechanics, which is what "each backend is one module exposing one
function with its own `Pass` chain" buys: nothing above the module knows a
row, a thread or a tile exists.

## 3. The linear operator, or why there is no matmul

JP, 2026-10-15: "Matmul is an implementation detail and we're a pull
renderer. I wanted to do ML in Hilbert space on kernels and let the
compiler worry about if or not it was a matmul."

Write it as the language does (DENOTATION §3):

```
A : Lattice(n, k) → Real        B : Lattice(k, m) → Real
C = λ(i, j). Σ_{d ∈ [0, k)} A(i, d) · B(d, j)   :  Lattice(n, m) → Real
```

`C` is a kernel. Its output lattice is `n × m`; its body is one fold over a
bound index `d` of a product of two applications. A pull renderer computes
each `C(i, j)` independently, and the naive reading of that is the triple
loop with no reuse. The factorization gives the reuse back without anyone
asking for it:

- `A(i, d)` does not depend on `j`: it factors through the projection onto
  `i`, so in a tile where `j` runs along `Lanes`, `A(i, d)` is computed (or
  loaded) once per `(i, d)` and broadcast across the lanes.
- `B(d, j)` does not depend on `i`: it factors through `j`, so it is a
  contiguous load along `Lanes` once per `d`, reused across every `i` in the
  tile.
- Refactoring `n × m` as `(n/t × m/t) × (t × t)` is the schedule's tiling;
  the inner `t × t` is where both hoists pay, and its size is the register
  budget's.

That is a blocked GEMM, and nothing in the kernel said so. What the
factorization does **not** reach is the bound index: splitting `d` into
panels (`Σ_{[0,k)} = Σ_{[0,p)} ⊕ Σ_{[p,k)}`) is a fold split, an e-graph
rule over the algorithm, not a projection of the domain. That is the one
place the schedule and the algorithm touch, and it is already where the
language puts it: extraction decides the fold's shape by cost (DENOTATION
§3), and the backend schedules whatever survives.

The Hilbert-space part has one honest limit. An inner product `⟨f, g⟩` on a
continuous domain is an integral, and the IR has none (deleted 2026-09-29,
and not coming back). So the inner products available are sums over
discrete kernels, which is the above, and closed forms over bases whose
inner product is known analytically, which is what `pixelflow-ml`'s
spherical-harmonic feature maps are. "ML on kernels" is then: a feature is
a discrete kernel, a linear operator is a kernel of kernels, and the
backend's factorization of its domain is the only sense in which a matmul
exists.

## 4. Module layout (the shape the code would take)

```
pixelflow-codegen/src/
  cpu/                      (kernel, lattice) → code
    nest.rs                 collapse + pack, moved from pixelflow-ir/src/passes/lattice.rs
    own.rs  layout.rs       program/ownership.rs, program/layout.rs
    select.rs allocate.rs assemble.rs
  gpu/                      later; same signature
  isa/                      the tier, as today
```

`pixelflow-ir` loses `passes::lattice`, `Write` and `Seq`, and with them the
last place a target-agnostic crate names a lane. `pixelflow-core::Lattice::
collapse` calls the one backend function. Nothing in the language changes.

## 5. Not decided here

- **Is time a factor?** A frame sequence would be `Frames × L`, with the
  uniform block the `Frames` projection's storage. The denotation already
  says a uniform factors through `L → 1`; whether to write `Frames` as an
  axis the schedule owns or leave it as "per call" is a naming choice with
  no code behind it yet.
- **May the schedule split a fold's range?** §3 says no: a fold is
  algorithm and only extraction reshapes it. A backend that wants panels
  would ask extraction for them through cost, not take them itself. This is
  the premise "a kernel's own folds are per pixel on every target" and may
  want restating once a contraction is actually scheduled.
- **The cost table per backend.** The latency prior is a CPU table.
  Extraction choosing among equal forms for a GPU needs that backend's
  table, which makes the cost model a component of the functor like the
  encoder is.

Decided in: `2026-10-15-kernels-evaluate-kernels.md` §3 (the premises),
`docs/DENOTATION.md` §7 (the claim), `designs/lattice-scheduling-types.md`
(the April form of the same idea, with the mechanism it got wrong).
