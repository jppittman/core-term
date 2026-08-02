# Plan: `pixelflow-ir` layering — subtract, then decide

**Status:** Proposed
**Date:** 2026-08-02
**Supersedes nothing.** Complements `docs/plans/2026-07-20-kernel-unification.md` (P6/P7+).

---

## The ask

> "I want to deduplicate the node representations. Pixelflow-ir should just be the ir I
> think? And I think the lowering pass belongs in compiler? The problem is too much shit
> is in the IR." … "the other crates operate on endomorphisms. They 'digest' the IR as it
> were. Maybe there's a marking for primary/secondary nodes (nodes that can be emitted and
> nodes that *must* be lowered before emission)."

Four claims. The evidence says **two are right, one is right about the wrong target, and
one is structurally impossible.** Details below, then the plan.

---

## What the evidence says about each claim

### ✅ "The optimizer takes IR → IR" — VERIFIED, and stronger than stated

The four passes in `pixelflow-ir/src/backend/emit/lowering.rs` each ship in two forms:
an accumulator form (`&mut ExprArena`) and an `_owned` form that is a genuine
endomorphism.

| Pass | endomorphic form | shape |
|---|---|---|
| `lower_dwrt` | `lowering.rs:467` | Kleisli arrow over `Result` |
| `expand_reduce` | `lowering.rs:286` | pure `(Arena, Root) → (Arena, Root)` |
| `expand_gather` | `lowering.rs:208` | pure |
| `expand_transcendentals` | `lowering.rs:167` | pure |

The `&mut` form is not destructive — the arena is append-only (`arena.rs:609-610`), so
`(&mut Arena, Root) → Root` is the state-passing spelling of the same thing. Each pass has
an identity fast-path when inapplicable (`lowering.rs:173-179, 209-215, 287-293, 471-481`),
making it **idempotent and free when it has nothing to do** — exactly the property a pass
manager needs to run everything unconditionally.

All four are built on **one shared skeleton** with a per-pass hook: `rebuild_arena`
(`lowering.rs:65`) / `try_rebuild_arena` (`lowering.rs:80`). That hook's type
(`lowering.rs:82`) **is already the `Pass` trait, spelled as a closure parameter.**

They are also **100% target-blind** — the only `cfg` in the entire file is `#[cfg(test)]`
at `:1176`, and the import list is `crate::{arena, kind, variance}` + `alloc`. The
target-abstraction trap I was worried about does not exist at this tier.

**What's missing is a *reified* pipeline.** The order is hand-copied at three call sites
(`emit/mod.rs:427-430`, `:2507-2510`, `:3750-3753`) and **three live paths bypass it
entirely** — see Phase 4.

### ✅ "There's a marking for primary/secondary nodes" — VERIFIED, and it's already total

The partition exists and is **provably exhaustive over the enum**:

| Category | Count | Where it lives today |
|---|---|---|
| **Legal** (backend must emit) | 30 | `backend/emit/coverage.rs:39-77` — **test-only infrastructure** |
| **Expand** (must be lowered first) | 16 | four independent `matches!` in `lowering.rs:34-49, 52-54, 197-201, 211-213` |
| **Structural** (never a `ScheduledOp`) | 4 | prose, in `coverage.rs:12-34`'s doc comment |

30 + 16 + 4 = 50 = the whole enum. This is LLVM's `setOperationAction` table, already
computed, just scattered across a test-only const, four predicates, and four backend
`match` arms. **Making it explicit is mostly promotion, not design.**

What is genuinely absent is the **target axis** — currently `#[cfg]` cascades and
per-emitter `match` arms, with no data anywhere. See Phase 6.

### ⚠️ "Too much shit is in the IR" — RIGHT, WRONG TARGET

The misfiled mass is not the emitter. It is **bucket E: a runtime SIMD intrinsics library,
4,061 lines (22% of the crate), that is not IR *or* codegen**:

| File | lines |
|---|---|
| `backend/x86.rs` | 2,172 |
| `backend/arm.rs` | 749 |
| `backend/mod.rs` | 340 |
| `backend/compounds.rs` | 273 |
| `backend/primitives.rs` | 181 |
| `backend/fastmath.rs` | 159 |

Zero coupling to the IR in either direction, verified:

```
grep 'crate::arena|crate::kind|ExprArena|OpKind' backend/{mod,x86,arm,primitives,compounds,fastmath}.rs  → 0 hits
grep 'SimdOps|Primitives|Compounds|backend::x86|backend::arm' backend/emit/                              → 0 hits
```

Its only consumer is `pixelflow-core`, through an 8-line blanket re-export
(`pixelflow-core/src/backend/mod.rs:8`). It is the representation of `Field`. It belongs
with `Field`. **Any "pixelflow-ir should just be the IR" that leaves 4,061 lines of
NEON/AVX intrinsics in the crate has not achieved the goal.**

### ❌ "The lowering pass belongs in compiler" — STRUCTURALLY IMPOSSIBLE

`pixelflow-compiler/Cargo.toml:10` is `proc-macro = true`. Verified by compiling a minimal
repro:

```
error: `proc-macro` crate types currently cannot export any items other than
functions tagged with `#[proc_macro]`, `#[proc_macro_derive]`, or `#[proc_macro_attribute]`
```

Nothing can `use` a function from it. And the runtime path genuinely needs these passes at
runtime — `Lattice::bake` JITs arenas built long after macro expansion. Worse, the
direction is a cycle: `pixelflow-compiler` already depends on `pixelflow-search`
(`Cargo.toml:26`), and `pixelflow-search/src/runtime.rs:106` calls `lower_dwrt_owned`.

**But the instinct is half right, and that half matters.** The passes are target-blind and
have no business under `backend/emit/`. Three external crates already reach *past* the
codegen namespace to get at them (`pixelflow-search/src/runtime.rs:106`,
`pixelflow-compiler/src/ir_bridge.rs:813`,
`pixelflow-graphics/tests/kernel_glyph_golden.rs:13`) — the tree telling you the file is
misfiled. The destination is `pixelflow-ir/src/passes/`, and it's a `git mv`.

### ❌ "Deduplicate the node representations" — the two node types cannot merge

`ExprNode` (`arena.rs:48-62`) is a fixed-arity enum over `ExprId` with a hard
`assert!(size_of::<ExprNode>() <= 16)`. `ENode` (`pixelflow-search/src/egraph/node.rs:24-34`)
is `Op { op: &'static dyn Op, children: Vec<EClassId> }` — a `Vec` (24 bytes) plus a fat
pointer (16 bytes), so it can never satisfy that assertion, and its children point at
**equivalence classes, not nodes**. That indirection *is* the e-graph. Making the e-graph
operate on `ExprArena` directly would delete the quotient, which is the whole point of the
data structure.

**So my earlier `Node<I>` suggestion was wrong** — `egg` gets away with it because its node
type carries a `Vec<Id>` unconditionally; this arena's 16-byte budget is load-bearing and
worth keeping.

What *is* genuinely duplicated, and worth killing:

1. **The chain rule, written twice.** `EGraph::build_derivative`
   (`pixelflow-search/src/egraph/graph.rs:1412+`) and `diff_node`
   (`lowering.rs:570+`) differ *only in the builder calls*:
   ```
   graph.rs:      un(self, &ops::Rsqrt, u);           op2(self, &ops::Mul, half, rs)
   lowering.rs:   arena.push_unary(OpKind::Rsqrt, a); arena.push_binary(OpKind::Mul, half, rs)
   ```
2. **Two parallel op vocabularies.** `pixelflow-ir/src/ops.rs:10` has a `define_op!` macro
   minting one ZST per op; `pixelflow-search/src/egraph/ops.rs` has *its own* `define_op!`
   minting a second family behind `&'static dyn Op`. They are bridged by `op_from_kind`
   (`ops.rs:150-206`) — a hand-maintained 50-arm table that is also a **third independent
   partition of `OpKind`**, whose `None` set matches neither the Legal set nor the Expand set.

Both die to one `ExprBuilder` trait. See Phase 5.

---

## Two real bugs found along the way — fix these first

### 🐛 `OpKind`'s index space is unsound

`kind.rs:11-118` defines **50 variants with sparse discriminants 0–52** (holes at 17, 31,
39; max is `Reduce = 52`). `COUNT = 50` (`kind.rs:122`) is the *cardinality*, not
`max + 1`. `index()` (`kind.rs:238`) returns the raw discriminant. Therefore:

- **`OpKind::from_index(17)` is UB** — `kind.rs:244-250` guards `idx >= COUNT` then
  `transmute`s, so 17 passes the guard and transmutes an invalid discriminant. The safety
  comment claims "contiguous 0..=40", which is wrong twice over.
- **`from_index(50..=52)` returns `None`** for the perfectly valid `Gather`/`RawGather`/`Reduce`.
- **`CostModel::cost(OpKind::Reduce)` is an out-of-bounds panic** —
  `pixelflow-search/src/egraph/cost.rs:216` does `self.costs[op.index()]` on a
  `[usize; 50]`, indexing at 52. Unreachable today only by luck: `arena_to_egraph` bails on
  any `Buffer` leaf first (`runtime.rs:305`), and every `Gather` has one.
- **`ALL_OPS` (`ops.rs:134-185`) is positional but carries sparse `INDEX` constants** —
  `define_op!(18, Sin, …)` sits at array position 17. Two "op-indexed" tables already
  disagree about what the index means.

**Any `[Action; N]` table keyed on opcode inherits all of this.** The legalization table in
Phase 6 is exactly such a table. Fix the key before building on it.

### 🐛 Split failure conventions, papered over by the test suite

| Convention | Sites |
|---|---|
| `Err(&'static str)` | `avx512.rs:349, 504, 536`; `avx2.rs:376, 401, 422`; `aarch64.rs:647, 664` |
| `panic!` | `x86_64.rs:611, 625, 737, 767`; `aarch64.rs:696, 740` |

**aarch64 is internally inconsistent** — unary/shift return `Err`, binary/ternary `panic!`.
The completeness test has to `catch_unwind` to paper over it (`emit/mod.rs:5753-5772`,
which says so explicitly). Given the fail-fast rule, nothing fails *silently* — but a
`panic!` inside a JIT emitter is the wrong shape for "this target does not implement this
op."

---

## Premise corrections (project memory and docs are stale)

- **SSE2 is not deprecated.** `X86Backend` is live and selected (`emit/mod.rs:2524-2527`),
  it is the **default build**, and `xtask/src/main.rs:439-443` lists it as ISA level 0.
  Anything sequenced on "SSE2 is going away" must be re-checked. *(Contradicts
  `jit-migration-decisions` memory — memory updated.)*
- **There is no interpreter fallback tier.** `pixelflow-core/src/lib.rs:309-318` is a
  `compile_error!`: "rendering goes through the JIT, which has no other targets and no
  interpreter fallback." `eval.rs` is gated `#[cfg(any(test, feature = "oracle"))]` and is
  an oracle, not a tier.
- **AVX-512 "Stage 2" is done.** `emit_shift_imm` exists (`avx512.rs:494`), `IAdd`/`BitAnd`/
  `BitOr` are in `emit_binary` (`:345-348`), `Rsqrt`/`Recip`/`TruncToInt`/`IntToFloat` in
  `emit_unary` (`:530-535`). The header comment at `avx512.rs:13-16` claiming otherwise is stale.
- **`lib.rs:18`'s `no_std` claim is already false.** `Cargo.toml:18-19` declares
  `libc = "0.2"` with default features, so the crate links `std` through libc even under
  `--no-default-features` — invisible to `cargo check`.

---

## The plan

Ordered so that **every phase is independently defensible and individually revertible.**
Each names what it **deletes**.

### Phase 0 — Subtract (~900 lines, zero risk)

All verified dead:

| Action | Evidence |
|---|---|
| Delete `pixelflow-ir/src/math.rs` (212 lines) | no `mod math` anywhere — not compiled; a third copy of the transcendental polynomials |
| Delete `backend/emit/patch.rs` (49 lines) | **no `mod patch;` anywhere** — not compiled, not lint-checked. Duplicates two `extern "C"` decls. Also the only file in the tree with a non-`unsafe extern "C"` block (`patch.rs:12`), which likely would not compile under edition 2024 if wired in |
| Delete `backend/primitives.rs` + `compounds.rs` (454 lines) | `grep 'impl Primitives'` → zero implementors; `compounds.rs:273` is a blanket impl over an empty set |
| Delete the `alloc` feature (`Cargo.toml:38-40`) | zero `cfg(feature = "alloc")` sites; `extern crate alloc` is unconditional at `lib.rs:20` |
| Delete the `serde` dep (`Cargo.toml:15`) | zero `serde` references in `src/` |
| Delete `pixelflow-runtime/Cargo.toml:21` | zero `pixelflow_ir` references in that crate |
| `libc = { version = "0.2", default-features = false }` | closes the silent `std` link |

Plus three stale comments corrected: `emit/mod.rs:227-231` (claims `Gather` is AVX-512-only;
all four backends implement it), `avx512.rs:13-16` (claims `ShiftImm` is rejected),
`docs/designs/2026-07-25-two-level-ir-and-backend-completeness.md` Part 2b (describes
emitters that have since been deleted).

**Deletes:** ~900 lines, 3 config entries, 3 false claims.

### Phase 1 — Fix the `OpKind` index space

Either make discriminants contiguous, or make `index()` a dense ordinal distinct from the
discriminant. Then fix `from_index`, `CostModel::cost`, and `ALL_OPS`'s positional/sparse
mismatch. Add a test asserting `from_index(i).index() == i` for all `i < COUNT` and that
`ALL_OPS[i].INDEX == i`.

**Deletes:** one UB site, one latent OOB panic, one wrong safety comment.

### Phase 2 — Bucket E → `pixelflow-core`

`git mv pixelflow-ir/src/backend/{mod,x86,arm,fastmath}.rs pixelflow-core/src/backend/`,
then delete the shim line `pixelflow-core/src/backend/mod.rs:8`. Every existing
`crate::backend::x86::F32x16` path in core keeps resolving.

Defensible with no reference to the rest of this plan: *a crate named "IR" should not
contain a runtime SIMD intrinsics library.* Leaves `pixelflow-ir/src/backend/` containing
only `emit/`, which makes the later structural question obvious.

**Deletes:** 3,600 lines from `pixelflow-ir` (22% → post-Phase-0 numbers), plus the
8-line re-export shim.

> ⚠️ Known pre-existing leak this move relocates but does not fix:
> `pixelflow-graphics/src/render/discrete.rs:4` does
> `use pixelflow_core::backend::{Backend, SimdOps, SimdU32Ops}` — SIMD through the public
> API, which CLAUDE.md forbids. Moving E puts the leak with its correct owner. **Do not fix
> it in this commit**; separate blast radius.

### Phase 3 — `lowering.rs` → `src/passes/`

Pure relocation: `git mv pixelflow-ir/src/backend/emit/lowering.rs pixelflow-ir/src/passes/mod.rs`
plus `use`-path fixes in three consumers. Verified safe — one `#[cfg(test)]`, imports are
`crate::{arena, kind, variance}` + `alloc`, and `pub mod lowering;` (`emit/mod.rs:38`) is not
cfg-gated.

> ⚠️ Constraint: `eval.rs:53-54` calls `expand_transcendentals_owned` **to define what
> `sin` means** — `eval.rs:40-52` explains why ("the interpreter and the JIT cannot
> disagree about `sin`"), and `kind.rs:649-656` makes `eval_unary` return `None` for every
> transcendental precisely because lowering owns that meaning. The expansion is the
> *denotation* of the opcode, not an optimization over it. `passes` must stay beside
> `eval`, never above it.

**Deletes:** the worst filing error in the tree, and three external crates' reach through
`backend::`.

### Phase 4 — Reify the pipeline

`try_rebuild_arena`'s hook type (`lowering.rs:82`) already *is* the `Pass` trait. Replace
the three hand-copied four-call sequences with one `PIPELINE: &[&dyn Pass]`. Safe to run
unconditionally because every pass has an identity fast-path.

This closes **three confirmed bypasses**:

- `CompileWorkspace::compile_arena` (`emit/mod.rs:504-610`) reimplements `arena_to_schedule`
  inline with **no lowering calls at all**, and unlike the real scheduler **has no `Dwrt`
  guard** — the guard at `emit/mod.rs:1483-1488` exists specifically to "fail loudly rather
  than as a cryptic instruction-emit panic."
- `compile_arena_dag_avx2` (`:3708`) and `compile_arena_dag_avx512` (`:3722`) also skip it.

**Deletes:** three copies of the pass order, three bypasses, one missing-guard hole.

### Phase 5 — `ExprBuilder`: write the derivative algebra once

```rust
pub trait ExprBuilder {
    type Id: Copy;
    fn constant(&mut self, v: f32) -> Self::Id;
    fn op(&mut self, k: OpKind, args: &[Self::Id]) -> Self::Id;
}
```

`ExprArena` implements it with `Id = ExprId`; `EGraph` with `Id = EClassId`. The rule table
lives once in `pixelflow-ir` next to `OpKind`; both tiers instantiate it. Same treatment
generalizes to the algebraic rewrite rules.

Also carry `OpKind` directly in `ENode`, deleting **both** `define_op!` ZST families and the
50-arm `op_from_kind` bridge (`egraph/ops.rs:150-206`).

> ⚠️ The two implementations differ in *strategy*, not just spelling:
> `build_derivative` is **lazy, one step** (emits fresh `Dwrt` for children and lets
> saturation expand them, `graph.rs:1423-1428`); `diff_node` is **eager, whole-tree** with a
> two-pass memo (`lowering.rs:574`) and inline peephole folding (`:833-869`). Unifying the
> rule *table* does not automatically unify the *driver*. Expect one table, two drivers.

> ⚠️ Do **not** delete `lower_dwrt` in favor of `ChainRule`. They are not substitutable:
> `runtime.rs:93-102` documents that `lower_dwrt` must run *before* saturation so
> `ConstantFold` can cascade over the constants differentiation manufactures, and it is also
> the no-alloc tier. Getting this ordering wrong previously shipped a real bug — the fonts'
> density-dependent AA ramp broke exactly this way (`runtime.rs:216-224`).

**Deletes:** one of two chain-rule implementations, two ZST op families, one 50-arm bridge
table, and demotes the differential test from load-bearing safety net to ordinary regression.

### Phase 6 — Promote the legalization table

Move `coverage.rs`'s Legal set out of test-only into `OpKind::category() -> Legal | Expand |
Structural`, and have `lowering.rs`'s four predicates *read* it instead of restating it.
Depends on Phase 1 (the key must be sound).

Then add the axis that is genuinely missing — **target**. Minimum honest target set, seeded
from `xtask/src/main.rs:439-470`'s `ISA_LEVELS`:

```
{ sse2-baseline(+avx), avx2, avx2+fma, avx512f+dq, neon }
```

Architecture is too coarse: `MulAdd` differs between `avx2` and `avx2+fma` (two whole
function bodies at `avx2.rs:455` vs `:469-477`), and `Eq` differs between `avx512f` and
`avx512f+dq` (comparisons need `vpmovm2d`, which is DQ — a requirement currently written
down *only* in `xtask/src/main.rs:460-465`).

A third axis exists and should be acknowledged: **legality attaches to the resolved form,
not the opcode.** `Binary(Shl, x, Const)` → `ShiftImm` is legal; `Shl` with a non-`Const`
RHS is illegal on every target (`emit/mod.rs:1461`).

**Optional, ambitious:** once the table exists, phase-type the arena —
`Arena<Surface> → Arena<NoDwrt> → Arena<Legal>`, with the emitter accepting only `Legal`.
That makes the pass ordering a compile error if violated and makes "a `Dwrt` reached
codegen" unrepresentable rather than a runtime panic. **Honest cost:** it is added
machinery, and Phase 4 already captures most of the safety. Treat as a follow-up, not a
prerequisite.

**Deletes:** a test-only const doing production work, four restated predicates, and
(if phase-typed) the surviving-`Dwrt` panic at `emit/mod.rs:1483-1488`.

### Phase 7 — One failure convention

Make every backend return `Err(&'static str)`. Delete the `catch_unwind` in
`emit/mod.rs:5753-5772` and the comment explaining why it was needed.

**Deletes:** two conventions, one `catch_unwind`, one internally-inconsistent backend.

### Phase 8 — `no_std` for real

The 219 errors under `--no-default-features` are mostly two missing lines:

| file | errors | cause |
|---|---|---|
| `backend/emit/aarch64.rs` | 135 | missing `use alloc::vec::Vec;` |
| `backend/emit/x86_64.rs` | 59 | missing `use alloc::vec::Vec;` |
| `arena.rs` | 18 | missing `use alloc::vec;` |
| `kind.rs` | 7 | route through the already-present `libm` |

**194 of 219 are those first two lines.** Their siblings `avx2.rs:32` and `avx512.rs:23`
already have the import. Notably, `backend/emit/executable.rs` — the `mmap`/`mprotect`/
`MAP_JIT` file — produces **zero** errors.

Combined with Phase 0's `libc` fix, this is the project's first real `no_std` guarantee.
Then promote `.github/workflows/rust.yaml:571` out of `continue-on-error`.

**Deletes:** a false claim in `lib.rs:18`, and `continue-on-error` from a CI job.

### Phase 9 — The crate split: **deferred, and conditional**

Re-evaluate after Phases 0–8. The evidence is genuinely mixed, and most of the motivating
wins get captured earlier and cheaper:

| Claimed win | Verdict |
|---|---|
| Deduplicate `ENode`/`ExprNode` | **Illusory** — structurally incompatible types; a crate boundary does nothing. Phase 5 is the real fix. |
| `no_std` reachability | **Illusory as argued** — Phase 8 gets it without a split |
| Compile time | **Weak** — whole crate builds in 3.12s; ceiling is ~1.5s cold, ~0 incremental |
| "Too much in the IR" | **Real, wrong target** — Phase 2 is the fix |
| Dissolve the search↔codegen inversion | **Real** — but see below |
| Testability | **Negative** — see below |
| Minimal public API | **Negative** — see below |

The one durable win is real: `pixelflow-ir` is two crates in a trenchcoat, and welding the
IR-defs half (which must sit *below* search) to the codegen half (which wants to sit
*above* it) is what forces the whole thing below search. That inversion has concrete costs
— `Lattice::bake` hand-sequences optimize-then-compile (`lattice/mod.rs:408-414`), and
`cell_grid.rs:284` calls `compile_collapse_cached` **without** the optimize step, so the
terminal's hot path gets no CSE and no FMA fusion.

But the cure fights the endgame. `pixelflow-search` pulls `rand`, `serde`, `serde_json`,
is `std`-only, and its `lib.rs:1-3` is `#![allow(warnings)] #![allow(unused)]` — meaning
the workspace's `unused_must_use = "deny"`, the lint the NO-SILENT-FAILURES rule leans on,
**is globally suppressed there.** Putting the bare-metal JIT emitter above that crate is
the opposite of the no_std direction.

Two further costs, both measured:

- **Testability is negative.** `pixelflow-ir/tests/transcendental_jit.rs` — the file
  CLAUDE.md cites as pinning the FP contract — asserts the JIT emitter and
  `OpKind::eval_binary` agree bit-for-bit (`:148-175`). Those sit on opposite sides of the
  proposed boundary. The invariant becomes cross-crate, testable only from the top.
- **It cements the over-broad API.** `pixelflow-ir` has **462 public items and 12
  `pub(crate)` sites**; measured cross-crate usage is **~44 paths**. So ~90% of the public
  surface is unused outside the crate. A split converts every internal-by-convention item
  into a permanent contract — the opposite of the minimal-public-API rule.

**Therefore the prerequisite for even considering it is: tighten visibility first.**
Demote the ~400 unused-externally items to `pub(crate)`. That delivers the "too much is
exposed" feeling directly, is reversible, and is the only way to know which ~44 items the
boundary would actually be made of.

> Also note the tripwire if this ever proceeds: `pixelflow-core/src/lib.rs:190` re-exports
> `pixelflow_ir as __ir`, and `pixelflow-compiler` stamps `::pixelflow_core::__ir::…` into
> **third-party user crates** at ~40 sites, spanning both halves (`__ir::OpKind` and
> `__ir::jit_cache::compile_cached`). One facade cannot name two crates, so codegen would
> have to `pub use` the defs crate — meaning from every downstream viewpoint, nothing was
> split. Separately, `substitute_params`/`substitute_vars_with`/`splice` are **inherent
> `impl ExprArena` methods** (`arena.rs:615, 741, 817`); Rust forbids inherent impls outside
> the defining crate, so they cannot move to a passes crate without becoming free functions
> — a breaking change to macro-stamped code.

**Naming, independent of the split:** consider renaming `pixelflow-compiler` →
`pixelflow-macros`. It is a macro front-end, and it is currently squatting the name the
passes+codegen layer wants.

### Phase 10 — Unified `CompiledKernel`: **it already exists; delete the alternative**

Verified. The proposed type — one region per kernel, RAII-freed on `Drop`, mechanism hidden
behind internal `cfg`, `Send + Sync` — **is `ExecutableCode`** (`executable.rs:11-19`, `Drop`
at `:119-126`). It is what the shipped path already uses. So this is not an addition; it is
**deleting everything that isn't it**, optionally renaming.

**No executable-memory leak exists anywhere** — every owner has a working `Drop`, directly
(`ExecutableCode:119`, `CodeBuffer:327`) or transitively (`CompileWorkspace` owns a
`CodeBuffer` by value; `JitManifold` owns an `ExecutableCode`).

**Claim corrections:**

- ❌ *`compile_cached`/`compile_collapse_cached` return a fresh `Arc` per call.* They return
  a **globally interned, shared** `Arc` out of `static CACHE: OnceLock<Mutex<HashMap<…>>>`
  (`jit_cache.rs:32`); a hit is a refcount bump (`:79-81`), and `Arc::ptr_eq` for
  canonically-identical kernels is an **asserted invariant** (`jit_cache.rs:205-215`).
- ✅ *…but the conclusion holds.* Each **distinct** kernel gets its own `mmap`ed,
  `munmap`ed region, and no shared buffer is reachable from `bake` at all. Live proof in the
  shipped terminal: `cell_grid.rs:283-286` compiles four channel kernels into
  `[Arc<JitManifold>; 4]` (`:235`) that coexist and are all called per frame.
- ✅ *`CompileWorkspace` is bench-only.* Confirmed — only callers are
  `pixelflow-pipeline/src/jit_bench.rs:480,482` and `bin/bench_jit_compile_cost.rs`. Zero
  hits in `core-term`, `pixelflow-graphics`, `pixelflow-runtime`, `pixelflow-core`,
  `pixelflow-compiler`, `pixelflow-search`; `pixelflow-pipeline` has **no reverse deps**.
- ❌ *The duplication is across the aarch64/x86_64 emitters.* The emitters and `jit_cache.rs`
  contain **zero** memory-management code. All of it is `executable.rs` (22 OS/arch `cfg`s +
  6 `cfg(unix)`) plus the orphan `patch.rs`.

**Two findings worth acting on independently:**

> 🔒 **The entitlements file depends on `CompileWorkspace` staying out of the binary.**
> `assets/entitlements.plist:5-24` reasons this out explicitly: *"`allow-jit` is NOT granted
> here because `CompileWorkspace`'s `MAP_JIT` path is not part of this binary."* If anyone
> ever wires it into the shipped app, the **notarized build breaks at runtime.** The two
> allocators use *different, incompatible* macOS mechanisms for the same OS —
> `ExecutableCode` uses `mprotect(PROT_EXEC)` + `allow-unsigned-executable-memory`,
> `CodeBuffer` uses `MAP_JIT` + `pthread_jit_write_protect_np`. Deleting `CodeBuffer`
> removes that standing hazard and shrinks the entitlements rationale from 20 lines to two.

> 🐛 **`benchmark_compile_reused` does not measure what it documents.**
> `jit_bench.rs:469-471` claims the region "is mmap'd once up front, and each compile pays
> only `pthread_jit_write_protect_np` toggles + icache invalidation … instead of
> mmap/munmap." But `CompileWorkspace::compile_arena` (`emit/mod.rs:594`) calls
> `compile_from_schedule` → `compile_dag_via_backend` (`:1023`), which does a **full
> `ExecutableCode::from_code`: `mmap` + `mprotect`, then `munmap` on drop** — and only *then*
> copies the bytes into the "reused" buffer (`:605-608`). So it pays the mmap/munmap it
> claims to avoid, **plus** an extra copy and the W^X toggles. **Any gate-G0 amortized-cost
> number from this harness is measuring something other than its stated claim** — relevant to
> the outstanding "redo P0 with a trustworthy timer" work, which now has a second reason to
> distrust the old numbers.

**Deletes (~430 lines, ~0 added):**

| Target | Lines |
|---|---|
| `backend/emit/patch.rs` (whole file — also in Phase 0) | 49 |
| `CodeBuffer` (`executable.rs:166-334`) | ~170 |
| `JitWriteState` + `toggle_jit_write` (`executable.rs:336-368`) | ~33 |
| The only `MAP_JIT` site (`executable.rs:192-196`) | — |
| `CompileWorkspace` (`emit/mod.rs:440-611`) incl. its duplicate of `arena_to_schedule` | ~170 |
| `benchmark_compile_reused` + `REUSED_CODE_CAPACITY` + the `reused` bin arm | ~50 |
| Two of three `sys_icache_invalidate` externs → one private helper | — |
| `docs/function-namespace-audit.md:134`'s remedy row (recommends growing the deleted type) | — |

Also removes one `#[cfg(target_arch = "aarch64")]`-only public type from the API surface,
and **subsumes Phase 4's `CompileWorkspace` bypass** — you cannot bypass the pipeline from a
type that no longer exists.

**The one addition worth arguing for:** `Drop` discards `munmap`'s return value
(`executable.rs:123`, `:331`) — the single place the no-silent-failures rule is bent. One
`debug_assert_eq!(rc, 0)`.

> ⚠️ Do **not** widen visibility to accomplish this. `ExecutableCode` is currently public
> only because cross-crate *tests* reach it (`pixelflow-core/tests/naked_scale.rs:19`,
> `pixelflow-search/tests/prod_kernel_jit.rs:80,87`,
> `pixelflow-ir/tests/avx512_evex_proof.rs:17`). Those could go through `JitManifold`
> instead — the correct direction here is *narrowing*.

**Do this before Phase 9, not after.** Every deleted line is one you don't have to relocate,
re-review, and re-`cfg` in a new crate — and `CompileWorkspace` straddles the seam worst of
all (it lives in `emit/mod.rs`, uses `regalloc`/`ScheduledOp`/`EmitCtx`, *and* owns
executable memory).

---

## Sequencing summary

```
Phase 0   Subtract dead code                    ~950 lines    zero risk
Phase 1   Fix OpKind index space                bug fix       blocks Phase 6
Phase 2   Bucket E → pixelflow-core             -3,600 lines  the real "too much in IR" fix
Phase 10  Collapse to one exec-memory type      -430 lines    subsumes Phase 4's worst bypass
Phase 3   lowering.rs → src/passes/             git mv
Phase 4   Reify the pipeline                    closes remaining bypasses
Phase 5   ExprBuilder, one derivative algebra   the real dedup
Phase 6   Legalization table + target axis      needs Phase 1
Phase 7   One failure convention                deletes catch_unwind
Phase 8   no_std for real                       ~25 mechanical fixes
──────── re-evaluate ────────
Phase 9   Crate split                           conditional; tighten visibility first
```

Phase 10 is renumbered-in-place rather than reordered — it was scoped last but sequences
third, because it is pure deletion and it removes the type that would otherwise be the most
painful thing to relocate in Phase 9.

**Running total before any structural decision: ~5,000 lines deleted, two bugs fixed, one
false benchmark retired.** That is the "clean up the old shit aggressively" pass, and it is
worth doing whether or not Phase 9 ever happens.

Phases 0–4 and 10 are mechanical and independently revertible. Phase 5 is the one with real
design content. Phases 6–8 are cleanup that becomes easy once the rest lands.

One additional data point for Phase 9, from the exec-memory audit: **`libc` is used in
exactly one file** — `executable.rs`, 18 references, declared under
`[target.'cfg(unix)'.dependencies]` (`pixelflow-ir/Cargo.toml:18-19`). Moving that file out
removes the `libc` dependency from `pixelflow-ir` **entirely**, along with all 6 `cfg(unix)`
gates and all 22 OS/arch `cfg`s. That is the cleanest single argument for the split — but it
still does not reach `no_std`, because the residual ~25 errors (`f32` inherent methods in
`kind.rs:629-637` and `kernel.rs:318,389`; the `ExprId` deref issue in `arena.rs`) live in
the code that *stays*. Phase 8 fixes those directly.
