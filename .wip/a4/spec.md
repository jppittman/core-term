# A4 spec: value bounds in codegen and the key

**Base:** `a747770` (main after A3, #1447). HEAD may carry `.wip/`-only commits on top; every
file:line below is at `a747770` (`git diff --quiet a747770 HEAD -- . ':!.wip'` holds). Re-grep
before trusting a line number; the grep in §9 is the authority.
**Plan:** `docs/language/2026-10-20-a-bound-is-a-value.md` (cited as "plan"), its §6 A4 bullet
(plan:627-650) and §8 decision 7 (plan:728-730).
**A3's record:** `/home/user/scratch/a3-wip-archive/.wip/a3/spec.md` (cited as "A3 spec", §13 J24-J28)
and `rulings.md` (cited as "A3 rulings").
**What binds the design:** CLAUDE.md "`If` guards its arms", "Floating point at the edges", "The
language is a DAG"; DENOTATION §3; L1-L9 (plan:169-216).

Every decision an implementer would otherwise make is made here. The one that is not mine to make
is §6's recommendation, which is marked **for review**; the rest of the spec does not depend on
which option review picks except where §6 says so.

---

## 0. What A4 is, and what it is not

**A4 is** (plan:627-650): a fold whose bound is a value compiles and runs on every tier, and has
a key.

1. **The guard** (L5, L2): a new last pass of `pixelflow_ir::passes::legalize`
   (`pixelflow-ir/src/passes/guard.rs`, new) wraps every fold with a value bound in
   `If(dom, If(Lt(lo, hi), fold, identity), NaN)`, built from arena nodes so every guard def
   takes its variance from `compute_arena_variance` (§1.3-§1.4).
2. **The operand and the carve**: `ScheduledOp::Reduce`'s `Bound::Value` reaches the schedule as
   a `ValueId`; `program::operands` yields it, so the carve, ownership and `assert_closed` see it;
   selection seeds the loop from `lo`'s lane and compares the latch against `hi`'s (§1.6, §2, §3).
3. **The post-collapse assert** in `arena_to_schedule`: L3 against the lattice's binders, and
   the single-reader property placement rests on (§1.5).
4. **`Trips`** (`Known(u64)`, `Estimated(u64)`) from selection to the allocator and to
   `EmitTraffic`, with a `Display` that prints a known count as the bare number (§1.6, §3).
5. **The key**: a value bound writes a marker (`u32::MAX`) in its count field of
   `FoldHeader::to_bits`, and `from_bits` reads it back; every serializer that wrote a header
   without its children learns to write them; the collapse corpus goes to v5 and the training
   corpus's schema text, whose hash is its version, names the marker (§4).
6. **`Kernel::over_values` is public** (decision 7) (§5).
7. **The extraction-side hole** (A3 spec J24): a recommendation, for review (§6).
8. The tests of plan:642-650 as named JIT tests, plus unit tests (§8); docs (§10).

**A4 is not:**
- **A5**: `<const N: usize>` and the structural range path (`Count::Structural`,
  `StructuralRange`, `instantiated_range`, `FoldRange`, `Holes`, `OPEN_RANGE`, `HOLE_STRIDE`,
  `ExpansionFold::Hole`, `lower.rs`'s `count`) stay exactly as they are. A4's one compiler edit
  (`arena_to_tokens`'s `Reduce` arm, §4.3) keeps the hole path byte for byte.
- **A6**: no `kernel!` bound surface; sema still refuses a non-literal bound; no `Args`-write
  refusal of `2²⁴ + 1`. `kernel!` cannot produce a value bound after A4 either.
- **A7**: the macro-tier template exemption (`pixelflow-compiler/src/emit.rs:154-156`) stays.
- **B1-B5**: no `Extent(p)` leaf, no discrete-kernel argument, `lower_gather`'s clamp stays, the
  glyph keeps its bucketed literal folds and `FontPrograms`; production kernels carry no value
  bound after A4 (byte identity, §7, rests on it).
- **T6**: bounds stay `Real`; `BoundArithmetic` stays at the user surfaces; the `Eq(Floor(v), v)`
  conjunct stays.
- **Decision 8**: `ANY`, `ALL`, `SEQ` keep refusing a value bound (`BoundSort`), so the guard
  meets only `SUM`, `PRODUCT`, `MIN`, `MAX`.
- **Not** a change to the lattice's own folds: `collapse` and `pack` keep literal `SEQ` folds,
  and the lattice's extent stays in the key (plan §4 "Deferred").
- **Not** an emitter change: no encoder, no `IsaBackend` method and no instruction is added on
  any tier (§3.4). So plan §6's "NEON under qemu where an emitter changes" does not trigger.

---

## 1. The semantics, as laws

### 1.1 What a value-bounded fold computes, for every bound value

`Reduce{⊕, b, lo, hi, body}` with at least one value bound, `⊕ ∈ {SUM, PRODUCT, MIN, MAX}`
(the only monoids `FoldHeader::new` admits beside a value, `fold.rs:836-865`), stride 1
(likewise). Write `v` for the value of each value bound at run time, as the lane holds it; a
literal bound is in the domain by construction (`FoldHeader::new`, `KnownRange::new`).

**D (the domain).** `v ∈ D ⟺ 0 ≤ v ∧ v ≤ 2²⁴ ∧ ⌊v⌋ = v`, with ordered `≤` and exact `=`.
So `D = {0, 1, …, 2²⁴}`, and `-0.0 ∈ D` (it is the count 0, as `Fold::count` says,
`fold.rs:660-682`).

| Every value bound in `D`? | `lo < hi`? | The fold is | Loop trips at run time |
|---|---|---|---|
| yes | yes | `⊕_{k ∈ [lo, hi)} body[b := k]` | exactly `hi − lo`, `1 ≤ hi − lo ≤ 2²⁴` |
| yes | no (`lo ≥ hi`) | the identity of `⊕` (L2): `0`, `1`, `+∞`, `−∞` | 0 (the loop's block is skipped) |
| no | any | **NaN** (L5) | 0 (the guard's block is skipped) |

Instances, for a uniform `n` as `hi` and `lo = 0` (the plan's gate list, plan:643-650):

| `n` | in `D` | result |
|---|---|---|
| `0`, `-0.0` | yes, empty | identity |
| `1`, `7`, `1000` | yes | the fold, `n` trips |
| `2²⁴` (16777216.0) | yes | the fold, 2²⁴ trips; the counter reaches `2²⁴` exactly and the latch's `Ge` fires there |
| `2²⁴ + 2` (16777218.0, an `f32`) | no (`≤ 2²⁴` fails) | NaN |
| `2.5` | no (`⌊2.5⌋ ≠ 2.5`) | NaN |
| `−1` | no (`0 ≤ −1` fails) | NaN |
| NaN | no (every ordered compare is false) | NaN |
| `+∞` | no (`≤ 2²⁴` fails) | NaN |
| `−∞` | no (`0 ≤ −∞` fails) | NaN |
| `1e30` | no (`≤ 2²⁴` fails) | NaN |
| `2²⁴ + 1` | — | not a case: no `f32` names it (plan:650, §1.4); `16777217.0` rounds to `2²⁴`, which is in `D` |
| uniforms `lo = 5`, `hi = 3` | yes, `lo ≥ hi` | identity |

**L5′ (never a hang, never an out-of-range trip count).** The loop body runs only when every value
bound is in `D` and `lo < hi`; then the f32 counter steps by `1.0` from `lo` and every value it
takes is an integer `≤ 2²⁴`, so `Ge(step, hi)` becomes true at `step = hi` exactly. Outside
`D` the loop is not entered at all. The trip count is never converted to an integer on any tier
(no `cvttps2dq`, no `FCVTZS`): the counter and the end stay `f32` lanes compared each latch, as
today, so CLAUDE.md's `TruncToInt` row is not reached.

**Signed zero.** A `lo` of `-0.0` is the count 0 and the first trip's binder lane holds `-0.0`
(the counter is seeded with `lo`'s lane, not with a literal `+0.0`). A body that tells `-0.0`
from `+0.0` (`1 / k`) sees CLAUDE.md's signed-zero rows; nothing is spent to normalize it
(§11, J9).

### 1.2 Why every tier computes the same answer

- The guard uses only `Le`, `Lt` (ordered: false for NaN on every tier), `Eq` (exact),
  `Floor` (exact; NaN stays NaN), `BitAnd` (bitwise on masks), `If`'s select (bitwise blend), and
  the constants `0`, `2²⁴`, the identity and NaN. None of them is in CLAUDE.md's per-target table.
  `Ge`, which is (x86 unordered), is never used by the guard (plan:370-371).
- The latch's `Ge(step, hi)` only ever compares two in-domain integers, so its NaN row is never
  reached.
- **Every mask the guard makes is uniform across a batch's lanes.** A value bound's variance shares
  no bit with the lattice's row, column or lane binders (L3; asserted after collapse, §1.5), so
  every lane of `lo`, `hi`, `dom` and `run` holds the same value: a uniform is a broadcast load,
  an enclosing binder a broadcast counter. So `If`'s "a mixed batch runs both arms" never happens
  for a guard: `any(dom) = all(dom)`, and the arm that holds the loop runs on every lane or on
  none. The loop's own exit test is "any lane done" (`select.rs:326-334`: back to the head while
  no lane of `done` is set), so even a mixed batch (impossible by L3) would exit at the first
  in-domain lane's end, never hang.

### 1.3 Where the guard lives, and why

**The guard is arena nodes, built by the last pass of `pixelflow_ir::passes::legalize`**
(`pixelflow-ir/src/passes.rs:84-117`; new file `pixelflow-ir/src/passes/guard.rs`), after
`expand_transcendentals_owned` (`passes.rs:116`). `pipeline.rs`'s chain is unchanged
(`Legalize → Lower → Nest → Own → Layout`, `pipeline.rs:138-144`): the guard is inside
`Legalize`.

Why an `If`, given "`If` guards its arms" (CLAUDE.md):

- `If(m, a, b)` compiles to: `if any(m)` run the true arm's own block, `if !all(m)` run the false
  arm's, then one select. The guarded fold is the true arm's only value, so `Own` gives the loop
  to that arm (`ownership.rs:186-207`) and `Layout` makes it a block (`layout.rs:47-87`): the
  existing arm skip *is* the guard. There is precedent: `a_fold_owned_by_an_arm_is_guarded`
  (`emit/mod.rs:1763-1802`).
- Because the masks are uniform (§1.2), "both arms run in a mixed batch" never applies, so the
  arm's block is skipped exactly when the fold must not run. The `If` is dispatch, not a fold, as
  CLAUDE.md says; that is what is wanted here: the loop must not run at all outside `D`.
- The identity and NaN arms are constants, which own nothing when the `If` sits in a fold's scope
  (a constant is `CONST`, placed in `Scope::Body`), and own a one-constant block when the `If` is
  itself in `Scope::Body`; either way the select reads them (§2).
- Built as arena nodes, every guard def takes its variance from `compute_arena_variance`
  (`lower.rs:153`) like any other node; nothing in `program/` hand-assigns one (plan:365-368).
- No accumulator parameter, no edge to `exit`, no change to the allocator's single-predecessor
  exit and head-only parameters (`regalloc/local.rs:1015-1022`, `:1114`).

Why `passes::legalize` and not a codegen `Pass`: the guard is L5 and L2 written in the
language's own ops, the same kind of legalization as `expand_transcendentals`' documented-domain
NaN for `sin` (CLAUDE.md "Precision is on the table; range is not"); it needs `rebuild_arena`
(`passes.rs:151`) and, after `collapse`, rebuilding a `Write` (`push_write` is `pub(crate)`,
`arena.rs:1044`), neither of which codegen can reach without a visibility change CLAUDE.md
forbids. Rejected alternatives are §11 J1.

**Why last.** Nothing after it can rebuild the guard (a rebuild through `rebuild_arena` is
correct but re-dedups), and lowering's single-reader assert (§1.5) then checks exactly what the
pass produced.

**Idempotence.** The guard is the one legalize pass that is not idempotent: run twice it wraps
the already-guarded fold again (correct, redundant). `legalize` runs it once. The module doc's
"Every pass is idempotent" (`passes.rs:21-22`) is amended to say so (§3.1).

### 1.4 The guard's exact shape

For each reachable `Reduce { fold, lo, hi, body }` with at least one `Bound::Value`, after its
children are rebuilt (`m` is `rebuild_arena`'s child map), push, **in this order**:

1. `R = push_reduce(fold, lo.map(m), hi.map(m), m(body))` (interned; usually the old id).
   Read its bounds back from the node pushed (`arena.node(R)`): discovery (`Bound::discover`,
   `fold.rs:375-391`) runs again in `push_reduce`, and if it made both bounds literals (it cannot
   for a bound that was already a value, but the pass does not rely on it) the pass returns `R`
   unguarded.
2. For each distinct value-bound term `v` (lo's first; a `hi` that is the same `ExprId` as `lo`
   is tested once), the domain test `d(v)`:
   `zero = push_const(0.0)`, `top = push_const(Fold::EXACT_BOUND as f32)` (2²⁴, exact),
   `a = push_binary(Le, zero, v)`, `b = push_binary(Le, v, top)`, `f = push_unary(Floor, v)`,
   `w = push_binary(Eq, f, v)`, `ab = push_binary(BitAnd, a, b)`, `d = push_binary(BitAnd, ab, w)`.
3. `dom = d(lo)` / `d(hi)` / `push_binary(BitAnd, d(lo), d(hi))` for one / the other / both.
   A literal bound adds no conjunct (plan:369-370).
4. `start`, `end`: a value bound's term, or `push_const(n as f32)` for `Bound::Lit(n)`
   (`n ≤ 2²⁴`, exact). `run = push_binary(Lt, start, end)`.
5. `identity = push_lit(fold.monoid().identity())` (`fold.rs:93`).
6. `ran = push_ternary(If, run, R, identity)`.
7. `nan = push_const(f32::NAN)`.
8. `guarded = push_ternary(If, dom, ran, nan)`; the hook returns `Some(guarded)`, so every parent
   of the old `R` is rebuilt over `guarded`, and `R`'s one reachable reader is `ran`'s true arm.

`identity` and `nan` are `Lit::Real` (`SUM`/`PRODUCT`/`MIN`/`MAX` have real identities), never a
mask. Every comparison yields a mask (`Lit::Mask` semantics, CLAUDE.md), and `BitAnd` of masks is
the bitwise op every backend selects (`avx512.rs:150-158` `kandw` on opmasks, `x86_64.rs:768`,
`aarch64.rs:918`).

### 1.5 What lowering asserts after collapse

`arena_to_schedule` (`pixelflow-codegen/src/program/lower.rs:121-384`) gains two pipeline-bug
panics, checked only when some reachable `Reduce` has a value bound (a pre-scan; literal kernels
skip both, §7):

- **P1, L3 after collapse.** For every reachable value bound with term `t` of a fold `fold` at
  `end`: `fold.admits_bound(end, variance[t])` holds (not the fold's own binder, no coordinate),
  and `variance[t]` shares no bit with `lattice`, the union of `Variance::from_var` of every
  reachable `Write`'s `row`, `col` and `lane` (collected in the lane scan, `lower.rs:132-146`).
  This is the property §1.2 relies on for uniform masks and a uniform latch. Message:
  `"arena_to_schedule: a fold's {end} bound varies with {reads:?}, a lattice binder's: L3 after
  collapse (docs/language/2026-10-20-a-bound-is-a-value.md); a pipeline bug, not a kernel's"`;
  the own-binder half panics with `BoundRefusal`'s `Display` prefixed `arena_to_schedule: `.
- **P2, the single reader.** Every reachable node that has a value-bounded `Reduce` `R` among its
  children (`arena.children`, checked before the `Write` skip at `lower.rs:177-179`) is
  `Ternary(If, m, t, f)` with `t == R`, `m != R`, `f != R`. Message: `"arena_to_schedule: {R:?},
  a fold whose bound is a value, is read by {node:?} other than as an If's true arm: an arena
  that was not legalized (passes::legalize ends with the guard that owns such a fold, L5)"`.
  This is the single-reader assumption placement rests on (§2): it is what puts the loop inside
  the arm.

P2 checks the reader's shape, not the guard's contents (that the mask is `Lt(lo, hi)` under a
domain test); the contents are `guard.rs`'s unit tests' (§8.1). A hand-built arena that wraps a
value-bounded fold in some other `If` passes P2; only crate-internal tests can build one (§11,
J4).

**Not asserted: "`stride ≠ 1` only with known bounds"** (plan:377-378). `FoldHeader::new` already
refuses a stride other than 1 beside a value bound (`fold.rs:848-852`), and `from_bits` will
(§4.1), and a `Reduce` node holds only a `FoldHeader`: the type refuses the shape, so the assert
could not fire (CLAUDE.md "Subtract before you add"; §11, J5).

### 1.6 How the trip count reaches the loop

**Today.** Lowering refuses a value bound (`lower.rs:48-52`, `:351-359`, `known`). Selection
reads `KnownRange::of(fold, lo, hi).expect(KNOWN_RANGE)` (`select.rs:172`) and emits a
do-while (`select.rs:251-337`): `lo = constant(start)` (`:275`), the head's parameters are the
f32 counter and the accumulator, the body runs, `step = binder + constant(stride)` (`:311-312`),
`end = constant(end)` (`:313`), `done = Ge(step, end)` (`:314`), and `B::branch(Test { cond: done,
dead: IfArm::True }, Edges { taken: head, next: exit })` (`:327-334`). The count itself is never an
operand: `trips = parent_trips * len` (`:285-289`) is a `u64` handed to `Builder::open_loop`
(`build.rs:321-330`) for the allocator's carry weights (`regalloc/local.rs:125-138`, `:195-205`,
`:384-433`) and for `EmitTraffic.trips` (`traffic.rs:76-79`, `:122-130`).

**After A4.**
- **Operand.** `arena_to_schedule`'s `Reduce` arm maps each bound: `lo: lo.map(map_child)`,
  `hi: hi.map(map_child)` (the value child is earlier in arena order, so it is mapped). `known`
  goes.
- **Carve.** `program::operands` yields `lo`'s then `hi`'s `ValueId` for a `Reduce`
  (`mod.rs:391-411`); `all_operands` and `structural_children` (`mod.rs:428-443`) follow, so the
  carve's closure walk (`scopes.rs:271`), `assert_closed` (`scopes.rs:76-97`) and
  `ownership::reads` (`ownership.rs:116-119`) see the bounds as reads *of the `Reduce` def*, in
  the scope that def sits in.
- **Selection.** `lo` seeds the counter: `self.bound(lo)`, a constant for a literal and
  `self.lookup(v)` for a value. `hi` is read at the latch: `self.bound(hi)`, after the fold's
  frame has popped, so `lookup` resolves in the parent's frame or an enclosing one (§2). The
  stride is the known range's, or `UNIT_STRIDE` (1) beside a value bound. The assert at
  `select.rs:268-271` becomes "a known range is non-empty" (`range.is_none_or(|r| !r.is_empty())`).
- **Liveness.** The latch's compare is an instruction inside the loop that reads `hi`, so the
  allocator's own rule ("a value defined before a loop and read inside it lives to the loop's
  latch", `regalloc/local.rs:44-46`) keeps `hi` live across the loop: "the `Reduce` def's interval
  spans the loop" (plan:360-362) is what already happens, with no carve change beyond `operands`.
  `hi` is not added to the fold's own `ScopeFold::outer` (its body does not read it). Pinned by
  §8.2's `a_bound_read_only_by_the_loop_reaches_its_latch`.
- **`Trips`.** `Trips::of(range)` is `Known(range.len())` or `Estimated(Fold::ESTIMATED_TRIPS)`;
  `trips = Trips::of(range).nested_in(parent_trips)` (saturating product; an estimate if either
  is). `open_loop` takes a `Trips`; the allocator reads `trips.weight()` (the number, known or
  estimated) into `LoopFacts.trips`; `EmitTraffic.trips` is `Vec<Trips>`.
- **`Display`.** `Known(n)` prints as `n` with the formatter's width honoured
  (`fmt::Display::fmt(&n, f)`), so `byte_probe`'s `trips={trips}` (`examples/byte_probe.rs:306-307`)
  and `constant_traffic`'s `{trips:>8}` are unchanged; `Estimated(n)` prints `~n`
  (`f.pad(&format!("~{n}"))`).
- **`EmitTraffic`.** `dynamic_memory_ops(&self) -> Option<u64>`: the sum of each scope's memory
  ops times its trips, `None` when any scope's trips are an estimate (`traffic.rs:154-163`), so no
  measurement shows an estimate as a count (plan:379-381).

---

## 2. Placement (plan §3.3)

Placement has one new rule, "a fold reads its value bounds" (plan:351), realized by `operands`
(§1.6). Everything else is today's: `Nest` puts a value in the outermost scope binding every
binder in its variance (`scopes.rs:181-314`); variance follows L4 (A3).

**Facts the guard makes true, by L4 and §1.4.** `var(R) = (var(body) \ b) ∪ var(lo) ∪ var(hi)`;
`var(dom) = var(run) = var(lo) ∪ var(hi) ⊆ var(R)`; `var(ran) = var(guarded) = var(R)`. So `R`,
`ran` and `guarded` always land in **one scope**, and the masks in it or an enclosing one.

**P-a. A bound read only by the loop** (the body does not read it). `hi` is read by `R` (as an
operand), by `run` and by `dom`. Its def is placed by its own variance: `Scope::Body` for a
uniform, the enclosing fold's scope for an enclosing binder. `R`'s parent scope reads it as an
operand of `R`; if that scope is a fold, the carve's walk reaches `hi` through
`structural_children(R)` and records it in that fold's `outer` when it is hoisted. The latch reads
it from the frame it was bound in. Its interval spans the loop by the allocator's rule (§1.6).

**P-b. A guarded fold whose body reads no coordinate** (body and bounds `CONST`). `R`, `ran`,
`guarded`, `dom`, `run` are all `CONST`, so all in `Scope::Body`. The lattice's row fold is a
`Reduce` def in `Scope::Body` whose closure walk (`scopes.rs:247-282`) reaches `guarded` (the
kernel's value reads it), finds it hoisted, and stops: `placeholder_here[guarded] =
is_fold(guarded) = false` (`scopes.rs:276-278`). **The carve sees the `If`, not the `Reduce`, as
the def a deeper fold reads.** `R` is reached by no deeper closure (its one reader is `ran`,
P2), so it is not a placeholder anywhere and is carved at the body level as its own fold
(`scopes.rs:212-301`, iterating Body's `Reduce` defs backwards: the row fold claims nothing of
`R`'s, `R` is unclaimed and opens `ScopeFold { parent: Scope::Body, .. }`). It runs once per call
under the body's arm blocks; its scope's trips are `Estimated(32)` (parent `ONCE`). Without the
single reader, `R` would sit in the scope's own region, outside the arm, and run unguarded: P2
(§1.5) is the assert.

**P-c. A bound that varies with nothing but uniforms is computed once per call.** A uniform's
variance is `CONST`; so are `dom`, `run` and every arithmetic node of bounds over uniforms. They
land in `Scope::Body` and are computed once per call, before the lattice's loops, whatever scope
`R` lands in. If the body varies with X, `R`, `ran` and `guarded` sit in the column fold's scope
(the scope that holds the `Write`, which binds the lane), and the arm tests read `dom`/`run` from
the body's frame. Pinned at schedule level by §8.3's
`a_guard_on_a_uniform_count_is_computed_once_per_call`.

**P-d. A bound on an enclosing binder** (triangular `Σ_i Σ_{j<i}`). `hi = Var(i)` varies with
`i`; `R_in`, its guard and `Var(i)`'s def land in the outer fold's scope; the outer fold's
closure walk reaches `guarded_in → ran_in → R_in` (not hoisted: `i` is the outer binder), follows
`structural_children(R_in)` to `Var(i)` and into `R_in`'s body, and the recursion carves `R_in` one
level down (`scopes.rs:288-290`). The guard runs once per outer trip.

**Ownership and layout.** `R`'s only reader is `ran`'s true arm, so `R` is owned by that arm's
region; `ran` is read only by `guarded`'s true arm; `run` only by `ran` (as its mask, in
`guarded`'s true region); `dom` by `guarded` (as its mask, in the scope region); `lo`/`hi` by
`dom` (scope region), so they are the scope's. `Layout` puts `guarded`'s true block before
`guarded`, holding `run`, `ran`'s true block (holding `R`'s loop) and `ran`. Every value an arm
reads was constructed before it (§1.4's push order puts `R` first and `guarded` last), which is
`layout.rs`'s invariant (`layout.rs:10-18`).

---

## 3. Changes by file

### 3.1 `pixelflow-ir`

**`src/passes/guard.rs` (new, ~120 lines with tests).** Module doc: what the guard denotes (L5,
L2), that it is `legalize`'s last pass and why (§1.3), that it is not idempotent and why that is
harmless. Items:

```rust
use crate::arena::{ExprArena, ExprId, ExprNode};
use crate::fold::{Bound, Fold};
use crate::kind::OpKind;

/// The last pass of [`super::legalize`]: every reachable fold with a value bound,
/// in its guard (L5, L2). Takes the arena by value, so the identity (no node of
/// the arena is such a fold) moves it back untouched.
pub(crate) fn guard_value_bounds_owned(arena: ExprArena, root: ExprId) -> (ExprArena, ExprId) {
    if !arena.nodes().any(|(_, node)| has_a_value_bound(&node)) {
        return (arena, root);
    }
    let mut arena = arena;
    let root = super::rebuild_arena(&mut arena, root, |arena, node, m| {
        // `ExprNode` is `Clone`, not `Copy`; a `Reduce`'s fields are `Copy`.
        let &ExprNode::Reduce { fold, lo, hi, body } = node else { return None };
        if lo.value().is_none() && hi.value().is_none() { return None }
        let folded = arena.push_reduce(fold, lo.map(|id| m(id)), hi.map(|id| m(id)), m(body));
        Some(guard(arena, folded))
    });
    (arena, root)
}

fn has_a_value_bound(node: &ExprNode) -> bool { /* a Reduce with lo or hi a Value */ }
/// §1.4 steps 1-8, in that push order.
fn guard(arena: &mut ExprArena, folded: ExprId) -> ExprId
/// §1.4 step 2.
fn in_domain(arena: &mut ExprArena, v: ExprId) -> ExprId
/// A bound as a term: its child, or a `Const` of the literal.
fn term(arena: &mut ExprArena, bound: Bound<ExprId>) -> ExprId
```

(Write the `dom` conjunction as an explicit loop: the iterator form borrows `arena` twice.)

**`src/passes.rs`.**
- `:3-13` module doc: "runs six passes" → seven; the table gains a last row: "`guard` (crate-private,
  `passes/guard.rs`) | a `Reduce` with a value bound | `If(dom, If(Lt(lo, hi), Reduce, identity),
  NaN)`: NaN outside `[0, 2²⁴]`, the identity over an empty range (L5, L2)". Write the pass name in
  backticks without a link (a `pub(crate)` item linked from a `pub` doc is a rustdoc warning).
- `:15-27` order paragraph: add "the guard goes last, so no pass after it rebuilds the `If` lowering
  relies on"; amend "Every pass is idempotent" to "Every pass but the guard is idempotent (it wraps
  every value-bounded fold it meets, its own output's included, so `legalize` runs it once)".
- After `:67` `pub mod lattice;`: `mod guard;`.
- `:84-117` `legalize`: the last line becomes
  `let (arena, root) = expand_transcendentals_owned(&arena, root); Ok(guard::guard_value_bounds_owned(arena, root))`,
  with a comment: last, so nothing rebuilds the guard after it; lowering asserts its shape.
  `legalize`'s doc gains one sentence: a fold whose bound is a value comes back inside its guard.

**`src/fold.rs`** (the key, §4.1): `VALUE_FIELD`, `to_bits`, `from_bits`, `admits_a_value`, the
const assert, `VALUE_BOUND` deleted (`:1094-1102`), `KNOWN_RANGE`'s text (`:1104-1113`),
`BoundOutOfDomain`'s doc (`:440-444`: "(A4)" → "(`passes::legalize`'s guard)"), and tests (§8.1).

**`src/arena.rs:971`**: "(L5, A4)" → "(L5: the guard `passes::legalize` ends with)".

**`src/key.rs`** (§4.2): docs and tests only.

**`src/kernel.rs`** (§5): `over_values` public.

**`tests/value_bounds.rs`** (§8.1): module doc, the key test flipped, public `over_values` pins.

**`intent.md`** (§10.4).

### 3.2 `pixelflow-codegen/src/program/`

**`lower.rs`.**
- `:12` imports: drop `VALUE_BOUND`, keep `KNOWN_RANGE` (the lane fold, `:340`) and `KnownRange`
  (`is_empty_fold`), add `BoundEnd` (`OpKind` is already imported, `:13`).
- `:48-52` `known`: deleted.
- `:132-146` lane scan: also accumulate `lattice: Variance` from each `Write`'s `row`, `col`,
  `lane` (`Variance::from_var(b.var())`).
- After `:155` (`lane_uniform`): `let value_folds: Vec<bool>` over reachable ids, true for a
  `Reduce` with a value bound; `let any_value_fold = value_folds.contains(&true);`.
- In the main loop, before the `Write` skip (`:177-179`): `if any_value_fold {
  assert_read_as_an_arm(&node, arena.children(expr_id), &value_folds) }` (P2).
- `:351-359` the surviving-fold arm: for each `(end, Bound::Value(t))`, inline (P1): `let reads =
  variance[t.0 as usize]; if let Err(refusal) = fold.admits_bound(end, reads) {
  panic!("arena_to_schedule: {refusal}") }` and `assert!(reads.intersection(lattice).is_const(), ..)`
  with §1.5's message; then `ScheduledOp::Reduce { fold, lo: lo.map(map_child), hi:
  hi.map(map_child), body: map_child(body) }`. The comment at `:353-355` is replaced: "a value
  bound is an operand of its fold (`operands`): `lo` seeds the loop, `hi` is read at its latch".
- One private helper, with a doc that says what it protects (the single reader placement rests
  on): `fn assert_read_as_an_arm(node: &ExprNode, children: impl IntoIterator<Item = ExprId>,
  value_folds: &[bool])` (P2), three arguments (CLAUDE.md: fewer than four). P1 stays inline in the
  arm: a helper would take four (`fold`, `end`, `reads`, `lattice`).
- `arena_to_schedule`'s doc (`:84-120`): a paragraph "A fold whose bound is a value arrives inside
  its guard (`passes::legalize`'s last pass): its bounds are operands, and it is read only as an
  `If`'s true arm"; `# Panics` gains P1 and P2.
- Tests (`:386-485`): `a_value_bound_is_refused_by_name_before_it_is_scheduled` (`:463-484`) is
  replaced by three (§8.3).

**`mod.rs`.**
- `:136-153` `ScheduledOp::Reduce` doc: last paragraph → "`lo` and `hi` are the arena node's
  bounds, copied where the node is in hand (`lower::arena_to_schedule`). A literal is selection's
  constant; a value is an operand ([`operands`]): `lo` is the loop's seed, read where the loop is
  entered, and `hi` is read at its latch, both in the scope this def sits in. Selection reads a
  known range with `KnownRange::of`."
- `:379-411` `operands`: `ScheduledOp::Reduce { lo, hi, .. }` leaves the leaf group and yields
  `(lo.value(), hi.value(), None)`; the doc's first paragraph: "A `Reduce`'s operands are its value
  bounds, the seed and the end its loop reads; its body is not an operand: by the time …" (keep
  the rest).
- `:432-443` `structural_children`: code unchanged (it chains `all_operands`); doc: "plus the body a
  `Reduce` folds (its value bounds are operands already)".

**`scopes.rs`, `ownership.rs`, `layout.rs`, `tree.rs`**: no code change. `scopes.rs:239-246`'s
comment ("A nested `Reduce` … `structural_children`, which the walk below follows, yields it")
gains "and its value bounds, which a deeper fold may compute".

**`intent.md`** (§10.4).

### 3.3 `pixelflow-codegen/src/emit/`

**`traffic.rs`.** New `pub enum Trips { Known(u64), Estimated(u64) }` (derive `Clone, Copy, Debug,
PartialEq, Eq`), doc as §1.6, with:
- `pub(in crate::emit) const ONCE: Self = Self::Known(1);`
- `pub(in crate::emit) fn of(range: Option<KnownRange>) -> Self`
- `pub(in crate::emit) fn nested_in(self, parent: Self) -> Self` (`parent.weight().saturating_mul(self.weight())`, `Known` iff both are)
- `pub(in crate::emit) fn weight(self) -> u64`
- `#[must_use] pub fn known(self) -> Option<u64>`
- `impl core::fmt::Display` (§1.6).
`EmitTraffic.trips: Vec<Trips>` (`:76-79`, doc: "a fold whose range is not known, and every fold
inside it, is an estimate"); `EmitTraffic::of` initializes `vec![Trips::ONCE; ..]` (`:122`) and
copies `looped.trips` (`:129`); `dynamic_memory_ops` returns `Option<u64>` (`:154-163`, via
`Option`'s `Sum`). Tests `:206-217` updated (§8.4).

**`mod.rs`.** `:55` `pub use traffic::{EmitTraffic, ScopeTraffic, Trips};`. `:509-519` `Loop.trips:
Trips` (doc: "its own trips nested in its parent's"). The module doc's "## The loop nest"
(`:19-29`) gains: "A fold whose bound is a value arrives inside the guard `passes::legalize` builds
(an `If`'s arm): its counter starts at `lo`'s lane and its latch compares against `hi`'s, and the
arm is skipped when the range is empty or outside `[0, 2²⁴]`."

**`build.rs:321-330`.** `open_loop(&mut self, head: Label, parent: Option<usize>, trips: Trips)`.

**`select.rs`.**
- `:21` imports: `Binder, Bound, Fold, KnownRange, Monoid` (drop `KNOWN_RANGE`); `:10`
  `use super::traffic::{Trips, scope_ix};`.
- `const UNIT_STRIDE: u64 = 1;` with doc "the step beside a value bound: `FoldHeader::new` admits no
  other there".
- `Selector.loops: Vec<(usize, Trips)>` (`:66-68`).
- `:171-183` the `Reduce` arm: no `KnownRange` read; pass `Looped { fold, lo, hi, body }` (a new
  private `Copy` struct, one field per `ScheduledOp::Reduce` field) to `fold`.
- `:251-337` `fold(&mut self, parent: Scope, j: usize, looped: Looped)`: as §1.6. The LaneOp
  sequence on a known range is today's, in today's order: constant(lo), constant(identity), the
  body, combine, constant(stride), `Add`, constant(end), `Ge` (§7). Doc gains: "A value bound
  starts the counter at its lane and ends it at its lane; the guard around this fold (an `If`'s
  arm, `passes::legalize`) made the range non-empty and in `[0, 2²⁴]`, so the do-while runs
  `hi − lo` trips."
- New `fn bound(&mut self, bound: Bound<ValueId>) -> Result<B::Lane, CompileError>`: `Lit(n)` →
  `self.constant(n as f32)`, `Value(v)` → `Ok(self.lookup(v))`.

**`regalloc/local.rs:201`.** `trips: l.trips.weight(),`. `LoopFacts.trips` stays `u64` (doc:
"… per call, an estimate's number for a fold whose range is not known"). Nothing else in
`regalloc/` changes: the latch's read of `hi` is an ordinary read.

**Encoders (`avx2.rs`, `avx512.rs`, `x86_64.rs`, `aarch64.rs`, `asm.rs`)**: no change (§3.4).

**`intent.md`** (§10.4).

### 3.4 Per ISA: what a value-bounded fold emits, registers and labels

Nothing below is new code; it is what the existing selection produces for the new shape.

| | AVX2 | AVX-512 | aarch64 NEON |
|---|---|---|---|
| guard masks | `vcmpps` (LE_OS, LT_OS, EQ_OQ) into vectors, `vandps`, `vroundps` imm floor | `vcmpps` into opmasks, `kandw` (`avx512.rs:144-158`), `vrndscaleps` | `FCMGE`/`FCMGT` (operand order for `Le`/`Lt`), `FCMEQ`, `AND`, `FRINTM` |
| arm skip (×2 per fold entry) | `vmovmskps` + `test`/`cmp 0xFF` + `jcc` (`avx2.rs:310-328`) | `kortest` (+`vptestm` for a vector mask) + `jcc` (`avx512.rs:543-562`) | `umaxv`/`uminv`(+`mvn`) + `fmov` + `CbzFar` (`aarch64.rs:1247-1275`), a fixed two-word pair, so label resolution (`asm.rs`) is unchanged |
| seed | `lo`'s register moved into the counter parameter (no pool load) | same | same |
| latch | `vcmpps step, end` with `end` a register or its 32-byte slot | same, opmask, 64-byte slot | `FCMGE` against a `q` register or an `ldr q` reload |

- **Registers.** On a known range `end` is a constant, which every tier rematerializes at its read
  (`IsaBackend::rematerializable`), so it costs no register across the loop. A value `hi` is a live
  vector value across the loop: a candidate for `plan_carries` (kept in a register up to the
  latch, priced by its one read per trip × the loop's `Trips::weight()`), or slot-homed and
  reloaded at the latch. `lo` is read once, at the preheader jump. Nothing is added to the
  general-purpose or opmask files beyond the guard's masks (AVX-512: two or three opmask values,
  short-lived).
- **Labels and blocks.** `fold()` makes the same blocks as today (head, exit; the preheader is the
  block it is called in). The guard adds, through `open_arm`/`close_arm` (`select.rs:342-366`), an
  arm block and a past block per guarded `If` that owns work: two pairs per value-bounded fold
  (four labels), and in `Scope::Body` up to two more pairs for the one-constant false arms (§1.3).
- **Frame.** At most one more vector slot (`hi` spilled); `MAX_FRAME` is not approached by it.
- **Immediates.** None: no bound reaches an immediate field (no `imm8`, no `imm12` change); the
  stride is the pool constant `1.0`.

### 3.5 `pixelflow-search` (only if review takes §6's recommendation; see §6 for the code)

`src/egraph/and_or.rs` (`BoundUse`, `AndOrGraph::bounds`, `bound_uses`), `src/egraph/extract.rs`
(`Settlement::refused`, `relax`, `run`), the two pins in `src/egraph/fold_rules.rs:2201-2257`,
`src/egraph/intent.md`. Nothing else in search changes for A4: A3 built the IR and e-graph side,
and the key no longer refuses a value bound, so `jit_cache::compile` now hands one to
`optimize_runtime_arena` (A3 pinned that path at e-graph level).

### 3.6 Other crates

`pixelflow-graphics`, `pixelflow-pipeline`, `pixelflow-compiler`: the serializers, §4.3.
`pixelflow-pipeline/src/collapse_bench/mod.rs:295-335` and
`pixelflow-graphics/examples/constant_traffic.rs:40-58`: `Trips`/`Option` consumers, §4.4.

---

## 4. The key and the serializers

### 4.1 `FoldHeader::to_bits` and `from_bits` (`pixelflow-ir/src/fold.rs`)

Layout unchanged: `stride << 80 | op << 72 | binder << 64 | lo << 32 | hi`, nothing at or above
bit 112 (`fold.rs:984-987`, `UNUSED_BITS` `:1048`).

- **The marker.** New private `const VALUE_FIELD: u32 = u32::MAX;` beside `UNUSED_BITS`, doc: "A
  count field's bits when its bound is a value: all ones, which no count reaches (the assert
  below), so a literal and a value never share a field. The value's child is the node's next edge
  after the body, written by whoever writes the node's children." The const assert at `:1056`
  becomes `const _: () = assert!(Fold::EXACT_BOUND < VALUE_FIELD as u64);`, and its comment
  (`:1050-1055`) is rewritten: the 32-bit fields hold every count with the all-ones field left
  over, which is the value marker; lifting `EXACT_BOUND` past it is the key-format change `to_bits`
  documents.
- **Two private helpers**: `fn count_field(bound: Bound<()>) -> u32` (`Lit(n)` → `field(n)`,
  `Value(())` → `VALUE_FIELD`) and `fn field_bound(field: u32) -> Bound<()>` (its inverse:
  `VALUE_FIELD` → `Value(())`, else `Lit(u64::from(field))`).
- **`to_bits` (`:996-1006`)**: total. `u128::from(self.stride) << 80 | u128::from(self.monoid_byte())
  << 72 | u128::from(self.binder.slot()) << 64 | u128::from(count_field(self.lo())) << 32 |
  u128::from(count_field(self.hi()))`. For a literal pair this is today's expression term for term
  (`field(lo)`, `field(hi)`). Doc `:991-995` (`# Panics … until A4 gives it a marker`) → "A bound
  that is a value writes [`VALUE_FIELD`]'s all-ones count; its child is not in the bits." (Write
  `VALUE_FIELD` in backticks, not a link: it is private.)
- **`admits_a_value` (new, private)**: `fn admits_a_value(fold: Fold, lo: Bound<()>, hi: Bound<()>)
  -> bool` = `fold.monoid.takes_a_value_bound() && fold.stride == 1 && [lo.lit(),
  hi.lit()].into_iter().flatten().all(Fold::is_exact)`. Doc: "What a value bound asks of its fold,
  as one predicate: [`FoldHeader::from_bits`] refuses where it fails, and `with_a_value` panics
  there naming which part, as `KnownRange::admits` is to `KnownRange::new`." `with_a_value`
  (`:836-865`) keeps its three asserts and messages (pinned by A3's tests).
- **`from_bits` (`:1019-1037`)**: decode stride, monoid, binder as today; `lo = field_bound(((bits
  >> 32) & 0xffff_ffff) as u32)`, `hi = field_bound((bits & 0xffff_ffff) as u32)`; refuse a zero or
  past-the-bound stride as today; then `match (lo, hi)`: two literals → today's
  `KnownRange::admits(&(lo..hi)) && (hi - lo).is_multiple_of(stride)` path, unchanged; otherwise
  `admits_a_value(fold, lo, hi).then(|| Self::store(fold, lo, hi))` where `fold =
  Fold::strided(monoid, binder, stride)`. Doc: the refusal list gains "a value marker under `ANY`,
  `ALL` or `SEQ`, at a stride other than 1, or beside a literal past the bound", and "an end field of
  all ones" moves from the refusal list to "a value bound".
- **Inverse law.** For every header `h`, `from_bits(h.to_bits()) == Some(h)`: a literal pair as
  today; a value bound stores `(Held::Value, 0)` (`Held::hold`, `:741-746`) and `store` rebuilds
  exactly that. A literal never decodes as a value (no count is `u32::MAX`) and a value never as a
  literal.
- **`VALUE_BOUND`** (`:1094-1102`): deleted (its two readers, `to_bits` and `lower.rs:51`, go).
- **`KNOWN_RANGE`** (`:1104-1113`): text → `"this fold's bounds are literals: a lattice's fold, a
  production glyph's before B2, or a test's (docs/language/2026-10-20-a-bound-is-a-value.md)"`, and
  its doc drops "a `kernel!` fold before A6" and "a schedule's (codegen refuses a value bound first,
  with [`VALUE_BOUND`])": after A4 neither has a non-test reader (§9's grep lists the rest).
- **`FoldHeader`'s doc** (`:758-802`): "A bound that is a value keeps its count at 0, so that equal
  bounds are equal in every field `Eq`, `Hash` and `Ord` read" gains ", and writes the all-ones
  marker in its bits".

### 4.2 `canonical` (`pixelflow-ir/src/key.rs`)

No code change: tag 10, `to_bits` (now total), the body's canonical id, then each value bound's
(`key.rs:454-461`) — already written by A3. Docs: `:447-453` "a bound that is a value is a child,
keyed after the body" → "a bound that is a value writes its header field's marker and its child's
key after the body's"; the parse-back argument (`:196-203`, "every tag has a fixed encoding
length") → "every tag has a fixed encoding length given its payload (an n-ary node carries its
count, a fold's header says how many children follow it, `FoldHeader::arity`)". A uniform bound
contributes a dense slot, not its default, so every `n` is one key (L9).

### 4.3 Every serializer that wrote a header without its children

The grep (`grep -rn 'fold_header(\|FoldHeader::from_bits\|\.to_bits()' --include=*.rs` filtered to
fold headers) finds these, and only these:

| File:line | Writes today | A4 |
|---|---|---|
| `pixelflow-ir/src/key.rs:454-461` (`canonical`) | header + body + value children | docs only (§4.2) |
| `pixelflow-graphics/src/test_support.rs:37` (`program_digest`) | header via `to_bits`, every child via `arena.children` (`:47-49`) | no code change: `to_bits` stops panicking; a pin (§8.6) |
| `pixelflow-graphics/src/render/cell_grid.rs:1166-1178` (`dump_arena`) | `R {bits} {body}` | `R {bits} {body}` then ` {d}` per value child, lo's first; the comment at `:1166-1168` → "a value bound's child follows the body, as the header's marker says (`FoldHeader::to_bits`)" |
| `pixelflow-graphics/tests/production_glyph_arena_dump.rs:167-174` (`dump_arena`) | `R {bits} {body}` | same as `cell_grid.rs` |
| `pixelflow-pipeline/src/collapse_bench/corpus.rs:268-279` (encode) | `R {bits} {body}` | same line format; `HEADER` (`:113`) → `"# pixelflow collapse corpus v5"`; `SUPERSEDED_HEADERS` (`:116-129`) gains `("# pixelflow collapse corpus v4", "v4's `R` line held a fold's body alone, so a fold whose bound is a value had no child to name")`; the comment at `:268-271` rewritten |
| `pixelflow-pipeline/src/collapse_bench/corpus.rs:497-505` (decode) | `["R", bits, body]`, `attach(empty)` | `["R", bits, body, values @ ..]`; after `from_bits`, refuse naming the file unless `values.len() == header.arity() - 1` ("{path}: fold bits {bits} name {} value bounds and the line gives {}"), then `header.attach(values.iter().map(|v| id(v)))`; the comment "`from_bits` names literal bounds only" goes |
| `pixelflow-pipeline/src/training/corpus.rs:403-415` (`write_node`) | refuses a value bound (`io::Error` "before A4's format version") | writes `TAG_REDUCE`, the bits, the body's id, then each value child's id (`u32` le), lo's first; the refusal goes |
| `pixelflow-pipeline/src/training/corpus.rs:603-615` (`read_node_into`) | bits, body, `attach(empty)` | after the body, read `header.arity() - 1` more ids through `resolve_child`, then `header.attach(those)`; a short record is `UnexpectedEof` as for any short record |
| `pixelflow-pipeline/src/bin/corpus_gaps.rs:456-506` (`hash_cons`) | `Key::Reduce(u128, u32)` from `to_bits` and the body | `Key::Reduce(u128, Vec<u32>)`: the bits, then the body's and each value child's mapped id; the build closure maps them and calls `push_reduce(header.fold(), lo, hi, body)` with `header.attach(mapped values)`; the comment `:496-498` → "the key is the header's bits and the children's ids" |
| `pixelflow-compiler/src/emit.rs:577-611` (`arena_to_tokens`'s `node`) | `from_bits(#bits)` + `attach(::core::iter::empty())` | the hole lookup becomes `arena.known_range(id).and_then(|range| context.holes.range_of(range))` (drop `KNOWN_RANGE`, `:51`, `:591`); a literal fold's tokens are **unchanged** (`attach(::core::iter::empty())`); a fold with value bounds emits `__header.attach([#(#values),*])` with `values` its value children's `node_var`s, lo's first. `kernel!` cannot produce one before A6, so this is reached only by §8.7's unit test |

Not serializers of a header, checked and left: `pixelflow-pipeline/tests/shader_and_psychedelic_arena_dump.rs:207-262`
(panics on any `Reduce`: its kernels have none); `pixelflow-pipeline/src/bin/extraction_witnesses.rs:442`
and `strict_label_output_class_recount.rs:169` (print the header's `Display` and every child);
`pixelflow-compiler/src/emit.rs:701-728` (`fold_terms`, a literal header for its monoid only);
`pixelflow-ir/src/expr.rs:327` (copies every child); `pixelflow-codegen/src/jit_cache.rs:262` (a
comment).

**Format versions.**
- **Training corpus**: its version is the hash of `CorpusFormat::SCHEMA` (`training/corpus.rs:82-114`):
  editing the text is the bump. Add, after the `7=Buffer …` clause: `9=Reduce(FoldHeader::to_bits
  u128 le: a count field of all ones marks a bound that is a value; body ExprId le; then one ExprId
  le per bound so marked, lo's first)`. The module doc's tag list (`:22-23`) gains `7=Buffer` (refused
  at write) and `9=Reduce`. Every on-disk corpus is then refused with the regeneration hint; none is
  checked in (`git ls-files` holds no corpus file).
- **Collapse corpus**: v4 → v5 as above (a v4 file holds no value bound, but §4 of the plan says a
  format that stores a `Value` bound bumps; no fixture is checked in).
- **Arena dumps** (`# pixelflow arena dump v1`, `cell_grid.rs:1116`,
  `production_glyph_arena_dump.rs:132`): **not bumped**. A literal fold's `R` line is unchanged, and
  no reader of v1 parses an `R` line at all (`pixelflow-search/src/runtime.rs:2283`,
  `pixelflow-search/src/arena_corpus.rs:26`, `pixelflow-pipeline/src/bin/corpus_gaps.rs:208` each
  panic on it as unparseable), so nothing that reads v1 reads differently (§11, J7).

### 4.4 Consumers of `EmitTraffic.trips` outside codegen

- `pixelflow-codegen/examples/byte_probe.rs:306-307`: unchanged (prints `Display`).
- `pixelflow-graphics/examples/constant_traffic.rs:44-56`: the `{trips:>8}` column unchanged;
  `t.dynamic_memory_ops()` (`:49`) prints through `map_or_else(|| "an estimate".to_owned(),
  |n| n.to_string())`.
- `pixelflow-pipeline/src/collapse_bench/mod.rs:309-314` (`dyn_weighted`) and `:329`: read
  `trips.known()` and `dynamic_memory_ops()` with `.expect(LITERAL_FOLDS)`, a new `const
  LITERAL_FOLDS: &str = "a collapse corpus kernel's folds have known ranges: its kernels are
  production glyphs and the synthetic families, every bound a literal, until B2 measures a glyph
  whose bound is a value";`. A value-bounded kernel in a collapse corpus is refused loudly at
  measurement rather than reported with an estimate as a count.

---

## 5. `Kernel::over_values` public (decision 7)

`pixelflow-ir/src/kernel.rs`:
- `:816` `pub(crate) fn over_values` → `pub fn over_values`; delete the `cfg_attr(not(test),
  expect(dead_code, …))` at `:812-815` and at `:865-868` (`user_bound` stays private, now reached
  from a `pub` caller). Signature unchanged: `(monoid: Monoid, lo: &Kernel, hi: &Kernel, body: impl
  FnOnce(&Kernel) -> Kernel) -> Self` (A3 spec J16).
- Doc (`:777-811`), kept, plus:
  - after the first paragraph: "A bound that is a value is read per call: `n` a [`Uniform`] is one
    compiled program for every `n`, and the fold runs `n` trips. Outside `[0, 2²⁴]` at run time —
    a uniform set to `2.5`, `−1`, NaN or `2²⁴ + 2` — the fold is NaN (L5), and a pair that runs
    backwards at run time is the identity (L2)."
  - the `# Panics` list stays A3's (by name: `BoundArithmetic`, `BoundNotANumber`,
    `BoundOutOfDomain`, `BoundSort` for a mask literal, then from `close_over` `BoundVaries`,
    `BoundSort` under `ANY`/`ALL`/`SEQ`, and the full index space); its links become explicit paths
    (`crate::fold::BoundRefusal::BoundArithmetic`, …), per A3 rulings C5.
  - a runnable doctest:
    ```rust
    use pixelflow_ir::{Kernel, Monoid, Uniform};
    let n = Uniform::new(8.0);
    // Σ_{i < n} X·i, one program for every n.
    let sum = Kernel::over_values(Monoid::SUM, &Kernel::constant(0.0), &n.kernel(), |i| {
        Kernel::x().mul(i)
    });
    assert_eq!(sum.uniforms().len(), 1);
    ```
- `Kernel::over`'s doc (`:704-751`): "a bound that is a value is `Kernel::over_values`'" becomes an
  intra-doc link [`Kernel::over_values`].
- **The user-surface refusals** are A3's and unchanged: `user_bound` (`:857-880`) refuses
  arithmetic, a `Param`/`Buffer` root, a constant naming no count and a mask literal before the body
  runs; `close_over` refuses L3 (`X`, `Y`), a value bound under `ANY`/`ALL`/`SEQ`, and a full index
  space. No new refusal: a per-call value outside the domain is the guard's (NaN), never a
  construction-time panic, and a host-written count past 2²⁴ is A6's.
- `pixelflow-core` re-exports `Kernel` (`pixelflow-core/src/lib.rs:87`), so `over_values` is public
  there with no change.

---

## 6. The extraction-side hole (A3 spec J24): options, and a recommendation for review

**The hole.** Extraction picks one member per class by cost (`Settlement`,
`extract.rs:1913-2040`). A bound's pick is not constrained to keep L3, so a bound class whose
cheapest member reads a coordinate the class does not vary with (the class fact is the meet over
members) is materialized through it, and `ExprArena::push_reduce` refuses the fold with
`BoundVaries` (`arena.rs:1012-1020`): loud, inside `jit_cache::compile`. Two pins show it
(`fold_rules.rs:2201-2213`, no lawful member: `i + X` beside `i + Y`; `fold_rules.rs:2228-2257`, a
dearer lawful member: `Pow(i, 2)` beside `((i·i) + X) − X`). Before A4 no production path carried
a value bound (the key refused it); after A4 `jit_cache::compile` does, so the hole becomes
reachable by a lawful user kernel if saturation ever leaves such a class.

**How reachable it is.** Not by any bound A4-B3 produce, as far as can be argued: user bounds are
leaves (`BoundArithmetic` at the user surfaces until T6), a leaf costs 0 and every op costs more, so
a leaf class's cheapest member is the leaf; the rules' only bound arithmetic is `HalveFold`'s
`binder + s` (one op), and a coordinate-reading member equal to it must cancel the coordinate, so
it needs more ops, each priced per sample (`LatticeShape::evals` of an X-reading form) where the
lawful one is priced per call or per outer trip. That is an argument, not a proof (a
`LatticeShape::POINT` extraction prices every form once), and T6 brings user bound arithmetic.

| Option | What | Cost | Residual |
|---|---|---|---|
| **(a) Leave it loud** | nothing; reword the two pins' docs ("A4 decides") to state it | 0 lines | a lawful kernel can panic in `jit_cache::compile` if a bound class gets a cheaper coordinate-reading member; the argument above says not before T6 |
| **(b) Constrain the pick at the settle** (recommended) | a class read as a value bound settles only on a candidate whose **chosen-form variance** (`Priced::var`, which both pricers already compute, `extract.rs:2111`, `:2368`) keeps L3 for every fold reading it; with none, extraction refuses by name | ~40 lines (`and_or.rs`, `extract.rs`) + two pins | a lawful member whose own child class settled on a coordinate-reading member (two levels down) is still refused, loudly, from the settlement; research paths (`Extraction::from_backfill`, `try_swap`, `IncrementalExtractor`) are not constrained and refuse at `push_reduce` as today |
| (b2) Constrain and propagate | as (b), and propagate the constraint to every class reachable from a bound class through members whose fact-variance keeps it | +100 lines: a fixpoint over classes (admissibility shrinks as the constraint grows, so a worklist); constrains a class shared with a body to its lawful member everywhere | complete up to stale facts (A3 spec J27); deferred to T6, when bound arithmetic makes two-level cancellations possible |
| (c) A per-use pick | the bound use materializes through its own lawful pick, the class's other uses through the cheapest | breaks one-node-per-class: `Selection` (`extract.rs:657-666`), both DP pricers, the claim/price audit (`costed`), the reranker's choice maps | rejected |
| (d) Repair at materialization | `Selection::to_arena` (`extract.rs:934-990`) rebuilds an unlawful bound through lawful members, as `Site::member` does for rules | the arena holds a term the DP never priced, so the claimed cost and the term's price disagree (gate 2x's `extracted_latency_prior_cost` and `costed`'s audit) | rejected |
| (e) The `Reranker` seam | a reranker that swaps unlawful bound picks | no `Reranker` ships (`extract.rs:230`), and a reranker is a research path the runtime tier never sets | rejected |

**Recommendation: (b). This is a decision for review.** Reasons: it uses the variance the DP
already computes, which is exactly what materialization checks (the e-graph's `ENode::variance`
is never narrower than the arena's transfer; A3 rulings S2 records the one place it is wider, a
known empty fold, which only makes the settle stricter); it keeps one pick per class; it costs
nothing on a literal kernel (no class is a value bound, so the check never runs and every pick,
price and `SETTLEMENT_DIGESTS` row is unchanged); and it turns the dearer-lawful-member pin from a
panic into a lawful extraction. If review picks (a), §3.5's search changes, §8.5's tests and
§10's egraph intent line are dropped, and the two pins' docs say the hole is left loud with the
reachability argument above; nothing else in this spec changes.

**(b) exactly.**

`pixelflow-search/src/egraph/and_or.rs`:
```rust
/// A class some candidate reads as a fold's value bound, the fold, and the end: what settlement
/// asks of the class's chosen form (L3 of docs/language/2026-10-20-a-bound-is-a-value.md).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct BoundUse { pub(crate) live: Live, pub(crate) fold: Fold, pub(crate) end: BoundEnd }
```
- `AndOrGraph` (`:130-155`) gains `bounds: Vec<BoundUse>`, doc "every class some candidate reads
  as a value bound, sorted by `live`; empty for a graph whose folds are all literal".
- `AndOrGraph::of` (`:167-257`), inside `for tag in egraph.tags(class)` after `let node =
  egraph.node(tag)` (`:187`): `if let Some((fold, lo, hi, _)) = node.as_reduce()` (`node.rs:412`),
  for `(BoundEnd::Lo, lo), (BoundEnd::Hi, hi)`, push `BoundUse { live:
  live_of.position(egraph.find(child)), fold, end }` for each `Bound::Value(child)`. After the loop:
  `bounds.sort_by_key(|u| u.live); bounds.dedup(); bounds.shrink_to_fit();`.
- `pub(crate) fn bound_uses(&self, live: Live) -> &[BoundUse]` by two `partition_point`s.

`pixelflow-search/src/egraph/extract.rs`:
- `Settlement` (`:1913-1929`) gains `refused: Option<(EClassId, BoundRefusal)>`, doc "the first
  candidate a bound's L3 refused, and why: what `run` reports when the root is left unsettled for
  want of a lawful one"; `new` sets `None`.
- `relax` (`:1944-1964`), right after `let priced = self.pricer.price(candidate, node);`:
  ```rust
  let refusal = self.problem.graph().bound_uses(live).iter()
      .find_map(|u| u.fold.admits_bound(u.end, priced.var).err());
  if let Some(refusal) = refusal {
      self.refused.get_or_insert((class, refusal));
      return;
  }
  ```
  (The pricer's recorder has already seen the candidate priced; a refused candidate is priced and
  not taken, and research recorders see that.)
- `run` (`:1966-2040`), before the root assert at `:2017-2023`: `if !settled[root] { if let
  Some((class, refusal)) = self.refused { panic!("{refusal}: extraction settled e-class {} on no
  member whose chosen form keeps it, so the fold it bounds has no lawful term", class.0) } }`.
- `Reranker`'s doc (`:222-229`) gains: "A reranker's swaps are not held to a value bound's L3; a
  term that reads a coordinate in a bound is refused when it is materialized
  (`ExprArena::push_reduce`)."
- Imports: `pixelflow_ir::fold::BoundRefusal` in `extract.rs`; `pixelflow_ir::fold::{Bound,
  BoundEnd, Fold}` in `and_or.rs`.

---

## 7. Byte identity on today's kernels

Today every bound in every production and pinned kernel is a literal (A3, plan:606-608; B2 is
what changes that). For such a kernel each changed site does exactly what it did:

| Site | Why it moves no byte, key, price or saturation result |
|---|---|
| `FoldHeader::to_bits` | a literal pair writes `field(lo)`, `field(hi)` as before; the marker only for a value. Same `u128` → same canonical bytes, `KernelKey`, JIT-cache key, `kernel!` tokens, corpus bytes, `program_digest` |
| `FoldHeader::from_bits` | a literal pair takes today's branch; only all-ones fields (`None` before) gain a meaning |
| `passes::legalize`'s guard | `guard_value_bounds_owned` moves the arena back untouched when no node is value-bounded (by value: no clone, no rebuild, no re-dedup) |
| `arena_to_schedule` | `lo.map(map_child)` of a `Lit` is the `Lit`; P1 and P2 run only when a value-bounded fold is reachable (`any_value_fold`); the `lattice` accumulation reads `Write`s already scanned; no def, `ValueId` or variance changes |
| `operands`/`structural_children` | a `Lit` bound yields nothing, so the carve, ownership, `assert_closed` and `Layout` see today's reads |
| `select.rs` `fold` | on a known range: `bound(Lit(lo))` is `constant(lo as f32)` = today's `constant(bounds.start as f32)` (the node's literals are the range's ends: `KnownRange::of` builds from them, `FoldHeader::new` refused a reversed pair); same for `end`; stride `range.stride()`; the LaneOp order is unchanged; `Trips::Known(parent · len)` equals today's `u64` below overflow (saturating, and no pinned kernel's product approaches 2⁶⁴) |
| `regalloc` | `LoopFacts.trips = Known(n).weight() = n` |
| `EmitTraffic` | same numbers in a new type; `Display` prints them bare, so `byte_probe`'s text is unchanged |
| extraction (§6 (b)) | `bounds` is empty, so `bound_uses` is `&[]`: no candidate is refused, every relax is today's; picks, prices, `SETTLEMENT_DIGESTS`, gate 2x's fields unchanged |
| saturation | no rule, budget or class count changes: gate 2 unchanged |
| pricing | unchanged (A3's, `cost.rs`, `extract.rs:1652-1730`) |
| `arena_to_tokens` | `known_range(id).and_then(range_of)` takes the branch `expect(KNOWN_RANGE)` then `range_of` took, and a literal fold emits the same tokens |
| serializers | a literal fold's `R` line and record are unchanged; only the collapse corpus's header line (v5) and the training corpus's identity move, and no corpus is checked in |

**Gates** (the harness in `/home/user/pixelflow/.wip/gates/`, README there; tags `a4-*`): gate 1
`byteprobe.sh` (both tiers, `cmp`-equal), gate 1b `gate1b.sh` (29 keys and digests), gate 2
`gate2.sh` and 2x `gate2x.sh` (saturation and extraction telemetry), all exit 0 against the bases
verified at `a747770`. Gate 5 `measure.sh`: report total Ir, `saturate` and `extract_dag_over`
inclusive against the `a747770` base (6,310,238,494 / 2,076,952,032 / 1,139,515,928); the
expectation is within +0.1 % of total (one enum match per candidate in `AndOrGraph::of`, one empty
slice per relax, one `nodes().any` scan per legalize); a rise over +0.5 % is a finding to explain
before submitting, as in A3.

---

## 8. Tests

### 8.0 Where they run

New JIT tests go in **`pixelflow-codegen/tests/value_bounds_jit.rs`** (`#![cfg(any(target_arch =
"x86_64", target_arch = "aarch64"))]`, a whole-file gate, which `check-cfg-encapsulation.sh`
allows), against scalar Rust. Tiers:
- **x86-64, AVX2 and AVX-512**: CI's `ISA matrix` job (`cargo run -p xtask -- isa-matrix --clippy
  --smoke`), whose smoke set runs `pixelflow-codegen`'s tests under `PIXELFLOW_ISA=avx2` and
  `=avx512` (a tier the runner lacks is reported NOT RUN, `xtask/src/main.rs:655-661`). Locally:
  `PIXELFLOW_ISA=avx2 cargo test -p pixelflow-codegen --test value_bounds_jit` and `=avx512` (this
  host has AVX-512).
- **aarch64 NEON**: executed by the `Test on macos-latest` leg (Apple-silicon runner, `cargo nextest
  run --workspace`); type-checked by `Cross-target ABI (aarch64)` (`cargo check --target
  aarch64-unknown-linux-gnu --all-targets -p pixelflow-codegen …`, which executes nothing). This
  host has no qemu and no aarch64 target, and A4 changes no emitter, so no local NEON run is
  required (plan §6's qemu clause).

Helpers in the file: `fn count(v: f32) -> Kernel { Uniform::new(v).kernel() }`; `const SHAPE:
LatticeShape = LatticeShape::new([64, 4])`, `const ORIGIN: [f32; 2] = [1.0, 2.0]` (integer origin,
so every value below is an integer below 2²⁴ and exact in any summation order: assertions are
`assert_eq!` on bits where the answer is finite, `is_nan()` where it is NaN);
`fn run_linked(kernel: &Kernel, shape) -> Vec<f32>` (through `jit_cache::compile`, the block
`linked.uniforms.iter().map(|d| d.default)`, `ctx = [block, ORIGIN]`, pitch = width);
`fn run_raw(kernel: &Kernel, shape) -> (Vec<f32>, EmitTraffic)` (through `emit::compile(arena, root,
shape)` on `kernel.parts()`, the block `arena.uniforms()` in slot order); and a watchdog `fn
within<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> T` that runs `f` on a thread
and `recv_timeout(Duration::from_secs(60))`s it, failing with "a fold whose bound is outside its
domain never hangs" (so a hang is a failed assertion, not a CI timeout; the worker's send is
`if let Err(e) = tx.send(..) { panic!(..) }`, never `let _ =`). Every call goes through `within`.
Oracles are explicit folds from the identity (`(0..n).fold(0.0f32, |a, k| a + …)`), not
`Iterator::sum` (whose float seed is not the JIT's `+0.0`).

### 8.1 `pixelflow-ir`

`src/fold.rs` tests (`:1500-2255`):
- `a_value_bound_writes_the_marker_and_reads_back` (replaces `a_value_bound_has_no_bits_until_a4`,
  `:2118-2126`): for `(Value, Lit 4)`, `(Lit 3, Value)`, `(Value, Value)` under `SUM` stride 1:
  `from_bits(h.to_bits()) == Some(h)`, and the value's field is `0xffff_ffff`.
- `a_value_header_decodes_from_its_layout_assembled_by_hand`:
  `0x0000_0000_0001_0200_0000_0000_ffff_ffff` is `SUM`, slot 0, `lo = Lit(0)`, `hi = Value`, and
  `to_bits` writes it (beside `a_header_decodes_from_its_layout_assembled_by_hand`, `:1887`).
- `an_end_field_of_all_ones_is_a_value_bound` (replaces `from_bits_refuses_an_end_field_of_all_ones`,
  `:1589-1600`): the same `hi_ones` and `lo_ones` now decode, to `hi` resp. `lo` a value.
- `from_bits_refuses_a_marker_its_fold_cannot_take`: the marker under `ANY`, `ALL`, `SEQ`; with stride
  2; beside a literal `EXACT_BOUND + 1`; and a field of `u32::MAX - 1` (past the bound, not the
  marker): each `None`.
- `from_bits_inverts_to_bits_on_every_header_shape`: every monoid × {literal pair `0..8`, `Value/Lit`,
  `Lit/Value`, `Value/Value`} that `FoldHeader::new` admits: round trip.

`src/key.rs` tests (`:525-1009`):
- `a_value_bounded_folds_key_record_is_tag_ten_its_bits_and_its_children`: the bytes assembled by
  hand, `[10] ++ bits(marker).to_le_bytes() ++ body id ++ bound id`.
- `every_count_is_one_key`: `[0, u)` with two uniform instances of different defaults: equal
  `canonical(..).key` and `KernelKey`.
- `which_end_is_a_value_is_in_the_key`: `[u, 4)` and `[0, u)` differ.

`src/passes/guard.rs` tests:
- `a_kernel_with_no_value_bound_moves_back_untouched`: the returned arena's `len()` and root equal the
  input's (identity fast path, §7).
- `a_value_bounded_fold_is_read_only_by_its_guard`: root is `If(dom, If(Lt(0, u), R, 0), NaN)`; `R`
  is the original fold; `R` has exactly one reachable reader.
- `a_literal_bound_adds_no_domain_test`: `[0, u)`: `dom` reads `u` only (count `Le` nodes: 2).
- `two_value_bounds_are_each_in_the_domain`: `[m, n)`: `dom = BitAnd(d(m), d(n))`; `[u, u)`: one `d`.
- `the_guard_varies_with_the_bounds_and_the_fold_with_its_body`: `compute_arena_variance`: `dom`,
  `run` `CONST` for uniforms; `guarded`'s variance equals `R`'s (body over `X`).
- `the_identity_arm_is_the_monoids`: `SUM`, `PRODUCT`, `MIN`, `MAX` → `0`, `1`, `+∞`, `−∞`.

`src/passes.rs` tests: `legalize_guards_a_value_bounded_fold_inside_the_lattice` (legalize
`X + Σ_{k<u} k` at `[4, 2]` with 4 lanes: the `Write`'s value reaches the guard; the lattice's `SEQ`
folds have literal ranges).

`tests/value_bounds.rs` (public API): the module doc's "The private pieces (`over_values`, …)" →
"The rooted variance walk is pinned in the crate's own unit tests"; `a_value_bounded_kernel_has_no_key_until_a4`
(`:146-155`) → `a_value_bounded_kernel_has_a_key` (`KernelKey::of` on the fixture returns, and equals
the key of the fixture rebuilt with another default for `u`); `over_values_is_public`
(`Kernel::over_values(SUM, &Kernel::constant(0.0), &Uniform::new(8.0).kernel(), |i| i.clone())`'s arena
holds one `Reduce` whose `hi` is a `Value`); `over_values_refuses_arithmetic_at_the_public_surface`
(`#[should_panic(expected = "BoundArithmetic")]`, `hi = u + 1`).

The existing `over_values_*` unit tests in `src/kernel.rs` (`:1814-2070`) stay; the doctest of §5
runs under `cargo test --doc -p pixelflow-ir`.

### 8.2 `pixelflow-codegen/tests/value_bounds_jit.rs` — the gate list, verbatim

Each bullet of plan:643-650, as a named test:

- **"a uniform `n` at 0, 1, 7, 1000, `2²⁴` (in the domain)"**:
  - `a_count_in_the_domain_runs_its_fold`: `Σ_{k<n} (X + k)` over `SHAPE` through `run_linked`, `n ∈
    {0, -0.0, 1, 7, 1000}`; oracle `(0..n).fold(0.0, |a, k| a + (x + k as f32))` per sample.
  - `a_count_at_the_exact_bound_runs_every_trip`: `Σ_{k<n} 1` at `LatticeShape::POINT`, `n = 2²⁴`:
    exactly `16777216.0` (2²⁴ trips once per call; the `CONST` fold is hoisted, P-b).
- **"`2²⁴ + 2`, NaN, 2.5, −1 (NaN, no hang)"**:
  - `a_count_outside_the_domain_is_nan_and_never_hangs`: the first kernel, `n ∈ {16777218.0, NaN,
    2.5, −1.0, +∞, −∞, 1e30}` (no subnormal: whether one reads as `0` depends on the
    thread's denormal mode, `pixelflow_core::FastMathGuard`, not on the guard): every sample NaN, inside the watchdog.
  - `a_literal_a_rule_made_outside_the_domain_is_nan`: a hand-built arena, `push_reduce(SUM, Lit(0),
    Value(Const(2.5)), X + i)` (kept a value by `push_reduce`, A3 rulings N4), through `emit::compile`:
    NaN, not a panic (plan:637-638).
- **"uniforms 5 and 3 as `lo` and `hi` (the identity)"**:
  - `a_pair_that_runs_backwards_is_the_identity`: `over_values(⊕, &count(5.0), &count(3.0), X + k)` for
    `SUM`, `PRODUCT`, `MIN`, `MAX`: `0`, `1`, `+∞`, `−∞` per sample; and `[4, 4)`.
  - `a_value_lo_starts_the_fold`: `[3, 7)` from two uniforms: `Σ_{k=3}^{6} (X + k)`.
- **"a triangular `Σ_i Σ_{j<i}`"**:
  - `a_triangular_sum_runs_each_inner_fold_to_its_outer_index`: `Kernel::sum_over(8, |i|
    Kernel::over_values(SUM, &zero, i, |j| X + j))` through both `run_raw` (the guard on a binder
    bound, once per outer trip) and `run_linked` (the optimizer may peel the outer fold and make the
    inner bounds known, L8); oracle `Σ_{i<8} Σ_{j<i} (x + j)`.
  - `a_triangular_sum_under_a_count`: the outer bound a uniform `n ∈ {0, 5}` and NaN.
- **"a uniform-bounded fold whose body reads no coordinate, so the guarded fold is hoisted to
  `Scope::Body` and the carve sees the `If`, not the `Reduce`, as its def"**:
  - `a_fold_that_reads_no_coordinate_is_guarded_once_per_call`: `X + Σ_{k<n} k` at `SHAPE` through
    `run_raw`, `n ∈ {0, 7, NaN}`: values, and `traffic.trips` holds exactly one
    `Trips::Estimated(Fold::ESTIMATED_TRIPS)` (a loop run once per call; inside the lattice's loops its
    trips would be `Estimated(32 · rows · batches)`), and no other estimate.
- **"a uniform-bounded fold whose body varies with X and whose `hi` is read only by the loop"**:
  - `a_bound_read_only_by_the_loop_reaches_its_latch`: `Σ_{k<n} w(X, k)` through `run_raw`, where
    `w` is a wall of 48 terms `(X + k)·c_m`, `c_m = m + 1`, summed in both orders (as
    `tests/loops.rs:28-81`), so every vector register is reused inside the body and `hi` survives
    only if its interval spans the loop; `n ∈ {0, 1, 7}` exact, NaN.
- **"one program and one code pointer for every `n`"**:
  - `every_count_is_one_program`: the first kernel with fresh `Uniform`s at `n ∈ {0, 1, 7, 1000}`
    through `jit_cache::compile`: `Arc::ptr_eq` on every `Linked::kernel`, and each evaluates to its
    own `n`'s answer with its own block.
- **"`2²⁴ + 1` is not a case here"**: no test (plan:650).

Plus, per monoid outside `SUM`: `each_monoid_runs_under_its_guard`: `MAX_{k<n}(X − k) = X`,
`MIN_{k<n}(X − k) = X − (n − 1)`, `Π_{k<n}(k + 1) = n!` (`n ≤ 10`, exact) at `n ∈ {0, 1, 7}`, NaN at
`2.5`.

### 8.3 `pixelflow-codegen` unit tests

`src/program/lower.rs` tests, replacing `a_value_bound_is_refused_by_name_before_it_is_scheduled`:
- `a_value_bound_is_an_operand_of_its_fold`: the hand-built `If(Lt(0, u), Σ_{i<u} i, 0)`: the
  schedule's `Reduce` has `hi: Bound::Value(v)` with `v` the uniform's def, and `operands` yields `v`.
- `an_unguarded_value_bounded_fold_fails_loudly`: `Σ_{i<u} i + 1` (read by an `Add`):
  `#[should_panic(expected = "other than as an If's true arm")]`.
- `a_bound_on_a_lattice_binder_fails_loudly`: a hand-built `Write { row: 1, col: 2, lane: 3, value }`
  under literal `SEQ` folds over slots 1-3, `value = If(Lt(0, Var(slot 1)), Σ_{i<Var(slot 1)} i, 0)`
  (`push_reduce` admits it: slot 1 is not `COORDS`): `#[should_panic(expected = "a lattice binder's")]`.

`src/pipeline.rs` tests (beside `schedule_for`, `:154-162`), each through `schedule_for` then `Nest`:
- `a_guard_on_a_uniform_count_is_computed_once_per_call`: `X·Σ_{k<n}(X + k)` (body over X): every
  `ScheduledOp::Compare` def (the guard's) is in `scoped.body.schedule`; the `Reduce`'s `ScopeFold`
  has a column fold as `parent`; the two guard `If`s are in that column fold's schedule (P-c).
- `a_fold_whose_body_reads_no_coordinate_is_carved_in_the_body`: `X + Σ_{k<n} k`: the `Reduce` and
  both guard `If`s are in `scoped.body.schedule`, the `Reduce`'s `ScopeFold.parent` is
  `Scope::Body`, and the row fold's `outer` names the outer `If`'s `ValueId`, not the `Reduce`'s (P-b).

### 8.4 `src/emit/traffic.rs` tests

- `a_known_trip_count_prints_as_its_number`: `format!("{}", Known(42)) == "42"`,
  `format!("{:>5}", Known(42)) == "   42"`.
- `an_estimate_prints_marked`: `"~32"`, and `format!("{:>5}", Estimated(32)) == "  ~32"`.
- `nesting_an_estimate_is_an_estimate`: `Estimated(32).nested_in(Known(4)) == Estimated(128)`,
  `Known(3).nested_in(Known(4)) == Known(12)`.
- `nesting_saturates`: `Known(u64::MAX).nested_in(Known(2)) == Known(u64::MAX)`.
- `dynamic_memory_ops_weights_each_scope_by_its_trip_count` (`:202-217`): trips
  `[Known(1), Known(6), Known(42)]`, result `Some(1 + 3·6 + 5·42)`.
- `dynamic_memory_ops_is_no_count_over_an_estimate`: one `Estimated` scope → `None`.

### 8.5 `pixelflow-search` (with §6 (b))

`src/egraph/fold_rules.rs`:
- `extracting_a_bound_class_whose_cheapest_member_reads_a_coordinate_is_refused_by_name`
  (`:2228-2257`) → `extracting_a_bound_class_takes_its_cheapest_member_that_keeps_l3`: no
  `should_panic`; `extract(&f.eg, class, &CostModel::latency_prior())` returns; the arena's inner
  fold has `hi: Bound::Value(b)` with `compute_arena_variance(&arena)[b] ==
  Variance::from_var(binder(1).var())` and `arena.node(b)` the `Pow` spelling; `free_index(root) ==
  None`. Docs drop "A4 decides".
- `extracting_a_bound_class_with_no_member_that_keeps_l3_is_refused_by_name` (`:2201-2213`): stays
  `should_panic`, `expected = "on no member whose chosen form keeps it"` (the settlement's refusal,
  not `push_reduce`'s); doc says extraction refuses it.

`src/egraph/and_or.rs` tests: `a_class_read_as_a_value_bound_records_its_fold` (the `over_a_joined_bound`
shape: `bound_uses(live_of(bound))` is one `BoundUse { fold: inner, end: Hi }`);
`a_literal_graph_has_no_bound_use` (`bounds.is_empty()`).

`src/egraph/extract.rs` tests: `a_shared_bound_class_takes_its_lawful_member_for_every_use` (the
bound class of the `Pow` fixture also read as a body operand: one pick, the lawful one, and the
extraction succeeds).

### 8.6 `pixelflow-graphics`

- `src/test_support.rs`: `a_value_bound_is_in_the_program_digest` (two arenas, `[0, u)` and `[0, v)`
  with `u`, `v` different uniform slots: different digests; neither panics).
- `src/render/cell_grid.rs` (its test module, beside `dump_arena`, `:1090-1190`):
  `the_dump_writes_a_value_bounds_child` (dump a `[0, u)` fold to a file under
  `std::env::temp_dir()` named for the test, read it back: the `R` line has four fields and the last
  is the uniform's dense id).
- `tests/production_glyph_arena_dump.rs`: the same test, not `#[ignore]`.

### 8.7 `pixelflow-pipeline` and `pixelflow-compiler`

- `src/training/corpus.rs`: `round_trip_a_value_bounded_fold` (a triangular `Σ_{i<4} Σ_{j<i} j` by
  `push_reduce` with `Value(Var(i))`: written and read back structurally equal, `subtree_eq`);
  `a_fold_record_with_an_all_ones_end_names_no_fold` (`:757-772`) → split into
  `a_fold_record_marking_a_value_reads_its_child` (the same bits, now with a child id after the body:
  decodes) and `a_fold_record_with_an_end_past_the_bound_names_no_fold` (`hi = 0x0100_0001`:
  `InvalidData`); `schema_identity_changes_when_the_schema_text_does` (`:949`) stays green.
- `src/collapse_bench/corpus.rs`: `a_value_bounded_kernel_round_trips` (a kernel with an `A` line and
  `[0, A)`: encode, decode, `subtree_eq`); `a_v4_corpus_names_the_version_it_is` (beside `:741`,
  `:762`).
- `src/bin/corpus_gaps.rs` tests (`:240`): `hash_cons_keeps_a_value_bound` (two folds differing only
  in their bound's child stay two nodes; two identical ones merge).
- `pixelflow-compiler/src/emit.rs` tests (`:863`): `a_value_bounded_fold_emits_its_children` (an
  arena with one uniform and `[0, u)`, `EntryContext { arguments: vec![quote!(3.0f32)], structural:
  &[], holes: &Holes::default() }`, `node(..)` for the fold contains `attach ([__e`), and
  `a_literal_fold_emits_no_children` (contains `attach (:: core :: iter :: empty ())`).

---

## 9. Order of work, and the grep-gated checklist

**Order** (each step leaves its crates' tests green; never `cargo test --workspace`; `df -h /`
before each build, stop under 2 GB):

1. **IR key**: `fold.rs` (§4.1), `key.rs` docs, the IR tests of §8.1 for the key;
   `cargo test -p pixelflow-ir`. `VALUE_BOUND`'s deletion breaks `pixelflow-codegen` until step 4:
   do steps 1-4 before building codegen.
2. **IR guard**: `passes/guard.rs`, `passes.rs` (§3.1), tests; `cargo test -p pixelflow-ir`;
   `cargo check -p pixelflow-ir --no-default-features`.
3. **IR `over_values`**: `kernel.rs` (§5), `tests/value_bounds.rs`; `cargo test -p pixelflow-ir`
   (doctests included).
4. **Codegen**: `program/lower.rs`, `program/mod.rs`, `emit/traffic.rs`, `emit/mod.rs`,
   `emit/build.rs`, `emit/select.rs`, `emit/regalloc/local.rs`, examples; unit tests (§8.3, §8.4),
   then `tests/value_bounds_jit.rs` (§8.2); `PIXELFLOW_ISA=avx2 cargo test -p pixelflow-codegen` and
   `=avx512`; `bash scripts/check-emit-boundary.sh`; `RUSTDOCFLAGS="-D warnings" cargo doc -p
   pixelflow-codegen --no-deps --document-private-items`.
5. **Search** (if §6 (b)): `and_or.rs`, `extract.rs`, the pins; `cargo test -p pixelflow-search
   --all-features`.
6. **Serializers and consumers**: graphics, pipeline, compiler (§4.3, §4.4); `cargo test -p
   pixelflow-graphics --lib`, `cargo test -p pixelflow-graphics --test production_glyph_arena_dump`,
   `cargo test -p pixelflow-pipeline` (lib, bins, the corpus tests), `cargo test -p
   pixelflow-compiler`, `cargo test -p pixelflow-core` (it bakes through `jit_cache`).
7. **Docs** (§10).
8. **Gates**: §7's gates 1, 1b, 2, 2x, 5; `cargo clippy --workspace --all-targets --all-features --
   -D warnings` (on this host's clippy 1.97 the three `must_use_candidate` reports A3 rulings R1 name
   are the only allowed errors, and no bare `#[must_use]` is added on an `impl Iterator`);
   `cargo fmt --all -- --check`; `bash scripts/check-intent.sh`, `check-doc-paths.sh`,
   `check-cfg-encapsulation.sh`, `check-emit-boundary.sh`, `check-bin-declarations.sh`; rustdoc for
   `pixelflow-ir` and `pixelflow-search` with no warning outside
   `/home/user/pixelflow/.wip/a4/rustdoc-base.<crate>.txt` (`cargo doc -p <crate> --no-deps
   --document-private-items 2>&1 | python3 -I /home/user/pixelflow/.wip/a4/rustdoc_norm.py --compare
   …`); the checklist and greps below.

**The `Reduce {` site checklist.** Copy `/home/user/scratch/a3-wip-archive/.wip/a3/gen/{checklist.py,
classes.json,README.md}` to `/home/user/pixelflow/.wip/a4/gen/` and run `python3 -I
/home/user/pixelflow/.wip/a4/gen/checklist.py --root /home/user/pixelflow` (its own root is the
archive's). At `a747770` it reports 175 sites, `A4` 15, unclassified 0, stale 0. Add a class
`changed-by-A4` to the copy's class list; reclassify each of the 15 `A4` rows with a reason:
`changed-by-A4` for `select.rs:171` (`definition`), `program/mod.rs:148` (doc), `:398`
(`operands`), `compiler emit.rs:581`, `cell_grid.rs:1169`, `production_glyph_arena_dump.rs:167`,
`corpus_gaps.rs:493`, `collapse_bench/corpus.rs:272`; `already-correct` (with the reason) for
`select.rs:147`, `program/mod.rs:438` (chains `all_operands`), `scopes.rs:213`, `:250`, `:268`
(match the def, read no bound), `test_support.rs:37` (feeds every child), `key.rs:454` (writes every
child). Also reclassify `training/corpus.rs:404` (`changed-by-A3` → `changed-by-A4`) and
`program/lower.rs:351` (likewise), and add rows for the new sites (`passes/guard.rs`'s hook and
`has_a_value_bound`, `lower.rs`'s `value_folds` scan, test files). Gate: `A4` count 0,
UNCLASSIFIED 0, STALE 0.

**Greps** (each, at head, gives exactly what is stated):

| Grep | Expected at head |
|---|---|
| `grep -rn 'VALUE_BOUND' --include=*.rs .` | nothing |
| `grep -rn 'KNOWN_RANGE' --include=*.rs pixelflow-* \| grep -v '/tests/\|mod tests\|#\[test\]'` then read each hit | non-test uses only in `fold.rs` (the definition), `passes/lattice.rs` (lattice folds), `pixelflow-codegen/src/program/lower.rs` (the lane fold, `:340`), `pixelflow-pipeline/examples/glyph_phase_split.rs` (production glyph before B2); none in `emit/select.rs` or `pixelflow-compiler/src/emit.rs` |
| `grep -rnE 'known\(\|known_range\(\)\.expect\|KnownRange::of\([^)]*\)\.expect\|\.expect\(KNOWN_RANGE\)' pixelflow-codegen/src` | `lower.rs`'s lane-fold `known_range(expr_id).expect(KNOWN_RANGE)` only |
| `grep -rnwE 'A4' --include=*.rs pixelflow-*` and `grep -rnw 'A4' --include=intent.md .` | nothing (A3 rulings B1's rule, one step on) |
| `grep -rn 'expect(dead_code, reason = "pub(crate) until A4' pixelflow-ir` | nothing |
| `grep -rn 'iter::empty()' pixelflow-pipeline pixelflow-graphics pixelflow-compiler/src` | the compiler's literal branch only |
| `grep -rn 'fold_header(id).to_bits()' --include=*.rs .` | each hit writes the node's value children too (§4.3's table) |
| `grep -rn 'FoldHeader::from_bits' --include=*.rs pixelflow-pipeline pixelflow-compiler` | each decoder attaches the children its header counts |
| `grep -rn 'trips' pixelflow-codegen/src/emit` | every per-loop count is a `Trips`, or `LoopFacts.trips` read through `weight()` |
| `grep -rn 'dynamic_memory_ops' --include=*.rs .` | each caller handles `Option` |
| `grep -rn 'no corpus encoding before\|has no bits\|no code and no key' --include=*.rs .` | nothing |

---

## 10. Docs

### 10.1 The plan (`docs/language/2026-10-20-a-bound-is-a-value.md`)

- **A4 bullet** (plan:627-650): append a `**Done:**` line stating facts only (A3 rulings C16: no J
  labels, nothing cites `.wip`): byte_probe empty in both columns under `PIXELFLOW_ISA=avx512` and
  `=avx2`; the 29 font programs' keys, digests and saturation telemetry, extraction fields included,
  unchanged; the guard is `pixelflow_ir::passes::legalize`'s last pass (`passes/guard.rs`); lowering
  asserts after collapse that a value bound reads no lattice binder and that a value-bounded fold is
  read only as its guard's true arm; `FoldHeader::new` already refusing a stride other than 1 beside a
  value, no stride assert is added; a value bound is an operand of its fold, `lo` seeding the loop
  and `hi` read at the latch, live there by the allocator's own rule; `Trips::{Known, Estimated}`
  each carry a number, an estimate printed `~n`; `EmitTraffic::dynamic_memory_ops` is `None` over an
  estimate; the key's marker is a count field of all ones; the collapse corpus is v5 and the training
  corpus's schema names the marker; the arena dumps stay v1; and the extraction decision as review
  rules it (with (b): "extraction settles a class read as a value bound only on a member whose chosen
  form keeps L3, and refuses by name when none does").
- **§3.3** (plan:342-383), amended to what was built: "a legalize pass in `program::build`" →
  "the last pass of `pixelflow_ir::passes::legalize` (`passes/guard.rs`)"; "Either the carve adds
  `hi` to the fold's `outer`, or the `Reduce` def's interval spans the loop; A4 picks one" → "The
  `Reduce` def's interval spans the loop: the latch's compare reads `hi`, so the allocator keeps it
  live to the latch (`emit/regalloc/local.rs:44-46`), and `operands()` yields the bounds so the carve
  and ownership see them"; the post-collapse assert sentence as §1.5; `Trips::{Known(u64),
  Estimated(u64)}`.
- **§3.2** pricing paragraph (plan:329-336), with (b): one sentence, "Extraction settles a class read
  as a value bound only on a member whose chosen form keeps L3 (A4)."
- **§6 A3 bullet** (plan:640-641's sentence in the A4 bullet, "A4 decides whether a bound's pick is
  constrained"): replaced by the decision.
- **§5's rows marked A4**: edited in their own files (§10.2-§10.3); the table itself is unchanged.

### 10.2 `CLAUDE.md`, `pixelflow-core/README.md`, `docs/DENOTATION.md`, the totality doc, BACKLOG

- **`CLAUDE.md:180`** ("The language is a DAG"), plan §5 row 1 "A4 as built": replace "Today every
  bound is a constant, so a trip count that must change is a recompile through the shape-keyed
  cache. The plan that makes bounds values placed by their variance is
  docs/language/2026-10-20-a-bound-is-a-value.md." with "A bound is a value placed by its variance: a
  literal is the fold's data, which extraction may unroll; a per-call count is a uniform, computed
  once per call, so one program serves every count; outside `[0, 2²⁴]` the fold is NaN
  (docs/language/2026-10-20-a-bound-is-a-value.md)."
- **`pixelflow-core/README.md:63-67`**: "Bounded reductions have constant extents today, …
  [a bound is a value](…)." → "A reduction's bounds are values: a literal, a per-call count
  (`Kernel::over_values` with a `Uniform`), or an enclosing reduction's index, and none may be a
  function of a coordinate or of a lattice binder. One compiled program serves every count, and a
  count outside `[0, 2²⁴]` makes the reduction NaN. General unbounded recursion is intentionally
  outside the language: **the language is a DAG**, and the fold is its one binder ([a bound is a
  value](../docs/language/2026-10-20-a-bound-is-a-value.md))."
- **`docs/DENOTATION.md:60-68`** (plan §5 "A4 cost"): "Every one is a constant today; [a bound is a
  value](…) makes a fold's bounds values placed by their variance. Every program's cost is a
  closed-form function of static extents, computable without running it. … the budget decides
  whether the number is acceptable." → "A fold's bounds are values placed by their variance ([a bound
  is a value](language/2026-10-20-a-bound-is-a-value.md)), each fixed before the lattice is
  traversed. Every program's cost is a closed-form function of values known before the traversal,
  computable without running it; nothing evaluates it per call, and the static model prices a fold
  whose range is not a known constant at an estimate (`Fold::ESTIMATED_TRIPS`). "Estimatable by the
  cost model", "a valid shader" and "`bake` terminates" are one property. Total is not cheap:
  totality gives a number, and a value-bounded program is priced at the estimate, never reported over
  budget."
- **`docs/DENOTATION.md:125-127`** (plan §5 `:123-125`): "Loop bounds, extents and ranges are never
  uniforms." → "A uniform may bound a fold, as a per-call count ([a bound is a
  value](language/2026-10-20-a-bound-is-a-value.md) §1.2); it is never an extent, and the lattice's
  extent stays in the program and the key (§7)."
- **`docs/DENOTATION.md:142-144`** (plan §5 `:140-142`): "the range is a static natural; the body is
  the only child; the monoid, binder and range are the node's identity" → "each bound is a literal,
  which the node holds, or a term, a child after the body, fixed before the lattice is traversed and
  reading no coordinate and no lattice binder; outside `[0, 2²⁴]` a fold over `+`, `×`, `min` or
  `max` is NaN, and `∧`, `∨` take literal bounds only; the monoid, the binder and the literal bounds
  are the node's identity".
- **`docs/DENOTATION.md:174`**: "every range constant" → "every bound fixed before the lattice is
  traversed".
- **Totality doc** (`docs/language/2026-07-24-totality-and-the-cost-model.md`), plan §5 `:191-200`
  and `:214` (now `:199-203`, `:216-217`): the "Dynamic behavior survives" bullet → "**Dynamic
  behavior survives; only coordinate-dependent *bounds* die.** A per-call count is a value, not a
  static max: `Σ_{k<n}` with `n` a uniform is one program for every `n`, priced at the cost model's
  estimate ([a bound is a value](2026-10-20-a-bound-is-a-value.md) §1.5). A count that varies with
  the pixel (ray-march until you hit the surface) still becomes **static max + early-out**: iterate
  up to `N`, mask off finished lanes, and the cost is the static worst case `N`. This is the same
  trick GPUs already play, so almost no expressiveness is lost."; item 5 → "Every eliminator that
  could iterate (`⊕_{i∈D}`, `iterate[N]`, field reify) carries a bound fixed before the lattice is
  traversed, a literal or a per-call value and never a coordinate, so the cost walk is closed-form in
  values known per call."
- **`docs/BACKLOG.md:164`** (C8), plan §5 "A4 (NaN)": append "Since a-bound-is-a-value A4 a bound
  that is a value is not refused at construction, where its count is not known: outside `[0, 2²⁴]`
  its fold is NaN in the language (L5, the guard `passes::legalize` ends with), and a host-written
  count is refused at its write in A6."
- **`docs/BACKLOG.md:32`** (plan table): move A4 to the landed column ("…, A3 (value bounds in the IR
  and the e-graph), A4 (value bounds in codegen, and the key)"); next: "A1's the-language-is-kernel
  rows after #1323; **A5** (`<const N>` deleted); A6, A7; B1-B5 after typed-arena T3 and item 2".

### 10.3 The ontology (`.claude/skills/ontology/language.md`)

- **"Binding time" (`:116-143`)**, plan §5 "A4": "Follows" gains: "A fold's bound is placed by its
  variance like any value (a-bound-is-a-value §1.2): a uniform bound is a per-call count, so one
  program serves every count."
- **"Fold (Reduce)" (`:417-470`)**: "Is": "its body is its only child until a bound is a value
  (a-bound-is-a-value A3)" → "its children are its body and each bound that is a value". "Is not":
  "Dynamically bounded." → "Bounded by a coordinate or a lattice binder, or by its own binder (L3); a
  per-call bound is a value, not a dynamic bound." "Follows": "A fold costs `len × body`, and its size
  for budgeting is also `len × body`." → "A fold costs `n·body + (n−1)·combine` plus its value bounds
  once, `n` its known length or `Fold::ESTIMATED_TRIPS`; outside `[0, 2²⁴]` a value-bounded fold is
  NaN, by the guard `passes::legalize` ends with, which is an `If` whose arm holds the loop." "Lives"
  gains `pixelflow-ir/src/passes/guard.rs` and `Trips` (`pixelflow-codegen/src/emit/traffic.rs`).
- **"Range (trip count)" (`:499-528`)**: rewritten. "Is: a fold's half-open `[lo, hi)` with a
  stride; each end a literal (node data) or a value (a child): a per-call count or an enclosing fold's
  index. The range is the half that carries the load: peeling over a range leaves the body unchanged
  and therefore shared." "Is not: a function of a coordinate, a lattice binder or the fold's own
  binder (L3). Past 2²⁴: a literal is refused at construction, a value makes the fold NaN. Not
  implicitly zero-based. Not a value of its own in the IR: a range node shared by sibling folds is
  decision 9 (a-glyph-is-a-formula §4.2)." "Follows: a known range is in the key; a value bound is a
  dense slot, so one program serves every count. Constructors refuse a literal range past 2²⁴
  (`KnownRange::admits`). The allocator weighs carries by `Trips`, an estimate for a loop whose range
  is not known. `bucketed_trip_count`'s padding existed only because counts were syntax, and goes
  (B2)." "Lives": as today, plus the guard and `Trips`; drop "Today every bound is a literal".
- **"Totality" (`:909-925`)**: "Ranges are constant, and uniforms are never extents." → "Every trip
  count is fixed before the lattice is traversed, so a program's cost is closed-form per call; a
  per-call count is a value, never a coordinate, and the lattice's extent is never a uniform." "Is"'s
  quote stays (it is JP's); "Is not"'s "dynamic trip counts" → "coordinate-dependent trip counts".

### 10.4 `intent.md` files (each ≤ 80 lines, `bash scripts/check-intent.sh`)

- **`pixelflow-ir/intent.md`** (80 lines, at the budget): `:72-73` becomes one line, "- `Var(u8)` means
  an axis or a binder by range; `Binder` a `u8`, against the 64-bit control-plane rule." (the "until
  A4" sentence goes); the fold bullet in "Owns" (`:13-18`) gains "and its domain: outside `[0, 2²⁴]` a
  value-bounded fold is NaN, by the guard `passes::legalize` ends with", wrapped so the file stays at
  80 lines.
- **`pixelflow-codegen/src/program/intent.md`** (64): "Owns" `:20-22` → "A fold's bounds in the
  schedule: `ScheduledOp::Reduce` carries the arena node's `lo` and `hi`; a value bound is an operand
  of the fold (`operands`), its loop's seed and its latch's end, and selection reads a known range with
  `KnownRange::of`."; "Never: No open term" gains "or a value-bounded fold read other than as an `If`'s
  true arm"; "Enforced by: the three `fails_loudly` tests" → five.
- **`pixelflow-codegen/src/emit/intent.md`** (61): "Contract": `EmitTraffic`, `ScopeTraffic`, `Trips`.
- **`pixelflow-codegen/intent.md`** (58): "Contract": `emit` (`compile`, `CompiledKernel`,
  `CompileResult`, `EmitTraffic`, `ScopeTraffic`, `Trips`).
- **`pixelflow-codegen/src/emit/regalloc/intent.md`** (76): no change.
- **`pixelflow-search/src/egraph/intent.md`** (75), with (b): "Owns: Extraction and cost" gains "a class
  read as a value bound settles only on a member whose chosen form keeps L3 (`Settlement::relax`), and
  extraction refuses by name when none does"; "Enforced by" gains
  `extracting_a_bound_class_takes_its_cheapest_member_that_keeps_l3`. Stays ≤ 78.

### 10.5 Code docs under changed code

Every doc §3-§6 names, plus: `fold_rules.rs:2202-2207` and `:2228-2238` (the pins' "A4 decides"),
`training/corpus.rs:408` (the refusal message goes with the refusal), `corpus_gaps.rs:496-498`,
`collapse_bench/corpus.rs:268-271` and `:503`, `cell_grid.rs:1166-1168`, `lower.rs:463-468` (the old
test's doc goes with it), `program/mod.rs:147`, `program/intent.md:21` (§10.4). §9's `A4` grep is the
check that none is missed.

---

## 11. Judgement calls, each with the alternative rejected

- **J1. The guard is `passes::legalize`'s last pass.** It is L5/L2 in the language's own ops, a
  documented-domain legalization like the transcendental expansions, and it needs `rebuild_arena` and
  a `Write` rebuild that only `pixelflow-ir` can do. Rejected: (a) a codegen `Pass` between `Legalize`
  and `Lower`, which would need `push_write` and `rebuild_arena` made public (a visibility change) or a
  second copier (a copy is a future divergence); (b) selection emitting the guard (`select.rs`), which
  hand-assigns variance, bypasses `Own`/`Layout` and the arm machinery, and is what plan:365-368
  rejected; (c) a hang-proof loop with no guard (`Lt(step, min(hi, 2²⁴))`), which never makes the
  NaN L5 asks for and spends instructions on the latch.
- **J2. The guard runs last in `legalize`, after `expand_transcendentals`.** Nothing after it can
  rebuild the `If` lowering asserts. Rejected: right after `lower_dwrt` (before `collapse`), where
  `collapse`, `pack` and the expansions would rebuild the guard before lowering reads it.
- **J3. Every value bound gets a domain test, a binder's included.** One rule; a binder is in `D` by
  its enclosing fold's guard, so `d(i)` is redundant but costs four ops per outer trip. Rejected:
  special-casing a `Var` bound, a second rule for a saving no profile asked for.
- **J4. P2 checks the reader's shape, not the guard's contents.** It is the property placement
  rests on (the loop inside the arm), cheap, and catches the realistic failure (an arena that skipped
  `legalize`). Rejected: recognizing the full guard (`If(dom, If(Lt(lo, hi), R, id), NaN)`) in
  lowering, a second statement of `guard.rs`'s shape that the two would have to keep agreeing on.
- **J5. No "stride ≠ 1 only with known bounds" assert.** `FoldHeader::new` and `from_bits` refuse
  the shape, and a node holds only a `FoldHeader`. Rejected: the plan's assert, which could not fire.
- **J6. `Trips::Estimated` carries a number.** Nested loops multiply (plan §4), and the allocator
  weighs a nested fold's reads by the product; `Estimated` alone would lose it. Rejected: the plan's
  bare `Estimated`, with the number recomputed wherever a weight is needed.
- **J7. The arena dumps stay v1.** A literal `R` line is unchanged and no v1 reader parses an `R`
  line. Rejected: a v2, which changes nothing any reader does. The collapse corpus does bump (its
  decoder parses `R` and its v4 files are superseded by name) and the training corpus's identity
  moves with its schema text, which is how that format versions itself.
- **J8. `dynamic_memory_ops` returns `Option<u64>`.** "No measurement shows an estimate as a count"
  is then the type; its two callers decide (print "an estimate", refuse with `LITERAL_FOLDS`).
  Rejected: returning `Trips` (a memory-op count is not a trip count) or summing weights (an estimate
  printed as a count).
- **J9. `-0.0` as `lo` is not normalized.** It is the count 0 (`Fold::count`), the result is right
  except for a body that tells signed zeros apart, which CLAUDE.md's rows already cover. Rejected:
  seeding with `lo + 0.0`, an instruction per fold entry for a case nothing asked about.
- **J10. The guard is not idempotent, and the module doc says so.** `legalize` runs it once, and a
  second application is correct (the outer guard passes exactly when the inner does). Rejected:
  recognizing an already-guarded fold, the J4 recognizer again.
- **J11. `over_values` keeps A3's refusals and adds none.** A per-call value outside `D` is the
  guard's NaN; a construction-time panic could not know the value. Rejected: refusing a `Uniform`
  whose default is outside `D` at construction, which judges the default (a placeholder) instead of
  the call's value.
- **J12. `arena_to_tokens` keeps a literal fold's tokens byte for byte** (`attach(::core::iter::empty())`)
  and emits the children array only beside a value. Rejected: one `attach([...])` form for every
  fold, which changes every `kernel!` expansion for nothing.
- **J13. §6: (b), for review**, with (a), (b2), (c), (d), (e) and their costs in §6.
- **J14. The JIT tests use exact integer arithmetic and a watchdog.** Exactness makes the oracle
  order-free (the optimizer may reassociate); the watchdog turns "no hang" into a failing assertion.
  Rejected: tolerances (which could hide a trip off by one at large `n`) and relying on nextest's
  per-test timeout (minutes, and reported as a timeout, not as the law it breaks).
