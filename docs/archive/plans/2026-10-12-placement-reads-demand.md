> **Archived 2026-10-15 — status `Superseded`.** Superseded by `docs/language/2026-10-15-kernels-evaluate-kernels.md`. Proposed D1 placement-by-demand design; its literals, demand and projected regions are withdrawn by the 2026-10-15 kernels ruling. Kept for rationale; not authoritative about the tree.

# Placement reads demand

## Metadata

- **Author**: JP (direction), Claude (draft)
- **Status**: `Proposed`. **Superseded in part by `docs/language/2026-10-15-kernels-evaluate-kernels.md` (2026-10-15):** demand weighting, arm probability and projected regions are deleted, so this plan's `Literal`, `Demand`, projected regions and demand-weighted price are withdrawn, and the per-batch test of a mask is the shape of `If` rather than a case to remove.
- **Created**: 2026-10-12
- **Verified against**: `7acdcaee`, which is `dd621dab` (#1367) plus β. β merged as #1368 (`d3a8fbe0`), whose tree is `7acdcaee`'s, so every citation here holds on `main` at `d3a8fbe0`.
  - β touched `pixelflow-codegen/src/program/{guards,layout,mod,ownership,scopes}.rs`, `pixelflow-core/tests/{guard_fold_price,guard_sibling_fold}.rs`, the ontology's `../../../.claude/skills/ontology/codegen.md`, `../../../.claude/skills/ontology/language.md` and `../../../.claude/skills/ontology/meta.md`, and the demand, one-conditional and selection plans. Every other `file:line` here is the same at `dd621dab`.
  - The task named `8f916d2d` (#1366). `dd621dab` is one commit past it and touches only `jit_cache.rs`, `pipeline.rs`, `pixelflow-ir/src/passes/lattice.rs` and `pixelflow-ir/src/variance.rs`.
- **Citations**: `file:line` at `7acdcaee`, always with the symbol. Paths are relative to `pixelflow-codegen/src/` unless another crate is named. `sel:` is `docs/archive/plans/2026-10-08-selection-is-a-phase.md`. Nothing here cites uncommitted work.
- **Ontology**: `.claude/skills/ontology` as of `7acdcaee`. The entries this plan meets are ../../../.claude/skills/ontology/codegen.md Schedule, Placement, Root, Park, Frame, Arm ownership, Guard and Block; ../../../.claude/skills/ontology/compiler.md Variance and Demand; ../../../.claude/skills/ontology/language.md Binding time, Uniform (adjective) and If; ../../../.claude/skills/ontology/lattice.md Region. Where this plan changes an entry, §7 names the line.
- **Name**: "D1" is the parallel track's name for this work: "D1: placement that reads demand" (one-pipeline :922-923; the-language-is-kernel :840; ../../../.claude/skills/ontology/codegen.md:93). The same name means three other things:
  - selection-is-a-phase D1, the deletion of the legacy pipeline (#1362);
  - one-conditional-three-lowerings D1, mask ⟹ index range (`pixelflow-ir/src/mask_support.rs`; ../../../.claude/skills/ontology/language.md If, Lives);
  - the-language-is-kernel decision D1, binding times (:643).

  This plan is called "placement reads demand" so it collides with none of them.
- **Continues**:
  - [one-conditional-three-lowerings](../../codegen/2026-09-08-one-conditional-three-lowerings.md). This plan is its lowering 2, "hoist and guard" (§1, :49), built at every scope. It is that plan's D5 ("lowering 2 on the general predicate", §8, :420) restricted to conjunctions. Its §6 keeps "scope projection" (:336-337) from the plan it superseded; this is that projection, on whole mask literals (§7A.13).
  - [demand-is-a-dag-property](../../compiler/2026-09-07-demand-is-a-dag-property.md), superseded by the above. Its §2 "Scope projection" (:141) is the rule built here. Its recorded question, JP's *"Why isn't the DAG over the whole program?"* (:28-29), is answered per scope, not by one pass (§7A.14, Q4).
  - [a-glyph-is-a-formula](2026-09-23-a-glyph-is-a-formula.md) §4.3: "an arm may own a root when it owns every scope that reads it" (:434-435), and "its jump belongs once per row, where placement already puts its *value*; the branch should land there too" (:441-442).
  - [an-integral-is-a-fold](2026-09-23-an-integral-is-a-fold.md) §1: `E(v) = π_var(v)(dem(v))` (:96). Its §4 "Placement" is where this plan departs (§7A.6).
  - [selection-is-a-phase](2026-10-08-selection-is-a-phase.md) §0.12, §0.13, D2 and D3. This plan generalizes §0.13 (§7A.5). Selection, the allocator and the per-ISA emitters are not changed, apart from one count (§3.9).
- **Builds on β** (#1368, "an arm owns the roots only its loops read"). JP approved β with α (#1367) and γ. It was CL1 in an earlier draft of this plan, and CL1 is now the instrument (§10). β:
  - deleted `Ownership::of`'s `external` pin, the `roots` fields of `ScopeRegion` and `ScopeFold`, and `place_roots`' computing-scope search;
  - added `ScopedSchedule::assert_closed` (`program/scopes.rs:80`), which `lay_out` ran per scope and, since #1371, the `Own` pass runs (`ownership.rs:89`);
  - retired the zero-park rule, on an argument this plan makes false (§7A.2);
  - grew `pixelflow-codegen/src` by 21 production lines and 96 in-file test lines.
- **Comes after γ** (#1372). JP approved the series α, β, γ, then this plan. γ carried the arena's variance into codegen, deleted `schedule_variance` and typed `ScheduledOp::Var(Binder)`. Nothing here needs γ, but nothing here reorders the series.
- **Rebased on #1371**, the passes as one typed pipeline (`Legalize -> Lower -> Nest -> Own -> Layout -> Select -> Allocate -> Assemble`). `Own` is the ownership pass this plan changes, and `Flow::of` already takes an `Ownership` rather than computing one, so that item of CL2 is done. What CL2 still changes is the order: `Own` is a `ScopedSchedule::analyze` (`program/mod.rs:242-264`), one independent call per scope; this plan makes it children-first, handing each scope's `Demand` to its parent.
- **Departs from recorded decisions.** Each is named in §7 with its source. None is undone silently.

**The request.** The parallel track names the work: *"D1: placement that reads demand. It puts a piece's row work inside its row-scope band guard. That is 79% of U_g's executed work"* (one-pipeline §5, :922-923). JP approved this write-up on 2026-10-12, in the series α, β, γ, then this plan. §1.1 says what the 79% does and does not measure.

**JP's rulings this plan is held to.**

- There are three binding times: constants; the uniform block, constant per collapse; and coordinates, per sample. Work that depends only on uniforms runs once per collapse.
  - Rows and columns are both coordinates. So the ruling places uniform-only work in the body. It does not decide row against column. That rule is the ontology's: "a frame-uniform value is placed in the body, a row-uniform one in the row fold", and "a row-uniform guard belongs at row scope" (../../../.claude/skills/ontology/language.md Uniform (adjective), Follows, :230-233); "Placement is 'outermost scope binding all its binders'" (collapse-is-a-fold §2.2, :180); and Hoist (LICM) (../../../.claude/skills/ontology/codegen.md:125-144).
- Placement and policy do not go into the per-ISA emitters: *"None of those things are going in the damn emitter."*
- `If` contains an if (CLAUDE.md). `If(m, a, b)` means `if m then a else b`. A uniform mask takes one arm, a jump. The blend is the lane-varying fallback.
- Selection plan §0.12: an `If` is a blend, its arms are blocks, and there is no bound. §0.13: an `If`'s arm blocks go immediately before the `If`; nothing permutes the schedule.

**The diagnosis, in one sentence.** An arm is a literal of its mask, and a literal is testable at every scope from its mask's binding time inward; the code ties each arm to the scope of the `If` that opens it, so work hoisted out of that scope has nothing left to guard it.

That one sentence explains these, each verified at `7acdcaee`:

1. **The font's row work runs on every row.** A def stays in a scope's schedule when its variance allows (`extract_folds_bound_by`, `program/scopes.rs:494-497`), and `place_roots` removes from a fold every def whose variance does not name its binder (:266-277). Neither reads demand. Each piece's `up`, `t0` and `t1` (`pixelflow-graphics/src/fonts/loop_blinn/program.rs:75`, :80-81) vary only by row, so they are row-fold defs. Their only readers sit in the column fold, under the piece's band `If` (:97). The row fold has no node for that `If`'s arm. So nothing guards them.
2. **An arm owns a root only within one scope.** β lets an arm own what only its loops read (`Ownership::of`'s doc, `program/ownership.rs:15-17`). A loop reads at its own `Reduce`, so a root read only under an `If` arm *inside* the loop is owned by the `Reduce`'s region, not by that arm. β is the one-scope case of this plan.
3. **A fold's reads lose their guard.** `FoldReads` makes a loop's `Reduce` def a reader of everything the loop names, at the `Reduce`'s own region (`program/guards.rs:13-45`; `reads`, :96-106). The arms inside the loop are forgotten. A read of a value the scope does not define is dropped (`ownership.rs:124-126`, `positions.get(value) else return`).
4. **A uniform or row-uniform mask is tested per batch.** Selection opens every arm where the `Flow` puts it (`emit/select.rs:118-123`; `open_arm`, :332-348). The `Flow` puts an arm only at its `If` (`program/layout.rs:46-58`, `arm.at == pos`). An `If` whose arm reads `X` sits in the column fold, so its test runs once per batch, whatever its mask's variance. This plan does not change that test (§7A.11, §7A.12).
5. **Two doc rules defend against reads of skipped work.** "The frame prologue is never guarded" (../../../.claude/skills/ontology/codegen.md:382, :425), and β's Park argument, "the join after the arm never holds it" (../../../.claude/skills/ontology/codegen.md:157-165). A guard that every reader's own guard implies makes those reads impossible, so the first rule has nothing left to defend, and the second needs a new argument (§7A.2, §7A.3).

The price disagreement the review also found (`pixelflow-search/src/egraph/extract.rs:1878-1888`) is a separate fact about the scope, not about demand. §8 treats it, and leaves its fix to one-pipeline M14.

---

## 1. The problem

### 1.1 What is measured, and what is not

One-pipeline §1.4, under "What unrolling does not fix", measured `'8'` at 32 px on AVX-512, with per-address counts:

- "Row-tier work is 79% of U_g's executed instructions: 109 of 138.5 per pixel" (:506-507).
- "Placement lifted it above the band select, into the row fold, where nothing guards it" (:509-510).
- "The band select does jump: 12,311 of 12,314 exclusive values are guarded … But the arms it skips run on only 2-4 of 64 batches" (:511-513).

That measurement is motivation from a predecessor program, and nothing more:

- One-pipeline is verified against `8b7b75a` (:48). U_g is the unrolled term at that commit: "fold-free, no integral, every read constant" (:474-475).
- `8b7b75a` predates the closed form. `piece_area`, with its four `monotone_root`s, landed in `33c6509b`. The-language-is-kernel O4 makes the same caveat about its frame numbers (:1157-1159).
- U_g read every value as a constant, so its uniform-only work could fold away. The per-count programs #1339 shipped read uniforms and keep body work.
- The per-address counter behind the 79% is not in the tree.

So the per-count programs' row share is unknown. What the source does say, per piece (§9):

- **Row tier:** `s·y`, `up`, `up ± ½`, two `monotone_root`s and two clamps (`program.rs:56-58`, :74-81). That is 2 square roots, 2 exact divides and about 20 other operations per row.
- **Column tier, in the band arm:** `across`, `left`, `right`, two more `monotone_root`s, two clamps and the area terms (:74-91). That is 2 square roots, 2 divides and about 45 other operations per batch in band.
- Out of band, the column work is skipped per batch today. The row work is not.

The row share is therefore a function of each piece's band fraction and of the batches per row. CL1 computes it exactly on the host (§10).

### 1.2 Translation: the description against the code

The denotation (§2) was written before the placement code was reread. This table is what the code has for each of its parts.

| part of the denotation | at `7acdcaee` |
|---|---|
| variance `var(v)` | `Def::variance`, carried from `compute_arena_variance` since γ (#1372), which deleted `schedule_variance`, the second copy of the arena's transfer function. It is orthogonal to this plan. |
| scope by variance | `extract_folds_bound_by` keeps in a scope what its variance allows (`scopes.rs:494-497`); `place_roots` removes from a fold what does not name its binder (:266-277). It matches the denotation. It is kept. |
| demand, within one scope | `Ownership::of` (`ownership.rs:102-165`): one reverse pass, each value owned by the lowest common ancestor of the regions that read it. It matches the denotation, confined to one scope. |
| demand, across scopes | Nowhere. `FoldReads` stands in for a fold's readers, at the `Reduce`'s own region (`guards.rs:96-106`). A read of a value the scope does not define is dropped (`ownership.rs:124-126`). |
| a fold's result read | `Selector::scope` reads an accumulating fold's body root after the scope's items (`emit/select.rs:126`; `Selector::fold`, :296), and that root may be a value an enclosing scope computed (:89-92). Today that read reaches `FoldReads` only through `outer` (`ScopeFold::reads`, `program/mod.rs:205-213`), and `assert_closed` pins it (`scopes.rs:217-222`). |
| a literal | Half there. `Arm { at, mask, arm, region }` (`ownership.rs:55-66`) and `ArmBlock { arm, mask, items }` (`program/mod.rs:245-255`) name a mask and an arm, never the `If`. Ownership's regions are grown per `If` met in the scope (`ownership.rs:134-136`, `Tree::grow`). |
| projection of a literal to an outer scope | Nowhere. demand-is-a-dag-property §2's "scope projection" was never built, and `passes/demand.rs` was deleted on 2026-10-03 (../../../.claude/skills/ontology/compiler.md Demand, Lives, :437-443). |
| a block's place | `Flow::of` splices an arm's block at its `If`, into the block of the `If`'s region (`layout.rs:46-58`). A projected region has no `If` in its scope. |
| selection of a block | Generic over any `ArmBlock` in any scope. The mask is looked up through the enclosing frames (`select.rs:120`; `lookup`, :382-390). |
| the allocator at a skipped block | Already general (§3.7). |

What the code has that the denotation does not:

- `ScopeFold::outer` (`mod.rs:195-198`) and `ScopeFold::reads` (:205-213). `outer`'s production readers are `reads` and `assert_closed`. `reads`'s only reader is `Own`'s `FoldReads` (`ownership.rs:91-96`).
- `FoldReads::new`'s named-id table and filter (`guards.rs:57-90`), with its admitted over-approximation: "a leaf the body rebuilds for itself … over-approximates" (:39-42).
- The doc rule "the frame prologue is never guarded", and β's Park argument (§7A.2, §7A.3).

### 1.3 Why this is one fault

This is global code motion over a structured loop nest (Click 1995). Schedule-early is variance: the outermost scope binding every binder a value reads. Schedule-late is demand: the common ancestor of a value's uses, where a use through an `If` arm counts as that arm. GCM chooses the least deeply nested block on the path between them. On a dominator tree, the outer loop level has no block for an inner `If`'s control dependence. So "least nested" can only mean "unguarded": the value is hoisted out of the conditional into the loop's preheader and runs on paths that never needed it. This codebase hits that with the font's row work.

The fix is not a price choosing between the two bounds. It is to give the outer level the node it lacks: the inner `If`'s literal, which the outer scope can test because it already computes the mask.

---

## 2. The denotation

Written in the domain's words, before the placement code was reread. §1.2 is the translation.

### 2.1 The nest

A collapse is a nest of folds (collapse-is-a-fold §2.1). Each fold is a scope. The body is the scope that runs once per call. A scope `S` binds `B(S)`, the binders of the folds on its chain. `Ω` is the product of every binder's range. A value `v` denotes `⟦v⟧ : Ω → lane`.

The lanes are bound in every scope. A vector computes every lane, and a block runs or is skipped whole.

### 2.2 Variance and demand

Both are read off the extracted DAG. Variance is also an e-graph fact; demand is not (an-integral-is-a-fold §1, "Why they are not one pass").

- **Variance** `var(v)` is the set of binders `v` reads, computed forward. Law: if `b ∉ var(v)`, then `⟦v⟧` is constant along `b`.
- **Demand** `dem(v) ⊆ Ω` is where `v` is observed, computed backward. `dem(root) = Ω`. `If(m, a, b)` passes `dem ∧ m` to `a`, `dem ∧ ¬m` to `b` and `dem` to `m`. A `Reduce` passes its demand to its body, whose root its combine reads on every trip. Every other edge passes demand unchanged. A value with several readers joins their demands by `∨`. Law: if `ω ∉ dem(v)`, replacing `⟦v⟧(ω)` changes no output.
- **What a value needs** is `E(v) = π_var(v)(dem(v))`: the demanded points projected onto `v`'s own axes. Hoisting realizes the projection. Guarding realizes the restriction.

### 2.3 A literal, and where it can be tested

A literal `ℓ = (m, a)`, with `a ∈ {True, False}`, means "some lane of mask `m` selects arm `a`".

`ℓ` is **testable** at scope `S` iff `m` is computed at `S` or at a scope enclosing `S`: `scope_of(var m)` is `S` or an ancestor of `S`.

The test has one answer per trip of the scope that computes `m`, whichever scope inside it asks. `m` is constant along every binder it does not read (the variance law), and every binder bound inside `scope_of(var m)` is one of those. So an outer test and an inner test of `ℓ` agree on every trip they share.

### 2.4 A region, and the product law

A region of `S` is a pair `(S, g)`, where `g` is a conjunction of literals testable at `S`, written as a path from the scope's top. `⟦(S, g)⟧` is the set of trips of `S` on which every literal of `g` holds.

**Product law.** If every literal of `g` is invariant along an inner binder `k`, then `⟦(S.k, g)⟧ = ⟦(S, g)⟧ × range(k)`.

So an `If`'s arm whose mask is computed at an outer scope *is* a region of that outer scope: the same literal, seen at a coarser scope. It is not a second `If`. There is no new node, no second mask def, no second blend and no join. There are two tests of one mask value, and each serves the work placed at its own scope.

### 2.5 A placement

The placement of `v` is a block `P(v) = (s(v), g(v))`.

- **The scope** `s(v) = scope_of(var v)`: the outermost scope whose binders contain `var(v)`. This is the ontology's placement rule (../../../.claude/skills/ontology/language.md Uniform (adjective), Follows; collapse-is-a-fold §2.2, :180). For uniform-only work it is also JP's binding-time ruling: such work goes to the body. Demand never chooses the scope in this plan (§2.7 says what that leaves open).
- **The guard** `g(v) = ⋀ over the readers r of v of proj_s(v)(path_s(v)(r) ++ e(r, v))`.
  - `path_S(r)` is the literals of every block that encloses `r` from `S` inward, outermost first: `S`'s own blocks, then each fold's blocks on the way to `r`'s scope. Literals tested outside `S` are not in it. They already decide whether a trip of `S` happens at all.
  - A fold's combine is a reader of its body root, at the fold's top: its path is empty.
  - `e(r, v)` is the literal an `If`'s arm edge adds: `(m, True)` for `If(m, v, _)`, `(m, False)` for `If(m, _, v)`, and nothing for any other edge.
  - `proj_S` drops the literals that are not testable at `S`. Dropping a literal is a sound widening: `∃x. a(y) ∧ b(x) ⇒ a(y)`.
  - The meet `⋀` of paths is their longest common prefix.
- **Executions**: `v` runs once per trip of `s(v)` on which `g(v)` holds.

### 2.6 The law

**Availability.** A block `(S1, g1)` is available at `(S2, g2)` iff `S1` is `S2` or encloses it, and `g2 ⇒ g1`. This is not tree ancestry. `(row, band)` is available at `(column, band)`, although the column loop does not sit inside the row's band block.

**The law.** Every read that observes `v` runs in a block at which `v`'s block is available.

*Proof.* Let `r` read `v` other than as an `If` reads its own arm operand, and let `r` run. Every literal of `path_s(v)(r) ++ e(r, v)` held where it was tested. Projection only drops literals and the meet only weakens, so every literal of `g(v)` is among them. Each has its mask computed at `s(v)` or outside it, so `s(v)`'s test in the current trip reads the same mask value `r`'s chain read (§2.3). So `v`'s block ran in the trip of `s(v)` that encloses `r`. ∎

**The one read that does not observe `v`.** An `If`'s blend reads its own arm operand on every batch, including a batch where that arm's block was skipped. The blend is bitwise on every backend (`vandps`/`vandnps`/`vorps`, `vblendmps` under `k`, `BSL`; selection §0.12, sel:119), so it takes every lane from the arm that ran. That read is what `If` means, not an exception to the law.

**Corollaries.**

- No write is needed into a skipped block's registers or slots, and no phi. Stale bits there, NaN included, never reach a gather index or an arithmetic reader.
- Soundness never rests on demand. Arms are total, because gathers clamp (one-conditional §4). So every widening is sound, and a precise guard is an optimization.

### 2.7 What is optimal, and what is left open

- `s(v)` is the outermost scope binding `var(v)`, so nothing is recomputed along an axis `v` does not vary on.
- `g(v)` is the strongest conjunction the structure can test at `s(v)`.
- So `v` runs exactly `|E(v)|` times when `dem(v)` is a conjunction of literals testable at `s(v)`, and every inner scope has at least one trip.
- Otherwise it runs more often, by the literals `s(v)` cannot test. That is speculation. It is sound.
- **Left open: sinking `v` below `s(v)`**, into an arm whose mask varies faster than `v`. That recomputes `v` along an axis it does not vary on, and buys the deeper literal's guard. It is a real trade, and not one-sided: where the deeper scope has one trip (§6.14), a sink into `[inside, band]` runs less than `v` at `s(v)` under `[band]`.
  - The static price cannot choose it. It has no guard term (`P = 1`, §8), so a sink multiplies a value's price by the deeper scope's trips, which are at least 1. The price we have never prices a sink lower.
  - So this plan does not sink, and puts the question to JP (Q7) rather than refusing it.

In Click's terms: "least nested" is variance, and "latest" is the strongest guard on the late path that the scope can test. The guard lattice replaces the dominator tree for everything but the sink.

---

## 3. The rule

Placement is all in `program/`. `emit/` gains one count, which decides nothing (§3.9), and a reworded doc comment (§3.7).

### 3.1 Scope: unchanged

`extract_folds_bound_by` (`scopes.rs:376-499`) and `place_roots` (:227-279) stay the only code that chooses a scope, and variance is their only input. Masks are placed by the same rule as every other value.

### 3.2 Regions: a tree of literals per scope

Each scope keeps its own tree of regions (`program/tree.rs`, `Tree`; `ownership.rs:69-76`). A region is the scope's top, or a child of a region keyed by one literal.

- An `If` at region `ρ` reads its mask at `ρ` and its arm operands at its two arm regions under `ρ`, as today (`ownership.rs:134-154`).
- A fold's read under a path of literals reaches a **projected region**: `under(ρ, ℓ)`, the child of `ρ` keyed by `ℓ`, made once and then found.
- A literal is keyed by `(the If's ValueId, arm)`. Each region records the literal it was grown by, so its path is read off its ancestors.

Keying by the `If`'s `ValueId` is what makes the two column folds `pack` builds, when it builds two, one region in the row scope. They share one `Write` (`pixelflow-ir/src/passes/lattice.rs:212`, :219), so their schedules share the band `If`'s `ValueId`, and both report the same key.

Keying by literal alone, so that two `If`s on one mask share a block, is CL4. It needs a block split this plan does not (§3.6).

### 3.3 Ownership runs children first, and returns its demand

`Own` runs the folds in reverse index order, then the body. A parent's index is always below its children's (`attach_fold`'s doc, `scopes.rs:418-420`), so every child is done before its parent. Today `Own` is a `ScopedSchedule::analyze`, which visits the body first, then the folds in index order, each without the others' results (`mod.rs:242-246`). CL2 gives `Own` a children-first traversal that carries each child's `Demand` to its parent; `analyze` stays as it is for the passes that need no such hand-off.

`Ownership::of` runs one reverse pass per scope, as today, with two changes:

- **It seeds the scope's result.** For a fold whose monoid is not `SEQ`, it records the body root as read at the scope's top before the sweep, since the combine reads it there on every trip (`select.rs:126`, :296). When the root is a def of the scope, that is where the pass already puts it: a value nothing in its own scope reads is the scope's (`ownership.rs:118-121`). When it is not, the read is the case below. A `SEQ` fold has no root to read (`Selector::scope` takes `None`).
- **It returns the scope's demand on its parent:** for each value the scope reads but does not define, the region where all its reads meet, as a path of literals. Today such a read is dropped (`ownership.rs:124-126`).

The seed is the one cross-scope read no def makes. Without it, a fold whose body root is outer reports no demand for it. Selection plan D2 names that shape: "A fold whose body is another fold's result has an empty schedule" (sel:1782), and `FactorFold` builds it: "A body invariant as a whole keeps its right operand inside" (`pixelflow-search/src/egraph/fold_rules.rs:330-331`). The miscompile it would allow is §11.1.

### 3.4 A fold's reads carry their path

When the parent's pass meets a `Reduce` def at region `ρ_R`, the fold reads each value `v` of its child's demand at `under(... under(ρ_R, ℓ1) ..., ℓk)`, where `ℓ1 .. ℓk` is the path the child reported for `v`.

- If the parent defines `v`, that is an edge here, and `v`'s region is the meet over its reads, as for any value.
- If it does not, the read goes into the parent's own demand on its parent. A value read two scopes in is forwarded one scope at a time.

### 3.5 Projection

When a scope hands its demand to its parent, every path drops the literals whose mask the scope defines. The parent cannot test those. A literal whose mask an enclosing scope computes stays, so it is testable in the parent, and it travels outward until it reaches its mask's scope.

### 3.6 Layout: one rule

Every value stays in the order it was constructed, within the block that owns it.

**A block goes into the block of its nearest ancestor region that owns a value: immediately before its first reader there, or at that block's end when its first reader lies outside it. Blocks placed before one reader go outer first, then in the order of their `If`s, true arm first.**

- **For an `If`'s arm**, the first reader is the `If`, which its parent region owns. An arm-owned value is read only inside its arm, by a loop whose `Reduce` the arm holds, or by its `If` (`layout.rs:11-15`). So the arm lands immediately before its `If`, true arm first, which is selection §0.13 exactly.
- **For a projected region**, the first reader is the first `Reduce` def in the scope whose fold reads under it.
- **For a projected region nested in another** with the same reader, the reader lies outside the outer block, so the inner block goes at the outer block's end.
- A region that owns no value is not a block. A block nested in it goes into its nearest ancestor that is one.

This is not §0.13 kept verbatim. §0.13 says an `If`'s arms before the `If` is "the whole placement rule", and that a `Flow`'s arms are "in the order of their `If`s" (sel:123-124). This rule has §0.13 as its arm case and adds the projected case, whose reader is a `Reduce`. §7A.5 names the change and Q3 puts it to JP.

`Arm::at` (`ownership.rs:58-59`, "the position of the `If`") becomes the position of the block's first reader. `Flow::of`'s splice loop (`layout.rs:46-58`) stays one loop, with two changes:

- it pushes a block into the block of its parent region's nearest owning ancestor. Today it pushes into the block of the def at `at`, which for an arm is the same block;
- it takes the blocks pending at one `at` outer first. An inner block is then appended to an outer block whose members are all placed, which is its end.

Why the rule is valid for a projected block:

- Every member comes before the reading `Reduce` in construction order. The fold's body reads the member, a `Reduce` follows its body's cone in the flat schedule, and a scope's schedule keeps flat order.
- No reader outside the block comes earlier. Such a reader would have weakened the member's region below the block.
- An outer block's members never read an inner block's members, or those members' region would be the outer one. So an inner block at the outer block's end follows everything it reads there.
- The block's mask is available before the block. The `If` inside the fold reads the mask, and that read is forwarded with its own path, so the mask's region is the block's parent region or an ancestor of it (§3.8 asserts it).

With projected regions keyed by `If`, a block never needs splitting. Every member lies in the cone of one `If`'s arm operand inside the fold, so every member precedes every `Reduce` whose fold holds that `If`. Under literal keying (CL4), two `If`s on one mask in sibling folds share a block, a member read only by the later fold can be built after the earlier `Reduce`, and the block must split there.

### 3.7 Selection, the allocator and the emitters: unchanged

- **Selection.** `Selector::scope` walks `Item::Arm` in any scope (`emit/select.rs:116-123`). `open_arm` branches on the mask it finds through the enclosing frames (:332-348; `lookup`, :382-390), and mints its blocks in the scope being selected (:336). A row-scope block needs nothing new.
- **Structure.** `Builder::finish` refuses a forward branch that leaves its scope, or enters or leaves a loop (`emit/build.rs:417-437`). A projected block holds row defs, or whole loops (§6.3), so its skip does neither.
- **The allocator.** A spilled value is stored right after its definition, never at the eviction point (`emit/regalloc/local.rs:15-20`). A join takes a value's register from the paths that define it (:34-41). In `Scan::arrive`, a path on which the definition comes after the branch keeps the value: `kept &= life.def > edge.from && life.remat.is_none()` (:1148-1153). So a value defined in a skipped row block and live into the column loop is handled as an arm value read by its blend is today. On the skip path its register or slot holds stale bits, and by the law nothing that runs there reads them.
- **Untested today.** No test builds a value defined in a skippable block and live into a loop after the join. CL2 adds one (§10).
- **The `Function` doc.** It says "Dominance is relaxed in one way, and only one" (`emit/mod.rs:607-616`). A row-block value read in a column arm on the same literal is not dominated in the CFG either. Selection §2.13 already names the cause: "A correlated branch is not in the CFG" (sel:1194). The doc is restated as the law of §2.6, not extended to a second relaxation. It is a doc comment.

### 3.8 What `Own` and `Layout` assert

In production code, linear, as selection §0.6 asks of an invariant no production input can break (sel:100). `ScopedSchedule::assert_closed` (`scopes.rs:163-224`) is where they live; it is β's, extended rather than joined by a second mechanism. Today `Own` runs it before `Ownership::of` (`ownership.rs:89`). Checks 2-4 need the ownership tree, the flow and the scope's `Demand`, so CL2 runs them in `Layout`, after `Flow::of`.

1. **Order.** Every read follows its definition in flow order.
2. **Nesting.** Every def runs in the block of its own region, and every block is spliced into the block of its parent's nearest owning ancestor. One walk of the flow with a stack of open regions.
3. **Availability of a block's mask.** In the block's scope, the mask's region is the block's parent region or an ancestor of it; otherwise the mask is not a def of this scope, so an enclosing scope computes it. One `common_ancestor` query per block on the `Tree` ownership already holds. Flow order alone would pass a mask defined earlier inside a sibling block, which a skip leaves stale, and `Selector::lookup` panics only on a value never bound, not on a stale one.
4. **Closure, restated over `Demand`.** Inward: every value a scope reads, its defs' operands and its accumulating root included, is a def of it or a key of its `Demand`. Outward: every key of a fold's `Demand` is defined in the parent ahead of the fold's `Reduce`, or is a key of the parent's own `Demand`. This is β's check with `outer` replaced by `Demand`. It is not dropped.

The test-only checker `laid_out_with` (`layout.rs:150-216`) checks 1 and 2 today, quadratically. CL2 moves them into production in linear form, and `laid_out_with` shrinks to what production cannot see: hand-built fixtures that read before they define, and the random-DAG test.

### 3.9 The count that makes a guard observable

A regression that unguards the font's row work would change no value, so no test that checks values can see it. CLAUDE.md: "A gap in CI is a check to write, not a caveat to attach." So CL2 adds one count:

- `ScopeTraffic` (`emit/traffic.rs:35-58`) gains the scheduled ops each scope runs inside a block. Selection already counts scheduled ops per scope (`emit/select.rs:146`).
- `traffic.rs` exists to hand a measurement harness the static features a cost model needs (:3-10), and says "It is a *count*, never a decision" (:12). This is one more such feature.
- Its production consumer is `collapse_bench`, which records `EmitTraffic` per kernel (`pixelflow-pipeline/src/collapse_bench/mod.rs:295-330`) and records this field beside the rest. So production decides the API (selection §0.6).
- A test in `pixelflow-codegen/tests/` asserts through `EmitTraffic` that the font-shaped kernel's row scope runs its row work inside a block.
- `ScopeTraffic` is a public struct, so the field is JP's to approve (Q10).

### 3.10 Worked example: the font

`pixelflow-graphics/src/fonts/loop_blinn/program.rs`, at the per-count programs #1339 shipped:

- `piece_term` is `if (y > p.lo) & (y < p.hi) { term } else { 0.0 }` (:95-98). Call its mask `band_p`.
- `glyph` is `if inside(bounds, x, y) { coverage(ink(x, y)) } else { 0.0 }` (:132-135). `inside` reads `x` and `y` (:111-114).
- `kernel!` lowers `if` to `OpKind::If` and `&` to `BitAnd` (`pixelflow-compiler/src/lower.rs:14`, :642, :732).
- `y = Uniform(y0) + Var(row)` and `x = (Uniform(x0) + Var(col)) + Var(lane)` (`pixelflow-ir/src/passes/lattice.rs:119-121`).

So `band_p` reads only the row binder and uniforms, and is a row value. `inside` reads the column, and is a column value. The band `If` reads `x` through `term`, so it sits in each column fold.

1. In a column fold, `t0_p` is read under `[inside: True, band_p: True]`.
2. Leaving the column fold drops `inside`, whose mask the column fold defines. It keeps `band_p`, a row value.
3. Every column fold `pack` builds, one or two, reports the same key. In the row fold, `up_p`, `t0_p`, `t1_p` and what only they read (`s·y`, `up ± ½`) are owned by one projected region on `band_p`, in a block immediately before the first column fold's `Reduce`.
4. `band_p` itself is read by the band `If` at `[inside: True]`, which projects to `[]`. So it stays unguarded in the row fold.
5. Leaving the row fold drops `band_p`. Each piece's uniform-only work stays unguarded in the body and runs once per call.

The column fold is unchanged. Its band arm is still tested and skipped per batch.

Where the column fold has one trip, this is not the whole story. §6.14 says what changes there, and why it matters for three of the six cells CL2 measures.

---

## 4. The types

All in `program/`, `pub(super)` unless stated. One noun is added, and two old ones carry it.

```rust
/// One arm of one `If`, named across scopes by the `If`'s value. A guard
/// tests `mask`; a region is keyed by `(choice, arm)`, so two folds that
/// share an `If` report one region. `mask` is carried so that a scope
/// without the `If` can test it.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct Literal {
    /// The `If` this is an arm of.
    choice: ValueId,
    mask: ValueId,
    arm: IfArm,
}

/// One block a scope's flow can skip: an `If`'s arm, or a literal the
/// scope's loops read under.
pub(super) struct Arm {
    /// The position of the block's first reader.
    pub(super) at: usize,
    pub(super) literal: Literal,
    pub(super) region: Region,
}

/// What a scope reads from the scopes around it, and under which literals:
/// for each value it reads but does not define, its accumulating root
/// included, the path from its top to the region where those reads meet,
/// with the literals on its own masks dropped. Built only by `Ownership::of`.
pub(super) struct Demand(BTreeMap<ValueId, Vec<Literal>>);

/// What each loop a scope opens reads from it: the loop's demand, keyed by
/// its `Reduce` def.
pub(super) struct FoldReads(BTreeMap<ValueId, Demand>);

impl Ownership {
    /// The ownership of `schedule`, whose result is `root`, and its demand
    /// on the scope around it.
    pub(super) fn of(schedule: &[Def], root: Option<ValueId>, folds: &FoldReads) -> (Self, Demand);
}
```

- `Region` (`ownership.rs:43-53`) stays a node of one scope's tree. It gains one way to be made: `under(parent, literal)`, made once and then found.
- `Arm` (`ownership.rs:55-66`) replaces its `mask` and `arm` fields with one `literal`, so the pair has one home, and `under(ρ, ℓ)` reads it directly. Its list gains the projected regions.
- `ArmBlock { arm, mask, items }` (`mod.rs:245-255`) is unchanged. It is selection's view of a block: a test and what it skips. Which `If` a literal belongs to is a placement fact that selection never needs, so it stays out of the type selection reads. An arm block and a projected block are one type, and selection never asks which it has (fold before you dispatch).
- `Flow::of` already takes the scope's `Ownership` (`layout.rs:46`, #1371). `Own` calls `Ownership::of` children first, with the `FoldReads` built from the children's `Demand`, and keeps the scope's own `Demand` in its `Ownership` for the parent. `Layout` lays out the flow and then runs §3.8's checks. Children first makes every input exist when it is needed.

**What becomes unrepresentable.**

- **A scope chosen by demand.** Scope choice reads only variance. No function maps a `Demand` to a `Scope`.
- **A literal a scope cannot test.** A parent receives literals only inside a `Demand`, and only the child's `Ownership::of` builds one, after dropping the literals on the child's own masks.
- **A guard narrower than its readers need.** A region is grown only at an `If`'s arm edge or along a path a child reported. Projection removes literals, and the meet weakens.
- **"Computed here for the scopes inside, unguarded."** β deleted `roots`. CL2 deletes `outer`: what a fold reads from outside is its `Demand`, with the paths it reads under.

**What stays a loud runtime check**, since a wrong value would panic on the next line rather than be silently representable (CLAUDE.md, "Denote before you build"):

- a mask not bound before its block panics in `Selector::lookup` (`emit/select.rs:382-390`);
- §3.8's assertions, which cover the stale-mask and missed-read cases `lookup` cannot see.

**What is not typed, and is not needed here.** Whether a `ValueId` is a mask. The IR has no mask type, and `OpKind::is_bitwise_domain()` stands in for one (CLAUDE.md). That is one reason splitting a `BitAnd` mask into conjunct literals is deferred (§12, Q9).

---

## 5. What it deletes, and what it adds

**Already deleted by β (`7acdcaee`):** `Ownership::of`'s `external` input and its pin loop; `ScopeRegion::roots` and `ScopeFold::roots`; the ancestor chain, `body_binds` and the computing-scope search in `place_roots`; the rule "no arm owns a root"; `no_arm_owns_a_root`, rewritten as `an_arm_owns_the_roots_only_its_loops_read`. Production +21 lines, in-file tests +96 (β's commit message).

**CL2 deletes:**

- `ScopeFold::outer` (`mod.rs:195-198`), `ScopeFold::reads` (:205-213) and `place_roots`' write to `outer` (`scopes.rs:277`).
- `FoldReads::new`'s named-id table and its filter (`guards.rs:57-90`), with the doc's admitted over-approximation (:39-42). `FoldReads` is built from the children's `Demand`, which holds their actual reads.
- `ownership.rs:124-126`'s early return, which becomes the record of the scope's demand.
- The order and nesting checks in `laid_out_with` (`layout.rs:150-216`), which move into production (§3.8).
- The doc rules §7 retires or restates: "the frame prologue is never guarded", β's Park argument, "relaxed in one way, and only one", and "a block's place … is the `If`".

**CL2 adds:** `Literal`, `Demand`, projected regions, the root seed, the first-reader position with the nearest-owning-ancestor splice, the children-first order, the production assertions in linear form, the `ScopeTraffic` count and its recording in `collapse_bench`, and the tests.

**Lines.** CL2 adds more than it deletes: about +80 to +130 production lines net in `pixelflow-codegen/src`, estimated from reading, not from a patch. With β's +21, the series grows production by about +100 to +150. That is said plainly rather than promised as a subtraction. CL1's harness lines are in `pixelflow-pipeline`.

What this plan subtracts is rules: the zero-park argument's special case, the frame-prologue rule, a second kind of dominance, `outer` as a list to defend, and the idea that what a scope computes for the scopes inside is unguarded by construction.

---

## 6. Consequences

Followed until nothing else falls out.

1. **The outer scope gets a block for an inner `If`'s literal.** It is not a copy of the `If`. There is no blend and no second mask def: a test of a mask the scope already computes, and a branch. Each font piece's row work moves under its row-level band test.
2. **Roots stop being special across scopes.** β made a root's readers the loops that read it, within one scope. With paths, a root's region is read off like any other value's, across scopes. The ontology's Root rule, "an arm owns a root when it owns every scope that reads it" (../../../.claude/skills/ontology/codegen.md:112-117), holds for an arm in another scope too.
3. **A whole loop can sit in a projected block.** A loop whose result is read only under a projected literal is owned by that region, so the whole loop sits inside the outer block and is skipped with it. A skip over a whole loop neither enters nor leaves it, which `Builder::finish` allows (`emit/build.rs:429-436`). Arms that hold loops already compile (`pixelflow-core/tests/guard_fold_price.rs`, first section).
4. **A fold's reads are its real reads.** `FoldReads` carries the child's `Demand`, so `outer`, `reads()` and the named-id over-approximation go. JP's question of 2026-09-07, "Why isn't the DAG over the whole program?", is answered per scope: one reverse pass per scope, composed through `Demand`, not one pass over the whole DAG. That departs from the recorded answer (§7A.14), because sibling folds share `ValueId`s (Q4).
5. **Layout has one rule.** A block goes immediately before its first reader. Selection §0.13 is its arm case, generalized (§7A.5).
6. **Ownership runs children first.** A child's demand is an input to its parent's pass.
7. **Selection, the allocator and every per-ISA emitter are unchanged.** The guard is decided in `program/`. `emit/` gains one count that decides nothing, and the `Function` doc becomes one law. JP's emitter ruling holds.
8. **A guarded value can be live past its block's join.** β retired the zero-park rule because an arm-owned value's "life, its slot's lease and its store end inside the block … so the join after the arm never holds it" (../../../.claude/skills/ontology/codegen.md:160-163). A projected block's values, `t0_p` and `t1_p`, are live past the row block's join into the column loop. So that argument stops holding, and Park is restated on the law: on a path that skipped the block, the only readers that run are the blend, which discards bitwise, and nothing (§7A.2). A gather index cannot read a skipped value, because a gather is not a blend.
9. **The body is a scope like any other.** Uniform-only work under a uniform mask whose `If` runs per batch gets a body block, tested once per call. HEAD already makes a body block for an `If` whose value is uniform (`pixelflow-core/tests/guard_parked_reads.rs` §3, :400-506).
10. **Demand becomes resident.** It is conjunctive and per scope. The ontology's "Lives: nowhere resident" (../../../.claude/skills/ontology/compiler.md:437) changes.
11. **A mask computed in a deeper scope guards nothing outside it.** The glyph's `inside` reads `x`, so it guards nothing in the row fold. Its row-only conjuncts could, if a conjunction's true arm contributed its conjuncts as literals. That is a follow-up (Q9), and a departure from what the superseded plan promised (§7A.13).
12. **A uniform or row-uniform mask is still tested per batch in the column fold.** A block's test runs once per trip of its scope, and the column arm keeps its own test. For a frame-uniform mask that test is uniform-only work run per batch, against JP's ruling. For a row-uniform mask it is a row-uniform guard at column scope, against the ontology. §7A.11 and §7A.12 name both, and Q6 puts them to JP.
13. **Lane-varying masks are consistent across scopes.** A mask that varies by row and lane is a row value, since the lanes are bound in every scope (`scopes.rs:243-251`). The row block and the column arm both test "some lane selects" over the same full vector. The remainder's lane fold `[0, r)` stores fewer lanes but computes the same vector.
14. **A one-trip column fold keeps the band `If` out of the row scope.** `pack` builds a main fold only when `w > r`, and a remainder fold of one trip (`pixelflow-ir/src/passes/lattice.rs:199-220`). So at 7 px on both x86 tiers there is only a one-trip remainder fold, and at 16 px on AVX-512 the main fold `[0, 16)` with stride 16 has one trip. NEON at 7 px has two one-trip folds. In those cells:
    - every value that reads `x` carries the column bit: `across`, `left`, `right`, and `monotone_root(left|right, bx, ax)` (`program.rs:74-77`, :82-83), which is 2 square roots and 2 divides per piece. The band arm owns them, so today they run once per in-band row;
    - if the one column value were substituted (Q7), `x = x0 + lane` would be lane-only, and those values would go to the body, once per call (`place_roots`' doc, `scopes.rs:243-251`). That is a prediction from the variance rule, to be confirmed from per-scope traffic on a font program;
    - the band `If` would then sit in the row scope, where HEAD's one-scope ownership already gives `up`, `t0` and `t1` the region `[inside: True, band_p: True]`, a stronger guard than this plan's projected `[band_p]`, with no second test of `band_p`;
    - this plan alone gives those cells a row block plus a second test of the same mask.

    So at three of the six cells CL2 measures, removing one-trip folds delivers more than this plan, more cheaply. This plan's own win is at 16 px on AVX2 (2 batches a row), at 32 px on both tiers (2 or 4), and at every width that is not one batch. CL2 measures four builds (§10), and Q7 asks whether one-trip folds go first.
15. **What placement executes, against the static price.** On the guard axis, the DAG ops executed are at most the price, which takes every guard as held. Each projected block adds one test and one branch per trip of its scope, which no DAG node carries and the price does not see. So a piece whose band covers every row executes slightly more than its price. The price and placement also differ in units for every lattice kernel, the font included (§8).
16. **What this plan adds dynamically is bounded.** Per call, it adds at most as many tests as the inner `If` executes, since an outer scope's trips never exceed an inner scope's, and exactly as many where a row is one batch. In code it adds at most one test and one branch per projected region per scope, and copies no value, since loop versioning is not built (Q6).

---

## 7. Recorded decisions this plan changes, and options it does not take

### 7A. Recorded decisions

Each is named with its source. P0 marks each where it stands. Items 1 and 2 follow from β, already approved. Items 3, 5, 6, 9, 10, 11, 12, 13 and 14 go to JP (§12).

1. **"A scope's roots are read by the scope itself, so no arm owns one."** Retired by β (`7acdcaee`). This plan extends β's one-scope case across scopes (§6.2).
2. **The zero-park rule, and β's argument for retiring it.** The rule, "a skipped region leaves zero in its parks, never a mask" (one-conditional §4, :259-263; §9, :447; demand §7, :340), was retired by β on this argument: an arm-owned value is read only inside its arm's block, so "the join after the arm never holds it" (../../../.claude/skills/ontology/codegen.md Park, :157-165; β's notes at one-conditional :448-452, demand :341-344 and sel:1781). §6.8 makes that argument false for projected blocks. Restated on the law of §2.6: on a path that skipped a block, no reader that runs observes its values except the bitwise blend. HEAD already leaves stale values: `check_per_call` collapses a skipped call after one that ran, and its doc says "a park or accumulator the skipped call failed to write still holds the previous call's value" (`guard_parked_reads.rs:183-185`). P0 edits Park and the three notes. One-conditional §4's sentence, which β left unmarked, is marked too.
3. **"The frame prologue is never guarded. It runs once per call."** One-conditional §9 (:453); demand-is-a-dag-property §7 (:338-339, "a branch there buys nothing and the allocator must not be pinned for it"); ontology Frame, Follows (../../../.claude/skills/ontology/codegen.md:382), and Guard, "Is not: … Ever in the frame prologue" (:425). Retired. HEAD already guards in the body: a uniform `If`'s arm is a body block (`guard_parked_reads.rs` §3). Its reasons are gone: no `Trips::Once` remains in any `.rs` file, and §0.12 deleted the mispredict bound. JP confirms (Q1).
4. **"Dominance is relaxed in one way, and only one"** (`emit/mod.rs:607-616`; selection §1.3, sel:210; §2.9, sel:864; Risks item 4, sel:1806). Restated, not extended: a read observes a def only where the def's guard holds, and the blend is the one reader that may run elsewhere.
5. **Selection §0.13 and the ontology's Block and Schedule entries.** §0.13: an `If`'s arms before the `If` is "the whole placement rule", and "A `Flow`'s arms are in the order of their `If`s" (sel:123-124). Block: "A block's place is not computed: it is the `If` it belongs to" (../../../.claude/skills/ontology/codegen.md:467-468). Schedule, Follows: "an `If`'s arm blocks go immediately before it" (../../../.claude/skills/ontology/codegen.md:48-50). Generalized: a block goes immediately before its first reader, and blocks before one reader go outer first, then in the order of their `If`s, true arm first. §0.13 is the arm case and holds for every arm. §0 claims precedence over later sections, so P0 marks §0.13 itself, and JP confirms (Q3).
6. **"Scope placement … with cost choosing between"** (ontology Placement, ../../../.claude/skills/ontology/codegen.md:79-82) and an-integral-is-a-fold §4, "choosing the scope is purely a cost question" (:389-390). Not refused: left open. The static price has no guard term, so it never prices a sink below the variance scope lower than the outer bound, and this plan does not sink (§2.7). A guard-aware price could choose one (D6's `cost·P(demanded)`). JP decides whether a sink is ever wanted (Q7).
7. **Selection §0.12's clause "and by its mask being computed in the arm's scope"** (sel:120). Already dead in code: D2 dropped it (sel:1783), and `layout.rs:4-7` reads "whatever scope computes its mask". This plan makes it false by construction, since a projected block's mask is always computed outside its `If`'s scope. §0 claims precedence over later sections, so P0 fixes the clause itself.
8. **The mispredict bound in one-conditional** §6 (:334-335, "the one profitability bound") and §9 (:442-444). Already superseded by §0.12 ("There is no bound"). `MISPREDICT_PENALTY_CYCLES` exists in no `.rs` file. P0 marks it.
9. **"A Y-range demand belongs in a loop bound, not a per-row guard"** (ontology Demand, ../../../.claude/skills/ontology/compiler.md:434-435). This plan adds per-row guards. Lowering 1 still comes first: a mask with a static range is split, not guarded (one-conditional §1). The font's band ends are uniforms, and "a range depending on a uniform is a bind-time answer" (../../../.claude/skills/ontology/lattice.md Derived range, :276-277), whose tier is unbuilt (one-conditional D3, :407). Until it exists, the per-row guard is the only lowering the band has. JP confirms (Q2).
10. **Demand "as a predicate … is DNF over mask literals, capped and widened"** (../../../.claude/skills/ontology/compiler.md:420-422; one-conditional §6, :317-319). This plan builds conjunctions only. The meet of two paths is their common prefix, which can be weaker than set intersection (`[a, b]` meet `[b]` is `true`). That is sound. DNF stays unbuilt (Q9).
11. **The per-batch test of a frame-uniform mask.** It reads only a uniform-only value and runs once per batch, so it is uniform-only work run per batch, against JP's binding-time ruling. This plan guards the uniform-only work under such a mask once per call (§6.9), but leaves the mask's own test where its `If` is. No lowering 1 answer exists for it today: a mask built only from uniforms "has no coordinate-only range at all … so it is the best case for lowering 2" (one-conditional §3, :240-247). JP decides (Q6).
12. **The per-batch test of a row-uniform mask.** Row is a coordinate, so JP's ruling does not cover it. The ontology does: "A row-uniform guard belongs at row scope" (../../../.claude/skills/ontology/language.md:232-233). This plan puts a guard on the band at row scope, and leaves the same mask's per-batch test in the column fold. For the band, lowering 1's bind-time tier (one-conditional D3) is the answer that removes it outside each band. JP decides (Q6).
13. **Projection drops whole mask literals.** One-conditional §6 keeps "dropping an X-dependent literal in the row prologue and guarding on the Y-only remainder" (:336-337), and demand §1 defines a literal as `(mask: ValueId, polarity)` (:108-110). So dropping a whole literal is that rule. But demand §2 promised more from it: "the 1,217 entries get their rows-outside-the-glyph guard with no font change" (:144-146). That needs a conjunction's row-only conjuncts as literals of their own, which this plan does not build. So the glyph's `inside` guards nothing at row scope (§6.11; Q9).
14. **"Why isn't the DAG over the whole program?"** (demand :28-29; ../../../.claude/skills/ontology/compiler.md Demand, Lives, :441-443). The recorded answer is one backward pass over the DAG. This plan runs one pass per scope and composes them through `Demand`, because sibling folds share binder slots and `ValueId`s (Q4).
15. **Region, sense 1: "a scope, or one arm of one `If` in it"** (../../../.claude/skills/ontology/lattice.md:287). A projected region is one arm of an `If` in another scope. P0 rewords the sense: a scope, or a conjunction of literals the scope can test.
16. **If, Lives: "the split itself, the bind-time tier, the interval tier and general guards are unbuilt (D2–D5)"** (../../../.claude/skills/ontology/language.md:644-649). CL2 builds part of D5: lowering 2 on conjunctions of whole literals. The "Lives" line changes in CL2.

### 7B. Options not taken

- **Sinking below the variance scope** to use a literal a deeper scope tests. The price we have cannot choose it (§2.7). Not refused: Q7 asks.
- **Versioning a loop per invariant mask** (full unswitching). `k` independent masks give `2^k` copies of the column fold, to remove one test per batch. A glyph has one band mask per piece. Not built here. Since leaving the per-batch test departs from a ruling and an ontology line (§7A.11, §7A.12), Q6 asks rather than refuses.
- **Placement, guard or unswitch decisions in the per-ISA emitters.** JP's ruling. A projected block is an `ArmBlock` that `select.rs` already walks.
- **A second block type for projected blocks.** `ArmBlock` already denotes one literal's block at one scope.
- **An IR `Guard` node or a one-armed conditional in the e-graph.** Retired 2026-10-05 (ontology Guard, sense 4). Demand is a fact of the extracted DAG, not an e-graph fact.
- **A static guard discount** set by a tuned constant. No tuned constant stands in for a bound (one-conditional §9). The price takes every guard as held.
- **A clause cap or a widening knob.** Guards are conjunctions, and the meet widens on its own.

---

## 8. The price

**The static price is `Σ cycles(v) × evals(var v) × P(g(v))`, with `P = 1`.** Every guard is taken as held. `P = 1` is a bound, not a tuned discount. So this plan changes no extraction decision.

Why not a guard fraction:

- Demand is not an e-graph fact. It is exact only on the extracted DAG (../../../.claude/skills/ontology/compiler.md Demand, "Is not").
- §0.12 deleted every arm price (`Arm::cycles`, `Ownership::price`, `def_cycles`).
- The font's band ends are uniforms, so no static fraction exists for them. The static fraction `cost·P(demanded)` stays unbuilt (../../../.claude/skills/ontology/compiler.md:433-434, D6).

**The bound that holds.** On the guard axis, the DAG ops placement executes are at most the price. Each projected block adds one test and one branch per trip of its scope, which the price does not see (§6.15). That is the whole claim: "what runs is never more than the price" is not true as stated.

**Price and placement are not in the same units, for any lattice kernel.**

- `LatticeShape::evals` prices a node that reads `X` at every sample, `W × H`, and one that reads only `Y` at `H` (`pixelflow-ir/src/variance.rs:418-456`). No lane width reaches the extractor: `pixelflow-search/src/runtime.rs` and `egraph/extract.rs` never name a lane.
- Codegen runs column work once per batch, `⌈W/L⌉ × H` times. So relative to row work, column work is over-priced by `W / ⌈W/L⌉`: 7× at 7 px, 16× at 16 px on AVX-512, 8× at 32 px on AVX2.
- After CL2, guarded row work is priced at `H` and runs `f·H` times, for a band fraction `f`. That error and the lane error have opposite signs and do not cancel.

**The disagreement inside a surviving fold predates this plan.**

- The tree DP multiplies a fold's whole child cost by its trip count, uniform-only descendants included. `fold_body_multiple` (`pixelflow-search/src/egraph/extract.rs:2217-2222`) is applied at three sites: `extract` (:946-954), `cost_of_choices` (:1878-1888) and `TreePricer::price` (:2633-2645). Placement runs that uniform-only work once per trip of the fold's parent.
- `LatticeShape::evals` prices any node that reads a binder as per sample, and carries no trip count (`variance.rs:430-437`, :441-442).
- `cost_of_choices`' `dag` adds each node's own cost with no trip count (`extract.rs:1891`).
- `fold_body_multiple` stays. Its own doc gives the reason: the trip count is local to the fold node, and "a per-binder-slot table would not work" (`extract.rs:2211-2216`). The fix is the `(class, level)` DP of schedule-cost-model-denotation (:283, :309), which is not built.
- The font is not on this one path. A per-count program has no kernel fold: a glyph's pieces are instances the host sums with `sum2` (`fonts/loop_blinn/program.rs:116-127`). It is on the lane path above, like every lattice kernel.

**A guard-blind consumer in codegen.** `plan_carries` ranks a loop's carries by the reloads they save, weighting each read by `trips_at` of its innermost loop (`emit/regalloc/local.rs:446-505`). A read inside a rarely taken row block keeps its full weight. CL2 changes no ranking; this is recorded so a carry regression in CL2's measurements has a named suspect.

**M14 is not in this plan.** One-pipeline's M14 (its CL9, :888-900) prices the arbiter's chosen term by what codegen executes. It stays there, as the follow-up for the `extract.rs:1878-1888` disagreement. Two notes for whoever lands it:

- After this plan, M14 is still an upper bound on the guard axis, so the arbiter is biased against guardable terms by the band fraction. One-pipeline measured that error on a guarded select: "modelled 1.9–2.1×, measured 2.1–5.6×" (:478-480).
- Its gates name `glyph_is_closed` (:893), which no `.rs` file has. Its gate should time every kernel whose extraction changes, with `collapse_cost`, on both x86 tiers, before and after.

`IfHoistUnary` (`pixelflow-search/src/math/round2_rules.rs:588-633`) keeps pulling shared work out of arms, and no cost term opposes it (../../../.claude/skills/ontology/language.md If, Follows; D7). It does not touch the font: the band `If`'s false arm is the constant `0.0`, not a unary of its true arm.

---

## 9. The font

`pixelflow-graphics/src/fonts/loop_blinn/program.rs`. Every count here is read from the source, not measured. Extraction may fuse or reorder them. `every_count_s_program_optimizes_estimate_free` pins that no reciprocal estimate replaces an exact divide (`fonts/loop_blinn/program/tests.rs:193`).

**The currency is the divider.** Each `monotone_root` is a square root and an exact divide (:56-58). Each piece holds four: two on the row tier, `t0` and `t1` (:80-81), and two on the column tier, `t_left` and `t_right` (:82-83). The divide and square-root unit's throughput, not the operation count, is the likely bound. A row value is lane-uniform but computed on a full vector, so a `zmm` spends the divider on it with no more useful lanes than a `ymm`.

**Per piece, per row, today, unguarded in the row fold:** `s·y`, `up = s·y − y0`, `up ± ½`, two `monotone_root`s and two clamps. That is 2 square roots, 2 divides and about 20 other operations.

**After CL2.**

- Outside a piece's band, its row work is one test and one branch, about 2 instructions.
- Inside the band, the same 2 instructions are added.
- Divides and square roots removed per draw: `4 × Σ_p (h − rows_in_band_p)`. Tests added: `N × h`. Both are exact on the host from each piece's `lo`, `hi` and the tile height. CL1 computes them.

**What does not change.**

- Each piece's band mask, `(y > lo) & (y < hi)`, about 3 operations, stays unguarded in the row fold.
- Each piece's uniform-only work (`b`, `bx`, `a`, `ax`, `b·b`, `bx·bx`, `2b`, `2bx`, `⅓(bx·a − b·ax)`, `σ`) stays unguarded in the body, once per call.
- The column work stays in the column fold's band arm, skipped per batch. Its test of `band_p` stays.
- `t0` and `t1` stay row values that the column loop reads, carried or stored by the allocator (`plan_carries`, `emit/regalloc/local.rs:22-32`).

**The one-trip cells.** At 7 px on both x86 tiers and 16 px on AVX-512, removing one-trip folds would move the column tier's two roots to the body and guard the row tier under `[inside, band]` (§6.14). This plan alone does less there.

**A prediction that discriminates.** C1 measured AVX-512 drawing slower than AVX2 at every size, "not yet explained": mean draws 2.9 against 1.6 µs at 7 px, 4.9 against 2.9 at 16, 9.3 against 6.4 at 32 (the-language-is-kernel :774-787). The ratio shrinks as the column share grows.

- If row-uniform divider work on full vectors explains the gap, this plan (and removing one-trip folds at the one-batch cells) closes most of it at 16 and 32 px.
- If `zmm` frequency licensing on the shared host explains it, both tiers speed up about equally and the gap stays.

CL2's commit body states which happened.

**What it rests on.** The band `If` must survive extraction as an `If`. No rule turns `If(m, a, 0)` into mask arithmetic: the `If` rules are `IfSameBranch`, `IfLtToMin`, `IfLtToMax` and `IfHoistUnary` (`round2_rules.rs:467-633`). The gates do not trust that. They check pixels in and out of the band, the `ScopeTraffic` count checks the guard, and CL2 measures the time.

---

## 10. CLs and gates

The gate letters are selection-is-a-phase §5's: **G** (every commit), **V** (G plus the suites of record), **Q** (the NEON suites under `qemu-aarch64`, §0.9), **L** (the line delta, stated) and **M** (CI's `Test on macos-latest`). Its rules hold: tests drive the production API and live in a crate's `tests/` (§0.6); goldens are images, and bytes are reported, never pinned (§0.11).

**Order.** α (#1367), β (#1368), γ; then P0, CL1, CL2. The removal of one-trip folds (Q7) is its own CL, and may land before CL2 if JP agrees.

**P0, docs only: this plan.**

- Adds this document. It landed first on its own, as `Proposed`, so the questions of §12 could be answered against it; the markings and ontology edits below follow JP's answers, since Q1 and Q3 change his recorded rulings.
- Marks every clause §7A changes, where it stands: selection §0.12's clause (sel:120) and §0.13 (sel:123-124); one-conditional §4's zero-park sentence (:259-263), §6's bound, parks and projection bullets (:334-338), and §9's frame-prologue constraint (:453) and β's note (:448-452); demand-is-a-dag-property's β note (:341-344), with a pointer to this plan in its banner; an-integral-is-a-fold §4 "Placement".
- Edits the ontology's meaning lines: Schedule (../../../.claude/skills/ontology/codegen.md:48-50), Placement (:79-82, :91-93), Root (:112-117), Park (:157-165), Frame (:382), Arm ownership (:404-406), Guard (:425, :429-432), Block (:467-468); Demand (../../../.claude/skills/ontology/compiler.md:420-422, :433-437); If (../../../.claude/skills/ontology/language.md:644-649); Region sense 1 (../../../.claude/skills/ontology/lattice.md:287). Each "Lives" line changes in the CL that makes it true.
- **Gate:** `scripts/check-doc-paths.sh`. The docs-only scope skips the build jobs (`scripts/ci-change-scope.sh`).

**CL1: the instrument.** No in-tree harness can produce CL2's measurement today:

- `pixelflow-graphics/benches/font_rendering.rs` times the builder's `text()`, never `FontPrograms::draw`. C1's draw table came from a harness that is not in the tree.
- `collapse_cost capture` captures the builder's glyph kernels through `GlyphAtlas` at 16 pt (`pixelflow-pipeline/src/bin/collapse_cost.rs:37-40`, :180), not the per-count programs.
- `Trips::of` asserts `extent[0] >= lanes` (`pixelflow-pipeline/src/collapse_bench/corpus.rs:69-74`), and `CollapseSession::measure` calls it on every kernel (`collapse_bench/mod.rs:233`), so a 7 px tile panics on both x86 tiers.
- `features_of`'s `dyn_instructions` is scope instructions times `EmitTraffic::trips` (`collapse_bench/mod.rs:309-330`). Both are guard-blind, so CL2's added test and branch would read as more work while executed work falls.

CL1 names what CL2's gate needs, and builds no more:

- `collapse_cost` captures the per-count programs: a glyph composed with its rows as uniform defaults is the same compile key as `program(n)` (the-language-is-kernel §1.7), and `dummy_context` already replays defaults (`collapse_bench/mod.rs:408-411`).
- `Trips::of` accepts `extent[0] < lanes`.
- The collapse call is timed alone, beside draw time, which includes the block write and the `DiscreteManifold` allocation (`program.rs:223-234`).
- A guard-aware executed count, computed on the host from each piece's `lo` and `hi` and the tile height: rows in band times the row block's ops, plus the static per-scope counts, and divides and square roots per draw (§9).
- `perf stat` instructions and branch-misses around the timed loop, where the host allows it.
- The font-shaped kernel, a row-only value read only in a per-batch `If` arm on a row-uniform mask, becomes a row of `pixelflow-codegen/tests/support/sibling_rows.rs`. Today no `byte_probe` row has that shape: `guarded_if_in_fold`'s mask is lane-varying (:152-176). The probe and CL2's first test then share one definition, and the probe's per-scope traffic shows the row work moving into a block.
- **Baseline:** CL1's commit body records the six cells (7, 16, 32 px; AVX2, AVX-512) and 20 px at HEAD. 20 px is a width no x86 lane count divides, so `pack` builds a main and a remainder column fold on both tiers, and §3.2's shared key is costed on x86, not only on NEON at 7 px.
- **Gate:** V, L.

**CL2: projection, "an inner literal guards outer work".**

- **Change:** §3.2-§3.9 and §5's CL2 list. `Ownership::of` seeds the accumulating root and returns its `Demand`. `FoldReads` carries the children's. `Own` runs children first. Projected regions are keyed by `(If, arm)` and placed before their first reading `Reduce`. `Layout` asserts §3.8, `assert_closed` restated over `Demand` included. `ScopeTraffic` gains the guarded-op count and `collapse_bench` records it. The `Function` doc is restated as §2.6's law. Nothing else in `emit/` changes.
- **New tests**, in `pixelflow-core/tests/` unless stated, each against a plain-f64 reference, at `WIDTH` and `RAGGED_WIDTH`:
  - the font's shape: a row-only value read only in the arm of a per-batch `If` on a row-uniform mask, on rows all in, all out and mixed;
  - a body value read in the column fold under a uniform mask, collapsed `check_per_call`-style, so a skipped call follows one that ran;
  - a value defined in a skippable block and live into a loop after the join, which no test builds today;
  - a mask that varies by row and lane, and a mask that varies by lane only;
  - **a fold whose body root is outer:** `out = select(X < T, W·sin(X/10), 0) + D`, with `W = Σ_{j<N} |X − j/2|` and `D = Σ_{k<N} W`, in `guard_sibling_fold.rs` §3, on a batch where the mask is false in every lane. Without the seed of §3.3, `W`'s loop is arm-owned and `D` adds `N` copies of a stale `W`;
  - in `pixelflow-codegen/tests/`: the font-shaped kernel's row scope runs its row work inside a block, through `EmitTraffic`.
- **Existing gates:** `guard_parked_reads`, `guard_sibling_fold` and `guard_fold_price` (in V); B7's equivalence gate, `every_ascii_glyph_drawn_by_its_count_s_program_draws_the_builders_pixels` (`fonts/loop_blinn/program/tests.rs:118`); `glyph_atlas_golden` and `freetype_oracle`. Images do not move.
- **Tiers:** `cargo xtask isa-matrix` runs the V suites on both x86 tiers. Q runs them under `qemu-aarch64`: the change reaches NEON through what it selects.
- **Commit body**, all with CL1's instrument, against CL1's baseline:
  - `byte_probe` before and after;
  - collapse time and draw time per glyph at 7, 16, 20 and 32 px on AVX2 and AVX-512;
  - the guard-aware executed counts, and divides and square roots per draw;
  - four builds where Q7's CL exists: HEAD, one-trip folds removed alone, this plan alone, and both;
  - branch-misses at 7 px on AVX2, where the margin is thinnest (§11.3);
  - zoom compile time for every per-count program at one tile size on both x86 tiers, against C1's table (the-language-is-kernel :774-779). An increase above noise is a finding;
  - which branch of §9's prediction held.
- **Gate:** V, Q, L.

**CL4, optional: literal keying.** CL3, the price of a scope (one-pipeline M14), was in an earlier draft and is dropped (§8); its number is not reused. A projected or arm region is keyed by `(mask, arm)` alone, so two `If`s on one mask share a block, and a block splits where a member is built after its first outside reader. It lands only with a measured production kernel that needs it. None has one today. `scene3d.rs` lowers a colour choice to one `If` on packed words precisely so that four channel `If`s never share a mask (:78-89). The checker's per-channel selects on `light` have constant arms that own nothing (:459-461).

**Not in this plan.**

- One-pipeline M14 (its CL9): the price of a scope (§8).
- Conjunct literals: the true arm of `If(a & b, …)` contributing `a` and `b` separately, so a row-only conjunct of a lane-varying mask can guard row work (Q9).
- Removing one-trip folds in `pack` or `legalize`: a fold over `[lo, lo+1)` is its body at `lo` (Q7). Its own CL.
- Lowering 1's bind-time tier for the font's bands (one-conditional D3), which would remove the remaining per-row and per-batch band tests outside each band (Q6).

---

## 11. Risks

1. **A cross-scope read no def makes.** An accumulating fold's body root is read by its combine, not by any def. If `Demand` missed it, an arm could own a value a loop outside the arm accumulates, and the loop would add stale bits: a wrong value, not a slow one. It is guard_sibling_fold's hazard 1 reached through the root. §3.3 seeds it, §3.8's closure assertion makes a miss loud, and CL2 tests the shape on an all-false batch.
2. **The allocator path is untested.** A value defined in a skippable block and live into a later loop relies on `Scan::arrive`'s rule for a path where the definition comes after the branch (`local.rs:1148-1153`) and on the store after the definition (:15-20). Both exist and are exercised for an arm read by its blend. Neither is exercised for a read inside a loop after the join. CL2's third test is the check.
3. **A row guard flips at its band's edges, and so does the column test.** At one-batch rows (7 px on every tier, 16 px on AVX-512), the column arm's test of `band_p` already flips on the band's edge rows. After CL2 the row block's test of the same mask flips on the same rows. §3.6 lays the `N` pieces' blocks out together before the column `Reduce`, so piece `p`'s two branches are about `N − 1` branches apart, and correlation through global history is not guaranteed. A glyph has 12.8 pieces on average (1,213 over 95; the-language-is-kernel :766-768). So a call can take up to about `4N` mispredicts instead of about `2N`, against a 1.6 µs AVX2 draw at 7 px. That is reasoned, not measured. CL2 measures branch-misses at 7 px on AVX2. Removing one-trip folds removes the second test in those cells.
4. **A cheap block still costs a branch.** §0.12 ruled that there is no bound: a region that owns a value is a block. A projected block that owns one cheap op costs a test and a branch per trip of its scope. If a measured kernel shows that matters, the answer is a measurement, not a constant.
5. **Compile time.** Ownership stays one reverse pass per scope, O(reads × log depth) (`ownership.rs:23-25`). The demand maps add one entry per value a scope reads from outside, the production assertions are linear, and no search is added. That is an asymptotic argument. Placement code has dominated glyph compile time before: `cluster_if_arms` was 73% of a glyph bake (CLAUDE.md, X1). So CL2 measures zoom compile time against C1's table.
6. **Bytes move.** Blocks appear in outer scopes: at most one test and one branch per projected region per scope, and no value is copied. Bytes are reported, not pinned (§0.11).
7. **The guard count is static.** CL2's `ScopeTraffic` count shows that work is inside a block. It does not show how often the block runs. CL1's host-computed count is the dynamic half.
8. **The scope is still read through `Variance` bits.** `place_roots` compares bits (`scopes.rs:267`). γ's typed binder lands first and makes "bound" a set of binders. Nothing here depends on γ's change of type.

---

## 12. Open questions for JP

Each has a recommended answer.

**Q1. Retire "the frame prologue is never guarded"?** (§7A.3) **Recommended: yes.** HEAD already guards in the body, its reasons are deleted, and with it gone uniform-only work under a uniform mask runs at most once per collapse.

**Q2. Is a per-row guard acceptable where the ontology says "a Y-range demand belongs in a loop bound"?** (§7A.9) **Recommended: yes, as the bridge.** This plan is the mechanism for any mask an outer scope computes, a range or not. Lowering 1 runs first and takes any mask with a static range. The font's band ends are uniforms, so a loop bound for them needs the bind-time tier, and a partition rule wherever bands overlap (one-conditional §5). The guard costs about 2 instructions per piece per row.

**Q3. Generalize §0.13?** (§7A.5) The rule becomes: a block goes immediately before its first reader; blocks before one reader go outer first, then in the order of their `If`s, true arm first. §0.13 is the arm case and holds for every arm. **Recommended: yes.** Without it a projected block has no place, since its scope holds no `If`.

**Q4. Per-scope ownership, or binders as fold identities?** (§7A.14) The placement review left this as your call. Sibling folds share binder slots and `ValueId`s (selection §2.9, "Bindings are private to `select.rs`, and scoped"), so one value can have a different owner on each path. **Recommended: per scope now.** Ownership is then per path, and "one backward pass over the DAG" becomes one pass per scope composed through `Demand`. A whole-nest pass keyed by `ValueId` would give a shared fold body one region for all its paths, and when one sibling's `Reduce` sat in an arm, that body would lose its own arm blocks. Revisit if binders become fold identities.

**Q5. Key projected regions by `If` now, by literal later?** **Recommended: by `If` (CL2), and by literal only with CL4.** It never splits a block. No production kernel has two `If`s on one mask that own work.

**Q6. The per-batch test of a uniform or row-uniform mask.** (§7A.11, §7A.12) Leaving it departs from your binding-time ruling for a frame-uniform mask, and from the ontology's "a row-uniform guard belongs at row scope" for a row-uniform one. The ways to remove it:
- **(a) Version the loop per mask** (full unswitching): `2^k` copies of the column fold for `k` masks.
- **(b) A frame-uniform mask: specialize the call.** At bind the mask is one boolean, so the host can choose between the residues `m ≡ true` and `m ≡ false`, each compiled once. That is lowering 1's bind-time answer for a mask with no coordinate range, and it is `2^k` programs too.
- **(c) The bands: lowering 1's bind-time tier** (one-conditional D3), which removes the band tests outside each band.
- **(d) Leave the test:** one test and one branch per batch, perfectly predicted for a frame-uniform mask.

**Recommended:** (d) now, recording that the ruling is waived for a branch test, which is control and computes no value; (c) for the bands after this plan; (a) and (b) only on a measured kernel. If you do not waive it, (b) is the form that keeps code out of the emitters.

**Q7. Sinking, and one-trip folds.** (§2.7, §6.14, §7A.6)
- *Should a value ever sink below its variance scope to use a deeper literal?* The price we have cannot choose a sink (§2.7), but a sink can win: at a one-trip column fold, `[inside, band]` is stronger than `[band]` at no extra trips. **Recommended: no sink now.** The same win comes from removing the one-trip fold, which keeps the variance rule.
- *Remove one-trip folds in `pack` or `legalize`, as its own CL?* A fold over `[lo, lo+1)` is its body at `lo`. At 7 px on both x86 tiers and 16 px on AVX-512 it moves each piece's column-tier roots to the body and lets HEAD's ownership guard the row work under `[inside, band]`. **Recommended: yes, measured against this plan at the six cells (§10), and landed first if it delivers what §6.14 predicts.**

**Q8. Leave M14 out of this plan?** **Recommended: yes.** With every guard taken as held, this plan changes no extraction decision. M14 stays one-pipeline's CL9, gated on time (§8). `fold_body_multiple`'s multiply in the DP stays until the `(class, level)` DP exists.

**Q9. Disjunctive demand, or conjunct literals from a `BitAnd` mask?** (§7A.10, §7A.13) **Recommended: neither now.** The common-prefix meet is sound, and widening `m1 ∨ ¬m2` only loses a guard. `inside`'s row conjuncts would add little for the font. A band is its piece's rows widened by half a pixel (`fonts/loop_blinn.rs:983-984`), the box is the outline's, widened the same way (`program.rs:111-114`), and a piece lies inside its outline's box. The IR has no mask type to split on safely. Revisit on a measured kernel.

**Q10. Approve one public field on `ScopeTraffic`?** (§3.9) CL2 needs it to make "this value is guarded" observable to CI. `collapse_bench` records it, so production decides the API. **Recommended: yes.** It is a change to a public struct, so it is yours to approve.
