# Demand is a property of the DAG

**Date:** 2026-09-07
**Status:** Draft
**Author:** JP (direction), Claude (draft)
**Refines:** §5 of [2026-09-06-lattice-is-the-index.md](2026-09-06-lattice-is-the-index.md)
("exclusivity does not survive the partition"), whose first implementation
(#1187) this plan replaces before it merges.
**Depends on:** the uniform leaf (#1183) for the frame prologue's shape;
index-range unions (#1184) for the loop-nest destination in §6.
**Decision it records (JP, 2026-09-07):** *"This operands work being
select-specific feels wrong to me. Why isn't the DAG over the whole
program? Why isn't that what we're topologically sorting to find out
dependencies?"* and, on whether to merge the per-select version first:
*"Deleting code is expensive once it's wrong and it's merged. Why not build
the control demand predicate on the DAG and use that?"*

---

## The shape

Every value in a kernel's DAG is observed under some condition. Call it
the value's **demand**:

```
demand(root)                 = true
demand(m)   for S = Select(m, a, b)   ⊇ demand(S)
demand(a)                              ⊇ demand(S) ∧ m
demand(b)                              ⊇ demand(S) ∧ ¬m
demand(v)   for consumers c₁ … cₙ      = ⋁ᵢ demand_edge(cᵢ → v)
```

One backward pass over the DAG in reverse topological order computes it
for every value. It is control dependence, and it is a property of the
graph — not of any select, not of any scope's slice of the schedule.

What #1187 built, `Exclusivity { selects: Vec<ArmOwnership> }`, is the
special case where `demand(v)` is a single literal: the values a select's
arm owns outright. A value read by the true arm of `S₁` and the false arm
of `S₂` has `demand = m₁ ∨ ¬m₂`, and the special case has no way to say
that, so it says "not exclusive" and the value is computed on every batch.
Every rule that follows from the general object had to be re-derived as a
repair in the special case: a value's consumers had to be re-discovered
per select, an arm's entries had to be *made* contiguous by iterative
partitioning (`cluster_select_arms`), and how many rounds of repair to
allow became a constant (`CLUSTER_ROUNDS_PER_SELECT = 2`) in a module
whose doc promised none.

**The invariant that makes demand a scheduler.** For every edge `u → v`
(`u` a producer of `v`), `demand(u) ⊇ demand(v)`: `u` has `v` among its
consumers, and demand is the union over consumers. So an order that lists
values by demand, weakest first, and by topological order within one
demand, is itself a topological order — every producer sorts no later
than its consumer. Values with equal demand are contiguous *by
construction*. There is nothing to cluster and no round count to choose.
This is the answer to "why isn't the topological sort finding this out":
it is the same sort, keyed by one more thing.

## Why now

C1's measurements, all on `O` at 32 px, AVX-512:

- Row-prologue guard telemetry went from `selects=60 guarded=0` to
  `selects=125 guarded=2801`. The mechanism works, and the per-row cost
  halved (0.69 → 0.37 µs/row; `S` 1.07 → 0.54).
- The largest single arm, 1,217 entries under the glyph's
  `unit_square_mask`, stayed unguarded in the row prologue because its
  mask is X-dependent. A Y-only *projection* of that mask would gate the
  rows outside the glyph; the per-select shape has no notion of
  projecting a predicate, so this was filed as a font follow-up.
- Values shared between selects are refused; nested selects are handled
  by nesting the special case; both are the general object's ordinary
  cases.

The adversarial review of #1187 found, in the per-select machinery:

- A skipped *false* arm parks the mask into the arm's root — all-ones,
  which read as `f32` is NaN. Every arithmetic consumer is annihilated
  by the blend, but a gather index is not arithmetic: through a commuted
  `Min`/`Max` a NaN index becomes `i32::MIN`, and the address is
  `base + i32::MIN · 4`. Unreachable with today's masks; forbidden by
  nothing.
- The clustering round cap is chosen, not derived, and the module doc
  says otherwise.
- The frame prologue is allocated against guards that `Trips::Once`
  never emits.

None of these is a bug in the idea. They are the cost of implementing a
graph property as a per-node repair.

## 1. Representation

A predicate is in disjunctive normal form: a set of clauses, each a set of
`(mask: ValueId, polarity)` literals. Disjunction is clause union with
subsumption; conjunction with a literal adds it to every clause and drops
any clause that now holds both polarities of one mask; the empty clause is
`true`.

Predicates are **capped** at a small number of clauses and **widened to
`true`** when the cap is exceeded. Widening is always sound: it can only
lose a guard, never skip a demanded value. The cap is a compile-budget
knob and is documented as one; it is not a correctness bound and must not
be described as derived.

Ordering the distinct predicates that occur (a few dozen in a glyph
kernel) for the sort: subsumption-based implication where it decides,
and the value DAG's own topological order between the two groups where it
does not. The producer-superset invariant is pinned by a property test
over random DAGs, because a representation that breaks it makes the
schedule silently non-topological.

## 2. A guard is a region of equal demand

Not an arm of a select. Within a scope, after the sort, the values with
one demand form one run; a branch over that run tests the predicate on
the batch: per lane, `⋁ᵢ ⋀ⱼ litᵢⱼ` with the mask ops the backends already
have (`and`, `andn`, `or`), then one any-true test. A single literal is
the one test the special case already emits; a conjunction costs one
`and` per extra literal; a disjunction one `or` per extra clause. The
profitability bound is unchanged: the region's latency-prior cost against
`MISPREDICT_PENALTY_CYCLES`. Nested regions are nested predicates and fall
out of the sort.

**Scope projection.** In the row prologue an X-dependent literal is not
yet computed. Drop it — weaken the predicate to its Y-only and
frame-only literals — and guard on the remainder. Weakening is sound.
This is the general form of the `unit_square_mask` follow-up: the 1,217
entries get their rows-outside-the-glyph guard with no font change.

**Parks.** C1's rule stands: a guard that skips a hoisted root's
definition writes that root's park ahead of the branch, so no inner
scope reads a slot nothing wrote. The value written is **zero**. For a
skipped true arm the mask is already zero; for a skipped false arm the
select takes the true operand and the false one is annihilated bitwise
whatever it holds. Zero is equally correct and removes NaN from the data
flow for one `xorps`.

**Allocation.** C1's liveness rule, "a guard site is a read of its
mask", becomes "a guard site is a read of every literal in its
predicate". The frame prologue is `Trips::Once`; guard analysis is gated
on the same `Trips` the emitter uses so nothing is pinned for a branch
that is never emitted.

## 3. What this deletes

`Exclusivity` and `ArmOwnership`; `cluster_select_arms` and
`CLUSTER_ROUNDS_PER_SELECT`; `SelectGuard::select_idx: Option<_>` and the
absent-select special-casing; `analyze_select_guards` in its
select-centric form. What remains of `guards.rs` is: compute demand,
sort by it, emit a branch per region that clears the bound.

What it keeps from #1187: `HoistCtx::Prologue { Trips }` and the
corrected comment; the liveness rule, generalized; the park-write rule,
with zero; the tight `y_extent` and `EXTENT_SLOP` on `AnalyticalQuad`;
both regression tests, re-stated over demand regions; the `Uniform` arms
in the operand helpers, including `arm_cycles`' cost arm.

## 4. The e-graph is currently against it

`pixelflow-search` has `SelectHoistUnary`:

```
Select(m, f(a), f(b))  →  f(Select(m, a, b))
```

It pulls shared work *out* of arms. That is right for operation count and
exactly wrong for guarding when `f` is expensive and `m` is coherent, and
there is no term in the extraction cost that could oppose it. The
observation in #1184's review that "CSE can break exclusivity" was not an
accident of CSE; the optimizer has a rule whose direction is anti-guard
and nothing weighing the other way.

Two levels, both later stages:

**Cost.** Extraction is additive per node. The demand-aware cost is
`cost(node) · P(demanded)`. `P = 1` where the node is unguardable. Where
the node's demand is a row-uniform mask with a known extent, `P` is
**static**: a segment gated on `y_lo ≤ Y < y_hi` over a 45-row glyph is
demanded on `(y_hi − y_lo)/45` of rows, and both numbers are in the
program. Coherence beyond the static fraction is a data property and is
the "first profile-dependent term" the S3b commit named for the
schedule-cost residual. The static part is available today from the
extents L3 and C1 put into kernels.

**Structure.** Make demand something the graph can hold and rewrite:
denote `Select(m, a, b)` as `Guard(m, a) ⊕ Guard(¬m, b)`. Then the sink
rule is the inverse of `SelectHoistUnary`, applied when
`cost(f) · (1 − P(m))` exceeds the duplicated work; guards with equal
masks merge; and extraction *chooses* a guarded form rather than codegen
recovering one from whatever form was extracted.

## 5. Guards and index ranges are one thing

A Y-only mask `y_lo ≤ Y < y_hi` *is* an index range. L3 put that range
in the loop nest so the pixel is never asked; C1 recovered it from a mask
so the pixel is asked and skipped when the batch is coherent. Same
denotation, two implementations, and demand predicates unify them: a
value whose demand is a Y-range should not be *guarded inside* a loop over
every row, it should be *scheduled into* a loop over those rows. That is
the one-loop-nest refinement §3 of the lattice plan names, and it is why
the row-prologue guard is a bridge and not the destination. The bridge is
worth crossing now because the loop-nest form needs the same demand
computation; nothing built for §2 is thrown away by §6.

## 6. Stages

**C1a — the gate, first and alone.** The `'8'@17` finding from #1187's
review is a live bug on `main`: the runtime e-graph's `MulAdd` fusion
flips `disc.ge(0)` at a horizontal-tangent row and moves coverage by 0.5,
and `optimized_glyph_matches_raw_within_reassociation_noise` cannot see it
because it runs at 32 px only. Extend it to the sizes the goldens use and
the one that bites; it must fail on `main` at 17 px. If it fails on the
branch, the tangent gate is ill-conditioned at `disc ≈ 0` by construction
and is replaced by the tight `y_extent` (which contains the vertex) or
compared against a scaled `−eps`. The `y_extent` commit rides here. Small
PR; lands before anything else.

**C1b — demand replaces exclusivity.** §1–§3, on a fresh branch from
`main` after C1a. Gate: row-prologue telemetry on `O`@32 no worse than
2,801 guarded entries; the 1,217-entry arm partially guarded by
projection; a value shared by two selects guarded (the refused case);
nested selects yield a conjunction; the producer-superset property test;
a Y-only-demanded region containing a gather never reads outside its
buffer; differential against the oracle over all-true, all-false and
mixed batches for every polarity; identity on the packed programs; ISA
matrix.

**C2a — static demand in the extraction cost.** §4, cost half. Gate:
extraction on the glyph corpus prefers the guarded form where the static
fraction says so, and the row-prologue guard count on `O` rises again
without a font change.

**C2b — `Guard` in the graph and the sink rule.** §4, structure half.
Gate: `SelectHoistUnary` and its inverse are both present and the cost
decides between them; no golden moves.

**C2c — demand becomes loop structure.** §5. A Y-range demand schedules
into a loop over that range, and the per-row call overhead L3 measured
on narrow summands (1.46× at 15-px cells) closes. Depends on L3's union
being the loop-nest primitive.

## 7. Constraints

- **Widening is always sound.** Any predicate may be replaced by a weaker
  one, up to `true`. Nothing may ever strengthen one.
- **No tuned constant stands in for a bound.** The clause cap is a
  compile-budget knob and says so; `MISPREDICT_PENALTY_CYCLES` stays the
  one profitability bound, with its existing derivation.
- **The frame prologue is never guarded.** It runs once per call; a
  branch there buys nothing and the allocator must not be pinned for it.
- **A skipped region leaves zero in its parks.** Never a mask.
- **Select stays a blend.** Its value semantics do not change; demand
  only decides what is computed, never what is selected.

## 8. Non-goals

- Profile-guided coherence. The dynamic part of `P(demanded)` is the
  schedule-cost residual and is not this plan.
- Guarding in the frame prologue.
- Changing `Select`'s NaN or bit-pattern semantics (CLAUDE.md's
  floating-point section).
