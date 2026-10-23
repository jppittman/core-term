# Critique of the A3 spec (`spec.md`)

Reviewed against `f772852c` (export at `tree/`), the plan
(`docs/language/2026-10-20-a-bound-is-a-value.md` §1, §3, §4, §6 A3, §8) and CLAUDE.md.
Every `file:line` is at `f772852c` unless it says "spec".

**Verdict.** The design is sound and byte-neutral on today's kernels. Two blocking
findings, both in the §11 grep gates, which cannot pass as written (B1, B2). Both are
cheap to fix. Ten non-blocking findings follow, ranked. The first three are design
holes that A3 introduces and later steps (A4, A6, B3) would otherwise meet. I found no
over-building into A4 or A6.

---

## (a) Byte, key, price and saturation identity for all-literal folds: cleared

I checked each place the spec changes on the literal path. None moves a byte, a key, a
price or a saturation result.

- **`close_over` rewrite (spec §3.5).** It yields the same arena node for node. Today
  the dag holds `[Var(binder), renamed body…, Reduce]`, and `to_arena` iterates every dag
  node in push order (`expr.rs:381`), pushing the `Reduce` through
  `arena.push_reduce(header.fold(), lo, hi, child_ids[0])` (`expr.rs:393-396`). In the
  spec, `roots == [body]` and `from_arena_roots(self, &[body])` is `from_arena`
  (`expr.rs:274-279`). The dag is `[Var(binder), renamed body…]`, which `to_arena_roots`
  pushes in the same order, and then the same `push_reduce` interns the fold last.
  `Builder::finish` records entries and does not reorder (`dag.rs:545-560`), so changing
  the entry from the `Reduce` to the body changes no node. The uniform and buffer tables
  are the same `Environment`.
- **`push_reduce` and `FoldHeader::new`.** For a literal pair, `discover` returns its
  input without calling `literal`, and `FoldHeader::new` takes the same `KnownRange::new`
  path (`fold.rs:559-563`). `refuse_a_varying_bound` returns early. The panic messages
  are unchanged. Only the point where a reversed or past-2²⁴ range panics moves, from
  `fold_at` into `push_reduce`.
- **Variance.** In `of_var`, an index below 64 is still `from_var`. Only placeholders
  (64..128) change, from `ALL` to `BINDERS`. No arena that reaches codegen
  (`program/lower.rs:147`), the unit check (`runtime.rs:289`), `lower_dwrt`
  (`passes.rs:600`) or the e-graph holds a placeholder. All of them see closed arenas.
  L4 adds `CONST` for each `Bound::Lit`.
- **`lowest_free_binder(roots)` marking `Var` reads (J8).** This is neutral for every
  builder. A placeholder (64 and up) gives `Binder::from_var(..) == None`. A slot `Var`
  in a well-formed body sits under the `Reduce` that binds it, and that slot is already
  marked.
- **E-graph.** `ENode::variance` folds over `children[1..]`, which is empty for a literal
  fold. The `EGraph::add` backstop skips a node with no value bound. `KnownBound::apply`
  returns `None` before reading anything. The application budget counts commits only
  (`graph/runner.rs:1157`, after the mask checks), and a rule that returns `None`
  commits nothing. Appending the rule last keeps every `RuleIdx`. The changed
  fingerprint reaches only `SaturationKey` (`optimizer.rs:638`), which keys the
  in-process `CACHE` (`runtime.rs:640`, `:655`). `FactorFold`'s
  `is_some_and(is_empty)` agrees with today's `is_none_or(is_empty)` whenever the range
  is known.
- **Pricing.** `below`: when `bounds == 0` and `usize` is 64 bits, the new formula is
  today's `usize::try_from((body as u64).saturating_mul(len))`
  (`extract.rs:1682-1688`). `fold_cost`: `usize::try_from(trips)` and today's
  `range.len() as usize` agree on 64-bit hosts (`cost.rs:425-431`).
- **`kernel!`.** `arena_to_tokens`' fold arm (`emit.rs:581-614`) is unchanged.
  `Staged`'s token shape changes, but the arena it builds does not, and no test
  snapshots those tokens (grep for `__binder`, `fold_at` and `__range` outside
  `src/`: none).

The A2b gate rows (byte_probe on both ISAs, the 29 font programs' keys) remain the
right arbiter.

---

## Blocking

### B1. The §11 "A3 mentions" grep cannot come out empty, and it misses real hits

Spec §11, fourth bullet. At head, the pattern also matches
`docs/language/2026-09-24-one-pipeline.md:681` and `:1019`, where "(A3)" is a different
plan's step. The required "empty" result is therefore unreachable. The pattern also
misses hits that A3 must reword: `fold.rs:574` ("A3's `new`"), `fold.rs:793`
("…a-bound-is-a-value.md, A3)"), `fold.rs:1524` ("A3's constructor") and
`passes.rs:1600` ("moves with A3"). The spec's own list of today's hits names three of
these four, but its regex catches none of them.

**Fix.** Scope the grep to code and the intent files, and match the word:
`grep -rnwE 'A3' --include=*.rs pixelflow-* ; grep -nw 'A3' pixelflow-ir/intent.md`.
After that, require only hits that name A4 or later.

### B2. The §11 `Reduce {` checklist fails at head: §5 classifies about 17 of about 45 files

Spec §11, first bullet: "every hit is one of §5's classified sites". Running the
spec's grep (plus `tests/`) at `f772852c` gives hits in about 45 files. §5 leaves these
unclassified. I checked each one, and none is a body-only walk that would skip a value
bound:

| Site | What it is | Class |
|---|---|---|
| `pixelflow-ir/src/passes/lattice.rs` (18 hits besides `:184`) | `collapse_shape`, `pack` and tests, on SEQ lattice folds only | left: SEQ refuses a value bound |
| `passes.rs:705`, `:949` | `lower_dwrt`: no derivative of a fold (`Err`) | left: refuses folds |
| `passes.rs:1595`, `:1933`, `:1994-2012` | tests | tests |
| `kernel.rs:1292`, `:1325`, `:1570`; `key.rs:657`; `kind.rs:204`; `term.rs:134` | tests, a doc and the `Shape` definition | n/a |
| `runtime.rs:2335`, `telemetry.rs:265`, `graphics/examples/glyph_saturation_cost.rs:49` | op-kind classifiers that walk `children()` | already correct |
| `extraction_gap.rs:187`, `witness.rs:356` | a research fold flag, and a stored-node key (`witness.rs:350-354` keeps its "as stored" for measurement continuity) | left: research |
| `nnue/mod.rs:890` | the `embed` inside the `:888` arm | already correct |
| `pipeline/src/bin/{extraction_witnesses.rs:442, strict_label_output_class_recount.rs:169}` | print every child | already correct |
| `pipeline/src/bin/phase3_at_budget_eval.rs:443`, `training/factored.rs:623`, `math/pict_rewrite_tests.rs:120` | op-kind or refuses folds | left |
| codegen `program/mod.rs`, `program/scopes.rs`, `emit/select.rs`, `program/lower.rs` | the schedule | A4 (lowering refuses first) |
| compiler `src/lower.rs` (6), `tests/*.rs` | `Holes` and tests | A5 / tests |
| `pipeline/examples/glyph_phase_split.rs:79` | `known_range().expect(KNOWN_RANGE)` | see N9 |

**Fix.** Paste this table into §5 so the checklist starts green. Otherwise narrow the
grep to `ExprNode::Reduce {|ENode::Reduce {|Shape::Reduce {` outside `tests` and
`passes/lattice.rs`.

---

## Non-blocking, ranked

### N1. The L3 check scans the whole arena on every value-bounded push. The repository has already measured this shape at 86 s

Spec §3.4 (`refuse_a_varying_bound`) and J4. Every `push_reduce` with a value bound
runs `compute_arena_variance(self)`, which is O(arena). That includes
`referent_variance` (`variance.rs:271-283`), which resolves **every** `Ref` in the arena
through the store and recomputes each referent's whole-arena variance. Refs that the
bound does not reach are resolved too.

Every rebuild re-pushes every value-bounded fold: `splice_with` (`arena.rs:1599-1602`),
`substitute_vars_with` (`:1839-1842`), `relink`, `to_arena_roots` (`expr.rs:393-396`),
`passes::copy_node` (`passes.rs:238-240`), and extraction's `Selection::to_arena`
(`extract.rs:971-981`). B3's run, unrolled over its glyphs, is k value-bounded folds in
an arena of n nodes holding unit Refs. That costs O(k·(n + Σ referents)).
`passes.rs:571-574` records this exact pattern, an arena-sized table per site: "86 s, of
a 1.5 M-node arena" on a 613-piece run. That precedent answers J4's "no profile asks for
it".

J4's objection, that a second definition of the per-node rule would be needed, does not
hold. `ENode::variance` (`node.rs:526-555`) is already one transfer function with two
drivers (the class fact and the extractor).

**Fix.** Factor `compute_arena_variance`'s match into a per-node
`fn node_variance(node, child: impl Fn(ExprId) -> Variance)`. Run it once in the forward
pass and once as a rooted, memoized walk from the bound. That keeps one rule and two
traversal orders. Alternatively, keep J4 and state the hazard in the spec and in the
plan's B3 bullet.

### N2. J9's reason for rejecting discovery in `close_over` is backwards; discovering there before the copy removes code
Shell cwd was reset to /home/user/pixelflow