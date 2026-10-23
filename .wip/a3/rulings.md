# A3: rulings on the critique (authoritative over spec.md where they differ)

Base: main after #1446 (fold header counts stored in 32 bits; readers return u64). Re-grep; never trust a line number.

- **B1** — the "A3 mentions" check: `grep -rnwE 'A3'` scoped to `pixelflow-*` and `pixelflow-ir/intent.md`. Reword `fold.rs:574`, `:793`, `:1524`, `passes.rs:1600` (and any other hit) as A3 makes them true.
- **B2** — classify every `Reduce {` site: paste the critique's table into the checklist; the generator reports 0 unclassified, 0 stale at head.
- **N1** — L3 is checked by one per-node transfer function, run as today's forward pass and as a rooted, memoized walk from the bound. Never a whole-arena `compute_arena_variance` per value-bounded fold; never re-resolve every `Ref` per fold.
- **N2** — discovery moves into `close_over`, before the copy: a `Const` bound is read into `Bound::Lit` and never copied. `over_values` then needs no discovery of its own. Update J9's text.
- **N3** — pricing: a `Value` bound whose child extracts to an in-domain integer literal is priced at that count; otherwise `ESTIMATED_TRIPS`. The extractor's DP holds the bound child's pick before it prices the fold. `FoldHeader::priced_trips` stays the arena/header-side reader (literal or estimate); the extractor computes the effective trips from the picks and passes them to `fold_cost(fold, trips, body)` and `below`. Amend the plan's §3.2 / §4 sentence "no price site needs a graph" to say pricing reads the bound children's extracted picks. Pin it: a class holding both twins of a fold with n > 32 prices both at n.
- **N4** — `push_reduce` (the internal constructor) keeps an out-of-domain or non-integer `Const` bound as a `Value` bound instead of refusing it, so a rule- or fold-made literal never panics at extraction and A4's guard yields NaN (L5). `BoundOutOfDomain` is refused at the user surfaces only (`over_values`, `Kernel::over`'s literal range, and `kernel!` at A6). A `Lit::Mask` bound is still `BoundSort` everywhere. Update plan §3.1's push_reduce sentence and add one line to the A4 bullet: the guard handles a rule-made out-of-domain literal.
- **N5** — state `substitute_vars_with`'s precondition (never substitutes a slot a nested fold rebinds) in §3.4 and its doc; do not claim L7 for it.
- **N6** — T-S-6 ends by extracting (`extract(&eg, root, &CostModel::latency_prior())`) and asserts the arena holds the fold with `hi` a `Value` reading the outer binder (or a literal after unrolling) and `free_index(root) == None`.
- **N7** — add T-S-4b (a rebinding inner fold is named, not entered, after `PeelFold` on the outer).
- **N8** — add `pixelflow-graphics/tests/production_glyph_arena_dump.rs:171` to the A4 bullet's serializer list; §10 says the L8/HalveFold pins are e-graph-level (no `ConstantFold`, no production path) because the key refuses value bounds until A4.
- **N9** — fix `kernel.rs` `Kernel::over`'s doc ("range is static… backend unrolls"), give `KNOWN_RANGE` a true reason for `glyph_phase_split.rs` (production glyphs before B2), narrow `pixelflow-search/src/egraph/intent.md`'s E6 restatement to known ranges.
- **N10** — `Fold::count` is `pub(crate)`; one function holds `n <= EXACT_BOUND` and both `KnownRange::admits` and `Fold::count` call it; rename `real_sorted` to `takes_a_value_bound`; `over_values` refuses a `Param`/`Buffer` root with its own message (not `BoundArithmetic`).
- Everything stays byte-identical on today's kernels (every bound literal): byte_probe both tiers, the 29 font programs' keys, saturation telemetry including extraction fields.
- Every intent.md stays within 80 lines.

# Round 2: rulings on the rewritten spec (.wip/a3/spec.md, 2,466 lines) and its two critiques

These are authoritative over spec.md and over round 1 where they differ. `S#` = critique.soundness.md finding, `C#` = critique.completeness.md finding.

**The spec's own judgement calls**
- **J7 accepted; N3's mechanism amended.** The extractor prices a value-bounded fold as its *literal twin*: the fold with each value bound replaced by its child's extracted in-domain integer literal, priced through `node_cost` and `below` like any literal fold. A bound child without such a pick, or a literal past the domain, leaves the fold priced at `priced_trips()` (`ESTIMATED_TRIPS`). `fold_cost` stays in `cost.rs` alone (gate e reads that way). The twins pin stays. The plan's "no price site needs a graph" sentence becomes: pricing reads the bound children's extracted picks.
- **J5 accepted.** A placeholder `Var` (64..128) has variance `CONST`. The reason goes in `of_var`'s doc: a placeholder is a binder-to-be, never a coordinate, and L3 is checked again when `close_over` renames it into a slot and re-pushes the fold.
- **J12, J3 accepted. N5's placement is as the spec wrote it.**
- **J13 accepted; N4 amended.** `Kernel::over` refuses a literal range past the domain with `KnownRange::new`'s existing message. `BoundOutOfDomain` is refused at `over_values` (and `kernel!` at A6). The plan text says so; it does not claim `BoundOutOfDomain` at `Kernel::over` (C6, S5).
- **B1 stricter accepted.** `grep -rnwE 'A3' --include=*.rs pixelflow-*` and `grep -nw 'A3' pixelflow-ir/intent.md` come out empty. That means no "before A3" in `below`'s doc (write "before value bounds"; C1), and new text for `insert.rs:521`'s `ValueBounded` doc (C9).

**Gate breakers (must be done)**
- **C2.** `over_values` and `user_bound` get `#[cfg_attr(not(test), expect(dead_code, reason = "pub(crate) until A4 makes it public (decision 7)"))]`. Visibility does not change.
- **C3.** `Fold::count`'s sign test is clippy-clean, e.g. `real.is_nan() || real < 0.0` (keeps `-0.0` as 0, rejects NaN).
- **C4.** The rustdoc gate is CI's: `RUSTDOCFLAGS="-D warnings" cargo doc -p pixelflow-codegen --no-deps --document-private-items` must pass. For pixelflow-ir and pixelflow-search, the head's rustdoc warning set (`cargo doc -p <crate> --no-deps --document-private-items 2>&1`, warnings normalized to file + message) is a subset of the base's: no new warning.
- **C5.** No broken or private intra-doc link in new docs: use explicit paths (`crate::fold::BoundRefusal::…`). `Fold::count` is written in backticks without a link from a `pub` item's doc.
- **C11.** Remove the imports A3 leaves unused (`cost.rs`'s `KnownRange`; the test-local `KNOWN_RANGE` once nothing uses it).

**Tests**
- **C7 / S4.** T-S-6 also asserts that the extracted arena reaches at least one slot-0 `Reduce`, and states the expected shape. Its `hi` is a `Value` whose variance is the outer binder's alone, or a `Lit` after unrolling. The rule-made `0 + 1` stays allowed.
- **C8.** T-IR-24's display text is what `fmt_expr` writes (`reduce[{header}](…`, as at `arena.rs`'s existing pin). Read the real output; never guess it.

**Holes documented, not closed** (spec §1.1 and §13 state each one; nothing new is built)
- **S1 / C12.** A3 checks a bound's sort syntactically: a `Lit::Mask` *literal* is `BoundSort`. A computed mask-valued bound, which only a hand-built `push_reduce` can make (`over_values` refuses anything but `Var`/`Const`/`Uniform`), is admitted. If the folder turns it into a mask literal, materialization panics with `BoundSort`. That is loud, not silent. T6 (the typed arena) closes the hole by giving a bound the `Real`/`Int` sort. No partial root-comparison check: a check that catches `Lt` but not `BitAnd` of two comparisons is a convention, not a sort.
- **S2.** The arena's transfer types a fold over a known empty range as `CONST`, and `ENode::variance` does not. So `EGraph::add`'s backstop can refuse a hand-built bound that contains such a fold, though `push_reduce` admitted it. State that exception exactly where the spec claims the two checkers agree. Do not give `ENode::variance` an empty arm in A3.
- **C14.** `variance_from` resolves each `Ref` it reaches through `referent_variance`, which runs over the referent's whole arena, on every push. No production bound holds a `Ref` before B3. State it in §13 and in the plan's B3 bullet: a referent's variance belongs to the referent, so memoize it per referent when B3's profile asks.

**Docs**
- **S3.** §1.3 gains a row: `Saturation::total_evals` rises on every kernel with a fold, because `KnownBound` is evaluated on every `Reduce`. No gate reads it, but the anytime curves and the Guide research bins do, so a moved curve is not a regression.
- **S6.** `FactorFold`'s doc lists every runtime `n = 0` edge decision 5 accepts:
  - an infinite `c`;
  - a NaN `c`, where `NaN·0` is NaN;
  - a negative `c`, where `c·(+0)` is `-0.0`;
  - for MIN and MAX, a NaN `c`.
- **C15.** Also update:
  - `ENode::variance`'s doc;
  - `fold_rules.rs`'s module doc (discovery joins the laws);
  - `fold_cost`'s kept paragraph and the test doc that say `fold_cost(fold, range, 0)`;
  - §2.11's list of `VALUE_BOUND` expecters (T-IR-26).
- **C16.** The plan's A3 "Done" line states facts, not J-labels; nothing in the repo may cite `.wip`. The A3 bullet's refusal list is amended per N4.
- **C10.** The Reduce-site checklist is `.wip/a3/gen/checklist.py` plus `.wip/a3/gen/classes.json`, keyed by (file, enclosing fn, trimmed hit text) with a per-key count and built from §5's table. It reports UNCLASSIFIED and STALE and exits non-zero on either. The rows A3 rewrites include `arena.rs:1757`, `extract.rs:1684` and `insert.rs:173`.

# Round 3 (CI on 31d4b58)

- **R1 — the toolchain is CI's.** This container's stable is rustc/clippy 1.97; CI's (`dtolnay/rust-toolchain@stable`) is 1.99. Local 1.97 clippy reports `must_use_candidate` on `ExprArena::nodes` (arena.rs), `Dag::iter` and `Rooted::entries` (dag.rs). All three return `impl Iterator`, which is already `#[must_use]`, and the same report fires at base, where none of the three is in A3's diff. CI's clippy rejects an added bare `#[must_use]` there as `double_must_use`, which turned Clippy and both Test jobs red on 31d4b58. So: never add `#[must_use]` to those three. A local clippy run passes when those three `must_use_candidate` reports are its only errors; any other error is real. The attributes were removed in the commit that adds this ruling.
