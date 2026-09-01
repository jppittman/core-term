# Phase 3 oracle headroom at corpus scale (2026-08-30)

Reproduce:
```
cargo run --release -p pixelflow-pipeline --features training --bin guide_headroom -- \
    --limit 800 --min-expressions 500 \
    --out docs/results/2026-08-30-guide-headroom.json
```
Harness: `pixelflow-pipeline/src/bin/guide_headroom.rs` (new, additive — no changes to
`provenance.rs`/`labeler.rs` semantics). Corpus: `corpus_train.bin` (6,480 expressions) +
`corpus_dev.bin` (1,512), regenerated this session (`gen_bench_corpus --target 8000`, since
`data/` was empty) because `--target 2000` fails the per-family integrity check at that scale
(quarantine noise pushes several families over the 20% refusal threshold when each family's
quota is only ~7-9 expressions — not a bug, just too small a target for the generator's
per-family floor). 800 expressions were **stride-sampled** evenly across the full 7,992-entry
train+dev population (not a prefix — corpus write order is band-by-band in increasing
complexity, so a prefix would silently sample only the simplest band and misrepresent "at
scale"). Every number below is a deterministic count or a `CostModel::latency_prior()` cost —
no wall-clock timing. `EGraph::saturate()`'s internal 500ms/iteration deadline is the same
stopping condition every production caller (including `run_episode`) already saturates under;
it bounds the *algorithm*, not anything we measure.

Pre-run check: `pgrep -fl "bootstrap_extraction_head|bench_extraction_3way"` was clean before
starting (no paper-capstone timed run in flight).

## The headline number(s) — there are two, and they disagree by 20x

| Bound | Pooled ratio (ΣLB / ΣApplications) | Per-expression median | Q1 | Q3 | Implied oracle savings (1/median) |
|---|---|---|---|---|---|
| **Labeler** (`derivation_ancestors`, over-approximate — the label a Guide would actually train on) | 0.7557 | **0.382** | 0.333 | 0.527 | **2.6x** |
| **Strict lower bound** (application's output node is literally on the extracted derivation path) | 0.0017 | **0.029** | 0.006 | 0.058 | **34x** |

800 expressions, 7,096,848 total rule applications recorded, 5,363,145 labeler-load-bearing,
12,368 strict-load-bearing. (These pooled totals carry run-to-run noise of a few tenths of a
percent — a re-run on the same corpus with the same harness produced 7,104,300 applications and
a 0.7552 pooled labeler ratio — because the handful of combinatorial-blowup expressions that
dominate the pooled sum are sensitive to iteration-order effects in how many times an
already-resolved match gets rescanned before quiescence. The per-expression median/quartiles
below are unaffected: they reproduced bit-for-bit across re-runs.)

**Read the median, not the pooled ratio.** Applications-per-expression is heavy-tailed: median
195, mean 8,871 (top 50 of 800 expressions — 6.25% — account for 65% of all applications
corpus-wide; one expression alone fired 353,542 applications). The pooled ratio is dominated by
a handful of blowup cases where comm/assoc/distribute saturate combinatorially; it answers "what
fraction of all firings across the whole corpus were load-bearing" (relevant to raw compute
spent), while the per-expression median answers "for a typical expression, what fraction of its
saturation's applications mattered" (the more relevant number for a per-episode Guide). Report
both; lead with the median.

**Compare to the toy-kernel baseline** (`docs/results/2026-07-08-rule-report.md`, 5 hand-picked
kernels, 61 rules, labeler bound only): aggregate ratios there ranged 10%-87% per rule with a
warning that "~75% of ALL applications load-bearing on circle_sdf is not credible." At corpus
scale the *median* expression's labeler ratio (38%) sits inside that same range, so the toy
kernels weren't wildly unrepresentative in aggregate — but the corpus surfaces the pooled/median
divergence the 5-kernel sample was too small to show, and the strict bound (not computed in the
toy-kernel report) reveals the over-approximation is far larger than "ratios read high":
**pooled labeler credits ~434x more applications than the strict walk finds on the winning
derivation path.**

## Two further diagnostics: budget exhaustion and expression size

The harness also reports (numbers from the final run,
`docs/results/2026-08-30-guide-headroom.json`, `quiesced_before_cap_count` /
`exhausted_budget_count` / per-expression `quiesced_before_cap`):

- **All 800 expressions quiesced before hitting either budget cap** (0 exhausted the 100-iteration
  or 10,000-class limit; `saturate_with_limits`'s own `if unions == 0 { break }` convergence check
  fired first every time). So the ratios above are not an artifact of truncating some expressions
  mid-saturation — every episode ran to its own quiescence point (a diagnostic condition, not a
  certified fixpoint; this optimizer is budget-only by design) (or, for the rare case this proxy
  can't distinguish, to the 500ms deadline; see the harness's module doc). This also means the
  heavy tail described above is a property of *how large the graph gets before quiescing*, not of
  budget clipping.
- **The labeler/strict divergence gets worse, not better, as expressions grow.** Splitting the 800
  expressions into size terciles by arena node count (small/medium/large, ~266-268 each):

  | Tercile | labeler median | labeler Q1-Q3 | strict median | strict Q1-Q3 |
  |---|---:|---:|---:|---:|
  | small | 0.375 | 0.333-0.389 | 0.056 | 0.021-0.097 |
  | medium | 0.359 | 0.297-0.399 | 0.035 | 0.017-0.059 |
  | large | 0.539 | 0.412-0.634 | 0.010 | 0.002-0.026 |

  Large expressions look *more* load-bearing under the labeler bound (54% vs 38%) but *less*
  under the strict one (1.0% vs 5.6%) — a 54x gap at the large tercile vs an 11x gap at the small
  one. This is the same heavy-tail/blowup effect as the pooled-vs-median split, now shown to be
  monotonic in expression size rather than a handful of unrelated outliers: bigger expressions
  build proportionally larger combinatorial equivalence classes (more comm/assoc churn per useful
  rewrite), which is exactly the regime a budget-bounded Guide is meant to help with — and exactly
  where this measurement says the labeler bound is least trustworthy as a training target.

## What the strict bound actually measures

`derivation_ancestors` (the labeler's substrate) is a deliberately conservative
over-approximation on three named axes (`provenance.rs` lines ~214-239): it credits every node
in a *class*, not just the node actually chosen; it pulls in union events by class membership;
it has no fixed-point pruning. The strict bound instead walks *only* the chosen derivation
tree — root, then each chosen node's chosen children, recursively (identical walk to
`labeler::chosen_tagged_nodes`, reimplemented against the crate's public API in the harness
since that internal function isn't `pub` and this round doesn't touch `labeler.rs`) — and
credits an application only if its output node is literally one of those chosen nodes.

This makes the strict bound a genuine *lower* bound, not a better estimate: it is blind to
"enabling" contributions — a `commutative` firing that never becomes the chosen node but was
necessary for congruence closure to later discover the equivalence that *did* get chosen is
real credit the strict walk cannot see. The gap between the two bounds is exactly that
enabling-credit mass, and it turns out to be enormous:

- **Structural/combinatorial rules (commutative, associative, reverse-associative, distribute,
  identity, annihilator, involution) score 73-93% under the labeler bound and ~0.0% under the
  strict bound**, for every one of them. These rules essentially never produce the literal node
  that survives extraction — extraction almost always resolves to a canonical, unrewritten
  form — but they fire in enormous volume (commutative alone: 2.3M firings) building the
  equivalence classes that let *other* rules' rewrites become reachable from the root.
- **Numeric/transcendental rules (power-recip, power-sqrt, power-rsqrt, recip-sqrt,
  even-negation) are the opposite shape**: 13-18% under the labeler bound but their strict
  scores (6-13%) are the *closest* to their labeler scores of any rule family — these rules'
  products more often survive directly into the extracted expression.
- **Rank correlation between the two bounds, per rule instance, is ≈0** (Spearman ρ = 0.023,
  n = 55 rules that fired). A Guide trained on labeler labels and one trained on strict labels
  would learn almost unrelated rule priorities. This is the load-bearing risk this measurement
  surfaces for Phase 3 (see below) — not "the ratio is inflated," which the toy-kernel report
  already knew, but "the *ordering* the two bounds imply is uncorrelated," which is new.

## Per-rule table, full 62-rule library (`pixelflow_search::math::all_rules()`)

39 distinct rule *names* fired (several are instantiated once per operator — e.g. `commutative`
covers both `Add` and `Mul` as separate `Rewrite` objects/rule indices; this table sums across
all instances sharing a name). 7 rule names never fired on any of the 800 sampled expressions:
**involution, odd-negation, ln-homomorphism, power-zero, log-power, differentiate** (odd-negation
and involution each have an instance that fired and one that didn't — `power_kind`-gated variants
that this corpus's generator families never happen to trigger; `differentiate` is inert without a
`Dwrt` node, matching its doc comment). Full per-rule-instance breakdown (62 rows, keyed by
rule index) is in the sibling JSON (`docs/results/2026-08-30-guide-headroom.json`, `per_rule`).

| rule | fired | labeler LB | labeler % | strict LB | strict % |
|---|---:|---:|---:|---:|---:|
| identity | 121,572 | 108,569 | 89.3% | 0 | 0.0% |
| commutative | 2,329,614 | 1,918,615 | 82.4% | 0 | 0.0% |
| fma-fusion | 1,403,501 | 1,148,569 | 81.8% | 1,483 | 0.1% |
| associative | 1,070,772 | 786,164 | 73.4% | 145 | 0.0% |
| distribute | 236,390 | 173,554 | 73.4% | 11 | 0.0% |
| reverse-associative | 1,145,762 | 839,677 | 73.3% | 151 | 0.0% |
| doubling | 41,325 | 26,994 | 65.3% | 83 | 0.2% |
| constant-fold | 130,201 | 82,575 | 63.4% | 1,607 | 1.2% |
| annihilator | 31,924 | 19,243 | 60.3% | 0 | 0.0% |
| factor | 308,002 | 156,732 | 50.9% | 192 | 0.1% |
| halving | 54,566 | 27,142 | 49.7% | 1 | 0.0% |
| half-angle-product | 3,299 | 1,608 | 48.7% | 1 | 0.0% |
| cos-angle-addition | 2,026 | 937 | 46.2% | 2 | 0.1% |
| sin-angle-addition | 2,633 | 986 | 37.4% | 1 | 0.0% |
| canonicalize | 6,539 | 2,141 | 32.7% | 57 | 0.9% |
| pythagorean | 644 | 200 | 31.1% | 0 | 0.0% |
| exp-homomorphism | 895 | 247 | 27.6% | 2 | 0.2% |
| odd-negation | 3,780 | 887 | 23.5% | 29 | 0.8% |
| reverse-angle-addition | 8,034 | 1,842 | 22.9% | 34 | 0.4% |
| power-combine | 4,827 | 1,000 | 20.7% | 143 | 3.0% |
| inverse-annihilation | 7,110 | 1,425 | 20.0% | 20 | 0.3% |
| diff-of-squares | 102 | 20 | 19.6% | 0 | 0.0% |
| power-recip | 9,182 | 1,664 | 18.1% | 1,159 | 12.6% |
| power-recurrence | 6 | 1 | 16.7% | 0 | 0.0% |
| expand-square | 6 | 1 | 16.7% | 0 | 0.0% |
| power-sqrt | 13,995 | 2,332 | 16.7% | 1,526 | 10.9% |
| power-rsqrt | 3,303 | 543 | 16.4% | 422 | 12.8% |
| involution | 37,650 | 5,703 | 15.1% | 0 | 0.0% |
| recip-sqrt | 3,363 | 504 | 15.0% | 408 | 12.1% |
| even-negation | 77,521 | 11,085 | 14.3% | 4,897 | 6.3% |
| exp-ln-cancel | 231 | 33 | 14.3% | 0 | 0.0% |
| power-identity | 919 | 124 | 13.5% | 0 | 0.0% |
| exp2-log2-cancel | 270 | 36 | 13.3% | 0 | 0.0% |
| idempotent | 1,012 | 82 | 8.1% | 0 | 0.0% |
| cancellation | 449 | 36 | 8.0% | 0 | 0.0% |
| log2-power | 14 | 1 | 7.1% | 0 | 0.0% |
| ln-exp-cancel | 231 | 3 | 1.3% | 0 | 0.0% |
| log2-exp2-cancel | 267 | 0 | 0.0% | 0 | 0.0% |
| power-expand-2 | 6 | 0 | 0.0% | 0 | 0.0% |

(Sorted by labeler %, names merged across operator instances — see JSON for the un-merged,
rule-index-level table the harness actually computed from.)

**Rule-triage reading, with the caveat that "reading" differs by which bound you trust:**
- Under the labeler bound alone, nothing looks safe to drop — even the bottom rows
  (`log2-exp2-cancel`, `power-expand-2`) simply have low fire counts (267, 6), not low ratios
  distinguishable from noise.
- Under the strict bound, the entire top of the labeler table (identity, commutative,
  fma-fusion, associative, distribute, reverse-associative, annihilator, involution) reads as
  **wasted** — 0.0-0.1%. Taken at face value that would argue for aggressively rule-masking
  exactly the rules the labeler bound says matter most. Taking it at face value would be a
  mistake (see above): these are the rules whose entire job is enabling other rewrites via
  congruence closure, which the strict walk cannot see by construction. **This table cannot be
  used for rule triage until the labeler/strict disagreement is understood** — that is this
  report's main finding, not a footnote to it.

## Design implications for Phase 3 (measurement-only conclusions; no Guide built this round)

1. **Headroom exists and is large under either bound** — 2.6x-34x fewer applications would
   suffice for the median expression's extraction. This clears the bar the extraction-head
   program's static/noswap=0.54 cleared for Phase 2: there is real slack for a Guide to
   recover. Recommendation: **Phase 3 is worth running**, contingent on point 2.
2. **The labeler/strict gap is the central open risk, not a rounding error.** A Guide is only
   ever going to be trainable on the labeler bound (the strict bound isn't a candidate training
   signal — it's blind to real enabling credit, as noted above). But this measurement shows the
   labeler bound's *ranking* of rules is uncorrelated with the strict bound's, and the two
   bounds' *magnitudes* differ by 20-400x depending on how you pool. Before spending Phase 3
   budget training against labeler labels, the follow-up docs/plans/2026-07-07 lines 88-90
   already called for — tightening the union-causality over-approximation — should be
   understood well enough to know whether a *tighter* over-approximation (still safe, i.e.
   never under-crediting) would substantially change the rule ranking above. This round
   deliberately did not touch that logic (house rule: measure the looseness, don't redesign it
   yet); the next round should.
3. **Per-expression heterogeneity is itself a design input.** The heavy tail (65% of all
   applications from 6.25% of expressions) means a fixed per-episode saturation budget hits
   wildly different "how much of this could a Guide have skipped" targets depending on
   expression size/shape. A budget-bounded Guide (as Phase 3 proposes) should be evaluated
   across the size distribution, not just on a pooled or median number — a Guide that only
   helps small expressions (where full saturation is already cheap) is a different result than
   one that helps the blowup cases (where it's most needed).
4. **7/62 rules never fired on 800 corpus-representative expressions.** Not evidence they're
   dead — this corpus doesn't specifically target the differentiation/homomorphism cases they
   guard — but worth a wider or targeted corpus pass before any rule-masking decision leans on
   "never fires" as a signal.

## Related concurrent scoping measurements (same session, same worktree, 2026-08-30)

Two other Phase-3-scoping measurements were produced alongside this one and are cross-referenced
here rather than duplicated:

- **`docs/results/2026-08-30-guide-scope-saturation-delta.json`** — asks whether the Stockfish
  incrementality argument (the reason NNUE-style accumulators are cheap) applies to the Guide's
  *state* representation (the e-graph as it grows during saturation), as distinct from the
  extraction-head program's negative finding for *extraction candidates*
  (`docs/plans/2026-08-17-egraph-vsa-nnue-research-notes.md`: ~2x, not ~98x, because sibling
  extraction candidates differ by median 44.9% of their edge multiset). Measured against the real
  production (batched) saturation algorithm: 91.1% of all recorded rule applications are
  idempotent re-fires that create zero new nodes/edges (an even stronger form of incrementality
  than "small delta" — no update needed at all), and among the remainder, the per-application
  edge-delta fraction has median 0.14% (~731x implied speedup) — but eval economics are
  separately expensive: median 153 match evaluations per saturation round, 90.4% of all evaluated
  candidates producing no committed rewrite action (the same idempotent-refire mechanism viewed
  from the candidate side; see that file's harness doc for the root cause — rules like
  `commutative` have no "already applied" check and re-match their own already-installed output
  forever).
- **`docs/results/2026-08-30-oracle-filtered-budget-curves.csv`** (may still be running at the
  time this document was written — check the file's presence/timestamp) — the most direct dry
  run of the Phase 3 thesis test at small scale: anytime extraction-cost curves comparing
  unguided (all 62 rules) saturation against a rule set restricted to only the rules a hindsight
  pass found load-bearing, sampled at standardized work fractions. This is the closest thing in
  this session's output to actually answering "does rule filtering keep pace with full
  saturation" — this report's headroom numbers establish *that there is room* for such filtering
  to help; that file (or its successor once complete) is where "and does it actually work,
  greedily, at a budget" gets a first empirical answer.

Neither of the above changes this report's numbers or conclusions; both were produced by other
agents working the same worktree/task concurrently and are noted here for a single point of
entry into everything measured this session.

## Reproducibility

- Corpus identity: regenerated this session, `gen_bench_corpus --target 8000 --seed 42`
  (default seed). `corpus_train.bin`: 6,480 entries; `corpus_dev.bin`: 1,512 entries; quarantine
  1.17% excluded, 0 JIT-vs-oracle miscompiles.
- Rule set: `pixelflow_search::math::all_rules()`, 62 rules (40 algebra/parity/trig/exp/power +
  2 fusion + 1 differentiation — see that module's doc comment for the exact split; the older
  "40+2=42" comment in `egraph/mod.rs` is stale relative to `math/mod.rs`'s "59+2+1=62", which
  this measurement confirms by construction (`all_rules().len() == 62`)).
- Extraction: `CostModel::latency_prior()` (the compiler's default, per Phase 2's still-standing
  gate).
- Full structured output: `docs/results/2026-08-30-guide-headroom.json` (per-expression rows,
  per-rule-index rows, quartiles, pooled ratios — this document's tables are derived from it).
- Harness: `pixelflow-pipeline/src/bin/guide_headroom.rs`.
