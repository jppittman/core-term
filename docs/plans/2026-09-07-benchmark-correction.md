# Benchmark correction: real shaders are the corpus

**Date:** 2026-09-07
**Status:** plan of record for the measurement side of the e-graph program. Retractions in §D are applied in the same PR.
**Authority:** JP, 2026-09-07 — *"It's an egraph for shaders project, and we have an app with shaders and the moment we pointed it at the real shaders it sucked. Fonts are real shaders. They're an example of what we're targeting. ... the synthetic benchmarks need to be made better. Gaps identified and closed. We need to act like when our research hit the real world, it shit the bed, so we need to go back and correct."*
**Inputs (read in full, not restated here):**
- `docs/results/2026-09-07-claims-ledger.md` (+ `.csv`, 88 rows; PR #1207, branch `claude/claims-ledger` until it merges) — every quantitative claim, its corpus, units, instrument and verdict.
- `docs/results/2026-09-07-corpus-structural-gaps.md` (+ `.csv` 777 × 90, `.json`; PR #1212, branch `claude/corpus-gaps`) — the seven ranked structural gaps between the 207 real kernels and the 570 synthetic ones.
- `docs/results/2026-09-07-egraph-off-vs-on-real-shaders.md` (+ rows; PR #1210, branch `claude/egraph-off-real-2`) — F, the e-graph on/off/cse-only/with-select-hoist arms on every shipped kernel (partial: chrome, psychedelic, cell grid, glyph16 landed; glyph32/bench/shader_bench in flight when the run was killed).
- `docs/plans/2026-09-06-egraph-at-production-scale.md` §7 (the F column and the A–F table), `docs/plans/2026-09-01-phase3-registration.md` (the registration format §E follows).

---

## A. The verdict on the method

Of the 88 quantitative claims the program minted since the self-play era, **53 do not stand**: 7 failed on a real shader, 14 were priced in tree cost where the kernel pays DAG cost, 18 came from an instrument later shown wrong (a 41.67× then 4× timebase, a null context so `Gather` was never priced, a tile-mode loop that inverts against production's scanline, an overhead-biased ratio, cross-population Spearman, artifacts not in the repository), and 14 were never taken on a shipped kernel at all — including the most-quoted number in the program, "extraction is worth ~2×". The 35 that stand are almost all measurements taken on real shaders in the last week or findings that an instrument was broken; **no claim that an optimizer improvement helps a shipped shader survives except sharing-aware extraction (#1192) and the S3b schedule win, which is codegen**. The structural inventory explains why every synthetic headline collapsed on contact: the 570 synthetic kernels contain **zero** selects, compares, gathers, buffers or uniforms, share at a median ratio of 1.0 against 5.2 on real kernels, are 32 nodes against 1,215, and 45% of them name coordinate axes the emitter refuses; the synthetic corpus said saturation was worth −37.5% of `dag_cost` where the real kernels pay −4.7%. And F, now measured, says what the e-graph is worth on real shaders today: **hash-consing is the product** — the `cse-only` arm (insert, extract, zero rewrite rounds) already delivers −41% of glyph bytes, −70% on the cell grid, −77% on psychedelic, and without it the chrome scene does not compile at all (a 335,411-node tree whose un-saturated schedule overflows the aarch64 branch range); the rewrite rules add −4% bytes / −2% `dag_cost` on glyphs beyond that, and on the scenes they *raise* bytes slightly (psychedelic 2,176 → 2,368, cell grid 2,512 → 2,576, chrome_R 2,608 → 2,640) while `dag_cost` falls — extraction optimizes the latency prior, not the emitted schedule. Saturation is 86–97% of compile time on every headline kernel, stops on the class cap after 2–3 rounds on all of them, and the one rule the demand plan indicted (`SelectHoistUnary`) is not registered in production and, when added, changes the emitted bytes of 0 of 97 kernels. The method's failure is not any one claim; it is that the benchmark was a draw from an op distribution and the product is a composition over a lattice, and nothing minted on the first can be read on the second.

---

## B. The corrected benchmark

### B.1 The corpus is the set of shipped kernels, by family

| family | members | source (all already dumpable) | what it exercises |
|---|---|---|---|
| **glyph bakes** | 95 printable ASCII × tile 16 / 32, per font (DejaVu Sans Mono: 190; NotoSansMono-Regular: 190) | `production_glyph_arena_dump`, `glyph_compile_report` | selects over band predicates (95% of kernels), row prologue (73% of the schedule at row scope), `Dwrt` sharing (5.2×), the class cap binding on the input (`U+0040`: 0 iterations) |
| **scene kernels** | `bench_scene_chrome`, `bench_scene_psychedelic` — the ONE packed kernel `PackedManifold::compile` hands the optimizer, at 1920×1080 | `dump_production_scene_arenas` | splice duplication (chrome 1,082×), guards (chrome 3.6× from S3b), the uniform (psychedelic), transcendentals |
| **cell grid** | 80×24 @1×, 80×24 @2×, 120×40 @2× | `dump_production_cell_grid_arenas` | gathers (14) over buffers (2), integer index arithmetic, trip-weighted memory traffic (the term `dyn_memory_ops` was named for and never trained on) |
| **shader_bench** | the 12 ShaderToy ports from their live definitions | `pixelflow-pipeline/examples/saturation_worth.rs`, `shader_and_psychedelic_arena_dump` | the only transcendental-bearing real kernels (4% of real vs 88% of synthetic); `mandelbrot_distance` shares 3,915× |

Fonts are an instance of the target, not the target. The scene and shader families need more members, and the gap doc's structural columns say what a new member must have to count. **Every new member is held out at birth** (§B.2) until the dev set needs it, and the promotion is journaled.

### B.2 The split — family-aware, and honest about contamination

Every prior decision was made on some part of this corpus; a "held-out" set that pretends otherwise is the paper's FINAL tier again (ledger L026: never opened, then consumed for a rule-order decision). So the split states what each member has already been used for, and the held-out set is what has been used for *nothing*.

**DEV** (every selection decision, every tuning, every registration's B and Y):
- DejaVuSansMono (190 bakes) — the gap inventory's font;
- NotoSansMono-**Regular** (190 bakes) — the font of the 2026-09-01 saturation telemetry (L053–L056) and the #1192 rebase gate (`glyph_compile_report`); it has driven two decisions and is not virgin;
- the cell grid at 80×24 @1× and 80×24 @2×;
- `bench_scene_psychedelic` — drove the S1 `bake` number (L070) and the rule-order reversal reading (L043);
- the 12 `shader_bench` ports — consumed by the rule-order decision (L057).

**HELD-OUT** (opened only by a pre-registered publication run; every opening journaled; a decision that reads it promotes the member to DEV permanently and the program owes a new held-out member before its next claim):
- **font: `assets/font/Noto_Sans_Mono/static/NotoSansMono-Bold.ttf`** — a distinct set of outlines that no harness, gate, dump or report has ever loaded (`grep -rn Bold docs/ pixelflow-*/tests pixelflow-*/examples` is empty). NotoSansMono-Regular does not qualify: it is in two results docs on main. 95 printable ASCII × tile 16 / 32 at core-term's 16 pt cell height, both atlas densities — the production shapes.
- **scene: `bench_scene_chrome`** — held out because *every* prior decision touched it (S3 shipped it at 0.32×, S3b hand-found the 3.6×, the cap 5,000 → 60,000 A/B, the #1192 250 ms compile gate). It is the kernel the program failed on, so the question it answers is the one JP asked — does a correction made on DEV generalize to the shader that broke us — and it is **frozen from 2026-09-07**: no decision reads a chrome number until a registered publication run. Its contamination is stated, not hidden: the S3b guards were hand-written against it, so a chrome win that is "S3b again" is not evidence.
- **cell grid: 120×40 @2×** — the third geometry; the two 80×24 geometries stay DEV.
- The next scene kernel written for the benchmark is held out at birth; the next ShaderToy port likewise.

The publication run reports DEV and HELD-OUT side by side, never pooled. A claim whose HELD-OUT row is missing is a DEV claim and is labelled one.

### B.3 The metrics, per family, in the order the shader pays them

Deterministic columns are the claim. The clock is a sign, taken only at 1-minute load < 8, labelled with its load, and reported as "not taken" otherwise; per-kernel clock ratios under ~10% are not trusted (L076). Compile time is reported **beside** every row and is never the objective.

| family | 1st | 2nd | 3rd | beside (never the objective) | external oracle |
|---|---|---|---|---|---|
| glyph bakes | **row-prologue guarded fraction** — guarded entries / schedule entries at row scope (`PIXELFLOW_GUARD_TELEMETRY`), because a curved glyph is all row prologue (L080) and `O`@32 is 16× slower per pixel at 40 px than at 640 | **µs/row** at the production lattice (tile 16 / 32, scanline loop) | bytes, spill slots, schedule entries, `dag_cost` | compile ms per glyph and Σ warm over 95 (the "5 ms per kernel = half a second" yardstick) | **FreeType** coverage agreement per pixel (the #1187 oracle — the only instrument that saw the `'8'` waist); same-form scalar oracle with NaN mismatch count |
| scene kernels | **ns/px** at 1920×1080, scanline, 1 T and N T | bytes, guarded / selects, spill slots | trip-weighted memory ops, `dag_cost` | compile ms (chrome's 250 ms gate stays a gate) | pixel agreement against the `off`/`cse` arm at a 1-byte tolerance per channel (FMA rounding is the contract; 58/256 pixels by ≤1 byte on chrome is agreement, not divergence) |
| cell grid | **µs/frame** at 80×24 @2× (the frame core-term draws) | trip-weighted memory ops | bytes, spills | compile ms | byte-identity of the rendered grid against the `off` arm (no transcendentals: it must be exact) |
| shader_bench | **ns/px** at the bench shape | bytes, `dag_cost` | guarded / selects | compile ms | same-form scalar oracle (NaN count, max |Δ|) **and** the range bound (`trig_range.rs`: a value outside the function's range is not imprecise, it is wrong) |

Every number carries its instrument in the row: ISA level, loop shape (scanline, never tile — L083 inverts), timebase (the 4× `INPUT_TUPLES`/`INPUT_VECTORS` confusion of L065 fixed or not), context (real buffers bound, so `Gather` is priced), load, and which oracle.

### B.4 The harness — instruments already in the tree

- **`collapse_cost bench`** at real shapes — a valid e-graph instrument (the ledger's correction of L078: `compile_as_baked` calls `optimize_runtime_arena`). This is D of production-scale §7: the oracle, not the product.
- **`corpus_gaps`** (PR #1212) — the 90 structural columns per kernel; the acceptance instrument for §C.
- **`egraph_off_on`** (PR #1210) — the off / cse-only / on / with-rule arms through the production path's three calls, asserted byte-identical to `Manifold::compile`.
- **`PIXELFLOW_GUARD_TELEMETRY`** per scope, **`--features saturation-telemetry`** per compile (applications, iterations, classes, typed stop), **`glyph_compile_report`** (compile ms, byte hashes per glyph).
- **Oracles**: the FreeType comparison test from #1187; `eval_scalar` same-form with the NaN count reported, never a clamp; the rendered-frame byte comparison for the packed kernels.

### B.5 The gate

A gap in CI is a check to write. The journal already names the gap (`ci_gap`, 2026-09-06): no presubmit job compiles a production scene or a glyph warm. The check: a job that compiles the DEV corpus and diffs the **deterministic** columns against the base commit — bytes, schedule entries, guard counts per scope, spill slots, `dag_cost`, saturation stop reason — and fails on any row that regresses without a journaled reason. No clock in CI (flakes are real failures; a clock at unknown load is a flake generator). This is re-validation item 0 in §E because everything after it is measured against it.

---

## C. The generator, retargeted — or demoted

The gap doc measured seven properties the real kernels have and the generators do not. For each: the concrete change, the acceptance test (the gap doc's own column, closed when the synthetic median lands inside the real [p10, p90]), or the statement that the property cannot come from a random generator and the synthetic set is demoted to coverage on that axis.

| # | gap (real vs synthetic, median) | generator change | acceptance (from `corpus_gaps` columns) or demotion |
|---|---|---|---|
| 1 | selects / 100 nodes 7.2 vs 0; kernels with a select 95% vs 0%; arm-exclusive 0.978 vs 0; guarded / schedule 0.88 vs 0 | **Construction, not draw.** A kernel is `p.select(a, b)` where `p` is a band predicate on `Y` or the sign of a drawn scalar field, and `a`, `b` are drawn *disjointly* (no shared subterm across arms). Nested to depth 2–3 as glyphs are. | *Static* half closes when synthetic medians land in real [p10, p90]: selects / 100 nodes ∈ [3.95, 7.83], arm-exclusive ≥ 0.96, guarded / schedule ≥ 0.41. *Dynamic* coherence (which pixels actually take which arm) is a property of the function's level sets, not its syntax — **cannot be drawn; coverage only**. Coherence is measured on real kernels (E of production-scale §7) and nowhere else. |
| 2 | extracted tree / dag 5.2 vs 1.0; kernels sharing 99% vs 44%; input tree / dag 16.6 vs 1.0 | Build DAGs on purpose: a drawn value `v` is reused ≥ 2× and `v.dx()` / `v.dy()` are kept beside `v` (`Dwrt` is 6.5% of real nodes, 0% synthetic); compose by value so splice copies appear. `sh_family` (5.5×) and `bezier_family` (4.2×) are the templates. | Closes when extracted tree / dag median ∈ [2.12, 17] and fraction sharing ≥ 0.9. The *shape* of real sharing (a hit-path normalization the whole kernel hangs from) is coverage only. |
| 3 | schedule hoisted out of the pixel loop 78% vs 41% (bimodal, p10 0 / p90 1); row-scope 73% vs 6%; LICM values 72 vs 1; 45% of synthetic name `Var(2)`/`Var(3)` and cannot be emitted | Production's variable vocabulary only — two axes plus uniforms; `Var(2)`/`Var(3)` deleted from the generator (259 / 570 rows were unmeasurable, not unrepresentative). A deliberate two-level construction: a row part (a function of `Y` alone) consumed by a pixel part, with the row / pixel work ratio a parameter fitted per family (glyph 0.74 / 0.21, scene 0.16 / 0.39, grid 0.29 / 0.71, shader 0.12 / 0.89). | Closes when hoisted fraction median ∈ [0.6, 0.9] **and** p10 > 0.3 (no longer bimodal), and LICM-hoisted values median ≥ 20. |
| 4 | splice duplication 1.72 glyph / 4.8 grid / 11 psychedelic / 1,082 chrome vs 1.11; ClassCap stop 93% vs 24%; `U+0040` 0 iterations | **Cannot come from a generator** — it is a property of how `Kernel::at` / `select` / `sum` splice copies at construction. Demoted. The fix is engineering on the production side (hash-cons at construction, production-scale §4 item 1), after which the row disappears from the *real* column too. | Acceptance is on the real side: duplication factor → 1.0 on every family, and `U+0040` gets > 0 iterations at production budget. Until then the class cap must count rewrite-minted classes, not input classes. |
| 5 | transcendental-bearing 4% vs 88%; real `MulAdd` 15.9%, `Select` 7.4%, compares 10.9%, `Dwrt` 6.5%, `Sqrt` 4.5%, `Div` 3.3% vs synthetic `Neg` 16.3%, `Abs` 7.9%, `Pow` 5.0%, `Div` 0.15%; 15 rules fire only on synthetic; `identity(Add)` in 93% of real vs 2% | Fit the op weights to the **real op column** of the gap table (font geometry and ray–sphere arithmetic, not ShaderToy). Emit the splice seams composition leaves (`x + 0`, `x · 0`, `x − x` at boundaries) as the construction does. | Closes when no rule fires on one population only at ≥ 0.5% of firings and `identity(Add)` fires in ≥ 50% of synthetic kernels. Where the ops *sit* (a `Sqrt` shared by a value and its derivative; a compare that is a select mask) is a pattern, not a histogram — coverage only. |
| 6 | gathers / buffers / uniforms 3 / 3 / 1 of 207 vs 0 of 570 | **Cannot, usefully.** A buffer plus `Gather` at drawn coordinates is one function; the index arithmetic (atlas slot from cell, packed-word channel extraction) is a construction. Demoted. The corpus for this axis is the real cell grid at more geometries. | Coverage only. `dyn_memory_ops` and any successor are fitted on the cell-grid family and nothing else. |
| 7 | arena nodes 1,215 vs 32; e-graph input 823 vs 29; depth 42 vs 13; schedule entries 1,048 vs 16; bytes 7,952 vs 768; largest band 63 unique nodes vs median glyph 734 | Size bands re-cut to the real column: the construction of rows 1–3 composed until the hash-consed size is 500–3,000 nodes. | Closes when synthetic median e-graph input nodes ≥ 500 and the ClassCap stop fraction ≥ 50% — the regime production runs in. Scale is a *stress* property: a synthetic kernel at scale is for finding panics and budget cliffs, not for a cost number. |

**The rule, binding from this date:** **the synthetic corpus is never a headline.** A number minted on a generated kernel may appear in a results doc only in a section titled *Coverage / stress*, with the real column beside it. The abstract, the verdict table, the journal `finding`, and the PR title name a shipped shader — or the claim is NEVER TESTED ON REAL by the ledger's definition and is labelled so in its own row. A generator axis is *open* until its acceptance column overlaps; while open, results on that axis are coverage, and the doc says which axes are open. No refit of a synthetic corpus revalidates a retracted claim; only the real column does.

---

## D. Retractions applied in this PR

Each doc below now opens with a dated **Retracted/Superseded (2026-09-07)** block — one sentence naming the ledger row and pointing at `docs/results/2026-09-07-claims-ledger.md`. Where a doc carries rows that HELD beside the retracted one, the block names the row it retracts and leaves the rest standing.

| rows | doc(s) on main | scope of the block |
|---|---|---|
| L004 | `docs/plans/2026-07-07-guided-saturation-redesign.md` | the Apr-2 "4.9% over LLVM" number (41.67× timebase); L001–L003 stand |
| L007 | `docs/results/2026-07-08-extraction-3way.md` | whole doc (timebase; audit H1–H3 apply) |
| L028 | `docs/results/2026-08-30-guide-headroom.md` | the ρ = 0.35 per-rule correlation (two draws give 0.35 and 0.19); L027/L029 are untested-on-real and labelled |
| L031 | `docs/results/2026-08-30-oracle-filtered-budget-curves.md` | whole doc (grid sampled after 97.8% of runs ended; tree units) |
| L033 | `docs/results/2026-09-01-phase3-unguided-baseline.md` | whole doc (tree units; B and Y were derived from it) |
| L033, L034 | `docs/plans/2026-09-01-phase3-registration.md` | the registered constants B = 100 / 200 and Y = 16.3% / 9.0% |
| L035 | `docs/results/2026-09-01-phase3-at-budget-eval-report.md`, `-at-budget-eval.md` | the "Guide saves 16%" headline (tree units; loss on every structured family in DAG units) |
| L036 | `docs/results/2026-09-01-control-guide-comparison.md`, `2026-09-01-train-guide-report.md` | whole doc (own banner: predates fixes, never re-run) |
| L038 | `docs/results/2026-09-01-phase3-round1b-domain-shift.md` | the H_null verdict (both arms lose on the real metric); L039 (half-angle pays on sh) stands |
| L041 | `docs/results/2026-09-01-tightened-labeler-rank.md` | whole doc (own banner) |
| L042 | `docs/results/2026-09-01-round2-unguided-vs-rulecount-v2.md` | whole doc (order confound; tree units) |
| L043 | `docs/results/2026-09-01-round2-unguided-vs-rulecount-v3.md` | the 86× order headline (≈3% on real, reversed on psychedelic; tree units) |
| L044 | `docs/results/2026-09-01-r2g-dataset.md`, `-r2g-trajectory-mint-full.md` | the dataset gate (tree-cost return labels) |
| L045 | `docs/results/2026-09-01-guide-return-to-go.md`, `-counterfactual-credit.md` | the Claim B delta (tree-cost difference) |
| L046 | `docs/results/2026-09-01-guide-r2g-spread-first.md`, `-counterfactual-credit-multimask.md` | the regime-restricted Spearman (tree units) |
| L047 | `docs/results/2026-09-02-bilinear-guide.md`, `-bilinear-guide-training.md` | the H_form-null verdict (extractor was tree-objective; re-take under #1192 only if §E item 5 is non-null) |
| L054 | `docs/results/2026-09-01-production-saturation-telemetry.md` | the "class cap costs 8.66%" figure only (tree-DP; sign contradicted by the chrome clock); L053, L055, L056 stand |
| L065 | `docs/results/2026-09-02-extraction-objective.md` | the absolute cost-table numbers it quotes from #1134 (4× optimistic; `Gather` unpriced); L066 (0 of 206 extractions move) stands |
| L078 | `docs/plans/2026-09-06-kernel-with-a-lattice.md` | the S4b-1 landing-block statement that `collapse_cost bench` bypasses the optimizer — wrong on main |
| L082 | `docs/plans/2026-09-07-demand-is-a-dag-property.md` | §9's "a tight Y-extent gate fixes the waist" (an always-true gate fixed it equally; font thread, out of scope here) |
| L087 | `docs/plans/2026-08-31-guide-design-revision.md` | §5's "Y = half the truncation loss at B = 100 / 200" framing (85× below the real regime; tree units) |
| L088 | `docs/plans/2026-09-01-schedule-cost-model-denotation.md` | the "cost model does 95% of the work" premise and the reranker-as-next-step (cross-population synthetic ρ; production-scale §7: B is not a step toward C) |

**Marked in the ledger only** (no doc on main): L006 (project memory), L013–L017 and L022 (the paper, branch `claude/workshop-writeup`, PR #1072 closed), L057–L061 and L065's own doc (closed branches `all-rules-numeric-first`, `class-cap-live`, `cap-break-ab`, `upward-congruence`, `extraction-objective`).

---

## E. The re-validation plan

Ordered cheapest-and-most-load-bearing first. Each item is registered in the phase-3 format before its run: environment fixed, metric verbatim, decision rule pre-committed, the null stated as a result. Every item runs on the DEV corpus of §B.2 with the HELD-OUT rows reported beside only at a registered publication run. Deterministic columns are the claim; the clock is a sign at load < 8 or "not taken".

| # | item | ledger rows | what runs | pre-committed decision rule | what a null means |
|---|---|---|---|---|---|
| 0 | **The CI gate** (§B.5) | ledger §6 "artifacts in the repository" | a presubmit job diffing deterministic columns of the DEV corpus against base | the job exists, blocks, and fails on a seeded synthetic regression (one glyph's bytes bumped) | not applicable — this is the instrument everything below is read on |
| 1 | **Finish F** — glyph32, `bench`, `shader_bench` rows (driver scripts in the F worker's scratchpad: `run_class.sh`, `diff_commit.sh`), then the clock on chrome, psychedelic, cell grid and `O`@32 at load < 8 | L020 | `egraph_off_on` off / cse / on / +hoist; `collapse_cost` scanline for the sign | if rules-beyond-CSE move ns/px by < 5% on every DEV family, the research target is **extraction of schedules (C)** and the saturation half waits (production-scale §7); if ≥ 5% on any family, both halves matter and the family is named | the e-graph's product on real shaders is hash-consing + FMA fusion + constant folding + canonicalization; that is a result, and it re-prices every rule-quality claim to ~0 |
| 2 | **Rule order on real kernels in DAG units** | L057, L043 | the existing dumps through `optimize_runtime_arena` under numeric-first vs default, `dag_cost` and bytes per kernel | keep the default order unless numeric-first is ≤ 0.97 median `dag_cost` **and** worse on no family **and** bytes do not grow on any scene | order is not a lever on shipped kernels (expected: ≈3%, reversing on psychedelic); closes L043 for good |
| 3 | **Class cap in DAG units, against the chrome clock** | L054, L058–L060, L072 | cap 5,000 vs live-counted vs 4× lifted on DEV; `dag_cost`, bytes, compile ms; the clock on the DEV scene and `O`@32 | ship a cap change only if `dag_cost` does not rise on any family, bytes do not grow on any scene, compile ms stays under chrome's 250 ms, and the clock is not worse; a class cap that counts *rewrite-minted* classes is engineering and ships on `U+0040` getting > 0 iterations | more saturation is not the lever (expected — L055, L072); the class-cap program closes and the budget stays where it is |
| 4 | **The latency prior on a fixed instrument** | L065, L083, L066 | fix the 4× timebase, bind a real context so `Gather` is priced, scanline not tile; re-measure; re-price DEV | the new table is adopted only if it changes ≥ 1 DEV extraction *and* the changed kernels are not worse in bytes or the clock; if it changes 0 of 190 glyph extractions again (L066: a 33× perturbation moved none), the table is recorded as not a lever on glyphs and the scene delta is the whole claim | the cost table is not where glyph quality lives; the saturated graph holds no alternative lowering, so extraction of *schedules* is the only place a model can act (agrees with item 1's expected reading) |
| 5 | **Guide-at-budget in DAG units at the real regime** — only after 0–4 | L034, L035, L038, L087 | the linear Guide vs unguided at production's actual stop (median ~8,400 applications, class cap as environment) on DEV, `dag_cost` per kernel | the Guide is a production candidate only if guided ≤ unguided `dag_cost` on every DEV family (fonts, grid, scene, shader) with a per-family domain-shift ablation; one family lost = not shipped | expected null; a null closes the Guide as a production feature honestly (the library and the labeler stay as instruments; rules are domain-conditional, so no global per-rule prior is ever quoted again) |
| 6 | **Bilinear under #1192** — only if item 5 is non-null | L047 | as item 5 with the bilinear head | same rule as item 5 | closed with item 5 |
| 7 | **The paper's parity** — only if the paper is revived | L013 | paired, same-search arms; ratio without the shared overhead term | not registered here | stays retracted |

**Not worth re-taking**, with reasons:
- **R2G** (L044–L046): "the target was not the bottleneck" is a negative result unlikely to flip in DAG units; the trajectory apparatus is kept as an instrument only.
- **Round 2 v2** (L042): superseded by v3, which itself failed on real (item 2 covers what is left).
- **NNUE extraction overhead and incrementality** (L021, L024): the architecture is deleted (#1093); the `Reranker` seam is not a step toward C (production-scale §7).
- **The self-play era** (L004–L007) and **L005**: apparatus deleted; nothing to re-take.
- **The `SelectHoistUnary` indictment** (demand plan §4): F already answered it — not registered in production, and with it added, 0 of 97 kernels change a byte. Closed, not re-taken.
- **L082** (the `'8'` waist gate): font thread; §F.

---

## F. Out of scope

- **Font geometry** — the `'8'` waist bug (#1187, open on main) and every rasterizer question stay in the font thread. This plan uses FreeType as an oracle and touches no outline code.
- **The "8 waist" bug** itself: referenced as the reason an external oracle is mandatory, not worked here.
- **New optimizer features** — no rule, no cap change, no cost-table change, no Guide training, no reranker — until items 0–1 of §E have landed and the benchmark of §B is the thing they are measured on. Hash-cons at construction (production-scale §4 item 1) is engineering, not an optimizer feature, and is the one production change §C item 4 asks for.

---

## Return to the orchestrator

- **Verdict:** §A, one paragraph.
- **Held-out:** NotoSansMono-Bold (never loaded by anything), `bench_scene_chrome` (frozen from 2026-09-07; contamination by S3/S3b/cap/#1192 stated), cell grid 120×40 @2×; every new family member held out at birth. NotoSansMono-Regular is DEV (two results docs on main).
- **First three re-validation items:** 0 the CI gate on deterministic columns; 1 finish F and take its clock at load < 8 with the < 5% rules-beyond-CSE rule; 2 rule order on real kernels in DAG units with the ≤ 0.97-and-no-family-worse rule.
