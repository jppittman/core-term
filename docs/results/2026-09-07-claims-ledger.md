# The claims ledger: what the research arm said, and what real shaders said back

**Date:** 2026-09-07. **Companion:** `2026-09-07-claims-ledger.csv` (88 rows, one per quantitative claim since the self-play era; columns `id, group, claim, minted, corpus, instrument, metric_units, kind, real_check, verdict, note`).
**Why this exists (JP, 2026-09-07):** *"This isn't an egraph for fonts project. It's an egraph for shaders project, and we have an app with shaders and the moment we pointed it at the real shaders it sucked. ... We need to act like when our research hit the real world, it shit the bed, so we need to go back and correct."* This is the first artifact of that review: every claim, its corpus, its units, its instrument, and its verdict against a shipped shader.

## 1. The verdict

| verdict | rows | meaning |
|---|---:|---|
| **HELD** | 35 | stands on a real shader in correct units, or is a fact about the method that survives every re-unit |
| **FAILED** | 7 | tested on a real shader and contradicted |
| **UNITS INVALID** | 14 | priced in tree cost (`ExtractedDAG::total_cost` before #1192) — the kernel pays DAG cost; the number is in the wrong units even where the corpus was real |
| **INSTRUMENT DEFECT** | 18 | the instrument that produced it was later shown wrong: 41.67× then 4× timebase, null-context gathers, tile-vs-scanline inversion, overhead-biased ratio, cross-population Spearman, unreproducible artifacts, hash-against-prose gates |
| **NEVER TESTED ON REAL** | 14 | minted on a generated corpus and never taken on a shipped kernel |

**53 of 88 do not stand.** Of the 35 that do, **16** are measurements taken on real shaders between 2026-09-01 and 09-06 (`verdict=HELD` and `kind=real`), 8 are findings that an instrument or method was broken (`kind=n/a`), and 11 are synthetic-corpus results that survive re-unit (8 `kind=synthetic`, 3 structured `sh`/`bezier`). Every figure in this section is a `verdict`/`kind` count over the companion CSV and should be recomputed from it rather than restated — an earlier revision said 21/8/6, which double-counted synthetic rows as real-shader measurements. **No claim that an optimizer improvement helps a shipped shader survives**, with two exceptions: sharing-aware extraction (#1192, L067: 55/206 improved, 0 worse, chrome schedule 401→385 entries) and the schedule-side S3b win (L073), which is codegen, not the e-graph.

By group, the failure is concentrated where the headlines were: the extraction-head/paper rows (16) are **10** defect-or-untested (6 INSTRUMENT DEFECT + 4 NEVER TESTED ON REAL); Phase 3 Round 1/1b/2/R2G/bilinear (20) are **10** units-invalid-or-failed (8 UNITS INVALID + 2 FAILED). The self-play era (8) is **5** timebase-defective or unrecorded, not all of it — L001–L003 are HELD, and they are findings *about* the method (the loop closes byte-exact; the extraction head was dead in production; the RL half cannot learn) rather than claims it produced.

The lattice programme's rows (17, real shaders) are **10** HELD. The seven that are not divide three ways: L074–L077 are findings *about* instruments rather than claims that fell, L078 and L082 are FAILED, and L081 is NEVER TESTED ON REAL.

## 2. The five ways the method failed, each with the rows that show it

1. **Units.** Every cost number before #1192 is a tree cost. The kernel pays DAG cost; `julia_set` is 1.4e7 tree against 716 DAG. Fourteen rows are in the wrong units outright (L033–L035, L038, L042, L044–L046, L054, L057–L061), and the Guide's registered constants (B=100/200, Y=16.3%/9.0%) were derived from them and "port as-is, not re-derivable" (L034). The bisect (L049) is the load-bearing correction: the sh family's 0.90 in tree is 1.10 in DAG — every guided arm *loses* on trig kernels, and guides trained on tree labels steer toward unshared terms.
2. **Regime.** The Guide programme was registered at B=100–200 applications. Real kernels fire a median 8,446 before production stops (L056), 85× the registered budget, and 68.4% stop on the class cap (L053). Every quality-at-budget claim (L033–L035, L087) is about a regime production never runs in.
3. **Instrument.** The bench timebase was wrong by 41.67× before 2026-07-20 (L010) and by 4× again after #1071 (L065); the latency prior was measured with a null context pointer so `Gather` has never been priced; labels were minted in tile mode, and Horner-vs-Estrin (L083) shows tile/latency ordering *inverts* against the scanline production loop. The paper's "parity" ratio is `(n+c)/(s+c)`, biased toward 1 (L013); its calibration Spearman is cross-population (L017); three of its rounds rest on artifacts not in the repository (L014, L015, L023). The corpus predictor `dyn_memory_ops` ranks kernels at ρ=0.98 and gets the paired sign right 27% of the time (L075); it reported +40% on a change whose clock moved −7% (L074).
4. **Corpus.** Fourteen rows were never taken on a real shader, including the single most-quoted number in the program — "extraction with the static table is worth ~2× over not extracting" (L020) — for which the raw-versus-optimized delta on a shipped kernel *does not exist*. Where real shaders were finally used, synthetic headlines collapsed: rule order 86× → 0.97× and *reversed* on the psychedelic shader (L043/L057); the class-cap A/Bs that improved tree cost on glyphs were contradicted by the clock on chrome, where 12× more classes made the kernel 15% slower (L072 vs L054/L058/L059). One row (L005) is a claim with no corpus, no number, and a deleted harness.
5. **Provenance.** Five rows carry their own banner that they predate a fix that "changes every number below" and were never re-run (L028, L031, L036, L041); the paper's FINAL tier was never opened (L026), and its 12 ShaderToy ports were then consumed for a rule-order decision, so it no longer holds out ordering claims.

## 3. What stands on real shaders, in correct units

- Production saturation is **class-cap bound** on real kernels (68.4%), fires a median **8,446 applications**, and the cap binds on the **spliced input** — chrome's 416,420 nodes hash-cons to ~4,900 classes, within 90 of the cap, on iteration 2 (L053, L056, L086).
- **More saturation sometimes extracts worse code** (L055), and on the one clocked real shader, 12× more classes cost 30× compile and **+15% ns/px** (L072). Congruence closure and the rebuild orphan do not explain the cap (L062, L063).
- **Extraction was minimizing the wrong quantity.** 95% of the proved greedy gap is tree-vs-DAG (L064); pricing sharing closes 95.9% of it, 55/206 real kernels improve, 0 worsen, and it holds on both production shapes (L067).
- **The cost table is not the lever on glyphs**: a 33× perturbation changes 0 of 190 extracted terms — the saturated graph holds no alternative lowering (L066).
- **No research artifact had ever changed a shipped kernel** as of 09-01: the static table was live, the NNUE head opt-in and never set, the Guide library-only (L002, L068).
- The **schedule** is where the real wins were: S3 shipped chrome at 0.32× (L071), S3b recovered 3.6× via one select per colour, arm clustering and a mispredict bound (L073) — none of it an e-node. A curved glyph is all row prologue (L080); `text()` as a sum is O(n)/px (L079).
- **Additivity is ISA-conditional**: exactly right where AVX-512 is throughput-bound (slope 1.03), wrong by up to 1.9× where latency is exposed (L083) — and the tile-mode instrument inverts against production's loop.
- Budgets are now denominated in applications, sized from real counts (L069); `bake` beats `kernel!` 2.6–2.9× at SSE2 (L070) — a JIT-vs-LLVM fact whose e-graph share is unmeasured.

## 4. Retractions

Each of these gets a dated "Retracted/Superseded" block at the top of its results doc, pointing here (applied by `2026-09-07-benchmark-correction.md`; closed-branch docs are marked here only).

| row | claim | why |
|---|---|---|
| L022 | the additive objective is nearly solved by DP; residual at the noise floor | premise false in its own units until #1192; additivity is ISA-dependent |
| L088 | "the cost model does 95% of the work"; a reranker over swap neighbourhoods is next | rests on a cross-population synthetic Spearman; production-scale §7: B is not a step toward C |
| L087 | Guide framing "Y = half the truncation loss" at B=100/200 | wrong regime (85×) and wrong units |
| L033–L035 | Round 1 baseline, registered constants, "Guide saves 16%" | tree units; DAG reading is a smaller DEV win and a loss on every structured family |
| L038 | domain shift is H-null | compared two arms that both lose on the real metric |
| L042, L043 | Round 2 v2/v3 — |R| effect; order dominates 86× | order confound; 86× is ~3% on real and reverses on psychedelic |
| L044–L047 | R2G dataset gate, R2G Guide, regime-restricted R2G, bilinear H_form-null | tree-cost deltas; bilinear must be re-taken under #1192 |
| L054, L058–L061 | class-cap cost 8.66%; ghost recovery 2.35%; live cap +2.03%; cap-break; congruence −0.07% | tree-DP costs on real kernels; sign contradicted by the chrome clock |
| L004, L006, L007 | self-play era ns numbers | corpus unrecorded; the 08-05 audit's H1/H2/H3. **Not the 41.67× timebase**: all three are ratios (L004 `ns ratio`, L006 `speedup`, L007 1.0669/0.6676/31×), and `2026-07-20-jit-compile-cost.md:29-33` states that a uniform scale factor leaves ratios and orderings unaffected. The verdicts stand on the other grounds |
| L013–L017 | the paper's parity rounds and calibration | overhead-biased ratio, cross-population ρ, unreproducible artifacts, confounded init |
| L028, L031, L036, L041 | headroom ρ, oracle-filtered curves, linear-vs-lookup AUC, tightened labeler | own banner: predates fixes, never re-run |
| L065 | latency-prior remeasure absolute numbers | 4× optimistic; table shift was sound but #1157 moved trig again — re-measure |
| L078 | "collapse_cost bench bypasses the e-graph" | wrong on main: `compile_as_baked` calls `optimize_runtime_arena` — collapse_cost IS an e-graph instrument |
| L082 | a tight Y-extent gate fixes the `'8'` waist | an always-true gate fixed it equally; font thread |

## 5. What must be re-taken, in order

Cheapest and most load-bearing first. Each on the corrected corpus (real shaders, family-split held-out), deterministic columns as the claim, clock as a sign, decision rule pre-committed before the run.

1. **F — the e-graph on vs off, on every shipped shader** (L020). The most-quoted number in the program and it has never been taken. *Running.*
2. **Rule order on real kernels in DAG units** (L057). Dumps exist; deterministic; the direction (≈3%, not 86×) is expected to hold and the magnitude to move.
3. **Class-cap cost and the live-cap A/B in DAG units on real kernels** (L054, L059), against the chrome clock. Until this is done, "more classes" carries a measured contradiction.
4. **The latency prior, on a fixed instrument**: the 4× timebase, a real context so `Gather` is priced, scanline mode not tile (L065, L083). Then re-measure; #1157 changed trig after the last one.
5. **Guide-at-budget in DAG units at the real regime** (~8,000 applications, not 100) on real kernels (L034, L035, L038). Only after 1–4. Expected null; a null closes the question honestly.
6. **The paper's parity** (L013) — only if the paper is revived; otherwise it stays retracted.

**Not worth re-taking:** R2G (L044–L046: "the target was not the bottleneck" is a negative result unlikely to flip in DAG units); Round 2 v2 (superseded by v3, which itself failed on real); NNUE extraction overhead and incrementality (L021, L024: architecture deleted, #1093); the self-play era (apparatus deleted); L005 (nothing to re-take).

## 6. Rules the ledger imposes on the program

- **The synthetic corpus is never a headline.** Coverage and stress only; a claim's row must name a shipped shader or it is NEVER TESTED ON REAL by definition.
- **Units in the row.** `dag_cost` or the clock; a tree cost is not a cost.
- **Regime in the row.** Applications at production's actual stop, not a registered budget.
- **The instrument is a claim too.** Timebase, context, loop shape (scanline), and the oracle (external or same-form) are stated beside every number.
- **Artifacts in the repository.** A number whose per-kernel rows are uncommitted is a note, not a result.
