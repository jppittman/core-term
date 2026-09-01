# Integration audit: what the cost-model research actually changed in the compile paths (2026-09-01)

Audited at `origin/main` 5568a807, in a detached read-only worktree. Three traces (compile-time,
runtime, inventory) were run independently and then synthesized here; **every `file:line` below was
re-opened during synthesis**, and the places where a trace was wrong are listed in §6. No source
file was modified; this document is the only write.

Status vocabulary, used throughout:

- **LIVE-DEFAULT** — runs in the default compile path with no opt-in.
- **OPT-IN** — env var, feature flag, or explicit API a caller must choose.
- **LIBRARY-ONLY** — public API with no production caller in `pixelflow-compiler`,
  `pixelflow-runtime`, `pixelflow-graphics`, or `core-term` (research/test/bench callers only).
- **DEAD** — unreachable.

JP's question, verbatim: *"What's actually wired up today? Is our cost table that does 95% actually
active?"*

---

## 1. Flat answers

### Q: Is the measured latency table the one `kernel!` uses today?

**Yes.** The chain is: `kernel!` → [`pixelflow-compiler/src/lib.rs:113`](../../pixelflow-compiler/src/lib.rs#L113)
`optimize::optimize` → [`optimize.rs:97,103`](../../pixelflow-compiler/src/optimize.rs#L97-L103)
`env_extraction_policy()` → [`pixelflow-search/src/egraph/extraction.rs:86-92`](../../pixelflow-search/src/egraph/extraction.rs#L86-L92),
whose `Err(_)` arm (env var unset) is `ExtractionPolicy::Static(Box::new(CostModel::latency_prior()))`
→ the table at [`pixelflow-search/src/egraph/cost.rs:81-133`](../../pixelflow-search/src/egraph/cost.rs#L81-L133).

That table **is** the Round-1 measured table. `git show e4388c54 -- pixelflow-search/src/egraph/cost.rs`
(commit "remeasure the latency prior", 2026-08-14) shows the handwritten values it replaced, and
`git diff e4388c54..HEAD -- pixelflow-search/src/egraph/cost.rs | grep -E '^[+-].*OpKind::'` shows
**no `OpKind::X => N` table line has changed since** (only test and accessor refactors). Headline
entries, today vs pre-Round-1:

| Op | Today | Pre-Round-1 | Location |
|---|---:|---:|---|
| `Pow` | **196** | 12 | [`cost.rs:112`](../../pixelflow-search/src/egraph/cost.rs#L112) |
| `Sqrt` | 15 | 15 | [`cost.rs:90`](../../pixelflow-search/src/egraph/cost.rs#L90) |
| `Rsqrt` | **21** | 5 | [`cost.rs:91`](../../pixelflow-search/src/egraph/cost.rs#L91) |
| `Recip` | **16** | 10 | [`cost.rs:96`](../../pixelflow-search/src/egraph/cost.rs#L96) |
| `Div` | **11** | 15 | [`cost.rs:88`](../../pixelflow-search/src/egraph/cost.rs#L88) |
| `Exp` / `Exp2` | **75** / **69** | 10 / 10 | [`cost.rs:106-107`](../../pixelflow-search/src/egraph/cost.rs#L106-L107) |
| `Ln` / `Log2` | **128** / **122** | 10 / 10 | [`cost.rs:108-109`](../../pixelflow-search/src/egraph/cost.rs#L108-L109) |
| `Sin` / `Cos` / `Tan` | **70** / **75** / **87** | 10 / 10 / 10 | [`cost.rs:100-102`](../../pixelflow-search/src/egraph/cost.rs#L100-L102) |
| `Neg` / `Abs` | **3** / **3** | 1 / 1 | [`cost.rs:89,92`](../../pixelflow-search/src/egraph/cost.rs#L89) |
| `Min` / `Max` | **3** / **3** | 4 / 4 | [`cost.rs:93-94`](../../pixelflow-search/src/egraph/cost.rs#L93-L94) |
| `MulAdd` | 5 | 5 | [`cost.rs:95`](../../pixelflow-search/src/egraph/cost.rs#L95) |

Two tests pin this: [`optimize.rs:1539`](../../pixelflow-compiler/src/optimize.rs#L1539)
`default_path_extraction_is_static_latency_prior` (default `optimize()` output is byte-identical to an
explicit `Static(latency_prior())` run) and
[`pixelflow-search/tests/latency_prior_regression.rs`](../../pixelflow-search/tests/latency_prior_regression.rs)
(`Pow(x, 0.5)` must extract as `Sqrt` through the runtime tier).

The learned model is never selected: `PIXELFLOW_NNUE_WEIGHTS` appears in exactly three source files
(`grep -rn PIXELFLOW_NNUE_WEIGHTS --include='*.rs' --include='*.toml' --include='*.yaml' --include='*.yml' --include='*.py' --include='*.sh' .`
→ `pixelflow-compiler/src/optimize.rs`, `pixelflow-search/src/egraph/extraction.rs`,
`pixelflow-search/src/runtime.rs`); `.cargo/config.toml` has no `[env]` section and
`grep -rln PIXELFLOW_NNUE_WEIGHTS .github .cargo` is empty.

**One caveat none of the traces caught.** There is a *second* macro-tier e-graph. For any
non-composing kernel whose arena contains `Dwrt` (i.e. uses `DX`/`DY`),
[`pixelflow-compiler/src/ir_bridge.rs:479-484`](../../pixelflow-compiler/src/ir_bridge.rs#L479-L484)
`ast_to_runtime_arena` calls
[`differentiate_in_optimizer`](../../pixelflow-compiler/src/ir_bridge.rs#L661-L740), which builds its own
`EGraph::with_rules(standard_rules())`, saturates with
[`eg.saturate()`](../../pixelflow-compiler/src/ir_bridge.rs#L722) (the `graph.rs` defaults —
**100 iterations / 10,000 classes / 500 ms**, [`graph.rs:811-813`](../../pixelflow-search/src/egraph/graph.rs#L811-L813))
and extracts with [`extract(&eg, root_class, &CostModel::default())`](../../pixelflow-compiler/src/ir_bridge.rs#L724).
`CostModel::default()` is `latency_prior()` ([`cost.rs:193-210`](../../pixelflow-search/src/egraph/cost.rs#L193-L210)),
so **the answer to Q1 is still yes** — but this path does not go through `env_extraction_policy()`, so
the NNUE opt-in is silently ignored there, and its budget is not `config_for_node_count`'s. It is reached
from all three optimizing macros: `kernel!` via
[`codegen/emitter.rs:1246`](../../pixelflow-compiler/src/codegen/emitter.rs#L1246), `kernel_jit!` via
[`jit_backend.rs:218`](../../pixelflow-compiler/src/jit_backend.rs#L218) (`emit_jit`), `kernel_value!` via
[`jit_backend.rs:338`](../../pixelflow-compiler/src/jit_backend.rs#L338) (`emit_kernel_value`). It is
live in core-term's build: the glyph winding segment is a `kernel_value!` using `DX`/`DY`
([`pixelflow-graphics/src/fonts/ttf_curve_analytical.rs:106-129`](../../pixelflow-graphics/src/fonts/ttf_curve_analytical.rs#L106-L129)).

### Q: Does the runtime JIT path run the same optimizer, and when?

**Yes — same rules, same budget tiers, same extraction policy.**
[`pixelflow-codegen/src/jit_cache.rs:39-57`](../../pixelflow-codegen/src/jit_cache.rs#L39-L57) forces
`pixelflow_search::runtime::optimize_runtime_arena` inside `compile` ("not a step callers get to
sequence"). [`pixelflow-search/src/runtime.rs:106-137`](../../pixelflow-search/src/runtime.rs#L106-L137)
uses `EGraph::with_rules(all_rules())` (the 62-rule set,
[`math/mod.rs:117-122`](../../pixelflow-search/src/math/mod.rs#L117-L122), pinned at 62 by
[`math/mod.rs:457-466`](../../pixelflow-search/src/math/mod.rs#L457-L466)), `config_for_node_count`
([`runtime.rs:126-133`](../../pixelflow-search/src/runtime.rs#L126-L133)) and `env_extraction_policy()`
([`runtime.rs:135`](../../pixelflow-search/src/runtime.rs#L135)) — the identical three calls the macro
tier makes at [`optimize.rs:49-50,103,143-150`](../../pixelflow-compiler/src/optimize.rs#L143-L150).

Differences, all documented in code, none changing the rule set or policy:

- The tier-selection proxy: macro tier counts raw AST nodes
  ([`optimize.rs:62-88`](../../pixelflow-compiler/src/optimize.rs#L62-L88)); runtime tier counts
  reachable arena nodes after `lower_dwrt` ([`runtime.rs:472`](../../pixelflow-search/src/runtime.rs#L472)).
- Derivatives: runtime tier resolves `Dwrt` *before* the e-graph
  ([`runtime.rs:120`](../../pixelflow-search/src/runtime.rs#L120)); macro tier differentiates *inside*
  a separate e-graph (the `ir_bridge.rs` path above).
- Bail-outs: `Param` and `Nary` (the `Reduce` binder) return `None` → arena compiles unoptimized
  ([`runtime.rs:413,441`](../../pixelflow-search/src/runtime.rs#L413)). No reachable core-term kernel
  uses `Kernel::over`/`Reduce` (runtime trace grep of `cell_grid.rs`, `atlas.rs`, `ttf.rs`,
  `ttf_curve_analytical.rs`: no hits).
- The `SaturationResult` is computed and dropped
  ([`runtime.rs:128-133`](../../pixelflow-search/src/runtime.rs#L128-L133) — call not bound; return type
  at [`runtime.rs:74`](../../pixelflow-search/src/runtime.rs#L74) is `Option<Arc<(ExprArena, ExprId)>>`).
  No production caller can observe whether a real kernel quiesced or hit a cap.

**When it runs** (core-term):

| Trigger | Count | Evidence |
|---|---|---|
| Cell-grid packed kernel: any geometry change (cols/rows/cell px/atlas extents) | compiles **on each transition to a different geometry** — a resize that leaves cols/rows/cell dims/atlas extents unchanged compiles nothing; only the current program is retained (single `Option` slot, not a cache) and the kernel's `Buffer` leaves make it uncacheable in `jit_cache`'s global dedup ([`jit_cache.rs:16-18,60-64`](../../pixelflow-codegen/src/jit_cache.rs#L60-L64)), so an A→B→A resize recompiles geometry A twice, not once | [`core-term/src/terminal_app.rs:362-388`](../../core-term/src/terminal_app.rs#L362-L388) |
| Glyph SDF kernel: first use of a codepoint | 1 per distinct `char`, memoized in `slots` | [`pixelflow-graphics/src/fonts/atlas.rs:168-184`](../../pixelflow-graphics/src/fonts/atlas.rs#L168-L184) |
| Startup ASCII warm at density 1.0 | 95 bakes | [`terminal_app.rs:201-205`](../../core-term/src/terminal_app.rs#L201-L205) |
| Atlas rebuild after density change | +95 bakes on any HiDPI display | [`terminal_app.rs:239-245`](../../core-term/src/terminal_app.rs#L239-L245), callers `:320,468,515` |
| Per frame | **never** | [`terminal_app.rs:373`](../../core-term/src/terminal_app.rs#L373) reuse gate |

Every production bake enters through `jit_cache::compile`:
[`pixelflow-core/src/lattice/mod.rs:402`](../../pixelflow-core/src/lattice/mod.rs#L402) (`Lattice::bake`),
[`cell_grid.rs:393,598`](../../pixelflow-core/src/lattice/cell_grid.rs#L598) (`CellGridProgram`/`CellGridPackedProgram::compile`).
The packed cell-grid kernel is 623 reachable nodes (pinned,
[`cell_grid.rs:1427-1446`](../../pixelflow-core/src/lattice/cell_grid.rs#L1427-L1446)) → classical tier.
The runtime trace measured ASCII glyph kernels at 271–3,687 raw nodes (all classical) with an
out-of-tree probe against the repo's fixture font; those numbers are carried forward from that
trace, not re-measured here.

### Q: Which research artifacts are LIVE-DEFAULT / OPT-IN / LIBRARY-ONLY / DEAD?

| Artifact | Status | Evidence |
|---|---|---|
| `CostModel::latency_prior()` measured table | **LIVE-DEFAULT** (both tiers, plus the `Dwrt` path) | [`extraction.rs:91`](../../pixelflow-search/src/egraph/extraction.rs#L91), [`ir_bridge.rs:724`](../../pixelflow-compiler/src/ir_bridge.rs#L724), [`cost.rs:208-209`](../../pixelflow-search/src/egraph/cost.rs#L208-L209) |
| 62-rule set `math::all_rules()` | **LIVE-DEFAULT** | [`optimize.rs:49-50`](../../pixelflow-compiler/src/optimize.rs#L49-L50), [`runtime.rs:122`](../../pixelflow-search/src/runtime.rs#L122), [`ir_bridge.rs:720`](../../pixelflow-compiler/src/ir_bridge.rs#L720) |
| `config_for_node_count` tiers | **LIVE-DEFAULT** (macro + runtime); **not used** by the `Dwrt` path | [`saturate.rs:196-202`](../../pixelflow-search/src/egraph/saturate.rs#L196-L202) |
| `extract_dag` (static DP extraction) | **LIVE-DEFAULT** | [`extraction.rs:58-64`](../../pixelflow-search/src/egraph/extraction.rs#L58-L64) |
| `Extraction` validated type (`from_dp`) | **LIVE-DEFAULT** as a wrapper; its `try_swap` refinement is NNUE-only | [`extraction.rs:63`](../../pixelflow-search/src/egraph/extraction.rs#L63); `try_swap` only at [`extract.rs:460`](../../pixelflow-search/src/egraph/extract.rs#L460) inside `IncrementalExtractor` |
| Legalization passes (`legalize`) | **LIVE-DEFAULT**, lowering only, after extraction | [`pixelflow-ir/src/passes.rs:39,63`](../../pixelflow-ir/src/passes.rs#L63); callers [`pixelflow-codegen/src/emit/mod.rs:1892,1967,1985,2008`](../../pixelflow-codegen/src/emit/mod.rs#L1892) |
| Rule-provenance **recording** | **LIVE-DEFAULT** — unconditional in `add`/`union`, no cfg gate | [`graph.rs:227,317`](../../pixelflow-search/src/egraph/graph.rs#L227), `Provenance::new()` at [`graph.rs:137,157`](../../pixelflow-search/src/egraph/graph.rs#L137); `grep -n "cfg(feature" graph.rs provenance.rs` → none |
| Rule-provenance **consumption** (`derivation_ancestors`, `labeler.rs`) | **LIBRARY-ONLY** | callers: `pixelflow-search/src/egraph/{graph,labeler,mod,provenance}.rs`, `pixelflow-search/examples/oracle_filtered_budget_curves.rs`, `pixelflow-pipeline/src/bin/guide_headroom.rs` (grep `derivation_ancestors --include='*.rs'`) |
| NNUE extraction head (`ExprNnue`, `IncrementalExtractor`, #1048 `try_swap`) | **OPT-IN** via `PIXELFLOW_NNUE_WEIGHTS`, never set; hard-panics on a bad file; **ignored** by the `Dwrt` path | [`extraction.rs:86-128`](../../pixelflow-search/src/egraph/extraction.rs#L86-L128) |
| `SaturationGuide` / `Guide` / `GraphAccumulator` / mask head | **LIBRARY-ONLY** (self-described "inert today") | [`nnue/guide/mod.rs:1-30,88`](../../pixelflow-search/src/nnue/guide/mod.rs#L11-L18); files: `pixelflow-search/src/nnue/{factored.rs,guide/*}`, `pixelflow-pipeline/src/{bin/guide_scope_saturation_delta.rs,training/unified_backward.rs}` |
| `saturate_guided*` / `anytime.rs` | **does not exist on main** | `grep -rn saturate_guided --include='*.rs' .` → none; `ls pixelflow-search/src/egraph/` has no `anytime.rs` |
| `EGraph::saturate()` / `saturate_with_limit` (100 / 10,000 / 500 ms) | **LIVE-DEFAULT via the `Dwrt` path only**; otherwise research/test | production caller [`ir_bridge.rs:722`](../../pixelflow-compiler/src/ir_bridge.rs#L722); other callers: `pixelflow-codegen/tests/prod_kernel_jit.rs`, `bench_extraction_3way.rs`, `profile_extraction.rs`, `labeler.rs`, `derivative.rs` (tests), `tests/fold_exactness.rs` |
| `kernel_raw!` | LIVE macro, **no production caller** | `grep -rn 'kernel_raw!' core-term/src pixelflow-graphics/src pixelflow-runtime/src` → none; users are `pixelflow-compiler/tests`, `pixelflow-pipeline/benches`, `pixelflow-runtime/examples`, `pixelflow-search/src/egraph/codegen.rs` |
| `eval_scalar` oracle | **TEST-ONLY** in `pixelflow-ir`/`-compiler`/`-codegen`/`-graphics`/`-search` (the `oracle`-enabling edge is `[dev-dependencies]` in each); **HARNESS/TRAINING code, not TEST-ONLY, in `pixelflow-pipeline`** — `training = ["pixelflow-search/std", "pixelflow-ir/oracle"]` is enabled by `default`, is an ordinary `[dependencies]` feature (not dev-only), and gates real `[[bin]]` targets (`gen_bench_corpus`, `bench_extraction_3way`) that call `eval_scalar` directly. Still outside the shipped core-term path in both cases; "under dev-dependencies everywhere" is false. | [`pixelflow-ir/src/eval.rs:1-7`](../../pixelflow-ir/src/eval.rs#L1-L7); [`pixelflow-ir/Cargo.toml:29`](../../pixelflow-ir/Cargo.toml#L29); edges in compiler/codegen/graphics/search `Cargo.toml` are dev-deps; [`pixelflow-pipeline/Cargo.toml:12,19,26-31`](../../pixelflow-pipeline/Cargo.toml#L19) (default `training` feature as a non-dev dependency edge), bin targets at [`Cargo.toml:68-70,89-91`](../../pixelflow-pipeline/Cargo.toml#L89-L91) |
| `BenchSession` / sentinel / `jit_bench` | **HARNESS-ONLY** | `pixelflow-pipeline` only (grep) |
| `Fence` / `FenceKey` / `SchemaIdentity` / `JournalEntry` | **HARNESS-ONLY** | `pixelflow-pipeline/src/{schema.rs,bin/*,training/*}` only (grep) |
| `CellGridProgram` | **LIBRARY-ONLY**, not DEAD — it is a public type in `pixelflow-core`, re-exported from `pixelflow-core`'s `lib.rs`, so a normal downstream build can reach `CellGridProgram::compile`; no production caller exists, which is exactly this document's LIBRARY-ONLY definition. Only its one in-tree usage — the retired four-plane parity-oracle benchmark in `scene.rs` — is test-only. | [`pixelflow-core/src/lattice/cell_grid.rs:367`](../../pixelflow-core/src/lattice/cell_grid.rs#L367) (`compile`); re-export [`pixelflow-core/src/lib.rs:267-268`](../../pixelflow-core/src/lib.rs#L267-L268); test-only usage at [`pixelflow-graphics/src/render/scene.rs:373-384,447`](../../pixelflow-graphics/src/render/scene.rs#L373-L384) (`#[cfg(test)]` module); core-term uses `CellGridPackedProgram` instead ([`terminal_app.rs:14,119,373`](../../core-term/src/terminal_app.rs#L14)) |

### Verdict

The one research result that changed what a user runs is Round 1's remeasured table: `Pow` 12→196,
`Rsqrt` 5→21, `Recip` 10→16, transcendentals 10→70–134, and it is live in every compile path core-term
has — the `kernel!`/`kernel_jit!`/`kernel_value!` macros at build time, the runtime `Lattice::bake`
path for every glyph and cell-grid kernel, and the macro-time `Dwrt` e-graph. Everything else the
program produced — the trained extraction head, incremental extraction, the Guide, the labeler — is
either behind an env var nobody sets or has no caller outside research binaries. Provenance recording
is the odd one out: paid on every production compile, read by nothing production-facing. Two things
should temper "the table does 95%": the 0.9438/0.9486 Spearman comparison
([`docs/plans/2026-08-17-egraph-vsa-nnue-research-notes.md:105-106`](../plans/2026-08-17-egraph-vsa-nnue-research-notes.md#L105-L106))
was measured under a saturation budget production does not use (§2), and the `Dwrt` path shows the
macro tier is not one e-graph but two, with different budgets and only one of them honoring the
extraction-policy switch.

---

## 2. Production saturation configuration vs. the research tier

### 2a. What production actually runs

| Constant | Value | Where set | Who uses it |
|---|---:|---|---|
| `SaturationConfig::blitz()` (≤10 nodes) | 20 iters / 500 classes / 10 ms | [`saturate.rs:158-164`](../../pixelflow-search/src/egraph/saturate.rs#L158-L164) | macro tier [`optimize.rs:143-150`](../../pixelflow-compiler/src/optimize.rs#L143-L150); runtime tier [`runtime.rs:126-133`](../../pixelflow-search/src/runtime.rs#L126-L133) |
| `SaturationConfig::rapid()` (11–50) | 50 / 2,000 / 50 ms | [`saturate.rs:167-173`](../../pixelflow-search/src/egraph/saturate.rs#L167-L173) | same |
| `SaturationConfig::classical()` (51+) | 100 / 5,000 / 200 ms | [`saturate.rs:176-182`](../../pixelflow-search/src/egraph/saturate.rs#L176-L182) | same — every measured core-term kernel (623-node cell grid, 271–3,687-node glyphs) lands here |
| Tier thresholds | `0..=10`, `11..=50`, `_` | [`saturate.rs:196-202`](../../pixelflow-search/src/egraph/saturate.rs#L196-L202) | same |
| `EGraph::saturate()` defaults | 100 / 10,000 / 500 ms | [`graph.rs:811-813`](../../pixelflow-search/src/egraph/graph.rs#L811-L813) | **`Dwrt` macro-time path only** ([`ir_bridge.rs:722`](../../pixelflow-compiler/src/ir_bridge.rs#L722)) |
| Stop conditions inside the one loop | iteration cap, `classes.len() > max_classes`, elapsed ≥ timeout, `unions == 0` | [`graph.rs:846-879`](../../pixelflow-search/src/egraph/graph.rs#L846-L879) | all of the above |
| What the loop *counts* | `iterations`, `total_unions` only | [`graph.rs:881-884`](../../pixelflow-search/src/egraph/graph.rs#L881-L884) | — there is no rule-application counter in the production stats |
| Extraction | `extract_dag` + `latency_prior` | [`extraction.rs:58-64,91`](../../pixelflow-search/src/egraph/extraction.rs#L58-L64) | macro + runtime; `Dwrt` path uses tree `extract` ([`extract.rs:951`](../../pixelflow-search/src/egraph/extract.rs#L951)) |
| NNUE top-k when opted in | 8 | [`extraction.rs:55`](../../pixelflow-search/src/egraph/extraction.rs#L55) | opt-in only |

### 2b. What the research harnesses ran, and what the pre-registration assumes

The "registration doc" is [`docs/plans/2026-08-31-guide-design-revision.md`](../plans/2026-08-31-guide-design-revision.md)
§5 ("Pre-registered Phase 3 experiment"). Its three supporting measurements are listed in its §1 table
(lines 19-30).

| Source | Saturation configuration | Matches production? |
|---|---|---|
| `bench_extraction_3way.rs` — the harness behind the 0.9438 / 0.9486 Spearman numbers | `eg.saturate_with_limit(SATURATE_LIMIT)`, `SATURATE_LIMIT = 40` → **40 / 10,000 / 500 ms**, one fixed operating point for every kernel size ([`bench_extraction_3way.rs:132-133,1318-1320`](../../pixelflow-pipeline/src/bin/bench_extraction_3way.rs#L132-L133)) | **MISMATCH.** The comment says "matches `prod_kernel_jit.rs`" — that file is [`pixelflow-codegen/tests/prod_kernel_jit.rs:58`](../../pixelflow-codegen/tests/prod_kernel_jit.rs#L58), a test, not a production path. Versus classical: 40 vs 100 iterations, 10,000 vs 5,000 classes, 500 vs 200 ms; versus blitz/rapid the gap is larger in the other direction. |
| `guide_headroom.rs` (§2.1 headroom bounds) | `SATURATE_MAX_ITERS = 100`, `SATURATE_MAX_CLASSES = 10_000`, `SATURATE_TIMEOUT = 60 s` ([`guide_headroom.rs:120-127`](../../pixelflow-pipeline/src/bin/guide_headroom.rs#L120-L127)) | **MISMATCH, and mislabeled.** The module doc calls 100 / 10,000 / 500 ms "`EGraph::saturate()`'s production default" ([`:29-34`](../../pixelflow-pipeline/src/bin/guide_headroom.rs#L29-L34)); no production compile reaches `EGraph::saturate()` except the `Dwrt` path. Production caps are size-tiered and 2×–20× smaller in classes. In wall-clock they are 2.5×–50× shorter (50×/10×/2.5× for blitz/rapid/classical's 10/50/200 ms) against that mislabeled 500 ms value, and 300×–6,000× shorter (6,000×/1,200×/300×) against this harness's actual 60 s timeout. |
| `guide_scope_saturation_delta.rs` (§2.2 incrementality) | CLI defaults `max_iterations = 100`, `max_classes = 3000`, timeout 3600 s, driven one `saturate_with_limits(1, ..)` round at a time ([`:150-157,299,310`](../../pixelflow-pipeline/src/bin/guide_scope_saturation_delta.rs#L150-L157)) | **MISMATCH** on class cap (3,000 vs 500/2,000/5,000) and timeout; the batched per-round algorithm itself is the production one. |
| `oracle_filtered_budget_curves.rs` (§2.3 anytime curves) | `FRACS = [0.25, 0.5, 1.0, 2.0, 3.0]` × the expression's own `config_for_node_count` nominal iterations and classes; 300 s safety ceiling ([`pixelflow-search/examples/oracle_filtered_budget_curves.rs:140-147,495-519`](../../pixelflow-search/examples/oracle_filtered_budget_curves.rs#L140-L147)) | **MATCH on iteration and class nominals** (the only harness that reads the production tiers). Wall-clock ceiling deliberately not production's. Its own report calls the run inconclusive as calibrated (97.8% of expressions quiesce before the first checkpoint). |
| `training/episodes.rs` (corpus generation) | randomized per episode: 10–60 iterations, 50–2,000 classes, ≤5 s ([`episodes.rs:236-247`](../../pixelflow-pipeline/src/training/episodes.rs#L236-L247)) | **MISMATCH by design** — documented as "budget-capped for training-corpus diversity". |
| §5 pre-registered experiment | Budget **denominated in rule applications** ("median ~195, heavy-tailed"), cost = `CostModel::latency_prior()`, "no wall-clock timing", family-held-out split ([`guide-design-revision.md:289-345`](../plans/2026-08-31-guide-design-revision.md#L289-L345)) | **MISMATCH in the budget's unit.** Production budgets rounds, classes, and milliseconds, selected by node count. It does count rule applications — every rule action calls `Provenance::record_application` ([`graph.rs:1014`](../../pixelflow-search/src/egraph/graph.rs#L1014)), and the public `application_count()` reports the running total ([`provenance.rs:158-182`](../../pixelflow-search/src/egraph/provenance.rs#L158-L182)) — but that count is not used as a stop condition and is not surfaced in `SaturationStats`/`SaturationResult` (the loop's own counters remain just `iterations`/`total_unions`, [`graph.rs:881-884`](../../pixelflow-search/src/egraph/graph.rs#L881-L884)). The wall-clock ceiling that production has (10/50/200 ms) is excluded from the experiment by construction. The cost model matches. |

What the mismatch means: a Guide validated under §5's protocol is validated on a machine whose budget
is "N applications, unlimited time" while production's is "≤100 rounds, ≤5,000 classes, ≤200 ms". Until
one of the two adopts the other's denomination, the experiment's accept gate does not speak to what
`kernel!` or `Lattice::bake` will do. The same applies retroactively to the 0.9438/0.9486 numbers:
they characterize extraction after 40 rounds at a 10,000-class / 500 ms ceiling, a point no
production kernel is saturated to.

---

## 3. What it would take

No code here — files that change, in dependency order.

### 3a. A budget knob on production saturation

1. [`pixelflow-search/src/egraph/saturate.rs`](../../pixelflow-search/src/egraph/saturate.rs) — `SaturationConfig` grows the new dimension (e.g. an application count) and `config_for_node_count` gets an override/scale input.
2. [`pixelflow-search/src/egraph/graph.rs:835-885`](../../pixelflow-search/src/egraph/graph.rs#L835-L885) — `saturate_with_limits` must *count* the new dimension and stop on it; today's `SaturationStats` has only `iterations` and `total_unions`.
3. [`pixelflow-compiler/src/optimize.rs:143-150`](../../pixelflow-compiler/src/optimize.rs#L143-L150) and [`pixelflow-search/src/runtime.rs:126-133`](../../pixelflow-search/src/runtime.rs#L126-L133) — the two `config_for_node_count` callers.
4. [`pixelflow-compiler/src/ir_bridge.rs:720-722`](../../pixelflow-compiler/src/ir_bridge.rs#L720-L722) — the third saturation site, currently hardcoded to `EGraph::saturate()`; it needs to take the same config or the knob has a hole.
5. [`pixelflow-search/src/runtime.rs:74`](../../pixelflow-search/src/runtime.rs#L74) — if the knob is to be observable, the `SaturationResult` currently dropped at `:128` has to come back out (a `_with_stats` sibling or a log line) — but surfacing that value alone is not sufficient: `saturate_with_full_budget` sets `saturated` whenever `stats.iterations < max_iterations` ([`saturate.rs:122`](../../pixelflow-search/src/egraph/saturate.rs#L122)), which is also true after an early timeout or class-limit exit, so the flag cannot distinguish quiescence from a capped run. Observability needs an explicit termination-cause field, not just this dropped value exposed. (PR #1083, open as of this writing, adds exactly that — a `stop_reason` read off `SaturationResult` behind a `saturation-telemetry` feature flag — but it is not yet merged, so the gap described here is current on `main`.)

### 3b. Guided saturation behind `SaturationGuide`, in both compile paths

1. [`pixelflow-search/src/egraph/graph.rs:860-871`](../../pixelflow-search/src/egraph/graph.rs#L860-L871) — the rule loop (`for rule_idx in 0..n_rules { batch.apply_rule(rule_idx, ..) }`) is the seam: ordering or masking `rule_idx` by a score is where a Guide acts.
2. [`pixelflow-search/src/nnue/guide/mod.rs:88`](../../pixelflow-search/src/nnue/guide/mod.rs#L88) — the `SaturationGuide` trait already exists; `score_candidates` needs a `GraphSummary`/`RuleCandidate` producer wired from the loop above.
3. [`pixelflow-search/src/egraph/saturate.rs:104`](../../pixelflow-search/src/egraph/saturate.rs#L104) — `saturate_with_full_budget` takes an optional guide.
4. The three callers: [`optimize.rs:145`](../../pixelflow-compiler/src/optimize.rs#L145), [`runtime.rs:128`](../../pixelflow-search/src/runtime.rs#L128), [`ir_bridge.rs:722`](../../pixelflow-compiler/src/ir_bridge.rs#L722).
5. A weights-selection function modeled on [`extraction.rs:86-128`](../../pixelflow-search/src/egraph/extraction.rs#L86-L128) (env var → `OnceLock` → hard panic on a bad file), so the Guide gets the same one-policy-two-tiers treatment as the extraction head.
6. A trainer that produces those weights: [`pixelflow-pipeline/src/training/unified_backward.rs`](../../pixelflow-pipeline/src/training/unified_backward.rs) currently has no gradient path into `nnue/guide` (by design, per the module doc), and `saturate_guided*` does not exist on `main`.

### 3c. Provenance cost, if it must be on for the Guide

The Guide's *training* labels come from `derivation_ancestors` offline; a *deployed* Guide scores
candidates from graph state and does not need the journal at compile time. So provenance can be off
in production and on in the harness — if that separation is wanted:

1. [`pixelflow-search/src/egraph/graph.rs:137,157,227,317`](../../pixelflow-search/src/egraph/graph.rs#L227) — `Provenance::new()` in both constructors and the unconditional `record_origin`/`record_union` calls, plus `record_application` at [`graph.rs:1014`](../../pixelflow-search/src/egraph/graph.rs#L1014), gain a switch (a Cargo feature, or a constructor variant).
2. [`pixelflow-search/src/egraph/provenance.rs:137-144`](../../pixelflow-search/src/egraph/provenance.rs#L137-L144) — the store itself (`origins: HashMap`, `applications: Vec`, `unions: Vec`).
3. Consumers that must keep it on: [`labeler.rs`](../../pixelflow-search/src/egraph/labeler.rs), [`guide_headroom.rs`](../../pixelflow-pipeline/src/bin/guide_headroom.rs), [`examples/oracle_filtered_budget_curves.rs`](../../pixelflow-search/examples/oracle_filtered_budget_curves.rs), [`examples/rule_report.rs`](../../pixelflow-search/examples/rule_report.rs).
4. The cost itself was not measured by any trace or by this synthesis; the module doc's "O(1) per event" claim ([`provenance.rs:39-44`](../../pixelflow-search/src/egraph/provenance.rs#L39-L44)) is a claim.

---

## 4. Top 5 results that changed nothing a user runs

| # | Artifact | Why it is not live | Smallest change that would make it live |
|---|---|---|---|
| 1 | Trained NNUE extraction head (`ExprNnue`, `bootstrap_extraction_head` output) | `PIXELFLOW_NNUE_WEIGHTS` is set nowhere in the tree | Add an `[env]` entry to `.cargo/config.toml` pointing at a checked-in weights file — **and** route [`ir_bridge.rs:724`](../../pixelflow-compiler/src/ir_bridge.rs#L724) through `env_extraction_policy()`, or the `Dwrt` path keeps using the static table regardless. |
| 2 | `IncrementalExtractor` / validated-`Extraction` `try_swap` fast path (#1048) | Reached only inside the NNUE policy ([`extract.rs:345-470`](../../pixelflow-search/src/egraph/extract.rs#L345-L470)); the static path is one DP pass with no swap loop | Same gate as #1; nothing else changes. |
| 3 | `SaturationGuide` / `GraphAccumulator` / mask head | No callers, no trainer, no weights; `saturate_guided*` does not exist on `main` | Nothing flips it — it is §3b in full, starting with the `graph.rs` rule-loop seam. |
| 4 | Provenance journal → hindsight labeler | Write side is already live on every compile; read side is `labeler.rs` and research bins only | Same as #3: a deployed Guide is the only production consumer the design names. |
| 5 | Corpus/holdout apparatus (`Fence`, `FenceKey`, `SchemaIdentity`, `JournalEntry`) | `pixelflow-pipeline`-internal bookkeeping by design | None directly; its only route to a user is producing the weights file in #1. |

For contrast, the one that *did* change what a user runs: the remeasured table (§1), shipped in
e4388c54 and untouched since.

---

## 5. Open questions and what the traces could not determine

1. **No production telemetry on saturation.** `optimize_runtime_arena` drops its `SaturationResult`
   ([`runtime.rs:128`](../../pixelflow-search/src/runtime.rs#L128)); the macro tier drops it too
   ([`optimize.rs:145`](../../pixelflow-compiler/src/optimize.rs#L145)). Nobody knows, for the 623-node
   cell-grid kernel or a 3,687-node glyph, whether saturation quiesces, hits 100 rounds, hits 5,000
   classes, or hits 200 ms. This is the first measurement to take before arguing about budgets.
2. **Was the `Dwrt` path's budget intentional?** [`ir_bridge.rs:719`](../../pixelflow-compiler/src/ir_bridge.rs#L719)
   says "Standard optimizer budget" and then uses `EGraph::saturate()` (100 / 10,000 / 500 ms), which is
   not the budget the rest of the macro tier calls standard. Not determinable from code.
3. **Timing figures are carried, not re-run.** The compile-time trace's ~52 ms classical-tier probe
   (102-node synthetic, one machine) and the runtime trace's `bench_jit_compile_cost` split
   (optimize+cache 97.5–99.7% of a cold compile at 8–512 nodes, one run, aarch64) were not
   reproduced by this synthesis. The harness exists
   ([`pixelflow-pipeline/src/jit_bench.rs:1311-1332`](../../pixelflow-pipeline/src/jit_bench.rs#L1311-L1332),
   [`bin/bench_jit_compile_cost.rs:47`](../../pixelflow-pipeline/src/bin/bench_jit_compile_cost.rs#L47)).
4. **The paper is not on `main`.** `docs/paper/2026-08-egraph-nnue-parity.md` exists only on
   `origin/claude/workshop-writeup` (commits b970c8b1, ff1e5f2c; `git log --all -- 'docs/paper/*'`).
   The Spearman numbers on `main` live in the research notes cited in §1.
5. **The "48% truncation loss at B=100" figure** the runtime trace was asked about does not appear in
   any `docs/` file on `main` (`grep -rn "48%" docs` → none); the nearest measurement
   ([`2026-08-30-oracle-filtered-budget-curves.md`](2026-08-30-oracle-filtered-budget-curves.md))
   reports zero observable regret because 97.8% of its corpus settles before the first checkpoint.
6. **Provenance overhead** was not benchmarked by anyone.
7. **x86 transfer of the measured table** is asserted in [`cost.rs:77-79`](../../pixelflow-search/src/egraph/cost.rs#L77-L79)
   as an expectation, not re-verified.
8. **`kernel!` sites in `pixelflow-graphics`** (`scene3d.rs`, `transform.rs`) compile in every
   core-term build (proc-macro expansion), but core-term does not call either module at runtime
   (`grep -rn "scene3d\|transform::" core-term/src` → none). Their tier was not measured.
9. **Doc drift, out of scope but seen:** [`runtime.rs:62`](../../pixelflow-search/src/runtime.rs#L62)
   describes the common path as "through `GlyphCache`"; core-term has no `GlyphCache` reference
   (`grep -rn GlyphCache core-term/src pixelflow-runtime/src` → none) and uses `GlyphAtlas`.
10. **The duplicate saturation loop** flagged as J11 in
    [`2026-08-17-cost-model-domain.md`](../plans/2026-08-17-cost-model-domain.md) is not dead after
    all: `graph.rs::saturate()` is a live production entry via the `Dwrt` path, so unifying it with
    `saturate_with_full_budget` is a behavior change for `DX`/`DY` kernels, not a cleanup.

---

## 6. Corrections to the three traces

Resolved by reading the code; the trace named was wrong on the point listed.

- **Compile-time trace:** `cost.rs` line numbers were off by 2–3 throughout (cited `Sqrt` at 88,
  actual 90; `Recip` 93→96; `Rsqrt` 89→91; `Exp` 104→106; `Log2`/`Ln` 106-107→109/108; `Sin` 98→100;
  `Div` 86→88; `MulAdd` 92→95; table extent 81-113 → 81-133). It listed
  `fonts/ttf_curve_analytical.rs` as a `kernel!` caller; it uses `kernel_value!` (same optimize
  pipeline, [`lib.rs:223-235`](../../pixelflow-compiler/src/lib.rs#L223-L235), so the substantive
  claim stands). It missed the `Dwrt` second e-graph.
- **Runtime trace:** missed the `Dwrt` second e-graph. Otherwise every cited line re-opened clean.
- **Inventory trace:** "`EGraph::saturate`/`saturate_with_limit(s)` has no production caller" is
  wrong — [`ir_bridge.rs:722`](../../pixelflow-compiler/src/ir_bridge.rs#L722), outside the
  `#[cfg(test)]` module that starts at line 809, reached from all three optimizing macros. Its
  description of `CostModel::default()` at `ir_bridge.rs:724` as "test-support extraction path" is
  wrong for the same reason. Its `derivation_ancestors` caller list omitted
  `pixelflow-search/examples/oracle_filtered_budget_curves.rs` (classification unchanged).
- **All three:** agreed, and were right, that the static table is LIVE-DEFAULT, NNUE is OPT-IN and
  unset, the Guide is LIBRARY-ONLY, and the macro and runtime tiers share rules, tiers, and policy.
