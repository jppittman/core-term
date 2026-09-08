# Guided-regression bisect: the sh ratio crosses 1.0 at the cost metric, in every era, on the same rows

**Date:** 2026-09-02  
**Question:** PR #1124's bilinear eval measured every guided arm at ~1.10 of unguided on `sh` (B=100, dag cost) where Round 1b measured ~0.90 (tree cost), and DEV linear went 0.537 → 0.653. Four candidate levers landed in between — (L1) `dag_cost` #1117, (L2) the forward-port #1121, (L3) mid-scan application budget #1118, (L4) `rebuild_budgeted` orphan fix #1120. Which one moved the number?  
**Answer:** **L1, alone — verdict (1) INSTRUMENT, §Verdict.** On Round 1b's exact source, corpus and checkpoint — reproducing its tree numbers to four decimals — the same rows priced as a DAG are already 1.098 (control) / 1.114 (linear). No behavioral lever crosses 1.0 anywhere; the tree ratio is ~0.90 in every era and the dag ratio is ~1.10 in every era. DEV's 0.537 → 0.653 is the same metric change (0.5366 tree → 0.6528 dag on the Round 1b source itself); the only behavioral movement on DEV is at B=200 and is L3.  
**Rows:** per-expression checkpoints in `2026-09-02-guided-regression-bisect-rows.csv`; aggregate table in `.csv`/`.json`. No wall clock in any number. FINAL untouched.

## Method

One binary per era (`phase3_bisect_eval`, on `claude/guided-bisect` for the main era and `claude/guided-bisect-retired{,-l3,-l4}` / `claude/guided-bisect-r1b-exact` for the retired base) runs the three Phase 3 arms — `unguided`, `control` (per-rule TRAIN strict-positive rate mapped by `rule_idx`, exactly Round 1b's `PerRuleRateGuide::from_train_guide_report`), `linear` (additive strict-label Guide) — on the guided grid [25, 50, 100, 200, 400, 800] and records **both** cost shapes at every checkpoint: `tree` = `ExtractedDAG::total_cost` (the DP objective every Round-1 number was read from) and `dag` = the materialized choice map priced once per distinct e-class (`ChoiceCost::dag` on main; on the retired base the same post-order walk over `ExtractedDAG::choices`, restated in the binary since that era has no `cost_of_choices`). On the main era there is a fourth arm, `control_label`, which is the control arm as `per_rule_rate_guide_from_report` builds it there (see §Bug).

Everything else is held fixed across every row:

- `corpus_dev.bin` MD5 `3026133ebba066eeca10f658da554400` (Round 1b's and the bilinear eval's — identical), `corpus_train.bin` MD5 `0ed6cf16abcbc006cd7a3ee2365b15b4` (fence source, 3,359 structures, 0 collisions on every load).
- The OOD corpus is **not** one file across the two reports: Round 1b ran `corpus_dev_ood.bin` MD5 `0c7cbe710c50175afb3cd91f60960b64` (sh: 100 entries, 95 classical, 51–171 nodes) and the bilinear eval a regenerated `corpus_dev_ood.bin` MD5 `81934b499e97664a165ab639ba42e156` (sh: 100 classical, 51–288 nodes; a different draw — 94 of the 95 shared names carry different node counts, the numbering is shifted). Both are run in every era, so the corpus is its own row pair.
- Checkpoints: the retired base can only load `guide_checkpoint_strict_v1.json` (MD5 `dcc79b59cfe00bc62df031924382e279`, 61 bare rule names, what Round 1b ran); main refuses it (`missing field rule_fingerprint`) and can only load the same-recipe re-mint `guide_checkpoint_strict_remint.json` (MD5 `88e8909abbbcbe4a91a2c3c83d369ddb`, what the bilinear eval ran). So the **linear** arm carries a checkpoint difference across the A/D boundary and the **control** arm — a lookup table from the MD5-identical `2026-09-01-train-guide-report.json` (`fffa56b92adfe0a47709a1c219d213cd`) — is the checkpoint-free comparator. Its median is bit-identical on `sh` in all six eras.
- Class cap 5000 (classical tier), sweep ceiling 10,000, `CostModel::latency_prior()`, deterministic.

Reproduction gates, per expression and per grid point (0 mismatches means every cost and every `app_actual` equal):

| gate | result |
|---|---|
| A0 tree vs `2026-09-01-phase3-round1b-{sh,dev}.jsonl` (95 + 334 expressions × 3 arms × 6 points) | **0 mismatches** — Round 1b reproduced bit-for-bit |
| E dag (`control_label`, `linear`, `unguided`) vs `2026-09-02-bilinear-guide.jsonl` (100 + 334 expressions) | **0 mismatches** — the alarm reproduced bit-for-bit |
| A (retired head) tree vs Round 1b rows | 157 mismatches on sh — the branch head is **not** Round 1b's source (`3e89ad50`, the post-run review fix, changed the guided loop: all targets sharing a candidate key are now attempted, not the first) |
| D (pre-#1120) dag vs bilinear rows | 13 mismatches, all `linear` at 400/800 apps — #1120 is real but invisible at B ≤ 200 |

## The table

Median `arm@B / unguided@B`, then [improved / unchanged / worse]. Tree and dag side by side on the same rows.

### sh (Round 1b corpus, n=95)

| row | B | control tree | control dag | linear tree | linear dag | control_label dag | unguided app_actual@B |
|---|---:|---|---|---|---|---|---:|
| A0 | 100 | **0.9028** [95/0/0] | **1.0980** [4/0/91] | **0.9039** [95/0/0] | **1.1138** [0/0/95] | — | 124 |
| A0 | 200 | **0.8940** [95/0/0] | **1.1334** [0/0/95] | **0.8959** [95/0/0] | **1.1355** [0/0/95] | — | 216 |
| A | 100 | **0.9031** [95/0/0] | **1.0994** [5/0/90] | **0.9039** [95/0/0] | **1.1138** [0/0/95] | — | 124 |
| A | 200 | **0.8940** [95/0/0] | **1.1334** [0/0/95] | **0.8959** [95/0/0] | **1.1355** [0/0/95] | — | 216 |
| A+L3 | 100 | **0.9031** [95/0/0] | **1.0994** [5/0/90] | **0.9039** [95/0/0] | **1.1138** [0/0/95] | — | 100 |
| A+L3 | 200 | **0.8943** [95/0/0] | **1.1334** [0/1/94] | **0.8971** [95/0/0] | **1.1355** [0/1/94] | — | 200 |
| A+L4 | 100 | **0.9031** [95/0/0] | **1.0994** [5/0/90] | **0.9039** [95/0/0] | **1.1138** [0/0/95] | — | 124 |
| A+L4 | 200 | **0.8940** [95/0/0] | **1.1334** [0/0/95] | **0.8959** [95/0/0] | **1.1355** [0/0/95] | — | 216 |
| D | 100 | **0.9031** [95/0/0] | **1.0994** [5/0/90] | **0.8988** [95/0/0] | **1.1352** [3/0/92] | **1.0987** [8/0/87] | 100 |
| D | 200 | **0.8943** [95/0/0] | **1.1334** [0/1/94] | **0.8999** [94/0/1] | **1.1355** [0/0/95] | **1.0988** [8/0/87] | 200 |
| E | 100 | **0.9031** [95/0/0] | **1.0994** [5/0/90] | **0.8988** [95/0/0] | **1.1352** [3/0/92] | **1.0987** [8/0/87] | 100 |
| E | 200 | **0.8943** [95/0/0] | **1.1334** [0/1/94] | **0.9000** [94/0/1] | **1.1355** [0/0/95] | **1.0988** [8/0/87] | 200 |

### sh (bilinear-eval corpus, n=100)

| row | B | control tree | control dag | linear tree | linear dag | control_label dag | unguided app_actual@B |
|---|---:|---|---|---|---|---|---:|
| A0 | 100 | **0.9064** [100/0/0] | **1.0995** [4/0/96] | **0.9047** [100/0/0] | **1.1062** [0/0/100] | — | 124 |
| A0 | 200 | **0.8974** [100/0/0] | **1.1197** [0/0/100] | **0.9006** [100/0/0] | **1.1197** [0/0/100] | — | 214 |
| A | 100 | **0.9095** [100/0/0] | **1.0968** [7/0/93] | **0.9047** [100/0/0] | **1.1053** [3/0/97] | — | 124 |
| A | 200 | **0.8974** [100/0/0] | **1.1197** [2/0/98] | **0.8996** [100/0/0] | **1.1197** [3/0/97] | — | 214 |
| A+L3 | 100 | **0.9095** [100/0/0] | **1.0968** [7/0/93] | **0.9047** [100/0/0] | **1.1053** [3/0/97] | — | 100 |
| A+L3 | 200 | **0.9000** [100/0/0] | **1.1197** [2/1/97] | **0.9007** [100/0/0] | **1.1197** [3/1/96] | — | 200 |
| A+L4 | 100 | **0.9095** [100/0/0] | **1.0968** [7/0/93] | **0.9047** [100/0/0] | **1.1053** [3/0/97] | — | 124 |
| A+L4 | 200 | **0.8974** [100/0/0] | **1.1197** [2/0/98] | **0.8996** [100/0/0] | **1.1197** [3/0/97] | — | 214 |
| D | 100 | **0.9095** [100/0/0] | **1.0968** [7/0/93] | **0.9123** [100/0/0] | **1.1063** [17/0/83] | **1.0978** [17/0/83] | 100 |
| D | 200 | **0.9000** [100/0/0] | **1.1197** [2/1/97] | **0.9097** [99/0/1] | **1.1234** [0/0/100] | **1.0921** [17/0/83] | 200 |
| E | 100 | **0.9095** [100/0/0] | **1.0968** [7/0/93] | **0.9123** [100/0/0] | **1.1063** [17/0/83] | **1.0978** [17/0/83] | 100 |
| E | 200 | **0.9000** [100/0/0] | **1.1197** [2/1/97] | **0.9097** [99/0/1] | **1.1234** [0/0/100] | **1.0921** [17/0/83] | 200 |

### DEV classical (n=334)

| row | B | control tree | control dag | linear tree | linear dag | control_label dag | unguided app_actual@B |
|---|---:|---|---|---|---|---|---:|
| A0 | 100 | **0.5655** [321/4/9] | **0.6584** [329/4/1] | **0.5366** [323/4/7] | **0.6528** [329/4/1] | — | 111 |
| A0 | 200 | **0.6991** [245/71/18] | **0.7374** [244/68/22] | **0.6959** [245/71/18] | **0.7307** [243/68/23] | — | 208 |
| A | 100 | **0.5747** [316/6/12] | **0.6614** [330/4/0] | **0.5312** [329/4/1] | **0.6496** [330/4/0] | — | 111 |
| A | 200 | **0.7281** [233/86/15] | **0.7445** [246/78/10] | **0.6918** [243/81/10] | **0.7267** [246/76/12] | — | 208 |
| A+L3 | 100 | **0.5768** [312/9/13] | **0.6627** [326/7/1] | **0.5391** [325/7/2] | **0.6515** [326/7/1] | — | 100 |
| A+L3 | 200 | **0.8846** [217/102/15] | **0.8165** [239/83/12] | **0.7952** [224/96/14] | **0.8003** [241/80/13] | — | 200 |
| A+L4 | 100 | **0.5747** [316/6/12] | **0.6614** [330/4/0] | **0.5312** [329/4/1] | **0.6496** [330/4/0] | — | 111 |
| A+L4 | 200 | **0.7281** [233/86/15] | **0.7445** [246/78/10] | **0.6958** [242/81/11] | **0.7267** [246/76/12] | — | 208 |
| D | 100 | **0.6223** [325/7/2] | **0.6637** [326/7/1] | **0.6084** [326/7/1] | **0.6530** [326/7/1] | **0.6680** [326/6/2] | 100 |
| D | 200 | **0.7897** [231/95/8] | **0.8165** [237/85/12] | **0.7855** [233/91/10] | **0.8003** [241/80/13] | **0.8061** [231/92/11] | 200 |
| E | 100 | **0.6223** [326/6/2] | **0.6637** [327/6/1] | **0.6084** [327/6/1] | **0.6530** [327/6/1] | **0.6680** [327/5/2] | 100 |
| E | 200 | **0.7897** [231/95/8] | **0.8165** [237/85/12] | **0.7855** [233/90/11] | **0.8003** [241/80/13] | **0.8061** [231/92/11] | 200 |

### bezier (Round 1b corpus, n=80)

| row | B | control tree | control dag | linear tree | linear dag | control_label dag | unguided app_actual@B |
|---|---:|---|---|---|---|---|---:|
| A0 | 100 | **0.9098** [80/0/0] | **0.8000** [80/0/0] | **0.9098** [80/0/0] | **0.8000** [80/0/0] | — | 110 |
| A0 | 200 | **0.9098** [60/20/0] | **0.8140** [60/20/0] | **0.8855** [60/20/0] | **0.7846** [60/20/0] | — | 210 |
| A | 100 | **0.9098** [80/0/0] | **0.8000** [80/0/0] | **0.9098** [80/0/0] | **0.8000** [80/0/0] | — | 110 |
| A | 200 | **0.9098** [60/20/0] | **0.8140** [45/35/0] | **0.8855** [60/20/0] | **0.7846** [60/20/0] | — | 210 |
| A+L3 | 100 | **0.9098** [80/0/0] | **0.8000** [80/0/0] | **0.9098** [80/0/0] | **0.8000** [80/0/0] | — | 100 |
| A+L3 | 200 | **0.9648** [60/20/0] | **0.8669** [45/35/0] | **0.9048** [60/20/0] | **0.7857** [60/20/0] | — | 200 |
| A+L4 | 100 | **0.9098** [80/0/0] | **0.8000** [80/0/0] | **0.9098** [80/0/0] | **0.8000** [80/0/0] | — | 110 |
| A+L4 | 200 | **0.9098** [60/20/0] | **0.8140** [45/35/0] | **0.8855** [60/20/0] | **0.7846** [60/20/0] | — | 210 |
| D | 100 | **0.9098** [80/0/0] | **0.8000** [80/0/0] | **0.8568** [80/0/0] | **0.7978** [80/0/0] | **0.7978** [80/0/0] | 100 |
| D | 200 | **0.9648** [60/20/0] | **0.8669** [45/35/0] | **0.9048** [60/20/0] | **0.8701** [60/20/0] | **0.7857** [60/20/0] | 200 |
| E | 100 | **0.9098** [80/0/0] | **0.8000** [80/0/0] | **0.8568** [80/0/0] | **0.7978** [80/0/0] | **0.7978** [80/0/0] | 100 |
| E | 200 | **0.9648** [60/20/0] | **0.8669** [45/35/0] | **0.9048** [60/20/0] | **0.8701** [60/20/0] | **0.7857** [60/20/0] | 200 |

### bezier (bilinear-eval corpus, n=80)

| row | B | control tree | control dag | linear tree | linear dag | control_label dag | unguided app_actual@B |
|---|---:|---|---|---|---|---|---:|
| A0 | 100 | **0.9098** [80/0/0] | **0.8000** [80/0/0] | **0.9098** [80/0/0] | **0.8000** [80/0/0] | — | 110 |
| A0 | 200 | **0.9098** [60/20/0] | **0.8140** [60/20/0] | **0.8855** [60/20/0] | **0.7846** [60/20/0] | — | 210 |
| A | 100 | **0.9098** [80/0/0] | **0.8000** [80/0/0] | **0.9098** [80/0/0] | **0.8000** [80/0/0] | — | 110 |
| A | 200 | **0.9098** [60/20/0] | **0.8140** [45/35/0] | **0.8855** [60/20/0] | **0.7846** [60/20/0] | — | 210 |
| A+L3 | 100 | **0.9098** [80/0/0] | **0.8000** [80/0/0] | **0.9098** [80/0/0] | **0.8000** [80/0/0] | — | 100 |
| A+L3 | 200 | **0.9648** [60/20/0] | **0.8669** [45/35/0] | **0.9048** [60/20/0] | **0.7857** [60/20/0] | — | 200 |
| A+L4 | 100 | **0.9098** [80/0/0] | **0.8000** [80/0/0] | **0.9098** [80/0/0] | **0.8000** [80/0/0] | — | 110 |
| A+L4 | 200 | **0.9098** [60/20/0] | **0.8140** [45/35/0] | **0.8855** [60/20/0] | **0.7846** [60/20/0] | — | 210 |
| D | 100 | **0.9098** [80/0/0] | **0.8000** [80/0/0] | **0.8568** [80/0/0] | **0.7978** [80/0/0] | **0.7978** [80/0/0] | 100 |
| D | 200 | **0.9648** [60/20/0] | **0.8669** [45/35/0] | **0.9048** [60/20/0] | **0.8701** [60/20/0] | **0.7857** [60/20/0] | 200 |
| E | 100 | **0.9098** [80/0/0] | **0.8000** [80/0/0] | **0.8568** [80/0/0] | **0.7978** [80/0/0] | **0.7978** [80/0/0] | 100 |
| E | 200 | **0.9648** [60/20/0] | **0.8669** [45/35/0] | **0.9048** [60/20/0] | **0.8701** [60/20/0] | **0.7857** [60/20/0] | 200 |

Rows: **A0** = Round 1b's exact source `14125f10` (retired branch before its post-run review fix; base `453d2a6e`) (linear checkpoint: strict_v1); **A** = retired branch head `b26506bd` (`claude/phase3-domain-shift` as-is: + `3e89ad50` sibling-target fix; base `3ced09be`) (linear checkpoint: strict_v1); **A+L3** = A + the unguided application budget bound mid-scan (#1118's semantics, 13-line patch on graph.rs) (linear checkpoint: strict_v1); **A+L4** = A + `rebuild_budgeted` write-back to `find(id)` (#1120's 3-line fix) (linear checkpoint: strict_v1); **D** = main `57decc0b` = #1117 + #1118 + forward-port #1121/#1122, before #1120 (linear checkpoint: remint); **E** = current main `c06aa3f6` (+ #1120, #1123) (linear checkpoint: remint).

### Applications at stop, stop reasons, classes at stop (guided arms run the grid to 800; the unguided arm here also runs to 800)

| row | set | arm | B=100 row: apps at stop; stop reasons; median classes at last checkpoint |
|---|---|---|---|
| A0 | sh_r1b | control | 800 apps; app_budget 94, quiesced 1; 454 classes |
| A0 | sh_r1b | linear | 800 apps; app_budget 94, quiesced 1; 462 classes |
| A0 | sh_r1b | unguided | — |
| A0 | sh_bil | control | 800 apps; app_budget 99, quiesced 1; 500 classes |
| A0 | sh_bil | linear | 800 apps; app_budget 99, quiesced 1; 516 classes |
| A0 | sh_bil | unguided | — |
| A0 | dev | control | 294 apps; app_budget 75, quiesced 259; 232 classes |
| A0 | dev | linear | 293 apps; app_budget 78, quiesced 256; 226 classes |
| A0 | dev | unguided | — |
| A | sh_r1b | control | 800 apps; app_budget 95; 450 classes |
| A | sh_r1b | linear | 800 apps; app_budget 95; 491 classes |
| A | sh_r1b | unguided | — |
| A | sh_bil | control | 800 apps; app_budget 100; 504 classes |
| A | sh_bil | linear | 800 apps; app_budget 100; 526 classes |
| A | sh_bil | unguided | — |
| A | dev | control | 800 apps; app_budget 169, quiesced 165; 292 classes |
| A | dev | linear | 800 apps; app_budget 168, quiesced 166; 297 classes |
| A | dev | unguided | — |
| A+L3 | sh_r1b | control | 800 apps; app_budget 95; 450 classes |
| A+L3 | sh_r1b | linear | 800 apps; app_budget 95; 491 classes |
| A+L3 | sh_r1b | unguided | — |
| A+L3 | sh_bil | control | 800 apps; app_budget 100; 504 classes |
| A+L3 | sh_bil | linear | 800 apps; app_budget 100; 526 classes |
| A+L3 | sh_bil | unguided | — |
| A+L3 | dev | control | 800 apps; app_budget 169, quiesced 165; 292 classes |
| A+L3 | dev | linear | 800 apps; app_budget 168, quiesced 166; 297 classes |
| A+L3 | dev | unguided | — |
| A+L4 | sh_r1b | control | 800 apps; app_budget 95; 450 classes |
| A+L4 | sh_r1b | linear | 800 apps; app_budget 95; 491 classes |
| A+L4 | sh_r1b | unguided | — |
| A+L4 | sh_bil | control | 800 apps; app_budget 100; 504 classes |
| A+L4 | sh_bil | linear | 800 apps; app_budget 100; 524 classes |
| A+L4 | sh_bil | unguided | — |
| A+L4 | dev | control | 800 apps; app_budget 169, quiesced 165; 292 classes |
| A+L4 | dev | linear | 800 apps; app_budget 168, quiesced 166; 297 classes |
| A+L4 | dev | unguided | — |
| D | sh_r1b | control | 800 apps; app_budget 95; 447 classes |
| D | sh_r1b | linear | 800 apps; app_budget 94, quiesced 1; 537 classes |
| D | sh_r1b | unguided | — |
| D | sh_bil | control | 800 apps; app_budget 100; 508 classes |
| D | sh_bil | linear | 800 apps; app_budget 99, quiesced 1; 566 classes |
| D | sh_bil | unguided | — |
| D | dev | control | 800 apps; app_budget 169, quiesced 165; 296 classes |
| D | dev | linear | 800 apps; app_budget 168, quiesced 166; 295 classes |
| D | dev | unguided | — |
| E | sh_r1b | control | 800 apps; app_budget 95; 447 classes |
| E | sh_r1b | linear | 800 apps; app_budget 94, quiesced 1; 533 classes |
| E | sh_r1b | unguided | — |
| E | sh_bil | control | 800 apps; app_budget 100; 508 classes |
| E | sh_bil | linear | 800 apps; app_budget 99, quiesced 1; 565 classes |
| E | sh_bil | unguided | — |
| E | dev | control | 800 apps; app_budget 169, quiesced 165; 296 classes |
| E | dev | linear | 800 apps; app_budget 168, quiesced 166; 295 classes |
| E | dev | unguided | — |

`unguided` on the retired base overshoots its target (Round 1b's between-rules check): median `app_actual` 124 at B=100 / 216 at B=200 on `sh`, 111 / 208 on DEV; from A+L3 onward every arm lands on its target exactly. On `sh` the unguided dag cost does not move between 25 and 800 applications in any era (`dag@4B / dag@B` = 0.999 / 1.000), so the overshoot has nothing to buy there; on DEV it does (0.72–0.79 at B=100), which is where L3 shows.

### The 4B control: guided at 4B against unguided at B (dag)

| row | set | control @4B/ung@B, B=100 | B=200 | linear @4B/ung@B, B=100 | B=200 |
|---|---|---:|---:|---:|---:|
| A0 | sh_r1b | 1.1320 | 1.1318 | 1.1319 | 1.1320 |
| A0 | sh_bil | 1.1127 | 1.1244 | 1.1110 | 1.1244 |
| A0 | dev | 0.6480 | 0.7272 | 0.6470 | 0.7271 |
| A | sh_r1b | 1.1336 | 1.1258 | 1.1336 | 1.1318 |
| A | sh_bil | 1.1195 | 1.1199 | 1.1126 | 1.1264 |
| A | dev | 0.6483 | 0.7304 | 0.6468 | 0.7281 |
| A+L3 | sh_r1b | 1.1336 | 1.1271 | 1.1336 | 1.1323 |
| A+L3 | sh_bil | 1.1195 | 1.1199 | 1.1126 | 1.1264 |
| A+L3 | dev | 0.6490 | 0.8035 | 0.6481 | 0.7933 |
| A+L4 | sh_r1b | 1.1336 | 1.1258 | 1.1336 | 1.1318 |
| A+L4 | sh_bil | 1.1195 | 1.1199 | 1.1126 | 1.1264 |
| A+L4 | dev | 0.6483 | 0.7304 | 0.6468 | 0.7281 |
| D | sh_r1b | 1.1336 | 1.1271 | 1.1316 | 1.1295 |
| D | sh_bil | 1.1195 | 1.1199 | 1.1256 | 1.1237 |
| D | dev | 0.6490 | 0.8035 | 0.6481 | 0.7976 |
| E | sh_r1b | 1.1336 | 1.1271 | 1.1336 | 1.1295 |
| E | sh_bil | 1.1195 | 1.1199 | 1.1275 | 1.1237 |
| E | dev | 0.6490 | 0.8035 | 0.6481 | 0.7976 |

On `sh`, a guided arm given **four times** the budget is still 1.11–1.13 of unguided-at-B on dag cost, in every era — and 0.89 on tree cost. The guided loop is not spending its applications differently under mid-scan accounting; it is reaching terms that are cheaper as trees and dearer as DAGs, and it reaches more of them with more budget.

## Reading it

1. **The `sh` ratio crosses 1.0 at L1 and nowhere else.** On the Round 1b source (A0), Round 1b corpus, Round 1b checkpoint, the rows that reproduce Round 1b's 0.9028 / 0.9039 tree ratios price as 1.0980 / 1.1138 dag ratios — 91 / 95 of 95 expressions are tree-better **and** dag-worse at B=100 (95 / 95 at B=200). Every later row says the same thing in both units: control tree 0.903–0.910, control dag 1.097–1.100, at B=100 on either corpus. The control arm's `sh` medians are bit-identical from A through E.
2. **DEV's 0.537 → 0.653 is the same metric change.** A0 linear at B=100: tree 0.5366 (= Round 1b), dag 0.6528; E: tree 0.6084, dag 0.6530. Under the unit the bilinear eval reports, DEV linear at B=100 did not move at all across four eras (0.6528, 0.6496, 0.6515, 0.6530). Under tree it did move (0.5366 → 0.6084), and that movement is split between the sibling-target fix (A0→A: −0.005 tree for linear, +0.009 for control), L3 (A→A+L3: +0.008 / +0.002), and the forward-port's remaining loop and target-identity changes plus the checkpoint (A+L3→D: +0.069 linear, +0.046 control). None of it reaches the dag number at B=100.
3. **The one behavioral lever with a dag-visible effect is L3, on DEV at B=200.** A → A+L3 moves DEV control dag 0.7445 → 0.8165 and linear 0.7267 → 0.8003, and A+L3 = D = E to three decimals. That is the unguided arm losing its overshoot: at B=200 the unguided curve on DEV is still improving (dag@800/dag@200 = 0.83 on the retired base), so the 8-application median overshoot (heavy-tailed to a whole rule sweep) was buying it real cost; with the cap binding mid-scan the denominator is honest and the guided advantage shrinks from 0.74 to 0.82. The registration's instrument-change banner already says this in words; this is the number.
4. **L4 (#1120) is invisible at B ≤ 200.** A+L4 equals A on every set at both tiers; 13 of 100 `sh` expressions differ from E on the linear arm at 400/800 only.
5. **The corpus regeneration and the post-Round-1b review fix are not it either.** `sh` on the bilinear corpus vs the Round 1b corpus: control dag 1.0968 vs 1.0994. A0 → A (sibling targets attempted, not the first): `sh` dag 1.0980 → 1.0994.

So: same row (L1) for `sh` and for DEV at B=100; DEV at B=200 additionally moves at L3. The guided path was never dag-better on `sh` — the Guides (and the strict labels they were trained on) were minted and evaluated against the tree objective, and on trig-dominant kernels with 4–6× tree/dag sharing they steer the tree-DP toward **unshared** forms: median tree/dag ratio of the extracted term is 4.25 unguided vs 3.29 guided on `sh` (Round 1b corpus), 6.05 vs 4.2–4.5 on the bilinear corpus. The bilinear eval's precondition (a) — "guidance of any kind is a net cost on `sh`" — is correct under the unit that matters and was equally true on 2026-09-01; Round 1b's 0.90 was the same behavior measured in the unit the DP optimizes.

## Verdict: (1) INSTRUMENT

**Evidence row: A0, sh (Round 1b corpus, n=95), B=100** — Round 1b's exact source `14125f10`, corpus MD5 `0c7cbe71…`, checkpoint `strict_v1`, reproducing Round 1b's tree ratios to four decimals (control 0.9028, linear 0.9039; 0 mismatches over 429 expressions × 3 arms × 6 points). Those same rows, priced once per distinct e-class, are control dag **1.0980** [4/0/91] and linear dag **1.1138** [0/0/95]. The crossing from 0.90 to 1.10 happens inside a single run, between two ways of pricing the same choice map — it is L1 (#1117, `total_cost` → `dag_cost`) and nothing else. The control arm's `sh` median is bit-identical from A through E; no behavioral lever (A0→A sibling-target fix, L3, L4, the L2 forward-port bracketed A+L3→D) moves `sh` dag by more than 0.002 at B=100, and none crosses 1.0 in either direction.

So the sh ~0.90 was a tree-cost artifact, and guidance never helped trig kernels on the real metric. Round 1b's H_null and the bilinear eval's H_form-null are both measuring arms that lose to nothing on `sh`: the honest statement is that **no Guide — control, additive, or bilinear — has yet beaten unguided on trig kernels at B ≤ 200, and a guided arm at 4B is still 1.11–1.13 of unguided at B, in every era.** The bilinear eval's precondition (a) attributed the crossing to #1117 + #1118 + #1120 jointly; it is #1117 alone.

**Round 1's DEV linear 0.537 is not recoverable on current main, and there is nothing to recover.** It is the tree-unit reading of a run whose dag-unit reading is 0.6528 (A0, same rows); main's 0.6530 is that same number. The tree number did drift (0.5366 → 0.6084, split across the sibling-target fix, L3, and the forward-port + re-mint), but tree cost is not the objective anyone should be reading, and under dag cost DEV linear at B=100 is flat across every era (0.6528 / 0.6496 / 0.6515 / 0.6530). The one dag-visible behavioral change is L3 on DEV at B=200 (0.74 → 0.82) and it is the unguided denominator becoming honest, not the guided arm getting worse.

**Sequencing consequence:** before any guided number is quoted again, the label the Guide was trained on and the ratio it is scored by must be minted in the same unit — `dag_cost` — on the same rows, with the control arm mapped by `rule_idx` (#1127), so that a guided/unguided ratio can only move because the guided path did something different.

## Bug found (filed, not fixed here)

On main, `pixelflow_pipeline::training::guide_linear::per_rule_rate_guide_from_report` builds the control arm by **label** (`RuleId::from_label(row.rule_name)`), but `2026-09-01-train-guide-report.json` carries Round 1b's bare family names (`associative`, `commutative`, `even-negation`, …) while main's rule identity is the specialized label (`associative(Add)`). 27 of the 52 rows resolve to no live rule and silently fall to `PerRuleRateGuide`'s `unwrap_or(0.0)`; duplicated bare names collapse last-wins. The rates that vanish are small (largest: `even-negation` 0.055/0.056, `canonicalize(Sub)` 0.016, `inverse-annihilation(Sub)` 0.015, `odd-negation(Sin)` 0.014), so the effect is bounded — `control` (by `rule_idx`) vs `control_label` (as shipped): `sh` dag 1.0994 vs 1.0987, DEV dag 0.6637 vs 0.6680 at B=100 — but the bilinear eval's control arm is not the registered control arm, and the loader has no check that a report row named a rule that exists. Round 1b's loader mapped by `rule_idx` and could not miss.

## What this does and does not license

- The production extraction digest is unaffected by any of this (the bilinear eval verified it byte-identical over 206 kernels); nothing here changes production.
- It does **not** say the Guides are useless: on DEV they are 0.65 of unguided at B=100 in dag units in every era. It says their `sh` advantage was an artifact of pricing trees, and that a label minted on tree cost teaches the wrong preference on kernels with heavy sharing. The R2G return label's unit is the same open question the instrument banner names.
- Row C (main at the merge-base with the old loop ported) was not built: the retired branch's own merge-base is `3ced09be` (not `2e82cdc2`, which is `phase3-guide`'s), the old loop's dependencies are the whole research surface, and the ladder above already brackets L2 from both sides (A+L3 → D) with no dag movement at B=100 on any set.

## Reproduce

```
# retired base (claude/guided-bisect-retired; -l3 / -l4 branches carry the one-lever patches; claude/guided-bisect-r1b-exact is 14125f10 + the binary)
cargo build --release -p pixelflow-pipeline --features training --bin phase3_bisect_eval
phase3_bisect_eval --corpus <corpus_dev_ood.bin> --corpus-dir <dir with corpus_train.bin> --name-prefix dev_sh_ \
   --checkpoint guide_checkpoint_strict_v1.json --train-guide-report docs/results/2026-09-01-train-guide-report.json --out-jsonl rows.jsonl --era A
# main era (claude/guided-bisect): same flags, --checkpoint guide_checkpoint_strict_remint.json; emits the extra control_label arm
```

Host load during the runs: 3.8–11.5 on 12 cores (shared machine; no metric depends on it).
