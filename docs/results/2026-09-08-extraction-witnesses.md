# Extraction witnesses: the extractor walks past terms it provably holds

**Date:** 2026-09-08
**Denotation:** `docs/plans/2026-09-08-extraction-witnesses.md` — written
before the instrument; nothing in it was revised after the run.
**Instrument:** `pixelflow-pipeline`'s `extraction_witnesses` bin over
`pixelflow-search`'s `egraph::witness` (feature-gated on
`provenance-journal`).
**Data:** `2026-09-08-extraction-witnesses.{csv,json}` — witness rows and
per-class explanations — and `2026-09-08-extraction-witnesses-budgets.csv` —
the budget ladder, including the tie-break A/B.

> **What is deterministic.** `dag_cost`, `objective`, `live_classes`,
> `tied_classes`, both choice maps and every classification are functions of
> the term and the graph, so every column below is exact and reproducible.
> Only `seconds` is wall clock, and it was taken on a shared box whose load
> average ran between 24 and 168 during the run — it is reported for scale,
> never as a claim.

## 0. The answer, first

**The extractor gets worse with more budget because most of its decisions
are not decisions.** As saturation grows a class, the extra e-nodes it gains
are overwhelmingly *cost-equal* rewrites of what was already there —
commutations, reassociations, the same arithmetic in another order — so the
DP's strict `<` never fires and the pick falls out to the first admissible
node, which is insertion order. The fraction of live classes settled that
way rises with the budget on every kernel that regresses: `mandelbrot_
distance` 47 % → 81 %, `julia_set` 63 % → 83 %, `metaballs` 50 % → 68 %.
Those same rewrites also close cycles (commutativity alone is enough), and a
class whose child is still on the single DFS stack is priced at
`CYCLE_COST` by *both* passes — so it is not chosen at all, it is filled in
afterwards by `repair_choices_well_founded`, which optimizes nothing. At
every frontier class where a witness diverges from greedy on
`mandelbrot_distance`, that is what happened: **both** candidates priced
`CYCLE`, the pick made by repair. Two effects, one cause — a bigger graph
puts more classes into the region where the extractor has no opinion, and
the term it returns is then decided by hash-cons insertion order and a
repair pass rather than by cost.

## 1. What a witness is

The e-graph was *expected* to be monotone: a run at a larger class cap
performs every application the smaller run performed and then more, so the
larger graph represents a superset of the terms (L2,
`docs/plans/2026-09-02-optimizer-api.md`). §1b is what happened to that
assumption. Where it does hold — checked, not assumed — the extractor's own
cheaper output at the smaller cap is a term present in the bigger graph that
the extractor walked past. The
instrument looks each of its subterms up in the bigger graph by hash-cons —
read-only, and a miss is a loud failure rather than a skip, since
monotonicity is exactly what is being tested — which yields a second choice
map `C_T` over the same classes. Greedy's is `C_G`. The **divergence set**
is where they differ; the **frontier** is the divergent classes below which
they agree, and at a frontier class the extractor's own comparison can be
read off directly.

**Two witness kinds, never conflated.** An *objective* witness is cheaper in
`ChoiceCost::dag` — the shape-weighted number the extractor actually
minimizes — and indicts the search. A *static-only* witness is cheaper only
in the sweep's unweighted `dag_cost` column while being dearer under the
weighting; that indicts the objective's weighting, not the search, and
realizability is not a question one can ask of it. Both are tabulated,
apart.

## 1b. A finding that arrived before any classification: **the e-graph is not monotone in the class cap**

The whole method rests on L2 (`docs/plans/2026-09-02-optimizer-api.md`): a
bigger budget yields a graph that represents a superset of the terms. The
instrument does not assume it — it looks each subterm up and fails loudly on
a miss — and it **found misses**.

| corpus | candidate pairs where the smaller budget's term was cheaper | of those, subterm absent from the bigger graph |
|---|---:|---:|
| shaders + glyphs (`--out /tmp/ew-dev`) | see §7 | 19 |
| glyphs (independent run) | see §7 | 20 |
| chrome (held out) | 10 | 3 |

Every chrome miss is the same node — `MulAdd(c1429, c0, c5181)`, minted in
`G(10k)` and absent from `G(20k)`, `G(50k)` and `G(100k)` — and the glyph
misses are the same shape: an `fma-fusion` output, or a `Select`, present at
the smaller cap and never minted at the larger one.

That is not a bug in the lookup; it is the claim being false as the loop is
actually run. **A different class cap is a different trajectory, not a
prefix-extension of the same one.** The cap changes which classes get
allocated and in what order, which changes which candidates the scan reaches
before a round ends, which changes which rules fire — so `G(b_hi)` is a
*different* graph that happens to be bigger, not a superset of `G(b_lo)`.

Two consequences, both load-bearing:

- **For the witness argument.** Roughly three quarters of the cheaper-at-a-
  smaller-budget pairs do map, and for those the monotonicity conclusion
  holds constructively — the term *is* in the bigger graph, by exhibition
  rather than by appeal to L2. Those are the witnesses this document
  reports. The rest are not witnesses at all, and are excluded rather than
  assumed.
- **For the optimizer's own contract.** L2 is stated as a property and is
  used as one. It should be either restated (monotone in *applications with
  a fixed cap*, which is a much weaker and probably true claim) or made
  true. Right now the class cap is a budget dimension that silently changes
  the answer's *shape*, not only its quality — which is the same class of
  problem `docs/plans/2026-09-01-production-budget-determinism.md` closed for
  wall clock.

## 2. The shader family: the ladder

Every column is exact. `tied` is the count of live classes where two or more
candidates shared the winning DP cost, so insertion order decided.

| kernel | live 5k → 100k | dag 5k | 10k | 20k | 50k | 100k | tied 5k | tied 100k |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `metaballs` | 50 → 73 | **155** | 140 | 136 | 255 | **270** | 25/50 (50 %) | 50/73 (68 %) |
| `julia_set` | 123 → 174 | **717** | 791 | 697 | 904 | **948** | 77/123 (63 %) | 144/174 (83 %) |
| `mandelbrot_distance` | 96 → 115 | **518** | 558 | 595 | 595 | **595** | 45/96 (47 %) | 93/115 (81 %) |
| `smooth_min_scene` | 39 → 39 | **132** | 132 | 132 | 144 | **144** | 19/39 | 20/39 |
| `psychedelic_packed` | 112 → 111 | **825** | 826 | 829 | 829 | **829** | 33/112 | 34/111 |
| `domain_warp_fbm` | 63 → 58 | **457** | 431 | 446 | 439 | **436** | 22/63 | 19/58 |
| `cosine_palette` | 25 → 25 | 292 | 292 | 292 | 292 | 292 | 7/25 | 9/25 |
| `star_sdf` | 58 → 58 | 172 | 172 | 172 | 172 | 172 | 21/58 | 21/58 |
| `gyroid_slice` | 33 → 33 | 932 | 932 | 932 | 932 | 932 | 12/33 | 12/33 |
| `plasma` | 29 → 29 | 359 | 359 | 359 | 359 | 359 | 9/29 | 9/29 |
| `kaleidoscope_fold` | 40 → 40 | 560 | 560 | 560 | 560 | 560 | 12/40 | 12/40 |
| `smoothstep_vignette` | 49 → 49 | 194 | 194 | 194 | 194 | 194 | 17/49 | 17/49 |
| `torus_slice` | 31 → 31 | 130 | 130 | 130 | 130 | 130 | 11/31 | 11/31 |

Five of thirteen regress from the smallest cap to the largest —
`metaballs` by **+74 %**, `julia_set` by **+32 %**, `mandelbrot_distance` by
**+15 %** — one improves, and seven are flat because saturation quiesced
inside the smallest cap and the ladder never changed the graph. Note
`julia_set` is not even monotone in the budget: 717 → 791 → **697** → 904 →
948. A search whose result moves like that with the size of the space it is
searching is not converging on anything.

## 3. Why: the class-level explanation for `mandelbrot_distance`

The kernel: `dag_cost` 518 at the 5k cap, 595 at 20k and above (the graph
quiesces at 6,287 allocated classes / 115 live). The witness is
greedy's own 5k term, mapped into `G(20k)`.

| | |
|---|---|
| witness `objective` / `dag_cost` | 32,707,328 / 518 |
| greedy at 20k | 37,492,480 / 595 |
| divergence set `\|D\|` | 27 classes |
| frontier `\|F\|` | **3** classes |
| every frontier class's label | **CYCLE-PRICED** (3/3) |
| realizable by local search | **no** — best single swap 34.8 M, greedy swaps 34.8 M after 2 accepted moves, against the witness's 32.7 M |

The first divergence, class 84:

| | greedy | witness |
|---|---|---|
| node | `Mul(19, 7)` | `Add(19, 19)` |
| weighted own cost | 327,680 | 262,144 |
| **tree-pass DP cost** | **CYCLE** | **CYCLE** |
| shared-pass DP cost | 589,824 | 524,288 |
| minted by | `seed` | — |
| stage that settled it | **repair** | — |
| one-swap delta from greedy | −256 | |

Read that row carefully, because it contains the whole finding.

1. The **tree pass priced both candidates at the cycle sentinel**. A class
   whose child is still on the DFS stack is not scored; `x + x` and `x · 2`
   are in one class here, and commutativity has closed a cycle through it.
   So the tree pass expressed *no preference at all* — the pick came out of
   `repair_choices_well_founded`, whose job is well-foundedness, not cost.
2. The **shared pass did price them, and preferred the witness's node**
   (524,288 < 589,824). The information was in the extractor, in the same
   run.
3. **`min-of-two` threw that arm away.** At 5k the winning objective is
   `Shared`; at 10k, 20k, 50k and 100k it is `TreeCheaper`. As the graph
   grows the arm that cycle-prices classes wins the comparison more often,
   and the arm that priced this class correctly is discarded whole. The
   choice is per-extraction, not per-class: there is no way for the shared
   arm's opinion about class 84 to survive when the tree arm's total is
   lower.
4. The witness is **not reachable by local search**. One swap buys 262 k of
   4.8 M; greedily accepting improving frontier swaps buys the same and
   stops. A reranker over single swaps could not have found this term.

## 4. The mechanical fix, A/B'd: canonical tie-breaking

Ties break to the first admissible node, i.e. insertion order. The variant
replaces that with a total order on the node's own content (leaf/op tag,
then payload or `OpKind` ordinal, then arity, then canonical child ids), so
the pick is independent of when a node was inserted. It is the same DP
otherwise, and `Ties::Insertion` is pinned byte-identical to production by
`insertion_tie_break_is_productions_extraction`.

On the shader family it is a **net win, and not a uniform one**:

| kernel | cap | production | canonical | Δ |
|---|---:|---:|---:|---:|
| `smooth_min_scene` | 50k, 100k | 144 | **124** | **−13.9 %** |
| `smoothstep_vignette` | all | 194 | **178** | **−8.2 %** |
| `mandelbrot_distance` | 20k–100k | 595 | **576** | **−3.2 %** |
| `julia_set` | 50k | 904 | **877** | −3.0 % |
| `julia_set` | 10k | 791 | **760** | −3.9 % |
| `domain_warp_fbm` | 5k | 457 | **448** | −2.0 % |
| `julia_set` | 5k | 717 | 723 | **+0.8 %** |
| `metaballs` | 100k | 270 | 271 | **+0.4 %** |

It never recovers a whole witness — `mandelbrot_distance` goes 595 → 576,
not 595 → 518 — because the classes it decides are ones the DP *scored*,
and the frontier classes that hold the witness are ones the DP scored at
`CYCLE`, where a tie-break has nothing to compare. It is worth having: it
removes the dependence on insertion order that the `'8'` bisect found, it
is free, and it costs nothing at run time. It is **not** the fix for the
regression.

## 5. Glyphs: the budget mostly helps, and where it does not it is the same story

Glyphs are the counter-example that keeps the headline honest. Over the 95
warm-range glyphs at each of two tile sizes, raising the cap from 5k to 20k
takes Σ `dag_cost` from **392,762 to 332,115 (−15.4 %)**; 74 of 95 improve,
17 are flat (quiesced inside the smallest cap), and **4 regress**:

| glyph | dag 5k | 10k | 20k | live 5k → 20k | tied 5k → 20k |
|---|---:|---:|---:|---|---|
| `U+006C` (`l`) | 1,773 | 1,827 | **2,172** (+22.5 %) | 432 → 526 | 162 → 185 |
| `U+0066` (`f`) | 1,821 | 1,875 | **2,158** (+18.5 %) | 447 → 526 | 162 → 184 |
| `U+0074` (`t`) | 1,798 | 1,907 | **1,935** (+7.6 %) | 441 → 484 | 165 → 178 |
| `U+006A` (`j`) | 1,828 | 1,958 | **1,956** (+7.0 %) | 450 → 488 | 170 → 182 |

and the biggest wins are large: `U+0024` 12,119 → 9,848, `U+0053` 11,756 →
9,669, `U+0067` 11,540 → 9,485.

**Tie density falls with the budget on glyphs** — 44.9 % → 38.9 % → 35.0 %
of live classes — the opposite of the shaders. So "more ties" is not a law
of bigger graphs; it is what happens when the rules a kernel's shape admits
are predominantly cost-neutral ones. Glyph kernels are winding-number
arithmetic over many Bézier segments, where saturation finds real
factorizations; the regressing shaders are transcendental-heavy, where the
reachable rewrites are mostly reassociations of the same instruction count.

**Which arm wins is a coin flip on glyphs** — `Shared` on 44–48 of 95
kernels per cap, `TreeCheaper` on the rest — so the `min-of-two` decision is
load-bearing on roughly half of every glyph bake.

## 6. Method notes and what this does not show

- Every budget arm is production's optimizer at its own tier's round cap
  with the class cap and `40 ×` application cap moved — the sweep's
  `cap{b}-app{40b}` arms — with the safety ceiling disabled, since the
  ceiling asserts something about *production's* budget and firing it here
  would say nothing about the extractor.
- Both budget-ladder runs used the shipped extractor; the traced re-run
  asserts its cost equals `Optimizer::run`'s on every pair analysed, so a
  drift between the instrument and the shipped path would abort rather than
  be quietly analysed.
- **No monotonicity violation was observed**: every witness subterm was
  found in the bigger graph by hash-cons lookup. The instrument would have
  said so loudly otherwise.
- This measures `dag_cost` and the extraction objective, not wall clock of
  the emitted kernel. The corrected-benchmark work
  (`docs/plans/2026-09-07-benchmark-correction.md`) is where a clock claim
  would come from, and none is made here.

## 7. The numbers

*(filled below from `2026-09-08-extraction-witnesses.json`.)*
