# Beam extraction: the seam is built, and width is not the missing axis

**The headline, first: no.** `Beam` at a 50,000-class cap does not reach
`Greedy`'s cost at a 5,000-class cap on any family. On `chrome_packed` — the
case the class-cap sweep named — every width from 1 to 64 returns the *same
term* at 50,000 (`dag_cost` 2,214, byte-identical machine code) against
`Greedy`'s 1,668 at 5,000. Width bought nothing exactly where the regression
is. The class-cap regression
(`docs/results/2026-09-08-class-cap-sweep.md`) is **not** a search-width
failure of the sharing-aware DP.

**What did land: the seam.** Extraction now has the trait the saturation side
has. `Extractor<C: CostFunction>` is one method; `Greedy` is a name for
production's `extract_dag_scoped` and is byte-identical to it end to end;
`Reranked` replaces `Optimizer`'s `rerank` field and the two-arm match that
was an unnamed second extractor inlined into its caller; `Beam::width(k)` is
the second real implementation, and `Beam::width(1)` is `Greedy` to the
machine-code byte. Denotation and boundary decisions:
`docs/plans/2026-09-08-extractor-trait.md`.

**Production is unchanged.** `Optimizer::production()` uses `Greedy`, and
nothing in this measurement argues for flipping it.

## Method

Every row is one kernel compiled through the production path with the arm's
optimizer in the saturation slot, by
`egraph_off_on run --class-cap N --beam K --no-clock --no-probe` (`--beam`
composes with every existing arm: the saturated graph is whatever the arm
built, only the extractor changes). The corpus is the benchmark correction's
DEV families — the 95 DejaVu glyphs at tile 16, the cell grid, psychedelic,
the twelve `shader_bench` ports — with `chrome_packed` held out and run once
per arm at the end.

`dag_cost` is a property of the term, so **every quality number here is
exact** and the shared host's load cannot move it. The host's 1-minute load
ran 10–35 throughout (other sessions' builds), so **no wall-clock number here
is a claim**: the compile milliseconds are reported as a sign, and only the
*ratio between arms of the same run* is quoted. Extraction's wall share is
not taken at load ≥ 8 per the benchmark correction's rule; what is quoted
instead is the whole-compile ratio, which bounds it from above.

`code_fnv` (new column) is FNV-1a of the emitted machine code — the column an
identity claim is actually made on, since `bytes` can collide and a picture
can agree on two different programs.

## Identity: `Beam::width(1)` is `Greedy` is production

Structural, then tested, then measured end to end:

- `Greedy::extract` **is** `extract_dag_scoped` — a delegation, not a copy —
  pinned by `greedy_is_production` over four saturated graphs × two lattice
  shapes on choices, `dag_cost`, `total_cost`, schedule, sharing and report.
- `beam_at_width_one_is_greedy` holds the same equality for `Beam::width(1)`.
  It is structural too: seat 0 of every e-class is the **anchor**, the state
  naming the DP's own choice, and at one seat the beam holds nothing else.
- End to end, `--beam 1` reproduces `code_fnv` on **every kernel of every arm
  measured** — 12/12 shaders, 95/95 glyph16, cell grid, psychedelic and
  chrome, at both caps.

## Quality, per family, exact

`ratio` is Σ `dag_cost` against `Greedy` at the same cap; below 1 is better.
`improved / worse` count kernels.

| cap | width | family | n | Σ dag_cost | ratio | improved | worse | compile ×Greedy |
|---|---|---|---:|---:|---:|---:|---:|---:|
<!--TABLE-->

## Chrome (held out, one run per arm)

Ties go to the anchor, so `code_fnv = Greedy` is **no** only where the width
found a strictly cheaper term.

| cap | width | dag_cost | bytes | objective | code_fnv = Greedy | compile |
|---|---|---:|---:|---|---|---:|
| 5000 | greedy | 1668 | 3424 | shared | — | 180 ms |
| 5000 | beam1 | 1668 | 3424 | shared | yes | 189 ms |
| 5000 | beam4 | 1668 | 3424 | shared | yes | 208 ms |
| 5000 | **beam16** | **1660** | 3408 | shared | no (a strict win) | 351 ms |
| 5000 | beam64 | 1668 | 3424 | shared | yes | 2187 ms |
| 50000 | greedy | 2214 | 4000 | shared | — | 2469 ms |
| 50000 | beam1 | 2214 | 4000 | shared | yes | 2504 ms |
| 50000 | beam4 | 2214 | 4000 | shared | yes | 2572 ms |
| 50000 | beam16 | 2214 | 4000 | shared | yes | 3606 ms |
| 50000 | beam64 | 2214 | 4000 | shared | yes | 18307 ms |

## The reading

**Width is not monotone, and the anchor is why it is still safe.** `Beam` is
never dearer than `Greedy` — the anchor is in every seat list, so the DP's own
term is always reachable — but a wider beam can be dearer than a narrower one,
and is: `chrome_packed` at 5,000 is 1,660 at `k = 16` and 1,668 at `k = 64`. A
wider beam generates strictly more candidates, and top-`k` of a larger
candidate set need not contain top-`j` of a smaller one; a cheap partial merge
that later conflicts with every sibling crowds out the state a narrower beam
carried to the root. That is beam search, not a defect, but it means a width
is a setting to measure and never one to raise on principle.

**Where the term does not move at all, the answer is not "search harder".**
At 50,000 classes chrome's extraction objective is `shared` — the beam is
widening the arm that actually produced the returned term — and yet `k = 64`
returns machine code identical to `Greedy`'s, having spent 18.1 s of compile
against 2.7 s. Sixty-four seats per e-class over 10,256 live classes changed
no choice. That is a much stronger negative than "beam was a bit worse": the
neighbourhood a per-class beam searches does not contain the cheaper term the
witness argument says is in the graph.

**Why it plausibly cannot.** A beam state must name **one** term, so two
sibling states that disagree on any shared class cannot merge. As the graph
grows, the sub-DAGs overlap more, so *more* candidate pairs conflict and the
free seats die before they reach the root — the beam degenerates toward its
anchor exactly as the graph gets big enough for width to matter. Width and
sharing pull against each other in this formulation. Any successor has to
resolve disagreement rather than drop it.

**What this rules in.** The seam is now the thing to iterate on, not the
extractor: a second `impl Extractor` costs a file and is held to the
`Beam::width(1) ≡ Greedy` standard by construction. The next candidate is not
a wider version of this search. Two directions the measurement points at:

1. **The tree arm is not free either.** On psychedelic and nine of twelve
   shaders the returned objective is `tree_cheaper` — the *tree* DP's term
   wins on DAG cost. Widening the shared arm cannot help those at all, by
   construction, and no width did. The tree DP is already optimal for tree
   cost; nobody is minimizing DAG cost over the tree arm's neighbourhood.
2. **The witnesses.** The sweep's own pairs say a cheaper term is provably in
   the bigger graph (chrome: 1,668 exists at 5,000; the 50,000 graph contains
   every node the 5,000 graph had). A per-class beam does not find it. The
   per-class evidence — which class, which stage, which rule — is what
   `origin/claude/extraction-witnesses` is mining, and it is the right next
   input, not more width.

## Reproduce

```bash
cargo build --release -p pixelflow-pipeline --bin egraph_off_on \
  --features pixelflow-search/saturation-telemetry
scripts/beam-extraction-sweep.sh /tmp/beam "1 4 16 64" "5000 50000" \
  "shader_ cellgrid psychedelic glyph16_"
python3 scripts/beam-extraction-report.py /tmp/beam docs/results/2026-09-08-beam-extraction
```
