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

**One thing the `dag_cost` column is not.** `egraph_off_on`'s `dag_cost` is
the *unweighted* latency-prior sum over the linked arena, while the production
path extracts under `for_lattice(shape)` — where a node's cost is multiplied
by how many times that lattice evaluates it. The two are the same objective
only at `LatticeShape::POINT`, which no glyph, shader or chrome row is. So
`Beam` is guaranteed no dearer than `Greedy` **in the extractor's own
objective** (asserted at both `POINT` and the bake shape by
`pixelflow-pipeline/tests/beam_never_loses_to_greedy.rs`), and the column
below can still move either way for an individual kernel — a term cheaper per
sample is not automatically cheaper per node. That caveat is not specific to
this document: `docs/results/2026-09-08-class-cap-sweep.md` reads the same
column as "the extractor's objective", and at a lattice it is a correlate of
it, not it.

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
`improved / worse` count kernels **in this column**, which is not the
extractor's objective at a lattice (see above) — the four `glyph16` kernels
that go the wrong way at `k = 64` (`f`, `j`, `l`, `~`) are cheaper per sample
and dearer per node, not a broken guarantee. `chrome` is the held-out row,
included here for one table and discussed on its own below.

| cap | width | family | n | Σ dag_cost | ratio | improved | worse | compile ×Greedy |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 5000 | greedy | cellgrid | 1 | 428 | 1.0000 | 0 | 0 | 1.0× |
| 5000 | beam1 | cellgrid | 1 | 428 | 1.0000 | 0 | 0 | 0.9× |
| 5000 | beam4 | cellgrid | 1 | 428 | 1.0000 | 0 | 0 | 1.6× |
| 5000 | beam16 | cellgrid | 1 | 428 | 1.0000 | 0 | 0 | 7.2× |
| 5000 | beam64 | cellgrid | 1 | 420 | 0.9813 | 1 | 0 | 111.3× |
| 5000 | greedy | chrome | 1 | 1668 | 1.0000 | 0 | 0 | 1.0× |
| 5000 | beam1 | chrome | 1 | 1668 | 1.0000 | 0 | 0 | 1.0× |
| 5000 | beam4 | chrome | 1 | 1668 | 1.0000 | 0 | 0 | 1.2× |
| 5000 | beam16 | chrome | 1 | 1660 | 0.9952 | 1 | 0 | 2.0× |
| 5000 | beam64 | chrome | 1 | 1668 | 1.0000 | 0 | 0 | 12.4× |
| 5000 | greedy | glyph16 | 95 | 392757 | 1.0000 | 0 | 0 | 1.0× |
| 5000 | beam1 | glyph16 | 95 | 392757 | 1.0000 | 0 | 0 | 1.0× |
| 5000 | beam4 | glyph16 | 95 | 392741 | 1.0000 | 1 | 0 | 1.2× |
| 5000 | beam16 | glyph16 | 95 | 392633 | 0.9997 | 8 | 0 | 2.7× |
| 5000 | beam64 | glyph16 | 95 | 392725 | 0.9999 | 6 | 4 | 26.0× |
| 5000 | greedy | psychedelic | 1 | 825 | 1.0000 | 0 | 0 | 1.0× |
| 5000 | beam1 | psychedelic | 1 | 825 | 1.0000 | 0 | 0 | 1.0× |
| 5000 | beam4 | psychedelic | 1 | 825 | 1.0000 | 0 | 0 | 1.5× |
| 5000 | beam16 | psychedelic | 1 | 825 | 1.0000 | 0 | 0 | 6.4× |
| 5000 | beam64 | psychedelic | 1 | 825 | 1.0000 | 0 | 0 | 72.2× |
| 5000 | greedy | shader | 12 | 4618 | 1.0000 | 0 | 0 | 1.0× |
| 5000 | beam1 | shader | 12 | 4618 | 1.0000 | 0 | 0 | 1.0× |
| 5000 | beam4 | shader | 12 | 4618 | 1.0000 | 0 | 0 | 1.2× |
| 5000 | beam16 | shader | 12 | 4618 | 1.0000 | 0 | 0 | 3.2× |
| 5000 | beam64 | shader | 12 | 4602 | 0.9965 | 1 | 0 | 30.0× |
| 50000 | greedy | cellgrid | 1 | 427 | 1.0000 | 0 | 0 | 1.0× |
| 50000 | beam1 | cellgrid | 1 | 427 | 1.0000 | 0 | 0 | 1.1× |
| 50000 | beam4 | cellgrid | 1 | 427 | 1.0000 | 0 | 0 | 2.3× |
| 50000 | beam16 | cellgrid | 1 | 427 | 1.0000 | 0 | 0 | 11.1× |
| 50000 | beam64 | cellgrid | 1 | 419 | 0.9813 | 1 | 0 | 134.9× |
| 50000 | greedy | chrome | 1 | 2214 | 1.0000 | 0 | 0 | 1.0× |
| 50000 | beam1 | chrome | 1 | 2214 | 1.0000 | 0 | 0 | 1.0× |
| 50000 | beam4 | chrome | 1 | 2214 | 1.0000 | 0 | 0 | 1.0× |
| 50000 | beam16 | chrome | 1 | 2214 | 1.0000 | 0 | 0 | 1.4× |
| 50000 | beam64 | chrome | 1 | 2214 | 1.0000 | 0 | 0 | 8.0× |
| 50000 | greedy | psychedelic | 1 | 829 | 1.0000 | 0 | 0 | 1.0× |
| 50000 | beam1 | psychedelic | 1 | 829 | 1.0000 | 0 | 0 | 1.0× |
| 50000 | beam4 | psychedelic | 1 | 829 | 1.0000 | 0 | 0 | 1.4× |
| 50000 | beam16 | psychedelic | 1 | 829 | 1.0000 | 0 | 0 | 5.8× |
| 50000 | beam64 | psychedelic | 1 | 829 | 1.0000 | 0 | 0 | 68.2× |
| 50000 | greedy | shader | 12 | 4976 | 1.0000 | 0 | 0 | 1.0× |
| 50000 | beam1 | shader | 12 | 4976 | 1.0000 | 0 | 0 | 0.9× |
| 50000 | beam4 | shader | 12 | 4976 | 1.0000 | 0 | 0 | 1.0× |
| 50000 | beam16 | shader | 12 | 4976 | 1.0000 | 0 | 0 | 1.2× |
| 50000 | beam64 | shader | 12 | 4869 | 0.9785 | 2 | 0 | 4.0× |

## Chrome (held out, one run per arm)

Ties go to the anchor, so `code_fnv = Greedy` is **no** only where the width
found a strictly cheaper term.

| cap | width | dag_cost | bytes | objective | code_fnv = Greedy | compile |
|---|---|---:|---:|---|---|---:|
| 5000 | greedy | 1668 | 3424 | shared | — | 177 ms |
| 5000 | beam1 | 1668 | 3424 | shared | yes | 185 ms |
| 5000 | beam4 | 1668 | 3424 | shared | yes | 205 ms |
| 5000 | **beam16** | **1660** | 3408 | shared | no (a strict win) | 347 ms |
| 5000 | beam64 | 1668 | 3424 | shared | yes | 2148 ms |
| 50000 | greedy | 2214 | 4000 | shared | — | 2487 ms |
| 50000 | beam1 | 2214 | 4000 | shared | yes | 2471 ms |
| 50000 | beam4 | 2214 | 4000 | shared | yes | 2535 ms |
| 50000 | beam16 | 2214 | 4000 | shared | yes | 3371 ms |
| 50000 | beam64 | 2214 | 4000 | shared | yes | 19824 ms |

## What the gate caught, and why it had to exist

`Beam::width(k)` is never dearer than `Greedy` **in the extractor's own
objective**. That property held on four hand-built saturated graphs in
`pixelflow-search`'s unit tests while being false three separate ways on real
kernels, because the situations that break it only occur at scale:

1. **The repair runs after the beam's comparison.** `repair_choices_well_founded`
   rewrites picks, and a state cheaper pre-repair can repair to a term dearer
   than the DP's. `Beam` now folds the DP's own repaired term back in as a
   third arm — `Greedy`'s fold with one more candidate — so the property is
   structural.
2. **A singleton reach set means three different things.** The sharing-aware
   DP leaves one for a leaf, for a cycle-priced winner, and for a class where
   *no* candidate beat the initial `usize::MAX` sentinel — and in the third
   the recorded node index (0) and own cost (0) are initial values, not facts.
   The beam's anchor was re-deriving the condition from `cost == CYCLE_COST`
   and walked node 0's whole subtree where the DP had priced a singleton. It
   is now a `ReachShape` enum recorded where the DP decides it.
3. **A member's price was per-state.** It was computed from the variance the
   state's own children induced, so two states agreeing on a class's node
   could price it differently. `own` is now a function of the class and the
   node alone, and the per-state variance field is gone.

Two of the three are the same lesson: *a convention written in a sentinel
value is an invariant something else eventually breaks*. All three were found
by a test, not by reading — and (3) only once that test ran at the **bake
shape** rather than only at `POINT`, which is the shape-is-part-of-the-
objective point above, in its sharpest form.

`pixelflow-pipeline/tests/beam_never_loses_to_greedy.rs` runs the property on
the six glyphs that regressed, at the cap they regressed at, plus the twelve
shaders, at both shapes — 22 s in release.

## The reading

**Width is not monotone, and the third arm is why it is still safe.** `Beam`
is never dearer than `Greedy` in the extractor's objective — the DP's own
repaired term is folded back in as a third arm — but a wider beam can be
dearer than a narrower one,
and is: `chrome_packed` at 5,000 is 1,660 at `k = 16` and 1,668 at `k = 64`. A
wider beam generates strictly more candidates, and top-`k` of a larger
candidate set need not contain top-`j` of a smaller one; a cheap partial merge
that later conflicts with every sibling crowds out the state a narrower beam
carried to the root. That is beam search, not a defect, but it means a width
is a setting to measure and never one to raise on principle.

**Where the term does not move at all, the answer is not "search harder".**
At 50,000 classes chrome's extraction objective is `shared` — the beam is
widening the arm that actually produced the returned term — and yet `k = 64`
returns machine code identical to `Greedy`'s, having spent 19.8 s of compile
against 2.5 s. Sixty-four seats per e-class over 10,256 live classes changed
no choice. That is a much stronger negative than "beam was a bit worse": the
neighbourhood a per-class beam searches does not contain the cheaper term the
witness argument says is in the graph.

**And what width does buy is under a percent, at 26–135× the compile.** The
best cells in the whole matrix: `cellgrid` −1.9% at `k = 64` (both caps),
`shader` −2.2% at `k = 64` and 50,000 with 2 of 12 improved, `glyph16` −0.03%
at `k = 16` with 8 of 95 improved, `chrome` −0.5% at `k = 16` and 5,000.
`psychedelic` never moves at any width, because its objective is
`tree_cheaper` — the tree arm's term wins, and widening the shared arm cannot
reach it by construction. Nothing here is worth 26× compile, which is why the
default does not move.

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
