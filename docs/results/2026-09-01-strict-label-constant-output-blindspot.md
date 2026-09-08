# The strict hindsight label cannot see constant-output rewrites — mechanism, fixtures, and the `sh` re-count (2026-09-01)

> **Instrument change (2026-09-02 forward-port).** Two things about how a
> Phase 3 anytime curve is measured changed with the port, so a re-run does
> not reproduce the numbers below even if nothing else changed: the
> application budget now binds **mid-scan** rather than between rule sweeps
> (`app_actual == app_target` exactly, no overshoot), and the reported cost
> is the **DAG** cost the emitted kernel pays rather than the extraction DP's
> tree total (#1117). Full statement:
> [docs/results/2026-09-02-phase3-instrument-changes.md](2026-09-02-phase3-instrument-changes.md).

**Status: confirmed, pinned by tests, deliberately not fixed.** Referral from Round 1b
(`docs/results/2026-09-01-phase3-round1b-domain-shift.md`, "Referral (labels)"): `pythagorean`
(sin²x + cos²x → 1, `all_rules()` idx 40) fired 231 times across 19 `sh` expressions under unguided
saturation and was strict-positive 0 / 231 times. Ruling (JP, after launch): verify only; do not repair
the label with another hand-drawn bound — hindsight provenance yields exact *ancestry*, not exact
*credit*, and the Guide's training target moves to hindsight return-to-go over diverse trajectories,
validated by counterfactual replay (separate workflow). This note is the evidence.

Reproduce (deterministic; the per-rule table below was byte-identical across two runs):
```
cargo test -p pixelflow-search --lib labeler                       # the five pinned fixtures
cargo run --release -p pixelflow-pipeline --features training --bin strict_label_output_class_recount -- \
    --corpus pixelflow-pipeline/data/corpus_dev_ood.bin --name-prefix dev_sh_ \
    --out-json docs/results/2026-09-01-strict-label-constant-output-blindspot.json
```
`corpus_dev_ood.bin` is the Round-1b file (MD5 `0c7cbe710c50175afb3cd91f60960b64`; regenerate with
`gen_sh_corpus` then `gen_bezier_corpus` if `data/` is empty). The recount's unguided arm is the
Round-1b harness's unguided arm — same `all_rules()`, `APP_CHECKPOINT_GRID`, class cap
(`config_for_node_count`), `CostModel::latency_prior()` `extract_dag` — and reproduces Round 1b's
pythagorean count exactly: 231 firings over the same 19 expressions.

## 1. The mechanism, at the code level

`strict_load_bearing` (`pixelflow-search/src/egraph/labeler.rs`) credits an application iff a chosen
node's `Origin` is `Rule(app)`. An `Origin::Rule` is recorded in exactly one place: `EGraph::add`,
when a node is **minted**. So strict credit is *minted* credit, and three things follow:

1. **A memo-hit right-hand side mints nothing.** `Pythagorean::apply` returns
   `RewriteAction::Create(Const(1.0))`; `apply_action` calls `self.add(node)`, which finds `Const(1)`
   in the memo and returns its existing class — no `ENodeId`, no origin. The firing's only effect is
   `union_counted(match_root, existing_class)`, a `UnionEvent` carrying the `ApplicationId`. The strict
   label never reads union events; the tightened label does (axis 2), which is why `tight` sees every
   effective firing below and `strict` sees none.
2. **Every SH basis function carries a literal `1.0`.** `sh_family.rs` pushes `Const(1.0)` in the
   l = 2, 3, 4 forms (`3cos²θ − 1`, `5cos³θ − 3cosθ`, …). All 19 `sh` expressions on which pythagorean
   matched have a seed `1` (`seed_has_const_one = 19 / 19`), so on this family the identity's RHS
   pre-exists **always**, not "almost always" as the referral guessed — pythagorean minted **0** nodes in
   231 firings. And even without a seed `1`, only the *first* firing per e-graph can mint it; every later
   re-match memo-hits the node it minted (fixture (a): 1 credited of 6 fired).
3. **When the identity does mint its node, the fold consumes it.** In `(sin²x + cos²x + 0.5)·y`
   pythagorean mints `Const(1)`, constant-fold immediately rewrites `1 + 0.5 → 1.5`, and the extracted
   path holds `1.5`, whose origin is the fold. The fold is strict-positive; the identity that made the
   fold possible is not (fixture (a3)).

The same construction makes `identity` (RHS = a child class, always exists: minted **0 / 35,387**),
`annihilator` (RHS = the seed `0`: **0 / 176**), `cancellation` (**0 / 1,158**), `exp-ln-cancel`
(RHS = the argument) and most `inverse-annihilation` firings strict-invisible on `sh` regardless of
whether they paid — see the table in §3.

## 2. Fixtures (pinned, `labeler.rs` tests — the documented blind spot, not the intended semantics)

All five saturate with `all_rules()` to a fixed application budget (no wall clock in the stop
condition) and extract with the latency prior. `strict ⊆ tight ⊆ loose` and `strict ⊆ minted` are
asserted in each.

| | fixture | extracted | rule under test | fired | minted | strict | tight | credit went to |
|---|---|---|---|---|---|---|---|---|
| (a) | `sin²x + cos²x + y` | `1 + y` | pythagorean | 6 | 1 | **1** | 1 | pythagorean — the ONE firing that minted the path's `1` |
| (a2) | `(sin²x + cos²x)·(y + 1)` — the `sh` shape | `y + 1` (seed `1`) | pythagorean | 8 | 0 | **0** | 1 | nobody: the path is all seed nodes |
| (a3) | `(sin²x + cos²x + 0.5)·y` | `1.5 · y` | pythagorean | 6 | 1 | **0** | 1 | constant-fold (minted `1.5`) |
| (b) | `(x · 0) + z` | `z` | annihilator, identity | ≥1 each | 0 | **0** | 1 each | nobody: strict set empty |
| (c) | `exp(ln x) · w` | `x · w` | exp-ln-cancel | 3 | 0 | **0** | 1 | nobody: strict set empty |

Fired/minted counts are what the fixtures observed at the time of writing; the tests pin the
*structure* (which rule is credited, through which node, and that the inclusions hold), not the exact
firing counts. Tight credits the union in every case — through axis 2 (the exact `application_id` on
the `UnionEvent`) plus axis-3 union chasing, which is broad: on `sh` it credits pythagorean in all
19 expressions, including the 17 where the identity's value never reaches the extracted path (§3).

## 3. The `sh` re-count: by output node vs by output class

Rules of interest (full 27-row table with every rule that fired, and per-expression pythagorean rows,
in the JSON). *effect*: `minted` = created ≥ 1 e-node; `union-only` = memo-hit RHS whose union merged
two distinct classes; `no-op` = memo hit into an already-equal class. *strict* = output node on the
path (the label as minted). *class on path* = the firing's output e-class (canonical match root at
extraction time) is one of the extracted path's classes — the step-(2) candidate semantics.
*effective* = class on path and not a no-op.

| idx | rule | fired | minted | union-only | no-op | **strict (node)** | class on path | class on path, effective | tight | loose | exprs |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 40 | `pythagorean` | 231 | **0** | 30 | 201 | **0** | **0** | 0 | 30 | 55 | 19 |
| 14 | `identity` | 34,521 | 0 | 3,820 | 30,701 | 0 | 8,981 | 544 | 3,820 | 24,399 | 88 |
| 13 | `identity` | 866 | 0 | 82 | 784 | 0 | 242 | 17 | 82 | 456 | 12 |
| 15 | `annihilator` | 176 | 0 | 17 | 159 | 0 | 0 | 0 | 17 | 89 | 8 |
| 6 | `cancellation` | 1,104 | 0 | 2 | 1,102 | 0 | 115 | 0 | 2 | 10 | 94 |
| 7 | `inverse-annihilation` | 1,080 | 3 | 111 | 966 | 0 | 0 | 0 | 114 | 124 | 72 |
| 8 | `constant-fold` | 35,783 | 1,808 | 864 | 33,111 | 699 | 16,380 | 1,018 | 2,634 | 28,705 | 100 |
| 0 | `canonicalize` | 3,733 | 319 | 0 | 3,414 | 40 | 1,922 | 170 | 319 | 1,844 | 93 |
| 20 | `doubling` | 37,314 | 14,133 | 521 | 22,660 | 0 | 8,771 | 2,856 | 14,647 | 36,044 | 94 |
| 36 | `sin-angle-addition` | 1,025 | 124 | 11 | 890 | 0 | 944 | 124 | 135 | 230 | 94 |
| 38 | `reverse-angle-addition` | 1,134 | 53 | 0 | 1,081 | 50 | 1,043 | 50 | 53 | 53 | 94 |
| 39 | `half-angle-product` | 3,275 | 192 | 52 | 3,031 | 72 | 728 | 51 | 244 | 501 | 94 |
| 58 | `diff-of-squares` | 578 | 93 | 0 | 485 | 33 | 573 | 92 | 93 | 99 | 89 |

(100 `sh` entries, 1,854,839 applications total; 91 % of all recorded applications are no-ops, matching
the 2026-08-30 scope measurement.)

**Pythagorean on `sh`, per firing:** 231 fired = 30 union-only + 201 no-op; 0 minted; strict 0;
**output class on path 0**; tight 30 (exactly the 30 effective firings). Per expression: 19 / 19 carry
a seed `1.0`; every one has ≥ 1 effective firing (1–4); none has `Const(1)` on its extracted path.

**Where the value went.** Two of the 19 (`dev_sh_00057`, `dev_sh_00078`) extract to
`Add(0.23873243, …)`: 0.23873243 = 0.48860252² = 3/(4π), the folded band energy
k²·(sin²θ + cos²θ) of the l = 1 shell — the pythagorean chain (pythagorean → identity → constant-fold)
completed, and strict credits `constant-fold` (6 and 7 positives respectively) for the constant the
identity produced. The other 17 extract the *explicit* form, e.g.
`MulAdd(0.4886, Add(Mul(Cos θ, Mul(0.4886, Cos θ)), Mul(Sin θ, Mul(0.4886, Sin θ))), …)`: the
pythagorean union happened in a `cos²+sin²` class the path never routes through, because the
factoring that would connect `k·(cos·(k·cos) + sin·(k·sin))` to `k·k·1` was never found — all 19 runs sat at
the class cap (4,933–5,000 of 5,000 classes) while `saturate_until_applications` reported `quiesced`
(the per-rule class-cap skip freezes a sweep with zero unions, which the stop condition reads as
quiescence). So on `sh` the 0 / 231 has two layers: in 2 expressions the identity paid and the label
handed the credit to the fold (the blind spot); in 17 the identity's value never reached the path at
all under this saturation regime (a true non-payer here — the enabler-starvation the Round-1b
`EnablerDiag` already measures, now with the cap named as the reason).

## 4. Why "credit through the output class" would not have fixed it

The original brief's step (2) proposed: an application is strictly load-bearing iff its output
e-class is on the extracted path. Measured on `sh`, that definition is wrong in both directions:

- **Still 0 / 231 for pythagorean.** Its output class — the merged `sin²+cos² = 1` class — is never on
  the path: when the chain completes, the path takes the *fold's* constant one level up, and the `1`
  class is a child of the folded-away product; when it does not, nothing near it is on the path.
- **Massive over-credit elsewhere.** `identity` idx 14 would go from 0 to 8,981 positives, of which
  8,437 are no-op re-matches of an already-merged class; `canonicalize` idx 0 from 40 to 1,922 (170
  effective); `constant-fold` from 699 to 16,380. A no-op that happens to re-match a class on the path
  is not credit. Restricting to effective firings still credits every re-derivation of a value the
  path already had.

That is the ruling in numbers: the union journal tells us exactly which firing merged which classes
(ancestry), and no on-path predicate — node, class, or class-and-effective — turns that into a
per-firing payoff (credit) without either omitting the identity that enabled a fold or admitting the
re-matches that did nothing. A trajectory-level target (hindsight return-to-go, with counterfactual
replay — remove the firing, re-saturate, re-extract, measure the cost delta — as validation) is the
formulation in which "the fold got the credit the identity earned" is not a question, because both
are on the same trajectory.

## 5. Not done here, on purpose

- No label change, no re-mint of TRAIN strict labels, no retrain, no Guide evaluation (ruling).
- What a retrain would have tested had a fix landed: whether pythagorean's TRAIN rate leaves 0.0 and
  whether a Guide that fires it on `sh` closes any of the +0.34 ratio shift Round 1b measured. Under
  the ruling that question moves to the return-to-go workflow, where the first check is simpler: on
  `dev_sh_00057`/`00078`, does the trajectory containing the pythagorean firings have lower return than
  the one without them, and does counterfactual replay agree.
- The class-cap-reported-as-quiesced observation (§3) is a saturation/harness finding, referred to that
  stream; it changes how "full saturation" should be read in Round 1b's `full_run` columns for classical
  `sh` (90 of the 95 classical rows ended within 100 classes of the 5,000 cap; all 19 pythagorean rows
  did, at 4,933–5,000), not any number in this note.
