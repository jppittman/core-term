# A3 fixes, round 1 (TAG=a3-fix1)

Code base `3bfdaded`; HEAD at the start of the round `be70736`. The A3 diff is
`git diff 3bfdaded -- . ':!.wip'`. Each item was checked against the code before it was changed.
Nothing here touches a `Reduce {` site, so the checklist is unchanged (169 sites, 0 UNCLASSIFIED,
0 STALE).

## Gate failures

### Gate 6: `cargo clippy --workspace --all-targets --all-features -- -D warnings` (real, fixed)

Reproduced with `cargo clippy -p pixelflow-ir --all-targets --all-features -- -D warnings`:
exit 101, three `clippy::must_use_candidate` errors at `pixelflow-ir/src/arena.rs` (`ExprArena::nodes`),
`pixelflow-ir/src/dag.rs` (`Dag::iter`, `Rooted::entries`); output kept in `clippy-fix1-repro.txt`.

Root cause: clippy 0.1.97 flags a `pub fn` returning `impl Iterator` that carries no `#[must_use]`, and the
workspace sets `must_use_candidate = "warn"`, which `-D warnings` turns into an error. None of the three is
a line A3 added: `nodes` has the identical signature and attributes at base (`git show 3bfdaded:...`, line
1067), `dag.rs` is not in the A3 diff. They are toolchain drift, and the literal gate command stops at
`pixelflow-ir`, so it also hid everything downstream.

Fix: `#[must_use]` on the three functions (`ExprArena::nodes`, `Dag::iter`, `Rooted::entries`). This is the
CLAUDE.md convention (`#[must_use]` on a `pub fn` returning a value), changes no signature and no behavior,
and puts three attributes in `dag.rs`, which the A3 diff otherwise does not touch. Listed in
`public-surface.md`.

Result: the literal command exits 0 for the whole workspace (run twice, the second after the last edit).

### Gate 7: `bash scripts/check-feature-matrix.sh` (was not run; now run, passes)

Round 1 could not run it: `cargo hack` was not installed. Installed with
`cargo install cargo-hack --locked --root /home/user/scratch/cargo-hack-root` (0.6.45, outside the repo) and
ran the script with that `bin` on `PATH`: exit 0, "OK: every crate's every feature builds in isolation".
cargo-hack rewrites every `Cargo.toml` and `Cargo.lock` while it runs and restores them; after the run
`git status` shows no manifest or lock change (checked before anything was committed).

## Blocking findings

1. **`KnownBound` can move the stop label (bytes; fold_rules.rs:665). Real, narrower than stated;
   documented, not changed in code.**
   Verified in `graph/runner.rs`: `sweep` checks `num_classes() > minted` before every rule's scan
   (:797-801), so the rule appended after `FactorFold` adds a check right after `FactorFold`'s commit and
   partial rebuild. The scan holds `num_classes + sum(predicted)` under the cap, and
   `predicted_growth`'s doc says the sum can under-count (`graph.rs:946-953`), so `FactorFold` can finish
   Completed and leave the graph past `minted`. On the last allowed iteration base then falls out of the
   `for` at its default `IterationCeiling`, and head reports `MintedCap` from the new check. On any earlier
   iteration both report `MintedCap` with the same count (base from the loop-top check, :726).
   Checked and narrower than the finding's wording in one respect: the application allowance has no such
   corner. `FactorFold`'s own scan breaks with `ApplicationBudget` as soon as `applications + queued >=
   cap` (`allowance_spent(ceilings, updates.len())`), and the commits it makes cannot exceed `queued`, so
   `KnownBound`'s entry check (`allowance_spent(ceilings, 0)`) never sees a spent allowance that
   `FactorFold` had not already reported. Only `class_cap` for `iteration_ceiling` can move, and only as
   telemetry's `stop_reason`; the graph, applications, iterations and extraction are the same.
   Why not code: putting `KnownBound` before `FactorFold` shifts `FactorFold`'s `RuleIdx` (spec 6.4 and
   `known_bound_is_the_last_fold_rule` pin the order); removing or gating `sweep`'s per-rule check changes
   base behavior for the existing rules. No test: the corner needs an under-counting growth prediction on
   the final iteration, and no pinned kernel reaches it (gate 2: 28 of 29 records stop at `class_cap` in 2
   iterations, `n=0` quiesces in 1; unchanged).
   Edits: `KnownBound`'s doc (a paragraph, "Inert is not absent", no private link); the plan's A3 "Done" line
   (the identity sentence now reads "unchanged on every pinned kernel", and a sentence states the corner,
   beside the `total_evals` one); `.wip/a3/spec.md` 1.3 (the `KnownBound` row no longer says stop reasons
   are unchanged, and a new row states the corner with its evidence). `critique.soundness.md` is left as
   the record it is.

2. **Plan 3.1 claims `BoundOutOfDomain` at `Kernel::over` (semantics; plan:259). Real; fixed.**
   Ruling J13 says it must not. Code agrees with J13: `Kernel::over` builds `FoldTerms` with `Bound::Lit`
   ends, so a range past 2^24 panics from `KnownRange::new` ("a fold's range ends at most at 16777216
   (2^24)"), pinned by `kernel.rs`'s `a_fold_past_the_exact_bound_is_refused_by_the_builder_too`.
   `BoundRefusal::BoundOutOfDomain`'s own doc already names only `over_values` and `kernel!` (A6). Plan
   3.1 now reads: refused at `over_values` and `kernel!` (A6); `Kernel::over`'s literal range is refused
   past 2^24 with `KnownRange::new`'s message, as today. The A3 bullet (`BoundOutOfDomain ... at
   over_values`) was already right.

3. **Same claim in `push_reduce`'s doc (conformance; arena.rs:972). Real; fixed.** The doc now says
   `Kernel::over_values` refuses one by name (linked) and `Kernel::over` refuses a literal range past 2^24 in
   `KnownRange::new`'s words. `KnownRange::new` is `pub(crate)`, so it is in backticks without a link (a
   first draft linked it; the rustdoc comparison against base caught it, NEW 0 after the change).

4. **A `.wip` ruling label in a tracked test doc (conformance and craft; variance.rs:956). Real; fixed.**
   "N1:" removed; the sentence now starts "The rooted walk and the forward pass are one transfer function
   in two traversal orders". Scanned the added lines of the A3 diff for other J/N/C/S/T-label citations,
   spec/ruling/critique words, `.wip` and model names: none besides this one (the matches are the plan's
   own item names and `CLAUDE.md`).

5. **`public-surface.md` omits changed pub items (conformance; run/public-surface.md:60). Real; fixed.**
   Rows added: `egraph::FactorFold` (declines only a known empty range; decision 5), `kernel::Kernel::over`,
   `arena::ExprArena::known_range` (doc only), `arena::ExprArena::substitute_vars_with` (precondition in the
   doc; not L7). A scan of every `pub` item touched by the diff outside test modules found five more the
   file lacked, also added: `FoldHeader::to_bits` / `from_bits` (doc text and a private predicate),
   `KnownRange::admits` (same predicate through `Fold::is_exact`), `egraph::HalveFold` and `egraph::ENode`
   (doc only), and this round's three `#[must_use]` additions. Nothing is added to the public surface by this
   round.

6. **Bare "(1.2)" in a test doc is the spec's section, not the plan's (craft; arena.rs:2872). Real; fixed.**
   The sort refusal for `ANY`/`ALL` is the plan's 1.4. The doc now reads "a sort error (the plan's 1.4), not
   a variance error, because `push_reduce` makes the cheap check first".

7. **Plan still names `fold_at` and calls it unchanged (craft; plan:389). Real; fixed.**
   `git grep 'fold_at\b' -- . ':!.wip' ':!docs/archive'` had only this line. It now names `fold_terms` at
   `:697-728` (checked in `emit.rs`: the doc comment starts at 697, the function ends at 728), says it emits
   a `FoldTerms` value since A3 where `fold_at` emitted a closure, and builds the same arena. The hole
   filling range `:576-598` is untouched by the diff and still right.

8. **Boolean-mode closure in T-S-13 (conformance and craft; extract.rs:3063). Real; fixed.**
   `build(true)` / `build(false)` become two named closures: `value_twin_only()` (returns graph, the value
   fold's class and its body) and `both_twins()` (that, plus the fold over the literal 40 unioned in). The
   shadowing name is gone. `both_twins_of_a_fold_price_at_the_count_their_bound_names` still passes in all
   three pixelflow-search configurations.

(The list had ten findings. Three defects were each reported by two lenses: the `BoundOutOfDomain` claim in
the plan (semantics and conformance, items 2 and 3 here also cover the code doc the conformance lens
named), the `.wip` label in variance.rs (conformance and craft, item 4), and the boolean closure
(conformance and craft, item 8).)

## Gates after the fixes (TAG=a3-fix1)

| gate | result |
|---|---|
| 1 `byteprobe.sh a3-fix1` | exit 0; avx512 and avx2 IDENTICAL to base |
| 1b `gate1b.sh a3-fix1` | exit 0; 29 lines IDENTICAL; self-check n=2 and n=40 equal the pins |
| 2 `gate2.sh a3-fix1` | exit 0; IDENTICAL to base (29 records) |
| 2x `gate2x.sh a3-fix1` | exit 0; IDENTICAL to base |
| after gates | `git status --short pixelflow-graphics` empty |
| `cargo test -p pixelflow-ir` | 375 passed, 0 failed, 2 ignored |
| `cargo test -p pixelflow-search` | 424 passed, 0 failed, 15 ignored |
| `... --all-features` | 427 passed, 0 failed, 15 ignored |
| `... --features saturation-telemetry` | 427 passed, 0 failed, 15 ignored |
| `cargo clippy --workspace --all-targets --all-features -- -D warnings` | exit 0 (gate 6, now passing) |
| `bash scripts/check-feature-matrix.sh` | exit 0 (gate 7, now run) |
| `cargo fmt --all -- --check` | exit 0 |
| `RUSTFLAGS="-D warnings" cargo check -p pixelflow-ir / -p pixelflow-search --no-default-features` | exit 0, exit 0 |
| rustdoc, C4 method | ir: base 38, head 38, NEW 0, GONE 0; search: base 28, head 28, NEW 0, GONE 0; `RUSTDOCFLAGS="-D warnings" cargo doc -p pixelflow-codegen --no-deps --document-private-items` exit 0 |
| `scripts/check-intent.sh`, `check-doc-paths.sh`, `check-cfg-encapsulation.sh` | OK, OK, OK |
| checklist | 169 sites, 0 UNCLASSIFIED, 0 STALE |
| spec 11 greps | `A3` in `pixelflow-*` `.rs` and in `pixelflow-ir/intent.md`: empty; `real_sorted`/`var_variance`: empty; no tracked file cites `.wip`; no intent.md over 80 lines |
| LFS | `git lfs ls-files | grep -c .` is 68 before and after |

Outputs: `../../gates/out/*.a3-fix1.*`.

## CI on 31d4b58 (after round 1)

CI clippy 1.99 rejected the three `#[must_use]` added to `nodes`, `iter` and `entries` (`double_must_use`); Clippy and both Test jobs were red. Removed; see rulings.md round 3 R1. Local clippy 1.97 reports `must_use_candidate` on them at base too: a toolchain difference, not an A3 defect.

# A3 fixes, round 2 (TAG=a3-fix2)

Code base `3bfdaded`; HEAD at the start of the round `8d1ed9e`. The A3 diff is
`git diff 3bfdaded -- . ':!.wip'` (24 files, +3217 -419, unchanged by this round: no tracked file outside
`.wip/` was edited). Nothing here touches a `Reduce {` site: the checklist is unchanged (169 sites,
0 UNCLASSIFIED, 0 STALE, `checklist-fix2.txt`).

## Gate failures

### Gate 6: `cargo clippy --workspace --all-targets --all-features -- -D warnings` on this host (reproduced; not an A3 defect; rejected, no code change)

Reproduced on this host's stable (clippy 0.1.97, rustc 1.97.0): exit 101, three `clippy::must_use_candidate`
errors and nothing else, `pixelflow-ir/src/arena.rs:1144` (`ExprArena::nodes`), `dag.rs:226` (`Dag::iter`),
`dag.rs:579` (`Rooted::entries`), the build stopping at `pixelflow-ir` (`clippy-fix2-197-literal.txt`).

Root cause, checked three ways:
- The three signatures are byte-identical to base. `nodes` is at `arena.rs:1067` in `3bfdaded` with the same
  `#[inline]` and no `#[must_use]`; `pixelflow-ir/src/dag.rs` is not in the A3 diff at all; `git diff 3bfdaded`
  of `arena.rs` has no added or removed line naming `nodes` or `must_use`.
- All three return `impl Iterator`, and `Iterator` is already `#[must_use]`. Clippy 1.97 still reports
  `must_use_candidate` on them; clippy 1.99 reports the opposite, `double_must_use`, if they carry the
  attribute. No single source text satisfies both toolchains, so the question is which toolchain governs.
- Ruling R1 (round 3) answers it: CI's. `.github/workflows/rust.yaml:421` runs this literal command with
  `dtolnay/rust-toolchain@stable`, which is 1.99 today. R1 also says what a local run on this older clippy
  must show: those three reports and no other error.

Evidence on CI's toolchain. Installed `1.99.0` beside the default (`rustup toolchain install 1.99.0 --profile
minimal -c clippy --no-self-update`; the default stays 1.97; rust-toolchain.toml says `stable`, and `cargo
+1.99.0` overrides it per command). Then, with its own target dir (`CARGO_TARGET_DIR=/home/user/scratch/a3-fix2/t199`):
- `cargo +1.99.0 clippy --workspace --all-targets --all-features -- -D warnings` at HEAD: **exit 0**, no warning,
  no error (`clippy-fix2-199-literal.txt`). This is the gate as CI runs it.
- The attribute the earlier round added, tried on a plain extraction of `3bfdaded` (`/home/user/scratch/a3-fix2/dm-src`,
  not a worktree): `cargo +1.99.0 clippy -p pixelflow-ir ... -D warnings` exits 101 with "this function has a
  `#[must_use]` attribute with no message, but returns a type already considered as `#[must_use]`" at the added
  line (`clippy-fix2-199-double-must-use.txt`). So adding the attribute is what is wrong on CI's toolchain; it is
  the reason `cc913d5` removed it, and the reason this round does not put it back.
- CI itself: the `Clippy` job (and both `Test on` jobs) is `success` on `cc913d5` (run 39932387202) and on
  `8d1ed9e` (run 39934526235), where `8d1ed9e` is HEAD at the start of this round.
- Nothing hidden behind the stopped build on 1.97: a plain `cargo clippy --workspace --all-targets --all-features`
  (no `-D`) exits 0 with exactly three warnings, the three above, and no other, so no A3-added `pub fn` lacks a
  needed `#[must_use]` (`clippy-fix2-197-plain.txt`).

Why no code change: an `#[allow(clippy::must_use_candidate)]` on the three would satisfy both toolchains, but it
would add three lines to `dag.rs` (not an A3 file) and one to `arena.rs` to quiet an older clippy about code A3
did not write, which R1 already rules a pass ("those three reports are its only errors"). The gate stays as
written; its governing toolchain is CI's. For the next round: run gate 6 as `cargo +1.99.0 clippy --workspace
--all-targets --all-features -- -D warnings` with a separate `CARGO_TARGET_DIR` (it needs the toolchain installed
above), and use the 1.97 run only as the plain, no-`-D` count of three.

## Blocking findings

None were listed.

## A note defect found while verifying (real, fixed)

`public-surface.md` still carried a row saying `#[must_use]` was added to `ExprArena::nodes`, `Dag::iter` and
`Rooted::entries`. `cc913d5` removed all three attributes but not the row, so the file listed a changed pub item that is
not changed. The row now says the three are byte-identical to base. Notes only; no code.

## Gates after this round (TAG=a3-fix2)

| gate | result |
|---|---|
| 1 `byteprobe.sh a3-fix2` | exit 0; avx512 and avx2 IDENTICAL to base |
| 1b `gate1b.sh a3-fix2` | exit 0; 29 lines IDENTICAL; self-check n=2 `22ad8819798461a6` and n=40 `e95137a9d464391e` equal the pins |
| 2 `gate2.sh a3-fix2` | exit 0; IDENTICAL to base (29 records) |
| 2x `gate2x.sh a3-fix2` | exit 0; IDENTICAL to base |
| after gates | `git status --short pixelflow-graphics` empty; no tracked file modified |
| `cargo test -p` ir / search / search `--all-features` / search `--features saturation-telemetry` | 375 / 424 / 427 / 427 passed, 0 failed (ignored 2 / 15 / 15 / 15) |
| `cargo test -p` codegen / compiler / graphics / pipeline / core / runtime | 160 / 304 / 236 / 221 / 136 / 152 passed, 0 failed (ignored 0 / 6 / 15 / 2 / 1 / 6) |
| test compiler warnings | 0 in every log (`tests-fix2-summary.txt`) |
| clippy, 1.99.0 (CI's), literal `-D warnings` | exit 0 |
| clippy, 1.97.0 (this host's), literal `-D warnings` | exit 101, exactly the three base reports (R1: a pass) |
| clippy, 1.97.0, plain | exit 0, exactly those three warnings |
| `cargo fmt --all -- --check` | exit 0 |
| `scripts/check-intent.sh`, `check-doc-paths.sh`, `check-cfg-encapsulation.sh` | exit 0, exit 0, exit 0 (121 grandfathered) |
| checklist (`python3 -I .wip/a3/gen/checklist.py`) | 169 sites, 0 UNCLASSIFIED, 0 STALE (already-correct 39, changed-by-A3 22, left 13, tests 75, n/a 5, A4 15, A5 0) |
| LFS | `git lfs ls-files | grep -c .` is 68 before and after |

Outputs: `../../gates/out/*.a3-fix2.*`; this directory's `*-fix2*` files.

# A3 closing fixes, round 1 (TAG=a3-close1)

Code base `3bfdaded`; HEAD at the start of the round `2083450` (it adds only the review notes and the workflow script under `.wip/` to the reviewed `8da7c86`; no tracked file outside `.wip/` differs).
The A3 diff is now 25 files, +3755 -450 (`diffstat-close1.txt`): `pixelflow-search/src/runtime.rs` is new to it. Rulings: `rulings.md`, "Round 4" (F1-F5). The A3 diff is `git diff 3bfdaded -- . ':!.wip'`.

## F1 (blocking): a leaked placeholder must not pass the unit check

**Done.**
- `pixelflow-search/src/runtime.rs`, `Program::of`: after the existing `variance.without(COORDS).is_const()` assertion (kept first, so
  every message that existed still comes out for the input that had it), `body.free_index(body_root)` is asked; `Some(index)` panics
  "the unit {key:?} reads Var({index}), an index no fold inside it binds — a fold index that escaped its builder …". `Program::of`'s doc
  says why two checks.
- Pin (pixelflow-search): `runtime::tests::a_unit_that_reads_an_escaped_fold_index_is_refused`, `#[should_panic(expected = "an index no
  fold inside it binds")]`, the repro program (`Kernel::sum_over(2, |i| { escaped = Some(i.clone()); i.clone() })`, `intern`, `push_ref(key)
  + X`, `optimize_runtime_arena(.., LatticeShape::POINT)`), with the premise asserted first (`compute_arena_variance` of the escaped index is
  `CONST`).
- `Variance::of_var`'s doc (`pixelflow-ir/src/variance.rs`): the false sentence ("Nothing that places a node sees a placeholder …") is
  replaced by the true one: a placeholder can escape its builder, its variance is exact for L3's question and is not "closed", and
  closedness checks use `free_index`.

**Audit** of every reader of a variance (`grep -rn 'compute_arena_variance\|of_var\|variance_from\|admits_bound\|is_invariant_in'`, and
`.variance(` / `is_const()` for the e-graph and the extractor). Each either refuses a placeholder or asks a question a placeholder answers
exactly; none treats a leaked one as a constant without a refusal elsewhere on its path.

| reader | question it asks | placeholder | verdict |
|---|---|---|---|
| `runtime.rs` `Program::of` (`:289` at base, the unit check) | is the unit closed? | CONST, admitted | **fixed**: `free_index` (`:307`) |
| `graph.rs` `EGraph::admit_unit` (`:310`) | is this variance closed over the coordinates? | cannot see one (it gets a `Variance`) | its only caller outside tests is `Program::of`, now guarded; the doc says so (no code change: it has no term to ask) |
| `program/lower.rs` `arena_to_schedule` (`:150` variance, `:190-196` the `Var` arm) | placement by each def's variance | CONST, but never reached | the `Var` arm refuses any `Var` that is no fold's binder (`Binder::from_var(i) == None` → panic) before a def exists for it. Existing refusal; its message now also names an escaped fold index; pinned by new `a_surviving_placeholder_fails_loudly` (`#[should_panic(expected = "escaped its builder")]`) |
| `program/scopes.rs` | placement by `Def::variance` | n/a | reads defs only, so only what `arena_to_schedule` admitted |
| `passes.rs` `lower_dwrt` → `Rules::tabulation` (`:641`, table at `:600`) | does a gather's index move with the variable being differentiated? | CONST | not a closedness or placement check. A placeholder is none of the coordinates, so "does not move" is exact, and `diff_node`'s `Var` arm (`:714`) already differentiates every other `Var` (a fold binder included) to 0. Unchanged |
| `nnue/factored.rs` `variance_histogram` (`:823`), `extract.rs` `node_variance` (`:1276`), `price`, `best_var` | a statistic; a price weight (`shape.evals`) | CONST | judge nothing closed or placed; unchanged |
| `fold_rules.rs` `FactorFold`/`substitute` (`variance(class).depends_on(binder)`) | does this class read the binder? | CONST: no | exact: a placeholder is not a binder slot |
| `arena.rs` `push_reduce` (`:1015`) and `graph.rs` `refuse_a_varying_bound` (`:428`) | L3 | CONST: admitted | L3 itself, by design (J5); `close_over` re-pushes after the rename |
| `kernel.rs` `Kernel::by_ref` (`:1058`), `:1196` | closedness | n/a | already `free_index` |

## F2 (blocking): L7 never builds an unlawful bound

**Done.** `PlanBuilder::substitute` (`pixelflow-search/src/egraph/fold_rules.rs`) now carries a `Site` per task:
- `Site::Body`: the first member (`representative`), exactly the walk it was. Its memo is a separate `BTreeMap<EClassId, Done>` (`Memo.body`),
  so a fold over literals builds no wider key and takes no new branch but one `match`.
- `Site::Bound { fold }`: a nested fold's value bound and every class under one. The member is the first whose own variance
  (`ENode::variance` over its children's class facts) `fold.admits_bound` accepts; none ⇒ `substitute` returns `None`, the rule declines.
  The site is decided in one place (`Site::of_child`): a Reduce's body keeps the site, its value bounds are bounds; a fold under a value
  bound declines (no builder makes one).
- "after the substitution's own effect is accounted for": no code is needed and the doc says why. A leaf becomes a literal (peel) or the
  substituted binder plus its stride (halve); that binder is neither a coordinate nor the nested fold's own (a fold that rebinds it is
  named before it is entered, by L4), so it is never in the refused set, and a member's variance before the substitution is the law's
  answer after it.
- **The ruling's mechanism, extended one step**: it says to choose the member of the *bound class*. The same disease recurs one level
  down (a bound `i + P` whose `P` has a coordinate-reading first member), so the discipline is applied to every class under a bound.
  Without that the repro's family would still panic at depth. `Task::Build` carries the class and the site only and re-reads the member
  (the choice is a function of the graph, which a walk does not change), so a `Task` stays small (measured below).
- `HalveFold`'s doc and the module doc's "A representative suffices" bullet say it; plan L7 and the A3 Done line say it.

Pins (`fold_rules::tests`); the first three were run with the member choice reverted to `representative` and each failed (two by assertion,
the runtime one with the `BoundVaries` panic), then restored:
- `a_nested_bound_is_rebuilt_through_the_member_that_keeps_l3`: the **two minimal hand-made-union cases** (`HalveFold` alone at n=4,
  `PeelFold` alone at n=1). **Deviation from the ruling's wording, stated:** with the repro's union the bound class holds `i + 1`
  beside `((i + X) + 1) − X`, a member that keeps L3, so the rule does not decline; it **plans through `i + 1`** and the graph takes the plan
  (one saturation round, no panic; no `Sub` in the plan). The ruling expected a decline here; the rule's own text ("if none does, the plan
  declines") gives a plan, and declining would have been less complete.
- `a_nested_bound_with_no_member_that_keeps_l3_declines`: the decline case proper, `i + X` beside `i + Y` (the class reads neither, so L3
  admitted the fold, and no member keeps it): both rules `apply` to `None`, and a two-round run applies nothing and does not panic.
- `the_runtime_rules_saturate_a_triangular_fold_with_a_varying_spelling`: the repro (`Σ_{i<8} ( Σ_{j<i} X·j + ((i + X) + 1) − X )`, the
  inner bound `Var(i)`, `Vocabulary::Runtime`, `RuleSet::runtime()`, `compatibility(12)`) and its three variants (bound `i + 1`; the extra
  term before or after): saturates, the graph holds more than 3 value-bounded slot-0 folds (the second level of halving, the step that
  panicked), extraction is closed (`free_index == None`), and **every extracted value bound reads the outer index and nothing else**.
- `extracting_a_bound_class_with_no_member_that_keeps_l3_is_refused_by_name` (`#[should_panic(expected = "BoundVaries")]`): the
  extraction-side hole, below.

**Extraction-side hole: stated, not closed, and the pin shows it is real.** A bound class with no member that keeps L3 is materialized
through one that reads a coordinate, and `ExprArena::push_reduce` refuses with `BoundVaries` (`arena.rs:1015`): loud. With a non-negative
cost model a class that holds a lawful member extracts it in the cases tried (the repro extracts `i + 1`), so extraction does **not**
"already never pick such a member": a class whose fact is a meet of members each reading a different coordinate has none to pick. So
the statement is in spec §13 (J24) and as one sentence in the plan's A4 bullet.

## F3: one literal twin

**Done.** `ENode::literal_twin(&self, literal: impl Fn(EClassId) -> Option<Lit>) -> Option<ENode>` (`node.rs`, `pub(crate)`); the private
`extract.rs` function is deleted (and its now-unused `Bound` import), the three price sites call `node.literal_twin(..)`, and
`KnownBound::apply` is `node.literal_twin(literal).map(RewriteAction::Create)`. The arity-1 early return is inside the method, so a fold
over literals still returns on its header before anything is read. Docs: the method says why there is one; `KnownBound`'s doc says the fold
it adds is that function's; the old "the two price alike" sentence is gone with the function; references in `cost.rs`, `fold.rs`, `below`
point at the method.

## F4: doc nits

All done, as written: `KnownBound`'s doc no longer says "a single label" and states the `total_evals` rise; law labels anchored at first
use per label per file in `fold_rules.rs`, `graph.rs`, `node.rs`, `runtime.rs` ("L3 of docs/language/2026-10-20-a-bound-is-a-value.md",
including the three graph.rs test docs, where the optimizer's L1-L4 live); `FoldHeader::new`'s summary line says what it checks;
`ENode::reduce`'s Panics is "As [`FoldHeader::new`]."; `is_the_fixture`'s doc names `copy`; the comment above `transfer`'s `Var` arm
names the placeholder; `close_over`'s "Constness is discovered first" paragraph now follows "The one definition of that rule and of the
rename" so its antecedent is intact.

## F5: public-surface.md

Done: the `must_use` row dropped; rows added for `egraph::extract`/`extract_dag` (and `extract_dag_scoped` through `Optimized::to_arena`),
`HalveFold`/`PeelFold`, `runtime::optimize_runtime_arena` (F1), the private `FoldHeader::with_a_value`, `ENode::literal_twin`, and
`Site`/`Site::member`/`Site::of_child`/`Memo`; the codegen tail row names the new test and the message change.

## Other .wip edits

`rulings.md` "Round 4"; `spec.md` §1.1 (L7 row), §1.3 (the `of_var` row's false claim), §3.1, §6.7, §6.8, §7.3 (`literal_twin` moved), §10
(T-S-11..14, T-R-1, T-C-2), §13 (J5 corrected, J24-J26); `gen/classes.json`: removed 3 rows whose sites left (`extract.rs` `literal_twin`,
`KnownBound::apply`'s header test, T-S-6's walk), added 3 (`node.rs` `literal_twin` changed-by-A3, `Site::of_child` changed-by-A3,
`nested_bounds` tests). Checklist: 169 sites, 0 UNCLASSIFIED, 0 STALE.

## Gates after this round (TAG=a3-close1), on the final tree

| gate | result |
|---|---|
| 1 `byteprobe.sh a3-close1` | exit 0; avx512 and avx2 IDENTICAL to base |
| 1b `gate1b.sh a3-close1` | exit 0; 29 lines IDENTICAL; self-check n=2 `22ad8819798461a6`, n=40 `e95137a9d464391e` equal the pins |
| 2 `gate2.sh a3-close1` | exit 0; IDENTICAL to base (29 records) |
| 2x `gate2x.sh a3-close1` | exit 0; IDENTICAL to base (extraction fields) |
| 5 `measure.sh a3-close1` | total Ir 6,311,544,165 (+0.247 percent; limit +0.5: PASS; r3 was +0.158); `Saturation::saturate` inclusive 2,076,042,600 (-0.163); `extract_dag_over` inclusive 1,142,798,209 (+1.713; r3 +1.157); median peak RSS 96,876 KiB (+0.415); median wall 1127 ms (reported only) |
| `cargo test -p` ir / search / search `--all-features` / search `--features saturation-telemetry` / codegen | 375 / 429 / 432 / 432 / 161 passed, 0 failed (ignored 2 / 15 / 15 / 15 / 0); 5 more search tests and 1 more codegen test than r3; no compiler warning in any log (`tests-close1.*.txt`) |
| `cargo +1.99.0 clippy -p pixelflow-ir -p pixelflow-search -p pixelflow-codegen --all-targets --all-features -- -D warnings` (CI's toolchain; `CARGO_TARGET_DIR=/home/user/scratch/a3-close1/t199`) | exit 0, no warning (`clippy-close1-199.txt`); re-run for `pixelflow-search` after the last doc edit, exit 0 (`clippy-close1-199-search-recheck.txt`) |
| `cargo fmt --all -- --check` | exit 0 |
| `RUSTFLAGS="-D warnings" cargo check -p pixelflow-{ir,search} --no-default-features` | exit 0, exit 0 |
| `RUSTDOCFLAGS="-D warnings" cargo doc -p pixelflow-codegen --no-deps --document-private-items` | exit 0 |
| rustdoc C4 (`rustdoc_norm.py --compare`) | pixelflow-ir base 38, head 38, NEW 0; pixelflow-search base 28, head 28, NEW 0 (the first run had NEW 1: `KnownBound`'s public doc linked the `pub(crate)` `ENode::literal_twin`; now written without a link, as C5 says) |
| `scripts/check-intent.sh`, `check-doc-paths.sh`, `check-cfg-encapsulation.sh` | exit 0, exit 0, exit 0 |
| checklist (`python3 -I .wip/a3/gen/checklist.py`) | 169 sites, 0 UNCLASSIFIED, 0 STALE (already-correct 39, changed-by-A3 22, left 13, tests 75, n/a 5, A4 15, A5 0) |
| B1 greps, `fold_cost(` only in `cost.rs`, no `real_sorted`/`var_variance`, no tracked file citing `.wip` | empty / empty / pass / pass / pass |
| `git status --short pixelflow-graphics` after the gates; `git lfs ls-files \| grep -c .` | empty; 68 |

### The compile-cost regression found and fixed on the way

The first measure of this round (before the commit, a throwaway tag) read total Ir +0.641 percent, over the +0.5 limit, with
`extract_dag_over` +28 M Ir above r3. The profile diff (`cg-annotate.close1-preinline.txt` against r3's) showed `ENode::literal_twin::<..>` as three
new out-of-line functions (8.1 M + 7.7 M + 2.2 M Ir): moving the function from `extract.rs`, where the compiler had inlined it into its
callers, to `node.rs` made it a call per candidate priced. `#[inline]` on `ENode::literal_twin` brought the total to +0.321 percent; making
`Task::Build` carry the class and the site only (it re-reads the member, a pure function of the graph) shrank the walk's `Task` and brought it
to +0.247 percent. Throwaway tags `a3-close1-inline` and `a3-close1-b` hold those two measurements in `gates/out/`.

Outputs: `../../gates/out/*.a3-close1.*`; this directory's `*-close1*` files.

# A3 closing fixes, round 2 (TAG=a3-close2)

Code base `3bfdaded`; HEAD at the start of the round `1301e5a` (the round-1 closing gates; no tracked file outside `.wip/` differs from `e84979c`'s tree).
Rulings: `rulings.md` rounds 1-4, unchanged; F1-F5 stay authoritative. Reviewer's scratch crate: `/home/user/scratch/a3-close-review` (copied, not edited, to
`/home/user/scratch/a3-close2/probe`). The A3 diff is `git diff 3bfdaded -- . ':!.wip'`.

## Gate failures

None were listed.

## Blocking finding: F2 is incomplete (soundness; `fold_rules.rs:214`)

**Real. Fixed.** Verified before changing anything: the three scratch cases reproduce at HEAD from a fresh build of the probe copy, each with the
backtrace the finding names (`refuse_a_varying_bound <- EGraph::add <- NodeSink::make <- plan_shape <- apply_action`):

| case | rule set | result at HEAD |
|---|---|---|
| `stale-halve2` (`P = ((i + X) − X) + 1`, `(i + X) − X = i`, `P = ((i + X) + 1) − X`; nested bound `i`) | `HalveFold` alone, n = 4 | `BoundVaries ... Variance{X,i5}` |
| `stale-peel` (`P = (X − X) + 1`, `X − X = 0`; nested bound `i + 1`) | `PeelFold` alone, n = 1 | `BoundVaries ... Variance{X}` |
| `stale-halve3` (both spellings of `P`'s sum before `K = i`) | `RuleSet::runtime()`, n = 2, 4, 8 | `BoundVaries ... Variance{X,i5}`; also at **one** iteration |

The mechanism is as stated. `var_fact` is seeded at birth and never repaired upward (`columns.rs`, "Never repaired upward"); `rebuild` re-keys a node it
processes and leaves its parents' facts alone; a rebuilt bound `i + 1` / `0 + 1` hash-conses into that class, and `add`'s backstop reads the class's fact.
Three docs claimed otherwise and are corrected (below). Nothing else in the finding was wrong; the "no arena-only trigger" claim I re-ran (below).

**The fix** (`pixelflow-search/src/egraph/graph.rs`, with `fold_rules.rs` docs and `rewrite.rs`'s doc):
- `NodeSink` gains `refuses(&self, node) -> bool`: whether `make(node)` would panic on it. `EGraph`'s answers after the same canonicalization and memo lookup `add` makes, then
  the backstop; `GrowthPredictor`'s after the memo and `pending` lookups, reading the fact of a class it minted off the node it minted it for (`ENode::variance` over the
  children's facts, exactly what `add` seeds; `GrowthPredictor::fact`). A fold over literals returns on its arity before anything is read (`has_a_value_bound`).
- `plan_shape` asks `sink.refuses(&node)` before making a `Reduce` and returns `None`; `apply_action`'s `Plan` arm unions nothing on `None` (0), and the prediction, which runs
  the same function, counts the classes built before the refusal, as the applier builds them. So the exactness `apply_measured` asserts holds for a declined plan.
- The backstop is one definition, `varying_bound(node, fact)`, asked by `add` (through `refuse_a_varying_bound`, out of line as before) of the graph's facts and by the predictor of the
  facts `add` would seed. The panic text is unchanged (`BoundVaries: ...: Fold {..}`; `VaryingBound`'s `Display`).
- The check is exactly the condition under which `add` would have panicked, asked at the same point of the sequence, so **a run that did not panic does no different thing**: the
  reviewer's 456 generated arenas and 160 swept ones, `RuleSet::runtime()` at `compatibility(12)`, give byte-identical extracted terms to the reviewer's pre-fix outputs
  (`sweep-close2.gen-0..3.txt`, `sweep-close2.sweep.txt`: `cmp` equal; the sweep file differs only by the reviewer's own trailing `exit 0` line), 0 panics. And the gates below.

**Rejected alternatives** (spec J27 has the long form):
1. *Planner-side memo replay* (the finding's second way): `substitute` resolving the bound's nodes to the classes they will land in and returning `None`. It mints nothing on a decline,
   but it predicts where `add` lands at scan time, and a commit earlier in the same batch changes what a node canonicalizes to (a union merging a child), so `add` could still reach
   the panic. Deciding at the call that would panic cannot be raced. Cost accepted: a declined plan leaves the nodes it built before the refusal (orphans), which cost classes,
   never soundness, and only on a plan that used to panic.
2. *Refresh a class's fact on a memo hit*: sound by the law, but it adds work to every `add` of a literal kernel.
3. *Loosen the backstop to "some member keeps L3"*: wrong one level down (a bound `W + 1` whose child `W` is stale has one member and no lawful one) and it weakens the one check that catches a rule bug.

**Performance, and a draft that was rejected for it.** The first draft made `add` a thin wrapper over `try_add -> Result<EClassId, VaryingBound>` and `NodeSink::try_make`
(one definition of `add`). Gate 5 read total Ir +0.506 percent against base (limit +0.5; round 1 was +0.247): the `Result` return and the moved panic formatting cost +16 M Ir in `add`,
the predictor's `make` and their callers on the 29 programs (`add` is the hottest call in the crate). A second draft (`add` calling an inlined `refusal`) was worse, +0.664, because it
changed LLVM's inlining of `Columns::push`, `absorb`, `merge_kinds` and `write_back`. The shipped shape keeps `add` and `make` and their return types exactly: `add`'s guard is spelled as in
round 1 (`if let ENode::Reduce { header, .. } = &node && header.arity() > 1` then the out-of-line `refuse_a_varying_bound`); spelling the same predicate through `has_a_value_bound` measured
+0.308 (4.5 M Ir more in `add` alone; the rest is noise-sized), so the comment above the guard says why it is in place. Final: total Ir **+0.193** percent (limit +0.5; below round 1's +0.247),
`Saturation::saturate` inclusive -0.131, `extract_dag_over` inclusive +1.274 (round 1: +1.713), median peak RSS +1.289 (5 runs; round 1 +0.415; the experiments of this round read
+0.427 to +1.289 on the same code shape, so it is run noise; reported only). Experiments' outputs: `gates/out/measure.a3-close2-{b,c,d,e,f}.*`, profiles `cg.a3-close2-*.out`.

**Docs corrected** (the three the finding names, and the sentences that depended on them):
- `PlanBuilder::substitute`'s doc: "the member's variance judged before the substitution is the law's answer after it" is replaced. The substitution adds no refused bit, so a node the plan
  *mints* is as lawful as the member it was read from; where a node *lands* is the graph's to say, and the graph declines (`plan_shape`) before `add`'s backstop could panic.
- `HalveFold`'s doc: "`EGraph::add` accepts it because the rule builds it only through a member ... that already keeps L3" now says the member choice is half of it and the graph declines the
  other half. The module doc's "A representative suffices" bullet gains the same sentence.
- The plan's L7 (`docs/language/2026-10-20-a-bound-is-a-value.md`): the rebuilt bound is a term the graph hash-conses; it can land in a class seeded wider than the term; the rewrite declines then too;
  `EGraph::add` panics only for a caller that is not a plan. The A3 bullet's pin list gains "a rule-made bound that lands in a class whose fact fails L3 (declined, not built)".
- `EGraph::add`'s doc: the "never becomes illegal" paragraph lists the second exception (a stale class) and says why a rule's plan does not reach it. `RewriteAction::Plan`'s doc says a plan can be declined.
- `.wip/a3/spec.md` §1.1 (L7 row), §6.7, §6.8 (the false sentence is quoted and corrected), §13 J27; `run/public-surface.md`; `gen/classes.json`.

**Tests added** (all fail with the check removed: `plan_shape` reverted to `sink.make` for the `Reduce` arm, each with the `BoundVaries` panic; restored):
- `pixelflow-search`, `egraph::fold_rules::tests::a_rebuilt_bound_that_lands_in_a_stale_class_is_declined`: the two single-rule cases, `HalveFold` at n = 4 and `PeelFold` at n = 1, each over its
  hand-made unions, with premises asserted (the landing class's fact reads X, a member of it keeps L3, the rule still plans), a control with no stale class where the same rule builds the
  nested fold (2 value-bounded inner folds), and the declined run building none (1), every bound in the graph keeping L3.
- `egraph::fold_rules::tests::the_runtime_rules_saturate_a_graph_with_a_stale_bound_class`: `stale_halve3` under `RuleSet::runtime()` at n = 2, 4, 8, `compatibility(6)` (the panic was at iteration 1;
  twelve iterations cost 8 s in a debug build, six cost tens of milliseconds), extracting a closed term whose bounds read the outer index alone.
- `egraph::graph::variance_fact_tests::a_plan_whose_bound_lands_in_a_varying_class_is_declined_and_predicted_so`: three plans built by hand over one body (a bound that lands in the stale class, a bound
  this plan mints that reads X, a constant): the first two are declined (`apply_action` returns 0, no fold made) and the third is built; for each, `predicted_growth` equals the classes actually built.
  The second case is what exercises `GrowthPredictor::fact` on a class the prediction minted.

## Gates after this round (TAG=a3-close2), on the final tree

| gate | result |
|---|---|
| 1 `byteprobe.sh a3-close2` | exit 0; avx512 and avx2 IDENTICAL to base |
| 1b `gate1b.sh a3-close2` | exit 0; 29 lines IDENTICAL; self-check n=2 `22ad8819798461a6`, n=40 `e95137a9d464391e` equal the pins |
| 2 `gate2.sh a3-close2` | exit 0; IDENTICAL to base (29 records) |
| 2x `gate2x.sh a3-close2` | exit 0; IDENTICAL to base (extraction fields) |
| 5 `measure.sh a3-close2` | total Ir 6,308,131,681 (+0.193 percent; limit +0.5: PASS); saturate inclusive -0.131; extraction inclusive +1.274; median peak RSS +1.289 (reported only) |
| `cargo test -p pixelflow-search` / `--all-features` / `--features saturation-telemetry` | 432 / 435 / 435 passed, 0 failed (ignored 15 each); 3 more than round 1 (429 / 432 / 432); 0 compiler warnings (`tests-close2.*`) |
| `cargo +1.99.0 clippy -p pixelflow-search --all-targets --all-features -- -D warnings` (CI's toolchain; `CARGO_TARGET_DIR=/home/user/scratch/a3-close1/t199`) | exit 0, no warning (`clippy-close2-199.txt`) |
| clippy 1.97 plain, same command without `-D` | the three base `must_use_candidate` reports in `pixelflow-ir` (R1) and nothing else |
| `cargo fmt --all -- --check` | exit 0 |
| `RUSTFLAGS="-D warnings" cargo check -p pixelflow-search --no-default-features` | exit 0 |
| `RUSTDOCFLAGS="-D warnings" cargo doc -p pixelflow-codegen --no-deps --document-private-items` | exit 0 |
| rustdoc C4 (`rustdoc_norm.py --compare`), pixelflow-search | base 28, head 28, NEW 0, GONE 0 |
| `scripts/check-intent.sh`, `check-doc-paths.sh`, `check-cfg-encapsulation.sh` | exit 0, exit 0, exit 0 |
| checklist (`python3 -I .wip/a3/gen/checklist.py`) | 172 sites, 0 UNCLASSIFIED, 0 STALE (already-correct 38, changed-by-A3 25, left 13, tests 76, n/a 5, A4 15, A5 0). Rows: `add`'s guard (changed-by-A3), `has_a_value_bound` and `varying_bound` (new, changed-by-A3), `plan_shape`'s `Reduce` arm (text changed), `a_sum_to_the_bound_built_by` (new test) |
| B1 greps (`A3` in `pixelflow-*` `.rs` and `pixelflow-ir/intent.md`), no tracked file outside `.wip` citing `.wip`, no ruling or J/N/C/S/T label in the added tracked lines, no model names | empty / empty / pass / pass / pass |
| `git status --short pixelflow-graphics` after the gates; `git lfs ls-files \| grep -c .` | empty; 68 |

`pixelflow-ir` and `pixelflow-codegen` are untouched this round (no tracked file of either differs from the previous commit), so their tests were not re-run; both were built by the gates.
Outputs: `../../gates/out/*.a3-close2*`; this directory's `*-close2*` files.

# A3 polish, round 1 (TAG=a3-polish1)

Code base `3bfdaded`; HEAD at the start of the round `3a696db` (the closing gates of round 2, `7e1a2f3`, plus the polish inputs). The polish diff is
`git diff 7e1a2f3 -- . ':!.wip'`. Input: `run/closing-nonblocking.json`, 20 findings. Not applied: #7 (the blocking one, fixed in `d4bcd11`) and #17 ("no defect").
Every other finding was checked against the code before a line changed; the outcomes follow, in the order the round's brief listed them.

## F3: the literal twin's fast path (finding #0). Applied, measured, and the claim "reduced, not removed" stands

**Verified.** The numbers in the finding are right for `close1` (total Ir 6,311,544,165 against r3's 6,305,898,607, +5.65 M) and wrong about the tip it was filed
against: closing round 2 had already brought the total to 6,308,131,681 (+2.23 M over r3; +0.193 percent over base), so the gap the finding asks to close was 2.2 M, not 5.6 M.

**Done.** `ENode::literal_twin` (`node.rs`) is an `#[inline(always)]` guard (not a `Reduce`, or arity 1: `None`) over `ENode::value_bounded_twin` (private), the
`discover`/`reduce` work. A leaf and an operation answer on the first test, a fold over literals on the second, and only a fold with a value bound reaches the rest.
The doc says what the guard is and no longer says "answer on the first test" of a fold over literals.

**Measured**, three forms of the body behind the one guard, on the same workload (`zz_prof_all_cold`; throwaway tags kept in `gates/out/`):

| tree | tag | total Ir (vs base) | `extract_dag_over` inclusive | `Saturation::saturate` inclusive |
|---|---|---|---|---|
| base | `base` | 6,295,969,957 | 1,123,555,965 | 2,079,429,510 |
| r3 (before the move to `node.rs`) | `a3-r3` | 6,305,898,607 (+0.158) | 1,136,559,328 (+1.157) | 2,076,320,833 (-0.149) |
| close1 (the finding's tree) | `a3-close1` | 6,311,544,165 (+0.247) | 1,142,798,209 (+1.713) | 2,076,042,600 (-0.163) |
| close2 (HEAD at the start of this round) | `a3-close2` | 6,308,131,681 (+0.193) | 1,137,870,806 (+1.274) | 2,076,709,606 (-0.131) |
| guard + `#[inline(never)]` body, `node.rs` change alone | `a3-polish1-f3` | 6,312,486,737 (+0.262) | 1,139,849,121 (+1.450) | 2,077,266,197 (-0.104) |
| guard + `#[inline(never)]` body, final tree | `a3-polish1-a` | 6,310,530,002 (+0.231) | 1,140,855,973 (+1.540) | 2,075,533,808 (-0.187) |
| guard + `#[inline]` body (**kept**), final tree | `a3-polish1-b`, `a3-polish1` | **6,307,995,455 (+0.191)** | **1,138,113,466 (+1.296)** | 2,076,970,165 (-0.118) |
| guard + one `&dyn Fn` `#[inline(never)]` body | `a3-polish1-c` | 6,350,414,320 (+0.865) | 1,143,588,336 (+1.783) | 2,111,563,930 (+1.545) |

The brief's literal shape (an out-of-line body) measured **worse** than the tip, by 2.4 M Ir (+0.231 percent), and the single non-generic body hit an inlining
cliff far from the twin (`Saturation::saturate` +35 M inclusive: an iterator adapter in saturation's class-set walk, a `FlatMap` over `Enumerate<Iter<u64>>`, stopped being inlined, +118 M
outlined against -82 M in `saturate`; round 2 met the same kind of cliff around `add`). The kept form, with the body `#[inline]`,
equals close2 within 0.14 M. So the guard at the call site is worth nothing measurable: it was already what the compiler emitted. **The closing pass's remaining cost,
recorded and not removed:** +2,096,848 Ir over r3 (+0.033 percent of the total), in `extract_dag_over`'s own cost (+1.36 M) and `BinaryHeap::pop` (+0.88 M, which no
change of this round or the last touches), against `memset` -0.65 M; every one is code placement in the settle loops, since gates 1b, 2 and 2x are identical to base
(no pricing moved). It is inside the +0.5 percent limit with a factor of 2.6 to spare. A fold over literals still does no new work: it returns on its header in the
guard, before anything is read.

## Docs made false (findings #1/#10, #2/#11, #3, #4, #5, #9, #12, #16, #18). All applied; two corrected beyond the finding's wording

| finding | where | what is true now |
|---|---|---|
| #1, #10 (`graph.rs`, `admit_unit`) | the caller | verified: `runtime::saturate` is the only caller outside tests (`runtime.rs:734`) and it holds `job.leaves` (key to variance), no body. The doc says `saturate` admits the leaves `Program::of` built, and `Program::of` refused an open unit by `free_index` when it resolved the body |
| #2, #11 (`variance.rs`, `of_var`) | the paragraph | verified false against `Program::of` (its variance assertion reads a variance) and against placement. Now: **a variance never decides that a term is closed** (a closedness check asks `free_index`, which counts a placeholder free: `Kernel::by_ref`, and the unit walk beside its variance assertion); placement does read a variance, and only of a term from which a `Var` that is no fold's binder has been refused (codegen's `Var` arm) |
| #3 (`codegen/program/lower.rs`) | the panic | split by case on `Variance::VARIABLES`: a coordinate or retired axis (collapse was skipped, "built without the lowering pipeline") and an index past the binder slots ("a fold index that escaped its builder ... the kernel is open"); the doc of the function and the test's `expected` follow. `Binder::from_var`'s doc lists a `Placeholder` and an index past the space among its `None`s. **Not done:** the finding's aside that `pipeline::compile` could refuse `free_index(root)` by name: it adds a walk to every compile and the brief did not ask for it |
| #4 (`node.rs` `literal_twin`, `fold_rules.rs` `KnownBound`) | "cannot disagree", "are the same" | now what one construction buys: given the same literal for each bound class, the rule and the price build the same fold; which literal each finds is its own (a class member for the rule, the chosen form for the price), and so is whether it finds one |
| #5 (`fold_rules.rs`) | `representative`, the `substitute` summary, plan L7, "no builder makes one" | `representative` is gone (inlined, #13); the summary says "one member per e-class and site"; the module doc's bullet and L7 say "a class reached as a bound, or under one"; the justification is gone with the case (#14). **Verified in passing**: the finding's example (`PeelFold` unions `Σ_{k<1} i` with `i`) is not `PeelFold` alone: `RuleSet::runtime()` derives it and `PeelFold`/`EmptyFold` alone do not (`/home/user/scratch/polish1/peelprobe`); the docs say "the runtime rules derive" |
| #9 (`graph.rs`, `GrowthPredictor`) | "never iterated" | amended, not given a lookup: `pending` is iterated by one reader, `fact`, whose answer is order-independent because ids are unique, so the capacity is still unobservable; the cost scales with the capacity and is paid only for a fold's value bound |
| #12 (`fold_rules.rs` `substitute`, `HalveFold`; `graph.rs` `plan_shape`, `add`, `NodeSink::refuses`, `VaryingBound`; `rewrite.rs` `Plan`) | the decline condition | stated in full: the class fact of a value bound fails L3, whether the bound landed in a stale class or was minted over a child that did. The `(i + 1)·2` case is pinned (below) |
| #16 (`node.rs`) | "first test" | "a leaf and an operation answer on the first test and a fold over literals on the second" |
| #18 (`docs/language/2026-10-20-a-bound-is-a-value.md`, L7) | "building no fold and making no union" | verified false (`plan_shape` refuses a fold only when it reaches it; the reviewer's orphan probe, re-run: value-bounded inner folds 2 -> 3, classes 13 -> 14, the matched class gains no member). Now: it makes no union, and the nodes it made before that fold stay, as orphans that cost classes and never soundness (a lawful nested fold can be among them). The A3 bullet's pin list says "declined: the fold is not built" |

## Code (findings #6, #13, #14)

- `Site` derives `Clone, Copy` only (the memo keys on `(EClassId, Fold)`; nothing compared a `Site`).
- **`Fold::bound_may_read(self, reads) -> bool`** (`fold.rs`, `pub` because the e-graph is another crate, `#[must_use]`): L3 stated once and end-free; `admits_bound` is
  `match self.bound_may_read(reads)`; `Site::member` calls it, so no `BoundEnd::Hi` is fabricated. The test helpers that used `admits_bound` for a yes or no use it too,
  and `fold_rules.rs` no longer imports `BoundEnd`.
- **`Site::member`'s `Bound` arm skips an `ENode::Reduce`.** Verified: the decline it replaces lost a rewrite whenever a later member of the class kept L3, and saturation
  does put a fold into a bound's class. With the skip a fold under a bound is never chosen, so `Site::of_child` is total (it returns a `Site`, and `substitute` drops a `?`)
  and the old arm `(Site::Bound, _) => None` is gone with the case it covered. Soundness: the skip only removes candidates the walk would have refused, and the graph's
  refusal (`plan_shape`) is untouched.
- `representative` is inlined into `Site::member`'s body arm (one caller).

## Tests (findings #6, #15, #12, #19)

Each assertion below was checked by mutation (the mutants are not kept):

- `a_nested_bound_is_rebuilt_through_the_member_that_keeps_l3`: the loop asserting L3 on every bound in the graph cannot fail (`add`'s backstop makes it true or panics first). Replaced by:
  the graph took the plan (value-bounded inner folds 2; a plan through the varying spelling is declined and builds none), and the rebuilt bound classes hold no `Sub` member.
- `a_rebuilt_bound_that_lands_in_a_stale_class_is_declined`: the vacuous `num_classes() >= classes` is gone. Added: the control's rule unions something (`total_unions > 0`), the stale run
  unions **nothing** (`total_unions == 0`: the plan was declined), the stale graph holds fewer value-bounded inner folds than its control (1 against 2), and `P`'s fact still reads `X`.
  A third case, `HalveFold` with the bound `i * 2`, pins the bound the plan **mints** over a child that landed in `P` (`(i + 1) * 2`). Mutants: removing `plan_shape`'s refusal fails the test
  (at its first case) with the `BoundVaries` panic; removing it **and** `add`'s backstop fails `total_unions == 0` instead, so the new assertions are not carried by the panic.
- `the_runtime_rules_saturate_a_graph_with_a_stale_bound_class`: now asserts the landing (a halved copy's `i + 1` is `P`), the decline (fewer value-bounded inner folds than a control over the bare
  index, 2 against 11 / 18 / 36 at n = 2 / 4 / 8), that `P`'s fact still reads `X`, and that the extracted term holds an inner fold (its bound loop could not be vacuous). The doc's claim that the
  bound lands in `P` "at every level" is corrected: the halving is declined at its first level, whatever the extent. Same mutants, same results.
- New `a_fold_in_a_bound_class_is_skipped_for_the_next_member` (the skip: the plan holds two folds, the outer and the shifted copy, and none for the bound; without the skip it holds a third) and
  `extracting_a_bound_class_whose_cheapest_member_reads_a_coordinate_is_refused_by_name` (finding #19's optional second pin, `should_panic("BoundVaries")` beside the first).
- Search tests: 434 / 437 / 437 passed (default / `--all-features` / `--features saturation-telemetry`), two more than close2's 432 / 435 / 435.

## `Kernel::over` and the capture hole (finding #8). Applied; the finding is right and reproduced

`Kernel::over`'s doc no longer says an index cannot escape. **Reproduced** at HEAD with a scratch crate (`/home/user/scratch/polish1/capture`): a closure keeps a clone of its index; the fold closes and
releases its placeholder (`BinderScope`'s drop); `Kernel::sum_over(3, |j| j.add(&escaped))` takes the same placeholder, and `close_over` renames the escaped clone with it: the result is
`reduce[add_4[0..3)](add(Var(4), Var(4)))`, `free_index` is `None`, and nothing refuses. Before it is captured the escaped index is open (`free_index` is `Some(64)`) and caught where closedness is asked.
Pre-existing at `3bfdaded` (`BinderScope` and the sentence are both in the base), so it is tracked and not fixed: **`docs/BACKLOG.md` C10**, in the table's own format (id, what, where), says what closing it needs:
placeholders that are never reissued, which a `u8` index cannot supply for a process's lifetime, or an index identity outside `Var(u8)`.

## `.wip` only (findings #18's option, #19)

`spec.md`: J24 states the extraction hole generally (the cheapest member of a bound class reads a coordinate; a class whose lawful member costs more as well as one with no lawful member; the `dear` probe
reproduced: no rules, `BoundVaries`; `RuleSet::runtime()` rewrites that graph's bound away and extracts `6X`), J25 drops "no builder makes one", J27 says a declined plan leaves orphans, §1.1, §2.6 and §6.8
follow the code, J28 records the round. `gen/classes.json`: the `of_child` row follows its new text, rows added for `Site::member` and for the two sites of the new test. `run/public-surface.md`: `Fold::bound_may_read`
(new `pub`), the `HalveFold`/`PeelFold` row, `value_bounded_twin`, and a stale line corrected (`refuse_a_varying_bound` is the name in the code; the file said it was gone in favour of a `refusal` that does not exist).

## Gates after this round (TAG=a3-polish1), on the final tree

| gate | result |
|---|---|
| 1 `byteprobe.sh a3-polish1` | exit 0; avx512 and avx2 IDENTICAL to base |
| 1b `gate1b.sh a3-polish1` | exit 0; 29 lines IDENTICAL; self-check n=2 `22ad8819798461a6`, n=40 `e95137a9d464391e` equal the pins |
| 2 `gate2.sh a3-polish1` | exit 0; IDENTICAL to base (29 records) |
| 2x `gate2x.sh a3-polish1` | exit 0; IDENTICAL to base (extraction fields) |
| 5 `measure.sh a3-polish1` | total Ir 6,307,995,455 (+0.191 percent; limit +0.5: PASS; close2 was +0.193, r3 +0.158); `Saturation::saturate` inclusive 2,076,970,165 (-0.118); `extract_dag_over` inclusive 1,138,113,466 (+1.296; close2 +1.274, r3 +1.157); median peak RSS 97,768 KiB (+1.339; five runs 97,168 to 98,712, close2 +1.289; reported only); median wall 1100 ms (runs 1009 to 1247; uncontrolled, reported only) |
| `cargo test -p` ir / search / search `--all-features` / search `--features saturation-telemetry` / codegen / compiler | 375 / 434 / 437 / 437 / 161 / 304 passed, 0 failed (ignored 2 / 15 / 15 / 15 / 0 / 6); search is two tests above close2 (432 / 435 / 435); no compiler warning in any log (`tests-polish1.*.txt`) |
| `cargo +1.99.0 clippy -p pixelflow-ir -p pixelflow-search -p pixelflow-codegen --all-targets --all-features -- -D warnings` (CI's toolchain; `CARGO_TARGET_DIR=/home/user/scratch/a3-close1/t199`) | exit 0, no warning (`clippy-polish1-199.txt`) |
| `cargo +1.99.0 clippy --workspace --all-targets --all-features -- -D warnings`, same target dir | exit 0, no warning (`clippy-polish1-199-workspace.txt`) |
| `cargo fmt --all -- --check` | exit 0 |
| `RUSTFLAGS="-D warnings" cargo check -p pixelflow-{ir,search} --no-default-features` | exit 0, exit 0 |
| `RUSTDOCFLAGS="-D warnings" cargo doc -p pixelflow-codegen --no-deps --document-private-items` | exit 0 |
| rustdoc C4 (`rustdoc_norm.py --compare`) | pixelflow-ir base 38, head 38, NEW 0, GONE 0; pixelflow-search base 28, head 28, NEW 0, GONE 0 |
| `scripts/check-intent.sh`, `check-doc-paths.sh` (checks `docs/BACKLOG.md` C10's paths), `check-cfg-encapsulation.sh`, `check-bin-declarations.sh` | exit 0 each |
| `PATH=/home/user/scratch/cargo-hack-root/bin:$PATH scripts/check-feature-matrix.sh` | exit 0, "every crate's every feature builds in isolation" |
| checklist (`python3 -I .wip/a3/gen/checklist.py`) | 175 sites, 0 UNCLASSIFIED, 0 STALE (already-correct 38, changed-by-A3 26, left 13, tests 78, n/a 5, A4 15, A5 0) |
| B1 greps (`A3` in `pixelflow-*` `.rs` and `pixelflow-ir/intent.md`), `fold_cost(` only in `cost.rs`, no tracked file outside `.wip` citing `.wip`, no model name, `.wip` label or ruling label in the added tracked lines | empty / empty / only `cost.rs` / empty / empty |
| `git status --short pixelflow-graphics` after the gates; `git lfs ls-files \| grep -c .` | empty; 68 |

Outputs: `../../gates/out/*.a3-polish1*` (the throwaway measurements `a3-polish1-f3`, `-a`, `-b`, `-c` too); this directory's `*-polish1*` files. Scratch: `/home/user/scratch/polish1/` (`capture`: the
placeholder-capture repro; `peelprobe`: which rule set derives `Σ_{k<1} i = i`).

