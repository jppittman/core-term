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
