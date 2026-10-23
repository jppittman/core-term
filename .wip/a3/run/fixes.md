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
