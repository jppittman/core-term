# A6 gates: the implementation in the working tree

Tree: `/home/user/pixelflow`, branch `claude/integral-area-work-ojiej6`, HEAD `cbfda81` (base `29980bf` plus `.wip/` commits;
`git diff --quiet 29980bf HEAD -- . ':!.wip'` holds). The A6 implementation is uncommitted in the working tree (20 tracked files
modified). Spec `.wip/a6/spec.md` §8 step 7; rulings `.wip/a6/rulings.md` win (D-A6-1 = arm (b); G2 with S1/C2; G9 with C4).
Nothing was fixed, committed, stashed, reset or checked out. Gate scripts ran one at a time. Disk: `df -h /` showed 3.9 GB free
before every build (never under the 2 GB floor); sccache was up.

**Verdict: every gate passes. No failure to report.**

## Summary

| # | Gate | Command (abridged) | Exit | Result |
|---|---|---|---|---|
| 1 | byte probe, both tiers | `.wip/gates/byteprobe.sh a6-impl` | 0 | IDENTICAL to base, avx512 and avx2 (127 lines each) |
| 1b | keys and digests | `.wip/gates/gate1b.sh a6-impl` | 0 | IDENTICAL to `gate1b.base.txt` (29 lines); self-check digests equal |
| 2 | saturation | `.wip/gates/gate2.sh a6-impl` | 0 | IDENTICAL to `gate2.base.txt` (29 lines, 29 records) |
| 2x | extraction fields | `.wip/gates/gate2x.sh a6-impl` | 0 | IDENTICAL to `gate2x.base.txt` |
| 5 | measure | `.wip/gates/measure.sh a6-impl` | 0 | total Ir 6,317,091,356 vs base 6,317,091,756: **-400 Ir (-0.000006 percent)**; limit +0.5 percent |
| T1 | compiler tests | `PIXELFLOW_ISA=avx2 cargo test -p pixelflow-compiler` | 0 | 149 lib + 169 integration = **318 passed**, 0 failed, 6 doc-tests ignored |
| T2 | compiler tests | `PIXELFLOW_ISA=avx512 cargo test -p pixelflow-compiler` | 0 | the same 318 passed, 0 failed |
| T3 | ir tests | `cargo test -p pixelflow-ir` | 0 | 310 + 1 + 75 + 9 + 1 + 1 doc = 397 passed, 0 failed (1 + 1 ignored) |
| T4 | core tests | `cargo test -p pixelflow-core` | 0 | lib 68 + integration 68 = 136 passed, 0 failed (1 doc-test ignored) |
| T5 | graphics tests, avx2 | `PIXELFLOW_ISA=avx2 cargo test -p pixelflow-graphics` | 0 | 239 passed, 0 failed, 15 ignored |
| T6 | graphics tests, avx512 | `PIXELFLOW_ISA=avx512 cargo test -p pixelflow-graphics` | 0 | 239 passed, 0 failed, 15 ignored |
| C1 | check | `cargo check --workspace --all-targets` | 0 | `Finished dev profile`; the one warning line is core-term's build script ("Building with X11 display driver") |
| L1 | clippy 1.99.0 | `CARGO_TARGET_DIR=/home/user/scratch/a3-close1/t199 cargo +1.99.0 clippy --workspace --all-targets --all-features -- -D warnings` | 0 | `Finished`, no diagnostic |
| L2 | fmt | `cargo fmt --all -- --check` | 0 | clean |
| S1 | intent | `bash scripts/check-intent.sh` | 0 | OK: 51 intent files |
| S2 | doc paths | `bash scripts/check-doc-paths.sh` | 0 | OK: 692 source paths across 170 documents (1 baselined) |
| S3 | cfg | `bash scripts/check-cfg-encapsulation.sh` | 0 | OK: no new violation (121 grandfathered) |
| S4 | emit boundary | `bash scripts/check-emit-boundary.sh` | 0 | OK: self-test 66 cases, 25 files |
| S5 | bin declarations | `bash scripts/check-bin-declarations.sh` | 0 | OK: all 29 `src/bin` targets declared |
| S6 | feature matrix | `PATH=/home/user/scratch/cargo-hack-root/bin:$PATH bash scripts/check-feature-matrix.sh` | 0 | OK: every crate's every feature builds in isolation; `git status --short` shows no `Cargo.toml` (or `Cargo.lock`) changed |
| R1 | rustdoc, `pixelflow-compiler` | `cargo doc ... \| rustdoc_norm.py --compare` | 0 | `warnings: base 4, head 4`, NEW 0, GONE 0 |
| R2 | rustdoc, `pixelflow-core` | same | 0 | `warnings: base 5, head 5`, NEW 0, GONE 0 |
| R3 | rustdoc, `pixelflow-ir` | same | 0 | `warnings: base 38, head 38`, NEW 0, GONE 0 |
| G1 | grep: renamed and replaced | see below | - | nothing (expected nothing) |
| G2 | grep: step labels | see below | - | nothing with S1/C2's filter (expected nothing) |
| G3 | grep: "constant" bound wording | see below | - | only `tests/rustc_is_the_oracle.rs:324` (expected exactly that) |
| G4 | grep: two-item `usize` lists | see below | - | nothing (expected nothing) |
| G5 | grep: `KnownRange::admits` | see below | - | the four expected code sites, plus doc links and test strings |
| G6 | grep: production has no count | see below | - | `0` (expected 0) |
| G7 | grep: decision-2 placeholders | see below | - | nothing (expected nothing) |
| G8 | grep: `ArityMismatch` in emit.rs | see below | - | the one in `write_into`'s signature, `emit.rs:360` (expected, arm (b)) |
| G9 | grep: `BOUND_PLAN` and plan path | see below | - | as expected, with C4's list |
| G10 | grep: every `open_fold` | see below | - | the trait, `Expansion`, `Staged`, `lower_fold`'s call, each arena call |
| N1 | test count | `grep -c '#\[test\]'` vs `passed` | - | compiler 318 = 318; core 138 greps, 136 run, both explained; matches impl.md's table |
| D1 | diff stat | `git diff --stat 29980bf -- . ':!.wip'` | - | 20 files, all in the spec's list; nothing else |
| D2 | core public surface | `git diff 29980bf -- pixelflow-core/src` | - | empty (0 lines); `pixelflow-core/README.md` is the only core file changed |

## 1. The byte, key and saturation gates and the measure gate

All five scripts restore `pixelflow-graphics/src/fonts/loop_blinn/program/tests.rs` by trap; after the last,
`git status --short pixelflow-graphics` printed nothing. Outputs are `.wip/gates/out/*.a6-impl.*` (untracked, as for the other tags).

### Gate 1: `bash /home/user/pixelflow/.wip/gates/byteprobe.sh a6-impl`, exit 0

```
avx512: wrote .../byteprobe.a6-impl.avx512.txt (127 lines, sha256 11636fa51e4ea172...)
== avx512
IDENTICAL to byteprobe.base.avx512.txt
avx2: wrote .../byteprobe.a6-impl.avx2.txt (127 lines, sha256 4b65f46998718aa1...)
== avx2
IDENTICAL to byteprobe.base.avx2.txt
```

### Gate 1b: `bash /home/user/pixelflow/.wip/gates/gate1b.sh a6-impl`, exit 0

```
selfcheck n=2  probe=22ad8819798461a6  pinned=22ad8819798461a6  equal
selfcheck n=40  probe=e95137a9d464391e  pinned=e95137a9d464391e  equal
wrote .../gate1b.a6-impl.txt (29 lines, sha256 4101ba0bd24c1ae0...)
IDENTICAL to gate1b.base.txt
```

### Gate 2 then 2x: `gate2.sh a6-impl` exit 0, then `gate2x.sh a6-impl` exit 0 (run in turn, never together)

```
wrote .../gate2.a6-impl.txt (29 lines, sha256 1e3ab0b5ee9b7016...) and .../gate2.a6-impl.jsonl (29 records)
IDENTICAL to gate2.base.txt
wrote .../gate2x.a6-impl.txt (29 lines, sha256 29834e148fbac219...)
IDENTICAL to gate2x.base.txt
```

### Gate 5: `bash /home/user/pixelflow/.wip/gates/measure.sh a6-impl`, exit 0 (measured, never asserted)

| Figure | Base (`29980bf`, re-recorded) | `a6-impl` | Change |
|---|---|---|---|
| total Ir | 6,317,091,756 | 6,317,091,356 | -400 Ir, -0.000006 percent (script prints `-0.0`) |
| `Saturation::saturate` inclusive Ir | 2,076,022,575 | 2,076,022,575 | 0 |
| `extract_dag_over` inclusive Ir | 1,139,268,582 | 1,139,268,582 | 0 |
| median peak RSS (KiB) | 96,824 | 96,704 | -0.124 percent |
| median wall (ms) | 1,203 | 1,171 | -2.66 percent (VM drift, never decisive) |

The limit was +0.5 percent total Ir: met with 0.5 percent to spare. Runs: peak RSS 96704, 96400, 98456, 96488, 99020 KiB; wall
1176, 1068, 1228, 1141, 1171 ms. Gate 5's workload runs no expansion (spec §5.3), so an unchanged figure is what the spec predicts.

## 2. Tests

### `PIXELFLOW_ISA=avx2 cargo test -p pixelflow-compiler` and `PIXELFLOW_ISA=avx512 ...`, both exit 0, identical counts

| Target | passed | failed |
|---|---|---|
| lib (unit) | 149 | 0 |
| tests/bake_vs_scalar_truth | 5 | 0 |
| tests/binding_times | 12 | 0 |
| tests/derivative_under_warp | 6 | 0 |
| tests/fold_is_kernel_over | 16 | 0 |
| tests/items_block | 6 | 0 |
| tests/jit_parity | 12 | 0 |
| tests/kernel_5params | 5 | 0 |
| tests/kernel_composition | 6 | 0 |
| tests/kernel_macro | 30 | 0 |
| tests/kernel_typed_parameters | 17 | 0 |
| tests/optimized_agrees_with_raw | 11 | 0 |
| tests/rustc_is_the_oracle | 34 | 0 |
| tests/the_library_is_the_builders | 9 | 0 |
| doc-tests | 0 (6 ignored) | 0 |
| **total** | **149 + 169 = 318** | **0** |

### `cargo test -p pixelflow-ir`, exit 0

lib 310 passed; `dag_scratch_allocation` 1; `dyadic_spec` 75 (1 ignored); `value_bounds` 9; `variance_empty_fold` 1; doc-tests 1
passed (1 ignored). 0 failed.

### `cargo test -p pixelflow-core`, exit 0

lib 68; `bind_allocates_nothing` 1; `fold_factoring` 7; `guard_fold_price` 13; `guard_parked_reads` 16; `guard_sibling_fold` 16;
`kernel_bake` 3; `kernel_by_reference` 6; `reduce_binder_reads_bound_buffer` 6; doc-tests 0 passed (1 ignored). 136 passed, 0 failed.

### `cargo test -p pixelflow-graphics` on each ISA, both exit 0

`PIXELFLOW_ISA=avx2`: lib `121 passed; 0 failed; 3 ignored` (108 s), every integration target `ok`, doc-tests `0 passed; 11 ignored`;
sum over the run 239 passed, 0 failed, 15 ignored. `PIXELFLOW_ISA=avx512`: the same 239 / 0 / 15.

### `cargo check --workspace --all-targets`, exit 0

```
Checking pixelflow-graphics v0.1.0
warning: core-term@0.1.0: Building with X11 display driver
Finished `dev` profile [unoptimized + debuginfo] target(s) in 3.44s
```

(The `Finished` time is short because the units were fresh from the implementer's own run of the same sources; the command ended
exit 0 with the one build-script line and no compiler warning.)

## 3. Clippy 1.99.0 and fmt

```
CARGO_TARGET_DIR=/home/user/scratch/a3-close1/t199 cargo +1.99.0 clippy --workspace --all-targets --all-features -- -D warnings
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 47.11s      exit=0
```

The log holds no `warning:` or `error` line other than core-term's build-script note; the `saturation-telemetry(macro): {...}`
lines are the `--all-features` macro expansions printing their records. `cargo fmt --all -- --check`: exit 0, no output.

## 4. Scripts

| Script | Exit | Last line |
|---|---|---|
| `scripts/check-intent.sh` | 0 | `OK: 51 intent files present, shaped to the template, and every path they name resolves` |
| `scripts/check-doc-paths.sh` | 0 | `OK: 692 source paths cited across 170 plans, designs, notes and instructions resolve (1 accepted via scripts/doc-paths-baseline.txt)`; stderr also prints `head: cannot open '../AGENTS.md'` and the same for `CLAUDE.md` and `GEMINI.md`: noise from the script, exit 0, nothing to do with A6 |
| `scripts/check-cfg-encapsulation.sh` | 0 | `OK: no new platform cfg encapsulation violations (121 pre-existing, grandfathered)` |
| `scripts/check-emit-boundary.sh` | 0 | `OK: emit/ and program/ boundaries hold (25 files)` after `self-test OK: 66 cases` |
| `scripts/check-bin-declarations.sh` | 0 | `OK: all 29 src/bin targets are declared in their Cargo.toml` |
| `check-feature-matrix.sh` (cargo-hack on `PATH`) | 0 | `OK: every crate's every feature builds in isolation` |

After the feature matrix: `git status --short | grep -i cargo` printed nothing, and `Cargo.lock` is clean. The matrix ran under a
disk watchdog (kill under 2 GB); it never fired (3.8 GB free at the end).

## 5. Rustdoc against the bases

`cargo doc -p <crate> --no-deps --document-private-items 2>&1 | python3 -I /home/user/pixelflow/.wip/a6/rustdoc_norm.py --compare /home/user/pixelflow/.wip/a6/rustdoc-base.<crate>.txt`

| Crate | cargo doc exit | compare exit | Output |
|---|---|---|---|
| `pixelflow-compiler` | 0 | 0 | `warnings: base 4, head 4` / `NEW 0` / `GONE 0` |
| `pixelflow-core` | 0 | 0 | `warnings: base 5, head 5` / `NEW 0` / `GONE 0` |
| `pixelflow-ir` | 0 | 0 | `warnings: base 38, head 38` / `NEW 0` / `GONE 0` |

## 6. Greps (from `/home/user/pixelflow`)

**G1**, `grep -rnE 'Binding::Index|\bBinding<.*Index|parameter\.record|scalar\.(param|field)|Scalar \{' pixelflow-compiler/`: no output
(rc 1). Expected: nothing. PASS.

**G2**, `grep -rnwE 'A6' --include=*.rs --include=intent.md . | grep -v '^./target' | grep -v '^./.wip/' | grep -v 0xA6 | grep -v guide_scope_saturation_delta.rs`
(S1/C2): no output. Expected: nothing. PASS. Without the S1 filter the one hit is
`./pixelflow-pipeline/src/bin/guide_scope_saturation_delta.rs:11: //! chess-engine ~98% (`docs/archive/plans/2026-08-17-cost-model-domain.md`, A6).`,
a citation of an unrelated archived plan, which is why S1 filters it. The base's hit, `pixelflow-ir/src/fold.rs:441`, is gone.

**G3**, `grep -rniE "ranges are constant|bounds are constant|over a constant range|iterates a constant range|over constant ranges" pixelflow-compiler/`:

```
pixelflow-compiler/tests/rustc_is_the_oracle.rs:324:// ───────────────────── B2: folds over constant ranges ─────────────────────
```

Expected: only that line. PASS.

**G4**, ``grep -rn "index or a \`usize\`" pixelflow-compiler/``: no output (rc 1). Expected: nothing. PASS.

**G5**, `grep -rn 'KnownRange::admits\|KnownRange :: admits' pixelflow-compiler/src`:

```
emit.rs:225   ::pixelflow_core::__macro::ir::KnownRange::admits(&(0..#count as u64)),        (the host function's refusal)
emit.rs:322   ::pixelflow_core::__macro::ir::KnownRange::admits(&(0..self.#count as u64)),   (write_into's refusal)
emit.rs:1054, 1066   the emit unit test's assertions on those two token strings
lower.rs:243  Some(lo..hi).filter(KnownRange::admits)        (literal_range)
lower.rs:258  .filter(|&end| KnownRange::admits(&(0..end)))  (literal_bound)
lower.rs:23, 235, 253, 703   doc links
```

Expected: `lower.rs` `literal_range` and `literal_bound`, `emit.rs` the host refusal and the `write_into` refusal. PASS.

**G6**, `sed -n 31,141p pixelflow-graphics/src/fonts/loop_blinn/program.rs | grep -cE 'usize|\.map\(|\.any\(|\.all\('`: `0`. Expected: 0. PASS.

**G7**, `grep -rnE "decision 2 picks|Waits on decision 2|which A6 builds|refused at its write in A6" docs .claude CLAUDE.md | grep -v 'docs/archive/'`: no
output (rc 1). Expected: nothing. PASS.

**G8**, `grep -n 'ArityMismatch' pixelflow-compiler/src/emit.rs`:

```
360:            ) -> ::core::result::Result<(), ::pixelflow_core::ArityMismatch> {
```

Expected under (b): the one `ArityMismatch` in the signature. PASS.

**G9**, `grep -rn 'BOUND_PLAN' pixelflow-compiler/src` finds: `sema.rs:79` (import), `:698` (M7), `:709` (M6), `:1021` (M2), `:1770` (M1), `:1787` (M3),
`:1989` (M4), `:2029` (M5); `emit.rs:57` (import), `:233` (the `# Panics` text), `:277` (`count_refusal`), `:330` (`write_into`'s `# Panics`);
`lib.rs:77` (the definition); `parser.rs:203` (import), `:660` (`generic_refusal`); `lower.rs:68` (import), `:719` (decision 8).
`grep -rn 'docs/language/2026-10-20' pixelflow-compiler/src` finds `lib.rs:77` (the definition), `lib.rs:289` (the `# Binding times` line),
`parser.rs:112` (the module doc), and test assertions only (`sema.rs:2879, 2880, 3047, 3071, 3286, 3293`; `emit.rs:1056`; `parser.rs:2758`).
Expected, with C4: exactly those. PASS.

**G10**, `grep -rn 'fn open_fold\|\.open_fold(' pixelflow-compiler/src`:

```
emit.rs:788   fn open_fold(&mut self, depth: usize, monoid: Monoid, bounds: [Bound<Ident>; 2]) -> Ident     (Staged)
emit.rs:793   .open_fold(#depth)                         (the arena call Staged emits into the expansion)
lower.rs:183  fn open_fold(&mut self, depth, monoid, bounds: [Bound<Self::Term>; 2]) -> Self::Term       (the trait)
lower.rs:362  fn open_fold(&mut self, depth: usize, monoid: Monoid, [lo, hi]: [Bound<ExprId>; 2]) -> ExprId   (Expansion)
lower.rs:365  .open_fold(depth)                          (Expansion's arena call; then pushes FoldTerms { monoid, lo, hi })
lower.rs:664  .open_fold(self.open_folds, monoid(fold.reduction), bounds)                                (lower_fold's call)
```

Read each: the trait and both sites take `[Bound<Term>; 2]`; both resolve the bounds before the arena's `open_fold(depth)`, which takes none.
PASS.

## 7. Test count (spec §6.7)

`grep -c '#\[test\]' pixelflow-compiler/src/*.rs pixelflow-compiler/tests/*.rs` sums to **318** (src: ast 4, emit 8, lib 4, lower 20,
parser 43, sema 63, symbol 7; tests: bake_vs_scalar_truth 5, binding_times 12, derivative_under_warp 6, fold_is_kernel_over 16,
items_block 6, jit_parity 12, kernel_5params 5, kernel_composition 6, kernel_macro 30, kernel_typed_parameters 17,
optimized_agrees_with_raw 11, rustc_is_the_oracle 34, the_library_is_the_builders 9). At `29980bf` the same count is 299. Summed
`passed`: 149 + 169 = **318**, on each tier. The per-file figures equal impl.md's table row by row (sema 63, lower 20, emit 8, oracle 34,
binding_times 12, fold_is_kernel_over 16, kernel_typed_parameters 17, optimized_agrees_with_raw 11; the other twelve files 137;
the eight unchanged test files 79).

`pixelflow-core`: 138 `#[test]` lines at the base and now (0 added, arm (b)); 136 run. The two that do not run are not defects:
`tests/units_past_the_class_limit.rs` declares `required-features = ["saturation-telemetry"]` in `Cargo.toml`, and
`tests/bind_allocates_nothing.rs:87` is the text `#[test]` inside a doc comment.

## 8. Diff

`git diff --stat 29980bf -- . ':!.wip'`: 20 files changed, 1626 insertions(+), 384 deletions(-).

```
.claude/skills/ontology/language.md                | 35     docs/BACKLOG.md                                | 6
docs/language/2026-09-25-the-language-is-kernel.md | 11     docs/language/2026-10-20-a-bound-is-a-value.md | 59
pixelflow-compiler/intent.md                       | 12     pixelflow-core/README.md                       | 10
pixelflow-compiler/src/{ast 21, emit 204, lib 64, lower 282, parser 64, sema 798, symbol 8}.rs
pixelflow-compiler/tests/{binding_times 131, fold_is_kernel_over 64, items_block 22, kernel_typed_parameters 27,
                          optimized_agrees_with_raw 33, rustc_is_the_oracle 154}.rs
pixelflow-ir/src/fold.rs                           | 5
```

Every file is in the spec's list (§2.6, §3-§4, §6.1-§6.4, §7.1-§7.8). Not touched, as the spec says: `CLAUDE.md`, `docs/DENOTATION.md`,
`pixelflow-core/intent.md`, `.claude/skills/ontology/SKILL.md`, any `pixelflow-core/src`, `-search`, `-codegen`, `Cargo.toml`, `Cargo.lock`.
`git diff 29980bf -- pixelflow-core/src` is **empty (0 lines)**: no public item added, as D-A6-1 arm (b) requires.
`pixelflow-ir/src/fold.rs` is a doc-comment edit only (5 lines). Every added item in `pixelflow-compiler/src` that is `pub` is
`pub(crate)` (`Parameter`, `ParameterKind`, `Scalar`, `Count`, `bound_names`, `range_bounds`) except the `pub kind` field of the
`pub(crate) struct Parameter`; `lib.rs`'s exported items are still the two `#[proc_macro]` functions.

## Observations for the coordinator (none is a gate failure)

- `.wip/a6/public-surface.md`, which spec §8 step 7 asks the final pass to write, does not exist yet; impl.md records it as not done.
- The plan's Done line still ends `GATES: pending.` (impl-docs.md); the measured clause belongs to the coordinator. Figures above:
  gates 1, 1b, 2 and 2x identical to base; total Ir -400 of 6,317,091,756.
- The untracked `.wip/gates/out/*.a6-impl.*` files are this run's outputs; the tracked tree is exactly the 20 modified files listed in §8.
