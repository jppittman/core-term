# A5 implementation gates (spec 7, step 6)

Tree: base `7e9531e` plus the `.wip/` commits (HEAD `3e5570f`) plus the uncommitted A5 working tree
(24 files, below). Branch `claude/integral-area-work-ojiej6`. Run from `/home/user/pixelflow`. Nothing was edited,
committed, stashed, reset or deleted; the only files written are this one, the gate outputs in `.wip/gates/out/`
(tag `a5-impl`) and logs in the session scratchpad. `df -h /` before every build: 5.6 GB free throughout.
Host: AVX-512 capable; the default ISA is AVX-512, `PIXELFLOW_ISA` set explicitly where the task names a tier.

## Summary

| # | Gate | Result | Detail |
|---|---|---|---|
| 1 | `byteprobe.sh a5-impl` | PASS (exit 0) | avx512 and avx2 each 127 lines, IDENTICAL to base (sha256 `11636fa51e4ea172`, `4b65f46998718aa1`) |
| 1 | `gate1b.sh a5-impl` | PASS (exit 0) | 29 `ZZGATE` lines IDENTICAL to base (`4101ba0bd24c1ae0`); self-check n=2, n=40 digests equal the suite's pins |
| 1 | `gate2.sh a5-impl` | PASS (exit 0) | 29 records IDENTICAL to base (`1e3ab0b5ee9b7016`) |
| 1 | `gate2x.sh a5-impl` | PASS (exit 0) | extraction fields IDENTICAL to base (`29834e148fbac219`) |
| 1 | `measure.sh a5-impl` | PASS (limit +0.5 percent) | total Ir 6,314,563,169 vs base 6,315,838,727: **-0.020 percent**. Reported, see "Gate 5" for the two inclusive figures |
| 2 | `PIXELFLOW_ISA=avx2 cargo test -p pixelflow-compiler` | PASS | 14 targets ok, 298 passed, 0 failed, doc-tests 6 ignored |
| 2 | `PIXELFLOW_ISA=avx512 cargo test -p pixelflow-compiler` | PASS | identical counts, 298 passed, 0 failed |
| 2 | `cargo test -p pixelflow-ir` | PASS | 6 targets: 310, 1, 75 (1 ignored), 9, 1, 1 (1 ignored); 0 failed |
| 2 | `cargo test -p pixelflow-core` | PASS | 10 targets: 68, 1, 7, 13, 16, 16, 3, 6, 6, 0 (1 ignored); 0 failed |
| 2 | `cargo test -p pixelflow-graphics` (whole crate), avx2 | PASS | 23 targets (incl. doc-tests), 239 passed, 0 failed, 15 ignored |
| 2 | `cargo test -p pixelflow-graphics` (whole crate), avx512 | PASS | 23 targets, 239 passed, 0 failed, 15 ignored |
| 3 | `cargo +1.99.0 clippy --workspace --all-targets --all-features -- -D warnings` (target dir `a3-close1/t199`) | PASS (exit 0) | `Finished dev profile`; the only `warning:` line is core-term's build-script note "Building with X11 display driver" (not a lint) |
| 3 | `cargo fmt --all -- --check` | PASS (exit 0) | no output |
| 4 | `scripts/check-intent.sh` | PASS | OK: 51 intent files present, shaped to the template, every path resolves |
| 4 | `scripts/check-doc-paths.sh` | PASS | OK: 682 source paths cited across 170 documents resolve (1 accepted via baseline). The `head:`/`grep: ../CLAUDE.md` lines are the script's own relative-path noise (script unchanged) |
| 4 | `scripts/check-cfg-encapsulation.sh` | PASS | OK: no new violations (121 grandfathered) |
| 4 | `scripts/check-emit-boundary.sh` | PASS | self-test OK: 66 cases; OK: emit/ and program/ boundaries hold (25 files) |
| 4 | `scripts/check-bin-declarations.sh` | PASS | OK: all 29 src/bin targets are declared |
| 4 | `check-feature-matrix.sh` (cargo-hack on PATH) | PASS (exit 0) | OK: every crate's every feature builds in isolation. `git status --short` afterwards shows **no Cargo.toml / Cargo.lock changed** |
| 5 | rustdoc `pixelflow-compiler` vs `.wip/a5/rustdoc-base.pixelflow-compiler.txt` | PASS | warnings: base 4, head 4; NEW 0; GONE 0 |
| 5 | rustdoc `pixelflow-ir` vs A4 base | PASS | warnings: base 38, head 38; NEW 0; GONE 0 |
| 5 | rustdoc `pixelflow-codegen` vs A4 base | PASS | warnings: base 0, head 0; NEW 0; GONE 0 |
| 5 | `RUSTDOCFLAGS="-D warnings" cargo doc -p pixelflow-codegen --no-deps --document-private-items` | PASS (exit 0) | `Finished`, `Generated .../pixelflow_codegen/index.html` |
| 6 | G1 | PASS | hits only in the bound plan (below) |
| 6 | G2 | PASS | nothing |
| 6 | G3 | PASS | only `pixelflow-compiler/src/lib.rs:356` ("the template" vocabulary of `macro_tier`) |
| 6 | G4 | PASS | only `src/emit.rs:440,442`, the `node` panic arm |
| 6 | G5 | PASS | nothing |
| 6 | G6 | PASS | nothing from either grep |
| 6 | G7 | PASS | every `FoldTerms` builder is `monoid`, `lo`, `hi`; none has `stride` |
| 6 | G8 (R2 pipe) | PASS | only the parser's new test, `parser.rs:2709-2713, 2741, 2743` |
| 6 | G9 | PASS | nothing |
| 6 | G10 (R3 files) | PASS | hits only in the bound plan, the-language-is-kernel, ontology `language.md` and `SKILL.md` (below) |
| 7 | test count | PASS | `grep -c '#\[test\]'` sums to **298** (base 306); summed `passed` of `cargo test -p pixelflow-compiler` is **298** on both tiers; per-file counts equal impl.md's table |
| 8 | `git diff --stat 7e9531e -- . ':!.wip'` | PASS | 24 files, 537 insertions, 1261 deletions; every file is on the spec's list; no Cargo file, no untracked file outside `.wip/` |

Every gate passed; none was stopped.

## Gate 1, 1b, 2, 2x, 5 (the base is the re-recorded one at `7e9531e`)

```
$ bash .wip/gates/byteprobe.sh a5-impl                      -> exit 0
avx512: wrote .../byteprobe.a5-impl.avx512.txt (127 lines, sha256 11636fa51e4ea172...)
== avx512 IDENTICAL to byteprobe.base.avx512.txt
avx2: wrote .../byteprobe.a5-impl.avx2.txt (127 lines, sha256 4b65f46998718aa1...)
== avx2 IDENTICAL to byteprobe.base.avx2.txt

$ bash .wip/gates/gate1b.sh a5-impl                         -> exit 0
selfcheck n=2  probe=22ad8819798461a6  pinned=22ad8819798461a6  equal
selfcheck n=40  probe=e95137a9d464391e  pinned=e95137a9d464391e  equal
wrote .../gate1b.a5-impl.txt (29 lines, sha256 4101ba0bd24c1ae0...)
IDENTICAL to gate1b.base.txt

$ bash .wip/gates/gate2.sh a5-impl && bash .wip/gates/gate2x.sh a5-impl   -> exit 0 and exit 0
wrote .../gate2.a5-impl.txt (29 lines, sha256 1e3ab0b5ee9b7016...) and .../gate2.a5-impl.jsonl (29 records)
IDENTICAL to gate2.base.txt
wrote .../gate2x.a5-impl.txt (29 lines, sha256 29834e148fbac219...)
IDENTICAL to gate2x.base.txt
```

`sha256sum` of each `a5-impl` output equals its `base` file's (checked separately of the scripts' own `cmp`). After each
build `git status` showed `program/tests.rs` unmodified (probe restored).

### Gate 5 (measured, never asserted)

```
$ bash .wip/gates/measure.sh a5-impl                        -> exit 0
total_ir                6,314,563,169   (base 6,315,838,727)   -0.020 percent
saturate_inclusive_ir   2,076,382,641   (base 2,075,982,919)   +0.019 percent
extraction_inclusive_ir 1,137,377,703   (base 1,139,023,257)   -0.144 percent
median_peak_rss_kib     97,464          (base 97,592)          -0.131 percent
median_wall_ms          1,116           (base 1,000)           +11.6 percent  (VM drift, never decisive; runs 1084 1290 1067 1116 1243)
```

Total Ir is far inside the +0.5 percent limit. One point for the reader, not a failure: the spec's 4.2 expected the
`Saturation::saturate` and `extract_dag_over` inclusive figures to be identical to an unmodified-tree run, and they are not
(+0.019 and -0.144 percent). The work they do is identical (gates 1b, 2 and 2x are byte-equal, including the
`extraction_objective`, `extracted_latency_prior_cost` and `shared_pass_bytes` of all 29 records), so the difference is the
binary's (the compiler and IR crates changed, and the callgrind binary is rebuilt), not the saturation. No `a5-pre` run
exists in `out/` to separate it from the base's own build; the comparison here is against the recorded base, as asked.

## Tests (step 2)

```
$ PIXELFLOW_ISA=avx2 cargo test -p pixelflow-compiler        -> exit 0
  lib 142; bake_vs_scalar_truth 5; binding_times 7; derivative_under_warp 6; fold_is_kernel_over 14; items_block 6;
  jit_parity 12; kernel_5params 5; kernel_composition 6; kernel_macro 30; kernel_typed_parameters 16;
  optimized_agrees_with_raw 10; rustc_is_the_oracle 30; the_library_is_the_builders 9  = 298 passed, 0 failed
  Doc-tests: 0 passed; 0 failed; 6 ignored
$ PIXELFLOW_ISA=avx512 cargo test -p pixelflow-compiler      -> exit 0, the same 14 counts, 298 passed, 0 failed
$ cargo test -p pixelflow-ir                                 -> exit 0
  310 / 1 / 75 (1 ignored) / 9 / 1 / 1 (1 ignored) passed; 0 failed
$ cargo test -p pixelflow-core                               -> exit 0
  68 / 1 / 7 / 13 / 16 / 16 / 3 / 6 / 6 / 0 (1 ignored) passed; 0 failed
$ PIXELFLOW_ISA=avx2   cargo test -p pixelflow-graphics      -> exit 0   (3m36s)
$ PIXELFLOW_ISA=avx512 cargo test -p pixelflow-graphics      -> exit 0   (3m14s)
  both: 23 targets (the lib, 21 integration targets, the doc-tests), 239 passed, 0 failed, 15 ignored
```

## Clippy and fmt (step 3)

```
$ CARGO_TARGET_DIR=/home/user/scratch/a3-close1/t199 cargo +1.99.0 clippy --workspace --all-targets --all-features -- -D warnings
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 1m 01s      -> exit 0
$ cargo fmt --all -- --check                                                  -> exit 0, no output
```

## Scripts (step 4)

```
$ bash scripts/check-intent.sh               -> 0  OK: 51 intent files present, shaped to the template, and every path they name resolves
$ bash scripts/check-doc-paths.sh            -> 0  OK: 682 source paths cited across 170 plans, designs, notes and instructions resolve (1 accepted via scripts/doc-paths-baseline.txt)
$ bash scripts/check-cfg-encapsulation.sh    -> 0  OK: no new platform cfg encapsulation violations (121 pre-existing, grandfathered)
$ bash scripts/check-emit-boundary.sh        -> 0  self-test OK: 66 cases / OK: emit/ and program/ boundaries hold (25 files)
$ bash scripts/check-bin-declarations.sh     -> 0  OK: all 29 src/bin targets are declared in their Cargo.toml
$ PATH=/home/user/scratch/cargo-hack-root/bin:$PATH bash scripts/check-feature-matrix.sh   -> 0
    OK: every crate's every feature builds in isolation        (1m51s)
$ git status --short | grep -iE 'Cargo\.(toml|lock)'          -> (empty) before and after
```

## rustdoc (step 5)

```
$ cargo doc -p pixelflow-compiler --no-deps --document-private-items 2>&1 | python3 -I .wip/a5/rustdoc_norm.py --compare .wip/a5/rustdoc-base.pixelflow-compiler.txt
    warnings: base 4, head 4 / NEW 0 / GONE 0                  -> exit 0
    (the 3 distinct: emit.rs "redundant explicit link target" x1, emit.rs "unresolved link to `pixelflow_core::Kernel`" x2, lib.rs the same x1)
$ ... -p pixelflow-ir      --compare /home/user/scratch/a4-wip-archive/.wip/a4/rustdoc-base.pixelflow-ir.txt
    warnings: base 38, head 38 / NEW 0 / GONE 0                -> exit 0
$ ... -p pixelflow-codegen --compare /home/user/scratch/a4-wip-archive/.wip/a4/rustdoc-base.pixelflow-codegen.txt
    warnings: base 0, head 0 / NEW 0 / GONE 0                  -> exit 0
$ RUSTDOCFLAGS="-D warnings" cargo doc -p pixelflow-codegen --no-deps --document-private-items   -> exit 0
```

No NEW line, so no judgement was needed on whether a warning came from an edit. GONE is 0 (no intra-doc link to a deleted
name was left in a documented item, none of the base warnings pointed at one).

## Greps G1-G10 (step 6), run exactly as the spec gives them, with R2 for G8 and R3 for G10

**G1** (expected: only the bound plan, at its sections 0, 2's row, the rewritten 3.4 sentence, and the A2a bullet). Hits, all in
`docs/language/2026-10-20-a-bound-is-a-value.md`:

```
:31, :32   section 0 ("`StructuralRange` in ...sema.rs", "`instantiated_range` in ...emit.rs")
:231       section 2's row ("HOLE_STRIDE")
:417, :418 the rewritten 3.4 sentence (Count::Structural, StructuralRange, instantiated_range, FoldRange, Holes, OPEN_RANGE, HOLE_STRIDE, ExpansionFold)
:574, :581 the A2a bullet ("Holes::range_of", "the compiler's `Holes` tests"; spec lines :578/:585 moved with the edits above)
```

No code file, no other doc. **PASS.**

**G2** `grep -rni 'structural' pixelflow-compiler/`: no hit (grep status 1). **PASS.**

**G3** `grep -rniE 'template|instantiat|\bholes?\b' pixelflow-compiler/`: one hit, the expected one:
`pixelflow-compiler/src/lib.rs:356:/// The macro tier's optimizer: equality saturation over the template`. **PASS.**

**G4** `grep -rnE 'push_param|ExprNode::Param' pixelflow-compiler/`: `src/emit.rs:440` (`ExprNode::Param(k) => {`) and `:442`
(the panic text, R1's: "a `Param` is a rewrite rule's metavariable, substituted before any arena reaches emission; lowering
builds none"). Only the `node` panic arm. **PASS.**

**G5** step label in `.rs` and `intent.md`: no hit. **PASS.**

**G6** `grep -rn 'stride' pixelflow-compiler/`: no hit; `grep -rn 'stride:' pixelflow-ir/src/kernel.rs`: no hit. **PASS.**

**G7** `grep -rn 'FoldTerms' --include=*.rs .`, each builder read:

```
pixelflow-ir/src/arena.rs:177      the struct: monoid, lo, hi (no stride)
pixelflow-ir/src/arena.rs:1743     close_over destructure `FoldTerms { monoid, lo, hi }`
pixelflow-ir/src/arena.rs:2449     test helper `terms`: FoldTerms { monoid, lo, hi }
pixelflow-ir/src/kernel.rs:765     Kernel::over:        monoid, lo: Bound::Lit, hi: Bound::Lit
pixelflow-ir/src/kernel.rs:863     Kernel::over_values: monoid, lo: Bound::Value, hi: Bound::Value
pixelflow-ir/src/kernel.rs:1684    test:                monoid: Monoid::SUM, lo: Lit(0), hi: Lit(3)
pixelflow-compiler/src/lower.rs:340  Expansion::open_fold: monoid, lo: Lit(range.start), hi: Lit(range.end)
pixelflow-compiler/src/emit.rs:616   Staged::fold_terms tokens: monoid, lo: Bound::Lit(#lo), hi: Bound::Lit(#hi)
(the rest are imports, type annotations and docs: arena.rs:1741,1829; kernel.rs:22; lower.rs:66,254; emit.rs:564,602)
```

No `stride` in any. **PASS.**

**G8** (R2: `| grep -v 'as_chunks::<'`): six hits, all in the parser's new test
`generics_are_refused_and_a_const_generic_says_what_a_count_is`:

```
pixelflow-compiler/src/parser.rs:2709  pub fn f<const N: usize>() -> f32 { X }
:2710  pub fn f<const N: usize, const M: usize>(r: f32) -> f32 { r }
:2712  pub fn f<const N: usize = 4>() -> f32 { X }
:2713  fn h<const N: usize>(x: f32) -> f32 { x } pub fn f() -> f32 { h(X) }
:2741  fn h(x: f32) -> f32 { x } pub fn f() -> f32 { h::<4>(X) }                       (R6)
:2743  pub fn f(k: impl Fn(f32, f32) -> f32) -> f32 { k::<4>(X, Y) }                  (R6)
```

(`pub fn f<const N: u32>` at `:2711` and `DX::<f32>` do not match the pattern.) Nothing in pixelflow-graphics or
pixelflow-core. **PASS.**

**G9** `structural param|structural hole|structural count` in `.rs` and `intent.md`: no hit. **PASS.**

**G10** (R3: expected files are the-language-is-kernel, the bound plan, ontology `SKILL.md` and `language.md`). Hits, each read:

```
docs/language/2026-10-20-a-bound-is-a-value.md  :417 (rewritten 3.4: "went with `<const N: usize>`"), :434 (4 "Deleted"),
    :501 (5 row quoting the ontology heading), :693 (6 A5 bullet)                         -> the plan's own text
docs/language/2026-09-25-the-language-is-kernel.md  :58 (the new banner sentence), :163 (1.1 binding-time bullet),
    :298 (1.5 "Ranges are constant"), :329 (1.6), :499 (1.7)                              -> history under that document's
    "Reversed in part" banner, each section named in the banner sentence
.claude/skills/ontology/language.md:149  "### Structural parameter (retired)"             -> the ontology's retired entry
.claude/skills/ontology/SKILL.md:59      "- Structural parameter (retired)"              -> the same
```

`compiler.md`, `terminal.md`, `CLAUDE.md`, BACKLOG: no hit. **PASS.**

## Test count (step 7)

`grep -c '#\[test\]' pixelflow-compiler/src/*.rs pixelflow-compiler/tests/*.rs`:

```
src: ast 4, emit 7, lib 4, lower 18, parser 43, sema 59, symbol 7
tests: bake_vs_scalar_truth 5, binding_times 7, derivative_under_warp 6, fold_is_kernel_over 14, items_block 6, jit_parity 12,
       kernel_5params 5, kernel_composition 6, kernel_macro 30, kernel_typed_parameters 16, optimized_agrees_with_raw 10,
       rustc_is_the_oracle 30, the_library_is_the_builders 9
sum = 298     (the same count at 7e9531e = 306)
```

Summed `passed` of `cargo test -p pixelflow-compiler`: 298 (avx2) and 298 (avx512), plus 6 ignored doc-tests. impl.md's table
says 306 -> 298 with parser 43, sema 59, lower 18, emit 7, binding_times 7 and 164 elsewhere (ast 4 + lib 4 + symbol 7 + the
other 149): all of it reproduces. A name-level diff of `#[test]` functions against `7e9531e` shows exactly the 9 removals
and 4 renames/1 addition the table lists (no other test added, removed or renamed).

## Changed files (step 8)

`git diff --stat 7e9531e -- . ':!.wip'`: 24 files changed, 537 insertions(+), 1261 deletions(-). Every one is on the spec's list:

```
pixelflow-ir/src/{arena,fold,kernel}.rs                                      (2.7)
pixelflow-compiler/src/{ast,parser,symbol,sema,lower,emit,lib}.rs            (2.1-2.6)
pixelflow-compiler/tests/{binding_times,items_block,kernel_typed_parameters,rustc_is_the_oracle}.rs   (5.1, 5.2)
pixelflow-compiler/intent.md                                                 (6.4)
pixelflow-codegen/src/program/lower.rs, pixelflow-graphics/src/fonts/loop_blinn/program.rs   (2.8)
docs/BACKLOG.md, docs/language/2026-09-25-the-language-is-kernel.md, docs/language/2026-10-20-a-bound-is-a-value.md   (6.1, 6.2, 6.8)
.claude/skills/ontology/{SKILL,compiler,language,terminal}.md                (6.6)
```

No Cargo.toml or Cargo.lock, no script, no CI file, no file outside this list. `git ls-files --others` outside `.wip/` is empty.
Under `.wip/` the untracked files are `a5/impl.md` and the `a5-impl` outputs in `gates/out/`.
