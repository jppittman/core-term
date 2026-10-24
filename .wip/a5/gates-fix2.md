# Round 2 gates (tag `a5-fix2`)

Tree: `0c3b52a` plus the Round 2 working-tree edits (24 files outside `.wip`, nothing committed by this run). HEAD is `f3233fc`
(rulings; `git diff --stat 0c3b52a HEAD -- . ':!.wip'` is empty). Host: AVX-512 capable. `df -h /` before every build: 4.8-4.9 GB
free. sccache was up, so it was not restarted. No fix was made; this run wrote only `.wip/gates/out/*.a5-fix2.*` and this file.
`pixelflow-graphics/.../program/tests.rs` is clean after the probe builds.

## Result table

| # | Check | Result | Detail |
|---|---|---|---|
| 1a | `byteprobe.sh a5-fix2` | PASS, exit 0 | avx512 and avx2 each IDENTICAL to base (127 lines; sha256 `11636fa51e4e...` / `4b65f4699871...`) |
| 1b | `gate1b.sh a5-fix2` | PASS, exit 0 | IDENTICAL (29 lines, `4101ba0bd24c...`); self-check n=2 `22ad8819798461a6`, n=40 `e95137a9d464391e` equal to the pins |
| 1c | `gate2.sh a5-fix2 && gate2x.sh a5-fix2` | PASS, exit 0 / 0 | gate 2 IDENTICAL (29 lines, `1e3ab0b5ee9b...`); gate 2x IDENTICAL (29 records, `29834e148fba...`) |
| 2a | `PIXELFLOW_ISA=avx2 cargo test -p pixelflow-compiler` | PASS | 298 passed, 0 failed (lib 142; targets 5, 7, 6, 14, 6, 12, 5, 6, 30, 16, 10, 30, 9); doc-tests 6 ignored |
| 2b | `PIXELFLOW_ISA=avx512 cargo test -p pixelflow-compiler` | PASS | 298 passed, 0 failed, same per-target counts |
| 2c | `cargo test -p pixelflow-ir` | PASS | 397 passed, 0 failed, 2 ignored (310, 1, 75, 9, 1, doc 1) |
| 2d | `cargo test -p pixelflow-search --features provenance-journal` | PASS | 434 passed, 0 failed, 15 ignored |
| 2e | `cargo test -p pixelflow-core` | PASS | 136 passed, 0 failed, 1 ignored |
| 2f | `PIXELFLOW_ISA=avx2 cargo test -p pixelflow-codegen` | PASS | 213 passed, 0 failed, 0 ignored |
| 2g | `PIXELFLOW_ISA=avx512 cargo test -p pixelflow-codegen` | PASS | 213 passed, 0 failed, 0 ignored |
| 3a | `CARGO_TARGET_DIR=.../a3-close1/t199 cargo +1.99.0 clippy --workspace --all-targets --all-features -- -D warnings` | PASS, exit 0 | `Finished`; ir, search, codegen, core, ml, compiler, graphics, pipeline, runtime were re-checked; only `warning:` line is core-term's "Building with X11 display driver" |
| 3b | `cargo fmt --all -- --check` | PASS, exit 0 | no diff |
| 4a | `check-intent.sh` | PASS | 51 intent files |
| 4b | `check-doc-paths.sh` | PASS | 688 paths across 170 files resolve, 1 accepted via baseline. Stderr has the six `cannot open '../{AGENTS,CLAUDE,GEMINI}.md'` lines from the script itself, as in Round 1; exit 0 |
| 4c | `check-cfg-encapsulation.sh` | PASS | no new violation (121 grandfathered) |
| 4d | `check-emit-boundary.sh` | PASS | self-test 66 cases; 25 files |
| 4e | `check-bin-declarations.sh` | PASS | all 29 bin targets declared |
| 5a | rustdoc compare `pixelflow-compiler` | PASS | base 4, head 4; NEW 0, GONE 0 |
| 5b | rustdoc compare `pixelflow-ir` | PASS | base 38, head 38; NEW 0, GONE 0 |
| 5c | rustdoc compare `pixelflow-search` | PASS | base 28, head 28; NEW 0, GONE 0 |
| 5d | rustdoc compare `pixelflow-codegen` | PASS | base 0, head 0; NEW 0, GONE 0 |
| 6 | greps G1-G10 (R2 on G8, R3 on G10) and the `Param` proximity grep | all as expected; proximity grep 1 hit | below |
| 7 | `#[test]` count vs `passed`, `pixelflow-compiler` | MATCH | 298 `#[test]` (src 142, tests 156) = 298 passed on both ISAs; libtest printed 304 `test ... ` lines = 298 + 6 ignored doc-tests |
| 8 | `git diff --stat 0c3b52a -- . ':!.wip'` | 24 files, +148 / -146 | below |

## Greps (from `/home/user/pixelflow`, patterns verbatim from `spec.md` §7)

| Grep | Expectation | Found | Verdict |
|---|---|---|---|
| G1 deleted names | only the bound plan: §0 :31-32, row :231, the §3.4 sentence, the A2a bullet | `a-bound-is-a-value.md` :31, :32, :231, :417, :418, :574, :581; no other file | as expected (:574/:581 are the A2a bullet's lines at this tree, as in Round 1) |
| G2 `structural` in `pixelflow-compiler/` | nothing | nothing | as expected |
| G3 template / instantiat / hole | only `lib.rs` `macro_tier` doc, "the template vocabulary" | `pixelflow-compiler/src/lib.rs:357` ("equality saturation over the template", "vocabulary" on :358) | as expected |
| G4 `push_param` / `ExprNode::Param` | only `emit.rs`'s `node` panic arm | `emit.rs:440` (the arm) and `:441` (its message) | as expected. The message now reads "kernel! produced ExprNode::Param({k}): lowering builds none"; it no longer carries R1's "a `Param` is a rewrite rule's metavariable, substituted before any arena reaches emission" clause (`emit.rs:439` comment: "A slot with no value: lowering builds none.") |
| G5 the step label in `.rs` / `intent.md` | nothing | nothing | as expected |
| G6 `stride` in the compiler; `stride:` in `kernel.rs` | nothing from either | nothing from either | as expected |
| G7 every `FoldTerms` built | `monoid`, `lo`, `hi`, no `stride` | built at `kernel.rs:765` (Lit/Lit), `:863` (Value/Value), `:1684` (SUM, Lit 0, Lit 3), `emit.rs:612` (the emitted tokens), `lower.rs:340`, `arena.rs:2449` (test helper `terms`); destructured at `arena.rs:1743`; the struct (`arena.rs:177`) has exactly `monoid`, `lo`, `hi` | as expected |
| G8 (R2: `\| grep -v 'as_chunks::<'`) | only the parser's new test, in its `quote!` cases | `parser.rs` :2735, :2736, :2738, :2739 (`<const N: usize` cases), :2768, :2770, :2771 (`::<4>` cases); nothing else (unfiltered the grep has 10 lines, 3 of them `as_chunks::<`) | as expected. Round 1's extra hit at `parser.rs:664` (the doc comment of `generic_arguments_refusal`) is gone |
| G9 structural param / hole / count | nothing | nothing | as expected |
| G10 (R3) live docs | every hit history under a banner, the bound plan's own text, or the ontology's retired entry; files: the-language-is-kernel, the bound plan, ontology `SKILL.md` and `language.md` | plan :417, :434, :501, :693; the-language-is-kernel :58 (the banner sentence), :163 (§1.1 binding-time bullet), :298 (§1.5), :329 (§1.6), :499 (§1.7), all named by the banner at :54-:59; `ontology/language.md:149` and `ontology/SKILL.md:59` ("Structural parameter (retired)") | as expected; no `compiler.md` hit |

### `Param` proximity grep

Method: every `*.rs` and `*.md` from `git ls-files -co --exclude-standard`, minus `docs/archive/`, `docs/results/`, `.wip/` and
`target*`; a line matching `metavariable|substitut|specializ` within 3 lines (either side) of a line containing `Param`. Run with
`Param` as a whole word and as a substring, case-sensitive and case-insensitive: the four runs give the same single hit.

| File:line | Text | Note |
|---|---|---|
| `docs/language/2026-09-24-one-pipeline.md:667` | "the `Param` family, about 120 references in 40 files, including the dead Dag-side `substitute_params`" (row F4 of a table) | `substitute_params` is a function name, not the concept; the file is unchanged since `7e9531e`; `substitute_params` is not in any `.rs` file |

No hit in any `.rs` file and none in any file this change touches.

## Diff stat against `0c3b52a` (outside `.wip`)

```
 .claude/skills/ontology/compiler.md             |  3 +-
 .claude/skills/ontology/language.md             | 19 +++++-----
 docs/language/2026-10-20-a-bound-is-a-value.md  | 22 ++++++------
 pixelflow-codegen/src/program/lower.rs          |  8 ++---
 pixelflow-compiler/intent.md                    |  3 +-
 pixelflow-compiler/src/emit.rs                  | 20 +++++------
 pixelflow-compiler/src/lib.rs                   | 33 +++++++++---------
 pixelflow-compiler/src/lower.rs                 |  2 +-
 pixelflow-compiler/src/parser.rs                | 46 ++++++++++++++++---------
 pixelflow-compiler/src/sema.rs                  |  5 +--
 pixelflow-compiler/tests/binding_times.rs       | 17 ++++-----
 pixelflow-ir/src/expr.rs                        |  4 +--
 pixelflow-ir/src/kind.rs                        | 22 ++++++------
 pixelflow-ir/src/term.rs                        |  8 ++---
 pixelflow-ir/src/variance.rs                    |  4 +--
 pixelflow-pipeline/src/collapse_bench/corpus.rs | 12 +++----
 pixelflow-pipeline/src/training/factored.rs     |  2 +-
 pixelflow-search/src/egraph/cost.rs             |  2 +-
 pixelflow-search/src/egraph/insert.rs           | 12 +++----
 pixelflow-search/src/egraph/node.rs             | 21 ++++-------
 pixelflow-search/src/nnue/factored.rs           |  5 ++-
 pixelflow-search/src/runtime.rs                 |  4 +--
 pixelflow-search/src/saturate_pass.rs           |  4 +--
 pixelflow-search/tests/ir_insert.rs             | 16 ++++-----
 24 files changed, 148 insertions(+), 146 deletions(-)
```

## Outputs kept

`/home/user/pixelflow/.wip/gates/out/{byteprobe,gate1b,gate2,gate2x}.a5-fix2.*` (gate 2's `.jsonl` too). Cargo and rustdoc logs are in
the session scratchpad, not in the repo.
