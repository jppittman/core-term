# Round 1 gates (tag `a5-fix1`)

Tree: `38703e0` plus the Round 1 working-tree edits (18 files outside `.wip`, nothing committed by this run). HEAD is `44a60e0`
(notes only). Host: AVX-512 capable. `df -h /` before every build: 5.0 GB free. sccache was up (`--show-stats` worked), so it was not
restarted. No fix was made; nothing outside `.wip/gates/out/` (new `a5-fix1` files) and this file was written.

## Result table

| # | Check | Result | Detail |
|---|---|---|---|
| 1a | `byteprobe.sh a5-fix1` | PASS, exit 0 | avx512 and avx2 columns each IDENTICAL to base (127 lines; sha256 `11636fa51e4e...` / `4b65f4699871...`) |
| 1b | `gate1b.sh a5-fix1` | PASS, exit 0 | IDENTICAL to `gate1b.base.txt` (29 lines, sha256 `4101ba0bd24c...`); self-check n=2 `22ad8819798461a6`, n=40 `e95137a9d464391e` equal to the pins |
| 1c | `gate2.sh a5-fix1` then `gate2x.sh a5-fix1` | PASS, exit 0 / 0 | gate 2 IDENTICAL (29 lines, sha256 `1e3ab0b5ee9b...`); gate 2x IDENTICAL (29 records, sha256 `29834e148fba...`) |
| 2a | `PIXELFLOW_ISA=avx2 cargo test -p pixelflow-compiler` | PASS, exit 0 | 298 passed, 0 failed (lib 142; targets 5, 7, 6, 14, 6, 12, 5, 6, 30, 16, 10, 30, 9); doc-tests 6 ignored |
| 2b | `PIXELFLOW_ISA=avx512 cargo test -p pixelflow-compiler` | PASS, exit 0 | 298 passed, 0 failed, same per-target counts |
| 2c | `cargo test -p pixelflow-ir` | PASS, exit 0 | 397 passed, 0 failed, 2 ignored (310, 1, 75, 9, 1, doc 1) |
| 2d | `cargo test -p pixelflow-search --features provenance-journal` | PASS, exit 0 | 434 passed, 0 failed, 15 ignored (lib 397 with 8 ignored; 5, 13, 2, 7, 1, 2, 7; doc-tests 6 + 1 ignored) |
| 2e | `cargo test -p pixelflow-core` | PASS, exit 0 | 136 passed, 0 failed, 1 ignored (68, 1, 7, 13, 16, 16, 3, 6, 6) |
| 3a | `CARGO_TARGET_DIR=/home/user/scratch/a3-close1/t199 cargo +1.99.0 clippy --workspace --all-targets --all-features -- -D warnings` | PASS, exit 0 | `Finished`; the only `warning:` line is core-term's build-script notice "Building with X11 display driver" |
| 3b | `cargo fmt --all -- --check` | PASS, exit 0 | no diff |
| 4a | `scripts/check-intent.sh` | PASS | 51 intent files |
| 4b | `scripts/check-doc-paths.sh` | PASS | 685 paths resolve, 1 accepted via baseline. Stderr carries six lines (`head`/`grep: cannot open '../AGENTS.md'`, `../CLAUDE.md`, `../GEMINI.md`) from the script's own line 148; `scripts/` is not in the diff and the exit is 0 |
| 4c | `scripts/check-cfg-encapsulation.sh` | PASS | no new violation (121 grandfathered) |
| 4d | `scripts/check-emit-boundary.sh` | PASS | self-test 66 cases; 25 files |
| 4e | `scripts/check-bin-declarations.sh` | PASS | all 29 bin targets declared |
| 5a | rustdoc compare, `pixelflow-compiler` | PASS | base 4, head 4; NEW 0, GONE 0 |
| 5b | rustdoc compare, `pixelflow-ir` | PASS | base 38, head 38; NEW 0, GONE 0 |
| 5c | rustdoc compare, `pixelflow-search` | PASS | base 28, head 28; NEW 0, GONE 0 |
| 6 | greps G1-G10 (R2 on G8, R3 on G10) | 9 as expected, G8 one extra line | see below |
| 7 | `#[test]` count vs `passed` | MATCH | 298 `#[test]` lines (src 142, tests 156) = 298 passed (142 + 156), on both ISAs |
| 8 | `git diff --stat 38703e0 -- . ':!.wip'` | 18 files, +126 / -102 | see below |

## Greps

Run from `/home/user/pixelflow`, patterns verbatim from `spec.md` §7, R2 appended to G8 and R3 applied to G10.

| Grep | Expectation | Found | Verdict |
|---|---|---|---|
| G1 deleted names | only the bound plan: §0 :31-32, row :231, the §3.4 sentence, the A2a bullet | `a-bound-is-a-value.md` :31, :32, :231, :417, :418, :574, :581 and no other file | as expected. The A2a bullet's line numbers are :574 and :581, not the spec's :578 and :585: the same two hits sit at :574 and :581 already at `38703e0`, because the plan's own §3.4 rewrite moved them, and Round 1 did not touch lines above :698 |
| G2 `structural` in `pixelflow-compiler/` | nothing | nothing | as expected |
| G3 template / instantiat / hole | only `lib.rs` `macro_tier` doc, "the template vocabulary" | `pixelflow-compiler/src/lib.rs:356` ("equality saturation over the template", continued "vocabulary" on :357) | as expected |
| G4 `push_param` / `ExprNode::Param` | only `emit.rs`'s `node` panic arm | `emit.rs:440` (the arm) and `:442` (its message) | as expected |
| G5 the step label in `.rs` / `intent.md` | nothing | nothing | as expected |
| G6 `stride` in the compiler; `stride:` in `kernel.rs` | nothing from either | nothing from either | as expected |
| G7 every `FoldTerms` built | `monoid`, `lo`, `hi`, no `stride` | built at `kernel.rs:765, 863, 1684`, `emit.rs:616` (the emitted tokens), `lower.rs:340`, `arena.rs:2449` (test helper); destructured at `arena.rs:1743`; the struct (`arena.rs:177`) has exactly those three fields | as expected |
| G8 (R2: `\| grep -v 'as_chunks::<'`) | only the parser's new test, in its `quote!` cases | `parser.rs` :2721, :2722, :2724, :2725 (`<const N: usize` cases), :2754, :2756, :2757 (`::<4>` cases), plus **`parser.rs:664`** | seven test lines as expected. One extra line, :664, is in the doc comment of `generic_arguments_refusal` (non-test code): "spanned on the arguments, `::<4>`, the token that is wrong." It is a Round 1 sentence naming the span, not a use of the syntax. Strictly, the grep's expectation ("only the new test") is not met by this one line |
| G9 structural param / hole / count | nothing | nothing | as expected |
| G10 (R3) live docs | every hit history under a banner, the bound plan's own text, or the ontology's retired entry; files: the-language-is-kernel, the bound plan, ontology `SKILL.md` and `language.md` | plan :417, :434, :501, :693; the-language-is-kernel :58 (the banner sentence itself), :163 (§1.1 binding-time bullet), :298 (§1.5), :329 (§1.6), :499 (§1.7), all named by the banner; `ontology/language.md:149` ("Structural parameter (retired)"); `ontology/SKILL.md:59` ("Structural parameter (retired)") | as expected; no `compiler.md` hit and none at `language.md:125-126` |

## Diff stat against `38703e0` (outside `.wip`)

```
 docs/BACKLOG.md                                |  1 +
 docs/language/2026-10-20-a-bound-is-a-value.md | 10 ++-
 pixelflow-compiler/src/emit.rs                 |  4 +-
 pixelflow-compiler/src/lower.rs                |  5 +-
 pixelflow-compiler/src/parser.rs               | 88 ++++++++++++++++----------
 pixelflow-compiler/src/sema.rs                 | 12 ++--
 pixelflow-ir/src/arena.rs                      |  6 +-
 pixelflow-ir/src/expr.rs                       |  3 +-
 pixelflow-ir/src/fold.rs                       | 16 ++---
 pixelflow-ir/src/kernel.rs                     |  2 +-
 pixelflow-ir/src/kind.rs                       | 21 +++---
 pixelflow-ir/src/term.rs                       |  5 +-
 pixelflow-search/src/egraph/extract.rs         |  2 +-
 pixelflow-search/src/egraph/graph.rs           |  4 +-
 pixelflow-search/src/egraph/insert.rs          | 26 ++++----
 pixelflow-search/src/egraph/node.rs            | 11 ++--
 pixelflow-search/src/runtime.rs                |  4 +-
 pixelflow-search/tests/ir_insert.rs            |  8 +--
 18 files changed, 126 insertions(+), 102 deletions(-)
```

18 files, as `impl.md` Round 1 states. `pixelflow-graphics/.../program/tests.rs` is unmodified after the probe builds (`git diff --quiet` clean).
`git status` outside `.wip` lists only the 18 modified files above.

## Outputs kept

`/home/user/pixelflow/.wip/gates/out/{byteprobe,gate1b,gate2,gate2x}.a5-fix1.*` (gate 2's `.jsonl` too). Cargo and rustdoc logs are in the
session scratchpad, not in the repo.
