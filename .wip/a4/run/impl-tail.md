# A4 implementer 3: the serializers, the search docs, the docs, the first gates

Base a747770 plus implementer 1 (pixelflow-ir) and implementer 2 (pixelflow-codegen), committed locally. Design:
`rulings.md`, then `spec.md`, then the three critiques. Scope: spec 4.3 and 4.4 (the serializers and the consumers of
`EmitTraffic`), 8.6 and 8.7, ruling (a)'s pin and the egraph `intent.md` line, C5, C8, C2, C9, and the docs of spec 10 with
the rulings applied. Then the workspace compile, the crate tests, and gates 1-3. The workspace compiles and gates 1, 1b,
2 and 2x are identical to base.

## Gates run

| Gate | Result |
|---|---|
| `cargo check --workspace --all-targets --all-features` | clean (no error, no warning). Before this tail it stopped in `pixelflow-pipeline` (the two `Trips`/`Option` errors implementer 2 named); `constant_traffic` was the third. Nothing else in the workspace broke |
| `cargo +1.99.0 clippy --workspace --all-targets --all-features -- -D warnings` (target dir `/home/user/scratch/a3-close1/t199`) | exit 0 |
| `cargo fmt --all -- --check` | clean (after `cargo fmt` on the three crates I touched; every diff was in a hunk of mine) |
| `scripts/check-intent.sh`, `check-doc-paths.sh`, `check-cfg-encapsulation.sh`, `check-emit-boundary.sh`, `check-bin-declarations.sh` | OK (51 intent files; the largest are `pixelflow-ir/intent.md` and `desloppify/src/review/intent.md` at 80; `egraph/intent.md` is 77) |
| `PATH=/home/user/scratch/cargo-hack-root/bin:$PATH bash scripts/check-feature-matrix.sh` | OK: every crate's every feature builds in isolation |
| `cargo test -p pixelflow-compiler` | 148 lib (incl. `a_value_bounded_fold_emits_its_children`, `a_literal_fold_emits_no_children`) and every integration file, green |
| `cargo test -p pixelflow-graphics` | lib 121 passed (3 ignored), every integration file green, `production_glyph_arena_dump` 1 passed (the new test; the telemetry dumper stays ignored) |
| `cargo test -p pixelflow-pipeline` | lib 180 passed, `corpus_gaps` bin 1 passed (its first test module), every other bin and test file green |
| `cargo test -p pixelflow-search --all-features` | lib 397 passed (8 ignored), every integration file green; both extraction pins ran |
| `cargo test -p pixelflow-core` | green |
| `PIXELFLOW_ISA=avx2` and `=avx512` `cargo test -p pixelflow-codegen`, `cargo test -p pixelflow-ir` | green on the integrated tree (rerun here, since nothing was run against the serializers' tree before) |
| gate 1 `byteprobe.sh a4-impl` | IDENTICAL to base on `avx512` and on `avx2` (127 lines each) |
| gate 1b `gate1b.sh a4-impl` | IDENTICAL: 29 keys, nodes, digests; self-check against the pinned `22ad8819798461a6`, `e95137a9d464391e` equal |
| gate 2 `gate2.sh a4-impl`, gate 2x `gate2x.sh a4-impl` | IDENTICAL: saturation counters and the extraction fields |
| `git status --short pixelflow-graphics` after the gates | only my four files (`examples/constant_traffic.rs`, `src/render/cell_grid.rs`, `src/test_support.rs`, `tests/production_glyph_arena_dump.rs`); `fonts/loop_blinn/program/tests.rs` (the probe file) is restored. Empty once they are committed |
| gate 5 `measure.sh a4-impl` (not asked for; run for the +0.5% constraint) | total Ir **+0.114 %** (6,317,448,479 against 6,310,238,494) against the a747770 base (implementer 2 measured +0.061 % at its tree), `Saturation::saturate` -0.001 %, `extract_dag_over` -0.013 %. RSS +0.89 %, wall -7.1 % (VM drift, not decisive). Nothing on the font programs' path changed in this tail, so the 0.05 point between the two measurements is code layout of the test binary the probe is built into |
| `python3 -I .wip/a4/gen/checklist.py --root /home/user/pixelflow` | 190 sites: already-correct 45, changed-by-A3 24, changed-by-A4 15, left 13, tests 88, n/a 5, **A4 0**, UNCLASSIFIED 0, STALE 0 |
| spec 9 greps | `VALUE_BOUND`: nothing. `fn known(bound`: nothing. `\bA4\b` in `.rs` and `intent.md`: nothing. `expect(dead_code, reason = "pub(crate) until A4`: nothing. `iter::empty()` in pipeline, graphics, compiler/src: the compiler's literal branch only. Every `fold_header(..)` writer writes the value children (`key.rs`, `test_support.rs`, both dumps, both corpora, `corpus_gaps`, the compiler). Every `FoldHeader::from_bits` in pipeline and compiler attaches the children its header counts (`fold_terms`' literal header excepted). `dynamic_memory_ops`: each caller handles the `Option`. `KNOWN_RANGE` non-test uses: `fold.rs`, `passes/lattice.rs`, codegen `lower.rs`'s lane fold, `glyph_phase_split.rs`; none in `select.rs` or the compiler's `emit.rs`. `has no bits")` is the two `shl` texts only (C3) |
| rustdoc, `--no-deps --document-private-items` on pipeline, compiler, graphics | every warning is in a line I did not write (the same files warn at base); none from `training/corpus.rs`'s or `collapse_bench`'s new text |

Mutation, run once and restored: dropping the value children from `corpus_gaps`'s `Key::Reduce` fails `hash_cons_keeps_a_value_bound`.

## What changed

### Serializers (spec 4.3)

- `pixelflow-graphics/src/render/cell_grid.rs` and `tests/production_glyph_arena_dump.rs`, `dump_arena`: `R {bits} {body}` then
  ` {id}` per value child, `lo`'s first; the comment now says the child follows the body as the marker says. The dumps stay
  `# pixelflow arena dump v1`: a literal fold's line is unchanged and no v1 reader parses an `R` line.
- `pixelflow-graphics/src/test_support.rs`: no code change (`to_bits` is total and every child is fed); the pin is
  `a_value_bound_is_in_the_program_digest`.
- `pixelflow-pipeline/src/collapse_bench/corpus.rs`: v5. `HEADER`, `SUPERSEDED_HEADERS` gains v4 with the spec's reason, the
  history comment gains the v5 paragraph. `encode` writes the children; `decode` matches `["R", bits, body, values @ ..]`,
  refuses naming the file unless `values.len() == header.arity() - 1`, and attaches.
- `pixelflow-pipeline/src/training/corpus.rs`: `write_node` writes the children after the body and no longer refuses;
  `read_node_into` reads `header.arity() - 1` more ids through `resolve_child` (a short record is `UnexpectedEof`);
  `CorpusFormat::SCHEMA` gains the `9=Reduce(..)` clause (that edit is the version bump), its history sentence and the module doc's
  tag list and `Reduce` record are updated.
- `pixelflow-pipeline/src/bin/corpus_gaps.rs`: `Key::Reduce(u128, Vec<u32>)` (bits, then the body's and each value child's
  mapped id); the build closure attaches the mapped children. New `#[cfg(test)] mod tests`.
- `pixelflow-compiler/src/emit.rs`: the `Reduce` arm asks `arena.known_range(id).and_then(|r| context.holes.range_of(r))` (the
  `KNOWN_RANGE` import is gone); a literal fold's tokens are unchanged (`attach(::core::iter::empty())`), a fold with a value
  bound emits `attach([__eN, ..])`; the hole branch is byte for byte what it was.

### Consumers of `Trips` (spec 4.4)

- `collapse_bench/mod.rs`: `features_of` reads `trips.known()` and `dynamic_memory_ops()` through a private `LITERAL_FOLDS`
  message; a `# Panics` section says so. `collapse_bench/intent.md` gains a "Never" line.
- `examples/constant_traffic.rs` prints `an estimate` where `dynamic_memory_ops()` is `None`; the `{trips:>8}` column is
  unchanged (`Trips`' `Display` honours width).

### Search (ruling (a); C5)

- `fold_rules.rs`: `KnownBound`'s doc and `known_bound_declines_a_count_out_of_domain`'s doc say "by the guard
  `passes::legalize` ends with" (C5). The two extraction pins keep `#[should_panic(expected = "BoundVaries")]`; their docs no longer
  say A4 decides, and say it is left loud and why. Both pins existed at a747770: the second one is the ruling's "add one
  pin" (`Pow(i, 2)` beside `((i·i)+X)−X`, no rules), so no test was added.
- `egraph/intent.md` ("Behind"): one bullet in place of the (b) line. No search code changed (spec 3.5 and 8.5 are dropped).

### Docs (spec 10, with the rulings)

- The plan: the A4 bullet's decision sentence and a `**Done:**` line of facts; section 3.3 amended (the guard is
  `legalize`'s last pass; the `Reduce` def's interval spans the loop, with the reload-per-latch fact; the post-collapse assert
  as built, the stride assert not added, the mask-valued bound refused; `Trips`); the A3 bullet's two "key refuses" sentences in the
  past tense.
- `CLAUDE.md` ("The language is a DAG"), `pixelflow-core/README.md`, `docs/DENOTATION.md` (the totality paragraph, the
  uniform paragraph, the `Reduce` paragraph, the binder-returns clause), the totality doc (the "Dynamic behavior" bullet and item
  5), `docs/BACKLOG.md` (the plan table row, C8, and a new row **E8** for the extraction hole), the ontology (Binding time,
  Fold, Range, Totality), and `.claude/agents/algebraist.md` (the same sentence as `CLAUDE.md`).
- `intent.md`: `pixelflow-ir`, `program/`, `emit/`, `codegen/` were done by implementers 1 and 2; this tail adds
  `egraph/intent.md` (77 lines) and `collapse_bench/intent.md`.

## Tests added

`cell_grid.rs`: `the_dump_writes_a_value_bounds_child`. `production_glyph_arena_dump.rs`: the same name, not ignored.
`test_support.rs`: `a_value_bound_is_in_the_program_digest`. `collapse_bench/corpus.rs`: `a_v4_corpus_names_the_version_it_is`,
`a_value_bounded_kernel_round_trips` (the oracle is `encode(&kernel) == encode(&decoded)`, ruling C2, plus the bound is still a
value after the trip), `a_fold_line_with_too_few_value_bounds_is_refused_naming_the_file`. `training/corpus.rs`:
`a_fold_record_marking_a_value_reads_its_child`, `a_fold_record_missing_its_value_child_is_unexpected_eof`,
`a_fold_record_with_an_end_past_the_bound_names_no_fold` (the first two replace
`a_fold_record_with_an_all_ones_end_names_no_fold`), `round_trip_a_value_bounded_fold` (`Σ_{i<4} Σ_{j<i} j`, `subtree_eq`).
`corpus_gaps.rs`: `hash_cons_keeps_a_value_bound` (only the half that can fail: two folds that differ only in a bound's child stay
two nodes, ruling C8). Compiler `emit.rs`: `a_value_bounded_fold_emits_its_children`, `a_literal_fold_emits_no_children`.

## Judgement calls, with the alternative rejected

- **J1. DENOTATION's pointer for the lattice's extent is section 5, not 7.** The spec's rewrite said "(§7)"; the statement that the
  extent is part of the program and the compile key is in section 5 ("A lattice is an extent and nothing else"). Rejected: the
  spec's pointer, which names a section about variance.
- **J2. The training reader does not pre-check L3 of a decoded value bound.** A hostile file that names a coordinate-reading
  term as a bound reaches `push_reduce`, which panics `BoundVaries` by name. The `of_arity` precedent refuses on the reader's own
  `io` channel, but the check would need `variance_from`, which is `pub(crate)` in the IR; widening it is outside the
  spec. A corpus the writer made cannot hold such a bound (its arena came through `push_reduce`).
- **J3. The second extraction pin was not added; it exists.** The ruling asks to "add one pin"; a747770 already holds
  `extracting_a_bound_class_whose_cheapest_member_reads_a_coordinate_is_refused_by_name`, with `should_panic(expected =
  "BoundVaries")` and the ruling's exact fixture. Rejected: a duplicate test. The docs on both pins changed (the work the ruling
  names).
- **J4. The reader's count check is an `assert_eq!` in the collapse corpus** (the module panics on every malformed line, naming the
  file) and an `UnexpectedEof` in the training corpus (the module returns `io::Result`). Each follows its module.
- **J5. `E8` is a new BACKLOG row id** (E7 is the last). The ruling's sentence is its body.
- **J6. `.claude/agents/algebraist.md` was edited.** It repeats `CLAUDE.md:180`'s sentence ("Today every bound is a constant ...") and
  A4 makes it false; it is not in the plan's section 5 table. Rejected: leaving a persona file that teaches the old rule.
- **J7. The `LITERAL_FOLDS` const is named in backticks, not linked,** from `features_of`'s public doc (`private_intra_doc_links`).

## Departures from the spec and the rulings

1. Spec 3.5 and 8.5 (extraction (b)) and the 10.4 `egraph/intent.md` (b) clause are dropped, per ruling (a); the plan's 3.2 pricing
   sentence is not added.
2. The `program_digest` pin builds its two arenas with `declare_uniform` twice and a hand-built fold, so the bounds sit in
   different slots; two kernels with one uniform each would digest equal.
3. The training-corpus round trip uses `subtree_eq` as the spec says (no uniform is in it); ruling C2 applies to the collapse
   corpus only, where it is used.
4. The `Done` line names the mask refusal (implementer 2's E1) beside the asserts the spec lists.

## Not run

- aarch64: no target on this host (`rustup target list --installed` is x86_64 only), as implementer 2 recorded. This tail changes no
  emitter and no `cfg`-gated code.
- The `Cross-target ABI (aarch64)`, `Test on macos-latest` and `ISA matrix` CI jobs.
