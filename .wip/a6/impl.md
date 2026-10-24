# A6 implementation record: code and tests

Base `29980bf` plus `.wip/` commits. Spec `.wip/a6/spec.md` §2-§6, the code docs it names (§2.6, §4.8, §7.8), and
`.wip/a6/rulings.md` (which wins), with D-A6-1 as arm (b): `write_into` panics on a count past 2^24 and no public
item is added. No `.md` file was edited by this pass except this one. Nothing was committed, stashed, reset or checked
out. Every file:line below is in the working tree as left, not at `29980bf`.

## Step 0: the macro-tier probe at the base

`git diff --quiet 29980bf HEAD -- . ':!.wip'` held. The probe `/home/user/scratch/a5-probe/tests/a6_macro_tier.rs`
(repo target dir, `RUSTFLAGS="-C llvm-args=-fp-contract=fast"`):

| Run | Result line |
|---|---|
| `PIXELFLOW_ISA=avx2 cargo test --test a6_macro_tier` | `test result: ok. 2 passed; 0 failed` |
| `PIXELFLOW_ISA=avx512 cargo test --test a6_macro_tier` | `test result: ok. 2 passed; 0 failed` |
| `/home/user/scratch/a6-stages-probe` `--test shapes`, avx2 and avx512 | `test result: ok. 13 passed; 0 failed` (each) |

Greps on the unmodified tree found what the spec said: G2 found `pixelflow-ir/src/fold.rs:441`, G3 and G4 found the
listed sites, G6 printed `0`. Baseline `cargo test -p pixelflow-compiler`: 143 lib + 156 integration = 299 passed.

## What changed, per spec section

### §7.8 the IR's one doc
- `pixelflow-ir/src/fold.rs:439-445`: `BoundRefusal::BoundOutOfDomain`'s doc loses the step label ("from A6"); it now
  says `kernel!` refuses "in its own words, at expansion for a literal and where the host writes a count". No code.

### §2 sema (`pixelflow-compiler/src/sema.rs`, `symbol.rs`)
- §2.1 `Count { Known(u64), Name(Ident) }` with `name()` and `From<u64>`: `sema.rs:1804-1832`. `UsizeScope` gains
  `type Output: From<u64>` and `operand`: `:1839-1912`. `RangeScope` yields `Count` and stays consts-only: `:2001-2043`.
  `ConstEvaluator` keeps `Output = u64`: `:2081-2106`. `range_bounds` returns `[Count; 2]`, refusing only a known
  reversed pair: `:1953-1971`. `bound_arithmetic` (M3): `:1777`. The `use std::ops::Range` is gone.
- §2.2 `type_of_fold` types the names of both bounds before the fold's scope opens, then `range_bounds`, then decision
  8: `:961-979`. `bound_named` `:985`, `not_a_usize_bound` (M2) `:999`, `a_coordinate_bounds_no_fold` (M1) `:1762`.
  `resolve_ident` is a guard clause and `not_in_scope` the moved tail: `:1144`, `:1170`; no message changed.
- §2.3 `refuse_a_value_bound_under_a_mask` (M4), by `Reduction::Any | Reduction::All`: `:1974-1993`.
  `takes_a_value_bound` stays `pub(crate)` in the IR (decision 8).
- §2.4 `n: usize` is an entry's parameter of the items form: `param_type` `:628`, `count_parameter` (M6, M7) `:688`.
  `Parameter { name, kind }`, `ParameterKind { Value, Count, Record }`, `Scalar { Value, Count, Field }`: `:244-299`;
  `parameter_kind`: `:321`.
- §2.5 messages M1-M7 cite `BOUND_PLAN`; the widened notes are in `of_a_value` (`:154`), `a_count_is_not_a_value`
  (`:1746`), `type_of_cast` (`:1030`) and `convert_cast`/`generic_refusal` in `parser.rs`.
- §2.6 text edits: every row applied in `sema.rs` (module doc, `Ty` docs, `signature` doc, the stray duplicated doc line
  at the old `:1893`, the `collect_calls` comment), `symbol.rs` (table row, Parameter Symbols paragraph,
  `register_parameter`), `ast.rs` (five sites), and `parser.rs` (grammar: `'usize'`, `iexpr` with `CONST`, `range`/`bound`,
  the legend, the binding-times paragraph, sema's and lowering's lists, the loop and collection notes, `convert_range`).

### §3 lowering (`lower.rs`)
- §3.1 `Binding::Index` is `Binding::Count` (`:208`); `resolve`'s arm says "`{name}` is a `usize`, where a value is
  expected" (`:876`).
- §3.2 `lower_entry` builds each binding from its kind, with no `expect` (`:403-426`, ruling S4).
- §3.3 `Site::open_fold` takes `[Bound<Term>; 2]` (`:183`); `Expansion::open_fold` pushes `FoldTerms { monoid, lo, hi }`
  (`:362`). `Staged::open_fold` is §4.7.
- §3.4 `lower_fold` resolves both bounds first (`:638`); `fold_bounds` (`:707`), `bound` (`:727`), `literal_bound`
  (`:256`, beside `literal_range`). Ruling S3: `refuse_a_bound_that_is_no_count` (`:681`) and `count_term` (`:692`)
  check every name of both bounds before `range_bounds` judges arithmetic.
- §3.5 `lower_cast`, `record_named`, `kernel_named`, module doc (`:28-40`).

### §4 emission and the host surface (`emit.rs`)
- §4.1 `argument` over the `Scalar` enum: `:170`. A count is `n as f32`; `write_into` reads it as `self.n as f32`.
- §4.2 `emit_entry` takes the analysis; a count gets `::core::assert!(KnownRange::admits(&(0..n as u64)), msg, n)`
  before the body and a `# Panics` doc: `:209-246`, `count_parameters` `:254`, `COUNT_DOMAIN` `:267`, `count_refusal`
  `:273`, `panics_section` `:249`.
- §4.3 `Args` field is the declared `usize`; values via `argument`.
- §4.4 and §4.5, **D-A6-1 arm (b)**: `emit_args` (`:290`) asserts each count before `set_declared`, with the same
  message, and carries a `# Panics` section ("Nothing is written"). The signature stays
  `Result<(), ArityMismatch>` (G8: the one `ArityMismatch`, at `emit.rs:360`). `# Errors` is reworded (surface S3):
  "Of the block, only the number of arguments is checked". Nothing in `pixelflow-core`.
- §4.6 the host function panics (D-A6-4); the signature stays `-> Kernel`.
- §4.7 `Staged::fold_terms` takes `[Bound<Ident>; 2]` (`:688`), `Staged::open_fold` (`:788`).
- §4.8 text edits: emit.rs module doc, `emit_kernel`, `emit_closure`'s comment, `emit_entry`, `emit_args`;
  `lib.rs`: `BOUND_PLAN`'s doc (`:75`), `# Folds` (`:157`), `# The closure form` (`:191`), `# Binding times` (`:260`).

### §5 the macro tier
No code. `kernel!` offers every entry that takes no kernel to `Saturate::macro_tier()`; the new entries (triangle,
`sum_to`, `triangle_to`, `sum_from`, `sum_between`, `counted`) all saturate and keep their meaning. Telemetry
(`--features saturation-telemetry`, a scratch crate over the base test files for the comparison): `rustc_is_the_oracle`
33 -> 38 records and `items_block` 8 -> 9, all five and one new records `quiesced`; the `class_cap` counts are 1 and 4
at the base and 1 and 4 now (pre-existing entries). No `BoundVaries` at materialization (E8) was hit.

## Rulings that touch code, and what each became

| Ruling | Where |
|---|---|
| D-A6-1 arm (b) | `emit.rs` `emit_args`; no `pixelflow-core` change; the offered arm (a) stays in the spec |
| S2 | `not_a_usize_bound` picks its help by `(kind, type, role, spelling)`: items-form entry gets today's text, the closure form a pointer to the items form (M7's help, one `WRITE_AN_ENTRY` const), a helper none. The pinned help substring is on an items-form row. `generic_arguments_refusal` (the turbofish help) is unchanged: it never got the appended clause. |
| S3 | lowering's name check, `lower.rs:681`; pins `|| (0..X + 1)` and `|r: f32| (0..r - 1)` in `a_bound_that_is_no_count_is_refused_here_too` |
| S4 | `lower_entry`: both `expect`s gone |
| S5 | `parser.rs` grammar: `bound ::= iexpr | INDEX | COUNT`, `CONST` as `iexpr`'s leaf, legend lines |
| C1 | the emit unit test was written without the `write_into` asserts, run in step 2, and the `refused < written` check added in step 4 |
| C7 | `param_type`'s entry refusal now reads "an `f32`, a count `n: usize` or a record"; comment in `rustc_is_the_oracle.rs` widened. **Unchanged**: `FnItem::has_args_record` (`ast.rs:200`, a count-only entry has an `Args` record, as a test shows), and the three `Ty::Usize` arms (`sema.rs:440` a `const`'s declared type, `:577` `name_of`, `:1590` the type of a statement expression). |
| LAWS-1 | the code docs that stated L2/L5 for `kernel!` were deleted or qualified: `range_bounds` says "decided per call (L2)"; `lib.rs` `# Folds` says "up to two optimizer edges `Kernel::over_values` documents"; the oracle test doc says "at these samples (L2, which holds up to the two optimizer edges ...)"; M4 no longer says "a fold is NaN" |
| LAWS-2 | `COUNT_DOMAIN` (`emit.rs:267`) is the one sentence in the host function's panic message, its `# Panics`, and `write_into`'s `# Panics`: a count is an integer in `0..=2^24`; `2^24 + 1` would be written as `2^24`, a fold bounded by it would run one trip short with no NaN, and a larger count would name none. "Refusal of plausible pixels" appears nowhere. |
| LAWS-3 | no code doc repeats the §5.1 claim that a fold rule runs at expansion |
| surface S3, S4 | `write_into`'s `# Errors`; the `# Panics` line names the count and does not claim to be the whole list |

**A `usize` const read as `N as f32` past 2^24 stays accepted** and rounds as Rust's `as` rounds (`lower_cast`,
`sema.rs` `eval_f32`'s `Cast` arm; `an_f32_const_converts_a_usize_const_by_as` already builds `N = 16777217`). It
predates A6, and it is a value read, not a bound or a host write.

## Departures from the spec, each with its reason

1. **One shared walk, `bound_names` (`sema.rs:1918`), instead of `type_of_bound` in sema and a second walk in lowering.**
   Ruling S3 asks lowering to check names the way `type_of_bound` does; a second copy of "which shapes a bound is built
   from" would be a place two stages can disagree. Both stages iterate `bound_names` over `lo` then `hi` and judge each
   name in their own scope; the shapes and the order of refusals are the spec's.
2. **M2's help is gated on the type being `f32`** (and by `(role, spelling)` per S2). The spec's help by kind alone would
   tell `(0..p)` over a record parameter to declare `p: usize`. Every row the spec pins is an `f32`.
3. **M4's note** reads "a mask has no NaN to answer with when a bound is no count" instead of "outside a count's domain a
   fold is NaN, and a mask has no NaN to be": the first half stated L5 for `kernel!` without LAWS-1's qualification, and
   deleting beat qualifying.
4. **The panic message and `# Panics` texts** are LAWS-2's and S4's, not §4.2's. The `# Panics` paragraph begins "If a
   count argument is past 2^24" under the heading, and `write_into`'s says "If a count in this record is past 2^24 ...
   Nothing is written".
5. **`panics_section`, `COUNT_DOMAIN`, `WRITE_AN_ENTRY`** are shared helpers (two emission sites, three message sites, two
   refusal sites): one sentence each, so they cannot drift.
6. **The sema bound test has 16 rows** (the spec's 11, with the `n: f32` row split into an items-form row and a
   closure-form row per S2, plus a helper row asserting no `help:`, a record row, a kernel row and the unknown inner
   binder `(0..j)`). The fold-index test also pins `(0..i).map(|i| ...)` in an outer index's body.
7. **Oracle test docs**: the empty-count doc no longer says "+0.0 against rustc's -0.0" (for a runtime-empty range
   FactorFold can give X's sign); it says the zero may differ in sign, which `==` cannot see.
8. **`binding_times.rs`** keeps a local `within` (with a `HANG` constant) as the spec allows; its Args-write test compiles
   a program first, as arm (a)'s test does, though arm (b) needs only a block.
9. **`generic_refusal`'s help** gets the spec's appended clause; the turbofish help does not (S2).
10. Not done here, because they are not code or tests: `.wip/a6/public-surface.md` (the surface is below), the docs of §7.1-§7.7, and
    the gates 1, 1b, 2, 2x, 5 (a gate script appends a probe to a tracked file, so the coordinator runs them), the
    `+1.99.0` clippy over the whole workspace (run here for `pixelflow-compiler` and `pixelflow-ir`) and
    `check-feature-matrix.sh`.

## Public surface (arm b)

Added to the generated API of a block, none to a crate: a host `fn` with an `n: usize` parameter (items form only) that
panics on a count past 2^24 and says so under `# Panics`; the `Args` field `pub n: usize`; `write_into`'s signature is
unchanged and it panics on a count past 2^24 before writing anything. `pixelflow-core` is untouched. Visibility: `Count`,
`ParameterKind`, `Scalar`, `Parameter`, `bound_names`, `range_bounds` and `RangeScope` are `pub(crate)`/crate-private in
a proc-macro crate; `Monoid::takes_a_value_bound` stays `pub(crate)`.

## Test counts, before and after (`#[test]` per file; sum checked with `grep -c`)

| File | Before | After | Added |
|---|---|---|---|
| src/sema.rs | 60 | 63 | `a_range_bound_may_be_an_enclosing_index_or_a_count_parameter`, `arithmetic_on_a_value_bound_is_refused_until_bounds_are_integers`, `any_and_all_take_bounds_known_at_expansion` (renamed: `a_range_bound_is_a_usize_and_reads_no_coordinate`, `a_usize_parameter_is_an_items_entrys_count`) |
| src/lower.rs | 18 | 20 | `a_value_bound_is_a_child_of_its_fold`, `a_bound_that_is_no_count_is_refused_here_too` |
| src/emit.rs | 7 | 8 | `a_count_parameter_is_refused_on_the_host_and_declared_as_an_f32` |
| tests/rustc_is_the_oracle.rs | 30 | 34 | `a_triangular_fold_folds_as_rustcs_does`, `a_per_call_count_folds_as_rustcs_does`, `a_range_between_counts_folds_as_rustcs_does`, `a_per_call_count_around_an_application_is_rusts` |
| tests/binding_times.rs | 7 | 12 | `every_count_is_one_program`, `a_program_compiled_at_one_count_is_rebound_to_another`, `the_host_function_refuses_a_count_past_the_domain`, `a_count_written_raw_outside_the_domain_is_nan`, `the_args_write_refuses_a_count_past_the_domain` |
| tests/fold_is_kernel_over.rs | 14 | 16 | `a_per_call_count_is_over_values`, `a_triangular_fold_is_over_values_inside_over` |
| tests/kernel_typed_parameters.rs | 16 | 17 | `a_count_beside_a_kernel_is_one_program_for_every_count` |
| tests/optimized_agrees_with_raw.rs | 10 | 11 | `a_per_call_count_agrees` |
| the other twelve files (ast 4, lib 4, parser 43, symbol 7, eight test files 79) | 137 | 137 | |
| **Total** | **299** | **318** | **19** |

`pixelflow-core`: 0 added (arm (b)). Both canonical-key equalities of §6.3 (`kernel_raw!` against `over_values`, for a
per-call count and for the triangle, with binders `[5, 4]`) held on the first run; no test was loosened.

## Commands and results

| Command | Result line |
|---|---|
| `cargo test -p pixelflow-compiler --lib` (after step 2) | `test result: ok. 149 passed; 0 failed` |
| `PIXELFLOW_ISA=avx2 cargo test -p pixelflow-compiler` (after step 3, then after step 4, then at the end) | 149 lib + 169 integration = 318 passed, 0 failed, 6 ignored (doc-tests), each time |
| `PIXELFLOW_ISA=avx512 cargo test -p pixelflow-compiler` (after step 4 and at the end) | 318 passed, 0 failed, 6 ignored |
| `cargo test -p pixelflow-compiler` (default tier) | 318 passed, 0 failed, 6 ignored |
| `cargo test -p pixelflow-core` | lib `68 passed`, integration 68 passed, doc-tests `0 passed; 1 ignored`; 0 failed |
| `cargo test -p pixelflow-graphics --lib` | `test result: ok. 121 passed; 0 failed; 3 ignored` |
| `cargo check --workspace --all-targets` | `Finished dev profile`; no warning (the one line is core-term's build script saying it builds the X11 driver) |
| `cargo +1.99.0 clippy -p pixelflow-compiler --all-targets --all-features -- -D warnings` | `Finished`, no diagnostic (all-features prints saturation telemetry from the macro expansions) |
| `cargo +1.99.0 clippy -p pixelflow-compiler -p pixelflow-ir --all-targets -- -D warnings` | no diagnostic |
| `cargo fmt --all` (twice), then `cargo fmt --all -- --check` | clean |
| `cargo doc -p pixelflow-compiler --no-deps --document-private-items` through `rustdoc_norm.py --compare` | `warnings: base 4, head 4`, `NEW 0`, `GONE 0` |
| the same for `pixelflow-ir` | `warnings: base 38, head 38`, `NEW 0`, `GONE 0` |
| `scripts/check-intent.sh`, `check-doc-paths.sh`, `check-emit-boundary.sh`, `check-bin-declarations.sh`, `check-cfg-encapsulation.sh` | each `OK` (run while the docs pass had edited `pixelflow-compiler/intent.md`) |

Greps (from the repo root, at the end): G1 nothing; G2 (with `grep -v guide_scope_saturation_delta.rs`, ruling S1) nothing;
G3 only `tests/rustc_is_the_oracle.rs:324`; G4 nothing; G5 `lower.rs:243` (`literal_range`), `lower.rs:258`
(`literal_bound`), `emit.rs:225` (host), `emit.rs:322` (`write_into`); G6 `0`; G8 the one `ArityMismatch` in the signature;
G9 `BOUND_PLAN` at the M1-M7 sites, `emit.rs` x3, `lower.rs:719` and `parser.rs:660`, and `docs/language/2026-10-20` only
at `lib.rs:77`, `lib.rs:289`, `parser.rs:112` and the test assertions (ruling C4); G10 the trait's, `Expansion`'s and
`Staged`'s `open_fold`, `lower_fold`'s call, and each site's arena call.

Scratch, outside the repo: `/home/user/scratch/a6-impl-telemetry` (the base test files with the telemetry feature);
telemetry records under the session scratchpad.
