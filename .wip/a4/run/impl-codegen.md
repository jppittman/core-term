# A4 implementer 2: pixelflow-codegen

Base a747770 plus implementer 1's pixelflow-ir commit. Design: `rulings.md`, then `spec.md`, then the three critiques.
Scope: lowering, the operand and the carve, `Trips` and `EmitTraffic`, selection, the AVX-512 mask predicate, the
tests of spec 8.2-8.4 and the rulings that land here (F1/E2, F3/C10, F4, E1, E3/E4, C7/E5, C9). Other crates were not
touched; two of their callers stop compiling until implementer 3 (see "Left for the others").

## Gates run

| Gate | Result |
|---|---|
| `PIXELFLOW_ISA=avx2 cargo test -p pixelflow-codegen` | all green: lib 102, 22 in `value_bounds_jit`, every other integration file as at base, doctests 0 |
| `PIXELFLOW_ISA=avx512 cargo test -p pixelflow-codegen` | the same counts, all green (this host has AVX-512) |
| `cargo +1.99.0 clippy -p pixelflow-codegen --all-targets --all-features -- -D warnings` | clean (target dir `/home/user/scratch/a3-close1/t199`) |
| `cargo fmt -p pixelflow-codegen -- --check` | clean |
| `RUSTDOCFLAGS="-D warnings" cargo doc -p pixelflow-codegen --no-deps --document-private-items` | clean (base 0 warnings) |
| `scripts/check-intent.sh`, `check-doc-paths.sh`, `check-emit-boundary.sh`, `check-cfg-encapsulation.sh` | OK (`program/intent.md` is 70 lines, `emit/intent.md` 61, `codegen/intent.md` 58). `check-intent.sh` resolves a test file by `git ls-files`, so it passes once `tests/value_bounds_jit.rs` is added (`git add -N` did it here) |
| `cargo hack check -p pixelflow-codegen --each-feature --no-dev-deps` (cargo-hack from `/home/user/scratch/cargo-hack-root/bin`) | both rows OK. The workspace-wide `check-feature-matrix.sh` cannot pass until the two callers below are fixed |
| gate 1 `byteprobe.sh a4-cg1` | IDENTICAL to base on `avx512` and on `avx2` (127 lines each) |
| gate 1b `gate1b.sh a4-cg1` | IDENTICAL: 29 keys, nodes, digests (self-check against the pinned `22ad8819798461a6`, `e95137a9d464391e` equal) |
| gate 2 `gate2.sh a4-cg1`, gate 2x `gate2x.sh a4-cg1` | IDENTICAL: saturation counters and the extraction fields |
| gate 5 `measure.sh a4-cg1` | total Ir **+0.061 %**, `Saturation::saturate` -0.031 %, `extract_dag_over` -0.192 % against the a747770 base (this includes implementer 1's IR work). RSS +1.1 %, wall -9.7 %: VM drift, not decisive |
| aarch64 | **not run, and not type-checked as a target.** `rustup target list --installed` is `x86_64-unknown-linux-gnu` only; `cargo check -p pixelflow-codegen --target aarch64-unknown-linux-gnu` stops at E0463 (no `core` for the target). What stands in for it: the NEON backend is not `cfg`-gated, so it is type-checked and unit-tested on this host, and `every_backend_emits_a_fold_whose_bound_is_a_value` selects, allocates and assembles three value-bounded kernels for `aarch64::Neon` (see "Tests"). No NEON encoder, instruction form or `IsaBackend` method changed. `tests/value_bounds_jit.rs` is `cfg(any(x86_64, aarch64))` and runs natively on the macOS leg |

Two mutations, run to see the tests can fail (both restored): seeding the counter with a literal `0` instead of `lo`
fails `a_value_lo_starts_the_fold` and `an_inner_fold_starting_at_the_outer_index`; letting `ScheduledOp::is_mask` ignore
its second operand fails `a_bound_is_refused_exactly_when_it_computes_a_mask` and the AVX-512 exactness test.

## What changed

### `program/lower.rs`
- `known` and the `VALUE_BOUND` import are gone; the surviving-fold arm maps each bound with `Bound::map(map_child)`.
- The lane scan (already walking reachable nodes) also collects `lattice`, the union of every reachable `Write`'s row, column
  and lane binder variance, and the ids of folds with a value bound. A literal kernel pays one `match` arm per reachable node.
- **P1** in the arm, per value bound: `fold.admits_bound(end, variance[term])` (panic = the `BoundRefusal`'s `Display`, prefixed),
  then `assert_off_the_lattice(end, reads, lattice)` ("a lattice binder's").
- **P2 / F1 / E2**: `ValueFolds { readers: Vec<ValueFold> }`, `ValueFold { Not, Unread, ReadBy(ExprId) }`. `read_by(arena, reader)`
  runs before the `Write` skip for every reachable node, counts each *edge* to a value-bounded fold, and panics on a second
  reader (naming both: "is read by A and by B"), or on a reader that is not `Ternary(If, m, t, f)` with `t == fold`, `m != fold`,
  `f != fold` ("other than as an If's true arm"). After the loop `assert_every_fold_was_read` panics on a fold nothing read
  ("is read by nothing": the root, say). Readers that are empty folds are skipped, as `mark_reachable` skips their children;
  unreachable parents (a rebuilt arena keeps them) are never visited, which implementer 1's note J1 asked for.
- **E1**: a value bound that computes a mask panics with `BoundSort`'s `Display` (the arena's own words for a mask literal) on
  every tier. `MaskValues` extends a `Vec<bool>` along the schedule as far as a bound asks (nothing is allocated for a literal
  kernel) using `ScheduledOp::is_mask`, so a reinterpretation, which lowering erases, answers as its operand.
- Tests: the old refusal test is replaced; see "Tests".

### `program/mod.rs`
- `ScheduledOp::Reduce` doc (a value bound is an operand; `lo` seeds, `hi` is read at the latch). `operands` yields
  `(lo.value(), hi.value(), None)` for a `Reduce`, so `all_operands`, `structural_children`, the closure walk, `assert_closed`
  and `ownership::reads` see the bounds with no code change in `scopes.rs`/`ownership.rs`/`layout.rs` (one comment in
  `scopes.rs` gains "and its value bounds").
- `MaskOp { And, Or }`, `BinaryOp::on_masks`, `ScheduledOp::is_mask` (crate-private): the one statement of "a comparison is a
  mask, and `BitAnd`/`BitOr` of two masks is a mask".

### `emit/`
- `traffic.rs`: `Trips`, `EmitTraffic.trips: Vec<Trips>`, `dynamic_memory_ops -> Option<u64>`, the tests of spec 8.4.
- `mod.rs`: `pub use traffic::{..., Trips}`; `Loop.trips: Trips`; the module doc's loop-nest section gains the value-bounded
  paragraph, written with the E3/E4 corrections (below). `build.rs`: `open_loop(.., trips: Trips)`.
- `select.rs`: `UNIT_STRIDE`, `Looped { fold, lo, hi, body }`, `Selector::bound`. `fold` reads `KnownRange::of(fold, lo, hi)`
  (an `Option`), seeds the counter with `bound(lo)`, steps by the range's stride or `UNIT_STRIDE`, ends at `bound(hi)` after
  the body's frame has popped, and nests `Trips::of(range)` in the parent's. For literal bounds the LaneOp sequence and its
  order are today's, which gates 1/1b/2/2x confirm.
- `regalloc/local.rs`: `LoopFacts.trips` is `l.trips.weight()` (still a `u64`; doc says an estimate's price).
- `avx512.rs`: `KOp::of` is `BinaryOp::on_masks` mapped (same two ops, same encodings), and `Lane`'s doc points at
  `ScheduledOp::is_mask`. This is the whole emitter change.

### Docs and intent
`program/intent.md` (Owns, Never, Enforced by: the `*_fails_loudly` tests are named, not counted: ruling C6),
`emit/intent.md` and `codegen/intent.md` (Contract names `Trips`).

## The corrected cells (E3, E4), measured

Spec 3.4 said the latch compares against a slot and that the guard's opmasks are short-lived. Measured with the compile's own
traffic, kernel as written (`emit::compile`), `[64, 4]`:

- **Latch (E3).** A compare reads two registers, so a `hi` the allocator does not carry is reloaded at every latch. The
  48-term wall over `Σ_{k<7}`: fold scope `loads` 45 with a value `hi`, 44 with a literal, on AVX2 and AVX-512 alike (the
  literal's end is a remat, one fewer on AVX2). Pinned by `a_bound_that_is_not_carried_is_reloaded_at_every_latch`
  (`value == literal + 1`).
- **Register and frame (E4), P-c.** `X · Σ_{k<n}(X + k)`: dom/run are `CONST`, computed in the body, read by the arm tests in
  every batch. AVX2: column fold `loads` 1 against 0 for the literal, frame 96 against 48 bytes. AVX-512: `loads` 4 against 0,
  frame 64 against 48 (the opmask file has no carry budget, so each mask is stored at its definition and reloaded with
  `kmovw` per batch; the narrow slots come out of the peak). Pinned, in sign and not in number, by
  `a_guard_computed_once_per_call_is_read_again_in_every_batch`. The module doc says all of this instead of the table's cells.
- P-b, `X + Σ_{k<n} k` (the loop is a body-level fold under the arm blocks): the written fold's trips are the one
  `Estimated(32)` in the kernel, and `dynamic_memory_ops` is `None` (`a_fold_that_reads_no_coordinate_is_guarded_once_per_call`).

## Tests added

`program/lower.rs`: `a_value_bound_is_an_operand_of_its_fold`, `an_unguarded_value_bounded_fold_fails_loudly`,
`a_value_bounded_fold_in_the_false_arm_fails_loudly`, `a_value_bounded_fold_with_two_guards_fails_loudly` (F1/E2: expects
"and by", so the shape message cannot satisfy it), `a_value_bounded_fold_nothing_reads_fails_loudly`,
`a_reader_that_is_not_reachable_is_not_a_reader`, `a_bound_on_a_lattice_binder_fails_loudly` (C7/E5: the `Write` is built with
`Ir::embed(Shape::Write { .. })`, inside a literal `SEQ` lane fold), `a_bound_is_refused_exactly_when_it_computes_a_mask` (a table: comparison,
`BitAnd`/`BitOr` of masks at depth, `Bits(mask)` refused; `mask & 3.0`, `u + 1`, a uniform admitted), `a_bound_that_is_a_mask_fails_by_name`.
`pipeline.rs` (C9: `[8, 2]` at 4 lanes): `a_guard_on_a_uniform_count_is_computed_once_per_call` (4 compares, 0 `If`s in the body;
the written fold's parent is the column fold, which holds the `Write`, and so are the two guard `If`s),
`a_fold_whose_body_reads_no_coordinate_is_carved_in_the_body` (fold parent `Scope::Body`; the lattice's folds' `outer` names
the outer `If` and never the `Reduce`).
`emit/traffic.rs`: the six of spec 8.4. `emit/avx512.rs`: `a_value_is_an_opmask_exactly_when_is_mask_says_it_is_a_mask` (selects
every `BinaryOp` over mask/number operands and compares the lane file with `ScheduledOp::is_mask`). `emit/mod.rs`:
`every_backend_emits_a_fold_whose_bound_is_a_value` (Neon, Avx2, Avx512 from this host: a seed that is another loop's counter,
a wall over a uniform bound, a ragged `[37, 3]` row).
`tests/value_bounds_jit.rs` (22, through `within`, a 60 s watchdog that resumes a worker's panic): the whole of spec 8.2 plus
F3/C10 (`each_monoid_runs_under_its_guard` expects the identity at `n = 0`: `-inf` MAX, `+inf` MIN, `1` PRODUCT; through
`run_raw` **and** `run_linked`, via `both`), F4 (`an_inner_fold_starting_at_the_outer_index`: `Σ_{i<8} Σ_{j∈[i,8)} (X + j)`, both
runners), E1 (`a_mask_and_a_number_is_a_count_on_every_tier` runs `BitAnd(Lt, 3.0)` as `lo` and as `hi`;
`a_bound_that_is_a_mask_is_refused_by_name`), E3/E4 (the two traffic pins), and four of my own: `a_ragged_row_runs_the_guard_in_both_column_folds`,
`two_counts_nest`, `a_bound_the_body_reads_too`, `a_literal_lo_beside_a_value_hi`.

## Judgement calls, with the alternative rejected

- **J1. E1 is exact, not a list.** The emitter decides a lane is an opmask in two places: a `Compare` (always) and
  `Binary(op, Opmask, Opmask)` when `KOp::of(op)` exists. I made `BinaryOp::on_masks` the one definition of which ops keep a mask a
  mask; `KOp::of` is now that function mapped, and `ScheduledOp::is_mask` is the same rule over a schedule. The compiler
  forces a third `MaskOp` to be given an encoding. Rejected: a root-op list in lowering (a partial list, A3's S1 ruling), and the
  documented fallback (not needed; the `Lane` file is a pure function of the op and its operands' files, so it can be extracted).
  A test per side pins the agreement. The fallback's "pin the AVX-512 panic" is therefore moot: the refusal comes first.
- **J2. The mask question is asked of the lowered defs, not of arena nodes.** `Bits(Lt)` has no def (lowering aliases it to its
  operand), so on AVX-512 it *is* the opmask lane; an arena-node predicate would have to re-derive the erasure. Rejected:
  filling a mask table for every kernel (new work on the literal path); it is extended lazily, only when a value bound asks.
- **J3. P2 counts edges, not reader nodes.** `If(m, R, R)` has one reader node and reads `R` from both arms, which owns it outside
  the arm. Rejected: a `Vec<u8>` of counts (cannot name both readers) and per-node dedup.
- **J4. The traffic pins compare with a literal twin** rather than fix absolute load counts, so a harmless allocator change
  moves both sides. The `+1` is exact today; if an allocator change breaks it, the message names what it meant.
- **J5. `Looped`** keeps `Selector::fold` at three parameters (CLAUDE.md), one field per `ScheduledOp::Reduce` field.
- **J6. The NEON stand-in** is a cross-backend compile, since no target or emulator exists here. It proves selection, allocation
  and assembly succeed for the new shapes (the label/branch-range, frame and class paths), not that the words run.
- **J7. Test-only helper constants.** The lowering tests declare their count as uniform 0, so `RAW_ORIGIN` (`UniformId(0)`, `(1)`)
  would read it as an origin axis; they name `AFTER_ONE_UNIFORM` instead. The mask table uses `catch_unwind`, so a passing run
  prints the refusals' panic lines under `--nocapture`.

## Departures from the spec and the rulings

1. Spec 3.4's table is not reproduced anywhere in the tree; the corrected facts are in `emit/mod.rs`'s module doc and the two pins.
2. P1's first half (`admits_bound`: the fold's own binder, a coordinate) has no test: `ExprArena::push_reduce` refuses both before
   any arena reaches lowering, so it cannot fire from a builder. The lattice half has one.
3. `a_bound_on_a_lattice_binder_fails_loudly` builds only the lane fold around the `Write` (lowering needs no row/column folds).
4. The spec's `a_guard_on_a_uniform_count...` says "every `Compare` def is in `scoped.body.schedule`": the count is 4 (three domain
   tests, one `Lt`), asserted.
5. More tests than the spec lists, all in the files it names (plus `emit/avx512.rs`'s, because the exactness claim needs one on each
   side).

## Left for the others

- `pixelflow-pipeline/src/collapse_bench/mod.rs` (`:313` `trips` is a `Trips`; `:329` `dynamic_memory_ops` is an `Option`) and
  `pixelflow-graphics/examples/constant_traffic.rs` (`:50`) do not compile: spec 4.4 gives the fix (`trips.known()` and
  `dynamic_memory_ops()` with `.expect(LITERAL_FOLDS)`; `map_or_else(|| "an estimate".to_owned(), |n| n.to_string())`).
  `cargo check -p pixelflow-pipeline -p pixelflow-graphics --all-targets` shows these three errors and no other from codegen's
  types (pipeline's later targets are not reached until its lib compiles, so there may be more of the same kind).
- Checklist rows for codegen: `select.rs` (`definition`'s Reduce arm and `fold`), `program/mod.rs` (`ScheduledOp::Reduce` doc, `operands`)
  and `lower.rs`'s surviving-fold arm are `changed-by-A4`; `select.rs`'s `matches!(.., Reduce)` counting arm, `structural_children`
  and `scopes.rs`'s `is_fold`/closure-walk matches are already correct (they match the def and read no bound).
- The plan's Done line may cite: byte_probe identical on both tiers, gates 1b/2/2x identical, Ir +0.061 %; P2 counts readers;
  a mask-valued bound is refused through the predicate the AVX-512 selector uses.
