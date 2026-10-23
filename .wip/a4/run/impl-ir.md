# A4 implementer 1: pixelflow-ir

Base a747770; design: `rulings.md`, then `spec.md`, then the three critiques. Scope: the key's Value marker,
the guard pass, `Kernel::over_values` public, the VALUE_BOUND/KNOWN_RANGE texts, docs under changed code,
pixelflow-ir's tests. pixelflow-ir is green alone; other crates were not touched.

## Gates run (all green)

| Gate | Result |
|---|---|
| `cargo test -p pixelflow-ir` (lib, 4 integration files, doctests) | 309 lib; integration: 1 `dag_scratch_allocation`, 75 `dyadic_spec` (1 ignored), 9 `value_bounds`, 1 `variance_empty_fold`; 1 doctest (the new `over_values` one) |
| `cargo +1.99.0 clippy -p pixelflow-ir --all-targets --all-features -- -D warnings` | clean (target dir `/home/user/scratch/a3-close1/t199`) |
| `RUSTFLAGS="-D warnings" cargo check -p pixelflow-ir --no-default-features` | clean (CI's command). `--tests` under that flag fails at the base too (`by_ref` needs `std`): not a CI job, not mine |
| `cargo fmt -p pixelflow-ir -- --check` | clean |
| rustdoc, `--no-deps --document-private-items`, `rustdoc_norm.py --compare` | base 38, head 38, NEW 0 |
| `scripts/check-intent.sh`, `check-doc-paths.sh`, `check-cfg-encapsulation.sh` | OK (`pixelflow-ir/intent.md` is 80 lines) |
| `cargo check -p pixelflow-search --all-targets`, `cargo test -p pixelflow-search --lib` | compile; 397 passed, 8 ignored (a cross-crate check that the IR changes break nothing there) |
| greps in pixelflow-ir: `A4`, `VALUE_BOUND`, `has no bits")`, `expect(dead_code, reason = "pub(crate) until A4` | nothing |

Byte identity (byte_probe, the 29 font programs, telemetry, Ir) is the integrator's gate: it needs codegen. By
construction here: a literal pair's `to_bits` is the old expression term for term; `from_bits` takes the old branch
for two literals; `guard_value_bounds_owned` moves the arena back when no node is value-bounded (no clone, no
rebuild). The only new work on a literal kernel is one `arena.nodes().any(..)` scan per `legalize`, the pattern of
the three scans `legalize` already makes before it (each of which also clones the arena).

## What changed

### `fold.rs` (the key)
- `VALUE_FIELD = u32::MAX` (private), `count_field(Bound<()>) -> u32` and `field_bound(u32) -> Bound<()>` (private
  inverses), `admits_a_value(Fold, Bound<()>, Bound<()>) -> bool` (private, the one predicate).
- `FoldHeader::to_bits` is total; `from_bits` reads the marker. Inverse law: `from_bits(h.to_bits()) == Some(h)`.
- The const assert is `Fold::EXACT_BOUND < VALUE_FIELD as u64`, its comment rewritten.
- `VALUE_BOUND` deleted. `KNOWN_RANGE` text and doc as the spec says. `BoundOutOfDomain`'s doc, `FoldHeader`'s doc,
  `store`'s doc (its callers now include `from_bits`) updated.
- `from_bits` is a guard clause: a value bound returns through `admits_a_value`; the literal range code that follows is
  the old expression with `lo`/`hi` renamed `first`/`end`.

### `passes/guard.rs` (new) and `passes.rs`
- `guard_value_bounds_owned(ExprArena, ExprId) -> (ExprArena, ExprId)` (`pub(crate)`), the last line of `legalize`:
  `Ok(guard::guard_value_bounds_owned(arena, root))` after `expand_transcendentals_owned`.
- The hook pushes the rebuilt `Reduce`, then `guard(arena, folded)` builds, in the spec's push order: domain tests
  (`zero`, `top = 2^24`, `Le`, `Le`, `Floor`, `Eq`, `BitAnd`, `BitAnd`) per distinct value-bound term, `dom`, `start`/`end`,
  `runs = Lt(start, end)`, `identity`, `ran = If(runs, R, identity)`, `nan`, `guarded = If(dom, ran, nan)`.
- `passes.rs` module doc: seven passes, the table's last row, "the guard goes last", "every pass but the guard is
  idempotent". `legalize`'s doc: one sentence on the guard.

### `kernel.rs`
- `over_values` is `pub`; both `expect(dead_code)` attributes are gone. Doc: the run-time law paragraph, with the backwards
  pair as "the identity, up to the factoring edge `FactorFold` accepts at a runtime-empty range (decision 5)" (F5), and a
  doctest. `Kernel::over` links `Kernel::over_values`.

### `arena.rs`, `key.rs`
- `arena.rs`: the `push_reduce` doc's "(L5, A4)" names the guard. `key.rs`: docs only (the value bound's marker and
  child; "a fold's header says how many children follow it"; a uniform bound is a dense slot, so every count is one key).

### `intent.md`
- The fold bullet gains the domain (NaN outside `[0, 2^24]` by `passes::legalize`'s guard); the `Var(u8)` bullet loses
  its "until A4" sentence. Exactly 80 lines (the spec's "wrapped so the file stays at 80" needed a tighter sentence than
  the spec's wording: the two edits net +1 line otherwise).

## Tests added (all in pixelflow-ir)

- `fold.rs`: `a_value_bound_writes_the_marker_and_reads_back` (replaces `a_value_bound_has_no_bits_until_a4`),
  `a_value_header_decodes_from_its_layout_assembled_by_hand` (both ends), `an_end_field_of_all_ones_is_a_value_bound`
  (replaces `from_bits_refuses_an_end_field_of_all_ones`), `from_bits_refuses_a_marker_its_fold_cannot_take`,
  `from_bits_inverts_to_bits_on_every_header_shape` (7 monoids x 7 shapes that `new` admits), and
  `new_panics_exactly_where_the_predicate_refuses` (see J2).
- `key.rs` (C1): `a_value_bounded_folds_key_record_is_tag_ten_its_bits_and_its_children` (bytes by hand),
  `every_count_is_one_key` (equal `canonical(..).key`; **different** `KernelKey`s), `which_end_is_a_value_is_in_the_key`.
- `passes/guard.rs`: the six of spec 8.1 (`a_kernel_with_no_value_bound_moves_back_untouched`,
  `a_value_bounded_fold_is_read_only_by_its_guard`, `a_literal_bound_adds_no_domain_test`,
  `two_value_bounds_are_each_in_the_domain`, `the_guard_varies_with_the_bounds_and_the_fold_with_its_body`,
  `the_identity_arm_is_the_monoids`) plus three: `a_bound_on_an_enclosing_index_is_guarded_inside_that_fold`,
  `a_literal_outside_the_domain_is_still_guarded` (a `Const(2.5)` bound stays a value and is guarded, never panics),
  `a_parent_of_the_fold_reads_its_guard`.
- `passes.rs`: `legalize_guards_a_value_bounded_fold_inside_the_lattice` (`X + sum_{k<u} k` at `[4, 2]`, 4 lanes).
- `tests/value_bounds.rs` (public API): module doc fixed; `a_value_bounded_kernel_has_a_key` (replaces
  `..._has_no_key_until_a4`; C1: byte-equal `canonical(..).key` across defaults, `uniforms` differ, `KernelKey`s differ);
  `over_values_is_public`; `over_values_refuses_arithmetic_at_the_public_surface`; and
  `legalize_guards_what_over_values_built` (user door to guard: one value-bounded fold in the legalized kernel, one reader,
  an `If`'s true arm).

## Judgement calls, with the alternative rejected

- **J1. The test for the legalize shape walks from the new root.** `rebuild_arena` appends to the arena and leaves the nodes it
  replaced. My first `legalize` test took the first `Write` in `nodes()` order and found the stale, unguarded one. The tests
  walk reachable nodes only. Rejected: compacting the arena after the guard, which would change ids for no reader's benefit.
  *This is the thing implementer 2 must hold to:* `value_folds` and the P2 reader count must scan reachable ids only (the
  spec says so; the stale `Add(x, R)` parent would otherwise be a second reader of `R`).
- **J2. `admits_a_value` and `with_a_value`'s three asserts are two statements of one rule** (the spec keeps both so a panic
  names the failing part). Rather than leave their agreement to parallel test lists, one test runs `FoldHeader::new` under
  `catch_unwind` for 7 monoids x 2 strides x 5 bound pairs and asserts it panics exactly when the predicate is false. It is
  `#[cfg(feature = "std")]`, as the crate's other `std`-needing tests are. Rejected: deriving `with_a_value` from the
  predicate (loses the named panic), or a `Result`-returning checker that both call (a third function for a rule that
  has one caller each way).
- **J3. `guard(arena, folded)` takes the already-pushed fold and reads its bounds back from the node** (the spec's
  sketch), rather than `(arena, fold, lo, hi, body)`: five arguments against CLAUDE.md's "fewer than four", and the
  read-back is what lets the pass return a fold unguarded if discovery made both bounds literals.
- **J4. `dom` is built by a three-way `match (lo.value(), hi.value())`**, not a loop over an `Option` accumulator: the
  `(None, None)` arm is the "discovery made it literal" return, so there is no `expect` for "at least one value bound".
  `(Some(l), Some(h)) if l != h` is the two-test case; equal terms fall to the one-test arm.
- **J5. `from_bits` uses a let-else guard clause** (a value pair returns through the predicate) rather than the spec's
  `match (lo, hi)`: the literal path stays the old statement, a diff reviewer sees it unchanged.
- **J6. Public doc links for `over_values` are the short `[`BoundRefusal::X`]`**, not the explicit paths A3's ruling named:
  rustdoc's `redundant_explicit_links` (warn by default) fires on an explicit path when the short one resolves, and
  `BoundRefusal` is imported in `kernel.rs`. The doc-warning gate (base 38, head 38) is what the explicit paths were for,
  and it holds. `close_over` (crate-private) is named in backticks, not linked, from the now-public doc.
- **J7. The `legalize` end-to-end pin at the public API** (`legalize_guards_what_over_values_built`) was added beside the
  spec's tests: it is the one test that goes user surface -> arena -> `legalize` -> guard shape without a private import,
  so a later visibility change cannot quietly detach the guard from `over_values`.

## Departures from the spec and the rulings

1. **`intent.md` wording** (above): the spec's phrase did not fit 80 lines; the shorter sentence says the same.
2. **Key tests are not the spec's two `KernelKey` equalities** (ruling C1, as directed): bytes equal, `KernelKey`s differ.
3. **The doctest is the spec's snippet** (`use pixelflow_ir::{Kernel, Monoid, Uniform}`); it lives in the `# Panics`-adjacent
   position, after that section, so it renders as the method's example.
4. The ruling items that live outside pixelflow-ir (F1/E2 reader count, F3/C10 monoid oracle, F4 `Σ_{j∈[i,8)}`, E1 mask-valued
   bound, E3/E4, C2-C9, C12, the plan and doc edits of spec 10) are not here: they belong to codegen, search, the
   serializers and the docs implementers.

## For the other implementers

- `pixelflow-codegen/src/program/lower.rs:12` and `:51` still import and use `fold::VALUE_BOUND`; it is gone. That site is
  spec 3.2 (the `known` helper goes with it).
- The guard emits `Const(Lit::Real(f32::NAN))` and `Const(2^24)` as ordinary pool constants; nothing in
  `pixelflow-codegen/src` refuses a non-finite constant (`grep is_nan\|is_finite` is empty there).
- `legalize` of a value-bounded kernel leaves the unguarded parents in the arena, unreachable. Anything that scans
  `arena.nodes()` rather than the nodes reachable from the root will see them.
- `canonical`/`KernelKey::of` no longer panic on a value bound, so `jit_cache::compile` now reaches `optimize_runtime_arena`
  and `legalize` with one (spec 3.5's remark about A3's e-graph pin applies).
