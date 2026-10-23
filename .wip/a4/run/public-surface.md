# A4 public surface

Every new or changed `pub` item, by crate. A section per implementer; an implementer
appends its own and edits no other. "Changed" means the item's signature, visibility,
behavior or documented contract differs from a747770.

## pixelflow-ir (implementer 1)

### Changed visibility (authorized: spec section 5, decision 7)

| Item | Before | After |
|---|---|---|
| `pixelflow_ir::Kernel::over_values(monoid: Monoid, lo: &Kernel, hi: &Kernel, body: impl FnOnce(&Kernel) -> Kernel) -> Kernel` (also `pixelflow_core::Kernel::over_values`, by re-export) | `pub(crate)`, with `#[cfg_attr(not(test), expect(dead_code, ..))]` | `pub`, `#[must_use]`, signature unchanged; the doc gains the run-time law (NaN outside `[0, 2^24]`, the identity for a backwards pair up to `FactorFold`'s edge) and a doctest; its `# Panics` links are the `BoundRefusal` variants |

`Kernel::user_bound` stays private (its `expect(dead_code)` went: it now has a `pub` caller).

### Removed

| Item | Why |
|---|---|
| `pixelflow_ir::fold::VALUE_BOUND` (`pub const &str`) | Its three readers are gone: `FoldHeader::to_bits` is total, and lowering (implementer 2) schedules a value bound. A caller that imported it stops compiling by name. |

### Changed behavior or contract (no signature change)

| Item | Change |
|---|---|
| `pixelflow_ir::fold::FoldHeader::to_bits(self) -> u128` | Total. A bound that is a value writes the all-ones count field (private `VALUE_FIELD`, `u32::MAX`); a literal pair writes what it always wrote, bit for bit. Was: panics with `VALUE_BOUND` on a value bound. The `# Panics` section is gone. |
| `pixelflow_ir::fold::FoldHeader::from_bits(bits: u128) -> Option<FoldHeader>` | Reads a count field of all ones as a value bound, `None` where its fold cannot take one (under `ANY`/`ALL`/`SEQ`, at a stride other than 1, beside a literal past `Fold::EXACT_BOUND`). Was: `None` for any all-ones field. `from_bits(h.to_bits()) == Some(h)` now holds for every header. |
| `pixelflow_ir::fold::KNOWN_RANGE` (`pub const &str`) | Text only: `"this fold's bounds are literals: a lattice's fold, a production glyph's before B2, or a test's (docs/language/2026-10-20-a-bound-is-a-value.md)"`. Was: also named `kernel!` before A6 and "a schedule's". |
| `pixelflow_ir::passes::legalize(arena, root, collapse)` | A fold with a value bound comes back inside `If(dom, If(Lt(lo, hi), fold, identity), NaN)`; the fold has exactly one reader, the inner `If`'s true arm. Identical (the arena moved back, not rebuilt) when no node is such a fold. Signature unchanged. |
| `pixelflow_ir::KernelKey::of`, `pixelflow_ir::key::canonical` | Total over a value-bounded fold (they reach `FoldHeader::to_bits`). Was: panicked by name. A uniform bound contributes its dense slot, so every count is one `Canonical::key`; `KernelKey`s of two uniform instances still differ (the instance is in the kernel's identity). |

### Doc-only

`ExprArena::push_reduce`, `fold::BoundRefusal::BoundOutOfDomain` (no longer name "A4"), `Kernel::over` (links `Kernel::over_values`), the `passes` module doc (seven passes, the guard last, every pass but the guard idempotent), `key::canonical`'s doc (a fold's header says how many children follow it).

### Not public (listed so a reviewer can see nothing leaked)

- `passes::guard` is a private module; `guard_value_bounds_owned` is `pub(crate)`; `has_a_value_bound`, `guard`, `in_domain`, `term` are private.
- `fold.rs`: `VALUE_FIELD`, `count_field`, `field_bound`, `admits_a_value` are private.
- No new `pub` item in pixelflow-ir. No `pub` field, no re-export, no visibility widening beyond `Kernel::over_values`.

## pixelflow-codegen (implementer 2)

### New (authorized: spec section 3.3)

| Item | Visibility and shape |
|---|---|
| `pixelflow_codegen::emit::Trips` | `pub enum Trips { Known(u64), Estimated(u64) }`, derives `Clone, Copy, Debug, PartialEq, Eq`; re-exported by `emit/mod.rs` beside `EmitTraffic` and `ScopeTraffic`. `Known(n)`: every bound a literal, the body runs `n` times. `Estimated(n)`: a bound is a value, here or in an enclosing fold, and `n` is the price (`Fold::ESTIMATED_TRIPS`, nested products). |
| `Trips::known(self) -> Option<u64>` | `pub`, `#[must_use]`. `Some(n)` for `Known(n)`, `None` for an estimate. |
| `impl Display for Trips` | `Known(n)` prints as the bare number, the formatter's width and fill honoured (`byte_probe`'s `trips={trips}` and `constant_traffic`'s `{trips:>8}` print what they did when the field was a `u64`); `Estimated(n)` prints `~n` through `Formatter::pad`. |

### Changed signature of a public item (authorized: spec section 3.3)

| Item | Before | After |
|---|---|---|
| `pixelflow_codegen::emit::EmitTraffic::trips` (pub field) | `Vec<u64>` | `Vec<Trips>`; doc: a fold whose range is not known, and every fold inside it, is an estimate |
| `pixelflow_codegen::emit::EmitTraffic::dynamic_memory_ops(&self)` | `-> u64` | `-> Option<u64>`: `None` when any scope's trips are an estimate, so no measurement shows an estimate as a count. `#[must_use]` kept |

Two callers outside this crate stop compiling until they handle the new types, both named by spec section 4.4 and **not touched here**: `pixelflow-pipeline/src/collapse_bench/mod.rs` (`:313` multiplies a `Trips` by a `u64`, `:329` stores the `Option`) and `pixelflow-graphics/examples/constant_traffic.rs` (`:50` prints the `Option` with `{}`).

### Changed behavior or contract (no signature change)

| Item | Change |
|---|---|
| `pixelflow_codegen::emit::compile` (re-exported from `pipeline::compile`; the entry `pixelflow-core`'s compile and bake reach through `jit_cache`) | A fold whose bound is a value compiles, inside its guard, and computes its fold or NaN on every tier. A bound that computes a mask panics with `BoundSort` on every tier (a new sentence in the `# Panics` section); the pipeline-bug panics (a bound reading a lattice binder, a value-bounded fold read other than once) are for an arena `legalize` did not produce, and this entry always runs it. Was: panicked with `VALUE_BOUND` in lowering. |
| `pixelflow_codegen::jit_cache::compile` | Accepts a value-bounded kernel (`Kernel::over_values`): `canonical` is total over it, the optimizer keeps its bound, and every count of one kernel shares one program. Was: panicked in `canonical` (and before that in lowering). A computed mask bound is not reachable here (`Kernel::over_values` refuses a computed bound), so no new panic. The doc gains one sentence. |

### Crate-private additions (listed so a reviewer can see nothing leaked)

- `program/mod.rs`: `pub(crate) enum MaskOp { And, Or }`; `BinaryOp::on_masks<M>(self, Option<M>, Option<M>) -> Option<(MaskOp, M, M)>` (`pub(crate)`): the one place that says a bitwise op on two masks leaves a mask, called by AVX-512's selector (with each operand's opmask) and by `ScheduledOp::is_mask(&self, impl Fn(ValueId) -> bool) -> bool` (`pub(crate)`, with the schedule's answer). `#[cfg(test)] ALL` on each `legal_ops!` enum (`pub(crate)`, test builds only), which the AVX-512 agreement test iterates. `program::operands` (already `pub(crate)`) now yields a `Reduce`'s value bounds.
- `emit/traffic.rs`: `Trips::ONCE`, `Trips::of`, `Trips::nested_in`, `Trips::weight`, all `pub(in crate::emit)`.
- `emit/select.rs`: private `UNIT_STRIDE`, `Looped`, `Selector::bound`. `emit/build.rs`: `Builder::open_loop` takes a `Trips` (`pub(super)`). `emit/mod.rs`: private `Loop.trips: Trips`. `regalloc/local.rs`: `LoopFacts.trips` stays a `u64`, read through `Trips::weight`.
- `program/lower.rs`: private `ValueFold`, `ValueFolds`, `MaskValues`, `assert_off_the_lattice`.
- `emit/avx512.rs`: `KOp::of` is `impl From<MaskOp> for KOp` (same ops, same encodings; no byte moves); `Lane::opmask` (private) is a lane as the mask it is held in, if it is one.

### Removed

Nothing public. Private: `lower.rs`'s `known` helper (the VALUE_BOUND refusal) and `select.rs`'s `KNOWN_RANGE` expect.

### Not widened

No `pub(crate)` item became `pub`; no `pub` field was added; `ScopedSchedule`/`ScopeFold` fields keep their `pub(super)`. `push_write` is untouched (the test builds its `Write` through `Ir::embed`).

## pixelflow-pipeline, pixelflow-compiler, pixelflow-graphics, pixelflow-search (implementer 3)

No new `pub` item, no `pub` field, no visibility widening, no re-export. Every change is a behavior or a
documented contract of an item that already existed, or private.

### Changed behavior or contract (no signature change)

| Item | Change |
|---|---|
| `pixelflow_pipeline::collapse_bench::features_of(&CompileResult) -> StaticFeatures` | Panics (new `# Panics` section) on a kernel with a fold whose range is not known: its trip counts are estimates, and `EmitTraffic::dynamic_memory_ops` is `None`. Reads both through the private `LITERAL_FOLDS` message. Was: summed `u64` trips and a `u64` count. A kernel whose bounds are all literals (every corpus kernel) is unchanged. |
| `pixelflow_pipeline::collapse_bench::corpus::{encode, read_dir, write_dir}` (and the private `decode`) | The fixture format is v5: an `R` line is `R <header bits> <body>` then one id per bound that is a value, `lo`'s first. A literal fold's line is what it was. A v4 file is refused by name (`SUPERSEDED_HEADERS`). |
| `pixelflow_pipeline::training::corpus::{write_corpus, read_corpus}` | A `Reduce` record (tag 9) is the header's bits, the body's id, then one id per bound that is a value, `lo`'s first. `write_corpus` no longer refuses a value bound. `CorpusFormat::SCHEMA` (private) names tag 9, so `corpus_identity()` moves and every earlier corpus is refused with the regeneration hint. |
| `pixelflow_compiler::emit::arena_to_tokens(&ExprArena, ExprId, &EntryContext) -> TokenStream` | A fold with a bound that is a value emits `attach([child ids])` with those children after its body; a literal fold's tokens are unchanged (`attach(::core::iter::empty())`), as is an open fold's hole filling. `kernel!` cannot produce a value bound before its bounds are typed expressions, so only a hand-built arena reaches the new branch. |

### Private, listed so a reviewer can see nothing leaked

- `collapse_bench/mod.rs`: `const LITERAL_FOLDS`. `collapse_bench/corpus.rs`: `HEADER` is v5; `SUPERSEDED_HEADERS` gains v4.
- `bin/corpus_gaps.rs`: `Key::Reduce(u128, Vec<u32>)` (was `(u128, u32)`) in the private `hash_cons`.
- Tests and one example only elsewhere (`pixelflow-graphics`: `cell_grid.rs`'s and `production_glyph_arena_dump.rs`'s
  `dump_arena` stay v1 and write a value bound's child; `examples/constant_traffic.rs` prints an estimate as "an
  estimate"). `pixelflow-search`: docs and one intent line only; no code.
