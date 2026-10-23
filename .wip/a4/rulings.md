# A4: rulings on the spec and its critiques (authoritative over spec.md where they differ)

Base: main a747770 (A3, #1447). `spec.md`'s file:line references are at a747770. Re-grep before trusting one. `F#` = critique.semantics.md, `C#` = critique.completeness.md, `E#` = critique.emitter.md.

## The spec's open decision: the extraction-side hole (§6)

**Ruled: (a), leave it loud.** No extraction change in A4.
- The hole fails loudly, never silently: a bound class whose chosen member reads a coordinate panics with `BoundVaries` at materialization, by name.
- (b), constraining a bound class's pick, adds about 40 lines. It trades that loud failure for another (F2, C11): a class read as a bound by some candidate, with no member that keeps L3, never settles, and every other user of the class fails with it, including users that (a) extracts lawfully.
- The coordinate-reading member has to be the class's *cheapest*. Such a member spells X with a cancellation, so it carries extra operations; no profile or kernel shows one winning.
- Subtract before you add: the fix waits until a real kernel hits it, or until T6 gives a bound a sort.
- Drop §3.5 and §8.5. In the egraph `intent.md`, write one line in place of the (b) line. Keep A3's pin that a bound class with no lawful member is refused by name, and add one pin: a class whose cheapest member reads a coordinate *beside* a dearer lawful one also panics by name (`Pow(i,2)` beside `((i·i)+X)−X`, no rules). That makes the documented hole a tested one.
- The plan's A4 bullet says A4 decided (a), with this reason in one sentence. BACKLOG gets a row: "extraction picks one member per class; a value bound's pick is unconstrained (A4 decided loud); revisit with T6 or a kernel that hits it."

## The spec's other calls: accepted

- The guard is the last pass of `pixelflow_ir::passes::legalize`, in `passes/guard.rs`, shaped `If(dom, If(Lt(lo, hi), R, identity), NaN)`. `dom` uses only ordered `Le`, exact `Eq` and `Floor`.
- Lowering keeps the post-collapse asserts. P1: a bound reads no lattice binder. P2: see F1/E2 below.
- `Trips { Known(u64), Estimated(u64) }`; an estimate prints as `~n`.
- The key marker is `u32::MAX` in the count field, read back through `admits_a_value`.
- The collapse corpus goes to v5, and the training corpus schema text changes. The arena dumps stay at v1, because no reader parses an `R` line.
- `hi` is kept live by the latch's read, with no carve change beyond `operands()` yielding the bounds.
- NEON runs in CI's macOS job only. The 2²⁴ in-domain case stays (16.7 M trips per call).

## Blocking

- **F3 / C10.** In `each_monoid_runs_under_its_guard`, the oracle at `n = 0` is the monoid's identity (−∞ for MAX, +∞ for MIN, 0 for SUM, 1 for PRODUCT); the formulas are kept for `n ≥ 1`. Name the runner: through `run_raw` and `run_linked` both, where the harness has both.
- **C1.** The key tests assert byte equality of `canonical(..).key`. They assert that `KernelKey`s of two uniform instances *differ*, because a uniform's identity is part of the key.
- **C2.** The collapse-corpus round trip asserts `encode(&kernel) == encode(&decoded)`, not `subtree_eq`.
- **C3, C4, C12.** Fix the three §9 greps:
  - Grep the exact old text (`has no bits")`).
  - Anchor `known(` on the deleted helper (`fn known(bound`), or list `traffic.rs` among the expected hits.
  - Name both header-writer sites the `to_bits` grep misses.

## Must do

- **F1 / E2 — P2 counts readers.** Beside `value_folds`, count the reachable readers of each value-bounded fold. Assert exactly one, an `If` whose true arm is the fold. On a second reader of any shape, panic naming both. This is the single-reader property placement rests on, so it must be what the assert checks.
- **F4.** Add `Σ_{i<8} Σ_{j∈[i,8)} (X + j)` (a value `lo` reading an enclosing index), run through `run_raw` and `run_linked` on avx2 and avx512, and on NEON in CI. It is the first test where a loop parameter is another loop's entry argument (the `LinearScan::pass` → `home` → `copy_of` path).
- **F5.** `over_values`' public doc reads: a backwards pair is "the identity, up to the factoring edge `FactorFold` accepts at a runtime-empty range (decision 5)".
- **E1 — a mask-valued bound, one definition.** On AVX-512 a value bound whose lane the emitter holds as an opmask panics in selection, while AVX2 and NEON read its bits as a float. Refuse it in lowering, by name, on every tier, *using the same predicate the emitter uses to decide a value is a mask*: one definition, so the refusal is exact for the emitter, not a list of root ops.
  - If no such single predicate exists (the classification lives only inside the AVX-512 selector), extract it into a shared function first and have both call it.
  - If even that cannot be exact, do not add a partial list (A3 ruled the same on S1). Keep the hole documented, with the tier divergence stated (AVX-512 panics in selection; AVX2 and NEON read the bits), and pin the AVX-512 panic.
- **E3, E4.** Correct §3.4's latch, register and frame cells as the findings say. Any traffic pin on a value-bounded kernel expects the per-trip reload of a `hi` that is not carried, and the narrow-slot frame growth for the guard's live masks under P-c on AVX-512.
- **C5.** Update `fold_rules.rs`'s two docs that A4 makes false (≈:544 and ≈:1812): "by the guard `passes::legalize` ends with".
- **C6.** `program/intent.md` names the `fails_loudly` tests instead of counting them.
- **C7 / E5.** The hand-built `Write` in the codegen unit test is built with `pixelflow_ir::Ir::embed(&mut arena, Shape::Write { row, col, lane, value })`. `push_write`'s visibility does not widen.
- **C8.** `corpus_gaps` gets a new `#[cfg(test)] mod tests`. Keep only the half that can fail: two folds that differ only in their bound's child stay two nodes.
- **C9.** The pipeline tests state a valid shape and lane count (for example `LatticeShape::new([8, 2])` at 4 lanes).

## Fixed constraints

- **Byte-identical on today's kernels.** Every bound is a literal: byte_probe on both tiers, the 29 font programs' nodes, keys and digests, and saturation telemetry including the extraction fields. Total Ir within +0.5% of the a747770 base.
- **Clippy is CI's.** Run `cargo +1.99.0 clippy` (1.99 is installed beside the default 1.97). For `check-feature-matrix.sh`, put `/home/user/scratch/cargo-hack-root/bin` on PATH.
- **Docs and labels.** Every intent.md stays within 80 lines. No tracked file outside `.wip` cites a `.wip` path, a ruling label or a model name. The plan's A4 Done line states facts.
