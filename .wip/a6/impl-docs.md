# A6 docs: what changed

Markdown only. No `.rs` file was touched by this pass. Every sentence below was checked against the
code as it stood after the code agent's edits (`sema.rs` `type_of_fold`/`count_parameter`/
`bound_arithmetic`, `lower.rs` `lower_fold`, `emit.rs` `emit_entry`/`emit_args`, the new tests by name).
`bash scripts/check-intent.sh` and `bash scripts/check-doc-paths.sh` both print OK.

## Files

- `docs/language/2026-10-20-a-bound-is-a-value.md` (the plan)
  - §1.4 (host write): "at `kernel!`'s host write of a per-call count, an entry's `n: usize` (decision 2),
    by its host function and by its `Args` record's `write_into` (A6)". Paragraph reflowed.
  - §3.4: a bound's names are typed before the fold's scope is pushed and before arithmetic is judged;
    `lower_fold` resolves `lo`/`hi` before `open_fold`, which takes them; a known end beside a value is
    checked alone against 2^24; a helper and the closure form take no count. Reflowed to the page width.
  - §5 row for the-language-is-kernel §1.3: "or an entry's per-call `n: usize`".
  - §6 A6 bullet: "Waits on decision 2" became "Decision 2's spelling, an `n: usize` entry parameter".
    The **Done** line is appended and ends in the marker sentence `GATES: pending.` for the coordinator.
    It carries no byte_probe/keys/digests clause; that clause is the marker's to replace.
  - Not edited, as the spec says: §0, §2's row ("`kernel!` refuses `(0..4).map(|i| (0..i)…)`", the tree the
    plan was verified against), §8 decision 2's record, plan L2/L5 (the denotation, not a statement about
    `kernel!`).
- `docs/language/2026-09-25-the-language-is-kernel.md` (surface S1, complete C3)
  - `:52-59` reversal note: appended "A6 made a `kernel!` bound a value: an enclosing fold's index, or an
    entry's `n: usize`."
  - `:111`: "Until then the quote's rule, constant ranges, is what the tree does for a font; a `kernel!` bound
    has been a value since A6: an enclosing fold's index, or an entry's `n: usize`." "For a font" is true:
    the glyph path folds with `Kernel::over` over literal ranges (`loop_blinn.rs:423`) and composes pieces
    with `sum2`.
  - `:197` §1.3 `usize` row: count includes "an entry's per-call parameter `n: usize`"; IR column is
    "a binder's `Var(REDUCE_BINDER_BASE + slot)`, a `usize` const's `Const`, a count parameter's `Uniform`"
    (`lower_cast` builds `site.constant(count as f32)` for a const; `Binding::Count` holds the uniform).
  - `:262-264`: "`write_into` ... checks the arity (`ArityMismatch`); it panics on a count parameter past 2²⁴."
- `.claude/skills/ontology/language.md`
  - `kernel!` Follows: a count the host writes is refused past 2^24 by `KnownRange::admits`, in the entry's
    host function and its `write_into`.
  - Args Is: "(`f32`s, counts `n: usize` and records)". Follows: "checks only the arity and each count's
    domain". Lives: "`write_into` returns `Result<(), ArityMismatch>` and panics on a count past 2²⁴".
  - Binding time Is: a count parameter `n: usize` is a uniform, read as a fold's bound.
  - Structural parameter (retired): "which A6 builds" became "which A6 built" (G7 clean).
  - Fold Is: "In `kernel!` its bounds are `usize`s: a literal or a `usize` const, an enclosing fold's index
    (`0..i`), or an entry's count parameter `n: usize`."
  - Range Is: "(in `kernel!`, an entry's `n: usize`)". Range Is not (LAWS-4): "Past 2²⁴: a literal is refused
    at construction, a count `kernel!`'s host function or `Args` record writes is refused there; any other
    write is an `f32`, and one naming no count makes the fold NaN (E9 aside)."
- `pixelflow-compiler/intent.md` (63 to 67 lines, budget 80): Owns (the count is an entry's `n: usize`, a
  uniform whose host function and `write_into` panic past 2^24); Never ("whatever its counts"); Enforced by
  (`sema.rs` `a_range_bound_is_a_usize_and_reads_no_coordinate`, which exists); Behind (a value bound is a
  single name until bounds are typed integers: `0..i + 1` refused, `BOUND_PLAN` §1.4, T6); Decided in
  (adds §1.4). No step label.
- `pixelflow-core/intent.md`: not edited. Under arm (b) no public item is added, so the Contract's re-export
  list is unchanged.
- `pixelflow-core/README.md:64`: "(`Kernel::over_values` with a `Uniform`; in `kernel!`, an entry's
  `n: usize`)". Paragraph reflowed.
- `docs/BACKLOG.md`
  - `:32` plan-table row: A6 moved from "next" to "landed" with the spec's parenthetical; next is
    "A1's the-language-is-kernel D-table rows after #1323; B1–B5 after typed-arena T3 and item 2".
  - E9 (`:153`, LAWS-1): widened. `kernel!` takes the fold at expansion too: the macro tier's rule set is
    `RuleSet::production()` = `all_rules()`, which holds `Annihilator::new(OpKind::Mul)`
    (`math/algebra.rs:1108`; `contains_const` on either operand), so a zero factor written beside a
    value-bounded fold (`0.0 * (0..n).map(|i| ...).sum()`) takes the fold with it, with no `FactorFold`
    and no `jit_cache::compile`. `FactorFold` is runtime-tier only (`RuleSet::runtime()`), which is why
    the row says "beside", not "in the body". No test is claimed to pin it.
  - C8 (`:166`): "a count the host writes through `kernel!` is refused at the write (a-bound-is-a-value A6):
    the entry's host function panics past 2²⁴, and so does its `Args::write_into`, before writing anything."
- Skipped on purpose: `CLAUDE.md` and `docs/DENOTATION.md` (spec §7.3: `CLAUDE.md:180` and DENOTATION
  `:129-131` are already true), `.claude/skills/ontology/SKILL.md` (no entry added or renamed).
- Left to the code agent (`.rs`, spec §7.6/§7.8), both landed and agree with the docs above: `lib.rs`
  `# Folds`, `# Binding times`, `# The closure form`, `BOUND_PLAN`'s doc; `pixelflow-ir/src/fold.rs`
  `BoundOutOfDomain`'s doc.

## Rulings applied

- **surface S1, complete C3:** the three extra the-language-is-kernel edits (`:262-264`, `:52-59`, `:111`),
  each read against the code.
- **D-A6-1 arm (b):** `write_into` panics; nothing called `WriteRefused` appears in any file. Checked:
  `emit_args` keeps `Result<(), ArityMismatch>` and asserts before `set_declared`.
- **LAWS-1:** the Done line says the raw-write NaN holds "up to the two optimizer edges `Kernel::over_values`
  documents (BACKLOG E9's annihilation, decision 5's factoring)"; the ontology Range row says "(E9 aside)";
  E9 widened. Claim checked against `Annihilator::apply` and `RuleSet::{production, runtime}`.
- **LAWS-2:** the domain is the reason; it is in the plan's §1.4 already (no new sentence needed in these
  docs) and in the generated `# Panics`/`write_into` doc, which is the code agent's. "Refusal of plausible
  pixels" is in none of the files I wrote.
- **LAWS-3:** no doc here says a fold rule runs at expansion (grepped `KnownBound|HalveFold|PeelFold|FactorFold`
  with expansion/macro).
- **LAWS-4:** the ontology Range row, verbatim intent.
- **surface S3/S4/S6, stages S1-S5, complete C1/C2/C4-C7:** code or spec-only; no Markdown surface.

## Open items for the coordinator

- The Done line ends in `GATES: pending.`; replace it with the measured gate clause (keep no clause no gate
  checked).
- `CLAUDE.md:180` and `pixelflow-core/README.md:67` say "a reduction with a bound that is not [a count] is
  NaN" with no E9 qualifier. That is A4's wording, about folds in general, so LAWS-1's "for `kernel!`" did
  not reach it and I left it. If the coordinator wants LAWS-1 read wider, those two lines and the ontology
  Fold "Follows" bullet (`language.md:463`) are the sites.
- `scripts/check-doc-paths.sh` prints `head: cannot open '../AGENTS.md'` style noise to stderr and exits 0;
  not from these edits.
