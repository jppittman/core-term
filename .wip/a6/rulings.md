# A6 rulings

These are rulings on `spec.md` and the four critiques (`critique.{laws,stages,surface,complete}.md`). Where a
ruling and the spec differ, the ruling wins. The design's R-A6 section was already folded into the spec.

## Decisions

- **D-A6-1: arm (b). `write_into` panics on a count past 2²⁴; no public item is added.**
  - **Why (b):** it is the smallest public-API change: nothing is added to `pixelflow-core`. It is consistent
    with D-A6-4 (the host function panics) and with `pixelflow-core`'s existing per-call refusals: `with_uniforms`
    and `bind` panic (critique.surface S2).
  - **The objection to (b):** it gives one call two failure modes (an `Err` for arity, a panic for a count).
    That is the same split `bind` already has, and nothing in production calls `write_into`.
  - **(a) stays open for JP:** a `WriteRefused` enum is a design JP may prefer. It goes in A6's PR body as an
    offered upgrade, with spec §4.4-4.5 holding both arms in full. A6 builds (b), and the order of work does
    not stop at step 4.
- **D-A6-2, D-A6-3, D-A6-4:** as the spec decides. A helper takes no `usize`, the closure form takes none,
  and the host function panics past 2²⁴.
- **Decision 8:** as the spec decides. Sema and lowering check `Reduction::Any | Reduction::All`, and
  `takes_a_value_bound` stays `pub(crate)`.

## Gates and order (blocking)

- **stages S1, complete C2:** G2 gets `| grep -v guide_scope_saturation_delta.rs`, and its expectation is
  "nothing".
- **complete C1:** step 2 writes §6.5's emit unit test without the `write_into` refusal asserts. Step 4 adds
  them with arm (b), which asserts the panic's text in the generated tokens.
- **complete C4:** G9's expectation lists `lib.rs:77`, the `lib.rs` `# Binding times` line, parser.rs's module
  doc, and the test assertions.

## Docs the spec misses (blocking)

- **surface S1, complete C3:** the-language-is-kernel gets three more edits.
  - `:262-264`: "`write_into` checks the arity (`ArityMismatch`) and panics on a count parameter past 2²⁴".
  - `:52-59`: an A6 clause beside A5's: "A6 made a `kernel!` bound a value: an enclosing fold's index, or an
    entry's `n: usize`".
  - `:111`: a matching qualification.
  - **Check:** read each line against the code after A6.

## Front end

- **stages S2:** M2's help is chosen by `(role, spelling)`.
  - **Items-form entry:** today's text.
  - **Closure form:** a pointer to the items form (M7's help).
  - **Helper:** no count help.
  - **Pins:** the pinned help substring moves to an items-form row. The parser's turbofish help drops its
    appended "or take it as an entry parameter" clause.
- **stages S3:** lowering's backstop resolves names before it judges arithmetic, mirroring `type_of_bound`.
  - **Rule:** every Ident in both bounds is checked against `self.frame.locals`. Only a `Binding::Count`, a
    fold index, or a `usize` const passes; anything else is refused as "`{name}` as a range's bound".
  - **Pins:** `|| (0..X + 1)` and `|r: f32| (0..r - 1)` go in the lowering unit test.
- **stages S4:** `lower_entry` builds each binding from its kind (`Value`, `Count`, `Record`), and both `expect`s go.
- **stages S5:** the grammar row becomes `bound ::= iexpr | INDEX | COUNT`, with `CONST` as `iexpr`'s leaf. The
  legend is updated.
- **complete C7:**
  - `param_type`'s entry refusal names a count: "an `f32`, a count `n: usize` or a record".
  - `has_args_record` and the three `Ty::Usize` arms are recorded as unchanged.
  - The comment at `rustc_is_the_oracle.rs:595` is widened.

## Laws and wording

- **LAWS-1:** L2 and L5 hold "up to the two optimizer edges `Kernel::over_values` documents (E9's annihilation,
  decision 5's factoring)" everywhere the spec or a doc states them for `kernel!`.
  - §1.3 row 0 reads "±0.0 (X's sign under `FactorFold`)".
  - E9's BACKLOG row is widened: a zero factor written in `kernel!` takes the fold at expansion too.
  - Every claim is checked against `fold_rules.rs`.
- **LAWS-2, surface S5:** the reason a count past 2²⁴ is refused is the domain. A count is an integer in
  `0..=2²⁴` (plan §1.4). 2²⁴ + 1 would be written as 2²⁴ and run one trip short with no NaN, and a larger
  count would name none.
  - This text goes in the generated `# Panics`, in `write_into`'s doc, and in J10.
  - "Refusal of plausible pixels" goes.
  - A `usize` const read as `N as f32` past 2²⁴ stays accepted, rounding as Rust's `as` rounds. It predates
    A6, it is a value read rather than a bound, and impl.md records it.
- **LAWS-3:**
  - §5.1 is corrected: no fold rule runs at expansion, and `KnownBound`/`HalveFold` act at bake.
  - E8 is unreachable because every member of an opaque leaf's class contains the leaf, so none is cheaper.
  - No code doc may repeat the wrong version.
- **LAWS-4:** the ontology Range row reads "a count `kernel!`'s host function or `Args` record writes is refused
  there; any other write is an `f32`, and one naming no count makes the fold NaN (E9 aside)".
- **surface S3:** the generated `write_into` doc reads "Of the block, only the number of arguments is checked:
  a block of another program that declares as many takes these values by position". A count past 2²⁴ panics,
  under `# Panics`.
- **surface S4:** the generated `# Panics` line names the count without claiming to be the whole list, e.g.
  "Panics if a count argument is past 2²⁴". A staged entry's other panics are as today.
- **surface S6:** §5.1's relink reason: the declared table is copied whole, so the count keeps its slot, read
  or not, and reaching the bound through the fold's children remaps its `Uniform` leaf.
- **complete C5:** step 5's reason is dropped. The expansion checks outside the compiler are
  `cargo test -p pixelflow-graphics --lib` and `cargo check --workspace --all-targets`.
- **complete C6:** "eight test files, 79 in all".

## Docs discipline (standing, from A4 and A5)

- **Every sentence:** every doc sentence written or edited is checked against the code, not against a ruling's
  wording, because a ruling can be wrong (A5's C1 and D3-9 were).
- **Delete before qualifying:** prefer deleting a sentence to qualifying it.
- **Prose that describes code:**
  - A `Param` is a leaf with no value, built by nothing in the pipeline.
  - A declined term compiles as written.
  - `kernel!` offers each entry to the optimizer at expansion.
- **Never commit while a gate runs:** the gate scripts append a probe to a tracked file.
