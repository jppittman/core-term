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

## Round 1 (reviews `review1.{laws,stages,docs,complete}.md`, on 6c60dfa): applied by the coordinator

No behavioural defect. The laws and stages were probed through the real `kernel!` on both x86 tiers; the
blockers are doc sentences, several of them ruling text copied verbatim.

- **S1/D6/B1 (blocking):** the grammar gets `bound ::= … | '(' bound ')'`, since `(0..(n))` is accepted.
- **D1/B2 (blocking):** the-language-is-kernel's "A6" now names the plan, "a-bound-is-a-value's A6"
  (`:59`) and "that plan's A6" (`:113`). That document's own A6 is the `Select`/`If` rename.
- **D2 (blocking):** `fold_bounds`' doc and the lowering test's doc. Sema refuses the names and decision 8
  first, and the 2²⁴ checks on a known end are lowering's alone.
- **D3 (blocking):** each law has one edge. L2 (identity) has the factoring edge, at lib.rs `# Folds` and the
  oracle test's doc. L5 (NaN) has the annihilation edge, in the plan's Done line. ast.rs's `FoldExpr` doc
  points once to `# Folds` (complete N4).
- **D4 (blocking):** `# Folds`: "the node `Kernel::over` builds, or `Kernel::over_values` for a bound that
  is a value".
- **D5 (blocking):** `BOUND_PLAN`'s doc is cut to "The plan that made a fold's bound a value." The clause
  went because it was false.
- **D7 (blocking):** M1's note: "its trip count the same at every sample". The triangular fold's inner
  count changes per outer trip.
- **D8:** `range_bounds`: "decided at run time (L2)".
- **D9 (blocking):** `not_a_usize_bound`'s doc says the help is chosen by what the name is, which is what
  the code does.
- **N1:** core README's NaN sentence gets "(up to BACKLOG E9's annihilation)".
- **N2:** fold_is_kernel_over.rs: the lead sentence names `over_values`, and `Fold::Range` becomes `Fold`.
- **N3:** lib.rs: "the number of arguments differs".
- **N4:** "and, if it has one, its `Args` record". This is in count_parameter's doc and plan §1.4/§3.4.
  M7's note drops the `Args` record, since the closure form's subject is the host function.
- **N5:** `param_type`'s unknown-type refusal lists "an entry's count `n: usize`".
- **N6:** `register_parameter`'s doc: "of the type its declaration names".
- **N7:** `BoundRefusal`'s doc: a count the host writes is refused at the write, before it is a value.
- **N8:** M3's help adds "arithmetic on an index waits for bounds typed as integers".
- **S2:** lowering's const-first lookup is documented as relying on sema's refusal of a binding that shadows a
  const.
- **S3 (subtract):** `RangeScope::usize_named`'s `F32` arm was unreachable, because both callers judge an
  `f32` const first. It is folded into the `Name` arm, and `Count::Name` is "no `usize` const". The f32-const
  bound row in sema's test still pins M2's help.
- **S4, refused:** the span of a literal end past 2²⁴ beside a value. J13 put the domain check in lowering,
  and BACKLOG C12 covers spans.
- **complete N1 (pins):**
  - The emit test asserts `write_into`'s own `# Panics` and its domain text.
  - New `a_staged_entry_refuses_a_count_past_the_domain` (`kernel_typed_parameters.rs`, `#[should_panic]`).
  - The compiler goes from 318 to 319 tests.
- **complete N2:** public-surface. A6 adds no public item: `git diff 29980bf -- pixelflow-core/src` is
  empty, and the generated host function's and `write_into`'s `# Panics` sections are the only surface change.
- **complete N3:** intent.md's count sentence names the entry's host function and its `Args` record's
  `write_into`.
- **L-2:** the pre-existing nest-depth gap goes in BACKLOG C13.
- **Polish:** `COUNT_DOMAIN`'s inner colon becomes a semicolon, so the generated text reads
  "past 2^24: a count is …; `2^24 + 1` …".

Checks after round 1:
- `PIXELFLOW_ISA=avx2` and `=avx512` compiler tests: 319 + 319 passed.
- IR lib: 310 passed.
- clippy 1.99 on the compiler and IR, all targets and all features: exit 0.
- Rustdoc: NEW 0 for compiler, IR and core.
- fmt, check-intent and check-doc-paths: OK.
- Gates `a6-fix1` (byteprobe, gate1b, gate2, gate2x): all exit 0, identical to base.
