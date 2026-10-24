# A5 spec critique: front-end semantics

Base: `7e9531e` (HEAD differs only under `.wip/`). Subject: `.wip/a5/spec.md`, with `.wip/a6/design.md`
where it constrains A5, against the plan (`docs/language/2026-10-20-a-bound-is-a-value.md`),
`CLAUDE.md` and `pixelflow-compiler/intent.md`. Lens: parser to sema to lower to emit, after A5.
Evidence probes live in `/home/user/scratch/a5-semantics-probe` (outside the repo; built against
`target.noindex` with the workspace `RUSTFLAGS`).

## Verdict

The semantics are sound. No stage accepts what a later one refuses as a result of A5. The parser is
the right place for the refusal. The message names §1.2 of the bound plan, and §5.3 tests it. The
byte-identity argument for production kernels holds. Deleting the template exemption sends no
entry to an optimizer that must not see it.

Two of the spec's own grep gates cannot pass as written (G3, G8). Both are trivial to fix. One
parser hole sits right next to A5's subject, a turbofish at a call inside a block. It predates A5
and is not blocking, but the spec should close it while it owns "no `fn` takes generics".

## What I checked and found correct

- **Trace after A5.**
  - Parser: `refuse_generics` covers every `fn`, entry and helper alike.
  - Sema: `RangeScope { consts }` and `range_bounds` return `Range<u64>`; the reversed-range
    refusal is kept.
  - Lowering: `literal_range` and the depth check run before `open_fold`, which can no longer fail.
  - Emission: the `Reduce` arm always goes through `FoldHeader::from_bits` + `attach`.
  - Load time: `push_reduce` refuses only what sema and lowering already refused (reversed, past
    2^24).
  - Sema's `type_of_cast` (`Index` or `Const` of `Ty::Usize`) and lowering's `lower_cast`
    (`Binding::Index`, then a `usize` const) still agree once the structural branches go.
  - `refuse_shadowing_an_item` and `resolve` agree on order.
- **Unchanged paths.** These all keep their behaviour:
  - literal bounds, and `usize` consts with checked `+ - * /`;
  - `i as f32` in nested folds;
  - an enclosing index as a bound (refused before and after);
  - records, `Args` and `write_into`;
  - helper inlining;
  - kernel-typed `Staged` entries.

  On token identity: `#vis struct #args #generics {` and `impl #generics #args #arguments {` with
  empty generics are token-identical to the A5 forms. `Staged::fold_terms` changes tokens but not
  the arena, because `Fold::new` is `Fold::strided(.., 1)` (`fold.rs:580-582`, pinned at
  `:1766-1769`).
- **The exemption's deletion.** After A5 every entry that reaches `entry_arena` has only `Lit`
  pairs. A `Dwrt` is declined by `DwrtFree` (`lib.rs:381-392`). A kernel-typed entry is `Staged` and
  never reaches `entry_arena` (`emit.rs:105-116`).

  A5 actually removes a latent hazard. Before A5, `holes.range_of` panicked on any optimizer-returned
  fold over `0..0` at stride 2 or more in a non-template entry, because `Holes` was empty
  (`lower.rs:333-349`, `emit.rs:592-594`).
- **Byte identity.** The one production block (`pixelflow-graphics/src/fonts/loop_blinn/program.rs:31-141`)
  has no generic and no fold, so every branch A5 deletes is unreachable for it. `byte_probe` does not
  use `kernel!`. The gate workload is `program(n)` only.

## Findings, ranked

1. **G3 cannot pass with J14's panic text (blocking: gate).** §2.5 prescribes this `Param` panic for
   `emit.rs`: "a `Param` is a rewrite rule's metavariable, **instantiated** before any arena reaches
   emission" (spec.md:344). G3 runs `grep -rniE 'template|instantiat|\bholes?\b' pixelflow-compiler/`
   and expects only `lib.rs`'s `macro_tier` doc (spec.md:1029-1034). The panic line will match.
   - Fix: say "substituted before any arena reaches emission", or add the hit to G3's expectation.

2. **G8 cannot pass on today's tree (blocking: gate).** G8's pattern `::<[0-9]+>` over
   `pixelflow-graphics/` matches three ordinary-Rust lines in
   `pixelflow-graphics/tests/common/mod.rs`: `:145`, `:148` and `:193`, all `.as_chunks::<3>()`. G8
   expects only the parser's new test (spec.md:1061-1066).
   - Fix: list them as expected, or narrow the grep to `kernel!` sources.

3. **A turbofish at a call inside a block is silently dropped (pre-existing, not blocking; it falls
   under A5's subject).** The parser's `Expr::Call` arm takes `segments[0].ident` whenever the path
   has one segment and no `qself` (`parser.rs:820-833`). It never checks `segments[0].arguments`,
   and it never checks `leading_colon`. Probed at `7e9531e` (`tests/turbofish.rs`,
   `tests/leading_colon.rs`), these all compile and evaluate as the call without the qualifier:
   - `h::<4>(X)` is `h(X)`;
   - `DX::<f32>(X * X)` is `DX(X * X)`;
   - `k::<4>(X, Y)` is `k(X, Y)`;
   - `::h(X)` is `h(X)`.

   The `Path` arm has the same `leading_colon` gap: `::X` is `X` (`parser.rs:724-733`).

   rustc refuses each of these (E0107; `::name` is a crate-root path). §1.2's table answers
   `::<4>` only for host code, and J1/J2 make the parser the one place generics are refused. This
   is the call-site half of that refusal, and it is missing.
   - Fix: in the `Call` arm, refuse a callee whose segment carries arguments, naming "no `fn`
     takes generics", and add the three cases to §5.3. While there, refuse `leading_colon` in both
     arms.

4. **The new note contradicts sema's vocabulary (not blocking).** §1.1's const-generic note says "a
   count is a value, not a program of its own" (spec.md:108). In the same crate after A5:
   - `Ty`'s doc says a `usize` "is a count and not a value" (`sema.rs:93`);
   - `a_count_is_not_a_value` refuses a count where a value is expected (`sema.rs:1641-1652`);
   - `of_a_value`'s note, as the spec itself rewrites it, says "a `usize` is a count, not a value".

   One crate's two refusals then say opposite things. "Value" in the bound plan means an IR term,
   not a `kernel!` expression.
   - Fix: "note: a count no longer makes a program of its own: an entry is one program (§1.2 of …)".

5. **A generic `const` item never reaches `refuse_generics` (not blocking).** syn turns a `const`
   item with generics or a `where` clause into `Item::Verbatim` (syn 2.0.101 `src/item.rs:1029-1042`;
   probed in `tests/generic_const_item.rs`). It then hits `refuse_item`'s `_ => "item"` arm and the
   user sees "a `item` in a `kernel!` block" (`parser.rs:296-322`). So `convert_const`'s call
   (`parser.rs:407`) is dead.

   The spec's parser-doc rewrite (spec.md:182) and J2 both say a `const` with a generic gets the
   generic refusal. It does not.
   - Fix: drop "or a `const`" from the doc and from J2. Alternatively, name a generic `const` in
     `refuse_item`'s `Verbatim` arm.

6. **The ontology's new "Entry" text overstates optimization (not blocking).** §6.6 says the host
   function "rebuilds its lowered, optimized arena" (spec.md:917-920). That is false in two cases:
   - under `kernel_raw!` (`Identity`, `lib.rs:340-343`);
   - for an entry carrying a `Dwrt` (`DwrtFree` declines, `lib.rs:385-391`).
   - Fix: "rebuilds its lowered arena, saturated at expansion unless it carries a `Dwrt` (or the
     macro is `kernel_raw!`)".

7. **Spec code that does not compile as written (not blocking).**
   - **`Frame`'s lifetime.** `Frame<'a, S>` uses `'a` only in the `structural` field
     (`lower.rs:593-598`). Deleting the field as §2.4 says leaves `'a` unused, which is E0392.
     `Frame<'a, S>` becomes `Frame<S>`, and `Lowering`'s `frame: Frame<'a, S>` (`lower.rs:603`)
     follows.
   - **`literal_range`.** Spec.md:285 describes it as `Some(range).filter(KnownRange::admits)`
     with the message formatted from `range.start`/`range.end` afterwards, which is a use after
     move. Destructure first.

## Outside the lens, noted only

G10's "expected files" (spec.md:1074-1080) leaves out `.claude/skills/ontology/SKILL.md:59`. The
spec's own edit keeps "Structural parameter (retired)" there, which matches `structural
parameter`. The hit is acceptable under G10's "read each hit" criteria; only the file list is off.
