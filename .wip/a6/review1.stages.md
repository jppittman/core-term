# A6 review 1: front-end stage agreement, by experiment

Working tree against `29980bf` (excluding `.wip/`), read against `.wip/a6/spec.md`, `.wip/a6/rulings.md` (which win),
the plan, CLAUDE.md and `pixelflow-compiler/intent.md`. Read-only on the repo; no gate script was run.

## How it was tested

Scratch crate `/home/user/scratch/a6-review-stages` (repo target dir, the workspace `RUSTFLAGS`):

- `gen_cases.py` writes 78 refusal cases, each once under `kernel!` and once under `kernel_raw!` (156 examples), in
  the closure form or the items form. `cargo check --examples --keep-going --message-format=json`, summarized by
  `report.py` into `report1.txt`: the first error of each, its span (line:col and the highlighted text) and its
  text.
- `tests/accepted.rs`: 46 entries (37 without a kernel, 9 staged; a helper and a record among their inputs), each defined under
  both macros, built and run against Rust's iterator over the same tokens. Counts 0, 1, 3, 7 and a pair table,
  at four sample points with `Lattice::eval_at`, and again over a 19 by 3 bake (wider than one batch). It also
  covers the host panics (Expansion and Staged, both macros, an unread count, a value-only count, the second of
  two counts), the `write_into` panic, `2^24` accepted, and rebinding through `Args`.
- `tests/adversarial.rs`: bound classes the macro tier could merge with a member that reads a coordinate.
  Examples are `(i as f32) + X - X`, `(n as f32) + Y - Y` and `(n as f32) * (X / X)`.
- `tests/mask_outer.rs`: a value-bounded sum inside an `any`/`all` whose index bounds it.
- `compiler-copy/`: a verbatim copy of `pixelflow-compiler/src` (checked with `diff -r`), plus one probe module
  appended to `lower.rs`. For each input it runs `parse` + `sema::analyze`, and lowering with no sema in front (the
  unit tests' `lower_unanalyzed`). Output is in `probe_lowering.txt`.

The runtime tests pass under `PIXELFLOW_ISA=avx512` and `=avx2`: `accepted` 13/13, `adversarial` 1/1,
`mask_outer` 1/1.

## What holds

**Every new refusal is refused at expansion, by the right stage, with the right text, span and citation.** It is the
same under `kernel!` and `kernel_raw!`: the 156 examples agree pairwise in text and in the token spanned (the
lowering refusals differ only in the macro name their call-site span covers).

- **M1 (L3):** `(0..X)`, `(Y..4)` and `(0..X + 1)`, in both forms; `(0..1 + X)`, `(n..X)` and `(0..n + X)`; `X` in a
  helper's bound; `X` in a staged entry. Each is spanned at the coordinate and cites "L3 of BOUND_PLAN". `(0..X + 1)`
  and `(0..n + X)` are L3 and not arithmetic, as R-A6 a6fit 2 requires.
- **M2 (type):** an `f32` parameter in the items form (help `n: usize`), in the closure form (help: the items form)
  and in a helper (no help). Also a `let`, an `f32` const (help `const R: usize`), a record (found `P`), a kernel
  (found `impl Fn(f32, f32) -> f32`), a `bool` helper parameter or `let` (found `bool`), and a `let` shadowing a
  count. `(0..r - 1)` in the closure form is M2 at `r`, not arithmetic. Each is spanned at the name and cites §3.4.
- **M3:** `(0..i + 1)`, `(0..n - 1)`, `(LO + n..n)`, `2 * n`, `n / 1`, `(n) + 0` and `i + n`. Each is spanned at
  the operator and names the first value operand; `n / 0` is M3 before division by zero.
- **M4:** `(0..n).any`, `(n..4).all`, and `(0..i).any` inside a fold. Each is spanned at the name and cites
  decision 8. `(0..n - 1).any` is M3 first, as the spec orders it, and `(16777217..n).any` is M4 in sema and in
  lowering alike.
- **M5:** a method, a comparison, `n as f32`, a call, a block, a field and an `if`.
- **M6 and M7:** `fn h(n: usize)`, whether called or not, and `|n: usize|` as the first or second parameter. Each is
  spanned at the type and cites §1.4; M7 carries its help.
- **Scope and shadowing:** the fold's own index in its bound, a count read inside a helper, and an unknown name all
  give `not_in_scope`'s text. So do a `fn` name and `Z`. A count named after a `usize` or `f32` const, after `X`,
  after `DX` or after a `fn` is refused as shadowing, and so is a duplicate count.
- **A count as a value:** a let alias of a count, a count passed to a helper, `-> usize`, `let n: usize`, a generic
  (the help now names `n: usize`), `as usize` and `..=` are refused as before.

**Every accepted spelling builds and means what rustc means, at both sites and under both macros.** No load-time
`push_reduce` panic, no host panic, no bake panic and no value disagreement appeared. The spellings: triangles with
value `lo`, value `hi`, a value pair, a backwards pair per trip, three deep, an index two folds out, `(i..i)`, a
shadowed index, product and min outers. Helpers with a triangle, inlined at depth, twice, and under a count. Counts
under sum, product, min and max, at either end or both, `(n..n)`, unread, read only as `n as f32`, twice, shadowed
by an index, inside `if`, beside const arithmetic, parenthesized, beside a record, and under an `any` with known
bounds. A count `lo` with an index `hi`. Staged: a count, a count declared first, a triangle, `from`, `between`,
index plus count, a helper, an unread count, `any`, a count kernel passed to a count kernel, and a triangle passed
to a triangle. The macro tier extracted the leaf in every adversarial class: no `BoundVaries` (E8) at load time.

**Host refusal:** the host function panics past `2^24` for every count, read or not, in Expansion and Staged entries
alike. The message names the entry, the parameter and the value: "kernel!: `between`'s count `hi` is 16777218, past
16777216 (2^24): ... (§1.4 of docs/language/2026-10-20-a-bound-is-a-value.md)". `write_into` panics too, and
`2^24` is accepted.

**R-A6 order:** sema and lowering judge the same things in the same sequence. Both walk `bound_names(lo)` then
`bound_names(hi)` in source order and judge every name before `range_bounds` judges any arithmetic. The first
refused name agrees for `(0..X + n)`, `(0..n + X)`, `(nn..X)`, `(X..nn)`, a `bool` and an `f32` const. Decision 8
comes after `range_bounds` in both, and lowering's literal-past-`2^24` check (beside a value) comes after decision
8. The one place the two stages do not agree is how a name is resolved (finding 2 below).

## Findings

1. **Blocking (a doc sentence the diff writes is false).** `parser.rs:69-72`, the grammar the diff writes:
   `range ::= '(' bound '..' bound ')'`, `bound ::= iexpr | INDEX | COUNT`, and `iexpr`'s leaves `CONST | INTEGER`.
   This derives no parenthesized value bound, yet `(0..(n))` and `((n)..(8))` are accepted:
   - sema pins `(0..(n))` as accepted (`sema.rs:3002`);
   - lowering builds a `Bound::Value` for it;
   - `tests/accepted.rs` `paren(n)` matches rustc under both macros.

   The ruling S5 wording was copied without checking it against the code. Fix: `bound ::= iexpr | INDEX | COUNT |
   '(' bound ')'`.
2. **Non-blocking.** Lowering resolves a bound's name `const` first (`lower.rs:681-688`, then `range_bounds`'
   consts-only `RangeScope`). Sema resolves it innermost scope first, and lowering's own `resolve` and `lower_cast`
   resolve a binding before a `const`. The comment at `lower.rs:889-891` says that order is "the same order sema
   documents". With sema absent, `const N: usize = 2; … (0..4).map(|N| (0..N)…)` lowers the inner bound as
   `[0..2)`, the const, while `(0..4).map(|N| N as f32)` reads the index, `Var(4)`. `let N = 3.0; (0..N)` lowers as
   `[0..2)` too (`probe_lowering.txt`). Through `kernel!` this cannot be reached: sema refuses every binder that
   shadows a const, as rustc does. It is still the "right only because an earlier stage refused its input" shape
   that `resolve`'s doc warns against. Fix: resolve `self.frame.locals` first in the bound's name check, and keep
   `range_bounds` from evaluating a name a local shadows; or state there that this order relies on sema's
   no-shadowing rule.
3. **Non-blocking.** Two refusals no input reaches.
   - `RangeScope::usize_named`'s `ConstValue::F32` arm (`sema.rs:2007-2014`, "`R` is an `f32` const, and a range's
     bounds are `usize`s") is unreachable from both of its callers. Sema refuses an `f32` const first with M2, and
     lowering refuses it first with "`R` as a range's bound" (probe). No test pins its text. The spec's "lowering's
     backstop now" for it stopped being true once ruling S3 put the name check ahead of `range_bounds`.
   - Likewise, `bound()`'s `count_term` error (`lower.rs:727-731`) cannot fire after the name loop at `:641-646`.

   This is a guard defending a state the earlier checks already exclude. Fix: fold the `F32` arm into the `None` arm
   (a name that is no `usize` const is the caller's scope's to judge, and both callers have judged it), and say so
   in `Count::Name`'s doc.
4. **Non-blocking.** Sema accepts a literal end past `2^24` beside a value, and lowering refuses it at
   `Span::call_site()` (`lib.rs:421`). An items block `kernel! { pub fn f(n: usize) -> f32 { (16777217..n)… } }` is
   reported at `kernel! {` (3:1), not at the bound, and the same holds for `(n..BIG)` with `const BIG: usize =
   16777216 + 1` and for a staged entry (`report1.txt`). This was already so for a known pair (`literal_range`), and
   the spec puts the check in lowering (J13), so nothing is silent. But it is the one A6 refusal without a span, and
   it is a stage accepting what a later one refuses. Fix, if wanted: check each known end with `KnownRange::admits`
   inside `range_bounds`, which has the range's span and is already "the one evaluation" both stages share.

Not findings, but noted:
- No sema unit test pins a `bool` bound (it is M2, "found `bool`", by experiment) or a count parameter shadowing a
  `usize` const (it is refused by the shared `register_parameter` path).
- The fold-index shadow help "rename this a fold's index" predates A6.
- The host-side `assert!` on a `usize` is the runtime guard decision 2 and D-A6-1/D-A6-4 chose over a typed count
  (rejected arm (c)).
