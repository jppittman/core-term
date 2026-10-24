# A5 round-1 fixes: code review, by experiment

**Tree:** `38703e0` plus the round-1 working-tree edits (18 files outside `.wip/`), branch `claude/integral-area-work-ojiej6`.
I made no edits to the repo. This file is the only one I wrote under it.
**Against:** `.wip/a5/rulings.md` (every round), `.wip/a5/spec.md`, `docs/language/2026-10-20-a-bound-is-a-value.md`,
`CLAUDE.md`.
**Lens:** the round-1 code changes (S3, S4, S5, C3), with the F4 test and F5/C4 as they touch them.

**Probe crate:** `/home/user/scratch/a5-review-fix1`, outside the repo. It uses the repo's `Cargo.lock` (syn 2.0.101,
proc-macro2 1.0.95), the repo's target dir and the workspace's rustflags:

```
cd /home/user/scratch/a5-review-fix1
CARGO_TARGET_DIR=/home/user/pixelflow/target.noindex RUSTFLAGS="-C llvm-args=-fp-contract=fast" \
  [PIXELFLOW_ISA=avx2|avx512] cargo test --test <target> [--no-run]
```

The targets:
- `refuse_kernel.rs`: 35 refusal cases, one module each.
- `refuse_raw.rs`: the same file with `kernel!` replaced by `kernel_raw!` (`sed`).
- `accept.rs`: 16 entries under both macros, plus the closure form. Each entry spells names with `::<>`, and each is
  compared bit for bit with its plain twin.
- `edge.rs`: 7 neighbouring spellings.

Raw rustc output is kept beside the crate (`refuse_kernel.out`, `refuse_raw.out`). A rustc-only probe is at
`/tmp/claude-0/-home-user-pixelflow/cd6598e1-de47-5cbe-90e2-60d028e023c0/scratchpad/rp/m.rs`.

## Verdict

The round-1 code does what the rulings ask. Results:
- **Empty turbofish:** `h::<>`, `N::<>`, `X::<>`, `DX::<>`, `k::<>`, `i::<>`, `b::<>.x0` and `a::<>` are all accepted as the
  bare name. Under both macros and on both x86 tiers, they compute what the plain spelling computes, bit for bit.
- **Non-empty generic arguments:** refused by one function with one message, in both arms. The span is the arguments.
- **`where` on a `fn`:** refused at the `where` keyword, whether the clause is empty or not.
- **Shadowing a `const`:** still refused with the same text as before.
- **The new test:** asserts both plan paths.
- **Stage agreement:** no stage now accepts what a later one refuses. The only behaviour changes are the ones S3, S4 and S5
  rule, plus the header wording the implementer recorded as departure 3. I agree with that departure.

There are three findings:
1. **A round-1 BACKLOG sentence is false of the code.** The empty-`where` wrong span is fixed on a `fn`. A record still has it.
2. **G8 does not pass as written.** A round-1 doc line matches the turbofish pattern.
3. **The method arms disagree with S3 (pre-existing, not blocking).** They still refuse an empty `::<>` and name type
   arguments that are not there.

## What was checked and holds

### S3: empty generic arguments are the name (`accept.rs`, both ISAs)

`PIXELFLOW_ISA=avx2` passes 2/2, and `PIXELFLOW_ISA=avx512` passes 2/2. Each pair below is compared with `to_bits()`
equality at four samples, under both `kernel!` and `kernel_raw!`:

| Empty-argument spelling | Plain twin |
|---|---|
| `h::<>(X)` | `h(X)` |
| `DX::<>(X * X * Y)` (a projection) | `DX(X * X * Y)` |
| `k::<>(X, Y) + k::<>(Y, X)` (a kernel parameter, a `Staged` entry) | the same with `k` |
| `N::<> as f32 + X::<> * Y::<> + NF + (M as f32)`, with `pub const M: usize = N::<> + 1` and `const NF: f32 = N::<> as f32` | the same with bare names |
| `(1..N::<>).map(\|i\| X * (i::<> as f32))` | the same with bare names |
| `b::<>.x0 * X + b.x1` | `b.x0 * X + b.x1` |
| `let a = X + 1.0; a::<> * a::<>.sqrt()` | the same with `a` |
| `(0..N::<>).map(\|i\| k::<>(X + (i as f32), Y))` (a fold around an application) | the same with bare names |
| the closure form `\|\| X::<> * 2.0 + DX::<>(X * X)` | the same with bare names |

Four checks keep "same" from being vacuous. The plain twins were checked against Rust values:
- `p_call` is 7 at x = 2.
- `p_dx` is 12 at (2, 3).
- `p_value` is 19.
- `p_bound` is 12.

`k::M == 5`, so the host const is the evaluated value and not re-emitted tokens. `::h::<>(X)` also builds (`edge.rs` e07),
consistent with R6's "`leading_colon` is refused nowhere".

### S4 and R6: non-empty arguments, one builder, both arms

Every case below gets the same three lines:

> generic arguments in a `kernel!` block / note: nothing in a kernel is generic, so no `fn` takes generic arguments (§1.3 of
> docs/language/2026-09-25-the-language-is-kernel.md) / help: a count is a literal, or a `usize` const of the block

Each is underlined on the arguments alone:
- **Call arm:**
  - `h::<4>(X)` (`^^^^^` under `::<4>`)
  - `DX::<f32>(X * X)`
  - `k::<4>(X, Y)`
  - an entry `g::<4>()`
  - an unknown `zz::<4>(X)`
  - `h::<{ 4 }>(X)`
  - the closure form `DX::<f32>(X * X)`
- **Path arm:**
  - `N::<4> as f32`
  - `(0..N::<4>)`
  - `X::<f32>`
  - `const M: usize = N::<4> + 1` (a const initializer)
  - a fold index `i::<4>`
  - a record base `b::<4>.x0`
  - an uncalled helper `let g = h::<4>;`
  - `Y::<'static>`
  - a method receiver `X::<f32>.sqrt()`
  - `a::<f32>` (a `let`)
  - `if c::<bool>`
  - an unknown `Z::<4>`
  - the closure form `X::<f32> + 1.0`
  - `::N::<4>` (`edge.rs` e04)

`generic_arguments_refusal` (`parser.rs:665`) is the only function that builds the error, and both arms call it (`:686`,
`:782`). Both arms test `is_empty()`, and syn's `PathArguments::is_empty` is true for `None` and for `<>`.

`kernel_raw!` gives the same 35 messages on the same lines. Every caret moves four columns (the `_raw`) except one: the
record-`where` case, whose span is the whole call (finding 1).

The header is "generic arguments in a `kernel!` block", not R6's "… at a call in …". `impl.md` records this as departure 3:
on a value position, "at a call" would be false. That follows the docs rule ("every sentence … must be true of the code"),
so I count it as a correct departure. It is not a defect. The rulings text still quotes the old header.

Multi-segment and qualified spellings keep the outside-the-block refusal, as before:
- `N::<4>::M` gives "a path in a kernel body: `N :: < 4 > :: M`".
- `<f32>::X` gives the same refusal.

### S5: `where` on a `fn`

| Input | Message | Caret |
|---|---|---|
| `pub fn f() -> f32 where { X }` | "a `where` clause in a `kernel!` block: there are no generics to bound" | `^^^^^` under `where` |
| `pub fn f() -> f32 where f32: Copy { X }` | same | under `where` only, where the base underlined the whole clause; S5 rules the keyword for both |
| helper `fn h(x: f32) -> f32 where { x }` | same | under `where` |
| helper with `where f32: Copy,` | same | under `where` |
| `pub fn f<const N: usize>() -> f32 where { X }` | the `const` generic message (R9) | `const N: usize`; the parameter is checked first |
| `const N: usize = 4 where;` | "a `item` in a `kernel!` block" | the whole item: syn returns `Item::Verbatim` when `where_clause.is_some()` (syn 2.0.101 `item.rs:1038-1057`), as the module doc now says |

The new comment at `parser.rs:630-631` is true: syn's `ToTokens for WhereClause` prints nothing when there are no predicates
(`generics.rs:1310-1317`), and `new_spanned` then falls back to the call site. Finding 1 is that fallback, still live.

### C3: shadowing a `const`, message unchanged

The guard is now `lookup(&text).is_some_and(|symbol| symbol.kind == SymbolKind::Const)`. That is the same predicate as the
deleted one-arm `match`. The text is identical: "{binder} `{text}` shadows the `const {text}` of this block / help: rename this
{binder} to something else". Observed for:
- a parameter
- a helper's parameter
- `let N`
- a tuple `let (N, b)`
- a fold index `|N|`

The `fn` arm below it is unchanged ("parameter `h` shadows the `fn h` of this block"). `an_item_is_not_shadowed` passes.

### F4: the new test

`cargo test -p pixelflow-compiler --lib` passes 142/142 on this tree. `generics_are_refused_and_a_const_generic_says_what_a_count_is`
asserts:
- `§1.2 of docs/language/2026-10-20-a-bound-is-a-value.md` in the `const`-generic message;
- `§1.3 of docs/language/2026-09-25-the-language-is-kernel.md` in the other-generic message and in the generic-arguments
  message;
- five generic-argument cases: three call cases and two path cases (`N::<4> as f32`, `X::<f32>`);
- two empty-argument parses.

Reverting S3 would fail the `h::<>(X)` parse. Reverting S4 would fail both path cases on the header. Changing either arm to
`is_none()` would fail the second parse. Spans are unpinned, as ruled (C12).

### Does any stage now accept what a later one refuses? Is any path changed beyond the rulings?

No:
- **Empty arguments:** the parser hands an empty-argument name to sema and lowering as the same `Ident` the bare name gives.
  `accept.rs` shows the same values through every later stage, on both macros and both tiers.
- **Non-empty arguments:** a name with them never leaves the parser.

Outside the parser, round 1 changed only:
- comments (`lower.rs`, `emit.rs`, `arena.rs`, `fold.rs`);
- the `BoundNotANumber` display text in `pixelflow-ir/src/fold.rs:502`. Its test (`fold.rs:2150-2159`) still finds
  `BoundNotANumber`, `` `Param` `` and "a fold's hi bound".

`gates-fix1.md`'s byte gates are IDENTICAL to base. That agrees with this: no path a previously accepted program takes
was changed.

## Findings, ranked

### 1. BACKLOG C12 says the empty-`where` span is fixed; on a record it is not (blocking: a round-1 sentence false of the code)

**The doc.** `docs/BACKLOG.md:170`, row C12: "One wrong span got through that way: a `where` clause with no predicates, which syn
prints as no tokens, so `Error::new_spanned` fell back to the whole `kernel!` call; a review's experiment found it, and it is
spanned at the `where` keyword now."

**The behaviour.** `convert_record` (`parser.rs:341-345`) still uses `syn::Error::new_spanned(where_clause, …)`:
- `pub struct Bx where { pub x0: f32 }` (`refuse_kernel.rs:82`, `refuse_raw.rs:82`) is underlined across the whole
  `pixelflow_compiler::kernel! { … }` invocation, from column 5.
- `pub struct Bx where f32: Copy { … }` (`:85`) underlines the clause, not the keyword. So the two `where` refusals in the
  file disagree on where they point.

**History.** This code predates A5: `7e9531e`'s `parser.rs:350-353` is the same. But C12 is a round-1 sentence, and it says
the defect is gone. The comment the round added at `parser.rs:630-631` describes exactly this fallback.

**Fix (preferred).** `syn::Error::new(where_clause.where_token.span, "a `where` clause on a record: …")` at `parser.rs:342`. That
makes C12 true, and the two `where` refusals then agree.

**Alternative.** Narrow C12 to "on a `fn` it is spanned at the `where` keyword now; on a record it is not", and count two
wrong spans.

### 2. G8 hits a non-test line (blocking under "the spec's own gates must pass as written")

**The grep.** R2's pattern, run here:

```
grep -rnE '<const [A-Za-z_]+: *usize|::<[0-9]+>|Args::<' pixelflow-compiler/ pixelflow-graphics/ pixelflow-core/ | grep -v 'as_chunks::<'
```

It returns the seven `quote!` lines of the new test, plus `pixelflow-compiler/src/parser.rs:664`:

> `/// arguments, ` + "`::<4>`" + `, the token that is wrong.`

That line is `generic_arguments_refusal`'s doc, written this round. R2 keeps the expectation "only the parser's new test
hits". `gates-fix1.md` row 6 records the miss but does not resolve it.

**A second problem on the same line.** `::<4>` is not one token. It is `::`, `<`, `4` and `>`, which syn holds as
`colon2_token`, `lt_token`, `args` and `gt_token`. So "the token that is wrong" is also inexact.

**Fix.** "It is spanned on the turbofish, the tokens that are wrong." This drops the `::<digit>` literal and corrects the count.

### 3. An empty turbofish on a method is still refused, naming type arguments that are not there (pre-existing; not blocking)

**The behaviour.** S3 made `h::<>` and S4's arm made `N::<>` read as the bare name. The method arms still treat an empty
`::<>` as type arguments (`edge.rs`):
- `X.max::<>(Y)` gives "type arguments on a method in a kernel body". This is the `MethodCall` arm, `parser.rs:749-757`.
- `(0..4).map(|i| i as f32).sum::<>()` gives "`.sum` over anything but `f32`s". This is `sum_or_product`, `:1145-1160`.
- `(0..4).map::<>(|i| i as f32).sum::<f32>()` gives "`.map` takes no type arguments in a kernel body". This is
  `refuse_turbofish`, `:1241-1251`.

rustc accepts all three. The probe `m.rs` prints `3 6 6 6`, and `s.rs` (`.sum::<>()`) compiles.

**Impact.** No wrong value, and no stage disagreement: the parser refuses, and nothing later sees the input. But it is the
same class S3 fixed (review1.semantics finding 3: "the refusal names generic arguments that are not there"). It now sits
beside the free-function and path arms, which accept the same spelling.

**History.** None of these lines changed in round 1.

**Fix, if wanted.** Test `arguments.args.is_empty()` in those three places, as the name arms test `is_empty()`. Otherwise,
record the asymmetry.

## Noted only (pre-existing, unchanged by round 1)

- The shadowing message for a fold index reads "help: rename this a fold's index to something else": the binder string
  is "a fold's index" (`sema.rs:886`), and the template prefixes "this". The text is the same as at `7e9531e`.
- `refuse_item` says "a `item` in a `kernel!` block" for a generic or `where`-carrying `const`. Review 1 already noted this.
