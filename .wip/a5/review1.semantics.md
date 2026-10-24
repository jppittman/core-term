# A5 implementation review: front-end semantics, by experiment

**Tree:** `7e9531e` plus the `.wip/` commits plus the uncommitted A5 working tree (24 files), branch
`claude/integral-area-work-ojiej6`. Read-only on the repo; this file is the only one written under it.
**Against:** `.wip/a5/spec.md`, `.wip/a5/rulings.md` (rulings win), `docs/language/2026-10-20-a-bound-is-a-value.md`,
`CLAUDE.md`, `pixelflow-compiler/intent.md`.
**Probe crate:** `/home/user/scratch/a5-review-sem` (outside the repo), built against the repo's target dir with the
workspace flags:

```
cd /home/user/scratch/a5-review-sem
CARGO_TARGET_DIR=/home/user/pixelflow/target.noindex RUSTFLAGS="-C llvm-args=-fp-contract=fast" \
  [PIXELFLOW_ISA=avx2|avx512] cargo test --test <target> [--no-run]
```

Targets: `refuse_kernel.rs` and `refuse_raw.rs` (38 refusal cases each, one module per case, the second a `sed` of
the first onto `kernel_raw!`), `refuse_types.rs` (10 type-position generics), `refuse_fold_turbofish.rs` (6),
`host_turbofish.rs` (host code naming an instantiation), `accept.rs` (17 entries under both macros, plus a
`#[no_implicit_prelude]` module, compared bit-for-bit against the same tokens over Rust consts), `accept2.rs`
(shadowed indices, a three-deep nest, a helper's fold under a staged fold), `accept_empty.rs` (`N::<>`, `X::<>`).
A plain `rustc` probe (`/tmp/claude-0/.../scratchpad/rustc-probe/{p,q}.rs`) checks what rustc itself accepts.

## Verdict

The semantics are right. Every accepted form computes what rustc computes, bit for bit, on both x86 tiers and under
both macros. Every refusal is at the right stage, and every A5 refusal carries the R6/R9 text and cites the right
plan section. No stage accepts what a later one refuses as a result of A5. The R1 `Param` panic is unreachable from
any input. The production kernels are byte-identical.

There are five findings, all small. Two are doc lines that contradict the rulings (R7 and R9), one line each, and
so are blocking under the gate's rubric. Three are refusal-quality nits with no wrong value; two of those predate
A5.

## What was checked and holds

### Refusals: stage, span, message (refuse_kernel.rs, refuse_raw.rs)

38 cases, 38 errors under `kernel!`. `kernel_raw!` produces the same 38 messages: the texts diff empty, and only
the columns move, by the four characters of `_raw`.

| Input | Stage | Span | Message |
|---|---|---|---|
| `pub fn f<const N: usize>`, `fn h<const N: usize>`, `<const N: u32>`, `<const N: usize = 4>`, `<const N: usize, T>`, `<const N: usize>(x: [f32; N])`, `<const N>` + `where`, a staged entry's `<const N>` | parser | the first parameter (`const N: usize`, `const N: usize = 4`) | "a `const` generic in a `kernel!` block / note: a count does not make a program of its own: an entry is one program, whatever its counts (§1.2 of docs/language/2026-10-20-a-bound-is-a-value.md) / help: … `const N: usize = 4;`", which is R9 verbatim |
| `<T>`, `<'a>`, helper `<T>`, `<T, const N: usize>` | parser | the first parameter | "generics in a `kernel!` block / note: nothing in a kernel is generic … (§1.3 of the-language-is-kernel)"; §1.3 is "Types" |
| `where f32: Copy` | parser | the clause | "a `where` clause in a `kernel!` block: there are no generics to bound" |
| `h::<4>(X)`, `DX::<f32>(X * X)`, `k::<4>(X, Y)`, entry `g::<4>()`, closure form `DX::<f32>` | parser (`Expr::Call`) | `::<4>` | R6 verbatim, citing §1.3 of `PLAN` |
| a generic `const` item | parser (`refuse_item`) | the whole item | "a `item` in a `kernel!` block" (R7: syn's `Verbatim`) |
| `(0..i)`, `(0..n)`, an `f32` const bound, `X * N`, `X * i`, underflow, `/ 0`, `5..3`, a helper naming an undeclared `N` | sema | the token | unchanged texts, "structural" gone |
| `0..16777217`, a const `N = 16777217` | lowering | the call site (a `String` error) | unchanged |
| `n: usize` parameter | sema | the type | "a `usize` is a count, not a value: a fold's index or a `usize` const" |

The type-position generics (`Bx<4>`, `f32<4>` as a parameter, return, field or `let` type, the `impl Fn` return,
`usize<4>` on a const, `.sum::<f32<4>>()`, `as f32<4>`, a generic record) are all refused in the parser or sema
(`refuse_types.rs`). So are the turbofish spellings on a fold (`.map::<…>`, `.fold::<…>`, `.any::<_>`) and on its
monoid paths (`refuse_fold_turbofish.rs`). Host code `ramp::<4>(1.0)` and `RampArgs::<4> { … }` are rustc's
E0107, "function/struct takes 0 generic arguments", as spec §1.2 says (`host_turbofish.rs`).

### Accepted forms (accept.rs, accept2.rs, accept_empty.rs)

Results:
- `PIXELFLOW_ISA=avx2`: 3/3 pass, and 1/1 for `accept2`.
- `PIXELFLOW_ISA=avx512`: 3/3 pass, and 1/1 for `accept2`.
- Every comparison is `to_bits()` equality against the same tokens over Rust consts, under `kernel!` and
  `kernel_raw!` alike.

What the entries cover:
- **Bounds:** literal bounds (`0..4`, `2..6`, `3..3`); `LO..LO + N`; `N - 1..N * 2 / ONE`, `N / 2..N`;
  `BIG - 1..BIG` at 2²⁴.
- **Counts as values:** `N as f32`; a `const NF: f32 = N as f32`, which goes through `eval_f32`'s rewritten `Cast`
  arm; consts declared out of order; a `pub const PN: usize = N + LO`, whose host value is 7.
- **Nesting:** a nested fold reading the outer index; a nested min-fold under a product; a helper with its own fold
  inlined under an entry fold; a shadowed index (`|i| … map(|i| …) + i`); a three-deep nest.
- **Monoids and empty ranges:** `.any`/`.all` over consts, with empty ranges giving their identities.
- **Records:** a record parameter with `RecArgs::write_into` rebinding equal to the bake.
- **Staged entries:** folds in staged entries, nested, reading the outer index, a helper's fold around an
  application, and the `#[no_implicit_prelude]` module (a staged nested fold, a record fold, a `kernel_raw!` fold).

### Does any stage accept what a later one refuses?

Not as a result of A5:
- **Parser:** generics never reach the AST. `FnItem` has no field for them.
- **Sema and lowering agree on `as f32`:** sema's `type_of_cast` (a symbol of `Ty::Usize`, which is now only
  `Index` or `Const`) agrees with lowering's `lower_cast` (`Binding::Index`, then `ConstValue::Usize`;
  `lower.rs:643-662`).
- **Name resolution order:** sema's agrees with lowering's `resolve`, binding, then const, then coordinate
  (`lower.rs:777-822`).
- **Bounds:** `range_bounds` is the one evaluation of a bound. Lowering reads the `Range<u64>` it returns, and
  `literal_range` adds only the 2²⁴ refusal, which the parser doc lists as lowering's (pre-existing).
- **Staged entries:** a staged fold's tokens are now `FoldTerms { monoid, lo: Bound::Lit(#lo), hi: Bound::Lit(#hi) }`
  (`emit.rs:613-624`), which build the same arena as before.

### R1: is the `Param` panic unreachable?

Yes. `node` (`emit.rs:422`) is reached only from `arena_to_tokens` (`emit.rs:378`), which is reached only from
`entry_arena` (`emit.rs:157`) on the arena `Expansion::lower` built, as the optimizer returns it:
- **Lowering:** builds no `Param`. `grep -rn 'push_param\|ExprNode::Param' pixelflow-compiler/src
  pixelflow-ir/src/library.rs` hits only the panic arm.
- **Insertion:** the e-graph adds an `ENode::Param` only from an input arena's `Param`
  (`pixelflow-search/src/egraph/insert.rs:131`, `Vocabulary::Templates`).
- **Rewrites:** a rewrite's right-hand side cannot add one. `instantiate_template` panics on a `Param` there
  (`pixelflow-search/src/egraph/graph.rs:1720-1721`).
- **Extraction:** yields a `Param` only if the graph holds one.

The panic text is R1's ("substituted").

### Byte identity

The gate outputs are `cmp`-equal to their bases, and their timestamps (05:37 to 05:40) postdate the last source edit
(`emit.rs` at 05:35:31):
- `byteprobe.a5-impl.avx512.txt` and `byteprobe.a5-impl.avx2.txt`: 127 lines each;
- `gate1b.a5-impl.txt`: 29;
- `gate2.a5-impl.txt`: 29;
- `gate2x.a5-impl.txt`: 29.

The only `kernel!` user outside the compiler crate is `pixelflow-graphics`: no other manifest depends on
`pixelflow-compiler`. `cargo test -p pixelflow-compiler --lib`, re-run here: 142 passed.

## Findings, ranked

1. **The parser's module doc still says a generic `const` gets the count message (R7; blocking by the rubric, one
   line).**
   - **The doc:** `parser.rs:117-123` lists, among what is "Refused here, with a span, at the token", "a generic or
     a `where` clause on a `fn` or a `const` (a `const` generic by a message saying a count is a literal or a
     `usize` const …)".
   - **The behaviour:** `pub const C<const N: usize>: usize = 4;` gets "a `item` in a `kernel!` block", spanning
     the whole item (`refuse_kernel.rs:47`, from `refuse_item` at `parser.rs:301-311`). It never reaches
     `refuse_generics`.
   - **What R7 asks:** that the parser doc stop claiming this, and say instead that syn gives the item refusal
     first. Only `refuse_generics`' own doc (`parser.rs:620-623`) says so.
   - **Fix:** "a generic or a `where` clause on a `fn` (a `const` generic by a message …); a `const` with generics
     is an item syn does not parse as one, refused as such".

2. **`generic_refusal`'s doc says "a count, which is a value now" (R9; blocking by the rubric, one line).**
   - **The doc:** `parser.rs:637-639`.
   - **Why it matters:** R9 changed the note away from exactly this sentence, because in the front end "a value is
     an `f32` or a `bool`, and a `usize` is a count". `Ty`'s doc says so too: "The third, `usize`, is a count and
     not a value" (`sema.rs:89`).
   - **Fix:** "a `const` one was a count, which a block now writes as a literal or a `usize` const".

3. **`h::<>(X)` is refused as "generic arguments at a call", while `N::<>` and `X::<>` are accepted as the name (not
   blocking).**
   - **The cause:** the new `Expr::Call` check uses `!segment.arguments.is_none()` (`parser.rs:761`). The `Path`
     arm beside it uses `segment.arguments.is_empty()` (`parser.rs:666`).
   - **Evidence:**
     - `refuse_kernel.rs:33` is refused, spanning `::<>`.
     - `accept_empty.rs` passes: `(0..N::<>).map(|i| X * (i as f32)).sum::<f32>() + (N::<> as f32) + X::<>` is
       18 at `X = 2`.
     - rustc accepts all three and prints `1 4 6` for `h::<>(1.0)`, `N::<> as f32` and
       `(0..N::<>)…sum::<f32>()` (`rustc-probe/p.rs`).
   - **Impact:** no wrong value. But the refusal names generic arguments that are not there, and R6 refuses "a
     callee whose one segment carries arguments".
   - **Fix:** `!segment.arguments.is_empty()` in the `Call` arm.

4. **A path with generic arguments in value position is refused for the wrong reason (pre-existing, not blocking).**
   - **The input:** `N::<4> as f32` and `(0..N::<4>)`, where `N` is a `const` of the block (`refuse_kernel.rs:36`,
     `:39`).
   - **The message:** "a path in a kernel body … a name from the enclosing Rust scope is not captured / help:
     declare the value as a `const` of this block". `N` already is one.
   - **Why now:** A5 gave the call site R6's message, but the value site of the same spelling still gets the
     outside-the-block message (`parser.rs:672-686`).
   - **Fix:** in the `Path` arm, give a one-segment path with non-empty arguments R6's message.

5. **An empty `where` clause is refused at the whole macro invocation (pre-existing in spirit, not blocking).**
   - **The cause:** `refuse_generics` spans the clause with `Error::new_spanned(where_clause, …)`
     (`parser.rs:628-632`). syn prints a `WhereClause` with no predicates as no tokens, so the span falls back to
     `Span::call_site()`.
   - **Evidence:** `refuse_kernel.rs:88`, `pub fn f() -> f32 where { X }`, underlines the whole `kernel!` call.
     rustc accepts that `fn` (`rustc-probe/q.rs` prints `1`).
   - **History:** `7e9531e`'s `structural_parameters` had the same `new_spanned`, and spec §1.1 prescribes it.
   - **Fix:** `syn::Error::new(where_clause.where_token.span, …)`.

Out of this lens, noted only:
- `refuse_item`'s "a `item` in a `kernel!` block" reads "a `item`" (pre-existing grammar).
- `lib.rs`'s `rings` example keeps the column padding of the turbofish it lost
  (`lattice.bake(&rings(b, 1.0, 0.0));            // one call, baked`).
