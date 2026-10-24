# R14 probe: a block `const` or an entry parameter spelled like an emission local

**Tree:** unmodified `7e9531e` (HEAD `2bbfc99`; `git status --short` shows nothing outside `.wip/`).
**rustc:** 1.97.0 (2d8144b78 2026-07-07), dev profile, `RUSTFLAGS=-C llvm-args=-fp-contract=fast`, repo
`target.noindex`. Host AVX-512, `PIXELFLOW_ISA` unset: every outcome is a compile error or an exact
integer, so no tier matters.
**Crate:** `/home/user/scratch/a5-r14-probe` (outside the repo), one integration-test target per case,
because a `kernel!` rustc refuses stops its whole file. Run one case with
`CARGO_TARGET_DIR=/home/user/pixelflow/target.noindex RUSTFLAGS="-C llvm-args=-fp-contract=fast" cargo test --test <case> -- --nocapture`
from the crate. Logs: `/tmp/claude-0/-home-user-pixelflow/cd6598e1-de47-5cbe-90e2-60d028e023c0/scratchpad/r14-<case>.log`.

## Verdict

R14's claim holds: a block `pub const __lo: usize = 2;` beside an entry with a fold does not compile on the
base, so the failure predates A5. Three corrections to R14's text, for `impl.md`:

1. **The code is E0308, not E0530/E0005.** `__lo` and `__hi` give E0308 ("`__lo` is interpreted as a
   constant, not a new binding"); `__header` gives E0308 plus two E0599s that follow from it. E0530 appears
   only where the emission's binding is `let mut` (`__arena`). E0005 appears nowhere.
2. **The hazard is the whole `__`-prefixed family and needs no fold.** `pub const __arena: usize = 1;` beside
   `pub fn g() -> f32 { X }` fails the same way (E0530 + 2x E0308), and `__root` likewise (2x E0308). The fold only matters for
   `__lo`/`__hi`/`__header`, which are emitted by `node`'s `Reduce` arm.
3. **A parameter named `__lo`, `__hi` or `__header` is fine** (value correct, 16 = 3*2 + 10). Only
   `__arena` collides with a parameter (E0308, at the parameter's own token). Reason: an entry's arguments are
   read once, in the `declare_uniform` statements at the top of the emitted block, before any fold; the
   `let (__lo, __hi)` and `let __header` sit inside the fold's own `{ ... }` and read no argument.

The error spans are the whole `kernel! { ... }` call (the const is labelled "constant defined here" on the
whole block too), except `param___arena`, which points at the parameter. "At the author's own line" is true
at the macro's granularity, not the const's.

## Cases

| # | Block | Outcome |
|---|---|---|
| control | `pub const N: usize = 2;` + `pub fn f() -> f32 { (0..3).map(\|i\| (i as f32) * X).sum::<f32>() }` | compiles; `eval_at(f(), 2, 0)` = **6** |
| 1 | `pub const __lo: usize = 2;` + the fold entry | **E0308** x2, `__lo` read as a constant (below) |
| 2 | `pub const __hi: usize = 2;` + the fold entry | **E0308** x2, identical shape to 1 |
| 3 | `pub const __header: usize = 2;` + the fold entry | **E0308** (expected `FoldHeader`, found `usize`) + **E0599** (no method `attach` for `usize`) + **E0599** (`usize` is not an iterator) |
| 4 | `pub fn f(__lo: f32) -> f32 { (0..3)...sum::<f32>() + __lo }` | compiles; `f(10.0)` at X=2 = **16** |
| 5 | same with `__hi` | compiles; **16** |
| 6 | same with `__header` | compiles; **16** |
| x1 | `pub const __lo: usize = 2;` + `pub fn g() -> f32 { X }` (no fold) | compiles (only `non_upper_case_globals` warning); `g` at X=2 = **2** |
| x2 | `pub fn g(__lo: f32) -> f32 { X + __lo }` (no fold) | compiles; **12** |
| x3 | `pub const __lo: usize = 2;` + `(__lo..4).map(...)` (the const is the bound) | **E0308** x2, same as 1 |
| x4 | closure form `kernel!(\|__lo: f32\| (0..3)...sum::<f32>() + __lo)` | compiles; **16** |
| x5 | `pub const __lo: usize = 2;` + `pub fn g(k: impl Fn(f32, f32) -> f32) -> f32 { (0..3).map(\|i\| k(X, Y + (i as f32))).sum::<f32>() }` (`Staged`, a fold) | compiles; **9** (the `Staged` path emits no `__lo`/`__hi`/`__header`) |
| x6 | `pub const __arena: usize = 1;` + `pub fn g() -> f32 { X }` | **E0530** ("let bindings cannot shadow constants") + **E0308** x2 (expected `ExprArena`, found `usize`) |
| x7 | `pub fn g(__arena: f32) -> f32 { X + __arena }` | **E0308** at `tests/param_arena.rs:6:14`, "expected `f32`, found `ExprArena`" |
| x8 | `pub const __root: usize = 1;` + `pub fn g() -> f32 { X }` | **E0308** x2 (expected `ExprId`, found `usize`; the second at `Kernel::from_parts`) |

Cases 1-6 are the task's; the control and x1-x8 are this probe's additions, and discriminate (the const is
harmless without a collision site, the fold is the trigger for `__lo`/`__hi`/`__header`, the `Staged` path is
exempt, the family is wider).

### Verbatim, case 1 (`const_lo`), first error

```
error[E0308]: mismatched types
 --> tests/const_lo.rs:5:1
  |
5 | / kernel! {
6 | |     pub const __lo: usize = 2;
7 | |     pub fn f() -> f32 { (0..3).map(|i| (i as f32) * X).sum::<f32>() }
8 | | }
  | | ^
  | | |
  | | expected `Bound<_>`, found `usize`
  | |_constant defined here
  |   `__lo` is interpreted as a constant, not a new binding
  |
  = note: expected enum `pixelflow_core::__macro::pixelflow_ir::fold::Bound<_>`
             found type `usize`
```

The second E0308 is `__arena.push_reduce(..., __lo, ...)`: "expected `Bound<ExprId>`, found `usize`", "arguments
to this method are incorrect", pointing at `pixelflow-ir/src/arena.rs:1003`. Final line:
`error: could not compile ... (test "const_lo") due to 2 previous errors`.

### Verbatim, case 3 (`const_header`), headlines

```
error[E0308]: mismatched types ... expected `FoldHeader`, found `usize` ... `__header` is interpreted as a constant, not a new binding
error[E0599]: no method named `attach` found for type `usize` in the current scope
error[E0599]: `usize` is not an iterator
error: could not compile `a5-r14-probe` (test "const_header") due to 3 previous errors
```

### Verbatim, case x6 (`const_arena_no_fold`), headlines

```
error[E0530]: let bindings cannot shadow constants ... the constant `__arena` is defined here
error[E0308]: mismatched types ... expected `ExprArena`, found `usize` ... `__arena` is interpreted as a constant, not a new binding
error[E0308]: mismatched types ... expected `ExprArena`, found `usize` ... arguments to this function are incorrect (Kernel::from_parts)
```

## Why, in the emission (line numbers at `7e9531e`)

- `emit.rs:609-611`, the `Reduce` arm's known-range branch: `let __header = #decoded;` then
  `let (__lo, __hi) = __header.attach(#children);`. With a `const __lo`/`__hi`/`__header` in scope, rustc
  resolves the pattern to the constant, so the `let` is a constant pattern of the wrong type.
- `emit.rs:196-197` and `:113-114`: `let (__arena, __root) = ...`, and `arena_to_tokens`' own
  `let mut __arena`: `__arena` and `__root` collide with a const for every entry, fold or not (x6, x8).
- The block's `pub const` is emitted as a module item (`emit_const`), so it is in scope inside every host `fn`
  the block emits; that is the whole mechanism. Nothing in the parser or sema refuses a `__` name (x1, x2
  compile).
- A parameter is only read in `arena_to_tokens`' `declare_uniform` statements, after `let mut __arena` and
  before any node, so of the names probed only `__arena` shadows it (x7). The emission's other locals
  (`__u<n>`, `__e<n>`, `__t<n>`, `__f<n>`, `__k<n>`) were not probed.

## What A5 changes

Nothing for these cases. The spec's §2.5 keeps the `Reduce` arm's known-range branch as the whole arm (only
the hole branch and `instantiated_range`, whose own `__lo`/`__hi` were the old template's, go), so the surviving
`let __header` and `let (__lo, __hi)` still meet a block's `pub const __lo`. Inferred from the spec; the tree
is unmodified, so not measured after A5. The old hygiene test's subject, a `const` generic spelled `lo`/`hi`,
is gone with the generic; the collision above is a different one and predates A5.

## Proposed BACKLOG row

Next free id in "Correctness and CI" is C11 (C10 is the last row; BACKLOG has no section named "compiler").
One line, with the repro:

```
| **C11** | **A block `const` spelled like one of `kernel!`'s emission locals does not compile.** `kernel! { pub const __lo: usize = 2; pub fn f() -> f32 { (0..3).map(\|i\| (i as f32) * X).sum::<f32>() } }` is E0308 ("`__lo` is interpreted as a constant, not a new binding"), likewise `__hi` and `__header`, and `__arena` or `__root` beside any entry (E0530 + E0308 / E0308), at the whole `kernel!` call; an entry parameter named `__lo`/`__hi`/`__header` is fine, `__arena` is E0308 at the parameter. Loud and the author's own, so it earns no type; the open question is only whether the message should name the expansion's local. | `pixelflow-compiler/src/emit.rs` (`node`'s `Reduce` arm, `arena_to_tokens`) |
```
