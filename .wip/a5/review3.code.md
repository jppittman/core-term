# A5 round-2 fixes: code review, by experiment

**Tree:** `f3233fc` (= `0c3b52a` outside `.wip/`) plus the round-2 working-tree edits (24 files outside `.wip/`), branch
`claude/integral-area-work-ojiej6`. I made no edits to the repo; this file is the only one I wrote under it.
**Against:** `.wip/a5/rulings.md` (every round), `docs/language/2026-10-20-a-bound-is-a-value.md`, `CLAUDE.md`.
**Lens:** the round-2 code changes (R2C-1, R2C-3, D1's two panics), by experiment; and whether any stage now accepts what a
later one refuses.

## How it was run

**Current tree:** the round-1 probe crate `/home/user/scratch/a5-review-fix1`, extended. Same command as round 1:

```
cd /home/user/scratch/a5-review-fix1
CARGO_TARGET_DIR=/home/user/pixelflow/target.noindex RUSTFLAGS="-C llvm-args=-fp-contract=fast" \
  [PIXELFLOW_ISA=avx2|avx512] cargo test --test <target> [--no-run]
```

**Base (`0c3b52a`):** `git archive 0c3b52a pixelflow-compiler` into `/home/user/scratch/a5-review3/base-src/`, its two path deps
pointed at the repo's `pixelflow-ir` and `pixelflow-search`. Round 2 changed only comments and message strings in those two
crates, so the parser's refusals are `0c3b52a`'s. A second probe crate, `/home/user/scratch/a5-review3/base-probe`, uses the
same `Cargo.lock` and copies of the same test files. It is not a git worktree.

**Validation of the base:** its `refuse_kernel` and `refuse_raw` outputs equal the round-1 review's recorded `refuse_kernel.out`
and `refuse_raw.out` byte for byte, once the `Compiling` lines are dropped.

**Outputs and tools:**
- Current-tree outputs: `*.r2.out`, beside the probe crate.
- Base outputs: `*.base.out`, in `base-probe/`.
- `/home/user/scratch/a5-review3/norm.py` reduces rustc's output to (case, message, the source text under the caret).
- `gen.py` writes the method cases.
- rustc-only probes: `rv3/{m,t,t2,t3}.rs` in this session's scratchpad.

New targets:

| Target | Cases | What |
|---|---|---|
| `method_empty.rs` | 2 tests, 30 + 2 pairs | `::<>` at every method site, compared bit for bit with the plain spelling, `kernel!` and `kernel_raw!` |
| `method_turbofish_{kernel,raw}.rs` | 31 each | non-empty arguments at every method site, plus `::<>` on methods whose plain spelling is refused |
| `method_plain_kernel.rs` | 20 | the plain twin of each `::<>` refusal case, plus `mm31` (non-empty, closure form) |
| `record_where.rs` | 8 | a record's `where` clause in every shape |
| `param_panic.rs` | 1 | a baked `Kernel` holding a `Param`, `should_panic(expected = <the new text>)` |
| `empty_in_types*.rs`, `turbofish_on_type_path.rs`, `empty_generics_items.rs`, `leading_colon_types.rs`, `order_probe.rs` | 3 to 12 each | neighbouring spellings, for the last question |

## Verdict

The round-2 code does what R2C-1, R2C-3 and D1 ask, and nothing else in behaviour changed. Round 2 adds no stage that
accepts what a later one refuses.

There are two findings, both pre-existing and identical at `0c3b52a`, neither blocking:
1. **`::f32` (leading colon):** the front end accepts it as a parameter's or a field's type, and rustc refuses the emitted
   code.
2. **`<>` on a type's path:** four sites still read it as arguments. R2C-3's "`::<>` means no arguments everywhere in the front
   end" holds for names and methods, not for types.

## What was checked and holds

### The 35 round-1 cases, before and after

These are `refuse_kernel.rs` and `refuse_raw.rs`, diffed base against current. The only difference is in two cases, and only
in the span:

| Case | Base caret | Current caret |
|---|---|---|
| w05 `pub struct Bx where { pub x0: f32 }` | the whole `kernel! { … }` call (col 5), with the "originates in the macro" note | `where` (col 49 / 53 raw) |
| w06 `pub struct Bx where f32: Copy { … }` | `where f32: Copy` | `where` |

The other 33 are unchanged: message, line, column and caret, under both macros. That includes:
- `m01` (`X.max::<f32>(Y)`);
- the four `fn` `where` cases;
- the six shadowing cases.

### R2C-1: a record's `where`, empty or not, is refused at `where`

`record_where.rs` (current against base):

| Input | Current caret | Base caret |
|---|---|---|
| tuple `pub struct T(f32) where f32: Copy;` | `where` | `where f32: Copy` |
| unit `pub struct U where;` | `where` | whole call |
| `where f32: Copy,` (trailing comma) | `where` | `where f32: Copy,` |
| two predicates | `where` | `where f32: Copy, f32: Clone` |
| `kernel_raw!`, empty | `where` | whole call |
| private record, empty | `where` | whole call |
| `pub struct Bx<T> where {…}` | `T` ("a generic record") | same |
| `#[repr(C)] pub struct Bx where {…}` | `#[repr(C)]` | same |

The message is unchanged throughout. The record's check order (attributes, then generics, then `where`, then fields) is
unchanged. BACKLOG C12's "it is spanned at the `where` keyword now" is true of both the `fn` and the record refusal.

### R2C-3: `::<>` on a method is the method

**Acceptance** (`method_empty.rs`): 2/2 pass under `PIXELFLOW_ISA=avx2` and 2/2 under `=avx512`. Each `e_*` entry equals its
plain `p_*` twin by `to_bits()` at five samples, under `kernel!` and `kernel_raw!`. The spellings:
- unary methods: `abs`, `sqrt`, `sin`, `floor`, `clone`;
- binary methods: `max`, `min`, `atan2`, `mul_add`;
- the library methods: `fract`, `hypot`, `clamp`;
- `lt` and `select`;
- a helper's body;
- folds: `.map::<>(…).sum::<>()`, `.product::<>()`, `.fold::<>(f32::INFINITY, f32::min)` and `.fold::<>(f32::NEG_INFINITY, f32::max)`;
- `(0..N).any::<>(…)` and `.all::<>(…)` as an `if` condition;
- a method on a record's field;
- a `Staged` entry: a fold around an application, `k(…).max::<>(0.0)`;
- `DX((X * X).sin::<>())`;
- the closure form.

`.sum::<f32>()` is compared with `.sum()` too. The plain twins are checked against hand-computed values: sum 12, product 24,
min 0.25, max 6, and `any`/`all` on each side. That keeps "same" from being vacuous. rustc accepts every one of these `::<>`
spellings (`rv3/m.rs`).

**Non-empty arguments are refused as before.** `mm01`–`mm11` and `mm31` give the same message and caret on the base and on
the current tree:
- `sqrt::<f32>`, `max::<f32, f32>`, `max::<'static>`;
- `sum::<f64>`, `sum::<f32, f32>`, `product::<i32>`;
- `map::<f32>`, `map::<f32, _>`, `fold::<f32, _>`;
- `any::<_>`, `all::<bool>`;
- the closure form's `X.sqrt::<f32>()`.

**An empty turbofish on a method that is refused anyway.** Round 2 moves the refusal from the turbofish to the cause the plain
spelling gets. For each such case, the current message and caret text equal the plain twin's (`method_plain_kernel.rs`), under
both macros. These cases changed from base:

| Case | Base | Now (= plain twin) |
|---|---|---|
| `.sum::<>(X)`, `.product::<>(1.0)` | "over anything but `f32`s" | "`.sum()` takes no arguments", at the argument |
| `X.sqrtt::<>()` | "type arguments on a method" | "unknown method 'sqrtt'" |
| `X.at::<>(X, Y)` | same | the retired-method message |
| `X.sqrt::<>(1.0)` | same | "`sqrt` takes 0 arguments, but 1 was supplied" |
| `const S: f32 = 2.0.max::<>(1.0);` | same | "a `const` initializer is evaluated at expansion" |
| `X.clone::<>(Y)` | same | "unknown method 'clone'" (see Noted) |
| `(0..4).any::<>(\|i\| X + …)` | "`.any` takes no type arguments" | "expected `bool`, found `f32`" |
| `.fold::<>(0.0, f32::min)` | "`.fold` takes no type arguments" | "names no monoid" |
| `.map::<>(\|i, j\| …)` | "`.map` takes no type arguments" | "a fold's closure takes one parameter" |
| `X.lt::<>(Y).sqrt::<>()`, `select::<>` with a `bool` arm, `.map::<>(\|i\| X > …).sum::<>()`, `(0..4).any::<>(…).sqrt()` | turbofish refusals | the plain spelling's type mismatch |
| `X.map::<>(\|i\| i).sum::<>()` | "type arguments" | "`.map` over `X`, which is not a range" |

These four cases are unchanged from base: an earlier or outer refusal applies in both trees.
- `(0..4).map::<>(…)` with no terminal;
- `(0..4).rev::<>()…sum::<f32>()`;
- `[1.0f32].iter::<>().sum::<f32>()`;
- `(0..4).into_iter::<>()…`.

**The three sites, and their test.** `turbofish_of` is the one predicate, and all three method sites call it:
- `parser.rs:756`, the `MethodCall` arm;
- `:1152`, `sum_or_product`;
- `:1249`, `refuse_turbofish`, which `convert_range_method`, `convert_fold_terminal`'s `map` and `fold_monoid` share.

The new test cases (`parser.rs:2668-2673`) reach all three. `X.max::<>(Y)` is the arm, `map::<>` is `refuse_turbofish`, and
`sum::<>` is `sum_or_product`, so reverting any one of them fails the test. `cargo test -p pixelflow-compiler --lib`:
142 passed.

`edge.rs`:
- e01–e03 are now accepted.
- e04 (`::N::<4>`), e05 (`N::<4>::M`) and e06 (`<f32>::X`) are unchanged.
- e07 (`::h::<>(X)`) builds.

### D1's two panics, and tests that assert them

**The JIT emitter.** `param_panic.rs` builds `X + Param(0)` by `ExprArena::push_param` and `Kernel::from_parts`, the macro's
own entry point. It bakes it with `Lattice::eval_at`. The bake reaches the JIT emitter's panic, and
`should_panic(expected = "ExprNode::Param(0) reached the JIT emitter: a `Param` has no value")` passes. Nothing earlier
intercepts the `Param`; the runtime tier's `Declined::Param` does not.

**The compiler's emission.** The panic in `emit.rs:441` cannot be reached from `kernel!`: nothing in `pixelflow-compiler/src`
builds a `Param` (G4).

**Assertions.** No test asserts either panic's text, old or new, and none asserts the three sibling texts round 2 changed:
- `arena_to_kernel_code`'s;
- the edge walker's;
- `collapse_bench` `encode`'s.

I grepped every `.rs` for each old and new phrase and every `should_panic`. The only hits are the panic sites themselves.
`ir_insert.rs`'s two changed assert messages run: `cargo test -p pixelflow-search --test ir_insert`, 13 passed.

### Nothing else changed in behaviour

**The non-comment diff.** `git diff 0c3b52a -U0 -- '*.rs'` with comment lines filtered out leaves exactly:
- the two D1 panics;
- `convert_record`'s span;
- `turbofish_of` and its three call sites;
- the parser test's new cases;
- three sibling panic texts (training `factored.rs`, `nnue/factored.rs`, `corpus.rs`);
- the two `ir_insert.rs` assert messages;
- one trailing comment in `cost.rs`.

**Programs that were accepted before.** A previously accepted program cannot hold `::<>` on a method, and for any other
turbofish `turbofish_of` returns what `&call.turbofish` did. So nothing a previously accepted program does can change.
`gates-fix2.md`'s byte gates are IDENTICAL to base, which agrees. The round-1 targets, `accept.rs` (both ISAs) and `edge.rs`,
behave as before, apart from e01–e03.

### Does any stage accept what a later one refuses?

Not through round 2:
- An empty method turbofish is dropped by the parser.
- `MethodCallExpr` holds no turbofish, so sema, lowering and emission see exactly the plain call.
- Values agree bit for bit through both macros and both ISA tiers, including a `Staged` entry and a `const` initializer.
- A record's `where` never leaves the parser.

Looking past round 2 for the same question turned up finding 1.

## Findings, ranked

### 1. `::f32` as a parameter's or a field's type passes sema and fails rustc on the emitted code (pre-existing; not blocking)

**Evidence.** `leading_colon_types.rs`:
- `pub fn f(a: ::f32) -> f32 { a + X }` expands, and rustc then fails with E0425 "cannot find crate `f32` in the list of
  imported crates", at the user's `f32`.
- `pub struct Bx { pub x0: ::f32 }` does the same.
- `const C: ::f32`, `const N: ::usize`, `let a: ::f32` and `-> ::f32` compile, because the emission writes its own type
  there. In plain Rust, rustc refuses `::f32` in a `const` and in a parameter alike, with the same E0425 (`rv3/t5.rs`).

The output is identical at `0c3b52a`.

**Cause.** `Ty::from_syn` (`pixelflow-compiler/src/sema.rs:126-143`) rejects a `qself` and more than one segment, but never
looks at `path.path.leading_colon`, so `::f32` is `Ty::F32`. The emission then re-emits the declared type's tokens:
- `emit_entry`'s `#name: #ty` (`emit.rs:206`);
- the `Args` record's fields (`:240`);
- `emit_record`'s fields (`:294`).

The line is the same at `7e9531e`.

**Impact.** No wrong value, and the failure is loud and at the author's token. By CLAUDE.md's priority that earns no type.
But it is a stage accepting what a later stage refuses: rustc's message names a missing crate, where the language would have
said "a kernel parameter is an `f32` or a `bool`". R6 left `leading_colon` unrefused for *expression* paths, where `::h` is
the block's `h` and nothing fails. This is the *type* position, which R6 did not consider.

**Fix, if wanted.** Add `|| path.path.leading_colon.is_some()` to `Ty::from_syn`'s first guard. sema then refuses with its
own parameter and field messages. Otherwise, a BACKLOG row beside C11.

### 2. `<>` on a type's path is still read as arguments at four sites (pre-existing; not blocking)

**Background.** R2C-3's rationale is "so `::<>` means no arguments everywhere in the front end". `turbofish_of`'s doc says
"`::<>` names none, as it names none on a path". That holds for a name in a body and for a method. It does not hold for a
*type's* path.

**How the front end reads `f32<>`.** It depends on which helper reads the type.

These read `<>` as none and accept, computing what the plain twin computes (`empty_in_types_accept.rs`, 1/1). They use
`segment.arguments.is_empty()`:
- `Ty::from_syn` (`sema.rs:134`): a parameter `a: f32<>`, a return `-> f32<>`, `const C: f32<>`, `const N: usize<>`, a field
  `x0: f32<>`, `let a: f32<>`, a helper's `fn h(a: f32<>) -> f32<>`;
- `is_path` (`parser.rs:1244`): `.fold(f32::<>::INFINITY, f32::min::<>)`.

These read `<>` as arguments and refuse, each naming a cause that is not the problem. They use `Path::is_ident` or
`get_ident`, which require `arguments.is_none()` (syn 2.0.101 `path.rs:82-91`):

| Site | Input | Refusal |
|---|---|---|
| `parser.rs:1181` `is_f32` | `N as f32<>`, `N as f32::<>` | "`as` converts a `usize` to an `f32`, and to nothing else … help: write `as f32`" |
| `parser.rs:1181` `is_f32` | `.sum::<f32<>>()`, `.sum::<f32::<>>()` | "`.sum` over anything but `f32`s … the one type argument it takes is `::<f32>`" |
| `ast.rs:342` `is_f32` | `impl Fn(f32<>, f32) -> f32`, `impl Fn(f32, f32) -> f32<>` | "a kernel-typed parameter is `impl Fn(f32, f32) -> f32`, and this one is spelled otherwise" |
| `ast.rs:78` `record_named` | `b: Bx<>` for the block's `Bx` | "a kernel parameter is an `f32` or a `bool`" |

rustc accepts every input in the table (`rv3/t.rs`, `t2.rs`, `t3.rs`, `t4.rs`). The outputs are identical at `0c3b52a`. All three helpers
are unchanged since `7e9531e`.

**Impact.** No wrong value, and no stage disagreement: the parser or sema refuses, and nothing later sees the input. It is the
class review2.code's finding 3 raised and R2C-3 fixed for methods: a refusal that names arguments, or a type, that is not
there.

**Fix, if wanted.** One predicate for "this type path is exactly the name `n`": no `qself`, no `leading_colon` (see finding 1),
one segment, `arguments.is_empty()`, ident `n`. Use it in both `is_f32`s, in `record_named` and in `Ty::from_syn`. Otherwise,
scope `turbofish_of`'s comparison to what is true: "as it names none on a name in a body".

## Noted only (pre-existing, identical at `0c3b52a`, not A5's)

- **A refusal names the documented spelling.** `(0..4).rev().map(|i| i as f32).sum::<f32>()` is refused as "type arguments on a
  method in a kernel body … no method is generic", at `::<f32>`. That is the spelling the fold docs prescribe, and the same
  happens for `[1.0f32].iter().sum::<f32>()`. With `.sum()` they say "`.rev` on a range" and "an array in a `kernel!` block"
  (`order_probe.rs`). The `MethodCall` arm returns its turbofish refusal before converting the receiver. The comment above it
  ("the receiver's own refusal, the innermost, comes first") is about `collection`, which is returned after; it is true of that,
  but the turbofish return breaks the principle it states.
- **A typo suggestion names the method itself.** `X.clone(Y)` (and now `X.clone::<>(Y)`) says "unknown method 'clone' / help: did
  you mean 'clone'?".
- **A lifetime is called a type argument.** `X.max::<'static>(Y)` is refused as "type arguments on a method".
- **The parser's module doc omits one refusal.** Its refusal list (`parser.rs:117-150`) names a `where` clause on a `fn` and on a
  `const`, but not on a record. The list is incomplete there, not false.
