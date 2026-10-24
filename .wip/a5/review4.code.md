# A5 closing review: code since round 1, by experiment

**Tree:** `20d8144` (branch `claude/integral-area-work-ojiej6`), lens `git diff 0c3b52a -- '*.rs'`. I made no edit to the repo;
this file is the only one I wrote under it.
**Against:** `.wip/a5/rulings.md` (every round), main `7e9531e` as the base, `CLAUDE.md`.
**Lens:** the turbofish refusals at calls, values and methods; the `where` spans on `fn` and record; `Ty::from_syn`'s
leading-colon refusal (R3C-1); the new sema test and the parser tests' asserts. Does any stage accept what a later one
refuses? Is anything accepted at `7e9531e`, and still meaningful, now refused?

## Verdict

The behaviour is what the rulings ask, and I found no behavioural defect:
- Every non-empty turbofish on a name, at a call or in a value position, is refused with R6's one message, at the turbofish.
- Every empty `::<>`, on a name or a method, computes what the plain spelling computes, bit for bit, under both macros and
  both x86 ISA tiers.
- Every `where` clause, on a `fn` or a record, empty or not, is refused at the keyword.
- `Ty::from_syn` now refuses `::f32`, `::bool` and `::usize` everywhere a declared type is read. That closes the two
  positions where sema accepted and rustc refused the emitted code. The spellings it newly refuses compiled at `7e9531e`,
  but rustc refuses every one of them in plain Rust in every edition, so nothing meaningful is lost.
- No stage accepts what a later one refuses through the changed code.
- CI's clippy and fmt pass.

There are three findings, all sentences in `.rs` files, ranked below. The first two are blocking: an A5 sentence false of the
code. The third is blocking only by the letter of the rule: the false clause predates A5, in a sentence A5 edited.

## How it was run

**Current tree.** Probe crate `/home/user/scratch/a5-review4/head-probe`, with path deps on the repo's crates (as
`a5-review-fix1` has):

```
CARGO_INCREMENTAL=0 CARGO_TARGET_DIR=/home/user/pixelflow/target.noindex RUSTFLAGS="-C llvm-args=-fp-contract=fast" \
  [PIXELFLOW_ISA=avx2|avx512] cargo test --offline --test <target> [--no-run]
```

**Base `7e9531e`, not `0c3b52a`.**
- **The archive:** `git archive 7e9531e` of `pixelflow-{compiler,ir,search,core,codegen}`, unpacked under
  `/home/user/scratch/a5-review4/base-ws/b7/`. It has a workspace root of its own, with the base's `Cargo.lock`.
- **The probe:** `base-ws/b7/probe` is a member of that workspace. The same command builds it, with
  `-p a5-review4-b7-probe`.
- **Why the nesting:** cargo hashes a path package by its path relative to the workspace root. The repo's `pixelflow-ir`
  is `pixelflow-ir`; the base's is `b7/pixelflow-ir`, so the two never share an artifact.
- **Smoke test:** `zz_smoke.rs` passes on both trees.

**Other tools.**
- **Normalizer:** `gen/norm.py` reduces rustc's output to three fields: case module, first message line, and the source
  text under the caret.
- **Outputs:** `*.head.out`, `*.base.out` and `*.norm`, all in `/home/user/scratch/a5-review4/`.
- **Generators:** `gen/*.py`.
- **Mutants:** a copy of HEAD's `pixelflow-compiler` in `mut-ws/m4/`, run by `gen/mut.sh`.
- **Plain-rustc probe:** `rustc/colon.rs`, compiled at editions 2015, 2018 and 2024.

**Two side effects on the machine.** Neither touches the repo.
1. **A shared fingerprint.** My first base attempt (`a5-review4/base7e/`, abandoned) shared the repo's
   relative paths. Its failed build reset the fingerprint `target.noindex/debug/.fingerprint/pixelflow-compiler-8274446ced60db6e/lib-pixelflow_compiler`
   to empty. The `.so` is untouched. The next repo build of that unit recompiles it once.
2. **`/dev/null`.** A stray line in my mutant runner deleted `/dev/null`. The redirections that followed recreated it as a
   regular file. I recreated the character device with `mknod -m 666 /dev/null c 1 3`, and it now reads back 0 bytes.
   Any process that wrote to `/dev/null` between 07:56:2x and 07:56:38 UTC wrote into that regular file instead, which is
   now gone.

## Findings, ranked

### 1. "so a slot with no value never reaches the emitter" is false (blocking)

**Where.** Three sentences, all written in `2121763` (D3-9):
- `pixelflow-search/src/egraph/insert.rs:54-55` (`Declined::Param`): "Only `Vocabulary::Runtime` declines it, so a slot
  with no value never reaches the emitter."
- `insert.rs:88-89` (`insert`'s doc): "Under `Vocabulary::Runtime` a `Param` declines, so a slot with no value never
  reaches the emitter."
- `pixelflow-search/tests/ir_insert.rs:149-150`: "`Runtime` declines it, so a slot with no value never reaches the
  emitter."

**Evidence: the code.** A runtime decline is exactly what sends the term, `Param` and all, to the emitter.
`pixelflow-codegen/src/jit_cache.rs:150-155` calls `optimize_runtime_arena` and, on `None`, compiles `linked_arena` unchanged. Four docs say so:

| Doc | Text |
|---|---|
| `pixelflow-search/src/runtime.rs:167-168` | "Callers compile the original arena unchanged in that case" |
| `saturate_pass.rs:19-21` (A5 edited it in round 2) | "Declining is ordinary: the caller compiles the term unoptimized" |
| `insert.rs:21-23`, the enum the variant belongs to | "a caller that meets one compiles the term unoptimized" |
| `insert.rs:47-48`, the sibling `ShiftCount` | "The term compiles unoptimized, and the emitter refuses it there" |

**Evidence: the experiment.** `head-probe/tests/param_reaches.rs` builds `X + Param(0)` with `push_param` and
`Kernel::from_parts`, then calls `Lattice::eval_at` (`param_reaches.out`, `RUST_BACKTRACE=1`):

```
panicked at pixelflow-codegen/src/program/lower.rs:374:17:
ExprNode::Param(0) reached the JIT emitter: a `Param` has no value
   2: pixelflow_codegen::program::lower::arena_to_schedule
  13: pixelflow_codegen::pipeline::compile
  14: pixelflow_codegen::jit_cache::compile::{closure#2}
```

`ir_insert.rs`'s own test asserts that the same term declines under `Runtime`
(`a_param_is_held_by_the_macro_vocabulary_and_declined_by_the_runtime_one`, passing).

**What is true.** Under `Runtime` the e-graph never holds a `Param`: nothing rewrites around a slot with no value, or extracts
one. The term then compiles as written, and the JIT emitter panics on it.

**Fix.** Say that, or drop the "so" clause. The variant could read: "Only `Vocabulary::Runtime` declines it: the term then
compiles as written, and the JIT emitter panics on the slot."

### 2. "a count is a literal or a `usize` const" is false (blocking)

**Where.** `pixelflow-compiler/src/parser.rs:621-623` is `refuse_generics`' doc, written by A5 (`38703e0`): "A `fn` takes no
generics: a value is an `f32` or a `bool`, a record is one of the block's, a kernel is `impl Fn(f32, f32) -> f32`, and a
count is a literal or a `usize` const, evaluated at expansion."

**Evidence.** A count is any constant `usize` expression:
- The parser's own grammar says so (`parser.rs:36-37`, `iexpr ::= iexpr ('+' | '-' | '*' | '/') iexpr -- a usize: a range's
  bounds`).
- `head-probe/tests/cnt.rs` passes (`cnt.head.out`). It checks `(0..N + 1)` → 6.5 at `X = 0.5`,
  `(N - 1..2 * N)` → 14, and `(0..(2 + 2) / 2)` with `.product()` → 2, each against a hand-computed value.

**Precedent.** D3-10 judged this exact wording false and corrected it in `lib.rs` and `intent.md`, but not here.

**Second site.** The parser's module doc at `parser.rs:121-122` attributes the same claim to the `const` generic refusal: "a
message saying a count is a literal or a `usize` const". Since D3-3 and D3-10, that message (`parser.rs:644-650`) says only
two things: a count does not make a program of its own, and how to write one.

**Fix.** At `:622-623`: "and a count is built from literals and `usize` consts with `+ - * /`". At `:121-122`: "(a `const`
generic by a message saying a count makes no program of its own, §1.2 of the bound plan)".

**Outside this lens, the same wording:**
- `.claude/skills/ontology/language.md:149-150`, the retired Structural parameter's Is;
- `docs/language/2026-10-20-a-bound-is-a-value.md:492`, §5's Becomes cell for the-language-is-kernel `:179`.

### 3. `emit.rs`'s Binding-times sentence: "takes its parameters by their declared types" (blocking by the letter; lowest)

**Where.** `pixelflow-compiler/src/emit.rs:38`: "A host function takes its parameters by their declared types, and every
`f32` and record one is a uniform: …".

**Evidence.** A kernel-typed parameter is declared `impl Fn(f32, f32) -> f32`. The host function takes it as
`&::pixelflow_core::Kernel` (`emit_entry`, `emit.rs:205`). The same file says so twice:
- `emit`'s doc (`:75-76`): "a kernel-typed one as a `&Kernel`";
- `emit_entry`'s doc.

**History.** The false clause is `7e9531e`'s (`emit.rs:38` there). Round 2's D5 edited the same sentence to scope its
uniform clause to `f32` and record parameters, and left this clause.

**Fix.** "A host function takes each `f32` and record parameter by its declared type, and a kernel-typed one as a
`&Kernel`; every `f32` and record one is a uniform: …".

## What was checked and holds

### Generic arguments on a name (R6, S3, S4): one message, at the turbofish

`tf_kernel.rs` and `tf_raw.rs` hold 30 cases, as their own `mod`s. Of those, 25 are non-empty arguments on a name.

**Call arm (12):**
- a helper, with `::<4>`, `::<f32>`, `::<'static>`, `::<4, 5>`, `::<{ 4 }>`, `::<N>` and `::<Item = f32>`;
- `DX::<f32>(X * X)`;
- `k::<4>(X, Y)` on a kernel parameter;
- `::h::<4>(X)`;
- `nope::<4>(X)`, an unknown name;
- a turbofish inside a helper's body.

**Path arm (11):**
- `X::<f32>` and `::X::<f32>`;
- `N::<4> as f32`;
- `p::<4>.x0`;
- `(0..N::<4>)`, a range bound;
- `let q = p::<4>;`;
- a `usize` and an `f32` const's initializer;
- `h2(k::<4>)`, a kernel passed by name;
- `i::<4>`, a fold's index;
- `r::<f32>`, an `f32` parameter.

**Closure form (2):** `X::<f32>` and `DX::<f32>(X * X)`.

**Result.**
- All 25 are refused with one text: "generic arguments in a `kernel!` block" / "note: nothing in a kernel is generic, so no
  `fn` takes generic arguments (§1.3 of docs/language/2026-09-25-the-language-is-kernel.md)" / "help: write a count with
  literals and the block's `usize` consts".
- `uniq -c` over the 25 message blocks gives one block.
- The caret's text is exactly the turbofish (`::<4>`, `::<Item = f32>`, `::<{ 4 }>`, …), never the segment.
- `kernel!` and `kernel_raw!` agree case for case.

**At `7e9531e` (`tf_kernel.base.norm`):**
- The 12 call-arm cases and the closure form's `DX::<f32>` compiled. The arm dropped the arguments, the defect R6 fixed.
  None of them is meaningful: no helper takes generics at `7e9531e` either.
- `nope::<4>` was refused as an unknown function.
- The value cases were refused as "a path in a kernel body".

**Methods keep their per-site messages, unchanged from base.**
- `X.max::<f32>(Y)`: "type arguments on a method in a kernel body".
- `.sum::<f64>()`: "`.sum` over anything but `f32`s".
- `.map::<f32>`, `.fold::<f32, _>`, `.any::<_>`: "`.<m>` takes no type arguments".

R6's one message is for names, and the rulings ask nothing more of methods.

### An empty `::<>` is the bare name, on a name and a method (S3, R2C-3)

`te_kernel.rs` and `te_raw.rs` hold 16 tests. Each compares an entry written with `::<>` against its plain twin by
`to_bits()` at four points:
- a helper call, plus `DX::<>` and `V::<>`;
- `k::<>(X, Y)` on a kernel parameter, alone and inside a fold (both `Staged`);
- `N::<> as f32 + X::<> * Y::<>`;
- `p::<>.x0`;
- a range bound `(1..N::<>)`;
- `let q = p::<>;`;
- a `usize` and an `f32` const's initializer;
- `h2(k::<>, X, Y)`;
- `::h::<>(X) + ::Y`;
- `i::<> as f32` on a fold's index;
- an `f32` parameter `r::<>`;
- `X.max::<>(Y)` with `.map::<>(…).sum::<>()`;
- the closure form.

**Result:** 16/16 pass for each macro under `PIXELFLOW_ISA=avx2` and under `=avx512`.

**The values are not vacuous.** I checked several by hand:
- the helper: 5;
- `k::<>`: -1;
- the fold of `k::<>`: 7.5;
- `h2`: 0.5;
- the range: 3;
- `M + X`: 5.5.

**At `7e9531e`:** every name case compiles, and only `X.max::<>(Y)` is refused ("type arguments on a method"). So R2C-3 adds
an acceptance and takes none away.

### `where` (S5, R2C-1): at the keyword, `fn` and record alike

`wh_kernel.rs` and `wh_raw.rs` hold 11 cases.

| Item | Spellings | Caret's text |
|---|---|---|
| entry | a predicate; empty | `where` |
| helper | empty; two predicates with a trailing comma | `where` |
| named record | empty; a predicate | `where` |
| tuple record | `where` | `where` |
| unit record | `where` | `where` |
| `fn f<>() … where {}` | — | `where` |
| `struct Bx<> where {}` | — | `where` |
| `fn f<T>() … where T: Copy` | — | `T` (the generic comes first) |

The messages are unchanged. Both macros agree.

At `7e9531e` the empty cases pointed at the whole invocation, and the others at the whole clause.

### `Ty::from_syn` and a leading `::` (R3C-1), against `7e9531e`

Sources: `lc_kernel.rs` and `lc_raw.rs` (30 cases), and `lcv.rs` (values at base).

**Spellings sema now refuses, which compiled at `7e9531e`.** Each one, at base, computes its plain twin's value bit for bit
at four points (`lcv.base.out`, 10/10):
- a helper's `::f32` and `::bool` parameter;
- `let a: ::f32`, `let m: ::bool`, and the tuple `let (a, b): (::f32, f32)`;
- an entry's `-> ::f32`, and a helper's `-> ::bool`;
- `const C: ::f32` and `const N: ::usize`;
- the closure form's `|a: ::f32|`.

At HEAD each is refused at the type, by sema's own message for that position.

**These are not meaningful Rust.** `rustc/colon.rs` puts `::f32`, `::bool` or `::usize` in each of those positions. Plain
rustc refuses every one:
- at edition 2015, "cannot find type `f32` in the crate root";
- at 2018 and 2024, E0425 "cannot find crate `f32` in the list of imported crates".

So nothing accepted at `7e9531e` and still meaningful is now refused.

**The two disagreements R3C-1 fixed.** At `7e9531e`, `pub fn f(a: ::f32)` and `pub struct Bx { pub x0: ::f32 }` pass the
macro, and rustc then fails on the emitted code with E0425, at the user's `f32`. At HEAD sema refuses both:
- the parameter: "a kernel parameter is an `f32` or a `bool`";
- the field: "a record's field is an `f32`, and `x0` is a type the language does not have".

Those are the only positions whose declared type emission spells back: `emit_entry`, the `Args` fields and `emit_record`. The
closure form, `const`s, `let`s and returns write their own type, which is why the base compiled them. With these two
refused, no position remains where sema accepts and rustc refuses a leading `::`.

**Already refused at `7e9531e`, same at HEAD.** The header is the same in each; some notes differ.
- `::usize` as an entry or helper parameter, a `let` or a return. The note changes from "a `usize` is a count" to "every
  value in a kernel body is one of the two".
- `::bool` as an entry parameter. The header changes from "an entry's parameter is an `f32` or a record" to "a kernel
  parameter is an `f32` or a `bool`".
- `::bool` and `::usize` as a field. "is a `bool`" becomes "is a type the language does not have".
- `const B: ::bool = true`, refused at `true` in both.

**Unchanged and identical in both trees.**
- `impl ::core::ops::Fn(f32, f32) -> f32`, as an entry's or a helper's parameter: "a kernel-typed parameter is
  `impl Fn(f32, f32) -> f32`, and this one is spelled otherwise", from `FunctionType::of_bounds`'s "`Fn`, by its own name".
- The same for `impl ::Fn(…)`, `impl core::ops::Fn(…)`, `impl Fn(::f32, f32) -> f32` and `impl Fn(f32, f32) -> ::f32`.
- `b: ::Bx`, refused by `get_ident` in `record_named`.
- `N as ::f32`.
- `.sum::<::f32>()`.
- `a: std::primitive::f32`.

rustc accepts `impl ::core::ops::Fn(f32, f32) -> f32` and `std::primitive::f32` (`colon.rs` at 2018 and 2024), so these two
are meaningful spellings refused. But they are refused by code A5 did not touch, identically at `7e9531e`. R3C-1 adds no
refusal of a kernel type.

**No newly accepted spelling.** The change only adds a refusal.

### The tests, and whether their asserts discriminate

**Runs:**
- `cargo test -p pixelflow-compiler --lib`: 143 passed, including `a_type_named_from_the_crate_root_is_refused`,
  `generics_are_refused_and_a_const_generic_says_what_a_count_is` and `type_arguments_on_a_method_are_refused`.
- `cargo test -p pixelflow-compiler`: 299 passed, 0 failed, over 15 targets. This matches R3C-1's count.
- `cargo test -p pixelflow-search --test ir_insert`: 13 passed.

**Mutants** (`mut.*.out`; the pristine copy also gives 143):

| Mutant | Result |
|---|---|
| m1: drop `Ty::from_syn`'s leading-colon guard | `a_type_named_from_the_crate_root_is_refused` fails |
| m2: the call arm refuses an empty `::<>` too | `generics_are_refused…` fails (`h::<>(X)`) |
| m3: the path arm refuses an empty `::<>` too | `generics_are_refused…` fails (`N::<> as f32 + X::<>`) |
| m5: the call arm drops arguments, as at `7e9531e` | `generics_are_refused…` fails (`h::<4>(X)`) |
| m6: the path arm sends non-empty arguments to "a path in a kernel body", as at round 1 | `generics_are_refused…` fails |
| m7: a `fn`'s `where` spanned on the clause | all 143 pass |

m7 surviving is ruled: spans stay unpinned (F4, BACKLOG C12). Round 3's review already showed that `turbofish_of`'s three
method sites are each pinned by `type_arguments_on_a_method_are_refused`.

### CI

- `CARGO_TARGET_DIR=/home/user/scratch/a3-close1/t199 cargo +1.99.0 clippy --workspace --all-targets --all-features -- -D warnings`
  exits 0. Its only `warning:` line is core-term's X11 build note (`clippy199.out`).
- `cargo fmt --check` over compiler, search, ir, codegen and pipeline exits 0.
- `git diff 0c3b52a HEAD -- pixelflow-graphics` is empty: the gate probe `2121763` committed is gone.
- No assert, test or `should_panic` was removed since `0c3b52a`. The two replaced asserts pin new text:
  - the parser's R6 help line;
  - `ir_insert.rs`'s assert messages.

### Does any stage accept what a later one refuses?

Not through the code since round 1.
- **Empty turbofish:** an empty `::<>` on a name becomes the `Ident`. On a method it becomes a `MethodCallExpr`, which holds
  no turbofish. Sema, lowering and emission see the plain spelling, and the values agree.
- **Non-empty turbofish:** it never leaves the parser.
- **`where`:** it never leaves the parser either.
- **A leading `::` in a type:** it never leaves sema. `lower.rs:897` reads a `const`'s type through the same
  `Ty::from_syn`, after sema accepted it.
- **`f32<>`:** this is R3C-2's subject, refused as a change. `Ty::from_syn` still accepts it, and emission spells it back,
  but rustc accepts `f32<>` (round 3, `empty_in_types_accept.rs`), so the stages agree.
- **Empty generic lists:** `pub fn f<>()`, `fn h<>()` and `pub struct Bx<>` compile and give 4.5 in both trees (`eg.rs`).

## Noted only (not blocking)

- **R3C-1's pin is an entry parameter only.** The record field is the other position whose type is spelled back to rustc,
  and it is not pinned. m1 shows the guard is pinned while it lives in `Ty::from_syn`. A guard moved into `param_type`
  would leave `x0: ::f32` free to reach rustc again, with no test failing.
- **`generic_refusal`'s doc** (`parser.rs:640-641`): "a `const` one says what a count is now". Since D3-3 the message says
  what a count does not do, and how to write one. That is a stretch, not a falsehood.
- **The `::bool` entry-parameter message** says "a kernel parameter is an `f32` or a `bool`". The author's next try,
  `bool`, is then refused with "an entry's parameter is an `f32` or a record". This is loud and at the token, and it is the
  same shape every non-scalar entry parameter gets (`u8`, …).
