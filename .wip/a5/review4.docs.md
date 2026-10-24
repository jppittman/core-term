# A5 review, round 4: docs truth (closing)

**Tree:** HEAD `20d8144`, branch `claude/integral-area-work-ojiej6`. The diff is `7e9531e..HEAD` outside `.wip/`.
**Order:** round 3's hand edits (`git diff 08047ae`) first. Then every sentence of the whole A5 diff that
`review{1,2,3}.*.md` did not already clear, and the items the task names.
**Method:** each sentence was checked against the code, not against the rulings. Read-only on the repo; this file is the
only one written there.

**What was run:**
- The existing probe crate `/home/user/scratch/a5-review-fix1`, `--test param_panic`, against `target.noindex`.
- In the checkout:
  - `cargo test --locked -p pixelflow-compiler --lib`: 143 passed.
  - `cargo test --locked -p pixelflow-search --test ir_insert`: 13 passed.
  - `cargo fmt --check` on the three crates round 3 touched: clean.
  - `scripts/check-doc-paths.sh` and `scripts/check-intent.sh`: both OK. Both are read-only, and `git status` was
    clean after them.

## Findings, ranked

### 1. Blocking: round 3's reason for "only Runtime declines a `Param`" is false

**Where:**
- `pixelflow-search/src/egraph/insert.rs:54-55` (`Declined::Param`).
- `:88-89` (`insert`'s doc).
- `pixelflow-search/tests/ir_insert.rs:149-150`.

All three were written by D3-9. The sentence is: "Only `Vocabulary::Runtime` declines it, so a slot with no value never
reaches the emitter."

**The decline is what sends the `Param` to the emitter.**
- `saturate_pass.rs:20-21`: "Declining is ordinary: the caller compiles the term unoptimized."
- `runtime.rs:150-151` lists `Param` among the constructs for which `optimize_runtime_arena` returns `None`, and
  `:167` says "Callers compile the original arena unchanged in that case".
- `jit_cache.rs:137-141` names "a `Param`" among the declines that "still compile … just without the extra fusion".
  `emit_fn` (`:150-157`) falls back to the linked arena with `unwrap_or((linked_arena, linked_root))` and calls
  `emit::compile`.
- `pixelflow-codegen/src/program/lower.rs:373-374` is where it stops:
  `panic!("ExprNode::Param({i}) reached the JIT emitter: a `Param` has no value")`.

**By experiment, on HEAD.** `tests/param_panic.rs` builds `X + Param(0)` with `ExprArena::push_param` and
`Kernel::from_parts`, which is the macro's own entry point. It bakes the kernel with `Lattice::eval_at` and expects the
emitter's panic:

```
cd /home/user/scratch/a5-review-fix1
CARGO_TARGET_DIR=/home/user/pixelflow/target.noindex RUSTFLAGS="-C llvm-args=-fp-contract=fast" \
  cargo test --offline --test param_panic
test a_bake_of_a_param_meets_the_jit_emitters_panic - should panic ... ok
```

`review3.code.md:159-161` found the same on the round-2 tree: the runtime tier's `Declined::Param` does not intercept
it.

**The other vocabulary does no better.**
- `Templates` holds a `Param` as `ENode::Param` and extracts it.
- Its output goes to the compiler's emission (`arena_to_tokens`), which panics at `emit.rs:440-441`.

So neither vocabulary's choice keeps a slot with no value away from an emitter. What keeps it out of a compiled kernel is
the emitter's panic. D3-9 asked for "the true reason". The code has no reason that tells the two vocabularies apart;
review3's finding 9 found that too, for the previous wording.

**Fix:** state what happens and give no reason. For example: "Only `Vocabulary::Runtime` declines it: a declined term
compiles unoptimized, and the emitter panics on the `Param` there. `Vocabulary::Templates` still holds one as
`ENode::Param`, because nothing has deleted it (the-language-is-kernel D2)." Make the same change at `insert.rs:88-89`
and `ir_insert.rs:149-150`. The asserted substrings at `ir_insert.rs:165` and `:172` need no change.

### 2. Non-blocking: the diff gives two answers for what a count is

**After D3-10:**
- `lib.rs:278-279` says a count is "built from literals and `usize` consts with `+ - * /`".
- `intent.md:14-15` says it is "built from literals and `usize` consts at expansion".

**Five sentences A5 wrote still say a count *is* a literal or a `usize` const:**
- **The-language-is-kernel `:197`** (the §1.3 table): "a fold binder or a count (a literal or a `usize` const)".
- **Plan `:492`** (§5's `:179` row): the Becomes and CL cells.
- **Ontology `language.md:149-150`** (Structural parameter (retired)): "A count is a literal or a `usize` const, known
  at expansion".
- **`parser.rs:621-623`** (`refuse_generics`' doc): "a count is a literal or a `usize` const, evaluated at expansion".
- **`parser.rs:121-122`** (the module doc's refusal list): "a `const` generic by a message saying a count is a literal
  or a `usize` const". Since D3-3, the message at `:644-649` says neither. Its note is "a count does not make a program
  of its own", and its help says how to write a count.

**What the code accepts:**
- `UsizeScope::eval_usize` (`sema.rs:1651-1700`) evaluates literals, names, parentheses and checked `+ - * /`.
- `range_bounds` (`sema.rs:1715`) evaluates each bound with it.
- The oracle ranges over `FIRST..FIRST + N` (`rustc_is_the_oracle.rs:662`, `:667`).

**Why it is not blocking:** read as "the value, once evaluated, is a literal", each sentence is true. Read as the
spelling, `FIRST + N` is neither a literal nor a const. Review3 ranked the same gap non-blocking (its finding 10), and
D3-10 fixed only some of the sites.

**Fix:** use D3-10's phrase at all five sites.

### 3. Non-blocking: the-language-is-kernel `:202-204` and its D3 (`:681`) now disagree, and the tree's count is D3's

**The two texts:**
- A5's new `:204`: "A count of things is not a type. It is a fold's bound, a value (§1.2)."
- D3 at `:681` still says "a count is how many instances the host composed … (§1.3, §1.6)". It cites §1.3 for the
  sentence A5 replaced.

**The tree's one count of things is D3's.** `program.rs:240-246` builds a glyph's program from `(0..pieces)` instances,
composed by `ink`'s `sum2` tree. No fold runs over the count.

**Inside `kernel!`, `:204` is true:** a count appears only as a bound or as `N as f32`. D3 is scheduled (§5 `:494`,
"A1, after #1323"), so this is not blocking.

**But the edit dropped "(§1.7)",** the pointer to the section that describes the tree.

**Fix:** "In `kernel!` it is a fold's bound, a value (…§1.2). The font's piece count is how many instances the host
composed (§1.7) until B2."

### 4. Non-blocking, beside an A5 edit: `symbol.rs:37` still says "An entry's parameters are its uniforms"

**The twin:** this is the sentence that D3-12 scoped at `lower.rs:33`.

**What A5 did:** it deleted this paragraph's last sentence (`symbol.rs:42-44` at the base), the structural exception.
That left the opening sentence universal.

**Counterexamples:**
- The file's own table has the row `Kernel | Per call (host) | The argument's term…` (`:15`).
- `AnalyzedKernel::parameters` filters kernel-typed parameters out (`sema.rs:285-291`).

**Fix:** "An entry's `f32` and record parameters are its uniforms".

### 5. Non-blocking, cleared before but not true: the ontology's "Args is each entry's record of its uniform arguments"

**Where:** `.claude/skills/ontology/language.md:95`. A5 changed one word of this sentence ("non-structural" to
"uniform"), and review1 cleared it.

**Why it is not true:** an entry with no parameters has no `Args` record, and neither has one that takes a kernel.
- `has_args_record` is `!params.is_empty() && !takes_a_kernel()` (`ast.rs:195-202`).
- `emit_args` returns nothing otherwise (`emit.rs:224-226`).
- `lib.rs` says the same for a kernel-typed entry ("It has no `Args` record").

**Fix:** "an entry's record of its uniform arguments, when it has some and takes no kernel".

### 6. Outside A5's diff, for the record only: `pixelflow-ir/src/arena.rs:404-411`

**What it says:** `ExprNode::Var`'s doc, seven lines above `Param(u8)`, lists a third meaning of `Var`: "in the macro
front end only, before substitution — a parameter placeholder".

**What other docs say:**
- CLAUDE.md says that meaning "went out with the macro parameter that needed it".
- The ontology's `Var` entry says: "Not a macro parameter (`Var(16+i)`, `PARAM_VAR_BASE`, deleted 2026-09-08)".

**Why no sweep caught it:** it is not A5's sentence, and round 2's D7 proximity sweep missed it because its subject is
`Var`, not `Param`.

## Checked, and holds

### Round 3, edit by edit

- **compiler.md, Program.**
  - A template is `arena_to_tokens`' output, or `Staged`'s for an entry that takes a kernel (`emit.rs:106-115`).
  - An entry that takes no kernel is one key and one compiled region (`binding_times.rs:99-117`).
- **language.md, `kernel!` Lives and Entry.**
  - `macro_tier()` is `DwrtFree(Saturate::macro_tier())` (`lib.rs:368-401`).
  - `Saturate` declines only when `insert` returns `Err` (`saturate_pass.rs:74`). From `kernel!`, that happens on a
    `Dwrt`, or on `&`/`|`, which `Templates` does not admit.
  - A `Staged` entry never reaches the optimizer (`emit.rs:108`).
- **language.md, Binding time.** The count sentence is gone, and nothing left in the entry contradicts its "a uniform
  bound is a per-call count".
- **BACKLOG's next cell, the Order paragraph, A1's Done line and §5's `:186` row.**
  - They agree with each other and with §5's Hunk column.
  - #1431 is `9c33002`, an ancestor of `7e9531e`. #1323 is not on `origin/main`.
- **The-language-is-kernel banner, "a count makes no program of its own".** It holds as "no entry is a family of
  programs over a count". Two different literal counts are still two programs, because a known bound is node data in
  the key (plan §1.2's table), so the sentence does not mean "a count is not in the key". Not a finding.
- **The plan's A5 bullet.**
  - At `7e9531e`, the `parser.rs` unit tests (`:2387`, `:2545`, `:2765`) and the `lower.rs` ones (`:1167`) use
    `<const N`.
  - "L3 refuses it" holds: `close_over` calls `push_reduce`, which calls `admits_bound` on `variance_from`
    (`arena.rs:1000-1015`), and the `Param` arm returns `Variance::ALL`.
- **intent.md.**
  - Every refusal in `parser.rs`, `sema.rs` and `lower.rs` that cites a plan names it by `{PLAN}` or `{BOUND_PLAN}`.
    The only other plan file named in a message, `KERNELS_AND_LATTICES.md`, is in a panic.
  - Behind matches `emit_kernel`.
- **lib.rs.**
  - An entry's parameters are bound at two times.
  - The count sentence matches `eval_usize`.
  - `N as f32` converts only a name (`type_of_cast`, `sema.rs:905-912`).
- **The scoping edits** at `lower.rs:33`, `sema.rs:30-31`, `parser.rs:97-103`, `binding_times.rs:4-10` and `:99-100`
  hold.
- **parser.rs.** The two messages, the test's doc and its asserts agree. `turbofish_of`'s doc holds.
- **sema.rs, the leading-`::` refusal and its test.**
  - The test passes.
  - `record_named` already refused `::Bounds` through `get_ident`, and `is_f32` refuses `::f32` through `is_ident`.
  - The test's reason holds: `emit_entry` spells `#name: #ty` back out.
- **emit.rs, `Staged::open`.** It matches `open_fold`/`close_fold` (`emit.rs:699-722`).
- **variance.rs.** The L3 half holds at `push_reduce`. The e-graph still classes a `Param` as `CONST`
  (`node.rs:589-595`), which is the split D3-7 left as it is.

### The rest of A5

- **The Done line's gate clause after round 3's code change.** `cmp` finds the a5-fix3b outputs identical to a5-main:
  byteprobe on both tiers, gate1b and gate2 `.lines`, and gate2x. The commit-mid-gate slip of `2121763` is reverted:
  the graphics diff against `7e9531e` is `program.rs` alone.
- **The A5 Done line's names, the A7 bullet, §3.4 and §4** are as review3 cleared them. The exemption at base is
  `emit.rs:154-156` (`if !entry.structural.is_empty()`).
- **BACKLOG C11 and C12** are unchanged since review2 cleared them. C12's premises still hold:
  `pixelflow-compiler/Cargo.toml` has no `trybuild`, no `compile_fail` and no `span-locations`.
- **The `Param` docs.** These are each true:
  - In `pixelflow-ir`: `kind.rs`, `expr.rs`, `term.rs`, `fold.rs` and `kernel.rs`.
  - In `pixelflow-search`: `node.rs`, `runtime.rs`, `cost.rs`, `extract.rs`, `graph.rs` and `saturate_pass.rs`.
  - The codegen and compiler panics.
  - The pipeline's `factored.rs` and `corpus.rs`.

  The only exceptions are `insert.rs` and `ir_insert.rs` (finding 1).
- **The ontology edits** are as above, plus the retired entry's "§4, 'Deferred'" (the plan's §4 has that paragraph),
  `terminal.md`, and `SKILL.md`'s index line.
