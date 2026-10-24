# A6 spec critique: host surface and public API

Spec: `.wip/a6/spec.md`. Checked against main `29980bf` (the tree is that plus `.wip/`), the plan
`docs/language/2026-10-20-a-bound-is-a-value.md`, `.wip/a6/design.md` (its R-A6 section wins),
CLAUDE.md and `pixelflow-compiler/intent.md`. Read-only. I built nothing: every answer below comes
from the code at `29980bf`, with file:line.

## Verdict

The host code the spec emits is sound. The host function's assert, both `write_into` arms, the
`Args` field, `n as f32` as the default and as the written value, and the `# Panics` doc all expand
without a prelude, and they stay in step with how lowering declares uniforms. One program serves
every `n`, and relink keeps the count in its declared slot. Every public item the spec adds or
changes is named. D-A6-1's arm (a) is flagged for JP as beyond decision 7.

The recommendation is the weak point. **(a) is not the least public-API change that refuses
`2²⁴ + 1` where it is written: (b) is.** The reasons the spec gives for (a) are contradicted by the
spec's own D-A6-4 and by `pixelflow-core`'s existing per-call refusals. One doc site the spec's own
standard says must change is missing (finding 1). Six findings, ranked. Only finding 1 blocks, and
its fix is one sentence.

## The lens questions, answered

**Is D-A6-1's recommendation the least public-API change?** No. (b) adds no item and changes no
existing signature. (a) adds `pixelflow_core::WriteRefused`. It also changes the generated
`write_into` signature of every `Args` record in every crate, including every record with no count:
`OnePieceArgs` and all the test records. For those records the type then admits a `Count` refusal
they can never produce. See finding 2.

**Is every public item named, and flagged where it goes beyond decision 7?** Yes. The spec names:
- `WriteRefused` (§4.4, §4.5, §9 Decisions);
- the generated signature change under (a) (§4.4, §5.3, §9);
- the host function's new panic and `# Panics` doc (§4.2, J11);
- the `Args` field `pub n: usize` (§4.3);
- (b)'s panic on `write_into` (§4.5).

`Monoid::takes_a_value_bound` stays `pub(crate)` (J5). `KnownRange::admits` is already public at
the IR's root and is reached through the doc-hidden `__macro` (J9). `Count`, `Parameter`,
`ParameterKind`, `Scalar` and `range_bounds` stay `pub(crate)`. The compiler is a proc-macro crate,
so nothing else can leak from it. Only `WriteRefused` goes beyond decision 7's list
(plan:783-785), and the spec flags it.

**Is D-A6-4's panic consistent with how the host function treats bad input today?** Yes in kind.
A staged host function already panics on a bad argument: `ExprArena::admit` asserts that a
kernel-typed argument reads no table and is closed (`pixelflow-ir/src/kernel.rs:1197-1219`, pinned
by `tests/kernel_typed_parameters.rs:400-415` with `#[should_panic]`). It also panics on binder
exhaustion (`Staged::close_fold`'s `expect`, `emit.rs:719-723`). The builders panic too
(`Kernel::over`, `Kernel::over_values`, `kernel.rs:809-844`). Two details are inconsistent:
- the generated `# Panics` section is complete for an entry that takes no kernel, and incomplete for
  one that does (finding 4);
- J10's stated reason contradicts how a `usize` const past 2²⁴ is treated (finding 5).

**Does one program serve every `n`?** Yes. The JIT cache keys on the canonical form of the arena
as handed in, plus the shape (`pixelflow-codegen/src/jit_cache.rs:122-126, 158-165`). The canonical
form numbers uniforms by slot and holds no defaults (`binding_times.rs:70-75`). A value-bounded
fold's key is the same for two defaults (`pixelflow-ir/tests/value_bounds.rs:154-170`). Nothing in
`pixelflow-ir`, `-search` or `-codegen` reads a `UniformDecl`'s default except equality, hashing,
ordering and one test (`grep -rn '\.default\b'`). So `sum_to(3)` and `sum_to(7)` share one key and
one compiled region, and the second call is a cache hit. The spec's `every_count_is_one_program`
pins this.

**Does relink keep the count in its declared slot?** Yes. `relink` copies the whole declared
table: `out.uniforms = uniforms.to_vec()` (`arena.rs:2101-2102`). It remaps every reachable
`Uniform` leaf by identity (`:2084-2098`), and a value bound is reachable through `children`, the
same walk `relink` and `reached` use. Lowering declares uniforms in `parameters().scalars()` order
(`lower.rs:393-400`, and the spec's §3.2 keeps it). `arena_to_tokens` declares slot `i` with
`arguments[i]` from that same order (`emit.rs:138-143, 368-376`). So `scaled_sum(b, n, s)` is
`b.x0..b.y1, n, s` in the kernel, the block and `Args`. The spec gives the wrong reason for this
(finding 6), but its conclusion holds.

## Findings

### 1. A doc the spec's own standard requires to change is left stating one check (blocking)

`docs/language/2026-09-25-the-language-is-kernel.md:262-264` says:

> `write_into` streams the values by position into any `UniformBlock` (`set_declared`), and the one
> check is the count (`ArityMismatch`).

After A6 both arms add a second check, the domain of each count parameter. Under (a) the refusal's
type is `WriteRefused`, not `ArityMismatch`. The spec's §7.4 rewrites the ontology's copy of this
sentence (`.claude/skills/ontology/language.md:107-109`, "checks only the arity and each count's
domain"), and the ontology sentence quotes this doc. But §7.2 edits this doc only at `:197`.

**Fix:** add `:262-264` to §7.2, worded like the ontology edit. Under (a), name `WriteRefused`.

### 2. D-A6-1's recommendation is not the least change, and its stated reasons cut the other way (non-blocking: flagged for JP)

The spec rejects (b) because a write would have "two failure modes on one call (an `Err` for arity,
a panic for a count)" (§4.4, spec:876-884). It recommends (a) because "errors stay values on the
per-call path". The tree argues the other way.

- **D-A6-4 already makes the same value a panic on the per-call path.** `sum_to(16_777_217)` panics
  (§4.2, §4.6). A caller who bakes per call (`lattice.bake(&sum_to(n))`, as every rebind test
  compares against) meets the count as a panic. Under (a), the same out-of-domain `n` is a panic at
  one host write and an `Err` at the other. Under (b), both host writes refuse it the same way.
- **`pixelflow-core`'s per-call path already panics on plausible-pixel refusals.**
  - `BoundManifold::with_uniforms` panics on a block from another program, and its doc names
    plausible pixels as the reason (`manifold.rs:670-681`). Every rebind calls it right after
    `write_into`, so the sequence already mixes an `Err` with a panic.
  - `Manifold::bind` panics on a buffer of the wrong length (`manifold.rs:528-536`).
  - `pixelflow-core/intent.md:30-31` records the convention.
  - The plan's other host-written count, B1's extent, is refused at `bind` (plan:728-730), which
    panics.
- **(a) widens every existing `write_into`.** A record with no count gets an error type with a
  variant it cannot produce. That is the reverse of "if a type's signature already refuses the wrong
  shape" (CLAUDE.md, "Subtract before you add"). It also changes `OnePieceArgs::write_into` in
  `pixelflow-graphics`, which has no count.

There is also an option between the two that the spec does not list: (a′). Only an entry with a
count would return `Result<(), WriteRefused>`, so no existing generated signature changes. It still
adds the core item.

**Fix:** recommend (b), the least change, consistent with D-A6-4 and with core's per-call refusals.
If (a) stays the recommendation, §4.4 should put the evidence above in front of JP, list (a′), and
drop "two failure modes on one call" as a reason against (b).

### 3. The generated `write_into` doc keeps "The count is all that is checked" (non-blocking)

`emit.rs:266-273`, emitted on every `Args::write_into`, ends: "Nothing is written. The count is all
that is checked: a block of another program that declares as many arguments takes these values by
position". Both arms keep this text:
- (a) keeps "[the rest of :268-273, unchanged]" (spec:981-988);
- (b) keeps the whole `# Errors` section (spec:1002).

After A6 the sentence is false, because the count parameters' domain is checked too. It is also
ambiguous, because "count" now names a parameter type. Under (b) it sits directly above a
`# Panics` section about a count.

**Fix:** reword it in both arms, for example: "Of the block, only the number of arguments is
checked: a block of another program that declares as many takes these values by position."

### 4. A staged entry with a count gets an incomplete `# Panics` section (non-blocking)

J11 rejects "an undocumented panic on a generated public function" (spec:2058-2059), and §4.2
emits `# Panics` naming only the count (spec:798-805). But `emit_entry` is shared with staged
entries (spec:852). A staged host function also panics on a table-reading or open argument
(`kernel.rs:1197-1219`, `kernel_typed_parameters.rs:400-415`) and on binder exhaustion
(`emit.rs:719-723`). Today none of these is documented. So `columns_to(k, n)`'s generated doc would
present the count as its only panic.

**Fix:** for a staged entry, have the generated `# Panics` section also name the argument refusals
and binder exhaustion (or emit one for every staged entry). Otherwise, word the count line so it
does not read as the complete list.

### 5. J10's stated reason contradicts how a `usize` const past 2²⁴ is treated (non-blocking)

J10 and §1.3 justify checking every count parameter, read or not, as "a count's domain is its
type's: `n as f32` past 2²⁴ is another count whether a fold or an `as` reads it" (spec:106-107,
2055-2057). For the same type, `Ty::Usize`, the tree does the opposite:
- a `usize` const `as f32` follows Rust's `as`, "exact up to 2²⁴", and is accepted past it
  (`sema.rs:1850-1856`; `lower.rs:640-660`, `count as f32` with no domain check);
- only as a bound is a const past 2²⁴ refused (`literal_range`).

So `const N: usize = 16_777_217;` with `X * (N as f32)` expands, while `pub fn f(n: usize)` with
`X * (n as f32)` panics at `f(16_777_217)`. In the second case the kernel would compute exactly
rustc's `x * (16777217usize as f32)`. One type would have two domains, split by binding time.

**Fix:** keep the single check, but give the reason as the plan's denotation of a count parameter
(plan §1.4, "A count denotes an integer in `[0, 2²⁴]`") rather than "its type's". Record the const
path's asymmetry as pre-existing, or refuse a `usize` const past 2²⁴ at expansion too.

### 6. The spec's reason the count keeps its slot is wrong; the conclusion holds (non-blocking)

§5.1 says: "`relink` … restores the declaration order; a count read only as a bound is reached
through its fold's children, so it is kept" (spec:1095-1096, carried from design §4). `relink`
keeps every declared uniform, read or not (`arena.rs:2044-2050, 2101-2102`). That is why
`an_unread_parameter_binds_positionally` holds. Reaching the bound through `children` buys
something else: the `Uniform` leaf under a value bound is remapped to its slot, so `lo.map(m)` at
`arena.rs:2129` finds it densified.

**Fix:** say that: the table is copied whole, and reachability through the fold's children remaps
the bound's leaf.

## Not findings (checked)

- **Host-code mechanics.** `0..n as u64` parses as `0..(n as u64)`. `self.n as f32` is
  `(self.n) as f32`. `u64`, `f32` and range syntax resolve in a `#[no_implicit_prelude]` module (the
  language prelude). `map_err` is inherent. The format string is a literal token with one `{}`.
  `BOUND_PLAN` holds no brace.
- **The test spellings.** `#[should_panic]` works under `panic = "abort"`, since tests ignore it.
  `let _refused = …` trips neither `unused_must_use` nor `let_underscore_must_use`.
- **The domain predicate.** `KnownRange::admits(&(0..n))` is `n <= Fold::EXACT_BOUND`
  (`fold.rs:563, 660, 1235`).
- **What core must change under (a).** It is in the §4.5 switch: the re-export (behind the existing
  `cfg`), intent.md's Contract, and one test. No README edit is needed: none names `ArityMismatch`.
- **Callers.** `write_into` has no production caller (`grep -rn write_into --include=*.rs`), so
  neither arm breaks production. `OnePieceArgs` lives in a private module (`loop_blinn.rs:1040-1041`).
