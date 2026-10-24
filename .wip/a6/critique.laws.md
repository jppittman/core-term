# A6 spec critique: semantics and the laws

**Against:** `.wip/a6/spec.md` at `29980bf` (tree = main plus `.wip/`), the plan
`docs/language/2026-10-20-a-bound-is-a-value.md`, `.wip/a6/design.md` (R-A6 wins),
`CLAUDE.md`, `pixelflow-compiler/intent.md`. Read-only; nothing tracked was edited.

**Verdict.** The pipeline the spec prescribes computes the fold's value or NaN for every
spelling and every value class of `n` that was asked about. It does not hang and does not
run an out-of-range trip count, on AVX2 and AVX-512. NEON was not run here. L3 holds by type
and by scope, L2 by the guard's `run` test, L5 by the guard plus the host refusal, and L9
by the uniform's dense slot. `2^24 + 1` is refused before `n as f32` can round it into the
domain. No finding is blocking. Five are about what the spec *says*. In four of them, a
statement of L2, L5 or the domain is broader than what the code does, or contradicts the
spec's own table. In the fifth (§5.1), the claims about which rules run at expansion are
wrong, and so is the stated reason E8 cannot be reached. Each should be fixed before
those sentences go into public docs, the plan's Done line and the ontology.

## How it was checked

`/home/user/scratch/a6-laws-probe/tests/laws.rs`, built against the repo's
`target.noindex` with the workspace `RUSTFLAGS`, 12 tests, all green under
`PIXELFLOW_ISA=avx2` and `=avx512`. The probe mirrors `kernel!`'s pipeline as the spec
prescribes it, step by step, through the IR's public calls:

- **Lowering** declares the count uniforms first, with the UNBOUND NaN default. It resolves
  each bound to a term *before* `open_fold`, then calls `open_fold(depth)` and
  `close_fold(open, body, FoldTerms { lo, hi })`. A triangle's inner bound is the outer
  `open.index()`.
- **The macro tier** is `Saturate::macro_tier()`, then `relink`.
- **Emission** follows `arena_to_tokens`. Uniforms are re-declared with the call's
  `n as f32`, every reached node is re-pushed in arena order, and each fold goes through
  `FoldHeader::from_bits(..).attach(..)` and `push_reduce`.
- **Two paths** are run for every case: the `kernel!` path (macro tier, then emission) and
  the `kernel_raw!` path (emission only). Both are baked through `jit_cache`, so the
  runtime tier and `legalize`'s guard apply.

Results:

| Case | Result, both paths, both tiers |
|---|---|
| `KnownRange::admits(&(0..n))` | true at 0 and 2^24; false at 2^24+1, `u64::MAX`, `usize::MAX as u64` |
| `n as f32` | exact for 0, 1, 7, 2^24−1, 2^24; `16777217usize as f32 == 16777216.0` |
| `0..n`, n ∈ {0,1,3,7} | rustc's value at the four `FOLD_SAMPLES` |
| `0..n`, n = 2^24, body `1` | `16777216.0`: exactly 2^24 trips |
| raw write: 16777218, NaN, −1, 2.5, ±∞, 1e30, `u64::MAX as f32`, `usize::MAX as f32` | NaN, answered within the worker's deadline (no hang) |
| raw write −0.0 | 0 (the count 0); raw `16777217.0` is 2^24 and counts 2^24 trips |
| `n..8`, n ∈ {0,3,8,10}; `lo..hi` ∈ {(0,0),(2,5),(5,2),(3,3),(0,7),(1,0),(2^24,0)} | rustc's value; a reversed pair is the identity |
| triangle `Σ_{i<4} Σ_{j<i}` (i = 0 included), and `Σ_{i<n} Σ_{j<i}`, n ∈ {0,1,7} | rustc's value; 26 at X = 3 |
| canonical key of the lowered `Σ_{i<n}(X·i+Y)` vs `over_values` over a `Uniform`; lowered triangle vs `over(0..4, over_values(.., i, ..))` | equal (the spec's "riskiest part", §9, holds) |
| bound classes the body widens with `(n as f32)+X−X` and `(i as f32)+X−X` | lawful, rustc's value |
| `triangle_to(3)` vs `triangle_to(7)` | one key, one compiled region (L9) |

L3, item by item:

- **`X`/`Y`** are `f32`, and nothing converts to `usize` (`parser.rs:1188-1198`).
- **A `let`** is never a `usize`. `let n: usize` is refused by `of_a_value`, `let m = n` by
  `a_count_is_not_a_value`, and `let n = X` is an `f32` local, which M2 refuses.
- **A const can't be shadowed**, so sema's symbol lookup and `RangeScope`'s const-first
  lookup cannot disagree. No parameter, `let` or index may shadow a const
  (`sema.rs:735-772`).
- **The fold's own index** is out of scope while its bounds are typed.
- **An enclosing index** varies only with its binder, whatever its fold's body reads. Its
  guard's masks are uniform across lanes (`passes/guard.rs:24-27`), and codegen's mask
  predicate is false for a binder `Var` (`program/mod.rs:433-444`).

## Findings, ranked

### 1. The L2 and L5 statements leave out the two optimizer edges the IR documents, and A6 makes one of them reachable at expansion (non-blocking)

The statements are spec.md:69-78 (§1.1 L2, L5), :100 and :103 (§1.3), :1798 (§7.1's Done
line) and :1843 (§7.4's Range row). They say a value bound naming no count makes the fold
NaN, and a backwards pair is the identity, with no exception. `Kernel::over_values`' doc
carries two exceptions (`pixelflow-ir/src/kernel.rs:790-797`), as does BACKLOG E9
(`docs/BACKLOG.md:153`):

- **E9.** The `Annihilator` (`0·y → 0`) is in `all_rules`, which is what the macro tier
  runs (`pixelflow-search/src/egraph/rules.rs:192-194`). So once `kernel!` saturates
  value-bounded entries, a `0.0 * (0..n).map(..).sum()` loses its fold at expansion.
  - Probe: the macro tier returns `Const(0)` with no fold left. A raw NaN count gives `0` on
    the `kernel!` path, and on the `kernel_raw!` path at bake.
  - E9's row names only `FactorFold` plus `jit_cache::compile`. The user-written zero
    factor at expansion needs neither.
- **Decision 5.** `FactorFold` at a runtime-empty range gives `c ⊗ identity`.
  - Probe: `(n..8).map(|i| X * (i as f32)).sum()` at `n = 10` (an in-domain pair running
    backwards) is NaN at `X = +∞`, where rustc's empty sum is `-0.0`.
  - Probe: `(0..0)` at `X = -2.5` is `-0.0`. §1.3's "+0.0 for a sum" is wrong as stated:
    the sign is X's.

**Fix:**
- §1.1 L2 and L5, and §1.3's last column: state the NaN and the identity "up to the two
  optimizer edges `Kernel::over_values` documents (E9's annihilation, decision 5's
  factoring)".
- §1.3 row 0: "±0.0".
- Widen E9's row: a zero factor written in `kernel!` takes the fold at expansion too.
- The Done line may keep its sentence, since the test it records is `X·i` at `X = 3`.

### 2. "Past 2^24 is another count" is true only at 2^24 + 1, yet it is the reason the generated public docs give (non-blocking)

The claim appears in five places:

- spec.md:800, the `# Panics` text emitted on every count-taking host `fn`: "past 2^24,
  which its `f32` uniform would hold as another count".
- :907-911, `WriteRefused`'s public doc: "Each is a refusal of plausible pixels … a count
  past 2²⁴, written as the `f32` its uniform holds, is another count".
- :986, `write_into`'s `# Errors`.
- :760, `argument`'s doc.
- :107 and J10 (:2056).

Every `n ≥ 2^24 + 2` converts to an `f32` above 2^24, which names no count, and the guard
makes the fold NaN. These pixels are not plausible ones. The spec's own §1.3 row at :103
says exactly this, and the probe shows NaN for 16777218, `usize::MAX as f32` and 1e30.
Only `2^24 + 1` rounds into the domain. The refusal is right (J10: a count's domain is its
type's); the reason given is wrong.

**Fix:** "A count is an integer in `0..=2^24` (§1.4). Past it, `2^24 + 1` would be written
as `2^24` and run one trip short with no NaN, and every larger count would name none."
Drop "refusal of plausible pixels" from `WriteRefused`'s `Count` variant.

### 3. §5.1 misstates which rules run at expansion, and why E8 cannot be reached (non-blocking)

- **Bullet 2** (:1091-1094) says `KnownBound` may make a triangular bound known, and that
  `HalveFold` makes an inner `0..(i + s)` that "the emitted code rebuilds … at load time".
  But the macro tier's set holds no fold rule: `RuleSet::production` is `all_rules()`, and
  only `RuleSet::runtime()` adds `fold_rules()` (`rules.rs:192-211`; `fold_rules.rs:530-534`,
  "Runtime tier only"). So at expansion no fold rule touches a value-bounded fold, and
  these rewrites happen at bake only.
- **The loud-hole bullet** (:1099-1100) says "No rule today makes a member reading `X` in a
  `Uniform`'s class or a binder `Var`'s". The probe shows that the macro tier does exactly
  that: for `Σ_{i<n} X·i + ((n as f32) + X − X)` it extracts
  `add(reduce(..), Uniform(0), Uniform(0))`, so the uniform's class gained `u + X − X`.
- E8 is still unreachable, for a different reason. A leaf is opaque, so every member of its
  class contains it, and the leaf is the cheapest member. Extraction therefore materializes
  the leaf, and the probe's widened bound classes come out lawful.

**Fix:** say both. The instruction "if a test hits one, report it" is then about a member
*cheaper* than the leaf, which cannot exist.

### 4. The ontology's Range row would claim every host write refuses past 2^24 (non-blocking)

spec.md:1841-1843 rewrites the row to: "Past 2²⁴: a literal is refused at construction, a
count the host writes is refused at the write, and a value makes the fold NaN". Only
`kernel!`'s host function and `Args::write_into` refuse. The other host writes all take an
`f32` and refuse nothing: `Uniform::new(v)` (the `over_values` spelling),
`UniformBlock::set_declared` (`manifold.rs:241-268`), and a handle write. A builder user
writing `16777217usize as f32` gets 2^24 silently, which is the case plan §1.4 says can be
refused "only there", at a typed write.

**Fix:** "a count `kernel!`'s host function or `Args` record writes is refused there".

## Not defects (checked)

- **Shadowing.** `refuse_shadowing_an_item` refuses a parameter, `let` or index named like
  a const, so the const-first `RangeScope` and sema's innermost-binding lookup agree on
  every name.
- **Decision 8 order.** Sema refuses before `range_bounds` returns, and lowering refuses
  after the known-pair arm. Both precede `push_reduce`'s `BoundSort`.
- **A literal end past 2^24 beside a value.** Lowering refuses it (`literal_bound`) before
  `FoldHeader::new` would panic. This is the same stage placement as `literal_range` today
  (the parser's doc lists lowering for it).
- **`(n..n)`.** Both ends are one term. The guard tests it once, and `Lt(n, n)` gives the
  identity. `EmptyFold` never fires on class equality (L6). For a raw NaN `n` the fold is
  NaN.
- **Uniform identity.** A count uniform is keyed by identity plus default bits
  (`arena.rs:338-362`). Two counts with the same UNBOUND default never hash-cons into one
  class, so `lo..hi` never collapses to `u..u`.
- **The emitted host check.** `(0..#count as u64)` parses as `0..(n as u64)`. It precedes
  `#body`, and so it precedes every `n as f32`, in both `entry_arena` and `Staged` bodies.
- **The `write_into` check.** It precedes the `set_declared` array under both arms.
- **`mismatch.fmt(f)`** in the arm (a) `Display` compiles as written (checked with
  `rustc --edition 2024`).
