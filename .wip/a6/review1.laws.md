# A6 review, lens: the laws, by experiment

**Against:** the working tree (`git diff 29980bf -- . ':!.wip'`, 20 files), `.wip/a6/spec.md`,
`.wip/a6/rulings.md` (rulings win; D-A6-1 is arm (b)), the plan
`docs/language/2026-10-20-a-bound-is-a-value.md`, `CLAUDE.md`, `pixelflow-compiler/intent.md`.
Read-only on the repo. Nothing tracked was edited, and no gate script was run.

**Verdict.** I found no blocking defect. Every law I was asked to check holds through the real `kernel!` and
`kernel_raw!` now in the tree, under `PIXELFLOW_ISA=avx2` and `=avx512`. That covers rustc's values at every
count, the identity on a reversed pair, NaN with no hang for a raw write that names no count, one key and one code
pointer for every `n`, and the domain panic in both host writes. One finding is about a stale marker, and one
predates A6. Neither is blocking. NEON was not run here.

## How it was checked

There are two scratch crates outside the repo, built against the repo's `target.noindex` with the workspace
`RUSTFLAGS="-C llvm-args=-fp-contract=fast"`:

- `/home/user/scratch/a6-review-laws`, the runtime laws.
  - `tests/entries.rs` is one block of 44 entries, included twice. One module has
    `use pixelflow_compiler::kernel as the_macro` and the other `kernel_raw`, so every entry runs saturated at
    expansion and as written.
  - `tests/laws.rs` holds 15 tests against scalar Rust over the same tokens.
  - `tests/helpers.rs`, `egraph_shapes.rs`, `closure_form.rs`, `deep.rs`, `deep2.rs` and `deep3.rs` are the
    targeted extras.
- `/home/user/scratch/a6-review-laws-cf/src/lib.rs`, 27 refusal and acceptance cases. One `cargo check`
  reports each case's error at its own line.

Command, per ISA:

```sh
cd /home/user/scratch/a6-review-laws
CARGO_TARGET_DIR=/home/user/pixelflow/target.noindex RUSTFLAGS="-C llvm-args=-fp-contract=fast" \
  cargo test --test laws --no-run
PIXELFLOW_ISA=avx2   target.noindex/debug/deps/laws-2fa56a3c5aab08d4   # 15 passed
PIXELFLOW_ISA=avx512 target.noindex/debug/deps/laws-2fa56a3c5aab08d4   # 15 passed
```

`helpers` (1), `egraph_shapes` (1), `closure_form` (1) and `deep2` (2) also pass on both tiers.

## What holds, law by law

**Denotation, against rustc.**
- **Counts.** `n ∈ {0, 1, 7, 1000, 2²⁴}` at `hi` (`0..n`), at `lo` (`n..1000`, and `5..n` with a literal `lo`)
  and at both ends (`n..n`). Every value equals rustc's, at 4 samples on each tier. At `n = 2²⁴`,
  `Σ_{i<n} X` at `X = 1` is exactly `16777216`, and `max_{i<n} i·X` is exactly `16777215`: the trip count and
  the last index are both exact.
- **Monoids.** MIN, MAX and PRODUCT under a count (`min_hi`, `max_hi`, `prod_hi`, `fact`, `prod_lo`) match
  rustc, including the empty fold's `±∞` and `1`.
- **Triangles.**
  - With literal outers: `Σ_{i<4} Σ_{j<i}` with `i = 0` included, `Σ_{i<4} Σ_{i≤j<4}`,
    `Σ_{i<1000} Σ_{j<i}`, and an odd const outer of `37` that is also a three-deep index triangle.
  - Under a count: `tri_n`, `tri_trips`, `tri_rev` (an index at `lo` and a count at `hi`), `tri_max`,
    `tri_prod`, and `tri2idx` (two enclosing indices as one fold's bounds).
  - Three deep: `deep3`, `Σ_{i<n} Σ_{j<i} Σ_{k<j}`.
  - Every one matches rustc for `n ∈ {0, 1, 7, 1000}` (`{0, 1, 7, 100}` for the three-deep ones).
  - By rebinding, `tri_n` also matches for every `n` in `0..=40`.
- **A count read as a bound and as a value in one entry.** `both_ways`, `diff_noy` and `d` match exactly at
  every count, `2²⁴` included where tested.
  - `diff_ways(2²⁴)` differs from rustc: 0 against 0.5 at `(1, 0.5)`. This is a reassociation, not a trip
    count, and is pinned by `diff_ways_at_the_top_is_a_reassociation`. The kernel's value equals
    `(n·X + Y) − S` bit for bit, where S is rustc's own sequential sum. `diff_noy` (no `+ Y`) is 0, as rustc's
    is. This is the reassociation CLAUDE.md permits, and A6 did not cause it.
- **Nesting and scope.**
  - A count bounding an inner fold inside the outer it bounds (`nested`) matches rustc.
  - So does a fold index that shadows the count bounding it (`(0..n).map(|n| (0..n)…)`).
  - So does an inner index that shadows the outer while the inner bound reads the outer (`own_shadow`, which
    is 4, as in Rust).
  - So do three counts at once (`three(a, b, c)`, reversed `(5, 2, 3)` included) and a record beside a count.
  - So do helpers whose own index-bounded folds are inlined inside a caller's count fold (`helpers.rs`).
  - So does a count fold in each arm of an `if` (`cond`), and `any` over a literal range inside a count fold
    (`any_in`).
- **Shapes that let the e-graph mix a count's value class with its bound class.** `(n as f32 + X) − X`,
  `(i − j) + X − X`, and two counts nested both ways (`egraph_shapes.rs`) all match rustc on both macros. No
  saturation at expansion panicked.
- **Staged.** `cols(&k, n)`, `stri(&k, n)` (a staged triangle) and `spair(&k, lo, hi)` over every pair below
  match rustc.
- **Composition keeps the count.**
  - `cols(&trips_hi(m), n)` has two counts from two entries, declared `[n, m]`. It holds one key across
    `(m, n)` and matches rustc when rebound by position.
  - The argument's count written raw as `2.5` gives NaN.
  - `stri(&tri_n(m), n)` matches rustc.
  - Two warps match: `max_hi(n).at(2X, Y)` and `xi(n).at(X+1, Y)`.
  - `Uniform + trips_hi(n)` matches.
- **The closure form.** `kernel!(|| …)` and `kernel!(|s: f32| …)` triangles match rustc, and a position rebind
  works.

**L2 (empty).** Reversed per-call pairs give each monoid's identity on both macros, rebound through `Args` too.
The pairs tried were `(1000, 7)`, `(1, 0)`, `(2²⁴, 0)` and `(2²⁴, 2²⁴−1)`, plus every pair in
`0..=20 × 0..=20` and 25 pairs near `2²⁴`, rebound by position (466 pairs × 4 monoid entries × 2 macros). A known
reversed pair is refused at expansion. The zero's sign may differ from rustc's, which the docs say ("may differ
from rustc's in sign").

**L3 (legality)**, refused through the real macro (`a6-review-laws-cf`):
- `0..X`, `Y..4` and `0..i + X`: "mismatched types: expected `usize`, found `f32`".
- `let m = X; 0..m`: M2.
- `0..X as usize`: the parser's `as` refusal.
- The fold's own index in its own bound: "cannot find `i`".
- A caller's index inside a helper: "cannot find `i`".
- `0..k(X, Y)` and `0..{n}`: M5.

**The leaf rule.** `0..n + 1`, `0..n * m` and `kernel_raw!`'s `0..n - 1` all give "arithmetic on `n` in a
range's bound". `0..-n` is the unary refusal.

**Decision 8.** `(0..n).any`, `(0..i).all` and `kernel_raw!`'s `(n..4).all` are refused with M4. `(0..4).any`
inside a count fold is accepted and matches rustc.

**Domain at expansion.**
- `(N..n)` with `N = 16777217`, and `(n..16777217)`: "the bound `16777217` reaches past 16777216".
- `(0..16777217)`: the literal-range refusal.
- `(16777216..i)` is accepted.

**Counts elsewhere.**
- `kernel!(|n: usize| X)` is refused with M7, and a helper's `n: usize` with M6.
- `let m = n` and `… + n` are refused as "`n` is a `usize`, where a value is expected".

**L5 (domain, no hang).** A raw `set_declared` makes the fold NaN on both macros and both tiers. The values
written were `16777218`, NaN, `−1`, `2.5`, `±∞`, `1e30`, `±0.5`, `8388607.5` and `usize::MAX as f32`.
- **Every entry that has a count was tried:** sum, MIN, MAX, PRODUCT, lo-count, the literal-lo mix, the
  triangles, `nested`, `deep3`, the shadowing entry, `n..n`, and a pair with the bad value at either end or at
  both. No case took more than about 50 µs.
- **Writes inside the domain.** `−0.0` runs 0 trips. `16777217.0` runs exactly `2²⁴` trips, one short with no
  NaN, which is why only the host can refuse `2²⁴ + 1`.
- **E9.** `kernel!`'s expansion of `Σ … * 0.0 + Y` is `Var(1)`: the fold is gone at expansion, the count's slot
  is kept (one uniform, and its `Args` writes), and a raw NaN gives `Y`. BACKLOG E9's widened row says exactly
  this.

**L9 (one program).** Across every `n` in each entry's set, including `2²⁴`, both macros give one
`canonical(..).key` and one `jit_cache::compile(..).kernel.as_bytes().as_ptr()` at `[11, 7]`. The sets cover the
22 single-count entries, the four pair entries over 12 pairs, staged `cols` and `stri`, and the composed
`cols ∘ trips_hi`.

**Host refusal.** For every single-count entry on both macros, `2²⁴ + 1`, `2²⁴ + 2`, `2⁴⁰` and `usize::MAX`
panic with `` `{entry}`'s count `{param}` is {value}, past 16777216 (2^24) ``. The message also contains
"a count is an integer in `0..=2^24`" and "§1.4 of docs/language/2026-10-20-a-bound-is-a-value.md". The rest
of the host refusal:
- `2²⁴` itself is accepted.
- Each end of a pair, the third count of `three`, an unread count, and a staged entry's count are all refused.
- A raw identifier `r#type` is named `type`.

**`Args::write_into`.**
- It panics with the same message for `2²⁴ + 1`, `2²⁴ + 2` and `usize::MAX`, and the block keeps its previous
  values.
- Two-count `TripsPairArgs { lo: 3, hi: 2²⁴+1 }` writes neither count, and `RecArgs` with a record before the
  count writes nothing.
- `n = 2²⁴` writes `[16777216.0]`.
- A program compiled at `n = 0` and rebound through `Args` matches rustc at every `n ∈ {0, 1, 7, 1000, 2²⁴}`, and
  at every pair for `min_pair`.

## Findings

1. **Non-blocking. Plan Done line marker:** `docs/language/2026-10-20-a-bound-is-a-value.md:729` still ends
   "GATES: pending." The gates have run (`.wip/gates/out/*.a6-impl.*`, `.wip/a6/gates-impl.md`). This is the
   coordinator's tracked handoff (`gates-impl.md:287`, `impl-docs.md:91`). As written, the sentence is not true
   of the tree, so it must be replaced with the measured clause before submit.
2. **Non-blocking, predates A6. Fold depth past 57 panics at bake.** Lowering admits a nest up to
   `Binder::COUNT` (60) deep (`pixelflow-compiler/src/lower.rs:657`). `passes::lattice::collapse` needs three
   free binders (`pixelflow-ir/src/passes/lattice.rs:95`), so a kernel nesting 58 to 60 folds that survive
   optimization expands and then panics at bake: "collapse: no free binder slot for the col fold".
   - Through `kernel!`/`kernel_raw!` with value bounds, `deep.rs`'s 59- and 60-deep nests panic this way, and
     57 passes (`deep2.rs`).
   - With literal bounds only (`deep3.rs`, 59 × `(0..100)`), `Manifold::compile` panics the same way, so A6
     did not introduce it.
   - It is a stage accepting what a later one refuses, and no BACKLOG row records it.
   - Out of A6's scope; listed for a BACKLOG row or a lowering refusal at `Binder::COUNT − 3`.

## Seen, not findings

- **Derivatives.** `Kernel::dx()` of any fold is refused at bake ("lower_dwrt: no derivative rule for a bounded
  fold", `pixelflow-ir/src/passes.rs:964`). This predates A6 and is unrelated to value bounds.
- **Error spans.** Lowering's refusals (a literal past `2²⁴`, alone or in a pair) are spanned at the `kernel!`
  call rather than at the bound, as `literal_range`'s were before A6.
