# A4 spec critique: run-time semantics and soundness

Read-only review of `.wip/a4/spec.md` against the code at `a747770` (`git diff --quiet a747770 HEAD -- . ':!.wip'` holds), the plan (`docs/language/2026-10-20-a-bound-is-a-value.md`), A3's spec and rulings, and CLAUDE.md. The lens is what the emitted code computes for every class of bound value, on every tier, and whether the guard, the loop, the key marker and the extraction recommendation are sound.

## Verdict

The core design is sound. For every bound value class the guard plus the do-while computes the fold's value or NaN on AVX2, AVX-512 and NEON. It cannot hang, it never converts a trip count to an integer, and it never reads past a buffer. The key marker round-trips and collides with no literal. Option (b) never materializes an unlawful arena from the production DP.

I found no defect that yields a wrong value or a hang with the spec's own guard pass. There are five findings. One makes a listed test fail as written: the MIN/MAX oracle at n = 0. The other four are a safety assert weaker than the spec says it is, a completeness regression that (b)'s residual column leaves out, a missing test for the one new allocator path, and a public doc that overstates L2 once the optimizer has run.

## What I checked and found right

- **The domain test, per class** (`Le(0,v) & Le(v,2^24) & Eq(Floor(v),v)`):
  - In-domain integers, including 0, -0.0, 1 and 2^24, pass.
  - 2^24 + 2, 1e30 and +inf fail on the second conjunct.
  - -1 and -inf fail on the first.
  - 2.5 fails on the third.
  - NaN fails all three: x86 `Le` is `Pred::Le = 2` (LE_OS, ordered) and `Eq` is `Pred::Eq = 0` (EQ_OQ) (`x86_64.rs:741-751`, `:1125-1133`); aarch64 uses swapped `FCMGE`/`FCMGT` and `FCMEQ`, all ordered (`aarch64.rs:884-905`).

  The test is robust beyond what it needs: even if an operand swap turned `Le(0, v)` into x86's unordered `Ge` (imm 5), the `Eq` conjunct alone still rejects NaN.
- **Floor.** It is a direct instruction on every tier (`x86_64.rs:699` `vroundps`/`vrndscaleps` imm 1; `aarch64.rs:930` `FRINTM`). No legalize pass after the guard would need to expand it.
- **The latch.** `done = Ge(step, end)` (`select.rs:314`) compares two in-domain integers only. The f32 counter is exact up to 2^24, so `Ge` fires at `step = hi`. The exit is "any lane done" (`select.rs:326-334`). Nothing converts a trip count to an integer: `Trips` is a static weight (`local.rs:386-433`, `traffic.rs:154-163`), so the `TruncToInt` row (`cvttps2dq` giving `i32::MIN`, `FCVTZS` saturating) is never reached.
- **Uniform masks.** P1 checks, after collapse, that a bound shares no bit with the row, column and lane binders. `collapse` picks lattice binder slots that no reachable fold or bound `Var` uses (`passes/lattice.rs:90-105`, `:157-185`), so P1 has no false positives. A uniform is a broadcast (`aarch64.rs:1141-1145`, `LdrS` + `DupLane0`). A fold counter is seeded and stepped by broadcasts. So every guard mask is uniform, and "a mixed batch runs both arms" never applies to the guard.
- **Placement.** `If`'s variance is the union of its children (`variance.rs:380`). A value-bounded fold's variance is L4's (`variance.rs:395-400`). So `R`, `ran` and `guarded` always share a variance, are never split across scopes by `extract_folds_bound_by`'s `remaining` filter, and are never placeholders (only folds are). `dom` and `run` vary with a subset of that, so they land in the same scope or an enclosing one. The bounds are always read by `dom` in the scope region, so they are never arm-owned.
- **Ownership and Layout.** An arm region that owns a def gets a block, with no price (`layout.rs:4-8`). R's only reader is `ran`'s true arm, and `ran`'s only reader is `guarded`'s true arm, so the loop sits under both tests. The precedent `a_fold_owned_by_an_arm_is_guarded` (`emit/mod.rs:1763-1802`) exists.
- **Liveness of `hi`.** `Liveness::read` (`local.rs:296-310`) extends any value read inside `[head, latch]` and defined before `head` to the latch, for every enclosing loop. "The `Reduce` def's interval spans the loop" therefore holds with no carve change. Selection resolves `hi` after the fold's frame pops (`select.rs:304-313`), so a leaf the body recomputes in its own frame cannot shadow it.
- **Hash-consing inside the guard pass.** I worked through a user who writes `If(Lt(0,n), Σ_{k<n} f, 0)` themselves. The guard's `ran` interns to the user's old `If` id, and the user's `If` is rebuilt over `guarded`. `R` still has one reader. The pass maps distinct old nodes to distinct new ones, so it cannot give `R` a second reader.
- **The key marker.** `VALUE_FIELD = u32::MAX` is not reachable by `field(n)` for `n ≤ 2^24`. `with_a_value` stores `(Held::Value, 0)` and `from_bits` rebuilds exactly that through `store`, so `from_bits(to_bits(h)) == Some(h)`. `admits_a_value` refuses the marker under ANY/ALL/SEQ, at a stride other than 1, or beside a literal past the bound. The hand-assembled `0x…0001_0200_0000_0000_ffff_ffff` decodes to SUM, slot 0, lo = 0, hi = Value.
- **No read past a buffer.** A fold body reaches memory only through `lower_gather`, whose `Min(Max(Floor(i),0),w-1)` clamp (`passes.rs:466-490`) stays. The loop only ever visits in-domain integer indices.
- **The optimizer and L5.** The e-graph runs before `legalize` (`jit_cache.rs:145-154`, then `pipeline.rs:122-144`), so no rule ever sees the guard. Of the fold rules, only `FactorFold` touches a value-bounded fold, and `c ⊗ NaN` is NaN, so L5 survives optimization (L2 does not exactly; see F5).
- **Option (b)'s soundness.** `Priced::var` is `node_variance` over the children's *chosen* forms (`extract.rs:1276-1293`), which is the term materialization builds. `ENode::variance` is never narrower than the arena's transfer, apart from A3 S2, where it is wider. So a settle that admits a candidate implies `push_reduce` admits the term.

## Findings

### F1. P2 does not check what the spec calls it: a single reader (non-blocking)

The spec calls P2 "the single reader" (spec.md:215-221). §2 P-b says that "Without the single reader, R would sit in the scope's own region, outside the arm, and run unguarded: P2 (§1.5) is the assert" (spec.md:305-307). The assert as specified checks only that every reader of `R` is `Ternary(If, m, t, f)` with `t == R`. It does not count readers.

Two `If`s both reading `R` as their true arm pass P2. Ownership then places `R` at the lowest common ancestor of the two arm regions (`ownership.rs:178-183`), which is the scope region. That leaves the loop unguarded. An unguarded do-while hangs on several tiers:

- With `hi = +inf` or `hi = 2^24 + 2`, the counter stalls at 2^24 (`2^24 + 1.0` rounds back) and `Ge(2^24, hi)` is false on every tier.
- With `hi = NaN`, aarch64's ordered `FCMGE` never fires.

Today's guard pass cannot produce a second reader, which is why this is not blocking. But the assert is the stated defence against a future pass that does, and as written it does not catch the failure it is named for.

Fix: P2 also counts reachable readers per value-bounded fold and panics on a second one, whatever its shape, naming both. That needs one more `Vec<u8>` (or `bool`) beside `value_folds`.

### F2. Option (b) can turn today's lawful extraction into a panic, and the residual column omits it (non-blocking)

`relax` refuses every candidate of a class that some candidate reads as a bound (spec.md:755-763). It does so whether or not a fold that reads it as a bound is ever chosen.

Suppose a bound class has no member that keeps L3, which is the meet-over-members case A3's F2 describes. Under (b) it never settles. Its other parents, for example a body that reads the same class, are relaxed only after it settles (`extract.rs:1999-2013`), so they never are either. The root can then be left unsettled, and (b)'s new panic fires.

Today that class settles on its cheapest member. If the fold reading it as a bound is not chosen (its class has a cheaper sibling, such as an unrolled form), the extraction succeeds lawfully.

The spec's residual column (spec.md:717) names only the two-levels-down case. The new panic message also blames the bound class even when the root is unsettled for another reason. The message itself is still loud and correct about what (b) refused, so this is completeness, not soundness.

Fix: state this residual in §6's table and in the egraph intent line. If review takes (b), add a pin in which the class is also read outside the bound and the fold has a cheaper sibling, showing the panic.

### F3. `each_monoid_runs_under_its_guard`'s oracle is wrong at n = 0 for MAX and MIN (blocking as written, a one-line fix)

The test asserts `MAX_{k<n}(X − k) = X` and `MIN_{k<n}(X − k) = X − (n − 1)` "at n ∈ {0, 1, 7}" (spec.md:935-937). At n = 0 the fold is the identity, −∞ for MAX and +∞ for MIN. That is the spec's own §1.1 (spec.md:81, :88), and it is also what the factored form gives (`X + max_∅` is `X + (−∞)`).

A test written to the bullet fails on every tier. Fix: at n = 0 the oracle is `fold.monoid().identity()`. Only `Π_{k<n}(k+1) = n!` holds as stated at n = 0.

### F4. No test reaches the one allocator path A4 makes new: a loop parameter passed as another loop's entry argument (non-blocking)

Before A4, every head entry argument is a fresh constant (`select.rs:275-281`). After A4, `lo`'s lane is the seed (spec.md:506), and when `lo` is an enclosing fold's index that lane is the outer loop's head parameter. It is live after the jump (the outer latch reads it), so `LinearScan::pass` takes the `home` → `copy_of` branch for a parameter argument (`local.rs:1105-1147`). No selected function took that branch before.

§8.2 tests `lo` a uniform (`a_value_lo_starts_the_fold`) and `hi` an enclosing index (the triangular tests), never `lo` an enclosing index. `HalveFold` only rewrites `hi` (`i + s`), so `run_linked` does not reach it either.

Fix: add `Σ_{i<8} Σ_{j∈[i,8)} (X + j)` through `run_raw` and `run_linked`, on both x86 tiers and on NEON in CI.

### F5. The public `over_values` doc and §1.1's L2 row promise the identity, which an optimized kernel does not always give (non-blocking)

§5 adds to `Kernel::over_values`' rustdoc: "a pair that runs backwards at run time is the identity (L2)" (spec.md:664-667). Through `jit_cache::compile`, `FactorFold` factors an invariant operand out of a value-bounded fold (decision 5, `fold_rules.rs:496-506`). At a runtime-empty range the kernel then computes `c ⊗ identity`:

- `-0.0` for a negative SUM factor;
- NaN for an infinite one;
- NaN for MIN with `c = −∞`, and for MAX with `c = +∞`.

That is decision 5's accepted edge, but the public doc states the identity unconditionally. L5's NaN does survive, since `c ⊗ NaN` is NaN.

Fix: the doc says "the identity, up to the factoring edge `FactorFold` accepts at a runtime-empty range (decision 5)".

## Not findings (checked, and the spec handles them)

- **A subnormal bound reads as 0 under DAZ/FZ and as NaN without.** That depends on the thread's mode, not the tier, it is safe either way, and §8.2 already excludes subnormals for this reason.
- **`-0.0` as `lo` seeds the counter with `-0.0`.** J9 documents it.
- **A computed mask-valued bound** (A3 S1, hand-built only) would reach codegen. At run time its guard makes the fold NaN. On AVX-512 the head parameter's class would be the opmask's; that is a loud compile failure at worst.
- **`Trips::nested_in` saturates, but `plan_carries`' `reloads` sum and `dynamic_memory_ops`' product do not.** That is only reachable with literal trip products past 2^64, which is pre-existing: today the multiply at `select.rs:287` overflows first.
