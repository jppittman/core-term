# A4 spec critique: codegen mechanics (program/ and emit/)

Read-only review of `.wip/a4/spec.md` against the code at `a747770` (`git diff --quiet a747770 HEAD -- . ':!.wip'` holds), the plan, A3's spec and rulings, and CLAUDE.md. The lens is codegen mechanics:

- `pixelflow-codegen/src/program/`: lowering, the carve, ownership, layout.
- `pixelflow-codegen/src/emit/`: selection, the builder, the allocator, traffic, and the three backends.

## Verdict

The operand-and-carve change is sound for every bound the user surfaces can build. It needs no encoder change on any tier. The existing arm skip really does do the guard's work. I found five defects, none blocking:

- **E1.** AVX-512 diverges from AVX2/NEON on a mask-valued `lo`.
- **E2.** P2 checks the reader's shape but does not count readers. This is the same defect as critique.semantics F1, seen here from the scope side.
- **E3, E4.** Two rows of §3.4's per-ISA table misstate what the allocator does.
- **E5.** One §8.3 test names a constructor this crate cannot reach.

## How the trip count is held today, and after A4

**Today.** The counter is an `f32` vector lane, the head's first parameter (`select.rs:284-300`). It is seeded with a pool constant, `lo` (`:275`). It is stepped by a pool constant, the stride (`:311-312`), and compared against a third pool constant, `end` (`:313-314`), with `vcmpps`/`CmpK`/`FCMGE`. The backward branch is taken while no lane of `done` is set (`:327-334`).

No count is ever an integer, an immediate or a general-purpose register. Every constant is rematerializable (`regalloc/local.rs:9-13`), so `end` costs no register across the loop. The `u64` count exists only as the allocator's carry weight (`build.rs:321-330`, `local.rs:381-449`) and as `EmitTraffic.trips` (`traffic.rs:115-130`).

**After A4.** A value `lo` is passed as the head's argument. A value `hi` is read by the latch's compare. Neither changes an instruction form:

- `Extra::Cmp` and `CmpK` read two vector registers (`avx2.rs:48-53`, `avx512.rs:65-70`).
- NEON's `FCMGE` is the same `Alu` form (`aarch64.rs:884-905`).

There is no immediate whose limit could bite: no `imm8`, `imm12` or `rel32` field. The aarch64 skip is the fixed two-word `CbzFar` (a CBNZ over a `B`, ±128 MiB, `aarch64.rs:302-306`, `:664-731`, `:1520-1522`). It is unchanged and reaches any loop an arm holds.

## What I checked and found right

- **Liveness of `hi`.** The latch's compare is an instruction at a position in `[head, latch]`. `Liveness::read` (`local.rs:304-318`) extends any value defined before `head` to `latch` and marks it hoisted. `read_again` (`:1455-1464`) keeps it held through the latch.
  - It is then either a carry candidate, priced by `trips_at` (`:381-449`), or stored at its definition and reloaded by `flush` (`:1430-1442`) and `reload_into` (`:1237`).
  - `ScopeFold::outer` is read nowhere in `emit/` (`grep -rn '\.outer\b' emit` is empty). Leaving `hi` out of the fold's own `outer` therefore changes nothing the allocator sees. §1.6's choice is right.
- **The seed.** `pass` → `home` (`local.rs:1105-1147`) takes the argument's lease when the value is dead after the jump. Otherwise it copies (`copy_of`); for a parameter that is not carried, it stores to the slot.
  - A value `lo` is either a uniform, live per batch when R sits in the column fold, or an enclosing loop's parameter, live to that loop's latch. Both take the copy path. That is sound.
  - No backward-branch cycle is introduced: the latch still passes a fresh `step` and `accumulated`. The test gap here is semantics F4.
- **Frames.** `hi` is looked up after `Scope::Fold(j)`'s frame pops (`select.rs:303-313`). By L4, `var(hi) ⊆ var(R)`, so `hi` is defined in R's scope or an enclosing one, never in R's own fold.
- **The carve with `operands` yielding the bounds.** Nest, ownership and assert_closed see the bounds as reads of the `Reduce` def, as §2 says:
  - `structural_children` (`program/mod.rs:436-443`) chains `all_operands`. The closure walk (`scopes.rs:260-282`) therefore follows a nested R's bounds.
  - `assert_closed` (`scopes.rs:76-97`) and `ownership::reads` (`ownership.rs:116-119`) see them as R's reads.
  - `dom` always reads each value bound in the scope region, so the bounds are never arm-owned.
- **The guard nodes share R's scope.** `R`, `ran` and `guarded` have equal variance, so `extract_folds_bound_by`'s `remaining` filter (`scopes.rs:309-312`) never splits them across scopes.
- **P-b, a fold whose body reads no coordinate.** It was traced through `extract_folds_bound_by`:
  - The row fold's walk stops at `guarded` with `placeholder_here[guarded] = false` (`:276-278`).
  - R is unclaimed and carved at the body (`:212-301`).
  - R's loop is opened inside the arm blocks' `Item::Arm` walk (`select.rs:111-127`), with no parent loop. Its trips are `ONCE`-nested (`select.rs:285-289`).
  - The precedent `a_fold_owned_by_an_arm_is_guarded` (`emit/mod.rs:1769-1802`) exercises exactly this layout at `POINT`.
- **The builder's invariants hold for a loop inside an arm.** These are: an exit has one predecessor (`local.rs:1015-1022`), and a forward branch stays in one scope and does not cross a loop extent (`build.rs:417-437`). The arm's skip branch and its `past` block are both in the parent scope, and the loop lies entirely between them.
- **Byte identity.** `bound(Lit(n))` is `constant(n as f32)`, at today's points and in today's order. `KnownRange::of` builds `range.lo`/`range.hi` from the node's literals (`fold.rs:1181-1183`), so `bounds.start`/`.end` equal them. `operands` of a literal `Reduce` is empty.

## Findings

### E1. A mask-valued `lo` panics in selection on AVX-512 and compiles on AVX2/NEON (non-blocking)

`ExprArena::push_reduce` admits a computed value bound that is a mask (`arena.rs:984-988`: "a computed bound that is a mask (only a hand-built call makes one…) is admitted … T6 closes the hole"). It reaches codegen through public `emit::compile`, and through `Kernel::from_parts` (`kernel.rs:331`) into `jit_cache::compile`.

`Selector::fold` derives the head's parameter classes from the seeds' names (`select.rs:281-284`). On AVX-512 a comparison's lane is `Lane::Opmask` (`avx512.rs:427-433`), and so is a `BitAnd`/`BitOr` of two opmasks (`:413-421`). The head's first parameter is therefore minted with class `Opmask`. `Avx512::param_lane` then does `Lane::Vector(param.typed())` (`avx512.rs:507-509`), and `typed` asserts the class (`emit/mod.rs:306-307`). The result is a panic saying "is not of the class asked for": a selection-bug message, not a refusal by name.

On AVX2 and NEON the same arena compiles, and the mask's bits are read as a float:

- A true mask is NaN bits, so `dom` is false and the fold is NaN.
- A false mask is `+0.0`, which is in D, so the fold runs from 0.

A mask-valued `hi` does not panic on AVX-512, because `Compare` widens its operands with `.vector(b)`. So the asymmetry is `lo` only.

This contradicts §1.2 ("every tier computes the same answer") and §3.4's "seed: `lo`'s register moved into the counter parameter". critique.semantics lists the case under "not findings" as "its guard makes the fold NaN". That holds for a true mask but not for a false one.

**Fix.** In P1's arm (`lower.rs:351-359`), refuse by name any value bound whose node can yield a mask lane: a `Compare`, or a `BitAnd`/`BitOr`. No lawful count is either, and the user surfaces already refuse both as `BoundArithmetic`. Pin the refusal with a lowering unit test. This keeps §0's "no `IsaBackend` method" and closes the divergence until T6 gives a bound a sort.

### E2. P2 permits two `If`-true-arm readers, and then the loop is not in an arm (non-blocking; same as critique.semantics F1)

§1.5 and §2 P-b call P2 "the single-reader property placement rests on". P2 checks only that every reader is `If(m, R, f)`.

If two such `If`s read the same R, ownership puts R at their lowest common ancestor (`ownership.rs:176-183`), which is the scope region. `Layout` then gives it no arm block (`layout.rs:47-58`), and selection opens the loop unguarded. An unguarded do-while hangs:

- With `hi ∈ {+∞, 1e30, 2²⁴+2}`, the counter stalls at 2²⁴, and `Ge(2²⁴, hi)` is false on every tier.
- With `hi` NaN on NEON, `FCMGE` is ordered and never fires.

The guard pass cannot make a second reader today. The assert exists for the day something can.

**Fix.** Count reachable readers per value-bounded fold beside `value_folds`, and assert exactly one.

### E3. §3.4's latch row says the compare reads `end` from "its 32-byte slot" / an "`ldr q` reload"; no compare reads a slot (non-blocking)

`Stage::Read` is "a register the instruction reads" (`emit/mod.rs:262-263`). Selection may not name a frame slot (`build.rs:221-223`). `Cmp` and `CmpK` take `S::Read<Vector>` for both operands.

A `hi` that is not carried is reloaded by the allocator into a register before the compare: `Origin::Reload`, every trip. `EmitTraffic::of` counts that reload in the fold scope's `loads` (`traffic.rs:123`), weighted by that scope's trips.

**Fix.** The row should say: "a register: carried, or reloaded from its slot each trip (one `loads` per trip in the fold's scope)". Any traffic pin a value-bounded kernel gets can then be written against what is emitted.

### E4. On AVX-512 the guard's opmasks are not "short-lived" in P-c; they are reloaded every batch (non-blocking)

§3.4 says "AVX-512: two or three opmask values, short-lived". In P-c, `dom` and `run` are `CONST`, so they are computed in `Scope::Body` (§2), while the two arm tests that read them run in the column fold once per batch. They are therefore live across the whole lattice.

The opmask file's carry budget is 0 (`local.rs:495-506`), and `plan_carries` skips a file with no budget (`:405`). At the row loop's head, `flush` stores each opmask at its definition and drops it (`:1430-1442`). Each batch's `kortestw` then reloads it with `kmovw k, [rsp+slot]` (`avx512.rs:136-140`, `:586-593`), giving two loads per batch per guarded fold.

`layout` also gives every narrow value that is read (opmask, and AVX2's `vmovmskps` integer) a narrow slot from the peak (`local.rs:321-352`). So §3.4's "Frame: at most one more vector slot" also understates the narrow region.

This is correct code, not a defect in what is emitted. But the table is where a reviewer would read the cost.

**Fix.** Correct the two cells.

### E5. `a_bound_on_a_lattice_binder_fails_loudly` asks for "a hand-built `Write`", and `push_write` is `pub(crate)` (non-blocking)

`ExprArena::push_write` is crate-private in pixelflow-ir (`arena.rs:1044`; §1.3 of the spec cites this itself). The test lives in `pixelflow-codegen/src/program/lower.rs`.

`collapse` can never build the shape either. It picks lattice slots that no reachable `Var` or fold uses (`passes/lattice.rs:90-105`, `:152-189`). So a bound on a lattice binder is unconstructible through the pipeline. That is also why P1's lattice half cannot fire from any public path.

The one public route is `<ExprArena as pixelflow_ir::Ir>::embed(Shape::Write { row, col, lane, value })` (`term_arena.rs:86-91`, `lib.rs:109`). An implementer who misses it may reach for the visibility change CLAUDE.md forbids.

**Fix.** Name `Ir::embed` in §8.3.

## Not findings

- **`lo` an enclosing loop's parameter.** Handled by `home` → `copy_of`, as traced above. The missing test is semantics F4.
- **The traffic layer.** `EmitTraffic`'s `Vec<Trips>`, `Trips::{ONCE, of, nested_in, weight, known}`, `Display` honouring width, and `dynamic_memory_ops` as `Option<u64>` through `Option`'s `Sum` are specified precisely enough to implement. Their consumers (`byte_probe.rs:306`, `constant_traffic.rs:44-56`, `collapse_bench/mod.rs:309-329`) are the only readers (`grep -rn 'dynamic_memory_ops\|\.trips\b'`).
- **P1's message.** P1 is implementable as written: `BoundEnd: Display` (`fold.rs:420-427`) and `Variance::intersection`/`is_const` exist.
- **The plan's qemu clause.** It applies to byte_probe. A4 leaves literal kernels' LaneOps identical on every tier, and the `Test on macos-latest` leg runs `value_bounds_jit.rs` natively on NEON (`rust.yaml:503-602`).
