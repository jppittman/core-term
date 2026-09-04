# Register allocation outside the register allocator

**Goal: no register allocation outside the register allocator.**

Today two systems assign registers. The allocator assigns DAG values over the
pool `RegisterFile` declares. Beside it, roughly a dozen `const`s in the ISA
files assign registers by hand — invisible to the allocator, unvalidated by
`RegisterFile::checked()`, and correct only by arguments written in comments.

This is not a list of bugs. It is a list of invariants a person maintains that
a type could maintain instead — CLAUDE.md's stated failure mode, in the one
place the codebase most claims to have solved.

## What the allocator actually gets

| backend | file | inputs | pool | reload | select | **pool share** |
|---|---|---|---|---|---|---|
| SSE2 | 16 xmm | 0–3 | **4–9** | 11,12 | 13 | 6/16 |
| AVX2 | 16 ymm | 0–3 | **4–7** | 11,12 | 13 | 4/16 |
| AVX-512 | 32 zmm | 0–3 | **4–9** | 11,12 | 13 | **6/32** |
| aarch64 | 32 v | 0–3 | **16–25** | 26,27 | 28 | 10/32 |

AVX-512 has 32 vector registers and allocates six of them. That is the cost of
the escape hatches stated as a number: the pool was shrunk to leave room for
registers taken by hand, so **the allocator is already paying for them — it
just does not know their names.**

And the bill arrives somewhere specific. A pool that small has no room to hold
loop-invariant values, so the collapse loop's LICM parks every hoisted value in
memory instead. The hand-picked scratch and the hand-managed hoist slots are the
same defect, and the cost lands on LICM.

## The escape hatches

### A. Instruction-level temps — the ISA needs a scratch to express one op

| const | backend | why it exists |
|---|---|---|
| `X86_SCRATCH = Reg(10)` | sse2 | SSE2 is 2-operand destructive; `emit_binary_safe` needs a temp when `dst` aliases the RHS |
| `X86_BUILTIN_SCRATCH = [10,13,14,15]` | sse2 | transcendental expansion holds several live temps |
| `UNARY_SCRATCH = Reg(15)` | avx2, avx512 | same, VEX/EVEX tier |
| `FMOV_FALLBACK_SCRATCH = [28,29,30,31]` | aarch64 | non-encodable constant → MOVZ/MOVK/DUP needs a vector temp |
| `IDX_INT`, `GATHER_DST` | avx512, aarch64 | gather's lane-extract/insert sequence |

### B. Unmodeled register classes

| const | class | note |
|---|---|---|
| `BASE_GPR=x9`, `IDX_GPR=x10`, `VAL_GPR=x11`, `CTX_GPR=x0` | general | aarch64 gather addressing |
| `RAX`, `RDI` | general | avx512 gather |
| `SCRATCH_K = k1` | mask | avx512 compare destination |
| `r9`/`r10`/`r11`, `x5`/`x6` | general | collapse-loop counters (`x86_64::scaffold`) |

`RegisterFile` has no vocabulary for either class. They are not allocated; they
are chosen.

### C. Reload targets — class A in disguise

`reload: [Reg; 2]` and `select_reload` are *declared* and *checked*, but fixed
rather than allocated. The usual justification is a bootstrap: materialising a
spilled value needs a register, and choosing that register by allocation would
need a register.

That bootstrap is an artifact, not a law. It exists only because spilling is an
**annotation on the input schedule** rather than **operations in an output
schedule**. Rewrite the program instead — insert spill/reload ops, as LLVM's
`InlineSpiller` and most modern linear-scan allocators do — and a reload simply
*defines a value*, allocated like any other, with a very short live range. Short
live ranges are what linear scan is best at.

Which makes the real statement: **a reload result and an instruction temp are
the same thing** — a value the input DAG did not contain. Classes A and C are
one missing concept, not two.

> **Correction, 2026-09-03 — half of class C is class D.** The paragraph above
> is right about `select_reload`, which is why that one came out (see the
> landed note under step 2). It is wrong about `reload`, and tracing every
> *read* rather than trusting this section is what shows it. `reload[1]` has
> four use-sites and `reload[0]` two:
>
> | site | where | class |
> |---|---|---|
> | `resolve_operands`' `tmp_op` | inside an instruction | **C** |
> | `resolve_operands`' spilled `dst` | inside an instruction | **C** |
> | the Select guard's mask reload (two call sites) | *between* instructions, in the guard scaffold | **D** |
> | the guarded Select's spilled `dst` | same | **D** |
> | parking a hoist root in its slot | after a def, prologue mode | **D** |
> | the root's return reload | after the whole schedule | **D** |
>
> Only the first two are "class A in disguise" — a need of one instruction,
> which an instruction can declare. The rest happen at points the schedule does
> not contain, so there is no instruction to hang a reservation on, and
> `RegisterFile` has to keep a register for them.
>
> The consequence is the ordering one, and it is sharp: **closing the rest of
> class C on its own frees zero registers.** `reload` stays in the file for the
> scaffold sites either way, so the two registers come back only when class D
> lands with it. That is a second reason to do step 4, independent of
> reclaiming LICM's trip to memory — and it is why the remaining work is one
> piece rather than two.
>
> None of those scaffold sites needs a *dedicated* register, which is what makes
> the combined step tractable: each happens where a free-register set is
> perfectly well defined (the root's reload runs after every value is dead), and
> what they lack is not a register but an allocator that can see the point they
> run at.

### D. Outside the DAG

`SCAFFOLD_ACC = Reg(0)`, `SCAFFOLD_SCRATCH = Reg(1)` — the collapse loop's own
X/Y arithmetic, reusing the *input* registers after the body has run. The reuse
is legitimate; the problem is that a person established it, not the allocator.

It is outside the allocator's view because it is outside the DAG, and it is
outside the DAG because **the DAG is loop-free for the e-graph's sake and the
allocator inherited that constraint for free.** Those are different
requirements: equality saturation needs acyclicity, while liveness is a fixpoint
dataflow analysis that is perfectly well-defined over a loop nest. One object
(`Vec<Def>`) serves both today, so the loop is excluded from an analysis that
could handle it.

The two scaffold registers are the small prize. The large one is `hoist_slots`:
LICM'd values from `frame_hoist`/`row_hoist` are **unconditionally spilled to
memory**, because the scaffold hand-manages them as slots. Hoisting a value out
of a loop *into a load* is a far weaker optimisation than hoisting it into a
register.

## Three apparent collisions — two of them phantom

Three backends have a fixed register whose number equals `select_reload`, whose
doc reads *"Must be untouched by the backend's own Select emission."*

| backend | `select_reload` | named collider | real? |
|---|---|---|---|
| sse2 | `Reg(13)` | `X86_BUILTIN_SCRATCH[1]` | **no** — `emit_const` takes the array as `_scratch` and ignores it |
| aarch64 | `Reg(28)` | `FMOV_FALLBACK_SCRATCH[0]` | **no** — `emit_fmov_imm` likewise ignores it |
| avx512 | `Reg(13)` | gather `IDX_INT` | **yes** |
| aarch64 | `Reg(28)` | gather `IDX_INT` | **yes** |

Tracing each parameter to a *read* rather than trusting the constant's name is
what separates these. Two of the "reserved" arrays are passed to functions that
never look at them, and both `emit_unary`s read only slot `[0]` of a `[Reg; 4]`.

That `[Reg; 4]` is itself a fossil. At the JIT's birth (`3626343f`) `emit_unary`
genuinely read all four slots — the encoder expanded transcendental polynomials
inline. `c781dc44`, *"the last hand-written polynomials leave the assemblers"*,
moved those expansions into `pixelflow-ir`'s passes. The requirement left; the
signature did not follow. A hand-rolled "this op needs N temps" frozen at a
width that stopped being true, with nothing able to notice — which is step 2 in
miniature.

**So the fixed set is far smaller than the constants suggest**: one register on
sse2 and avx2, three on avx512, two on aarch64.

## What the honest accounting reveals

Once the phantom reservations are removed and the survivors declared, every
register is accounted for — and what is left over is provably free:

| backend | file | pool now | provably free | pool could be |
|---|---|---|---|---|
| SSE2 | 16 xmm | 6 | 2 — `xmm14,15` | 8 |
| AVX2 | 16 ymm | 4 | 4 — `ymm8,9,10,14` | 8 |
| **AVX-512** | **32 zmm** | **6** | **16** — `zmm10, zmm17–31` | **22** |
| aarch64 | 32 v | 10 | 5 — `v4–7, v31` | 15 |

(aarch64 excludes `v8–v15`: AAPCS64 callee-saves their low 64 bits, and these
are leaf kernels emitted with no prologue that preserves them.)

AVX-512 allocates six of thirty-two registers while sixteen are untouched by
anything at all. **This is a performance change, not hygiene** — and the win
does not wait for step 2.

One obstruction, and it is another missing denotation. The pool is
`scratch_base: u8` plus `scratch_count: u8` — a *contiguous range*. The free
registers are not contiguous (`{14,15}` sits above `select_reload` at 13;
avx2's are `{8,9,10,14}`), so base+count structurally cannot express the pool
that is actually available. Either renumber everything to keep the range
contiguous, or let the pool become a **set**. A representation that cannot say
the true thing rounds the true thing down.

## How they formed

One cause. The allocator's input language cannot express an operation's
*scratch requirement*:

- `ScheduledOp` variants name only value operands.
- `Def` is one value plus one op.
- `Placement` is one register per value.

So an op needing a temp takes one from outside the pool, by hand, and writes
the disjointness argument in a comment. The pool then shrinks to leave room.
Every hatch above is that same workaround, repeated per backend.

## Closing them

Four steps, each independently valuable, in dependency order.

### 1. Declare the fixed registers (cheap, standalone)

Add `fixed: [Reg; N]` to `RegisterFile` naming every register a backend
reserves for its own use. Extend `checked()` to assert `fixed` is outside the
pool *and* disjoint from `inputs`, `reload`, `select_reload`.

Structurally this closes nothing — it converts three latent collisions into
const-eval failures. **This is the step that would have caught them.** Expect
it not to compile at first: each collision then resolves as either a real bug
or a documented, deliberate exception.

### 2. Let the allocator name values its input did not contain — dissolves A *and* C

The allocator's output becomes a *new* schedule rather than an annotation on
the old one: it may introduce defs the input DAG never had — instruction temps,
and spill/reload results — each allocated like any other value.

- **Temps** dissolve class A: `X86_SCRATCH`, `UNARY_SCRATCH`,
  `X86_BUILTIN_SCRATCH`, `FMOV_FALLBACK_SCRATCH` and the gathers' vector temps
  become allocated values, and their registers rejoin the pool.
- **Reload defs** dissolve class C: `Placement::Spilled` disappears (a spilled
  value is one with a `Spill` op and `Reload` ops), and `reload`/`select_reload`
  leave `RegisterFile` entirely.

Slots stay abstract in that output — a `SlotId`, not an offset — so
`FrameLayout` remains the memory manager and gains the ability to *coalesce*
disjoint live ranges onto one slot, which it cannot do today.

Register pressure *falls* even though more values are allocated: most ops need
no temp, so the pool roughly doubles on x86 and quintuples on AVX-512.

> **Landed 2026-09-03 — the temps half.** `RegisterFile::temps_for(&ScheduledOp)
> -> u8` declares the demand; the allocator reserves a slot before it places
> the destination, excluding every operand, and `Allocation::temp(i)` hands it
> to the encoder through `InstructionPlan::temp`. `UNARY_SCRATCH` is gone from
> all three of AVX2, AVX-512 and aarch64.
>
> The pool effect is smaller than the paragraph above predicts, and worth
> recording rather than rounding up:
>
> | backend | pool before | pool after | why |
> |---|---|---|---|
> | SSE2 | 6 | 6 | still `no_temps` — see below |
> | AVX2 | 5 | 5 | its `UNARY_SCRATCH` *was* `GATHER_IDX` |
> | AVX-512 | 22 | 23 | zmm15 joins `scratch` |
> | aarch64 | 15 | 16 | v29 joins `scratch` |
>
> AVX2 gains no register because ymm15 wore two names: it appeared in `fixed`
> twice, once as the gather's index and once as the unary mask, so the sign
> mask and the select blend had been *borrowing the gather's register* — safe
> only because no single instruction is both, exactly the convention step 1
> exists to make checkable. What landed there is one owner per register, not a
> larger pool.
>
> SSE2 keeps `X86_SCRATCH` in `fixed` deliberately. Its two-operand hazard
> (`emit_binary_safe`) and decomposed `MulAdd` demand a scratch as a function
> of *which registers allocation picked* (`dst == right`), not of the op — so
> the demand cannot be stated as `temps_for` yet, and drawing a second pool
> register for the unary mask meanwhile would cost pressure and free nothing.
> That backend closes when the hazard does.
>
> One thing the design forced out into the open: a temp cannot spill. Every
> value survives a small pool by going to memory, which is why a one-register
> budget was previously legal; scratch the encoder destroys mid-instruction has
> no such escape, so the pool must hold a ternary's three operands plus the
> temp. That floor is `RegisterFile::MIN_SCRATCH = 4`, asserted in `checked()`
> and clamped in `capped()` — `EmitCtx::max_regs` can no longer reach through
> it. Found by probe, not by reasoning: `with_max_regs(1)` on a `Neg` of a
> computed value hit the "unreachable" arm on AVX2 and AVX-512.
>
> **Reload defs (class C) are not done** — `Placement::Spilled` and
> `reload`/`select_reload` are still there. That is where the paragraph above's
> "roughly doubles" actually lives.

> **Landed 2026-09-03 — `select_reload`, the first of class C.** The scratch a
> backend declares became two named roles rather than one
> (`regalloc::Scratch { temp, arm_reload }`), and `select_reload` left
> `RegisterFile`. It was the clearest case in the class: one register held out
> of *every* kernel's pool so that the rare kernel with a `Select` whose result
> and both arms were all spilled had a third reload target, chosen at emit time
> by a three-way case analysis over two fixed registers. It is now a
> per-instruction reservation the allocator makes, disjoint by construction
> from that instruction's operands, destination and temp.
>
> | backend | pool before | pool after |
> |---|---|---|
> | SSE2 | 6 | **7** |
> | AVX2 | 5 | **6** |
> | AVX-512 | 23 | **24** |
> | aarch64 | 16 | 16 — see below |
>
> `arm_reload` is reserved for *every* `Select`, not only the ones that turn out
> to need it, and that is forced rather than lazy. A value resident when its
> reader is allocated can still be evicted by a later instruction — `Placement`
> is one answer per value for its whole life — so residency read at the point of
> reservation is not final. **This is the constraint that shapes the rest of
> class C**: the fixed point between "which values spill" and "how many reload
> registers each instruction needs" is why the remaining work is a redesign of
> the allocator's output contract rather than another field, and why it is not
> attempted here.
>
> Over-reserving could have cost more spilling than the extra pool register
> buys, so it was measured on the shape that maximizes the risk — *w*
> independent `Select` chains all live at once:
>
> | live selects | before (pool 6) | after (pool 7) |
> |---|---|---|
> | 8 | 4 | 4 |
> | 12 | 9 | **8** |
> | 16 | 15 | **14** |
> | 24 | 27 | **26** |
>
> Unchanged or slightly better throughout: the register the pool gains pays for
> the register a `Select` transiently borrows.
>
> aarch64 gains nothing, and finding out why is the point of step 1's
> accounting. v28 was `select_reload` *and* the register
> `emit_skip_if_all_false`/`_true` reduce a mask into with `UMAXV`/`UMINV` —
> two roles, one register, and only one of them declared. Returning v28 to the
> pool would have let the allocator hand it to a live value that a `Select`'s
> own short-circuit guard then clobbered. It is now `GUARD_SCRATCH` in
> `fixed`, where `RegisterFile::checked` can see it. The x86 tiers have no
> equivalent: their guards go through `movmskps`/`kortest` and the flags, so
> they need no vector register at all.

> **Landed 2026-09-03 — the gathers, and class A is closed.** `Scratch` carries
> up to `MAX_TEMPS` registers instead of one, so an encoding can ask for
> several, and every gather's scratch became a per-instruction reservation. The
> gathers were the last of class A: a 256-bit AVX2 gather needs four registers
> (a 128-bit half's index and value, plus one of each to carry the high half),
> AVX-512 and SSE2 two, aarch64 one — each held out of the pool for the whole
> kernel, for an instruction most kernels do not contain at all.
>
> | backend | pool at step 1 | now | of |
> |---|---|---|---|
> | SSE2 | 6 | **9** | 16 |
> | AVX2 | 4 | **10** | 16 |
> | AVX-512 | 6 | **26** | 32 |
> | aarch64 | 10 | **17** | 32 |
>
> Every one is past the "pool could be" column in the table above — that column
> was drawn before temps were allocatable, so it still charged each backend for
> the registers its own encodings borrow. `fixed` is now **empty** on AVX2 and
> AVX-512, and holds exactly one register on SSE2 (`X86_SCRATCH`, whose demand
> is a function of which registers allocation picked rather than of the op) and
> one on aarch64 (`GUARD_SCRATCH`).
>
> `MIN_SCRATCH` rose from 4 to 6 with it, and the reason moved: it is no longer
> a ternary's three operands plus a temp but AVX2's gather — one operand plus
> four temps — with room for the destination either way.
>
> Three tests had a pool size written into them as a literal and stopped
> testing anything when the pool grew past it. `muladd_and_clamp_spilled` said
> "Sethi-Ullman number > 6" and now sizes its fillers from `pool_size()`;
> `belady_evicts_the_value_used_farthest_out` and
> `the_pool_may_straddle_a_reserved_register` are stated in terms of
> `MIN_SCRATCH`. A fourth, `a_spilled_muladd_rounds_twice_on_every_target`, was
> worse than stale: its wall was `(l − r) − (l − r)`, which `legalize` folds to
> a single constant, so there was no wall — it reached the decomposed arm only
> because the pool was smaller than three live values. Its terms are now
> `(X + i) · W`, worth exactly zero at W = 0 and not foldable, because a
> variable is the one thing the folder cannot see through.

> **Landed 2026-09-03 — SSE2's last fixed register, and the hazard it was held
> for was not real.** `fixed` is now **empty on all three x86 tiers**; aarch64's
> `GUARD_SCRATCH` is the only one left in the workspace. The temps block
> predicted "that backend closes when the hazard does" — and the hazard turned
> out not to exist.
>
> `emit_binary_safe` stashed the right operand whenever the allocator chose
> `dst == right` for a non-commutative binary, because SSE2's `dst op= right`
> would corrupt it. **The allocator never makes that choice.** A destination
> takes a pool slot with no live owner, or evicts one — and an evicted value's
> placement becomes `Spilled` for its whole life, so `resolve_operands` reads
> it back from memory rather than from the register the destination took. The
> right operand is therefore a pool register no destination can alias, an
> input register (outside the pool), or `reload[1]` (outside the pool), or —
> when both operands are spilled — `dst` itself, in which case `left` is `dst`
> too. So the case was not a demand for scratch. It was a fallback for
> something unrepresentable, and one xmm was held out of every kernel's pool
> to serve it.
>
> The register moved rather than the argument: the invariant is now stated
> where the registers are *chosen* (a `debug_assert` in `resolve_operands`,
> which fails loudly in every debug build if the allocator stops guaranteeing
> it) and tested where they are *allocated*
> (`a_destination_never_lands_on_a_resident_operand`, which asserts a
> destination never lands on a pool-resident operand and counts the contested
> pairs so it cannot pass vacuously). The three real demands on that
> register — `Neg`/`Abs`'s sign mask, the select blend, and the `movaps`/
> `mulps`/`addps` stand-in's product, this tier having no FMA — are
> `temps_for` answers like every other backend's.
>
> | backend | pool before | pool after | of |
> |---|---|---|---|
> | SSE2 | 9 | **10** | 16 |
>
> Worth measuring rather than assuming, because the ops now reserving a temp
> are `Neg`, `Abs`, `Select` and `MulAdd` — not the gathers most kernels lack —
> and a reservation under pressure evicts an occupant permanently, which is
> exactly what sank the reload-target attempt below. It does not repeat here:
>
> | kernel | before (pool 9) | after (pool 10) |
> |---|---|---|
> | wide 16 | 8 spills / 1340 B | **7** / **1302 B** |
> | wide 32 | 24 / 2876 B | **23** / **2832 B** |
> | abs 8 | 0 / 835 B | 0 / **812 B** |
> | abs 16 | 8 / 1715 B | 8 / **1452 B** (−15.3%) |
> | muladd 8 | 0 / 711 B | 0 / **680 B** |
> | muladd 16 | 8 / 1471 B | 8 / **1192 B** (−19.0%) |
> | select 8 / 12 / 16 | 2 / 6 / 10 spills | unchanged, same bytes |
>
> Same or fewer spills everywhere and never more code. The gain is largest on
> the unary and `MulAdd` shapes, where the temp is now whichever pool register
> is free instead of always xmm10 — a low register needs no REX prefix, so the
> encoding shrinks. `Select` keeps its size because its temp lands on a high
> register either way; the bytes differ, the count does not.
>
> Verified by execution, not identity, per the note below: the whole workspace
> on the SSE2 baseline, and `pixelflow-codegen` plus `pixelflow-ir` on
> `+avx2,+fma` and `+avx512f,+avx512dq` — the JIT-vs-interpreter suites
> (`spill_pressure`, `prod_kernel_jit`, `transcendental_jit`,
> `oracle_reference`) are the differential half.

> **Built and measured 2026-09-03 — frequency-weighted allocation is worse, and
> the reason names what is actually missing.** The carry policy that landed
> above picks roots by raw body-read count and caps them at
> `pool − MIN_SCRATCH`. Both are hand-picked constants, and the symptom was
> shape-sensitivity: a kernel with 6 invariants got 4 of them resident, one
> with 202 got zero, because the *choice* of carry register was made from
> whatever the producing region happened to leave free.
>
> The diagnosis was that the allocator has no cost model, only a heuristic.
> `LinearScan` evicts by Belady — farthest out in **schedule order** — which is
> optimal for cache replacement under the assumption that every miss costs the
> same. A loop nest breaks that assumption: a value read once per inner
> iteration costs one reload *per iteration*, one read in the prologue costs
> one reload, and `last_use` is a schedule index that cannot tell them apart.
>
> So `WeightedScan` prices the reload instead of measuring the distance to it.
> Every read contributes `10^depth`; eviction takes whichever resident value is
> cheapest to bring back; a root is live past its region (`Pricing::live_out`)
> so ordinary eviction keeps it. Carries stop being budgeted and simply fall
> out of the eviction rule.
>
> **It is substantially worse.** Real glyph bakes (257 kernels): carries rise
> 578 → 1021, and emitted code rises **3.01 MB → 4.12 MB, +37%**. Corroborated
> on a synthetic invariant kernel, where the mechanism is legible:
>
> | invariants | LinearScan | WeightedScan |
> |---|---|---|
> | 4 | 0 spills / 498 B | 0 / 492 B |
> | 16 | 0 / 1422 B | **13 spills** / 1937 B |
> | 48 | 0 / 3886 B | **77 spills** / 6641 B |
>
> The cause is a half-priced trade. Carrying is charged to the body — every
> inner scope gets its pool as `file.inside(carried)` — so a carried invariant
> displaces a body value. Both are read at the same depth, so they are worth
> the *same* per iteration; but the displaced one is spilled, which costs a
> store as well as a reload. Pricing the benefit of carrying without pricing
> its cost therefore trades equal-weight reads and pays store traffic on top.
>
> Two things worth keeping from it. First, **`MIN_SCRATCH` is load-bearing in a
> second way**: `LinearScan`'s `pool − MIN_SCRATCH` budget is not only policy,
> it is what keeps the body above the encoding floor, and removing it as a
> "constant to be eliminated" made the allocator run out of evictable registers
> entirely. Second, the floor is *not* a sufficient cap — carrying right up to
> it leaves the body enough registers to emit and not enough to avoid spilling,
> which is the whole 37%.
>
> What the result actually asks for is not a better weight. It is that the
> regions and the body be allocated **against one pool at once**, so the
> invariant's demand and the body's compete directly, instead of the body
> receiving whatever the regions left. That is a global allocation over the
> nest rather than a sequential one, and it is a larger change than this was.
> The code is reverted; the trait sharpening it motivated is not (see the
> commit that made `allocate_nest` the required method), and it is what a
> second allocator will plug into when one is worth shipping.

> **Methodology note.** Byte identity carried #1059–#1062 because those were
> pure refactors: the emitted code was supposed to be unchanged, so an empty
> diff was the whole proof. From here on the emitted code is *supposed* to
> change — freeing registers is the point — so an empty diff would mean the
> change did nothing. The successor technique is a **structured** diff (assert
> that the only bytes that moved are the register fields intended to move)
> backed by differential execution across all four backends.

### 3. The class axis — dissolves class B

`RegClass { Vector, General, Mask }`; `RegisterFile` keyed by class; `Placement`
carrying one. `Gpr`/`Xr` already exist as newtypes; add `KReg`. A *newtype*,
not `type KReg = u8` — a transparent alias is a comment with syntax and would
let any `u8` through.

The gathers' GPRs and `SCRATCH_K` become allocated. This is also the
prerequisite for real k-register predication on AVX-512 — masked ops instead of
blend sequences — which is where that backend should eventually go, and which
is impossible while `k1` is a hardcoded transient.

### 4. Show the allocator the loop — dissolves class D

Give the allocator the whole emitted function, scaffold included, rather than
just the loop-free body. The e-graph keeps its acyclic domain; the allocator
gets a loop nest and computes liveness over it.

`SCAFFOLD_ACC`/`SCAFFOLD_SCRATCH` stop being a hand-argued reuse and become an
allocation the allocator can prove. More importantly, `hoist_slots` stops being
an unconditional trip to memory: loop-invariant values compete for registers on
merit, which is what LICM was supposed to buy.

Until then, class D should at least be *declared* under step 1, so "nothing is
live here" is written where `checked()` can see it.

> **Landed 2026-09-03 — the allocator can see the loop, and LICM stops going to
> memory.** `allocate_nest` was three independent straight-line allocations
> against the same pool; it is one problem now. Regions are allocated
> outermost-first, and a root the innermost body reads can be **carried** — kept
> in a register for the whole of the loops inside, instead of parked in a slot.
> `RegSet::without` and `RegisterFile::inside(carried)` are the vocabulary: the
> pool a scope inside a loop sees is the file's pool minus whatever is carried
> across it, which is what allocating every region against the full pool used to
> ignore.
>
> **What the old shape actually cost, measured before changing anything.**
> `HoistCtx::Body` pinned every hoisted value to `Loc::Spill`, so every *use*
> became a `Reload::FromStack` — on every iteration of the inner loop, for a
> value that provably does not change within it. Uses tracked hoist count almost
> exactly across the suite, and the worst kernel measured hoists **48** values:
> 48 stack loads per batch iteration, for 48 constants.
>
> | kernel | hoisted | uses in body |
> |---|---|---|
> | small | 1–2 | 1–2 |
> | mid | 5, 10 | 5, 10 |
> | prod | **48** | **48** |
>
> Also measured, and worth recording because it is *not* where the cost is: the
> scaffold itself is a constant **268 bytes** on every kernel checked (twelve of
> them, 268 or 269 every time). The loop's overhead is not its size, it is the
> traffic inside it.
>
> **Result, inner-loop bytes, `n` loop-invariant terms each read once:**
>
> | n | SSE2 (pool 10) | AVX2 (pool 10) | AVX-512 (pool 26) |
> |---|---|---|---|
> | 4 | 589 → 498 | 560 → 460 | 634 → **526** |
> | 8 | 897 → 806 | 820 → 720 | 938 → **722** (−23%) |
> | 16 | 1513 → 1422 | 1340 → 1240 | 1546 → **1114 (−28%)** |
> | 48 | 3977 → 3886 | 3420 → 3320 | 3978 → **3438** |
>
> **Zero added spills at every size on every tier.** Code size understates the
> win: each byte removed is a load that was executing every iteration.
>
> The shape of that table is the point. The carry budget is
> `pool − MIN_SCRATCH`, so the 128-bit tiers saturate at four carries while
> AVX-512 gets twenty — **the backend with the most registers gains the most,
> having previously gained nothing from having them.** That is the same
> observation step 1's accounting opened with, arriving from the other end.
>
> **The safety property, and the test that states it.** A carried register is
> read by the body on every iteration, so anything inside the loop writing it is
> a miscompile that surfaces only as wrong pixels. It holds by construction —
> the body's pool is `file.inside(carried)` — but the carry is chosen from what
> the *producing* region leaves free, and that had a trap worth recording: the
> first version took the complement of `placements`, which does not include
> instruction **temps**. A temp is a pool register no `Placement` records, so
> that version could hand out a register the prologue destroys.
> `a_carried_register_is_untouched_by_everything_inside_the_loop` checks
> placements *and* temps *and* `arm_reload`, and asserts something was carried
> at all — which immediately caught its own first draft, written against a file
> whose pool is exactly `MIN_SCRATCH` and whose budget is therefore zero.
>
> **What did not land: `SCAFFOLD_ACC`/`SCAFFOLD_SCRATCH`.** They are still
> hand-chosen, and they are still safe, for a reason this work makes precise
> rather than removes: they are `Reg(0)`/`Reg(1)`, *input* registers, which
> `checked()` already holds outside the pool — so a carried pool register
> survives the scaffold untouched, and the two of them survive the body. What
> makes them safe is that every coordinate round-trips to a slot and is reloaded
> at the top of each iteration. Keeping the **coordinates** in registers is the
> other half of class D — induction variables, not invariants — and it is what
> would finally make those two an allocation rather than an argument. Separable,
> and not attempted here.

## Order of attack

1 first, alone: it is small, it is a strict improvement, and it tells us
whether the three collisions are bugs before any redesign is built on top of
them.

Then 2 — the largest single win, and it retires two of the four classes at
once.

Then 3 when k-predication is actually wanted, and 4 when LICM's trip to memory
is worth reclaiming. Neither is a prerequisite for the other; both need 2
first.

**Revised 2026-09-03.** Step 2's class-A half is done and `select_reload` with
it; what is left of 2 is `reload` and `Placement::Spilled`. Two findings move
the rest:

1. **The remainder of 2 has to land with 4, or it buys nothing** — three of
   `reload`'s use-sites are outside the DAG (see the correction under class C).
2. **It needs live-range splitting, and this was measured, not argued.**
   Eviction writes `Placement::Spilled` for a value's *whole life*, so an
   operand resident when its reader is allocated can be spilled retroactively
   by a later instruction. Every reservation that landed so far dodged this
   because its demand is a function of the *op* — `temps_for` can state it.
   A reload target's demand is a function of *which operands are in registers
   at that point*, which is not final when the reservation is made.

   Restricting eviction to values nothing has read yet does make an operand's
   residency final, and it is cheap: +1 spill across every size measured. The
   whole change was built on that and it still failed, for a reason worth
   writing down rather than rediscovering.

   **Reserving a register that is needed for one instruction costs a value its
   register for the whole program.** Under pressure no pool slot is free, so
   the reservation evicts an occupant — and eviction is permanent. Every
   spilled operand therefore causes another spill, and constants are operands:

   | wide kernel | pool 9, `reload` fixed | pool 11, targets reserved |
   |---|---|---|
   | 4 terms | **0** | 4 |
   | 8 terms | **0** | 8 |

   Two more registers do not come close to paying for that. The two fixed
   registers are cheaper than the pool ones precisely *because* they are
   outside the pool: taking one costs nothing.

   And the destination cannot be reserved for at all. `evictable` makes an
   operand's residency final from its first *read*, but a value's own
   definition is a write — so a destination allocated a register can still lose
   it later, which is exactly what `reload[0]` exists to catch.

   So the remainder is not a reservation problem. A reload has to become a
   *value with a short live range* — the "output schedule rather than an
   annotation" this plan already names — because only then does materialising
   one cost a register for an instruction instead of for a program.

### The splitting version, built and measured

Built next, since that is what the paragraph above asks for. Eviction stops
writing `Placement::Spilled` over a whole life and instead **splits**: the value
keeps its register up to the eviction, a store hands the register on there, and
later reads name a memory-resident half the input never contained. On top of
that, every reload target and the destination scratch become per-instruction
reservations, `reload` leaves `RegisterFile`, and each pool grows by two.

Splitting is what makes the reservations sound, and it is worth writing down
why: a value evicted later keeps the register it held *earlier*, so residency
read when an instruction is allocated is final, and a destination holds a
register at its own definition or never. Both were exactly what the previous
attempt could not guarantee.

Measured, same kernels as above, on the SSE2 tier:

| kernel | before (pool 9) | after (pool 11) | code size |
|---|---|---|---|
| wide 16 | 8 spills / 2127 B | 8 / 1903 B | **−10.5%** |
| wide 32 | 32 / 4480 B | 32 / 3834 B | **−14.4%** |
| anchored 12×40 | 5 / 1329 B | 4 / 1132 B | **−14.8%** |
| anchored 16×64 | 9 / 2047 B | 8 / 1640 B | **−19.9%** |

Same or fewer spills everywhere and 0.7–20% less code, the gap widening with
pressure. The whole codegen suite passes on all three x86 tiers, and the
workspace passes apart from the case below.

**It is not merged, because of one hole in guarded regions.** A `Select`'s
short-circuit arms are instructions a branch can skip, which constrains where a
split's store may go:

- A value defined *before* an arm can store before the arm's branch — that path
  always runs.
- A value defined *inside* an arm has no such point. Storing at its definition
  is right only if no *nested* guard sits between there and the eviction;
  ignoring that nesting miscompiles glyph bakes, which
  `kernel_glyph_golden.rs` catches.
- A `Select`'s own operands cannot be split at all: the guard reads its mask
  under the name the pre-redirect analysis recorded, and the arms are read at
  the `Select`, outside both ranges.

Those three leave a residue of registers that can be neither split nor spilled
whole — the latter needs the value to be unread so far — and an instruction
inside a guarded region can find every register in that state and have nowhere
to put its scratch. It is reachable: a warm glyph cache reaches it.

Closing it means the store point has to be chosen against the *nesting* of
guard ranges rather than a flat "is this index guarded", and the `Select`
operands need the guard analysis to name values by their post-split identity.
Neither is deep, but both are the kind of thing that miscompiles quietly, and
the measurement above is the reason it is worth doing properly rather than
quickly.
