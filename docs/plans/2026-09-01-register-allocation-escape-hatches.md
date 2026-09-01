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

## Order of attack

1 first, alone: it is small, it is a strict improvement, and it tells us
whether the three collisions are bugs before any redesign is built on top of
them.

Then 2 — the largest single win, and it retires two of the four classes at
once.

Then 3 when k-predication is actually wanted, and 4 when LICM's trip to memory
is worth reclaiming. Neither is a prerequisite for the other; both need 2
first.
