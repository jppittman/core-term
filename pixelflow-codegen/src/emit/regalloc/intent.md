# pixelflow-codegen/src/emit/regalloc

**Is.** The register allocator of the CPU backend: a function from a machine
function over *values* to the same function over *registers and slots*
(DENOTATION §10). At every point the map from live value to location is
injective, every operand constraint holds, and every instruction it inserts
preserves a value from its definition to every read.

## Owns
- Every register. A `Reg` is minted once per compile by `Pool::mint`, leased
  only by `Leases::new`, and reaches an encoder only as an `In`, `Out` or
  `InOut` borrowed from a lease. Whoever owns a resource makes every choice
  about it.
- Every frame slot (`Frame`, `SlotLease`), the frame's size (`MAX_FRAME`),
  and the stack pointer.
- Eviction, rematerialization, spill placement (right after the definition,
  never at the eviction point a guard could skip), carries across a loop,
  and the join at a forward branch.

## Never
- No register is named outside this directory. A requirement the ABI or an
  instruction fixes (`cl`, `rax`/`rdx`, the argument registers) is a
  constraint on a *value* that selection states; the choice of any other
  register is the allocator's, with no exception (JP, 2026-10-16: "No to
  register allocation out of the allocator. Ever.").
- No prediction from an earlier phase: it is told nothing about temps,
  scratch counts or reserves. Everything an instruction needs is a value
  selection defined; a reload result is a value the input DAG did not
  contain.
- No knowledge of what a value means. It sees classes, lives and
  constraints, never an `OpKind`, a mask's domain or a lattice.
- No register is guessed behind the scan. A store placed after a definition
  names the stored value's register and, for a slot past the store's reach,
  one the allocator's own record of that point says was lent to nothing
  (`Life::free_general`, `Scan::free_general_at_entry`) — never one picked
  for having gone unused so far (JP, 2026-10-15: "ask the allocator for a
  register, or don't have one").
- Never silent: running out of frame is `Err(BudgetExceeded)`; failing to
  find a register for a value is a panic naming the instruction, a selection
  bug, never a fact about a kernel. It never picks a register outside its
  leases.

## Contract
`mod.rs`: `local` is the allocator (`Allocate`, the `Pass` from a selected
`Function` to an allocated one), `policy` is what it prefers
(`EvictionRank`, `carried`), `resource` is what it allocates from (`Reg`,
`Pool`, `Lease`, `Frame`, the `In`/`Out`/`InOut` borrows). Everything is
`pub(super)` or `pub(in crate::emit)`; nothing escapes `emit/`.

## Enforced by
- Types: `Reg` is not `Copy`, has one constructor, and is branded by backend,
  so a register chosen elsewhere, two holders of one register, or an AVX-512
  token in an AVX2 encoder does not compile (`resource.rs`).
- `scripts/check-emit-boundary.sh` (CI job "Emit boundary"): `program/`
  names no allocator and no register.
- desloppify `registers-come-from-the-allocator`, `phases-in-order`,
  `invariant-in-comment` (`copyable_resource`).
- The as-if contract: tests observe values, refusals and no-crash through
  the production API, never spill counts or register choices
  (`pixelflow-codegen/tests/register_pressure.rs`, `deep_frame.rs`,
  `residency.rs`, `loops.rs`; `docs/process/2026-10-13-tests-check-the-as-if-contract.md`).

## Behind
- An encoder's register field takes a bare `u8` from `Reg::number`, so the
  header's promise is one step weaker than it reads (S3).
- Nothing in CI checks the ABI's callee-saved registers (S1).

## Decided in
`docs/archive/plans/2026-10-08-selection-is-a-phase.md` (§0.5 fixed registers, §2.1
classes, the allocation guarantees), `docs/archive/plans/2026-09-01-register-allocation-escape-hatches.md`,
`docs/archive/plans/2026-09-22-a-pointer-is-a-value.md`,
`docs/language/2026-10-15-kernels-evaluate-kernels.md` §3 (the CPU module alone
owns register allocation).
