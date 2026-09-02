# Design Doc: Preemption as Transaction Abort

## Metadata
- **Author**: jppittman (with Claude)
- **Status**: Exploratory — **not adopted**. Captured so the reasoning is not lost.
- **Created**: 2026-08-31
- **Companion**: `docs/designs/2026-08-31-hardware-sandboxed-kernel-preemption.md` — the KVM
  alternative. This design covers most of the same ground far more cheaply; that one covers the
  case this one structurally cannot.
- **Amends**: `docs/designs/actor-scheduler-mealy-transducer.md` §5 — specifically the claim
  that a signal "cannot interrupt a handler mid-body (nor should it need to)."
- **Reviewers**: —

---

## 1. The insight

Preemption is normally expensive because it means **suspension**: capture the program counter,
the stack, the register file — for a SIMD kernel, ZMM0–31 and k0–k7 — stash it somewhere, and
be able to restore all of it later. That is why the KVM design is attractive: hardware does the
capture for you.

The Mealy formulation makes the capture unnecessary. An actor is `δ : State × Input → State ×
Output`. `mealy.rs`'s own module doc states the premise exactly:

> always runs to completion, so there is no live stack to suspend: everything that survives
> between inputs is already in `self`.

If all surviving state is in `self`, and the input message is still owned by the scheduler, then
a step interrupted midway needs **nothing preserved**. The actor is still in `state`; the message
is still unconsumed. Discard the partial computation, return to the sweep loop, and retry the
step later from scratch.

**Preemption becomes a transaction abort, not a context switch.** No register file to save, no
stack to keep, no coroutine, no second address space. The cost is the discarded work, bounded by
the quantum.

The second half: the thing you abort *to* already exists. `impl Transducer for Host` makes the
host itself a Mealy machine whose input is `RunSweep`. The abort target is the point in
`Node::poll` / `Host::sweep` where `Step::Ran` would have been returned. `DedicatedThread` and
`GreenThread` need no changes — the preempted step lands exactly where a completed one would
have. The scheduler layers on itself; the "hypervisor" is just the next Mealy machine up, in
userspace.

## 2. It reduces to one invariant

> **A step must not commit until it completes.**

Everything else in this design follows from that, and the current trait does not enforce it.
`fn step_data(&mut self, msg: Self::Data) -> Result<Self::Out, HandlerError>` takes `&mut self`,
so a handler is free to mutate incrementally. Abort such a step and the actor is left in a torn
intermediate state — neither `state` nor `state'`.

Per `CLAUDE.md`'s "when you extend a type's meaning, extend its type," the fix is a signature,
not a comment. A preemptible transducer wants the commit to be a move the scheduler performs:

```rust
// Sketch, not a proposal in final form.
fn step_data(&self, msg: &Self::Data) -> Result<(Self::State, Self::Out), HandlerError>;
```

The handler computes into a value; `Node` installs it. Abort is then *dropping a local*, and
abort-safety is a property the type system guarantees rather than a discipline that survives
until someone writes an optimization.

Note this is the same invariant that makes a transducer table-testable with no scheduler in the
loop — which the mealy design already claims as a goal. **The preemptible signature and the
pure-step goal are the same signature.**

## 3. Mechanisms

### 3.1 Peek/commit instead of take

`Node::poll` currently does `self.data.take()` and moves the message into the dispatch closure,
so an abort drops it. The message should never be consumed until the step commits:

- Add `peek` to the `Inbox` trait: read the slot at `head` without advancing.
- Run the step against the peeked message.
- Advance `head` only on successful commit.

The SPSC ring makes this nearly free — the consumer already owns `head`, so a peek is "read the
slot, don't store the new head." It is the input-side mirror of the parked `outbox` already used
for blocked flushes.

With this, an aborted step loses **only the partial computation**, never the message. The
semantics become at-least-once execution of a pure function, which is safe precisely because of
§2's invariant.

### 3.2 The RIP-range gate — what makes this sound rather than hopeful

The obvious objection to signal preemption is that the signal lands wherever it lands: inside the
allocator, inside a `Wiring::flush`, inside libc, inside the doorbell. `longjmp`ing out of those
is at best a leak (Rust destructors are skipped) and at worst a torn invariant.

**Because pixelflow JITs its kernels, the runtime knows their exact address range.** In the
handler, read `uc_mcontext.gregs[REG_RIP]` (the third `SA_SIGINFO` argument is a `ucontext_t*`
holding the state `sigreturn` will restore) and:

- **RIP inside the emitted kernel's `[start, end)`** → abort: jump back to the sweep loop.
- **Anywhere else** → set a deferred flag, return normally, let the scheduler notice at the next
  step boundary.

This converts async-signal-safety from a global property you hope holds into a **predicate you
evaluate**, and the two sides line up exactly: the JIT'd kernel region is precisely the code with
no `Drop` locals, no allocation, no locks, and no syscalls. Everything outside it is simply not a
preemption point. This is the load-bearing safety argument of the whole design.

### 3.3 Signal plumbing

- **Not `SIGALRM`.** `setitimer(ITIMER_REAL)` raises a *process-directed* signal — the kernel
  picks an arbitrary unblocked thread — and there is only one per process. Use a realtime signal
  (`SIGRTMIN+n`): they queue rather than coalesce and nothing else in the ecosystem claims them.
  (Go picked `SIGURG` for its async preemption on the same "nobody else wants it" reasoning.)
- **Handler disposition is per-process; the signal mask is per-thread.** Register once with
  `sigaction`, then block the signal on every thread that must not be preempted — the Cocoa main
  thread, PTY I/O threads.
- **Targeting**: `pthread_kill(tid, sig)` from a ticker thread works everywhere including macOS.
  Linux additionally offers `timer_create` with `sigev_notify = SIGEV_THREAD_ID` for per-thread
  timers with no ticker thread.
- **Handler stack**: `sigaltstack`, so a preemption landing on a deep kernel frame has room.

### 3.4 `setjmp` placement and the mask trap

Put the `setjmp` **per sweep, not per step** — one save at the top of the sweep loop, not on
every `poll`.

The trap: when a handler runs, that signal is blocked, and returning through `sigreturn` is what
unblocks it. `longjmp` out of the handler and `sigreturn` never happens — **the preemption signal
stays masked forever and you get exactly one preemption, then silence.** Either pair
`sigsetjmp(env, 1)` with `siglongjmp` (correct, but the mask save costs a syscall per sweep), or
use `_setjmp`/`_longjmp` and explicitly `pthread_sigmask` the signal back on immediately after
landing.

This is the same species as the doorbell lost-wakeup (`doorbell.rs`): **the state that says "I
can be signalled again" must be restored on the path actually taken, not the path expected.**

## 4. The failure mode to design for

**A step that takes longer than the quantum can never complete.** Abort, retry, abort, retry —
zero progress, forever. This is the transactional-memory fallback problem, and it fails
*silently*: a frame that never renders, not a crash.

It needs an explicit escape valve. After N consecutive aborts of the same step, either:

- run that step to completion uninterrupted (mask the preemption signal for its duration), or
- escalate the quantum geometrically until it fits.

`actor-scheduler` already has the vocabulary — this is a burst budget by another name, and the
abort counter belongs next to `slot_progress` in `Node`.

## 5. What this is and is not

**It is a starvation backstop.** It guarantees that one long-running actor cannot freeze its
host indefinitely.

**It is not fair timeslicing.** Fairness would require finishing work that this design keeps
discarding. An actor whose steps routinely approach the quantum will burn CPU on retries and
make progress only through the §4 escape valve. If workloads like that are common, the answer is
tiling the work into smaller steps (natural in a pull-based sampler — render bounded coordinate
tiles and return between them), not leaning harder on preemption.

## 6. Relationship to the mealy design's §5

The mealy design's placement table says preemption granularity "is the process/thread, never the
green actor," and that a signal "cannot interrupt a handler mid-body (nor should it need to)."

The first clause stays true in spirit; the second is what this design contradicts. A signal
*can* interrupt a handler mid-body safely, given (a) the commit-at-the-end invariant, and (b) the
RIP-range gate restricting aborts to JIT'd kernel code. If this is ever adopted, §5 needs
amending: a green actor becomes preemptible at kernel granularity, without acquiring a thread,
a stack, or a `Send` bound.

## 7. Why this over the KVM design

For pixelflow's actual constraints, this wins on every axis except one:

| | Transaction abort | KVM sandbox |
|---|---|---|
| Platforms | Linux + macOS (BSD signals) | Linux only |
| Permissions | none | `/dev/kvm`, `kvm` group |
| Emitter work | none | freestanding long-mode guest ABI |
| Cost when idle | one `_setjmp` per sweep | one VM-entry per kernel |
| Partial work | **discarded** | **preserved** |

That last row is the whole difference. This design is cheap *because* it throws work away, which
is exactly why the KVM design retains a narrow justification: computations where discarding
partial results is unacceptable, or code whose abort-safety cannot be required because it is not
ours.

## 8. What is worth doing regardless

Two pieces of this earn their keep **with no signal machinery at all**, and could be built
independently:

1. **The pure-step signature** (§2) makes the transducer genuinely transactional and
   table-testable, which the mealy design already wants.
2. **Peek/commit on the inbox** (§3.1) makes a failed or refused step non-destructive, which
   pairs with the existing parked-outbox backpressure story.

The signal machinery (§3.2–3.4) only earns its keep once a step is *actually* observed to
overrun. Nothing has been measured overrunning yet — that measurement is the gate on building
any of this.
