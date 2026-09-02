# Design Doc: Hardware-Sandboxed Kernel Preemption (KVM)

## Metadata
- **Author**: jppittman (with Claude)
- **Status**: Exploratory — **not adopted**. Captured so the reasoning is not lost.
- **Created**: 2026-08-31
- **Companion**: `docs/designs/2026-08-31-preemption-as-transaction-abort.md` — a cheaper
  design that covers most of the same ground. Read both; they are alternatives, and the
  companion is the one more likely to be built.
- **Amends**: `docs/designs/actor-scheduler-mealy-transducer.md` §5 (the "own process" row of
  the placement table)
- **Reviewers**: —

---

## 1. The problem

A green actor's step is run-to-completion. `Host::sweep` advances each adopted `Node` by one
`poll()`, and a `Node::poll` that enters a long compute does not come back until it is done.
Every other green actor that host owns is frozen for the duration. The mealy-transducer design
names this and accepts it (§1.3 Non-Goals: "Preemption *inside* a handler"), on the reasoning
that anything needing preemption gets its own thread or process.

That reasoning holds right up until the long thing is a **fused SIMD kernel**, which is
pixelflow's entire output. The usual fix — insert cooperative safe points, check a flag every N
iterations — is unavailable here on purpose: instrumenting the inner loop of a kernel wrecks
exactly the codegen the compiler exists to produce. `CLAUDE.md` is explicit that SIMD is an
implementation detail and users write equations; a yield check is neither.

So the question is whether a kernel can be preempted **with zero instrumentation** — no safe
points, no flag polls, no compiler cooperation.

## 2. The mechanism

Run the kernel inside a KVM guest and let hardware do the preempting.

1. `open("/dev/kvm")` → `KVM_CREATE_VM` → `KVM_CREATE_VCPU`.
2. Map a host arena as guest physical memory via `KVM_SET_USER_MEMORY_REGION`. This is the
   kernel's input/output buffers — one mapping, established once, no per-frame work (which
   preserves the zero-allocation rule).
3. Load the JIT-emitted kernel at a known guest physical address, set `rip` to its entry.
4. Arm a per-thread timer. `KVM_RUN` enters guest mode; the CPU executes the kernel at native
   speed with full AVX-512/NEON available (KVM exposes host CPU features through CPUID).
5. Timer fires → signal → the vCPU is forced out of guest mode. `KVM_RUN` returns `-EINTR`
   (exit reason `KVM_EXIT_INTR`), with the **entire vCPU state — including the vector register
   file — checkpointed by hardware**.
6. The host scheduler does whatever it likes, then calls `KVM_RUN` again to resume exactly
   where the kernel left off.
7. A guest `hlt` at the end of the kernel exits with `KVM_EXIT_HLT` — completion and preemption
   arrive through the same path, distinguished by exit reason.

## 3. Why it is attractive

**Resumability.** This is the property nothing else on the table has. A bare `SIGALRM` to a
worker thread can *interrupt* a SIMD computation but cannot cleanly *resume* it — you cannot
`longjmp` out of the middle of a vectorized loop and come back. KVM checkpoints the whole
vCPU, so "pause this computation, run something else, resume mid-flight" is a solved problem
rather than a project.

**No async-signal-safety problem, structurally.** The usual terror of signal-based preemption is
"what was the interrupted thread holding?" — a lock, a half-initialized allocation, a `Drop`
guard. Here the question is unanswerable in the bad direction: the guest runs in a different
address space's execution context. It cannot hold one of the host's locks, cannot be inside the
host allocator, and cannot corrupt host state. Preemption safety stops being a property you
audit and becomes a property of the boundary.

**Zero instrumentation.** The kernel stays a pure fused blob. Nothing in `pixelflow-core`,
`pixelflow-ir`, or the emitted code knows preemption exists.

**Negligible throughput cost.** A VM-exit/entry round trip is roughly 1–1.5k cycles. At a 10ms
preemption quantum that is noise. The kernel body itself runs at full native speed.

## 4. What it costs

**The platform matrix, which is the disqualifying one.** KVM is Linux plus virtualization
extensions. `pixelflow-runtime` targets macOS (Cocoa — would need Hypervisor.framework, a
different API with a different memory model), Linux/X11, and WASM (which cannot host a guest at
all). A mechanism that exists on one of three targets cannot be a *scheduling primitive* —
anything above it would have to be written twice, and the green tier's semantics would differ
by platform. This is what keeps it out of `actor-scheduler` and `pixelflow-core`.

**A second emitter target.** The guest blob is not the current JIT output. It needs a
freestanding ABI: no libc, no syscalls, its own stack set up by the loader, a fixed entry, and
`hlt` to signal completion. That is real work in `pixelflow-ir`'s emitter, and a second code
path to keep correct against the ISA matrix.

**Deployment friction.** `/dev/kvm` access means `kvm` group membership. A terminal emulator
that needs a virtualization device node to render is a hard sell, and nested virtualization
(running inside a VM or container) may not be available.

**Debuggability.** A guest-mode fault is not a normal Rust panic with a backtrace. Diagnosing
a miscompiled kernel through `KVM_EXIT_*` reason codes and register dumps is materially worse
than the current story.

## 5. Prototype notes — traps found the hard way

A prototype was drafted (by Gemini) and does **not** work as written. Recording the failures
because they are the traps anyone building this will hit:

**Real mode vs. long mode.** The prototype's loop body is `0x48, 0xff, 0xc0`, intended as
`inc %rax`. `0x48` is a REX.W prefix *only in long mode*. The setup never establishes long mode
— no paging, no `cr3`, no `EFER.LMA`, no 64-bit code segment — so the guest runs in 16-bit real
mode, where `0x40–0x4F` are the single-byte inc/dec encodings. The blob decodes as
`dec ax; inc ax; jmp -5`: net zero per iteration. The reported result (`RAX: 42183921`,
presented as an iteration count) is impossible; the run would abort on its own final assert.
**A guest that is going to run JIT'd 64-bit SIMD code needs the full long-mode bringup** —
page tables, `cr0.PG`, `cr4.PAE`, `EFER.LME/LMA`, a 64-bit CS descriptor — which is most of the
real work and is exactly what the prototype skipped.

**The lost-signal race.** The prototype arms a one-shot `setitimer` and installs a plain
handler. If the signal is delivered in the window after the kernel commits to entering guest
mode but before `KVM_RUN` is actually in the guest, it is lost — and with a one-shot timer and
an infinite-loop guest, the process hangs forever. This is the *same species* as the doorbell
lost-wakeup fixed in `actor-scheduler` (see `doorbell.rs`): a wake asserted against a sleeper
that has not yet committed to sleeping. KVM's answer is the atomic-unblock-on-entry pattern —
`KVM_SET_SIGNAL_MASK` so the signal is blocked everywhere except inside the guest run (the
`pselect` trick), or setting `run->immediate_exit` from the handler. **Do not build this without
one of those.**

**`OwnedFd` from an unchecked ioctl.** `OwnedFd::from_raw_fd(ioctl(...))` wraps the return
value before checking it, so a failure produces an `OwnedFd(-1)` whose `Drop` calls
`close(-1)`. Check, then wrap.

## 6. Verdict

Not adopted. The idea is sound and the resumability argument is genuinely unique, but the
value is niche and the platform cost is structural, not incidental.

If it is ever built, the shape is: **a feature-gated, Linux-only experimental backend in its own
crate**, parallel to the display drivers — never a dependency of `pixelflow-core` or
`actor-scheduler`. It slots into the mealy design's §5 placement table as a refinement of the
existing "own process" row (`SIGSTOP`-pausable, fault-isolated) with hardware preemption and
sub-process granularity.

The case that would justify revisiting it: **executing kernels whose abort-safety cannot be
required** — untrusted or externally-supplied expressions, or unbounded computations where
discarding partial work is unacceptable. That is precisely the gap the companion design cannot
close, because the companion buys its cheapness by *throwing partial work away*.
