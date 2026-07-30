# Design Doc: Actor Scheduler — Backpressure Is a Protocol, Not a Politeness

## Metadata
- **Author**: jppittman (with Claude)
- **Status**: Accepted — describes the runtime as built
- **Created**: 2026-07-30
- **Related**: `docs/designs/actor-scheduler-mealy-transducer.md`

---

## 0. Why this document exists

The send path looks like it is made of local decisions. It is not. Every question of
the form *"this queue is full — what should **this** caller do?"* has a local answer that
is wrong, because the property being protected is global. Readers who reason one call
site at a time — humans in a hurry, and language models nearly always — reliably invent
a local fix (drop it, spin on it, wait forever on it) that is individually plausible and
collectively destroys the thing that makes the system work.

So the invariant is written here, once, in full:

> **Every sender backs off, on the same ladder. The properties below hold only because
> that is true of everyone. A sender that opts out does not merely misbehave — it
> revokes the guarantee for every other sender in the system.**

---

## 1. The mechanism

`send_with_backoff` (`actor-scheduler/src/lib.rs`) is the one send path, and it is a
ladder with three rungs, tuned in `SchedulerParams::DEFAULT`:

| Rung | Default | What it is for |
|---|---|---|
| Spin | 458 attempts | Contention measured in nanoseconds; cheaper than a context switch |
| Yield | 191 attempts | Hand the CPU to the consumer, who may be runnable right now |
| Sleep | exponential from 2.11ms, jittered 73–98%, until 2.97s | Real contention; step aside and let the queue drain |

Exhausting the ladder returns `SendError::Timeout`. The total cascade is bounded at
~8.6s, and that number is a tuned domain constraint, not an accident — the optimizer was
penalized for exceeding a 12s degradation window, and separately for a `min_backoff`
above one frame at 155fps, because a sender that sleeps through a frame is a dropped
frame.

Jitter is part of the protocol, not decoration: without it, N senders woken by the same
drain retry in lockstep and collide again. 25% spread is the thundering-herd fix.

---

## 2. Three properties, and all of them are consequences of symmetry

### 2.1 The heaviest sender is the one that gets timed out

Not by accounting — nothing counts messages per sender — but by construction. A sender
that transmits constantly meets a full queue constantly, and each meeting climbs *its
own* ladder. A sender that transmits rarely almost always finds space, because the space
was freed while the heavy sender was asleep. Under sustained overload the heavy sender
is the one still climbing when the ladder ends.

The penalty is therefore proportional to how hard you push, self-administered, and needs
no arbiter, no quota, no priority table, and no accounting state anywhere. This is the
same shape as Ethernet's collision backoff and TCP's AIMD, and it is why fairness
survives as a *measured* metric in the tuning results (100%, maintained) rather than an
aspiration.

### 2.2 The sleeping is productive — it is scheduling, not waiting

While a heavy sender sleeps, two things happen that could not otherwise:

1. The **consumer** gets uncontended access to its own queue and drains it.
2. **Light senders** find the space that drain created, and get through immediately.

A backing-off sender is not idle. It has yielded its slot to the two parties that need
it. Backoff is how the send path schedules access to a contended queue, and the sleep is
the mechanism, not the cost of the mechanism.

### 2.3 A blown ladder is a diagnosis, not backpressure

~8.6s is not "how long we are willing to wait." It is *longer than any live consumer
could plausibly take* — at roughly 1µs per message, a 118-deep control ring drains in
microseconds; even a heavily loaded actor is three orders of magnitude inside the
window. So exhausting the ladder does not mean "busy." It means **the actor on the other
end is not running.** It is wedged, deadlocked, or dead.

That is a liveness assertion failing, and the response to a failed assertion is to
crash, loudly, at the point of discovery. There is no supervisor to escalate to (see
§4), no restart that could help a peer whose state is already unknown, and nothing
useful to do with a message addressed to something that no longer exists.

---

## 3. What defection costs, concretely

Three local "fixes" break §2, and each has been proposed, written, and reverted at least
once in this codebase's history:

**Shedding** (`try_send`, and on `Full`, drop it). The shedder never sleeps, so it never
creates the window in §2.2 — light senders starve behind a producer that has stopped
participating. It also never climbs a ladder, so it can never be the one timed out in
§2.1: the heaviest sender becomes the *only* one immune to the penalty aimed at it. And
because the loss is silent, §2.3's diagnosis never fires — a wedged consumer presents as
a slow one, forever.

**Spinning** (`try_send` in a retry loop). Worse on every axis: it holds the CPU the
consumer needs to drain, so it actively prevents the recovery it is waiting for, and it
still never times out.

**Unbounded blocking** (retry until success). Preserves fairness but discards §2.3
entirely: a wedged peer becomes a wedged system with no diagnosis and no crash, which is
the failure mode hardest to debug from a core dump.

The rule that falls out, and the reason it is a rule rather than a guideline:

> `send` is the only way an actor may push into another actor's inbox. `try_send` exists
> solely so `Wiring::flush` can answer "not now" to the runtime — which parks the outbox
> and stops stepping that actor, participating in backpressure by a different mechanism
> (see §5). Every other use of `try_send` is defection.

---

## 4. Why panicking deletes work rather than creating it

Enforcing invariants by crashing is usually presented as a tradeoff — safety bought with
brittleness. Here it is the opposite, because of what it makes *unnecessary*.

Every degraded-mode path is code that exists to survive a state that should not occur:
retry policies for peers that are gone, fallbacks for queues that cannot drain,
reporting channels for failures nobody can act on, supervision hierarchies to receive
that reporting, restart logic to respond to it, and — because a restarted actor's peers
now hold stale handles — reconnection logic beneath all of it. Each layer exists to
service the layer above, and the whole stack is rooted in the decision not to crash.

Refuse that decision and the stack has nothing to stand on. It is not that these paths
become simpler; they become *unreachable*, and unreachable code can be deleted. The
supervision machinery this codebase carried for months — `Exit::Failed`,
`HandlerError::Recoverable`, `Supervision`/`Stuck`/`NodeId`, `Host::exits()` — was
~900 lines servicing a supervisor that was never built, and could not have worked if it
had been: every build profile sets `panic = "abort"`, so nothing unwinds, nothing
restarts, nothing resumes. The types described a capability the runtime does not have.

There is also a review-economics argument, and it is not a small one. Questions of the
form *"what happens if this half-fails?"* are unbounded — there is always another
interleaving, another partial state, another peer that might vanish mid-operation. A
design that answers them all with **"it panics"** converts an open-ended class of review
findings into a closed one. The remaining questions are about behavior the system
actually has.

---

## 5. The green tier participates by a different mechanism, for a reason

`Wiring::flush` may not block, and this is not an exception to §1 — it is the same
principle applied where blocking is impossible.

A `flush` runs inside `Node::poll`, inside `Host::sweep`, on a thread shared by N
co-located green actors. A blocking flush freezes all of them: one slow consumer stalls
actors that have nothing to do with it, and co-location — the green tier's whole measured
advantage — becomes its liability.

So a green actor exerts backpressure by *declining*: `send_port` returns `Flush::Blocked`,
the undelivered word stays in the node's outbox, and the scheduler **stops stepping that
actor** until it drains. The producer is suspended exactly as a backing-off sender is,
but the suspension is scheduler state rather than a sleeping thread. §2.2's property
holds — the actor has yielded its slot — with none of the thread cost.

This is why the parked outbox exists at all, and why `Flush::Blocked`, `Node::outbox`,
the outbox-first gate, and `Step::Blocked` are all one mechanism. Make `flush` block and
every one of them becomes dead code, replaced by a thread-blocking primitive that
violates the tier's defining constraint.

---

## 6. Open question: should the timeout panic in the crate?

§2.3 says a blown ladder means the peer is dead, and §4 says dead means panic. But
`send_with_backoff` returns `SendError::Timeout` and leaves the panic to the caller,
which every production caller then performs — identically. That is caller-chosen policy
for a condition the crate has already diagnosed, and §3's rule says the crate owns send
policy.

Moving the panic inside would leave `SendError` with only `Disconnected`, which is a
genuinely different case: a disconnected peer during a shutdown cascade is ordinary, not
a failure. Unresolved, and deliberately recorded rather than quietly settled.

---

## 7. Paths not taken

- **Per-sender quotas / accounting.** §2.1 gets proportional fairness with no state.
  Counting would add exactly the bookkeeping the backoff ladder makes unnecessary.
- **Priority inheritance across senders.** The three lanes already encode priority where
  it belongs — in the message, not the sender.
- **`Delivery::Droppable`.** The crate's own shedding policy: zero production users
  across its entire lifetime, while the one place production actually shed hand-rolled it
  through `try_send`, bypassing the abstraction meant to govern it. Removed; §3 explains
  why no replacement is wanted.
- **Growing the queue on overload.** There is no such thing as an unbounded queue, only a
  queue whose bound you have not written down. Growth converts a fast, loud §2.3
  diagnosis into a slow memory leak.
