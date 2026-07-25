# Design Doc: pixelflow-runtime Engine — from Mediator to Mesh

## Metadata
- **Author**: jppittman (with Claude)
- **Status**: Draft — plan + topology proof only, no live actor code changed
- **Created**: 2026-07-25
- **Builds on**: `docs/designs/actor-scheduler-mealy-transducer.md` §9 (the audit)

---

## 1. Goal

Delete `EngineHandler` as a central mediator. Today `driver`, `vsync`, `rasterizer`, and `app`
all talk *through* engine; the target is that they talk to **each other** directly, over edges
the `Topology` validator checks at bootstrap. Scoped as its own body of work, separate from the
actor-scheduler primitives it depends on (§9 of the Mealy doc): those primitives (`Node`,
`ports!`, `Credit`, `Topology`, priority lanes) are now in place. This doc is the plan for
spending them on the real system.

## 2. Why this doc exists before any engine code changes

This is production code driving a real, running terminal renderer, not a prototype. The audit
(§9) found four real bidirectional cycles held together today by an untyped global atomic and a
hand-rolled `stale` flag. Reclassifying each edge is a per-edge *judgment call* about what may be
lost — the audit was explicit that the type system cannot make this call automatically
(`PresentComplete` dropped is catastrophic; `AppData::RenderSurface` dropped is nothing, the code
already treats it that way). Getting one of these wrong is worse than the status quo, not better.
So: classify every edge on paper first, prove the resulting graph is actually a DAG (mechanically,
not by re-reading it), and only then touch a live actor.

## 3. The target topology

| Edge | Direction | Kind | Why |
|------|-----------|------|-----|
| Input + window-lifecycle events (`DisplayEvent` minus `PasteData`: key, mouse, scroll, focus, resize, close, scale, window created/destroyed, clipboard-data-requested) | driver → engine | **Blocking** | Discrete, non-idempotent events. None may be lost, and none coalesce with another. |
| `Present` | engine → driver | **Credit(1) + Droppable backstop** | Not a judgment call this doc is inventing: `DisplayData::Present`'s own doc comment already specifies the receiver's contract as "3. NOT block the sender (use backpressure if buffer full)". The original table lumped it into a generic "commands, never lose" bucket and missed that the code already disagrees. Exactly one window is ever in flight (the ping-pong buffer), so this is the *same* structurally-unreachable shape as the row below, not a new risk. |
| `PresentComplete` (window return) | driver → engine | **Credit(1) + Droppable backstop** | The reply half of `Present`, sharing its credit. Exactly one window is ever in flight, so the reply ring's capacity is provisioned ≥ 1 and can **never actually be full** when the reply lands — the droppable backstop exists for `Topology`'s proof, but is unreachable by construction, not merely rare. Not the same case as a true "acceptable loss" droppable edge (§9.2) — safe only because the credit bound is airtight. |
| `SetTitle` / `SetSize` / `SetCursor` / `Copy` | engine → driver | **Droppable, no credit needed** | Idempotent, "latest write wins" state — the exact shape already handled for `pending_manifold` below. Two queued `SetTitle`s only ever needed the second one anyway. |
| `RequestPaste` | engine → driver | **Droppable, no credit** — *revised, see §3.2* | Originally planned as Credit(1); that would deadlock paste permanently. |
| `PasteData` (reply to `RequestPaste`) | driver → engine | **Droppable, no credit** — *revised, see §3.2* | The reply is not merely losable, it is *routinely absent*: pasting from an empty clipboard produces no reply at all, by design. A credit released only by the reply would never come back. |
| `Create` (window creation) | engine → driver | **Not part of the steady-state graph — see §3.1** | One-shot bootstrap per window, not a member of the continuously-live mesh at all. |
| VSync tick | vsync → engine | **Credit(100) + Droppable backstop** | Direct replacement for `VSYNC_TOKEN_BUCKET`. A dropped tick under a bug is a missed/late frame, genuinely tolerable — this *is* an ordinary acceptable-loss droppable edge, unlike the row above. |
| Frame-rendered notice (`RenderedResponse`, FPS tracking only) | engine → vsync | **Droppable, no credit needed** | Pure telemetry. Losing a sample changes nothing but a displayed FPS number. |
| Render request | engine → rasterizer | **Credit(1) + Droppable backstop** | Mirrors the window-return case: `pending_render` tracks exactly one outstanding render, so ring capacity ≥ 1 makes the backstop unreachable by construction. |
| Render complete | rasterizer → engine | **Credit(1) + Droppable backstop** (same credit as above; it's the reply half) | Same reasoning as `PresentComplete`. If this bound is ever wrong (more than 1 outstanding), the failure mode changes from "spin-forever under adversarial timing" (today) to "one dropped frame, `pending_render` never clears, next resize/vsync recovers" — recorded as an explicit design trade-off, not an oversight. |
| Frame request (`RequestFrame`) | engine → app | **Credit(N) + Droppable backstop** | N-outstanding-requests bound; a dropped request is recovered by the next vsync tick. |
| Manifold submission (`AppData::RenderSurface`) | app → engine | **Droppable, no credit needed** | The *existing* code already implements "always keep the most recent, drop old ones" for `pending_manifold` by hand. This edge doesn't need `Credit` — it needs to be declared `[drop]` and the hand-rolled staleness logic deleted, because the port *is* the staleness policy. |

Every reply edge above is the closing edge of a cycle; every one is legal under the DAG rule
(§3.1 of the Mealy doc) specifically because it's Droppable, whether or not `Credit` additionally
makes that backstop unreachable by construction. **Two different reasons an edge is safe to mark
`[drop]`, and this doc keeps them distinct on purpose:**

- **Structurally unreachable** (window return, render complete): the credit bound is airtight,
  so the drop path is dead code in the well-behaved system and only fires on an actual bug.
- **Genuinely acceptable loss** (vsync tick, FPS notice, manifold submission): dropping is a
  normal, expected outcome, not a bug symptom.

Conflating these two would be exactly the mistake §9.2 warned against.

### 3.1 driver ↔ engine: resolved by decomposing, not by picking an excuse

`pixelflow-runtime/tests/target_topology.rs` was first written with `DisplayEvent` and "display
commands" each as one aggregate edge, both Blocking — and immediately proved that wrong.
`Topology` correctly rejects it: **two independent blocking edges between the same pair of
actors are a cycle regardless of whether the two message flows are logically unrelated.** driver
blocked pushing an event into a full engine inbox, at the same moment engine is blocked pushing a
command into a full driver inbox, is the identical shape to the app↔compiler example the Mealy
doc uses to motivate the DAG rule — "they're different conversations" does not save it.

The fix is not picking one lucky excuse to drop the whole aggregate — it's noticing "display
commands" was never one thing. Per-message, it decomposes entirely into edge kinds this doc had
already established for other actors:

- **`Present` already has a non-blocking contract in the code itself** (`DisplayData`'s own doc
  comment — see the table), which the original aggregate framing simply missed by lumping it in
  with "commands, must never be lost." It's the same Credit(1) shape as `PresentComplete`,
  because the ping-pong buffer means the two were always one relationship, not two.
- **`SetTitle`/`SetSize`/`SetCursor`/`Copy` are idempotent state pushes** — exactly the
  "keep latest, drop stale" shape `pending_manifold` already implements by hand elsewhere in
  this file. Plain `[drop]`, no credit needed, nothing new invented.
- **`RequestPaste`/`PasteData` is its own tiny 1:1 request/reply** — originally classified here
  as "structurally identical to every other credit-bounded pair, just lower-stakes." That was
  wrong, and §3.2 corrects it: the reply is *routinely absent*, so a credit released only by the
  reply is a deadlock, not a bound.
- **`Create` is a bootstrap-phase message**, not a steady-state one — the same category the
  *current* code already puts the rasterizer's handshake in (`engine_troupe.rs`: "Rasterizer is
  NOT in the troupe - it uses a bootstrap handshake pattern"). `Topology`'s DAG check governs the
  continuously-live mesh; a message sent once per window before that mesh is even fully up
  doesn't belong in the same graph, the same way the rasterizer handshake doesn't today.

With every engine → driver edge now either Credit+Droppable, plain Droppable, or bootstrap-only,
**driver → engine is the only edge left that's genuinely Blocking** — and a single blocking
direction between a pair is never a cycle by itself. Nothing that matters was made droppable to
get there: every discrete, non-idempotent, must-not-lose message (input, window lifecycle) stays
exactly as reliable as it is today.

### 3.2 The paste credit was a deadlock, not a weaker guarantee

§3.1 originally flagged `RequestPaste`/`PasteData` as carrying a *weaker* credit bound than its
siblings — "nobody issues a second `RequestPaste` before the first replies" — and accepted that
as a recorded trade-off. Implementation found that framing to be wrong in kind, not just in
degree, and this section is the correction.

**A reply is not merely rare to lose here. It is routinely, correctly absent.** Pasting from an
empty or unowned clipboard produces *no reply at all*, by design and at the protocol level. On
X11, `request_paste()` issues an async `XConvertSelection`; when nothing owns the selection the
server answers `SelectionNotify` with `property == None`, and
`platform/linux/events.rs::handle_selection_notify` correctly returns `None` rather than
inventing an empty `PasteData`. That is the normal, specified path for "clipboard is empty" —
not an error, not an edge case worth engineering around.

A `Credit(1)` released only by `PasteData` would therefore be consumed and never returned the
first time a user hits paste with an empty clipboard, and **every subsequent paste for the life
of the process would be silently gated off**. The bound's failure mode is not "a dropped paste";
it is "paste stops working permanently." That is strictly worse than the unbounded status quo.

The fix is to *subtract*, not to add a rescue mechanism. A timeout that releases the credit when
no reply arrives would work, but it answers a question this edge never needed to ask: `Credit`
exists to make a droppable edge's drop **unreachable**, and it earns its keep only where the
drop would be catastrophic (`PresentComplete` losing the sole window buffer). Here the drop is
*already* the acceptable outcome — §3.1's own words, "a dropped reply costs one retried
keystroke" — so there is nothing for a credit to protect. The edge is plain Droppable, exactly
like `SetTitle`/`SetSize`/`SetCursor`/`Copy`: lose it and the user presses paste again.

This is the two-category distinction from §3 doing its job in the direction it was meant to.
Those categories — *structurally unreachable* vs. *genuinely acceptable loss* — exist so that
"safe to `[drop]`" is never allowed to blur into one word. Paste was filed under the first and
belongs under the second; a credit bolted onto an acceptable-loss edge bought no safety and
introduced a hang.

Worth noting what did *not* change: `pixelflow-runtime/tests/target_topology.rs` needed no edit
to its edges for this correction, because `Credit` is a sender-side discipline and not a
`Topology` concept at all — both directions were already `droppable_edge`. The proof was
indifferent to a distinction the prose had gotten wrong, which is a useful reminder that the
mechanical check constrains the *shape* of the graph and never the judgment about what may be
lost.

## 4. Placement

- **driver**: `[main]` / dedicated thread. Unchanged — Cocoa/X11 requires it (`CLAUDE.md`:
  "Platform on main thread").
- **vsync**: dedicated thread. It already runs one (plus its own clock thread sending `Tick` as
  an ordinary message — this precedent already answers the Mealy doc's §9.4 open question about
  where timer ticks belong, and does not change here).
- **rasterizer**: dedicated thread(s) — `render_threads`/work-stealing is real (§9.4), the
  `Send`-bounded migratable-pool exception the Mealy doc's §5.2 carved out, not the owned-green
  default.
- **engine**: ceases to exist as a distinct actor. `EngineHandler` today isn't a pure router —
  it holds real coordination state (`window`, `pending_render`, `pending_manifold`,
  `frame_number`) and makes real decisions (stale-render discarding, catch-up rendering).
  Deleting it means that state and logic goes *somewhere*: split across the actor that
  naturally owns each piece (driver already owns window lifecycle; app already owns the
  manifold it produces), or a new, deliberately small coordinator that does only the glue
  `EngineHandler` cannot shed — **not decided here**. The edge table in §3 is written against
  "whatever ends up owning that state," under the working name `engine`, because every edge in
  it holds regardless of how that question resolves; resolving it is explicitly step 5, once
  the other four actors already speak the new protocol and it's clear what's actually left to
  coordinate.
- **app**: unchanged placement (a real actor the embedding crate, e.g. `core-term`, owns) — only
  the wire types at the boundary matter here, and they're already concrete within
  `pixelflow-runtime` (§9.4 of the Mealy doc already found no gap here).

## 5. Rollout order

Big-bang replacement of a live, running actor system is the risk this whole doc exists to avoid.
Order of attack, each step independently shippable and reviewable:

1. **This doc + the topology proof** (§6) — no live code changes. *Landed.*
2. **`vsync` first**: smallest actor, already isolated, already had the credit-shaped token
   bucket crying out to become real `Credit`. *Landed, in two slices:*
   - **2a.** A real-`VsyncActor` regression harness, before touching any production code —
     `vsync_actor_tests.rs` admitted outright it only ever tested a hand-rolled mock, never
     the actual implementation. Writing it surfaced that the bucket is one process-wide
     global (a real cross-test coupling bug, not flakiness) and that `RenderedResponse` never
     actually returned a token — the return happened via a raw global-static call from
     `engine_troupe.rs`, bypassing vsync's message interface entirely.
   - **2b.** `VsyncActor`'s decision logic extracted into `VsyncCore`, a real `Transducer` —
     pure, table-tested, no thread/clock/scheduler in the loop. `VSYNC_TOKEN_BUCKET` replaced
     by a `Credit` field; the raw global mutation replaced by a real message
     (`VsyncCommand::ReturnToken`), which is what the cross-actor state-poking actually
     required once it could no longer reach into a shared global. `VsyncActor` itself stays
     on the old `Actor`/`ActorScheduler` shell — a thin adapter translating `VsyncCore`'s
     output into the one real send it still makes — so `EngineHandler`'s field types don't
     change at all; only `engine_troupe.rs`'s two `return_vsync_token()` call sites become
     message sends. Not yet on a real `Node`/`Host`; that migration is deferred until the
     adapter pattern earns its keep on more than one actor. The step-2a harness, run
     unmodified against the refactored code, passed without a single assertion changing —
     proof nothing observable broke — plus one new capability it can now check for the first
     time: `ReturnToken` genuinely unblocking a tick, previously untestable.
3. **`rasterizer`**: same shape as vsync (request/reply, already isolated via its bootstrap
   handshake — which was itself a symptom of not fitting the old static-topology model, so this
   should *simplify*, not complicate).
4. **`driver`**: convert the command/event edges; window-return `Credit` last, since it's the
   one edge where getting the bound wrong is worst. **See §5.1 — the shape of this step is not
   what steps 2–3 assumed.**
5. **`app`-facing edge + delete `EngineHandler`**: once every satellite speaks the new
   protocol directly to its real peer, the mediator has nothing left to route and is deleted,
   not refactored down.

Each of steps 2–5 lands as its own PR with its own before/after behavioral tests against the
running terminal, not just unit tests of the actor in isolation.

### 5.1 Step 4 extracts the *send*, not a decision core

Steps 2 and 3 both worked the same way: find the decision logic tangled up with I/O, pull it out
into a pure `Transducer` (`VsyncCore`, `RasterCore`), leave a thin adapter holding the channels.
Applying that template to the driver looks, at first, like it fails — `DriverActor` is a pure
delegation shell with no state and no decisions, and the real logic lives in `PlatformOps` impls
doing raw X11 and Cocoa calls, which will never be a pure transducer over a `*mut xlib::Display`.

That reads as "step 4 is harder." It is the opposite, and the reason is worth writing down:
**there is no decision core to extract because the platform state is already properly
encapsulated.** `LinuxOps` holds `window: Option<X11Window>` and a waker; `MetalOps` holds
`app`, `windows`, `window_map`. That is all OS resource, already owned by exactly the thing that
should own it. Nothing wants moving.

What the Mealy conversion actually asks for here is the *other* half of its rule — **effects are
return values, not calls** — and the effect in question is the one thing these types do that
couples them to another actor: `engine_handle.send(...)`. That is the whole of step 4.

The sites are few and mechanical:

| Impl | Site | Emits |
|---|---|---|
| `LinuxOps` | `handle_data` (Present) | `PresentComplete(window)` |
| `LinuxOps` | `handle_management` (Create) | `FromDriver(WindowCreated)` |
| `LinuxOps` | `park` — event pump | `FromDriver(event)` |
| `LinuxOps` | `park` — drain loop | `FromDriver(event)` × N |
| `MetalOps` | 7 sites, same two shapes | — |

So: `PlatformOps`' methods stop sending and start yielding their outbound events, and
`PlatformActor` — which **already exists as exactly this adapter**, wrapping `PlatformOps` into
an `Actor` — performs the sends. Identical in shape to steps 2 and 3, with less work, because
the encapsulation those steps had to create is already present here.

The payoff is concrete and testable: `EngineActorHandle` disappears from `PlatformOps` entirely.
Today every platform impl holds a live handle to the engine and can only be exercised with one
running; afterwards they return `DisplayEvent`s and can be tested with no engine at all.

Two real constraints on the implementation, neither a blocker:

- **`park` emits *N* events**, not zero-or-one — the drain loop pulls every pending X11 event.
  So the output is a sequence, unlike `VsyncCoreOut`/`RasterCoreOut`'s `Option`. Since `park`
  runs every frame, this must not allocate per call (`CLAUDE.md`: "no per-frame heap
  allocation"); the adapter owns one reusable buffer that the ops push into, so allocation is
  amortized to zero in steady state rather than paid per frame.
- **Only two impls exist** (`LinuxOps`, `MetalOps`) — headless and web still use the older
  `DisplayDriver` trait and are untouched by this. Both are covered by CI (ubuntu + macOS), but
  only the X11 half compiles on a Linux dev box, so the macOS half is verified in CI rather than
  locally. That is the reason to land this as one deliberate PR rather than piecemeal: a
  half-converted `PlatformOps` trait breaks the platform that cannot be compiled locally.

---

## 6. Proof: the target topology is a DAG

Before any of §5's steps touch a real actor, the target graph above is checked mechanically —
`pixelflow-runtime/tests/target_topology.rs` builds exactly the nine edges in the table with
their declared kinds and asserts `Topology::validate()` succeeds. This is deliberately built
from the *same* five actor names and *same* edge classification this doc commits to, so a
future change to either has to touch both or the proof drifts from the plan it's supposed to
prove.
