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
  That does *not* make `Out` a list: it stays one output word, `DriverOut { events: Vec<..> }`,
  exactly like `VsyncCoreOut { tick: Option<..> }` and `RasterCoreOut { response: Option<..> }`.
  The struct's *fields* are the ports; a step still returns one `Out`. Keeping that uniform
  matters more than it looks — `Out` being sometimes-a-word and sometimes-a-sequence is the kind
  of special case that later has to be handled everywhere `Out` is touched.

  This also disposes of an allocation concern that turned out not to exist. An earlier draft of
  this section had the adapter own a reusable buffer for the ops to push into, on the grounds
  that `park` runs every frame and `CLAUDE.md` forbids per-frame heap allocation. But an empty
  `Vec` never touches the heap — `Vec::new()` allocates on first push, not on construction —
  and the overwhelming majority of frames have no input events at all. Those frames allocate
  nothing. A frame where the user actually types or moves the mouse pays one small amortized
  growth, which is not a per-frame cost. The sink was machinery to solve a problem that
  measurement of the type's own semantics dissolves, and it is deleted before being written.
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

---

## 7. Step 5: where `EngineHandler`'s state goes

§3 deliberately left this open — the mediator holds real state, so deleting it means that state
moves somewhere, and deciding before the satellites spoke the new protocol would have been
architecture on paper. Steps 2–4 have landed, so this is now answerable from the code.

The useful discovery is that **most of it isn't coordination state at all.** Two of the five
fields dissolve rather than relocate.

### 7.1 `pending_render` is half of a torn `Window`

`trigger_render_with_window` destructures `Window { id, frame, width_px, height_px, scale }`,
sends **only `frame`** to the rasterizer, and stashes the remaining four fields in
`pending_render` so `RenderComplete` can reassemble a `Window` from the cooked frame plus the
metadata. The field exists solely because a value is split apart and put back together across an
actor boundary.

It does not need an owner. It needs to not be created: if the request carries the metadata and
the response returns it untouched, there is nothing to stash. `RenderRequest`/`RenderResponse`
live in `pixelflow-graphics`, which must stay runtime-agnostic and so cannot name `Window` — but
it does not have to. An opaque round-trip payload (`RenderRequest<P, Meta>` →
`RenderResponse<P, Meta>`, `Meta` passed through untouched) keeps the rasterizer a pure
`(manifold, frame, meta) -> (frame, render_time, meta)` and lets the runtime put a `Window`'s
metadata in it. `pending_render` is then deleted, not moved.

### 7.2 `stale` is a comparison, not a flag

`stale` is set by the resize handler reaching **into** an in-flight `pending_render` to mutate
it, and read once at completion. With metadata riding along, completion can compare the returned
dimensions against the current window's directly: they differ ⟺ a resize happened mid-render.
Same decision, no mutable flag, and no cross-message reach-in — which is the same class of
coupling as the `VSYNC_TOKEN_BUCKET` global that step 2 removed, just local instead of static.

### 7.3 The rest

| Field | Goes to | Why |
|---|---|---|
| `pending_manifold` | the app → rasterizer edge, as a droppable keep-latest port | §3 already called this: the *port is* the staleness policy, so the hand-rolled "keep newest, drop old" logic is deleted along with the field. |
| `window` | **driver** | It already creates the window (`WindowCreated`) and presents it. With the mediator gone the buffer circulates driver → rasterizer → driver; the driver is the only actor that outlives every stage of that loop. |
| `frame_number` | **rasterizer** | It is a count of completed renders, and its only consumer is the FPS telemetry edge to vsync. It belongs to the thing doing the counting. |
| `render_threads` | **bootstrap config, needs a real owner** | *Corrected during implementation.* This row first claimed the field was a redundant copy of `RasterCore::num_threads`, deletable outright. It isn't: the engine passes it to `spawn_with_setup` at bootstrap, so it is the *source* of the rasterizer's value, not a duplicate of it. It has to be supplied by whoever spawns the rasterizer once the engine no longer does — small, but a genuine open question rather than a deletion. |
| `driver` / `vsync` / `rasterizer` / `app_handle` / `self_handle` / `rasterizer_forward_handle` | topology edges | Handles are what a mediator is made of. They are the thing being deleted, not state needing a home. |

So `EngineHandler` collapses to: two fields deleted outright (`pending_render`, `render_threads`),
one flag replaced by a comparison (`stale`), one absorbed into a port's semantics
(`pending_manifold`), and two genuine relocations (`window` → driver, `frame_number` →
rasterizer). Nothing needs a new home invented for it, which is the sign the mediator was
holding state on behalf of actors that should have held it themselves.

### 7.4 Order

The metadata round-trip (7.1/7.2) is a `pixelflow-graphics` change and lands first, on its own —
it is independently correct, deletes `pending_render` and `stale` while `EngineHandler` still
exists, and shrinks the surface the collapse has to move. Only then do `window` and
`frame_number` relocate and the mediator go away.

---

## 8. The render pipeline's actual edges — proven before wiring anything

§7 decided where `EngineHandler`'s *state* goes. It didn't decide who performs the one
remaining piece of logic that isn't state: matching "a window is free" against "a manifold is
waiting" to decide when to render. That decision, and the edges it implies, needed the same
treatment every edge in §3 got — proven before any live actor changes — because it invents two
edges (`driver` ↔ `rasterizer`) that never existed in the mesh before.

**The match logic lives in `rasterizer`, and the window is pulled rather than pushed** (§8.5
explains why). `app` pushes its latest manifold (droppable, keep-latest, same semantics as
today). `rasterizer` holds that manifold and a `Credit(1)`; when it has work and the credit
allows, it *requests* a window. `driver` keeps the window between frames, stays its allocator,
and grants the one it already holds — always correctly sized, because the driver resized it in
place when the OS said so. `rasterizer` renders into it and `Present`s it back; the driver
presents and retains it for the next request.

The resulting edges, added to `pixelflow-runtime/tests/target_topology.rs`:

| Edge | Direction | Kind |
|---|---|---|
| Window request ("I have work, give me a buffer") | rasterizer → driver | Droppable — genuinely losable, carries nothing |
| Window grant (correctly sized, driver-allocated) | driver → rasterizer | Droppable, **unreachable**: driver can't grant what it doesn't hold |
| `Present` | rasterizer → driver | Droppable, **unreachable**: coordinator can't present what it doesn't hold |
| Manifold submission | app → rasterizer | Droppable, keep-latest |
| Frame-rendered notice (FPS telemetry) | rasterizer → vsync | Droppable, no credit |
| `ReturnToken` | app → vsync | Droppable, no credit |

**There is no `Credit` on this loop.** Ownership of the single `Window` is already the bound —
see §8.6. Rust will not let an actor send a window it does not hold, so "at most one in flight"
is a property of the type rather than a counter that has to be kept in sync with it.

Two of these delete an existing edge rather than add one: `engine → driver` (`Present`) and
`driver → engine` (`PresentComplete`) are gone — `rasterizer` presents directly now. `engine ↔
rasterizer` and `engine → vsync` (`RenderedResponse`) are gone entirely; nothing uses that pair
any more. `app → engine` (`AppData::RenderSurface`) is also gone: the message that used to
double as both "here is the manifold" and "return my vsync token" now travels as two direct
sends, to the two actors that actually need each half.

`ReturnToken` closes a small pre-existing gap in this proof: it was already an `engine → vsync`
Control-lane send, but never modeled as its own edge — modeled now, since it travels a new edge
anyway.

`the_target_engine_mesh_is_a_dag` is **updated in place**, not duplicated, matching this file's
own stated purpose of staying valid across the rollout. A new `regressed_render_pipeline_
framing_is_cyclic` test records the shape of mistake most likely to recur here: treating "only
one window ever circulates" as license to make the driver ↔ rasterizer hand-off synchronous in
both directions. It's the identical error §3.1 caught for driver ↔ engine, just with a fresh
pair of actors — two blocking edges between the same pair is a cycle regardless of whether the
conversations are logically unrelated.

### 8.1 Explicitly out of scope for this slice

`engine` remains a node in the proven graph — this is a render-pipeline-only cut, not the full
deletion. Left alone, and routed through `engine` exactly as today:

- Input-event forwarding (`driver → engine`, Blocking).
- `AppManagement` commands — `SetTitle`, `SetSize`, `SetCursor`, `Copy`, `RequestPaste`,
  `CreateWindow` — and their replies (`PasteData`).
- The vsync-tick → `RequestFrame` relay (`vsync → engine → app`).

Moving these is a separate, later slice: none of them touch the state this doc's §7 already
redistributed, and folding them in here would make one change cover two independent decisions.

### 8.2 What implementing this actually requires

Wiring these edges for real is a larger change than §7's field deletions were, and is scoped as
its own follow-up rather than bundled with the proof:

- `rasterizer`'s bootstrap (today `EngineHandler::spawn_rasterizer`) moves out of
  `EngineHandler` entirely — the rasterizer becomes a directly-wired mesh participant with its
  own initialization, not something the mediator stands up on the mediator's behalf.
- A new piece of runtime-side state holds the latest manifold and the render `Credit(1)`, and
  performs the point-space → pixel-space dimap warp that `trigger_render_with_window` does today
  — using the dimensions of whichever window it was granted, since under §8.5 it holds no window
  of its own between frames and never allocates one. This
  logic stays in `pixelflow-runtime`, not `pixelflow-graphics`, since it's runtime coordinate
  mapping, not general rasterization (`CLAUDE.md`: no terminal-adjacent logic in the graphics
  crate).
- `app`'s handle setup gains a second target: today `Application` only reaches `engine`; it
  needs a way to reach `rasterizer` (manifold) and `vsync` (`ReturnToken`) directly.

### 8.4 What the `rasterizer` node actually is

The proof's `rasterizer` node is **not** `pixelflow-graphics`'s `RasterizerActor`. It can't be:
that crate is deliberately runtime-agnostic and names no runtime concept — no `Window`, no
`driver`, no `vsync`, no `Application` — and §8.2 already places the window/manifold pairing and
the point→pixel dimap warp in `pixelflow-runtime`. Left implicit, this reads as "driver sends
windows straight to `RasterizerActor`," which is not implementable and not what was proven.

So `rasterizer` in the graph is a **runtime-side coordinator**: it holds the latest manifold and
the render `Credit(1)`; it does the coordinate warp; it owns the graphics `RasterizerActor` as a
worker behind it, via the same bootstrap handshake used today. It does *not* hold a window
between frames — §8.5 explains why that would put the allocator on the wrong side.

Collapsing those two into one node is sound rather than convenient, and the reason is worth
stating because it's the property that must not silently change: **`RasterizerActor` is a leaf.**
Its only outbound channel is `response_tx`, back to whoever registered it — it holds no handle to
any other actor and initiates contact with nobody. A node whose sole peer is its caller, with a
droppable reply, cannot participate in a cycle with the rest of the graph, so it cannot affect
the DAG result either way.

The moment that stops being true — if the worker ever gains a handle to a third actor — it stops
being collapsible and has to enter the proof as its own node. That is the tripwire.

### 8.5 The window is pulled, not pushed — and why the first attempts failed

This section replaces two earlier ones (a "tear the Window, send metadata to engine and the
buffer to rasterizer" split, and a follow-up "resize must not mint a window" patch). Both are
deleted rather than annotated, because keeping a chain of superseded fixes would obscure what
turned out to be a single wrong decision underneath all three.

**What went wrong.** Automated review caught a run of defects that all turned out to share one
cause: a `rasterizer → app` edge invented to dodge a frame-buffer clone that was never an
acceptable option; resize windows sent over a droppable edge, where a full ring discards the
*new* message while keep-latest requires discarding the *old* one, losing the only
correctly-sized buffer with no replacement until the next resize; and then a fix for *that* which
left the coordinator responsible for reallocating on resize without any edge over which to learn
the dimensions had changed — where routing that metadata over a droppable edge would have
reproduced the previous defect exactly.

Each patch exposing the next is a decomposition error rather than a detail error. **The root
cause: frame allocation was on the coordinator, but window size is driver-authoritative** — the
OS decides it. Putting the allocator on the side that doesn't know the size forces size
information to chase it across the mesh, and every route it can take is either unreliable
(droppable, loses state permanently) or blocks the main thread.

**The fix is to move allocation, not to route information better.** The driver keeps the window
between frames and remains its allocator. The coordinator *pulls* one when it actually has work:

1. `rasterizer → driver`: window request, sent when a manifold is held and `Credit(1)` allows.
   **The credit is consumed here and held for the whole round trip.**
2. `driver → rasterizer`: the window it already holds — always correctly sized, because the
   driver resized it in place when the OS said so.
3. `rasterizer → driver`: `Present` once rendered. The driver presents it and retains it, ready
   for the next request. **The credit is released only here**, once the window is home.

**There is no resize notification anywhere in the render pipeline, so none can be dropped.** On
resize the driver swaps its held buffer for a correctly-sized one and tells nobody. The
coordinator never learns dimensions because it never allocates. The problem isn't solved, it's
absent.

**Why the credit must span the round trip, and why "recoverable" was the wrong justification.**

An earlier draft of this section argued the droppable edges were safe because a dropped message
costs one frame and *the sender still holds the authoritative copy to resend*. **That argument is
false**, and review was right to reject it: `Window` owns the framebuffer and is sent **by
value**. `send_port` destroys the message on a full droppable ring. So a dropped grant leaves the
driver with nothing — it moved the window out — and a dropped `Present` leaves nobody holding it.
The sole render buffer is destroyed, permanently, not delayed by a frame. There is exactly one
`Window`; no edge carrying it is ever safe on "recovery" grounds.

Safety here comes from the drop being **impossible**, which is §3's *structurally unreachable*
category — the argument this doc already established for `Present`/`PresentComplete` and which
the "recoverable" framing needlessly abandoned:

- The coordinator cannot request without consuming the single credit.
- The credit is not released until the window is back with the driver, so **at most one message
  carrying the window is in flight across all three edges at any instant.**
- Each ring is provisioned ≥ 1. A ring holding at most one message, sent only when the previous
  round trip completed, is never full at the moment of sending.

So the droppable backstop on all three edges is dead code in the well-behaved system, exactly as
§3 intends — present for `Topology`'s proof, unreachable by construction. Releasing the credit at
the *grant* instead would break this directly: a second manifold could trigger a request while
the window is still out being rendered, and the driver would have nothing to grant.

**Droppable is safe when the drop cannot occur, or when the message is genuinely replaceable.**
Never because a sender "could resend" something it moved away.

**Consequence for the coordinator.** It shrinks: manifold plus `Credit(1)`, no `latest_window`,
no reallocation logic, no dimension tracking. It still performs the point→pixel dimap warp, using
the dimensions of the window it was handed.

**Consequence for `Present`/`PresentComplete`.** `PresentComplete` disappears entirely as a
message. Previously the window made a full circuit — `Present` carried it to the driver and
`PresentComplete` carried it back to the engine, which was its resting owner between frames. Now
the **driver** is the resting owner, so `Present` *is* the return: it hands the window back to
the actor that already keeps it, and no separate acknowledgement is needed. The window still
moves by value on every hop; what's gone is the second hop.

That also retires an invariant §3 had been asserting but not holding: "exactly one window
circulates" was untrue while resize could mint a second one mid-flight. Now exactly one window
*exists*, and it is away from the driver only for the duration of a single render.

### 8.6 Ownership is the bound — delete the credit

§8.5 put a `Credit(1)` on the render loop, consumed at the request and released when `Present`
returned the window. Review showed that cannot work, and the reason is worth keeping:

**`Credit` is requester-local, and `send_port` reports `Flush::Done` whether it delivered or
dropped.** So the coordinator can never learn that its `Present` arrived. Release the credit when
*emitting* `Present` and a dropped delivery restores the credit while destroying the only window
— which makes the drop-impossibility argument circular, since it assumed the credit bounded the
edge. Never release it and rendering halts after one frame. With a coordinator-local credit and
no acknowledgement, there is no third option.

The available fixes were to reinstate a `PresentComplete` acknowledgement, or invent a
transferable permit. Both are machinery. **The subtraction is to notice the credit was modelling
something the type system already guarantees: you cannot send a `Window` you do not hold.**

There is exactly one `Window`, and it is moved, never copied. Therefore:

- **The driver can only grant while it holds the window.** Having granted, it holds nothing and
  cannot grant again until `Present` returns it. At most one grant is ever in flight, and the
  ring is provisioned ≥ 1 — so the grant ring is never full, and its droppable backstop is dead
  code. No counter required; `Option<Window>` *is* the counter.
- **The coordinator can only `Present` while it holds the window**, which it does only between a
  grant and that `Present`. Same argument, same conclusion.
- **The request carries no resource**, so it is the one genuinely losable message here. A dropped
  request costs one frame and the next vsync tick issues another — *actual* acceptable loss, of
  the kind §3 distinguishes from structural unreachability.

The coordinator's gate becomes a plain question about its own state: *do I hold a manifold, am I
not currently holding a window, and is no request outstanding?* The middle clause is what covers
the render window — while rendering it holds the window, so it cannot request another regardless
of how many manifolds arrive. That is the concern §8.5's round-trip credit was introduced to
address, handled by ownership instead of by a counter.

**Rule.** Where a resource is unique and moved rather than copied, ownership already enforces
the bound a `Credit` would encode. Adding one duplicates the invariant into a place it can drift
out of sync with — and, as here, into a place that needs an acknowledgement message to maintain.
Reach for `Credit` when the bound is *not* expressible as ownership: a count of outstanding
requests (vsync's tick budget), or a permit held across actors that never move a value.

### 8.7 Resize while the window is out being rendered

§8.5 said "on resize the driver swaps its held buffer." That is only true when it *has* one.
Between granting and `Present`, the driver holds nothing — and a resize in that gap is not a
corner case: it is the race today's `stale` flag exists to handle, and `engine_troupe.rs`
explicitly handles resizes arriving during both rendering and presentation.

Following §8.5 literally leaves two bad options: ignore the resize and reuse the stale-sized
buffer indefinitely, or allocate a second window and break the exactly-one invariant §8.6 now
depends on.

**The driver records pending dimensions instead of acting immediately.** It is already the size
authority and the allocator, so remembering "the OS told me 1920×1080 while the buffer was out"
is state that belongs to it:

- Resize while the driver **holds** the window: swap the buffer now, as §8.5 said.
- Resize while the window is **out**: store the new dimensions. Do nothing else.
- On `Present` with pending dimensions set: the returned frame was rendered at the old size, so
  **skip the blit** — presenting it would show a stretched or clipped frame — then replace the
  buffer with a correctly-sized one and clear the pending dimensions. The next grant is correct.

That reproduces today's behaviour exactly. The `stale` flag §7.2 deleted did precisely this job
from the engine's side; the work does not disappear, it moves to the actor that owns the buffer,
where it needs no cross-actor flag to coordinate — the driver observes the mismatch locally by
comparing what it stored against what came back.

One dropped frame per resize, same as today. The alternative — presenting a wrong-sized frame —
is visibly worse and is what today's code already declines to do.
