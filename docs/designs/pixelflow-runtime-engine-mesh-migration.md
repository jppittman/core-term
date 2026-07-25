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
| Input events (`DisplayEvent`) | driver → engine | **Blocking — ⚠️ see §3.1** | Input must never be lost. |
| Display commands (`SetTitle`/`SetSize`/`Present`/…) | engine → driver | **Blocking — ⚠️ see §3.1** | Commands must never be lost. |
| Window return (`PresentComplete`) | driver → engine | **Credit(1) + Droppable backstop** | The *reply* half of Present. Exactly one window is ever in flight (the ping-pong buffer), so the reply ring's capacity is provisioned ≥ 1 and can **never actually be full** when the reply lands — the droppable backstop exists for `Topology`'s proof, but is unreachable by construction, not merely rare. This is *not* the same case as a true "acceptable loss" droppable edge (§9.2) — it is safe only because the credit bound is airtight. |
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

### 3.1 Open finding: driver ↔ engine is not yet a DAG

`pixelflow-runtime/tests/target_topology.rs` was written to prove the table above validates —
and instead it proved the table wrong. Input events and display commands were both marked
Blocking on the reasoning "genuinely one-way, nothing here closes a cycle." That reasoning was
false: `Topology` correctly rejects it, because **two independent blocking edges between the
same pair of actors are a cycle regardless of whether the two message flows are logically
unrelated.** driver blocked pushing an event into a full engine inbox, at the same moment
engine is blocked pushing a command into a full driver inbox, is the identical deadlock shape
as the app↔compiler example the Mealy doc uses to motivate the DAG rule in the first place —
"they're different conversations" does not save it.

This is deliberately left **unresolved in this doc**, not quietly patched, because every fix
available is a real behavioral trade-off, not a mechanical one:

- Mark `DisplayEvent` droppable — user input can be silently lost under backpressure.
- Mark display commands droppable — a `Present` can be silently skipped, or `SetTitle`/`SetSize`
  coalesced to "latest wins" (plausible for those two; almost certainly wrong for `Present`).
- Deepen one ring so it is provisioned never to fill under realistic load, the way the
  window-return edge is provisioned at exactly 1 — plausible if input volume is bounded and
  measurable, but "provably never fills" needs an actual argument, not a big number.
- Split display commands into a genuinely droppable stream (`SetTitle`, cursor, coalescible
  state) and a separate, still-blocking one for `Present` alone, if `Present` turns out to be
  the only command that truly cannot tolerate loss.

**This is a decision for a human, not a default the DAG checker should pick.** Recorded here so
step 4 of the rollout (§5) starts from a known question instead of rediscovering it, and so
`target_topology.rs`'s test for the full mesh does not silently paper over it — it currently
tests everything except this pair, plus a dedicated test proving the pair really is cyclic as
specified, so the gap stays visible in `cargo test` output rather than in a comment.

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
2. **`vsync` first**: smallest actor, already isolated, already has the credit-shaped token
   bucket crying out to become real `Credit`. Convert to `Transducer`/`Node`, replace
   `VSYNC_TOKEN_BUCKET` with `Credit`, keep talking to the *unconverted* engine across an
   adapter that speaks the old `ActorHandle` shape on the engine-facing side. Proves the pattern
   on the smallest possible surface.
3. **`rasterizer`**: same shape as vsync (request/reply, already isolated via its bootstrap
   handshake — which was itself a symptom of not fitting the old static-topology model, so this
   should *simplify*, not complicate).
4. **`driver`**: convert the command/event edges; window-return `Credit` last, since it's the
   one edge where getting the bound wrong is worst.
5. **`app`-facing edge + delete `EngineHandler`**: once every satellite speaks the new
   protocol directly to its real peer, the mediator has nothing left to route and is deleted,
   not refactored down.

Each of steps 2–5 lands as its own PR with its own before/after behavioral tests against the
running terminal, not just unit tests of the actor in isolation.

---

## 6. Proof: the target topology is a DAG

Before any of §5's steps touch a real actor, the target graph above is checked mechanically —
`pixelflow-runtime/tests/target_topology.rs` builds exactly the nine edges in the table with
their declared kinds and asserts `Topology::validate()` succeeds. This is deliberately built
from the *same* five actor names and *same* edge classification this doc commits to, so a
future change to either has to touch both or the proof drifts from the plan it's supposed to
prove.
