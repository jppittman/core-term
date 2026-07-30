//! Proves the target engine-mesh topology from
//! `docs/designs/pixelflow-runtime-engine-mesh-migration.md` §3 — built from the *same* five
//! actor names and *same* edge classifications that document commits to, so a change to either
//! the plan or this test without the other is a diff someone has to notice and reconcile, not a
//! silent drift.
//!
//! No `pixelflow-runtime` actor code is exercised here. `Topology` only needs names and edge
//! kinds — this file is deliberately actor-agnostic, so it stays valid across the rollout in
//! the plan's §5 regardless of which actor is mid-conversion.
//!
//! # History: the first version of this proof failed, on purpose left visible
//!
//! The first cut declared `DisplayEvent` and "display commands" as one aggregate edge each,
//! both Blocking, and the proof immediately failed: two blocking edges between the same pair
//! are a cycle regardless of the flows being logically unrelated (§3.1 of the migration doc).
//! The fix wasn't picking one side to weaken — it was noticing "display commands" was never
//! one thing. Decomposed per message, it turned out to already fit the same Credit/droppable
//! vocabulary every other edge in §3 uses (`Present` even has its own non-blocking contract in
//! the code already, which the aggregate framing had simply missed). `regressed_aggregate_
//! framing_is_cyclic` below keeps that mistake on record so it isn't repeated.
//!
//! # `droppable_edge` is gone: non-blocking edges are annotated, not declared
//!
//! `Topology::droppable_edge` has been deleted (actor-scheduler no longer has a shedding
//! delivery kind — every port parks) and was never load-bearing for this proof's actual check
//! in the first place: `Topology::validate` always drew its cycle graph from `blocking_edge`
//! calls only, and `droppable_edge` recorded the rest purely for diagnostics. So the edges
//! below that used to be `droppable_edge` calls are now plain comments — same names, same
//! rationale, just not fed to a graph that no longer has a vocabulary for "cannot block": their
//! safety is a per-edge argument (ownership, `Credit`, or "latest wins") that the DAG checker
//! was never actually verifying for them, and the design doc's §3.2 revision says so explicitly.
//!
//! # §7: the render pipeline moves off `engine` entirely
//!
//! Deleting `EngineHandler` means its render-pipeline edges (window hand-off, manifold
//! submission, present, FPS telemetry, token return) need to become direct edges among
//! `driver`, `rasterizer`, `vsync`, and `app` — and those specific edges were never checked.
//! Every edge through `engine` up to this point was individually proven; a direct
//! `driver` ↔ `rasterizer` edge was not, because it didn't exist yet. `the_target_engine_mesh_
//! is_a_dag` below is updated in place (not duplicated) to the corrected edge set, per this
//! file's own stated purpose of staying valid across the rollout rather than accumulating a
//! stale copy per step.
//!
//! `engine` remains a node for what this slice deliberately leaves alone: input-event
//! forwarding, `AppManagement` commands (`SetTitle`/`CreateWindow`/etc.), and the
//! vsync-tick → `RequestFrame` relay. Those are a separate, later slice.

use actor_scheduler::mealy::Topology;

/// Every edge from §3/§7 of the migration doc, decomposed per message rather than lumped into
/// "input events" / "display commands". `Create` (bootstrap-only, §3.1) is deliberately not
/// part of this graph at all — the same way the rasterizer's bootstrap handshake isn't part of
/// today's `troupe!` topology.
#[test]
fn the_target_engine_mesh_is_a_dag() {
    let mut topo = Topology::new();

    let driver = topo.actor("driver");
    let engine = topo.actor("engine");
    // Every edge vsync participates in is annotated rather than declared (see the module doc's
    // note on `droppable_edge`'s removal), so its `ActorId` binding is otherwise unused — the
    // node still has to exist for `order.len() == 5` below to mean what it says.
    let _vsync = topo.actor("vsync");
    let rasterizer = topo.actor("rasterizer");
    let app = topo.actor("app");

    // The one edge that stays genuinely Blocking: discrete, non-idempotent input and
    // window-lifecycle events. Nothing else in this graph is Blocking, so this alone cannot
    // form a cycle with anything below.
    //
    // `WindowCreated`/`Resized` notify engine over this edge as `WindowMeta` (scalar, `Copy`)
    // purely so engine can relay dimensions to app. No frame buffer travels here, and the
    // render pipeline does not consume these at all — the driver keeps the window and hands it
    // out on request (§8, "The shape"), so nothing downstream needs telling that it was resized.
    topo.blocking_edge(driver, engine); // DisplayEvent (window lifecycle as WindowMeta)

    // Idempotent "latest wins" state pushes: no credit needed at all. `Present` used to live
    // here too; it is now a rasterizer -> driver edge below. `PresentComplete` is gone as a
    // message entirely: the driver is now the window's resting owner, so `Present` *is* the
    // return and no separate acknowledgement is needed (§8, "The shape").
    //
    // engine -> driver: SetTitle / SetSize / SetCursor / Copy — not declared to `Topology` (see
    // the module doc's note on `droppable_edge`'s removal): safe because the latest write always
    // wins, not because a graph check proved it.

    // RequestPaste / PasteData: deliberately *not* credit-bounded — see §3.2. A reply is not
    // merely rare-to-lose here, it is routinely absent: X11 answers a paste request against an
    // unowned clipboard with `property == None`, which the driver correctly turns into no event
    // at all. A Credit(1) released only by the reply would therefore be spent forever the first
    // time a user pastes from an empty clipboard.
    //
    // engine -> driver: RequestPaste, driver -> engine: PasteData — not declared to `Topology`;
    // safe because a lost request/reply here costs nothing worse than a paste the user retries.

    // vsync-tick -> RequestFrame relay: unchanged, still routed through engine. Out of scope
    // for this slice (§7.4's remaining-work note).
    //
    // vsync -> engine: tick (Credit-gated), engine -> app: RequestFrame (Credit-gated) — not
    // declared to `Topology`; safety is `Credit` bounding outstanding requests below the reply
    // ring's capacity (§3.2), which the graph check cannot see and was never actually asked to.

    // The render pipeline, wired directly instead of through engine (§7/§8). The window is
    // *pulled*, not pushed (§8, "The shape"): the driver keeps and allocates it, and rasterizer asks for
    // one only when it holds a manifold and is not already holding a window.
    //
    // There is no Credit on this loop: ownership of the single Window is already the bound
    // (§8, "The judgment calls"). An actor cannot send a window it does not hold, so "at most one grant in flight"
    // and "at most one Present in flight" are properties of the type, not of a counter — and
    // each ring is provisioned >= 1, so neither is ever full when sent. That matters because
    // `Window` moves by value: a park that somehow did fire would freeze the display, not merely
    // delay a frame — ownership, not a graph check, is what makes that unreachable.
    //
    // The request is the one genuinely losable message here — it carries no resource, so a drop
    // costs one frame and the next vsync tick re-asks.
    //
    // Note there is no resize edge at all. The driver owns the buffer and resizes it locally,
    // deferring to `Present` if the window is out at the time (§8, "The judgment calls"), so no resize message
    // exists that could be dropped.
    // The coordinator is reactive — it acts only on messages it receives. A tick edge is what
    // makes "re-request until granted" implementable at all: without it, a dropped request could
    // only be retried if another manifold happened to arrive, and the app may legitimately send
    // nothing for many frames. Genuinely losable: the next tick is the retry.
    //
    // vsync -> rasterizer: tick — drives request retries. Not declared to `Topology`.

    // Requests ride the Management lane, `Present` the Data lane — deliberately different rings
    // (§8, "Known-unsettled"). Both are rasterizer -> driver, so sharing one inbox would let queued retries fill
    // it and force a `Present` park, freezing the sole buffer's only path back. Ownership bounds
    // the window-bearing traffic; it says nothing about unrelated messages in the same ring.
    //
    // rasterizer -> driver: window request (Management lane; carries nothing), driver ->
    // rasterizer: window grant (unreachable: can't grant unheld) — not declared to `Topology`.
    //
    // Blocking would be wrong here for a different reason than deadlock risk: keeping a slot on
    // the receiver does not help if the *newest* submission is lost in transit — nothing then
    // arrives to overwrite it, and the coordinator renders a stale kernel indefinitely if the app
    // has nothing further to send. An app parking behind a busy coordinator is correct
    // backpressure. There is no rasterizer -> app edge, so one blocking direction here cannot
    // cycle.
    topo.blocking_edge(app, rasterizer); // manifold submission

    // rasterizer -> driver: Present (unreachable: can't present unheld), rasterizer -> vsync:
    // RenderedResponse (FPS telemetry only), app -> vsync: ReturnToken — previously an
    // engine->vsync Control-lane send this proof never modeled separately, modeled here now that
    // it travels a new edge anyway. None declared to `Topology`: their safety arguments (buffer
    // ownership; "losing a sample skews a number, not correctness") live in the code and this
    // comment, not in a graph the DAG checker walks.

    let order = topo.validate().expect(
        "each declared blocking edge (driver -> engine, app -> rasterizer) has no return path \
         declared alongside it, so this must be a DAG — if it isn't, an edge above needs \
         reclassifying before any live actor is converted",
    );
    assert_eq!(order.len(), 5);
}

/// The same mistake `regressed_aggregate_framing_is_cyclic` records, in the shape someone
/// converting the render pipeline is likely to make fresh: treating "exactly one window
/// circulates" as license to make the hand-off synchronous in both directions. It doesn't
/// matter that "hand me a window" and "here's your rendered frame" are logically distinct
/// conversations — two blocking edges between the same pair is a cycle regardless.
#[test]
fn regressed_render_pipeline_framing_is_cyclic() {
    let mut topo = Topology::new();
    let driver = topo.actor("driver");
    let rasterizer = topo.actor("rasterizer");

    topo.blocking_edge(driver, rasterizer); // "hand off the window synchronously"
    topo.blocking_edge(rasterizer, driver); // "and block until it's presented"

    let cycle = topo
        .validate()
        .expect_err("two blocking edges between the same pair must be rejected");
    assert!(cycle.actors.contains(&"driver") && cycle.actors.contains(&"rasterizer"));
}

/// The mistake the first draft of this proof made, kept as a permanent regression test rather
/// than deleted once fixed: declaring *both* directions between driver and engine Blocking —
/// even for logically unrelated message flows — is a real cycle, not a false positive from an
/// overly strict checker.
#[test]
fn regressed_aggregate_framing_is_cyclic() {
    let mut topo = Topology::new();
    let driver = topo.actor("driver");
    let engine = topo.actor("engine");

    topo.blocking_edge(driver, engine); // input events
    topo.blocking_edge(engine, driver); // "display commands", undifferentiated

    let cycle = topo
        .validate()
        .expect_err("two blocking edges between the same pair must be rejected");
    assert!(cycle.actors.contains(&"driver") && cycle.actors.contains(&"engine"));
}
