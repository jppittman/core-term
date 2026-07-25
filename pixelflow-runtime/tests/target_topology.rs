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
//! one thing. Decomposed per message, it turned out to already fit the same Credit/Droppable
//! vocabulary every other edge in §3 uses (`Present` even has its own non-blocking contract in
//! the code already, which the aggregate framing had simply missed). `regressed_aggregate_
//! framing_is_cyclic` below keeps that mistake on record so it isn't repeated.
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
    let vsync = topo.actor("vsync");
    let rasterizer = topo.actor("rasterizer");
    let app = topo.actor("app");

    // The one edge that stays genuinely Blocking: discrete, non-idempotent input and
    // window-lifecycle events. Nothing else in this graph is Blocking, so this alone cannot
    // form a cycle with anything below.
    //
    // `WindowCreated`/`Resized` notify engine over this edge as `WindowMeta` (scalar, `Copy`)
    // purely so engine can relay dimensions to app. No frame buffer travels here, and the
    // render pipeline does not consume these at all — the driver keeps the window and hands it
    // out on request (§8.5), so nothing downstream needs telling that it was resized.
    topo.blocking_edge(driver, engine); // DisplayEvent (window lifecycle as WindowMeta)

    // Idempotent "latest wins" state pushes: no credit needed at all. `Present` used to live
    // here too; it is now a rasterizer -> driver edge below. `PresentComplete` is gone as a
    // message entirely — the driver never gives the window away permanently, so there is
    // nothing to hand back (§8.5).
    topo.droppable_edge(engine, driver); // SetTitle / SetSize / SetCursor / Copy

    // RequestPaste / PasteData: plain droppable, deliberately *not* credit-bounded — see §3.2.
    // A reply is not merely rare-to-lose here, it is routinely absent: X11 answers a paste
    // request against an unowned clipboard with `property == None`, which the driver correctly
    // turns into no event at all. A Credit(1) released only by the reply would therefore be
    // spent forever the first time a user pastes from an empty clipboard.
    topo.droppable_edge(engine, driver); // RequestPaste
    topo.droppable_edge(driver, engine); // PasteData

    // vsync-tick -> RequestFrame relay: unchanged, still routed through engine. Out of scope
    // for this slice (§7.4's remaining-work note).
    topo.droppable_edge(vsync, engine); // vsync tick (Credit-gated)
    topo.droppable_edge(engine, app); // RequestFrame (Credit-gated)

    // The render pipeline, wired directly instead of through engine (§7/§8). The window is
    // *pulled*, not pushed (§8.5): the driver keeps and allocates it, and rasterizer asks for
    // one only when it holds a manifold and its Credit(1) allows. That is what makes every
    // droppable edge here recoverable rather than merely idempotent-sounding — a dropped
    // message costs one frame, and the sender still holds the authoritative copy to resend.
    //
    // Note there is no resize edge at all. On resize the driver swaps its own buffer and tells
    // nobody, so there is no resize message that can be dropped.
    topo.droppable_edge(rasterizer, driver); // window request (Credit(1))
    topo.droppable_edge(driver, rasterizer); // window grant, correctly sized (shares credit)
    topo.droppable_edge(app, rasterizer); // manifold submission (keep-latest, as today)
    topo.droppable_edge(rasterizer, driver); // Present, once rendered
    topo.droppable_edge(rasterizer, vsync); // RenderedResponse (FPS telemetry only)
    topo.droppable_edge(app, vsync); // ReturnToken — previously an engine->vsync Control-lane
                                     // send this proof never modeled separately; modeled here
                                     // now that it travels a new edge anyway.

    let order = topo.validate().expect(
        "every blocking edge has no return path and every reply is droppable, so this must be \
         a DAG — if it isn't, an edge above needs reclassifying before any live actor is \
         converted",
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
