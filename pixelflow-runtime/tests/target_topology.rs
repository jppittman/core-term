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

use actor_scheduler::mealy::Topology;

/// Every edge from §3 of the migration doc, decomposed per message rather than lumped into
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
    // window-lifecycle events. Nothing on the engine -> driver side is Blocking any more, so
    // this alone cannot form a cycle with anything below.
    topo.blocking_edge(driver, engine); // DisplayEvent, minus PasteData

    // Present / PresentComplete: structurally-unreachable droppable backstop, Credit(1) via
    // the single ping-pong window buffer.
    topo.droppable_edge(engine, driver); // Present
    topo.droppable_edge(driver, engine); // PresentComplete

    // Idempotent "latest wins" state pushes: no credit needed at all.
    topo.droppable_edge(engine, driver); // SetTitle / SetSize / SetCursor / Copy

    // RequestPaste / PasteData: plain droppable, deliberately *not* credit-bounded — see §3.2.
    // A reply is not merely rare-to-lose here, it is routinely absent: X11 answers a paste
    // request against an unowned clipboard with `property == None`, which the driver correctly
    // turns into no event at all. A Credit(1) released only by the reply would therefore be
    // spent forever the first time a user pastes from an empty clipboard.
    topo.droppable_edge(engine, driver); // RequestPaste
    topo.droppable_edge(driver, engine); // PasteData

    // The other three satellite relationships, unchanged from the first draft of this proof.
    topo.droppable_edge(engine, rasterizer); // render request
    topo.droppable_edge(rasterizer, engine); // render complete
    topo.droppable_edge(vsync, engine); // vsync tick (Credit-gated)
    topo.droppable_edge(engine, vsync); // RenderedResponse (FPS telemetry only)
    topo.droppable_edge(engine, app); // RequestFrame (Credit-gated)
    topo.droppable_edge(app, engine); // AppData::RenderSurface (already "keep latest" today)

    let order = topo.validate().expect(
        "every engine -> driver edge is droppable and driver -> engine is the only Blocking \
         one, so this must be a DAG — if it isn't, an edge above needs reclassifying before \
         any live actor is converted",
    );
    assert_eq!(order.len(), 5);
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
