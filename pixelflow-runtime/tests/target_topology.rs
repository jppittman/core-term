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
//! # Status: driver ↔ engine is a known-open gap, not a passing proof
//!
//! Writing the "does the whole mesh validate" test found that it doesn't: input events and
//! display commands were both specified Blocking, and two independent blocking edges between
//! the same pair of actors are a cycle regardless of the flows being logically unrelated (§3.1
//! of the migration doc). Fixing that requires a real behavioral trade-off — which stream may
//! lose messages under backpressure, or whether a ring can be proven never to fill — not a
//! mechanical edit, so it is **not resolved here**. The two tests below say so directly: one
//! proves everything *except* that pair is already a valid DAG, the other proves that pair, as
//! currently specified, is not.

use actor_scheduler::mealy::Topology;

/// Every edge from §3 of the migration doc except driver↔engine, which is not yet resolved
/// (see the module doc and §3.1). Confirms the four satellite relationships — vsync,
/// rasterizer, app, and driver's *own* structurally-unreachable window-return edge — validate
/// as specified, so the rollout in §5 can proceed on those independent of how driver↔engine is
/// eventually settled.
#[test]
fn the_resolved_part_of_the_target_mesh_is_a_dag() {
    let mut topo = Topology::new();

    let driver = topo.actor("driver");
    let engine = topo.actor("engine");
    let vsync = topo.actor("vsync");
    let rasterizer = topo.actor("rasterizer");
    let app = topo.actor("app");

    // driver -> engine (input events) and engine -> driver (display commands) are
    // deliberately NOT declared here — see `driver_and_engine_are_not_yet_resolved` below.

    // Structurally-unreachable droppable backstops: exactly one request ever outstanding,
    // so the ring can never actually be full when the reply lands (migration doc §3).
    topo.droppable_edge(driver, engine); // PresentComplete (window return)
    topo.droppable_edge(engine, rasterizer); // render request
    topo.droppable_edge(rasterizer, engine); // render complete

    // Genuinely acceptable loss: dropping is a normal outcome, not a bug symptom.
    topo.droppable_edge(vsync, engine); // vsync tick (Credit-gated)
    topo.droppable_edge(engine, vsync); // RenderedResponse (FPS telemetry only)
    topo.droppable_edge(engine, app); // RequestFrame (Credit-gated)
    topo.droppable_edge(app, engine); // AppData::RenderSurface (already "keep latest" today)

    let order = topo.validate().expect(
        "every edge here is droppable, so this must already be a DAG — if it isn't, an edge \
         above needs reclassifying before any live actor is converted",
    );
    assert_eq!(order.len(), 5);
}

/// The open finding itself, kept as a running test rather than a comment: as currently
/// specified (both directions Blocking), driver and engine form a real cycle. This should keep
/// failing to validate until someone makes the actual decision in §3.1 of the migration doc —
/// at which point this test should be rewritten to assert the *chosen* resolution validates,
/// not deleted.
#[test]
fn driver_and_engine_are_not_yet_resolved() {
    let mut topo = Topology::new();
    let driver = topo.actor("driver");
    let engine = topo.actor("engine");

    topo.blocking_edge(driver, engine); // input events — must not be lost
    topo.blocking_edge(engine, driver); // display commands — must not be lost

    let cycle = topo.validate().expect_err(
        "driver<->engine validates now — §3.1 of the migration doc has been resolved; \
         replace this test with one asserting the chosen fix (a droppable/coalesced stream, \
         a provably-bounded ring, or a split into blocking + droppable command streams)",
    );
    assert!(cycle.actors.contains(&"driver") && cycle.actors.contains(&"engine"));
}
