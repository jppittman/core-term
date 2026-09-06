//! The `Scene::Surface` lane's HiDPI contramap, from outside the crate.
//!
//! A surface is authored in point space, so a denser device frame has to be
//! contramapped by points/pixels or it renders at half density —
//! `RenderCoordinator`'s bind step is where that decision is made, and
//! forwarding unchanged is invisible on a 1:1 monitor and wrong on every
//! Retina one.
//!
//! This lives in `tests/` rather than beside the coordinator because no path
//! in `pixelflow-runtime/src` may construct a `Scene::Surface` any more
//! (`docs/plans/2026-09-06-kernel-with-a-lattice.md`, S2): packed scenes are
//! device-pixel space by construction and need no wrap, so the coordinator's
//! own fixtures are packed. The branch still exists for `scene3d`, and S3
//! deletes it along with this file — until then it stays covered, because a
//! branch nothing exercises is a branch nothing can catch.

use std::sync::Arc;

use pixelflow_core::{Discrete, Field, Manifold};
use pixelflow_graphics::render::scene::Scene;
use pixelflow_runtime::api::public::WindowId;
use pixelflow_runtime::display::messages::{Surface, Window};
use pixelflow_runtime::display::window_keeper::WindowKeeper;
use pixelflow_runtime::render_coordinator::{RenderCoordinator, Step};

/// A manifold that ignores its input — the coordinator never samples it, it
/// only decides whether to wrap it.
#[derive(Clone, Copy)]
struct Flat;

impl Manifold<(Field, Field, Field, Field)> for Flat {
    type Output = Discrete;
    fn eval(&self, _p: (Field, Field, Field, Field)) -> Discrete {
        Discrete::pack(
            Field::from(0.0),
            Field::from(0.0),
            Field::from(0.0),
            Field::from(1.0),
        )
    }
}

fn surface_scene() -> Scene {
    Scene::Surface(Arc::new(Flat))
}

/// One buffer at the given logical size over the given device frame.
fn window(logical: (u32, u32), frame: (u32, u32), scale: f64) -> Window {
    let mut keeper = WindowKeeper::new();
    keeper.surface_changed(Surface {
        id: WindowId(1),
        width_px: logical.0,
        height_px: logical.1,
        frame_width: frame.0,
        frame_height: frame.1,
        scale,
    });
    keeper.request();
    keeper.pending_grant().expect("a buffer is at rest")
}

/// Drive the coordinator to the point where it hands back a bound scene.
fn bind(scene: Scene, window: Window) -> Scene {
    let mut coord = RenderCoordinator::new();
    let Step::RequestWindow = coord.submit(scene) else {
        panic!("a coordinator with no buffer asks for one");
    };
    coord.request_sent();
    let Step::Render(request) = coord.granted(window) else {
        panic!("a granted buffer plus a submitted scene is a render");
    };
    request.scene
}

/// This pins the *decision*, not the numeric scale — reading the ratio back
/// out of the bound manifold would mean sampling a `Field`, and lane access is
/// deliberately not public.
#[test]
fn a_retina_frame_warps_a_surface_and_a_1_1_frame_does_not() {
    let flat = surface_scene();
    let (Scene::Surface(sent), Scene::Surface(got)) = (
        &flat,
        &bind(flat.clone(), window((100, 100), (100, 100), 1.0)),
    ) else {
        panic!("surface scenes in, surface scenes out");
    };
    assert!(
        Arc::ptr_eq(got, sent),
        "a 1:1 frame needs no warp, so the scene should pass through unwrapped"
    );

    // 2:1 — 100 points across a 200-pixel frame.
    let bound = bind(flat.clone(), window((100, 100), (200, 200), 2.0));
    let (Scene::Surface(sent), Scene::Surface(got)) = (&flat, &bound) else {
        panic!("surface scenes in, surface scenes out");
    };
    assert!(
        !Arc::ptr_eq(got, sent),
        "a Retina frame must be contramapped by points/pixels, not forwarded as authored"
    );
}
