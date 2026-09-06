//! The render coordinator: decides when a frame can be drawn, and with what.
//!
//! This is the renderer's half of the pull protocol whose other half is
//! [`WindowKeeper`](crate::display::window_keeper::WindowKeeper). The driver owns the framebuffer
//! and lends it out; this side asks for it, binds the current kernel to the buffer it was handed,
//! and gives it back to be shown.
//!
//! It holds no actor handles and performs no sends. Every decision comes out as a [`Step`] the
//! caller delivers, which is what lets the whole protocol — including the resize and
//! render-in-flight races that are awkward to provoke through a live troupe — be tested as plain
//! function calls. It is also the precondition for the coordinator becoming its own node
//! (`docs/designs/pixelflow-runtime-engine-mesh-migration.md` §8): a node cannot be carved out of
//! logic that is interleaved with the mediator's handles.

use crate::display::messages::{Window, WindowMeta};
use crate::platform::PlatformPixel;
use actor_scheduler::mealy::Credit;
use pixelflow_core::{At, W, X, Y, Z};
use pixelflow_graphics::render::rasterizer::{RenderRequest, RenderResponse};
use std::sync::Arc;
use std::time::Duration;

/// A scene as it travels between the app and the rasterizer.
use pixelflow_graphics::render::scene::Scene;

/// What the coordinator wants delivered, having decided everything it can on its own.
///
/// Returned by every input method rather than accumulated, because at most one of these is ever
/// outstanding: a coordinator holding the buffer renders, and one without it asks. There is no
/// state in which it wants both.
#[must_use = "a Step is a decision that only takes effect when the caller delivers it"]
pub enum Step {
    /// Nothing to do — no kernel to draw, or a render already in flight.
    Idle,
    /// Ask the driver for the buffer (Management lane; carries nothing).
    RequestWindow,
    /// Send this to the rasterizer. Carries the buffer, so losing it strands the display.
    Render(RenderRequest<PlatformPixel, WindowMeta>),
}

/// The outcome of a completed render.
pub struct Completed {
    /// The drawn buffer and how long it took, to be handed to the driver to show.
    ///
    /// `None` when the rasterizer was paused and returned the buffer undrawn — presenting it
    /// would blit stale pixels, so it is retained in the coordinator instead. Pairing the two
    /// keeps "was it drawn?" a single question: the buffer and its render time are either both
    /// here or neither is.
    pub present: Option<(Window, Duration)>,
    /// What to do once `present` has been delivered.
    pub next: Step,
}

/// Decides when to render and what to render into.
///
/// Two resources have to be in hand at once: a kernel to draw and a buffer to draw into. Every
/// path that acquires either ends at [`advance`](Self::advance), so "can we render yet?" is asked
/// in one place rather than open-coded at each arrival.
pub struct RenderCoordinator {
    /// The window buffer, while it is on loan from the driver.
    ///
    /// Borrowed, not owned: it arrives in answer to a request and goes back with `Present`.
    /// Holding it is the whole of "a render can start" — there is no separate flag, and no
    /// generation kept alongside it, because the buffer carries its own.
    window: Option<Window>,
    /// A window request has been emitted and not yet answered.
    ///
    /// Paired with the driver's own latch: it holds an unanswered request until a buffer frees,
    /// so asking twice cannot make an answer come sooner — it only leaves a request alive past
    /// the grant that satisfied it, and the next buffer to appear gets handed over unasked.
    ///
    /// Ownership bounds how many *buffers* exist and says nothing about how many *requests* do,
    /// which is why this flag is not redundant with holding the buffer.
    awaiting_grant: bool,
    /// The one-outstanding-render bound on the coordinator → rasterizer edge.
    render_credit: Credit,
    /// Latest kernel from the app — keep-latest, so a newer one replaces any frame that has not
    /// started rendering.
    ///
    /// Not deletable in favour of a droppable port: a full ring discards the *newest* message,
    /// while keep-latest needs the *oldest* gone. If the app's last state change were the dropped
    /// one, a stale surface would sit on screen indefinitely.
    pending_scene: Option<Scene>,
}

impl Default for RenderCoordinator {
    fn default() -> Self {
        Self::new()
    }
}

impl RenderCoordinator {
    /// A coordinator holding neither half of a render.
    #[must_use]
    pub fn new() -> Self {
        Self {
            window: None,
            awaiting_grant: false,
            render_credit: Credit::new(1),
            pending_scene: None,
        }
    }

    /// Whether the buffer is currently in hand.
    ///
    /// Read-only on purpose: the buffer leaves only inside a [`Step::Render`], so there is no
    /// accessor that hands it out. "Who has the buffer?" means "who is drawing?", and this is how
    /// the other side of that equivalence is checked without being able to break it.
    #[must_use]
    pub fn holds_buffer(&self) -> bool {
        self.window.is_some()
    }

    /// The app produced a new kernel. Keep-latest: it replaces any frame not yet started.
    pub fn submit(&mut self, scene: Scene) -> Step {
        self.pending_scene = Some(scene);
        self.advance()
    }

    /// The driver answered a request with the buffer.
    pub fn granted(&mut self, window: Window) -> Step {
        self.awaiting_grant = false;
        self.hold(window);
        self.advance()
    }

    /// The rasterizer finished with the buffer.
    ///
    /// `render_time` is `None` when the rasterizer was paused and handed the buffer back
    /// undrawn; the buffer's return is unconditional, its having been drawn into is not.
    pub fn completed(&mut self, response: RenderResponse<PlatformPixel, WindowMeta>) -> Completed {
        // The render is over, so the edge's one credit is free again.
        self.render_credit.release();

        let window = Window::rejoin(response.frame, response.meta);
        match response.render_time {
            Some(render_time) => Completed {
                present: Some((window, render_time)),
                // The buffer has just left, so this asks for it back.
                next: self.advance(),
            },
            None => {
                self.hold(window);
                Completed {
                    present: None,
                    next: self.advance(),
                }
            }
        }
    }

    /// A [`Step::RequestWindow`] reached the driver. Latch it so we do not ask twice.
    ///
    /// Deliberately separate from emitting the step: a request that was never delivered must not
    /// set the latch, or the retry that would have fixed it never happens.
    pub fn request_sent(&mut self) {
        self.awaiting_grant = true;
    }

    /// A [`Step::Render`] could not be delivered, and the buffer it carried is gone with it.
    ///
    /// Gives the credit back so the edge is not permanently blocked by a render that never
    /// started. It cannot give the buffer back: `send` takes the message by value and
    /// `SendError` carries nothing, so a failed render send destroys the framebuffer and the
    /// display cannot recover. See the note in `EngineHandler`'s caller.
    pub fn render_send_failed(&mut self) {
        self.render_credit.release();
    }

    /// Take the buffer into hand.
    ///
    /// An `assert!`, not a `debug_assert!`, because the release-build alternative is worse than a
    /// crash: the buffer being overwritten is the driver's only current one, and losing it leaves
    /// a display that never draws again and cannot recover without being killed.
    fn hold(&mut self, window: Window) {
        assert!(
            self.window.is_none(),
            "a second buffer (generation {}) arrived while one was already in hand",
            window.generation
        );
        self.window = Some(window);
    }

    /// Render if both halves are in hand, and otherwise go and get the missing one.
    ///
    /// Every path that acquires either half ends here, so "can we render yet?" is asked in one
    /// place rather than open-coded at each arrival. Nothing is consumed unless a render actually
    /// starts, which is what lets the next arrival pick up whatever this call put back.
    ///
    /// Public because re-driving is an input in its own right: a vsync tick has nothing new to
    /// offer but is what retries a request lost in transit, and a window appearing or resizing is
    /// the first moment a buffer can exist to draw into. Idempotent, so calling it on a hunch
    /// costs nothing.
    pub fn advance(&mut self) -> Step {
        if self.pending_scene.is_none() {
            // Nothing to draw. Deliberately does *not* ask for the buffer: holding it idle would
            // block nothing today, but it makes "who has the buffer?" stop meaning "who is
            // drawing?", and that equivalence is what bounds the loop.
            return Step::Idle;
        }
        let Some(window) = self.window.take() else {
            // Only ask for a buffer we could actually draw into *now*. Asking while a render is
            // in flight looks harmless — the answer is normally "nothing free" — but a resize
            // allocates a replacement, which would answer the outstanding ask and leave a second
            // buffer in hand while the first is still out. The completion then arrives with
            // nowhere to put its buffer, and a *paused* completion overwrites the replacement,
            // stranding the driver's only current buffer and freezing the display.
            //
            // The credit already knows: it is the one-outstanding-render bound.
            if self.render_credit.outstanding() == 0 && !self.awaiting_grant {
                return Step::RequestWindow;
            }
            return Step::Idle;
        };

        // Take the edge's single credit *before* taking the window apart — a refusal has to hand
        // both halves back intact.
        //
        // A refusal is ordinary backpressure, not an error: vsync keeps asking for frames at
        // 60Hz while a render is in flight, so a frame arriving mid-render is the common case.
        // Dropping the pair would be the expensive kind of mistake — the window in hand during an
        // in-flight render is the live buffer, so losing it strands the display at whatever the
        // in-flight render happens to produce.
        if !self.render_credit.try_consume() {
            self.window = Some(window);
            return Step::Idle;
        }

        let scene = self
            .pending_scene
            .take()
            .expect("pending_scene checked Some above");
        Step::Render(bind(scene, window))
    }
}

/// Bind a kernel to the buffer it will be drawn into.
///
/// The scene is authored in point space; the frame is the platform's sample lattice and may be
/// denser (device pixels on HiDPI displays). The lattice embedding is the measured ratio
/// points/pixels per axis — identity when the platform samples 1:1 (X11, non-Retina macOS).
/// Contramapping here keeps the app scale-agnostic: platform = dimap.
///
/// Everything comes from the buffer itself, so the result is correct for the frame it was handed
/// by construction — there is no separate scale to keep in step, and nothing to re-derive later.
fn bind(scene: Scene, window: Window) -> RenderRequest<PlatformPixel, WindowMeta> {
    let (frame, meta) = window.tear();
    let WindowMeta {
        width_px,
        height_px,
        ..
    } = meta;

    assert!(
        frame.width > 0 && frame.height > 0,
        "cannot render into an empty frame ({}x{})",
        frame.width,
        frame.height
    );
    let point_per_px_x = width_px as f32 / frame.width as f32;
    let point_per_px_y = height_px as f32 / frame.height as f32;
    let scene: Scene = match scene {
        // Point-space surfaces are contramapped into the denser device grid.
        Scene::Surface(manifold) => {
            if point_per_px_x == 1.0 && point_per_px_y == 1.0 {
                Scene::Surface(manifold)
            } else {
                Scene::Surface(Arc::new(At {
                    inner: manifold,
                    x: X * point_per_px_x,
                    y: Y * point_per_px_y,
                    z: Z,
                    w: W,
                }))
            }
        }
        // A packed scene is device-pixel space by construction: its kernel
        // was compiled against this frame's own lattice, so an author working
        // in points said so in the language — `Kernel::at` precomposition on
        // the channel kernels — before compiling. There is nothing left to
        // wrap here, and wrapping would mean recompiling the program every
        // frame.
        Scene::Packed(packed) => Scene::Packed(packed),
    };

    RenderRequest { scene, frame, meta }
}

#[cfg(test)]
mod tests {
    //! The coordinator, driven directly.
    //!
    //! No troupe, no threads, no display: the whole protocol is plain function calls, which is
    //! the point of the core holding no handles. Buffers come from a real
    //! [`WindowKeeper`] rather than being hand-built, so the two halves of the pull protocol are
    //! exercised against each other rather than against a fixture's idea of the other side.

    use super::*;
    use crate::api::public::WindowId;
    use crate::display::messages::Surface;
    use crate::display::window_keeper::WindowKeeper;
    use pixelflow_core::Field;
    use pixelflow_core::{Discrete, Manifold};
    use pixelflow_graphics::render::rasterizer::RenderResponse;

    /// A manifold that ignores its input — the coordinator never samples it, it only decides
    /// whether to wrap it.
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

    fn scene() -> Scene {
        Scene::Surface(Arc::new(Flat))
    }

    /// A keeper holding one buffer at the given logical size, sampled 1:1.
    fn keeper_at(logical: (u32, u32), frame: (u32, u32), scale: f64) -> WindowKeeper {
        let mut keeper = WindowKeeper::new();
        keeper.surface_changed(Surface {
            id: WindowId(1),
            width_px: logical.0,
            height_px: logical.1,
            frame_width: frame.0,
            frame_height: frame.1,
            scale,
        });
        keeper
    }

    /// Take the buffer out of a keeper, as a grant would.
    fn grant_from(keeper: &mut WindowKeeper) -> Window {
        keeper.request();
        keeper.pending_grant().expect("a buffer is at rest")
    }

    fn one_to_one_window() -> Window {
        grant_from(&mut keeper_at((100, 100), (100, 100), 1.0))
    }

    /// A render that finished normally, handing the buffer back drawn.
    fn drawn(
        request: RenderRequest<PlatformPixel, WindowMeta>,
    ) -> RenderResponse<PlatformPixel, WindowMeta> {
        RenderResponse {
            frame: request.frame,
            render_time: Some(Duration::from_millis(1)),
            meta: request.meta,
        }
    }

    /// A render the rasterizer skipped because it was paused.
    fn skipped(
        request: RenderRequest<PlatformPixel, WindowMeta>,
    ) -> RenderResponse<PlatformPixel, WindowMeta> {
        RenderResponse {
            frame: request.frame,
            render_time: None,
            meta: request.meta,
        }
    }

    fn expect_render(step: Step) -> RenderRequest<PlatformPixel, WindowMeta> {
        match step {
            Step::Render(request) => request,
            Step::RequestWindow => panic!("expected a render, got a window request"),
            Step::Idle => panic!("expected a render, got idle"),
        }
    }

    fn assert_idle(step: Step) {
        assert!(
            matches!(step, Step::Idle),
            "expected idle, got a step the caller would have delivered"
        );
    }

    fn assert_requests_window(step: Step) {
        assert!(
            matches!(step, Step::RequestWindow),
            "expected a window request"
        );
    }

    /// Holding the buffer has to keep meaning "drawing into it", so an idle coordinator does not
    /// hoard it.
    #[test]
    fn nothing_to_draw_does_not_ask_for_the_buffer() {
        let mut coord = RenderCoordinator::new();
        assert_idle(coord.advance());
        assert!(!coord.holds_buffer());
    }

    #[test]
    fn a_scene_with_no_buffer_asks_for_one() {
        let mut coord = RenderCoordinator::new();
        assert_requests_window(coord.submit(scene()));
    }

    /// The driver latches an unanswered request, so asking again cannot make the answer come
    /// sooner — it only outlives the grant that satisfied it, and the next buffer to appear is
    /// handed over unasked.
    #[test]
    fn an_unanswered_request_is_not_repeated() {
        let mut coord = RenderCoordinator::new();
        assert_requests_window(coord.submit(scene()));
        coord.request_sent();

        assert_idle(coord.advance());
        assert_idle(coord.submit(scene()));
    }

    /// A request that never reached the driver must be retried, or the display stops forever on
    /// one dropped message.
    #[test]
    fn an_undelivered_request_is_retried_on_the_next_tick() {
        let mut coord = RenderCoordinator::new();
        assert_requests_window(coord.submit(scene()));
        // No `request_sent()` — the send failed.
        assert_requests_window(coord.advance());
    }

    #[test]
    fn a_granted_buffer_and_a_scene_render() {
        let mut coord = RenderCoordinator::new();
        assert_requests_window(coord.submit(scene()));
        coord.request_sent();

        let request = expect_render(coord.granted(one_to_one_window()));
        assert_eq!((request.frame.width, request.frame.height), (100, 100));
        assert!(
            !coord.holds_buffer(),
            "the buffer left with the render request"
        );
    }

    /// The one-outstanding-render bound. A second scene arriving mid-render must not start
    /// another one — there is only one buffer, and it is out.
    #[test]
    fn a_second_scene_mid_render_does_not_start_a_second_render() {
        let mut coord = RenderCoordinator::new();
        assert_requests_window(coord.submit(scene()));
        coord.request_sent();
        let _in_flight = expect_render(coord.granted(one_to_one_window()));

        assert_idle(coord.submit(scene()));
        assert_idle(coord.advance());
    }

    /// The steady state: draw, hand the buffer straight back, ask for it again.
    #[test]
    fn a_completed_render_presents_and_asks_again() {
        let mut coord = RenderCoordinator::new();
        assert_requests_window(coord.submit(scene()));
        coord.request_sent();
        let request = expect_render(coord.granted(one_to_one_window()));

        // Something to draw next frame, so the coordinator has a reason to want it back. The
        // render is still in flight, so this starts nothing on its own.
        assert_idle(coord.submit(scene()));
        let Completed { present, next } = coord.completed(drawn(request));

        assert!(present.is_some(), "a drawn buffer goes to the driver");
        assert_requests_window(next);
        assert!(
            !coord.holds_buffer(),
            "the buffer went out with the present"
        );
    }

    /// A paused rasterizer hands the buffer back undrawn. Presenting it would blit stale pixels,
    /// so it is kept — and it is still in hand for the next render.
    #[test]
    fn a_skipped_render_retains_the_buffer_unpresented() {
        let mut coord = RenderCoordinator::new();
        assert_requests_window(coord.submit(scene()));
        coord.request_sent();
        let request = expect_render(coord.granted(one_to_one_window()));

        let Completed { present, next } = coord.completed(skipped(request));
        assert!(present.is_none(), "an undrawn buffer must not be blitted");
        assert_idle(next);
        assert!(coord.holds_buffer(), "the buffer stays here");
    }

    /// The credit is the render bound, so a render that never started must give it back —
    /// otherwise the edge is blocked forever by a request that was never sent.
    #[test]
    fn a_failed_render_send_frees_the_edge() {
        let mut coord = RenderCoordinator::new();
        assert_requests_window(coord.submit(scene()));
        coord.request_sent();
        let _lost = expect_render(coord.granted(one_to_one_window()));

        coord.render_send_failed();

        // The buffer went with the lost request, so the coordinator has nothing in hand and asks
        // again rather than sitting on a consumed credit.
        assert_requests_window(coord.submit(scene()));
    }

    /// The HiDPI case the design doc calls out as uncovered: the scene is authored in points and
    /// the frame is the device lattice, so a denser frame has to be contramapped or it renders at
    /// half density.
    ///
    /// This pins the *decision*, not the numeric scale — reading the ratio back out of the bound
    /// manifold would mean sampling a `Field`, and lane access is deliberately not public. The
    /// branch is what the doc warns about: forwarding unchanged is invisible on a 1:1 monitor and
    /// wrong on every Retina one.
    #[test]
    fn a_retina_frame_warps_the_scene_and_a_1_1_frame_does_not() {
        // 1:1 — the scene must arrive at the rasterizer untouched.
        let mut coord = RenderCoordinator::new();
        let flat = scene();
        assert_requests_window(coord.submit(flat.clone()));
        coord.request_sent();
        let request = expect_render(coord.granted(one_to_one_window()));
        let (Scene::Surface(got), Scene::Surface(sent)) = (&request.scene, &flat) else {
            panic!("surface scenes in, surface scenes out");
        };
        assert!(
            Arc::ptr_eq(got, sent),
            "a 1:1 frame needs no warp, so the scene should pass through unwrapped"
        );

        // 2:1 — 100 points across a 200-pixel frame.
        let mut coord = RenderCoordinator::new();
        let flat = scene();
        assert_requests_window(coord.submit(flat.clone()));
        coord.request_sent();
        let retina = grant_from(&mut keeper_at((100, 100), (200, 200), 2.0));
        let request = expect_render(coord.granted(retina));
        assert_eq!((request.frame.width, request.frame.height), (200, 200));
        let (Scene::Surface(got), Scene::Surface(sent)) = (&request.scene, &flat) else {
            panic!("surface scenes in, surface scenes out");
        };
        assert!(
            !Arc::ptr_eq(got, sent),
            "a Retina frame must be contramapped by points/pixels, not forwarded as authored"
        );
    }

    /// The metadata rides with the request and comes back untouched, which is what deleted the
    /// torn-`Window` stash (§7.1). If it did not survive the round trip the buffer could not be
    /// rejoined into a window at all.
    #[test]
    fn window_metadata_survives_the_round_trip() {
        let mut coord = RenderCoordinator::new();
        assert_requests_window(coord.submit(scene()));
        coord.request_sent();
        let request = expect_render(coord.granted(one_to_one_window()));
        let sent = request.meta;

        let Completed { present, .. } = coord.completed(drawn(request));
        let (window, _) = present.expect("a drawn buffer comes back to be presented");

        assert_eq!(window.width_px, sent.width_px);
        assert_eq!(window.height_px, sent.height_px);
        assert_eq!(window.generation, sent.generation);
    }
}
