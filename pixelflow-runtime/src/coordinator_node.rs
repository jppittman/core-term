//! The render coordinator's own green node: a second [`Node`](actor_scheduler::mealy::Node) on
//! the vsync green-host thread.
//!
//! Step 5c of `docs/designs/pixelflow-runtime-engine-mesh-migration.md` §8: [`RenderCoordinator`]
//! (`render_coordinator.rs`) leaves `EngineCore` and runs here as a `Transducer`
//! ([`CoordinatorCore`]) with its own `Wiring` ([`CoordinatorWiring`]) sending directly to the
//! renderer, the driver, and vsync's telemetry lane. Render completions arrive straight from
//! the renderer forwarder — no engine round-trip — while window grants, scene submissions, and
//! the "try rendering again" nudge still arrive from the engine shell, which is the one relay
//! this step leaves in place (`engine_core.rs`'s `coordinator` port).
//!
//! # Why two producers, two lanes
//!
//! [`CoordinatorData`] (engine shell → coordinator: submit/grant/advance) and
//! [`RenderResponse`] (renderer forwarder → coordinator: completions) are different producers,
//! so SPSC purity puts them on different lanes even before priority is considered. Management
//! outranking Data is not incidental, though: a completion frees the render credit and the
//! buffer the driver is waiting on, so finishing in-flight work before accepting a new
//! submission is the same continuation-first discipline `Node::poll` already applies to its own
//! self-edge — draining what is already committed before taking on more.
//!
//! # The delivery-feedback seam dissolved
//!
//! [`RenderCoordinator::request_sent`]/`render_send_failed` existed because `EngineHandler`'s own
//! sends could fail *after* the core had already decided — a pure core cannot see a send
//! outcome, so the shell had to report it back in. Under `Wiring`, delivery is flush's problem: a
//! blocked port parks in the `Node`'s outbox and retries on its own, so the core never needs to
//! learn about a transient failure. [`CoordinatorCore::route`] therefore calls
//! [`RenderCoordinator::request_sent`] the moment it decides to ask — delivery is now
//! guaranteed-or-parked, so "an ask is in flight" is already true at the moment of emission.
//! `render_send_failed` has no counterpart here: a renderer that is truly *gone* surfaces as
//! [`Flush::Disconnected`], which panics — there is no supervisor to escalate to, and a
//! recoverable outcome was never on offer once every build profile sets `panic = "abort"`. That
//! is still strictly better than the old warn-and-release-credit path right up to the panic: the
//! render's buffer is retained rather than destroyed (see [`CoordinatorWiring`]'s `render` port),
//! so the crash is at least legible from a core dump.

use crate::display::messages::{DisplayControl, DisplayData, DisplayMgmt, Window, WindowMeta};
use crate::platform::PlatformPixel;
use crate::render_coordinator::{Completed, RenderCoordinator, Step};
use crate::vsync_actor::RenderedResponse;
use actor_scheduler::mealy::Transducer;
use actor_scheduler::{ActorHandle, GreenSender, HandlerError, Message};
use pixelflow_graphics::render::renderer::{RenderRequest, RenderResponse, RendererHandle};
use std::time::{Duration, Instant};

/// A manifold as it travels between the app and the renderer.
///
/// Duplicates `render_coordinator`'s own private alias of the same name rather than exporting
/// it: that module's `Scene` is deliberately not part of even this crate's wider surface, and
/// re-exporting it just to share one line would widen `RenderCoordinator`'s API for a type
/// alias, not a capability.
use pixelflow_graphics::render::scene::Scene;

/// Frame counter logging cadence — moved verbatim from the old `EngineCore` copy: the telemetry
/// edge (`rendered`, below) is a coordinator → vsync edge now, so the counter that stamps it
/// lives with the coordinator (§7.3 of the migration doc).
const LOG_FRAME_INTERVAL: u64 = 60;

/// Input on the coordinator's data lane. Producer: the engine shell. Window grants, scene
/// submissions, and the "something might be drawable now" nudge all still arrive by way of
/// `EngineCore`'s `coordinator` port (driver `WindowGranted`, app `RenderSurface`, and the
/// vsync-tick / window-lifecycle advance, respectively) rather than directly from their
/// original sources — keeping this lane single-producer.
pub(crate) enum CoordinatorData {
    /// A new kernel from the app. Keep-latest — see [`RenderCoordinator::submit`].
    Submit(Scene),
    /// The driver granted the buffer this coordinator asked for.
    Granted(Window),
    /// Re-drive the pull protocol with nothing new to offer: retries a request lost in transit,
    /// or is the first chance to render after a window appears or resizes.
    Advance,
}

impl std::fmt::Debug for CoordinatorData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Submit(_) => f.write_str("Submit(<scene>)"),
            Self::Granted(window) => f.debug_tuple("Granted").field(window).finish(),
            Self::Advance => f.write_str("Advance"),
        }
    }
}

type RenderMsg = Message<
    RenderRequest<PlatformPixel, WindowMeta>,
    pixelflow_graphics::render::renderer::RenderControl,
    pixelflow_graphics::render::renderer::RenderManagement,
>;
type DisplayMsg = Message<DisplayData, DisplayControl, DisplayMgmt>;

actor_scheduler::ports! {
    Coordinator {
        render: RenderMsg -> RendererHandle<PlatformPixel, WindowMeta>,
        request_window: DisplayMsg -> ActorHandle<DisplayData, DisplayControl, DisplayMgmt>,
        present: DisplayMsg -> ActorHandle<DisplayData, DisplayControl, DisplayMgmt>,
        rendered: RenderedResponse -> GreenSender<RenderedResponse>,
    }
}

/// Owns the render coordinator plus the one piece of state that used to live on `EngineCore`:
/// the frame counter, because its only consumer — FPS telemetry — is a coordinator → vsync edge
/// now (§7.3 of the migration doc).
pub(crate) struct CoordinatorCore {
    render: RenderCoordinator,
    frame_number: u64,
}

impl CoordinatorCore {
    pub(crate) fn new() -> Self {
        Self {
            render: RenderCoordinator::new(),
            frame_number: 0,
        }
    }

    /// Whether the buffer is currently in hand. Test-only window into otherwise-private state,
    /// mirroring `RenderCoordinator::holds_buffer` (and the old `EngineCore::holds_buffer`).
    #[cfg(test)]
    pub(crate) fn holds_buffer(&self) -> bool {
        self.render.holds_buffer()
    }

    /// Map a `render_coordinator::Step` onto the port it belongs on.
    ///
    /// Calls `request_sent` the instant a `RequestWindow` is decided, not once it is delivered —
    /// see the module doc's "the delivery-feedback seam dissolved".
    fn route(&mut self, step: Step, out: &mut CoordinatorOut) {
        match step {
            Step::Idle => {}
            Step::RequestWindow => {
                out.request_window = Some(Message::Management(DisplayMgmt::RequestWindow));
                self.render.request_sent();
            }
            Step::Render(request) => out.render = Some(Message::Data(request)),
        }
    }

    /// Hand the drawn buffer to the `present` port and vsync's telemetry port together. Two
    /// different ports set in one step is fine — `Wiring::flush` delivers them independently;
    /// it is only ever a problem for two messages sharing *one* port in one step, which does not
    /// happen here.
    fn present_cooked_frame(
        &mut self,
        render_time: Duration,
        window: Window,
        out: &mut CoordinatorOut,
    ) {
        out.present = Some(Message::Data(DisplayData::Present { window }));

        self.frame_number += 1;
        out.rendered = Some(RenderedResponse {
            frame_number: self.frame_number,
            rendered_at: Instant::now(),
        });

        if self.frame_number.is_multiple_of(LOG_FRAME_INTERVAL) {
            log::info!("Frame {}: render={:?}", self.frame_number, render_time);
        }
    }
}

impl Transducer for CoordinatorCore {
    type Control = std::convert::Infallible;
    type Management = RenderResponse<PlatformPixel, WindowMeta>;
    type Data = CoordinatorData;
    type Out = CoordinatorOut;

    fn step_data(&mut self, data: CoordinatorData) -> Result<CoordinatorOut, HandlerError> {
        let mut out = CoordinatorOut::default();
        let step = match data {
            CoordinatorData::Submit(scene) => self.render.submit(scene),
            CoordinatorData::Granted(window) => self.render.granted(window),
            CoordinatorData::Advance => self.render.advance(),
        };
        self.route(step, &mut out);
        Ok(out)
    }

    fn step_management(
        &mut self,
        response: RenderResponse<PlatformPixel, WindowMeta>,
    ) -> Result<CoordinatorOut, HandlerError> {
        let mut out = CoordinatorOut::default();
        let Completed { present, next } = self.render.completed(response);
        match present {
            Some((window, render_time)) => self.present_cooked_frame(render_time, window, &mut out),
            // The renderer was paused, so it handed the buffer back unrendered. Presenting it
            // would blit whatever stale pixels it still holds; keeping it is the whole point of
            // the frame coming back at all. The coordinator has retained it.
            None => {
                log::debug!("Render skipped (paused); retaining the buffer unpresented")
            }
        }
        self.route(next, &mut out);
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    //! `CoordinatorCore` driven directly — no `Node`, no channels, no scheduler. Mirrors
    //! `render_coordinator.rs`'s own `mod tests` one layer up: these pin the *port mapping*
    //! (`Step` → `CoordinatorOut`), not the pull protocol itself, which is already covered there.

    use super::*;
    use crate::api::public::WindowId;
    use crate::display::messages::Surface;
    use crate::display::window_keeper::WindowKeeper;

    /// The lattice the fixture scene is compiled for. These tests never bake
    /// it — they assert which port fired, not what reached a buffer — so it
    /// only has to be a legal frame; the windows they grant are this size.
    const FIXTURE_FRAME: [u32; 2] = [100, 100];

    fn manifold() -> Scene {
        crate::testing::black_scene(FIXTURE_FRAME)
    }

    fn one_to_one_window() -> Window {
        let mut keeper = WindowKeeper::new();
        keeper.surface_changed(Surface {
            id: WindowId(1),
            width_px: 100,
            height_px: 100,
            frame_width: 100,
            frame_height: 100,
            scale: 1.0,
        });
        keeper.request();
        keeper.pending_grant().expect("a buffer is at rest")
    }

    #[test]
    fn a_granted_window_turns_a_submitted_scene_into_a_render_port() {
        let mut core = CoordinatorCore::new();
        let out = core.step_data(CoordinatorData::Submit(manifold())).unwrap();
        assert!(
            out.request_window.is_some(),
            "a scene with no buffer in hand must ask for one"
        );

        let out = core
            .step_data(CoordinatorData::Granted(one_to_one_window()))
            .unwrap();
        assert!(
            out.render.is_some(),
            "a granted window with a pending scene must render immediately"
        );
        assert!(
            !core.holds_buffer(),
            "the buffer left with the render request"
        );
    }

    #[test]
    fn render_complete_with_a_presentable_frame_emits_present_and_vsync_telemetry() {
        let mut core = CoordinatorCore::new();
        core.step_data(CoordinatorData::Submit(manifold())).unwrap();
        let out = core
            .step_data(CoordinatorData::Granted(one_to_one_window()))
            .unwrap();
        let Message::Data(request) = out.render.expect("expected a render request") else {
            panic!("expected Message::Data");
        };

        let out = core
            .step_management(RenderResponse {
                frame: request.frame,
                render_time: Some(Duration::from_millis(1)),
                meta: request.meta,
            })
            .unwrap();

        assert!(out.present.is_some(), "a drawn frame must be presented");
        let rendered = out
            .rendered
            .expect("presenting a frame must report FPS telemetry to vsync");
        assert_eq!(
            rendered.frame_number, 1,
            "the first presented frame must be numbered 1, not left at its zero default"
        );
    }

    #[test]
    fn the_frame_counter_advances_once_per_presented_frame() {
        let mut core = CoordinatorCore::new();

        for expected in 1..=3u64 {
            core.step_data(CoordinatorData::Submit(manifold())).unwrap();
            let out = core
                .step_data(CoordinatorData::Granted(one_to_one_window()))
                .unwrap();
            let Message::Data(request) = out.render.expect("expected a render request") else {
                panic!("expected Message::Data");
            };

            let out = core
                .step_management(RenderResponse {
                    frame: request.frame,
                    render_time: Some(Duration::from_millis(1)),
                    meta: request.meta,
                })
                .unwrap();

            let rendered = out.rendered.expect("a drawn frame reports telemetry");
            assert_eq!(
                rendered.frame_number, expected,
                "the counter must advance by exactly one per presented frame"
            );
        }
    }

    #[test]
    fn coordinator_data_debug_names_its_variant() {
        assert_eq!(format!("{:?}", CoordinatorData::Advance), "Advance");
        assert!(format!("{:?}", CoordinatorData::Submit(manifold())).contains("Submit"));
        assert!(format!("{:?}", CoordinatorData::Granted(one_to_one_window())).contains("Granted"));
    }

    /// The delivery-feedback seam dissolved (module doc): `route` calls `request_sent` the
    /// instant it decides to ask, not once wiring confirms delivery — so a second submission
    /// arriving before any reply must not repeat the ask.
    #[test]
    fn a_request_window_step_latches_immediately_on_emission() {
        let mut core = CoordinatorCore::new();
        let out = core.step_data(CoordinatorData::Submit(manifold())).unwrap();
        assert!(out.request_window.is_some());

        let out = core.step_data(CoordinatorData::Submit(manifold())).unwrap();
        assert!(
            out.request_window.is_none(),
            "the latch must already be set from the first ask"
        );
    }
}

#[cfg(test)]
mod node_tests {
    //! The coordinator as a real `Node`: a real driver scheduler + spy (owning a real
    //! `WindowKeeper`, exactly as `engine_troupe.rs`'s old `Rig` did) and a real renderer
    //! scheduler + spy on the other side, with `node.poll()` standing in for a `Host`'s sweep.
    //!
    //! These are the render-protocol interaction tests that used to live in `engine_troupe.rs`'s
    //! `Rig`, driving `EngineHandler` directly — moved here verbatim in name and intent now that
    //! the coordinator is its own `Node` rather than logic embedded in the engine. They found
    //! real races (see each test's own doc); losing that coverage in the move is not acceptable.

    use super::*;
    use crate::api::public::WindowId;
    use crate::display::messages::Surface;
    use crate::display::window_keeper::{Presented, WindowKeeper};
    use actor_scheduler::mealy::{Lanes, NoLane, Node, Step as NodeStep};
    use actor_scheduler::spsc::{spsc_channel, SpscReceiver, SpscSender};
    use actor_scheduler::{
        Actor, ActorScheduler, ActorStatus, HandlerResult, SchedulerParams, SystemStatus,
    };
    use pixelflow_graphics::render::renderer::{RenderControl, RenderManagement};
    use pixelflow_graphics::render::Frame;
    use std::collections::VecDeque;
    use std::convert::Infallible;

    /// Scheduler tuning for the fixture: nothing here approaches the buffer, and the burst
    /// limit only has to be big enough that one pump drains a test's worth of messages.
    const LANE_BURST: usize = 16;
    const LANE_BUFFER: usize = 16;

    /// A window size, as the tests talk about them.
    type Size = (u32, u32);

    #[derive(Default)]
    struct RendererSpy {
        requests: Vec<WindowMeta>,
    }

    impl Actor<RenderRequest<PlatformPixel, WindowMeta>, RenderControl, RenderManagement>
        for RendererSpy
    {
        fn handle_data(&mut self, req: RenderRequest<PlatformPixel, WindowMeta>) -> HandlerResult {
            self.requests.push(req.meta);
            Ok(())
        }
        fn handle_control(&mut self, _msg: RenderControl) -> HandlerResult {
            Ok(())
        }
        fn handle_management(&mut self, _msg: RenderManagement) -> HandlerResult {
            Ok(())
        }
        fn handle_os(&mut self, _status: SystemStatus) -> Result<ActorStatus, HandlerError> {
            Ok(ActorStatus::Idle)
        }
    }

    /// A driver, minus the platform: real buffer ownership, no blit.
    #[derive(Default)]
    struct DriverSpy {
        keeper: WindowKeeper,
        /// Buffers lent out, waiting to reach the coordinator. A real `PlatformActor` sends
        /// these straight back over its own engine handle; the fixture has no such handle, so
        /// `Rig` drains this in `pump`.
        granted: Vec<Window>,
        /// Sizes actually blitted — excluding superseded buffers, which never reach a screen.
        blitted: Vec<Size>,
    }

    impl DriverSpy {
        /// What `PlatformActor::flush` does at the end of every handler: issue whatever grant
        /// has come due.
        fn grant(&mut self) {
            if let Some(window) = self.keeper.pending_grant() {
                self.granted.push(window);
            }
        }
    }

    impl Actor<DisplayData, DisplayControl, DisplayMgmt> for DriverSpy {
        fn handle_data(&mut self, msg: DisplayData) -> HandlerResult {
            let DisplayData::Present { window } = msg;
            match self.keeper.presented(window) {
                Presented::Blit(window) => {
                    self.blitted.push((window.width_px, window.height_px));
                    self.keeper.rest(window);
                }
                Presented::Superseded => {}
            }
            self.grant();
            Ok(())
        }
        fn handle_control(&mut self, _msg: DisplayControl) -> HandlerResult {
            Ok(())
        }
        fn handle_management(&mut self, msg: DisplayMgmt) -> HandlerResult {
            if matches!(msg, DisplayMgmt::RequestWindow) {
                self.keeper.request();
                self.grant();
            }
            Ok(())
        }
        fn handle_os(&mut self, _status: SystemStatus) -> Result<ActorStatus, HandlerError> {
            Ok(ActorStatus::Idle)
        }
    }

    type CoordinatorNode = Node<
        CoordinatorCore,
        CoordinatorWiring,
        SpscReceiver<CoordinatorData>,
        NoLane<Infallible>,
        SpscReceiver<RenderResponse<PlatformPixel, WindowMeta>>,
    >;

    struct Rig {
        node: CoordinatorNode,
        data_tx: SpscSender<CoordinatorData>,
        mgmt_tx: SpscSender<RenderResponse<PlatformPixel, WindowMeta>>,
        render_sched: ActorScheduler<
            RenderRequest<PlatformPixel, WindowMeta>,
            RenderControl,
            RenderManagement,
        >,
        render_spy: RendererSpy,
        driver_sched: ActorScheduler<DisplayData, DisplayControl, DisplayMgmt>,
        driver_spy: DriverSpy,
        /// Never asserted on: these tests are not about FPS telemetry. Held only so the
        /// coordinator's sends have somewhere to land.
        _vsync_rx: SpscReceiver<RenderedResponse>,
        /// Renders the renderer has been asked for and not yet answered, oldest first.
        in_flight: VecDeque<WindowMeta>,
        request_log: Vec<Size>,
    }

    impl Rig {
        fn new() -> Self {
            let mut driver_builder = actor_scheduler::ActorBuilder::new(LANE_BUFFER, None);
            let driver_rw = driver_builder.add_producer();
            let driver_p = driver_builder.add_producer();
            let driver_sched = driver_builder
                .build_with_burst(LANE_BURST, actor_scheduler::ShutdownMode::default());

            let (renderer, render_sched) = ActorScheduler::new(LANE_BURST, LANE_BUFFER);

            // A `Waker` with nobody listening is harmless, so a throwaway scheduler dropped
            // immediately after minting one is enough — the plain channel below never needs a
            // real host to wake.
            let waker = {
                let (handle, _sched) =
                    ActorScheduler::<Infallible, Infallible, Infallible>::new(1, 1);
                handle.waker()
            };
            let (vsync_tx, vsync_rx) = spsc_channel::<RenderedResponse>(8);
            let vsync = GreenSender::new(vsync_tx, waker);

            let (data_tx, data_rx) = spsc_channel::<CoordinatorData>(64);
            let (mgmt_tx, mgmt_rx) = spsc_channel::<RenderResponse<PlatformPixel, WindowMeta>>(64);

            let node = Node::new_with_lanes(
                CoordinatorCore::new(),
                Lanes {
                    control: NoLane::new(),
                    management: mgmt_rx,
                    data: data_rx,
                },
                CoordinatorWiring {
                    render: renderer,
                    request_window: driver_rw,
                    present: driver_p,
                    rendered: vsync,
                },
                SchedulerParams::DEFAULT,
            );

            Self {
                node,
                data_tx,
                mgmt_tx,
                render_sched,
                render_spy: RendererSpy::default(),
                driver_sched,
                driver_spy: DriverSpy::default(),
                _vsync_rx: vsync_rx,
                in_flight: VecDeque::new(),
                request_log: Vec::new(),
            }
        }

        /// Drain the node's own outbox and lanes until it goes quiet — one input's worth of
        /// work, mirroring the old `Rig::feed`'s synchronous `handle_data` call.
        fn drain_node(&mut self) {
            while let NodeStep::Ran = self.node.poll() {}
        }

        fn feed(&mut self, data: CoordinatorData) {
            self.data_tx
                .try_send(data)
                .expect("coordinator data lane has room");
            self.drain_node();
        }

        /// Carry every message in flight to its destination, in both directions, until nothing
        /// moves. The request → grant → render loop takes several hops, so a single pass would
        /// leave the system mid-conversation and the assertions would read a partial state.
        fn pump(&mut self) {
            loop {
                let _ = self.render_sched.poll_once(&mut self.render_spy);
                for meta in self.render_spy.requests.drain(..) {
                    self.request_log.push((meta.width_px, meta.height_px));
                    self.in_flight.push_back(meta);
                }

                let _ = self.driver_sched.poll_once(&mut self.driver_spy);
                let granted = std::mem::take(&mut self.driver_spy.granted);
                if granted.is_empty() {
                    return;
                }
                for window in granted {
                    self.feed(CoordinatorData::Granted(window));
                }
            }
        }

        /// The platform reported a window at this geometry. Both halves of what
        /// `PlatformActor::flush` does: the keeper builds the buffer, the coordinator is nudged
        /// to look. Used for the initial window and for resizes alike, because the driver
        /// treats them identically.
        fn surface(&mut self, (width_px, height_px): Size) {
            let surface = Surface {
                id: WindowId(1),
                width_px,
                height_px,
                frame_width: width_px,
                frame_height: height_px,
                scale: 1.0,
            };
            self.driver_spy.keeper.surface_changed(surface);
            // A resize can answer a request that was refused while the old buffer was out, which
            // is why `PlatformActor` flushes here too.
            self.driver_spy.grant();
            self.feed(CoordinatorData::Advance);
            self.pump();
        }

        /// Answer the oldest outstanding render, returning its `meta` untouched — which is the
        /// renderer's actual contract.
        fn complete_render(&mut self) {
            self.pump();
            let meta = self
                .in_flight
                .pop_front()
                .expect("a render must be outstanding to complete");
            let sent = self.mgmt_tx.try_send(RenderResponse {
                frame: Frame::new(meta.width_px, meta.height_px),
                render_time: Some(Duration::from_millis(1)),
                meta,
            });
            assert!(sent.is_ok(), "coordinator management lane has room");
            self.drain_node();
            self.pump();
        }

        fn app_frame(&mut self) {
            self.queue_frame();
            self.pump();
        }

        /// A frame, without letting anything be delivered downstream. Two of these back to back
        /// is one round in which the coordinator reacts twice before the driver reacts once —
        /// which is ordinary, since they are separate actors.
        fn queue_frame(&mut self) {
            self.feed(CoordinatorData::Submit(manifold()));
        }

        /// A vsync tick (relayed through the engine as `CoordinatorData::Advance`), which is
        /// what re-drives a request the driver could not answer.
        fn tick(&mut self) {
            self.feed(CoordinatorData::Advance);
            self.pump();
        }

        /// Render requests emitted since the last call.
        fn render_requests(&mut self) -> Vec<Size> {
            self.pump();
            std::mem::take(&mut self.request_log)
        }

        /// Frames that actually reached the screen since the last call.
        fn blitted(&mut self) -> Vec<Size> {
            self.pump();
            std::mem::take(&mut self.driver_spy.blitted)
        }

        fn holds_buffer(&self) -> bool {
            self.node.actor().holds_buffer()
        }
    }

    /// The lattice the fixture scene is compiled for. These tests never bake
    /// it — they assert which port fired, not what reached a buffer — so it
    /// only has to be a legal frame; the windows they grant are this size.
    const FIXTURE_FRAME: [u32; 2] = [100, 100];

    /// A constant black scene — these tests care about which buffer a render is aimed at, not
    /// what is in it.
    fn manifold() -> Scene {
        crate::testing::black_scene(FIXTURE_FRAME)
    }

    /// The steady state: the coordinator asks for the buffer only when it has something to
    /// draw, draws, hands it straight back, and asks again next frame.
    #[test]
    fn frames_circulate_through_request_render_and_present() {
        let mut rig = Rig::new();
        rig.surface((100, 100));

        rig.app_frame();
        assert_eq!(rig.render_requests(), vec![(100, 100)]);

        rig.complete_render();
        assert_eq!(rig.blitted(), vec![(100, 100)]);

        rig.app_frame();
        assert_eq!(
            rig.render_requests(),
            vec![(100, 100)],
            "Present returned the buffer to the driver, so the next frame can have it"
        );
    }

    /// Nothing is requested until there is something to draw. If the coordinator held the
    /// buffer while idle, "who has the buffer" would stop meaning "who is drawing", and that
    /// equivalence is the entire bound on the loop.
    #[test]
    fn an_idle_engine_does_not_hold_the_buffer() {
        let mut rig = Rig::new();
        rig.surface((100, 100));

        rig.tick();
        rig.tick();
        assert!(rig.render_requests().is_empty());
        assert!(
            !rig.holds_buffer(),
            "no manifold to draw, so no reason to be holding the buffer"
        );
    }

    /// Vsync keeps asking for frames at 60Hz regardless of how long a render takes, so a new
    /// manifold routinely arrives while one is in flight. The refusal must keep it: dropping it
    /// would leave the app's last state change unrendered with nothing following to correct it.
    #[test]
    fn a_manifold_arriving_mid_render_is_kept_not_dropped() {
        let mut rig = Rig::new();
        rig.surface((100, 100));

        rig.app_frame();
        assert_eq!(rig.render_requests(), vec![(100, 100)]);

        rig.app_frame();
        assert!(
            rig.render_requests().is_empty(),
            "the in-flight render holds the edge's only credit"
        );

        rig.complete_render();
        assert_eq!(
            rig.render_requests(),
            vec![(100, 100)],
            "the deferred manifold renders once the credit comes back"
        );
    }

    /// The race the generation stamp exists for, end-to-end across both sides: a resize while
    /// the buffer is out with the renderer. The old-size frame must not reach the screen, and
    /// the next one must be at the new size.
    #[test]
    fn a_resize_mid_render_keeps_the_old_frame_off_the_screen() {
        let mut rig = Rig::new();
        rig.surface((100, 100));

        rig.app_frame();
        assert_eq!(rig.render_requests(), vec![(100, 100)]);

        // Resize lands while that render is in flight.
        rig.surface((200, 200));

        rig.complete_render();
        assert!(
            rig.blitted().is_empty(),
            "the old-size buffer is superseded and must not be blitted"
        );

        rig.app_frame();
        assert_eq!(
            rig.render_requests(),
            vec![(200, 200)],
            "and the driver's replacement buffer is what the next frame draws into"
        );
        rig.complete_render();
        assert_eq!(rig.blitted(), vec![(200, 200)]);
    }

    /// A resize with nothing in flight: the next frame simply comes out at the new size.
    #[test]
    fn a_resize_between_frames_renders_at_the_new_size() {
        let mut rig = Rig::new();
        rig.surface((100, 100));

        rig.app_frame();
        rig.complete_render();
        assert_eq!(rig.blitted(), vec![(100, 100)]);
        let _ = rig.render_requests(); // drain, so the next assertion is about what follows

        rig.surface((200, 200));
        rig.app_frame();
        assert_eq!(rig.render_requests(), vec![(200, 200)]);
    }

    /// A resize while a render is in flight, *with a frame already queued behind it*. The queued
    /// frame makes the coordinator ask for a buffer it cannot yet use; the resize then allocates
    /// one, so the ask is answered immediately and the coordinator is holding a second buffer
    /// while the first is still out. The old completion then arrives with nowhere to go.
    ///
    /// In a debug build that trips the "one buffer" assertion. In release, a *paused*
    /// completion — the arm that keeps its buffer rather than presenting it — overwrites the
    /// replacement, and since the driver has already handed its only current buffer over, the
    /// terminal never draws again.
    #[test]
    fn a_resize_does_not_grant_a_second_buffer_mid_render() {
        let mut rig = Rig::new();
        rig.surface((100, 100));

        rig.app_frame();
        assert_eq!(rig.render_requests(), vec![(100, 100)]);

        // A second frame queues behind the in-flight render, and asks for a buffer.
        rig.app_frame();
        // The resize allocates one, which could answer that ask.
        rig.surface((200, 200));

        assert!(
            !rig.holds_buffer(),
            "a buffer is already out with the renderer; a second one must not be granted"
        );

        // The original render completes into a coordinator that still has exactly one buffer's
        // worth of state to reconcile.
        rig.complete_render();
        assert!(rig.blitted().is_empty(), "the old-size frame is superseded");
        assert_eq!(
            rig.render_requests(),
            vec![(200, 200)],
            "and the queued frame renders into the resize's buffer"
        );
    }

    /// The same failure one step earlier in the conversation: two asks issued *before* the
    /// driver has answered either. The credit guard cannot see this one — no render is in flight
    /// yet, so both asks are legitimate at the moment they are made.
    ///
    /// The driver answers the first and has nothing for the second, which leaves its "somebody
    /// is waiting" latch set with nobody actually waiting any more. The next buffer to appear —
    /// a resize allocation, while the granted one is out being drawn into — is then handed over
    /// unasked, and the coordinator is holding two.
    #[test]
    fn a_second_ask_before_the_first_is_answered_does_not_earn_a_second_grant() {
        let mut rig = Rig::new();
        rig.surface((100, 100));

        // Two asks in one round, before the driver reacts to either.
        rig.queue_frame();
        rig.queue_frame();
        rig.pump();
        assert_eq!(
            rig.render_requests(),
            vec![(100, 100)],
            "the grant that did arrive is rendering"
        );

        // A resize now allocates a buffer that a stale ask would collect.
        rig.surface((200, 200));
        assert!(
            !rig.holds_buffer(),
            "the buffer is out with the renderer; an unanswered duplicate ask must not \
             collect the resize's replacement"
        );

        // The old buffer goes back and is discarded as superseded; the app answers the resize
        // with a new frame, which asks properly and gets the replacement.
        rig.complete_render();
        rig.app_frame();
        assert_eq!(
            rig.render_requests(),
            vec![(200, 200)],
            "and the next frame draws into the replacement, asked for properly"
        );
    }

    /// Two buffers can never be out at once — the old fixture had to model that possibility and
    /// order the returns. It is now a property of ownership: the driver cannot lend what it is
    /// not holding, so however many times it is asked, at most one buffer is out.
    #[test]
    fn the_driver_never_lends_a_second_buffer() {
        let mut rig = Rig::new();
        rig.surface((100, 100));

        rig.app_frame();
        assert_eq!(rig.render_requests(), vec![(100, 100)]);

        // Ask repeatedly while the buffer is out with the renderer.
        for _ in 0..5 {
            rig.tick();
        }
        assert!(
            !rig.holds_buffer(),
            "no grant can arrive while the buffer is out, however often it is requested"
        );

        rig.complete_render();
        assert_eq!(rig.blitted(), vec![(100, 100)]);
    }
}
