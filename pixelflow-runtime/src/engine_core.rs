//! `EngineCore` — the pure decision logic behind `EngineHandler` (`engine_troupe.rs`).
//!
//! Mirrors the `VsyncCore`/`RasterCore` split (`vsync_actor.rs`, `pixelflow-graphics`'s
//! `rasterizer/actor.rs`): a `step_*` call takes a message and **returns** what to emit, so the
//! whole mediator — request/render/present, the resize races, the app/vsync/driver relays — is
//! table-testable with no scheduler, no handles, and no threads in the loop. `EngineHandler`
//! shrinks to the thin adapter that owns the real channels and flushes the returned word.

use crate::api::private::{EngineControl, EngineData};
use crate::api::public::{
    AppData, AppManagement, EngineEvent, EngineEventControl, EngineEventData,
    EngineEventManagement, WindowId,
};
use crate::display::messages::{
    DisplayControl, DisplayData, DisplayEvent, DisplayMgmt, Window, WindowMeta,
};
use crate::input::MouseButton;
use crate::platform::PlatformPixel;
use crate::render_coordinator::{Completed, RenderCoordinator, Step};
use crate::vsync_actor::{RenderedResponse, VsyncCommand};
use actor_scheduler::mealy::Transducer;
use actor_scheduler::HandlerError;
use pixelflow_graphics::render::rasterizer::RenderRequest;
use std::time::{Duration, Instant};

const LOG_FRAME_INTERVAL: u64 = 60;

/// One optional slot per downstream edge. Default (all `None`, `quit: false`) is the silent
/// step — most inputs move state without telling any peer.
#[derive(Default)]
pub(crate) struct EngineOut {
    pub(crate) app: Option<EngineEvent>,
    pub(crate) driver_data: Option<DisplayData>,
    pub(crate) driver_control: Option<DisplayControl>,
    pub(crate) driver_mgmt: Option<DisplayMgmt>,
    pub(crate) vsync_data: Option<RenderedResponse>,
    pub(crate) vsync_control: Option<VsyncCommand>,
    pub(crate) rasterizer: Option<RenderRequest<PlatformPixel, WindowMeta>>,
    /// Run the shutdown cascade: vsync, rasterizer, forwarder, app-drop, driver, self.
    pub(crate) quit: bool,
}

/// Pure engine mediator: request/render/present bookkeeping and the app/driver/vsync relay
/// decisions, with no actor handle or channel in it. No self-port — retries of a dropped
/// `RequestWindow` ride the next VSync tick, exactly as today, so there is no continuation to
/// take.
pub(crate) struct EngineCore {
    render: RenderCoordinator,
    /// Frame counter for VSync feedback. See the field doc on the old `EngineHandler` copy of
    /// this (moved verbatim): it stays here rather than moving to the rasterizer because the
    /// only consumer, FPS telemetry, is a `rasterizer → vsync` edge that does not exist until
    /// the coordinator is its own node.
    frame_number: u64,
}

impl EngineCore {
    pub(crate) fn new() -> Self {
        Self {
            render: RenderCoordinator::new(),
            frame_number: 0,
        }
    }

    /// Whether the render coordinator currently holds the driver's buffer. Test-only window
    /// into otherwise-private state, mirroring `RenderCoordinator::holds_buffer`.
    #[cfg(test)]
    pub(crate) fn holds_buffer(&self) -> bool {
        self.render.holds_buffer()
    }

    /// A `Step::RequestWindow` port actually reached the driver.
    ///
    /// A pure core cannot see send outcomes — that is the shell's job, since only it holds the
    /// handle and calls `send`. This is the hand-written stand-in for the mealy runtime's
    /// parked-outbox/`Flush::Blocked` bookkeeping until the engine runs as a green node: the
    /// shell calls this after a successful send, and [`Self::render_undeliverable`] after a
    /// failed one, so the coordinator's latch tracks what actually left rather than what the
    /// core merely emitted.
    pub(crate) fn request_delivered(&mut self) {
        self.render.request_sent();
    }

    /// A `Step::Render` port could not be delivered (missing rasterizer, or the send failed).
    /// See [`Self::request_delivered`] for why this lives on the shell side of the seam.
    pub(crate) fn render_undeliverable(&mut self) {
        self.render.render_send_failed();
    }

    /// Map a `render_coordinator::Step` onto the port it belongs on.
    fn route(step: Step, out: &mut EngineOut) {
        match step {
            Step::Idle => {}
            Step::RequestWindow => out.driver_mgmt = Some(DisplayMgmt::RequestWindow),
            Step::Render(request) => out.rasterizer = Some(request),
        }
    }

    fn step_app_data(&mut self, app_data: AppData, out: &mut EngineOut) {
        match app_data {
            AppData::RenderSurface(manifold) | AppData::RenderSurfaceU32(manifold) => {
                log::debug!("Engine: Received RenderSurface from app");
                // The app has provided its compute graph, so permit VSync to request another
                // frame without waiting for rasterization to finish.
                out.vsync_control = Some(VsyncCommand::ReturnToken);

                // Keep-latest port: the newest compute graph replaces any frame that hasn't
                // started rendering yet, and renders now if a window is free to draw into.
                Self::route(self.render.submit(manifold), out);
            }
            AppData::Skipped => {
                // App says nothing to render - return token anyway
                out.vsync_control = Some(VsyncCommand::ReturnToken);
            }
        }
    }

    /// Hand the drawn buffer back to the driver to be shown, and tell vsync it landed.
    ///
    /// This *is* the return: the driver is the buffer's resting owner, so there is no separate
    /// acknowledgement to wait for. FPS telemetry rides the frame actually reaching the driver
    /// port, same as before — it used to ride `PresentComplete`, which no longer exists.
    fn present_cooked_frame(&mut self, render_time: Duration, window: Window, out: &mut EngineOut) {
        out.driver_data = Some(DisplayData::Present { window });

        self.frame_number += 1;
        out.vsync_data = Some(RenderedResponse {
            frame_number: self.frame_number,
            rendered_at: Instant::now(),
        });

        if self.frame_number.is_multiple_of(LOG_FRAME_INTERVAL) {
            log::info!("Frame {}: render={:?}", self.frame_number, render_time);
        }
    }

    fn step_driver_event(&mut self, event: DisplayEvent, out: &mut EngineOut) {
        match event {
            // Both window-lifecycle events are pure relays: the driver has already built the
            // buffer for the new geometry by the time this arrives, so there is nothing here to
            // take delivery of, stamp, or retire. What is left is telling the app its size.
            DisplayEvent::WindowCreated { surface } => {
                log::debug!(
                    "Relaying WindowCreated: id={}, {}x{}, scale={}",
                    surface.id.0,
                    surface.width_px,
                    surface.height_px,
                    surface.scale
                );

                // The app may already have handed us something to draw, in which case this is
                // the first moment a buffer can exist to draw it into.
                Self::route(self.render.advance(), out);

                out.app = Some(EngineEvent::Control(EngineEventControl::WindowCreated {
                    id: surface.id,
                    width_px: surface.width_px,
                    height_px: surface.height_px,
                    scale: surface.scale,
                }));
            }
            DisplayEvent::Resized { surface } => {
                log::debug!(
                    "Relaying Resized: id={}, {}x{}",
                    surface.id.0,
                    surface.width_px,
                    surface.height_px
                );

                // A buffer we are currently holding is now the wrong size, but it is not dropped
                // here: it goes back on the next `Present` and the driver recognises it as
                // superseded. Rendering into it once more first is harmless and one frame
                // cheaper than reaching for the new one mid-flight.
                Self::route(self.render.advance(), out);

                out.app = Some(EngineEvent::Control(EngineEventControl::Resized {
                    id: surface.id,
                    width_px: surface.width_px,
                    height_px: surface.height_px,
                }));
            }
            DisplayEvent::Key {
                symbol,
                modifiers,
                text,
                ..
            } => {
                out.app = Some(EngineEvent::Management(EngineEventManagement::KeyDown {
                    key: symbol,
                    mods: modifiers,
                    text,
                }));
            }
            DisplayEvent::MouseButtonPress { button, x, y, .. } => {
                out.app = Some(EngineEvent::Management(EngineEventManagement::MouseClick {
                    x: x as u32,
                    y: y as u32,
                    button: convert_mouse_button(button),
                }));
            }
            DisplayEvent::MouseButtonRelease { button, x, y, .. } => {
                out.app = Some(EngineEvent::Management(
                    EngineEventManagement::MouseRelease {
                        x: x as u32,
                        y: y as u32,
                        button: convert_mouse_button(button),
                    },
                ));
            }
            DisplayEvent::MouseMove {
                x, y, modifiers, ..
            } => {
                out.app = Some(EngineEvent::Management(EngineEventManagement::MouseMove {
                    x: x as u32,
                    y: y as u32,
                    mods: modifiers,
                }));
            }
            DisplayEvent::MouseScroll {
                dx,
                dy,
                x,
                y,
                modifiers,
                ..
            } => {
                out.app = Some(EngineEvent::Management(
                    EngineEventManagement::MouseScroll {
                        x: x as u32,
                        y: y as u32,
                        dx,
                        dy,
                        mods: modifiers,
                    },
                ));
            }
            DisplayEvent::CloseRequested { .. } => {
                log::debug!("Close requested");
                // Notify the app, then the shell drops the handle as part of the shutdown
                // cascade `quit` triggers — cleanup goes in the app's Drop impl.
                out.app = Some(EngineEvent::Control(EngineEventControl::CloseRequested));
                out.quit = true;
            }
            DisplayEvent::FocusGained { .. } => {
                out.app = Some(EngineEvent::Management(EngineEventManagement::FocusGained));
            }
            DisplayEvent::FocusLost { .. } => {
                out.app = Some(EngineEvent::Management(EngineEventManagement::FocusLost));
            }
            DisplayEvent::PasteData { text } => {
                out.app = Some(EngineEvent::Management(EngineEventManagement::Paste(text)));
            }
            DisplayEvent::ScaleChanged { id, scale } => {
                log::debug!("Relaying ScaleChanged: id={}, scale={}", id.0, scale);
                out.app = Some(EngineEvent::Control(EngineEventControl::ScaleChanged {
                    id,
                    scale,
                }));
            }
            DisplayEvent::ClipboardDataRequested => {
                unimplemented!("Clipboard data requested")
            }
            DisplayEvent::WindowDestroyed { .. } => {
                unimplemented!("window destroyed, forward to app unimplemented");
            }
        }
    }
}

impl Transducer for EngineCore {
    type Control = EngineControl;
    type Management = AppManagement;
    type Data = EngineData;
    type Out = EngineOut;

    fn step_data(&mut self, data: EngineData) -> Result<EngineOut, HandlerError> {
        let mut out = EngineOut::default();
        match data {
            EngineData::FromApp(app_data) => self.step_app_data(app_data, &mut out),
            EngineData::FromDriver(event) => self.step_driver_event(event, &mut out),
            EngineData::VSync {
                timestamp,
                target_timestamp,
                refresh_interval,
            } => {
                // ALWAYS request frame from app (app builds compute graphs fast). Token bucket
                // is managed atomically by VSync.
                out.app = Some(EngineEvent::Data(EngineEventData::RequestFrame {
                    timestamp,
                    target_timestamp,
                    refresh_interval,
                }));

                // The tick is also what retries a request that was dropped in transit.
                Self::route(self.render.advance(), &mut out);
            }
            EngineData::RenderComplete(response) => {
                // No staleness check here any more. Whether this buffer is still the one the
                // driver wants is the driver's question, asked against state the driver owns —
                // and it has to be asked there regardless, because a resize can land after this
                // point too. Asking it in both places would be two answers that can disagree.
                let Completed { present, next } = self.render.completed(response);
                match present {
                    Some((window, render_time)) => {
                        self.present_cooked_frame(render_time, window, &mut out)
                    }
                    // The rasterizer was paused, so it handed the buffer back unrendered.
                    // Presenting it would blit whatever stale pixels it still holds; keeping it
                    // is the whole point of the frame coming back at all. The coordinator has
                    // retained it.
                    None => {
                        log::debug!("Render skipped (paused); retaining the buffer unpresented")
                    }
                }
                Self::route(next, &mut out);
            }
            EngineData::WindowGranted(window) => {
                Self::route(self.render.granted(window), &mut out);
            }
        }
        Ok(out)
    }

    fn step_control(&mut self, ctrl: EngineControl) -> Result<EngineOut, HandlerError> {
        let mut out = EngineOut::default();
        match ctrl {
            EngineControl::Quit => out.quit = true,
            EngineControl::UpdateRefreshRate(rr) => {
                out.vsync_control = Some(VsyncCommand::UpdateRefreshRate(rr));
            }
            // Handle-carrying: the shell intercepts this before the core ever sees it, since a
            // pure core has no field to hold the handle in. Only the type split arriving with
            // the port/wiring phase removes this arm.
            EngineControl::VsyncActorReady(..) => {
                unreachable!("handled by the engine shell")
            }
            // A real port with a real consumer; policy later.
            EngineControl::GreenStuck(supervision) => {
                log::error!("green node stuck: {supervision:?}");
            }
            EngineControl::DriverAck => {
                unimplemented!("DriverAck not yet implemented");
            }
        }
        Ok(out)
    }

    fn step_management(&mut self, mgmt: AppManagement) -> Result<EngineOut, HandlerError> {
        let mut out = EngineOut::default();
        match mgmt {
            // Handle-carrying / bootstrap messages: the shell intercepts these before the core
            // ever sees them, since a pure core has no field to hold a handle or an `Arc<dyn
            // Application>` in. Only the type split arriving with the port/wiring phase removes
            // these arms.
            AppManagement::SetRasterizerForwardHandle(_) => {
                unreachable!("handled by the engine shell")
            }
            AppManagement::Configure(_) => {
                unreachable!("handled by the engine shell")
            }
            AppManagement::RegisterApp(_) => {
                unreachable!("handled by the engine shell")
            }
            AppManagement::SetTitle(title) => {
                out.driver_control = Some(DisplayControl::SetTitle {
                    id: WindowId::PRIMARY,
                    title,
                });
            }
            AppManagement::ResizeRequest(width, height) => {
                out.driver_control = Some(DisplayControl::SetSize {
                    id: WindowId::PRIMARY,
                    width,
                    height,
                });
            }
            AppManagement::CopyToClipboard(text) => {
                out.driver_control = Some(DisplayControl::Copy { text });
            }
            AppManagement::RequestPaste => {
                out.driver_control = Some(DisplayControl::RequestPaste);
            }
            AppManagement::SetCursorIcon(icon) => {
                out.driver_control = Some(DisplayControl::SetCursor {
                    id: WindowId::PRIMARY,
                    cursor: icon,
                });
            }
            AppManagement::CreateWindow(descriptor) => {
                // Engine assigns the window ID (for now, just use PRIMARY for single window)
                let id = WindowId::PRIMARY;
                log::info!(
                    "Relaying CreateWindow request: assigning id={}, {}x{} \"{}\"",
                    id.0,
                    descriptor.width,
                    descriptor.height,
                    descriptor.title
                );
                out.driver_mgmt = Some(DisplayMgmt::Create {
                    settings: descriptor,
                });
            }
            AppManagement::Quit => out.quit = true,
        }
        Ok(out)
    }
}

/// Convert raw mouse button code to MouseButton enum
fn convert_mouse_button(button: u8) -> MouseButton {
    match button {
        0 => MouseButton::Left,
        1 => MouseButton::Middle,
        2 => MouseButton::Right,
        _ => MouseButton::Other(button),
    }
}

#[cfg(test)]
mod tests {
    //! `EngineCore` in isolation — no handles, no threads, no scheduler. Mirrors
    //! `vsync_actor.rs`'s `mod tests` for `VsyncCore` and `rasterizer/actor.rs`'s `core_tests`
    //! for `RasterCore`.

    use super::*;
    use crate::display::messages::{Generation, Surface};
    use crate::platform::ColorCube;
    use pixelflow_core::{Discrete, Manifold};
    use pixelflow_graphics::render::Frame;
    use std::sync::Arc;

    /// A constant black scene — these tests care about which port fires, not what is drawn.
    fn manifold() -> Arc<dyn Manifold<Output = Discrete> + Send + Sync> {
        Arc::new(ColorCube::default().at(0.0f32, 0.0f32, 0.0f32, 1.0f32))
    }

    fn surface(width_px: u32, height_px: u32) -> Surface {
        Surface {
            id: WindowId(1),
            width_px,
            height_px,
            frame_width: width_px,
            frame_height: height_px,
            scale: 1.0,
        }
    }

    fn window(meta_size: (u32, u32)) -> Window {
        Window::rejoin(
            Frame::new(meta_size.0, meta_size.1),
            WindowMeta {
                id: WindowId(1),
                width_px: meta_size.0,
                height_px: meta_size.1,
                scale: 1.0,
                generation: Generation::NONE,
            },
        )
    }

    #[test]
    fn a_vsync_tick_requests_a_frame_from_the_app() {
        let mut core = EngineCore::new();
        let out = core
            .step_data(EngineData::VSync {
                timestamp: Instant::now(),
                target_timestamp: Instant::now(),
                refresh_interval: Duration::from_millis(16),
            })
            .unwrap();

        assert!(
            matches!(
                out.app,
                Some(EngineEvent::Data(EngineEventData::RequestFrame { .. }))
            ),
            "every tick must ask the app for a frame"
        );
    }

    #[test]
    fn green_stuck_is_logged_and_emits_nothing() {
        // A trivial green actor, adopted only so `Host::adopt` mints a real `NodeId` — the
        // supervision port's payload has to be a real identity, not a stand-in, for this test to
        // say anything about the actual variant shape.
        struct Noop;
        impl actor_scheduler::mealy::Transducer for Noop {
            type Control = std::convert::Infallible;
            type Management = std::convert::Infallible;
            type Data = std::convert::Infallible;
            type Out = ();
            fn step_data(&mut self, msg: std::convert::Infallible) -> Result<(), HandlerError> {
                match msg {}
            }
        }
        struct NoopWiring;
        impl actor_scheduler::mealy::Wiring for NoopWiring {
            type Out = ();
            fn flush(&mut self, _out: &mut ()) -> actor_scheduler::mealy::Flush {
                actor_scheduler::mealy::Flush::Done
            }
        }
        let (_tx, rx) = actor_scheduler::spsc::spsc_channel::<std::convert::Infallible>(1);
        let mut host = actor_scheduler::host::Host::new();
        let node_id = host.adopt(actor_scheduler::mealy::Node::new(Noop, rx, NoopWiring));

        let mut core = EngineCore::new();
        let out = core
            .step_control(EngineControl::GreenStuck(
                actor_scheduler::host::Supervision {
                    node: node_id,
                    reason: actor_scheduler::host::Stuck::TargetGone,
                },
            ))
            .unwrap();

        assert!(
            !out.quit && out.app.is_none() && out.driver_control.is_none(),
            "logging a stuck node must not itself trigger any other port"
        );
    }

    #[test]
    fn render_surface_returns_the_vsync_token() {
        let mut core = EngineCore::new();
        let out = core
            .step_data(EngineData::FromApp(AppData::RenderSurface(manifold())))
            .unwrap();

        assert!(
            matches!(out.vsync_control, Some(VsyncCommand::ReturnToken)),
            "submitting a scene must release the token so vsync can ask again"
        );
    }

    #[test]
    fn a_granted_window_turns_a_submitted_scene_into_a_rasterizer_port() {
        let mut core = EngineCore::new();
        // A window must exist before the coordinator will ask for it.
        core.step_data(EngineData::FromDriver(DisplayEvent::WindowCreated {
            surface: surface(100, 100),
        }))
        .unwrap();

        let out = core
            .step_data(EngineData::FromApp(AppData::RenderSurface(manifold())))
            .unwrap();
        assert!(
            matches!(out.driver_mgmt, Some(DisplayMgmt::RequestWindow)),
            "a scene with no buffer in hand must ask the driver for one"
        );
        core.request_delivered();

        let out = core
            .step_data(EngineData::WindowGranted(window((100, 100))))
            .unwrap();
        assert!(
            out.rasterizer.is_some(),
            "a granted window with a pending scene must render immediately"
        );
        assert!(
            !core.holds_buffer(),
            "the buffer left with the render request"
        );
    }

    #[test]
    fn render_complete_with_a_presentable_frame_emits_present_and_vsync_telemetry() {
        use pixelflow_graphics::render::rasterizer::RenderResponse;

        let mut core = EngineCore::new();
        core.step_data(EngineData::FromDriver(DisplayEvent::WindowCreated {
            surface: surface(100, 100),
        }))
        .unwrap();
        core.step_data(EngineData::FromApp(AppData::RenderSurface(manifold())))
            .unwrap();
        core.request_delivered();
        let out = core
            .step_data(EngineData::WindowGranted(window((100, 100))))
            .unwrap();
        let request = match out.rasterizer {
            Some(request) => request,
            None => panic!("expected a render request"),
        };

        let out = core
            .step_data(EngineData::RenderComplete(RenderResponse {
                frame: request.frame,
                render_time: Some(Duration::from_millis(1)),
                meta: request.meta,
            }))
            .unwrap();

        assert!(
            matches!(out.driver_data, Some(DisplayData::Present { .. })),
            "a drawn frame must be presented"
        );
        assert!(
            out.vsync_data.is_some(),
            "presenting a frame must report FPS telemetry to vsync"
        );
    }

    #[test]
    fn quit_control_sets_quit() {
        let mut core = EngineCore::new();
        let out = core.step_control(EngineControl::Quit).unwrap();
        assert!(
            out.quit,
            "EngineControl::Quit must trigger the shutdown cascade"
        );
    }

    #[test]
    fn set_title_emits_driver_control() {
        let mut core = EngineCore::new();
        let out = core
            .step_management(AppManagement::SetTitle("hello".into()))
            .unwrap();
        assert!(
            matches!(out.driver_control, Some(DisplayControl::SetTitle { .. })),
            "SetTitle must relay to the driver's control port"
        );
    }

    #[test]
    fn close_requested_sets_quit_and_notifies_the_app() {
        let mut core = EngineCore::new();
        let out = core
            .step_data(EngineData::FromDriver(DisplayEvent::CloseRequested {
                id: WindowId::PRIMARY,
            }))
            .unwrap();

        assert!(
            out.quit,
            "a close request must trigger the shutdown cascade"
        );
        assert!(
            matches!(
                out.app,
                Some(EngineEvent::Control(EngineEventControl::CloseRequested))
            ),
            "the app must be told the window is closing"
        );
    }
}
