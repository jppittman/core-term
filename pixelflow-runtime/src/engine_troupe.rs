//! Engine Troupe - Render pipeline actor coordination using troupe! macro.

use crate::api::private::{EngineControl, EngineData};
use crate::api::public::{
    AppData, AppManagement, Application, EngineEvent, EngineEventControl, EngineEventData,
    EngineEventManagement, WindowId,
};
use crate::config::EngineConfig;
use crate::display::driver::DriverActor;
use crate::display::messages::{
    DisplayControl, DisplayData, DisplayEvent, DisplayMgmt, Generation, Window, WindowMeta,
};
use crate::display::platform::PlatformActor;
use crate::error::RuntimeError;
use crate::input::MouseButton;
use crate::platform::{ActivePlatform, PlatformPixel};
use crate::vsync_actor::{
    RenderedResponse, VsyncActor, VsyncCommand, VsyncConfig, VsyncManagement,
};
use actor_scheduler::mealy::Credit;
use actor_scheduler::{
    Actor, ActorHandle, ActorStatus, ActorTypes, HandlerError, HandlerResult, Message, SendError,
    SystemStatus, TroupeActor,
};
use pixelflow_core::{At, Discrete, Manifold, W, X, Y, Z};
use pixelflow_graphics::render::rasterizer::{
    RasterizerActor, RasterizerHandle, RenderRequest, RenderResponse,
};
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Instant;

const LOG_FRAME_INTERVAL: u64 = 60;

/// A window buffer together with the generation it belongs to.
///
/// The two are paired everywhere the buffer goes because they are only useful together: the
/// buffer alone cannot say whether it is still the live one (see [`Generation`]).
struct StampedWindow {
    generation: Generation,
    window: Window,
}

/// Engine handler - coordinates app, rendering, display.
pub struct EngineHandler {
    /// Handle to the display driver actor.
    driver: ActorHandle<DisplayData, DisplayControl, DisplayMgmt>,
    /// Handle to the vsync actor (for feedback loop).
    vsync: ActorHandle<RenderedResponse, VsyncCommand, VsyncManagement>,
    /// Handle to the rasterizer actor (set after bootstrap completes).
    rasterizer: Option<RasterizerHandle<PlatformPixel, WindowMeta>>,
    /// Handle to self (for shutdown).
    self_handle: Option<ActorHandle<EngineData, EngineControl, AppManagement>>,
    /// Pre-created dedicated SPSC handle for rasterizer response forwarding thread.
    /// Set via SetRasterizerForwardHandle management message before Configure.
    rasterizer_forward_handle: Option<ActorHandle<EngineData, EngineControl, AppManagement>>,
    /// Handle to the application (for event forwarding).
    app_handle: Option<Arc<dyn Application + Send + Sync>>,
    /// Frame counter for VSync feedback.
    frame_number: u64,
    /// The active window (owns frame buffer, returned by driver after presentation).
    window: Option<StampedWindow>,
    /// The newest buffer the driver has handed over. Anything older is a buffer that a resize
    /// has already replaced, wherever it happens to be circulating.
    current_generation: Generation,
    /// Buffers sent to the driver to present and not yet returned, oldest first.
    ///
    /// A resize can put a second buffer in circulation, so `PresentComplete` is not
    /// self-describing — the returned `Window` carries no mark saying which one it is. The
    /// driver presents in the order it is asked to and returns in that same order (one actor,
    /// one FIFO lane each way), so the front of this queue names the buffer that just came back.
    presented: VecDeque<Generation>,
    /// The one-outstanding-render bound on the engine → rasterizer edge.
    ///
    /// `pending_render` used to serve double duty: carrying the torn-off window metadata *and*
    /// standing in as an "is a render in flight" flag. The metadata now rides with the request,
    /// leaving only the bound — which the target topology already specifies as `Credit(1)`
    /// (`docs/designs/pixelflow-runtime-engine-mesh-migration.md` §3), so it is that, rather
    /// than an `Option` being checked for `is_none`.
    render_credit: Credit,
    /// Number of render threads for work-stealing parallelism.
    render_threads: usize,
    /// Latest manifold from app - always keep the most recent, drop old ones.
    /// App sends manifolds fast (cheap algebra), engine rasterizes slow (expensive).
    pending_manifold: Option<Arc<dyn Manifold<Output = Discrete> + Send + Sync>>,
}

// ActorTypes impls - required for troupe! macro
impl ActorTypes for EngineHandler {
    type Data = EngineData;
    type Control = EngineControl;
    type Management = AppManagement;
}

impl ActorTypes for DriverActor<ActivePlatform> {
    type Data = DisplayData;
    type Control = DisplayControl;
    type Management = DisplayMgmt;
}

// Generate troupe structures using macro
// Note: Rasterizer is NOT in the troupe - it uses a bootstrap handshake pattern
// that enforces type-level guarantees about initialization order.
actor_scheduler::troupe! {
    driver: DriverActor<ActivePlatform> [main],
    engine: EngineHandler [expose],
    vsync: VsyncActor [expose],
}

// Implement Actor for EngineHandler
impl Actor<EngineData, EngineControl, AppManagement> for EngineHandler {
    fn handle_data(&mut self, data: EngineData) -> HandlerResult {
        match data {
            EngineData::FromApp(app_data) => self.handle_app_data(app_data),
            EngineData::FromDriver(event) => self.handle_driver_event(event),
            EngineData::VSync {
                timestamp,
                target_timestamp,
                refresh_interval,
            } => {
                // ALWAYS request frame from app (app builds compute graphs fast)
                // Token bucket is now managed atomically by VSync
                if let Some(app) = &self.app_handle {
                    app.send(EngineEvent::Data(EngineEventData::RequestFrame {
                        timestamp,
                        target_timestamp,
                        refresh_interval,
                    }))
                    .expect("failed to send to app. it probably crashed");
                }

                // Delegate rendering to rasterizer if we have a manifold and a window
                self.render_if_ready();
            }
            EngineData::RenderComplete(response) => {
                // The render is done, so the edge's one credit is free again.
                self.render_credit.release();

                // Staleness is something we *read*, not something the resize handler reaches in
                // and marks: the generation this render was aimed at travelled out with the
                // request and came back untouched, so comparing it against the newest buffer the
                // driver has handed over says exactly what the old `stale` flag said.
                if response.meta.generation != self.current_generation {
                    log::debug!(
                        "Discarding stale render ({}x{}, buffer {}) - buffer {} is current",
                        response.meta.width_px,
                        response.meta.height_px,
                        response.meta.generation,
                        self.current_generation
                    );
                    // The frame was drawn into a buffer that no longer exists as far as the
                    // driver is concerned. Drop it and re-render into the one that does.
                    drop(response);
                    self.render_if_ready();
                } else {
                    // Reassembled from the metadata that travelled with the request.
                    let window = Window {
                        id: response.meta.id,
                        frame: response.frame,
                        width_px: response.meta.width_px,
                        height_px: response.meta.height_px,
                        scale: response.meta.scale,
                    };
                    assert!(
                        self.window.is_none(),
                        "two live buffers of generation {}: one held, one just rendered",
                        response.meta.generation
                    );
                    let stamped = StampedWindow {
                        generation: response.meta.generation,
                        window,
                    };
                    match response.render_time {
                        Some(render_time) => self.present_cooked_frame(render_time, stamped),
                        // The rasterizer was paused, so it handed the buffer back unrendered.
                        // Presenting it would blit whatever stale pixels it still holds; keeping
                        // it is the whole point of the frame coming back at all. This arm exists
                        // because `render_time: Option<Duration>` forces the question — the
                        // buffer's return is unconditional, its having been drawn into is not.
                        None => {
                            log::debug!("Render skipped (paused); retaining the buffer unpresented");
                            self.window = Some(stamped);
                        }
                    }
                }
            }
            EngineData::PresentComplete(returned_window) => {
                // Driver returned the buffer it just presented — the oldest one still out, since
                // it presents and returns in order. Keeping it is only correct if it is still
                // the live buffer; a resize since it went out means the driver has replaced it
                // and this one is just an old-size allocation to drop.
                let generation = self
                    .presented
                    .pop_front()
                    .expect("PresentComplete for a buffer the engine never presented");
                if generation == self.current_generation {
                    assert!(
                        self.window.is_none(),
                        "two live buffers of generation {generation}: one held, one returned"
                    );
                    self.window = Some(StampedWindow {
                        generation,
                        window: returned_window,
                    });
                } else {
                    log::debug!(
                        "Discarding superseded buffer {} ({}x{}) - buffer {} is current",
                        generation,
                        returned_window.width_px,
                        returned_window.height_px,
                        self.current_generation
                    );
                }

                // Notify VSync for FPS tracking (actual rasterization completion)
                self.vsync
                    .send(Message::Data(RenderedResponse {
                        frame_number: self.frame_number,
                        rendered_at: Instant::now(),
                    }))
                    .expect("Failed to notify VSync of completed frame");

                // Catch up on whatever arrived while the buffer was with the driver.
                self.render_if_ready();
            }
        }
        Ok(())
    }

    fn handle_control(&mut self, ctrl: EngineControl) -> HandlerResult {
        match ctrl {
            EngineControl::Quit => {
                self.vsync
                    .send(Message::Shutdown)
                    .expect("Failed to shutdown vsync on Quit");
                if let Some(rasterizer) = &self.rasterizer {
                    rasterizer
                        .send(Message::Shutdown)
                        .expect("Failed to shutdown rasterizer on Quit");
                }
                self.app_handle = None;
                self.driver
                    .send(Message::Shutdown)
                    .expect("Failed to shutdown driver on Quit");
                if let Some(self_handle) = &self.self_handle {
                    self_handle
                        .send(Message::Shutdown)
                        .expect("Failed to shutdown engine on Quit");
                }
            }
            EngineControl::UpdateRefreshRate(rr) => {
                self.vsync
                    .send(VsyncCommand::UpdateRefreshRate(rr))
                    .expect("failed to update refresh rate");
            }
            EngineControl::VsyncActorReady(handle) => {
                self.vsync = handle;
            }
            EngineControl::DriverAck => {
                unimplemented!("DriverAck not yet implemented");
            }
        }
        Ok(())
    }

    fn handle_management(&mut self, mgmt: AppManagement) -> HandlerResult {
        match mgmt {
            AppManagement::SetRasterizerForwardHandle(handle) => {
                self.rasterizer_forward_handle = Some(handle);
            }
            AppManagement::Configure(config) => {
                self.render_threads = config.performance.render_threads;
                log::info!("Engine configured: {} render threads", self.render_threads);

                // Spawn rasterizer with bootstrap pattern
                self.spawn_rasterizer();
            }
            AppManagement::SetTitle(title) => {
                self.driver
                    .send(Message::Control(DisplayControl::SetTitle {
                        id: WindowId::PRIMARY,
                        title,
                    }))
                    .expect("Failed to relay SetTitle to driver");
            }
            AppManagement::ResizeRequest(width, height) => {
                self.driver
                    .send(Message::Control(DisplayControl::SetSize {
                        id: WindowId::PRIMARY,
                        width,
                        height,
                    }))
                    .expect("Failed to send SetSize to driver");
            }
            AppManagement::CopyToClipboard(text) => {
                self.driver
                    .send(Message::Control(DisplayControl::Copy { text }))
                    .expect("Failed to send Copy to driver");
            }
            AppManagement::RequestPaste => {
                self.driver
                    .send(Message::Control(DisplayControl::RequestPaste))
                    .expect("Failed to send RequestPaste to driver");
            }
            AppManagement::SetCursorIcon(icon) => {
                self.driver
                    .send(Message::Control(DisplayControl::SetCursor {
                        id: WindowId::PRIMARY,
                        cursor: icon,
                    }))
                    .expect("Failed to send SetCursor to driver");
            }
            AppManagement::RegisterApp(app) => {
                log::info!("Application handle registered");
                self.app_handle = Some(app);
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
                self.driver
                    .send(Message::Management(DisplayMgmt::Create {
                        settings: descriptor,
                    }))
                    .expect("Failed to relay CreateWindow to driver");
            }
            AppManagement::Quit => {
                self.vsync
                    .send(Message::Shutdown)
                    .expect("Failed to shutdown vsync on AppManagement::Quit");
                if let Some(rasterizer) = &self.rasterizer {
                    rasterizer
                        .send(Message::Shutdown)
                        .expect("Failed to shutdown rasterizer on AppManagement::Quit");
                }
                self.app_handle = None;
                self.driver
                    .send(Message::Shutdown)
                    .expect("Failed to shutdown driver on AppManagement::Quit");
                if let Some(self_handle) = &self.self_handle {
                    self_handle
                        .send(Message::Shutdown)
                        .expect("Failed to shutdown engine on AppManagement::Quit");
                }
            }
        }
        Ok(())
    }

    fn park(&mut self, _status: SystemStatus) -> Result<ActorStatus, HandlerError> {
        // engine has no external channels which might be busy
        Ok(ActorStatus::Idle)
    }
}

impl EngineHandler {
    /// Take delivery of a buffer from the driver, as the newest generation.
    ///
    /// Every window the engine holds comes through here, so "which buffer is live?" has exactly
    /// one writer. Stamping is what retires the previous buffer: it does not have to be found
    /// and marked wherever it is circulating, because everything downstream compares against
    /// this counter rather than carrying a flag of its own.
    fn stamp(&mut self, window: Window) -> StampedWindow {
        self.current_generation = self.current_generation.next();
        StampedWindow {
            generation: self.current_generation,
            window,
        }
    }

    /// Start a render if both halves of one are in hand: a manifold to draw, and a window to
    /// draw it into.
    ///
    /// Every path that acquires either half ends here, so "can we render yet?" is asked in one
    /// place instead of being open-coded at each arrival. Nothing is consumed unless a render
    /// actually starts — a refused render puts both back (see `trigger_render_with_window`),
    /// which is what lets the next completion pick them up.
    fn render_if_ready(&mut self) {
        if self.pending_manifold.is_none() || self.window.is_none() {
            return;
        }
        let manifold = self
            .pending_manifold
            .take()
            .expect("pending_manifold checked Some above");
        let stamped = self.window.take().expect("window checked Some above");
        self.trigger_render_with_window(manifold, stamped);
    }

    fn return_vsync_token(&self) {
        match self.vsync.send(Message::Control(VsyncCommand::ReturnToken)) {
            Ok(()) | Err(SendError::Disconnected) => {}
            Err(SendError::Timeout) => panic!("Timed out returning vsync token"),
        }
    }

    /// Spawn the rasterizer actor with bootstrap handshake.
    ///
    /// This sets up:
    /// 1. The rasterizer actor thread
    /// 2. A response channel for render results
    /// 3. A forwarding thread that receives responses and sends them to the engine
    fn spawn_rasterizer(&mut self) {
        if self.rasterizer.is_some() {
            log::warn!("Rasterizer already initialized");
            return;
        }

        let engine_handle = self
            .rasterizer_forward_handle
            .take()
            .expect("SetRasterizerForwardHandle must be sent before Configure");

        // Step 1: Spawn rasterizer with setup handle
        let (setup_handle, _rasterizer_thread) =
            RasterizerActor::<PlatformPixel, WindowMeta>::spawn_with_setup(self.render_threads);

        // Step 2: Create response channel (engine receives render results here)
        let (response_tx, response_rx) =
            std::sync::mpsc::channel::<RenderResponse<PlatformPixel, WindowMeta>>();

        // Step 3: Start forwarding thread - receives responses and sends to engine
        std::thread::spawn(move || {
            log::debug!("Rasterizer response forwarding thread started");
            while let Ok(response) = response_rx.recv() {
                // Forward to engine as RenderComplete
                if let Err(e) =
                    engine_handle.send(Message::Data(EngineData::RenderComplete(response)))
                {
                    log::warn!("Failed to forward render response to engine: {}", e);
                    break;
                }
            }
            log::debug!("Rasterizer response forwarding thread exiting");
        });

        // Step 4: Complete bootstrap - register response channel and get full handle
        let rasterizer_handle = setup_handle.register(response_tx);

        log::info!("Rasterizer actor initialized via bootstrap");
        self.rasterizer = Some(rasterizer_handle);
    }

    /// Handle app data messages (render surfaces, etc.)
    fn handle_app_data(&mut self, app_data: AppData) {
        match app_data {
            AppData::RenderSurface(manifold) | AppData::RenderSurfaceU32(manifold) => {
                log::debug!("Engine: Received RenderSurface from app");
                // The app has provided its compute graph, so permit VSync to request another
                // frame without waiting for rasterization to finish.
                self.return_vsync_token();

                // Keep-latest port: the newest compute graph replaces any frame that hasn't
                // started rendering yet, and renders now if a window is free to draw into.
                self.pending_manifold = Some(manifold);
                self.render_if_ready();
            }
            AppData::Skipped => {
                // App says nothing to render - return token anyway
                self.return_vsync_token();
            }
        }
    }

    /// Trigger asynchronous rendering on the rasterizer actor with a Window.
    fn trigger_render_with_window(
        &mut self,
        manifold: Arc<dyn Manifold<Output = Discrete> + Send + Sync>,
        stamped: StampedWindow,
    ) {
        // Take the edge's single credit *before* taking the window apart — a refusal has to
        // hand both halves back intact.
        //
        // A refusal here is ordinary backpressure, not an error: vsync keeps asking for frames
        // at 60Hz while a render is in flight, so a frame arriving mid-render is the common
        // case. Dropping the pair would be the expensive kind of mistake — the window in hand
        // during an in-flight render is the live buffer, so losing it strands the terminal at
        // whatever the in-flight render happens to produce. Both go back where they came from
        // and the next `release()` re-drives them.
        if !self.render_credit.try_consume() {
            log::debug!(
                "Render in flight; deferring frame for buffer {} ({}x{}) until it completes",
                stamped.generation,
                stamped.window.width_px,
                stamped.window.height_px
            );
            self.window = Some(stamped);
            self.pending_manifold = Some(manifold);
            return;
        }

        // Extract frame from window for rasterization; the rest travels with the request.
        let StampedWindow { generation, window } = stamped;
        let Window {
            id,
            frame,
            width_px,
            height_px,
            scale,
        } = window;

        // The scene is authored in point space; the frame is the platform's
        // sample lattice and may be denser (device pixels on HiDPI displays).
        // The lattice embedding is the measured ratio points/pixels per axis —
        // identity when the platform samples 1:1 (X11, non-Retina macOS).
        // Contramapping here keeps the app scale-agnostic: platform = dimap.
        assert!(
            frame.width > 0 && frame.height > 0,
            "cannot render into an empty frame ({}x{})",
            frame.width,
            frame.height
        );
        let point_per_px_x = width_px as f32 / frame.width as f32;
        let point_per_px_y = height_px as f32 / frame.height as f32;
        let manifold: Arc<dyn Manifold<Output = Discrete> + Send + Sync> =
            if point_per_px_x == 1.0 && point_per_px_y == 1.0 {
                manifold
            } else {
                Arc::new(At {
                    inner: manifold,
                    x: X * point_per_px_x,
                    y: Y * point_per_px_y,
                    z: Z,
                    w: W,
                })
            };

        // The window's other half travels with the frame instead of being stashed here.
        let request = RenderRequest {
            manifold,
            frame,
            meta: WindowMeta {
                id,
                width_px,
                height_px,
                scale,
                generation,
            },
        };

        // Send to rasterizer. On any failure the render never happens, so give the credit back
        // — otherwise the edge would be permanently blocked by a request that was never sent.
        if let Some(rasterizer) = &self.rasterizer {
            if let Err(e) = rasterizer.send(Message::Data(request)) {
                log::warn!("Failed to send render request to rasterizer: {}", e);
                self.render_credit.release();
            }
        } else {
            log::warn!("Rasterizer not initialized, dropping render request");
            self.render_credit.release();
        }
    }

    /// Present a window with cooked frame to the driver.
    fn present_cooked_frame(&mut self, render_time: std::time::Duration, stamped: StampedWindow) {
        // Send window to driver for presentation (transfers ownership). The generation stays
        // here, in presentation order, because the buffer comes back unmarked.
        let StampedWindow { generation, window } = stamped;
        self.presented.push_back(generation);

        let t1 = Instant::now();
        self.driver
            .send(Message::Data(DisplayData::Present { window }))
            .expect("Failed to send window to driver for presentation");
        let send_time = t1.elapsed();

        self.frame_number += 1;
        if self.frame_number.is_multiple_of(LOG_FRAME_INTERVAL) {
            log::info!(
                "Frame {}: render={:?}, send={:?}",
                self.frame_number,
                render_time,
                send_time
            );
        }
    }

    /// Handle events from the display driver
    fn handle_driver_event(&mut self, event: DisplayEvent) {
        match event {
            DisplayEvent::WindowCreated { window } => {
                log::debug!(
                    "Relaying WindowCreated: id={}, {}x{}, scale={}",
                    window.id.0,
                    window.width_px,
                    window.height_px,
                    window.scale
                );

                let id = window.id;
                let width_px = window.width_px;
                let height_px = window.height_px;
                let scale = window.scale;

                // Receive initial window from driver
                self.window = Some(self.stamp(window));
                log::debug!("Engine: Window stored from WindowCreated");

                // Render straight away if the app already handed us a frame to draw.
                self.render_if_ready();

                // Relay WindowCreated event to app
                if let Some(app) = &self.app_handle {
                    app.send(EngineEvent::Control(EngineEventControl::WindowCreated {
                        id,
                        width_px,
                        height_px,
                        scale,
                    }))
                    .expect("Failed to relay WindowCreated event to app");
                }
            }
            DisplayEvent::Resized { window } => {
                log::debug!(
                    "Relaying Resized: id={}, {}x{}",
                    window.id.0,
                    window.width_px,
                    window.height_px
                );

                let id = window.id;
                let width_px = window.width_px;
                let height_px = window.height_px;

                // Nothing to mark: this is a whole new buffer, and stamping it below retires
                // every older one wherever it is — mid-render, mid-presentation, or held here.
                // `RenderComplete` and `PresentComplete` each read that off the stamp.
                self.window = Some(self.stamp(window));

                // No gate on "is a render in flight" here. The credit is the single arbiter:
                // if one is in flight it refuses, and the refusal is now a deferral that leaves
                // this window and the queued manifold exactly where they are, to be re-driven
                // when the stale render completes and releases.
                self.render_if_ready();

                // Relay resize event to app
                if let Some(app) = &self.app_handle {
                    app.send(EngineEvent::Control(EngineEventControl::Resized {
                        id,
                        width_px,
                        height_px,
                    }))
                    .expect("Failed to relay Resized event to app");
                }
            }
            DisplayEvent::Key {
                symbol,
                modifiers,
                text,
                ..
            } => {
                if let Some(app) = &self.app_handle {
                    app.send(EngineEvent::Management(EngineEventManagement::KeyDown {
                        key: symbol,
                        mods: modifiers,
                        text,
                    }))
                    .expect("Failed to send KeyDown event to app");
                }
            }
            DisplayEvent::MouseButtonPress { button, x, y, .. } => {
                if let Some(app) = &self.app_handle {
                    let button = convert_mouse_button(button);
                    app.send(EngineEvent::Management(EngineEventManagement::MouseClick {
                        x: x as u32,
                        y: y as u32,
                        button,
                    }))
                    .expect("Failed to send MouseClick event to app");
                }
            }
            DisplayEvent::MouseButtonRelease { button, x, y, .. } => {
                if let Some(app) = &self.app_handle {
                    let button = convert_mouse_button(button);
                    app.send(EngineEvent::Management(
                        EngineEventManagement::MouseRelease {
                            x: x as u32,
                            y: y as u32,
                            button,
                        },
                    ))
                    .expect("Failed to send MouseRelease event to app");
                }
            }
            DisplayEvent::MouseMove {
                x, y, modifiers, ..
            } => {
                if let Some(app) = &self.app_handle {
                    app.send(EngineEvent::Management(EngineEventManagement::MouseMove {
                        x: x as u32,
                        y: y as u32,
                        mods: modifiers,
                    }))
                    .expect("Failed to send MouseMove event to app");
                }
            }
            DisplayEvent::MouseScroll {
                dx,
                dy,
                x,
                y,
                modifiers,
                ..
            } => {
                if let Some(app) = &self.app_handle {
                    app.send(EngineEvent::Management(
                        EngineEventManagement::MouseScroll {
                            x: x as u32,
                            y: y as u32,
                            dx,
                            dy,
                            mods: modifiers,
                        },
                    ))
                    .expect("Failed to send MouseScroll event to app");
                }
            }
            DisplayEvent::CloseRequested { .. } => {
                log::debug!("Close requested");
                // Stop vsync from generating more frame requests
                self.vsync
                    .send(Message::Shutdown)
                    .expect("Failed to shutdown vsync on CloseRequested");
                // Shutdown rasterizer
                if let Some(rasterizer) = &self.rasterizer {
                    rasterizer
                        .send(Message::Shutdown)
                        .expect("Failed to shutdown rasterizer on CloseRequested");
                }
                // Notify app, then drop it - cleanup goes in app's Drop impl
                if let Some(app) = self.app_handle.take() {
                    app.send(EngineEvent::Control(EngineEventControl::CloseRequested))
                        .expect("Failed to send CloseRequested to app");
                }
                // Shutdown the driver actor (terminates platform event loop)
                self.driver
                    .send(Message::Shutdown)
                    .expect("Failed to shutdown driver on CloseRequested");
                // Shutdown self
                if let Some(self_handle) = &self.self_handle {
                    self_handle
                        .send(Message::Shutdown)
                        .expect("Failed to shutdown engine on CloseRequested");
                }
            }
            DisplayEvent::FocusGained { .. } => {
                if let Some(app) = &self.app_handle {
                    app.send(EngineEvent::Management(EngineEventManagement::FocusGained))
                        .expect("Failed to send FocusGained event to app");
                }
            }
            DisplayEvent::FocusLost { .. } => {
                if let Some(app) = &self.app_handle {
                    app.send(EngineEvent::Management(EngineEventManagement::FocusLost))
                        .expect("Failed to send FocusLost event to app");
                }
            }
            DisplayEvent::PasteData { text } => {
                if let Some(app) = &self.app_handle {
                    app.send(EngineEvent::Management(EngineEventManagement::Paste(text)))
                        .expect("Failed to send Paste event to app");
                }
            }
            DisplayEvent::ScaleChanged { id, scale } => {
                log::debug!("Relaying ScaleChanged: id={}, scale={}", id.0, scale);
                if let Some(app) = &self.app_handle {
                    app.send(EngineEvent::Control(EngineEventControl::ScaleChanged {
                        id,
                        scale,
                    }))
                    .expect("Failed to relay ScaleChanged event to app");
                }
            }
            DisplayEvent::ClipboardDataRequested => {
                unimplemented!("Clipboard data requested")
            }
            DisplayEvent::WindowDestroyed { .. } => {
                unimplemented!("window destroyed, forward to app unimplemented");
                // Window was destroyed
            }
        }
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

// Implement TroupeActor for EngineHandler — takes ownership of per-actor Directory
impl TroupeActor<Directory> for EngineHandler {
    fn new(dir: Directory) -> Self {
        Self {
            driver: dir.driver,
            vsync: dir.vsync,
            rasterizer: None, // Set up separately via bootstrap
            self_handle: Some(dir.engine),
            rasterizer_forward_handle: None, // Set via SetRasterizerForwardHandle message
            app_handle: None,
            frame_number: 0,
            window: None,
            current_generation: Generation::NONE,
            presented: VecDeque::new(),
            render_credit: Credit::new(1),
            render_threads: 1, // Default, will be set by Configure message
            pending_manifold: None,
        }
    }
}

// Implement TroupeActor for DriverActor — takes ownership of per-actor Directory
impl TroupeActor<Directory> for DriverActor<ActivePlatform> {
    fn new(dir: Directory) -> Self {
        #[cfg(target_os = "macos")]
        {
            use crate::platform::MetalOps;
            let ops = MetalOps::new().expect("Failed to create Metal ops");
            let platform = PlatformActor::new(ops, dir.engine);
            DriverActor::new(platform)
        }
        #[cfg(target_os = "linux")]
        {
            use crate::platform::linux::LinuxOps;
            let ops = LinuxOps::new().expect("Failed to create Linux ops");
            let platform = PlatformActor::new(ops, dir.engine);
            DriverActor::new(platform)
        }
        #[cfg(not(any(target_os = "macos", target_os = "linux")))]
        {
            let _ = dir;
            panic!("Unsupported platform");
        }
    }
}

impl Troupe {
    /// Create troupe and configure vsync actor.
    pub fn with_config(config: EngineConfig) -> Result<Self, RuntimeError> {
        // Create troupe with platform-specific waker for the main (driver) actor
        #[cfg(target_os = "macos")]
        let mut troupe = {
            use crate::platform::waker::CocoaWaker;
            Self::new_with_waker(Some(std::sync::Arc::new(CocoaWaker::new())))
        };
        #[cfg(target_os = "linux")]
        let mut troupe = {
            use crate::platform::linux::set_shared_waker;
            use crate::platform::waker::X11Waker;
            let waker = X11Waker::new();
            set_shared_waker(waker.clone());
            Self::new_with_waker(Some(std::sync::Arc::new(waker)))
        };
        #[cfg(not(any(target_os = "macos", target_os = "linux")))]
        let mut troupe = Self::new();

        // Create SPSC handles for initialization (each exposed() creates unique channels)
        let init = troupe.exposed(); // engine + vsync handles for sending config messages
        let vsync_engine = troupe.exposed(); // engine handle for vsync→engine
        let clock_vsync = troupe.exposed(); // vsync handle for clock→vsync
        let rasterizer_fwd = troupe.exposed(); // engine handle for rasterizer→engine

        // Send rasterizer forwarding handle BEFORE Configure
        init.engine
            .send(Message::Management(
                AppManagement::SetRasterizerForwardHandle(rasterizer_fwd.engine),
            ))
            .map_err(|e| {
                RuntimeError::InitError(format!("Failed to set rasterizer fwd handle: {}", e))
            })?;

        // Configure the engine with window settings
        init.engine
            .send(Message::Management(AppManagement::Configure(
                config.clone(),
            )))
            .map_err(|e| RuntimeError::InitError(format!("Failed to configure engine: {}", e)))?;

        // Configure vsync with target FPS (auto-starts after configuration)
        init.vsync
            .send(Message::Management(VsyncManagement::SetConfig {
                config: VsyncConfig {
                    refresh_rate: config.performance.target_fps as f64,
                },
                engine_handle: Box::new(vsync_engine.engine),
                self_handle: Box::new(clock_vsync.vsync),
            }))
            .map_err(|e| RuntimeError::InitError(format!("Failed to configure vsync: {}", e)))?;

        Ok(troupe)
    }

    /// Get an unregistered engine handle.
    ///
    /// Creates a new SPSC producer for the engine actor.
    /// Must be called before `play()` (which consumes the builders).
    pub fn engine_handle(&mut self) -> crate::api::public::UnregisteredEngineHandle {
        let handles = self.exposed();
        crate::api::public::UnregisteredEngineHandle::new(handles.engine)
    }

    /// Get the raw engine actor handle for advanced use cases.
    ///
    /// Creates a new SPSC producer for the engine actor.
    /// Must be called before `play()` (which consumes the builders).
    pub fn raw_engine_handle(&mut self) -> crate::api::private::EngineActorHandle {
        let handles = self.exposed();
        handles.engine
    }
}

#[cfg(test)]
mod tests {
    //! The engine's buffer bookkeeping, driven directly.
    //!
    //! `EngineHandler` is exercised here without a troupe, a display, or a thread: the handles
    //! it holds are real channel endpoints whose schedulers stay in the fixture, so every
    //! message the engine emits is observable by polling the corresponding spy. Buffers make
    //! the full round trip — the rasterizer spy hands back the exact `WindowMeta` it was given
    //! and the driver spy hands back the exact `Window`, in the order they went out — so the
    //! races these tests are about are expressible as a plain sequence of arrivals.

    use super::*;
    use crate::platform::ColorCube;
    use actor_scheduler::ActorScheduler;
    use pixelflow_graphics::render::rasterizer::{RasterControl, RasterManagement};
    use pixelflow_graphics::render::Frame;
    use std::time::Duration;

    /// Scheduler tuning for the fixture: nothing here approaches the buffer, and the burst
    /// limit only has to be big enough that one `poll_once` drains a test's worth of messages.
    const LANE_BURST: usize = 16;
    const LANE_BUFFER: usize = 16;

    /// A window size, as the tests talk about them.
    type Size = (u32, u32);

    #[derive(Default)]
    struct RasterizerSpy {
        requests: Vec<WindowMeta>,
    }

    impl Actor<RenderRequest<PlatformPixel, WindowMeta>, RasterControl, RasterManagement>
        for RasterizerSpy
    {
        fn handle_data(&mut self, req: RenderRequest<PlatformPixel, WindowMeta>) -> HandlerResult {
            self.requests.push(req.meta);
            Ok(())
        }
        fn handle_control(&mut self, _msg: RasterControl) -> HandlerResult {
            Ok(())
        }
        fn handle_management(&mut self, _msg: RasterManagement) -> HandlerResult {
            Ok(())
        }
        fn park(&mut self, _status: SystemStatus) -> Result<ActorStatus, HandlerError> {
            Ok(ActorStatus::Idle)
        }
    }

    #[derive(Default)]
    struct DriverSpy {
        presented: Vec<Window>,
    }

    impl Actor<DisplayData, DisplayControl, DisplayMgmt> for DriverSpy {
        fn handle_data(&mut self, msg: DisplayData) -> HandlerResult {
            match msg {
                DisplayData::Present { window } => self.presented.push(window),
            }
            Ok(())
        }
        fn handle_control(&mut self, _msg: DisplayControl) -> HandlerResult {
            Ok(())
        }
        fn handle_management(&mut self, _msg: DisplayMgmt) -> HandlerResult {
            Ok(())
        }
        fn park(&mut self, _status: SystemStatus) -> Result<ActorStatus, HandlerError> {
            Ok(ActorStatus::Idle)
        }
    }

    struct Rig {
        engine: EngineHandler,
        raster_sched: ActorScheduler<
            RenderRequest<PlatformPixel, WindowMeta>,
            RasterControl,
            RasterManagement,
        >,
        raster_spy: RasterizerSpy,
        driver_sched: ActorScheduler<DisplayData, DisplayControl, DisplayMgmt>,
        driver_spy: DriverSpy,
        /// Never polled: vsync's token returns and FPS reports are not what these tests are
        /// about. It is held only so the engine's sends to it stay connected.
        _vsync_sched: ActorScheduler<RenderedResponse, VsyncCommand, VsyncManagement>,
        /// Renders the rasterizer has been asked for and not yet answered, oldest first.
        in_flight: VecDeque<WindowMeta>,
        /// Buffers the driver has been asked to present and not yet returned, oldest first.
        holding: VecDeque<Window>,
        request_log: Vec<Size>,
        present_log: Vec<Size>,
    }

    impl Rig {
        fn new() -> Self {
            let (driver, driver_sched) = ActorScheduler::new(LANE_BURST, LANE_BUFFER);
            let (vsync, _vsync_sched) = ActorScheduler::new(LANE_BURST, LANE_BUFFER);
            let (rasterizer, raster_sched) = ActorScheduler::new(LANE_BURST, LANE_BUFFER);

            let engine = EngineHandler {
                driver,
                vsync,
                rasterizer: Some(rasterizer),
                self_handle: None,
                rasterizer_forward_handle: None,
                app_handle: None,
                frame_number: 0,
                window: None,
                current_generation: Generation::NONE,
                presented: VecDeque::new(),
                render_credit: Credit::new(1),
                render_threads: 1,
                pending_manifold: None,
            };

            Self {
                engine,
                raster_sched,
                raster_spy: RasterizerSpy::default(),
                driver_sched,
                driver_spy: DriverSpy::default(),
                _vsync_sched,
                in_flight: VecDeque::new(),
                holding: VecDeque::new(),
                request_log: Vec::new(),
                present_log: Vec::new(),
            }
        }

        fn feed(&mut self, data: EngineData) {
            self.engine
                .handle_data(data)
                .expect("engine handled the message");
        }

        /// Collect whatever the engine has emitted since the last look.
        fn pump(&mut self) {
            let _ = self.raster_sched.poll_once(&mut self.raster_spy);
            for meta in self.raster_spy.requests.drain(..) {
                self.request_log.push((meta.width_px, meta.height_px));
                self.in_flight.push_back(meta);
            }

            let _ = self.driver_sched.poll_once(&mut self.driver_spy);
            for window in self.driver_spy.presented.drain(..) {
                self.present_log.push((window.width_px, window.height_px));
                self.holding.push_back(window);
            }
        }

        fn window_created(&mut self, size: Size) {
            self.feed(EngineData::FromDriver(DisplayEvent::WindowCreated {
                window: window(size),
            }));
        }

        /// The driver's resize: a brand-new buffer at the new size, as the real one sends.
        fn resized(&mut self, size: Size) {
            self.feed(EngineData::FromDriver(DisplayEvent::Resized {
                window: window(size),
            }));
        }

        fn app_frame(&mut self) {
            self.feed(EngineData::FromApp(AppData::RenderSurface(manifold())));
        }

        /// Answer the oldest outstanding render, returning its `meta` untouched — which is the
        /// rasterizer's actual contract.
        fn complete_render(&mut self) {
            self.pump();
            let meta = self
                .in_flight
                .pop_front()
                .expect("a render must be outstanding to complete");
            self.feed(EngineData::RenderComplete(RenderResponse {
                frame: Frame::new(meta.width_px, meta.height_px),
                render_time: Some(Duration::from_millis(1)),
                meta,
            }));
        }

        /// Hand back the oldest buffer the driver is holding, which is the order a real driver
        /// returns them in.
        fn return_buffer(&mut self) {
            self.pump();
            let window = self
                .holding
                .pop_front()
                .expect("a buffer must be presented before it can be returned");
            self.feed(EngineData::PresentComplete(window));
        }

        /// Render requests emitted since the last call.
        fn render_requests(&mut self) -> Vec<Size> {
            self.pump();
            std::mem::take(&mut self.request_log)
        }

        /// Frames handed to the driver since the last call.
        fn presented(&mut self) -> Vec<Size> {
            self.pump();
            std::mem::take(&mut self.present_log)
        }
    }

    fn window((width_px, height_px): Size) -> Window {
        Window {
            id: WindowId(1),
            frame: Frame::new(width_px, height_px),
            width_px,
            height_px,
            scale: 1.0,
        }
    }

    /// A constant black scene — these tests care about which buffer a render is aimed at, not
    /// what is in it.
    fn manifold() -> Arc<dyn Manifold<Output = Discrete> + Send + Sync> {
        Arc::new(ColorCube::default().at(0.0f32, 0.0f32, 0.0f32, 1.0f32))
    }

    /// The steady state, as a baseline for the race tests below: one render per app frame, each
    /// presented, with the one buffer circulating back through `PresentComplete`.
    #[test]
    fn frames_circulate_through_render_present_and_back() {
        let mut rig = Rig::new();
        rig.window_created((100, 100));

        rig.app_frame();
        assert_eq!(rig.render_requests(), vec![(100, 100)]);

        rig.complete_render();
        assert_eq!(rig.presented(), vec![(100, 100)]);

        // Frame arrives while the buffer is still with the driver: queued, not dropped.
        rig.app_frame();
        assert!(rig.render_requests().is_empty());

        rig.return_buffer();
        assert_eq!(rig.render_requests(), vec![(100, 100)]);
    }

    /// Vsync keeps asking for frames at 60Hz regardless of how long a render takes, so an
    /// `AppData::RenderSurface` routinely arrives while one is in flight. If a resize landed
    /// first, the window that arrival carries is the *new* buffer, and refusing the render used
    /// to drop it — after which the old-size render completed into an engine that no longer had
    /// anything newer to compare against, and the resize was gone for good.
    #[test]
    fn a_frame_refused_mid_render_keeps_the_resized_window() {
        let mut rig = Rig::new();
        rig.window_created((100, 100));

        rig.app_frame();
        assert_eq!(rig.render_requests(), vec![(100, 100)]);

        // Resize lands while that render is still in flight.
        rig.resized((200, 200));
        assert!(
            rig.render_requests().is_empty(),
            "the in-flight render holds the edge's only credit"
        );

        // Vsync asks again and the app answers, still mid-render.
        rig.app_frame();
        assert!(
            rig.render_requests().is_empty(),
            "still no credit — but the refusal must not consume anything"
        );

        // The old-size render finally completes.
        rig.complete_render();
        assert_eq!(
            rig.render_requests(),
            vec![(200, 200)],
            "the deferred frame re-renders into the buffer the resize delivered"
        );
        assert!(
            rig.presented().is_empty(),
            "the old-size frame is stale and must not reach the driver"
        );
    }

    /// A buffer coming back from the driver mid-render is the *older* one — the resize that
    /// started the current render is what superseded it. Parking it as the live buffer made the
    /// engine discard the good new-size frame and re-render into the stale one instead.
    #[test]
    fn a_buffer_returning_mid_render_is_not_mistaken_for_a_resize() {
        let mut rig = Rig::new();
        rig.window_created((100, 100));

        rig.app_frame();
        assert_eq!(rig.render_requests(), vec![(100, 100)]);
        rig.complete_render();
        assert_eq!(rig.presented(), vec![(100, 100)]);

        // Resize arrives before the driver hands the presented buffer back.
        rig.resized((200, 200));
        rig.app_frame();
        assert_eq!(
            rig.render_requests(),
            vec![(200, 200)],
            "the buffer the resize delivered is free to render into"
        );

        // Now the old buffer comes back, mid-render.
        rig.return_buffer();
        assert!(
            rig.engine.window.is_none(),
            "a superseded buffer must not be parked as the live one"
        );

        rig.complete_render();
        assert_eq!(
            rig.presented(),
            vec![(200, 200)],
            "the new-size frame is current and must be presented"
        );
        assert!(
            rig.render_requests().is_empty(),
            "nothing to re-render: that frame was not stale"
        );
    }

    /// Both buffers out at once, returning oldest-first. Nothing observable distinguishes them
    /// at the moment they come back — same `Window` shape, and "a render is outstanding" is
    /// true for the *stale* one's return as easily as the live one's. Only the generation does,
    /// which is why it is stamped rather than inferred.
    #[test]
    fn two_buffers_returning_in_order_do_not_revert_the_resize() {
        let mut rig = Rig::new();
        rig.window_created((100, 100));

        rig.app_frame();
        rig.complete_render();
        assert_eq!(rig.presented(), vec![(100, 100)]);

        // Resize while the first buffer is still with the driver, then render and present the
        // second one too: now both are out.
        rig.resized((200, 200));
        rig.app_frame();
        rig.complete_render();
        assert_eq!(rig.presented(), vec![(200, 200)]);
        let _ = rig.render_requests();

        // The stale buffer comes back first.
        rig.return_buffer();
        rig.app_frame();
        assert!(
            rig.render_requests().is_empty(),
            "the superseded buffer must not be rendered into"
        );

        // Then the live one.
        rig.return_buffer();
        assert_eq!(
            rig.render_requests(),
            vec![(200, 200)],
            "the live buffer resumes rendering at the resized dimensions"
        );
    }

    /// A resize with no render in flight renders into the new buffer immediately.
    #[test]
    fn a_resize_between_frames_renders_at_the_new_size() {
        let mut rig = Rig::new();
        rig.window_created((100, 100));

        rig.app_frame();
        assert_eq!(rig.render_requests(), vec![(100, 100)]);
        rig.complete_render();
        rig.return_buffer();

        rig.app_frame();
        assert_eq!(rig.render_requests(), vec![(100, 100)]);
        rig.complete_render();

        // Buffer is with the driver; the resize brings its own.
        rig.resized((200, 200));
        rig.app_frame();
        assert_eq!(rig.render_requests(), vec![(200, 200)]);
    }
}
