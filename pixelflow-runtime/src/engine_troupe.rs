//! Engine Troupe - Render pipeline actor coordination using troupe! macro.

use crate::api::private::{EngineControl, EngineData};
use crate::api::public::{
    AppData, AppManagement, Application, EngineEvent, EngineEventControl, EngineEventData,
    EngineEventManagement, WindowId,
};
use crate::config::EngineConfig;
use crate::display::driver::DriverActor;
use crate::display::messages::{
    DisplayControl, DisplayData, DisplayEvent, DisplayMgmt, Window, WindowMeta,
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
use std::sync::Arc;
use std::time::Instant;

const LOG_FRAME_INTERVAL: u64 = 60;

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
    /// The window buffer, while it is on loan from the driver.
    ///
    /// Borrowed, not owned: it arrives via `WindowGranted` in answer to a request and goes back
    /// with `Present`. Holding it is the whole of "a render can start" — there is no separate
    /// flag, and no generation kept alongside it, because the buffer carries its own.
    ///
    /// There is deliberately no "a request is outstanding" flag either. Re-asking is harmless: a
    /// grant requires the driver to be holding the buffer, and there is exactly one, so no
    /// number of requests can produce two grants. Ownership is the bound.
    window: Option<Window>,
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

                // No staleness check here any more. Whether this buffer is still the one the
                // driver wants is the driver's question, asked against state the driver owns —
                // and it has to be asked there regardless, because a resize can land after this
                // point too. Asking it in both places would be two answers that can disagree.
                let window = Window::rejoin(response.frame, response.meta);
                match response.render_time {
                    Some(render_time) => self.present_cooked_frame(render_time, window),
                    // The rasterizer was paused, so it handed the buffer back unrendered.
                    // Presenting it would blit whatever stale pixels it still holds; keeping it
                    // is the whole point of the frame coming back at all. This arm exists
                    // because `render_time: Option<Duration>` forces the question — the buffer's
                    // return is unconditional, its having been drawn into is not.
                    None => {
                        log::debug!("Render skipped (paused); retaining the buffer unpresented");
                        self.hold(window);
                    }
                }
            }
            EngineData::WindowGranted(window) => {
                self.hold(window);
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
    /// Take the buffer into hand.
    ///
    /// Every path that acquires one goes through here so the "at most one buffer" invariant has
    /// a single home rather than an assertion repeated at each arrival. It is a real `assert!`,
    /// not a `debug_assert!`, because the release-build alternative is worse than a crash: the
    /// buffer being overwritten is the driver's only current one, and losing it leaves a
    /// terminal that never draws again and cannot recover without being killed.
    fn hold(&mut self, window: Window) {
        assert!(
            self.window.is_none(),
            "a second buffer (generation {}) arrived while one was already in hand",
            window.generation
        );
        self.window = Some(window);
    }

    /// Render if both halves of one are in hand, and otherwise go and get the missing half.
    ///
    /// Every path that acquires either half ends here, so "can we render yet?" is asked in one
    /// place instead of being open-coded at each arrival. Nothing is consumed unless a render
    /// actually starts — a refused render puts both back (see `trigger_render_with_window`),
    /// which is what lets the next completion pick them up.
    fn render_if_ready(&mut self) {
        if self.pending_manifold.is_none() {
            // Nothing to draw. Deliberately does *not* ask for the buffer: holding it idle would
            // block nothing today, but it makes "who has the buffer?" stop meaning "who is
            // drawing?", and that equivalence is what bounds the loop.
            return;
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
            if self.render_credit.outstanding() == 0 {
                self.request_window();
            }
            return;
        };
        let manifold = self
            .pending_manifold
            .take()
            .expect("pending_manifold checked Some above");
        self.trigger_render_with_window(manifold, window);
    }

    /// Ask the driver for the buffer.
    ///
    /// Unanswered if the buffer is out — there is no refusal to handle, by design. The next
    /// vsync tick calls `render_if_ready` again, which asks again; that retry is why the request
    /// is allowed to be lost in the first place (it carries nothing, so losing it costs a frame
    /// rather than the only framebuffer).
    fn request_window(&self) {
        if let Err(e) = self
            .driver
            .send(Message::Management(DisplayMgmt::RequestWindow))
        {
            log::debug!("Window request not delivered ({e}); retrying on the next tick");
        }
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
        window: Window,
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
                window.generation,
                window.width_px,
                window.height_px
            );
            self.hold(window);
            self.pending_manifold = Some(manifold);
            return;
        }

        // The frame goes to the rasterizer; the rest travels with the request and comes back
        // untouched, so nothing has to be stashed here in the meantime.
        let (frame, meta) = window.tear();
        let WindowMeta {
            width_px, height_px, ..
        } = meta;

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

        let request = RenderRequest {
            manifold,
            frame,
            meta,
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

    /// Hand the drawn buffer back to the driver to be shown.
    ///
    /// This *is* the return: the driver is the buffer's resting owner, so there is no separate
    /// acknowledgement to wait for and nothing to remember about what went out. The engine
    /// simply has no buffer again afterwards, and asks for one when it next has something to
    /// draw.
    fn present_cooked_frame(&mut self, render_time: std::time::Duration, window: Window) {
        let t1 = Instant::now();
        self.driver
            .send(Message::Data(DisplayData::Present { window }))
            .expect("Failed to send window to driver for presentation");
        let send_time = t1.elapsed();

        self.frame_number += 1;
        // FPS telemetry, on the frame actually reaching the driver. It used to ride
        // `PresentComplete`, which no longer exists as a message.
        self.vsync
            .send(Message::Data(RenderedResponse {
                frame_number: self.frame_number,
                rendered_at: Instant::now(),
            }))
            .expect("Failed to notify VSync of completed frame");

        if self.frame_number.is_multiple_of(LOG_FRAME_INTERVAL) {
            log::info!(
                "Frame {}: render={:?}, send={:?}",
                self.frame_number,
                render_time,
                send_time
            );
        }

        // Catch up on whatever arrived while the render was in flight. The buffer has just gone
        // to the driver, so in practice this asks for it back.
        self.render_if_ready();
    }

    /// Handle events from the display driver
    fn handle_driver_event(&mut self, event: DisplayEvent) {
        match event {
            // Both window-lifecycle events are now pure relays: the driver has already built
            // the buffer for the new geometry by the time this arrives, so there is nothing here
            // to take delivery of, stamp, or retire. What is left is telling the app its size.
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
                self.render_if_ready();

                if let Some(app) = &self.app_handle {
                    app.send(EngineEvent::Control(EngineEventControl::WindowCreated {
                        id: surface.id,
                        width_px: surface.width_px,
                        height_px: surface.height_px,
                        scale: surface.scale,
                    }))
                    .expect("Failed to relay WindowCreated event to app");
                }
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
                self.render_if_ready();

                if let Some(app) = &self.app_handle {
                    app.send(EngineEvent::Control(EngineEventControl::Resized {
                        id: surface.id,
                        width_px: surface.width_px,
                        height_px: surface.height_px,
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
    //! The render pipeline, driven directly.
    //!
    //! `EngineHandler` is exercised here without a troupe, a display, or a thread: the handles
    //! it holds are real channel endpoints whose schedulers stay in the fixture, so every
    //! message the engine emits is observable by polling the corresponding spy.
    //!
    //! The driver spy owns a **real [`WindowKeeper`]** rather than imitating one. Buffer
    //! ownership is the driver's now, so a fixture that mocked it would be asserting against a
    //! second implementation of the thing under test — and the resize races below are precisely
    //! the interaction between the two sides, not the behaviour of either alone.

    use super::*;
    use crate::display::messages::Surface;
    use crate::display::window_keeper::{Presented, WindowKeeper};
    use crate::platform::ColorCube;
    use actor_scheduler::ActorScheduler;
    use pixelflow_graphics::render::rasterizer::{RasterControl, RasterManagement};
    use pixelflow_graphics::render::Frame;
    use std::collections::VecDeque;
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

    /// A driver, minus the platform: real buffer ownership, no blit.
    #[derive(Default)]
    struct DriverSpy {
        keeper: WindowKeeper,
        /// Buffers lent out, waiting to reach the engine. A real `PlatformActor` sends these
        /// straight back over its engine handle; the fixture has no such handle, so `Rig` drains
        /// this in `pump`.
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
        request_log: Vec<Size>,
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
                request_log: Vec::new(),
            }
        }

        fn feed(&mut self, data: EngineData) {
            self.engine
                .handle_data(data)
                .expect("engine handled the message");
        }

        /// Carry every message in flight to its destination, in both directions, until nothing
        /// moves. The request → grant → render loop takes several hops, so a single pass would
        /// leave the system mid-conversation and the assertions would read a partial state.
        fn pump(&mut self) {
            loop {
                let _ = self.raster_sched.poll_once(&mut self.raster_spy);
                for meta in self.raster_spy.requests.drain(..) {
                    self.request_log.push((meta.width_px, meta.height_px));
                    self.in_flight.push_back(meta);
                }

                let _ = self.driver_sched.poll_once(&mut self.driver_spy);
                let granted: Vec<_> = self.driver_spy.granted.drain(..).collect();
                if granted.is_empty() {
                    return;
                }
                for window in granted {
                    self.feed(EngineData::WindowGranted(window));
                }
            }
        }

        /// The platform reported a window at this geometry. Both halves of what
        /// `PlatformActor::flush` does: the keeper builds the buffer, the engine is told the
        /// size so it can relay it. Used for the initial window and for resizes alike, because
        /// the driver treats them identically.
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
            self.feed(EngineData::FromDriver(DisplayEvent::Resized { surface }));
            self.pump();
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
            self.pump();
        }

        fn app_frame(&mut self) {
            self.feed(EngineData::FromApp(AppData::RenderSurface(manifold())));
            self.pump();
        }

        /// A vsync tick, which is what re-drives a request the driver could not answer.
        fn tick(&mut self) {
            let now = Instant::now();
            self.feed(EngineData::VSync {
                timestamp: now,
                target_timestamp: now,
                refresh_interval: Duration::from_millis(16),
            });
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
    }

    /// A constant black scene — these tests care about which buffer a render is aimed at, not
    /// what is in it.
    fn manifold() -> Arc<dyn Manifold<Output = Discrete> + Send + Sync> {
        Arc::new(ColorCube::default().at(0.0f32, 0.0f32, 0.0f32, 1.0f32))
    }

    /// The steady state: the engine asks for the buffer only when it has something to draw,
    /// draws, hands it straight back, and asks again next frame.
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

    /// Nothing is requested until there is something to draw. If the engine held the buffer
    /// while idle, "who has the buffer" would stop meaning "who is drawing", and that
    /// equivalence is the entire bound on the loop.
    #[test]
    fn an_idle_engine_does_not_hold_the_buffer() {
        let mut rig = Rig::new();
        rig.surface((100, 100));

        rig.tick();
        rig.tick();
        assert!(rig.render_requests().is_empty());
        assert!(
            rig.engine.window.is_none(),
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
    /// frame makes the engine ask for a buffer it cannot yet use; the resize then allocates one,
    /// so the ask is answered immediately and the engine is holding a second buffer while the
    /// first is still out. The old completion then arrives with nowhere to go.
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
            rig.engine.window.is_none(),
            "a buffer is already out with the renderer; a second one must not be granted"
        );

        // The original render completes into an engine that still has exactly one buffer's
        // worth of state to reconcile.
        rig.complete_render();
        assert!(
            rig.blitted().is_empty(),
            "the old-size frame is superseded"
        );
        assert_eq!(
            rig.render_requests(),
            vec![(200, 200)],
            "and the queued frame renders into the resize's buffer"
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
            rig.engine.window.is_none(),
            "no grant can arrive while the buffer is out, however often it is requested"
        );

        rig.complete_render();
        assert_eq!(rig.blitted(), vec![(100, 100)]);
    }
}
