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
    /// The active window (owns frame buffer, returned by driver after presentation).
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
                if let Some(manifold) = self.pending_manifold.take() {
                    if let Some(window) = self.window.take() {
                        self.trigger_render_with_window(manifold, window);
                    } else {
                        // Window unavailable, keep manifold pending
                        self.pending_manifold = Some(manifold);
                    }
                }
            }
            EngineData::RenderComplete(response) => {
                // The render is done, so the edge's one credit is free again.
                self.render_credit.release();

                // Staleness is now something we *observe* rather than something the resize
                // handler reaches in and marks. The engine hands its window to the rasterizer
                // by value, so `self.window` is None for the whole duration of a render — the
                // only way it can be Some here is if a resize delivered a newer one meanwhile.
                // That is exactly the condition the `stale` flag encoded, read off the state
                // that already implies it.
                if let Some(current) = self.window.take() {
                    log::debug!(
                        "Discarding stale render ({}x{}) - resize to {}x{} happened during render",
                        response.meta.width_px,
                        response.meta.height_px,
                        current.width_px,
                        current.height_px
                    );
                    // Drop the stale frame and re-render at the correct size.
                    if let Some(manifold) = self.pending_manifold.take() {
                        self.trigger_render_with_window(manifold, current);
                    } else {
                        self.window = Some(current);
                    }
                } else {
                    // Reassembled from the metadata that travelled with the request.
                    let window = Window {
                        id: response.meta.id,
                        frame: response.frame,
                        width_px: response.meta.width_px,
                        height_px: response.meta.height_px,
                        scale: response.meta.scale,
                    };
                    self.present_cooked_frame(response.render_time, window);
                }
            }
            EngineData::PresentComplete(returned_window) => {
                // Driver returned the window after presentation.
                // IMPORTANT: If a resize happened while presenting, self.window already
                // contains the NEW resized window from the driver. Don't overwrite it
                // with the stale returned window - that would lose the new dimensions forever.
                if self.window.is_none() {
                    self.window = Some(returned_window);
                } else {
                    // A resize happened - we already have the new window.
                    // Discard the old returned window (its dimensions are stale).
                    log::debug!(
                        "Discarding stale window from PresentComplete (resize happened): {}x{}",
                        returned_window.width_px,
                        returned_window.height_px
                    );
                }

                // Notify VSync for FPS tracking (actual rasterization completion)
                self.vsync
                    .send(Message::Data(RenderedResponse {
                        frame_number: self.frame_number,
                        rendered_at: Instant::now(),
                    }))
                    .expect("Failed to notify VSync of completed frame");

                // Check if we have a pending manifold waiting for this window
                if let Some(manifold) = self.pending_manifold.take() {
                    let window = self.window.take().unwrap();
                    log::trace!("Engine: Catching up - rendering pending manifold");
                    self.trigger_render_with_window(manifold, window);
                }
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

                if let Some(window) = self.window.take() {
                    // Window available - render immediately
                    log::debug!("Engine: Window available, triggering render");
                    self.trigger_render_with_window(manifold, window);
                } else {
                    // Window still with driver - queue manifold
                    self.pending_manifold = Some(manifold);
                    log::debug!("Engine: Window busy, manifold queued");
                }
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
        // Extract frame from window for rasterization, store metadata for later
        let Window {
            id,
            frame,
            width_px,
            height_px,
            scale,
        } = window;

        // Take the edge's single credit. Callers already gate on `outstanding()`, so this
        // failing means a second render was started while one was in flight — the bound the
        // topology declares, now enforced rather than implied by an Option being Some.
        if !self.render_credit.try_consume() {
            log::warn!("Render already in flight, dropping request");
            return;
        }

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
    fn present_cooked_frame(&mut self, render_time: std::time::Duration, window: Window) {
        // Send window to driver for presentation (transfers ownership)
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
                self.window = Some(window);
                log::debug!("Engine: Window stored from WindowCreated");

                // Check if we have a pending manifold waiting for this window
                if let Some(manifold) = self.pending_manifold.take() {
                    log::debug!("Engine: Found pending manifold, triggering render");
                    if let Some(window) = self.window.take() {
                        self.trigger_render_with_window(manifold, window);
                    }
                } else {
                    log::debug!("Engine: No pending manifold");
                }

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

                // Nothing to mark: storing the new window below *is* the staleness signal, and
                // `RenderComplete` reads it off that rather than off a flag set from here.

                // Update window with new one from driver
                self.window = Some(window);

                // DON'T start a new render here if one is already in flight. The stale render
                // will complete, be discarded, and then we render at the correct dimensions.
                // Starting one now would be refused by the credit anyway — this check is what
                // keeps that from becoming a dropped frame and a warning.
                if self.render_credit.outstanding() == 0 {
                    if let Some(manifold) = self.pending_manifold.take() {
                        if let Some(window) = self.window.take() {
                            self.trigger_render_with_window(manifold, window);
                        }
                    }
                }

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
