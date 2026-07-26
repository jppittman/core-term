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
use actor_scheduler::{
    Actor, ActorHandle, ActorStatus, ActorTypes, HandlerError, HandlerResult, Message, SendError,
    SystemStatus, TroupeActor,
};
use crate::render_coordinator::{Completed, RenderCoordinator, Step};
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
    ///
    /// Stays here for now. §7.3 sends it to the rasterizer, whose only consumer is the FPS
    /// telemetry edge — but that edge is `rasterizer → vsync`, which does not exist until the
    /// coordinator is its own node, so the move belongs with that slice rather than this one.
    frame_number: u64,
    /// Number of render threads for work-stealing parallelism.
    render_threads: usize,
    /// When to render and what to render into.
    ///
    /// All of it — the borrowed buffer, the outstanding-request latch, the one-render credit,
    /// and the keep-latest kernel slot — lives behind this, and none of it is reachable from the
    /// mediator's other responsibilities. What remains of `EngineHandler` on this path is
    /// delivering the [`Step`]s it hands back.
    render: RenderCoordinator,
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

                // The tick is also what retries a request that was dropped in transit.
                let step = self.render.advance();
                self.deliver(step);
            }
            EngineData::RenderComplete(response) => {
                // No staleness check here any more. Whether this buffer is still the one the
                // driver wants is the driver's question, asked against state the driver owns —
                // and it has to be asked there regardless, because a resize can land after this
                // point too. Asking it in both places would be two answers that can disagree.
                let Completed { present, next } = self.render.completed(response);
                match present {
                    Some((window, render_time)) => {
                        self.present_cooked_frame(render_time, window)
                    }
                    // The rasterizer was paused, so it handed the buffer back unrendered.
                    // Presenting it would blit whatever stale pixels it still holds; keeping it
                    // is the whole point of the frame coming back at all. The coordinator has
                    // retained it.
                    None => {
                        log::debug!("Render skipped (paused); retaining the buffer unpresented")
                    }
                }
                self.deliver(next);
            }
            EngineData::WindowGranted(window) => {
                let step = self.render.granted(window);
                self.deliver(step);
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
    /// Deliver whatever the coordinator decided.
    ///
    /// The single place the render protocol reaches the wire, so the coordinator can stay
    /// handle-free and every send failure is handled once rather than at each call site.
    fn deliver(&mut self, step: Step) {
        match step {
            Step::Idle => {}
            Step::RequestWindow => {
                // A refused ask is not lost: the driver latches it and answers when a buffer
                // frees. A send that fails leaves the latch clear, so the next tick retries —
                // the latch only covers requests that actually arrived.
                match self
                    .driver
                    .send(Message::Management(DisplayMgmt::RequestWindow))
                {
                    Ok(()) => self.render.request_sent(),
                    Err(e) => {
                        log::debug!(
                            "Window request not delivered ({e}); retrying on the next tick"
                        );
                    }
                }
            }
            Step::Render(request) => self.send_render(request),
        }
    }

    /// Hand a bound frame to the rasterizer.
    ///
    /// NOTE: this send carries the framebuffer, and `ActorHandle::send` takes the message by
    /// value while `SendError` carries nothing back — so a failure here destroys the driver's
    /// only buffer and the display can never recover. Releasing the credit keeps the *edge*
    /// usable, but there is no buffer left to use it with. Pre-existing, and deliberately left
    /// as-is by the extraction that moved this code rather than changed mid-refactor; the
    /// `Present` send takes the opposite stance (`.expect`) for the same hazard, so the two
    /// disagree and one of them is wrong.
    fn send_render(&mut self, request: RenderRequest<PlatformPixel, WindowMeta>) {
        let Some(rasterizer) = &self.rasterizer else {
            log::warn!("Rasterizer not initialized, dropping render request");
            self.render.render_send_failed();
            return;
        };
        if let Err(e) = rasterizer.send(Message::Data(request)) {
            log::warn!("Failed to send render request to rasterizer: {}", e);
            self.render.render_send_failed();
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
                let step = self.render.submit(manifold);
                self.deliver(step);
            }
            AppData::Skipped => {
                // App says nothing to render - return token anyway
                self.return_vsync_token();
            }
        }
    }

    /// Hand the drawn buffer back to the driver to be shown.
    ///
    /// This *is* the return: the driver is the buffer's resting owner, so there is no separate
    /// acknowledgement to wait for and nothing to remember about what went out. The coordinator
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
                let step = self.render.advance();
                self.deliver(step);

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
                let step = self.render.advance();
                self.deliver(step);

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
            render: RenderCoordinator::new(),
            render_threads: 1, // Default, will be set by Configure message
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
    use pixelflow_core::{Discrete, Manifold};
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
                render: RenderCoordinator::new(),
                render_threads: 1,
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
            self.queue_frame();
            self.pump();
        }

        /// A frame, without letting anything be delivered. Two of these back to back is one
        /// scheduler pass in which the engine reacts twice before the driver reacts once —
        /// which is ordinary, since they are separate actors.
        fn queue_frame(&mut self) {
            self.feed(EngineData::FromApp(AppData::RenderSurface(manifold())));
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
            !rig.engine.render.holds_buffer(),
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
            !rig.engine.render.holds_buffer(),
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

    /// The same failure one step earlier in the conversation: two asks issued *before* the
    /// driver has answered either. The credit guard cannot see this one — no render is in flight
    /// yet, so both asks are legitimate at the moment they are made.
    ///
    /// The driver answers the first and has nothing for the second, which leaves its "somebody
    /// is waiting" latch set with nobody actually waiting any more. The next buffer to appear —
    /// a resize allocation, while the granted one is out being drawn into — is then handed over
    /// unasked, and the engine is holding two.
    #[test]
    fn a_second_ask_before_the_first_is_answered_does_not_earn_a_second_grant() {
        let mut rig = Rig::new();
        rig.surface((100, 100));

        // Two asks in one scheduler pass, before the driver reacts to either.
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
            !rig.engine.render.holds_buffer(),
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
            !rig.engine.render.holds_buffer(),
            "no grant can arrive while the buffer is out, however often it is requested"
        );

        rig.complete_render();
        assert_eq!(rig.blitted(), vec![(100, 100)]);
    }
}
