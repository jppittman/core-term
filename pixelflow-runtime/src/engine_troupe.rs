//! Engine Troupe - Render pipeline actor coordination using troupe! macro.
//!
//! `EngineHandler` is the thin adapter around [`EngineCore`] (`engine_core.rs`): the core
//! decides, in the mealy-transducer style, what should happen and returns it as an
//! [`EngineOut`] word; `flush` is this file's hand-written `Wiring` — the one place that word
//! meets real channels.

use crate::api::private::{EngineControl, EngineData};
use crate::api::public::{AppManagement, Application};
use crate::config::EngineConfig;
use crate::display::driver::DriverActor;
use crate::display::messages::{DisplayControl, DisplayData, DisplayMgmt, WindowMeta};
use crate::display::platform::PlatformActor;
use crate::engine_core::{EngineCore, EngineOut};
use crate::error::RuntimeError;
use crate::platform::{ActivePlatform, PlatformPixel};
use crate::vsync_actor::{
    RenderedResponse, VsyncActor, VsyncCommand, VsyncConfig, VsyncManagement,
};
use actor_scheduler::mealy::Transducer;
use actor_scheduler::{
    Actor, ActorBuilder, ActorHandle, ActorScheduler, ActorStatus, ActorTypes, HandlerError,
    HandlerResult, Message, SendError, SystemStatus, TroupeActor,
};
use pixelflow_graphics::render::rasterizer::{
    RasterizerActor, RasterizerHandle, RenderRequest, RenderResponse,
};
use std::convert::Infallible;
use std::sync::Arc;

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
    /// Handle to the rasterizer response-forwarding actor (spawned alongside the rasterizer
    /// in `spawn_rasterizer`; `None` until then, same as `rasterizer` itself).
    rasterizer_forwarder: Option<ActorHandle<Infallible, Infallible, Infallible>>,
    /// Handle to the application (for event forwarding).
    app_handle: Option<Arc<dyn Application + Send + Sync>>,
    /// Number of render threads for work-stealing parallelism.
    render_threads: usize,
    /// The pure mediator: decides what to do with each message and returns it as an
    /// [`EngineOut`] word for `flush` to deliver. Owns none of the handles above.
    core: EngineCore,
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
        let out = self.core.step_data(data)?;
        self.flush(out);
        Ok(())
    }

    fn handle_control(&mut self, ctrl: EngineControl) -> HandlerResult {
        // Handle-carrying: intercepted before the core ever sees it, since a pure core has no
        // field to hold an `ActorHandle` in.
        let ctrl = match ctrl {
            EngineControl::VsyncActorReady(handle) => {
                self.vsync = handle;
                return Ok(());
            }
            other => other,
        };
        let out = self.core.step_control(ctrl)?;
        self.flush(out);
        Ok(())
    }

    fn handle_management(&mut self, mgmt: AppManagement) -> HandlerResult {
        // Handle-carrying / bootstrap messages: intercepted before the core ever sees them, for
        // the same reason as `EngineControl::VsyncActorReady` above.
        let mgmt = match mgmt {
            AppManagement::SetRasterizerForwardHandle(handle) => {
                self.rasterizer_forward_handle = Some(handle);
                return Ok(());
            }
            AppManagement::Configure(config) => {
                self.render_threads = config.performance.render_threads;
                log::info!("Engine configured: {} render threads", self.render_threads);

                // Spawn rasterizer with bootstrap pattern
                self.spawn_rasterizer();
                return Ok(());
            }
            AppManagement::RegisterApp(app) => {
                log::info!("Application handle registered");
                self.app_handle = Some(app);
                return Ok(());
            }
            other => other,
        };
        let out = self.core.step_management(mgmt)?;
        self.flush(out);
        Ok(())
    }

    fn handle_os(&mut self, _status: SystemStatus) -> Result<ActorStatus, HandlerError> {
        // engine has no external channels which might be busy
        Ok(ActorStatus::Idle)
    }
}

/// Bridges the rasterizer's response channel into the engine's own actor system.
///
/// [`RasterizerActor`] hands completed frames back over a bare `mpsc::Sender` so
/// pixelflow-graphics stays decoupled from any particular consumer's message enum. This actor
/// is the one place that channel meets [`EngineData::RenderComplete`] — `handle_os` blocking on
/// `response_rx.recv()` *is* the actor, the same way `PtyReader::handle_os` blocking on
/// `epoll_wait` is that actor's entire job.
///
/// The scheduler only calls `handle_os` after the doorbell has woken at least once, so — same
/// as `PtyReader` waiting for its first `Bind` — this actor needs one doorbell ring to start its
/// loop; `spawn_rasterizer` rings it right after spawning the thread. Once `handle_os` returns
/// `Busy` the scheduler keeps re-entering it without waiting on the doorbell again, so from then
/// on the loop is self-sustaining.
struct RasterizerForwarder {
    response_rx: std::sync::mpsc::Receiver<RenderResponse<PlatformPixel, WindowMeta>>,
    engine: ActorHandle<EngineData, EngineControl, AppManagement>,
    /// Sends itself `Shutdown` once the rasterizer drops its sender, so the scheduler loop
    /// (and the OS thread underneath it) actually exits instead of parking on an empty
    /// doorbell forever. `Quit`/`AppManagement::Quit`/`CloseRequested` also send `Shutdown`
    /// here directly; this is the fallback for whichever signal arrives second.
    self_handle: Option<ActorHandle<Infallible, Infallible, Infallible>>,
}

impl ActorTypes for RasterizerForwarder {
    type Data = Infallible;
    type Control = Infallible;
    type Management = Infallible;
}

impl Actor<Infallible, Infallible, Infallible> for RasterizerForwarder {
    fn handle_data(&mut self, msg: Infallible) -> HandlerResult {
        match msg {}
    }

    fn handle_control(&mut self, msg: Infallible) -> HandlerResult {
        match msg {}
    }

    fn handle_management(&mut self, msg: Infallible) -> HandlerResult {
        match msg {}
    }

    fn handle_os(&mut self, _status: SystemStatus) -> Result<ActorStatus, HandlerError> {
        match self.response_rx.recv() {
            Ok(response) => {
                if let Err(e) = self
                    .engine
                    .send(Message::Data(EngineData::RenderComplete(response)))
                {
                    log::warn!("Failed to forward render response to engine: {}", e);
                    self.shut_down();
                    return Ok(ActorStatus::Idle);
                }
                Ok(ActorStatus::Busy)
            }
            Err(_) => {
                // The rasterizer shut down and dropped its sender; there is nothing left to
                // forward, so this actor's work is done too.
                self.shut_down();
                Ok(ActorStatus::Idle)
            }
        }
    }
}

impl RasterizerForwarder {
    fn shut_down(&mut self) {
        if let Some(handle) = self.self_handle.take() {
            if let Err(e) = handle.send(Message::Shutdown) {
                log::debug!("Rasterizer forwarder self-shutdown send failed: {}", e);
            }
        }
    }
}

impl EngineHandler {
    /// Deliver an [`EngineOut`] — the hand-written `Wiring` for [`EngineCore`]'s output word,
    /// until the engine runs as a green node behind the generated port/wiring machinery.
    ///
    /// Order matters here in one place: `app` is flushed first and `quit` last, so a
    /// `CloseRequested` (which sets both `app` and `quit` in the same word) reaches the app
    /// before the shutdown cascade drops its handle — matching what all three quit paths did
    /// before this split.
    fn flush(&mut self, out: EngineOut) {
        if let Some(event) = out.app {
            // Silently dropped if no app is registered yet, matching every `if let Some(app) =
            // &self.app_handle` guard this replaces.
            if let Some(app) = &self.app_handle {
                app.send(event)
                    .expect("failed to send to app. it probably crashed");
            }
        }

        if let Some(ctrl) = out.driver_control {
            self.driver
                .send(Message::Control(ctrl))
                .expect("Failed to relay control to driver");
        }

        if let Some(mgmt) = out.driver_mgmt {
            self.send_driver_mgmt(mgmt);
        }

        if let Some(request) = out.rasterizer {
            self.send_render(request);
        }

        if let Some(data) = out.driver_data {
            self.driver
                .send(Message::Data(data))
                .expect("Failed to send window to driver for presentation");
        }

        if let Some(response) = out.vsync_data {
            self.vsync
                .send(Message::Data(response))
                .expect("Failed to notify VSync of completed frame");
        }

        if let Some(cmd) = out.vsync_control {
            self.send_vsync_control(cmd);
        }

        if out.quit {
            self.shut_down();
        }
    }

    /// The `driver_mgmt` port. `RequestWindow` needs the delivery-feedback seam described on
    /// [`EngineCore::request_delivered`]; every other management message is a plain relay.
    fn send_driver_mgmt(&mut self, mgmt: DisplayMgmt) {
        match mgmt {
            DisplayMgmt::RequestWindow => {
                // A refused ask is not lost: the driver latches it and answers when a buffer
                // frees. A send that fails leaves the latch clear, so the next tick retries —
                // the latch only covers requests that actually arrived.
                match self
                    .driver
                    .send(Message::Management(DisplayMgmt::RequestWindow))
                {
                    Ok(()) => self.core.request_delivered(),
                    Err(e) => {
                        log::debug!(
                            "Window request not delivered ({e}); retrying on the next tick"
                        );
                    }
                }
            }
            other => {
                self.driver
                    .send(Message::Management(other))
                    .expect("Failed to relay management to driver");
            }
        }
    }

    /// The `vsync_control` port. `ReturnToken` tolerates a disconnected vsync (shutdown races
    /// are ordinary here); `UpdateRefreshRate` does not, matching each variant's original
    /// handling before the two were merged into one port.
    fn send_vsync_control(&mut self, cmd: VsyncCommand) {
        match cmd {
            VsyncCommand::ReturnToken => {
                match self.vsync.send(Message::Control(VsyncCommand::ReturnToken)) {
                    Ok(()) | Err(SendError::Disconnected) => {}
                    Err(SendError::Timeout) => panic!("Timed out returning vsync token"),
                }
            }
            other => {
                self.vsync
                    .send(Message::Control(other))
                    .expect("failed to update refresh rate");
            }
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
            self.core.render_undeliverable();
            return;
        };
        if let Err(e) = rasterizer.send(Message::Data(request)) {
            log::warn!("Failed to send render request to rasterizer: {}", e);
            self.core.render_undeliverable();
        }
    }

    /// The shutdown cascade all three quit paths (`EngineControl::Quit`,
    /// `AppManagement::Quit`, `DisplayEvent::CloseRequested`) share: vsync, rasterizer,
    /// forwarder, drop the app handle, driver, then self. Any app notification for a
    /// `CloseRequested` has already gone out via the `app` port earlier in [`Self::flush`].
    fn shut_down(&mut self) {
        self.vsync
            .send(Message::Shutdown)
            .expect("Failed to shutdown vsync on Quit");
        if let Some(rasterizer) = &self.rasterizer {
            rasterizer
                .send(Message::Shutdown)
                .expect("Failed to shutdown rasterizer on Quit");
        }
        if let Some(forwarder) = &self.rasterizer_forwarder {
            forwarder
                .send(Message::Shutdown)
                .expect("Failed to shutdown rasterizer forwarder on Quit");
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

        // Step 3: Run the forwarder as a real actor rather than a bare thread — it is now
        // addressable (a real Shutdown on the Quit paths, not an implicit exit whenever the
        // rasterizer happens to drop its sender) and supervisable the same way the rest of the
        // troupe is. Its lanes are `Infallible`: nothing but `Shutdown` and the doorbell ever
        // reaches it, so `data_buffer_size` of 1 is a formality.
        let mut builder = ActorBuilder::new(1, None);
        let self_handle = builder.add_producer();
        let forwarder_handle = builder.add_producer();
        let mut forwarder_scheduler: ActorScheduler<Infallible, Infallible, Infallible> =
            builder.build();
        let mut forwarder = RasterizerForwarder {
            response_rx,
            engine: engine_handle,
            self_handle: Some(self_handle),
        };
        std::thread::Builder::new()
            .name("rasterizer-forwarder".into())
            .spawn(move || {
                forwarder_scheduler.run(&mut forwarder);
            })
            .expect("failed to spawn rasterizer forwarder thread");
        // The scheduler blocks on its doorbell until woken; with `Infallible` lanes there is no
        // message to send, so ring the doorbell directly to start the loop.
        forwarder_handle.waker().wake();

        // Step 4: Complete bootstrap - register response channel and get full handle
        let rasterizer_handle = setup_handle.register(response_tx);

        log::info!("Rasterizer actor initialized via bootstrap");
        self.rasterizer = Some(rasterizer_handle);
        self.rasterizer_forwarder = Some(forwarder_handle);
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
            rasterizer_forwarder: None,      // Set up separately via bootstrap
            app_handle: None,
            render_threads: 1, // Default, will be set by Configure message
            core: EngineCore::new(),
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
    use crate::api::public::{AppData, WindowId};
    use crate::display::messages::{DisplayEvent, Surface, Window};
    use crate::display::window_keeper::{Presented, WindowKeeper};
    use crate::platform::ColorCube;
    use actor_scheduler::ActorScheduler;
    use pixelflow_core::{Discrete, Manifold};
    use pixelflow_graphics::render::rasterizer::{RasterControl, RasterManagement};
    use pixelflow_graphics::render::Frame;
    use std::collections::VecDeque;
    use std::time::{Duration, Instant};

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
        fn handle_os(&mut self, _status: SystemStatus) -> Result<ActorStatus, HandlerError> {
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
        fn handle_os(&mut self, _status: SystemStatus) -> Result<ActorStatus, HandlerError> {
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
                rasterizer_forwarder: None,
                app_handle: None,
                render_threads: 1,
                core: EngineCore::new(),
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
            !rig.engine.core.holds_buffer(),
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
            !rig.engine.core.holds_buffer(),
            "a buffer is already out with the renderer; a second one must not be granted"
        );

        // The original render completes into an engine that still has exactly one buffer's
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
            !rig.engine.core.holds_buffer(),
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
            !rig.engine.core.holds_buffer(),
            "no grant can arrive while the buffer is out, however often it is requested"
        );

        rig.complete_render();
        assert_eq!(rig.blitted(), vec![(100, 100)]);
    }

    /// The forwarder's whole reason to exist: turn the rasterizer's bare `mpsc` responses into
    /// real `EngineData::RenderComplete` sends on an addressable actor, rather than a bare
    /// thread nobody could signal. Exercises the actual `RasterizerForwarder`, not a stand-in.
    #[test]
    fn forwarder_relays_responses_and_exits_when_the_rasterizer_disconnects() {
        use crate::display::messages::Generation;

        let (engine_handle, mut engine_sched) = ActorScheduler::<
            EngineData,
            EngineControl,
            AppManagement,
        >::new(LANE_BURST, LANE_BUFFER);
        let (response_tx, response_rx) = std::sync::mpsc::channel();

        let mut builder = ActorBuilder::new(1, None);
        let self_handle = builder.add_producer();
        let starter_handle = builder.add_producer();
        let mut forwarder_sched: ActorScheduler<Infallible, Infallible, Infallible> =
            builder.build();
        let mut forwarder = RasterizerForwarder {
            response_rx,
            engine: engine_handle,
            self_handle: Some(self_handle),
        };
        let thread = std::thread::spawn(move || forwarder_sched.run(&mut forwarder));
        starter_handle.waker().wake();

        let meta = WindowMeta {
            id: WindowId(1),
            width_px: 4,
            height_px: 4,
            scale: 1.0,
            generation: Generation::NONE,
        };
        response_tx
            .send(RenderResponse {
                frame: Frame::new(4, 4),
                render_time: Some(Duration::from_millis(1)),
                meta,
            })
            .expect("forwarder thread should still be listening");

        #[derive(Default)]
        struct TargetSpy {
            received: Vec<(u32, u32)>,
        }
        impl Actor<EngineData, EngineControl, AppManagement> for TargetSpy {
            fn handle_data(&mut self, data: EngineData) -> HandlerResult {
                if let EngineData::RenderComplete(response) = data {
                    self.received
                        .push((response.meta.width_px, response.meta.height_px));
                }
                Ok(())
            }
            fn handle_control(&mut self, _msg: EngineControl) -> HandlerResult {
                Ok(())
            }
            fn handle_management(&mut self, _msg: AppManagement) -> HandlerResult {
                Ok(())
            }
            fn handle_os(&mut self, _status: SystemStatus) -> Result<ActorStatus, HandlerError> {
                Ok(ActorStatus::Idle)
            }
        }

        let mut spy = TargetSpy::default();
        // The forwarder thread runs concurrently; poll until its send lands rather than
        // asserting after a single pass.
        for _ in 0..10_000 {
            let _ = engine_sched.poll_once(&mut spy);
            if !spy.received.is_empty() {
                break;
            }
            std::thread::yield_now();
        }
        assert_eq!(
            spy.received,
            vec![(4, 4)],
            "the forwarder must translate the rasterizer's response into RenderComplete"
        );

        // Dropping the sender is exactly what the real bootstrap path does when the rasterizer
        // shuts down; the forwarder must notice and exit rather than parking on an empty
        // doorbell forever.
        drop(response_tx);
        thread
            .join()
            .expect("forwarder must exit once the rasterizer disconnects");
    }
}
