//! Engine Troupe - Render pipeline actor coordination using troupe! macro.
//!
//! `EngineHandler` is the thin adapter around [`EngineCore`] (`engine_core.rs`): the core
//! decides, in the mealy-transducer style, what should happen and returns it as an
//! [`EngineOut`] word; `flush` is this file's hand-written `Wiring` — the one place that word
//! meets real channels.

use crate::api::private::{EngineControl, EngineData, GreenReadyBundle};
use crate::api::public::{AppManagement, Application};
use crate::config::EngineConfig;
use crate::coordinator_node::{CoordinatorCore, CoordinatorData, CoordinatorWiring};
use crate::display::driver::DriverActor;
use crate::display::messages::{DisplayControl, DisplayData, DisplayMgmt, WindowMeta};
use crate::display::platform::PlatformActor;
use crate::engine_core::{EngineCore, EngineOut};
use crate::error::RuntimeError;
use crate::platform::{ActivePlatform, PlatformPixel};
use crate::vsync_actor::{RenderedResponse, VsyncCommand, VsyncCore, VsyncManagement, VsyncWiring};
use actor_scheduler::actors::{Schedule, Timer};
use actor_scheduler::host::{GreenThread, Host, HostOut};
use actor_scheduler::mealy::{Flush, Lanes, NoLane, Node, Transducer, Wiring};
use actor_scheduler::{
    green_channel, Actor, ActorBuilder, ActorHandle, ActorScheduler, ActorStatus, ActorTypes,
    GreenSender, HandlerError, HandlerResult, Message, SchedulerParams, SystemStatus, TroupeActor,
    TrySendError,
};
use pixelflow_graphics::render::rasterizer::{RasterizerActor, RasterizerHandle, RenderResponse};
use std::convert::Infallible;
use std::ops::ControlFlow;
use std::sync::Arc;
use std::time::Duration;

/// Deliver a message on a credit-bounded green edge (design doc §3.2): the ring is provisioned
/// to never fill from that edge alone, so `Disconnected` is an ordinary shutdown race but `Full`
/// means the provisioning broke. Factored out because `vsync_data` and `vsync_control` make
/// exactly this argument for exactly this reason.
fn expect_credit_bounded_send<T: std::fmt::Debug>(
    result: Result<(), TrySendError<T>>,
    full_msg: &str,
) {
    match result {
        Ok(()) | Err(TrySendError::Disconnected(_)) => {}
        Err(err @ TrySendError::Full(_)) => panic!("{full_msg}: {err:?}"),
    }
}

/// Engine handler - coordinates app, rendering, display.
pub struct EngineHandler {
    /// Handle to the display driver actor.
    driver: ActorHandle<DisplayData, DisplayControl, DisplayMgmt>,
    /// The vsync green node's control edge (`UpdateRefreshRate`, `ReturnToken`, ...).
    vsync_control: Option<GreenSender<VsyncCommand>>,
    /// A handle to the green host itself, for the shutdown cascade — sending it
    /// `Message::Shutdown` stops both green nodes it hosts (vsync and the render coordinator)
    /// along with everything else the host owns.
    vsync_host: Option<ActorHandle<Infallible, Infallible, Infallible>>,
    /// The render coordinator's data lane (`coordinator_node.rs`): window grants, scene
    /// submissions, and advance nudges, relayed from the driver/app/vsync edges below. `None`
    /// until `EngineControl::GreenReady` arrives.
    coordinator: Option<GreenSender<CoordinatorData>>,
    /// Handle to self (for shutdown).
    self_handle: Option<ActorHandle<EngineData, EngineControl, AppManagement>>,
    /// Handle to the rasterizer response-forwarding actor (spawned in `with_config`, before the
    /// green host, via the free `spawn_rasterizer` function; kept only for the shutdown
    /// cascade — the rasterizer's own live handle belongs to the coordinator's wiring now, see
    /// `GreenReadyBundle`'s doc).
    rasterizer_forwarder: Option<ActorHandle<Infallible, Infallible, Infallible>>,
    /// Handle to the application (for event forwarding).
    app_handle: Option<Arc<dyn Application + Send + Sync>>,
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
//
// `driver` is `[expose]` as well as `[main]` now: the render coordinator's own green node
// (`coordinator_node.rs`, step 5c) needs a driver handle of its own, for `Present` and
// `RequestWindow` — sends the engine no longer makes on its behalf.
actor_scheduler::troupe! {
    driver: DriverActor<ActivePlatform> [main, expose],
    engine: EngineHandler [expose],
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
            EngineControl::GreenReady(bundle) => {
                let GreenReadyBundle {
                    vsync_control,
                    coordinator,
                    host,
                    forwarder,
                } = *bundle;
                self.vsync_control = Some(vsync_control);
                self.coordinator = Some(coordinator);
                self.vsync_host = Some(host);
                self.rasterizer_forwarder = Some(forwarder);
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
        // the same reason as `EngineControl::GreenReady` above.
        let mgmt = match mgmt {
            AppManagement::Configure(config) => {
                // The rasterizer is spawned by `with_config` itself now, before the green host —
                // it needs the coordinator's management-lane sender as its forwarder's target,
                // which does not exist until bootstrap builds it (`spawn_rasterizer`). Nothing
                // engine-side to do with this any more but log; kept as a message rather than
                // deleted outright in case a future engine-side setting wants to ride it.
                log::info!(
                    "Engine configured: {} render threads (rasterizer already spawned at bootstrap)",
                    config.performance.render_threads
                );
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

/// Bridges the rasterizer's response channel straight into the render coordinator's own green
/// node — no engine round-trip (step 5c of the mesh migration doc).
///
/// [`RasterizerActor`] hands completed frames back over a bare `mpsc::Sender` so
/// pixelflow-graphics stays decoupled from any particular consumer's message enum. This actor
/// is the one place that channel meets the coordinator's management lane — `handle_os` blocking
/// on `response_rx.recv()` *is* the actor, the same way `PtyReader::handle_os` blocking on
/// `epoll_wait` is that actor's entire job.
///
/// The scheduler only calls `handle_os` after the doorbell has woken at least once, so — same
/// as `PtyReader` waiting for its first `Bind` — this actor needs one doorbell ring to start its
/// loop; `spawn_rasterizer` rings it right after spawning the thread. Once `handle_os` returns
/// `Busy` the scheduler keeps re-entering it without waiting on the doorbell again, so from then
/// on the loop is self-sustaining.
struct RasterizerForwarder {
    response_rx: std::sync::mpsc::Receiver<RenderResponse<PlatformPixel, WindowMeta>>,
    coordinator: GreenSender<RenderResponse<PlatformPixel, WindowMeta>>,
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
            Ok(response) => match self.coordinator.try_send(response) {
                Ok(()) => Ok(ActorStatus::Busy),
                // This is the render credit's reply edge: at most one render is ever in flight,
                // so the ring can never fill from this edge alone — `Full` is a provisioning
                // bug, not backpressure to wait out.
                Err(TrySendError::Full(_)) => {
                    panic!("coordinator management ring unexpectedly full: the render-complete edge is credit(1)-bounded")
                }
                Err(TrySendError::Disconnected(_)) => {
                    log::warn!("Coordinator gone; rasterizer forwarder shutting down");
                    self.shut_down();
                    Ok(ActorStatus::Idle)
                }
            },
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
            self.driver
                .send(Message::Management(mgmt))
                .expect("Failed to relay management to driver");
        }

        if let Some(data) = out.coordinator {
            self.send_coordinator(data);
        }

        if let Some(cmd) = out.vsync_control {
            self.send_vsync_control(cmd);
        }

        if out.quit {
            self.shut_down();
        }
    }

    /// The `coordinator` port: every window grant, scene submission, and advance nudge the
    /// engine relays to the render coordinator's own green node.
    ///
    /// Submissions are bounded by the vsync token bucket (`MAX_TOKENS`, `vsync_actor.rs`) and
    /// grants by the one-buffer-in-flight bound the driver already enforces, so this edge can
    /// never see more outstanding sends than its ring holds — `Full` is therefore a provisioning
    /// bug, the same credit argument `expect_credit_bounded_send` makes for `vsync_control`.
    fn send_coordinator(&mut self, data: CoordinatorData) {
        let Some(tx) = &self.coordinator else {
            return; // Not configured yet, or the green host has already shut down.
        };
        expect_credit_bounded_send(tx.try_send(data), "coordinator ring unexpectedly full");
    }

    /// The `vsync_control` port — `ReturnToken` (a credit return: the ring holds `>= MAX_TOKENS`,
    /// design doc §3.2) and `UpdateRefreshRate` alike. Both tolerate a disconnected vsync
    /// (shutdown races are ordinary), and both treat `Full` as a bug rather than backpressure to
    /// wait out, since the ring was sized to never fill from this edge.
    fn send_vsync_control(&mut self, cmd: VsyncCommand) {
        let Some(tx) = &self.vsync_control else {
            return; // Not configured yet, or the green host has already shut down.
        };
        expect_credit_bounded_send(tx.try_send(cmd), "vsync control ring unexpectedly full");
    }

    /// The shutdown cascade all three quit paths (`EngineControl::Quit`,
    /// `AppManagement::Quit`, `DisplayEvent::CloseRequested`) share: the green host, the
    /// rasterizer forwarder, drop the app handle, driver, then self. Any app notification for a
    /// `CloseRequested` has already gone out via the `app` port earlier in [`Self::flush`].
    ///
    /// The rasterizer itself is *not* sent an explicit `Shutdown` here — it cannot be: its one
    /// live handle belongs to the coordinator's wiring (see `GreenReadyBundle`'s doc), which
    /// this cascade already reaches indirectly. Shutting down the green host drops `Host`, which
    /// drops the coordinator `Node` and its `CoordinatorWiring` along with it — including that
    /// sole rasterizer handle. The rasterizer's own scheduler sees every one of its lanes
    /// disconnect and halts gracefully (the same "all disconnected ⇒ normal shutdown" path the
    /// forwarder below already relies on one hop downstream), so the cascade still reaches it,
    /// just by the handle dropping rather than a message arriving.
    fn shut_down(&mut self) {
        if let Some(host) = &self.vsync_host {
            host.send(Message::Shutdown)
                .expect("Failed to shutdown green host on Quit");
        }
        // The host shutting down drops its lanes' receivers; drop our ends too so nothing
        // downstream mistakes a dead node for one still configured.
        self.vsync_control = None;
        self.coordinator = None;
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
}

/// Spawn the rasterizer actor with its bootstrap handshake, and the forwarder that turns its
/// bare `mpsc` responses into direct sends on the coordinator's management lane.
///
/// A free function, not a method on `EngineHandler`: nothing here needs the engine any more once
/// the forwarder's target is the coordinator instead of it (step 5c of the mesh migration doc).
/// Called once from `Troupe::with_config`, before the green-host thread spawns — it needs only
/// `render_threads` from config and the coordinator's already-built management-lane sender.
///
/// Returns the rasterizer's own handle (which goes straight into `CoordinatorWiring`, never to
/// the engine — see `GreenReadyBundle`'s doc for why it can't be shared) and the forwarder's
/// handle (which the engine keeps for its shutdown cascade).
fn spawn_rasterizer(
    render_threads: usize,
    coordinator: GreenSender<RenderResponse<PlatformPixel, WindowMeta>>,
) -> (
    RasterizerHandle<PlatformPixel, WindowMeta>,
    ActorHandle<Infallible, Infallible, Infallible>,
) {
    // Step 1: Spawn rasterizer with setup handle
    let (setup_handle, _rasterizer_thread) =
        RasterizerActor::<PlatformPixel, WindowMeta>::spawn_with_setup(render_threads);

    // Step 2: Create response channel (the forwarder receives render results here)
    let (response_tx, response_rx) =
        std::sync::mpsc::channel::<RenderResponse<PlatformPixel, WindowMeta>>();

    // Step 3: Run the forwarder as a real actor rather than a bare thread — it is addressable (a
    // real Shutdown on the Quit paths, not an implicit exit whenever the rasterizer happens to
    // drop its sender) and supervisable the same way the rest of the troupe is. Its lanes are
    // `Infallible`: nothing but `Shutdown` and the doorbell ever reaches it, so
    // `data_buffer_size` of 1 is a formality.
    let mut builder = ActorBuilder::new(1, None);
    let self_handle = builder.add_producer();
    let forwarder_handle = builder.add_producer();
    let mut forwarder_scheduler: ActorScheduler<Infallible, Infallible, Infallible> =
        builder.build();
    let mut forwarder = RasterizerForwarder {
        response_rx,
        coordinator,
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
    (rasterizer_handle, forwarder_handle)
}

// Implement TroupeActor for EngineHandler — takes ownership of per-actor Directory
impl TroupeActor<Directory> for EngineHandler {
    fn new(dir: Directory) -> Self {
        Self {
            driver: dir.driver,
            vsync_control: None, // Set via EngineControl::GreenReady once the green host is up
            vsync_host: None,
            coordinator: None, // Set via EngineControl::GreenReady once the green host is up
            self_handle: Some(dir.engine),
            rasterizer_forwarder: None, // Set via EngineControl::GreenReady
            app_handle: None,
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

/// Delivers a [`Host`]'s supervision findings to the engine — the wired half of "Host as
/// transducer" (design doc §5.4): a real port with a real consumer, where `handle_os` could only
/// compute the event and drop it.
///
/// `HostOut::again` is the host's own continuation; it never reaches wiring at all — see the
/// `debug_assert!` in [`Wiring::flush`](Wiring::flush), which is what makes that a checkable fact
/// rather than a comment.
struct EngineHostWiring {
    engine: crate::api::private::EngineActorHandle,
}

impl Wiring for EngineHostWiring {
    type Out = HostOut;

    fn flush(&mut self, out: &mut HostOut) -> Flush {
        debug_assert!(
            out.again.is_none(),
            "GreenThread drains its own continuation; wiring never sees it"
        );

        let Some(event) = out.supervision else {
            return Flush::Done;
        };

        match self
            .engine
            .try_send(Message::Control(EngineControl::GreenStuck(event)))
        {
            Ok(()) => {
                out.supervision = None;
                Flush::Done
            }
            // Neither clears `out.supervision` — it is already `Some`, so leaving it be is
            // putting it back, exactly like `VsyncWiring`'s tick port.
            Err(TrySendError::Full(_)) => Flush::Blocked,
            Err(TrySendError::Disconnected(_)) => Flush::Disconnected,
        }
    }
}

impl Troupe {
    /// Create troupe, configure the engine, and bring vsync *and the render coordinator* up as
    /// green nodes (docs/designs/actor-scheduler-mealy-transducer.md §5, and step 5c of
    /// `pixelflow-runtime-engine-mesh-migration.md` §8): one "green-host" thread runs a [`Host`]
    /// hosting both [`Node`]s, stepped by its own [`GreenThread`] rather than either living on a
    /// dedicated OS thread of its own. The one thread this bootstrap still spawns for vsync is
    /// the [`Timer`] clock — a green actor may not block, and a clock has to wait.
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

        // Create SPSC handles for initialization (each exposed() creates unique channels).
        // `init.driver` is the coordinator's own driver handle — grabbed here rather than with a
        // separate `exposed()` call, since one call already mints every exposed actor's handle.
        let init = troupe.exposed(); // engine: config messages; driver: the coordinator's handle
        let vsync_engine = troupe.exposed(); // dedicated engine producer for vsync ticks
        let host_engine = troupe.exposed(); // dedicated engine producer for host supervision

        // Configure the engine with window settings
        init.engine
            .send(Message::Management(AppManagement::Configure(
                config.clone(),
            )))
            .map_err(|e| RuntimeError::InitError(format!("Failed to configure engine: {}", e)))?;

        // ── The green host: vsync AND the render coordinator ────────────────────────────
        //
        // A host has no lanes of its own (all three types are `Infallible`); work reaches it by
        // pushing straight into an adopted node's inbox with a `GreenSender` and ringing the
        // host's doorbell. `host_shutdown` is kept (handed to the engine below, for the shutdown
        // cascade); `host_kick` only lends its `Waker` to the green channels and gives the
        // scheduler its initial kick once the thread is running.
        let mut host_builder = ActorBuilder::<Infallible, Infallible, Infallible>::new(1, None);
        let host_shutdown = host_builder.add_producer();
        let host_kick = host_builder.add_producer();
        let mut host_sched = host_builder.build();

        let waker = host_kick.waker();
        // Capacities >= MAX_TOKENS (100, `vsync_actor.rs`): the credit argument (design doc
        // §3.2) needs the ring to hold every possible outstanding tick/`ReturnToken`, so `Full`
        // on either is provably a bug rather than backpressure to tolerate.
        let (vsync_data_tx, vsync_data_rx) = green_channel::<RenderedResponse>(128, waker.clone());
        let (vsync_control_tx, vsync_control_rx) =
            green_channel::<VsyncCommand>(128, waker.clone());
        // Capacity 1 is enough: a queued tick is as good as a fresh one, and the clock is the
        // only producer.
        let (tick_tx, tick_rx) = green_channel::<VsyncManagement>(2, waker.clone());

        // The coordinator's two lanes (§5c): data, fed by the engine shell
        // (Submit/Granted/Advance) — 256 covers every vsync-token-bounded frame request plus
        // every buffer grant that could be outstanding; management, fed directly by the
        // rasterizer forwarder (`RenderResponse`) — bounded by the render credit (<=1 in
        // flight), so its capacity is a formality, same as the forwarder's own ring.
        let (coordinator_data_tx, coordinator_data_rx) =
            green_channel::<CoordinatorData>(256, waker.clone());
        let (coordinator_mgmt_tx, coordinator_mgmt_rx) =
            green_channel::<RenderResponse<PlatformPixel, WindowMeta>>(8, waker);

        // The rasterizer is spawned here now, not by the engine (§8 of the mesh migration doc):
        // it needs only `render_threads` from config and the coordinator's management-lane
        // sender as the forwarder's target, so the `SetRasterizerForwardHandle` round trip that
        // used to gate `Configure` is gone entirely.
        let (rasterizer, forwarder) =
            spawn_rasterizer(config.performance.render_threads, coordinator_mgmt_tx);

        init.engine
            .send(Message::Control(EngineControl::GreenReady(Box::new(
                GreenReadyBundle {
                    vsync_control: vsync_control_tx,
                    coordinator: coordinator_data_tx,
                    host: host_shutdown,
                    forwarder,
                },
            ))))
            .map_err(|e| {
                RuntimeError::InitError(format!("Failed to hand green handles to engine: {}", e))
            })?;

        let refresh_rate = config.performance.target_fps as f64;
        let tick_interval = Duration::from_secs_f64(1.0 / refresh_rate);

        // Built here, on the calling thread, then moved into the green-host thread below —
        // `Timer` is `Send`, so this doesn't need to happen on the thread that owns it.
        let clock = Timer::spawn("vsync-clock", Schedule::Every(tick_interval), move || {
            match tick_tx.try_send(VsyncManagement::Tick) {
                Ok(()) => ControlFlow::Continue(()),
                // A queued tick is as good as this one.
                Err(TrySendError::Full(_)) => ControlFlow::Continue(()),
                Err(TrySendError::Disconnected(_)) => ControlFlow::Break(()),
            }
        });

        let coordinator_driver = init.driver;

        std::thread::Builder::new()
            .name("green-host".to_string())
            .spawn(move || {
                let mut host = Host::new();

                let vsync_core = VsyncCore::started(refresh_rate);
                let vsync_wiring = VsyncWiring {
                    engine: vsync_engine.engine,
                    clock,
                };
                let vsync_node = Node::new_with_lanes(
                    vsync_core,
                    Lanes {
                        control: vsync_control_rx,
                        management: tick_rx,
                        data: vsync_data_rx,
                    },
                    vsync_wiring,
                    SchedulerParams::DEFAULT,
                );
                host.adopt(vsync_node);

                // Adopted after vsync: sweep order between the two is not load-bearing here —
                // vsync's tick still relays through the engine rather than the coordinator, so
                // neither node's completion in one sweep depends on the other having already
                // run in that same sweep.
                let coordinator_core = CoordinatorCore::new();
                let coordinator_wiring = CoordinatorWiring {
                    rasterizer,
                    driver: coordinator_driver,
                    vsync: vsync_data_tx,
                };
                let coordinator_node = Node::new_with_lanes(
                    coordinator_core,
                    Lanes {
                        control: NoLane::new(),
                        management: coordinator_mgmt_rx,
                        data: coordinator_data_rx,
                    },
                    coordinator_wiring,
                    SchedulerParams::DEFAULT,
                );
                host.adopt(coordinator_node);

                let mut green_thread = GreenThread::new(
                    host,
                    EngineHostWiring {
                        engine: host_engine.engine,
                    },
                );
                host_sched.run(&mut green_thread);
            })
            .expect("failed to spawn green host thread");
        // The scheduler blocks on its doorbell until woken; ring it once to start the sweep
        // loop, the same kick `spawn_rasterizer`'s `RasterizerForwarder` gives itself.
        host_kick.waker().wake();

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
    //! `EngineHandler`, driven directly.
    //!
    //! The render protocol itself — request/render/present, resize races, `holds_buffer` — no
    //! longer runs through the engine at all, so those interaction tests moved to
    //! `coordinator_node.rs`'s `node_tests` (see that module's doc for the full list and why
    //! each one still exists). What is left here is what the engine still actually does: relay
    //! driver/app/vsync events to the render coordinator's `CoordinatorData` port, plus its own
    //! shutdown cascade.

    use super::*;
    use crate::api::public::{AppData, WindowId};
    use crate::coordinator_node::CoordinatorData;
    use crate::display::messages::{DisplayEvent, Surface};
    use crate::platform::ColorCube;
    use actor_scheduler::spsc::{spsc_channel, SpscReceiver};
    use actor_scheduler::ActorScheduler;
    use pixelflow_core::{Discrete, Manifold};
    use std::time::Instant;

    /// Scheduler tuning for the fixture: small, since nothing here approaches a real ring.
    const LANE_BURST: usize = 16;
    const LANE_BUFFER: usize = 16;

    /// A constant black scene — these tests care about which port fires, not what is drawn.
    fn manifold() -> Arc<dyn Manifold<Output = Discrete> + Send + Sync> {
        Arc::new(ColorCube::default().at(0.0f32, 0.0f32, 0.0f32, 1.0f32))
    }

    /// `EngineHandler` wired to a real driver scheduler (so `driver_control`/`driver_mgmt`
    /// relays are observable) and a held `coordinator` receiver (so `CoordinatorData` arrivals
    /// are observable), with no green host, thread, or rasterizer in the loop.
    struct Rig {
        engine: EngineHandler,
        coordinator_rx: SpscReceiver<CoordinatorData>,
    }

    impl Rig {
        fn new() -> Self {
            let (driver, _driver_sched) = ActorScheduler::new(LANE_BURST, LANE_BUFFER);

            let waker = {
                let (handle, _sched) =
                    ActorScheduler::<Infallible, Infallible, Infallible>::new(1, 1);
                handle.waker()
            };
            let (coordinator_tx, coordinator_rx) = spsc_channel::<CoordinatorData>(64);
            let coordinator = GreenSender::new(coordinator_tx, waker);

            let engine = EngineHandler {
                driver,
                vsync_control: None,
                vsync_host: None,
                coordinator: Some(coordinator),
                self_handle: None,
                rasterizer_forwarder: None,
                app_handle: None,
                core: EngineCore::new(),
            };

            Self {
                engine,
                coordinator_rx,
            }
        }

        fn feed(&mut self, data: EngineData) {
            self.engine
                .handle_data(data)
                .expect("engine handled the message");
        }
    }

    #[test]
    fn render_surface_relays_a_submit_to_the_coordinator() {
        let mut rig = Rig::new();
        rig.feed(EngineData::FromApp(AppData::RenderSurface(manifold())));
        assert!(matches!(
            rig.coordinator_rx.try_recv(),
            Ok(CoordinatorData::Submit(_))
        ));
    }

    #[test]
    fn skipped_frame_does_not_touch_the_coordinator() {
        let mut rig = Rig::new();
        rig.feed(EngineData::FromApp(AppData::Skipped));
        assert!(rig.coordinator_rx.try_recv().is_err());
    }

    #[test]
    fn window_granted_relays_to_the_coordinator() {
        use crate::display::messages::{Generation, Window, WindowMeta};

        let mut rig = Rig::new();
        let window = Window::rejoin(
            pixelflow_graphics::render::Frame::new(100, 100),
            WindowMeta {
                id: WindowId(1),
                width_px: 100,
                height_px: 100,
                scale: 1.0,
                generation: Generation::NONE,
            },
        );
        rig.feed(EngineData::WindowGranted(window));
        assert!(matches!(
            rig.coordinator_rx.try_recv(),
            Ok(CoordinatorData::Granted(_))
        ));
    }

    #[test]
    fn vsync_tick_relays_advance_to_the_coordinator() {
        let mut rig = Rig::new();
        let now = Instant::now();
        rig.feed(EngineData::VSync {
            timestamp: now,
            target_timestamp: now,
            refresh_interval: Duration::from_millis(16),
        });
        assert!(matches!(
            rig.coordinator_rx.try_recv(),
            Ok(CoordinatorData::Advance)
        ));
    }

    #[test]
    fn window_created_and_resized_relay_advance_to_the_coordinator() {
        let mut rig = Rig::new();
        let surface = Surface {
            id: WindowId(1),
            width_px: 100,
            height_px: 100,
            frame_width: 100,
            frame_height: 100,
            scale: 1.0,
        };

        rig.feed(EngineData::FromDriver(DisplayEvent::WindowCreated {
            surface,
        }));
        assert!(matches!(
            rig.coordinator_rx.try_recv(),
            Ok(CoordinatorData::Advance)
        ));

        rig.feed(EngineData::FromDriver(DisplayEvent::Resized { surface }));
        assert!(matches!(
            rig.coordinator_rx.try_recv(),
            Ok(CoordinatorData::Advance)
        ));
    }

    /// The forwarder's whole reason to exist: turn the rasterizer's bare `mpsc` responses into
    /// direct sends on the coordinator's management lane, rather than a bare thread nobody could
    /// signal. Exercises the actual `RasterizerForwarder`, not a stand-in — simpler now than the
    /// engine-routed version, since a `GreenSender`'s receiving end is a plain `SpscReceiver`
    /// and no engine scheduler is needed to observe arrival.
    #[test]
    fn forwarder_relays_responses_and_exits_when_the_rasterizer_disconnects() {
        use crate::display::messages::Generation;

        let waker = {
            let (handle, _sched) = ActorScheduler::<Infallible, Infallible, Infallible>::new(1, 1);
            handle.waker()
        };
        let (coordinator_tx, mut coordinator_rx) =
            spsc_channel::<RenderResponse<PlatformPixel, WindowMeta>>(8);
        let coordinator = GreenSender::new(coordinator_tx, waker);

        let (response_tx, response_rx) = std::sync::mpsc::channel();

        let mut builder = ActorBuilder::new(1, None);
        let self_handle = builder.add_producer();
        let starter_handle = builder.add_producer();
        let mut forwarder_sched: ActorScheduler<Infallible, Infallible, Infallible> =
            builder.build();
        let mut forwarder = RasterizerForwarder {
            response_rx,
            coordinator,
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
                frame: pixelflow_graphics::render::Frame::new(4, 4),
                render_time: Some(Duration::from_millis(1)),
                meta,
            })
            .expect("forwarder thread should still be listening");

        // The forwarder thread runs concurrently; poll until its send lands rather than
        // asserting after a single pass.
        let mut received = None;
        for _ in 0..10_000 {
            if let Ok(response) = coordinator_rx.try_recv() {
                received = Some((response.meta.width_px, response.meta.height_px));
                break;
            }
            std::thread::yield_now();
        }
        assert_eq!(
            received,
            Some((4, 4)),
            "the forwarder must translate the rasterizer's response into a direct coordinator send"
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
