//! VSync Actor - Separate thread that generates vsync timing signals.
//!
//! The VSync actor runs in its own thread and can be controlled via message passing.
//! It sends periodic vsync signals that the engine uses for frame timing.
//!
//! # clock thread
//! To avoid scheduling starvation, the VSync timing is driven by a dedicated
//! clock thread that sends explicit `Tick` messages to the actor. This ensures
//! the actor wakes up reliably regardless of other system load, without relying
//! on blocking `park` calls that could stall the actor scheduler.
//!
//! # `VsyncCore`: the decision logic as a `Transducer`
//!
//! Rollout step 2 of `docs/designs/pixelflow-runtime-engine-mesh-migration.md` §5. The actual
//! vsync decision logic — should this tick fire, has the refresh rate changed, is FPS due for
//! an update — lives in [`VsyncCore`], a pure [`actor_scheduler::mealy::Transducer`]: it takes
//! a message and returns what to emit, touching no threads, no channels, no `EngineActorHandle`.
//! `VsyncActor` (below) is now a thin adapter: it owns the clock thread and the engine handle,
//! and turns `VsyncCore`'s `Some(tick)` output into the actual `engine_handle.send(...)` call.
//!
//! This is deliberately *not yet* wired onto a real `Node`/`Host` — `VsyncActor` still runs on
//! its own dedicated OS thread via the old `Actor`/`ActorScheduler`, exactly as before, so
//! `EngineHandler` needs zero changes to its field types. Only the *internals* moved onto the
//! new primitive. See the design doc for why that split is deliberate, not a shortcut.
//!
//! # `Credit` replaces the global token bucket
//!
//! The former `VSYNC_TOKEN_BUCKET` was a process-wide `AtomicU32`, decremented by vsync itself
//! but *returned* by `engine_troupe.rs` calling a free function directly — reaching into
//! vsync's state without a message, the exact hazard the mesh migration exists to remove (see
//! the design doc's case study). `VsyncCore` now holds a private, non-atomic
//! [`actor_scheduler::mealy::Credit`], consumed in [`Transducer::step_management`] and released
//! only in [`Transducer::step_control`] on a new [`VsyncCommand::ReturnToken`] — a real message,
//! not a shared global. `engine_troupe.rs`'s two call sites become sends of that command.

use actor_scheduler::actors::{Schedule, Timer};
use actor_scheduler::mealy::{Credit, Transducer};
use actor_scheduler::{Actor, ActorBuilder, ActorHandle, HandlerError, HandlerResult, SystemStatus};
use log::info;
use std::ops::ControlFlow;
use std::sync::mpsc::Sender;
use std::thread;
use std::time::{Duration, Instant};

/// Default refresh rate in Hz.
const DEFAULT_REFRESH_RATE: f64 = 60.0;

/// Vsync tick request budget. Was `VSYNC_TOKEN_BUCKET`'s cap; now `Credit`'s `max`.
const MAX_TOKENS: u32 = 100;

/// Configuration for VsyncActor
#[derive(Debug, Clone)]
pub struct VsyncConfig {
    pub refresh_rate: f64,
}

impl Default for VsyncConfig {
    fn default() -> Self {
        Self {
            refresh_rate: DEFAULT_REFRESH_RATE,
        }
    }
}

/// Messages TO the VSync actor (commands) - Control lane
#[derive(Debug, Default)]
pub enum VsyncCommand {
    /// Start sending vsync signals
    Start,
    /// Stop sending vsync signals (pause)
    Stop,
    /// Update refresh rate (for VRR displays)
    UpdateRefreshRate(f64),
    /// Request current FPS stats
    RequestCurrentFPS(Sender<f64>),
    /// Return a previously-consumed vsync token, allowing another tick through.
    ///
    /// Replaces the old `return_vsync_token()` free function — the same event (app responded
    /// to a frame request, rasterization no longer bottlenecks the tick rate), now a real
    /// message instead of a same-process global mutation.
    ReturnToken,
    /// Shutdown the actor
    #[default]
    Shutdown,
}
actor_scheduler::impl_control_message!(VsyncCommand);

/// Response from engine after rendering a frame - Data lane
#[derive(Debug, Clone, Copy)]
pub struct RenderedResponse {
    /// Frame number that was rendered
    pub frame_number: u64,
    /// When the frame was rendered
    pub rendered_at: Instant,
}
actor_scheduler::impl_data_message!(RenderedResponse);

/// Management messages
#[derive(Debug)]
pub enum VsyncManagement {
    /// Internal clock tick - wakes the actor to check vsync timing
    Tick,
    /// Configure the vsync actor (set refresh rate, engine handle, etc.)
    SetConfig {
        config: VsyncConfig,
        engine_handle: Box<crate::api::private::EngineActorHandle>,
        self_handle: Box<ActorHandle<RenderedResponse, VsyncCommand, VsyncManagement>>,
    },
}
actor_scheduler::impl_management_message!(VsyncManagement);

// ────────────────────────────────────────────────────────────────────────────
// VsyncCore: the pure decision logic
// ────────────────────────────────────────────────────────────────────────────

/// What a tick decides to emit: the payload of `EngineData::VSync`, without depending on
/// `EngineData` itself — `VsyncCore` knows nothing about the engine's wire types, only that it
/// produces a timestamp triple.
#[derive(Debug, Clone, Copy)]
pub(crate) struct VsyncTick {
    pub(crate) timestamp: Instant,
    pub(crate) target_timestamp: Instant,
    pub(crate) refresh_interval: Duration,
}

/// Output word for [`VsyncCore`]: at most one tick per step. `Default` (no tick) is the common
/// case — most steps (Start/Stop/FPS tracking/etc.) emit nothing.
#[derive(Debug, Default)]
pub(crate) struct VsyncCoreOut {
    pub(crate) tick: Option<VsyncTick>,
}

/// Pure vsync decision core. No threads, no channels, no `EngineActorHandle` — a `step_*` call
/// takes a message and returns what to emit, so it is table-testable with no scheduler, no
/// clock thread, and no actor machinery in the loop.
pub(crate) struct VsyncCore {
    refresh_rate: f64,
    interval: Duration,
    running: bool,
    next_vsync: Instant,

    // FPS tracking (actual rasterization rate, not token rate)
    frame_count: u64,
    fps_start: Instant,
    last_fps: f64,

    credit: Credit,
}

impl VsyncCore {
    fn new(refresh_rate: f64) -> Self {
        Self {
            refresh_rate,
            interval: Duration::from_secs_f64(1.0 / refresh_rate),
            running: false,
            next_vsync: Instant::now(),
            frame_count: 0,
            fps_start: Instant::now(),
            last_fps: 0.0,
            credit: Credit::new(MAX_TOKENS),
        }
    }

    /// Current tick interval, for the adapter to hand to the clock thread.
    fn interval(&self) -> Duration {
        self.interval
    }

    /// Applied on `SetConfig`: set the rate and auto-start. Distinct from the plain
    /// `UpdateRefreshRate` command, which changes the rate without affecting `running` — this
    /// mirrors the original code's "auto-start after configuration" comment exactly.
    fn configure(&mut self, refresh_rate: f64) {
        self.refresh_rate = refresh_rate;
        self.interval = Duration::from_secs_f64(1.0 / refresh_rate);
        self.running = true;
        self.next_vsync = Instant::now();
    }

    fn set_refresh_rate(&mut self, new_rate: f64) {
        self.refresh_rate = new_rate;
        self.interval = Duration::from_secs_f64(1.0 / new_rate);
    }

    fn update_fps(&mut self) {
        let elapsed = self.fps_start.elapsed();
        if elapsed >= Duration::from_secs(1) {
            self.last_fps = self.frame_count as f64 / elapsed.as_secs_f64();
            self.frame_count = 0;
            self.fps_start = Instant::now();
        }
    }
}

impl Transducer for VsyncCore {
    type Control = VsyncCommand;
    type Management = VsyncManagement;
    type Data = RenderedResponse;
    type Out = VsyncCoreOut;

    fn step_data(&mut self, response: RenderedResponse) -> Result<VsyncCoreOut, HandlerError> {
        // Count actual rendered frames for accurate FPS measurement. Deliberately does not
        // touch `credit` — see the module doc: replenishment is `ReturnToken`, a separate,
        // explicit command, not implied by a frame having rendered.
        self.frame_count += 1;
        log::trace!("VsyncActor: Frame {} rendered", response.frame_number);
        self.update_fps();
        Ok(VsyncCoreOut::default())
    }

    fn step_control(&mut self, cmd: VsyncCommand) -> Result<VsyncCoreOut, HandlerError> {
        match cmd {
            VsyncCommand::Start => {
                self.running = true;
                self.next_vsync = Instant::now(); // Reset timing
                info!("VsyncActor: Started");
            }
            VsyncCommand::Stop => {
                self.running = false;
                info!("VsyncActor: Stopped");
            }
            VsyncCommand::UpdateRefreshRate(new_rate) => {
                self.set_refresh_rate(new_rate);
                info!(
                    "VsyncActor: Updated refresh rate to {:.2} Hz ({:.2}ms interval)",
                    self.refresh_rate,
                    self.interval.as_secs_f64() * 1000.0
                );
                // Pushing the new interval to the clock thread is the adapter's job — this
                // core has no clock thread to push it to.
            }
            VsyncCommand::RequestCurrentFPS(sender) => {
                info!("VsyncActor: FPS requested - {:.2} fps", self.last_fps);
                if let Err(e) = sender.send(self.last_fps) {
                    log::warn!("VsyncActor: Failed to send FPS response: {:?}", e);
                }
            }
            VsyncCommand::ReturnToken => {
                self.credit.release();
                log::trace!("VsyncActor: Token returned");
            }
            VsyncCommand::Shutdown => {
                info!("VsyncActor: Shutting down");
                // Stopping the clock thread is the adapter's job — this core owns no thread.
            }
        }
        Ok(VsyncCoreOut::default())
    }

    fn step_management(&mut self, msg: VsyncManagement) -> Result<VsyncCoreOut, HandlerError> {
        match msg {
            VsyncManagement::Tick => {
                if !self.running {
                    return Ok(VsyncCoreOut::default());
                }

                let now = Instant::now();
                if now < self.next_vsync {
                    return Ok(VsyncCoreOut::default());
                }

                if !self.credit.try_consume() {
                    log::trace!("VsyncActor: No tokens available, skipping vsync");
                    return Ok(VsyncCoreOut::default());
                }

                let timestamp = now;
                let target_timestamp = now + self.interval;
                self.next_vsync = timestamp + self.interval; // no cumulative drift
                Ok(VsyncCoreOut {
                    tick: Some(VsyncTick {
                        timestamp,
                        target_timestamp,
                        refresh_interval: self.interval,
                    }),
                })
            }
            VsyncManagement::SetConfig { .. } => {
                // Intercepted by the adapter before it ever reaches the core (see
                // `VsyncActor::handle_management`) — configuring engine_handle/self_handle and
                // spawning the clock thread are adapter concerns the core has no fields for.
                Ok(VsyncCoreOut::default())
            }
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// VsyncActor: the thin adapter — clock thread, engine handle, real sends
// ────────────────────────────────────────────────────────────────────────────

/// VSync actor - generates periodic vsync timing signals.
///
/// Owns exactly the things [`VsyncCore`] cannot: the engine handle, the clock thread. Every
/// `handle_*` call delegates the decision to `core`, then turns `Some(tick)` into the one real
/// side effect this whole actor performs — `engine_handle.send(...)`.
pub struct VsyncActor {
    engine_handle: Option<crate::api::private::EngineActorHandle>,
    clock: Option<Timer>,
    core: VsyncCore,
}

impl VsyncActor {
    /// Spawn the clock: a plain [`Timer`] whose callback is the one thing this actor's clock
    /// ever did — send itself a `Tick`. The hand-rolled `recv_timeout` loop this replaces was
    /// a timer with a vsync-shaped name on it; nothing about waiting on a duration was
    /// specific to vsync.
    fn spawn_clock(
        interval: Duration,
        self_handle: ActorHandle<RenderedResponse, VsyncCommand, VsyncManagement>,
    ) -> Timer {
        Timer::spawn("vsync-clock", Schedule::Every(interval), move || {
            match self_handle.send(VsyncManagement::Tick) {
                Ok(()) => ControlFlow::Continue(()),
                // The actor is gone; there is nobody left to tick.
                Err(_) => ControlFlow::Break(()),
            }
        })
    }

    /// Create empty VsyncActor for troupe pattern - configured via SetConfig management message.
    #[must_use]
    pub fn new_empty() -> Self {
        Self {
            engine_handle: None,
            clock: None,
            core: VsyncCore::new(DEFAULT_REFRESH_RATE),
        }
    }

    /// Create a new VsyncActor. Takes the handle to itself (for the clock thread).
    pub fn new(
        refresh_rate: f64,
        engine_handle: crate::api::private::EngineActorHandle,
        self_handle: ActorHandle<RenderedResponse, VsyncCommand, VsyncManagement>,
    ) -> Self {
        let interval = Duration::from_secs_f64(1.0 / refresh_rate);

        info!(
            "VsyncActor: Started with refresh rate {:.2} Hz ({:.2}ms interval), token bucket max: {}",
            refresh_rate,
            interval.as_secs_f64() * 1000.0,
            MAX_TOKENS
        );

        // Spawn the clock — self_handle is a dedicated SPSC channel
        let clock = Self::spawn_clock(interval, self_handle);

        Self {
            engine_handle: Some(engine_handle),
            clock: Some(clock),
            core: VsyncCore::new(refresh_rate),
        }
    }

    /// Spawn VSync actor in a new thread.
    ///
    /// Returns an ActorHandle that can be used to send commands and responses.
    pub fn spawn(
        refresh_rate: f64,
        engine_handle: crate::api::private::EngineActorHandle,
    ) -> ActorHandle<RenderedResponse, VsyncCommand, VsyncManagement> {
        let mut builder =
            ActorBuilder::<RenderedResponse, VsyncCommand, VsyncManagement>::new(1024, None);
        let handle = builder.add_producer(); // For the caller
        let clock_handle = builder.add_producer(); // For the clock thread (self-handle)
        let mut scheduler = builder.build();

        thread::Builder::new()
            .name("vsync-actor".to_string())
            .spawn(move || {
                let mut actor = VsyncActor::new(refresh_rate, engine_handle, clock_handle);
                scheduler.run(&mut actor);
            })
            .expect("Failed to spawn vsync actor thread");

        handle
    }

    /// Turn a `VsyncCore` output word into the one real send this actor makes.
    fn emit(&mut self, out: VsyncCoreOut) {
        let Some(tick) = out.tick else {
            return;
        };
        let Some(ref engine_handle) = self.engine_handle else {
            return; // Not configured yet
        };

        use crate::api::private::EngineData;
        use actor_scheduler::Message;

        if engine_handle
            .send(Message::Data(EngineData::VSync {
                timestamp: tick.timestamp,
                target_timestamp: tick.target_timestamp,
                refresh_interval: tick.refresh_interval,
            }))
            .is_err()
        {
            // Engine dropped the receiver
            info!("VsyncActor: Engine disconnected");
        }
    }
}

impl Actor<RenderedResponse, VsyncCommand, VsyncManagement> for VsyncActor {
    fn handle_data(&mut self, response: RenderedResponse) -> HandlerResult {
        let out = self.core.step_data(response)?;
        self.emit(out);
        Ok(())
    }

    fn handle_control(&mut self, cmd: VsyncCommand) -> HandlerResult {
        // UpdateRefreshRate and Shutdown need an adapter-level side effect (the clock thread)
        // in addition to the core's state change — decide that up front, then delegate once.
        let update_clock_interval = matches!(cmd, VsyncCommand::UpdateRefreshRate(_));
        let stop_clock = matches!(cmd, VsyncCommand::Shutdown);

        let out = self.core.step_control(cmd)?;

        if update_clock_interval {
            if let Some(ref clock) = self.clock {
                clock.set_interval(self.core.interval());
            }
        }
        if stop_clock {
            // `stop` consumes the Timer and joins its thread, so once this returns no further
            // Tick can land — the shutdown is complete, not merely requested.
            if let Some(clock) = self.clock.take() {
                clock.stop();
            }
        }

        self.emit(out);
        Ok(())
    }

    fn handle_management(&mut self, msg: VsyncManagement) -> HandlerResult {
        match msg {
            VsyncManagement::Tick => {
                let out = self.core.step_management(VsyncManagement::Tick)?;
                self.emit(out);
            }
            VsyncManagement::SetConfig {
                config,
                engine_handle,
                self_handle,
            } => {
                // Configure the vsync actor (called via Management after construction).
                // Engine handle + clock thread are adapter-only state; the core just gets the
                // refresh rate and auto-starts, mirroring the original "auto-start after
                // configuration" behavior (avoids a priority inversion where Start on Control
                // could otherwise run before SetConfig on Management).
                self.engine_handle = Some(*engine_handle);
                self.core.configure(config.refresh_rate);

                info!("VsyncActor: Configured with {:.2} Hz", config.refresh_rate);

                self.clock = Some(Self::spawn_clock(self.core.interval(), *self_handle));

                info!("VsyncActor: Auto-started after configuration");
            }
        }
        Ok(())
    }

    fn park(
        &mut self,
        _status: SystemStatus,
    ) -> Result<actor_scheduler::ActorStatus, HandlerError> {
        // VSync is driven by internal clock thread sending messages
        Ok(actor_scheduler::ActorStatus::Idle)
    }
}

// ActorTypes impl for VsyncActor
impl actor_scheduler::ActorTypes for VsyncActor {
    type Data = RenderedResponse;
    type Control = VsyncCommand;
    type Management = VsyncManagement;
}

// TroupeActor impl for VsyncActor — ignores directory, configured via SetConfig message
impl<Dir> actor_scheduler::TroupeActor<Dir> for VsyncActor {
    fn new(_dir: Dir) -> Self {
        Self::new_empty()
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Tests: VsyncCore in isolation — no threads, no clock, no scheduler
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_tick_before_start_emits_nothing() {
        let mut core = VsyncCore::new(60.0);
        let out = core.step_management(VsyncManagement::Tick).unwrap();
        assert!(out.tick.is_none(), "must not tick while stopped");
    }

    #[test]
    fn start_then_tick_emits_exactly_one_tick() {
        let mut core = VsyncCore::new(60.0);
        core.step_control(VsyncCommand::Start).unwrap();
        let out = core.step_management(VsyncManagement::Tick).unwrap();
        assert!(out.tick.is_some(), "a started core must tick when due");
    }

    #[test]
    fn stop_silences_further_ticks() {
        let mut core = VsyncCore::new(60.0);
        core.step_control(VsyncCommand::Start).unwrap();
        core.step_control(VsyncCommand::Stop).unwrap();
        let out = core.step_management(VsyncManagement::Tick).unwrap();
        assert!(out.tick.is_none(), "must not tick after Stop");
    }

    /// Forces `next_vsync` into the past immediately before a `Tick`, so the time gate is open
    /// regardless of clock resolution — a prior version of these tests instead used an absurdly
    /// high refresh rate (1GHz, a ~1ns interval) hoping any real work between calls would exceed
    /// it. That depends on the platform's clock reading finer than 1ns, which isn't guaranteed
    /// under a virtualized CI runner: back-to-back `Instant::now()` calls returning an identical
    /// value made the time gate spuriously block ticks that should only have been gated by
    /// credit (seen as flaky `left: 79/81, right: 100` failures on macOS CI). Overriding
    /// `next_vsync` directly has no such dependency — it isolates the credit gate exactly.
    fn force_tick_due(core: &mut VsyncCore) {
        core.next_vsync = Instant::now() - Duration::from_secs(1);
    }

    /// Force the gate open, then take one `Tick` — the two steps every test below needs.
    fn ticked(core: &mut VsyncCore) -> bool {
        force_tick_due(core);
        core.step_management(VsyncManagement::Tick)
            .unwrap()
            .tick
            .is_some()
    }

    #[test]
    fn credit_caps_ticks_at_max_tokens_with_no_clock_involved() {
        // The whole point of extracting VsyncCore: this needs no thread, no clock, no sleep —
        // just call step_management as many times as we like and count.
        let mut core = VsyncCore::new(60.0);
        core.step_control(VsyncCommand::Start).unwrap();
        let mut count = 0;
        for _ in 0..(MAX_TOKENS as usize * 2) {
            if ticked(&mut core) {
                count += 1;
            }
        }
        assert_eq!(
            count, MAX_TOKENS as usize,
            "must cap at exactly the credit bound, not the number of Tick calls"
        );
    }

    #[test]
    fn return_token_command_unblocks_exactly_one_more_tick() {
        let mut core = VsyncCore::new(60.0);
        core.step_control(VsyncCommand::Start).unwrap();

        let exhausted = (0..MAX_TOKENS).filter(|_| ticked(&mut core)).count();
        assert_eq!(exhausted, MAX_TOKENS as usize);

        // Starved: one more Tick call ticks nothing.
        assert!(
            !ticked(&mut core),
            "must be starved once the bound is spent"
        );

        // A real message, not a shared global — this is the fix.
        core.step_control(VsyncCommand::ReturnToken).unwrap();

        assert!(
            ticked(&mut core),
            "returning one token via a message must unblock exactly one more tick"
        );
    }

    #[test]
    fn rendered_response_does_not_return_a_token() {
        // Matches the documented, intentional behavior in vsync_actor_real_tests.rs: frame
        // completion and credit return are deliberately separate signals.
        let mut core = VsyncCore::new(60.0);
        core.step_control(VsyncCommand::Start).unwrap();

        let exhausted = (0..MAX_TOKENS).filter(|_| ticked(&mut core)).count();
        assert_eq!(
            exhausted, MAX_TOKENS as usize,
            "must actually exhaust the bound, or the check below proves nothing"
        );

        core.step_data(RenderedResponse {
            frame_number: 1,
            rendered_at: Instant::now(),
        })
        .unwrap();

        assert!(
            !ticked(&mut core),
            "RenderedResponse must not affect the credit bound — only ReturnToken does"
        );
    }
}
