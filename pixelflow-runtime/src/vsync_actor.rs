//! VSync - a green [`actor_scheduler::mealy::Node`] hosted on its own `Host`/`GreenThread`.
//!
//! There is no dedicated OS thread for vsync itself: the node is stepped by the `Host` its
//! `GreenThread` sweeps, woken by [`actor_scheduler::GreenSender`] pushes into its lanes. The
//! one real thread this module still owns is the [`Timer`] clock (`clock thread`, below) —
//! ticks have to arrive on a schedule, and a green node cannot wait on the OS itself (design
//! doc §5).
//!
//! # clock thread
//! To avoid scheduling starvation, the VSync timing is driven by a dedicated
//! clock thread that sends explicit `Tick` messages to the node. This ensures
//! ticks arrive reliably regardless of other system load, without relying
//! on blocking `handle_os` calls that could stall a scheduler.
//!
//! # Decision logic
//!
//! [`VsyncCore`] implements the pure [`actor_scheduler::mealy::Transducer`] that decides whether
//! a tick should fire and when FPS updates are due. It does not interact with threads, channels,
//! or the engine handle. [`VsyncWiring`] (in `engine_troupe.rs`'s bootstrap) owns those runtime
//! resources and forwards the core's output to the engine and the clock.
//!
//! # Backpressure
//!
//! [`VsyncCore`] owns a [`actor_scheduler::mealy::Credit`] budget. Management ticks consume
//! credit, and [`VsyncCommand::ReturnToken`] replenishes it after the application responds to a
//! frame request.

use actor_scheduler::actors::Timer;
use actor_scheduler::mealy::{Credit, Transducer};
use actor_scheduler::{HandlerError, Message};
use log::info;
use std::sync::mpsc::Sender;
use std::time::{Duration, Instant};

/// Maximum number of outstanding VSync tick requests.
const MAX_TOKENS: u32 = 100;

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
    /// Return a previously consumed VSync token, allowing another tick through.
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
    /// Internal clock tick - wakes the node to check vsync timing
    Tick,
}
actor_scheduler::impl_management_message!(VsyncManagement);

// ────────────────────────────────────────────────────────────────────────────
// VsyncCore: the pure decision logic
// ────────────────────────────────────────────────────────────────────────────

type EngineMsg = Message<
    crate::api::private::EngineData,
    crate::api::private::EngineControl,
    crate::api::public::AppManagement,
>;

actor_scheduler::ports! {
    VsyncCore {
        tick: EngineMsg -> crate::api::private::EngineActorHandle,
        interval: Duration -> Timer,
    }
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

    /// New and already running. `SetConfig` is gone — the bootstrap that used to send it
    /// (`engine_troupe.rs`'s `with_config`) now builds a `VsyncCore` once, at construction, and
    /// this is the only entry point that does: there is no later message that (re-)configures
    /// one. Named for what it hands back, not what it does internally — "new+configure" is the
    /// implementation, not the contract.
    pub(crate) fn started(refresh_rate: f64) -> Self {
        let mut core = Self::new(refresh_rate);
        core.configure(refresh_rate);
        core
    }

    /// Set the rate and auto-start. Distinct from the plain `UpdateRefreshRate` command, which
    /// changes the rate without affecting `running`.
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
        log::trace!("vsync: Frame {} rendered", response.frame_number);
        self.update_fps();
        Ok(VsyncCoreOut::default())
    }

    fn step_control(&mut self, cmd: VsyncCommand) -> Result<VsyncCoreOut, HandlerError> {
        match cmd {
            VsyncCommand::Start => {
                self.running = true;
                self.next_vsync = Instant::now(); // Reset timing
                info!("vsync: Started");
            }
            VsyncCommand::Stop => {
                self.running = false;
                info!("vsync: Stopped");
            }
            VsyncCommand::UpdateRefreshRate(new_rate) => {
                self.set_refresh_rate(new_rate);
                info!(
                    "vsync: Updated refresh rate to {:.2} Hz ({:.2}ms interval)",
                    self.refresh_rate,
                    self.interval.as_secs_f64() * 1000.0
                );
                // The wiring relays this to the clock thread — the core has no clock of its own.
                return Ok(VsyncCoreOut {
                    interval: Some(self.interval),
                    ..VsyncCoreOut::default()
                });
            }
            VsyncCommand::RequestCurrentFPS(sender) => {
                info!("vsync: FPS requested - {:.2} fps", self.last_fps);
                if let Err(e) = sender.send(self.last_fps) {
                    log::warn!("vsync: Failed to send FPS response: {:?}", e);
                }
            }
            VsyncCommand::ReturnToken => {
                self.credit.release();
                log::trace!("vsync: Token returned");
            }
            VsyncCommand::Shutdown => {
                info!("vsync: Shutting down");
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
                    log::trace!("vsync: No tokens available, skipping vsync");
                    return Ok(VsyncCoreOut::default());
                }

                let timestamp = now;
                let target_timestamp = now + self.interval;
                self.next_vsync = timestamp + self.interval; // no cumulative drift
                Ok(VsyncCoreOut {
                    tick: Some(Message::Data(crate::api::private::EngineData::VSync {
                        timestamp,
                        target_timestamp,
                        refresh_interval: self.interval,
                    })),
                    ..VsyncCoreOut::default()
                })
            }
        }
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
        // Frame completion and credit return are deliberately separate signals — only
        // `VsyncCommand::ReturnToken` moves credit. (The now-deleted
        // `tests/vsync_actor_tests.rs` mocked the opposite, which is part of why it went.)
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

    #[test]
    fn update_refresh_rate_emits_the_new_interval_on_its_own_port() {
        let mut core = VsyncCore::new(60.0);
        let out = core
            .step_control(VsyncCommand::UpdateRefreshRate(120.0))
            .unwrap();
        assert_eq!(
            out.interval,
            Some(Duration::from_secs_f64(1.0 / 120.0)),
            "the wiring relays this to the clock; the core just reports the new interval"
        );
        assert!(out.tick.is_none(), "a rate change is not itself a tick");
    }

    #[test]
    fn other_control_messages_leave_the_interval_port_unset() {
        let mut core = VsyncCore::new(60.0);
        let out = core.step_control(VsyncCommand::Start).unwrap();
        assert!(
            out.interval.is_none(),
            "only UpdateRefreshRate should touch the interval port"
        );
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Tests: VsyncCoreWiring — the one place a real send happens
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod wiring_tests {
    use super::*;
    use crate::api::private::{EngineControl, EngineData};
    use crate::api::public::AppManagement;
    use actor_scheduler::actors::Schedule;
    use actor_scheduler::mealy::Wiring;
    use actor_scheduler::mealy::{send_port, Flush};
    use actor_scheduler::{Actor, ActorScheduler, ActorStatus, HandlerResult, SystemStatus};
    use std::ops::ControlFlow;

    /// A clock that never fires within a test's lifetime — these tests only care whether
    /// `set_interval` lands, not what the timer does with it.
    fn quiet_clock() -> Timer {
        Timer::spawn(
            "test-vsync-wiring-clock",
            Schedule::Every(Duration::from_secs(3600)),
            || ControlFlow::Continue(()),
        )
    }

    fn one_tick_msg() -> EngineMsg {
        let now = Instant::now();
        Message::Data(EngineData::VSync {
            timestamp: now,
            target_timestamp: now,
            refresh_interval: Duration::from_millis(16),
        })
    }

    /// Stop the borrowed clock so the test does not leak a timer thread.
    fn stop(wiring: VsyncCoreWiring) {
        let VsyncCoreWiring {
            interval: clock, ..
        } = wiring;
        clock.stop();
    }

    #[test]
    fn a_tick_reaches_the_engine_as_vsync_data() {
        let (engine, mut engine_sched) =
            ActorScheduler::<EngineData, EngineControl, AppManagement>::new(4, 8);
        let mut wiring = VsyncCoreWiring {
            tick: engine,
            interval: quiet_clock(),
        };

        let mut out = VsyncCoreOut {
            tick: Some(one_tick_msg()),
            interval: None,
        };
        assert_eq!(wiring.flush(&mut out), Flush::Done);
        assert!(
            out.tick.is_none(),
            "a delivered tick is cleared from the port"
        );

        struct Collector(bool);
        impl Actor<EngineData, EngineControl, AppManagement> for Collector {
            fn handle_data(&mut self, data: EngineData) -> HandlerResult {
                self.0 = matches!(data, EngineData::VSync { .. });
                Ok(())
            }
            fn handle_control(&mut self, _: EngineControl) -> HandlerResult {
                Ok(())
            }
            fn handle_management(&mut self, _: AppManagement) -> HandlerResult {
                Ok(())
            }
            fn handle_os(&mut self, _: SystemStatus) -> Result<ActorStatus, HandlerError> {
                Ok(ActorStatus::Idle)
            }
        }
        let mut collector = Collector(false);
        engine_sched.poll_once(&mut collector);
        assert!(collector.0, "the tick must arrive as EngineData::VSync");

        stop(wiring);
    }

    #[test]
    fn a_full_engine_inbox_parks_and_retains_the_tick() {
        let (engine, _engine_sched) =
            ActorScheduler::<EngineData, EngineControl, AppManagement>::new(4, 2);
        // Fill the ring without a consumer draining it, so the next send is genuinely full
        // rather than disconnected — via `send_port`, the one public non-blocking path to an
        // `ActorHandle`.
        let tick = || {
            Some(Message::Data(EngineData::VSync {
                timestamp: Instant::now(),
                target_timestamp: Instant::now(),
                refresh_interval: Duration::from_millis(16),
            }))
        };
        let mut port = tick();
        while send_port(&mut port, &engine) == Flush::Done {
            port = tick();
        }

        let mut wiring = VsyncCoreWiring {
            tick: engine,
            interval: quiet_clock(),
        };
        let mut out = VsyncCoreOut {
            tick: Some(one_tick_msg()),
            interval: None,
        };
        assert_eq!(wiring.flush(&mut out), Flush::Blocked);
        assert!(out.tick.is_some(), "a parked tick must stay in the outbox");

        stop(wiring);
    }

    #[test]
    fn a_gone_engine_reports_disconnected_and_retains_the_tick() {
        let (engine, engine_sched) =
            ActorScheduler::<EngineData, EngineControl, AppManagement>::new(4, 8);
        drop(engine_sched);

        let mut wiring = VsyncCoreWiring {
            tick: engine,
            interval: quiet_clock(),
        };
        let mut out = VsyncCoreOut {
            tick: Some(one_tick_msg()),
            interval: None,
        };
        assert_eq!(wiring.flush(&mut out), Flush::Disconnected);
        assert!(out.tick.is_some(), "the payload survives a dead peer");

        stop(wiring);
    }

    #[test]
    fn an_interval_change_reaches_the_clock_and_is_always_cleared() {
        let (engine, _engine_sched) =
            ActorScheduler::<EngineData, EngineControl, AppManagement>::new(4, 8);
        let mut wiring = VsyncCoreWiring {
            tick: engine,
            interval: quiet_clock(),
        };

        let mut out = VsyncCoreOut {
            tick: None,
            interval: Some(Duration::from_millis(8)),
        };
        assert_eq!(wiring.flush(&mut out), Flush::Done);
        assert!(
            out.interval.is_none(),
            "the interval port is always drained"
        );

        stop(wiring);
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Integration: a real clock, a real Node, a real EngineActorHandle
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod integration_tests {
    //! Replaces the deleted `tests/vsync_actor_real_tests.rs`, which proved a real `VsyncActor`
    //! reached the same core `VsyncCore`'s own unit tests exercise deterministically. That actor
    //! is gone, but the same real-clock, real-engine-handle guarantee still needs proving against
    //! the green replacement — and it has to live here, not in `tests/`, because `VsyncCore` and
    //! `VsyncWiring` are `pub(crate)` (minimal public API): nothing outside this crate can build
    //! the pieces this test wires together.
    //!
    //! What this deliberately does not re-prove: `Host`/`GreenThread` sweep semantics
    //! (`actor-scheduler`'s own `host.rs` tests own that) or the credit/token-bucket arithmetic
    //! (`VsyncCore`'s own deterministic tests above own that, with no thread/clock/sleep
    //! involved). This is only the wiring in between: a real [`Timer`] feeding a real [`Node`]
    //! that really sends to a real [`EngineActorHandle`] — the same three pieces
    //! `engine_troupe.rs`'s `with_config` assembles, minus the OS-thread `Host` layer, which has
    //! no vsync-specific behavior of its own to prove.
    use super::*;
    use crate::api::private::{EngineControl, EngineData};
    use crate::api::public::AppManagement;
    use actor_scheduler::actors::Schedule;
    use actor_scheduler::mealy::{Lanes, Node};
    use actor_scheduler::spsc::spsc_channel;
    use actor_scheduler::{
        Actor, ActorScheduler, ActorStatus, HandlerResult, SchedulerParams, SystemStatus,
        TrySendError,
    };
    use std::ops::ControlFlow;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex};

    /// Collects every `EngineData::VSync` tick; ignores everything else.
    struct TickCollector {
        ticks: Arc<Mutex<Vec<Instant>>>,
    }

    impl Actor<EngineData, EngineControl, AppManagement> for TickCollector {
        fn handle_data(&mut self, data: EngineData) -> HandlerResult {
            if let EngineData::VSync { timestamp, .. } = data {
                self.ticks.lock().unwrap().push(timestamp);
            }
            Ok(())
        }
        fn handle_control(&mut self, _: EngineControl) -> HandlerResult {
            Ok(())
        }
        fn handle_management(&mut self, _: AppManagement) -> HandlerResult {
            Ok(())
        }
        fn handle_os(&mut self, _status: SystemStatus) -> Result<ActorStatus, HandlerError> {
            Ok(ActorStatus::Idle)
        }
    }

    /// Poll `ticks` until it has at least `n` entries or `deadline` passes.
    fn wait_for_at_least(ticks: &Arc<Mutex<Vec<Instant>>>, n: usize, deadline: Instant) -> usize {
        loop {
            let count = ticks.lock().unwrap().len();
            if count >= n || Instant::now() >= deadline {
                return count;
            }
            std::thread::sleep(Duration::from_millis(2));
        }
    }

    /// Exercises a real `Node<VsyncCore, VsyncWiring, ..>` end to end: a real clock thread ticks
    /// it, a poller thread steps it (the same job a `Host`'s sweep does), and its wiring really
    /// sends to a real `EngineActorHandle`. Saturates the token bucket under sustained demand and
    /// confirms `ReturnToken` unblocks exactly one more — deliberately only "did this eventually
    /// happen" assertions (generous deadlines), never "did this NOT happen by some wall-clock
    /// point", for the same reason the file this replaces gave.
    ///
    /// 1000Hz reaches `MAX_TOKENS` quickly; deadlines stay generous throughout since a shared CI
    /// runner's real tick rate can fall well short of the requested rate.
    #[test]
    fn the_real_green_node_saturates_its_credit_and_return_token_unblocks_one_more() {
        let (engine, mut engine_sched) =
            ActorScheduler::<EngineData, EngineControl, AppManagement>::new(64, 256);
        let ticks = Arc::new(Mutex::new(Vec::new()));
        let ticks_clone = ticks.clone();

        std::thread::Builder::new()
            .name("test-engine-collector".to_string())
            .spawn(move || {
                let mut collector = TickCollector { ticks: ticks_clone };
                engine_sched.run(&mut collector);
            })
            .expect("failed to spawn engine collector thread");

        let (cmd_tx, cmd_rx) = spsc_channel::<VsyncCommand>(8);
        let (_data_tx, data_rx) = spsc_channel::<RenderedResponse>(8);
        let (tick_tx, tick_rx) = spsc_channel::<VsyncManagement>(2);

        let clock = Timer::spawn(
            "test-vsync-clock",
            Schedule::Every(Duration::from_micros(1_000)), // 1000Hz
            move || match tick_tx.try_send(VsyncManagement::Tick) {
                Ok(()) => ControlFlow::Continue(()),
                // A queued tick is as good as this one.
                Err(TrySendError::Full(_)) => ControlFlow::Continue(()),
                Err(TrySendError::Disconnected(_)) => ControlFlow::Break(()),
            },
        );

        let mut node = Node::new_with_lanes(
            VsyncCore::new(1000.0),
            Lanes {
                control: cmd_rx,
                management: tick_rx,
                data: data_rx,
            },
            VsyncCoreWiring {
                tick: engine,
                interval: clock,
            },
            SchedulerParams::DEFAULT,
        );

        cmd_tx
            .try_send(VsyncCommand::Start)
            .expect("room for Start");

        let stop = Arc::new(AtomicBool::new(false));
        let stop_clone = stop.clone();
        // Steps the node concurrently with the real clock thread — the same job a `Host`'s sweep
        // does for a real green node, minus `Host` itself, which has nothing vsync-specific to
        // prove.
        let poller = std::thread::spawn(move || {
            while !stop_clone.load(Ordering::Relaxed) {
                if node.poll() == actor_scheduler::mealy::Step::Idle {
                    std::thread::yield_now();
                }
            }
        });

        // 1. A started node ticks at least once.
        let first = wait_for_at_least(&ticks, 1, Instant::now() + Duration::from_secs(5));
        assert!(first > 0, "a started node must tick at least once");

        // 2. Sustained demand reaches exactly the cap.
        let saturated = wait_for_at_least(
            &ticks,
            MAX_TOKENS as usize,
            Instant::now() + Duration::from_secs(30),
        );
        assert_eq!(
            saturated, MAX_TOKENS as usize,
            "bucket should saturate at exactly its cap under sustained demand"
        );

        // 3. `ReturnToken` — a real message, not a shared global — unblocks exactly one more
        //    real tick.
        cmd_tx
            .try_send(VsyncCommand::ReturnToken)
            .expect("room for ReturnToken");
        let after_return = wait_for_at_least(
            &ticks,
            MAX_TOKENS as usize + 1,
            Instant::now() + Duration::from_secs(30),
        );
        assert_eq!(
            after_return,
            MAX_TOKENS as usize + 1,
            "ReturnToken must unblock exactly one more tick"
        );

        stop.store(true, Ordering::Relaxed);
        poller.join().unwrap();
    }
}
