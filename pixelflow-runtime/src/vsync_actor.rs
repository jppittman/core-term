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

use actor_scheduler::{Actor, ActorHandle};
use log::info;
use std::sync::mpsc::Sender;
use std::thread;
use std::time::{Duration, Instant};

use crate::platform::PlatformPixel;

/// Configuration for VsyncActor
#[derive(Debug, Clone)]
pub struct VsyncConfig {
    pub refresh_rate: f64,
}

impl Default for VsyncConfig {
    fn default() -> Self {
        Self { refresh_rate: 60.0 }
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
        engine_handle: crate::api::private::EngineActorHandle<crate::platform::PlatformPixel>,
        self_handle: ActorHandle<RenderedResponse, VsyncCommand, VsyncManagement>,
    },
}
actor_scheduler::impl_management_message!(VsyncManagement);

/// Internal commands sent to the clock thread
#[derive(Debug)]
enum ClockCommand {
    /// Update the tick interval
    SetInterval(Duration),
    /// Stop the clock thread
    Stop,
}

/// VSync actor - generates periodic vsync timing signals.
pub struct VsyncActor {
    engine_handle: Option<crate::api::private::EngineActorHandle<PlatformPixel>>,

    // VSync state
    refresh_rate: f64,
    interval: Duration,
    running: bool,
    next_vsync: Instant,

    // Token bucket for adaptive frame pacing
    tokens: u32,

    // FPS tracking
    frame_count: u64,
    fps_start: Instant,
    last_fps: f64,

    // Control for the clock thread
    clock_control: Option<Sender<ClockCommand>>,
}

const MAX_TOKENS: u32 = 3;

impl VsyncActor {
    /// Create empty VsyncActor for troupe pattern - configured via SetConfig management message.
    pub fn new_empty() -> Self {
        Self {
            engine_handle: None,
            refresh_rate: 60.0,
            interval: Duration::from_secs_f64(1.0 / 60.0),
            running: false,
            next_vsync: Instant::now(),
            tokens: MAX_TOKENS,
            frame_count: 0,
            fps_start: Instant::now(),
            last_fps: 0.0,
            clock_control: None,
        }
    }

    /// Create a new VsyncActor. Takes the handle to itself (for the clock thread).
    pub fn new(
        refresh_rate: f64,
        engine_handle: crate::api::private::EngineActorHandle<PlatformPixel>,
        self_handle: ActorHandle<RenderedResponse, VsyncCommand, VsyncManagement>,
    ) -> Self {
        let interval = Duration::from_secs_f64(1.0 / refresh_rate);

        info!(
            "VsyncActor: Started with refresh rate {:.2} Hz ({:.2}ms interval), token bucket max: {}",
            refresh_rate,
            interval.as_secs_f64() * 1000.0,
            MAX_TOKENS
        );

        // Spawn the clock thread
        let (clock_tx, clock_rx) = std::sync::mpsc::channel();
        let clock_handle = self_handle.clone();

        thread::Builder::new()
            .name("vsync-clock".to_string())
            .spawn(move || {
                let mut current_interval = interval;
                loop {
                    match clock_rx.recv_timeout(current_interval) {
                        Ok(ClockCommand::Stop) => break,
                        Ok(ClockCommand::SetInterval(d)) => current_interval = d,
                        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                            // Time to tick
                            if clock_handle.send(VsyncManagement::Tick).is_err() {
                                // Actor is gone
                                break;
                            }
                        }
                        Err(_) => break, // Channel disconnected
                    }
                }
            })
            .expect("Failed to spawn vsync clock thread");

        Self {
            engine_handle: Some(engine_handle),
            refresh_rate,
            interval,
            running: false,
            next_vsync: Instant::now(),
            tokens: MAX_TOKENS,
            frame_count: 0,
            fps_start: Instant::now(),
            last_fps: 0.0,
            clock_control: Some(clock_tx),
        }
    }

    /// Spawn VSync actor in a new thread.
    ///
    /// Returns an ActorHandle that can be used to send commands and responses.
    pub fn spawn(
        refresh_rate: f64,
        engine_handle: crate::api::private::EngineActorHandle<PlatformPixel>,
    ) -> ActorHandle<RenderedResponse, VsyncCommand, VsyncManagement> {
        let (handle, mut scheduler) =
            actor_scheduler::create_actor::<RenderedResponse, VsyncCommand, VsyncManagement>(
                1024, // Data buffer (RenderedResponse)
                None, // No wake handler
            );

        // We need to pass the handle to new(), so we clone it
        let actor_handle = handle.clone();

        thread::Builder::new()
            .name("vsync-actor".to_string())
            .spawn(move || {
                let mut actor = VsyncActor::new(refresh_rate, engine_handle, actor_handle);
                scheduler.run(&mut actor);
            })
            .expect("Failed to spawn vsync actor thread");

        handle
    }

    fn send_vsync(&mut self) {
        let Some(ref engine_handle) = self.engine_handle else {
            return; // Not configured yet
        };

        let now = Instant::now();
        let timestamp = now;
        let target_timestamp = now + self.interval;

        use crate::api::private::EngineControl;

        if engine_handle
            .send(EngineControl::VSync {
                timestamp,
                target_timestamp,
                refresh_interval: self.interval,
            })
            .is_ok()
        {
            // Consume token after successful send
            self.tokens -= 1;
            self.frame_count += 1;
            log::trace!(
                "VsyncActor: VSync sent, token consumed, {} remaining",
                self.tokens
            );

            // Calculate next vsync (no cumulative drift)
            self.next_vsync = timestamp + self.interval;
        } else {
            // Engine dropped the receiver
            info!("VsyncActor: Engine disconnected");
        }
    }

    fn update_fps(&mut self) {
        let elapsed = self.fps_start.elapsed();
        if elapsed >= Duration::from_secs(1) {
            self.last_fps = self.frame_count as f64 / elapsed.as_secs_f64();
            info!("VsyncActor: Current FPS: {:.2}", self.last_fps);
            self.frame_count = 0;
            self.fps_start = Instant::now();
        }
    }

    fn handle_tick(&mut self) {
        if self.running && self.tokens > 0 {
            let now = Instant::now();

            // If it's time for next vsync (or close enough/past due), send it
            // With clock thread, we are roughly at the right time.
            if now >= self.next_vsync {
                self.send_vsync();
                self.update_fps();
            }
        }
    }
}

impl Actor<RenderedResponse, VsyncCommand, VsyncManagement> for VsyncActor {
    type Error = ();

    fn handle_data(&mut self, response: RenderedResponse) -> Result<(), ()> {
        // Received frame rendered feedback - add token
        if self.tokens < MAX_TOKENS {
            self.tokens += 1;
            log::trace!(
                "VsyncActor: Token added (frame {}), now have {} tokens",
                response.frame_number,
                self.tokens
            );
        }
        Ok(())
    }

    fn handle_control(&mut self, cmd: VsyncCommand) -> Result<(), ()> {
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
                self.refresh_rate = new_rate;
                self.interval = Duration::from_secs_f64(1.0 / self.refresh_rate);
                info!(
                    "VsyncActor: Updated refresh rate to {:.2} Hz ({:.2}ms interval)",
                    self.refresh_rate,
                    self.interval.as_secs_f64() * 1000.0
                );

                // Update clock thread
                if let Some(ref tx) = self.clock_control {
                    let _ = tx.send(ClockCommand::SetInterval(self.interval));
                }
            }
            VsyncCommand::RequestCurrentFPS(sender) => {
                info!("VsyncActor: FPS requested - {:.2} fps", self.last_fps);
                if let Err(e) = sender.send(self.last_fps) {
                    log::warn!("VsyncActor: Failed to send FPS response: {:?}", e);
                }
            }
            VsyncCommand::Shutdown => {
                info!("VsyncActor: Shutting down");
                if let Some(ref tx) = self.clock_control {
                    let _ = tx.send(ClockCommand::Stop);
                }
                // Scheduler will exit when all senders are dropped
                // We should probably drop our own handles if we held any that loop back?
                // But we don't hold loopback handles in struct, only for clock thread.
            }
        }
        Ok(())
    }

    fn handle_management(&mut self, msg: VsyncManagement) -> Result<(), ()> {
        match msg {
            VsyncManagement::Tick => self.handle_tick(),
            VsyncManagement::SetConfig {
                config,
                engine_handle,
                self_handle,
            } => {
                // Configure the vsync actor (called via Management after construction)
                self.engine_handle = Some(engine_handle);
                self.refresh_rate = config.refresh_rate;
                self.interval = Duration::from_secs_f64(1.0 / config.refresh_rate);
                self.running = true;

                info!(
                    "VsyncActor: Configured with {:.2} Hz, auto-started",
                    config.refresh_rate
                );

                // Spawn clock thread
                let (clock_tx, clock_rx) = std::sync::mpsc::channel();
                let interval = self.interval;

                thread::Builder::new()
                    .name("vsync-clock".to_string())
                    .spawn(move || {
                        let mut current_interval = interval;
                        loop {
                            match clock_rx.recv_timeout(current_interval) {
                                Ok(ClockCommand::Stop) => break,
                                Ok(ClockCommand::SetInterval(d)) => current_interval = d,
                                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                                    if self_handle.send(VsyncManagement::Tick).is_err() {
                                        break;
                                    }
                                }
                                Err(_) => break,
                            }
                        }
                    })
                    .expect("Failed to spawn vsync clock thread");

                self.clock_control = Some(clock_tx);
            }
        }
        Ok(())
    }

    fn park(&mut self, _hint: actor_scheduler::ParkHint) -> Result<actor_scheduler::ParkHint, ()> {
        // No-op. We are driven by the clock thread messages.
        // The scheduler will block on the mailbox (doorbell) automatically.
        Ok(actor_scheduler::ParkHint::Wait)
    }
}

// ActorTypes impl for VsyncActor
impl actor_scheduler::ActorTypes for VsyncActor {
    type Data = RenderedResponse;
    type Control = VsyncCommand;
    type Management = VsyncManagement;
}

// TroupeActor impl for VsyncActor
impl<'a, Dir: 'a> actor_scheduler::TroupeActor<'a, Dir> for VsyncActor {
    fn new(_dir: &Dir) -> Self {
        Self::new_empty()
    }
}
