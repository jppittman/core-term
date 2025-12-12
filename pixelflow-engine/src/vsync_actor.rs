//! VSync Actor - Separate thread that generates vsync timing signals.
//!
//! The VSync actor runs in its own thread and can be controlled via message passing.
//! It sends periodic vsync signals that the engine uses for frame timing.

use actor_scheduler::{Actor, ActorHandle, ActorScheduler};
use log::info;
use std::thread;
use std::time::{Duration, Instant};

/// Messages TO the VSync actor (commands) - Control lane
#[derive(Debug)]
pub enum VsyncCommand {
    /// Start sending vsync signals
    Start,
    /// Stop sending vsync signals (pause)
    Stop,
    /// Update refresh rate (for VRR displays)
    UpdateRefreshRate(f64),
    /// Request current FPS stats
    RequestCurrentFPS,
    /// Shutdown the actor
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

/// Management messages (placeholder for future use)
#[derive(Debug)]
pub enum VsyncManagement {}
actor_scheduler::impl_management_message!(VsyncManagement);

/// VSync actor - generates periodic vsync timing signals.
pub struct VsyncActor<P: pixelflow_core::Pixel> {
    engine_handle: crate::api::private::EngineActorHandle<P>,

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
}

const MAX_TOKENS: u32 = 3;

impl<P: pixelflow_core::Pixel> VsyncActor<P> {
    fn new(refresh_rate: f64, engine_handle: crate::api::private::EngineActorHandle<P>) -> Self {
        let interval = Duration::from_secs_f64(1.0 / refresh_rate);

        info!(
            "VsyncActor: Started with refresh rate {:.2} Hz ({:.2}ms interval), token bucket max: {}",
            refresh_rate,
            interval.as_secs_f64() * 1000.0,
            MAX_TOKENS
        );

        Self {
            engine_handle,
            refresh_rate,
            interval,
            running: false,
            next_vsync: Instant::now(),
            tokens: MAX_TOKENS,
            frame_count: 0,
            fps_start: Instant::now(),
            last_fps: 0.0,
        }
    }

    /// Spawn VSync actor in a new thread.
    ///
    /// Returns an ActorHandle that can be used to send commands and responses.
    pub fn spawn(
        refresh_rate: f64,
        engine_handle: crate::api::private::EngineActorHandle<P>,
    ) -> ActorHandle<RenderedResponse, VsyncCommand, VsyncManagement>
    where
        P: 'static,
    {
        let (handle, mut scheduler) =
            actor_scheduler::create_actor::<RenderedResponse, VsyncCommand, VsyncManagement>(
                1024, // Data buffer (RenderedResponse)
                None, // No wake handler
            );

        thread::Builder::new()
            .name("vsync-actor".to_string())
            .spawn(move || {
                let mut actor = VsyncActor::new(refresh_rate, engine_handle);
                scheduler.run(&mut actor);
            })
            .expect("Failed to spawn vsync actor thread");

        handle
    }

    fn send_vsync(&mut self) {
        let now = Instant::now();
        let timestamp = now;
        let target_timestamp = now + self.interval;

        use crate::api::private::EngineControl;

        if self
            .engine_handle
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
}

impl<P: pixelflow_core::Pixel> Actor<RenderedResponse, VsyncCommand, VsyncManagement>
    for VsyncActor<P>
{
    fn handle_data(&mut self, response: RenderedResponse) {
        // Received frame rendered feedback - add token
        if self.tokens < MAX_TOKENS {
            self.tokens += 1;
            log::trace!(
                "VsyncActor: Token added (frame {}), now have {} tokens",
                response.frame_number,
                self.tokens
            );
        }
    }

    fn handle_control(&mut self, cmd: VsyncCommand) {
        match cmd {
            VsyncCommand::Start => {
                self.running = true;
                self.next_vsync = Instant::now() + self.interval;
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
            }
            VsyncCommand::RequestCurrentFPS => {
                info!("VsyncActor: FPS requested - {:.2} fps", self.last_fps);
                // TODO: Send FPS response back through a response channel
            }
            VsyncCommand::Shutdown => {
                info!("VsyncActor: Shutting down");
                // Scheduler will exit when all senders are dropped
            }
        }
    }

    fn park(&mut self) {
        // VSync timing loop - called after every work cycle
        if self.running && self.tokens > 0 {
            let now = Instant::now();

            // If it's time for next vsync, send it
            if now >= self.next_vsync {
                self.send_vsync();
                self.update_fps();
            } else {
                // Sleep until next vsync (but not too long to stay responsive)
                let sleep_time = self.next_vsync.duration_since(now);
                let sleep_time = sleep_time.min(Duration::from_millis(1)); // Cap at 1ms for responsiveness
                thread::sleep(sleep_time);
            }
        } else {
            // Not running or no tokens - sleep a bit to avoid busy loop
            thread::sleep(Duration::from_millis(1));
        }
    }

    fn handle_management(&mut self, _msg: VsyncManagement) {
        // No management messages yet
    }
}
