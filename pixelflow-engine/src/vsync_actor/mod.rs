//! VSync Timer - Sends periodic vsync signals with token-based backpressure.
//!
//! Architecture:
//! - VsyncActor runs on dedicated thread with simple mpsc channel
//! - Receives VsyncRequest messages (RenderedResponse, UpdateRefreshRate, Shutdown)
//! - Sends VsyncMessage to engine's data lane on each tick
//! - Token bucket prevents overwhelming the engine

use log::info;
use std::sync::mpsc::{channel, Sender, TryRecvError};
use std::thread;
use std::time::{Duration, Instant};

const MAX_TOKENS: u32 = 3;

// ============================================================================
// VSync Messages
// ============================================================================

/// Requests sent TO the VSync actor (inbound)
#[derive(Debug, Clone)]
pub enum VsyncRequest {
    /// Frame completed, restore a token
    RenderedResponse,
    /// Update the refresh rate
    UpdateRefreshRate(f64),
    /// Shutdown the timer
    Shutdown,
}

/// Messages sent FROM the VSync actor to the engine (outbound, goes to data lane)
#[derive(Debug, Clone, Copy)]
pub struct VsyncMessage {
    pub frame_id: u64,
    pub timestamp: Instant,
    pub target_timestamp: Instant,
    pub refresh_interval: Duration,
}

// ============================================================================
// VSync Handle (for sending requests TO VSync)
// ============================================================================

/// Handle for sending requests to the VSync actor
pub trait VsyncHandle: Send + Sync + Clone {
    /// Send a request to the VSync actor
    fn send(&self, req: VsyncRequest) -> Result<(), actor_scheduler::SendError>;
}

/// Concrete VSync actor handle
#[derive(Clone)]
pub struct VsyncActorHandle {
    tx: Sender<VsyncRequest>,
}

impl VsyncActorHandle {
    pub fn new(tx: Sender<VsyncRequest>) -> Self {
        Self { tx }
    }

    pub fn send(&self, req: VsyncRequest) -> Result<(), actor_scheduler::SendError> {
        self.tx.send(req).map_err(|_| actor_scheduler::SendError)
    }
}

impl VsyncHandle for VsyncActorHandle {
    fn send(&self, req: VsyncRequest) -> Result<(), actor_scheduler::SendError> {
        self.send(req)
    }
}

// ============================================================================
// VSync Actor
// ============================================================================

/// VSync timer actor that sends periodic timing signals
pub struct VsyncActor<F>
where
    F: Fn(VsyncMessage) -> Result<(), actor_scheduler::SendError>,
{
    /// Closure to send VsyncMessage to engine
    engine_sender: F,
    /// Current refresh rate
    refresh_rate: f64,
    /// Token bucket for backpressure
    tokens: u32,
    /// Next vsync time
    next_vsync: Instant,
    /// Frame counter for FPS reporting
    frame_count: u64,
    /// FPS calculation start time
    fps_start: Instant,
    /// Missed tick flag
    missed_tick: bool,
}

impl<F> VsyncActor<F>
where
    F: Fn(VsyncMessage) -> Result<(), actor_scheduler::SendError> + Send + 'static,
{
    /// Create a new VSync actor
    pub fn new(engine_sender: F, refresh_rate: f64) -> Self {
        info!(
            "VsyncActor: Created with refresh rate {:.2} Hz ({:.2}ms interval), token bucket max: {}",
            refresh_rate,
            1000.0 / refresh_rate,
            MAX_TOKENS
        );

        Self {
            engine_sender,
            refresh_rate,
            tokens: MAX_TOKENS,
            next_vsync: Instant::now(),
            frame_count: 0,
            fps_start: Instant::now(),
            missed_tick: false,
        }
    }

    /// Get the current refresh interval
    fn refresh_interval(&self) -> Duration {
        Duration::from_secs_f64(1.0 / self.refresh_rate)
    }

    /// Send a VSync message if we have tokens
    fn send_vsync(&mut self) {
        if self.tokens == 0 {
            if !self.missed_tick {
                self.missed_tick = true;
                log::trace!("VsyncActor: Tick missed due to backpressure (no tokens)");
            }
            return;
        }

        let now = Instant::now();
        let interval = self.refresh_interval();
        let timestamp = now;
        let target_timestamp = now + interval;

        log::trace!("VsyncActor: Sending VSync frame {} at {:?}, {} tokens available", self.frame_count, timestamp, self.tokens);

        let msg = VsyncMessage {
            frame_id: self.frame_count,
            timestamp,
            target_timestamp,
            refresh_interval: interval,
        };

        if (self.engine_sender)(msg).is_ok() {
            self.tokens -= 1;
            self.frame_count += 1;
            self.missed_tick = false;
            log::trace!("VsyncActor: VSync sent successfully, token consumed, {} remaining", self.tokens);
            self.next_vsync = timestamp + interval;
        } else {
            info!("VsyncActor: Engine disconnected");
        }
    }

    /// Check if it's time to send a VSync
    fn tick(&mut self) {
        let now = Instant::now();

        if now >= self.next_vsync {
            self.send_vsync();
        }

        // Update FPS stats
        let elapsed = self.fps_start.elapsed();
        if elapsed >= Duration::from_secs(1) {
            let fps = self.frame_count as f64 / elapsed.as_secs_f64();
            info!("VsyncActor: Current FPS: {:.2}", fps);
            self.frame_count = 0;
            self.fps_start = Instant::now();
        }
    }
}

impl<F> VsyncActor<F>
where
    F: Fn(VsyncMessage) -> Result<(), actor_scheduler::SendError> + Send + 'static,
{
    fn handle_request(&mut self, req: VsyncRequest) {
        match req {
            VsyncRequest::RenderedResponse => {
                if self.tokens < MAX_TOKENS {
                    self.tokens += 1;
                    log::trace!("VsyncActor: Token added, now have {} tokens", self.tokens);

                    // If we had a missed tick and now have tokens, send immediately
                    if self.missed_tick && self.tokens > 0 {
                        self.send_vsync();
                    }
                }
            }
            VsyncRequest::UpdateRefreshRate(rate) => {
                info!("VsyncActor: Updating refresh rate from {:.2} Hz to {:.2} Hz",
                      self.refresh_rate, rate);
                self.refresh_rate = rate;
                self.next_vsync = Instant::now();
            }
            VsyncRequest::Shutdown => {
                info!("VsyncActor: Shutdown requested");
            }
        }
    }
}

/// Spawn a VSync actor in a dedicated thread
pub fn spawn_vsync_actor<F>(
    engine_sender: F,
    refresh_rate: f64,
) -> VsyncActorHandle
where
    F: Fn(VsyncMessage) -> Result<(), actor_scheduler::SendError> + Send + 'static,
{
    let (tx, rx) = channel();

    thread::Builder::new()
        .name("vsync-actor".to_string())
        .spawn(move || {
            let mut actor = VsyncActor::new(engine_sender, refresh_rate);

            loop {
                // Process all pending messages
                loop {
                    match rx.try_recv() {
                        Ok(msg) => actor.handle_request(msg),
                        Err(TryRecvError::Empty) => break,
                        Err(TryRecvError::Disconnected) => {
                            info!("VsyncActor: Channel disconnected, shutting down");
                            return;
                        }
                    }
                }

                // Send VSync if needed
                actor.tick();

                // Sleep briefly to avoid busy loop
                thread::sleep(Duration::from_micros(100));
            }
        })
        .expect("Failed to spawn vsync actor thread");

    VsyncActorHandle::new(tx)
}

#[cfg(test)]
mod tests;
