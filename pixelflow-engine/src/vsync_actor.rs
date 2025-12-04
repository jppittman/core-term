//! VSync Actor - Separate thread that generates vsync timing signals.
//!
//! The VSync actor runs in its own thread and can be controlled via message passing.
//! It sends periodic vsync signals that the engine uses for frame timing.

use log::info;
use std::sync::mpsc::{Receiver, Sender, TryRecvError};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};

/// Messages TO the VSync actor (commands)
#[derive(Debug)]
pub enum VsyncCommand {
    /// Start sending vsync signals
    Start,
    /// Stop sending vsync signals (pause)
    Stop,
    /// Update refresh rate (for VRR displays)
    UpdateRefreshRate(f64),
    /// Shutdown the actor
    Shutdown,
}

/// Messages FROM the VSync actor (signals)
#[derive(Debug, Clone, Copy)]
pub struct VsyncSignal {
    /// When this vsync occurred
    pub timestamp: Instant,
    /// When the next frame should be ready
    pub target_timestamp: Instant,
    /// Current refresh interval (may vary with VRR)
    pub refresh_interval: Duration,
}

/// VSync actor handle - used to communicate with the vsync actor thread.
///
/// The actor runs in a separate thread and sends periodic timing signals.
/// Commands can be sent to control the actor (start/stop/update rate).
pub struct VsyncActor {
    cmd_tx: Sender<VsyncCommand>,
    signal_rx: Receiver<VsyncSignal>,
    shutdown_flag: Arc<AtomicBool>,
}

impl std::fmt::Debug for VsyncActor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VsyncActor")
            .field("shutdown", &self.shutdown_flag.load(Ordering::Relaxed))
            .finish()
    }
}

impl VsyncActor {
    /// Create and spawn a new VSync actor.
    ///
    /// # Arguments
    /// * `refresh_rate` - Initial display refresh rate in Hz (e.g., 60.0, 120.0)
    ///
    /// # Returns
    /// A handle to communicate with the actor. The actor starts in stopped state.
    pub fn spawn(refresh_rate: f64) -> Self {
        let (cmd_tx, cmd_rx) = std::sync::mpsc::channel();
        let (signal_tx, signal_rx) = std::sync::mpsc::channel();
        let shutdown_flag = Arc::new(AtomicBool::new(false));
        let shutdown_clone = shutdown_flag.clone();

        thread::Builder::new()
            .name("vsync-actor".to_string())
            .spawn(move || {
                run_vsync_actor(cmd_rx, signal_tx, shutdown_clone, refresh_rate);
            })
            .expect("Failed to spawn vsync actor thread");

        Self {
            cmd_tx,
            signal_rx,
            shutdown_flag,
        }
    }

    /// Send a command to the vsync actor.
    pub fn send_command(&self, cmd: VsyncCommand) -> Result<(), std::sync::mpsc::SendError<VsyncCommand>> {
        self.cmd_tx.send(cmd)
    }

    /// Try to receive the next vsync signal (non-blocking).
    ///
    /// Returns `Ok(signal)` if a signal is ready, `Err` otherwise.
    pub fn try_recv_signal(&self) -> Result<VsyncSignal, TryRecvError> {
        self.signal_rx.try_recv()
    }

    /// Wait for the next vsync signal (blocking).
    pub fn recv_signal(&self) -> Result<VsyncSignal, std::sync::mpsc::RecvError> {
        self.signal_rx.recv()
    }

    /// Check if there are any pending signals without removing them.
    pub fn has_pending_signal(&self) -> bool {
        matches!(self.signal_rx.try_recv(), Ok(_))
    }
}

impl Drop for VsyncActor {
    fn drop(&mut self) {
        let _ = self.cmd_tx.send(VsyncCommand::Shutdown);
        self.shutdown_flag.store(true, Ordering::Relaxed);
    }
}

/// VSync actor main loop - runs in separate thread.
fn run_vsync_actor(
    cmd_rx: Receiver<VsyncCommand>,
    signal_tx: Sender<VsyncSignal>,
    shutdown: Arc<AtomicBool>,
    initial_refresh_rate: f64,
) {
    let mut refresh_rate = initial_refresh_rate;
    let mut interval = Duration::from_secs_f64(1.0 / refresh_rate);
    let mut running = false;
    let mut next_vsync = Instant::now();

    info!(
        "VsyncActor: Started with refresh rate {:.2} Hz ({:.2}ms interval)",
        refresh_rate,
        interval.as_secs_f64() * 1000.0
    );

    loop {
        if shutdown.load(Ordering::Relaxed) {
            break;
        }

        // Process all pending commands
        loop {
            match cmd_rx.try_recv() {
                Ok(VsyncCommand::Start) => {
                    running = true;
                    next_vsync = Instant::now() + interval;
                    info!("VsyncActor: Started");
                }
                Ok(VsyncCommand::Stop) => {
                    running = false;
                    info!("VsyncActor: Stopped");
                }
                Ok(VsyncCommand::UpdateRefreshRate(new_rate)) => {
                    refresh_rate = new_rate;
                    interval = Duration::from_secs_f64(1.0 / refresh_rate);
                    info!(
                        "VsyncActor: Updated refresh rate to {:.2} Hz ({:.2}ms interval)",
                        refresh_rate,
                        interval.as_secs_f64() * 1000.0
                    );
                }
                Ok(VsyncCommand::Shutdown) => {
                    info!("VsyncActor: Shutting down");
                    return;
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    info!("VsyncActor: Command channel disconnected");
                    return;
                }
            }
        }

        if running {
            let now = Instant::now();

            if now >= next_vsync {
                // Time for vsync - send signal
                let timestamp = now;
                let target_timestamp = now + interval;

                if signal_tx
                    .send(VsyncSignal {
                        timestamp,
                        target_timestamp,
                        refresh_interval: interval,
                    })
                    .is_err()
                {
                    // Engine dropped the receiver - shutdown
                    info!("VsyncActor: Signal receiver disconnected, shutting down");
                    break;
                }

                // Calculate next vsync (no cumulative drift)
                next_vsync = timestamp + interval;
            } else {
                // Sleep until next vsync
                thread::sleep(next_vsync - now);
            }
        } else {
            // Not running, just wait for commands
            thread::sleep(Duration::from_millis(10));
        }
    }

    info!("VsyncActor: Exited");
}
