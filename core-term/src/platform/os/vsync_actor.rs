//! Platform-agnostic vsync actor that sends frame presentation commands at a fixed rate.
//!
//! The VsyncActor runs in a dedicated background thread and sends `PresentFrame` commands
//! to the Display thread at the configured target FPS. This decouples frame rendering from
//! frame presentation, allowing the terminal to update the framebuffer at any rate while
//! presenting at a consistent refresh rate.
//!
//! TODO: Replace thread::sleep() with platform-specific vsync APIs:
//! - macOS: CVDisplayLink
//! - Linux: DRM vblank ioctls or GNOME/KDE compositor APIs
//! - Windows: DwmFlush or DX12 present sync

use crate::orchestrator::OrchestratorSender;
use crate::term::ControlEvent;
use anyhow::{Context, Result};
use log::*;
use std::thread::{self, JoinHandle};
use std::time::Duration;

/// Vsync actor that sends frame presentation commands at a fixed rate.
pub struct VsyncActor {
    thread_handle: Option<JoinHandle<()>>,
}

impl VsyncActor {
    /// Spawns the vsync actor in a background thread.
    ///
    /// # Arguments
    ///
    /// * `orchestrator_tx` - Unified channel to send RequestSnapshot events to Orchestrator
    /// * `target_fps` - Target frames per second (e.g., 60)
    ///
    /// # Returns
    ///
    /// Returns a handle to the vsync actor for cleanup.
    pub fn spawn(orchestrator_tx: OrchestratorSender, target_fps: u32) -> Result<Self> {
        let frame_duration = Duration::from_secs_f64(1.0 / target_fps as f64);

        let thread_handle = thread::Builder::new()
            .name("vsync".to_string())
            .spawn(move || {
                info!("VsyncActor: Started (target: {} FPS)", target_fps);
                loop {
                    thread::sleep(frame_duration);

                    // Send RequestSnapshot event to Orchestrator thread
                    if orchestrator_tx.send(ControlEvent::RequestSnapshot).is_err() {
                        info!("VsyncActor: Orchestrator channel closed, exiting");
                        break;
                    }
                }
                debug!("VsyncActor: Thread exiting");
            })
            .context("Failed to spawn vsync thread")?;

        info!("VsyncActor spawned successfully");
        Ok(Self {
            thread_handle: Some(thread_handle),
        })
    }
}

impl Drop for VsyncActor {
    fn drop(&mut self) {
        debug!("VsyncActor dropped");
        if let Some(handle) = self.thread_handle.take() {
            if let Err(e) = handle.join() {
                error!("VsyncActor thread panicked: {:?}", e);
            }
        }
    }
}
