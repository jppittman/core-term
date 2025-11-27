// src/platform/mod.rs
//
// This module re-exports the platform-specific functionalities.

use crate::config::Config;
use crate::display::{DisplayManager, DriverRequest, DriverResponse};
use crate::orchestrator::OrchestratorSender;
use crate::platform::actions::PlatformAction;
use crate::renderer::{RenderChannels, RenderResult, RenderWork};
use anyhow::{Context, Result};
pub use backends::BackendEvent;
use log::*;
use std::sync::mpsc::Receiver;

/// Platform actions (Orchestrator -> Platform).
pub mod actions;
/// Backend event types and definitions.
pub mod backends;
// #[cfg(target_os = "linux")]
// pub mod console_platform;
/// Font management and fallback logic.
pub mod font_manager;
// #[cfg(target_os = "linux")]
// pub mod linux_x11;
#[cfg(use_cocoa_display)]
pub mod macos;
// pub mod os; // Moved to io module
/// Platform traits.
pub mod platform_trait;
/// Event loop waker abstraction.
pub mod waker;

#[cfg(use_cocoa_display)]
pub use macos::MacosPlatform;

// MockPlatform is obsolete after refactor to actor model
// #[cfg(test)]
// pub mod mock;

// Type alias for backward compatibility
/// Alias for orchestrator events.
pub type PlatformEvent = crate::orchestrator::OrchestratorEvent;

/// Groups all channels used for Platform ↔ Orchestrator communication.
/// This reduces function parameter counts and makes ownership clear.
pub struct PlatformChannels {
    /// Platform receives display actions from orchestrator (including RequestRedraw with snapshot)
    pub display_action_rx: std::sync::mpsc::Receiver<actions::PlatformAction>,
    /// Platform sends events to orchestrator (user input, FrameRendered with returned snapshots)
    pub platform_event_tx: OrchestratorSender,
}

/// Generic platform implementation that works for all display-based platforms.
/// Contains all the common logic for event polling, rendering, and action handling.
pub struct GenericPlatform {
    /// The display manager responsible for the window and driver.
    pub display_manager: DisplayManager,
    /// Channels for communicating with the render thread.
    pub render_channels: RenderChannels,
    /// Application configuration.
    pub config: Config,
    /// Channel to send events to the orchestrator.
    pub platform_event_tx: OrchestratorSender,
    /// Channel to receive actions from the orchestrator (Option because consumed in run).
    pub display_action_rx: Option<Receiver<PlatformAction>>,
    /// The current framebuffer (owned here or by display driver).
    pub framebuffer: Option<Box<[u8]>>,
}

impl GenericPlatform {
    /// Create a new GenericPlatform with the given channels.
    ///
    /// # Parameters
    /// * `channels` - Platform communication channels.
    /// * `render_channels` - Renderer communication channels.
    ///
    /// # Returns
    /// * A new `GenericPlatform` instance.
    pub fn new(channels: PlatformChannels, render_channels: RenderChannels) -> Result<Self> {
        info!("GenericPlatform::new() - Initializing display-based platform");

        // Create the display manager (initializes driver and window)
        let display_manager = DisplayManager::new().context("Failed to create DisplayManager")?;
        let metrics = display_manager.metrics();
        info!(
            "Display initialized: {}x{} px, scale={}",
            metrics.width_px, metrics.height_px, metrics.scale_factor
        );

        let config = Config::default();

        info!("GenericPlatform::new() - Initialization complete");

        Ok(Self {
            display_manager,
            render_channels,
            config,
            platform_event_tx: channels.platform_event_tx,
            display_action_rx: Some(channels.display_action_rx),
            framebuffer: None, // Framebuffer starts in display driver
        })
    }

    /// Run the platform event loop. This consumes self.
    ///
    /// This loop polls for display events, processes incoming actions from the orchestrator,
    /// handles rendering, and manages the framebuffer lifecycle.
    pub fn run(mut self) -> Result<()> {
        info!("GenericPlatform::run() - Starting event loop");

        // Take ownership of input channel from orchestrator
        let display_action_rx = self
            .display_action_rx
            .take()
            .context("display_action_rx already consumed")?;

        // Send initial Resize event
        let metrics = self.display_manager.metrics();
        self.platform_event_tx
            .send(BackendEvent::Resize {
                width_px: metrics.width_px as u16,
                height_px: metrics.height_px as u16,
                scale_factor: metrics.scale_factor,
            })
            .context("Failed to send initial Resize event")?;
        info!("Sent initial Resize event to orchestrator");

        let mut shutdown_complete = false;
        loop {
            // Poll display events via DisplayManager
            let response = self
                .display_manager
                .handle_request(DriverRequest::PollEvents)
                .context("DisplayManager event polling failed")?;

            if let DriverResponse::Events(display_events) = response {
                for display_event in display_events {
                    // Convert DisplayEvent to BackendEvent using From trait
                    let backend_event: BackendEvent = display_event.into();
                    debug!(
                        "GenericPlatform: Sending BackendEvent to orchestrator: {:?}",
                        backend_event
                    );
                    if let Err(e) = self.platform_event_tx.send(backend_event) {
                        warn!("Failed to send display event to orchestrator: {}", e);
                        info!("Platform event channel closed, shutting down");
                        shutdown_complete = true;
                        break;
                    }
                }
            }

            // Drain all available display actions
            while let Ok(action) = display_action_rx.try_recv() {
                trace!(
                    "GenericPlatform: Processing action: {:?}",
                    std::mem::discriminant(&action)
                );
                match action {
                    PlatformAction::RequestRedraw(snapshot) => {
                        let (cols, rows) = snapshot.dimensions;
                        debug!(
                            "GenericPlatform: Processing RequestRedraw ({}x{} grid)",
                            cols, rows
                        );

                        // Get framebuffer from display or self (ping-pong pattern)
                        let framebuffer = if let Some(fb) = self.framebuffer.take() {
                            fb
                        } else {
                            // Request framebuffer from display
                            let response = self
                                .display_manager
                                .handle_request(DriverRequest::RequestFramebuffer)?;
                            if let DriverResponse::Framebuffer(fb) = response {
                                fb
                            } else {
                                return Err(anyhow::anyhow!("Expected Framebuffer response"));
                            }
                        };

                        // Send rendering work to render thread
                        let metrics = self.display_manager.metrics();
                        let work = RenderWork {
                            snapshot,
                            framebuffer,
                            display_width_px: metrics.width_px as u16,
                            display_height_px: metrics.height_px as u16,
                            scale_factor: metrics.scale_factor,
                        };

                        if let Err(e) = self.render_channels.work_tx.send(work) {
                            return Err(anyhow::anyhow!("Failed to send render work: {}", e));
                        }
                        trace!("GenericPlatform: Sent render work to render thread");

                        // Receive rendered result from render thread
                        let RenderResult {
                            snapshot,
                            framebuffer,
                        } = self
                            .render_channels
                            .result_rx
                            .recv()
                            .context("Failed to receive render result")?;
                        trace!("GenericPlatform: Received render result from render thread");

                        // Create RenderSnapshot with framebuffer + dimensions
                        let metrics = self.display_manager.metrics();
                        let render_snapshot = crate::display::messages::RenderSnapshot {
                            framebuffer,
                            width_px: metrics.width_px,
                            height_px: metrics.height_px,
                        };

                        // Present RenderSnapshot and get it back, recovering on error
                        let response = self
                            .display_manager
                            .handle_request(DriverRequest::Present(render_snapshot));

                        match response {
                            Ok(DriverResponse::PresentComplete(render_snapshot)) => {
                                // Extract framebuffer from returned snapshot
                                self.framebuffer = Some(render_snapshot.framebuffer);
                            }
                            Ok(_) => {
                                return Err(anyhow::anyhow!("Expected PresentComplete response"));
                            }
                            Err(crate::display::DisplayError::PresentationFailed(
                                render_snapshot,
                                reason,
                            )) => {
                                // Recover the framebuffer even on error
                                self.framebuffer = Some(render_snapshot.framebuffer);
                                warn!("Presentation failed, buffer recovered: {}", reason);
                                // Continue execution - don't propagate the error
                            }
                            Err(crate::display::DisplayError::Generic(e)) => {
                                return Err(e);
                            }
                        }

                        // Return snapshot to orchestrator
                        use crate::term::ControlEvent;
                        self.platform_event_tx
                            .send(ControlEvent::FrameRendered(snapshot))
                            .context("Failed to return snapshot via FrameRendered")?;

                        debug!("GenericPlatform: Frame rendered and snapshot returned");
                    }
                    PlatformAction::SetTitle(title) => {
                        let _ = self
                            .display_manager
                            .handle_request(DriverRequest::SetTitle(title));
                    }
                    PlatformAction::RingBell => {
                        let _ = self.display_manager.handle_request(DriverRequest::Bell);
                    }
                    PlatformAction::CopyToClipboard(text) => {
                        let _ = self
                            .display_manager
                            .handle_request(DriverRequest::CopyToClipboard(text));
                    }
                    PlatformAction::SetCursorVisibility(visible) => {
                        let _ = self
                            .display_manager
                            .handle_request(DriverRequest::SetCursorVisibility(visible));
                    }
                    PlatformAction::RequestPaste => {
                        let _ = self
                            .display_manager
                            .handle_request(DriverRequest::RequestPaste);
                    }
                    PlatformAction::ShutdownComplete => {
                        info!("GenericPlatform: Received ShutdownComplete - exiting event loop");
                        shutdown_complete = true;
                    }
                    PlatformAction::Write(_) | PlatformAction::ResizePty { .. } => {
                        warn!(
                            "GenericPlatform received PTY action - this should go to PTY thread!"
                        );
                    }
                }
            }

            if shutdown_complete {
                break;
            }
        }

        info!("GenericPlatform::run() - Exiting normally");
        Ok(())
    }
}
