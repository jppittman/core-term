// src/platform/mod.rs
//
// This module re-exports the platform-specific functionalities.

use crate::config::Config;
use crate::display::{DisplayManager, DriverRequest, DriverResponse};
use crate::orchestrator::OrchestratorSender;
use crate::platform::actions::PlatformAction;
use crate::platform::backends::{PlatformState, RenderCommand};
use crate::rasterizer::{compile_into_buffer, SoftwareRasterizer};
use crate::renderer::Renderer;
use anyhow::{Context, Result};
pub use backends::BackendEvent;
use log::*;
use std::sync::mpsc::Receiver;

pub mod actions;
pub mod backends;
#[cfg(target_os = "linux")]
pub mod console_platform;
pub mod font_manager;
#[cfg(target_os = "linux")]
pub mod linux_x11;
pub mod macos;
pub mod os;
pub mod platform_trait;

pub use macos::MacosPlatform;

#[cfg(test)]
pub mod mock;

// Type alias for backward compatibility
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
    pub display_manager: DisplayManager,
    pub rasterizer: SoftwareRasterizer,
    pub renderer: Renderer,
    pub config: Config,
    pub platform_event_tx: OrchestratorSender,
    pub display_action_rx: Option<Receiver<PlatformAction>>,
    pub framebuffer: Option<Box<[u8]>>,
}

impl GenericPlatform {
    /// Create a new GenericPlatform with the given channels.
    pub fn new(channels: PlatformChannels) -> Result<Self> {
        info!("GenericPlatform::new() - Initializing display-based platform");

        // Create the display manager (initializes driver and window)
        let display_manager = DisplayManager::new().context("Failed to create DisplayManager")?;
        let metrics = display_manager.metrics();
        info!(
            "Display initialized: {}x{} px, scale={}",
            metrics.width_px, metrics.height_px, metrics.scale_factor
        );

        // Create rasterizer (gets font metrics from CONFIG/FontManager)
        let rasterizer = SoftwareRasterizer::new();
        let renderer = Renderer::new();
        let config = Config::default();

        info!("GenericPlatform::new() - Initialization complete");

        Ok(Self {
            display_manager,
            rasterizer,
            renderer,
            config,
            platform_event_tx: channels.platform_event_tx,
            display_action_rx: Some(channels.display_action_rx),
            framebuffer: None, // Framebuffer starts in display driver
        })
    }

    /// Get current platform state (dimensions, font metrics, etc.)
    pub fn get_current_platform_state(&self) -> PlatformState {
        let metrics = self.display_manager.metrics();

        // Font metrics come from rasterizer's internal state
        // TODO: Expose cell metrics from rasterizer or get from FontManager
        const FONT_CELL_WIDTH_PX: usize = 8;
        const FONT_CELL_HEIGHT_PX: usize = 16;

        PlatformState {
            event_fd: None, // Display-based platforms don't use FD-based events
            font_cell_width_px: FONT_CELL_WIDTH_PX,
            font_cell_height_px: FONT_CELL_HEIGHT_PX,
            scale_factor: metrics.scale_factor,
            display_width_px: metrics.width_px as u16,
            display_height_px: metrics.height_px as u16,
        }
    }

    /// Run the platform event loop. This consumes self.
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
                match action {
                    PlatformAction::RequestRedraw(snapshot) => {
                        let (cols, rows) = snapshot.dimensions;
                        debug!(
                            "GenericPlatform: Processing RequestRedraw ({}x{} grid)",
                            cols, rows
                        );

                        let platform_state = self.get_current_platform_state();

                        // Prepare render commands
                        let mut render_commands = self.renderer.prepare_render_commands(
                            &snapshot,
                            &self.config,
                            &platform_state,
                        );
                        render_commands.push(RenderCommand::PresentFrame);

                        // Get framebuffer from display or self (ping-pong pattern)
                        let mut framebuffer = if let Some(fb) = self.framebuffer.take() {
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

                        // Compile into framebuffer
                        let metrics = self.display_manager.metrics();
                        compile_into_buffer(
                            &mut self.rasterizer,
                            render_commands,
                            &mut framebuffer,
                            metrics.width_px as usize,
                            metrics.height_px as usize,
                            platform_state.font_cell_width_px,
                            platform_state.font_cell_height_px,
                        );

                        // Present framebuffer and get it back
                        let response = self
                            .display_manager
                            .handle_request(DriverRequest::Present(framebuffer))?;
                        if let DriverResponse::PresentComplete(fb) = response {
                            self.framebuffer = Some(fb); // Store for next frame
                        } else {
                            return Err(anyhow::anyhow!("Expected PresentComplete response"));
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
