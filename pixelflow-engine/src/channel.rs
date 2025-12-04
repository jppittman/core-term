// src/channel.rs
//!
//! Unified event channel for the engine with priority scheduling.
//! Implements the "Smart Sender" pattern to route display events to a bounded
//! slow lane with backpressure, and control commands to an unbounded fast lane.
//! Uses a "Doorbell" signal to wake the engine when display events are ready.

use crate::display::messages::WindowId;
use crate::display::DisplayEvent;
use crate::vsync_actor::VsyncActor;
use pixelflow_core::Pixel;
use pixelflow_render::Frame;
use std::sync::mpsc::{self, Receiver, SendError, Sender, SyncSender, TrySendError};
use std::time::{Duration, Instant};

/// Commands sent TO the engine (from driver or external control).
#[derive(Debug)]
pub enum EngineCommand<P: Pixel> {
    /// Display events from driver (slow lane, backpressure)
    DisplayEvent(DisplayEvent),

    /// Response: Presentation complete, framebuffer returned for reuse
    PresentComplete(Frame<P>),

    /// Response: Driver operation completed (SetTitle, etc.)
    DriverAck,

    /// Doorbell - wakes engine when display events are ready
    Doorbell,

    /// VSync signal from display driver with timing information.
    /// Routes through fast lane (control channel) for high-priority delivery.
    /// DEPRECATED: Use VsyncActorReady instead.
    VSync {
        /// Timestamp when this vsync occurred (monotonic clock)
        timestamp: Instant,
        /// When the next frame should be ready for display
        target_timestamp: Instant,
        /// Current refresh interval (may vary with VRR displays)
        refresh_interval: Duration,
    },

    /// VSync actor provided by platform/display driver.
    /// Engine will use this actor to receive vsync signals.
    VsyncActorReady(VsyncActor),
}

/// Commands sent TO the driver (from engine).
#[derive(Debug)]
pub enum DriverCommand<P: Pixel> {
    // ========================================================================
    // Lifecycle
    // ========================================================================
    /// Create a new window with the given ID and dimensions.
    CreateWindow {
        id: WindowId,
        width: u32,
        height: u32,
        title: String,
    },

    /// Destroy a window.
    DestroyWindow { id: WindowId },

    // ========================================================================
    // Rendering
    // ========================================================================
    /// Present a frame to a window.
    /// The Frame contains typed pixel data matching the driver's Pixel type.
    Present { id: WindowId, frame: Frame<P> },

    // ========================================================================
    // Window properties
    // ========================================================================
    /// Set window title.
    SetTitle { id: WindowId, title: String },

    /// Set window size.
    SetSize {
        id: WindowId,
        width: u32,
        height: u32,
    },

    // ========================================================================
    // Clipboard
    // ========================================================================
    /// Copy text to clipboard.
    CopyToClipboard(String),

    /// Request paste from clipboard.
    RequestPaste,

    // ========================================================================
    // Audio
    // ========================================================================
    /// Ring the terminal bell.
    Bell,

    // ========================================================================
    // Control
    // ========================================================================
    /// Shutdown the driver.
    Shutdown,
}

/// Type-safe sender wrapper that routes commands to the appropriate channel.
/// Display events go to `display_tx` (bounded), responses/control go to `control_tx` (unbounded).
pub struct EngineSender<P: Pixel> {
    control_tx: Sender<EngineCommand<P>>, // Unbounded, High Priority
    display_tx: SyncSender<EngineCommand<P>>, // Bounded, Low Priority (backpressure)
}

impl<P: Pixel> Clone for EngineSender<P> {
    fn clone(&self) -> Self {
        Self {
            control_tx: self.control_tx.clone(),
            display_tx: self.display_tx.clone(),
        }
    }
}

impl<P: Pixel> EngineSender<P> {
    /// Create a new EngineSender.
    /// Internal use only - use create_engine_channels() instead.
    pub fn new(
        control_tx: Sender<EngineCommand<P>>,
        display_tx: SyncSender<EngineCommand<P>>,
    ) -> Self {
        Self {
            control_tx,
            display_tx,
        }
    }

    /// Send a command, automatically routing to the appropriate channel.
    /// Blocks if sending display events and the slow lane is full (backpressure).
    pub fn send(&self, cmd: EngineCommand<P>) -> Result<(), SendError<EngineCommand<P>>> {
        match cmd {
            EngineCommand::DisplayEvent(_) => {
                // 1. Block on Slow Lane (Backpressure applies here)
                self.display_tx.send(cmd).map_err(|e| SendError(e.0))?;
                // 2. Ring Doorbell (Wake Engine)
                // Ignore send errors - if control channel is closed, engine is dead anyway
                let _ = self.control_tx.send(EngineCommand::Doorbell);
                Ok(())
            }
            // Fast Lane: responses, doorbell, ack
            _ => self.control_tx.send(cmd),
        }
    }

    /// Try to send a command without blocking.
    /// Returns Err if channel is full (display events only) or disconnected.
    pub fn try_send(&self, cmd: EngineCommand<P>) -> Result<(), TrySendError<EngineCommand<P>>> {
        match cmd {
            EngineCommand::DisplayEvent(_) => {
                // 1. Try Send to Slow Lane (Backpressure)
                self.display_tx.try_send(cmd)?;
                // 2. Ring Doorbell (Wake Engine)
                let _ = self.control_tx.send(EngineCommand::Doorbell);
                Ok(())
            }
            // Fast Lane
            _ => self
                .control_tx
                .send(cmd)
                .map_err(|e| TrySendError::Disconnected(e.0)),
        }
    }
}

/// Channel bundle for engine-side communication.
pub struct EngineChannels<P: Pixel> {
    /// Sender for driver -> engine (display events, responses)
    pub engine_sender: EngineSender<P>,
    /// Engine receives control/responses here (high priority)
    pub control_rx: Receiver<EngineCommand<P>>,
    /// Engine receives display events here (low priority, backpressure)
    pub display_rx: Receiver<EngineCommand<P>>,
}

/// Create channels for driver -> engine communication.
///
/// Returns EngineChannels containing:
/// - engine_sender: for driver to push events to engine
/// - control_rx/display_rx: for engine to receive events
///
/// Engine -> driver communication is via driver.send() directly.
///
/// # Arguments
/// * `display_buffer_size` - Size of the bounded display event buffer (backpressure threshold)
pub fn create_engine_channels<P: Pixel>(display_buffer_size: usize) -> EngineChannels<P> {
    let (control_tx, control_rx) = mpsc::channel();
    let (display_tx, display_rx) = mpsc::sync_channel(display_buffer_size);
    let engine_sender = EngineSender::new(control_tx, display_tx);

    EngineChannels {
        engine_sender,
        control_rx,
        display_rx,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::input::{KeySymbol, Modifiers};
    use pixelflow_render::color::Rgba;

    #[test]
    fn test_create_channels() {
        let _channels: EngineChannels<Rgba> = create_engine_channels(16);
    }

    #[test]
    fn test_responses_go_to_fast_lane() {
        let channels: EngineChannels<Rgba> = create_engine_channels(16);

        channels
            .engine_sender
            .send(EngineCommand::DriverAck)
            .unwrap();
        channels
            .engine_sender
            .send(EngineCommand::Doorbell)
            .unwrap();

        // Should be in control channel
        assert!(matches!(
            channels.control_rx.try_recv(),
            Ok(EngineCommand::DriverAck)
        ));
        assert!(matches!(
            channels.control_rx.try_recv(),
            Ok(EngineCommand::Doorbell)
        ));

        // Display channel should be empty
        assert!(channels.display_rx.try_recv().is_err());
    }

    #[test]
    fn test_display_event_goes_to_slow_lane_with_doorbell() {
        let channels: EngineChannels<Rgba> = create_engine_channels(16);

        let event = DisplayEvent::Key {
            id: WindowId::PRIMARY,
            symbol: KeySymbol::Char('a'),
            modifiers: Modifiers::empty(),
            text: Some("a".to_string()),
        };
        channels
            .engine_sender
            .send(EngineCommand::DisplayEvent(event))
            .unwrap();

        // Display event should be in display channel
        assert!(matches!(
            channels.display_rx.try_recv(),
            Ok(EngineCommand::DisplayEvent(_))
        ));

        // Doorbell should be in control channel
        assert!(matches!(
            channels.control_rx.try_recv(),
            Ok(EngineCommand::Doorbell)
        ));
    }

    #[test]
    fn test_backpressure() {
        let channels: EngineChannels<Rgba> = create_engine_channels(2);

        let event = || DisplayEvent::CloseRequested {
            id: WindowId::PRIMARY,
        };

        // Fill the buffer
        channels
            .engine_sender
            .send(EngineCommand::DisplayEvent(event()))
            .unwrap();
        channels
            .engine_sender
            .send(EngineCommand::DisplayEvent(event()))
            .unwrap();

        // Third should fail with try_send (buffer full)
        let result = channels
            .engine_sender
            .try_send(EngineCommand::DisplayEvent(event()));
        assert!(matches!(result, Err(TrySendError::Full(_))));
    }
}
