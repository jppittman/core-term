// src/orchestrator/orchestrator_channel.rs
//!
//! Unified event channel for the orchestrator with priority scheduling.
//! Implements the "Smart Sender" pattern to route high-priority events (UI, Control)
//! to an unbounded fast lane, and bulk PTY data to a bounded slow lane with backpressure.
//! Uses a "Doorbell" signal to wake the orchestrator when PTY data is ready.

use crate::ansi::AnsiCommand;
use crate::platform::backends::BackendEvent;
use crate::term::ControlEvent;
use std::sync::mpsc::{self, Receiver, SendError, Sender, SyncSender, TrySendError};

/// All event types that the orchestrator can receive.
/// This is the unified message type for the orchestrator input channels.
#[derive(Debug)]
pub enum OrchestratorEvent {
    /// Display/input events from the Platform thread (keyboard, mouse, resize, etc.)
    BackendEvent(BackendEvent),

    /// PTY output events - parsed ANSI command sequences
    IOEvent { commands: Vec<AnsiCommand> },

    /// Internal control flow events (vsync, snapshot handshake, etc.)
    Control(ControlEvent),
}

/// Type-safe sender wrapper that routes events to the appropriate channel.
/// High priority events go to `ui_tx` (unbounded), bulk PTY data goes to `pty_tx` (bounded).
#[derive(Clone)]
pub struct OrchestratorSender {
    ui_tx: Sender<OrchestratorEvent>,      // Unbounded, High Priority
    pty_tx: SyncSender<OrchestratorEvent>, // Bounded, Low Priority
}

impl OrchestratorSender {
    /// Create a new OrchestratorSender.
    /// Internal use only - use create_orchestrator_channels() instead.
    pub fn new(ui_tx: Sender<OrchestratorEvent>, pty_tx: SyncSender<OrchestratorEvent>) -> Self {
        Self { ui_tx, pty_tx }
    }

    /// Send a message, automatically converting to OrchestratorEvent and routing it.
    /// Blocks if sending to the slow lane (PTY data) and it's full (backpressure).
    pub fn send<T>(&self, msg: T) -> Result<(), SendError<OrchestratorEvent>>
    where
        OrchestratorEvent: From<T>,
    {
        let event = OrchestratorEvent::from(msg);
        match event {
            OrchestratorEvent::IOEvent { .. } => {
                // 1. Block on Slow Lane (Backpressure applies here)
                self.pty_tx.send(event).map_err(|e| SendError(e.0))?;
                // 2. Ring Doorbell (Wake Orchestrator)
                // Ignore send errors here - if UI channel is closed, orchestrator is dead anyway
                let _ = self.ui_tx.send(ControlEvent::PtyDataReady.into());
                Ok(())
            }
            // Fast Lane: Always succeeds (conceptually, unless disconnected)
            _ => self.ui_tx.send(event),
        }
    }

    /// Try to send a message without blocking.
    /// Returns Err if channel is full (slow lane only) or disconnected.
    pub fn try_send<T>(&self, msg: T) -> Result<(), TrySendError<OrchestratorEvent>>
    where
        OrchestratorEvent: From<T>,
    {
        let event = OrchestratorEvent::from(msg);
        match event {
            OrchestratorEvent::IOEvent { .. } => {
                // 1. Try Send to Slow Lane (Backpressure)
                self.pty_tx.try_send(event)?;
                // 2. Ring Doorbell (Wake Orchestrator)
                let _ = self.ui_tx.send(ControlEvent::PtyDataReady.into());
                Ok(())
            }
            // Fast Lane: Always succeeds (conceptually)
            _ => self
                .ui_tx
                .send(event)
                .map_err(|e| TrySendError::Disconnected(e.0)),
        }
    }
}

// Conversion traits for ergonomic sending

impl From<BackendEvent> for OrchestratorEvent {
    fn from(event: BackendEvent) -> Self {
        OrchestratorEvent::BackendEvent(event)
    }
}

impl From<ControlEvent> for OrchestratorEvent {
    fn from(event: ControlEvent) -> Self {
        OrchestratorEvent::Control(event)
    }
}

// Note: IOEvent is constructed directly since it has a field, not a tuple variant

/// Create the orchestrator channels and sender.
/// Returns (Sender, UI_Receiver, PTY_Receiver).
pub fn create_orchestrator_channels(
    pty_buffer_size: usize,
) -> (
    OrchestratorSender,
    Receiver<OrchestratorEvent>,
    Receiver<OrchestratorEvent>,
) {
    let (ui_tx, ui_rx) = mpsc::channel();
    let (pty_tx, pty_rx) = mpsc::sync_channel(pty_buffer_size);

    let sender = OrchestratorSender::new(ui_tx, pty_tx);

    (sender, ui_rx, pty_rx)
}
