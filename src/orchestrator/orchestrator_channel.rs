// src/orchestrator/orchestrator_channel.rs
//!
//! Unified event channel for the orchestrator.
//! All actors (Platform, PTY, Vsync) send events to a single channel using OrchestratorEvent.

use crate::ansi::AnsiCommand;
use crate::platform::backends::BackendEvent;
use crate::term::ControlEvent;
use std::sync::mpsc::{SendError, SyncSender, TrySendError};

/// All event types that the orchestrator can receive.
/// This is the unified message type for the single orchestrator input channel.
#[derive(Debug)]
pub enum OrchestratorEvent {
    /// Display/input events from the Platform thread (keyboard, mouse, resize, etc.)
    BackendEvent(BackendEvent),

    /// PTY output events - parsed ANSI command sequences
    IOEvent { commands: Vec<AnsiCommand> },

    /// Internal control flow events (vsync, snapshot handshake, etc.)
    Control(ControlEvent),
}

/// Type-safe sender wrapper that allows sending specific event types.
/// Automatically converts to OrchestratorEvent using From trait.
#[derive(Clone)]
pub struct OrchestratorSender {
    tx: SyncSender<OrchestratorEvent>,
}

impl OrchestratorSender {
    /// Create a new typed sender from a raw channel.
    pub fn new(tx: SyncSender<OrchestratorEvent>) -> Self {
        Self { tx }
    }

    /// Send a message, automatically converting to OrchestratorEvent.
    /// Blocks if channel is full (provides backpressure).
    pub fn send<T>(&self, msg: T) -> Result<(), SendError<OrchestratorEvent>>
    where
        OrchestratorEvent: From<T>,
    {
        let wrapped_msg = OrchestratorEvent::from(msg);
        self.tx.send(wrapped_msg)
    }

    /// Try to send a message without blocking.
    /// Returns Err if channel is full or disconnected.
    pub fn try_send<T>(&self, msg: T) -> Result<(), TrySendError<OrchestratorEvent>>
    where
        OrchestratorEvent: From<T>,
    {
        let wrapped_msg = OrchestratorEvent::from(msg);
        self.tx.try_send(wrapped_msg)
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
