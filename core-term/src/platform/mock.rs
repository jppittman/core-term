// src/platform/mock.rs

use crate::platform::actions::PlatformAction;
use crate::platform::PlatformChannels;
use crate::term::ControlEvent;
use crate::platform::BackendEvent;
use anyhow::{Context, Result};
use std::sync::mpsc::{Receiver, TryRecvError, RecvTimeoutError};
use std::time::Duration;

pub struct MockPlatform {
    pub display_action_rx: Receiver<PlatformAction>,
    pub platform_event_tx: crate::orchestrator::OrchestratorSender,
    // We store actions that are not RequestRedraw.
    // RequestRedraw contains a Box<TerminalSnapshot> which we must return, so we can't keep it.
    pub received_actions: Vec<PlatformAction>,
    pub redraw_count: usize,
}

impl MockPlatform {
    pub fn new(channels: PlatformChannels) -> Self {
        Self {
            display_action_rx: channels.display_action_rx,
            platform_event_tx: channels.platform_event_tx,
            received_actions: Vec::new(),
            redraw_count: 0,
        }
    }

    /// Process all pending actions in the channel.
    /// Non-blocking.
    pub fn process_actions(&mut self) -> Result<()> {
        loop {
            match self.display_action_rx.try_recv() {
                Ok(action) => {
                    match action {
                        PlatformAction::RequestRedraw(snapshot) => {
                             self.redraw_count += 1;
                             // Echo back immediately to simulate rendering completion
                             // This is critical for the orchestrator to reuse the snapshot buffer
                             self.platform_event_tx
                                .send(ControlEvent::FrameRendered(snapshot))
                                .context("Failed to return snapshot")?;
                        }
                        other => {
                            self.received_actions.push(other);
                        }
                    }
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    // Channel disconnected means sender (Orchestrator) has dropped the channel.
                    // This is expected during shutdown.
                    break;
                },
            }
        }
        Ok(())
    }

    /// Wait for the next action from the orchestrator.
    /// Handles RequestRedraw side-effects automatically.
    pub fn wait_for_next_action(&mut self, timeout: Duration) -> Result<Option<PlatformAction>> {
        match self.display_action_rx.recv_timeout(timeout) {
            Ok(action) => {
                match action {
                    PlatformAction::RequestRedraw(snapshot) => {
                        self.redraw_count += 1;
                        // Clone snapshot so we can return one and send one back
                        let snapshot_clone = snapshot.clone();
                        self.platform_event_tx
                           .send(ControlEvent::FrameRendered(snapshot))
                           .context("Failed to return snapshot")?;

                        Ok(Some(PlatformAction::RequestRedraw(snapshot_clone)))
                    }
                    other => {
                         Ok(Some(other))
                    }
                }
            }
            Err(RecvTimeoutError::Timeout) => Ok(None),
            Err(RecvTimeoutError::Disconnected) => {
                // Treated as end of stream
                Ok(None)
            }
        }
    }

    /// Inject a backend event (e.g. keyboard input, resize)
    pub fn send_event(&self, event: BackendEvent) -> Result<()> {
         self.platform_event_tx.send(event).context("Failed to send event")
    }
}
