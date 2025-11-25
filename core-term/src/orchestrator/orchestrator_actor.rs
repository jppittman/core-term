//! Orchestrator Actor - processes events and generates snapshots with backpressure.
//!
//! This actor owns the terminal state machine (TerminalEmulator) and processes
//! events from PTY, Vsync, and Platform threads. It uses a two-channel handshake with the Platform
//! to prevent wasted snapshot generation: only generates snapshots when the Platform signals
//! it's ready via the `ready_rx` channel.
//!
//! The PTY thread (EventMonitorActor) owns the AnsiProcessor and sends parsed AnsiCommands,
//! not raw bytes. This allows parallel parsing while the Orchestrator processes frames.

use crate::keys;
use crate::orchestrator::OrchestratorEvent;
use crate::platform::actions::PlatformAction;
use crate::platform::backends::{BackendEvent, MouseButton};
use crate::term::{ControlEvent, EmulatorAction, EmulatorInput, TerminalEmulator, UserInputAction};
use anyhow::{Context, Result};
use log::*;
use std::sync::mpsc::{Receiver, RecvError, SyncSender};
use std::thread::{self, JoinHandle};

/// Orchestrator actor that runs in a background thread.
///
/// Receives events from PTY, Vsync, and Platform threads, processes them through the terminal
/// emulator, and sends snapshots to the Platform only when it's ready (backpressure).
pub struct OrchestratorActor {
    thread_handle: Option<JoinHandle<()>>,
}

impl OrchestratorActor {
    /// Spawns the Orchestrator actor in a background thread.
    ///
    /// # Arguments
    ///
    /// * `term_emulator` - The terminal emulator (takes ownership)
    /// * `orchestrator_rx` - Unified channel to receive all events (IO, Vsync, Platform)
    /// * `display_action_tx` - Channel to send PlatformActions to Platform
    /// * `pty_action_tx` - Channel to send PlatformActions to PTY
    ///
    /// # Returns
    ///
    /// Returns `Self` (handle to the actor for cleanup)
    pub fn spawn(
        term_emulator: TerminalEmulator,
        orchestrator_rx: Receiver<OrchestratorEvent>,
        display_action_tx: SyncSender<PlatformAction>,
        pty_action_tx: SyncSender<PlatformAction>,
        waker: Box<dyn crate::platform::waker::EventLoopWaker>,
    ) -> Result<Self> {
        info!("OrchestratorActor: Spawning background thread");

        let thread_handle = thread::Builder::new()
            .name("orchestrator".to_string())
            .spawn(move || {
                if let Err(e) = Self::actor_thread_main(
                    term_emulator,
                    orchestrator_rx,
                    display_action_tx,
                    pty_action_tx,
                    waker,
                ) {
                    error!("OrchestratorActor thread error: {:#}", e);
                }
            })
            .context("Failed to spawn Orchestrator actor thread")?;

        info!("OrchestratorActor spawned successfully");

        Ok(Self {
            thread_handle: Some(thread_handle),
        })
    }

    /// Main loop for the Orchestrator actor thread (unified channel model).
    ///
    /// Receives events from a single unified channel and processes them.
    /// Manages snapshot lifecycle: receives FrameRendered (with snapshot), generates on RequestSnapshot.
    fn actor_thread_main(
        mut term_emulator: TerminalEmulator,
        orchestrator_rx: Receiver<OrchestratorEvent>,
        display_action_tx: SyncSender<PlatformAction>,
        pty_action_tx: SyncSender<PlatformAction>,
        waker: Box<dyn crate::platform::waker::EventLoopWaker>,
    ) -> Result<()> {
        debug!("OrchestratorActor: Starting event loop (unified channel model)");

        let mut pending_emulator_actions = Vec::new();

        loop {
            // Blocking receive from unified channel
            let event = match orchestrator_rx.recv() {
                Ok(event) => event,
                Err(RecvError) => {
                    info!("OrchestratorActor: Orchestrator channel disconnected, shutting down");
                    return Ok(());
                }
            };

            // Process the event
            Self::process_event(
                event,
                &mut term_emulator,
                &mut pending_emulator_actions,
                &display_action_tx,
                &waker,
            )?;

            // Handle any pending emulator actions
            for action in pending_emulator_actions.drain(..) {
                Self::handle_emulator_action(action, &display_action_tx, &pty_action_tx)?;
            }
        }
    }

    /// Process a single event and update state accordingly.
    ///
    /// Handles three event types:
    /// - IOEvent: ANSI commands from PTY
    /// - Control: RequestSnapshot (generates snapshot), FrameRendered (returns snapshot to terminal), Resize
    /// - BackendEvent: User input (keyboard, mouse)
    fn process_event(
        event: OrchestratorEvent,
        term_emulator: &mut TerminalEmulator,
        pending_emulator_actions: &mut Vec<EmulatorAction>,
        display_action_tx: &SyncSender<PlatformAction>,
        waker: &Box<dyn crate::platform::waker::EventLoopWaker>,
    ) -> Result<()> {
        match event {
            OrchestratorEvent::Control(control_event) => {
                match control_event {
                    ControlEvent::RequestSnapshot => {
                        debug!("OrchestratorActor: Received RequestSnapshot");

                        // Ask terminal for its snapshot (it owns the buffer)
                        if let Some(snapshot) = term_emulator.get_render_snapshot() {
                            display_action_tx
                                .send(PlatformAction::RequestRedraw(Box::new(snapshot)))
                                .context("Failed to send RequestRedraw to Platform")?;
                        } else {
                            debug!("OrchestratorActor: No snapshot available (synchronized_output or buffer out)");
                        }
                    }
                    ControlEvent::FrameRendered(boxed_snapshot) => {
                        debug!("OrchestratorActor: Received FrameRendered (returning snapshot to terminal)");
                        // Return the snapshot buffer to the terminal
                        term_emulator.return_snapshot(*boxed_snapshot);
                    }
                    ControlEvent::Resize {
                        width_px,
                        height_px,
                        scale_factor,
                    } => {
                        info!(
                            "OrchestratorActor: Resizing to {}x{} px (scale={})",
                            width_px, height_px, scale_factor
                        );
                        if let Some(action) = term_emulator.interpret_input(EmulatorInput::Control(
                            ControlEvent::Resize {
                                width_px,
                                height_px,
                                scale_factor,
                            },
                        )) {
                            pending_emulator_actions.push(action);
                        }
                    }
                }
            }
            OrchestratorEvent::IOEvent {
                commands: ansi_commands,
            } => {
                debug!(
                    "OrchestratorActor: Received {} ANSI commands from PTY",
                    ansi_commands.len()
                );

                for command in ansi_commands {
                    if let Some(action) =
                        term_emulator.interpret_input(EmulatorInput::Ansi(command))
                    {
                        pending_emulator_actions.push(action);
                    }
                }

                // Wake the platform event loop to handle OOB PTY input
                if let Err(e) = waker.wake() {
                    warn!("OrchestratorActor: Failed to wake platform event loop: {}", e);
                }
            }
            OrchestratorEvent::BackendEvent(backend_event) => {
                debug!(
                    "OrchestratorActor: Received BackendEvent: {:?}",
                    backend_event
                );

                if let BackendEvent::CloseRequested = backend_event {
                    info!("OrchestratorActor: CloseRequested received - sending ShutdownComplete and exiting");
                    display_action_tx
                        .send(PlatformAction::ShutdownComplete)
                        .context("Failed to send ShutdownComplete to Platform")?;
                    return Err(anyhow::anyhow!("CloseRequested - shutting down"));
                }

                let emulator_input = Self::process_backend_event(backend_event, term_emulator)?;

                if let Some(input) = emulator_input {
                    if let Some(action) = term_emulator.interpret_input(input) {
                        pending_emulator_actions.push(action);
                    }
                }
            }
        }
        Ok(())
    }

    /// Process a BackendEvent and return the corresponding EmulatorInput.
    fn process_backend_event(
        backend_event: BackendEvent,
        _term_emulator: &mut TerminalEmulator,
    ) -> Result<Option<EmulatorInput>> {
        match backend_event {
            BackendEvent::CloseRequested => {
                warn!("OrchestratorActor: CloseRequested reached process_backend_event (should not happen)");
                Ok(None)
            }
            BackendEvent::Resize {
                width_px,
                height_px,
                scale_factor,
            } => {
                // Forward physical dimensions to emulator, which will calculate cols/rows
                info!(
                    "OrchestratorActor: Forwarding resize {}x{} px (scale={}) to emulator",
                    width_px, height_px, scale_factor
                );

                Ok(Some(EmulatorInput::Control(ControlEvent::Resize {
                    width_px,
                    height_px,
                    scale_factor,
                })))
            }
            BackendEvent::Key {
                symbol,
                modifiers,
                text,
            } => {
                debug!(
                    "OrchestratorActor: Key: {:?} + {:?}, Text: {:?}",
                    modifiers, symbol, text
                );
                let key_input_action =
                    keys::map_key_event_to_action(symbol, modifiers, &crate::config::CONFIG)
                        .unwrap_or(UserInputAction::KeyInput {
                            symbol,
                            modifiers,
                            text: if text.is_empty() { None } else { Some(text) },
                        });
                Ok(Some(EmulatorInput::User(key_input_action)))
            }
            BackendEvent::MouseButtonPress {
                button,
                x,
                y,
                scale_factor,
                modifiers: _,
            } => {
                let input = match button {
                    MouseButton::Left => {
                        // Forward pixel coordinates to emulator, which will convert to cells
                        Some(EmulatorInput::User(UserInputAction::StartSelection {
                            x_px: x,
                            y_px: y,
                            scale_factor,
                        }))
                    }
                    MouseButton::Middle => {
                        Some(EmulatorInput::User(UserInputAction::RequestPrimaryPaste))
                    }
                    _ => None,
                };
                Ok(input)
            }
            BackendEvent::MouseButtonRelease {
                button,
                x: _x,
                y: _y,
                scale_factor: _scale_factor,
                modifiers: _,
            } => {
                if button == MouseButton::Left {
                    Ok(Some(EmulatorInput::User(
                        UserInputAction::ApplySelectionClear,
                    )))
                } else {
                    Ok(None)
                }
            }
            BackendEvent::MouseMove {
                x,
                y,
                scale_factor,
                modifiers: _,
            } => {
                // Forward pixel coordinates to emulator, which will convert to cells
                Ok(Some(EmulatorInput::User(
                    UserInputAction::ExtendSelection {
                        x_px: x,
                        y_px: y,
                        scale_factor,
                    },
                )))
            }
            BackendEvent::FocusGained => {
                Ok(Some(EmulatorInput::User(UserInputAction::FocusGained)))
            }
            BackendEvent::FocusLost => Ok(Some(EmulatorInput::User(UserInputAction::FocusLost))),
            BackendEvent::PasteData { text } => {
                Ok(Some(EmulatorInput::User(UserInputAction::PasteText(text))))
            }
            BackendEvent::ClipboardDataRequested => {
                // X11 clipboard protocol: another app is requesting our clipboard data
                // TODO: Send current clipboard text to display via SubmitClipboardData
                // For now, just ignore this event - clipboard won't work properly on X11
                warn!("ClipboardDataRequested: clipboard support not yet implemented");
                Ok(None)
            }
        }
    }

    /// Handle an EmulatorAction by sending it to the appropriate actor.
    fn handle_emulator_action(
        action: EmulatorAction,
        display_tx: &SyncSender<PlatformAction>,
        pty_tx: &SyncSender<PlatformAction>,
    ) -> Result<()> {
        debug!("OrchestratorActor: Handling EmulatorAction: {:?}", action);

        match action {
            EmulatorAction::WritePty(data) => {
                pty_tx
                    .send(PlatformAction::Write(data))
                    .context("Failed to send Write to PTY")?;
            }
            EmulatorAction::SetTitle(title) => {
                display_tx
                    .send(PlatformAction::SetTitle(title))
                    .context("Failed to send SetTitle to Display")?;
            }
            EmulatorAction::RingBell => {
                display_tx
                    .send(PlatformAction::RingBell)
                    .context("Failed to send RingBell to Display")?;
            }
            EmulatorAction::CopyToClipboard(text) => {
                display_tx
                    .send(PlatformAction::CopyToClipboard(text))
                    .context("Failed to send CopyToClipboard to Display")?;
            }
            EmulatorAction::SetCursorVisibility(visible) => {
                display_tx
                    .send(PlatformAction::SetCursorVisibility(visible))
                    .context("Failed to send SetCursorVisibility to Display")?;
            }
            EmulatorAction::RequestRedraw => {
                trace!("OrchestratorActor: RequestRedraw received (no-op in actor model)");
            }
            EmulatorAction::RequestClipboardContent => {
                display_tx
                    .send(PlatformAction::RequestPaste)
                    .context("Failed to send RequestPaste to Display")?;
            }
            EmulatorAction::ResizePty { cols, rows } => {
                info!("OrchestratorActor: Resizing PTY to {}x{}", cols, rows);
                pty_tx
                    .send(PlatformAction::ResizePty { cols, rows })
                    .context("Failed to send ResizePty to PTY")?;
            }
        }
        Ok(())
    }
}

impl Drop for OrchestratorActor {
    fn drop(&mut self) {
        debug!("OrchestratorActor dropped");
        if let Some(handle) = self.thread_handle.take() {
            if let Err(e) = handle.join() {
                error!("OrchestratorActor thread panicked: {:?}", e);
            }
        }
    }
}
