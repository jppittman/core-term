//! Orchestrator Actor - processes events and generates snapshots with backpressure.
//!
//! This actor owns the terminal state machine (TerminalEmulator) and processes
//! events from PTY, Vsync, and Platform threads. It uses a two-channel handshake with the Platform
//! to prevent wasted snapshot generation: only generates snapshots when the Platform signals
//! it's ready via the `ready_rx` channel.
//!
//! The PTY thread (EventMonitorActor) owns the AnsiProcessor and sends parsed AnsiCommands,
//! not raw bytes. This allows parallel parsing while the Orchestrator processes frames.

use crate::ansi::AnsiCommand;
use crate::keys;
use crate::platform::actions::PlatformAction;
use crate::platform::backends::{BackendEvent, MouseButton, PlatformState};
use crate::platform::PlatformEvent;
use crate::term::snapshot::RenderSnapshot;
use crate::term::{ControlEvent, EmulatorAction, EmulatorInput, TerminalEmulator, UserInputAction};
use anyhow::{Context, Result};
use log::*;
use std::sync::mpsc::{Receiver, Sender, SyncSender, TryRecvError};
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
    /// * `event_rx` - Channel to receive PlatformEvents (from Platform, PTY, Vsync)
    /// * `snapshot_tx` - Channel to send RenderSnapshots to Platform
    /// * `snapshot_pool_rx` - Channel to receive reusable RenderSnapshots from Platform
    /// * `display_action_tx` - Channel to send PlatformActions to Platform
    /// * `pty_action_tx` - Channel to send PlatformActions to PTY
    /// * `initial_platform_state` - Initial platform state
    ///
    /// # Returns
    ///
    /// Returns `Self` (handle to the actor for cleanup)
    #[allow(clippy::too_many_arguments)]
    pub fn spawn(
        term_emulator: TerminalEmulator,
        event_rx: Receiver<PlatformEvent>,
        snapshot_tx: SyncSender<RenderSnapshot>,
        snapshot_pool_rx: Receiver<RenderSnapshot>,
        display_action_tx: Sender<PlatformAction>,
        pty_action_tx: Sender<PlatformAction>,
        initial_platform_state: PlatformState,
    ) -> Result<Self> {
        info!("OrchestratorActor: Spawning background thread");

        let thread_handle = thread::Builder::new()
            .name("orchestrator".to_string())
            .spawn(move || {
                if let Err(e) = Self::actor_thread_main(
                    term_emulator,
                    event_rx,
                    snapshot_tx,
                    snapshot_pool_rx,
                    display_action_tx,
                    pty_action_tx,
                    initial_platform_state,
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

    /// Main loop for the Orchestrator actor thread.
    ///
    /// Blocks on `event_rx.recv()` for the first event, then drains additional events with
    /// `try_recv()` to coalesce state updates. Only generates snapshots when both:
    /// 1. A frame was requested (by Vsync or user input)
    /// 2. A snapshot buffer is available from the pool (Platform has returned it)
    #[allow(clippy::too_many_arguments)]
    fn actor_thread_main(
        mut term_emulator: TerminalEmulator,
        event_rx: Receiver<PlatformEvent>,
        snapshot_tx: SyncSender<RenderSnapshot>,
        snapshot_pool_rx: Receiver<RenderSnapshot>,
        display_action_tx: Sender<PlatformAction>,
        pty_action_tx: Sender<PlatformAction>,
        platform_state: PlatformState,
    ) -> Result<()> {
        debug!("OrchestratorActor: Starting event loop");

        let mut pending_emulator_actions = Vec::new();

        loop {
            let event = match event_rx.recv() {
                Ok(event) => event,
                Err(_) => {
                    info!("OrchestratorActor: Event channel closed, shutting down");
                    return Ok(());
                }
            };

            pending_emulator_actions.clear();
            let mut frame_requested = false;

            Self::process_event(
                event,
                &mut term_emulator,
                &platform_state,
                &mut pending_emulator_actions,
                &mut frame_requested,
                &display_action_tx,
            )?;

            loop {
                match event_rx.try_recv() {
                    Ok(event) => {
                        Self::process_event(
                            event,
                            &mut term_emulator,
                            &platform_state,
                            &mut pending_emulator_actions,
                            &mut frame_requested,
                            &display_action_tx,
                        )?;
                    }
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => {
                        info!("OrchestratorActor: Event channel disconnected, shutting down");
                        return Ok(());
                    }
                }
            }

            for action in pending_emulator_actions.drain(..) {
                Self::handle_emulator_action(action, &display_action_tx, &pty_action_tx)?;
            }

            if frame_requested {
                match snapshot_pool_rx.try_recv() {
                    Ok(mut snapshot) => {
                        // Populate the snapshot with current terminal state
                        if term_emulator.populate_snapshot(&mut snapshot) {
                            snapshot_tx
                                .send(snapshot)
                                .context("Failed to send snapshot to Display")?;
                        } else {
                            // synchronized_output is active, skip frame but return snapshot to pool
                            debug!("OrchestratorActor: Synchronized output active, skipping frame");
                            snapshot_tx
                                .send(snapshot)
                                .context("Failed to return snapshot to pool")?;
                        }
                    }
                    Err(TryRecvError::Empty) => {
                        trace!("OrchestratorActor: Snapshot pool empty (Platform still rendering), skipping frame");
                    }
                    Err(TryRecvError::Disconnected) => {
                        info!("OrchestratorActor: Snapshot pool channel disconnected, shutting down");
                        return Ok(());
                    }
                }
            }
        }
    }

    /// Process a single event and update state accordingly.
    ///
    /// Updates `frame_requested` and `pending_emulator_actions` based on the event type.
    /// Handles shutdown for CloseRequested by sending ShutdownComplete and returning an error.
    #[allow(clippy::too_many_arguments)]
    fn process_event(
        event: PlatformEvent,
        term_emulator: &mut TerminalEmulator,
        platform_state: &PlatformState,
        pending_emulator_actions: &mut Vec<EmulatorAction>,
        frame_requested: &mut bool,
        display_action_tx: &Sender<PlatformAction>,
    ) -> Result<()> {
        match event {
            PlatformEvent::RequestFrame => {
                debug!("OrchestratorActor: Received RequestFrame from Vsync");
                *frame_requested = true;
            }
            PlatformEvent::IOEvent { commands: ansi_commands } => {
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
            }
            PlatformEvent::BackendEvent(backend_event) => {
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

                let emulator_input =
                    Self::process_backend_event(backend_event, platform_state, term_emulator)?;

                if let Some(input) = emulator_input {
                    if let Some(action) = term_emulator.interpret_input(input) {
                        pending_emulator_actions.push(action);
                    }
                }

                *frame_requested = true;
            }
        }
        Ok(())
    }

    /// Process a BackendEvent and return the corresponding EmulatorInput.
    fn process_backend_event(
        backend_event: BackendEvent,
        platform_state: &PlatformState,
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
            } => {
                if platform_state.font_cell_width_px > 0 && platform_state.font_cell_height_px > 0 {
                    let cols =
                        (width_px as usize / platform_state.font_cell_width_px.max(1)).max(1);
                    let rows =
                        (height_px as usize / platform_state.font_cell_height_px.max(1)).max(1);
                    info!("OrchestratorActor: Resizing to {}x{} cells", cols, rows);

                    Ok(Some(EmulatorInput::Control(ControlEvent::Resize {
                        cols,
                        rows,
                    })))
                } else {
                    warn!("OrchestratorActor: Font dimensions are zero, cannot process resize");
                    Ok(None)
                }
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
                modifiers: _,
            } => {
                if platform_state.font_cell_width_px > 0 && platform_state.font_cell_height_px > 0 {
                    let cell_x =
                        (x as u32 / platform_state.font_cell_width_px.max(1) as u32) as usize;
                    let cell_y =
                        (y as u32 / platform_state.font_cell_height_px.max(1) as u32) as usize;

                    let input = match button {
                        MouseButton::Left => {
                            Some(EmulatorInput::User(UserInputAction::StartSelection {
                                x: cell_x,
                                y: cell_y,
                            }))
                        }
                        MouseButton::Middle => {
                            Some(EmulatorInput::User(UserInputAction::RequestPrimaryPaste))
                        }
                        _ => None,
                    };
                    Ok(input)
                } else {
                    warn!(
                        "OrchestratorActor: Font dimensions are zero, cannot process mouse press"
                    );
                    Ok(None)
                }
            }
            BackendEvent::MouseButtonRelease {
                button,
                x: _x,
                y: _y,
                modifiers: _,
            } => {
                if platform_state.font_cell_width_px > 0 && platform_state.font_cell_height_px > 0 {
                    if button == MouseButton::Left {
                        Ok(Some(EmulatorInput::User(
                            UserInputAction::ApplySelectionClear,
                        )))
                    } else {
                        Ok(None)
                    }
                } else {
                    warn!(
                        "OrchestratorActor: Font dimensions are zero, cannot process mouse release"
                    );
                    Ok(None)
                }
            }
            BackendEvent::MouseMove { x, y, modifiers: _ } => {
                if platform_state.font_cell_width_px > 0 && platform_state.font_cell_height_px > 0 {
                    let cell_x =
                        (x as u32 / platform_state.font_cell_width_px.max(1) as u32) as usize;
                    let cell_y =
                        (y as u32 / platform_state.font_cell_height_px.max(1) as u32) as usize;
                    Ok(Some(EmulatorInput::User(
                        UserInputAction::ExtendSelection {
                            x: cell_x,
                            y: cell_y,
                        },
                    )))
                } else {
                    warn!("OrchestratorActor: Font dimensions are zero, cannot process mouse move");
                    Ok(None)
                }
            }
            BackendEvent::FocusGained => {
                Ok(Some(EmulatorInput::User(UserInputAction::FocusGained)))
            }
            BackendEvent::FocusLost => Ok(Some(EmulatorInput::User(UserInputAction::FocusLost))),
            BackendEvent::PasteData { text } => {
                Ok(Some(EmulatorInput::User(UserInputAction::PasteText(text))))
            }
        }
    }

    /// Handle an EmulatorAction by sending it to the appropriate actor.
    fn handle_emulator_action(
        action: EmulatorAction,
        display_tx: &Sender<PlatformAction>,
        pty_tx: &Sender<PlatformAction>,
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
