//! Orchestrator Actor - runs in a background thread and coordinates terminal logic.
//!
//! The OrchestratorActor owns all application state (TerminalEmulator, AnsiProcessor, Renderer)
//! and processes events from both the Display driver and PTY I/O thread. It runs in a dedicated
//! background thread and communicates via message-passing channels.

use crate::ansi::{AnsiCommand, AnsiParser, AnsiProcessor};
use crate::config::Config;
use crate::keys;
use crate::platform::actions::PlatformAction;
use crate::platform::backends::{BackendEvent, MouseButton, PlatformState, RenderCommand};
use crate::platform::PlatformEvent;
use crate::renderer::Renderer;
use crate::term::{ControlEvent, EmulatorAction, EmulatorInput, TerminalEmulator, UserInputAction};
use anyhow::{Context, Result};
use log::*;
use std::sync::mpsc::{Receiver, Sender};
use std::thread::{self, JoinHandle};

/// Orchestrator actor that runs in a background thread.
///
/// This actor receives events from both the Display driver (UI events) and the PTY thread
/// (shell output), processes them through the terminal emulator, and sends back render
/// commands to the display and write commands to the PTY.
pub struct OrchestratorActor {
    thread_handle: Option<JoinHandle<()>>,
}

impl OrchestratorActor {
    /// Spawns the Orchestrator actor in a background thread.
    ///
    /// # Arguments
    ///
    /// * `term_emulator` - The terminal emulator (takes ownership)
    /// * `ansi_parser` - The ANSI parser (takes ownership)
    /// * `renderer` - The renderer (takes ownership)
    /// * `event_rx` - Channel to receive PlatformEvents (from Display and PTY)
    /// * `display_action_tx` - Channel to send PlatformActions to Display
    /// * `pty_action_tx` - Channel to send PlatformActions to PTY
    /// * `initial_platform_state` - Initial platform state for rendering
    ///
    /// # Returns
    ///
    /// Returns `Self` (handle to the actor for cleanup)
    pub fn spawn(
        term_emulator: TerminalEmulator,
        ansi_parser: AnsiProcessor,
        renderer: Renderer,
        event_rx: Receiver<PlatformEvent>,
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
                    ansi_parser,
                    renderer,
                    event_rx,
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
    /// Blocks on `event_rx.recv()` waiting for events from Display or PTY,
    /// processes them, and sends actions back via channels.
    fn actor_thread_main(
        mut term_emulator: TerminalEmulator,
        mut ansi_parser: AnsiProcessor,
        renderer: Renderer,
        event_rx: Receiver<PlatformEvent>,
        display_action_tx: Sender<PlatformAction>,
        pty_action_tx: Sender<PlatformAction>,
        platform_state: PlatformState,
    ) -> Result<()> {
        debug!("OrchestratorActor: Starting event loop");

        let mut pending_emulator_actions = Vec::new();

        loop {
            // Block on event channel - wakes when Display or PTY sends event
            let event = match event_rx.recv() {
                Ok(event) => event,
                Err(_) => {
                    info!("OrchestratorActor: Event channel closed, shutting down");
                    return Ok(());
                }
            };

            pending_emulator_actions.clear();
            let mut should_present_frame = false;

            // Process the event
            match event {
                PlatformEvent::RequestFrame => {
                    // Vsync is requesting a frame presentation
                    debug!("OrchestratorActor: Received RequestFrame from Vsync");
                    should_present_frame = true;
                }
                PlatformEvent::IOEvent { data: pty_data } => {
                    debug!(
                        "OrchestratorActor: Received {} bytes from PTY",
                        pty_data.len()
                    );

                    // Parse ANSI commands
                    let ansi_commands: Vec<AnsiCommand> = ansi_parser.process_bytes(&pty_data);
                    if !ansi_commands.is_empty() {
                        debug!(
                            "OrchestratorActor: Parsed {} ANSI commands",
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
                }
                PlatformEvent::BackendEvent(backend_event) => {
                    debug!(
                        "OrchestratorActor: Received BackendEvent: {:?}",
                        backend_event
                    );

                    // Check for CloseRequested before processing
                    if let BackendEvent::CloseRequested = backend_event {
                        info!("OrchestratorActor: CloseRequested received - exiting event loop");
                        return Ok(()); // Exit the event loop, causing channels to close
                    }

                    let emulator_input = Self::process_backend_event(
                        backend_event,
                        &platform_state,
                        &mut term_emulator,
                    )?;

                    if let Some(input) = emulator_input {
                        if let Some(action) = term_emulator.interpret_input(input) {
                            pending_emulator_actions.push(action);
                        }
                    }

                    // User input should be immediately visible
                    should_present_frame = true;
                }
            }

            // Handle emulator actions
            for action in pending_emulator_actions.drain(..) {
                Self::handle_emulator_action(action, &display_action_tx, &pty_action_tx)?;
            }

            // Generate render commands
            let Some(snapshot) = term_emulator.get_render_snapshot() else {
                continue;
            };

            let config = Config::default();
            let mut render_commands =
                renderer.prepare_render_commands(&snapshot, &config, &platform_state);

            // Append PresentFrame only when vsync requests it (rate-limited to 60 FPS)
            if should_present_frame {
                render_commands.push(RenderCommand::PresentFrame);
            }

            if !render_commands.is_empty() {
                debug!(
                    "OrchestratorActor: Sending {} render commands",
                    render_commands.len()
                );
                display_action_tx
                    .send(PlatformAction::Render(render_commands))
                    .context("Failed to send render commands to Display")?;
            }
        }
    }

    /// Process a BackendEvent and return the corresponding EmulatorInput.
    ///
    /// Returns `Ok(None)` for events that don't produce emulator input.
    /// Returns `Err` only for fatal errors.
    ///
    /// Note: CloseRequested is handled in the main event loop before this function is called.
    fn process_backend_event(
        backend_event: BackendEvent,
        platform_state: &PlatformState,
        _term_emulator: &mut TerminalEmulator,
    ) -> Result<Option<EmulatorInput>> {
        match backend_event {
            BackendEvent::CloseRequested => {
                // This should never be reached - CloseRequested is handled in the main loop
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
