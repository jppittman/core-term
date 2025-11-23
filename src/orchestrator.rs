// src/orchestrator.rs
//! The main application orchestrator.
//!
//! This module contains the `AppOrchestrator`, which drives the main event loop
//! of the terminal application. It is responsible for coordinating the other
//! major components: `platform`, `term_emulator`, `ansi_parser`, and `renderer`.

pub mod actor;
pub mod orchestrator_actor;

use anyhow::{Context, Result};
use log::{debug, info, trace, warn};

use crate::ansi::{AnsiCommand, AnsiParser, AnsiProcessor};
use crate::config::Config;
use crate::keys;
use crate::platform::actions::PlatformAction;
use crate::platform::backends::{BackendEvent, MouseButton};
use crate::platform::{platform_trait::Platform, PlatformEvent};
use crate::renderer::Renderer;
use crate::term::{ControlEvent, EmulatorAction, EmulatorInput, TerminalEmulator, UserInputAction};

/// Represents the status of the application after an event cycle.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum OrchestratorStatus {
    /// The application should continue running.
    Running,
    /// The application should shut down gracefully.
    Shutdown,
}

/// The core application state machine.
///
/// `AppOrchestrator` connects all the components of the terminal. It polls the
/// `Platform` for events, forwards them to the `TerminalEmulator` for state
/// updates, and then uses the `Renderer` to generate drawing commands, which
/// are sent back to the `Platform`.
pub struct AppOrchestrator<'a> {
    platform: &'a mut dyn Platform,
    term_emulator: &'a mut TerminalEmulator,
    ansi_parser: &'a mut AnsiProcessor,
    renderer: Renderer,
    pending_emulator_actions: Vec<EmulatorAction>,
}

impl<'a> AppOrchestrator<'a> {
    /// Creates a new `AppOrchestrator`.
    ///
    /// # Arguments
    ///
    /// * `platform` - A mutable reference to the platform driver.
    /// * `term_emulator` - A mutable reference to the terminal emulator.
    /// * `ansi_parser` - A mutable reference to the ANSI parser.
    /// * `renderer` - An instance of the renderer.
    pub fn new(
        platform: &'a mut dyn Platform,
        term_emulator: &'a mut TerminalEmulator,
        ansi_parser: &'a mut AnsiProcessor,
        renderer: Renderer,
    ) -> Self {
        info!("AppOrchestrator: Initializing...");
        AppOrchestrator {
            platform,
            term_emulator,
            ansi_parser,
            renderer,
            pending_emulator_actions: Vec::new(),
        }
    }

    fn handle_emulator_action_immediately(&mut self, action: EmulatorAction) -> Result<()> {
        debug!(
            "AppOrchestrator: Handling EmulatorAction immediately: {:?}",
            action
        );
        match action {
            EmulatorAction::WritePty(data) => self
                .platform
                .dispatch_actions(vec![PlatformAction::Write(data)])
                .context("Failed to dispatch PTY write action"),
            EmulatorAction::SetTitle(title) => self
                .platform
                .dispatch_actions(vec![PlatformAction::SetTitle(title)])
                .context("Failed to dispatch UI set title action"),
            EmulatorAction::RingBell => self
                .platform
                .dispatch_actions(vec![PlatformAction::RingBell])
                .context("Failed to dispatch UI ring bell action"),
            EmulatorAction::CopyToClipboard(text) => self
                .platform
                .dispatch_actions(vec![PlatformAction::CopyToClipboard(text)])
                .context("Failed to dispatch UI copy to clipboard action"),
            EmulatorAction::SetCursorVisibility(visible) => self
                .platform
                .dispatch_actions(vec![PlatformAction::SetCursorVisibility(visible)])
                .context("Failed to dispatch UI set cursor visibility action"),
            EmulatorAction::RequestRedraw => {
                trace!("EmulatorAction::RequestRedraw received.");
                Ok(())
            }
            EmulatorAction::RequestClipboardContent => self
                .platform
                .dispatch_actions(vec![PlatformAction::RequestPaste]),
        }
    }

    /// Executes a single cycle of the main event loop.
    ///
    /// This method performs the following steps:
    /// 1. Polls the platform for new I/O or UI events.
    /// 2. Processes those events, updating the terminal state. This may
    ///    generate `EmulatorAction`s (e.g., to write to the PTY).
    /// 3. Handles any generated `EmulatorAction`s.
    /// 4. Creates a `RenderSnapshot` of the terminal's new state.
    /// 5. Generates `RenderCommand`s from the snapshot and dispatches them
    ///    to the platform for drawing.
    ///
    /// # Returns
    ///
    /// A `Result` containing the `OrchestratorStatus`, which indicates whether
    /// the application should continue running or shut down.
    pub fn process_event_cycle(&mut self) -> Result<OrchestratorStatus> {
        trace!("AppOrchestrator: Starting new event cycle.");

        self.pending_emulator_actions.clear();

        let events = self
            .platform
            .poll_events()
            .context("Failed to poll platform events")?;

        debug!("processing {} events", events.len());
        for platform_event in events {
            match platform_event {
                PlatformEvent::RequestFrame => {
                    // Handled by actor-based architecture, not used in single-threaded orchestrator
                }
                PlatformEvent::IOEvent { commands: ansi_commands } => {
                    debug!(
                        "AppOrchestrator: Received {} ANSI commands from PTY.",
                        ansi_commands.len()
                    );
                    for command in ansi_commands {
                        if let Some(action) = self
                            .term_emulator
                            .interpret_input(EmulatorInput::Ansi(command))
                        {
                            self.pending_emulator_actions.push(action);
                        }
                    }
                }
                PlatformEvent::BackendEvent(backend_event) => {
                    debug!(
                        "AppOrchestrator: Received BackendEvent: {:?}",
                        backend_event
                    );
                    let mut emulator_input_to_process: Option<EmulatorInput> = None;

                    match backend_event {
                        BackendEvent::CloseRequested => {
                            info!("AppOrchestrator: CloseRequested event received. Signaling shutdown.");
                            return Ok(OrchestratorStatus::Shutdown);
                        }
                        BackendEvent::Resize {
                            width_px,
                            height_px,
                        } => {
                            let platform_state = self.platform.get_current_platform_state();
                            if platform_state.font_cell_width_px > 0
                                && platform_state.font_cell_height_px > 0
                            {
                                let cols = (width_px as usize
                                    / platform_state.font_cell_width_px.max(1))
                                .max(1);
                                let rows = (height_px as usize
                                    / platform_state.font_cell_height_px.max(1))
                                .max(1);
                                info!(
                                    "AppOrchestrator: Resizing to {}x{} cells ({}x{} px, char_size: {}x{})",
                                    cols,
                                    rows,
                                    width_px,
                                    height_px,
                                    platform_state.font_cell_width_px,
                                    platform_state.font_cell_height_px
                                );
                                self.platform
                                    .dispatch_actions(vec![PlatformAction::ResizePty {
                                        cols: cols as u16,
                                        rows: rows as u16,
                                    }])
                                    .context("Failed to dispatch PTY resize action")?;
                                emulator_input_to_process =
                                    Some(EmulatorInput::Control(ControlEvent::Resize {
                                        cols,
                                        rows,
                                    }));
                            } else {
                                warn!("AppOrchestrator: Font dimensions are zero, cannot process resize.");
                            }
                        }
                        BackendEvent::Key {
                            symbol,
                            modifiers,
                            text,
                        } => {
                            debug!("Key: {:?} + {:?}\nText: {:?}", modifiers, symbol, text);
                            let key_input_action = keys::map_key_event_to_action(
                                symbol,
                                modifiers,
                                &crate::config::CONFIG,
                            )
                            .unwrap_or(UserInputAction::KeyInput {
                                symbol,
                                modifiers,
                                text: if text.is_empty() { None } else { Some(text) },
                            });
                            emulator_input_to_process = Some(EmulatorInput::User(key_input_action));
                        }
                        BackendEvent::MouseButtonPress {
                            button,
                            x,
                            y,
                            modifiers: _,
                        } => {
                            let platform_state = self.platform.get_current_platform_state();
                            if platform_state.font_cell_width_px > 0
                                && platform_state.font_cell_height_px > 0
                            {
                                let cell_x = (x as u32
                                    / platform_state.font_cell_width_px.max(1) as u32)
                                    as usize;
                                let cell_y = (y as u32
                                    / platform_state.font_cell_height_px.max(1) as u32)
                                    as usize;

                                match button {
                                    MouseButton::Left => {
                                        emulator_input_to_process = Some(EmulatorInput::User(
                                            UserInputAction::StartSelection {
                                                x: cell_x,
                                                y: cell_y,
                                            },
                                        ));
                                    }
                                    MouseButton::Middle => {
                                        emulator_input_to_process = Some(EmulatorInput::User(
                                            UserInputAction::RequestPrimaryPaste,
                                        ));
                                    }
                                    _ => trace!("Unhandled mouse button press: {:?}", button),
                                }
                            } else {
                                warn!("AppOrchestrator: Font dimensions are zero, cannot process mouse press.");
                            }
                        }
                        BackendEvent::MouseButtonRelease {
                            button,
                            x: _x,
                            y: _y,
                            modifiers: _,
                        } => {
                            let platform_state = self.platform.get_current_platform_state();
                            if platform_state.font_cell_width_px > 0
                                && platform_state.font_cell_height_px > 0
                            {
                                if button == MouseButton::Left {
                                    emulator_input_to_process = Some(EmulatorInput::User(
                                        UserInputAction::ApplySelectionClear,
                                    ));
                                }
                            } else {
                                warn!("AppOrchestrator: Font dimensions are zero, cannot process mouse release.");
                            }
                        }
                        BackendEvent::MouseMove { x, y, modifiers: _ } => {
                            let platform_state = self.platform.get_current_platform_state();
                            if platform_state.font_cell_width_px > 0
                                && platform_state.font_cell_height_px > 0
                            {
                                let cell_x = (x as u32
                                    / platform_state.font_cell_width_px.max(1) as u32)
                                    as usize;
                                let cell_y = (y as u32
                                    / platform_state.font_cell_height_px.max(1) as u32)
                                    as usize;
                                emulator_input_to_process =
                                    Some(EmulatorInput::User(UserInputAction::ExtendSelection {
                                        x: cell_x,
                                        y: cell_y,
                                    }));
                            } else {
                                warn!(
                                    "AppOrchestrator: Font dimensions are zero, cannot process mouse move."
                                );
                            }
                        }
                        BackendEvent::FocusGained => {
                            emulator_input_to_process =
                                Some(EmulatorInput::User(UserInputAction::FocusGained));
                        }
                        BackendEvent::FocusLost => {
                            emulator_input_to_process =
                                Some(EmulatorInput::User(UserInputAction::FocusLost));
                        }
                        BackendEvent::PasteData { text } => {
                            emulator_input_to_process =
                                Some(EmulatorInput::User(UserInputAction::PasteText(text)));
                        }
                    }

                    if let Some(input) = emulator_input_to_process {
                        if let Some(action) = self.term_emulator.interpret_input(input) {
                            self.pending_emulator_actions.push(action);
                        }
                    }
                } // End of BackendEvent processing
            } // End of match platform_event
        } // End of for platform_event in events

        for action in self.pending_emulator_actions.drain(..).collect::<Vec<_>>() {
            self.handle_emulator_action_immediately(action)?;
        }

        let Some(snapshot) = self.term_emulator.get_render_snapshot() else {
            return Ok(OrchestratorStatus::Running);
        };

        // SOT: Extract authoritative dimensions from snapshot
        let (cols, rows) = snapshot.dimensions;

        let config = Config::default();
        let platform_state = self.platform.get_current_platform_state();
        let render_commands =
            self.renderer
                .prepare_render_commands(&snapshot, &config, &platform_state);

        if !render_commands.is_empty() {
            debug!(
                "AppOrchestrator: Sending {} render commands ({}x{} grid) to UI.",
                render_commands.len(),
                cols,
                rows
            );
            self.platform
                .dispatch_actions(vec![PlatformAction::Render {
                    commands: render_commands,
                    cols,
                    rows,
                }])
                .context("Failed to dispatch UI render action")?;
        }

        Ok(OrchestratorStatus::Running)
    }
}

#[cfg(test)]
mod tests;
