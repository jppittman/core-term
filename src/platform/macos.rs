use crate::ansi::AnsiProcessor;
use crate::orchestrator::actor::OrchestratorActor;
use crate::platform::actions::PlatformAction;
use crate::platform::backends::{
    cocoa::CocoaDriver, CursorVisibility, Driver, PlatformState,
};
use crate::platform::os::event_monitor_actor::EventMonitorActor;
use crate::platform::os::pty::{NixPty, PtyConfig};
use crate::platform::platform_trait::Platform;
use crate::platform::PlatformEvent;
use crate::renderer::Renderer;
use crate::term::TerminalEmulator;
use anyhow::{Context, Result};
use log::*;
use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};

pub struct MacosPlatform {
    driver: CocoaDriver,
    event_monitor_actor: EventMonitorActor,
    orchestrator_actor: OrchestratorActor,
    // Channels to Orchestrator
    orchestrator_event_tx: Sender<PlatformEvent>, // Display sends events here
    orchestrator_display_action_rx: Receiver<PlatformAction>, // Display receives actions from Orchestrator
    // Channel to PTY
    pty_action_tx: Sender<PlatformAction>, // Forward PTY actions from Orchestrator to PTY
}

impl Platform for MacosPlatform {
    fn new(
        initial_pty_cols: u16,
        initial_pty_rows: u16,
        shell_command: String,
        shell_args: Vec<String>,
    ) -> Result<(Self, PlatformState)>
    where
        Self: Sized,
    {
        info!(
            "MacosPlatform::new() - Initializing with PTY size {}x{}",
            initial_pty_cols, initial_pty_rows
        );
        info!("Shell command: '{}', args: {:?}", shell_command, shell_args);

        // Step 1: Create CocoaDriver
        let driver = CocoaDriver::new().context("Failed to create CocoaDriver")?;
        let initial_platform_state = driver.get_platform_state();
        info!("Initial platform state: {:?}", initial_platform_state);

        // Step 2: Calculate terminal dimensions
        let term_cols = (initial_platform_state.display_width_px as usize
            / initial_platform_state.font_cell_width_px.max(1))
        .max(1);
        let term_rows = (initial_platform_state.display_height_px as usize
            / initial_platform_state.font_cell_height_px.max(1))
        .max(1);
        info!(
            "Calculated terminal dimensions: {}x{} cells",
            term_cols, term_rows
        );

        // Step 3: Create unified event channel for Orchestrator
        let (orchestrator_event_tx, orchestrator_event_rx) = mpsc::channel();

        // Step 4: Create PTY and EventMonitorActor
        let shell_args_refs: Vec<&str> = shell_args.iter().map(String::as_str).collect();
        let pty_config = PtyConfig {
            command_executable: &shell_command,
            args: &shell_args_refs,
            initial_cols: initial_pty_cols,
            initial_rows: initial_pty_rows,
        };

        let pty = NixPty::spawn_with_config(&pty_config).context("Failed to create NixPty")?;
        info!("Spawned PTY");

        // PTY sends events to the unified Orchestrator event channel
        let (pty_action_tx, event_monitor_actor) =
            EventMonitorActor::spawn(pty, orchestrator_event_tx.clone())
                .context("Failed to spawn EventMonitorActor")?;
        info!("EventMonitorActor spawned successfully");

        // Step 5: Create channels for Orchestrator <-> Display communication
        let (orchestrator_display_action_tx, orchestrator_display_action_rx) = mpsc::channel();

        // Step 6: Create core components for Orchestrator
        let term_emulator = TerminalEmulator::new(term_cols, term_rows);
        let ansi_parser = AnsiProcessor::new();
        let renderer = Renderer::new();

        // Step 7: Spawn Orchestrator Actor
        let orchestrator_actor = OrchestratorActor::spawn(
            term_emulator,
            ansi_parser,
            renderer,
            orchestrator_event_rx,
            orchestrator_display_action_tx,
            pty_action_tx.clone(),
            initial_platform_state.clone(),
        )
        .context("Failed to spawn OrchestratorActor")?;
        info!("OrchestratorActor spawned successfully");

        Ok((
            Self {
                driver,
                event_monitor_actor,
                orchestrator_actor,
                orchestrator_event_tx,
                orchestrator_display_action_rx,
                pty_action_tx,
            },
            initial_platform_state,
        ))
    }

    fn poll_events(&mut self) -> Result<Vec<PlatformEvent>> {
        // This method is no longer used in the three-thread architecture
        // The Display thread forwards events directly via channels in run()
        warn!("MacosPlatform::poll_events() called but not used in three-thread architecture");
        Ok(vec![])
    }

    fn dispatch_actions(&mut self, actions: Vec<PlatformAction>) -> Result<()> {
        // This method is no longer used in the three-thread architecture
        // Actions are dispatched directly in run() loop
        warn!("MacosPlatform::dispatch_actions() called but not used in three-thread architecture");
        Ok(())
    }

    fn get_current_platform_state(&self) -> PlatformState {
        self.driver.get_platform_state()
    }

    fn run(mut self) -> Result<()> {
        info!("MacosPlatform::run() - Starting main event loop");

        // Main event loop: process Cocoa events and handle Orchestrator actions
        loop {
            // Step 1: Process Cocoa UI events (non-blocking)
            let cocoa_events = self
                .driver
                .process_events()
                .context("CocoaDriver event processing failed")?;

            // Step 2: Forward Cocoa events to Orchestrator
            for backend_event in cocoa_events {
                let platform_event = PlatformEvent::BackendEvent(backend_event);
                if let Err(e) = self.orchestrator_event_tx.send(platform_event) {
                    warn!("Failed to send Cocoa event to Orchestrator: {}", e);
                    // Orchestrator channel closed - time to shut down
                    info!("Orchestrator channel closed, shutting down");
                    return Ok(());
                }
            }

            // Step 3: Process actions from Orchestrator (non-blocking)
            loop {
                match self.orchestrator_display_action_rx.try_recv() {
                    Ok(action) => {
                        self.handle_display_action(action)?;
                    }
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => {
                        info!("Orchestrator action channel closed, shutting down");
                        return Ok(());
                    }
                }
            }

            // Small sleep to avoid spinning (Cocoa will handle timing via NSApp.run() later)
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    }

    fn cleanup(&mut self) -> Result<()> {
        info!("MacosPlatform: cleanup() called");
        self.driver.cleanup()
    }
}

impl MacosPlatform {
    /// Handle a display action from the Orchestrator.
    fn handle_display_action(&mut self, action: PlatformAction) -> Result<()> {
        match action {
            PlatformAction::Render(render_commands) => {
                use crate::rasterizer::compile_into_buffer;

                let (width_px, height_px) = self.driver.get_framebuffer_size();
                let platform_state = self.driver.get_platform_state();
                let framebuffer = self.driver.get_framebuffer_mut();

                let driver_commands = compile_into_buffer(
                    render_commands,
                    framebuffer,
                    width_px,
                    height_px,
                    platform_state.font_cell_width_px,
                    platform_state.font_cell_height_px,
                );

                self.driver
                    .execute_driver_commands(driver_commands)
                    .context("Failed to execute driver commands")?;
            }
            PlatformAction::SetTitle(title) => {
                self.driver.set_title(&title);
            }
            PlatformAction::RingBell => {
                self.driver.bell();
            }
            PlatformAction::CopyToClipboard(text) => {
                self.driver.own_selection(0, text);
            }
            PlatformAction::SetCursorVisibility(visible) => {
                let visibility = if visible {
                    CursorVisibility::Shown
                } else {
                    CursorVisibility::Hidden
                };
                self.driver.set_cursor_visibility(visibility);
            }
            PlatformAction::RequestPaste => {
                self.driver.request_selection_data(0, 0);
            }
            // PTY actions should go to PTY thread, not here
            PlatformAction::Write(_) | PlatformAction::ResizePty { .. } => {
                warn!("MacosPlatform received PTY action - this should go to PTY thread!");
            }
        }
        Ok(())
    }
}

impl Drop for MacosPlatform {
    fn drop(&mut self) {
        debug!("MacosPlatform dropped");
    }
}
