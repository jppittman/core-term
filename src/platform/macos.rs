use crate::config::Config;
use crate::orchestrator::orchestrator_actor::OrchestratorActor;
use crate::platform::actions::PlatformAction;
use crate::platform::backends::{cocoa::CocoaDriver, Driver, PlatformState, RenderCommand};
use crate::platform::os::event_monitor_actor::EventMonitorActor;
use crate::platform::os::pty::{NixPty, PtyConfig};
use crate::platform::os::vsync_actor::VsyncActor;
use crate::platform::platform_trait::Platform;
use crate::platform::PlatformEvent;
use crate::rasterizer::{compile_into_buffer, SoftwareRasterizer};
use crate::renderer::Renderer;
use crate::term::snapshot::RenderSnapshot;
use crate::term::TerminalEmulator;
use anyhow::{Context, Result};
use log::*;
use std::sync::mpsc::{self, sync_channel, Receiver, Sender, SyncSender, TryRecvError};

pub struct MacosPlatform {
    driver: CocoaDriver,
    event_monitor_actor: EventMonitorActor,
    orchestrator_actor: OrchestratorActor,
    vsync_actor: VsyncActor,
    rasterizer: SoftwareRasterizer,
    renderer: Renderer,
    config: Config,
    terminal_event_tx: Sender<PlatformEvent>,
    snapshot_rx: Receiver<RenderSnapshot>,
    snapshot_pool_tx: SyncSender<RenderSnapshot>,
    display_action_rx: Receiver<PlatformAction>,
    pty_action_tx: Sender<PlatformAction>,
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

        let driver = CocoaDriver::new().context("Failed to create CocoaDriver")?;
        let initial_platform_state = driver.get_platform_state();
        info!("Initial platform state: {:?}", initial_platform_state);

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

        let (terminal_event_tx, terminal_event_rx) = mpsc::channel();

        let shell_args_refs: Vec<&str> = shell_args.iter().map(String::as_str).collect();
        let pty_config = PtyConfig {
            command_executable: &shell_command,
            args: &shell_args_refs,
            initial_cols: initial_pty_cols,
            initial_rows: initial_pty_rows,
        };

        let pty = NixPty::spawn_with_config(&pty_config).context("Failed to create NixPty")?;
        info!("Spawned PTY");

        let (pty_action_tx, event_monitor_actor) =
            EventMonitorActor::spawn(pty, terminal_event_tx.clone())
                .context("Failed to spawn EventMonitorActor")?;
        info!("EventMonitorActor spawned successfully");

        let (display_action_tx, display_action_rx) = mpsc::channel();

        let (snapshot_tx, snapshot_rx) = sync_channel(1);
        let (snapshot_pool_tx, snapshot_pool_rx) = sync_channel(1);

        // Create initial snapshot and send to pool
        let initial_snapshot = RenderSnapshot {
            dimensions: (term_cols, term_rows),
            lines: Vec::new(),
            cursor_state: None,
            selection: crate::term::snapshot::Selection::default(),
        };
        snapshot_pool_tx
            .send(initial_snapshot)
            .context("Failed to send initial snapshot to pool")?;

        let term_emulator = TerminalEmulator::new(term_cols, term_rows);

        let orchestrator_actor = OrchestratorActor::spawn(
            term_emulator,
            terminal_event_rx,
            snapshot_tx,
            snapshot_pool_rx,
            display_action_tx.clone(),
            pty_action_tx.clone(),
            initial_platform_state.clone(),
        )
        .context("Failed to spawn OrchestratorActor")?;
        info!("OrchestratorActor spawned successfully");

        let target_fps = crate::config::CONFIG.performance.target_fps;
        let vsync_actor = VsyncActor::spawn(terminal_event_tx.clone(), target_fps)
            .context("Failed to spawn VsyncActor")?;

        let rasterizer = SoftwareRasterizer::new(
            initial_platform_state.font_cell_width_px,
            initial_platform_state.font_cell_height_px,
        );
        let renderer = Renderer::new();
        let config = Config::default();

        Ok((
            Self {
                driver,
                event_monitor_actor,
                orchestrator_actor,
                vsync_actor,
                rasterizer,
                renderer,
                config,
                terminal_event_tx,
                snapshot_rx,
                snapshot_pool_tx,
                display_action_rx,
                pty_action_tx,
            },
            initial_platform_state,
        ))
    }

    fn poll_events(&mut self) -> Result<Vec<PlatformEvent>> {
        warn!("MacosPlatform::poll_events() called but not used in actor architecture");
        Ok(vec![])
    }

    fn dispatch_actions(&mut self, actions: Vec<PlatformAction>) -> Result<()> {
        warn!("MacosPlatform::dispatch_actions() called but not used in actor architecture");
        Ok(())
    }

    fn get_current_platform_state(&self) -> PlatformState {
        self.driver.get_platform_state()
    }

    fn run(mut self) -> Result<()> {
        info!("MacosPlatform::run() - Starting main event loop");

        let mut shutdown_complete = false;
        loop {
            let cocoa_events = self
                .driver
                .process_events()
                .context("CocoaDriver event processing failed")?;

            for backend_event in cocoa_events {
                let platform_event = PlatformEvent::BackendEvent(backend_event);
                if let Err(e) = self.terminal_event_tx.send(platform_event) {
                    warn!("Failed to send Cocoa event to Terminal: {}", e);
                    info!("Terminal event channel closed, shutting down");
                    break;
                }
            }

            // Process at most one snapshot per iteration (no draining loop)
            if let Ok(snapshot) = self.snapshot_rx.try_recv() {
                // SOT: Extract authoritative dimensions from snapshot
                let (cols, rows) = snapshot.dimensions;
                debug!("MacosPlatform: Processing snapshot ({}x{} grid)", cols, rows);

                let platform_state = self.driver.get_platform_state();

                // TODO: Enforce size contract - ensure framebuffer matches snapshot dimensions
                // self.driver.ensure_size(cols, rows)?;

                let mut render_commands = self.renderer.prepare_render_commands(
                    &snapshot,
                    &self.config,
                    &platform_state,
                );

                render_commands.push(RenderCommand::PresentFrame);

                let (width_px, height_px) = self.driver.get_framebuffer_size();
                let framebuffer = self.driver.get_framebuffer_mut();

                let driver_commands = compile_into_buffer(
                    &mut self.rasterizer,
                    render_commands,
                    framebuffer,
                    width_px,
                    height_px,
                    platform_state.font_cell_width_px,
                    platform_state.font_cell_height_px,
                );

                trace!(
                    "MacosPlatform: Compiled into {} driver commands",
                    driver_commands.len()
                );

                self.driver
                    .execute_driver_commands(driver_commands)
                    .context("Failed to execute driver commands")?;

                trace!("MacosPlatform: Driver commands executed successfully");

                // Return snapshot to pool for reuse
                self.snapshot_pool_tx
                    .send(snapshot)
                    .context("Failed to return snapshot to pool")?;

                debug!("MacosPlatform: Returned snapshot to pool");
            }

            loop {
                match self.display_action_rx.try_recv() {
                    Ok(action) => {
                        match action {
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
                                use crate::platform::backends::CursorVisibility;
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
                            PlatformAction::ShutdownComplete => {
                                info!("MacosPlatform: Received ShutdownComplete - exiting event loop");
                                shutdown_complete = true;
                            }
                            PlatformAction::Write(_) | PlatformAction::ResizePty { .. } => {
                                warn!("MacosPlatform received PTY action - this should go to PTY thread!");
                            }
                            PlatformAction::Render { .. } => {
                                warn!("MacosPlatform received Render action - snapshots should come via snapshot channel!");
                            }
                        }
                    }
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => {
                        info!("Display action channel closed, shutting down");
                        shutdown_complete = true;
                        break;
                    }
                }
            }

            if shutdown_complete {
                break;
            }

            std::thread::sleep(std::time::Duration::from_millis(1));
        }

        info!("MacosPlatform::run() - Exiting normally, Drop will handle cleanup");
        Ok(())
    }

    fn cleanup(&mut self) -> Result<()> {
        info!("MacosPlatform: cleanup() called");
        self.driver.cleanup()
    }
}

impl Drop for MacosPlatform {
    fn drop(&mut self) {
        info!("MacosPlatform::drop() - Dropping, calling cleanup");
        let _ = self.driver.cleanup();
        info!("MacosPlatform::drop() - Drop complete");
    }
}
