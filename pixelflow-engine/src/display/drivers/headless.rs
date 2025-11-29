#![cfg(use_headless_display)]

//! Headless mock display driver implementation.
//!
//! Driver struct is just cmd_tx - trivially Clone.
//! run() reads Configure, runs a simple event loop.

use crate::channel::{DriverCommand, EngineCommand, EngineSender};
use crate::display::driver::DisplayDriver;
use crate::display::messages::{DisplayEvent, RenderSnapshot};
use anyhow::{anyhow, Result};
use log::info;
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};

// --- Run State (only original driver has this) ---
struct RunState {
    cmd_rx: Receiver<DriverCommand>,
    engine_tx: EngineSender,
}

// --- Display Driver ---

/// Headless display driver for testing.
///
/// Clone to get a handle for sending commands. Only the original can run().
pub struct HeadlessDisplayDriver {
    cmd_tx: SyncSender<DriverCommand>,
    /// Only present on original, None on clones
    run_state: Option<RunState>,
}

impl Clone for HeadlessDisplayDriver {
    fn clone(&self) -> Self {
        Self {
            cmd_tx: self.cmd_tx.clone(),
            run_state: None, // Clones can't run
        }
    }
}

impl DisplayDriver for HeadlessDisplayDriver {
    fn new(engine_tx: EngineSender) -> Result<Self> {
        let (cmd_tx, cmd_rx) = sync_channel(16);

        Ok(Self {
            cmd_tx,
            run_state: Some(RunState { cmd_rx, engine_tx }),
        })
    }

    fn send(&self, cmd: DriverCommand) -> Result<()> {
        self.cmd_tx.send(cmd)?;
        Ok(())
    }

    fn run(&self) -> Result<()> {
        let run_state = self
            .run_state
            .as_ref()
            .ok_or_else(|| anyhow!("Only original driver can run (this is a clone)"))?;

        run_event_loop(&run_state.cmd_rx, &run_state.engine_tx)
    }
}

// --- Event Loop ---

fn run_event_loop(cmd_rx: &Receiver<DriverCommand>, engine_tx: &EngineSender) -> Result<()> {
    // 1. Read Configure command first
    let config = match cmd_rx.recv()? {
        DriverCommand::Configure(c) => c,
        other => return Err(anyhow!("Expected Configure, got {:?}", other)),
    };

    info!("Headless: Creating resources with config");

    let width_px = (config.initial_cols * config.cell_width_px) as u32;
    let height_px = (config.initial_rows * config.cell_height_px) as u32;

    info!("Headless: Virtual window {}x{} px", width_px, height_px);

    // Send initial resize
    let _ = engine_tx.send(EngineCommand::DisplayEvent(DisplayEvent::Resize {
        width_px,
        height_px,
    }));

    // 2. Run simple event loop
    loop {
        match cmd_rx.recv()? {
            DriverCommand::Configure(_) => {
                // Already configured, ignore
            }
            DriverCommand::Shutdown => {
                info!("Headless: Shutdown command received");
                return Ok(());
            }
            DriverCommand::Present(snapshot) => {
                // Just return the framebuffer
                let _ = engine_tx.send(EngineCommand::PresentComplete(snapshot));
            }
            DriverCommand::SetTitle(title) => {
                info!("Headless: SetTitle '{}'", title);
                let _ = engine_tx.send(EngineCommand::DriverAck);
            }
            DriverCommand::CopyToClipboard(text) => {
                info!("Headless: CopyToClipboard '{}'", text);
                let _ = engine_tx.send(EngineCommand::DriverAck);
            }
            DriverCommand::RequestPaste => {
                info!("Headless: RequestPaste");
                // In headless mode, just acknowledge - no actual paste data
                let _ = engine_tx.send(EngineCommand::DriverAck);
            }
            DriverCommand::Bell => {
                info!("Headless: Bell");
                let _ = engine_tx.send(EngineCommand::DriverAck);
            }
        }
    }
}
