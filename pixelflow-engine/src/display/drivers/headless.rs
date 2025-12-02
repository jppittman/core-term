#![cfg(use_headless_display)]

//! Headless mock display driver implementation.
//!
//! Driver struct is just cmd_tx - trivially Clone.
//! run() reads CreateWindow, runs a simple event loop.

use crate::channel::{DriverCommand, EngineCommand, EngineSender};
use crate::display::driver::DisplayDriver;
use crate::display::messages::{DisplayEvent, WindowId};
use pixelflow_render::color::Rgba;
use pixelflow_render::Frame;
use anyhow::{anyhow, Result};
use log::info;
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};

// --- Run State (only original driver has this) ---
struct RunState {
    cmd_rx: Receiver<DriverCommand<Rgba>>,
    engine_tx: EngineSender<Rgba>,
}

// --- Display Driver ---

/// Headless display driver for testing.
///
/// Clone to get a handle for sending commands. Only the original can run().
pub struct HeadlessDisplayDriver {
    cmd_tx: SyncSender<DriverCommand<Rgba>>,
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
    type Pixel = Rgba;

    fn new(engine_tx: EngineSender<Rgba>) -> Result<Self> {
        let (cmd_tx, cmd_rx) = sync_channel(16);

        Ok(Self {
            cmd_tx,
            run_state: Some(RunState { cmd_rx, engine_tx }),
        })
    }

    fn send(&self, cmd: DriverCommand<Rgba>) -> Result<()> {
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

fn run_event_loop(
    cmd_rx: &Receiver<DriverCommand<Rgba>>,
    engine_tx: &EngineSender<Rgba>,
) -> Result<()> {
    // 1. Read CreateWindow command first
    let (window_id, width_px, height_px, title) = match cmd_rx.recv()? {
        DriverCommand::CreateWindow { id, width, height, title } => (id, width, height, title),
        other => return Err(anyhow!("Expected CreateWindow, got {:?}", other)),
    };

    info!("Headless: Creating window '{}' {}x{}", title, width_px, height_px);

    // Send WindowCreated event
    let _ = engine_tx.send(EngineCommand::DisplayEvent(DisplayEvent::WindowCreated {
        id: window_id,
        width_px,
        height_px,
        scale: 1.0,
    }));

    // 2. Run simple event loop
    loop {
        match cmd_rx.recv()? {
            DriverCommand::CreateWindow { .. } => {
                // Already created, ignore
            }
            DriverCommand::DestroyWindow { id } => {
                info!("Headless: DestroyWindow {:?}", id);
                return Ok(());
            }
            DriverCommand::Shutdown => {
                info!("Headless: Shutdown command received");
                return Ok(());
            }
            DriverCommand::Present { frame, .. } => {
                // Just return the framebuffer
                let _ = engine_tx.send(EngineCommand::PresentComplete(frame));
            }
            DriverCommand::SetTitle { title, .. } => {
                info!("Headless: SetTitle '{}'", title);
                let _ = engine_tx.send(EngineCommand::DriverAck);
            }
            DriverCommand::SetSize { width, height, .. } => {
                info!("Headless: SetSize {}x{}", width, height);
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
