#![cfg(use_web_display)]

//! Web DisplayDriver implementation using OffscreenCanvas and SharedArrayBuffer.
//!
//! Driver struct is cmd_tx - trivially Clone.
//! run() reads Configure, creates canvas resources, runs event loop.

pub mod ipc;

use crate::channel::{DriverCommand, EngineCommand, EngineSender};
use crate::display::driver::DisplayDriver;
use crate::display::messages::{DisplayEvent, RenderSnapshot};
use anyhow::{anyhow, Result};
use ipc::SharedRingBuffer;
use js_sys::SharedArrayBuffer;
use std::cell::RefCell;
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use wasm_bindgen::{Clamped, JsCast};
use web_sys::{ImageData, OffscreenCanvas, OffscreenCanvasRenderingContext2d};

// Thread-local storage for web resources passed from JS
thread_local! {
    static RESOURCES: RefCell<Option<(OffscreenCanvas, SharedArrayBuffer)>> = RefCell::new(None);
}

/// Initialize web resources. Must be called from JS before creating driver.
pub fn init_resources(canvas: OffscreenCanvas, sab: SharedArrayBuffer) {
    RESOURCES.with(|r| *r.borrow_mut() = Some((canvas, sab)));
}

// --- Run State (only original driver has this) ---
struct RunState {
    cmd_rx: Receiver<DriverCommand>,
    engine_tx: EngineSender,
}

// --- Display Driver ---

/// Web display driver using OffscreenCanvas.
///
/// Clone to get a handle for sending commands. Only the original can run().
pub struct WebDisplayDriver {
    cmd_tx: SyncSender<DriverCommand>,
    /// Only present on original, None on clones
    run_state: Option<RunState>,
}

impl Clone for WebDisplayDriver {
    fn clone(&self) -> Self {
        Self {
            cmd_tx: self.cmd_tx.clone(),
            run_state: None, // Clones can't run
        }
    }
}

impl DisplayDriver for WebDisplayDriver {
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

    // Get web resources from thread-local storage
    let (canvas, sab) = RESOURCES.with(|r| {
        r.borrow_mut()
            .take()
            .ok_or_else(|| anyhow!("Web resources not initialized. Call init_resources() first."))
    })?;

    let context = canvas
        .get_context("2d")
        .map_err(|_| anyhow!("Failed to get 2d context"))?
        .ok_or_else(|| anyhow!("Context is null"))?
        .dyn_into::<OffscreenCanvasRenderingContext2d>()
        .map_err(|_| anyhow!("Failed to cast context"))?;

    let ipc = SharedRingBuffer::new(&sab);

    let width_px = (config.initial_cols * config.cell_width_px) as u32;
    let height_px = (config.initial_rows * config.cell_height_px) as u32;

    // Resize canvas
    canvas.set_width(width_px);
    canvas.set_height(height_px);

    // Send initial resize
    let _ = engine_tx.send(EngineCommand::DisplayEvent(DisplayEvent::Resize {
        width_px,
        height_px,
    }));

    // 2. Create state and run event loop
    let mut state = WebState {
        context,
        ipc,
        width_px,
        height_px,
    };

    state.event_loop(cmd_rx, engine_tx)
}

// --- Web State (only exists during run) ---

struct WebState {
    context: OffscreenCanvasRenderingContext2d,
    ipc: SharedRingBuffer,
    width_px: u32,
    height_px: u32,
}

impl WebState {
    fn event_loop(
        &mut self,
        cmd_rx: &Receiver<DriverCommand>,
        engine_tx: &EngineSender,
    ) -> Result<()> {
        loop {
            // 1. Poll IPC events (from main thread via SharedArrayBuffer)
            match self.ipc.blocking_read_timeout(16) {
                Ok(Some(evt)) => {
                    if matches!(evt, DisplayEvent::CloseRequested) {
                        return Ok(());
                    }
                    let _ = engine_tx.send(EngineCommand::DisplayEvent(evt));
                }
                Ok(None) => {
                    // Timeout, no events
                }
                Err(e) => {
                    // Log error but continue
                    web_sys::console::error_1(&format!("IPC read error: {}", e).into());
                }
            }

            // 2. Process commands from engine
            while let Ok(cmd) = cmd_rx.try_recv() {
                match cmd {
                    DriverCommand::Configure(_) => {
                        // Already configured, ignore
                    }
                    DriverCommand::Shutdown => {
                        return Ok(());
                    }
                    DriverCommand::Present(snapshot) => {
                        if let Ok(snapshot) = self.handle_present(snapshot) {
                            let _ = engine_tx.send(EngineCommand::PresentComplete(snapshot));
                        }
                    }
                    DriverCommand::SetTitle(_) => {
                        // Not supported in worker context
                        let _ = engine_tx.send(EngineCommand::DriverAck);
                    }
                    DriverCommand::CopyToClipboard(_) => {
                        // Not supported in worker context
                        let _ = engine_tx.send(EngineCommand::DriverAck);
                    }
                    DriverCommand::RequestPaste => {
                        // Not supported in worker context
                        let _ = engine_tx.send(EngineCommand::DriverAck);
                    }
                    DriverCommand::Bell => {
                        // Not supported in worker context
                        let _ = engine_tx.send(EngineCommand::DriverAck);
                    }
                }
            }
        }
    }

    fn handle_present(&mut self, snapshot: RenderSnapshot) -> Result<RenderSnapshot> {
        let data = snapshot.framebuffer.as_ref();
        let image_data = ImageData::new_with_u8_clamped_array_and_sh(
            Clamped(data),
            snapshot.width_px,
            snapshot.height_px,
        )
        .map_err(|e| anyhow!("Failed to create ImageData: {:?}", e))?;

        self.context
            .put_image_data(&image_data, 0.0, 0.0)
            .map_err(|e| anyhow!("Failed to put image data: {:?}", e))?;

        Ok(snapshot)
    }
}
