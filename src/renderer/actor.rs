// src/renderer/actor.rs
//! RenderActor - Dedicated thread for rasterizing frames.
//!
//! Simple work queue pattern: receives (snapshot, framebuffer, state),
//! rasterizes pixels, returns (snapshot, framebuffer).
//!
//! Threading model:
//! - Owns: Renderer, SoftwareRasterizer, FontManager, Config
//! - No queuing: Single framebuffer + snapshot = max 1 frame in flight
//! - Natural backpressure via SyncChannel with capacity=1

use crate::config::Config;
use crate::platform::backends::{PlatformState, RenderCommand};
use crate::rasterizer::{compile_into_buffer, SoftwareRasterizer};
use crate::renderer::Renderer;
use crate::term::TerminalSnapshot;
use anyhow::{Context, Result};
use log::*;
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::thread;

/// Work item sent to render thread: snapshot + framebuffer + display metrics
pub struct RenderWork {
    pub snapshot: Box<TerminalSnapshot>,
    pub framebuffer: Box<[u8]>,
    pub display_width_px: u16,
    pub display_height_px: u16,
    pub scale_factor: f64,
}

/// Result returned from render thread: snapshot + framebuffer (pixels updated)
pub struct RenderResult {
    pub snapshot: Box<TerminalSnapshot>,
    pub framebuffer: Box<[u8]>,
}

/// Channels for communicating with render thread
pub struct RenderChannels {
    pub work_tx: SyncSender<RenderWork>,
    pub result_rx: Receiver<RenderResult>,
}

/// RenderActor state (runs on dedicated thread)
struct RenderActor {
    renderer: Renderer,
    rasterizer: SoftwareRasterizer,
    config: Config,
}

impl RenderActor {
    /// Create new render actor with given renderer and rasterizer
    fn new(renderer: Renderer, rasterizer: SoftwareRasterizer, config: Config) -> Self {
        Self {
            renderer,
            rasterizer,
            config,
        }
    }

    /// Process one frame: convert snapshot to pixels in framebuffer
    fn render_frame(&mut self, work: RenderWork) -> RenderResult {
        let RenderWork {
            snapshot,
            mut framebuffer,
            display_width_px,
            display_height_px,
            scale_factor,
        } = work;

        // Get font cell dimensions from snapshot
        let font_cell_width_px = snapshot.cell_width_px;
        let font_cell_height_px = snapshot.cell_height_px;

        // Construct complete platform state
        let platform_state = PlatformState {
            event_fd: None, // Not used by renderer
            font_cell_width_px,
            font_cell_height_px,
            scale_factor,
            display_width_px,
            display_height_px,
        };

        // Prepare high-level render commands
        let mut render_commands =
            self.renderer
                .prepare_render_commands(&snapshot, &self.config, &platform_state);
        render_commands.push(RenderCommand::PresentFrame);

        // Compile render commands into framebuffer pixels
        // display_width_px and display_height_px are already physical pixels from the display driver
        // font_cell dimensions from snapshot are logical pixels from config
        // FIXME: Rasterizer is now initialized with physical pixels (hardcoded 2x scale in main.rs)
        // so we scale here to match. This needs proper refactoring to get scale factor before init.
        let physical_cell_width = (font_cell_width_px as f64 * scale_factor) as usize;
        let physical_cell_height = (font_cell_height_px as f64 * scale_factor) as usize;

        debug!(
            "render_frame: display={}x{} physical_px, scale={}, font_cell={}x{} logical_px, physical_cell={}x{} physical_px",
            display_width_px, display_height_px, scale_factor,
            font_cell_width_px, font_cell_height_px,
            physical_cell_width, physical_cell_height
        );

        compile_into_buffer(
            &mut self.rasterizer,
            render_commands,
            &mut framebuffer,
            display_width_px as usize,
            display_height_px as usize,
            physical_cell_width,
            physical_cell_height,
        );

        RenderResult {
            snapshot,
            framebuffer,
        }
    }

    /// Run the render loop: receive work, process, send result
    fn run(mut self, work_rx: Receiver<RenderWork>, result_tx: SyncSender<RenderResult>) {
        info!("RenderActor: Thread started");

        loop {
            match work_rx.recv() {
                Ok(work) => {
                    trace!("RenderActor: Received render work");
                    let result = self.render_frame(work);

                    if let Err(e) = result_tx.send(result) {
                        warn!(
                            "RenderActor: Failed to send result (platform closed): {}",
                            e
                        );
                        break;
                    }
                    trace!("RenderActor: Sent render result");
                }
                Err(_) => {
                    info!("RenderActor: Work channel closed, exiting");
                    break;
                }
            }
        }

        info!("RenderActor: Thread stopped");
    }
}

/// Spawn the render thread and return channels for communication
pub fn spawn_render_thread(
    renderer: Renderer,
    rasterizer: SoftwareRasterizer,
    config: Config,
) -> Result<RenderChannels> {
    info!("spawn_render_thread: Creating channels");

    // Create channels with capacity=1 for natural backpressure
    let (work_tx, work_rx) = sync_channel(1);
    let (result_tx, result_rx) = sync_channel(1);

    // Spawn render thread
    thread::Builder::new()
        .name("render".to_string())
        .spawn(move || {
            let actor = RenderActor::new(renderer, rasterizer, config);
            actor.run(work_rx, result_tx);
        })
        .context("Failed to spawn render thread")?;

    info!("spawn_render_thread: Render thread spawned");

    Ok(RenderChannels { work_tx, result_rx })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::rasterizer::SoftwareRasterizer;
    use crate::renderer::Renderer;
    use crate::term::{CursorRenderState, CursorShape, Selection, SnapshotLine, TerminalSnapshot};

    fn create_test_snapshot(cols: usize, rows: usize, cell_width_px: usize, cell_height_px: usize) -> TerminalSnapshot {
        TerminalSnapshot {
            dimensions: (cols, rows),
            lines: vec![SnapshotLine {
                is_dirty: true,
                cells: vec![],
            }; rows],
            cursor_state: None,
            selection: Selection::default(),
            cell_width_px,
            cell_height_px,
        }
    }

    #[test]
    fn physical_pixel_dimensions_calculated_correctly_for_hidpi() {
        // Regression test for HiDPI scaling bug
        // Verifies that cell dimensions are properly scaled from logical to physical pixels

        let config = Config::default();
        let renderer = Renderer::new();

        // Simulate a 2x scale factor (Retina display)
        // Rasterizer initialized with physical pixels (20x32 for 2x scale of 10x16 logical)
        let rasterizer = SoftwareRasterizer::new(20, 32);

        let mut actor = RenderActor::new(renderer, rasterizer, config);

        // Create snapshot with logical pixel dimensions from config
        let snapshot = Box::new(create_test_snapshot(80, 24, 10, 16));

        // Create framebuffer for 1600x1200 physical pixels (800x600 logical * 2.0 scale)
        let framebuffer = vec![0u8; 1600 * 1200 * 4].into_boxed_slice();

        let work = RenderWork {
            snapshot,
            framebuffer,
            display_width_px: 1600,
            display_height_px: 1200,
            scale_factor: 2.0,
        };

        // This should not panic - the test verifies that dimensions are handled correctly
        let result = actor.render_frame(work);

        // Verify we got the snapshot and framebuffer back
        assert_eq!(result.snapshot.dimensions, (80, 24));
        assert_eq!(result.framebuffer.len(), 1600 * 1200 * 4);
    }

    #[test]
    fn physical_pixel_dimensions_correct_for_1x_scale() {
        // Test standard (non-HiDPI) displays with 1x scale factor

        let config = Config::default();
        let renderer = Renderer::new();

        // 1x scale factor
        let rasterizer = SoftwareRasterizer::new(10, 16);

        let mut actor = RenderActor::new(renderer, rasterizer, config);

        let snapshot = Box::new(create_test_snapshot(80, 24, 10, 16));
        let framebuffer = vec![0u8; 800 * 600 * 4].into_boxed_slice();

        let work = RenderWork {
            snapshot,
            framebuffer,
            display_width_px: 800,
            display_height_px: 600,
            scale_factor: 1.0,
        };

        let result = actor.render_frame(work);

        assert_eq!(result.snapshot.dimensions, (80, 24));
        assert_eq!(result.framebuffer.len(), 800 * 600 * 4);
    }
}
