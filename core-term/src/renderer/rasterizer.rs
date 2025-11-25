// src/renderer/rasterizer.rs
//! Bridge between core-term's RenderCommand API and pixelflow's Op API.

use crate::platform::backends::RenderCommand;
use pixelflow_render::Op;

/// Configuration for framebuffer rendering.
#[derive(Debug, Clone, Copy)]
pub struct FramebufferConfig {
    pub width_px: usize,
    pub height_px: usize,
    pub cell_width_px: usize,
    pub cell_height_px: usize,
}

/// Stateless rasterizer that converts RenderCommands to pixelflow Ops.
pub struct Rasterizer;

impl Rasterizer {
    /// Create a new rasterizer (all parameters are ignored, stateless now).
    pub fn new(
        _cell_width_px: usize,
        _cell_height_px: usize,
        _font_backend: FontBackend,
        _font_size_pt: f64,
    ) -> Self {
        Self
    }

    /// Compile render commands into framebuffer pixels.
    pub fn compile(
        &mut self,
        commands: &[RenderCommand],
        framebuffer: &mut [u8],
        config: FramebufferConfig,
    ) {
        // Convert framebuffer from [u8] to [u32]
        let fb_u32 = unsafe {
            std::slice::from_raw_parts_mut(
                framebuffer.as_mut_ptr() as *mut u32,
                framebuffer.len() / 4,
            )
        };

        // Convert RenderCommand to Op
        let mut ops: Vec<Op<Vec<u8>>> = Vec::new();

        for cmd in commands {
            match cmd {
                RenderCommand::ClearAll { bg } => {
                    ops.push(Op::Clear { color: *bg });
                }

                RenderCommand::DrawTextRun {
                    x,
                    y,
                    text,
                    fg,
                    bg,
                    flags: _,
                    is_selected: _,
                } => {
                    // Convert cell coordinates to pixel coordinates
                    let px_x = x * config.cell_width_px;
                    let px_y = y * config.cell_height_px;

                    // Generate Op::Text for each character
                    for (i, ch) in text.chars().enumerate() {
                        let char_x = px_x + i * config.cell_width_px;
                        ops.push(Op::Text {
                            ch,
                            x: char_x,
                            y: px_y,
                            fg: *fg,
                            bg: *bg,
                        });
                    }
                }

                RenderCommand::FillRect {
                    x,
                    y,
                    width,
                    height,
                    color,
                    is_selection_bg: _,
                } => {
                    // Fill rectangle by drawing individual cells
                    // This is less efficient but matches the current API
                    for row in 0..*height {
                        for col in 0..*width {
                            let px_x = (x + col) * config.cell_width_px;
                            let px_y = (y + row) * config.cell_height_px;

                            // Use a space character to fill the cell
                            ops.push(Op::Text {
                                ch: ' ',
                                x: px_x,
                                y: px_y,
                                fg: *color,
                                bg: *color,
                            });
                        }
                    }
                }

                // Ignore metadata commands
                RenderCommand::SetCursorVisibility { .. }
                | RenderCommand::SetWindowTitle { .. }
                | RenderCommand::RingBell
                | RenderCommand::PresentFrame => {}
            }
        }

        // Call pixelflow's stateless process_frame
        pixelflow_render::process_frame(
            fb_u32,
            config.width_px,
            config.height_px,
            config.cell_width_px,
            config.cell_height_px,
            &ops,
        );
    }
}

/// Font backend enum (kept for compatibility but not used).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum FontBackend {
    Headless,
    Cocoa,
    CoreText,
    X11,
    FreeType,
}
