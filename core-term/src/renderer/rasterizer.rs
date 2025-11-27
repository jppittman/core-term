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
                    // Pixelflow expects cell coordinates - it converts to pixels internally
                    for (i, ch) in text.chars().enumerate() {
                        ops.push(Op::Text {
                            ch,
                            x: x + i,
                            y: *y,
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
                    for row in 0..*height {
                        for col in 0..*width {
                            ops.push(Op::Text {
                                ch: ' ',
                                x: x + col,
                                y: y + row,
                                fg: *color,
                                bg: *color,
                            });
                        }
                    }
                }

                RenderCommand::SetCursorVisibility { .. }
                | RenderCommand::SetWindowTitle { .. }
                | RenderCommand::RingBell
                | RenderCommand::PresentFrame => {}
            }
        }

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color::Color;
    use crate::glyph::AttrFlags;

    fn make_config(cols: usize, rows: usize, cell_w: usize, cell_h: usize) -> FramebufferConfig {
        FramebufferConfig {
            width_px: cols * cell_w,
            height_px: rows * cell_h,
            cell_width_px: cell_w,
            cell_height_px: cell_h,
        }
    }

    fn pixel_at(fb: &[u8], x: usize, y: usize, stride: usize) -> u32 {
        let idx = (y * stride + x) * 4;
        u32::from_le_bytes([fb[idx], fb[idx + 1], fb[idx + 2], fb[idx + 3]])
    }

    // Black with full alpha (0xFF000000 in RGBA little-endian)
    const BLACK: u32 = 0xFF000000;

    #[test]
    fn clear_fills_entire_framebuffer() {
        let config = make_config(10, 10, 8, 16);
        let mut fb = vec![0u8; config.width_px * config.height_px * 4];

        let commands = vec![RenderCommand::ClearAll {
            bg: Color::Rgb(255, 0, 0),
        }];

        let mut rasterizer = Rasterizer::new(8, 16, FontBackend::Headless, 14.0);
        rasterizer.compile(&commands, &mut fb, config);

        let red = pixel_at(&fb, 0, 0, config.width_px);
        let center = pixel_at(&fb, 40, 80, config.width_px);
        let corner = pixel_at(&fb, 79, 159, config.width_px);

        assert_eq!(red, center);
        assert_eq!(red, corner);
        assert_ne!(red, 0);
    }

    #[test]
    fn text_renders_in_correct_cell_region() {
        let config = make_config(10, 5, 8, 16);
        let mut fb = vec![0u8; config.width_px * config.height_px * 4];

        let commands = vec![
            RenderCommand::ClearAll {
                bg: Color::Rgb(0, 0, 0),
            },
            RenderCommand::DrawTextRun {
                x: 0,
                y: 0,
                text: "A".to_string(),
                fg: Color::Rgb(255, 255, 255),
                bg: Color::Rgb(100, 100, 100),
                flags: AttrFlags::empty(),
                is_selected: false,
            },
        ];

        let mut rasterizer = Rasterizer::new(8, 16, FontBackend::Headless, 14.0);
        rasterizer.compile(&commands, &mut fb, config);

        // Cell (0,0) should have the bg color (100,100,100), not black
        let cell_0_0_bg = pixel_at(&fb, 0, 0, config.width_px);
        // Cell (2,0) should be black - check further away to avoid any glyph edge effects
        let cell_2_0 = pixel_at(&fb, 16, 0, config.width_px);

        assert_ne!(cell_0_0_bg, BLACK, "Cell (0,0) should have gray bg, not black");
        assert_eq!(cell_2_0, BLACK, "Cell (2,0) should be black from clear");
    }

    #[test]
    fn text_at_offset_renders_in_correct_region() {
        let config = make_config(10, 5, 8, 16);
        let mut fb = vec![0u8; config.width_px * config.height_px * 4];

        let commands = vec![
            RenderCommand::ClearAll {
                bg: Color::Rgb(0, 0, 0),
            },
            RenderCommand::DrawTextRun {
                x: 5,
                y: 2,
                text: "X".to_string(),
                fg: Color::Rgb(255, 255, 255),
                bg: Color::Rgb(50, 100, 150),
                flags: AttrFlags::empty(),
                is_selected: false,
            },
        ];

        let mut rasterizer = Rasterizer::new(8, 16, FontBackend::Headless, 14.0);
        rasterizer.compile(&commands, &mut fb, config);

        // Cell (5,2) = pixel (40, 32) should have the bg color
        let target_cell = pixel_at(&fb, 40, 32, config.width_px);
        // Cell (0,0) should be black
        let origin = pixel_at(&fb, 0, 0, config.width_px);
        // Cell (4,2) should be black (one cell to the left)
        let left_neighbor = pixel_at(&fb, 32, 32, config.width_px);

        assert_ne!(target_cell, BLACK, "Cell (5,2) should have colored bg");
        assert_eq!(origin, BLACK, "Origin should be black");
        assert_eq!(left_neighbor, BLACK, "Cell (4,2) should be black");
    }

    #[test]
    fn fillrect_fills_correct_cell_region() {
        let config = make_config(10, 5, 8, 16);
        let mut fb = vec![0u8; config.width_px * config.height_px * 4];

        let commands = vec![
            RenderCommand::ClearAll {
                bg: Color::Rgb(0, 0, 0),
            },
            RenderCommand::FillRect {
                x: 2,
                y: 1,
                width: 3,
                height: 2,
                color: Color::Rgb(0, 255, 0),
                is_selection_bg: false,
            },
        ];

        let mut rasterizer = Rasterizer::new(8, 16, FontBackend::Headless, 14.0);
        rasterizer.compile(&commands, &mut fb, config);

        // Cells (2,1), (3,1), (4,1), (2,2), (3,2), (4,2) should be green
        // Cell (2,1) = pixel (16, 16)
        let inside = pixel_at(&fb, 16, 16, config.width_px);
        // Cell (1,1) = pixel (8, 16) should be black
        let outside_left = pixel_at(&fb, 8, 16, config.width_px);
        // Cell (5,1) = pixel (40, 16) should be black
        let outside_right = pixel_at(&fb, 40, 16, config.width_px);

        assert_ne!(inside, BLACK, "Inside rect should be green");
        assert_eq!(outside_left, BLACK, "Left of rect should be black");
        assert_eq!(outside_right, BLACK, "Right of rect should be black");
    }
}
