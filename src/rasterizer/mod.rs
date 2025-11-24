//! Render command compiler for terminal rendering.
//!
//! This module acts as a **compiler** that transforms high-level RenderCommands
//! into minimal, RISC-like DriverCommands:
//!
//! ```text
//! RenderCommand[]  →  [Rasterizer/Compiler]  →  DriverCommand[]
//!   (high-level)                                   (low-level)
//!   DrawTextRun                                    BlitPixels
//!   FillRect                                       Clear
//!   etc.                                           Present
//! ```
//!
//! Benefits:
//! - Renderer and existing drivers stay unchanged
//! - New drivers only implement minimal DriverCommand interface
//! - Rasterizer can optimize (merge blits, eliminate redundant clears)
//! - Text rendering is done once, in one place
//! - Easy to add optimizations (damage tracking, caching, etc.)

// Font support modules
pub mod font_driver;
pub mod font_manager;

#[cfg(target_os = "macos")]
pub mod cocoa_font_driver;

use crate::color::Color;
use crate::glyph::AttrFlags;
use crate::platform::backends::{DriverCommand, RenderCommand};
use crate::rasterizer::font_driver::FontDriver;
use crate::rasterizer::font_manager::FontManager;
use log::{debug, trace};
use std::collections::HashMap;

// Platform-specific font driver selection
#[cfg(target_os = "macos")]
use crate::rasterizer::cocoa_font_driver::CocoaFontDriver;
#[cfg(target_os = "macos")]
type PlatformFontDriver = CocoaFontDriver;

// TODO: Linux font driver
#[cfg(not(target_os = "macos"))]
type PlatformFontDriver = (); // Placeholder for now

/// RGBA color in 32-bit format (8 bits per channel)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Rgba {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

impl Rgba {
    pub const fn new(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { r, g, b, a }
    }

    pub const fn opaque(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b, a: 255 }
    }

    /// Convert to RGBA byte array
    pub fn to_bytes(&self) -> [u8; 4] {
        [self.r, self.g, self.b, self.a]
    }
}

impl From<Color> for Rgba {
    fn from(color: Color) -> Self {
        match color {
            Color::Named(named) => {
                use crate::color::NamedColor::*;
                match named {
                    Black => Rgba::opaque(0, 0, 0),
                    Red => Rgba::opaque(205, 49, 49),
                    Green => Rgba::opaque(13, 188, 121),
                    Yellow => Rgba::opaque(229, 229, 16),
                    Blue => Rgba::opaque(36, 114, 200),
                    Magenta => Rgba::opaque(188, 63, 188),
                    Cyan => Rgba::opaque(17, 168, 205),
                    White => Rgba::opaque(229, 229, 229),
                    BrightBlack => Rgba::opaque(102, 102, 102),
                    BrightRed => Rgba::opaque(241, 76, 76),
                    BrightGreen => Rgba::opaque(35, 209, 139),
                    BrightYellow => Rgba::opaque(245, 245, 67),
                    BrightBlue => Rgba::opaque(59, 142, 234),
                    BrightMagenta => Rgba::opaque(214, 112, 214),
                    BrightCyan => Rgba::opaque(41, 184, 219),
                    BrightWhite => Rgba::opaque(255, 255, 255),
                }
            }
            Color::Rgb(r, g, b) => Rgba::opaque(r, g, b),
            Color::Indexed(idx) => {
                // Simple 256-color palette mapping
                // TODO: Use proper xterm 256 color palette
                if idx < 16 {
                    // Standard colors
                    Color::Named(crate::color::NamedColor::from_index(idx)).into()
                } else if idx < 232 {
                    // 216-color cube: 6x6x6
                    let idx = idx - 16;
                    let r = ((idx / 36) % 6) * 51;
                    let g = ((idx / 6) % 6) * 51;
                    let b = (idx % 6) * 51;
                    Rgba::opaque(r, g, b)
                } else {
                    // Grayscale ramp
                    let gray = 8 + (idx - 232) * 10;
                    Rgba::opaque(gray, gray, gray)
                }
            }
            Color::Default => Rgba::opaque(255, 255, 255), // Should be resolved before this
        }
    }
}

/// A key for caching rendered glyphs
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct GlyphKey {
    codepoint: char,
    fg: Rgba,
    bg: Rgba,
    flags: AttrFlags,
}

/// A pre-rendered glyph as RGBA pixel data
#[derive(Debug, Clone)]
pub struct RenderedGlyph {
    pub width_px: usize,
    pub height_px: usize,
    pub rgba_data: Vec<u8>, // width * height * 4 bytes
}

/// Software text rasterizer
pub struct SoftwareRasterizer {
    /// Glyph cache: codepoint + style -> rendered pixels
    glyph_cache: HashMap<GlyphKey, RenderedGlyph>,
    /// Font metrics
    cell_width_px: usize,
    cell_height_px: usize,
    /// Font manager for loading and caching fonts
    font_manager: Option<FontManager<PlatformFontDriver>>,
}

impl SoftwareRasterizer {
    /// Create a new rasterizer with specified cell dimensions
    pub fn new(cell_width_px: usize, cell_height_px: usize) -> Self {
        use crate::config::CONFIG;
        use log::*;

        let font_size_pt = CONFIG.appearance.font.size_pt;

        let safe_width = cell_width_px.max(1);
        let safe_height = cell_height_px.max(1);

        info!(
            "SoftwareRasterizer: Using cell metrics: {}x{} px (font size {} pt)",
            safe_width, safe_height, font_size_pt
        );

        // Initialize platform-specific font manager
        #[cfg(target_os = "macos")]
        let font_manager = {
            let driver = CocoaFontDriver::new();
            let _font_config = &CONFIG.appearance.font;

            let manager = FontManager::new(
                driver,
                "Menlo",            // regular (TODO: use font_config.normal)
                "Menlo-Bold",       // bold (TODO: use font_config.bold)
                "Menlo-Italic",     // italic (TODO: use font_config.italic)
                "Menlo-BoldItalic", // bold+italic (TODO: use font_config.bold_italic)
                font_size_pt,
            )
            .expect("Failed to initialize font manager");

            info!(
                "SoftwareRasterizer: Initialized with FontManager (Menlo {} pt from CONFIG)",
                font_size_pt
            );
            Some(manager)
        };

        #[cfg(not(target_os = "macos"))]
        let font_manager = {
            warn!(
                "SoftwareRasterizer: No font driver for this platform, using placeholder rendering"
            );
            None
        };

        Self {
            glyph_cache: HashMap::new(),
            cell_width_px: safe_width,
            cell_height_px: safe_height,
            font_manager,
        }
    }

    /// Render a single character cell to RGBA pixels
    pub fn render_cell(
        &mut self,
        ch: char,
        fg: Color,
        bg: Color,
        flags: AttrFlags,
    ) -> &RenderedGlyph {
        let fg_rgba = fg.into();
        let bg_rgba = bg.into();

        let key = GlyphKey {
            codepoint: ch,
            fg: fg_rgba,
            bg: bg_rgba,
            flags,
        };

        // Check if glyph is cached, if not render it
        if !self.glyph_cache.contains_key(&key) {
            let glyph = self.render_placeholder_glyph(ch, fg_rgba, bg_rgba, flags);
            self.glyph_cache.insert(key.clone(), glyph);
        }

        self.glyph_cache.get(&key).unwrap()
    }

    /// Render a glyph using the platform's font driver, or placeholder if unavailable
    fn render_placeholder_glyph(
        &mut self,
        ch: char,
        fg: Rgba,
        bg: Rgba,
        flags: AttrFlags,
    ) -> RenderedGlyph {
        let width = self.cell_width_px;
        let height = self.cell_height_px;

        // Try to use the font manager if available
        if let Some(ref mut font_manager) = self.font_manager {
            if let Some(resolved) = font_manager.get_glyph(ch, flags) {
                let font = font_manager.get_font(resolved.font_id);
                let glyph_pixels =
                    font_manager
                        .driver()
                        .rasterize_glyph(font, resolved.glyph_id, width, height);

                let rgba_data = Self::colorize_glyph(&glyph_pixels, fg, bg);

                return RenderedGlyph {
                    width_px: width,
                    height_px: height,
                    rgba_data,
                };
            }
        }

        // Fall back to placeholder rendering
        let mut rgba_data = Vec::with_capacity(width * height * 4);

        // Fill with background color
        for _ in 0..(width * height) {
            rgba_data.extend_from_slice(&bg.to_bytes());
        }

        // Add underline if needed
        if flags.contains(AttrFlags::UNDERLINE) {
            let underline_y = height - 2;
            for x in 0..width {
                let idx = (underline_y * width + x) * 4;
                rgba_data[idx..idx + 4].copy_from_slice(&fg.to_bytes());
            }
        }

        // Add strikethrough if needed
        if flags.contains(AttrFlags::STRIKETHROUGH) {
            let strike_y = height / 2;
            for x in 0..width {
                let idx = (strike_y * width + x) * 4;
                rgba_data[idx..idx + 4].copy_from_slice(&fg.to_bytes());
            }
        }

        RenderedGlyph {
            width_px: width,
            height_px: height,
            rgba_data,
        }
    }

    /// Colorize a white-on-transparent glyph with fg/bg colors.
    ///
    /// The glyph is expected to be white (255,255,255) on transparent background,
    /// with varying alpha representing the glyph's coverage at each pixel.
    /// This function blends fg and bg colors based on that alpha.
    fn colorize_glyph(white_glyph: &[u8], fg: Rgba, bg: Rgba) -> Vec<u8> {
        let mut result = Vec::with_capacity(white_glyph.len());

        // Sample first few pixels for debugging
        if white_glyph.len() >= 16 {
            debug!(
                "colorize_glyph: Input - first 4 pixels (RGBA): [{},{},{},{}] [{},{},{},{}] [{},{},{},{}] [{},{},{},{}]",
                white_glyph[0], white_glyph[1], white_glyph[2], white_glyph[3],
                white_glyph[4], white_glyph[5], white_glyph[6], white_glyph[7],
                white_glyph[8], white_glyph[9], white_glyph[10], white_glyph[11],
                white_glyph[12], white_glyph[13], white_glyph[14], white_glyph[15]
            );
            debug!("colorize_glyph: fg={:?}, bg={:?}", fg, bg);
        }

        for pixel in white_glyph.chunks_exact(4) {
            let alpha = pixel[3] as f32 / 255.0;

            // Blend foreground and background based on glyph alpha
            let r = (fg.r as f32 * alpha + bg.r as f32 * (1.0 - alpha)) as u8;
            let g = (fg.g as f32 * alpha + bg.g as f32 * (1.0 - alpha)) as u8;
            let b = (fg.b as f32 * alpha + bg.b as f32 * (1.0 - alpha)) as u8;

            result.extend_from_slice(&[r, g, b, 255]);
        }

        // Sample first few output pixels for debugging
        if result.len() >= 16 {
            debug!(
                "colorize_glyph: Output - first 4 pixels (RGBA): [{},{},{},{}] [{},{},{},{}] [{},{},{},{}] [{},{},{},{}]",
                result[0], result[1], result[2], result[3],
                result[4], result[5], result[6], result[7],
                result[8], result[9], result[10], result[11],
                result[12], result[13], result[14], result[15]
            );
        }

        result
    }

    /// Get cell dimensions
    pub fn cell_size(&self) -> (usize, usize) {
        (self.cell_width_px, self.cell_height_px)
    }

    /// Clear the glyph cache (e.g., when font changes)
    pub fn clear_cache(&mut self) {
        self.glyph_cache.clear();
    }
}

/// Compile high-level RenderCommands into a framebuffer.
///
/// This is the main compiler entry point. It takes the output from the Renderer
/// and writes pixels directly to the provided framebuffer. This is zero-copy:
/// no intermediate buffers or commands containing pixel data.
///
/// # Arguments
/// * `rasterizer` - The software rasterizer (owns font manager and glyph cache)
/// * `commands` - High-level render commands from the Renderer
/// * `framebuffer` - The target RGBA pixel buffer to write to (row-major, 4 bytes/pixel)
/// * `buffer_width_px` - Width of the framebuffer in pixels
/// * `buffer_height_px` - Height of the framebuffer in pixels
/// * `cell_width_px` - Width of a character cell in pixels
/// * `cell_height_px` - Height of a character cell in pixels
///
/// # Returns
/// A vector of minimal DriverCommands (just metadata, no pixel data)
pub fn compile_into_buffer(
    rasterizer: &mut SoftwareRasterizer,
    commands: Vec<RenderCommand>,
    framebuffer: &mut [u8],
    buffer_width_px: usize,
    buffer_height_px: usize,
    cell_width_px: usize,
    cell_height_px: usize,
) -> Vec<DriverCommand> {
    let mut driver_commands = Vec::new();

    // TODO(performance): Add software prefetching for dirty lines
    // When processing DrawTextRun commands, could use std::intrinsics::prefetch_read_data
    // to hint the CPU to load the next line's data into cache while processing current line.
    // Could reduce cache misses by 10-20% for large terminal repaints. Profile first!

    let num_render_commands = commands.len();

    for cmd in commands {
        match cmd {
            RenderCommand::ClearAll { bg } => {
                let rgba: Rgba = bg.into();
                let color_bytes = rgba.to_bytes();
                trace!(
                    "rasterizer: ClearAll with bg={:?} (rgba={},{},{},{})",
                    bg,
                    rgba.r,
                    rgba.g,
                    rgba.b,
                    rgba.a
                );
                // Fill entire framebuffer with background color
                for pixel in framebuffer.chunks_exact_mut(4) {
                    pixel.copy_from_slice(&color_bytes);
                }
            }
            RenderCommand::DrawTextRun {
                x,
                y,
                text,
                fg,
                bg,
                flags,
                is_selected: _,
            } => {
                debug!(
                    "rasterizer: DrawTextRun '{}' at col={} row={} fg={:?} bg={:?} flags={:?}",
                    text, x, y, fg, bg, flags
                );

                let y_px = y.saturating_mul(cell_height_px);
                if y_px >= buffer_height_px {
                    continue;
                }

                let mut x_offset = x;
                for ch in text.chars() {
                    let x_px = x_offset.saturating_mul(cell_width_px);
                    if x_px >= buffer_width_px {
                        break;
                    }

                    let glyph = rasterizer.render_cell(ch, fg, bg, flags);

                    let has_color = glyph
                        .rgba_data
                        .chunks_exact(4)
                        .any(|p| p[0] > 0 || p[1] > 0 || p[2] > 0 || p[3] > 0);

                    debug!(
                        "rasterizer: Rendered '{}' (U+{:04X}): {}x{} pixels, {} bytes, has_color={}",
                        ch, ch as u32, glyph.width_px, glyph.height_px, glyph.rgba_data.len(), has_color
                    );

                    blit_to_framebuffer(
                        framebuffer,
                        buffer_width_px,
                        buffer_height_px,
                        &glyph.rgba_data,
                        x_px,
                        y_px,
                        glyph.width_px,
                        glyph.height_px,
                    );
                    x_offset = x_offset.saturating_add(1);
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
                let rgba: Rgba = color.into();
                let color_bytes = rgba.to_bytes();
                let x_px = x.saturating_mul(cell_width_px);
                let y_px = y.saturating_mul(cell_height_px);
                let width_px = width.saturating_mul(cell_width_px);
                let height_px = height.saturating_mul(cell_height_px);

                if x_px >= buffer_width_px || y_px >= buffer_height_px {
                    continue;
                }

                debug!(
                    "rasterizer: FillRect at col={} row={} size={}x{} cells ({}x{} px) color={:?} (rgba={},{},{},{})",
                    x, y, width, height, width_px, height_px, color, rgba.r, rgba.g, rgba.b, rgba.a
                );

                // Fill rectangle directly in framebuffer
                for row in 0..height_px {
                    let y_pos = y_px + row;
                    if y_pos >= buffer_height_px {
                        break;
                    }
                    let row_start = (y_pos * buffer_width_px + x_px) * 4;
                    for col in 0..width_px {
                        if x_px + col >= buffer_width_px {
                            break;
                        }
                        let idx = row_start + col * 4;
                        if idx + 4 <= framebuffer.len() {
                            framebuffer[idx..idx + 4].copy_from_slice(&color_bytes);
                        }
                    }
                }
            }
            RenderCommand::SetCursorVisibility { visible: _ } => {
                // TODO: Handle cursor visibility
                // For now, cursor is drawn as a text cell with REVERSE flag
            }
            RenderCommand::SetWindowTitle { title } => {
                driver_commands.push(DriverCommand::SetTitle { title });
            }
            RenderCommand::RingBell => {
                driver_commands.push(DriverCommand::Bell);
            }
            RenderCommand::PresentFrame => {
                debug!("rasterizer: PresentFrame - requesting screen update");
                driver_commands.push(DriverCommand::Present);
            }
        }
    }

    debug!(
        "rasterizer: Compilation complete - {} render commands → {} driver commands",
        num_render_commands,
        driver_commands.len()
    );

    driver_commands
}

/// Helper function to blit a source buffer into a destination framebuffer
fn blit_to_framebuffer(
    dest: &mut [u8],
    dest_width_px: usize,
    dest_height_px: usize,
    src: &[u8],
    dest_x_px: usize,
    dest_y_px: usize,
    src_width_px: usize,
    src_height_px: usize,
) {
    // TODO(performance): Add SIMD acceleration for pixel copying
    // Could use portable_simd or wide crate for 4-8x speedup on alpha blending operations.
    // Currently relying on memcpy optimization and auto-vectorization, which works well for
    // simple copy_from_slice but may not vectorize complex blending. Benchmark first!

    for row in 0..src_height_px {
        let dest_y = dest_y_px + row;
        if dest_y >= dest_height_px {
            break;
        }

        let src_row_start = row * src_width_px * 4;
        let dest_row_start = (dest_y * dest_width_px + dest_x_px) * 4;
        let copy_width = src_width_px.min(dest_width_px.saturating_sub(dest_x_px));

        if dest_row_start + copy_width * 4 <= dest.len()
            && src_row_start + copy_width * 4 <= src.len()
        {
            dest[dest_row_start..dest_row_start + copy_width * 4]
                .copy_from_slice(&src[src_row_start..src_row_start + copy_width * 4]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color::{Color, NamedColor};
    use crate::glyph::AttrFlags;
    use crate::platform::backends::{DriverCommand, RenderCommand};

    // --- Test Constants ---

    const TEST_BUF_WIDTH: usize = 100;
    const TEST_BUF_HEIGHT: usize = 100;
    const TEST_CELL_WIDTH: usize = 8;
    const TEST_CELL_HEIGHT: usize = 16;
    const BYTES_PER_PIXEL: usize = 4; // RGBA

    // Standard colors for clear testing
    const COLOR_BLACK: Color = Color::Named(NamedColor::Black);
    const COLOR_WHITE: Color = Color::Named(NamedColor::White);
    const COLOR_RED: Color = Color::Named(NamedColor::Red);
    const COLOR_GREEN: Color = Color::Named(NamedColor::Green);

    /// Test harness to manage the rasterizer lifecycle and buffer state.
    /// Reduces argument clutter in tests and provides ergonomic assertions.
    struct TestHarness {
        rasterizer: SoftwareRasterizer,
        framebuffer: Vec<u8>,
        width: usize,
        height: usize,
        cell_w: usize,
        cell_h: usize,
    }

    impl TestHarness {
        fn new(w: usize, h: usize, cell_w: usize, cell_h: usize) -> Self {
            // Ensure we allocate enough space even for 0 dims to avoid Vec panics in setup
            let buf_size = std::cmp::max(w * h * BYTES_PER_PIXEL, 4);
            Self {
                rasterizer: SoftwareRasterizer::new(cell_w, cell_h),
                framebuffer: vec![0u8; buf_size],
                width: w,
                height: h,
                cell_w,
                cell_h,
            }
        }

        fn default() -> Self {
            Self::new(
                TEST_BUF_WIDTH,
                TEST_BUF_HEIGHT,
                TEST_CELL_WIDTH,
                TEST_CELL_HEIGHT,
            )
        }

        /// Runs the compiler against the internal framebuffer.
        fn compile(&mut self, commands: Vec<RenderCommand>) -> Vec<DriverCommand> {
            compile_into_buffer(
                &mut self.rasterizer,
                commands,
                &mut self.framebuffer,
                self.width,
                self.height,
                self.cell_w,
                self.cell_h,
            )
        }

        /// Gets a pixel at (x, y). Panics if out of bounds (use strictly for validation).
        fn get_pixel(&self, x: usize, y: usize) -> Rgba {
            let idx = (y * self.width + x) * BYTES_PER_PIXEL;
            let p = &self.framebuffer[idx..idx + 4];
            Rgba::new(p[0], p[1], p[2], p[3])
        }

        /// Asserts that every pixel in the given rect matches the expected color.
        #[track_caller]
        fn assert_rect_color(&self, x: usize, y: usize, w: usize, h: usize, expected: Color) {
            let expected_rgba: Rgba = expected.into();

            for cy in y..y + h {
                for cx in x..x + w {
                    if cx >= self.width || cy >= self.height {
                        continue; // Skip out of bounds checks for clipping tests
                    }
                    let pixel = self.get_pixel(cx, cy);
                    assert_eq!(
                        pixel, expected_rgba,
                        "Pixel mismatch at ({}, {}). Expected {:?}, got {:?}",
                        cx, cy, expected_rgba, pixel
                    );
                }
            }
        }

        /// Asserts that every pixel in the buffer matches the expected color.
        #[track_caller]
        fn assert_clear(&self, expected: Color) {
            self.assert_rect_color(0, 0, self.width, self.height, expected);
        }
    }

    // --- Correctness Tests ---

    #[test]
    fn test_compile_clear_all() {
        let mut harness = TestHarness::default();

        harness.compile(vec![RenderCommand::ClearAll { bg: COLOR_WHITE }]);
        harness.assert_clear(COLOR_WHITE);

        harness.compile(vec![RenderCommand::ClearAll { bg: COLOR_BLACK }]);
        harness.assert_clear(COLOR_BLACK);
    }

    #[test]
    fn test_fill_rect_geometry_exact() {
        let mut harness = TestHarness::default();
        harness.compile(vec![RenderCommand::ClearAll { bg: COLOR_WHITE }]);

        let rect_cmd = RenderCommand::FillRect {
            x: 1,
            y: 1,
            width: 1,
            height: 1,
            color: COLOR_RED,
            is_selection_bg: false,
        };
        harness.compile(vec![rect_cmd]);

        harness.assert_rect_color(8, 16, 8, 16, COLOR_RED);

        let white_rgba: Rgba = COLOR_WHITE.into();
        assert_eq!(harness.get_pixel(8, 15), white_rgba, "Bleed detected top");
        assert_eq!(harness.get_pixel(7, 16), white_rgba, "Bleed detected left");
        assert_eq!(
            harness.get_pixel(16, 16),
            white_rgba,
            "Bleed detected right"
        );
        assert_eq!(
            harness.get_pixel(8, 32),
            white_rgba,
            "Bleed detected bottom"
        );
    }

    #[test]
    fn test_overwrite_behavior() {
        let mut harness = TestHarness::default();

        let cmds = vec![
            RenderCommand::ClearAll { bg: COLOR_WHITE },
            RenderCommand::FillRect {
                x: 0,
                y: 0,
                width: 4,
                height: 4,
                color: COLOR_RED,
                is_selection_bg: false,
            },
            RenderCommand::FillRect {
                x: 1,
                y: 1,
                width: 2,
                height: 2,
                color: COLOR_GREEN,
                is_selection_bg: false,
            },
        ];
        harness.compile(cmds);

        harness.assert_rect_color(
            1 * TEST_CELL_WIDTH,
            1 * TEST_CELL_HEIGHT,
            2 * TEST_CELL_WIDTH,
            2 * TEST_CELL_HEIGHT,
            COLOR_GREEN,
        );
        harness.assert_rect_color(0, 0, TEST_CELL_WIDTH, TEST_CELL_HEIGHT, COLOR_RED);
    }

    #[test]
    fn test_driver_command_passthrough() {
        let mut harness = TestHarness::default();
        let cmds = vec![
            RenderCommand::SetWindowTitle {
                title: "Unit Test".into(),
            },
            RenderCommand::RingBell,
            RenderCommand::PresentFrame,
        ];

        let driver_cmds = harness.compile(cmds);

        assert_eq!(driver_cmds.len(), 3);
        assert!(matches!(driver_cmds[0], DriverCommand::SetTitle { .. }));
        assert!(matches!(driver_cmds[1], DriverCommand::Bell));
        assert!(matches!(driver_cmds[2], DriverCommand::Present));
    }

    // --- Edge Case & Robustness Tests ---

    #[test]
    fn test_clipping_right_edge() {
        let mut harness = TestHarness::default();
        harness.compile(vec![RenderCommand::ClearAll { bg: COLOR_WHITE }]);

        let cmd = RenderCommand::FillRect {
            x: 12,
            y: 0,
            width: 2,
            height: 1,
            color: COLOR_RED,
            is_selection_bg: false,
        };
        harness.compile(vec![cmd]);

        harness.assert_rect_color(96, 0, 4, 16, COLOR_RED);
    }

    #[test]
    fn test_clipping_bottom_edge() {
        let mut harness = TestHarness::default();
        harness.compile(vec![RenderCommand::ClearAll { bg: COLOR_WHITE }]);

        let cmd = RenderCommand::FillRect {
            x: 0,
            y: 6,
            width: 1,
            height: 1,
            color: COLOR_RED,
            is_selection_bg: false,
        };
        harness.compile(vec![cmd]);

        harness.assert_rect_color(0, 96, 8, 4, COLOR_RED);
    }

    #[test]
    fn test_massive_coordinates() {
        let mut harness = TestHarness::default();

        let cmds = vec![
            RenderCommand::FillRect {
                x: usize::MAX,
                y: usize::MAX,
                width: 10,
                height: 10,
                color: COLOR_RED,
                is_selection_bg: false,
            },
            RenderCommand::DrawTextRun {
                x: usize::MAX,
                y: usize::MAX,
                text: "Crash?".into(),
                fg: COLOR_WHITE,
                bg: COLOR_BLACK,
                flags: AttrFlags::empty(),
                is_selected: false,
            },
        ];

        harness.compile(cmds);
    }

    #[test]
    fn test_zero_size_rasterizer() {
        let mut harness = TestHarness::new(0, 0, 0, 0);

        let cmds = vec![
            RenderCommand::ClearAll { bg: COLOR_WHITE },
            RenderCommand::DrawTextRun {
                x: 0,
                y: 0,
                text: "A".into(),
                fg: COLOR_BLACK,
                bg: COLOR_WHITE,
                flags: AttrFlags::empty(),
                is_selected: false,
            },
        ];

        harness.compile(cmds);
    }

    #[test]
    fn test_empty_text_run() {
        let mut harness = TestHarness::default();
        harness.compile(vec![RenderCommand::ClearAll { bg: COLOR_BLACK }]);

        harness.compile(vec![RenderCommand::DrawTextRun {
            x: 0,
            y: 0,
            text: "".into(),
            fg: COLOR_WHITE,
            bg: COLOR_BLACK,
            flags: AttrFlags::empty(),
            is_selected: false,
        }]);

        harness.assert_clear(COLOR_BLACK);
    }

    #[test]
    fn test_caching_behavior() {
        let mut harness = TestHarness::default();

        let cmd = RenderCommand::DrawTextRun {
            x: 0,
            y: 0,
            text: "AA".into(),
            fg: COLOR_WHITE,
            bg: COLOR_BLACK,
            flags: AttrFlags::empty(),
            is_selected: false,
        };

        harness.compile(vec![cmd]);

        let bg_rgba: Rgba = COLOR_BLACK.into();

        let mut a1_has_content = false;
        for y in 0..TEST_CELL_HEIGHT {
            for x in 0..TEST_CELL_WIDTH {
                if harness.get_pixel(x, y) != bg_rgba {
                    a1_has_content = true;
                }
            }
        }

        let mut a2_has_content = false;
        for y in 0..TEST_CELL_HEIGHT {
            for x in TEST_CELL_WIDTH..TEST_CELL_WIDTH * 2 {
                if harness.get_pixel(x, y) != bg_rgba {
                    a2_has_content = true;
                }
            }
        }

        assert!(a1_has_content, "First char failed to render");
        assert!(a2_has_content, "Second char (cached) failed to render");
    }
}
