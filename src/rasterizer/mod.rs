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
use log::{debug, trace, warn};
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
struct RenderedGlyph {
    width_px: usize,
    height_px: usize,
    rgba_data: Vec<u8>, // width * height * 4 bytes
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
    /// Create a new rasterizer with given font metrics
    pub fn new(cell_width_px: usize, cell_height_px: usize) -> Self {
        use log::*;

        // Initialize platform-specific font manager
        #[cfg(target_os = "macos")]
        let font_manager = {
            let driver = CocoaFontDriver::new();
            let font_size_pt = 12.0; // TODO: Make configurable and calculate from cell height

            let manager = FontManager::new(
                driver,
                "Menlo",           // regular
                "Menlo-Bold",      // bold
                "Menlo-Italic",    // italic
                "Menlo-BoldItalic", // bold+italic
                font_size_pt,
            ).expect("Failed to initialize font manager");

            info!("SoftwareRasterizer: Initialized with FontManager (Menlo {} pt)", font_size_pt);
            Some(manager)
        };

        #[cfg(not(target_os = "macos"))]
        let font_manager = {
            warn!("SoftwareRasterizer: No font driver for this platform, using placeholder rendering");
            None
        };

        Self {
            glyph_cache: HashMap::new(),
            cell_width_px,
            cell_height_px,
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
                let glyph_pixels = font_manager.driver().rasterize_glyph(
                    font,
                    resolved.glyph_id,
                    width,
                    height,
                );

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
            debug!(
                "colorize_glyph: fg={:?}, bg={:?}",
                fg, bg
            );
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
                    bg, rgba.r, rgba.g, rgba.b, rgba.a
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
                // Rasterize each character and blit directly to framebuffer
                debug!(
                    "rasterizer: DrawTextRun '{}' at col={} row={} fg={:?} bg={:?} flags={:?}",
                    text, x, y, fg, bg, flags
                );
                let mut x_offset = x;
                for ch in text.chars() {
                    let glyph = rasterizer.render_cell(ch, fg, bg, flags);

                    // Sample pixel data for diagnostics: check if glyph has non-black pixels
                    let has_color = glyph.rgba_data.chunks_exact(4).any(|p| {
                        p[0] > 0 || p[1] > 0 || p[2] > 0 || p[3] > 0
                    });

                    debug!(
                        "rasterizer: Rendered '{}' (U+{:04X}): {}x{} pixels, {} bytes, has_color={}",
                        ch, ch as u32, glyph.width_px, glyph.height_px, glyph.rgba_data.len(), has_color
                    );

                    blit_to_framebuffer(
                        framebuffer,
                        buffer_width_px,
                        buffer_height_px,
                        &glyph.rgba_data,
                        x_offset * cell_width_px,
                        y * cell_height_px,
                        glyph.width_px,
                        glyph.height_px,
                    );
                    x_offset += 1;
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
                let x_px = x * cell_width_px;
                let y_px = y * cell_height_px;
                let width_px = width * cell_width_px;
                let height_px = height * cell_height_px;

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

    // Test helper to create basic RenderCommands
    fn make_clear(color: Color) -> RenderCommand {
        RenderCommand::ClearAll { bg: color }
    }

    fn make_text(x: usize, y: usize, text: &str, fg: Color, bg: Color) -> RenderCommand {
        RenderCommand::DrawTextRun {
            x,
            y,
            text: text.to_string(),
            fg,
            bg,
            flags: AttrFlags::empty(),
            is_selected: false,
        }
    }

    fn make_rect(x: usize, y: usize, w: usize, h: usize, color: Color) -> RenderCommand {
        RenderCommand::FillRect {
            x,
            y,
            width: w,
            height: h,
            color,
            is_selection_bg: false,
        }
    }

    #[test]
    fn test_compile_empty() {
        // Contract: Empty input should produce empty DriverCommand list
        let commands = vec![];
        let mut buffer = vec![0u8; 800 * 600 * 4];
        let mut rasterizer = SoftwareRasterizer::new(8, 16);
        let driver_commands = compile_into_buffer(&mut rasterizer, commands, &mut buffer, 800, 600, 8, 16);
        assert!(driver_commands.is_empty());
    }

    #[test]
    fn test_compile_clear_all() {
        // Contract: ClearAll should fill framebuffer with background color
        let commands = vec![make_clear(Color::Named(crate::color::NamedColor::Black))];
        let mut buffer = vec![255u8; 10 * 10 * 4]; // Start with white
        let mut rasterizer = SoftwareRasterizer::new(8, 16);
        compile_into_buffer(&mut rasterizer, commands, &mut buffer, 10, 10, 8, 16);

        // Verify all pixels are black
        for pixel in buffer.chunks_exact(4) {
            assert_eq!(pixel, &[0, 0, 0, 255]); // Black
        }
    }

    #[test]
    fn test_compile_clear_all_white() {
        // Contract: Different colors fill correctly
        let commands = vec![make_clear(Color::Named(crate::color::NamedColor::White))];
        let mut buffer = vec![0u8; 10 * 10 * 4]; // Start with black
        let mut rasterizer = SoftwareRasterizer::new(8, 16);
        compile_into_buffer(&mut rasterizer, commands, &mut buffer, 10, 10, 8, 16);

        // Verify all pixels are white
        for pixel in buffer.chunks_exact(4) {
            assert_eq!(pixel, &[229, 229, 229, 255]); // White
        }
    }

    #[test]
    fn test_compile_text_writes_to_buffer() {
        // Contract: DrawTextRun should write pixels to framebuffer at correct position
        let commands = vec![make_text(
            0,
            0,
            "A",
            Color::Named(crate::color::NamedColor::White),
            Color::Named(crate::color::NamedColor::Black),
        )];
        let mut buffer = vec![0u8; 100 * 100 * 4];
        let mut rasterizer = SoftwareRasterizer::new(8, 16);
        compile_into_buffer(&mut rasterizer, commands, &mut buffer, 100, 100, 8, 16);

        // Just verify something was written (placeholder rasterizer fills with bg)
        // Real test would verify actual glyph rendering
        assert!(buffer.iter().any(|&b| b != 0));
    }

    #[test]
    fn test_compile_fill_rect() {
        // Contract: FillRect should write correct color to framebuffer region
        let commands = vec![make_rect(
            0,
            0,
            2,
            2,
            Color::Named(crate::color::NamedColor::Red),
        )];
        let mut buffer = vec![0u8; 100 * 100 * 4];
        let mut rasterizer = SoftwareRasterizer::new(8, 16);
        compile_into_buffer(&mut rasterizer, commands, &mut buffer, 100, 100, 8, 16);

        // Verify some pixels in the filled region are red
        // The rect is 2x2 cells = 16x32 pixels starting at (0,0)
        for y in 0..32 {
            for x in 0..16 {
                let idx = (y * 100 + x) * 4;
                assert_eq!(&buffer[idx..idx + 4], &[205, 49, 49, 255]); // Red
            }
        }
    }

    #[test]
    fn test_compile_set_title() {
        // Contract: SetWindowTitle should produce SetTitle DriverCommand
        let commands = vec![RenderCommand::SetWindowTitle {
            title: "Test Terminal".to_string(),
        }];
        let mut buffer = vec![0u8; 10 * 10 * 4];
        let mut rasterizer = SoftwareRasterizer::new(8, 16);
        let driver_commands = compile_into_buffer(&mut rasterizer, commands, &mut buffer, 10, 10, 8, 16);

        assert_eq!(driver_commands.len(), 1);
        match &driver_commands[0] {
            DriverCommand::SetTitle { title } => {
                assert_eq!(title, "Test Terminal");
            }
            _ => panic!("Expected SetTitle command"),
        }
    }

    #[test]
    fn test_compile_bell() {
        // Contract: RingBell should produce Bell DriverCommand
        let commands = vec![RenderCommand::RingBell];
        let mut buffer = vec![0u8; 10 * 10 * 4];
        let mut rasterizer = SoftwareRasterizer::new(8, 16);
        let driver_commands = compile_into_buffer(&mut rasterizer, commands, &mut buffer, 10, 10, 8, 16);

        assert_eq!(driver_commands.len(), 1);
        assert!(matches!(driver_commands[0], DriverCommand::Bell));
    }

    #[test]
    fn test_compile_present() {
        // Contract: PresentFrame should produce Present DriverCommand
        let commands = vec![RenderCommand::PresentFrame];
        let mut buffer = vec![0u8; 10 * 10 * 4];
        let mut rasterizer = SoftwareRasterizer::new(8, 16);
        let driver_commands = compile_into_buffer(&mut rasterizer, commands, &mut buffer, 10, 10, 8, 16);

        assert_eq!(driver_commands.len(), 1);
        assert!(matches!(driver_commands[0], DriverCommand::Present));
    }

    #[test]
    fn test_compile_mixed_commands() {
        // Contract: Multiple commands should process in order
        let commands = vec![
            make_clear(Color::Named(crate::color::NamedColor::Black)),
            make_text(
                0,
                0,
                "Hi",
                Color::Named(crate::color::NamedColor::Green),
                Color::Named(crate::color::NamedColor::Black),
            ),
            RenderCommand::RingBell,
            RenderCommand::PresentFrame,
        ];
        let mut buffer = vec![0u8; 100 * 100 * 4];
        let mut rasterizer = SoftwareRasterizer::new(8, 16);
        let driver_commands = compile_into_buffer(&mut rasterizer, commands, &mut buffer, 100, 100, 8, 16);

        // Only metadata commands returned (no pixel data in commands)
        assert_eq!(driver_commands.len(), 2); // Bell + Present
        assert!(matches!(driver_commands[0], DriverCommand::Bell));
        assert!(matches!(driver_commands[1], DriverCommand::Present));

        // Framebuffer should be modified (black clear + green text)
        assert!(buffer.iter().any(|&b| b != 0));
    }

    #[test]
    fn test_rgba_conversion_named_colors() {
        // Contract: Named colors convert to correct RGBA values
        let test_cases = vec![
            (crate::color::NamedColor::Black, (0, 0, 0, 255)),
            (crate::color::NamedColor::White, (229, 229, 229, 255)),
            (crate::color::NamedColor::Red, (205, 49, 49, 255)),
            (crate::color::NamedColor::Green, (13, 188, 121, 255)),
        ];

        for (named, expected) in test_cases {
            let rgba: Rgba = Color::Named(named).into();
            assert_eq!((rgba.r, rgba.g, rgba.b, rgba.a), expected);
        }
    }

    #[test]
    fn test_rgba_conversion_rgb() {
        // Contract: RGB colors convert correctly
        let rgba: Rgba = Color::Rgb(100, 150, 200).into();
        assert_eq!((rgba.r, rgba.g, rgba.b, rgba.a), (100, 150, 200, 255));
    }

    #[test]
    fn test_rgba_conversion_indexed() {
        // Contract: Indexed colors convert to valid RGBA
        let rgba: Rgba = Color::Indexed(1).into(); // Should be red
        assert_eq!((rgba.r, rgba.g, rgba.b, rgba.a), (205, 49, 49, 255));

        // Test grayscale range (232-255)
        let rgba: Rgba = Color::Indexed(232).into();
        assert_eq!(rgba.a, 255); // Alpha should always be 255
    }

    #[test]
    fn test_render_cell_returns_non_empty_glyph() {
        // Contract: render_cell should produce RGBA pixels with non-transparent content
        let mut rasterizer = SoftwareRasterizer::new(8, 16);
        let cell_size = rasterizer.cell_size();

        let glyph = rasterizer.render_cell(
            'A',
            Color::Named(crate::color::NamedColor::White),
            Color::Named(crate::color::NamedColor::Black),
            AttrFlags::empty(),
        );

        assert_eq!(glyph.rgba_data.len(), cell_size.0 * cell_size.1 * 4);

        // Check that the glyph has some non-fully-transparent pixels
        let has_content = glyph.rgba_data.chunks_exact(4).any(|p| {
            // Not fully transparent AND not pure black background
            p[3] < 255 || p[0] > 10 || p[1] > 10 || p[2] > 10
        });

        assert!(has_content, "Rendered cell should have visible content (non-black pixels)");
    }

    #[test]
    fn test_render_cell_white_on_black_produces_bright_pixels() {
        // Contract: White foreground on black background should produce bright pixels
        let mut rasterizer = SoftwareRasterizer::new(8, 16);

        let glyph = rasterizer.render_cell(
            'A',
            Color::Rgb(255, 255, 255), // White foreground
            Color::Rgb(0, 0, 0),        // Black background
            AttrFlags::empty(),
        );

        // Check for bright pixels (R, G, or B > 200)
        let has_bright_pixels = glyph.rgba_data.chunks_exact(4).any(|p| {
            p[0] > 200 || p[1] > 200 || p[2] > 200
        });

        assert!(has_bright_pixels, "White-on-black glyph should have bright pixels, not all black");
    }

    #[test]
    fn test_render_cell_different_colors_produce_different_output() {
        // Contract: Different foreground colors should produce different colored glyphs
        let mut rasterizer = SoftwareRasterizer::new(8, 16);

        let white_data = rasterizer.render_cell(
            'A',
            Color::Rgb(255, 255, 255),
            Color::Rgb(0, 0, 0),
            AttrFlags::empty(),
        ).rgba_data.clone();

        let red_data = rasterizer.render_cell(
            'A',
            Color::Rgb(255, 0, 0),
            Color::Rgb(0, 0, 0),
            AttrFlags::empty(),
        ).rgba_data.clone();

        // The glyphs should be different (different colors)
        assert_ne!(white_data, red_data, "Different foreground colors should produce different output");
    }

    #[test]
    fn test_render_cell_caching_works() {
        // Contract: render_cell should cache glyphs for repeated requests
        let mut rasterizer = SoftwareRasterizer::new(8, 16);

        let data1 = rasterizer.render_cell(
            'A',
            Color::Named(crate::color::NamedColor::White),
            Color::Named(crate::color::NamedColor::Black),
            AttrFlags::empty(),
        ).rgba_data.clone();

        let data2 = rasterizer.render_cell(
            'A',
            Color::Named(crate::color::NamedColor::White),
            Color::Named(crate::color::NamedColor::Black),
            AttrFlags::empty(),
        ).rgba_data.clone();

        // Both should be identical (from cache)
        assert_eq!(data1, data2);
        assert!(!data1.is_empty());
    }

    #[test]
    fn test_colorize_glyph_produces_correct_blend() {
        // Contract: colorize_glyph should blend white glyph with fg/bg colors
        // White pixel with 50% alpha on white glyph
        let white_glyph = vec![255, 255, 255, 128]; // 50% alpha

        let fg = Rgba { r: 255, g: 0, b: 0, a: 255 }; // Red
        let bg = Rgba { r: 0, g: 0, b: 0, a: 255 };   // Black

        let result = SoftwareRasterizer::colorize_glyph(&white_glyph, fg, bg);

        // Should blend to 50% red, 50% black = (128, 0, 0, 255)
        assert_eq!(result.len(), 4);
        assert!(result[0] > 100 && result[0] < 150, "Red channel should be ~128");
        assert_eq!(result[1], 0, "Green should be 0");
        assert_eq!(result[2], 0, "Blue should be 0");
        assert_eq!(result[3], 255, "Alpha should be 255");
    }

    #[test]
    fn test_draw_text_run_actually_draws_pixels() {
        // This test is designed to fail if the rasterizer produces only background pixels.
        let cell_width = 8;
        let cell_height = 16;
        let buffer_width = cell_width * 10;
        let buffer_height = cell_height * 1;

        let mut rasterizer = SoftwareRasterizer::new(cell_width, cell_height);
        let mut buffer = vec![0u8; buffer_width * buffer_height * 4]; // Black buffer

        let commands = vec![make_text(
            0,
            0,
            "A",
            Color::Rgb(255, 255, 255), // White
            Color::Rgb(0, 0, 0),       // Black
        )];

        compile_into_buffer(
            &mut rasterizer,
            commands,
            &mut buffer,
            buffer_width,
            buffer_height,
            cell_width,
            cell_height,
        );

        // If the screen is black, this test will fail.
        // We expect that *some* pixels in the buffer are not the background color.
        let bg_rgba = Rgba::from(Color::Rgb(0,0,0)).to_bytes();
        let has_non_bg_pixels = buffer.chunks_exact(4).any(|p| p != &bg_rgba);

        assert!(has_non_bg_pixels, "compile_into_buffer with DrawTextRun should modify the framebuffer with non-background pixels.");
    }

    #[test]
    fn test_rasterizer_full_render_cycle() {
        // This test simulates a full render cycle, checking if a complex scene
        // results in a correctly modified framebuffer and appropriate driver commands.
        // It adheres to the public API contract of the rasterizer module.
        let cell_width = 8;
        let cell_height = 16;
        let buffer_width = cell_width * 80;
        let buffer_height = cell_height * 24;

        let mut rasterizer = SoftwareRasterizer::new(cell_width, cell_height);
        let mut buffer = vec![0u8; buffer_width * buffer_height * 4]; // Black buffer

        let mut commands = Vec::new();
        // Simulate a screen full of text
        for y in 0..24 {
            commands.push(make_text(
                0,
                y,
                &format!("Line {}", y),
                Color::Named(crate::color::NamedColor::White),
                Color::Named(crate::color::NamedColor::Black),
            ));
        }
        commands.push(RenderCommand::SetWindowTitle { title: "Test".to_string() });
        commands.push(RenderCommand::PresentFrame);


        let driver_commands = compile_into_buffer(
            &mut rasterizer,
            commands,
            &mut buffer,
            buffer_width,
            buffer_height,
            cell_width,
            cell_height,
        );

        // 1. Check for driver commands
        assert_eq!(driver_commands.len(), 2, "Should produce SetTitle and Present commands");
        assert!(matches!(driver_commands[0], DriverCommand::SetTitle { .. }));
        assert!(matches!(driver_commands[1], DriverCommand::Present));

        // 2. Check if framebuffer was modified
        let bg_rgba = Rgba::from(Color::Named(crate::color::NamedColor::Black)).to_bytes();
        let has_non_bg_pixels = buffer.chunks_exact(4).any(|p| p != &bg_rgba);
        assert!(has_non_bg_pixels, "Framebuffer should contain non-background pixels after rendering text.");

        // 3. Optional: More detailed check for a specific pixel
        // Check a pixel where we expect text to be.
        // For 'L' in "Line 0" at (0,0)
        // Let's assume top-left pixel of the glyph is not transparent.
        let first_pixel = &buffer[0..4];
        assert_ne!(first_pixel, &bg_rgba, "First pixel of a glyph should not be background color.");
    }
}
