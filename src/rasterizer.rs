// src/rasterizer.rs

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

use crate::color::Color;
use crate::glyph::AttrFlags;
use crate::platform::backends::{DriverCommand, RenderCommand};
use std::collections::HashMap;

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
}

impl SoftwareRasterizer {
    /// Create a new rasterizer with given font metrics
    pub fn new(cell_width_px: usize, cell_height_px: usize) -> Self {
        Self {
            glyph_cache: HashMap::new(),
            cell_width_px,
            cell_height_px,
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

    /// Placeholder glyph renderer (just fills with colors for now)
    fn render_placeholder_glyph(
        &self,
        _ch: char,
        fg: Rgba,
        bg: Rgba,
        flags: AttrFlags,
    ) -> RenderedGlyph {
        let width = self.cell_width_px;
        let height = self.cell_height_px;
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
    commands: Vec<RenderCommand>,
    framebuffer: &mut [u8],
    buffer_width_px: usize,
    buffer_height_px: usize,
    cell_width_px: usize,
    cell_height_px: usize,
) -> Vec<DriverCommand> {
    let mut rasterizer = SoftwareRasterizer::new(cell_width_px, cell_height_px);
    let mut driver_commands = Vec::new();

    // TODO(performance): Add software prefetching for dirty lines
    // When processing DrawTextRun commands, could use std::intrinsics::prefetch_read_data
    // to hint the CPU to load the next line's data into cache while processing current line.
    // Could reduce cache misses by 10-20% for large terminal repaints. Profile first!

    for cmd in commands {
        match cmd {
            RenderCommand::ClearAll { bg } => {
                let rgba: Rgba = bg.into();
                let color_bytes = rgba.to_bytes();
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
                let mut x_offset = x;
                for ch in text.chars() {
                    let glyph = rasterizer.render_cell(ch, fg, bg, flags);
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
                driver_commands.push(DriverCommand::Present);
            }
        }
    }

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
        let driver_commands = compile_into_buffer(commands, &mut buffer, 800, 600, 8, 16);
        assert!(driver_commands.is_empty());
    }

    #[test]
    fn test_compile_clear_all() {
        // Contract: ClearAll should fill framebuffer with background color
        let commands = vec![make_clear(Color::Named(crate::color::NamedColor::Black))];
        let mut buffer = vec![255u8; 10 * 10 * 4]; // Start with white
        compile_into_buffer(commands, &mut buffer, 10, 10, 8, 16);

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
        compile_into_buffer(commands, &mut buffer, 10, 10, 8, 16);

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
        compile_into_buffer(commands, &mut buffer, 100, 100, 8, 16);

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
        compile_into_buffer(commands, &mut buffer, 100, 100, 8, 16);

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
        let driver_commands = compile_into_buffer(commands, &mut buffer, 10, 10, 8, 16);

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
        let driver_commands = compile_into_buffer(commands, &mut buffer, 10, 10, 8, 16);

        assert_eq!(driver_commands.len(), 1);
        assert!(matches!(driver_commands[0], DriverCommand::Bell));
    }

    #[test]
    fn test_compile_present() {
        // Contract: PresentFrame should produce Present DriverCommand
        let commands = vec![RenderCommand::PresentFrame];
        let mut buffer = vec![0u8; 10 * 10 * 4];
        let driver_commands = compile_into_buffer(commands, &mut buffer, 10, 10, 8, 16);

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
        let driver_commands = compile_into_buffer(commands, &mut buffer, 100, 100, 8, 16);

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
}
