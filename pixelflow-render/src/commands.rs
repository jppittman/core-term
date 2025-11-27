//! Stateless rendering operations.
//!
//! These operations represent low-level primitives for the stateless renderer.
//! They are platform-agnostic and operate directly on framebuffer memory.

use crate::types::Color;

/// Low-level rendering operations for stateless processing.
///
/// The generic parameter `T` allows passing pixel data by reference or value
/// (e.g., `Vec<u8>` or `&[u8]`).
#[derive(Debug, Clone)]
pub enum Op<T: AsRef<[u8]>> {
    /// Clear the entire framebuffer to a single color.
    Clear {
        /// The color to clear with.
        color: Color,
    },

    /// Blit raw RGBA pixel data to the framebuffer.
    Blit {
        /// RGBA pixel data (4 bytes per pixel).
        data: T,
        /// Width of the source data in pixels.
        w: usize,
        /// X coordinate in framebuffer (pixels).
        x: usize,
        /// Y coordinate in framebuffer (pixels).
        y: usize,
    },

    /// Render a single character glyph from the built-in font.
    Text {
        /// Character to render.
        ch: char,
        /// X coordinate in pixels (top-left).
        x: usize,
        /// Y coordinate in pixels (top-left).
        y: usize,
        /// Foreground color.
        fg: Color,
        /// Background color.
        bg: Color,
    },
}
