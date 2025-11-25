//! Stateless rendering operations.
//!
//! These operations represent low-level primitives for the stateless renderer.
//! They are platform-agnostic and operate directly on framebuffer memory.

use crate::types::Color;

/// Low-level rendering operations for stateless processing.
///
/// The generic parameter `T` allows passing pixel data by reference or value.
#[derive(Debug, Clone)]
pub enum Op<T: AsRef<[u8]>> {
    /// Clear the entire framebuffer to a single color.
    Clear { color: Color },

    /// Blit raw RGBA pixel data to the framebuffer.
    ///
    /// # Parameters
    /// - `data`: RGBA pixel data (4 bytes per pixel)
    /// - `w`: Width of the data in pixels
    /// - `x`: X coordinate in framebuffer (pixels)
    /// - `y`: Y coordinate in framebuffer (pixels)
    Blit {
        data: T,
        w: usize,
        x: usize,
        y: usize,
    },

    /// Render a single character glyph from ROM.
    ///
    /// # Parameters
    /// - `ch`: Character to render
    /// - `x`: X coordinate in pixels
    /// - `y`: Y coordinate in pixels
    /// - `fg`: Foreground color (semantic Color enum)
    /// - `bg`: Background color (semantic Color enum)
    Text {
        ch: char,
        x: usize,
        y: usize,
        fg: Color,
        bg: Color,
    },
}
