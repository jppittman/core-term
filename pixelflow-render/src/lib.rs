//! # PixelFlow Render
//!
//! A high-performance, software-based rendering engine built on top of `pixelflow-core`.
//!
//! ## Architecture
//!
//! Everything is a lazy, infinite Surface until materialization:
//! - `Frame<P>` is both a target (write into) AND a Surface (read from)
//! - Colors (`Rgba`, `Bgra`) are constant Surfaces
//! - Compose with `Over`, `Offset`, `Skew`, `Max`, etc.
//! - Materialize via `execute()` or `render()`
//!
//! ## Example
//! ```ignore
//! use pixelflow_render::{Frame, Rgba, render};
//! use pixelflow_core::dsl::MaskExt;
//!
//! let mut frame = Frame::<Rgba>::new(800, 600);
//! let fg = Rgba::new(255, 0, 0, 255);
//! let bg = Rgba::new(0, 0, 255, 255);
//! let surface = glyph_mask.over::<Rgba>(fg, bg);
//! render(&surface, &mut frame);
//! ```

#![warn(missing_docs)]

/// Unified color types: semantic colors (Color, NamedColor), text attributes, and pixel formats.
pub mod color;
/// Framebuffer type generic over color format.
pub mod frame;
/// Embedded font access.
pub mod glyph;
/// Rasterization utilities.
pub mod rasterizer;

pub use color::{
    // Semantic colors
    AttrFlags,
    // Pixel formats
    Bgra,
    CocoaPixel,
    Color,
    NamedColor,
    Pixel,
    PlatformPixel,
    Rgba,
    WebPixel,
    X11Pixel,
};
pub use frame::Frame;
pub use glyph::{font, gamma_decode, gamma_encode, subpixel, SubpixelBlend, SubpixelMap};
pub use pixelflow_fonts::{Font, Glyph, GlyphBounds};
pub use rasterizer::{execute, render, render_pixel, render_to_buffer, render_u32};

// Re-export core types for convenience
pub use pixelflow_core::traits::Surface;
