//! # PixelFlow Render
//!
//! A high-performance, software-based rendering engine built on top of `pixelflow-core`.
//!
//! This crate provides functionality for:
//! - Rasterizing graphical primitives (rectangles, blits).
//! - Font access via pixelflow-fonts (glyphs are Surfaces).
//! - Managing rendering commands and types.
//! - Type-safe color format handling (Rgba, Bgra).

#![warn(missing_docs)]

/// Unified color types: semantic colors (Color, NamedColor), text attributes, and pixel formats.
pub mod color;
/// Rendering commands and operation definitions.
pub mod commands;
/// Framebuffer type generic over color format.
pub mod frame;
/// Embedded font access.
pub mod glyph;
/// Rasterization logic and frame processing.
pub mod rasterizer;

pub use color::{
    // Semantic colors
    AttrFlags, Color, NamedColor,
    // Pixel formats
    Bgra, CocoaPixel, Pixel, Rgba, WebPixel, X11Pixel,
};
pub use commands::Op;
pub use frame::Frame;
pub use glyph::font;
pub use pixelflow_fonts::{Font, Glyph, GlyphBounds};
pub use rasterizer::{process_frame, ScreenView, ScreenViewMut};
