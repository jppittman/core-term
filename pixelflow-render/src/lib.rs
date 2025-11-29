//! # PixelFlow Render
//!
//! A high-performance, software-based rendering engine built on top of `pixelflow-core`.
//!
//! This crate provides functionality for:
//! - Rasterizing graphical primitives (rectangles, blits).
//! - Rendering text glyphs using font atlases (legacy) and Loop-Blinn vectors (v11).
//! - Managing rendering commands and types.
//! - Type-safe color format handling (Rgba, Bgra).

#![warn(missing_docs)]

/// Color format types (Rgba, Bgra) with compile-time safety.
pub mod color;
/// Rendering commands and operation definitions.
pub mod commands;
/// Framebuffer type generic over color format.
pub mod frame;
/// Loop-Blinn vector font rendering (v11.0 architecture).
pub mod glyph;
/// Rasterization logic and frame processing.
pub mod rasterizer;
/// Common types and data structures used in rendering.
pub mod types;

pub use color::{Bgra, CocoaPixel, Pixel, Rgba, WebPixel, X11Pixel};
pub use commands::Op;
pub use frame::Frame;
pub use glyph::{
    get_glyph_metrics, render_glyph_direct, GlyphMetrics, GlyphRenderCoords, GlyphStyleOverrides,
    RenderTarget,
};
pub use rasterizer::{process_frame, ScreenView, ScreenViewMut};
pub use types::*;
