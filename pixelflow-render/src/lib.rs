//! # PixelFlow Render
//!
//! A high-performance, software-based rendering engine built on top of `pixelflow-core`.
//!
//! This crate provides functionality for:
//! - Rasterizing graphical primitives (rectangles, blits).
//! - Rendering text glyphs using font atlases.
//! - Managing rendering commands and types.

#![warn(missing_docs)]

/// Rendering commands and operation definitions.
pub mod commands;
/// Glyph rendering and font management.
pub mod glyph;
/// Rasterization logic and frame processing.
pub mod rasterizer;
/// Common types and data structures used in rendering.
pub mod types;

pub use commands::Op;
pub use glyph::{
    get_glyph_metrics, render_glyph_direct, GlyphMetrics, GlyphRenderCoords, GlyphStyleOverrides,
    RenderTarget,
};
pub use rasterizer::process_frame;
pub use types::*;
