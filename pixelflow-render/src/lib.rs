//! # PixelFlow Render
//!
//! A high-performance, software-based rendering engine built on top of `pixelflow-core`.
//!
//! This crate provides functionality for:
//! - Rasterizing graphical primitives (rectangles, blits).
//! - Rendering text glyphs using font atlases (legacy) and Loop-Blinn vectors (v11).
//! - Managing rendering commands and types.

#![warn(missing_docs)]

/// Rendering commands and operation definitions.
pub mod commands;
/// Loop-Blinn vector font rendering (v11.0 architecture).
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
pub use rasterizer::{process_frame, ScreenView, ScreenViewMut};
pub use types::*;
