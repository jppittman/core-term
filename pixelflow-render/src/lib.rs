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
/// Rasterization logic and frame processing.
pub mod rasterizer;
/// Glyph rendering and font management.
pub mod glyph;
/// Common types and data structures used in rendering.
pub mod types;

pub use commands::Op;
pub use types::*;
pub use rasterizer::process_frame;
pub use glyph::{render_glyph_direct, get_glyph_metrics, GlyphMetrics, RenderTarget, GlyphRenderCoords, GlyphStyleOverrides};
