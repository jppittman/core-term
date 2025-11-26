//! # PixelFlow Render
//!
//! Stateless pixel rendering with SIMD acceleration.
//!
//! ## Design Philosophy
//!
//! PixelFlow Render provides a single pure function: `process_frame()`.
//! It transforms a framebuffer based on a list of operations, using SIMD
//! instructions for maximum performance.
//!
//! **Key principles:**
//! - **Stateless**: No caching, no configuration, just pure data transformation
//! - **Fast**: CPU is fast, memory is slow. Decompress glyphs on-the-fly.
//! - **Simple**: One function, three operation types (Clear/Blit/Text)
//! - **SIMD**: Uses pixelflow-core for cross-platform acceleration
//!
//! ## Architecture
//!
//! ```text
//! Commands (Op[])  →  process_frame()  →  Pixels (framebuffer)
//!     (what)             (how)              (result)
//! ```
//!
//! ## Example
//!
//! ```ignore
//! use pixelflow_render::{Op, process_frame};
//!
//! let mut fb = vec![0u32; 1920 * 1080];
//! let ops = [
//!     Op::Clear { color: 0xFF000000 },
//!     Op::Text { ch: 'A', x: 0, y: 0, fg: 0xFFFFFFFF, bg: 0xFF000000 },
//! ];
//! process_frame(&mut fb, 1920, 1080, 12, 24, &ops);
//! ```

pub mod commands;
pub mod glyph;
pub mod rasterizer;
mod shader_impl;
pub mod simd_resize;
pub mod types;

// Re-export stateless API
pub use commands::Op;
pub use rasterizer::process_frame;

// Re-export types for convenience
pub use types::{AttrFlags, Color, NamedColor};

// Type aliases for ergonomic image views
pub type Frame<'a> = pixelflow_core::TensorView<'a, u8>;
pub type MutFrame<'a> = pixelflow_core::TensorViewMut<'a, u8>;

// Shader parameter types and rendering (GPU-style rendering)
pub mod shader {
    pub use super::shader_impl::render_glyph;
    /// Fixed-point coordinate projection (16.16 format).
    ///
    /// Maps destination coordinates to source coordinates:
    /// `src_coord = start + dst_coord * step`
    #[derive(Debug, Copy, Clone)]
    pub struct Projection {
        pub start: u32,  // Starting coordinate in 16.16 fixed-point
        pub step: u32,   // Step per pixel in 16.16 fixed-point
    }

    impl Projection {
        /// Create identity projection (1:1 mapping).
        pub fn identity() -> Self {
            Self {
                start: 0,
                step: 1 << 16,
            }
        }

        /// Create scaling projection from source to destination size.
        pub fn scale(src_size: usize, dst_size: usize) -> Self {
            let step = ((src_size as u32) << 16) / dst_size as u32;
            Self { start: 0, step }
        }
    }

    /// Font weight for rendering.
    #[derive(Debug, Copy, Clone, PartialEq, Eq)]
    pub enum FontWeight {
        Normal,
        Bold,
    }

    /// Visual style for glyph rendering.
    #[derive(Debug, Copy, Clone)]
    pub struct GlyphStyle {
        pub fg: u32,           // Foreground color (ARGB)
        pub bg: u32,           // Background color (ARGB)
        pub weight: FontWeight,
    }

    /// Complete parameters for glyph shader.
    #[derive(Debug, Copy, Clone)]
    pub struct GlyphParams {
        pub style: GlyphStyle,
        pub x_proj: Projection,
        pub y_proj: Projection,
    }
}

// Re-export shader types
pub use shader::{FontWeight, GlyphParams, GlyphStyle, Projection};
