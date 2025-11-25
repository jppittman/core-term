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
pub mod simd_resize;
pub mod types;

// Re-export stateless API
pub use commands::Op;
pub use rasterizer::process_frame;

// Re-export types for convenience
pub use types::{AttrFlags, Color, NamedColor};
