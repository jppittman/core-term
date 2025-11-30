//! pixelflow-fonts: TTF parsing and glyph Surface generation.
//!
//! The pixelflow way: `font.glyph('A', 24.0)` returns a `Surface<u8>`.
//!
//! ```ignore
//! use pixelflow_fonts::Font;
//! use pixelflow_core::dsl::MaskExt;
//!
//! let font = Font::from_bytes(font_data)?;
//! let glyph = font.glyph('A', 24.0)?;  // Surface<u8>
//! let rendered = glyph.over(fg, bg);   // Surface<u32>
//! ```

pub mod atlas;
pub mod curves;
pub mod font;
pub mod glyph;

pub use atlas::{Atlas, AtlasConfig, AtlasEntry, AtlasSampler};
pub use font::{Font, FontError, FontMetrics};
pub use glyph::{Glyph, GlyphBounds};
