//! pixelflow-fonts: TTF parsing and glyph Surface generation.
//!
//! The pixelflow way: `font.glyph('A', 24.0)` returns a `Surface<u8>`.
//!
//! ```ignore
//! use pixelflow_fonts::{Font, glyphs};
//! use pixelflow_core::dsl::MaskExt;
//!
//! let font = Font::from_bytes(font_data)?;
//! let glyph_factory = glyphs(font, 12, 16);
//! let glyph = glyph_factory('A');  // Lazy<Baked<u8>>
//! let rendered = glyph.over(fg, bg);   // Surface<u32>
//! ```

pub mod curves;
pub mod font;
pub mod glyph;
pub mod combinators;

pub use font::{Font, FontError, FontMetrics};
pub use glyph::{Glyph, GlyphBounds, CurveSurface};
pub use combinators::{Lazy, glyphs, Hint, Bold, Slant, Scale, CurveSurfaceExt};
