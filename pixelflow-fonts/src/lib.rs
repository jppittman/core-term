//! pixelflow-fonts: Functional glyph factory.
//!
//! char -> Surface<u8>.

pub mod baked;
pub mod curves;
pub mod font;
pub mod glyph;
pub mod lazy;

pub use baked::{BakedExt, BakedFactory, BakedMask};
pub use font::{Font, FontError, FontMetrics};
pub use glyph::{GlyphCache, GlyphSurface, glyph, glyphs, glyphs_cached, glyphs_scaled};
pub use lazy::Lazy;
