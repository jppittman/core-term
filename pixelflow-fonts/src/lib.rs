//! pixelflow-fonts: Functional glyph factory.
//!
//! char -> Surface<u8>.

pub mod curves;
pub mod font;
pub mod glyph;
pub mod lazy;
pub mod surface;

pub use font::{Font, FontError, FontMetrics};
pub use glyph::{glyph, glyphs, glyphs_scaled};
pub use lazy::Lazy;
pub use surface::CurveSurface;
