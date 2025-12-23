//! # PixelFlow Graphics
//!
//! Consolidated graphics crate merging rendering and font logic.

pub mod fonts;
pub mod image;
pub mod render;
pub mod shapes;
pub mod transform;

pub use transform::Scale;

// Re-export fonts
// TODO: These need to be fixed once combinators.rs is updated
// pub use fonts::combinators::{glyphs, Bold, CurveScale, CurveSurfaceExt, Hint, Lazy, Slant};
pub use fonts::font::{Font, FontError, FontMetrics};
// TODO: Update glyph exports once glyph.rs is fixed
// pub use fonts::glyph::{CurveSurface, Glyph, GlyphBounds};

// Re-export render
pub use render::color::{
    AttrFlags, Bgra8, CocoaPixel, Color, ColorManifold, ColorMap, Lift, NamedColor, Pixel, Rgba8,
    WebPixel, X11Pixel,
};
pub use render::frame::Frame;
pub use render::rasterizer::TensorShape;

// Re-export core types for convenience
pub use pixelflow_core::{Discrete, Field, Manifold, ManifoldExt, Map, W, X, Y, Z};
