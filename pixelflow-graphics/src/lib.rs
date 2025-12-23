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
pub use fonts::{CurveSurface, Font, FontError, FontMetrics, Glyph, GlyphBounds};

// Re-export render
pub use render::color::{
    AttrFlags, Bgra8, CocoaPixel, Color, ColorManifold, ColorMap, Lift, NamedColor, Pixel, Rgba8,
    WebPixel, X11Pixel,
};
pub use render::frame::Frame;
pub use render::rasterizer::TensorShape;

// Re-export core types for convenience
pub use pixelflow_core::{Discrete, Field, Manifold, ManifoldExt, Map, W, X, Y, Z};
