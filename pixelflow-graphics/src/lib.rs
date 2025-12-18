//! # PixelFlow Graphics
//!
//! Consolidated graphics crate merging rendering and font logic.

pub mod fonts;
pub mod primitives;
pub mod render;

// Re-export fonts
pub use fonts::combinators::{glyphs, Bold, CurveScale, CurveSurfaceExt, Hint, Lazy, Slant};
pub use fonts::font::{Font, FontError, FontMetrics};
pub use fonts::glyph::{CurveSurface, Glyph, GlyphBounds};

// Re-export render
pub use render::color::{
    AttrFlags, Bgra, CocoaPixel, Color, NamedColor, Pixel, Rgba, WebPixel, X11Pixel,
};
pub use render::frame::Frame;
pub use render::glyph::{font, gamma_decode, gamma_encode, subpixel, SubpixelBlend, SubpixelMap};
pub use render::rasterizer::{
    bake, execute, execute_stripe, render, render_pixel, render_to_buffer, render_u32, Rasterize,
    Stripe, TensorShape,
};

// Re-export core types for convenience
pub use pixelflow_core::{traits::Surface, Scale};
