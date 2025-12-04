//! `pixelflow-fonts`: High-performance, functional font rasterization.
//!
//! This crate provides tools to parse TTF/OTF fonts and render glyphs as
//! analytic surfaces. It leverages `pixelflow-core` to perform SIMD-accelerated
//! evaluation of algebraic curves (Loop-Blinn), allowing for infinite resolution
//! and dynamic styling.
//!
//! # Core Concepts
//!
//! - **[`Font`]**: A handle to a parsed font file. Used to retrieve glyph geometry.
//! - **[`Glyph`]**: A resolution-independent vector representation of a character.
//!   It implements [`Surface<u8>`](pixelflow_core::traits::Surface), returning coverage values (alpha).
//! - **Combinators**: Traits like [`CurveSurfaceExt`] allow transforming glyphs
//!   (e.g., [`Bold`], [`Slant`]) before rasterization.
//!
//! # Example
//!
//! ```ignore
//! use pixelflow_fonts::{Font, glyphs};
//! use pixelflow_core::dsl::MaskExt;
//!
//! let font = Font::from_bytes(include_bytes!("../assets/font.ttf"))?;
//!
//! // Create a factory for caching baked glyphs
//! let get_glyph = glyphs(font, 16, 24);
//!
//! // Retrieve and use a glyph
//! let glyph_surface = get_glyph('A');
//! ```

pub mod combinators;
pub mod curves;
pub mod font;
pub mod glyph;

pub use combinators::{glyphs, Bold, CurveSurfaceExt, Hint, Lazy, Scale, Slant};
pub use font::{Font, FontError, FontMetrics};
pub use glyph::{CurveSurface, Glyph, GlyphBounds};
