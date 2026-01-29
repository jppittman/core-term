//! # Font Rendering Pipeline
//!
//! Bridges vector font formats (TTF/OTF) to rasterized glyphs as manifolds.
//!
//! ## Architecture: Four Layers
//!
//! ```text
//! Application Layer (CachedText)
//!      ↓
//!      │  High-level text rendering with layout and caching
//!      │
//! Rasterization Layer (GlyphCache)
//!      ↓
//!      │  Lazy glyph caching per size
//!      │
//! Font Layer (Font, Glyph)
//!      ↓
//!      │  TTF/OTF parsing and glyph decomposition
//!      │
//! Loading Layer (FontSource, LoadedFont)
//!      ↓
//! In-Memory Font Data
//! ```
//!
//! ## Layer 1: Font Loading (`loader` module)
//!
//! The entry point for font data. Supports multiple sources:
//!
//! ### FontSource Variants
//!
//! - **Embedded**: Fonts baked into the binary (no I/O)
//! - **Filesystem**: Load from disk at runtime
//! - **Memory-mapped**: Zero-copy file access via mmap
//!
//! ```ignore
//! use pixelflow_graphics::fonts::{Font, FontSource};
//!
//! // Load embedded font (fast, no I/O)
//! let font = Font::from_source(FontSource::Embedded)?;
//!
//! // Load from file (slower, but flexible)
//! let font = Font::from_source(FontSource::Filesystem {
//!     path: "/path/to/font.ttf".into(),
//! })?;
//! ```
//!
//! ## Layer 2: Glyph Decomposition (`ttf` module)
//!
//! Parses TTF/OTF files and decomposes glyphs into curves.
//!
//! ### Glyph Structure
//!
//! Each glyph contains:
//! - **Contours**: Outlines composed of quadratic Bézier curves (TrueType) or cubic Bézier curves (PostScript)
//! - **Metrics**: Advance width, bounding box, bearing
//! - **Rasterization state**: Hints and grid-fitting instructions (optional)
//!
//! ### Font Trait
//!
//! ```ignore
//! pub trait Font {
//!     fn glyph(&self, codepoint: char) -> Option<Glyph>;
//!     fn glyph_metrics(&self, codepoint: char) -> Option<Metrics>;
//!     fn baseline_offset(&self) -> i32;
//! }
//! ```
//!
//! The `Font` trait provides:
//! - **Glyph lookup**: Get the vector outline for a character
//! - **Metrics**: Advance width and bounding box
//! - **Baseline**: Vertical offset for proper alignment
//!
//! ## Layer 3: Glyph Rasterization (`cache` module)
//!
//! Glyphs are expensive to rasterize (curves → pixels). The cache stores rasterized glyphs per size.
//!
//! ### GlyphCache
//!
//! Stores a 2D grid of rasterized glyphs, keyed by character and size.
//!
//! ```ignore
//! use pixelflow_graphics::GlyphCache;
//!
//! let cache = GlyphCache::new();
//! let rasterized = cache.get_or_rasterize('A', &font, 16)?;
//! // Returns: height_px, width_px, pixels
//! ```
//!
//! ### Caching Strategy
//!
//! - **Lazy**: Glyphs are rasterized on first use
//! - **Per-size**: Each font size gets its own cache (common for terminals)
//! - **Exponential growth**: Cache grows as needed; never shrinks
//! - **Memory-efficient**: Typical terminal use (ANSI colors + 100 glyphs) uses ~1MB per size
//!
//! ## Layer 4: Text Layout (`text` module and `CachedText`)
//!
//! Combines font, cache, and manifold composition for high-level text rendering.
//!
//! ### CachedText
//!
//! The primary interface for rendering text. It:
//! - **Caches glyphs**: Via internal `GlyphCache`
//! - **Composes manifolds**: Each glyph position gets a manifold that samples the cache
//! - **Handles layout**: Advance widths, kerning (if available), baseline alignment
//!
//! ```ignore
//! use pixelflow_graphics::CachedText;
//!
//! let text = CachedText::new(
//!     "Hello, World!",
//!     &font,
//!     size,
//!     foreground_color,
//!     background_color,
//! )?;
//!
//! // Text is now a Manifold<Output = Discrete> (a color manifold)
//! // Can be rendered directly or composed with other manifolds
//! ```
//!
//! ## Rendering Pipeline
//!
//! Typical usage flow:
//!
//! 1. **Load font** (once): `Font::from_source(FontSource::...)?`
//! 2. **Create text manifold** (per string): `CachedText::new(text, &font, ...)?`
//! 3. **Render** (every frame): `execute(&text_manifold, &mut framebuffer, shape)`
//!
//! The glyph cache is shared across all `CachedText` instances using the same font and size,
//! so repeated text doesn't re-rasterize glyphs.
//!
//! ## Integration with Colors
//!
//! `CachedText` produces manifolds with `Output = Discrete` (RGBA pixels), so they compose
//! seamlessly with color manifolds and other effects:
//!
//! ```ignore
//! use pixelflow_graphics::{CachedText, ColorCube};
//! use pixelflow_core::combinators::At;
//!
//! let text = CachedText::new(...)?;
//! let background = At { inner: ColorCube, x: 0.2, y: 0.2, z: 0.2, w: 1.0 };  // Dark gray
//!
//! // Compose: text over background
//! let scene = text.select(background);  // TextOn{text, background}
//! ```
//!
//! ## Performance Characteristics
//!
//! | Operation | Time | Cache Status |
//! |-----------|------|--------------|
//! | Load font | ~1-10ms | Single per app |
//! | First render (new size) | ~10-50ms per unique char | Cache miss |
//! | Second render (same size) | ~0.1ms per pixel | Cache hit (SIMD) |
//!
//! For a 1080p terminal with ~20 unique characters per frame, rasterization is negligible (<1% of frame time).
//!
//! ## Memory Layout: Glyphs as Manifolds
//!
//! A `CachedText` manifold doesn't store pixel data directly. Instead:
//! 1. It stores the glyph cache (shared across all text instances)
//! 2. At evaluation time, it samples the cache at the requested coordinates
//! 3. The compiler fuses this sampling into the main SIMD loop
//!
//! Result: No intermediate framebuffer, no memory copies. Text is rasterized directly into the final output.
//!
//! ## Supported Formats
//!
//! - **TTF** (TrueType): Quadratic Bézier curves
//! - **OTF** (OpenType): Cubic Bézier curves (via subsetting)
//! - **Variable fonts**: Supported if font has variation axes
//!
pub mod cache;
pub mod combinators;
pub mod loader;
pub mod text;
pub mod ttf;
pub mod ttf_curve_analytical;

// Re-export font types (user-facing only; internal geometry types stay in ttf module)
pub use ttf::{Font, Glyph};

// Re-export loader types
pub use loader::{DataSource, EmbeddedSource, FontSource, LoadedFont, MmapSource};

// Re-export text
pub use text::text;

// Re-export cache
pub use cache::{CachedGlyph, CachedText, GlyphCache};
