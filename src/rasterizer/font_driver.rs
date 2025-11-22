//! Platform-specific font loading and glyph rasterization primitives.
//!
//! This module defines the `FontDriver` trait, which provides thin FFI wrappers
//! around platform-specific font APIs (Core Text on macOS, fontconfig+freetype on Linux, etc.).

use anyhow::Result;

/// Platform-specific font driver trait.
///
/// Implementors provide thin wrappers around native font APIs, handling:
/// - Font loading by name/spec
/// - Glyph lookup for characters
/// - System font fallback queries
/// - Glyph rasterization to RGBA pixels
///
/// The `FontManager` uses this trait to implement shared caching and fallback logic.
pub trait FontDriver {
    /// Platform-specific font handle type (e.g., `CTFont`, `*mut XftFont`)
    type Font: Clone;

    /// Platform-specific glyph ID type (e.g., `CGGlyph`, `u32`)
    type GlyphId: Copy;

    /// Load a font by name and size.
    ///
    /// # Arguments
    /// * `name` - Font name (e.g., "Menlo", "Monaco", "Noto Sans Mono")
    /// * `size_pt` - Font size in points
    ///
    /// # Returns
    /// Platform-specific font handle, or error if font cannot be loaded
    fn load_font(&self, name: &str, size_pt: f64) -> Result<Self::Font>;

    /// Find a glyph for the given character in the specified font.
    ///
    /// # Arguments
    /// * `font` - Font to search in
    /// * `ch` - Character to find
    ///
    /// # Returns
    /// Some(glyph_id) if the font contains this character, None otherwise
    fn find_glyph(&self, font: &Self::Font, ch: char) -> Option<Self::GlyphId>;

    /// Query the system for a fallback font that contains the given character.
    ///
    /// # Arguments
    /// * `ch` - Character to find a font for
    ///
    /// # Returns
    /// A font that contains this character, or error if no suitable font is found
    fn find_fallback_font(&self, ch: char) -> Result<Self::Font>;

    /// Rasterize a glyph to RGBA pixel data.
    ///
    /// # Arguments
    /// * `font` - Font containing the glyph
    /// * `glyph_id` - Glyph to rasterize
    /// * `cell_width_px` - Target cell width in pixels
    /// * `cell_height_px` - Target cell height in pixels
    ///
    /// # Returns
    /// RGBA pixel data (4 bytes per pixel, row-major layout) with **straight (non-premultiplied) alpha**.
    /// The glyph should be rendered as white (255,255,255) on transparent background,
    /// with the alpha channel representing coverage.
    /// Length must be `cell_width_px * cell_height_px * 4`
    fn rasterize_glyph(
        &self,
        font: &Self::Font,
        glyph_id: Self::GlyphId,
        cell_width_px: usize,
        cell_height_px: usize,
    ) -> Vec<u8>;
}
