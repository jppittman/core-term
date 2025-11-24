//! Platform-specific font loading and glyph rasterization primitives.
//!
//! This module defines the `FontDriver` trait, which provides thin FFI wrappers
//! around platform-specific font APIs (Core Text on macOS, fontconfig+freetype on Linux, etc.).

use anyhow::Result;

/// Opaque font ID used to reference loaded fonts.
/// The FontDriver maintains an internal mapping from IDs to platform-specific font handles.
pub type FontId = usize;

/// Platform-specific font driver trait (object-safe version).
///
/// Implementors provide thin wrappers around native font APIs, handling:
/// - Font loading by name/spec (returns opaque FontId)
/// - Glyph lookup for characters (using FontId)
/// - System font fallback queries (returns FontId)
/// - Glyph rasterization to RGBA pixels (using FontId + glyph_id)
///
/// The driver maintains an internal cache of loaded fonts and maps FontIds to
/// platform-specific font handles.
///
/// Note: The driver is created and used entirely on the render thread, so it does not
/// need to be Send or Sync.
pub trait FontDriver {
    /// Load a font by name and size, returning an opaque font ID.
    ///
    /// # Arguments
    /// * `name` - Font name (e.g., "Menlo", "Monaco", "Noto Sans Mono")
    /// * `size_pt` - Font size in points
    ///
    /// # Returns
    /// Opaque FontId that can be used with other methods, or error if font cannot be loaded
    fn load_font(&self, name: &str, size_pt: f64) -> Result<FontId>;

    /// Find a glyph for the given character in the specified font.
    ///
    /// # Arguments
    /// * `font_id` - Font ID returned from load_font()
    /// * `ch` - Character to find
    ///
    /// # Returns
    /// Some(glyph_id) if the font contains this character, None otherwise
    fn find_glyph(&self, font_id: FontId, ch: char) -> Option<u32>;

    /// Query the system for a fallback font that contains the given character.
    ///
    /// # Arguments
    /// * `ch` - Character to find a font for
    ///
    /// # Returns
    /// FontId for a font that contains this character, or error if no suitable font is found
    fn find_fallback_font(&self, ch: char) -> Result<FontId>;

    /// Rasterize a glyph to RGBA pixel data.
    ///
    /// # Arguments
    /// * `font_id` - Font ID containing the glyph
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
        font_id: FontId,
        glyph_id: u32,
        cell_width_px: usize,
        cell_height_px: usize,
    ) -> Vec<u8>;
}
