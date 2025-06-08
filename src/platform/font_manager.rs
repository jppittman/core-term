// src/platform/font_manager.rs

//! Defines the `FontManager` trait and related structures for platform-agnostic font handling.

use crate::config::FontConfig;
use crate::glyph::AttrFlags;
use anyhow::Result;

/// Represents a glyph that has been successfully found in a font.
///
/// This structure is platform-agnostic, providing a common way to refer to
/// a glyph irrespective of the underlying font rendering engine.
#[derive(Debug, Clone, Copy)]
pub struct ResolvedGlyph {
    /// The platform-specific identifier for the glyph.
    /// For example, this could be an Xft glyph index or a FreeType glyph index.
    pub glyph_id: u32,
    /// An internal, platform-specific identifier assigned by the `FontManager`
    /// to the font from which this glyph was resolved. This ID is used with
    /// `FontManager::get_font_handle`.
    pub font_id: usize,
}

/// Defines the contract for a font management system.
///
/// Implementors of this trait are responsible for loading fonts, resolving
/// characters to specific glyphs within those fonts (potentially using fallbacks),
/// and providing access to platform-specific font handles.
pub trait FontManager {
    /// The type representing a platform-specific font handle.
    /// For example, for X11/Xft, this might be `*mut xft::XftFont`.
    type FontHandle;

    /// Creates a new instance of the font manager.
    ///
    /// Implementations should load necessary fonts based on the provided `FontConfig`.
    /// Depending on the platform, additional parameters (like X11 display connections)
    /// might be required, often handled by specific internal constructors rather than
    /// this trait method directly if the trait needs to remain generic.
    ///
    /// # Arguments
    /// * `config`: A reference to the `FontConfig` specifying which fonts to load.
    ///
    /// # Returns
    /// A `Result` containing the new `FontManager` instance or an error if initialization fails.
    fn new(
        config: &FontConfig,
        // Platform-specific implementations might require more context than just FontConfig.
        // For instance, an X11 implementation needs X server connection details.
        // Such details might be passed via a more specific internal constructor,
        // or this `new` method might be less suitable for direct use by generic code
        // if it can't encapsulate all necessary platform details via `FontConfig` alone.
    ) -> Result<Self>
    where
        Self: Sized;

    /// Resolves a character to a specific glyph based on the desired attributes.
    ///
    /// The manager will attempt to find the character in the primary font corresponding
    /// to the attributes. If not found, it may try fallback fonts or system font matching
    /// to find a suitable glyph.
    ///
    /// # Arguments
    /// * `character`: The `char` to resolve.
    /// * `attributes`: `AttrFlags` (e.g., bold, italic) specifying the desired style.
    ///
    /// # Returns
    /// An `Option<ResolvedGlyph>` containing information about the found glyph and its
    /// source font, or `None` if the character cannot be resolved.
    fn get_glyph(&mut self, character: char, attributes: AttrFlags) -> Option<ResolvedGlyph>;

    /// Retrieves a platform-specific font handle using an internal font ID.
    ///
    /// The `font_id` is obtained from `ResolvedGlyph::font_id`. This handle can then
    /// be used with platform-specific drawing APIs.
    ///
    /// # Arguments
    /// * `font_id`: The internal ID of the font to retrieve.
    ///
    /// # Returns
    /// The platform-specific font handle. Behavior for an invalid `font_id` is
    /// implementation-defined (e.g., may return a null handle or a default font handle).
    fn get_font_handle(&self, font_id: usize) -> Self::FontHandle;
}
