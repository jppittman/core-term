// src/platform/backends/x11/font_manager.rs

//! Implements the `FontManager` trait for the X11 platform using Xft and Fontconfig.

use crate::config::FontConfig;
use crate::glyph::AttrFlags;
use crate::platform::backends::x11::graphics::SafeXftFont; // Assuming SafeXftFont is pub(super) or pub
use crate::platform::font_manager::{FontManager, ResolvedGlyph};
use anyhow::{anyhow, Result}; // Removed Context as it's unused for now
use log::{debug, error, info, trace, warn}; // Added info
// CString is not used directly in this file after refactoring font loading to Graphics.
use std::ptr;

// X11 FFI
use x11::{xft, xlib};
use x11::xft::XftFontMatch; // Import XftFontMatch

// Fontconfig FFI
use fontconfig::fontconfig::{
    FcBool, FcCharSetAddChar, FcCharSetCreate, FcCharSetDestroy, FcConfigSubstitute, FcDefaultSubstitute,
    FcMatchPattern, FcObjectSetBuild, FcPatternAddBool, FcPatternAddCharSet, FcPatternAddObject, FcPatternCreate,
    FcPatternDestroy, FcPatternGetString, FcResultMatch, FC_CHARSET, FC_FILE, FC_INDEX, FC_SCALABLE,
};

// Constants for font_id mapping
const REGULAR_FONT_ID: usize = 0;
const BOLD_FONT_ID: usize = 1;
const ITALIC_FONT_ID: usize = 2;
const BOLD_ITALIC_FONT_ID: usize = 3;
// Start fallback font IDs after primary ones
const FALLBACK_FONT_ID_START: usize = 4;


#[derive(Debug)]
pub struct X11FontManager {
    display: *mut xlib::Display,
    xft_font_regular: SafeXftFont,
    xft_font_bold: SafeXftFont,
    xft_font_italic: SafeXftFont,
    xft_font_bold_italic: SafeXftFont,
    fallback_fonts: Vec<SafeXftFont>,
}

impl X11FontManager {
    /// Internal constructor for `X11FontManager`.
    ///
    /// This method is the designated way to create an `X11FontManager`,
    /// as it requires X11-specific resources (the display connection) and
    /// pre-loaded primary fonts which are typically handled during graphics initialization.
    /// The `FontManager::new` trait method is intentionally made to fail for this struct
    /// to guide towards this internal constructor.
    ///
    /// # Arguments
    /// * `display`: A raw pointer to the Xlib Display.
    /// * `regular`: The pre-loaded `SafeXftFont` for the regular style.
    /// * `bold`: The pre-loaded `SafeXftFont` for the bold style.
    /// * `italic`: The pre-loaded `SafeXftFont` for the italic style.
    /// * `bold_italic`: The pre-loaded `SafeXftFont` for the bold-italic style.
    ///
    /// # Returns
    /// A new instance of `X11FontManager`.
    pub(super) fn new_internal(
        display: *mut xlib::Display,
        regular: SafeXftFont, // Consumes the SafeXftFont
        bold: SafeXftFont,
        italic: SafeXftFont,
        bold_italic: SafeXftFont,
    ) -> Self {
        Self {
            display,
            xft_font_regular: regular,
            xft_font_bold: bold,
            xft_font_italic: italic,
            xft_font_bold_italic: bold_italic,
            fallback_fonts: Vec::new(),
        }
    }

    /// Helper to find a glyph in a specific XftFont using `XftCharIndex`.
    ///
    /// # Arguments
    /// * `font`: A reference to the `SafeXftFont` to search within.
    /// * `char_code`: The Unicode character to find.
    ///
    /// # Returns
    /// `Some(u32)` containing the glyph index if found, otherwise `None`.
    /// A glyph index of 0 from `XftCharIndex` typically means the glyph is not in the font.
    fn find_glyph_in_font(&self, font: &SafeXftFont, char_code: char) -> Option<u32> {
        if font.raw().is_null() {
            warn!("find_glyph_in_font called with a null font pointer for char '{}'.", char_code);
            return None;
        }
        // SAFETY: XftCharIndex is an FFI call.
        // `self.display` must be a valid X Display pointer.
        // `font.raw()` provides a valid XftFont pointer (or should be, if not null).
        let glyph_index = unsafe { xft::XftCharIndex(self.display, font.raw(), char_code as u32) };
        if glyph_index != 0 {
            Some(glyph_index)
        } else {
            // Character not found in this font.
            None
        }
    }
}

impl FontManager for X11FontManager {
    type FontHandle = *mut xft::XftFont;

    /// Creates a new `X11FontManager`.
    ///
    /// **Note:** This implementation intentionally returns an error.
    /// `X11FontManager` requires X11-specific setup (like a display connection and
    /// pre-loaded fonts) that isn't fully encapsulated by `FontConfig` alone.
    /// Use `X11FontManager::new_internal` for actual instantiation within the
    /// X11 backend, typically during graphics system initialization.
    fn new(_config: &FontConfig) -> Result<Self> {
        Err(anyhow!(
            format!("{} {}",
            "X11FontManager should not be constructed via FontManager::new.",
            "It requires pre-loaded Xft fonts and an X11 Display handle. Use X11FontManager::new_internal within the X11 backend."
            )
        ))
    }

    fn get_font_handle(&self, font_id: usize) -> Self::FontHandle {
        match font_id {
            REGULAR_FONT_ID => self.xft_font_regular.raw(),
            BOLD_FONT_ID => self.xft_font_bold.raw(),
            ITALIC_FONT_ID => self.xft_font_italic.raw(),
            BOLD_ITALIC_FONT_ID => self.xft_font_bold_italic.raw(),
            id if id >= FALLBACK_FONT_ID_START => {
                let fallback_idx = id - FALLBACK_FONT_ID_START;
                if fallback_idx < self.fallback_fonts.len() {
                    self.fallback_fonts[fallback_idx].raw()
                } else {
                    error!(
                        "Invalid fallback font_id: {} (index {} out of bounds for {} fallback fonts)",
                        font_id,
                        fallback_idx,
                        self.fallback_fonts.len()
                    );
                    ptr::null_mut()
                }
            }
            _ => {
                error!("Invalid font_id: {}", font_id);
                ptr::null_mut()
            }
        }
    }

    fn get_glyph(&mut self, character: char, attributes: AttrFlags) -> Option<ResolvedGlyph> {
        // 1. Primary Font Search
        let primary_font_info = if attributes.contains(AttrFlags::BOLD | AttrFlags::ITALIC) {
            (&self.xft_font_bold_italic, BOLD_ITALIC_FONT_ID)
        } else if attributes.contains(AttrFlags::BOLD) {
            (&self.xft_font_bold, BOLD_FONT_ID)
        } else if attributes.contains(AttrFlags::ITALIC) {
            (&self.xft_font_italic, ITALIC_FONT_ID)
        } else {
            (&self.xft_font_regular, REGULAR_FONT_ID)
        };

        if let Some(glyph_id) = self.find_glyph_in_font(primary_font_info.0, character) {
            trace!("Found glyph for '{}' in primary font_id {}", character, primary_font_info.1);
            return Some(ResolvedGlyph {
                glyph_id,
                font_id: primary_font_info.1,
            });
        }

        // 2. Cached Fallback Search
        for (idx, fallback_font) in self.fallback_fonts.iter().enumerate() {
            if let Some(glyph_id) = self.find_glyph_in_font(fallback_font, character) {
                let font_id = FALLBACK_FONT_ID_START + idx;
                trace!("Found glyph for '{}' in cached fallback font_id {}", character, font_id);
                return Some(ResolvedGlyph { glyph_id, font_id });
            }
        }

        // 3. Fontconfig Fallback Search: If not in primary or cached fallbacks, query Fontconfig.
        trace!("Glyph for '{}' not in primary or cached fallbacks. Querying Fontconfig.", character);

        // SAFETY: All Fontconfig FFI calls are unsafe and require careful handling of pointers and resources.
        unsafe {
            // Create a new Fontconfig pattern.
            let pat = FcPatternCreate();
            if pat.is_null() {
                warn!("FcPatternCreate failed. Cannot perform font fallback via Fontconfig for char '{}'.", character);
                return None;
            }

            // Request a scalable font.
            FcPatternAddBool(pat, FC_SCALABLE.as_ptr() as *const i8, FcBool::from(true));

            // Create a charset containing only the character we're looking for.
            let fc_charset = FcCharSetCreate();
            if fc_charset.is_null() {
                warn!("FcCharSetCreate failed for char '{}'. Aborting Fontconfig fallback.", character);
                FcPatternDestroy(pat); // Clean up pattern
                return None;
            }
            FcCharSetAddChar(fc_charset, character as u32);
            FcPatternAddCharSet(pat, FC_CHARSET.as_ptr() as *const i8, fc_charset);
            // Note: FcPatternAddCharSet copies the charset, so fc_charset must be destroyed by us later.

            // Perform Fontconfig substitutions (e.g., family aliases).
            FcConfigSubstitute(ptr::null_mut(), pat, FcMatchPattern); // Use default config
            FcDefaultSubstitute(pat);

            // Match the pattern to find a suitable font.
            let mut result = FcResultMatch;
            // `xlib::XDefaultScreen(self.display)` gets the default screen number for the display.
            let matched_pattern = XftFontMatch(self.display, xlib::XDefaultScreen(self.display), pat, &mut result);

            // The original pattern and charset are no longer needed after matching.
            FcPatternDestroy(pat);
            FcCharSetDestroy(fc_charset); // This is important!

            if matched_pattern.is_null() {
                trace!("XftFontMatch found no fallback font via Fontconfig for char '{}'", character);
                return None; // No font found by Fontconfig.
            }

            // Open the matched font pattern.
            // XftFontOpenPattern consumes `matched_pattern` on success.
            let new_xft_font_raw = xft::XftFontOpenPattern(self.display, matched_pattern);

            if new_xft_font_raw.is_null() {
                warn!("XftFontOpenPattern failed for matched pattern for char '{}'. Matched pattern was: {:p}", character, matched_pattern);
                FcPatternDestroy(matched_pattern); // Destroy `matched_pattern` if XftFontOpenPattern failed to consume it.
                return None;
            }

            // Successfully opened a new XftFont.
            debug!("Successfully opened fallback font via Fontconfig for char '{}'", character);
            let safe_font = SafeXftFont::new(new_xft_font_raw, self.display);

            // Check if the character exists in this newly opened font.
            if let Some(glyph_id) = self.find_glyph_in_font(&safe_font, character) {
                let font_id = FALLBACK_FONT_ID_START + self.fallback_fonts.len();
                self.fallback_fonts.push(safe_font); // Add to cache for future use.
                info!(
                    "Found glyph for char '{}' in new Fontconfig fallback (font_id {}). Caching font.",
                    character, font_id
                );
                Some(ResolvedGlyph { glyph_id, font_id })
            } else {
                // Character not found even in the font Fontconfig suggested. This is unusual.
                // The `safe_font` (and its underlying XftFont) will be closed on drop automatically.
                warn!(
                    "Glyph for char '{}' not found even in Fontconfig-matched fallback font. Fallback font not cached.",
                    character
                );
                None
            }
        }
    }
}
