//! Implements the `FontManager` trait for the X11 platform using Xft and Fontconfig.

use crate::config::FontConfig;
use crate::glyph::AttrFlags;
use crate::platform::backends::x11::graphics::SafeXftFont;
use crate::platform::font_manager::{FontManager, ResolvedGlyph};
use anyhow::{anyhow, Result};
use log::{error, info, trace, warn};
use std::ptr;

// Import specific FFI functions and constants from fontconfig-sys
use fontconfig_sys::{
    FcCharSetCreate, FcCharSetAddChar, FcConfigSubstitute, FcDefaultSubstitute,
    FcMatchPattern, FcPattern, FcPatternAddBool, FcPatternAddCharSet, 
    FcPatternCreate, FcPatternDestroy, FcBool, FcCharSetDestroy,
};
use fontconfig_sys::constants::{FC_CHARSET, FC_SCALABLE};

// X11 FFI
use x11::{xft, xlib};

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
    pub(super) fn new_internal(
        display: *mut xlib::Display,
        regular: SafeXftFont,
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
    fn find_glyph_in_font(&self, font: &SafeXftFont, char_code: char) -> Option<u32> {
        if font.raw().is_null() {
            warn!("find_glyph_in_font called with a null font pointer for char '{}'.", char_code);
            return None;
        }
        // SAFETY: XftCharIndex is an FFI call.
        let glyph_index = unsafe { xft::XftCharIndex(self.display, font.raw(), char_code as u32) };
        if glyph_index != 0 {
            Some(glyph_index)
        } else {
            None
        }
    }
}

impl FontManager for X11FontManager {
    type FontHandle = *mut xft::XftFont;

    fn new(_config: &FontConfig) -> Result<Self> {
        Err(anyhow!(
            "X11FontManager should not be constructed via FontManager::new. \
             Use X11FontManager::new_internal within the X11 backend."
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
                self.fallback_fonts
                    .get(fallback_idx)
                    .map_or_else(|| {
                        error!(
                            "Invalid fallback font_id: {} (index {} out of bounds for {} fonts)",
                            font_id, fallback_idx, self.fallback_fonts.len()
                        );
                        ptr::null_mut()
                    },
                    |font| font.raw())
            }
            _ => {
                error!("Invalid font_id: {}", font_id);
                ptr::null_mut()
            }
        }
    }

    fn get_glyph(&mut self, character: char, attributes: AttrFlags) -> Option<ResolvedGlyph> {
        // 1. Primary Font Search
        let (primary_font, primary_font_id) = if attributes.contains(AttrFlags::BOLD | AttrFlags::ITALIC) {
            (&self.xft_font_bold_italic, BOLD_ITALIC_FONT_ID)
        } else if attributes.contains(AttrFlags::BOLD) {
            (&self.xft_font_bold, BOLD_FONT_ID)
        } else if attributes.contains(AttrFlags::ITALIC) {
            (&self.xft_font_italic, ITALIC_FONT_ID)
        } else {
            (&self.xft_font_regular, REGULAR_FONT_ID)
        };

        if let Some(glyph_id) = self.find_glyph_in_font(primary_font, character) {
            trace!("Found glyph for '{}' in primary font_id {}", character, primary_font_id);
            return Some(ResolvedGlyph { glyph_id, font_id: primary_font_id });
        }

        // 2. Cached Fallback Search
        for (idx, fallback_font) in self.fallback_fonts.iter().enumerate() {
            if let Some(glyph_id) = self.find_glyph_in_font(fallback_font, character) {
                let font_id = FALLBACK_FONT_ID_START + idx;
                trace!("Found glyph for '{}' in cached fallback font_id {}", character, font_id);
                return Some(ResolvedGlyph { glyph_id, font_id });
            }
        }

        // 3. Fontconfig Fallback Search using raw FFI.
        trace!("Glyph for '{}' not in primary or cached fallbacks. Querying Fontconfig.", character);
        
        // SAFETY: All Fontconfig and Xft FFI calls are unsafe.
        unsafe {
            let pat = FcPatternCreate();
            if pat.is_null() {
                warn!("FcPatternCreate failed for char '{}'.", character);
                return None;
            }

            FcPatternAddBool(pat, FC_SCALABLE.as_ptr(), 1 as FcBool);
            let fc_charset = FcCharSetCreate();
            if fc_charset.is_null() {
                warn!("FcCharSetCreate failed for char '{}'.", character);
                FcPatternDestroy(pat);
                return None;
            }
            FcCharSetAddChar(fc_charset, character as u32);
            FcPatternAddCharSet(pat, FC_CHARSET.as_ptr(), fc_charset);

            FcConfigSubstitute(ptr::null_mut(), pat, FcMatchPattern);
            FcDefaultSubstitute(pat);
            
            // `result` must be of the specific type defined in the `x11` crate's bindings.
            let mut result: xft::FcResult = xft::FcResult::Match;
            
            // Cast `pat` to the `FcPattern` type expected by `xft`.
            let matched_pattern_xft = xft::XftFontMatch(
                self.display,
                xlib::XDefaultScreen(self.display),
                pat as *const xft::FcPattern, // Cast to xft's FcPattern type
                &mut result,
            );

            // Clean up original pattern and charset
            FcPatternDestroy(pat);
            FcCharSetDestroy(fc_charset);

            if matched_pattern_xft.is_null() {
                trace!("XftFontMatch found no fallback font for char '{}'", character);
                return None;
            }
            
            let new_xft_font_raw = xft::XftFontOpenPattern(self.display, matched_pattern_xft);

            if new_xft_font_raw.is_null() {
                warn!(
                    "XftFontOpenPattern failed for char '{}'. Matched pattern: {:p}",
                    character, matched_pattern_xft
                );
                // On failure, cast the pattern back to the type expected by fontconfig-sys
                // and destroy it.
                FcPatternDestroy(matched_pattern_xft as *mut FcPattern);
                return None;
            }
            
            let safe_font = SafeXftFont::new(new_xft_font_raw, self.display);

            if let Some(glyph_id) = self.find_glyph_in_font(&safe_font, character) {
                let font_id = FALLBACK_FONT_ID_START + self.fallback_fonts.len();
                info!(
                    "Found glyph for '{}' in new fallback (font_id {}). Caching.",
                    character, font_id
                );
                self.fallback_fonts.push(safe_font);
                Some(ResolvedGlyph { glyph_id, font_id })
            } else {
                warn!(
                    "Glyph for '{}' not found even in Fontconfig-matched font.",
                    character
                );
                // `safe_font` will be dropped, closing the font automatically.
                None
            }
        }
    }
}

