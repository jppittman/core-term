//! Shared font management and caching logic.
//!
//! The `FontManager` implements platform-agnostic font fallback and glyph caching
//! using a platform-specific `FontDriver` for primitives.

use super::font_driver::{FontDriver, FontId};
use crate::glyph::AttrFlags;
use anyhow::{Context, Result};
use log::*;
use std::collections::HashMap;

/// Indices for primary fonts array
const REGULAR_FONT_IDX: usize = 0;
const BOLD_FONT_IDX: usize = 1;
const ITALIC_FONT_IDX: usize = 2;
const BOLD_ITALIC_FONT_IDX: usize = 3;

/// A resolved glyph with font ID and glyph ID
#[derive(Debug, Clone, Copy)]
pub struct ResolvedGlyph {
    /// FontId from FontDriver (opaque handle)
    pub font_id: FontId,
    /// Platform-specific glyph ID
    pub glyph_id: u32,
}

/// Font manager with shared caching and fallback logic.
///
/// Uses a platform-specific `FontDriver` for primitives (load, find, rasterize).
/// Implements shared logic for:
/// - Font fallback (primary → cached fallbacks → system query)
/// - Glyph caching (HashMap of char+attrs → font_id+glyph_id)
/// - Font handle management
pub struct FontManager {
    /// Platform-specific driver for font primitives (object-safe trait)
    driver: Box<dyn FontDriver>,

    /// Primary fonts: [regular, bold, italic, bold+italic]
    primary_fonts: [FontId; 4],

    /// Dynamically discovered fallback fonts
    fallback_fonts: Vec<FontId>,

    /// Glyph cache: (char, attrs) → (font_id, glyph_id)
    glyph_cache: HashMap<(char, AttrFlags), (FontId, u32)>,
}

impl FontManager {
    /// Create a new font manager with the given driver and font specs.
    ///
    /// # Arguments
    /// * `driver` - Platform-specific font driver (boxed trait object)
    /// * `regular_name` - Name of regular font
    /// * `bold_name` - Name of bold font
    /// * `italic_name` - Name of italic font
    /// * `bold_italic_name` - Name of bold+italic font
    /// * `size_pt` - Font size in points
    ///
    /// # Returns
    /// Initialized FontManager or error if primary fonts cannot be loaded
    pub fn new(
        driver: Box<dyn FontDriver>,
        regular_name: &str,
        bold_name: &str,
        italic_name: &str,
        bold_italic_name: &str,
        size_pt: f64,
    ) -> Result<Self> {
        info!("FontManager: Loading primary fonts at {} pt", size_pt);

        let regular = driver
            .load_font(regular_name, size_pt)
            .with_context(|| format!("Failed to load regular font '{}'", regular_name))?;

        let bold = driver
            .load_font(bold_name, size_pt)
            .with_context(|| format!("Failed to load bold font '{}'", bold_name))?;

        let italic = driver
            .load_font(italic_name, size_pt)
            .with_context(|| format!("Failed to load italic font '{}'", italic_name))?;

        let bold_italic = driver
            .load_font(bold_italic_name, size_pt)
            .with_context(|| format!("Failed to load bold+italic font '{}'", bold_italic_name))?;

        info!("FontManager: Primary fonts loaded successfully");

        Ok(Self {
            driver,
            primary_fonts: [regular, bold, italic, bold_italic],
            fallback_fonts: Vec::new(),
            glyph_cache: HashMap::new(),
        })
    }

    /// Get a glyph for the given character and attributes.
    ///
    /// Implements the fallback strategy:
    /// 1. Check glyph cache
    /// 2. Try primary font (based on bold/italic attrs)
    /// 3. Try cached fallback fonts
    /// 4. Query system for new fallback font
    /// 5. Cache the result
    ///
    /// # Arguments
    /// * `ch` - Character to find
    /// * `attrs` - Text attributes (bold, italic, etc.)
    ///
    /// # Returns
    /// Some(ResolvedGlyph) if found, None if character cannot be rendered
    pub fn get_glyph(&mut self, ch: char, attrs: AttrFlags) -> Option<ResolvedGlyph> {
        // 1. Check cache first
        if let Some(&(font_id, glyph_id)) = self.glyph_cache.get(&(ch, attrs)) {
            trace!("FontManager: Cache hit for '{}' (U+{:X})", ch, ch as u32);
            return Some(ResolvedGlyph { font_id, glyph_id });
        }

        // 2. Try primary font (based on bold/italic)
        let primary_idx = Self::primary_font_index(attrs);
        let primary_font_id = self.primary_fonts[primary_idx];
        if let Some(glyph_id) = self.driver.find_glyph(primary_font_id, ch) {
            debug!(
                "FontManager: Found '{}' (U+{:X}) in primary font (idx={})",
                ch, ch as u32, primary_idx
            );
            self.glyph_cache
                .insert((ch, attrs), (primary_font_id, glyph_id));
            return Some(ResolvedGlyph {
                font_id: primary_font_id,
                glyph_id,
            });
        }

        // 3. Try cached fallback fonts
        for font_id in &self.fallback_fonts {
            if let Some(glyph_id) = self.driver.find_glyph(*font_id, ch) {
                debug!(
                    "FontManager: Found '{}' (U+{:X}) in cached fallback font (id={})",
                    ch, ch as u32, font_id
                );
                self.glyph_cache.insert((ch, attrs), (*font_id, glyph_id));
                return Some(ResolvedGlyph {
                    font_id: *font_id,
                    glyph_id,
                });
            }
        }

        // 4. Query system for new fallback font
        trace!(
            "FontManager: Querying system for fallback font for '{}' (U+{:X})",
            ch,
            ch as u32
        );

        if let Ok(fallback_font_id) = self.driver.find_fallback_font(ch) {
            if let Some(glyph_id) = self.driver.find_glyph(fallback_font_id, ch) {
                info!(
                    "FontManager: Found '{}' (U+{:X}) in new fallback font (id={}, caching)",
                    ch, ch as u32, fallback_font_id
                );
                self.fallback_fonts.push(fallback_font_id);
                self.glyph_cache.insert((ch, attrs), (fallback_font_id, glyph_id));
                return Some(ResolvedGlyph {
                    font_id: fallback_font_id,
                    glyph_id,
                });
            }
        }

        // Character not found in any font
        warn!(
            "FontManager: Could not find glyph for '{}' (U+{:X}) in any font",
            ch, ch as u32
        );
        None
    }

    /// Get a reference to the driver for direct access to primitives.
    pub fn driver(&self) -> &dyn FontDriver {
        &*self.driver
    }

    /// Determine which primary font to use based on attributes.
    fn primary_font_index(attrs: AttrFlags) -> usize {
        match (
            attrs.contains(AttrFlags::BOLD),
            attrs.contains(AttrFlags::ITALIC),
        ) {
            (true, true) => BOLD_ITALIC_FONT_IDX,
            (true, false) => BOLD_FONT_IDX,
            (false, true) => ITALIC_FONT_IDX,
            (false, false) => REGULAR_FONT_IDX,
        }
    }
}

// TODO: Add tests with updated trait (object-safe version)
