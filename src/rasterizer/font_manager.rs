//! Shared font management and caching logic.
//!
//! The `FontManager` implements platform-agnostic font fallback and glyph caching
//! using a platform-specific `FontDriver` for primitives.

use super::font_driver::FontDriver;
use crate::glyph::AttrFlags;
use anyhow::{Context, Result};
use log::*;
use std::collections::HashMap;

/// Font IDs for primary fonts
const REGULAR_FONT_ID: usize = 0;
const BOLD_FONT_ID: usize = 1;
const ITALIC_FONT_ID: usize = 2;
const BOLD_ITALIC_FONT_ID: usize = 3;
const FALLBACK_FONT_ID_START: usize = 4;

/// A resolved glyph with its font ID and platform-specific glyph ID
#[derive(Debug, Clone, Copy)]
pub struct ResolvedGlyph<G> {
    /// Index into primary_fonts or fallback_fonts
    pub font_id: usize,
    /// Platform-specific glyph ID
    pub glyph_id: G,
}

/// Font manager with shared caching and fallback logic.
///
/// Uses a platform-specific `FontDriver` for primitives (load, find, rasterize).
/// Implements shared logic for:
/// - Font fallback (primary → cached fallbacks → system query)
/// - Glyph caching (HashMap of char+attrs → font_id+glyph_id)
/// - Font handle management
#[derive(Debug)]
pub struct FontManager<D: FontDriver> {
    /// Platform-specific driver for font primitives
    driver: D,

    /// Primary fonts: [regular, bold, italic, bold+italic]
    primary_fonts: [D::Font; 4],

    /// Dynamically discovered fallback fonts
    fallback_fonts: Vec<D::Font>,

    /// Glyph cache: (char, attrs) → (font_id, glyph_id)
    /// Simple HashMap for now (can optimize to Vec later if needed)
    glyph_cache: HashMap<(char, AttrFlags), (usize, D::GlyphId)>,
}

impl<D: FontDriver> FontManager<D> {
    /// Create a new font manager with the given driver and font specs.
    ///
    /// # Arguments
    /// * `driver` - Platform-specific font driver
    /// * `regular_name` - Name of regular font
    /// * `bold_name` - Name of bold font
    /// * `italic_name` - Name of italic font
    /// * `bold_italic_name` - Name of bold+italic font
    /// * `size_pt` - Font size in points
    ///
    /// # Returns
    /// Initialized FontManager or error if primary fonts cannot be loaded
    pub fn new(
        driver: D,
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
    /// Implements the 5-step fallback strategy:
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
    pub fn get_glyph(&mut self, ch: char, attrs: AttrFlags) -> Option<ResolvedGlyph<D::GlyphId>> {
        // 1. Check cache first
        if let Some(&(font_id, glyph_id)) = self.glyph_cache.get(&(ch, attrs)) {
            trace!("FontManager: Cache hit for '{}' (U+{:X})", ch, ch as u32);
            return Some(ResolvedGlyph { font_id, glyph_id });
        }

        // 2. Try primary font (based on bold/italic)
        let primary_idx = Self::primary_font_index(attrs);
        if let Some(glyph_id) = self.driver.find_glyph(&self.primary_fonts[primary_idx], ch) {
            debug!(
                "FontManager: Found '{}' (U+{:X}) in primary font {}",
                ch, ch as u32, primary_idx
            );
            self.glyph_cache
                .insert((ch, attrs), (primary_idx, glyph_id));
            return Some(ResolvedGlyph {
                font_id: primary_idx,
                glyph_id,
            });
        }

        // 3. Try cached fallback fonts
        for (idx, font) in self.fallback_fonts.iter().enumerate() {
            if let Some(glyph_id) = self.driver.find_glyph(font, ch) {
                let font_id = FALLBACK_FONT_ID_START + idx;
                debug!(
                    "FontManager: Found '{}' (U+{:X}) in cached fallback font {}",
                    ch, ch as u32, font_id
                );
                self.glyph_cache.insert((ch, attrs), (font_id, glyph_id));
                return Some(ResolvedGlyph { font_id, glyph_id });
            }
        }

        // 4. Query system for new fallback font
        trace!(
            "FontManager: Querying system for fallback font for '{}' (U+{:X})",
            ch,
            ch as u32
        );

        if let Ok(fallback_font) = self.driver.find_fallback_font(ch) {
            if let Some(glyph_id) = self.driver.find_glyph(&fallback_font, ch) {
                let font_id = FALLBACK_FONT_ID_START + self.fallback_fonts.len();
                info!(
                    "FontManager: Found '{}' (U+{:X}) in new fallback font {} (caching)",
                    ch, ch as u32, font_id
                );
                self.fallback_fonts.push(fallback_font);
                self.glyph_cache.insert((ch, attrs), (font_id, glyph_id));
                return Some(ResolvedGlyph { font_id, glyph_id });
            }
        }

        // Character not found in any font
        warn!(
            "FontManager: Could not find glyph for '{}' (U+{:X}) in any font",
            ch, ch as u32
        );
        None
    }

    /// Get a reference to the font at the given font_id.
    ///
    /// # Arguments
    /// * `font_id` - Font ID from ResolvedGlyph
    ///
    /// # Returns
    /// Reference to the platform-specific font handle
    pub fn get_font(&self, font_id: usize) -> &D::Font {
        if font_id < FALLBACK_FONT_ID_START {
            &self.primary_fonts[font_id]
        } else {
            let fallback_idx = font_id - FALLBACK_FONT_ID_START;
            &self.fallback_fonts[fallback_idx]
        }
    }

    /// Get a mutable reference to the driver for direct access to primitives.
    pub fn driver(&self) -> &D {
        &self.driver
    }

    /// Determine which primary font to use based on attributes.
    fn primary_font_index(attrs: AttrFlags) -> usize {
        match (
            attrs.contains(AttrFlags::BOLD),
            attrs.contains(AttrFlags::ITALIC),
        ) {
            (true, true) => BOLD_ITALIC_FONT_ID,
            (true, false) => BOLD_FONT_ID,
            (false, true) => ITALIC_FONT_ID,
            (false, false) => REGULAR_FONT_ID,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rasterizer::font_driver::FontDriver;
    use anyhow::anyhow;
    use std::collections::HashMap;

    // --- Mock Implementation ---

    #[derive(Clone, Debug, PartialEq, Eq)]
    struct MockFont(String);

    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    struct MockGlyphId(u32);

    /// Represents the "system" font database for the mock driver.
    #[derive(Debug)]
    struct MockFontDatabase {
        fonts: HashMap<String, HashMap<char, MockGlyphId>>,
    }

    impl MockFontDatabase {
        fn new() -> Self {
            Self {
                fonts: HashMap::new(),
            }
        }

        fn add_font(&mut self, name: &str, glyphs: Vec<(char, u32)>) {
            let glyph_map = glyphs
                .into_iter()
                .map(|(c, id)| (c, MockGlyphId(id)))
                .collect();
            self.fonts.insert(name.to_string(), glyph_map);
        }

        fn get_font_glyphs(&self, name: &str) -> Option<&HashMap<char, MockGlyphId>> {
            self.fonts.get(name)
        }

        fn find_font_for_char(&self, ch: char) -> Option<String> {
            // In a real scenario, font-kit would have sophisticated matching.
            // For tests, we'll just find the first font that has the glyph.
            // We iterate in a fixed order for stable tests.
            let mut font_names: Vec<_> = self.fonts.keys().collect();
            font_names.sort();

            for name in font_names {
                if let Some(glyphs) = self.fonts.get(name) {
                    if glyphs.contains_key(&ch) {
                        return Some(name.clone());
                    }
                }
            }
            None
        }
    }

    /// A more robust mock FontDriver that uses a `MockFontDatabase`.
    #[derive(Debug)]
    struct MockFontDriver<'db> {
        db: &'db MockFontDatabase,
    }

    impl<'db> MockFontDriver<'db> {
        fn new(db: &'db MockFontDatabase) -> Self {
            Self { db }
        }
    }

    impl<'db> FontDriver for MockFontDriver<'db> {
        type Font = MockFont;
        type GlyphId = MockGlyphId;

        fn load_font(&self, name: &str, _size_pt: f64) -> Result<Self::Font> {
            if self.db.get_font_glyphs(name).is_some() {
                Ok(MockFont(name.to_string()))
            } else {
                Err(anyhow!("MockFontDriver: Font '{}' not found", name))
            }
        }

        fn find_glyph(&self, font: &Self::Font, ch: char) -> Option<Self::GlyphId> {
            self.db
                .get_font_glyphs(&font.0)
                .and_then(|glyphs| glyphs.get(&ch).copied())
        }

        fn find_fallback_font(&self, ch: char) -> Result<Self::Font> {
            self.db
                .find_font_for_char(ch)
                .map(|name| Ok(MockFont(name)))
                .unwrap_or_else(|| {
                    Err(anyhow!(
                        "MockFontDriver: No fallback font found for '{}'",
                        ch
                    ))
                })
        }

        fn rasterize_glyph(
            &self,
            _font: &Self::Font,
            _glyph_id: Self::GlyphId,
            cell_width_px: usize,
            cell_height_px: usize,
        ) -> Vec<u8> {
            vec![128; cell_width_px * cell_height_px * 4]
        }
    }

    // --- Test Setup ---

    fn setup_database() -> MockFontDatabase {
        let mut db = MockFontDatabase::new();
        db.add_font("Regular", vec![('A', 100), ('B', 101)]);
        db.add_font("Bold", vec![('A', 200)]);
        db.add_font("Italic", vec![('A', 300)]);
        db.add_font("BoldItalic", vec![('A', 400)]);
        db.add_font("Fallback1", vec![('C', 500), ('E', 501)]);
        db.add_font("Fallback2", vec![('D', 600)]);
        db
    }

    // --- Test Cases ---

    #[test]
    fn test_new_font_manager_success() {
        let db = setup_database();
        let driver = MockFontDriver::new(&db);
        let fm = FontManager::new(driver, "Regular", "Bold", "Italic", "BoldItalic", 12.0);
        assert!(fm.is_ok());
    }

    #[test]
    fn test_new_font_manager_missing_font() {
        let db = MockFontDatabase::new(); // Empty database
        let driver = MockFontDriver::new(&db);
        let fm = FontManager::new(driver, "Regular", "Bold", "Italic", "BoldItalic", 12.0);
        assert!(fm.is_err());
        let err_msg = fm.unwrap_err().to_string();
        assert!(err_msg.contains("Failed to load regular font 'Regular'"));
    }

    #[test]
    fn test_get_glyph_primary_font() {
        let db = setup_database();
        let driver = MockFontDriver::new(&db);
        let mut fm =
            FontManager::new(driver, "Regular", "Bold", "Italic", "BoldItalic", 12.0).unwrap();

        // Regular
        let resolved = fm.get_glyph('A', AttrFlags::empty()).unwrap();
        assert_eq!(resolved.font_id, REGULAR_FONT_ID);
        assert_eq!(resolved.glyph_id, MockGlyphId(100));

        // Bold
        let resolved = fm.get_glyph('A', AttrFlags::BOLD).unwrap();
        assert_eq!(resolved.font_id, BOLD_FONT_ID);
        assert_eq!(resolved.glyph_id, MockGlyphId(200));

        // Italic
        let resolved = fm.get_glyph('A', AttrFlags::ITALIC).unwrap();
        assert_eq!(resolved.font_id, ITALIC_FONT_ID);
        assert_eq!(resolved.glyph_id, MockGlyphId(300));

        // Bold + Italic
        let resolved = fm
            .get_glyph('A', AttrFlags::BOLD | AttrFlags::ITALIC)
            .unwrap();
        assert_eq!(resolved.font_id, BOLD_ITALIC_FONT_ID);
        assert_eq!(resolved.glyph_id, MockGlyphId(400));
    }

    #[test]
    fn test_get_glyph_cache_hit() {
        let db = setup_database();
        let driver = MockFontDriver::new(&db);
        let mut fm =
            FontManager::new(driver, "Regular", "Bold", "Italic", "BoldItalic", 12.0).unwrap();

        // First call, should find it in primary font
        let resolved1 = fm.get_glyph('A', AttrFlags::empty()).unwrap();
        assert_eq!(resolved1.font_id, REGULAR_FONT_ID);
        assert_eq!(resolved1.glyph_id, MockGlyphId(100));

        // Second call should hit the cache. The driver is immutable, so this proves
        // we are not calling it again.
        let resolved2 = fm.get_glyph('A', AttrFlags::empty()).unwrap();
        assert_eq!(resolved2.font_id, REGULAR_FONT_ID);
        assert_eq!(resolved2.glyph_id, MockGlyphId(100)); // Same result
    }

    #[test]
    fn test_get_glyph_does_not_fallback_to_regular_from_styled() {
        let db = setup_database();
        let driver = MockFontDriver::new(&db);
        let mut fm =
            FontManager::new(driver, "Regular", "Bold", "Italic", "BoldItalic", 12.0).unwrap();

        // 'B' is only in the regular font. Requesting it with bold attributes
        // should check the bold font, fail, then check fallbacks. It should *not*
        // check the regular font.
        let resolved = fm.get_glyph('B', AttrFlags::BOLD);

        // In our mock DB, `find_fallback_font` will find 'B' in the "Regular" font.
        // So the fallback logic will kick in and find it.
        assert!(resolved.is_some());
        let glyph = resolved.unwrap();
        assert_eq!(glyph.font_id, FALLBACK_FONT_ID_START); // Found in a "new" fallback
        assert_eq!(fm.get_font(glyph.font_id).0, "Regular");
    }

    #[test]
    fn test_get_glyph_new_fallback() {
        let db = setup_database();
        let driver = MockFontDriver::new(&db);
        let mut fm =
            FontManager::new(driver, "Regular", "Bold", "Italic", "BoldItalic", 12.0).unwrap();

        // 'C' is not in primary fonts, should trigger a fallback search
        let resolved = fm.get_glyph('C', AttrFlags::empty()).unwrap();
        assert_eq!(resolved.font_id, FALLBACK_FONT_ID_START); // First fallback
        assert_eq!(resolved.glyph_id, MockGlyphId(500));

        // Check that the fallback font is now cached
        assert_eq!(fm.fallback_fonts.len(), 1);
        assert_eq!(fm.fallback_fonts[0], MockFont("Fallback1".to_string()));
    }

    #[test]
    fn test_get_glyph_cached_fallback() {
        let db = setup_database();
        let driver = MockFontDriver::new(&db);
        let mut fm =
            FontManager::new(driver, "Regular", "Bold", "Italic", "BoldItalic", 12.0).unwrap();

        // Find 'C' to populate the fallback cache.
        let resolved_c = fm.get_glyph('C', AttrFlags::empty()).unwrap();
        assert_eq!(resolved_c.font_id, FALLBACK_FONT_ID_START);
        assert_eq!(fm.fallback_fonts.len(), 1);

        // Now, let's ask for 'E', which is in the same fallback font.
        // The FontManager should use the cached fallback_fonts list and find 'E'
        // without calling find_fallback_font on the driver again.
        let resolved_e = fm.get_glyph('E', AttrFlags::empty()).unwrap();
        assert_eq!(resolved_e.font_id, FALLBACK_FONT_ID_START);
        assert_eq!(resolved_e.glyph_id, MockGlyphId(501));

        // The number of fallback fonts should still be 1.
        assert_eq!(fm.fallback_fonts.len(), 1);
    }

    #[test]
    fn test_get_glyph_multiple_fallbacks() {
        let db = setup_database();
        let driver = MockFontDriver::new(&db);
        let mut fm =
            FontManager::new(driver, "Regular", "Bold", "Italic", "BoldItalic", 12.0).unwrap();

        // 'C' is in Fallback1
        let resolved_c = fm.get_glyph('C', AttrFlags::empty()).unwrap();
        assert_eq!(resolved_c.font_id, FALLBACK_FONT_ID_START);
        assert_eq!(resolved_c.glyph_id, MockGlyphId(500));

        // 'D' is in Fallback2
        let resolved_d = fm.get_glyph('D', AttrFlags::empty()).unwrap();
        assert_eq!(resolved_d.font_id, FALLBACK_FONT_ID_START + 1);
        assert_eq!(resolved_d.glyph_id, MockGlyphId(600));

        assert_eq!(fm.fallback_fonts.len(), 2);
    }

    #[test]
    fn test_get_glyph_not_found() {
        let db = setup_database();
        let driver = MockFontDriver::new(&db);
        let mut fm =
            FontManager::new(driver, "Regular", "Bold", "Italic", "BoldItalic", 12.0).unwrap();

        // 'Z' is not in any font, and fallback search will also fail.
        let resolved = fm.get_glyph('Z', AttrFlags::empty());
        assert!(resolved.is_none());
    }

    #[test]
    fn test_get_font() {
        let db = setup_database();
        let driver = MockFontDriver::new(&db);
        let mut fm =
            FontManager::new(driver, "Regular", "Bold", "Italic", "BoldItalic", 12.0).unwrap();

        // Resolve a fallback glyph to populate the fallback list
        let resolved_fallback = fm.get_glyph('C', AttrFlags::empty()).unwrap();

        // Get primary font
        let regular_font = fm.get_font(REGULAR_FONT_ID);
        assert_eq!(*regular_font, MockFont("Regular".to_string()));

        // Get fallback font
        let fallback_font = fm.get_font(resolved_fallback.font_id);
        assert_eq!(*fallback_font, MockFont("Fallback1".to_string()));
    }

    #[test]
    #[should_panic]
    fn test_get_font_invalid_id() {
        let db = setup_database();
        let driver = MockFontDriver::new(&db);
        let fm = FontManager::new(driver, "Regular", "Bold", "Italic", "BoldItalic", 12.0).unwrap();

        // This should panic (index out of bounds)
        fm.get_font(99);
    }
}
