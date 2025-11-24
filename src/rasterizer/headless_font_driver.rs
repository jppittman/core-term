//! Headless mock font driver implementation.

use crate::rasterizer::font_driver::FontDriver;
use anyhow::Result;

#[derive(Clone)]
pub struct HeadlessFontDriver;

impl HeadlessFontDriver {
    pub fn new() -> Self {
        Self
    }
}

impl FontDriver for HeadlessFontDriver {
    type Font = ();
    type GlyphId = u32;

    fn load_font(&self, _name: &str, _size_pt: f64) -> Result<Self::Font> {
        Ok(())
    }

    fn find_glyph(&self, _font: &Self::Font, ch: char) -> Option<Self::GlyphId> {
        Some(ch as u32)
    }

    fn find_fallback_font(&self, _ch: char) -> Result<Self::Font> {
        Ok(())
    }

    fn rasterize_glyph(
        &self,
        _font: &Self::Font,
        _glyph_id: Self::GlyphId,
        cell_width_px: usize,
        cell_height_px: usize,
    ) -> Vec<u8> {
        // Return blank/transparent buffer of correct size
        vec![0u8; cell_width_px * cell_height_px * 4]
    }
}
