//! Headless mock font driver implementation.

use super::font_driver::{FontDriver, FontId};
use anyhow::Result;

#[derive(Clone)]
pub struct HeadlessFontDriver;

impl HeadlessFontDriver {
    pub fn new() -> Self {
        Self
    }
}

impl FontDriver for HeadlessFontDriver {
    fn load_font(&self, _name: &str, _size_pt: f64) -> Result<FontId> {
        // Just return a dummy FontId
        Ok(0)
    }

    fn find_glyph(&self, _font_id: FontId, ch: char) -> Option<u32> {
        // Mock: use character codepoint as glyph ID
        Some(ch as u32)
    }

    fn find_fallback_font(&self, _ch: char) -> Result<FontId> {
        // Just return a dummy FontId
        Ok(0)
    }

    fn rasterize_glyph(
        &self,
        _font_id: FontId,
        _glyph_id: u32,
        cell_width_px: usize,
        cell_height_px: usize,
    ) -> Vec<u8> {
        // Return blank/transparent buffer of correct size
        vec![0u8; cell_width_px * cell_height_px * 4]
    }
}
