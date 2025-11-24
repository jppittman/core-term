#![cfg(use_x11_display)]

//! Minimal X11 font driver using fontconfig and freetype
//!
//! This driver provides basic glyph rasterization for X11 display.
//! For now, this is a placeholder that generates colored blocks.
//! A full implementation would use freetype-rs to load and render actual fonts.

use crate::rasterizer::font_driver::FontDriver;
use anyhow::Result;

#[derive(Clone)]
pub struct X11Font {
    name: String,
    size_pt: f64,
}

pub struct X11FontDriver {
    cell_width_px: usize,
    cell_height_px: usize,
}

impl X11FontDriver {
    pub fn new(cell_width_px: usize, cell_height_px: usize) -> Self {
        Self {
            cell_width_px,
            cell_height_px,
        }
    }
}

impl FontDriver for X11FontDriver {
    type Font = X11Font;
    type GlyphId = u32;

    fn load_font(&self, name: &str, size_pt: f64) -> Result<Self::Font> {
        // Placeholder: just store the font name and size
        // A real implementation would use fontconfig to find the font file
        // and freetype to load it
        Ok(X11Font {
            name: name.to_string(),
            size_pt,
        })
    }

    fn find_glyph(&self, _font: &Self::Font, ch: char) -> Option<Self::GlyphId> {
        // Placeholder: treat character code as glyph ID
        Some(ch as u32)
    }

    fn find_fallback_font(&self, _ch: char) -> Result<Self::Font> {
        // Placeholder: return a generic monospace font
        // A real implementation would query fontconfig for a font that contains this character
        Ok(X11Font {
            name: "monospace".to_string(),
            size_pt: 12.0,
        })
    }

    fn rasterize_glyph(
        &self,
        _font: &Self::Font,
        glyph_id: Self::GlyphId,
        cell_width_px: usize,
        cell_height_px: usize,
    ) -> Vec<u8> {
        // Placeholder: generate a colored block for visible characters
        let size = cell_width_px * cell_height_px * 4;
        let mut data = vec![0u8; size];

        let ch = char::from_u32(glyph_id).unwrap_or(' ');

        // For printable characters, render white on transparent
        if ch.is_ascii_graphic() {
            // Draw a rectangle in the center of the cell
            for y in cell_height_px / 4..cell_height_px * 3 / 4 {
                for x in cell_width_px / 4..cell_width_px * 3 / 4 {
                    let offset = (y * cell_width_px + x) * 4;
                    data[offset] = 255; // R
                    data[offset + 1] = 255; // G
                    data[offset + 2] = 255; // B
                    data[offset + 3] = 255; // A (full opacity)
                }
            }
        }

        data
    }
}
