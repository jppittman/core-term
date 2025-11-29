use ttf_parser::{Face, FaceParsingError};
pub use ttf_parser::GlyphId;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum FontError {
    #[error("Failed to parse font: {0}")]
    ParseError(#[from] FaceParsingError),
}

pub struct Font<'a> {
    face: Face<'a>,
}

#[derive(Debug, Clone, Copy)]
pub struct FontMetrics {
    pub units_per_em: u16,
    pub ascent: i16,
    pub descent: i16,
    pub line_gap: i16,
}

impl<'a> Font<'a> {
    pub fn from_bytes(data: &'a [u8]) -> Result<Self, FontError> {
        let face = Face::parse(data, 0)?;
        Ok(Self { face })
    }

    pub fn metrics(&self) -> FontMetrics {
        FontMetrics {
            units_per_em: self.face.units_per_em(),
            ascent: self.face.ascender(),
            descent: self.face.descender(),
            line_gap: self.face.line_gap(),
        }
    }

    pub fn glyph_id(&self, c: char) -> Option<GlyphId> {
        self.face.glyph_index(c)
    }

    pub fn advance(&self, glyph: GlyphId) -> i32 {
        self.face.glyph_hor_advance(glyph).unwrap_or(0) as i32
    }

    pub fn kern(&self, left: GlyphId, right: GlyphId) -> i32 {
        // Try to access kern table
        if let Some(kern) = self.face.tables().kern {
             for subtable in kern.subtables {
                 if let Some(k) = subtable.glyphs_kerning(left, right) {
                     return k as i32;
                 }
             }
        }
        0
    }

    pub(crate) fn face(&self) -> &Face<'a> {
        &self.face
    }
}
