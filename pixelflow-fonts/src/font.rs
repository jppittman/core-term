extern crate alloc;
use alloc::vec::Vec;
use crate::curves::{Segment, Line, Quadratic, Point};

#[derive(Clone)]
pub struct Font<'a> {
    face: ttf_parser::Face<'a>,
}

#[derive(Debug)]
pub enum FontError {
    ParseError(ttf_parser::FaceParsingError),
}

impl From<ttf_parser::FaceParsingError> for FontError {
    fn from(e: ttf_parser::FaceParsingError) -> Self {
        FontError::ParseError(e)
    }
}

pub struct FontMetrics {
    pub units_per_em: u16,
    pub ascent: i16,
    pub descent: i16,
    pub line_gap: i16,
}

impl<'a> Font<'a> {
    pub fn from_bytes(data: &'a [u8]) -> Result<Self, FontError> {
        let face = ttf_parser::Face::parse(data, 0)?;
        Ok(Self { face })
    }

    pub fn units_per_em(&self) -> u16 {
        self.face.units_per_em()
    }

    pub fn metrics(&self) -> FontMetrics {
        FontMetrics {
            units_per_em: self.face.units_per_em(),
            ascent: self.face.ascender(),
            descent: self.face.descender(),
            line_gap: self.face.line_gap(),
        }
    }

    pub fn glyph_index(&self, c: char) -> Option<ttf_parser::GlyphId> {
        self.face.glyph_index(c)
    }

    pub fn glyph_bounding_box(&self, id: ttf_parser::GlyphId) -> Option<ttf_parser::Rect> {
        self.face.glyph_bounding_box(id)
    }

    pub fn outline_segments(&self, id: ttf_parser::GlyphId, scale: f32) -> Vec<Segment> {
        struct Builder {
            segments: Vec<Segment>,
            current: Point,
            start: Point,
            scale: f32,
        }

        impl ttf_parser::OutlineBuilder for Builder {
            fn move_to(&mut self, x: f32, y: f32) {
                let p = [x * self.scale, y * self.scale];
                self.current = p;
                self.start = p;
            }

            fn line_to(&mut self, x: f32, y: f32) {
                let p = [x * self.scale, y * self.scale];
                self.segments.push(Segment::Line(Line { p0: self.current, p1: p }));
                self.current = p;
            }

            fn quad_to(&mut self, x1: f32, y1: f32, x: f32, y: f32) {
                let p1 = [x1 * self.scale, y1 * self.scale];
                let p2 = [x * self.scale, y * self.scale];
                if let Some(q) = Quadratic::try_new(self.current, p1, p2) {
                    self.segments.push(Segment::Quad(q));
                } else {
                    // Degenerate quad -> line
                    self.segments.push(Segment::Line(Line { p0: self.current, p1: p2 }));
                }
                self.current = p2;
            }

            fn curve_to(&mut self, x1: f32, y1: f32, x2: f32, y2: f32, x: f32, y: f32) {
                // Approximate cubic with quads (simplified)
                // For now, treat as line or split?
                // Proper way is adaptive subdivision or multiple quads.
                // Simple approximation: just line to end
                let p = [x * self.scale, y * self.scale];
                 // TODO: Implement cubic approximation
                self.segments.push(Segment::Line(Line { p0: self.current, p1: p }));
                self.current = p;
            }

            fn close(&mut self) {
                if self.current != self.start {
                    self.segments.push(Segment::Line(Line { p0: self.current, p1: self.start }));
                }
            }
        }

        let mut builder = Builder {
            segments: Vec::new(),
            current: [0.0, 0.0],
            start: [0.0, 0.0],
            scale,
        };

        if let Some(_bbox) = self.face.outline_glyph(id, &mut builder) {
            builder.segments
        } else {
            Vec::new()
        }
    }
}
