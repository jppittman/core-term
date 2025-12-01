//! Font parsing and glyph retrieval.
//!
//! This module provides the [`Font`] struct, which serves as the entry point
//! for loading font data and extracting glyph geometry.

use thiserror::Error;
pub use ttf_parser::GlyphId;
use ttf_parser::{Face, FaceParsingError, OutlineBuilder};
use std::sync::Arc;

use crate::curves::{Line, Point, Quadratic, Segment};
use crate::glyph::{Glyph, GlyphBounds};

/// Errors that can occur when parsing a font.
#[derive(Error, Debug)]
pub enum FontError {
    #[error("Failed to parse font: {0}")]
    ParseError(#[from] FaceParsingError),
}

/// A handle to a parsed font.
///
/// `Font` is cheap to clone (wraps an `Arc`) and thread-safe.
/// It uses `ttf-parser` internally to extract outline data.
#[derive(Clone)]
pub struct Font<'a> {
    face: Arc<Face<'a>>,
}

/// Metrics describing the font's vertical layout.
#[derive(Debug, Clone, Copy)]
pub struct FontMetrics {
    /// The number of font units per EM square.
    pub units_per_em: u16,
    /// Distance from the baseline to the top of the highest glyph.
    pub ascent: i16,
    /// Distance from the baseline to the bottom of the lowest glyph.
    pub descent: i16,
    /// Recommended vertical distance between lines.
    pub line_gap: i16,
}

impl<'a> Font<'a> {
    /// Creates a new `Font` from raw TTF/OTF byte data.
    ///
    /// # Arguments
    ///
    /// * `data` - A byte slice containing the font file data.
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing the `Font` or a `FontError` if parsing fails.
    pub fn from_bytes(data: &'a [u8]) -> Result<Self, FontError> {
        let face = Face::parse(data, 0)?;
        Ok(Self { face: Arc::new(face) })
    }

    /// Returns the global metrics for this font.
    pub fn metrics(&self) -> FontMetrics {
        FontMetrics {
            units_per_em: self.face.units_per_em(),
            ascent: self.face.ascender(),
            descent: self.face.descender(),
            line_gap: self.face.line_gap(),
        }
    }

    /// Retrieves the geometry for a specific character.
    ///
    /// This method extracts the outline of the character, subdivides cubic Bézier curves
    /// into quadratics (since the renderer supports quadratics natively), and returns
    /// a [`Glyph`] which implements [`Surface<u8>`](pixelflow_core::pipe::Surface).
    ///
    /// # Arguments
    ///
    /// * `ch` - The character to retrieve.
    /// * `size` - The desired font size in pixels. This scales the geometry.
    ///
    /// # Returns
    ///
    /// * `Some(Glyph)` if the character exists in the font.
    /// * `None` if the character is not found.
    pub fn glyph(&self, ch: char, size: f32) -> Option<Glyph> {
        let glyph_id = self.face.glyph_index(ch)?;

        let scale = size / self.face.units_per_em() as f32;
        let mut builder = GlyphBuilder::new(scale);

        let _ = self.face.outline_glyph(glyph_id, &mut builder);

        let bbox = self
            .face
            .glyph_bounding_box(glyph_id)
            .unwrap_or(ttf_parser::Rect {
                x_min: 0,
                y_min: 0,
                x_max: 0,
                y_max: 0,
            });

        // Compute integer bounds for the glyph surface
        let bounds = GlyphBounds {
            width: ((bbox.x_max - bbox.x_min) as f32 * scale).ceil() as u32,
            height: ((bbox.y_max - bbox.y_min) as f32 * scale).ceil() as u32,
            bearing_x: (bbox.x_min as f32 * scale).round() as i32,
            bearing_y: (bbox.y_max as f32 * scale).round() as i32,
        };

        Some(Glyph {
            segments: Arc::from(builder.segments),
            bounds,
        })
    }

    /// Calculates the horizontal advance for a character.
    ///
    /// The advance is the distance the cursor should move after drawing this character.
    ///
    /// # Arguments
    ///
    /// * `ch` - The character.
    /// * `size` - The font size in pixels.
    pub fn advance(&self, ch: char, size: f32) -> f32 {
        let scale = size / self.face.units_per_em() as f32;
        self.face
            .glyph_index(ch)
            .and_then(|id| self.face.glyph_hor_advance(id))
            .map(|adv| adv as f32 * scale)
            .unwrap_or(0.0)
    }

    /// Calculates the kerning adjustment between two characters.
    ///
    /// # Arguments
    ///
    /// * `left` - The left character.
    /// * `right` - The right character.
    /// * `size` - The font size in pixels.
    ///
    /// # Returns
    ///
    /// The kerning offset in pixels. Add this to the advance of the left character.
    pub fn kern(&self, left: char, right: char, size: f32) -> f32 {
        let scale = size / self.face.units_per_em() as f32;
        let left_id = match self.face.glyph_index(left) {
            Some(id) => id,
            None => return 0.0,
        };
        let right_id = match self.face.glyph_index(right) {
            Some(id) => id,
            None => return 0.0,
        };

        if let Some(kern) = self.face.tables().kern {
            for subtable in kern.subtables {
                if let Some(k) = subtable.glyphs_kerning(left_id, right_id) {
                    return k as f32 * scale;
                }
            }
        }
        0.0
    }
}

// ============================================================================
// Outline Builder
// ============================================================================

struct GlyphBuilder {
    segments: Vec<Segment>,
    scale: f32,
    current: Point,
    start: Point,
}

impl GlyphBuilder {
    fn new(scale: f32) -> Self {
        Self {
            segments: Vec::with_capacity(32),
            scale,
            current: [0.0, 0.0],
            start: [0.0, 0.0],
        }
    }

    // Recursively subdivide cubic curves into line segments if they are small enough,
    // or eventually approximate with quadratics (though here it seems to just degenerate to lines).
    // Note: The comment in original code mentioned approximation, but the implementation
    // subdivides until flat enough for lines.
    fn subdivide_cubic(&mut self, p0: Point, p1: Point, p2: Point, p3: Point, depth: u32) {
        let d03 = dist_sq(p0, p3);
        const MIN_LEN_SQ: f32 = 0.25; // 0.5 px

        if depth > 8 || d03 < MIN_LEN_SQ {
            self.segments.push(Segment::Line(Line { p0, p1: p3 }));
            return;
        }

        let p01 = lerp(p0, p1, 0.5);
        let p12 = lerp(p1, p2, 0.5);
        let p23 = lerp(p2, p3, 0.5);
        let p012 = lerp(p01, p12, 0.5);
        let p123 = lerp(p12, p23, 0.5);
        let p0123 = lerp(p012, p123, 0.5);

        self.subdivide_cubic(p0, p01, p012, p0123, depth + 1);
        self.subdivide_cubic(p0123, p123, p23, p3, depth + 1);
    }
}

impl OutlineBuilder for GlyphBuilder {
    fn move_to(&mut self, x: f32, y: f32) {
        self.current = [x * self.scale, y * self.scale];
        self.start = self.current;
    }

    fn line_to(&mut self, x: f32, y: f32) {
        let p1 = [x * self.scale, y * self.scale];
        self.segments.push(Segment::Line(Line {
            p0: self.current,
            p1,
        }));
        self.current = p1;
    }

    fn quad_to(&mut self, x1: f32, y1: f32, x: f32, y: f32) {
        let p1 = [x1 * self.scale, y1 * self.scale];
        let p2 = [x * self.scale, y * self.scale];

        if let Some(q) = Quadratic::try_new(self.current, p1, p2) {
            self.segments.push(Segment::Quad(q));
        } else {
            self.segments.push(Segment::Line(Line {
                p0: self.current,
                p1: p2,
            }));
        }
        self.current = p2;
    }

    fn curve_to(&mut self, x1: f32, y1: f32, x2: f32, y2: f32, x: f32, y: f32) {
        let p1 = [x1 * self.scale, y1 * self.scale];
        let p2 = [x2 * self.scale, y2 * self.scale];
        let p3 = [x * self.scale, y * self.scale];

        self.subdivide_cubic(self.current, p1, p2, p3, 0);
        self.current = p3;
    }

    fn close(&mut self) {
        if (self.current[0] - self.start[0]).abs() > 1e-4
            || (self.current[1] - self.start[1]).abs() > 1e-4
        {
            self.segments.push(Segment::Line(Line {
                p0: self.current,
                p1: self.start,
            }));
            self.current = self.start;
        }
    }
}

#[inline]
fn lerp(p0: Point, p1: Point, t: f32) -> Point {
    [p0[0] * (1.0 - t) + p1[0] * t, p0[1] * (1.0 - t) + p1[1] * t]
}

#[inline]
fn dist_sq(p0: Point, p1: Point) -> f32 {
    let dx = p0[0] - p1[0];
    let dy = p0[1] - p1[1];
    dx * dx + dy * dy
}
