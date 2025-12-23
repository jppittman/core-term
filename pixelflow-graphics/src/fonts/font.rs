//! Font parsing and glyph retrieval.
//!
//! This module provides the [`Font`] struct, which serves as the entry point
//! for loading font data and extracting glyph geometry.

use std::sync::Arc;
use thiserror::Error;
pub use ttf_parser::GlyphId;
use ttf_parser::{Face, FaceParsingError, OutlineBuilder};

use super::loopblinn::{Glyph, LineSegment, LoopBlinnQuad, Point};

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
        Ok(Self {
            face: Arc::new(face),
        })
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
    /// a [`Glyph`] which implements [`Surface<u8>`](pixelflow_core::traits::Surface).
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

        let units_per_em = self.face.units_per_em() as f32;
        let scale = size / units_per_em;

        let pixel_advance = self.face.glyph_hor_advance(glyph_id).unwrap_or(0) as f32 * scale;
        let bbox = self
            .face
            .glyph_bounding_box(glyph_id)
            .unwrap_or(ttf_parser::Rect {
                x_min: 0,
                y_min: 0,
                x_max: 0,
                y_max: 0,
            });

        // Step 1: Translate bbox to origin (in Builder, to get [0, W] x [0, H])
        // Note: transform is y * -scale + offset.
        // We want y_raw range [y_min, y_max] to map to [0, H].
        // y_max -> 0. y_min -> H.
        // We want (x_min, y_min) to map to (0,0) in the "raw" manifold.
        // So we translate by (-x_min, -y_min).
        let offset_x = -bbox.x_min as f32;
        let offset_y = bbox.y_max as f32; // Invert logic: map y_max to 0
        let mut builder = GlyphBuilder::new(1.0, [offset_x, offset_y]);
        let _ = self.face.outline_glyph(glyph_id, &mut builder);

        use super::loopblinn::AlgebraicGlyph;
        use crate::shapes::Square;
        use crate::transform::{Scale, Translate};

        let raw = AlgebraicGlyph {
            line_segments: Arc::from(builder.line_segments),
            quad_segments: Arc::from(builder.quad_segments),
        };
        // DEBUG: Check if we have segments
        // println!("Glyph Segments: Lines={}, Quads={}", raw.line_segments.len(), raw.quad_segments.len());
        // Force unbuffer stdout to ensure we see it
        use std::io::Write;
        let _ = std::io::stdout().flush();
        println!(
            "Glyph Segments: Lines={}, Quads={}",
            raw.line_segments.len(),
            raw.quad_segments.len()
        );

        // Step 2: Normalize to [0,1] box.
        // We use max_dim to maintain aspect ratio.
        let width = (bbox.x_max - bbox.x_min) as f32;
        let height = (bbox.y_max - bbox.y_min) as f32;
        let max_dim = width.max(height);

        // DEBUG: Print metrics
        println!(
            "Glyph BBox: {:?}, Width: {}, Height: {}, MaxDim: {}",
            bbox, width, height, max_dim
        );
        println!("Offsets: x={}, y={}", offset_x, offset_y);

        // Input x is [0, W]. x / max_dim is [0, W/max_dim] <= 1.0. Correct.
        let normalized = Scale {
            manifold: raw,
            factor: 1.0 / max_dim,
        };

        // Step 3: AABB clip (Unit Square)
        let boxed = Square {
            fg: normalized,
            bg: 0.0,
        };

        // Step 4: Scale to pixel size
        // Maps 1.0 (normalized) to `size`.
        // Effective scale: size / max_dim.
        let sized = Scale {
            manifold: boxed,
            factor: size,
        };

        // Step 5: Translate BACK to original EM position (scaled by size)
        let back_x = 0.0;
        let back_y = 0.0;

        let restored = Translate {
            manifold: sized,
            offset: [back_x, back_y],
        };

        Some(Glyph {
            advance: pixel_advance,
            manifold: restored,
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
    line_segments: Vec<LineSegment>,
    quad_segments: Vec<LoopBlinnQuad>,
    scale: f32,
    offset: Point,
    current: Point,
    start: Point,
}

impl GlyphBuilder {
    fn new(scale: f32, offset: Point) -> Self {
        Self {
            line_segments: Vec::with_capacity(32),
            quad_segments: Vec::with_capacity(32),
            scale,
            offset,
            current: [0.0, 0.0],
            start: [0.0, 0.0],
        }
    }

    // Recursively subdivide cubic curves into line segments if they are small enough,
    // or eventually approximate with quadratics.
    #[allow(clippy::too_many_arguments)]
    fn subdivide_cubic(&mut self, p0: Point, p1: Point, p2: Point, p3: Point, depth: u32) {
        let d03 = dist_sq(p0, p3);
        const MIN_LEN_SQ: f32 = 0.25; // 0.5 px

        if depth > 8 || d03 < MIN_LEN_SQ {
            self.line_segments.push(LineSegment::new(p0, p3));
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

    fn transform(&self, x: f32, y: f32) -> Point {
        [
            x * self.scale + self.offset[0],
            y * -self.scale + self.offset[1],
        ]
    }
}

impl OutlineBuilder for GlyphBuilder {
    fn move_to(&mut self, x: f32, y: f32) {
        self.current = self.transform(x, y);
        self.start = self.current;
    }

    fn line_to(&mut self, x: f32, y: f32) {
        let p1 = self.transform(x, y);
        self.line_segments.push(LineSegment::new(self.current, p1));
        self.current = p1;
    }

    fn quad_to(&mut self, x1: f32, y1: f32, x: f32, y: f32) {
        let p1 = self.transform(x1, y1);
        let p2 = self.transform(x, y);

        if let Some(q) = LoopBlinnQuad::new(self.current, p1, p2) {
            self.quad_segments.push(q);
        } else {
            self.line_segments.push(LineSegment::new(self.current, p2));
        }
        self.current = p2;
    }

    fn curve_to(&mut self, x1: f32, y1: f32, x2: f32, y2: f32, x: f32, y: f32) {
        let p1 = self.transform(x1, y1);
        let p2 = self.transform(x2, y2);
        let p3 = self.transform(x, y);

        self.subdivide_cubic(self.current, p1, p2, p3, 0);
        self.current = p3;
    }

    fn close(&mut self) {
        if (self.current[0] - self.start[0]).abs() > 1e-4
            || (self.current[1] - self.start[1]).abs() > 1e-4
        {
            self.line_segments
                .push(LineSegment::new(self.current, self.start));
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
