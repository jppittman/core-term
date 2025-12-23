//! Minimal TTF parser for pixelflow-fonts.
//!
//! Outputs `Vec<Segment>` directly in pixel coordinates.
//! No intermediate normalization. One transform during construction.

use std::collections::HashMap;
use std::ops::Range;
use std::sync::{Arc, RwLock};

use super::curves::{Line, Quadratic, Segment};
use super::glyph::GlyphBounds;

// ============================================================================
// PUBLIC API
// ============================================================================

/// A parsed TrueType font.
pub struct TtfFont<'a> {
    data: &'a [u8],
    tables: HashMap<Tag, Range<usize>>,
    cache: RwLock<HashMap<(u16, u32), Arc<CompiledGlyph>>>, // (glyph_id, size_bits)
}

/// A glyph ready for rendering.
#[derive(Clone)]
pub struct CompiledGlyph {
    pub segments: Arc<[Segment]>,
    pub bounds: GlyphBounds,
    pub advance: f32,
}

/// Font-wide metrics.
#[derive(Clone, Copy, Debug)]
pub struct FontMetrics {
    pub units_per_em: u16,
    pub ascent: i16,
    pub descent: i16,
    pub line_gap: i16,
}

impl<'a> TtfFont<'a> {
    /// Parse a TTF from bytes.
    pub fn parse(data: &'a [u8]) -> Option<Self> {
        let n = u16::from_be_bytes([data[4], data[5]]) as usize;
        let mut tables = HashMap::new();

        for i in 0..n {
            let off = 12 + i * 16;
            let tag = Tag([data[off], data[off + 1], data[off + 2], data[off + 3]]);
            let offset = u32::from_be_bytes([data[off + 8], data[off + 9], data[off + 10], data[off + 11]]) as usize;
            let length = u32::from_be_bytes([data[off + 12], data[off + 13], data[off + 14], data[off + 15]]) as usize;
            tables.insert(tag, offset..offset + length);
        }

        // Validate required tables
        for tag in [Tag::HEAD, Tag::MAXP, Tag::CMAP, Tag::LOCA, Tag::GLYF, Tag::HHEA, Tag::HMTX] {
            if !tables.contains_key(&tag) {
                return None;
            }
        }

        Some(Self {
            data,
            tables,
            cache: RwLock::new(HashMap::new()),
        })
    }

    /// Font metrics.
    pub fn metrics(&self) -> FontMetrics {
        let head = self.table(Tag::HEAD);
        let hhea = self.table(Tag::HHEA);

        FontMetrics {
            units_per_em: u16::from_be_bytes([head[18], head[19]]),
            ascent: i16::from_be_bytes([hhea[4], hhea[5]]),
            descent: i16::from_be_bytes([hhea[6], hhea[7]]),
            line_gap: i16::from_be_bytes([hhea[8], hhea[9]]),
        }
    }

    /// Horizontal advance for a character at given size.
    pub fn advance(&self, ch: char, size: f32) -> f32 {
        let glyph_id = match self.cmap_lookup(ch) {
            Some(id) => id,
            None => return 0.0,
        };

        let scale = size / self.metrics().units_per_em as f32;
        let hmtx = self.table(Tag::HMTX);
        let hhea = self.table(Tag::HHEA);
        let num_h_metrics = u16::from_be_bytes([hhea[34], hhea[35]]) as usize;

        let idx = (glyph_id as usize).min(num_h_metrics.saturating_sub(1));
        let advance = u16::from_be_bytes([hmtx[idx * 4], hmtx[idx * 4 + 1]]);

        advance as f32 * scale
    }

    /// Kerning between two characters.
    pub fn kern(&self, _left: char, _right: char, _size: f32) -> f32 {
        // TODO: implement kern table lookup
        0.0
    }
}

// ============================================================================
// INTERNAL: Table Access
// ============================================================================

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct Tag([u8; 4]);

impl Tag {
    const HEAD: Self = Self(*b"head");
    const MAXP: Self = Self(*b"maxp");
    const CMAP: Self = Self(*b"cmap");
    const LOCA: Self = Self(*b"loca");
    const GLYF: Self = Self(*b"glyf");
    const HHEA: Self = Self(*b"hhea");
    const HMTX: Self = Self(*b"hmtx");
}

impl<'a> TtfFont<'a> {
    fn table(&self, tag: Tag) -> &'a [u8] {
        let range = self.tables.get(&tag).expect("missing table");
        &self.data[range.clone()]
    }

    fn loca_format(&self) -> bool {
        // true = long (4 bytes), false = short (2 bytes)
        let head = self.table(Tag::HEAD);
        i16::from_be_bytes([head[50], head[51]]) != 0
    }

    fn num_glyphs(&self) -> u16 {
        let maxp = self.table(Tag::MAXP);
        u16::from_be_bytes([maxp[4], maxp[5]])
    }

    fn glyph_offset(&self, glyph_id: u16) -> Option<Range<usize>> {
        if glyph_id >= self.num_glyphs() {
            return None;
        }

        let loca = self.table(Tag::LOCA);
        let id = glyph_id as usize;

        let (start, end) = if self.loca_format() {
            let s = u32::from_be_bytes([loca[id * 4], loca[id * 4 + 1], loca[id * 4 + 2], loca[id * 4 + 3]]) as usize;
            let e = u32::from_be_bytes([loca[id * 4 + 4], loca[id * 4 + 5], loca[id * 4 + 6], loca[id * 4 + 7]]) as usize;
            (s, e)
        } else {
            let s = u16::from_be_bytes([loca[id * 2], loca[id * 2 + 1]]) as usize * 2;
            let e = u16::from_be_bytes([loca[id * 2 + 2], loca[id * 2 + 3]]) as usize * 2;
            (s, e)
        };

        Some(start..end)
    }

    fn glyph_data(&self, glyph_id: u16) -> Option<&'a [u8]> {
        let range = self.glyph_offset(glyph_id)?;
        if range.is_empty() {
            return None;
        }
        let glyf = self.table(Tag::GLYF);
        Some(&glyf[range])
    }
}

// ============================================================================
// INTERNAL: cmap
// ============================================================================

impl<'a> TtfFont<'a> {
    fn cmap_lookup(&self, ch: char) -> Option<u16> {
        let cmap = self.table(Tag::CMAP);
        let codepoint = ch as u32;

        if codepoint > 0xFFFF {
            return None; // BMP only for now
        }
        let c = codepoint as u16;

        // Find format 4 subtable
        let num_tables = u16::from_be_bytes([cmap[2], cmap[3]]) as usize;
        let mut subtable_offset = None;

        for i in 0..num_tables {
            let off = 4 + i * 8;
            let platform = u16::from_be_bytes([cmap[off], cmap[off + 1]]);
            let encoding = u16::from_be_bytes([cmap[off + 2], cmap[off + 3]]);
            let offset = u32::from_be_bytes([cmap[off + 4], cmap[off + 5], cmap[off + 6], cmap[off + 7]]) as usize;

            if (platform == 3 && encoding == 1) || (platform == 0 && encoding == 3) {
                subtable_offset = Some(offset);
                break;
            }
            if platform == 0 && subtable_offset.is_none() {
                subtable_offset = Some(offset);
            }
        }

        let offset = subtable_offset?;
        let format = u16::from_be_bytes([cmap[offset], cmap[offset + 1]]);
        if format != 4 {
            return None;
        }

        let seg_count = u16::from_be_bytes([cmap[offset + 6], cmap[offset + 7]]) as usize / 2;
        let end_off = offset + 14;
        let start_off = end_off + seg_count * 2 + 2;
        let delta_off = start_off + seg_count * 2;
        let range_off = delta_off + seg_count * 2;

        for i in 0..seg_count {
            let end_code = u16::from_be_bytes([cmap[end_off + i * 2], cmap[end_off + i * 2 + 1]]);
            if c > end_code {
                continue;
            }

            let start_code = u16::from_be_bytes([cmap[start_off + i * 2], cmap[start_off + i * 2 + 1]]);
            if c < start_code {
                return Some(0);
            }

            let delta = i16::from_be_bytes([cmap[delta_off + i * 2], cmap[delta_off + i * 2 + 1]]);
            let range = u16::from_be_bytes([cmap[range_off + i * 2], cmap[range_off + i * 2 + 1]]);

            if range == 0 {
                return Some((c as i16).wrapping_add(delta) as u16);
            } else {
                let glyph_off = range_off + i * 2 + range as usize + (c - start_code) as usize * 2;
                let glyph_id = u16::from_be_bytes([cmap[glyph_off], cmap[glyph_off + 1]]);
                if glyph_id == 0 {
                    return Some(0);
                }
                return Some((glyph_id as i16).wrapping_add(delta) as u16);
            }
        }

        Some(0)
    }
}

// ============================================================================
// INTERNAL: Glyph Compilation
// ============================================================================

#[derive(Clone, Copy)]
struct RawPoint {
    x: i16,
    y: i16,
    on_curve: bool,
}

struct RawContour {
    points: Vec<RawPoint>,
}

#[derive(Clone, Copy, Default)]
struct BBox {
    x_min: i16,
    y_min: i16,
    x_max: i16,
    y_max: i16,
}

impl<'a> TtfFont<'a> {
    fn compile_glyph(&self, glyph_id: u16, size: f32) -> Option<CompiledGlyph> {
        let metrics = self.metrics();
        let scale = size / metrics.units_per_em as f32;

        // Get raw contours (recursive for compound glyphs)
        let (contours, bbox) = self.parse_glyph(glyph_id)?;

        if contours.is_empty() {
            // Space or empty glyph
            return Some(CompiledGlyph {
                segments: Arc::from([]),
                bounds: GlyphBounds::default(),
                advance: self.advance_raw(glyph_id) as f32 * scale,
            });
        }

        // Transform: origin at (x_min, y_max), Y-flip, scale to pixels
        let ox = bbox.x_min as f32;
        let oy = bbox.y_max as f32; // Y-flip: y_max becomes 0

        let transform = |x: i16, y: i16| -> [f32; 2] {
            [
                (x as f32 - ox) * scale,
                (oy - y as f32) * scale, // Y-flip
            ]
        };

        // Convert contours to segments
        let mut segments = Vec::new();

        for contour in &contours {
            let expanded = expand_contour(&contour.points);
            segments.extend(contour_to_segments(&expanded, &transform));
        }

        let width = ((bbox.x_max - bbox.x_min) as f32 * scale).ceil() as u32;
        let height = ((bbox.y_max - bbox.y_min) as f32 * scale).ceil() as u32;

        Some(CompiledGlyph {
            segments: Arc::from(segments),
            bounds: GlyphBounds {
                width,
                height,
                bearing_x: (bbox.x_min as f32 * scale) as i32,
                bearing_y: (bbox.y_max as f32 * scale) as i32,
            },
            advance: self.advance_raw(glyph_id) as f32 * scale,
        })
    }

    fn advance_raw(&self, glyph_id: u16) -> u16 {
        let hmtx = self.table(Tag::HMTX);
        let hhea = self.table(Tag::HHEA);
        let num_h_metrics = u16::from_be_bytes([hhea[34], hhea[35]]) as usize;
        let idx = (glyph_id as usize).min(num_h_metrics.saturating_sub(1));
        u16::from_be_bytes([hmtx[idx * 4], hmtx[idx * 4 + 1]])
    }

    fn parse_glyph(&self, glyph_id: u16) -> Option<(Vec<RawContour>, BBox)> {
        let data = match self.glyph_data(glyph_id) {
            Some(d) if !d.is_empty() => d,
            _ => return Some((vec![], BBox::default())),
        };

        let num_contours = i16::from_be_bytes([data[0], data[1]]);
        let bbox = BBox {
            x_min: i16::from_be_bytes([data[2], data[3]]),
            y_min: i16::from_be_bytes([data[4], data[5]]),
            x_max: i16::from_be_bytes([data[6], data[7]]),
            y_max: i16::from_be_bytes([data[8], data[9]]),
        };

        if num_contours >= 0 {
            // Simple glyph
            let contours = self.parse_simple_glyph(&data[10..], num_contours as usize);
            Some((contours, bbox))
        } else {
            // Compound glyph
            self.parse_compound_glyph(&data[10..], bbox)
        }
    }

    fn parse_simple_glyph(&self, data: &[u8], num_contours: usize) -> Vec<RawContour> {
        if num_contours == 0 {
            return vec![];
        }

        let mut cursor = 0;

        // End points
        let mut end_points = Vec::with_capacity(num_contours);
        for _ in 0..num_contours {
            end_points.push(u16::from_be_bytes([data[cursor], data[cursor + 1]]) as usize);
            cursor += 2;
        }

        let num_points = end_points.last().map(|&e| e + 1).unwrap_or(0);
        if num_points == 0 {
            return vec![];
        }

        // Skip instructions
        let instr_len = u16::from_be_bytes([data[cursor], data[cursor + 1]]) as usize;
        cursor += 2 + instr_len;

        // Decode flags
        let (flags, flags_consumed) = decode_flags(&data[cursor..], num_points);
        cursor += flags_consumed;

        // Decode coordinates
        let (xs, rest) = decode_coords(&flags, &data[cursor..], 0x02, 0x10);
        let (ys, _) = decode_coords(&flags, rest, 0x04, 0x20);

        // Build contours
        let mut contours = Vec::with_capacity(num_contours);
        let mut start = 0;

        for &end in &end_points {
            let points: Vec<_> = (start..=end)
                .map(|i| RawPoint {
                    x: xs[i],
                    y: ys[i],
                    on_curve: flags[i] & 0x01 != 0,
                })
                .collect();
            contours.push(RawContour { points });
            start = end + 1;
        }

        contours
    }

    fn parse_compound_glyph(&self, data: &[u8], bbox: BBox) -> Option<(Vec<RawContour>, BBox)> {
        let mut cursor = 0;
        let mut all_contours = Vec::new();

        loop {
            let flags = u16::from_be_bytes([data[cursor], data[cursor + 1]]);
            let component_id = u16::from_be_bytes([data[cursor + 2], data[cursor + 3]]);
            cursor += 4;

            const ARG_1_AND_2_ARE_WORDS: u16 = 0x0001;
            const ARGS_ARE_XY_VALUES: u16 = 0x0002;
            const WE_HAVE_A_SCALE: u16 = 0x0008;
            const MORE_COMPONENTS: u16 = 0x0020;
            const WE_HAVE_AN_X_AND_Y_SCALE: u16 = 0x0040;
            const WE_HAVE_A_TWO_BY_TWO: u16 = 0x0080;

            let (dx, dy) = if flags & ARGS_ARE_XY_VALUES != 0 {
                if flags & ARG_1_AND_2_ARE_WORDS != 0 {
                    let dx = i16::from_be_bytes([data[cursor], data[cursor + 1]]);
                    let dy = i16::from_be_bytes([data[cursor + 2], data[cursor + 3]]);
                    cursor += 4;
                    (dx, dy)
                } else {
                    let dx = data[cursor] as i8 as i16;
                    let dy = data[cursor + 1] as i8 as i16;
                    cursor += 2;
                    (dx, dy)
                }
            } else {
                if flags & ARG_1_AND_2_ARE_WORDS != 0 {
                    cursor += 4;
                } else {
                    cursor += 2;
                }
                (0, 0)
            };

            // Skip scale/matrix (simplified: we only support translation)
            if flags & WE_HAVE_A_SCALE != 0 {
                cursor += 2;
            } else if flags & WE_HAVE_AN_X_AND_Y_SCALE != 0 {
                cursor += 4;
            } else if flags & WE_HAVE_A_TWO_BY_TWO != 0 {
                cursor += 8;
            }

            // Recursively get component contours
            if let Some((mut contours, _)) = self.parse_glyph(component_id) {
                // Apply translation
                for contour in &mut contours {
                    for pt in &mut contour.points {
                        pt.x += dx;
                        pt.y += dy;
                    }
                }
                all_contours.extend(contours);
            }

            if flags & MORE_COMPONENTS == 0 {
                break;
            }
        }

        Some((all_contours, bbox))
    }
}

// ============================================================================
// INTERNAL: Flag & Coordinate Decoding
// ============================================================================

fn decode_flags(data: &[u8], num_points: usize) -> (Vec<u8>, usize) {
    let mut flags = Vec::with_capacity(num_points);
    let mut i = 0;

    while flags.len() < num_points {
        let flag = data[i];
        i += 1;
        flags.push(flag);

        if flag & 0x08 != 0 {
            // REPEAT flag
            let repeat = (data[i] as usize).min(num_points - flags.len());
            i += 1;
            for _ in 0..repeat {
                flags.push(flag);
            }
        }
    }

    (flags, i)
}

fn decode_coords<'a>(flags: &[u8], data: &'a [u8], short_bit: u8, same_bit: u8) -> (Vec<i16>, &'a [u8]) {
    let mut coords = Vec::with_capacity(flags.len());
    let mut cursor = 0;
    let mut prev = 0i16;

    for &flag in flags {
        let is_short = flag & short_bit != 0;
        let is_same_or_positive = flag & same_bit != 0;

        let delta = match (is_short, is_same_or_positive) {
            (true, true) => {
                let v = data[cursor] as i16;
                cursor += 1;
                v
            }
            (true, false) => {
                let v = -(data[cursor] as i16);
                cursor += 1;
                v
            }
            (false, true) => 0,
            (false, false) => {
                let v = i16::from_be_bytes([data[cursor], data[cursor + 1]]);
                cursor += 2;
                v
            }
        };

        prev += delta;
        coords.push(prev);
    }

    (coords, &data[cursor..])
}

// ============================================================================
// INTERNAL: Contour → Segments
// ============================================================================

/// Insert implicit on-curve midpoints between consecutive off-curve points.
fn expand_contour(points: &[RawPoint]) -> Vec<RawPoint> {
    if points.is_empty() {
        return vec![];
    }

    let mut result = Vec::with_capacity(points.len() * 2);

    for i in 0..points.len() {
        let curr = points[i];
        let next = points[(i + 1) % points.len()];

        result.push(curr);

        // Two consecutive off-curve points → insert midpoint
        if !curr.on_curve && !next.on_curve {
            result.push(RawPoint {
                x: (curr.x + next.x) / 2,
                y: (curr.y + next.y) / 2,
                on_curve: true,
            });
        }
    }

    result
}

/// Convert expanded contour to Line/Quadratic segments.
fn contour_to_segments<F>(points: &[RawPoint], transform: &F) -> Vec<Segment>
where
    F: Fn(i16, i16) -> [f32; 2],
{
    if points.is_empty() {
        return vec![];
    }

    // Find first on-curve point
    let start = points.iter().position(|p| p.on_curve).unwrap_or(0);
    let n = points.len();

    let mut segments = Vec::new();
    let mut i = 0;

    while i < n {
        let idx0 = (start + i) % n;
        let idx1 = (start + i + 1) % n;

        let p0 = &points[idx0];
        let p1 = &points[idx1];

        if p1.on_curve {
            // Line: on → on
            segments.push(Segment::Line(Line {
                p0: transform(p0.x, p0.y),
                p1: transform(p1.x, p1.y),
            }));
            i += 1;
        } else {
            // Quadratic: on → off → on
            let idx2 = (start + i + 2) % n;
            let p2 = &points[idx2];

            let t0 = transform(p0.x, p0.y);
            let t1 = transform(p1.x, p1.y);
            let t2 = transform(p2.x, p2.y);

            match Quadratic::try_new(t0, t1, t2) {
                Some(q) => segments.push(Segment::Quad(q)),
                None => segments.push(Segment::Line(Line { p0: t0, p1: t2 })),
            }
            i += 2;
        }
    }

    segments
}

// ============================================================================
// COMPAT: Re-export for drop-in replacement
// ============================================================================

pub use TtfFont as Font;

#[derive(Debug)]
pub enum FontError {
    ParseError,
}

impl std::fmt::Display for FontError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Failed to parse font")
    }
}

impl std::error::Error for FontError {}

impl<'a> TtfFont<'a> {
    /// Drop-in replacement for ttf-parser Font::from_bytes
    pub fn from_bytes(data: &'a [u8]) -> Result<Self, FontError> {
        Self::parse(data).ok_or(FontError::ParseError)
    }
}

/// Glyph with pre-computed segments in pixel coordinates.
#[derive(Clone, Debug)]
pub struct Glyph {
    segments: Arc<[Segment]>,
    bounds: GlyphBounds,
    pub advance: f32,
}

impl Glyph {
    pub fn curves(&self) -> &[Segment] {
        &self.segments
    }

    pub fn bounds(&self) -> GlyphBounds {
        self.bounds
    }
}

impl super::glyph::CurveSurface for Glyph {
    fn curves(&self) -> &[Segment] {
        &self.segments
    }

    fn bounds(&self) -> GlyphBounds {
        self.bounds
    }
}

impl pixelflow_core::Manifold for Glyph {
    type Output = pixelflow_core::Field;

    fn eval_raw(
        &self,
        x: pixelflow_core::Field,
        y: pixelflow_core::Field,
        _z: pixelflow_core::Field,
        _w: pixelflow_core::Field,
    ) -> pixelflow_core::Field {
        super::glyph::eval_curves(
            self.curves(),
            self.bounds(),
            x,
            y,
            pixelflow_core::Field::from(0.0),
        )
    }
}

impl<'a> TtfFont<'a> {
    /// Get a glyph (CurveSurface) at a specific pixel size.
    pub fn glyph(&self, ch: char, size: f32) -> Option<Glyph> {
        let compiled = self.compile_glyph_for_char(ch, size)?;
        Some(Glyph {
            segments: compiled.segments,
            bounds: compiled.bounds,
            advance: compiled.advance,
        })
    }

    fn compile_glyph_for_char(&self, ch: char, size: f32) -> Option<CompiledGlyph> {
        let glyph_id = self.cmap_lookup(ch)?;

        // Cache key includes size (as bits to avoid float hashing)
        let size_bits = size.to_bits();
        let key = (glyph_id, size_bits);

        // Check cache
        if let Some(cached) = self.cache.read().unwrap().get(&key) {
            return Some((**cached).clone());
        }

        // Compile
        let compiled = self.compile_glyph(glyph_id, size)?;
        let compiled = Arc::new(compiled);

        // Cache and return
        self.cache.write().unwrap().insert(key, compiled.clone());
        Some((*compiled).clone())
    }
}
