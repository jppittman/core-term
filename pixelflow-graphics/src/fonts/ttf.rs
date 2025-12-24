//! pixelflow-graphics/src/fonts/ttf.rs
//!
//! Pure Manifold TTF Parser.
//!
//! Glyphs are parsed into unit space [0,1]², bounded with Select for
//! short-circuit evaluation, then wrapped in Affine transforms.

use crate::shapes::{square, Bounded};
use pixelflow_core::{Field, Jet2, Manifold, Numeric};
use std::sync::Arc;

// ═══════════════════════════════════════════════════════════════════════════
// Combinators
// ═══════════════════════════════════════════════════════════════════════════

/// Affine transform combinator - generic over any inner manifold.
#[derive(Clone, Debug)]
pub struct Affine<M> {
    pub inner: M,
    pub inv: [f32; 6], // [a b c d tx ty] inverted
}

impl<M> Affine<M> {
    pub fn new(inner: M, [a, b, c, d, tx, ty]: [f32; 6]) -> Self {
        let det = a * d - b * c;
        let inv_det = if det.abs() < 1e-6 { 0.0 } else { 1.0 / det };
        Self {
            inner,
            inv: [
                d * inv_det,
                -b * inv_det,
                -c * inv_det,
                a * inv_det,
                tx,
                ty,
            ],
        }
    }
}

/// Generic Affine implementation for any inner manifold.
impl<M, I> Manifold<I> for Affine<M>
where
    I: Numeric,
    M: Manifold<I>,
{
    type Output = M::Output;

    #[inline(always)]
    fn eval_raw(&self, x: I, y: I, z: I, w: I) -> Self::Output {
        let [a, b, c, d, tx, ty] = self.inv;
        let x2 = (x - I::from_f32(tx)) * I::from_f32(a) + (y - I::from_f32(ty)) * I::from_f32(b);
        let y2 = (x - I::from_f32(tx)) * I::from_f32(c) + (y - I::from_f32(ty)) * I::from_f32(d);
        self.inner.eval_raw(x2, y2, z, w)
    }
}

/// Monoid sum - accumulates winding numbers from multiple segments/glyphs.
#[derive(Clone, Debug)]
pub struct Sum<M>(pub Arc<[M]>);

/// Generic Sum implementation for any inner manifold.
impl<M, I> Manifold<I> for Sum<M>
where
    I: Numeric,
    M: Manifold<I, Output = I>,
{
    type Output = I;

    #[inline(always)]
    fn eval_raw(&self, x: I, y: I, z: I, w: I) -> I {
        self.0
            .iter()
            .fold(I::from_f32(0.0), |acc, m| acc + m.eval_raw(x, y, z, w))
    }
}

/// Threshold combinator - converts winding number to inside/outside (0 or 1).
///
/// Applies the non-zero winding rule: |winding| >= 0.5 means inside.
#[derive(Clone, Debug)]
pub struct Threshold<M>(pub M);

impl<M, I> Manifold<I> for Threshold<M>
where
    I: Numeric,
    M: Manifold<I, Output = I>,
{
    type Output = I;

    #[inline(always)]
    fn eval_raw(&self, x: I, y: I, z: I, w: I) -> I {
        let winding = self.0.eval_raw(x, y, z, w);
        // Non-zero winding rule: |winding| >= 0.5 means inside
        let inside = winding.abs().ge(I::from_f32(0.5));
        I::select(inside, I::from_f32(1.0), I::from_f32(0.0))
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Geometry
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Clone, Copy, Debug)]
pub struct Curve<const N: usize>(pub [[f32; 2]; N]);

pub type Line = Curve<2>;
pub type Quad = Curve<3>;

#[derive(Clone, Copy, Debug)]
pub enum Segment {
    Line(Line),
    Quad(Quad),
}

// ─── Field Implementation (Aliased / Hard Edges) ───────────────────────────

impl Manifold<Field> for Line {
    type Output = Field;
    fn eval_raw(&self, x: Field, y: Field, _: Field, _: Field) -> Field {
        let [[x0, y0], [x1, y1]] = self.0;
        let (dy, dx) = (y1 - y0, x1 - x0);
        if dy.abs() < 1e-6 {
            return Field::from(0.0);
        }
        // Use bitwise AND (&) for combining masks, not multiplication (*)
        // SIMD comparison results are bit masks (0xFFFFFFFF for true),
        // and multiplying them gives NaN, not a valid mask.
        let (y0f, y1f) = (Field::from(y0), Field::from(y1));
        let in_y = y.ge(y0f.min(y1f)) & y.lt(y0f.max(y1f));
        let x_int = Field::from(x0) + (y - y0f) * Field::from(dx / dy);
        let dir = if dy > 0.0 { Field::from(1.0) } else { Field::from(-1.0) };
        Field::select(in_y & x.lt(x_int), dir, Field::from(0.0))
    }
}

impl Manifold<Field> for Quad {
    type Output = Field;
    fn eval_raw(&self, x: Field, y: Field, _: Field, _: Field) -> Field {
        let [[x0, y0], [x1, y1], [x2, y2]] = self.0;
        let (ay, by, cy) = (y0 - 2.0 * y1 + y2, 2.0 * (y1 - y0), y0);
        let (ax, bx, cx) = (x0 - 2.0 * x1 + x2, 2.0 * (x1 - x0), x0);

        let eval_t = |t: Field| -> Field {
            let in_t = t.ge(Field::from(0.0)) & t.lt(Field::from(1.0));
            if !in_t.any() { return Field::from(0.0); }
            let x_int = (Field::from(ax) * t + Field::from(bx)) * t + Field::from(cx);
            let dy_dt = Field::from(2.0 * ay) * t + Field::from(by);
            let dir = Field::select(dy_dt.gt(Field::from(0.0)), Field::from(1.0), Field::from(-1.0));
            Field::select(in_t & x.lt(x_int), dir, Field::from(0.0))
        };

        if ay.abs() < 1e-6 {
            if by.abs() < 1e-6 { return Field::from(0.0); }
            eval_t((y - Field::from(cy)) / Field::from(by))
        } else {
            let c_val = Field::from(cy) - y;
            let d = Field::from(by * by) - Field::from(4.0 * ay) * c_val;
            let valid = d.ge(Field::from(0.0));
            let sd = d.abs().sqrt();
            let t1 = (Field::from(-by) - sd) / Field::from(2.0 * ay);
            let t2 = (Field::from(-by) + sd) / Field::from(2.0 * ay);
            Field::select(valid, eval_t(t1) + eval_t(t2), Field::from(0.0))
        }
    }
}

// ─── Jet2 Implementation (Anti-Aliased / Smooth Edges) ─────────────────────

impl Manifold<Jet2> for Line {
    type Output = Jet2;
    fn eval_raw(&self, x: Jet2, y: Jet2, _: Jet2, _: Jet2) -> Jet2 {
        let [[x0, y0], [x1, y1]] = self.0;
        let (dy, dx) = (y1 - y0, x1 - x0);
        if dy.abs() < 1e-6 { return Jet2::from_f32(0.0); }
        let (y0f, y1f) = (Jet2::from_f32(y0), Jet2::from_f32(y1));
        let in_y = y.ge(y0f.min(y1f)) & y.lt(y0f.max(y1f));
        if !in_y.val.any() { return Jet2::from_f32(0.0); }

        let x_int = Jet2::from_f32(x0) + (y - y0f) * Jet2::from_f32(dx / dy);
        let dir = if dy > 0.0 { Jet2::from_f32(1.0) } else { Jet2::from_f32(-1.0) };

        let dist = x_int - x;
        let grad_mag = (dist.dx * dist.dx + dist.dy * dist.dy).sqrt().max(Field::from(1e-6));
        let coverage = (dist.val / grad_mag + Field::from(0.5)).max(Field::from(0.0)).min(Field::from(1.0));
        
        Jet2::select(in_y, dir * Jet2::constant(coverage), Jet2::from_f32(0.0))
    }
}

impl Manifold<Jet2> for Quad {
    type Output = Jet2;
    fn eval_raw(&self, x: Jet2, y: Jet2, _: Jet2, _: Jet2) -> Jet2 {
        let [[x0, y0], [x1, y1], [x2, y2]] = self.0;
        let (ay, by, cy) = (y0 - 2.0 * y1 + y2, 2.0 * (y1 - y0), y0);
        let (ax, bx, cx) = (x0 - 2.0 * x1 + x2, 2.0 * (x1 - x0), x0);

        let eval_t = |t: Jet2| -> Jet2 {
            let in_t = t.ge(Jet2::from_f32(0.0)) & t.lt(Jet2::from_f32(1.0));
            if !in_t.val.any() { return Jet2::from_f32(0.0); }
            let x_int = (Jet2::from_f32(ax) * t + Jet2::from_f32(bx)) * t + Jet2::from_f32(cx);
            let dy_dt = Jet2::from_f32(2.0 * ay) * t + Jet2::from_f32(by);
            let dir = Jet2::select(dy_dt.gt(Jet2::from_f32(0.0)), Jet2::from_f32(1.0), Jet2::from_f32(-1.0));
            let dist = x_int - x;
            let grad_mag = (dist.dx * dist.dx + dist.dy * dist.dy).sqrt().max(Field::from(1e-6));
            let coverage = (dist.val / grad_mag + Field::from(0.5)).max(Field::from(0.0)).min(Field::from(1.0));
            Jet2::select(in_t, dir * Jet2::constant(coverage), Jet2::from_f32(0.0))
        };

        if ay.abs() < 1e-6 {
            if by.abs() < 1e-6 { return Jet2::from_f32(0.0); }
            eval_t((y - Jet2::from_f32(cy)) / Jet2::from_f32(by))
        } else {
            let c_val = Jet2::from_f32(cy) - y;
            let d = Jet2::from_f32(by * by) - Jet2::from_f32(4.0 * ay) * c_val;
            let valid = d.ge(Jet2::from_f32(0.0));
            let sd = d.abs().sqrt();
            let t1 = (Jet2::from_f32(-by) - sd) / Jet2::from_f32(2.0 * ay);
            let t2 = (Jet2::from_f32(-by) + sd) / Jet2::from_f32(2.0 * ay);
            Jet2::select(valid, eval_t(t1) + eval_t(t2), Jet2::from_f32(0.0))
        }
    }
}

impl Manifold<Field> for Segment {
    type Output = Field;
    fn eval_raw(&self, x: Field, y: Field, z: Field, w: Field) -> Field {
        match self {
            Self::Line(c) => c.eval_raw(x, y, z, w),
            Self::Quad(c) => c.eval_raw(x, y, z, w),
        }
    }
}

impl Manifold<Jet2> for Segment {
    type Output = Jet2;
    fn eval_raw(&self, x: Jet2, y: Jet2, z: Jet2, w: Jet2) -> Jet2 {
        match self {
            Self::Line(c) => c.eval_raw(x, y, z, w),
            Self::Quad(c) => c.eval_raw(x, y, z, w),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Glyph (Compositional Scene Graph)
// ═══════════════════════════════════════════════════════════════════════════

/// A simple glyph: segments in unit space, thresholded, bounded, then transformed.
///
/// The composition is: Affine<Select<UnitSquare, Threshold<Sum<Segment>>, 0.0>>
/// - Sum<Segment>: Accumulates winding numbers from curve segments
/// - Threshold: Converts winding to 0/1 via non-zero rule
/// - Select (via square): Bounds check with short-circuit
/// - Affine: Restores to font coordinate space
pub type SimpleGlyph = Affine<Bounded<Threshold<Sum<Segment>>>>;

/// A compound glyph: sum of transformed child glyphs.
pub type CompoundGlyph = Sum<Affine<Glyph>>;

/// A glyph is either empty, a simple outline, or a compound of sub-glyphs.
#[derive(Clone)]
pub enum Glyph {
    /// No geometry - evaluates to 0.
    Empty,
    /// Simple glyph: bounded, thresholded segments in unit space.
    Simple(SimpleGlyph),
    /// Compound glyph: sum of transformed child glyphs.
    Compound(CompoundGlyph),
}

impl core::fmt::Debug for Glyph {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Empty => write!(f, "Glyph::Empty"),
            Self::Simple(_) => write!(f, "Glyph::Simple(...)"),
            Self::Compound(_) => write!(f, "Glyph::Compound(...)"),
        }
    }
}

// Glyph evaluation - concrete impls because Line/Quad/Segment have
// different implementations for Field (hard) vs Jet2 (anti-aliased).

impl Manifold<Field> for Glyph {
    type Output = Field;

    #[inline(always)]
    fn eval_raw(&self, x: Field, y: Field, z: Field, w: Field) -> Field {
        match self {
            Self::Empty => Field::from(0.0),
            Self::Simple(g) => g.eval_raw(x, y, z, w),
            Self::Compound(g) => g.eval_raw(x, y, z, w),
        }
    }
}

impl Manifold<Jet2> for Glyph {
    type Output = Jet2;

    #[inline(always)]
    fn eval_raw(&self, x: Jet2, y: Jet2, z: Jet2, w: Jet2) -> Jet2 {
        match self {
            Self::Empty => Jet2::from_f32(0.0),
            Self::Simple(g) => g.eval_raw(x, y, z, w),
            Self::Compound(g) => g.eval_raw(x, y, z, w),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Reader
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Clone, Copy)]
struct R<'a>(&'a [u8], usize);

impl<'a> R<'a> {
    fn u8(&mut self) -> Option<u8> {
        let v = *self.0.get(self.1)?;
        self.1 += 1;
        Some(v)
    }
    fn i8(&mut self) -> Option<i8> {
        self.u8().map(|v| v as i8)
    }
    fn u16(&mut self) -> Option<u16> {
        let s = self.0.get(self.1..self.1 + 2)?;
        self.1 += 2;
        Some(u16::from_be_bytes(s.try_into().ok()?))
    }
    fn i16(&mut self) -> Option<i16> {
        self.u16().map(|v| v as i16)
    }
    fn u32(&mut self) -> Option<u32> {
        let s = self.0.get(self.1..self.1 + 4)?;
        self.1 += 4;
        Some(u32::from_be_bytes(s.try_into().ok()?))
    }
    fn skip(&mut self, n: usize) -> Option<()> {
        self.0.get(self.1..self.1 + n)?;
        self.1 += n;
        Some(())
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tables (Dependent Types)
// ═══════════════════════════════════════════════════════════════════════════

enum Loca<'a> {
    Short(&'a [u8]),
    Long(&'a [u8]),
}

impl Loca<'_> {
    fn get(&self, i: usize) -> Option<usize> {
        match self {
            Self::Short(d) => Some(R(*d, i * 2).u16()? as usize * 2),
            Self::Long(d) => Some(R(*d, i * 4).u32()? as usize),
        }
    }
}

enum Cmap<'a> {
    Fmt4(&'a [u8]),
    Fmt12(&'a [u8]),
}

impl Cmap<'_> {
    fn lookup(&self, c: u32) -> Option<u16> {
        match self {
            Self::Fmt4(d) if c <= 0xFFFF => {
                let n = R(*d, 6).u16()? as usize / 2;
                (0..n).find_map(|i| {
                    let end = R(*d, 14 + i * 2).u16()?;
                    if c as u16 > end {
                        return None;
                    }
                    let start = R(*d, 16 + n * 2 + i * 2).u16()?;
                    if (c as u16) < start {
                        return Some(0);
                    }
                    let delta = R(*d, 16 + n * 4 + i * 2).i16()?;
                    let range = R(*d, 16 + n * 6 + i * 2).u16()?;
                    Some(if range == 0 {
                        (c as i16).wrapping_add(delta) as u16
                    } else {
                        let off = 16 + n * 6 + i * 2 + range as usize + (c as u16 - start) as usize * 2;
                        let g = R(*d, off).u16()?;
                        if g == 0 { 0 } else { (g as i16).wrapping_add(delta) as u16 }
                    })
                })
            }
            Self::Fmt12(d) => (0..R(*d, 12).u32()? as usize).find_map(|i| {
                let (s, e, g) = (
                    R(*d, 16 + i * 12).u32()?,
                    R(*d, 20 + i * 12).u32()?,
                    R(*d, 24 + i * 12).u32()?,
                );
                (c >= s && c <= e).then(|| (g + c - s) as u16)
            }),
            _ => None,
        }
    }
}

enum Kern<'a> {
    /// Format 0: sorted pairs (left_glyph, right_glyph, value)
    Fmt0 { data: &'a [u8], n_pairs: usize },
    /// No kerning table
    None,
}

impl<'a> Kern<'a> {
    fn parse(data: &'a [u8]) -> Self {
        let Some(n_tables) = R(data, 2).u16() else { return Self::None };
        let mut off = 4;

        for _ in 0..n_tables {
            let Some(length) = R(data, off + 2).u16() else { return Self::None };
            let Some(coverage) = R(data, off + 4).u16() else { return Self::None };

            let format = coverage >> 8;
            let horizontal = coverage & 1;

            if format == 0 && horizontal == 1 {
                let Some(n_pairs) = R(data, off + 6).u16() else { return Self::None };
                return Self::Fmt0 {
                    data: &data[off + 14..], // Skip header to pairs
                    n_pairs: n_pairs as usize,
                };
            }
            off += length as usize;
        }
        Self::None
    }

    fn get(&self, left: u16, right: u16) -> i16 {
        match self {
            Self::Fmt0 { data, n_pairs } => {
                // Binary search: each pair is 6 bytes (left:2, right:2, value:2)
                let key = ((left as u32) << 16) | (right as u32);
                let (mut lo, mut hi) = (0, *n_pairs);

                while lo < hi {
                    let mid = (lo + hi) / 2;
                    let pair = ((R(*data, mid * 6).u16().unwrap_or(0) as u32) << 16)
                        | (R(*data, mid * 6 + 2).u16().unwrap_or(0) as u32);

                    match pair.cmp(&key) {
                        std::cmp::Ordering::Less => lo = mid + 1,
                        std::cmp::Ordering::Greater => hi = mid,
                        std::cmp::Ordering::Equal => {
                            return R(*data, mid * 6 + 4).i16().unwrap_or(0)
                        }
                    }
                }
                0
            }
            Self::None => 0,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Font
// ═══════════════════════════════════════════════════════════════════════════

pub struct Font<'a> {
    data: &'a [u8],
    glyf: usize,
    loca: Loca<'a>,
    cmap: Cmap<'a>,
    kern: Kern<'a>,
    hmtx: usize,
    num_hm: usize,
    pub units_per_em: u16,
    pub ascent: i16,
    pub descent: i16,
    pub line_gap: i16,
}

impl<'a> Font<'a> {
    pub fn parse(data: &'a [u8]) -> Option<Self> {
        // TTF header: sfntVersion(4) + numTables(2) + searchRange(2) + entrySelector(2) + rangeShift(2) = 12 bytes
        // Table record: tag(4) + checksum(4) + offset(4) + length(4) = 16 bytes
        let num_tables = R(data, 4).u16()? as usize;
        let mut t = std::collections::HashMap::new();

        for i in 0..num_tables {
            let rec = 12 + i * 16;
            let tag = [data[rec], data[rec + 1], data[rec + 2], data[rec + 3]];
            let offset = R(data, rec + 8).u32()? as usize;
            t.insert(tag, offset);
        }

        let head = *t.get(b"head")?;
        let loca = *t.get(b"loca")?;
        let hhea = *t.get(b"hhea")?;

        Some(Self {
            data,
            glyf: *t.get(b"glyf")?,
            loca: if R(data, head + 50).i16()? != 0 {
                Loca::Long(&data[loca..])
            } else {
                Loca::Short(&data[loca..])
            },
            cmap: Self::find_cmap(&data[*t.get(b"cmap")?..])?,
            kern: t.get(b"kern").map(|&off| Kern::parse(&data[off..])).unwrap_or(Kern::None),
            hmtx: *t.get(b"hmtx")?,
            num_hm: R(data, hhea + 34).u16()? as usize,
            units_per_em: R(data, head + 18).u16()?,
            ascent: R(data, hhea + 4).i16()?,
            descent: R(data, hhea + 6).i16()?,
            line_gap: R(data, hhea + 8).i16()?,
        })
    }

    fn find_cmap(d: &'a [u8]) -> Option<Cmap<'a>> {
        (0..R(d, 2).u16()? as usize)
            .filter_map(|i| {
                let (p, e, o) = (
                    R(d, 4 + i * 8).u16()?,
                    R(d, 6 + i * 8).u16()?,
                    R(d, 8 + i * 8).u32()? as usize,
                );
                let f = R(d, o).u16()?;
                match (p, e, f) {
                    (3, 10, 12) | (0, 4, 12) => Some((2, o, f)),
                    (3, 1, 4) | (0, 3, 4) => Some((1, o, f)),
                    _ => None,
                }
            })
            .max_by_key(|x| x.0)
            .and_then(|(_, o, f)| match f {
                4 => Some(Cmap::Fmt4(&d[o..])),
                12 => Some(Cmap::Fmt12(&d[o..])),
                _ => None,
            })
    }

    pub fn glyph(&self, ch: char) -> Option<Glyph> {
        self.compile(self.cmap.lookup(ch as u32)?)
    }

    pub fn glyph_scaled(&self, ch: char, size: f32) -> Option<Glyph> {
        let g = self.glyph(ch)?;
        let scale = size / self.units_per_em as f32;
        // Transform: scale X, flip Y (screen Y goes down), and translate by ascent
        // so the top of the text is at Y=0 in screen coordinates.
        let y_offset = self.ascent as f32 * scale;
        Some(Glyph::Compound(Sum(
            [Affine::new(g, [scale, 0.0, 0.0, -scale, 0.0, y_offset])].into(),
        )))
    }

    pub fn advance(&self, ch: char) -> Option<f32> {
        let id = self.cmap.lookup(ch as u32)?;
        let i = (id as usize).min(self.num_hm.saturating_sub(1));
        Some(R(self.data, self.hmtx + i * 4).u16()? as f32)
    }

    pub fn advance_scaled(&self, ch: char, size: f32) -> Option<f32> {
        Some(self.advance(ch)? * size / self.units_per_em as f32)
    }

    /// Get kerning adjustment between two characters in font units.
    pub fn kern(&self, left: char, right: char) -> f32 {
        let left_id = self.cmap.lookup(left as u32).unwrap_or(0);
        let right_id = self.cmap.lookup(right as u32).unwrap_or(0);
        self.kern.get(left_id, right_id) as f32
    }

    /// Get kerning adjustment between two characters, scaled to size.
    pub fn kern_scaled(&self, left: char, right: char, size: f32) -> f32 {
        self.kern(left, right) * size / self.units_per_em as f32
    }

    fn compile(&self, id: u16) -> Option<Glyph> {
        let (a, b) = (
            self.loca.get(id as usize)?,
            self.loca.get(id as usize + 1)?,
        );
        if a == b {
            return Some(Glyph::Empty);
        }
        let mut r = R(self.data, self.glyf + a);
        let n = r.i16()?;
        let x_min = r.i16()?;
        let y_min = r.i16()?;
        let x_max = r.i16()?;
        let y_max = r.i16()?;

        let width = (x_max - x_min) as f32;
        let height = (y_max - y_min) as f32;
        let max_dim = width.max(height).max(1.0); // Avoid div by 0

        // Normalize transform: map [x_min, x_min+max_dim] -> [0, 1]
        let norm_scale = 1.0 / max_dim;
        let norm_tx = -(x_min as f32) * norm_scale;
        let norm_ty = -(y_min as f32) * norm_scale;

        // The restore transform maps [0, 1] back to font units
        // x_world = x_local * max_dim + x_min
        let restore = [max_dim, 0.0, 0.0, max_dim, x_min as f32, y_min as f32];

        if n >= 0 {
            // Parse segments in normalized [0,1] space
            let sum_segs = self.simple(&mut r, n as usize, norm_scale, norm_tx, norm_ty)?;

            // Compose: Sum -> Threshold -> Bounded (via square) -> Affine
            // This gives us: Affine<Select<UnitSquare, Threshold<Sum<Segment>>, f32>>
            let thresholded = Threshold(sum_segs);
            let bounded = square(thresholded, 0.0f32);
            Some(Glyph::Simple(Affine::new(bounded, restore)))
        } else {
            // Compound glyphs: children are already fully composed with their own bounds
            self.compound(&mut r)
        }
    }

    fn simple(&self, r: &mut R, n: usize, scale: f32, tx: f32, ty: f32) -> Option<Sum<Segment>> {
        if n == 0 {
            return Some(Sum(vec![].into()));
        }
        let ends: Vec<_> = (0..n)
            .map(|_| r.u16().map(|v| v as usize))
            .collect::<Option<_>>()?;
        let np = *ends.last()? + 1;
        let instr_len = r.u16()? as usize;
        r.skip(instr_len)?;

        let mut fl = Vec::with_capacity(np);
        while fl.len() < np {
            let f = r.u8()?;
            fl.push(f);
            if f & 8 != 0 {
                for _ in 0..r.u8()?.min((np - fl.len()) as u8) {
                    fl.push(f);
                }
            }
        }

        let dec = |r: &mut R, s: u8, m: u8| {
            fl.iter()
                .try_fold((0i16, vec![]), |(mut v, mut out), &f| {
                    v += match (f & s != 0, f & m != 0) {
                        (true, true) => r.u8()? as i16,
                        (true, false) => -(r.u8()? as i16),
                        (false, true) => 0,
                        (false, false) => r.i16()?,
                    };
                    out.push(v);
                    Some((v, out))
                })
                .map(|(_, v)| v)
        };

        let (xs, ys) = (dec(r, 2, 16)?, dec(r, 4, 32)?);
        
        // Normalize points immediately
        let pts: Vec<_> = (0..np).map(|i| (
            (xs[i] as f32) * scale + tx, 
            (ys[i] as f32) * scale + ty, 
            fl[i] & 1 != 0
        )).collect();

        let segs: Vec<Segment> = ends
            .iter()
            .scan(0, |s, &e| {
                let c = &pts[*s..=e];
                *s = e + 1;
                Some(c.to_vec())
            })
            .flat_map(|c| to_segs(&c))
            .collect();

        Some(Sum(segs.into()))
    }

    fn compound(&self, r: &mut R) -> Option<Glyph> {
        let mut kids = vec![];
        loop {
            let fl = r.u16()?;
            let id = r.u16()?;
            let (dx, dy) = if fl & 2 != 0 {
                if fl & 1 != 0 {
                    (r.i16()?, r.i16()?)
                } else {
                    (r.i8()? as i16, r.i8()? as i16)
                }
            } else {
                r.skip(if fl & 1 != 0 { 4 } else { 2 })?;
                (0, 0)
            };
            let mut m = [1.0, 0.0, 0.0, 1.0, dx as f32, dy as f32];
            if fl & 0x08 != 0 {
                let s = r.i16()? as f32 / 16384.0;
                m[0] = s;
                m[3] = s;
            } else if fl & 0x40 != 0 {
                m[0] = r.i16()? as f32 / 16384.0;
                m[3] = r.i16()? as f32 / 16384.0;
            } else if fl & 0x80 != 0 {
                m[0] = r.i16()? as f32 / 16384.0;
                m[1] = r.i16()? as f32 / 16384.0;
                m[2] = r.i16()? as f32 / 16384.0;
                m[3] = r.i16()? as f32 / 16384.0;
            }
            if let Some(g) = self.compile(id) {
                kids.push(Affine::new(g, m));
            }
            if fl & 0x20 == 0 {
                break;
            }
        }
        Some(Glyph::Compound(Sum(kids.into())))
    }
}

fn to_segs(pts: &[(f32, f32, bool)]) -> Vec<Segment> {
    if pts.is_empty() {
        return vec![];
    }
    let exp: Vec<_> = pts
        .iter()
        .enumerate()
        .flat_map(|(i, &(x, y, on))| {
            let (nx, ny, non) = pts[(i + 1) % pts.len()];
            if !on && !non {
                vec![(x, y, on), ((x + nx) / 2.0, (y + ny) / 2.0, true)]
            } else {
                vec![(x, y, on)]
            }
        })
        .collect();

    if exp.is_empty() {
        return vec![];
    }

    let start = exp.iter().position(|p| p.2).unwrap_or(0);
    let mut out = vec![];
    let mut i = 0;
    while i < exp.len() {
        let p = |j: usize| {
            let (x, y, _) = exp[(start + j) % exp.len()];
            [x, y]
        };
        if exp[(start + i + 1) % exp.len()].2 {
            out.push(Segment::Line(Curve([p(i), p(i + 1)])));
            i += 1;
        } else {
            out.push(Segment::Quad(Curve([p(i), p(i + 1), p(i + 2)])));
            i += 2;
        }
    }
    out
}
