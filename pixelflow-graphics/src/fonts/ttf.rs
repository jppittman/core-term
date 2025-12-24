//! pixelflow-graphics/src/fonts/ttf.rs
//!
//! Pure Manifold TTF Parser.

use pixelflow_core::{Manifold, Numeric};
use std::sync::Arc;

// ═══════════════════════════════════════════════════════════════════════════
// Combinators
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Clone)]
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

impl<I: Numeric, M: Manifold<I>> Manifold<I> for Affine<M> {
    type Output = M::Output;
    fn eval_raw(&self, x: I, y: I, z: I, w: I) -> Self::Output {
        let [a, b, c, d, tx, ty] = self.inv;
        let x2 = (x - I::from_f32(tx)) * I::from_f32(a) + (y - I::from_f32(ty)) * I::from_f32(b);
        let y2 = (x - I::from_f32(tx)) * I::from_f32(c) + (y - I::from_f32(ty)) * I::from_f32(d);
        self.inner.eval_raw(x2, y2, z, w)
    }
}

#[derive(Clone)]
pub struct Sum<M>(pub Arc<[M]>);

impl<I: Numeric, M: Manifold<I, Output = I>> Manifold<I> for Sum<M> {
    type Output = I;
    fn eval_raw(&self, x: I, y: I, z: I, w: I) -> I {
        self.0
            .iter()
            .fold(I::from_f32(0.0), |acc, m| acc + m.eval_raw(x, y, z, w))
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

impl<I: Numeric> Manifold<I> for Line {
    type Output = I;
    fn eval_raw(&self, x: I, y: I, _: I, _: I) -> I {
        let [[x0, y0], [x1, y1]] = self.0;
        let (dy, dx) = (y1 - y0, x1 - x0);
        if dy.abs() < 1e-6 {
            return I::from_f32(0.0);
        }
        let (y0f, y1f) = (I::from_f32(y0), I::from_f32(y1));
        let in_y = y.ge(y0f.min(y1f)) * y.lt(y0f.max(y1f));
        let x_int = I::from_f32(x0) + (y - y0f) * I::from_f32(dx / dy);
        let dir = if dy > 0.0 {
            I::from_f32(1.0)
        } else {
            I::from_f32(-1.0)
        };
        I::select(in_y * x.lt(x_int), dir, I::from_f32(0.0))
    }
}

impl<I: Numeric> Manifold<I> for Quad {
    type Output = I;
    fn eval_raw(&self, x: I, y: I, z: I, w: I) -> I {
        let [[x0, y0], _, [x2, y2]] = self.0;
        // For winding number, endpoints matter most
        let line: Line = Curve([[x0, y0], [x2, y2]]);
        line.eval_raw(x, y, z, w)
    }
}

impl<I: Numeric> Manifold<I> for Segment {
    type Output = I;
    fn eval_raw(&self, x: I, y: I, z: I, w: I) -> I {
        match self {
            Self::Line(c) => c.eval_raw(x, y, z, w),
            Self::Quad(c) => c.eval_raw(x, y, z, w),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Glyph (Recursive Scene Graph)
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Clone)]
pub enum Glyph {
    Empty,
    Simple(Sum<Segment>),
    Compound(Sum<Affine<Glyph>>),
}

impl<I: Numeric> Manifold<I> for Glyph {
    type Output = I;
    fn eval_raw(&self, x: I, y: I, z: I, w: I) -> I {
        let winding = match self {
            Self::Empty => I::from_f32(0.0),
            Self::Simple(s) => s.eval_raw(x, y, z, w),
            Self::Compound(c) => c.eval_raw(x, y, z, w),
        };
        // Convert winding to coverage
        let winding_abs = winding.abs();
        let inside = winding_abs.ge(I::from_f32(0.5));
        I::select(inside, I::from_f32(1.0), I::from_f32(0.0))
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

// ═══════════════════════════════════════════════════════════════════════════
// Font
// ═══════════════════════════════════════════════════════════════════════════

pub struct Font<'a> {
    data: &'a [u8],
    glyf: usize,
    loca: Loca<'a>,
    cmap: Cmap<'a>,
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
        Some(Glyph::Compound(Sum(
            [Affine::new(g, [scale, 0.0, 0.0, -scale, 0.0, 0.0])].into(),
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
        r.skip(8)?;
        if n >= 0 {
            self.simple(&mut r, n as usize)
        } else {
            self.compound(&mut r)
        }
    }

    fn simple(&self, r: &mut R, n: usize) -> Option<Glyph> {
        if n == 0 {
            return Some(Glyph::Empty);
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
        let pts: Vec<_> = (0..np).map(|i| (xs[i], ys[i], fl[i] & 1 != 0)).collect();

        let segs: Vec<Segment> = ends
            .iter()
            .scan(0, |s, &e| {
                let c = &pts[*s..=e];
                *s = e + 1;
                Some(c.to_vec())
            })
            .flat_map(|c| to_segs(&c))
            .collect();

        Some(Glyph::Simple(Sum(segs.into())))
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

fn to_segs(pts: &[(i16, i16, bool)]) -> Vec<Segment> {
    if pts.is_empty() {
        return vec![];
    }
    let exp: Vec<_> = pts
        .iter()
        .enumerate()
        .flat_map(|(i, &(x, y, on))| {
            let (nx, ny, non) = pts[(i + 1) % pts.len()];
            if !on && !non {
                vec![(x, y, on), ((x + nx) / 2, (y + ny) / 2, true)]
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
            [x as f32, y as f32]
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

// ═══════════════════════════════════════════════════════════════════════════
// Compat
// ═══════════════════════════════════════════════════════════════════════════

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

pub struct FontMetrics {
    pub ascent: i16,
    pub descent: i16,
    pub line_gap: i16,
    pub units_per_em: u16,
}

impl<'a> Font<'a> {
    pub fn metrics(&self) -> FontMetrics {
        FontMetrics {
            ascent: self.ascent,
            descent: self.descent,
            line_gap: self.line_gap,
            units_per_em: self.units_per_em,
        }
    }

    /// Kerning - stub for now (TODO: parse kern/GPOS tables)
    pub fn kern(&self, _a: char, _b: char, _size: f32) -> f32 {
        0.0
    }
}
