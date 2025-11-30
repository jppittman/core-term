use crate::Font;
use crate::curves::{Point, Segment, Line, Quadratic};
use pixelflow_core::batch::Batch;
use pixelflow_core::pipe::Surface;
use ttf_parser::OutlineBuilder;
use crate::baked::{LazyBaked, BakedExt};

#[derive(Clone, Copy)]
pub struct GlyphSurface<'a> {
    pub font: &'a Font<'a>,
    pub codepoint: char,
    pub scale: f32,
}

impl<'a> GlyphSurface<'a> {
    pub fn new(font: &'a Font<'a>, codepoint: char) -> Self {
        Self { font, codepoint, scale: 1.0 }
    }

    pub fn with_scale(mut self, scale: f32) -> Self {
        self.scale = scale;
        self
    }
}

pub fn glyph<'a>(font: &'a Font<'a>, c: char) -> GlyphSurface<'a> {
    GlyphSurface::new(font, c)
}

pub fn glyphs<'a>(font: &'a Font<'a>) -> impl Fn(char) -> GlyphSurface<'a> + 'a {
    move |c| glyph(font, c)
}

pub fn glyphs_cached<'a>(font: &'a Font<'a>, width: u32, height: u32) -> impl Fn(char) -> LazyBaked<GlyphSurface<'a>, u8> + 'a {
    move |c| glyph(font, c).baked(width, height)
}

pub fn glyphs_scaled<'a>(font: &'a Font<'a>, size_px: f32) -> impl Fn(char) -> GlyphSurface<'a> + 'a {
    let scale = size_px / font.units_per_em() as f32;
    move |c| glyph(font, c).with_scale(scale)
}

impl<'a> Surface<u8> for GlyphSurface<'a> {
    fn eval(&self, x: Batch<u32>, y: Batch<u32>) -> Batch<u8> {
        let glyph_id = match self.font.face.glyph_index(self.codepoint) {
            Some(id) => id,
            None => return Batch::splat(0),
        };

        let mut builder = EvalBuilder::new(x, y, self.scale);
        let _ = self.font.face.outline_glyph(glyph_id, &mut builder);
        builder.result()
    }
}

struct EvalBuilder {
    xs: [f32; 4],
    ys: [f32; 4],
    winding: [i32; 4],
    min_dist: [f32; 4],
    current: Point,
    start: Point,
    scale: f32,
}

impl EvalBuilder {
    fn new(x: Batch<u32>, y: Batch<u32>, scale: f32) -> Self {
        let xa = x.to_array_usize();
        let ya = y.to_array_usize();

        let mut xs = [0.0; 4];
        let mut ys = [0.0; 4];
        for i in 0..4 {
            xs[i] = xa[i] as f32 + 0.5;
            ys[i] = ya[i] as f32 + 0.5;
        }

        Self {
            xs,
            ys,
            winding: [0; 4],
            min_dist: [1e9; 4],
            current: [0.0, 0.0],
            start: [0.0, 0.0],
            scale,
        }
    }

    fn process_segment(&mut self, segment: &Segment) {
        for i in 0..4 {
            let px = self.xs[i];
            let py = self.ys[i];

            self.winding[i] += segment.winding(px, py);
            let dist = segment.signed_pseudo_distance(px, py);
            if dist.abs() < self.min_dist[i].abs() {
                self.min_dist[i] = dist;
            }
        }
    }

    fn result(self) -> Batch<u8> {
        let mut res = [0u32; 4];
        for i in 0..4 {
            let inside = self.winding[i] != 0;
            let d = self.min_dist[i];
            let signed_dist = if inside { -d.abs() } else { d.abs() };
            let alpha = (0.5 - signed_dist).clamp(0.0, 1.0);
            res[i] = (alpha * 255.0) as u32;
        }
        Batch::new(res[0], res[1], res[2], res[3]).transmute()
    }

    fn subdivide_cubic(&mut self, p0: Point, p1: Point, p2: Point, p3: Point, depth: u32) {
         let d03_sq = (p0[0]-p3[0]).powi(2) + (p0[1]-p3[1]).powi(2);
         const MIN_LEN_SQ: f32 = 0.25;

        if depth > 8 || d03_sq < MIN_LEN_SQ {
            let seg = Segment::Line(Line { p0, p1: p3 });
            self.process_segment(&seg);
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

impl OutlineBuilder for EvalBuilder {
    fn move_to(&mut self, x: f32, y: f32) {
        self.current = [x * self.scale, y * self.scale];
        self.start = self.current;
    }

    fn line_to(&mut self, x: f32, y: f32) {
        let p1 = [x * self.scale, y * self.scale];
        let seg = Segment::Line(Line { p0: self.current, p1 });
        self.process_segment(&seg);
        self.current = p1;
    }

    fn quad_to(&mut self, x1: f32, y1: f32, x: f32, y: f32) {
        let p1 = [x1 * self.scale, y1 * self.scale];
        let p2 = [x * self.scale, y * self.scale];
        let seg = if let Some(q) = Quadratic::try_new(self.current, p1, p2) {
            Segment::Quad(q)
        } else {
             Segment::Line(Line { p0: self.current, p1: p2 })
        };
        self.process_segment(&seg);
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
        if (self.current[0] - self.start[0]).abs() > 1e-4 || (self.current[1] - self.start[1]).abs() > 1e-4 {
             let seg = Segment::Line(Line { p0: self.current, p1: self.start });
             self.process_segment(&seg);
             self.current = self.start;
        }
    }
}

#[inline]
fn lerp(p0: Point, p1: Point, t: f32) -> Point {
    [p0[0] * (1.0 - t) + p1[0] * t, p0[1] * (1.0 - t) + p1[1] * t]
}
