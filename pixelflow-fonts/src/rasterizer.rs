use crate::font::Font;
use crate::curves::{Segment, Line, Quadratic, Point};
use ttf_parser::{GlyphId, OutlineBuilder};
use pixelflow_core::{Batch, pipe::Surface};

#[derive(Clone, Copy, Debug)]
pub enum Hinting { None, Light, Full }

#[derive(Clone, Copy, Debug)]
pub struct RasterConfig {
    pub size: f32,
    pub hinting: Hinting,
}

#[derive(Clone, Copy, Debug)]
pub struct GlyphBounds {
    pub width: u32,
    pub height: u32,
    pub bearing_x: i32,
    pub bearing_y: i32,
}

pub struct Glyph {
    pub segments: Vec<Segment>,
    pub bounds: GlyphBounds,
}

#[derive(Clone, Copy)]
pub struct GlyphSurface<'a> {
    glyph: &'a Glyph,
}

impl<'a> Surface<u8> for GlyphSurface<'a> {
    fn eval(&self, x: Batch<u32>, y: Batch<u32>) -> Batch<u8> {
        let xs = x.to_array_usize();
        let ys = y.to_array_usize();
        let mut results = [0u32; 4];

        for i in 0..4 {
            let px = xs[i] as f32 + 0.5; // pixel center
            let py = ys[i] as f32 + 0.5;

            let mut winding = 0;
            let mut min_signed_dist: f32 = 1000.0;

            for segment in &self.glyph.segments {
                winding += segment.winding(px, py);

                let dist = segment.signed_pseudo_distance(px, py);
                if dist.abs() < min_signed_dist.abs() {
                    min_signed_dist = dist;
                }
            }

            let inside = winding != 0;

            let signed_dist = if inside {
                -min_signed_dist.abs()
            } else {
                min_signed_dist.abs()
            };

            let alpha = (0.5 - signed_dist).clamp(0.0, 1.0);

            results[i] = (alpha * 255.0) as u32;
        }

        Batch::new(results[0], results[1], results[2], results[3]).cast()
    }
}

impl Glyph {
    pub fn surface(&self) -> GlyphSurface<'_> {
        GlyphSurface { glyph: self }
    }
}

pub struct Rasterizer<'a> {
    font: &'a Font<'a>,
    config: RasterConfig,
}

impl<'a> Rasterizer<'a> {
    pub fn new(font: &'a Font<'a>, config: RasterConfig) -> Self {
        Self { font, config }
    }

    pub fn glyph(&self, id: GlyphId) -> Glyph {
        let mut builder = GlyphBuilder::new(self.config.size, self.font.metrics().units_per_em);

        let _ = self.font.face().outline_glyph(id, &mut builder);

        let bbox = self.font.face().glyph_bounding_box(id).unwrap_or(ttf_parser::Rect { x_min:0, y_min:0, x_max:0, y_max:0 });
        let scale = builder.scale;

        let bounds = GlyphBounds {
            width: ((bbox.x_max - bbox.x_min) as f32 * scale).ceil() as u32,
            height: ((bbox.y_max - bbox.y_min) as f32 * scale).ceil() as u32,
            bearing_x: (bbox.x_min as f32 * scale).round() as i32,
            bearing_y: (bbox.y_max as f32 * scale).round() as i32,
        };

        Glyph {
            segments: builder.segments,
            bounds,
        }
    }

    pub fn glyph_with_bounds(&self, id: GlyphId) -> (Glyph, GlyphBounds) {
        let g = self.glyph(id);
        let b = g.bounds;
        (g, b)
    }
}

struct GlyphBuilder {
    segments: Vec<Segment>,
    scale: f32,
    current: Point,
    start: Point,
}

fn lerp(p0: Point, p1: Point, t: f32) -> Point {
    [
        p0[0] * (1.0 - t) + p1[0] * t,
        p0[1] * (1.0 - t) + p1[1] * t,
    ]
}

impl GlyphBuilder {
    fn new(size: f32, units_per_em: u16) -> Self {
        let scale = size / units_per_em as f32;
        Self {
            segments: Vec::with_capacity(32),
            scale,
            current: [0.0, 0.0],
            start: [0.0, 0.0],
        }
    }

    fn subdivide_cubic(&mut self, p0: Point, p1: Point, p2: Point, p3: Point, depth: u32) {
        if depth > 2 {
            self.segments.push(Segment::Line(Line { p0, p1: p3 }));
            return;
        }

        let p01 = lerp(p0, p1, 0.5);
        let p12 = lerp(p1, p2, 0.5);
        let p23 = lerp(p2, p3, 0.5);

        let p012 = lerp(p01, p12, 0.5);
        let p123 = lerp(p12, p23, 0.5);

        let p0123 = lerp(p012, p123, 0.5);

        self.subdivide_cubic(p0, p01, p012, p0123, depth+1);
        self.subdivide_cubic(p0123, p123, p23, p3, depth+1);
    }
}

impl OutlineBuilder for GlyphBuilder {
    fn move_to(&mut self, x: f32, y: f32) {
        self.current = [x * self.scale, y * self.scale];
        self.start = self.current;
    }
    fn line_to(&mut self, x: f32, y: f32) {
        let p1 = [x * self.scale, y * self.scale];
        self.segments.push(Segment::Line(Line { p0: self.current, p1 }));
        self.current = p1;
    }
    fn quad_to(&mut self, x1: f32, y1: f32, x: f32, y: f32) {
        let p1 = [x1 * self.scale, y1 * self.scale];
        let p2 = [x * self.scale, y * self.scale];

        if let Some(q) = Quadratic::try_new(self.current, p1, p2) {
            self.segments.push(Segment::Quad(q));
        } else {
            // Degenerate, fallback to line
            self.segments.push(Segment::Line(Line { p0: self.current, p1: p2 }));
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
        if (self.current[0] - self.start[0]).abs() > 1e-4 || (self.current[1] - self.start[1]).abs() > 1e-4 {
             self.segments.push(Segment::Line(Line { p0: self.current, p1: self.start }));
             self.current = self.start;
        }
    }
}
