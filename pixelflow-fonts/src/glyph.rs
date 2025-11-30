use crate::Font;
use crate::curves::{Segment, Point};
use crate::lazy::{Lazy, Factory};
use pixelflow_core::batch::Batch;
use pixelflow_core::pipe::Surface;
use std::sync::{Arc, RwLock};
use std::collections::HashMap;
use crate::baked::{BakedExt, BakedMask, BakedFactory};

#[derive(Clone, Debug)]
pub struct ParsedGlyph {
    pub segments: Vec<Segment>,
}

#[derive(Clone)]
pub struct GlyphRenderer {
    pub data: Arc<ParsedGlyph>,
    pub scale: f32,
}

impl Surface<u8> for GlyphRenderer {
    fn eval(&self, x: Batch<u32>, y: Batch<u32>) -> Batch<u8> {
        let mut builder = EvalBuilder::new(x, y, self.scale);
        for seg in &self.data.segments {
            builder.process_segment(seg);
        }
        builder.result()
    }
}

pub type GlyphFactory<'a> = Arc<dyn Fn() -> GlyphRenderer + Send + Sync + 'a>;
pub type GlyphSurface<'a> = Lazy<GlyphFactory<'a>, GlyphRenderer>;

fn make_lazy<'a>(font: Font<'a>, c: char, scale: f32) -> GlyphSurface<'a> {
    let factory: GlyphFactory<'a> = Arc::new(move || {
        let glyph_id = font.glyph_index(c).unwrap_or(ttf_parser::GlyphId(0));
        let mut segments = font.outline_segments(glyph_id, 1.0);

        if let Some(bbox) = font.glyph_bounding_box(glyph_id) {
             let min_x = bbox.x_min as f32;
             let min_y = bbox.y_min as f32;
             for seg in &mut segments {
                 seg.translate(-min_x, -min_y);
             }
        }

        GlyphRenderer {
            data: Arc::new(ParsedGlyph { segments }),
            scale,
        }
    });

    Lazy::new(factory)
}

pub fn glyph<'a>(font: &'a Font<'a>, c: char) -> GlyphSurface<'a> {
    make_lazy(font.clone(), c, 1.0)
}

pub fn glyphs<'a>(font: &'a Font<'a>) -> impl Fn(char) -> GlyphSurface<'a> + 'a {
    let font = font.clone();
    move |c| make_lazy(font.clone(), c, 1.0)
}

pub fn glyphs_scaled<'a>(font: &'a Font<'a>, size_px: f32) -> impl Fn(char) -> GlyphSurface<'a> + 'a {
    let s = size_px / font.units_per_em() as f32;
    let font = font.clone();
    move |c| make_lazy(font.clone(), c, s)
}

pub fn glyphs_cached<'a>(font: &'a Font<'a>, width: u32, height: u32) -> impl Fn(char) -> Lazy<BakedFactory<'a>, BakedMask> + 'a {
    let font = font.clone();
    move |c| make_lazy(font.clone(), c, 1.0).baked(width, height)
}

#[derive(Clone)]
pub struct GlyphCache {
    cache: Arc<RwLock<HashMap<char, Lazy<BakedFactory<'static>, BakedMask>>>>,
    font: Font<'static>,
    width: u32,
    height: u32,
    scale: f32,
}

impl GlyphCache {
    pub fn new(font: Font<'static>, width: u32, height: u32, scale: f32) -> Self {
        Self {
            cache: Arc::new(RwLock::new(HashMap::new())),
            font,
            width,
            height,
            scale,
        }
    }

    pub fn get(&self, c: char) -> Lazy<BakedFactory<'static>, BakedMask> {
        if let Some(lazy) = self.cache.read().unwrap().get(&c) {
            return lazy.clone();
        }

        let mut map = self.cache.write().unwrap();
        if let Some(lazy) = map.get(&c) {
            return lazy.clone();
        }

        let lazy = make_lazy(self.font.clone(), c, self.scale).baked(self.width, self.height);

        map.insert(c, lazy.clone());
        lazy
    }
}

struct EvalBuilder {
    xs: [f32; 4],
    ys: [f32; 4],
    winding: [i32; 4],
    min_dist: [f32; 4],
    scale: f32,
}

impl EvalBuilder {
    fn new(x: Batch<u32>, y: Batch<u32>, scale: f32) -> Self {
        let xa = x.to_array_usize();
        let ya = y.to_array_usize();

        let mut xs = [0.0; 4];
        let mut ys = [0.0; 4];
        let inv_scale = if scale.abs() > 1e-6 { 1.0 / scale } else { 1.0 };

        for i in 0..4 {
            xs[i] = (xa[i] as f32 + 0.5) * inv_scale;
            ys[i] = (ya[i] as f32 + 0.5) * inv_scale;
        }

        Self {
            xs,
            ys,
            winding: [0; 4],
            min_dist: [1e9; 4],
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
            let d = self.min_dist[i] * self.scale;
            let signed_dist = if inside { -d.abs() } else { d.abs() };
            let alpha = (0.5 - signed_dist).clamp(0.0, 1.0);
            res[i] = (alpha * 255.0) as u32;
        }
        Batch::new(res[0], res[1], res[2], res[3]).cast()
    }
}
