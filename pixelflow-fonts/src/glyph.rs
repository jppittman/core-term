use crate::Font;
use crate::curves::Segment;
use pixelflow_core::batch::Batch;
use pixelflow_core::pipe::Surface;
use pixelflow_core::ops::{Offset, Scale};
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct ParsedGlyph {
    pub segments: Vec<Segment>,
}

#[derive(Clone)]
pub struct GlyphRenderer {
    pub data: Arc<ParsedGlyph>,
    pub scale: f32, // For AA metric
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

// Raw glyph (no offset, segments in font units)
fn glyph_raw(font: &Font, c: char, scale: f32) -> GlyphRenderer {
    let glyph_id = font.glyph_index(c).unwrap_or(ttf_parser::GlyphId(0));
    let segments = font.outline_segments(glyph_id, 1.0);
    GlyphRenderer {
        data: Arc::new(ParsedGlyph { segments }),
        scale,
    }
}

// Glyph with offset (normalized to 0,0)
pub fn glyph<'a>(font: &'a Font<'a>, c: char) -> Offset<GlyphRenderer> {
    let raw = glyph_raw(font, c, 1.0);

    let glyph_id = font.glyph_index(c).unwrap_or(ttf_parser::GlyphId(0));
    let bbox = font.glyph_bounding_box(glyph_id).unwrap_or(ttf_parser::Rect{x_min:0, y_min:0, x_max:0, y_max:0});

    Offset {
        source: raw,
        dx: bbox.x_min as i32,
        dy: bbox.y_min as i32,
    }
}

pub fn glyphs<'a>(font: &'a Font<'a>) -> impl Fn(char) -> Offset<GlyphRenderer> + 'a {
    let font = font.clone();
    move |c| glyph(&font, c)
}

pub fn glyphs_scaled<'a>(font: &'a Font<'a>, size_px: f32) -> impl Fn(char) -> Scale<Offset<GlyphRenderer>> + 'a {
    let s = size_px / font.units_per_em() as f32;
    let font = font.clone();
    move |c| {
        let raw = glyph_raw(&font, c, s);
        let glyph_id = font.glyph_index(c).unwrap_or(ttf_parser::GlyphId(0));
        let bbox = font.glyph_bounding_box(glyph_id).unwrap_or(ttf_parser::Rect{x_min:0, y_min:0, x_max:0, y_max:0});

        let offset = Offset { source: raw, dx: bbox.x_min as i32, dy: bbox.y_min as i32 };
        Scale::new(offset, s as f64)
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

        for i in 0..4 {
            xs[i] = xa[i] as f32 + 0.5;
            ys[i] = ya[i] as f32 + 0.5;
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
