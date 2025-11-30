use crate::Font;
use crate::surface::CurveSurface;
use crate::lazy::Lazy;
use pixelflow_core::ops::{Offset, Scale};
use alloc::sync::Arc;

extern crate alloc;

// Raw glyph (no offset, segments in font units)
fn glyph_raw(font: &Font, c: char, scale: f32) -> CurveSurface {
    let glyph_id = font.glyph_index(c).unwrap_or(ttf_parser::GlyphId(0));
    let segments = font.outline_segments(glyph_id, 1.0);
    CurveSurface {
        segments: Arc::from(segments),
        scale,
    }
}

// Glyph with offset (normalized to 0,0)
pub fn glyph<'a>(font: &'a Font<'a>, c: char) -> impl pixelflow_core::pipe::Surface<u8> + 'a {
    // We capture font and char, but defer parsing.
    // The Lazy surface expects a closure that returns a Surface.
    let font = font.clone();

    // We need to return something that implements Surface<u8>.
    // Lazy<T> implements Surface<P> if T: Surface<P>.
    // So we need Lazy to produce Offset<CurveSurface>.
    Lazy::new(move || {
        let raw = glyph_raw(&font, c, 1.0);
        let glyph_id = font.glyph_index(c).unwrap_or(ttf_parser::GlyphId(0));
        let bbox = font.glyph_bounding_box(glyph_id).unwrap_or(ttf_parser::Rect{x_min:0, y_min:0, x_max:0, y_max:0});

        Offset {
            source: raw,
            dx: bbox.x_min as i32,
            dy: bbox.y_min as i32,
        }
    })
}

// Helper: Box the factory closure to avoid unnamable types in return signature
pub fn glyphs<'a>(font: &'a Font<'a>) -> impl Fn(char) -> Lazy<Box<dyn Fn() -> Offset<CurveSurface> + Send + Sync + 'a>, Offset<CurveSurface>> + 'a {
    let font = font.clone();
    move |c| {
        let font = font.clone();
        Lazy::new(Box::new(move || {
            let raw = glyph_raw(&font, c, 1.0);
            let glyph_id = font.glyph_index(c).unwrap_or(ttf_parser::GlyphId(0));
            let bbox = font.glyph_bounding_box(glyph_id).unwrap_or(ttf_parser::Rect{x_min:0, y_min:0, x_max:0, y_max:0});

            Offset {
                source: raw,
                dx: bbox.x_min as i32,
                dy: bbox.y_min as i32,
            }
        }))
    }
}

pub fn glyphs_scaled<'a>(font: &'a Font<'a>, size_px: f32) -> impl Fn(char) -> Lazy<Box<dyn Fn() -> Scale<Offset<CurveSurface>> + Send + Sync + 'a>, Scale<Offset<CurveSurface>>> + 'a {
    let s = size_px / font.units_per_em() as f32;
    let font = font.clone();
    move |c| {
        let font = font.clone();
        Lazy::new(Box::new(move || {
            let raw = glyph_raw(&font, c, s);
            let glyph_id = font.glyph_index(c).unwrap_or(ttf_parser::GlyphId(0));
            let bbox = font.glyph_bounding_box(glyph_id).unwrap_or(ttf_parser::Rect{x_min:0, y_min:0, x_max:0, y_max:0});

            let offset = Offset { source: raw, dx: bbox.x_min as i32, dy: bbox.y_min as i32 };
            Scale::new(offset, s as f64)
        }))
    }
}
