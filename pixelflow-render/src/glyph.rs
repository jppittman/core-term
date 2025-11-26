//! Glyph decompression and rendering from pre-baked ROM.
//!
//! This module provides stateless functions for:
//! - Unpacking 4-bit compressed glyph data
//! - Downsampling glyphs to target size
//! - Applying bold/italic synthesis
//! - Converting to RGBA format

// Include the generated baked font data
#[cfg(feature = "fonts")]
include!(concat!(env!("OUT_DIR"), "/baked_font.rs"));

/// Unpack 4-bit packed grayscale data to 8-bit.
///
/// Each byte contains two pixels: high nibble is first pixel, low nibble is second.
/// Unpacking formula: `nibble * 17` expands 0-15 range to 0-255.
pub fn unpack_4bit(packed: &[u8], width: usize, height: usize) -> Vec<u8> {
    let pixel_count = width * height;
    let mut result = Vec::with_capacity(pixel_count);

    for byte in packed {
        // High nibble (first pixel)
        result.push(((byte >> 4) & 0x0F) * 17);
        // Low nibble (second pixel)
        result.push((byte & 0x0F) * 17);

        if result.len() >= pixel_count {
            break;
        }
    }

    result.truncate(pixel_count);
    result
}

/// Sample a grayscale image using bilinear interpolation.
fn sample_bilinear(src: &[u8], src_w: usize, src_h: usize, x: f32, y: f32) -> u8 {
    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let x1 = (x0 + 1).min(src_w - 1);
    let y1 = (y0 + 1).min(src_h - 1);

    let dx = x - x0 as f32;
    let dy = y - y0 as f32;

    // Clamp coordinates to prevent out of bounds
    let x0 = x0.min(src_w - 1);
    let y0 = y0.min(src_h - 1);

    let p00 = src[y0 * src_w + x0] as f32;
    let p10 = src[y0 * src_w + x1] as f32;
    let p01 = src[y1 * src_w + x0] as f32;
    let p11 = src[y1 * src_w + x1] as f32;

    let top = p00 * (1.0 - dx) + p10 * dx;
    let bottom = p01 * (1.0 - dx) + p11 * dx;
    let result = top * (1.0 - dy) + bottom * dy;

    result.round() as u8
}

/// Downsample a grayscale image using bilinear interpolation.
pub fn downsample(
    src: &[u8],
    src_w: usize,
    src_h: usize,
    dst_w: usize,
    dst_h: usize,
) -> Vec<u8> {
    let mut dst = vec![0u8; dst_w * dst_h];

    let x_ratio = src_w as f32 / dst_w as f32;
    let y_ratio = src_h as f32 / dst_h as f32;

    for y in 0..dst_h {
        for x in 0..dst_w {
            let src_x = (x as f32 + 0.5) * x_ratio - 0.5;
            let src_y = (y as f32 + 0.5) * y_ratio - 0.5;

            // Clamp to valid range
            let src_x = src_x.max(0.0).min((src_w - 1) as f32);
            let src_y = src_y.max(0.0).min((src_h - 1) as f32);

            dst[y * dst_w + x] = sample_bilinear(src, src_w, src_h, src_x, src_y);
        }
    }

    dst
}

/// Apply bold synthesis by double-rendering with 1px offset.
pub fn apply_bold(src: &[u8], width: usize, height: usize) -> Vec<u8> {
    let mut dst = vec![0u8; width * height];

    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            let original = src[idx] as u16;

            // Also blend with pixel to the right
            let right = if x + 1 < width {
                src[y * width + (x + 1)] as u16
            } else {
                0
            };

            // Take maximum (simulates darkening/thickening)
            dst[idx] = original.max(right).min(255) as u8;
        }
    }

    dst
}

/// Apply italic synthesis using shear transformation.
pub fn apply_italic(src: &[u8], width: usize, height: usize) -> Vec<u8> {
    let mut dst = vec![0u8; width * height];
    let shear_factor = 0.2; // ~11 degree slant

    for y in 0..height {
        let offset = ((height - y) as f32 * shear_factor).round() as isize;

        for x in 0..width {
            let src_x = (x as isize - offset).max(0).min(width as isize - 1) as usize;
            dst[y * width + x] = src[y * width + src_x];
        }
    }

    dst
}

/// Glyph rendering result with full metrics.
pub struct RenderedGlyph {
    pub data: Vec<u8>,
    pub width: usize,
    pub height: usize,
    pub bearing_x: i32,
    pub bearing_y: i32,
}

/// Decompress and render a glyph from ROM, scaled to target cell size.
///
/// Returns glyph with proper metrics for baseline alignment.
#[cfg(feature = "fonts")]
pub fn render_glyph_natural(ch: char, target_cell_height: usize, bold: bool, italic: bool) -> RenderedGlyph {
    use pixelflow_core::{TensorView, TensorViewMut};
    use crate::shader::{self, FontWeight, GlyphParams, GlyphStyle, Projection};

    // Binary search in GLYPH_METADATA
    let glyph_idx = match GLYPH_METADATA.binary_search_by_key(&ch, |meta| meta.c) {
        Ok(idx) => idx,
        Err(_) => {
            // Glyph not found, return empty
            return RenderedGlyph {
                data: vec![0u8; 1],
                width: 1,
                height: 1,
                bearing_x: 0,
                bearing_y: 0,
            };
        }
    };

    let meta = &GLYPH_METADATA[glyph_idx];

    // Calculate output dimensions
    let scale = target_cell_height as f32 / 24.0; // 24pt is our base size
    let width = (meta.width as f32 * scale).max(1.0) as usize;
    let height = (meta.height as f32 * scale).max(1.0) as usize;
    let bearing_x = (meta.bearing_x as f32 * scale) as i32;
    let bearing_y = (meta.bearing_y as f32 * scale) as i32;

    // Build shader parameters
    let params = GlyphParams {
        style: GlyphStyle {
            fg: 0xFF_FF_FF_FF,  // White (will extract alpha channel later)
            bg: 0x00_00_00_00,  // Transparent
            weight: if bold { FontWeight::Bold } else { FontWeight::Normal },
        },
        x_proj: Projection::scale(meta.width, width),
        y_proj: Projection::scale(meta.height, height),
    };

    // Allocate output buffer (u32 for shader pipeline)
    let mut output_u32 = vec![0u32; width * height];
    let mut dst_view = TensorViewMut::new(&mut output_u32, width, height, width);

    // Get packed atlas data (stays 4-bit packed!)
    let packed_len = (meta.width * meta.height + 1) / 2;
    let packed = &GLYPH_DATA[meta.offset..meta.offset + packed_len];
    let atlas_view = TensorView::new(packed, meta.width, meta.height, (meta.width + 1) / 2);

    // *** THE SHADER: One pass, zero intermediate buffers ***
    // Fuses: 4-bit unpack + bilinear resize + bold synthesis + alpha blend
    shader::render_glyph(&mut dst_view, &atlas_view, params);

    // Extract grayscale from R channel (after alpha blending, all RGB channels contain glyph intensity)
    let mut grayscale: Vec<u8> = output_u32
        .iter()
        .map(|&px| (px & 0xFF) as u8)
        .collect();

    // Apply italic if needed (TODO: could be moved into shader too)
    if italic {
        grayscale = apply_italic(&grayscale, width, height);
    }

    RenderedGlyph {
        data: grayscale,
        width,
        height,
        bearing_x,
        bearing_y,
    }
}

/// Decompress and render a glyph from ROM (legacy, forces exact size).
///
/// Returns grayscale alpha mask (8-bit per pixel).
#[cfg(feature = "fonts")]
pub fn render_glyph(ch: char, _width: usize, height: usize, bold: bool, italic: bool) -> Vec<u8> {
    render_glyph_natural(ch, height, bold, italic).data
}

/// Stub for when fonts is not enabled.
#[cfg(not(feature = "fonts"))]
pub fn render_glyph(_ch: char, width: usize, height: usize, _bold: bool, _italic: bool) -> Vec<u8> {
    vec![0u8; width * height]
}
