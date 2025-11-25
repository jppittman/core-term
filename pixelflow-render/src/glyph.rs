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
    use crate::simd_resize;

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

    // Calculate packed data length
    let src_pixel_count = meta.width * meta.height;
    let packed_len = (src_pixel_count + 1) / 2;

    // Extract packed data
    let packed_data = &GLYPH_DATA[meta.offset..meta.offset + packed_len];

    // Unpack 4-bit to 8-bit grayscale
    let mut grayscale = unpack_4bit(packed_data, meta.width, meta.height);
    let mut width = meta.width;
    let mut height = meta.height;
    let mut bearing_x = meta.bearing_x;
    let mut bearing_y = meta.bearing_y;

    // Scale glyph to target cell height
    // Font was baked at 24pt, typical cell height is ~18px at 24pt
    // We scale everything proportionally
    if height > 0 {
        let scale = target_cell_height as f32 / 24.0; // 24pt is our base size

        if scale != 1.0 {
            let new_width = (width as f32 * scale).max(1.0) as usize;
            let new_height = (height as f32 * scale).max(1.0) as usize;

            grayscale = simd_resize::resize_bilinear(&grayscale, width, height, new_width, new_height);

            bearing_x = (bearing_x as f32 * scale) as i32;
            bearing_y = (bearing_y as f32 * scale) as i32;
            width = new_width;
            height = new_height;
        }
    }

    // Apply synthesis
    if bold {
        grayscale = apply_bold(&grayscale, width, height);
    }

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
