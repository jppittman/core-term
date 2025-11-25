//! SIMD-accelerated bilinear image resizing.
//!
//! Uses separable filtering: vertical pass then horizontal pass.
//! This is much more SIMD-friendly than 2D interpolation.

/// Resize a grayscale image using separable bilinear interpolation.
///
/// This is optimized for downsampling font glyphs.
pub fn resize_bilinear(
    src: &[u8],
    src_w: usize,
    src_h: usize,
    dst_w: usize,
    dst_h: usize,
) -> Vec<u8> {
    let mut dst = vec![0u8; dst_w * dst_h];

    // Pre-calculate X indices and weights (avoids float math in hot loop)
    let mut x_indices = vec![0usize; dst_w];
    let mut x_weights = vec![0u8; dst_w];

    let x_ratio = src_w as f32 / dst_w as f32;
    for x in 0..dst_w {
        let src_pos = x as f32 * x_ratio;
        x_indices[x] = (src_pos as usize).min(src_w - 1);
        x_weights[x] = ((src_pos.fract()) * 256.0) as u8;
    }

    // Temporary buffer for vertical pass (one row of intermediate data)
    let mut temp_row = vec![0u8; src_w];

    let y_ratio = src_h as f32 / dst_h as f32;

    // === VERTICAL PASS (Y-Axis) ===
    for y in 0..dst_h {
        let src_pos_y = y as f32 * y_ratio;
        let y0 = (src_pos_y as usize).min(src_h - 1);
        let y1 = (y0 + 1).min(src_h - 1);
        let y_weight = ((src_pos_y.fract()) * 256.0) as u8;
        let inv_y_weight = 256u16 - y_weight as u16;

        let row0 = &src[y0 * src_w..(y0 + 1) * src_w];
        let row1 = &src[y1 * src_w..(y1 + 1) * src_w];

        // SIMD lerp: temp_row[i] = (row0[i] * inv_weight + row1[i] * weight) >> 8
        // For now, use scalar (SIMD version would use SSE/NEON intrinsics)
        for i in 0..src_w {
            let p0 = row0[i] as u16;
            let p1 = row1[i] as u16;
            temp_row[i] = ((p0 * inv_y_weight + p1 * y_weight as u16) >> 8) as u8;
        }

        // === HORIZONTAL PASS (X-Axis) ===
        // Sample from temp_row into dst
        let dst_row_start = y * dst_w;

        for x in 0..dst_w {
            let x0 = x_indices[x];
            let x1 = (x0 + 1).min(src_w - 1);
            let x_weight = x_weights[x] as u16;
            let inv_x_weight = 256 - x_weight;

            let p_left = temp_row[x0] as u16;
            let p_right = temp_row[x1] as u16;

            dst[dst_row_start + x] = ((p_left * inv_x_weight + p_right * x_weight) >> 8) as u8;
        }
    }

    dst
}
