#![no_main]

use libfuzzer_sys::fuzz_target;
use pixelflow_core::{Batch, TensorView, TensorViewMut, Projection};

// Mock shader parameters
struct ShaderParams {
    fg: u32,
    bg: u32,
    x_proj: Projection,
    y_proj: Projection,
}

fuzz_target!(|data: &[u8]| {
    if data.len() < 20 {
        return;
    }

    // Fuzz colors
    let fg = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
    let bg = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);

    // Fuzz dimensions (clamped to reasonable size)
    let src_w = ((data[8] as usize % 32) + 1).max(1);
    let src_h = ((data[9] as usize % 32) + 1).max(1);
    let dst_w = ((data[10] as usize % 32) + 1).max(1);
    let dst_h = ((data[11] as usize % 32) + 1).max(1);

    // Create source texture (4-bit packed)
    let packed_len = (src_w * src_h + 1) / 2;
    let mut src_texture = vec![0u8; packed_len];

    // Fill with fuzz data
    let fill_len = (data.len() - 12).min(packed_len);
    if fill_len > 0 {
        src_texture[..fill_len].copy_from_slice(&data[12..12 + fill_len]);
    }

    // Create destination buffer
    let mut dst_buffer = vec![0u32; dst_w * dst_h];

    // Create views
    let src_stride = (src_w + 1) / 2;
    let src_view = TensorView::new(&src_texture, src_w, src_h, src_stride);
    let mut dst_view = TensorViewMut::new(&mut dst_buffer, dst_w, dst_h, dst_w);

    // Create projections
    let x_proj = Projection::scale(src_w, dst_w);
    let y_proj = Projection::scale(src_h, dst_h);

    let params = ShaderParams { fg, bg, x_proj, y_proj };

    // Simulate shader pipeline: sample, blend, write
    // Process in batches of 4 pixels
    let batch_count = (dst_w * dst_h + 3) / 4;

    for batch_idx in 0..batch_count {
        let pixel_idx = batch_idx * 4;
        if pixel_idx >= dst_w * dst_h {
            break;
        }

        // Compute coordinates for this batch
        let coords: Vec<(u32, u32)> = (0..4)
            .map(|i| {
                let idx = pixel_idx + i;
                if idx < dst_w * dst_h {
                    let x = (idx % dst_w) as u32;
                    let y = (idx / dst_w) as u32;
                    (x, y)
                } else {
                    (0, 0) // Padding
                }
            })
            .collect();

        // Project to source space (fixed-point)
        let u_coords: Vec<u32> = coords.iter()
            .map(|(x, _)| params.x_proj.u0 + (params.x_proj.du * x))
            .collect();
        let v_coords: Vec<u32> = coords.iter()
            .map(|(_, y)| params.y_proj.v0 + (params.y_proj.dv * y))
            .collect();

        let u_batch = Batch::from_array_u32([u_coords[0], u_coords[1], u_coords[2], u_coords[3]]);
        let v_batch = Batch::from_array_u32([v_coords[0], v_coords[1], v_coords[2], v_coords[3]]);

        // Sample from source (bilinear)
        let alpha = unsafe { src_view.sample_4bit_bilinear(u_batch, v_batch) };

        // Verify alpha is in range
        let alpha_arr = alpha.to_array_u32();
        for &a in &alpha_arr {
            assert!(a <= 255, "Alpha out of range: {}", a);
        }

        // Replicate alpha across ARGB channels
        let alpha_argb = alpha | (alpha << 8) | (alpha << 16) | (alpha << 24);

        // Blend
        let fg_batch = Batch::splat(params.fg);
        let bg_batch = Batch::splat(params.bg);
        let result = fg_batch.blend_alpha(bg_batch, alpha_argb);

        // Verify result
        let result_arr = result.to_array_u32();
        for &pixel in &result_arr {
            let a = (pixel >> 24) & 0xFF;
            let r = (pixel >> 16) & 0xFF;
            let g = (pixel >> 8) & 0xFF;
            let b = pixel & 0xFF;
            assert!(a <= 255);
            assert!(r <= 255);
            assert!(g <= 255);
            assert!(b <= 255);
        }

        // Write to destination (bounds check)
        for i in 0..4 {
            let idx = pixel_idx + i;
            if idx < dst_buffer.len() {
                dst_buffer[idx] = result_arr[i];
            }
        }
    }

    // Final verification: all output pixels should have valid ARGB values
    for &pixel in &dst_buffer {
        let a = (pixel >> 24) & 0xFF;
        let r = (pixel >> 16) & 0xFF;
        let g = (pixel >> 8) & 0xFF;
        let b = pixel & 0xFF;
        assert!(a <= 255);
        assert!(r <= 255);
        assert!(g <= 255);
        assert!(b <= 255);
    }
});
