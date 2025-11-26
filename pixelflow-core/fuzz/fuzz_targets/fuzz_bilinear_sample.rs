#![no_main]

use libfuzzer_sys::fuzz_target;
use pixelflow_core::{Batch, TensorView};

fuzz_target!(|data: &[u8]| {
    if data.len() < 12 {
        return;
    }

    // Fuzz texture dimensions (clamped to reasonable range)
    let width = (data[0] as usize % 128).max(1);
    let height = (data[1] as usize % 128).max(1);

    // Fuzz coordinates (u, v in fixed-point 16.16)
    let u_fp = u32::from_le_bytes([data[2], data[3], data[4], data[5]]);
    let v_fp = u32::from_le_bytes([data[6], data[7], data[8], data[9]]);

    // Create a simple test texture (4-bit packed)
    let packed_len = (width * height + 1) / 2;
    let mut texture = vec![0u8; packed_len];

    // Fill with fuzz data if available
    let fill_len = (data.len() - 10).min(packed_len);
    texture[..fill_len].copy_from_slice(&data[10..10 + fill_len]);

    // Create tensor view with proper stride for 4-bit data
    let stride = (width + 1) / 2;
    let view = TensorView::new(&texture, width, height, stride);

    // Fuzz sample_4bit_bilinear - should never panic due to OOB
    let u_batch = Batch::splat(u_fp);
    let v_batch = Batch::splat(v_fp);

    let result = unsafe { view.sample_4bit_bilinear(u_batch, v_batch) };

    // Verify result is in valid range (0-255 for unpacked 4-bit)
    let result_arr = result.to_array_usize();
    for &val in &result_arr {
        assert!(val <= 255, "sample_4bit_bilinear returned value > 255: {}", val);
    }
});
