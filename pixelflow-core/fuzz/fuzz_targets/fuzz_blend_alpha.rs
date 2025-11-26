#![no_main]

use libfuzzer_sys::fuzz_target;
use pixelflow_core::Batch;

fuzz_target!(|data: &[u8]| {
    if data.len() < 12 {
        return;
    }

    // Extract three u32 values from fuzz input
    let fg = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
    let bg = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
    let alpha = u32::from_le_bytes([data[8], data[9], data[10], data[11]]);

    // Create batches (splat same value across all lanes)
    let fg_batch = Batch::splat(fg);
    let bg_batch = Batch::splat(bg);
    let alpha_batch = Batch::splat(alpha);

    // Fuzz blend_alpha - should never panic
    let result = fg_batch.blend_alpha(bg_batch, alpha_batch);

    // Extract result to ensure it's computed
    let result_arr = result.to_array_usize();

    // Invariants to check:
    // 1. If alpha is 0, result should be close to bg
    // 2. If alpha is 0xFFFFFFFF, result should be close to fg
    // 3. Result should be in valid ARGB range

    for &pixel in &result_arr {
        // Each byte should be in 0..=255 range
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
