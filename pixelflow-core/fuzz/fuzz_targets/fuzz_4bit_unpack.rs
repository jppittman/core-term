#![no_main]

use libfuzzer_sys::fuzz_target;
use pixelflow_core::{Batch, TensorView};

fuzz_target!(|data: &[u8]| {
    if data.len() < 6 {
        return;
    }

    // Fuzz dimensions
    let width = (data[0] as usize % 64).max(1);
    let height = (data[1] as usize % 64).max(1);

    // Fuzz coordinates (must be in bounds)
    let x = u32::from_le_bytes([data[2], data[3], 0, 0]) % (width as u32);
    let y = u32::from_le_bytes([data[4], data[5], 0, 0]) % (height as u32);

    // Create packed texture
    let packed_len = (width * height + 1) / 2;
    let mut texture = vec![0u8; packed_len];

    // Fill with remaining fuzz data
    let fill_len = (data.len() - 6).min(packed_len);
    if fill_len > 0 {
        texture[..fill_len].copy_from_slice(&data[6..6 + fill_len]);
    }

    // Create tensor view
    let stride = (width + 1) / 2;
    let view = TensorView::new(&texture, width, height, stride);

    // Create batch of coordinates (all same value)
    let x_batch = Batch::splat(x);
    let y_batch = Batch::splat(y);

    // Fuzz gather_4bit - should extract correct nibble
    let result = unsafe { view.gather_4bit(x_batch, y_batch) };
    let result_arr = result.to_array_u32();

    // All results should be in 0..=15 range (4-bit values)
    for &val in &result_arr {
        assert!(val <= 15, "gather_4bit returned value > 15: {}", val);
    }

    // Verify consistency: same coordinates should give same result
    let result2 = unsafe { view.gather_4bit(x_batch, y_batch) };
    let result2_arr = result2.to_array_u32();
    assert_eq!(result_arr, result2_arr, "gather_4bit not deterministic");

    // Test expanded version (unpacks to 0-255 range)
    let expanded = unsafe { view.gather_4bit_expanded(x_batch, y_batch) };
    let expanded_arr = expanded.to_array_u32();

    for &val in &expanded_arr {
        assert!(val <= 255, "gather_4bit_expanded returned value > 255: {}", val);
        // Should be a multiple of 17 (since we expand 4-bit by *17)
        assert_eq!(val % 17, 0, "gather_4bit_expanded not properly expanded: {}", val);
    }

    // Verify expansion matches: expanded[i] = result[i] * 17
    for i in 0..4 {
        let expected = result_arr[i] * 17;
        assert_eq!(expanded_arr[i], expected, "Expansion mismatch at lane {}", i);
    }
});
