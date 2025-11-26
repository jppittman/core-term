#![no_main]

use libfuzzer_sys::fuzz_target;
use pixelflow_core::Batch;

fuzz_target!(|data: &[u8]| {
    if data.len() < 8 {
        return;
    }

    let a = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
    let b = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);

    let a_batch: Batch<u32> = Batch::splat(a);
    let b_batch: Batch<u32> = Batch::splat(b);

    // Test saturating_add
    let add_result = a_batch.saturating_add(b_batch);
    let add_arr: [u32; 4] = add_result.to_array();

    for &val in &add_arr {
        // Result should be >= both operands (or saturated at u32::MAX)
        // For saturating add: min(a + b, u32::MAX)
        let expected = a.saturating_add(b);
        assert_eq!(val, expected, "saturating_add({}, {}) = {}, expected {}", a, b, val, expected);
    }

    // Test saturating_sub
    let sub_result = a_batch.saturating_sub(b_batch);
    let sub_arr: [u32; 4] = sub_result.to_array();

    for &val in &sub_arr {
        // Result should be <= a and >= 0 (saturated at 0)
        // For saturating sub: max(a - b, 0)
        let expected = a.saturating_sub(b);
        assert_eq!(val, expected, "saturating_sub({}, {}) = {}, expected {}", a, b, val, expected);
    }

    // Edge cases
    if a == u32::MAX {
        let add_max = a_batch.saturating_add(Batch::splat(1));
        let add_max_arr = add_max.to_array_u32();
        assert_eq!(add_max_arr[0], u32::MAX, "MAX + 1 should saturate to MAX");
    }

    if a == 0 {
        let sub_zero = a_batch.saturating_sub(Batch::splat(1));
        let sub_zero_arr = sub_zero.to_array_u32();
        assert_eq!(sub_zero_arr[0], 0, "0 - 1 should saturate to 0");
    }
});
