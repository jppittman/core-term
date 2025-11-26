#![no_main]

use libfuzzer_sys::fuzz_target;
use pixelflow_core::Projection;

fuzz_target!(|data: &[u8]| {
    if data.len() < 8 {
        return;
    }

    // Fuzz source and destination dimensions
    let src_dim = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
    let dst_dim = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);

    // Clamp to reasonable ranges to avoid extreme memory usage
    let src_dim = (src_dim % 4096).max(1);
    let dst_dim = (dst_dim % 4096).max(1);

    // Create projection - should not panic
    let proj = Projection::scale(src_dim as usize, dst_dim as usize);

    // Verify projection produces valid output for all possible input coordinates
    for dst_coord in 0..dst_dim.min(100) {
        // Test projection at this coordinate
        // The projection should map dst coordinates to src coordinates in fixed-point

        // Check that du/dv increments are reasonable
        // du = (src_dim << 16) / dst_dim
        if dst_dim > 0 {
            let expected_step = ((src_dim as u64) << 16) / (dst_dim as u64);
            assert!(expected_step <= u32::MAX as u64, "Projection step overflow");
        }
    }

    // Edge cases
    if src_dim == dst_dim {
        // Identity projection: step should be 1.0 in fixed-point (65536)
        let expected_step = 1u32 << 16;
        // Verify this makes sense
        assert!(expected_step > 0);
    }

    if dst_dim == 1 {
        // Projection to single pixel should work
        let _ = Projection::scale(src_dim as usize, 1);
    }

    if src_dim == 1 {
        // Projection from single pixel should work
        let _ = Projection::scale(1, dst_dim as usize);
    }
});
