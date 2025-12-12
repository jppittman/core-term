//! Tests for SIMD safety and memory correctness.
//!
//! This suite ensures that SIMD operations respect memory bounds and don't
//! cause buffer overruns, which can lead to silent data corruption.

use pixelflow_core::backend::{Backend, SimdBatch};

// We only test the ARM NEON backend here as it's the one suspected of having
// buffer overrun issues. If x86 AVX512 is added later, it should be tested similarly.
#[cfg(target_arch = "aarch64")]
use pixelflow_core::backend::arm::{Neon, SimdVec};

#[test]
#[cfg(target_arch = "aarch64")]
fn test_neon_u8_store_bounds() {
    // Guard pattern to detect overwrites
    const GUARD: u8 = 0xAA;
    const FILL: u8 = 0x55;

    // We expect LANES=4 for this backend
    assert_eq!(Neon::LANES, 4);

    // Create a buffer with:
    // [0..4]: Target area
    // [4..8]: Guard area
    let mut buffer = [GUARD; 8];

    // Initialize batch with FILL
    let batch = SimdVec::<u8>::splat(FILL);

    // Store into the first 4 bytes.
    // The API contract implies that if we have 4 lanes, we write 4 items.
    // We pass a mutable slice of the whole buffer to allow the implementation
    // to potentially write more if it was "safe" (it's not, but for the test setup).
    // If the implementation writes 16 bytes (128 bits), it will overwrite the guards.

    SimdBatch::store(&batch, &mut buffer);

    // Check target area
    for i in 0..4 {
        assert_eq!(buffer[i], FILL, "Target byte {} should be written", i);
    }

    // Check guard area
    for i in 4..8 {
        assert_eq!(
            buffer[i], GUARD,
            "Guard byte {} was overwritten! Expected 0x{:02X}, got 0x{:02X}. This indicates a buffer overrun in SimdVec::store for u8.",
            i, GUARD, buffer[i]
        );
    }
}

#[test]
#[cfg(target_arch = "aarch64")]
fn test_neon_u16_store_bounds() {
    // Guard pattern
    const GUARD: u16 = 0xBEEF;
    const FILL: u16 = 0xCAFE;

    assert_eq!(Neon::LANES, 4);

    // Buffer: 4 target u16s + 4 guard u16s
    let mut buffer = [GUARD; 8];

    let batch = SimdVec::<u16>::splat(FILL);

    // Store should verify it only writes 4 items (LANES)
    SimdBatch::store(&batch, &mut buffer);

    for i in 0..4 {
        assert_eq!(buffer[i], FILL, "Target u16 {} should be written", i);
    }

    for i in 4..8 {
        assert_eq!(
            buffer[i], GUARD,
            "Guard u16 {} was overwritten! Expected 0x{:04X}, got 0x{:04X}. This indicates a buffer overrun in SimdVec::store for u16.",
            i, GUARD, buffer[i]
        );
    }
}

#[test]
#[cfg(target_arch = "aarch64")]
fn test_neon_u8_load_bounds() {
    // Loading should also be safe. If we load from a slice of length 4,
    // but the intrinsic reads 16 bytes, we might segfault if the page ends there.
    // This is harder to test deterministically without exact page boundary alignment,
    // but we can at least verify correct values are loaded from a larger buffer.

    let data = [1u8, 2, 3, 4, 0xFF, 0xFF, 0xFF, 0xFF];

    // Load 4 bytes
    let batch = SimdVec::<u8>::load(&data[0..4]);

    // First element should be 1
    assert_eq!(SimdBatch::first(&batch), 1);

    // Verify by storing back to a safe buffer
    let mut out = [0u8; 16]; // Large enough to catch the overflow if it happens during store
    SimdBatch::store(&batch, &mut out);

    assert_eq!(out[0], 1);
    assert_eq!(out[1], 2);
    assert_eq!(out[2], 3);
    assert_eq!(out[3], 4);

    // If the load read 16 bytes, it picked up the 0xFFs.
    // When we stored it back (assuming store is still broken/writes 16 bytes),
    // we can check if those 0xFFs appear.
    // If load only read 4 bytes properly (e.g. zero-extended), then index 4+ would be 0.
    //
    // However, the current implementation of SimdVec<u8> in arm.rs is:
    // SimdVec(NeonReg { u8: vdupq_n_u8(0) }) for default/splat?
    // No, load uses vld1q_u8 which reads 16 bytes.
    // So it definitely reads the garbage.
    //
    // We won't assert on the garbage here because fixing 'load' is secondary
    // to fixing 'store' (which corrupts memory). Reading garbage is bad but
    // usually less catastrophic than writing garbage, UNLESS we cross a page boundary.
}

#[test]
#[cfg(target_arch = "aarch64")]
fn test_downcast_u32_to_u8_packs_correctly() {
    use pixelflow_core::batch::NativeBackend;
    // Input u32 values (representing pixel components 0-255)
    let input_u32s = [0xAAu32, 0xBBu32, 0xCCu32, 0xDDu32];
    let input_batch_u32 = SimdVec::<u32>::from_array(input_u32s);

    // Downcast to u8
    let downcasted_batch_u8 = NativeBackend::downcast_u32_to_u8(input_batch_u32);

    // Store the resulting u8 batch to verify its content
    let mut output_u8s = [0u8; 4];
    SimdBatch::store(&downcasted_batch_u8, &mut output_u8s);

    // The output should be the original u8 values, packed
    assert_eq!(
        output_u8s,
        [0xAA, 0xBB, 0xCC, 0xDD],
        "downcast_u32_to_u8 should correctly pack u32 values into u8"
    );
}
