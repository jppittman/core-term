//! Tests for numeric conversions and bitwise casting in the batch system.
//!
//! Verifies the behavior of `bitcast`, `to_f32`, and `to_u32` across different
//! backends to ensure consistent behavior for numeric vs bitwise operations.

use pixelflow_core::batch::Batch;
use pixelflow_core::backend::SimdBatch;

/// IEEE 754 floating point representation of 1.0.
const F32_ONE_BITS: u32 = 0x3F800000;

#[test]
fn test_bitcast_preserves_bits() {
    // Case 1: Bitcast u32 bits of 1.0 to f32 -> should be 1.0
    // This ensures that we can manipulate floats as bits (e.g. for masks) and cast back.
    let val_u32 = F32_ONE_BITS;
    let batch_u32 = Batch::<u32>::splat(val_u32);
    
    // Explicitly use bitcast - this must compile and run on all backends
    let batch_f32: Batch<f32> = batch_u32.bitcast();
    
    assert_eq!(batch_f32.first(), 1.0, "Bitcast of 0x3F800000 should result in 1.0");

    // Case 2: Bitcast 1u32 to f32 -> should be denormal (tiny), NOT 1.0
    // This verifies that bitcast is NOT performing a value conversion.
    let one_u32 = Batch::<u32>::splat(1);
    let one_as_float_bits: Batch<f32> = one_u32.bitcast();
    
    assert_ne!(one_as_float_bits.first(), 1.0, "Bitcast of 1u32 should not be 1.0");
    assert!(one_as_float_bits.first() < 1.0e-30, "Bitcast of 1u32 should be a denormalized float");
}

#[test]
fn test_numeric_conversion_preserves_value() {
    // Case 1: Convert 1u32 to f32 -> should be 1.0
    let batch_u32 = Batch::<u32>::splat(1);
    let batch_f32 = batch_u32.to_f32();
    
    assert_eq!(batch_f32.first(), 1.0, "Conversion of 1u32 should result in 1.0f32");
    
    // Case 2: Convert 2.5f32 to u32 -> should be 2 (truncation)
    let batch_float = Batch::<f32>::splat(2.5);
    let batch_int = batch_float.to_u32();
    
    assert_eq!(batch_int.first(), 2, "Conversion of 2.5f32 should truncate to 2u32");
}