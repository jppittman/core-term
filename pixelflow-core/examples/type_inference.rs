//! Demonstration of beautiful type inference
//!
//! Shows how removing constraints from the struct definition
//! lets Rust infer almost everything automatically.

use pixelflow_core::Batch;

fn main() {
    println!("Type Inference Demo\n");

    // Example 1: Rust infers everything from the first splat
    println!("=== Example 1: Zero annotations ===");
    let a = Batch::splat(100u32); // Explicitly u32 here
    let b = Batch::splat(50);      // Infers u32 from 'a'
    let c = a + b;                 // Infers Batch<u32>

    let mut result = [0u32; 4];
    unsafe { c.store(result.as_mut_ptr()) };
    println!("100 + 50 = {:?}", result);

    // Example 2: Bilinear interpolation with minimal annotations
    println!("\n=== Example 2: Bilinear interpolation (terse!) ===");
    let p00 = Batch::splat(0xFF000000u32);
    let p10 = Batch::splat(0x00000000u32);
    let dx = Batch::splat(64u32);       // 25% weight
    let inv_dx = Batch::splat(256u32) - dx; // 75% weight

    // Look how clean this is! No type annotations needed.
    // Rust infers that cast() returns Batch<u16> from context.
    let w0 = (p00.cast::<u16>() * inv_dx.cast::<u16>() +
              p10.cast::<u16>() * dx.cast::<u16>()) >> 8;

    let mut result = [0u32; 4];
    unsafe { w0.cast::<u32>().store(result.as_mut_ptr()) };
    println!("Interpolated: {:08X?}", result);

    // Example 3: gather() also benefits from inference
    println!("\n=== Example 3: Type inference in gather ===");
    let palette = [0xFFu8, 0xAAu8, 0x55u8, 0x00u8];

    // No turbofish! Rust infers T=u8 from the slice type
    let colors = unsafe { Batch::gather(&palette, [0, 1, 2, 3]) };

    let mut result = [0u32; 4];
    unsafe { colors.store(result.as_mut_ptr()) };
    println!("Gathered: {:02X?}", result);

    println!("\n✓ Notice: Almost zero type annotations!");
    println!("  The only explicit types are:");
    println!("  1. Initial value literals (100u32)");
    println!("  2. cast::<U>() turbofish (unavoidable - changing type)");
}
