//! Example demonstrating the type-driven SIMD API
//!
//! This shows how the same operators (`+`, `*`, `>>`) automatically select
//! the correct SIMD instructions based on the type parameter.

use pixelflow_core::Batch;

fn main() {
    println!("Type-Driven SIMD API Example\n");

    // Example 1: 32-bit addition
    println!("=== Example 1: u32 addition (uses paddd/vaddq_u32) ===");
    let a = Batch::<u32>::splat(100);
    let b = Batch::<u32>::splat(50);
    let c = a + b; // Uses _mm_add_epi32 on x86, vaddq_u32 on ARM

    let mut result = [0u32; 4];
    unsafe { c.store(result.as_mut_ptr()) };
    println!("100 + 50 = {:?}", result);

    // Example 2: 16-bit multiplication (the key operation for bilinear interpolation)
    println!("\n=== Example 2: u16 multiplication (uses pmullw/vmulq_u16) ===");
    let pixels = Batch::<u32>::new(0x01020304, 0x05060708, 0x090A0B0C, 0x0D0E0F10);
    let weights = Batch::<u32>::splat(256); // Fixed-point 1.0

    // Cast to u16 to exploit 16-bit SIMD multiply
    let pixels_16 = pixels.cast::<u16>();
    let weights_16 = weights.cast::<u16>();

    // This uses _mm_mullo_epi16 on x86, vmulq_u16 on ARM
    let scaled = pixels_16 * weights_16;

    println!("Pixels (as u16):  [view of memory as 8×u16]");
    println!("Weights (as u16): [256, 256, 256, 256, ...]");
    println!("Scaled product:   [computed using pmullw/vmulq_u16]");

    // Example 3: The bilinear interpolation pattern
    println!("\n=== Example 3: Bilinear interpolation pattern ===");
    let p00 = Batch::<u32>::new(0xFF000000, 0x00FF0000, 0x0000FF00, 0x000000FF);
    let p10 = Batch::<u32>::new(0x00000000, 0x00000000, 0x00000000, 0x00000000);

    let inv_dx = Batch::<u32>::splat(192); // 75% weight (256 - 64)
    let dx = Batch::<u32>::splat(64); // 25% weight

    // Old C-style API (what we had before):
    // let w0 = p00.mullo_u16(inv_dx).add_u16(p10.mullo_u16(dx)).shift_right_u16(8);

    // New Rust-style API (using standard operators):
    let w0 =
        ((p00.cast::<u16>() * inv_dx.cast::<u16>()) + (p10.cast::<u16>() * dx.cast::<u16>())) >> 8;

    let mut result_pixels = [0u32; 4];
    unsafe { w0.cast::<u32>().store(result_pixels.as_mut_ptr()) };
    println!("Interpolated: {:08X?}", result_pixels);

    println!("\n✓ Notice how the same operators (+, *, >>) automatically select:");
    println!("  - Batch<u32> + Batch<u32> → paddd (32-bit add)");
    println!("  - Batch<u16> * Batch<u16> → pmullw (16-bit multiply)");
    println!("  - Batch<u16> >> 8         → psrlw (16-bit shift)");
    println!("\nThe type IS the instruction selector!");
}
