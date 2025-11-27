//! Assembly proof: Verify that type-driven API compiles to optimal code
//!
//! This example demonstrates that `Batch<u32> + Batch<u32>` compiles to
//! the exact same assembly as calling `_mm_add_epi32` directly.

use pixelflow_core::Batch;

/// Old style: Direct intrinsic call (what we're comparing against)
#[cfg(target_arch = "aarch64")]
#[inline(never)]
pub fn add_u32_direct(a: [u32; 4], b: [u32; 4]) -> [u32; 4] {
    use core::arch::aarch64::*;
    unsafe {
        let va = vld1q_u32(a.as_ptr());
        let vb = vld1q_u32(b.as_ptr());
        let result = vaddq_u32(va, vb);
        let mut out = [0u32; 4];
        vst1q_u32(out.as_mut_ptr(), result);
        out
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(never)]
pub fn add_u32_direct(a: [u32; 4], b: [u32; 4]) -> [u32; 4] {
    use core::arch::x86_64::*;
    unsafe {
        let va = _mm_loadu_si128(a.as_ptr() as *const __m128i);
        let vb = _mm_loadu_si128(b.as_ptr() as *const __m128i);
        let result = _mm_add_epi32(va, vb);
        let mut out = [0u32; 4];
        _mm_storeu_si128(out.as_mut_ptr() as *mut __m128i, result);
        out
    }
}

/// New style: Type-driven operator (should compile to identical code)
#[inline(never)]
pub fn add_u32_generic(a: [u32; 4], b: [u32; 4]) -> [u32; 4] {
    unsafe {
        let va = Batch::<u32>::load(a.as_ptr());
        let vb = Batch::<u32>::load(b.as_ptr());
        let result = va + vb; // <-- Just use standard `+` operator!
        let mut out = [0u32; 4];
        result.store(out.as_mut_ptr());
        out
    }
}

/// Multiply using 16-bit SIMD (the bilinear interpolation pattern)
#[inline(never)]
pub fn mul_u16_generic(a: [u32; 4], b: [u32; 4]) -> [u32; 4] {
    unsafe {
        let va = Batch::<u32>::load(a.as_ptr());
        let vb = Batch::<u32>::load(b.as_ptr());

        // Cast to u16 to use pmullw/vmulq_u16
        let result = va.cast::<u16>() * vb.cast::<u16>();

        let mut out = [0u32; 4];
        result.cast::<u32>().store(out.as_mut_ptr());
        out
    }
}

/// Full bilinear interpolation weight computation
#[inline(never)]
pub fn bilinear_weights(p00: [u32; 4], p10: [u32; 4], dx: u32) -> [u32; 4] {
    unsafe {
        let p00 = Batch::<u32>::load(p00.as_ptr());
        let p10 = Batch::<u32>::load(p10.as_ptr());
        let dx = Batch::<u32>::splat(dx);
        let inv_dx = Batch::<u32>::splat(256) - dx;

        // This is the money shot: Clean syntax that compiles to optimal SIMD
        let w0 = ((p00.cast::<u16>() * inv_dx.cast::<u16>())
            + (p10.cast::<u16>() * dx.cast::<u16>()))
            >> 8;

        let mut out = [0u32; 4];
        w0.cast::<u32>().store(out.as_mut_ptr());
        out
    }
}

fn main() {
    println!("Assembly Proof Example\n");
    println!("To verify zero-cost abstraction, compile with:");
    println!("  cargo build --release --example assembly_proof");
    println!("  cargo asm --release --example assembly_proof add_u32_direct");
    println!("  cargo asm --release --example assembly_proof add_u32_generic");
    println!("\nThey should produce IDENTICAL assembly!");

    // Run the functions to verify correctness
    let a = [100, 200, 300, 400];
    let b = [50, 60, 70, 80];

    let direct = add_u32_direct(a, b);
    let generic = add_u32_generic(a, b);

    println!("\nCorrectness check:");
    println!("Direct intrinsic:  {:?}", direct);
    println!("Generic operator:  {:?}", generic);
    println!("Match: {}", direct == generic);

    // Test 16-bit multiply
    let mul_result = mul_u16_generic([0x0001_0002, 0x0003_0004, 0, 0], [256, 256, 0, 0]);
    println!("\n16-bit multiply (256 * [1,2,3,4]):");
    println!("Result: {:08X?}", mul_result);

    // Test bilinear weights
    let weights = bilinear_weights(
        [0xFF000000, 0x00FF0000, 0x0000FF00, 0x000000FF],
        [0x00000000, 0x00000000, 0x00000000, 0x00000000],
        64, // 25% blend
    );
    println!("\nBilinear weights (75% p00 + 25% p10):");
    println!("Result: {:08X?}", weights);
}
