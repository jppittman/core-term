//! # PixelFlow Core
//!
//! A zero-cost SIMD compute graph compiler for pixel operations.
//!
//! ## Design Philosophy
//!
//! PixelFlow is inspired by TensorFlow's compute graph model: you define **what**
//! computation you want (the graph), and the compiler fuses it into optimal machine
//! code for the target hardware (SSE/AVX on x86_64, NEON on ARM).
//!
//! Unlike traditional SIMD libraries, PixelFlow:
//! - Has **zero runtime overhead** (all abstractions inline away)
//! - Performs **compile-time graph fusion** via LLVM
//! - Provides a **declarative API** (operations, not loops)
//! - Supports **cross-platform** without conditional compilation in user code
//!
//! ## Example
//!
//! ```ignore
//! use pixelflow_core::Batch;
//!
//! // Define the computation graph
//! let fg = Batch::splat(0xFFFFFFFF);  // White foreground
//! let bg = Batch::splat(0xFF000000);  // Black background
//! let mask = unsafe { Batch::load(glyph_ptr) };
//!
//! // Compile-time fusion: this becomes 3 SSE instructions
//! let result = fg.select(bg, mask);
//! unsafe { result.store(dest_ptr) };
//! ```

#![no_std]
#![deny(unsafe_op_in_unsafe_fn)]

pub mod backends;

/// The core abstraction: A batch of pixels (typically 4) in SIMD registers.
///
/// This trait defines the "operation graph" available to the compiler.
/// All operations are marked `#[inline(always)]` to enable graph fusion.
///
/// # Safety
///
/// `load` and `store` operations require:
/// - Pointers must be valid for reads/writes
/// - No alignment requirements (we use unaligned load/store)
/// - Caller must ensure no data races
pub trait PixelBatch: Copy + Clone + Sized {
    // =========================================================================
    // Source Nodes (Inputs to the Compute Graph)
    // =========================================================================

    /// Create a batch where all pixels have the same value (broadcast).
    ///
    /// This is a "constant tensor" in TensorFlow terminology.
    #[must_use]
    fn splat(val: u32) -> Self;

    /// Load a batch of pixels from memory (input tensor).
    ///
    /// # Safety
    ///
    /// - `ptr` must be valid for reading 4 u32 values (16 bytes)
    /// - No alignment required
    #[must_use]
    unsafe fn load(ptr: *const u32) -> Self;

    // =========================================================================
    // Sink Nodes (Outputs from the Compute Graph)
    // =========================================================================

    /// Store a batch of pixels to memory (output tensor).
    ///
    /// # Safety
    ///
    /// - `ptr` must be valid for writing 4 u32 values (16 bytes)
    /// - No alignment required
    unsafe fn store(self, ptr: *mut u32);

    // =========================================================================
    // Operation Nodes (The Compute Graph)
    // =========================================================================

    /// Bitwise OR: `self | other`
    ///
    /// Useful for combining masks or setting bits.
    #[must_use]
    fn or(self, other: Self) -> Self;

    /// Bitwise AND: `self & other`
    ///
    /// Useful for masking or clearing bits.
    #[must_use]
    fn and(self, other: Self) -> Self;

    /// Bitwise NOT: `!self`
    ///
    /// Inverts all bits.
    #[must_use]
    fn not(self) -> Self;

    /// Conditional select: `(self & mask) | (other & !mask)`
    ///
    /// This is the core operation for alpha blending:
    /// - Where `mask` is set, use `self`
    /// - Where `mask` is clear, use `other`
    ///
    /// This single operation compiles to highly optimized code on all platforms.
    #[must_use]
    fn select(self, other: Self, mask: Self) -> Self;

    /// Alpha blend: Linear interpolation between `self` (fg) and `bg` based on `alpha`.
    ///
    /// Formula: `fg * alpha + bg * (1 - alpha)`
    ///
    /// This is the proper way to blend antialiased text glyphs.
    /// Each pixel is independently blended based on its alpha channel.
    ///
    /// # Arguments
    /// * `self` - Foreground color (RGBA pixels)
    /// * `bg` - Background color (RGBA pixels)
    /// * `alpha` - Alpha mask where alpha channel contains coverage (0-255)
    ///
    /// # Implementation Note
    /// Default implementation uses per-pixel scalar blending.
    /// Backends can override with SIMD-optimized versions using:
    /// 1. Expand u8 channels to u16 (prevent overflow)
    /// 2. Multiply fg by alpha, bg by (256 - alpha)
    /// 3. Add and shift right by 8 (divide by 256)
    /// 4. Pack back to u8
    #[must_use]
    fn blend_alpha(self, bg: Self, alpha: Self) -> Self {
        // Default scalar implementation
        // This processes each pixel independently
        // Backends can override with SIMD optimizations

        // Convert each u32 pixel to [u8; 4] for processing
        let fg_pixels = self.to_bytes();
        let bg_pixels = bg.to_bytes();
        let alpha_pixels = alpha.to_bytes();

        let mut result = [[0u8; 4]; 4];

        for i in 0..4 {
            let a = alpha_pixels[i][3] as u16; // Alpha channel
            let inv_a = 256 - a;

            // Blend each channel
            result[i][0] = ((fg_pixels[i][0] as u16 * a + bg_pixels[i][0] as u16 * inv_a) >> 8) as u8; // R
            result[i][1] = ((fg_pixels[i][1] as u16 * a + bg_pixels[i][1] as u16 * inv_a) >> 8) as u8; // G
            result[i][2] = ((fg_pixels[i][2] as u16 * a + bg_pixels[i][2] as u16 * inv_a) >> 8) as u8; // B
            result[i][3] = 255; // Output alpha is always opaque
        }

        Self::from_bytes(result)
    }

    // Helper methods for blend_alpha default implementation
    /// Convert batch to array of RGBA byte arrays
    fn to_bytes(self) -> [[u8; 4]; 4];

    /// Create batch from array of RGBA byte arrays
    fn from_bytes(bytes: [[u8; 4]; 4]) -> Self;
}

// ============================================================================
// Platform Selection (Type Aliasing)
// ============================================================================

/// The platform-optimal batch type.
///
/// On x86_64: Uses SSE/AVX intrinsics
/// On ARM: Uses NEON intrinsics
/// Fallback: Uses scalar operations
#[cfg(target_arch = "x86_64")]
pub use backends::x86::Batch;

#[cfg(target_arch = "aarch64")]
pub use backends::arm::Batch;

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub use backends::scalar::Batch;
