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
