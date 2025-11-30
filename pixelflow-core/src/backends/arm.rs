//! ARM NEON backend using `std::arch` intrinsics.
//!
//! This backend uses NEON instructions for AArch64 (64-bit ARM),
//! which includes Apple Silicon (M1/M2/M3) and modern ARM Linux systems.
//!
//! ## Type-Driven Instruction Selection
//!
//! Unlike x86 where all SIMD types are `__m128i`, ARM's NEON has distinct types:
//! - `uint32x4_t` for 4×u32 (uses `vadd`, `vmul`, etc.)
//! - `uint16x8_t` for 8×u16 (uses `vaddq_u16`, `vmulq_u16`, etc.)
//! - `uint8x16_t` for 16×u8 (uses `vaddq_u8`, `vmulq_u8`, etc.)
//!
//! We use newtype wrappers to implement the `SimdOps` trait for each type.

use crate::batch::{SimdFloatOps, SimdOps, SimdOpsU8};
use core::arch::aarch64::*;
use core::marker::PhantomData;

// ============================================================================
// Platform-Specific Types
// ============================================================================

/// Platform-specific SIMD vector wrapper.
///
/// On ARM: Each type has a distinct NEON register type.
#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct SimdVec<T: Copy>(pub(crate) NeonReg<T>);

/// The actual NEON register types (platform-specific).
///
/// ARM's NEON intrinsics are strongly typed, so we need distinct types
/// for each lane width.
#[derive(Copy, Clone)]
pub union NeonReg<T> {
    /// 4 lanes of u32.
    pub u32: uint32x4_t,
    /// 8 lanes of u16.
    pub u16: uint16x8_t,
    /// 16 lanes of u8.
    pub u8: uint8x16_t,
    /// 4 lanes of f32.
    pub f32: float32x4_t,
    _marker: PhantomData<T>,
}

// ============================================================================
// u32 Implementation (4 lanes, 32-bit operations)
// ============================================================================

impl SimdOps<u32> for SimdVec<u32> {
    /// Broadcasts a value to all lanes.
    #[inline(always)]
    fn splat(val: u32) -> Self {
        // SAFETY: vdupq_n_u32 is always safe
        unsafe {
            Self(NeonReg {
                u32: vdupq_n_u32(val),
            })
        }
    }

    /// Loads a vector from a pointer.
    #[inline(always)]
    unsafe fn load(ptr: *const u32) -> Self {
        // SAFETY: Caller guarantees ptr is valid for reading 16 bytes
        // vld1q_u32 handles unaligned loads
        unsafe {
            Self(NeonReg {
                u32: vld1q_u32(ptr),
            })
        }
    }

    /// Stores the vector to a pointer.
    #[inline(always)]
    unsafe fn store(self, ptr: *mut u32) {
        // SAFETY: Caller guarantees ptr is valid for writing 16 bytes
        unsafe { vst1q_u32(ptr, self.0.u32) }
    }

    /// Creates a new vector from values.
    #[inline(always)]
    fn new(v0: u32, v1: u32, v2: u32, v3: u32) -> Self {
        // SAFETY: Load from stack array
        unsafe {
            Self(NeonReg {
                u32: vld1q_u32([v0, v1, v2, v3].as_ptr()),
            })
        }
    }

    /// Adds two vectors.
    #[inline(always)]
    fn add(self, other: Self) -> Self {
        // SAFETY: vaddq_u32 (32-bit add)
        unsafe {
            Self(NeonReg {
                u32: vaddq_u32(self.0.u32, other.0.u32),
            })
        }
    }

    /// Subtracts two vectors.
    #[inline(always)]
    fn sub(self, other: Self) -> Self {
        // SAFETY: vsubq_u32 (32-bit subtract)
        unsafe {
            Self(NeonReg {
                u32: vsubq_u32(self.0.u32, other.0.u32),
            })
        }
    }

    /// Multiplies two vectors.
    #[inline(always)]
    fn mul(self, other: Self) -> Self {
        // SAFETY: vmulq_u32 (32-bit multiply)
        unsafe {
            Self(NeonReg {
                u32: vmulq_u32(self.0.u32, other.0.u32),
            })
        }
    }

    /// Bitwise AND.
    #[inline(always)]
    fn bitand(self, other: Self) -> Self {
        // SAFETY: vandq_u32 (bitwise AND)
        unsafe {
            Self(NeonReg {
                u32: vandq_u32(self.0.u32, other.0.u32),
            })
        }
    }

    /// Bitwise OR.
    #[inline(always)]
    fn bitor(self, other: Self) -> Self {
        // SAFETY: vorrq_u32 (bitwise OR)
        unsafe {
            Self(NeonReg {
                u32: vorrq_u32(self.0.u32, other.0.u32),
            })
        }
    }

    /// Bitwise NOT.
    #[inline(always)]
    fn not(self) -> Self {
        // SAFETY: vmvnq_u32 (bitwise NOT / MVN instruction)
        unsafe {
            Self(NeonReg {
                u32: vmvnq_u32(self.0.u32),
            })
        }
    }

    /// Logical shift right.
    #[inline(always)]
    fn shr(self, count: i32) -> Self {
        // SAFETY: vshrq_n_u32 (32-bit logical right shift)
        // Note: NEON shift amounts must be compile-time constants
        // For variable shifts, use vshlq with negative shift amount
        unsafe {
            let shift = vdupq_n_s32(-count);
            Self(NeonReg {
                u32: vshlq_u32(self.0.u32, shift),
            })
        }
    }

    /// Logical shift left.
    #[inline(always)]
    fn shl(self, count: i32) -> Self {
        // SAFETY: vshlq_n_u32 (32-bit logical left shift)
        unsafe {
            let shift = vdupq_n_s32(count);
            Self(NeonReg {
                u32: vshlq_u32(self.0.u32, shift),
            })
        }
    }

    /// Element-wise selection.
    #[inline(always)]
    fn select(self, other: Self, mask: Self) -> Self {
        // Implements: (self & mask) | (other & !mask)
        // NEON has BSL (Bit Select): result = (mask & a) | (!mask & b)
        // vbslq_u32(mask, a, b) means: (mask & a) | (!mask & b)
        // SAFETY: All bitwise operations are safe
        unsafe {
            Self(NeonReg {
                u32: vbslq_u32(mask.0.u32, self.0.u32, other.0.u32),
            })
        }
    }

    /// Minimum value.
    #[inline(always)]
    fn min(self, other: Self) -> Self {
        // SAFETY: vminq_u32 (unsigned 32-bit min)
        unsafe {
            Self(NeonReg {
                u32: vminq_u32(self.0.u32, other.0.u32),
            })
        }
    }

    /// Maximum value.
    #[inline(always)]
    fn max(self, other: Self) -> Self {
        // SAFETY: vmaxq_u32 (unsigned 32-bit max)
        unsafe {
            Self(NeonReg {
                u32: vmaxq_u32(self.0.u32, other.0.u32),
            })
        }
    }

    /// Saturating addition.
    #[inline(always)]
    fn saturating_add(self, other: Self) -> Self {
        // SAFETY: vqaddq_u32 (saturating add - Q prefix means saturating!)
        unsafe {
            Self(NeonReg {
                u32: vqaddq_u32(self.0.u32, other.0.u32),
            })
        }
    }

    /// Saturating subtraction.
    #[inline(always)]
    fn saturating_sub(self, other: Self) -> Self {
        // SAFETY: vqsubq_u32 (saturating subtract)
        unsafe {
            Self(NeonReg {
                u32: vqsubq_u32(self.0.u32, other.0.u32),
            })
        }
    }
}

// Inherent methods on SimdVec<u32>
impl SimdVec<u32> {
    #[inline(always)]
    pub fn to_f32(self) -> SimdVec<f32> {
        unsafe {
            SimdVec(NeonReg {
                f32: vcvtq_f32_u32(self.0.u32),
            })
        }
    }

    #[inline(always)]
    pub fn cmp_eq(self, other: Self) -> SimdVec<u32> {
        unsafe {
            SimdVec(NeonReg {
                u32: vceqq_u32(self.0.u32, other.0.u32),
            })
        }
    }

    #[inline(always)]
    pub fn cmp_ne(self, other: Self) -> SimdVec<u32> {
        unsafe {
            SimdVec(NeonReg {
                u32: vmvnq_u32(vceqq_u32(self.0.u32, other.0.u32)),
            })
        }
    }

    #[inline(always)]
    pub fn cmp_lt(self, other: Self) -> SimdVec<u32> {
        unsafe {
            SimdVec(NeonReg {
                u32: vcltq_u32(self.0.u32, other.0.u32),
            })
        }
    }

    #[inline(always)]
    pub fn cmp_le(self, other: Self) -> SimdVec<u32> {
        unsafe {
            SimdVec(NeonReg {
                u32: vcleq_u32(self.0.u32, other.0.u32),
            })
        }
    }

    #[inline(always)]
    pub fn cmp_gt(self, other: Self) -> SimdVec<u32> {
        unsafe {
            SimdVec(NeonReg {
                u32: vcgtq_u32(self.0.u32, other.0.u32),
            })
        }
    }

    #[inline(always)]
    pub fn cmp_ge(self, other: Self) -> SimdVec<u32> {
        unsafe {
            SimdVec(NeonReg {
                u32: vcgeq_u32(self.0.u32, other.0.u32),
            })
        }
    }
}

// ============================================================================
// f32 Implementation (4 lanes)
// ============================================================================

impl SimdOps<f32> for SimdVec<f32> {
    #[inline(always)]
    fn splat(val: f32) -> Self {
        unsafe {
            Self(NeonReg {
                f32: vdupq_n_f32(val),
            })
        }
    }

    #[inline(always)]
    unsafe fn load(ptr: *const f32) -> Self {
        unsafe {
            Self(NeonReg {
                f32: vld1q_f32(ptr),
            })
        }
    }

    #[inline(always)]
    unsafe fn store(self, ptr: *mut f32) {
        unsafe { vst1q_f32(ptr, self.0.f32) }
    }

    #[inline(always)]
    fn new(v0: f32, v1: f32, v2: f32, v3: f32) -> Self {
        unsafe {
            Self(NeonReg {
                f32: vld1q_f32([v0, v1, v2, v3].as_ptr()),
            })
        }
    }

    #[inline(always)]
    fn add(self, other: Self) -> Self {
        unsafe {
            Self(NeonReg {
                f32: vaddq_f32(self.0.f32, other.0.f32),
            })
        }
    }

    #[inline(always)]
    fn sub(self, other: Self) -> Self {
        unsafe {
            Self(NeonReg {
                f32: vsubq_f32(self.0.f32, other.0.f32),
            })
        }
    }

    #[inline(always)]
    fn mul(self, other: Self) -> Self {
        unsafe {
            Self(NeonReg {
                f32: vmulq_f32(self.0.f32, other.0.f32),
            })
        }
    }

    #[inline(always)]
    fn bitand(self, other: Self) -> Self {
        // Reinterpret as u32 for bitwise ops
        unsafe {
            let a = vreinterpretq_u32_f32(self.0.f32);
            let b = vreinterpretq_u32_f32(other.0.f32);
            Self(NeonReg {
                f32: vreinterpretq_f32_u32(vandq_u32(a, b)),
            })
        }
    }

    #[inline(always)]
    fn bitor(self, other: Self) -> Self {
        unsafe {
            let a = vreinterpretq_u32_f32(self.0.f32);
            let b = vreinterpretq_u32_f32(other.0.f32);
            Self(NeonReg {
                f32: vreinterpretq_f32_u32(vorrq_u32(a, b)),
            })
        }
    }

    #[inline(always)]
    fn not(self) -> Self {
        unsafe {
            let a = vreinterpretq_u32_f32(self.0.f32);
            Self(NeonReg {
                f32: vreinterpretq_f32_u32(vmvnq_u32(a)),
            })
        }
    }

    #[inline(always)]
    fn shr(self, _count: i32) -> Self {
        unimplemented!("Bitwise shift not supported on float batch")
    }

    #[inline(always)]
    fn shl(self, _count: i32) -> Self {
        unimplemented!("Bitwise shift not supported on float batch")
    }

    #[inline(always)]
    fn select(self, other: Self, mask: Self) -> Self {
        unsafe {
            let m = vreinterpretq_u32_f32(mask.0.f32);
            let a = vreinterpretq_u32_f32(self.0.f32);
            let b = vreinterpretq_u32_f32(other.0.f32);
            Self(NeonReg {
                f32: vreinterpretq_f32_u32(vbslq_u32(m, a, b)),
            })
        }
    }

    #[inline(always)]
    fn min(self, other: Self) -> Self {
        unsafe {
            Self(NeonReg {
                f32: vminq_f32(self.0.f32, other.0.f32),
            })
        }
    }

    #[inline(always)]
    fn max(self, other: Self) -> Self {
        unsafe {
            Self(NeonReg {
                f32: vmaxq_f32(self.0.f32, other.0.f32),
            })
        }
    }

    #[inline(always)]
    fn saturating_add(self, other: Self) -> Self {
        // No saturating add for floats
        self.add(other)
    }

    #[inline(always)]
    fn saturating_sub(self, other: Self) -> Self {
        self.sub(other)
    }
}

impl SimdFloatOps for SimdVec<f32> {
    #[inline(always)]
    fn sqrt(self) -> Self {
        unsafe {
            Self(NeonReg {
                f32: vsqrtq_f32(self.0.f32),
            })
        }
    }

    #[inline(always)]
    fn div(self, other: Self) -> Self {
        unsafe {
            Self(NeonReg {
                f32: vdivq_f32(self.0.f32, other.0.f32),
            })
        }
    }

    #[inline(always)]
    fn ceil(self) -> Self {
        unsafe {
            Self(NeonReg {
                f32: vrndpq_f32(self.0.f32),
            })
        }
    }

    #[inline(always)]
    fn floor(self) -> Self {
        unsafe {
            Self(NeonReg {
                f32: vrndmq_f32(self.0.f32),
            })
        }
    }

    #[inline(always)]
    fn round(self) -> Self {
        unsafe {
            Self(NeonReg {
                f32: vrndnq_f32(self.0.f32),
            })
        }
    }

    #[inline(always)]
    fn abs(self) -> Self {
        unsafe {
            Self(NeonReg {
                f32: vabsq_f32(self.0.f32),
            })
        }
    }

    #[inline(always)]
    fn to_u32(self) -> SimdVec<u32> {
        unsafe {
            SimdVec(NeonReg {
                u32: vcvtq_u32_f32(self.0.f32),
            })
        }
    }

    #[inline(always)]
    fn to_i32(self) -> SimdVec<u32> {
        unsafe {
            SimdVec(NeonReg {
                u32: vreinterpretq_u32_s32(vcvtq_s32_f32(self.0.f32)),
            })
        }
    }

    #[inline(always)]
    fn cmp_eq(self, other: Self) -> SimdVec<u32> {
        unsafe {
            SimdVec(NeonReg {
                u32: vceqq_f32(self.0.f32, other.0.f32),
            })
        }
    }

    #[inline(always)]
    fn cmp_ne(self, other: Self) -> SimdVec<u32> {
        unsafe {
            SimdVec(NeonReg {
                u32: vmvnq_u32(vceqq_f32(self.0.f32, other.0.f32)),
            })
        }
    }

    #[inline(always)]
    fn cmp_lt(self, other: Self) -> SimdVec<u32> {
        unsafe {
            SimdVec(NeonReg {
                u32: vcltq_f32(self.0.f32, other.0.f32),
            })
        }
    }

    #[inline(always)]
    fn cmp_le(self, other: Self) -> SimdVec<u32> {
        unsafe {
            SimdVec(NeonReg {
                u32: vcleq_f32(self.0.f32, other.0.f32),
            })
        }
    }

    #[inline(always)]
    fn cmp_gt(self, other: Self) -> SimdVec<u32> {
        unsafe {
            SimdVec(NeonReg {
                u32: vcgtq_f32(self.0.f32, other.0.f32),
            })
        }
    }

    #[inline(always)]
    fn cmp_ge(self, other: Self) -> SimdVec<u32> {
        unsafe {
            SimdVec(NeonReg {
                u32: vcgeq_f32(self.0.f32, other.0.f32),
            })
        }
    }
}

// ============================================================================
// u16 Implementation (8 lanes, 16-bit operations)
// ============================================================================

impl SimdOps<u16> for SimdVec<u16> {
    /// Broadcasts a value to all lanes.
    #[inline(always)]
    fn splat(val: u16) -> Self {
        // SAFETY: vdupq_n_u16 is always safe
        unsafe {
            Self(NeonReg {
                u16: vdupq_n_u16(val),
            })
        }
    }

    /// Loads a vector from a pointer.
    #[inline(always)]
    unsafe fn load(ptr: *const u16) -> Self {
        // SAFETY: Caller guarantees ptr is valid for reading 16 bytes (8×u16)
        unsafe {
            Self(NeonReg {
                u16: vld1q_u16(ptr),
            })
        }
    }

    /// Stores the vector to a pointer.
    #[inline(always)]
    unsafe fn store(self, ptr: *mut u16) {
        // SAFETY: Caller guarantees ptr is valid for writing 16 bytes
        unsafe { vst1q_u16(ptr, self.0.u16) }
    }

    /// Creates a new vector (partial initialization).
    #[inline(always)]
    fn new(v0: u16, v1: u16, v2: u16, v3: u16) -> Self {
        // SAFETY: Load from stack array (only 4 lanes, rest zero)
        unsafe {
            Self(NeonReg {
                u16: vld1q_u16([v0, v1, v2, v3, 0, 0, 0, 0].as_ptr()),
            })
        }
    }

    /// Adds two vectors.
    #[inline(always)]
    fn add(self, other: Self) -> Self {
        // SAFETY: vaddq_u16 (16-bit add)
        // This is the instruction that `add_u16()` was trying to expose!
        unsafe {
            Self(NeonReg {
                u16: vaddq_u16(self.0.u16, other.0.u16),
            })
        }
    }

    /// Subtracts two vectors.
    #[inline(always)]
    fn sub(self, other: Self) -> Self {
        // SAFETY: vsubq_u16 (16-bit subtract)
        unsafe {
            Self(NeonReg {
                u16: vsubq_u16(self.0.u16, other.0.u16),
            })
        }
    }

    /// Multiplies two vectors.
    #[inline(always)]
    fn mul(self, other: Self) -> Self {
        // SAFETY: vmulq_u16 (16-bit multiply)
        // This is the instruction that `mullo_u16()` was trying to expose!
        unsafe {
            Self(NeonReg {
                u16: vmulq_u16(self.0.u16, other.0.u16),
            })
        }
    }

    /// Bitwise AND.
    #[inline(always)]
    fn bitand(self, other: Self) -> Self {
        // SAFETY: vandq_u16 (bitwise AND)
        unsafe {
            Self(NeonReg {
                u16: vandq_u16(self.0.u16, other.0.u16),
            })
        }
    }

    /// Bitwise OR.
    #[inline(always)]
    fn bitor(self, other: Self) -> Self {
        // SAFETY: vorrq_u16 (bitwise OR)
        unsafe {
            Self(NeonReg {
                u16: vorrq_u16(self.0.u16, other.0.u16),
            })
        }
    }

    /// Bitwise NOT.
    #[inline(always)]
    fn not(self) -> Self {
        // SAFETY: vmvnq_u16 (bitwise NOT)
        unsafe {
            Self(NeonReg {
                u16: vmvnq_u16(self.0.u16),
            })
        }
    }

    /// Logical shift right.
    #[inline(always)]
    fn shr(self, count: i32) -> Self {
        // SAFETY: vshlq_u16 with negative shift (16-bit logical right shift)
        // This is the instruction that `shift_right_u16()` was trying to expose!
        unsafe {
            let shift = vdupq_n_s16(-count as i16);
            Self(NeonReg {
                u16: vshlq_u16(self.0.u16, shift),
            })
        }
    }

    /// Logical shift left.
    #[inline(always)]
    fn shl(self, count: i32) -> Self {
        // SAFETY: vshlq_u16 with positive shift (16-bit logical left shift)
        unsafe {
            let shift = vdupq_n_s16(count as i16);
            Self(NeonReg {
                u16: vshlq_u16(self.0.u16, shift),
            })
        }
    }

    /// Element-wise selection.
    #[inline(always)]
    fn select(self, other: Self, mask: Self) -> Self {
        // SAFETY: BSL instruction
        unsafe {
            Self(NeonReg {
                u16: vbslq_u16(mask.0.u16, self.0.u16, other.0.u16),
            })
        }
    }

    /// Minimum value.
    #[inline(always)]
    fn min(self, other: Self) -> Self {
        // SAFETY: vminq_u16 (unsigned 16-bit min)
        unsafe {
            Self(NeonReg {
                u16: vminq_u16(self.0.u16, other.0.u16),
            })
        }
    }

    /// Maximum value.
    #[inline(always)]
    fn max(self, other: Self) -> Self {
        // SAFETY: vmaxq_u16 (unsigned 16-bit max)
        unsafe {
            Self(NeonReg {
                u16: vmaxq_u16(self.0.u16, other.0.u16),
            })
        }
    }

    /// Saturating addition.
    #[inline(always)]
    fn saturating_add(self, other: Self) -> Self {
        // SAFETY: vqaddq_u16 (saturating add)
        unsafe {
            Self(NeonReg {
                u16: vqaddq_u16(self.0.u16, other.0.u16),
            })
        }
    }

    /// Saturating subtraction.
    #[inline(always)]
    fn saturating_sub(self, other: Self) -> Self {
        // SAFETY: vqsubq_u16 (saturating subtract)
        unsafe {
            Self(NeonReg {
                u16: vqsubq_u16(self.0.u16, other.0.u16),
            })
        }
    }
}

// ============================================================================
// u8 Implementation (16 lanes, 8-bit operations)
// ============================================================================

impl SimdOps<u8> for SimdVec<u8> {
    /// Broadcasts a value to all lanes.
    #[inline(always)]
    fn splat(val: u8) -> Self {
        // SAFETY: vdupq_n_u8 is always safe
        unsafe {
            Self(NeonReg {
                u8: vdupq_n_u8(val),
            })
        }
    }

    /// Loads a vector from a pointer.
    #[inline(always)]
    unsafe fn load(ptr: *const u8) -> Self {
        // SAFETY: Caller guarantees ptr is valid for reading 16 bytes (16×u8)
        unsafe { Self(NeonReg { u8: vld1q_u8(ptr) }) }
    }

    /// Stores the vector to a pointer.
    #[inline(always)]
    unsafe fn store(self, ptr: *mut u8) {
        // SAFETY: Caller guarantees ptr is valid for writing 16 bytes
        unsafe { vst1q_u8(ptr, self.0.u8) }
    }

    /// Creates a new vector (partial initialization).
    #[inline(always)]
    fn new(v0: u8, v1: u8, v2: u8, v3: u8) -> Self {
        // SAFETY: Load from stack array (only 4 lanes, rest zero)
        unsafe {
            Self(NeonReg {
                u8: vld1q_u8([v0, v1, v2, v3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0].as_ptr()),
            })
        }
    }

    /// Adds two vectors.
    #[inline(always)]
    fn add(self, other: Self) -> Self {
        // SAFETY: vaddq_u8 (8-bit add)
        unsafe {
            Self(NeonReg {
                u8: vaddq_u8(self.0.u8, other.0.u8),
            })
        }
    }

    /// Subtracts two vectors.
    #[inline(always)]
    fn sub(self, other: Self) -> Self {
        // SAFETY: vsubq_u8 (8-bit subtract)
        unsafe {
            Self(NeonReg {
                u8: vsubq_u8(self.0.u8, other.0.u8),
            })
        }
    }

    /// Multiplies two vectors.
    #[inline(always)]
    fn mul(self, other: Self) -> Self {
        // SAFETY: vmulq_u8 (8-bit multiply)
        unsafe {
            Self(NeonReg {
                u8: vmulq_u8(self.0.u8, other.0.u8),
            })
        }
    }

    /// Bitwise AND.
    #[inline(always)]
    fn bitand(self, other: Self) -> Self {
        // SAFETY: vandq_u8 (bitwise AND)
        unsafe {
            Self(NeonReg {
                u8: vandq_u8(self.0.u8, other.0.u8),
            })
        }
    }

    /// Bitwise OR.
    #[inline(always)]
    fn bitor(self, other: Self) -> Self {
        // SAFETY: vorrq_u8 (bitwise OR)
        unsafe {
            Self(NeonReg {
                u8: vorrq_u8(self.0.u8, other.0.u8),
            })
        }
    }

    /// Bitwise NOT.
    #[inline(always)]
    fn not(self) -> Self {
        // SAFETY: vmvnq_u8 (bitwise NOT)
        unsafe {
            Self(NeonReg {
                u8: vmvnq_u8(self.0.u8),
            })
        }
    }

    /// Logical shift right.
    #[inline(always)]
    fn shr(self, count: i32) -> Self {
        // SAFETY: vshlq_u8 with negative shift (8-bit logical right shift)
        unsafe {
            let shift = vdupq_n_s8(-count as i8);
            Self(NeonReg {
                u8: vshlq_u8(self.0.u8, shift),
            })
        }
    }

    /// Logical shift left.
    #[inline(always)]
    fn shl(self, count: i32) -> Self {
        // SAFETY: vshlq_u8 with positive shift (8-bit logical left shift)
        unsafe {
            let shift = vdupq_n_s8(count as i8);
            Self(NeonReg {
                u8: vshlq_u8(self.0.u8, shift),
            })
        }
    }

    /// Element-wise selection.
    #[inline(always)]
    fn select(self, other: Self, mask: Self) -> Self {
        // SAFETY: BSL instruction
        unsafe {
            Self(NeonReg {
                u8: vbslq_u8(mask.0.u8, self.0.u8, other.0.u8),
            })
        }
    }

    /// Minimum value.
    #[inline(always)]
    fn min(self, other: Self) -> Self {
        // SAFETY: vminq_u8 (unsigned 8-bit min)
        unsafe {
            Self(NeonReg {
                u8: vminq_u8(self.0.u8, other.0.u8),
            })
        }
    }

    /// Maximum value.
    #[inline(always)]
    fn max(self, other: Self) -> Self {
        // SAFETY: vmaxq_u8 (unsigned 8-bit max)
        unsafe {
            Self(NeonReg {
                u8: vmaxq_u8(self.0.u8, other.0.u8),
            })
        }
    }

    /// Saturating addition.
    #[inline(always)]
    fn saturating_add(self, other: Self) -> Self {
        // SAFETY: vqaddq_u8 (saturating add)
        unsafe {
            Self(NeonReg {
                u8: vqaddq_u8(self.0.u8, other.0.u8),
            })
        }
    }

    /// Saturating subtraction.
    #[inline(always)]
    fn saturating_sub(self, other: Self) -> Self {
        // SAFETY: vqsubq_u8 (saturating subtract)
        unsafe {
            Self(NeonReg {
                u8: vqsubq_u8(self.0.u8, other.0.u8),
            })
        }
    }
}

impl SimdOpsU8 for SimdVec<u8> {
    /// Shuffles bytes according to indices using ARM NEON's vqtbl1q_u8.
    ///
    /// For each byte position i:
    /// - If `indices[i] >= 16`, result[i] = 0
    /// - Otherwise, result[i] = self[indices[i]]
    #[inline(always)]
    fn shuffle_bytes(self, indices: Self) -> Self {
        // SAFETY: vqtbl1q_u8 is a table lookup instruction
        // Available on all AArch64 (ARM64) processors
        unsafe {
            Self(NeonReg {
                u8: vqtbl1q_u8(self.0.u8, indices.0.u8),
            })
        }
    }
}

// ============================================================================
// Bitcasting (Zero-Cost Type Conversion)
// ============================================================================

/// Bitcast between SIMD types.
///
/// On ARM, this is a `vreinterpret` operation (zero cost).
///
/// ```ignore
/// let pixels = SimdVec::<u32>::splat(0xFF00FF00); // 4×u32
/// let as_u16: SimdVec<u16> = cast(pixels);         // View as 8×u16
/// ```
///
/// # Parameters
/// * `v` - The source vector.
///
/// # Returns
/// * The vector bitcasted to type `U`.
#[inline(always)]
pub fn cast<T: Copy, U: Copy>(v: SimdVec<T>) -> SimdVec<U> {
    // SAFETY: Bitcast via union (same memory, different interpretation)
    // ARM's `vreinterpret` instructions compile to zero instructions
    // We transmute through the union to change the type parameter
    unsafe { core::mem::transmute_copy(&v) }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_u32_add() {
        let a = SimdVec::<u32>::splat(100);
        let b = SimdVec::<u32>::splat(50);
        let c = a.add(b);

        let mut output = [0u32; 4];
        unsafe { c.store(output.as_mut_ptr()) };
        assert_eq!(output, [150; 4]);
    }

    #[test]
    fn test_u16_multiply() {
        // This is the key operation for bilinear interpolation
        let a = SimdVec::<u16>::splat(100);
        let b = SimdVec::<u16>::splat(2);
        let c = a.mul(b); // Uses vmulq_u16!

        let mut output = [0u16; 8];
        unsafe { c.store(output.as_mut_ptr()) };
        assert_eq!(output, [200; 8]);
    }

    #[test]
    fn test_bitcast() {
        let pixels = SimdVec::<u32>::new(0x12345678, 0x9ABCDEF0, 0, 0);
        let as_u16: SimdVec<u16> = cast(pixels);

        let mut output = [0u16; 8];
        unsafe { as_u16.store(output.as_mut_ptr()) };

        // Little-endian byte order (same as x86)
        assert_eq!(output[0], 0x5678); // Low 16 bits of 0x12345678
        assert_eq!(output[1], 0x1234); // High 16 bits
    }

    #[test]
    fn test_shuffle_bytes_rgba_bgra() {
        // Test RGBA to BGRA conversion
        // Input: 4 RGBA pixels [R,G,B,A] = [0x11,0x22,0x33,0xFF] each
        let rgba_bytes: [u8; 16] = [
            0x11, 0x22, 0x33, 0xFF, // Pixel 0: R=0x11, G=0x22, B=0x33, A=0xFF
            0xAA, 0xBB, 0xCC, 0xDD, // Pixel 1
            0x00, 0x55, 0xAA, 0xFF, // Pixel 2
            0x12, 0x34, 0x56, 0x78, // Pixel 3
        ];
        // Shuffle mask: swap bytes 0↔2 in each 4-byte group
        let shuffle_mask: [u8; 16] = [
            2, 1, 0, 3, // Pixel 0
            6, 5, 4, 7, // Pixel 1
            10, 9, 8, 11, // Pixel 2
            14, 13, 12, 15, // Pixel 3
        ];

        let data = unsafe { SimdVec::<u8>::load(rgba_bytes.as_ptr()) };
        let indices = unsafe { SimdVec::<u8>::load(shuffle_mask.as_ptr()) };
        let result = data.shuffle_bytes(indices);

        let mut output = [0u8; 16];
        unsafe { result.store(output.as_mut_ptr()) };

        // Expected: BGRA [B,G,R,A]
        assert_eq!(output[0..4], [0x33, 0x22, 0x11, 0xFF]); // Pixel 0: B,G,R,A
        assert_eq!(output[4..8], [0xCC, 0xBB, 0xAA, 0xDD]); // Pixel 1
        assert_eq!(output[8..12], [0xAA, 0x55, 0x00, 0xFF]); // Pixel 2
        assert_eq!(output[12..16], [0x56, 0x34, 0x12, 0x78]); // Pixel 3
    }
}
