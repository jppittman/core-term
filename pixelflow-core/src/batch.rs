use core::ops::{Add, BitAnd, BitOr, Mul, Not, Shl, Shr, Sub};

// Use the top-level backends module.
// Note: We assume `mod backends` is declared in lib.rs.
use crate::backends;
use core::mem;

// Select backend based on target architecture
#[cfg(target_arch = "x86_64")]
use backends::x86 as backend;

#[cfg(target_arch = "aarch64")]
use backends::arm as backend;

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
use backends::scalar as backend;

// Re-export platform-optimal type
pub use backend::SimdVec;

/// The number of 32-bit lanes in a batch.
pub const LANES_U32: usize = 4;

/// Calculates the aligned stride for a given width, ensuring SIMD compatibility.
///
/// This aligns the width to the next multiple of `LANES_U32`.
///
/// # Parameters
/// * `width` - The requested width in pixels.
///
/// # Returns
/// * The aligned stride.
pub const fn calculate_aligned_stride(width: usize) -> usize {
    (width + LANES_U32 - 1) & !(LANES_U32 - 1)
}

/// A SIMD batch of values of type `T`.
///
/// This is the core type for vector operations. It wraps a platform-specific SIMD vector
/// (e.g., SSE2 __m128i on x86, NEON on ARM, or a fallback array).
// FIX: Removed Eq, Ord, PartialEq, PartialOrd derive macros.
// SIMD types generally don't support total ordering/equality due to mask results.
#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct Batch<T: Copy> {
    pub(crate) inner: backend::SimdVec<T>,
}

impl<T: Copy> Batch<T> {
    /// Casts the elements of the batch to another type `U`.
    ///
    /// # Returns
    /// * A new `Batch<U>` with cast elements.
    #[inline(always)]
    #[must_use]
    pub fn cast<U: Copy>(self) -> Batch<U> {
        Batch {
            inner: backend::cast(self.inner),
        }
    }
}

impl Batch<u32> {
    /// Casts the batch to `u16` elements.
    ///
    /// # Returns
    /// * A `Batch<u16>`.
    #[inline(always)]
    #[must_use]
    pub fn as_u16(self) -> Batch<u16> {
        self.cast()
    }

    /// Converts the batch to a fixed-size array of `usize`.
    ///
    /// Useful for extracting indices for scalar operations.
    ///
    /// # Returns
    /// * An array of 4 `usize` values.
    #[inline(always)]
    pub fn to_array_usize(self) -> [usize; 4] {
        let mut arr = [0u32; 4];
        unsafe { self.store(arr.as_mut_ptr()) };
        [
            arr[0] as usize,
            arr[1] as usize,
            arr[2] as usize,
            arr[3] as usize,
        ]
    }

    /// Converts the batch to a packed array of `u8` (first 4 bytes).
    ///
    /// # Returns
    /// * An array of 4 `u8` values.
    #[inline(always)]
    pub fn to_bytes_packed(self) -> [u8; 4] {
        let mut arr = [0u32; 4];
        unsafe { self.store(arr.as_mut_ptr()) };
        [arr[0] as u8, arr[1] as u8, arr[2] as u8, arr[3] as u8]
    }

    /// Extracts a single element from the batch at the given index.
    ///
    /// # Parameters
    /// * `index` - The lane index (0-3).
    ///
    /// # Returns
    /// * The `u32` value at the index.
    #[inline(always)]
    pub fn extract(self, index: usize) -> u32 {
        let mut arr = [0u32; 4];
        unsafe { self.store(arr.as_mut_ptr()) };
        arr[index]
    }

    /// Gathers values from a slice using a set of indices.
    ///
    /// # Parameters
    /// * `src` - The source slice.
    /// * `indices` - An array of 4 indices to gather from.
    ///
    /// # Returns
    /// * A `Batch<u32>` containing the gathered values.
    ///
    /// # Safety
    /// This function uses `get_unchecked` internally (conceptually), but the implementation here calls `get_unchecked`.
    /// Wait, the implementation calls `src.get_unchecked`. So it is unsafe if indices are out of bounds.
    #[inline(always)]
    #[must_use]
    pub unsafe fn gather<T: Gatherable>(src: &[T], indices: [usize; 4]) -> Self {
        unsafe {
            Self::new(
                src.get_unchecked(indices[0]).to_u32(),
                src.get_unchecked(indices[1]).to_u32(),
                src.get_unchecked(indices[2]).to_u32(),
                src.get_unchecked(indices[3]).to_u32(),
            )
        }
    }

    // === The "Safe Lie" Primitives ===

    /// Loads a batch from a slice safely, handling boundary conditions.
    ///
    /// If the slice has fewer than 4 elements, it loads partially and pads with zeros (or garbage).
    /// Actually, the implementation copies element by element if len < 4.
    ///
    /// # Parameters
    /// * `data` - The source slice.
    ///
    /// # Returns
    /// * A loaded `Batch<u32>`.
    #[inline(always)]
    pub fn load_from_slice(data: &[u32]) -> Self {
        if data.len() >= 4 {
            unsafe { Self::load(data.as_ptr()) }
        } else {
            let mut arr = [0u32; 4];
            for i in 0..data.len() {
                arr[i] = data[i];
            }
            Self::new(arr[0], arr[1], arr[2], arr[3])
        }
    }

    /// Stores the batch into a slice safely, handling boundary conditions.
    ///
    /// # Parameters
    /// * `dest` - The destination mutable slice.
    #[inline(always)]
    pub fn store_into_slice(self, dest: &mut [u32]) {
        if dest.len() >= 4 {
            unsafe { self.store(dest.as_mut_ptr()) }
        } else {
            let mut arr = [0u32; 4];
            unsafe { self.store(arr.as_mut_ptr()) };
            for i in 0..dest.len() {
                dest[i] = arr[i];
            }
        }
    }
}

/// Trait defining the core SIMD operations required by the backend.
pub trait SimdOps<T>: Copy + Clone + Sized {
    /// Creates a vector with all lanes set to `val`.
    fn splat(val: T) -> Self;
    /// Loads a vector from a pointer.
    ///
    /// # Safety
    /// Pointer must be valid and properly aligned.
    unsafe fn load(ptr: *const T) -> Self;
    /// Stores the vector to a pointer.
    ///
    /// # Safety
    /// Pointer must be valid and properly aligned.
    unsafe fn store(self, ptr: *mut T);
    /// Creates a vector from explicit values.
    fn new(v0: T, v1: T, v2: T, v3: T) -> Self;
    /// Adds two vectors.
    fn add(self, other: Self) -> Self;
    /// Subtracts two vectors.
    fn sub(self, other: Self) -> Self;
    /// Multiplies two vectors.
    fn mul(self, other: Self) -> Self;
    /// Bitwise AND.
    fn bitand(self, other: Self) -> Self;
    /// Bitwise OR.
    fn bitor(self, other: Self) -> Self;
    /// Bitwise NOT.
    fn not(self) -> Self;
    /// Bitwise shift right.
    fn shr(self, count: i32) -> Self;
    /// Bitwise shift left.
    fn shl(self, count: i32) -> Self;
    /// Selects elements based on a mask.
    fn select(self, other: Self, mask: Self) -> Self;
    /// Computes the minimum of two vectors.
    fn min(self, other: Self) -> Self;
    /// Computes the maximum of two vectors.
    fn max(self, other: Self) -> Self;
    /// Saturating addition.
    fn saturating_add(self, other: Self) -> Self;
    /// Saturating subtraction.
    fn saturating_sub(self, other: Self) -> Self;
}

/// Extended operations for byte-level SIMD (u8 only).
pub trait SimdOpsU8: SimdOps<u8> {
    /// Shuffles bytes according to indices.
    ///
    /// For each byte position i in the result:
    /// - If `indices[i] & 0x80` is set, result[i] = 0
    /// - Otherwise, result[i] = self[indices[i] & 0x0F]
    ///
    /// This maps to `pshufb` on x86 (SSSE3) and `vqtbl1q_u8` on ARM NEON.
    fn shuffle_bytes(self, indices: Self) -> Self;
}

impl<T: Copy> Add for Batch<T>
where
    backend::SimdVec<T>: SimdOps<T>,
{
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self {
            inner: self.inner.add(rhs.inner),
        }
    }
}
impl<T: Copy> Sub for Batch<T>
where
    backend::SimdVec<T>: SimdOps<T>,
{
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self {
            inner: self.inner.sub(rhs.inner),
        }
    }
}
impl<T: Copy> Mul for Batch<T>
where
    backend::SimdVec<T>: SimdOps<T>,
{
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self {
            inner: self.inner.mul(rhs.inner),
        }
    }
}
impl<T: Copy> BitAnd for Batch<T>
where
    backend::SimdVec<T>: SimdOps<T>,
{
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        Self {
            inner: self.inner.bitand(rhs.inner),
        }
    }
}
impl<T: Copy> BitOr for Batch<T>
where
    backend::SimdVec<T>: SimdOps<T>,
{
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        Self {
            inner: self.inner.bitor(rhs.inner),
        }
    }
}
impl<T: Copy> Not for Batch<T>
where
    backend::SimdVec<T>: SimdOps<T>,
{
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        Self {
            inner: self.inner.not(),
        }
    }
}
impl<T: Copy> Shr<i32> for Batch<T>
where
    backend::SimdVec<T>: SimdOps<T>,
{
    type Output = Self;
    #[inline(always)]
    fn shr(self, count: i32) -> Self {
        Self {
            inner: self.inner.shr(count),
        }
    }
}
impl<T: Copy> Shl<i32> for Batch<T>
where
    backend::SimdVec<T>: SimdOps<T>,
{
    type Output = Self;
    #[inline(always)]
    fn shl(self, count: i32) -> Self {
        Self {
            inner: self.inner.shl(count),
        }
    }
}

/// Trait for types that can be gathered into a `Batch<u32>`.
pub trait Gatherable: Copy {
    /// Converts the value to `u32`.
    fn to_u32(self) -> u32;
}
impl Gatherable for u8 {
    #[inline(always)]
    fn to_u32(self) -> u32 {
        self as u32
    }
}
impl Gatherable for u16 {
    #[inline(always)]
    fn to_u32(self) -> u32 {
        self as u32
    }
}
impl Gatherable for u32 {
    #[inline(always)]
    fn to_u32(self) -> u32 {
        self
    }
}

impl Batch<u16> {
    /// Casts the batch to `u32` elements.
    #[inline(always)]
    #[must_use]
    pub fn as_u32(self) -> Batch<u32> {
        self.cast()
    }
}

/// Shuffle mask for RGBA↔BGRA conversion (swaps bytes 0 and 2 in each 4-byte group).
///
/// This mask converts between:
/// - RGBA [R,G,B,A] → BGRA [B,G,R,A]
/// - BGRA [B,G,R,A] → RGBA [R,G,B,A]
///
/// The same mask works in both directions since it's a symmetric swap.
pub const SHUFFLE_RGBA_BGRA: [u8; 16] = [
    2, 1, 0, 3, // Pixel 0: swap R↔B
    6, 5, 4, 7, // Pixel 1: swap R↔B
    10, 9, 8, 11, // Pixel 2: swap R↔B
    14, 13, 12, 15, // Pixel 3: swap R↔B
];

impl Batch<u8>
where
    backend::SimdVec<u8>: SimdOpsU8,
{
    /// Shuffles bytes according to indices.
    ///
    /// For each byte position i in the result:
    /// - If `indices[i] & 0x80` is set, result[i] = 0
    /// - Otherwise, result[i] = self[indices[i] & 0x0F]
    #[inline(always)]
    #[must_use]
    pub fn shuffle_bytes(self, indices: Self) -> Self {
        Self {
            inner: self.inner.shuffle_bytes(indices.inner),
        }
    }

    /// Creates a batch from a 16-byte array.
    #[inline(always)]
    #[must_use]
    pub fn from_array(arr: [u8; 16]) -> Self {
        unsafe { Self::load(arr.as_ptr()) }
    }
}
impl<T: Copy> Batch<T>
where
    backend::SimdVec<T>: SimdOps<T>,
{
    /// Creates a batch with all lanes set to `val`.
    #[inline(always)]
    #[must_use]
    pub fn splat(val: T) -> Self {
        Self {
            inner: backend::SimdVec::<T>::splat(val),
        }
    }

    /// Loads a batch from a pointer.
    ///
    /// # Safety
    /// Pointer must be valid and aligned.
    #[inline(always)]
    #[must_use]
    pub unsafe fn load(ptr: *const T) -> Self {
        Self {
            inner: unsafe { backend::SimdVec::<T>::load(ptr) },
        }
    }

    /// Stores the batch to a pointer.
    ///
    /// # Safety
    /// Pointer must be valid and aligned.
    #[inline(always)]
    pub unsafe fn store(self, ptr: *mut T) {
        unsafe { self.inner.store(ptr) }
    }

    /// Creates a new batch from 4 explicit values.
    #[inline(always)]
    #[must_use]
    pub fn new(v0: T, v1: T, v2: T, v3: T) -> Self {
        Self {
            inner: backend::SimdVec::<T>::new(v0, v1, v2, v3),
        }
    }

    /// Computes the minimum of two batches.
    #[inline(always)]
    #[must_use]
    pub fn min(self, other: Self) -> Self {
        Self {
            inner: self.inner.min(other.inner),
        }
    }

    /// Computes the maximum of two batches.
    #[inline(always)]
    #[must_use]
    pub fn max(self, other: Self) -> Self {
        Self {
            inner: self.inner.max(other.inner),
        }
    }

    /// Computes the saturating addition of two batches.
    #[inline(always)]
    #[must_use]
    pub fn saturating_add(self, other: Self) -> Self {
        Self {
            inner: self.inner.saturating_add(other.inner),
        }
    }

    /// Computes the saturating subtraction of two batches.
    #[inline(always)]
    #[must_use]
    pub fn saturating_sub(self, other: Self) -> Self {
        Self {
            inner: self.inner.saturating_sub(other.inner),
        }
    }

    /// Selects elements from `self` or `other` based on `mask`.
    ///
    /// If a mask bit is 1, the corresponding bit from `other` is selected?
    /// Typically `select(a, b, mask)` selects b where mask is set, a where mask is clear? Or similar.
    /// Needs check on `SimdOps`. Usually `select` behaves like `(self & !mask) | (other & mask)`.
    #[inline(always)]
    #[must_use]
    pub fn select(self, other: Self, mask: Self) -> Self {
        Self {
            inner: self.inner.select(other.inner, mask.inner),
        }
    }
}

// Transmute doesn't need SimdOps, so separate impl block
impl<T: Copy> Batch<T> {
    /// Reinterprets the bits of this batch as a batch of type U.
    ///
    /// This is safe for repr(transparent) types like Rgba <-> u32.
    #[inline(always)]
    #[must_use]
    pub fn transmute<U: Copy>(self) -> Batch<U> {
        const {
            assert!(mem::size_of::<T>() == mem::size_of::<U>(), "Size mismatch");
        }
        unsafe { mem::transmute(self) }
    }
}
