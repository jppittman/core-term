use core::ops::{Add, BitAnd, BitOr, Mul, Not, Shl, Shr, Sub};

use crate::simd::{Simd, SimdElement};

// Use the top-level backends module.
// Note: We assume `mod backends` is declared in lib.rs.
use crate::backends;

// Select backend based on target architecture
#[cfg(target_arch = "x86_64")]
use backends::x86 as backend;

#[cfg(target_arch = "aarch64")]
use backends::arm as backend;

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
use backends::scalar as backend;

// Re-export platform-optimal type
pub use backend::SimdVec;
// Re-export AVX2/Fallback type
pub use backend::SimdVec256;

/// The number of 32-bit lanes in a batch.
pub const LANES_U32: usize = 4;

/// Calculates the aligned stride for a given width, ensuring SIMD compatibility.
pub const fn calculate_aligned_stride(width: usize) -> usize {
    (width + LANES_U32 - 1) & !(LANES_U32 - 1)
}

/// A SIMD batch of values of type `T`.
#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct Batch<T: Copy> {
    pub(crate) inner: backend::SimdVec<T>,
}

/// A 256-bit SIMD batch (8 lanes of u32).
#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct Batch256<T: Copy> {
    pub(crate) inner: backend::SimdVec256<T>,
}

impl<T: Copy> Batch<T> {
    #[inline(always)]
    #[must_use]
    pub fn cast<U: Copy>(self) -> Batch<U> {
        Batch {
            inner: backend::cast(self.inner),
        }
    }
}

impl Batch<u32> {
    #[inline(always)]
    #[must_use]
    pub fn as_u16(self) -> Batch<u16> {
        self.cast()
    }
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
    #[inline(always)]
    pub fn to_bytes_packed(self) -> [u8; 4] {
        let mut arr = [0u32; 4];
        unsafe { self.store(arr.as_mut_ptr()) };
        [arr[0] as u8, arr[1] as u8, arr[2] as u8, arr[3] as u8]
    }
    #[inline(always)]
    pub fn extract(self, index: usize) -> u32 {
        let mut arr = [0u32; 4];
        unsafe { self.store(arr.as_mut_ptr()) };
        arr[index]
    }
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

pub trait SimdOps<T>: Copy + Clone + Sized {
    fn splat(val: T) -> Self;
    unsafe fn load(ptr: *const T) -> Self;
    unsafe fn store(self, ptr: *mut T);
    fn new(v0: T, v1: T, v2: T, v3: T) -> Self;
    fn add(self, other: Self) -> Self;
    fn sub(self, other: Self) -> Self;
    fn mul(self, other: Self) -> Self;
    fn bitand(self, other: Self) -> Self;
    fn bitor(self, other: Self) -> Self;
    fn not(self) -> Self;
    fn shr(self, count: i32) -> Self;
    fn shl(self, count: i32) -> Self;
    fn select(self, other: Self, mask: Self) -> Self;
    fn min(self, other: Self) -> Self;
    fn max(self, other: Self) -> Self;
    fn saturating_add(self, other: Self) -> Self;
    fn saturating_sub(self, other: Self) -> Self;
}

pub trait SimdOps8<T>: Copy + Clone + Sized {
    fn splat(val: T) -> Self;
    unsafe fn load(ptr: *const T) -> Self;
    unsafe fn store(self, ptr: *mut T);
    fn new(v0: T, v1: T, v2: T, v3: T, v4: T, v5: T, v6: T, v7: T) -> Self;
    fn add(self, other: Self) -> Self;
    fn sub(self, other: Self) -> Self;
    fn mul(self, other: Self) -> Self;
    fn bitand(self, other: Self) -> Self;
    fn bitor(self, other: Self) -> Self;
    fn not(self) -> Self;
    fn shr(self, count: i32) -> Self;
    fn shl(self, count: i32) -> Self;
    fn select(self, other: Self, mask: Self) -> Self;
    fn min(self, other: Self) -> Self;
    fn max(self, other: Self) -> Self;
    fn saturating_add(self, other: Self) -> Self;
    fn saturating_sub(self, other: Self) -> Self;
}

impl<T: Copy> Add for Batch<T> where backend::SimdVec<T>: SimdOps<T> { type Output = Self; #[inline(always)] fn add(self, rhs: Self) -> Self { Self { inner: self.inner.add(rhs.inner) } } }
impl<T: Copy> Sub for Batch<T> where backend::SimdVec<T>: SimdOps<T> { type Output = Self; #[inline(always)] fn sub(self, rhs: Self) -> Self { Self { inner: self.inner.sub(rhs.inner) } } }
impl<T: Copy> Mul for Batch<T> where backend::SimdVec<T>: SimdOps<T> { type Output = Self; #[inline(always)] fn mul(self, rhs: Self) -> Self { Self { inner: self.inner.mul(rhs.inner) } } }
impl<T: Copy> BitAnd for Batch<T> where backend::SimdVec<T>: SimdOps<T> { type Output = Self; #[inline(always)] fn bitand(self, rhs: Self) -> Self { Self { inner: self.inner.bitand(rhs.inner) } } }
impl<T: Copy> BitOr for Batch<T> where backend::SimdVec<T>: SimdOps<T> { type Output = Self; #[inline(always)] fn bitor(self, rhs: Self) -> Self { Self { inner: self.inner.bitor(rhs.inner) } } }
impl<T: Copy> Not for Batch<T> where backend::SimdVec<T>: SimdOps<T> { type Output = Self; #[inline(always)] fn not(self) -> Self { Self { inner: self.inner.not() } } }
impl<T: Copy> Shr<i32> for Batch<T> where backend::SimdVec<T>: SimdOps<T> { type Output = Self; #[inline(always)] fn shr(self, count: i32) -> Self { Self { inner: self.inner.shr(count) } } }
impl<T: Copy> Shl<i32> for Batch<T> where backend::SimdVec<T>: SimdOps<T> { type Output = Self; #[inline(always)] fn shl(self, count: i32) -> Self { Self { inner: self.inner.shl(count) } } }

pub trait Gatherable: Copy { fn to_u32(self) -> u32; }
impl Gatherable for u8 { #[inline(always)] fn to_u32(self) -> u32 { self as u32 } }
impl Gatherable for u16 { #[inline(always)] fn to_u32(self) -> u32 { self as u32 } }
impl Gatherable for u32 { #[inline(always)] fn to_u32(self) -> u32 { self } }

impl Batch<u16> {
    #[inline(always)]
    #[must_use]
    pub fn as_u32(self) -> Batch<u32> { self.cast() }
}
impl<T: Copy> Batch<T> where backend::SimdVec<T>: SimdOps<T> {
    #[inline(always)] #[must_use] pub fn splat(val: T) -> Self { Self { inner: backend::SimdVec::<T>::splat(val) } }
    #[inline(always)] #[must_use] pub unsafe fn load(ptr: *const T) -> Self { Self { inner: unsafe { backend::SimdVec::<T>::load(ptr) } } }
    #[inline(always)] pub unsafe fn store(self, ptr: *mut T) { unsafe { self.inner.store(ptr) } }
    #[inline(always)] #[must_use] pub fn new(v0: T, v1: T, v2: T, v3: T) -> Self { Self { inner: backend::SimdVec::<T>::new(v0, v1, v2, v3) } }
    #[inline(always)] #[must_use] pub fn min(self, other: Self) -> Self { Self { inner: self.inner.min(other.inner) } }
    #[inline(always)] #[must_use] pub fn max(self, other: Self) -> Self { Self { inner: self.inner.max(other.inner) } }
    #[inline(always)] #[must_use] pub fn saturating_add(self, other: Self) -> Self { Self { inner: self.inner.saturating_add(other.inner) } }
    #[inline(always)] #[must_use] pub fn saturating_sub(self, other: Self) -> Self { Self { inner: self.inner.saturating_sub(other.inner) } }
    #[inline(always)] #[must_use] pub fn select(self, other: Self, mask: Self) -> Self { Self { inner: self.inner.select(other.inner, mask.inner) } }
}

impl<T: Copy> Batch256<T> where backend::SimdVec256<T>: SimdOps8<T> {
    #[inline(always)] pub fn splat(val: T) -> Self { Self { inner: backend::SimdVec256::splat(val) } }
    #[inline(always)] pub fn new(v0: T, v1: T, v2: T, v3: T, v4: T, v5: T, v6: T, v7: T) -> Self { Self { inner: backend::SimdVec256::new(v0, v1, v2, v3, v4, v5, v6, v7) } }
    #[inline(always)] pub unsafe fn load(ptr: *const T) -> Self { Self { inner: unsafe { backend::SimdVec256::load(ptr) } } }
    #[inline(always)] pub unsafe fn store(self, ptr: *mut T) { unsafe { self.inner.store(ptr) } }
    #[inline(always)] pub fn select(self, other: Self, mask: Self) -> Self { Self { inner: self.inner.select(other.inner, mask.inner) } }
    #[inline(always)] pub fn min(self, other: Self) -> Self { Self { inner: self.inner.min(other.inner) } }
    #[inline(always)] pub fn max(self, other: Self) -> Self { Self { inner: self.inner.max(other.inner) } }
    #[inline(always)] pub fn saturating_add(self, other: Self) -> Self { Self { inner: self.inner.saturating_add(other.inner) } }
    #[inline(always)] pub fn saturating_sub(self, other: Self) -> Self { Self { inner: self.inner.saturating_sub(other.inner) } }
    #[inline(always)] pub fn cast<U: Copy>(self) -> Batch256<U> { Batch256 { inner: backend::cast256(self.inner) } }
}

impl<T: Copy> Add for Batch256<T> where backend::SimdVec256<T>: SimdOps8<T> { type Output = Self; #[inline(always)] fn add(self, rhs: Self) -> Self { Self { inner: self.inner.add(rhs.inner) } } }
impl<T: Copy> Sub for Batch256<T> where backend::SimdVec256<T>: SimdOps8<T> { type Output = Self; #[inline(always)] fn sub(self, rhs: Self) -> Self { Self { inner: self.inner.sub(rhs.inner) } } }
impl<T: Copy> Mul for Batch256<T> where backend::SimdVec256<T>: SimdOps8<T> { type Output = Self; #[inline(always)] fn mul(self, rhs: Self) -> Self { Self { inner: self.inner.mul(rhs.inner) } } }
impl<T: Copy> BitAnd for Batch256<T> where backend::SimdVec256<T>: SimdOps8<T> { type Output = Self; #[inline(always)] fn bitand(self, rhs: Self) -> Self { Self { inner: self.inner.bitand(rhs.inner) } } }
impl<T: Copy> BitOr for Batch256<T> where backend::SimdVec256<T>: SimdOps8<T> { type Output = Self; #[inline(always)] fn bitor(self, rhs: Self) -> Self { Self { inner: self.inner.bitor(rhs.inner) } } }
impl<T: Copy> Not for Batch256<T> where backend::SimdVec256<T>: SimdOps8<T> { type Output = Self; #[inline(always)] fn not(self) -> Self { Self { inner: self.inner.not() } } }
impl<T: Copy> Shr<i32> for Batch256<T> where backend::SimdVec256<T>: SimdOps8<T> { type Output = Self; #[inline(always)] fn shr(self, count: i32) -> Self { Self { inner: self.inner.shr(count) } } }
impl<T: Copy> Shl<i32> for Batch256<T> where backend::SimdVec256<T>: SimdOps8<T> { type Output = Self; #[inline(always)] fn shl(self, count: i32) -> Self { Self { inner: self.inner.shl(count) } } }

macro_rules! impl_simd_batch {
    ($type:ty, $lanes:expr) => {
        impl Simd for Batch<$type> {
            type Element = $type;
            const LANES: usize = $lanes;
            type Cast<U: SimdElement> = Batch<U>;
            #[inline(always)] fn cast<U: SimdElement>(self) -> Self::Cast<U> { self.cast() }
            #[inline(always)] fn splat(val: $type) -> Self { Batch::splat(val) }
            #[inline(always)] fn select(self, other: Self, mask: Self) -> Self { self.select(other, mask) }
            #[inline(always)] fn min(self, other: Self) -> Self { self.min(other) }
            #[inline(always)] fn max(self, other: Self) -> Self { self.max(other) }
            #[inline(always)] fn saturating_add(self, other: Self) -> Self { self.saturating_add(other) }
            #[inline(always)] fn saturating_sub(self, other: Self) -> Self { self.saturating_sub(other) }
            #[inline(always)] fn iota() -> Self {
                const ARR: [$type; $lanes] = {
                    let mut arr = [0; $lanes];
                    let mut i = 0;
                    while i < $lanes { arr[i] = i as $type; i += 1; }
                    arr
                };
                unsafe { Self::load(ARR.as_ptr()) }
            }
            #[inline(always)] unsafe fn load(ptr: *const Self::Element) -> Self { unsafe { Self::load(ptr) } }
            #[inline(always)] unsafe fn store(self, ptr: *mut Self::Element) { unsafe { self.store(ptr) } }
        }
        impl Simd for Batch256<$type> {
            type Element = $type;
            const LANES: usize = $lanes * 2;
            type Cast<U: SimdElement> = Batch256<U>;
            #[inline(always)] fn cast<U: SimdElement>(self) -> Self::Cast<U> { self.cast() }
            #[inline(always)] fn splat(val: $type) -> Self { Batch256::splat(val) }
            #[inline(always)] fn select(self, other: Self, mask: Self) -> Self { self.select(other, mask) }
            #[inline(always)] fn min(self, other: Self) -> Self { self.min(other) }
            #[inline(always)] fn max(self, other: Self) -> Self { self.max(other) }
            #[inline(always)] fn saturating_add(self, other: Self) -> Self { self.saturating_add(other) }
            #[inline(always)] fn saturating_sub(self, other: Self) -> Self { self.saturating_sub(other) }
            #[inline(always)] fn iota() -> Self {
                const ARR: [$type; $lanes * 2] = {
                    let mut arr = [0; $lanes * 2];
                    let mut i = 0;
                    while i < $lanes * 2 { arr[i] = i as $type; i += 1; }
                    arr
                };
                unsafe { Self::load(ARR.as_ptr()) }
            }
            #[inline(always)] unsafe fn load(ptr: *const Self::Element) -> Self { unsafe { Self::load(ptr) } }
            #[inline(always)] unsafe fn store(self, ptr: *mut Self::Element) { unsafe { self.store(ptr) } }
        }
    }
}

impl_simd_batch!(u32, 4);
impl_simd_batch!(u16, 8);
impl_simd_batch!(u8, 16);
