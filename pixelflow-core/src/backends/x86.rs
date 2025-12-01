//! x86_64 SSE backend.

use crate::backend::{Backend, SimdBatch, BatchArithmetic, FloatBatchOps};
use core::arch::x86_64::*;
use core::marker::PhantomData;
use core::fmt::{Debug, Formatter};
use core::ops::*;
use core::any;
use core::ptr;

/// SSE2 Backend (4 lanes for u32/f32).
#[derive(Copy, Clone, Debug, Default)]
pub struct Sse2;

impl Backend for Sse2 {
    const LANES: usize = 4;
    const GATHER_IS_SLOW: bool = true;
    type Batch<T: Copy + Debug + Default + Send + Sync + 'static> = SimdVec<T>;

    #[inline(always)]
    fn downcast_u32_to_u8(b: SimdVec<u32>) -> SimdVec<u8> {
        unsafe {
            // Pack 32-bit integers to 16-bit integers (signed saturation)
            // Note: Since values are 0-255, signed/unsigned saturation to 16-bit doesn't matter much,
            // but we use packs_epi32 which treats inputs as signed.
            let b16 = _mm_packs_epi32(b.0, b.0);
            // Pack 16-bit integers to 8-bit integers (unsigned saturation)
            let b8 = _mm_packus_epi16(b16, b16);
            SimdVec(b8, PhantomData)
        }
    }

    #[inline(always)]
    fn upcast_u8_to_u32(b: SimdVec<u8>) -> SimdVec<u32> {
        unsafe {
            // Unpack low 8 bits to 16 bits
            let b16 = _mm_unpacklo_epi8(b.0, _mm_setzero_si128());
            // Unpack low 16 bits to 32 bits
            let b32 = _mm_unpacklo_epi16(b16, _mm_setzero_si128());
            SimdVec(b32, PhantomData)
        }
    }

    #[inline(always)]
    fn u32_to_f32(b: SimdVec<u32>) -> SimdVec<f32> {
        unsafe {
            // _mm_cvtepi32_ps converts signed i32 to f32
            // For u32, we need to handle the high bit carefully for values > i32::MAX
            // For simplicity, cast through scalar (font values won't exceed i32::MAX)
            let mut vals = [0u32; 4];
            _mm_storeu_si128(vals.as_mut_ptr() as *mut _, b.0);
            let floats = [vals[0] as f32, vals[1] as f32, vals[2] as f32, vals[3] as f32];
            SimdVec(cast_from_ps(_mm_loadu_ps(floats.as_ptr())), PhantomData)
        }
    }

    #[inline(always)]
    fn f32_to_u32(b: SimdVec<f32>) -> SimdVec<u32> {
        unsafe {
            // _mm_cvttps_epi32 truncates f32 to i32
            let i32_vec = _mm_cvttps_epi32(cast_to_ps(b.0));
            SimdVec(i32_vec, PhantomData)
        }
    }

    #[inline(always)]
    fn transmute_u32_to_f32(b: SimdVec<u32>) -> SimdVec<f32> {
        unsafe { b.transmute() }
    }

    #[inline(always)]
    fn transmute_f32_to_u32(b: SimdVec<f32>) -> SimdVec<u32> {
        unsafe { b.transmute() }
    }
}

/// Platform-specific SIMD vector wrapper.
#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct SimdVec<T>(pub(crate) __m128i, PhantomData<T>);

impl<T> Default for SimdVec<T> {
    fn default() -> Self {
        unsafe { Self(_mm_setzero_si128(), PhantomData) }
    }
}

impl<T> Debug for SimdVec<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        write!(f, "SimdVec({:?})", any::type_name::<T>())
    }
}

// Inherent methods for compatibility
impl<T: Copy> SimdVec<T> {
    #[inline(always)]
    pub unsafe fn transmute<U>(self) -> SimdVec<U> {
        SimdVec(self.0, PhantomData)
    }

    // Deprecated cast alias for transmute
    #[inline(always)]
    pub fn cast<U>(self) -> SimdVec<U> {
         unsafe { self.transmute() }
    }

    #[inline(always)]
    pub unsafe fn store(self, ptr: *mut T) {
        unsafe { _mm_storeu_si128(ptr as *mut __m128i, self.0) }
    }
}

impl SimdVec<u32> {
    #[inline(always)]
    pub fn splat(val: u32) -> Self { unsafe { Self(_mm_set1_epi32(val as i32), PhantomData) } }

    #[inline(always)]
    pub fn new(v0: u32, v1: u32, v2: u32, v3: u32) -> Self {
        unsafe { Self(_mm_set_epi32(v3 as i32, v2 as i32, v1 as i32, v0 as i32), PhantomData) }
    }
}

impl SimdVec<u16> {
    #[inline(always)]
    pub fn splat(val: u16) -> Self { unsafe { Self(_mm_set1_epi16(val as i16), PhantomData) } }
}

impl SimdVec<u8> {
    #[inline(always)]
    pub fn splat(val: u8) -> Self { unsafe { Self(_mm_set1_epi8(val as i8), PhantomData) } }
}

impl SimdVec<f32> {
    #[inline(always)]
    pub fn splat(val: f32) -> Self { unsafe { Self(cast_from_ps(_mm_set1_ps(val)), PhantomData) } }
}

// Helpers
#[inline(always)]
unsafe fn cast_to_ps(i: __m128i) -> __m128 {
    unsafe { _mm_castsi128_ps(i) }
}
#[inline(always)]
unsafe fn cast_from_ps(f: __m128) -> __m128i {
    unsafe { _mm_castps_si128(f) }
}

// ============================================================================
// u32 Implementation
// ============================================================================

impl Add for SimdVec<u32> { type Output = Self; fn add(self, rhs: Self) -> Self { unsafe { Self(_mm_add_epi32(self.0, rhs.0), PhantomData) } } }
impl Sub for SimdVec<u32> { type Output = Self; fn sub(self, rhs: Self) -> Self { unsafe { Self(_mm_sub_epi32(self.0, rhs.0), PhantomData) } } }
impl Mul for SimdVec<u32> { type Output = Self; fn mul(self, rhs: Self) -> Self { unsafe { Self(_mm_mullo_epi32(self.0, rhs.0), PhantomData) } } }
impl Div for SimdVec<u32> {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        let mut a = [0u32; 4];
        let mut b = [0u32; 4];
        unsafe { _mm_storeu_si128(a.as_mut_ptr() as *mut _, self.0) };
        unsafe { _mm_storeu_si128(b.as_mut_ptr() as *mut _, rhs.0) };
        let res = [
            if b[0] != 0 { a[0]/b[0] } else { 0 },
            if b[1] != 0 { a[1]/b[1] } else { 0 },
            if b[2] != 0 { a[2]/b[2] } else { 0 },
            if b[3] != 0 { a[3]/b[3] } else { 0 },
        ];
        unsafe { Self(_mm_loadu_si128(res.as_ptr() as *const _), PhantomData) }
    }
}
impl BitAnd for SimdVec<u32> { type Output = Self; fn bitand(self, rhs: Self) -> Self { unsafe { Self(_mm_and_si128(self.0, rhs.0), PhantomData) } } }
impl BitOr for SimdVec<u32> { type Output = Self; fn bitor(self, rhs: Self) -> Self { unsafe { Self(_mm_or_si128(self.0, rhs.0), PhantomData) } } }
impl BitXor for SimdVec<u32> { type Output = Self; fn bitxor(self, rhs: Self) -> Self { unsafe { Self(_mm_xor_si128(self.0, rhs.0), PhantomData) } } }
impl Not for SimdVec<u32> { type Output = Self; fn not(self) -> Self { unsafe { Self(_mm_xor_si128(self.0, _mm_set1_epi32(-1)), PhantomData) } } }
impl Shl<i32> for SimdVec<u32> { type Output = Self; fn shl(self, rhs: i32) -> Self { unsafe { Self(_mm_sll_epi32(self.0, _mm_cvtsi32_si128(rhs)), PhantomData) } } }
impl Shr<i32> for SimdVec<u32> { type Output = Self; fn shr(self, rhs: i32) -> Self { unsafe { Self(_mm_srl_epi32(self.0, _mm_cvtsi32_si128(rhs)), PhantomData) } } }

// Generic SimdBatch implementation
impl<T: Copy + Debug + Default + Send + Sync + 'static> SimdBatch<T> for SimdVec<T> {
    fn splat(val: T) -> Self {
        unsafe {
            if core::mem::size_of::<T>() == 4 {
                let v: u32 = core::mem::transmute_copy(&val);
                Self(_mm_set1_epi32(v as i32), PhantomData)
            } else if core::mem::size_of::<T>() == 2 {
                 let v: u16 = core::mem::transmute_copy(&val);
                Self(_mm_set1_epi16(v as i16), PhantomData)
            } else if core::mem::size_of::<T>() == 1 {
                let v: u8 = core::mem::transmute_copy(&val);
                Self(_mm_set1_epi8(v as i8), PhantomData)
            } else {
                Self(_mm_setzero_si128(), PhantomData)
            }
        }
    }

    fn sequential_from(start: T) -> Self {
        unsafe {
            if core::mem::size_of::<T>() == 4 {
                let s: u32 = core::mem::transmute_copy(&start);
                Self(_mm_set_epi32((s+3) as i32, (s+2) as i32, (s+1) as i32, s as i32), PhantomData)
            } else if core::mem::size_of::<T>() == 1 {
                // u8 seq
                let s: u8 = core::mem::transmute_copy(&start);
                let mut arr = [0u8; 16];
                for i in 0..16 { arr[i] = s.wrapping_add(i as u8); }
                Self(_mm_loadu_si128(arr.as_ptr() as *const _), PhantomData)
            } else {
                Self(_mm_setzero_si128(), PhantomData)
            }
        }
    }

    fn load(slice: &[T]) -> Self {
        unsafe {
            if core::mem::size_of::<T>() == 1 {
                // Load 4 bytes for u8 packed
                let ptr = slice.as_ptr() as *const i32;
                let v = ptr::read_unaligned(ptr);
                Self(_mm_cvtsi32_si128(v), PhantomData)
            } else if core::mem::size_of::<T>() == 2 {
                // Load 8 bytes for u16 packed
                let ptr = slice.as_ptr() as *const i64;
                let v = ptr::read_unaligned(ptr);
                Self(_mm_cvtsi64_si128(v), PhantomData)
            } else {
                Self(_mm_loadu_si128(slice.as_ptr() as *const _), PhantomData)
            }
        }
    }

    fn store(&self, slice: &mut [T]) {
        unsafe {
            if core::mem::size_of::<T>() == 1 {
                // Store 4 bytes for u8 packed
                let v = _mm_cvtsi128_si32(self.0);
                let ptr = slice.as_mut_ptr() as *mut i32;
                ptr::write_unaligned(ptr, v);
            } else if core::mem::size_of::<T>() == 2 {
                // Store 8 bytes for u16 packed
                let v = _mm_cvtsi128_si64(self.0);
                let ptr = slice.as_mut_ptr() as *mut i64;
                ptr::write_unaligned(ptr, v);
            } else {
                _mm_storeu_si128(slice.as_mut_ptr() as *mut _, self.0)
            }
        }
    }

    fn first(&self) -> T {
        unsafe {
            // Extract the first lane
            let v = _mm_cvtsi128_si32(self.0) as u32;
            core::mem::transmute_copy(&v)
        }
    }
}

impl BatchArithmetic<u32> for SimdVec<u32> {
    fn select(self, if_true: Self, if_false: Self) -> Self {
        unsafe {
            let masked_true = _mm_and_si128(self.0, if_true.0);
            let masked_false = _mm_andnot_si128(self.0, if_false.0);
            Self(_mm_or_si128(masked_true, masked_false), PhantomData)
        }
    }

    fn gather(base: &[u32], indices: Self) -> Self {
        unsafe {
            let mut idx = [0u32; 4];
            _mm_storeu_si128(idx.as_mut_ptr() as *mut _, indices.0);
            let v0 = base.get(idx[0] as usize).unwrap_or(&0);
            let v1 = base.get(idx[1] as usize).unwrap_or(&0);
            let v2 = base.get(idx[2] as usize).unwrap_or(&0);
            let v3 = base.get(idx[3] as usize).unwrap_or(&0);
            Self(_mm_set_epi32(*v3 as i32, *v2 as i32, *v1 as i32, *v0 as i32), PhantomData)
        }
    }

    fn gather_u8(base: &[u8], indices: Self) -> Self {
        unsafe {
            let mut idx = [0u32; 4];
            _mm_storeu_si128(idx.as_mut_ptr() as *mut _, indices.0);
            let v0 = base.get(idx[0] as usize).unwrap_or(&0);
            let v1 = base.get(idx[1] as usize).unwrap_or(&0);
            let v2 = base.get(idx[2] as usize).unwrap_or(&0);
            let v3 = base.get(idx[3] as usize).unwrap_or(&0);
            Self(_mm_set_epi32(*v3 as i32, *v2 as i32, *v1 as i32, *v0 as i32), PhantomData)
        }
    }

    #[inline(always)]
    fn min(self, other: Self) -> Self {
        unsafe {
             let mask = cmp_gt_u32(self.0, other.0);
             let masked_other = _mm_and_si128(mask, other.0);
             let masked_self = _mm_andnot_si128(mask, self.0);
             Self(_mm_or_si128(masked_self, masked_other), PhantomData)
        }
    }

    #[inline(always)]
    fn max(self, other: Self) -> Self {
        unsafe {
             let mask = cmp_gt_u32(self.0, other.0);
             let masked_self = _mm_and_si128(mask, self.0);
             let masked_other = _mm_andnot_si128(mask, other.0);
             Self(_mm_or_si128(masked_self, masked_other), PhantomData)
        }
    }

    #[inline(always)]
    fn saturating_add(self, other: Self) -> Self {
        unsafe {
            let sum = _mm_add_epi32(self.0, other.0);
            let mask = cmp_gt_u32(self.0, sum);
            Self(_mm_or_si128(sum, mask), PhantomData)
        }
    }

    #[inline(always)]
    fn saturating_sub(self, other: Self) -> Self {
        unsafe {
            let diff = _mm_sub_epi32(self.0, other.0);
            let mask = cmp_gt_u32(other.0, self.0);
            Self(_mm_andnot_si128(mask, diff), PhantomData)
        }
    }

    #[inline(always)]
    fn cmp_eq(self, other: Self) -> Self {
        unsafe { Self(_mm_cmpeq_epi32(self.0, other.0), PhantomData) }
    }
    #[inline(always)]
    fn cmp_ne(self, other: Self) -> Self {
        unsafe {
            let eq = _mm_cmpeq_epi32(self.0, other.0);
            Self(_mm_xor_si128(eq, _mm_set1_epi32(-1)), PhantomData)
        }
    }
    #[inline(always)]
    fn cmp_lt(self, other: Self) -> Self {
         unsafe { Self(cmp_gt_u32(other.0, self.0), PhantomData) }
    }
    #[inline(always)]
    fn cmp_le(self, other: Self) -> Self {
        unsafe {
             let gt = cmp_gt_u32(self.0, other.0);
             Self(_mm_xor_si128(gt, _mm_set1_epi32(-1)), PhantomData)
        }
    }
    #[inline(always)]
    fn cmp_gt(self, other: Self) -> Self {
        unsafe { Self(cmp_gt_u32(self.0, other.0), PhantomData) }
    }
    #[inline(always)]
    fn cmp_ge(self, other: Self) -> Self {
        unsafe {
             let lt = cmp_gt_u32(other.0, self.0);
             Self(_mm_xor_si128(lt, _mm_set1_epi32(-1)), PhantomData)
        }
    }
}

// Helper: Unsigned Greater Than for u32 (SSE2)
#[inline(always)]
unsafe fn cmp_gt_u32(a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        let offset = _mm_set1_epi32(i32::MIN);
        let a_off = _mm_add_epi32(a, offset);
        let b_off = _mm_add_epi32(b, offset);
        _mm_cmpgt_epi32(a_off, b_off)
    }
}

// ============================================================================
// u16 Implementation
// ============================================================================

impl Add for SimdVec<u16> { type Output = Self; fn add(self, rhs: Self) -> Self { unsafe { Self(_mm_add_epi16(self.0, rhs.0), PhantomData) } } }
impl Sub for SimdVec<u16> { type Output = Self; fn sub(self, rhs: Self) -> Self { unsafe { Self(_mm_sub_epi16(self.0, rhs.0), PhantomData) } } }
impl Mul for SimdVec<u16> { type Output = Self; fn mul(self, rhs: Self) -> Self { unsafe { Self(_mm_mullo_epi16(self.0, rhs.0), PhantomData) } } }
impl Div for SimdVec<u16> { type Output = Self; fn div(self, _rhs: Self) -> Self { unimplemented!("u16 div") } }
impl BitAnd for SimdVec<u16> { type Output = Self; fn bitand(self, rhs: Self) -> Self { unsafe { Self(_mm_and_si128(self.0, rhs.0), PhantomData) } } }
impl BitOr for SimdVec<u16> { type Output = Self; fn bitor(self, rhs: Self) -> Self { unsafe { Self(_mm_or_si128(self.0, rhs.0), PhantomData) } } }
impl BitXor for SimdVec<u16> { type Output = Self; fn bitxor(self, rhs: Self) -> Self { unsafe { Self(_mm_xor_si128(self.0, rhs.0), PhantomData) } } }
impl Not for SimdVec<u16> { type Output = Self; fn not(self) -> Self { unsafe { Self(_mm_xor_si128(self.0, _mm_set1_epi32(-1)), PhantomData) } } }
impl Shl<i32> for SimdVec<u16> { type Output = Self; fn shl(self, rhs: i32) -> Self { unsafe { Self(_mm_sll_epi16(self.0, _mm_cvtsi32_si128(rhs)), PhantomData) } } }
impl Shr<i32> for SimdVec<u16> { type Output = Self; fn shr(self, rhs: i32) -> Self { unsafe { Self(_mm_srl_epi16(self.0, _mm_cvtsi32_si128(rhs)), PhantomData) } } }

impl BatchArithmetic<u16> for SimdVec<u16> {
    fn select(self, if_true: Self, if_false: Self) -> Self {
         unsafe {
            let masked_true = _mm_and_si128(self.0, if_true.0);
            let masked_false = _mm_andnot_si128(self.0, if_false.0);
            Self(_mm_or_si128(masked_true, masked_false), PhantomData)
        }
    }
    fn gather(base: &[u16], _indices: Self) -> Self {
         unimplemented!("u16 gather")
    }

    fn min(self, other: Self) -> Self { unsafe { Self(_mm_min_epi16(self.0, other.0), PhantomData) } }
    fn max(self, other: Self) -> Self { unsafe { Self(_mm_max_epi16(self.0, other.0), PhantomData) } }

    fn saturating_add(self, other: Self) -> Self { unsafe { Self(_mm_adds_epu16(self.0, other.0), PhantomData) } }
    fn saturating_sub(self, other: Self) -> Self { unsafe { Self(_mm_subs_epu16(self.0, other.0), PhantomData) } }

    fn cmp_eq(self, other: Self) -> Self { unsafe { Self(_mm_cmpeq_epi16(self.0, other.0), PhantomData) } }
    fn cmp_ne(self, other: Self) -> Self {
         unsafe {
             let eq = _mm_cmpeq_epi16(self.0, other.0);
             Self(_mm_xor_si128(eq, _mm_set1_epi32(-1)), PhantomData)
         }
    }
    fn cmp_lt(self, other: Self) -> Self { unsafe { Self(_mm_cmplt_epi16(self.0, other.0), PhantomData) } }
    fn cmp_le(self, other: Self) -> Self {
         let lt = self.cmp_lt(other);
         let eq = self.cmp_eq(other);
         lt | eq
    }
    fn cmp_gt(self, other: Self) -> Self {
         unsafe { Self(_mm_cmpgt_epi16(self.0, other.0), PhantomData) }
    }
    fn cmp_ge(self, other: Self) -> Self {
         let gt = self.cmp_gt(other);
         let eq = self.cmp_eq(other);
         gt | eq
    }
}

// ============================================================================
// u8 Implementation
// ============================================================================

impl Add for SimdVec<u8> { type Output = Self; fn add(self, rhs: Self) -> Self { unsafe { Self(_mm_add_epi8(self.0, rhs.0), PhantomData) } } }
impl Sub for SimdVec<u8> { type Output = Self; fn sub(self, rhs: Self) -> Self { unsafe { Self(_mm_sub_epi8(self.0, rhs.0), PhantomData) } } }
impl Mul for SimdVec<u8> { type Output = Self; fn mul(self, _rhs: Self) -> Self { unimplemented!("u8 mul") } }
impl Div for SimdVec<u8> { type Output = Self; fn div(self, _rhs: Self) -> Self { unimplemented!("u8 div") } }
impl BitAnd for SimdVec<u8> { type Output = Self; fn bitand(self, rhs: Self) -> Self { unsafe { Self(_mm_and_si128(self.0, rhs.0), PhantomData) } } }
impl BitOr for SimdVec<u8> { type Output = Self; fn bitor(self, rhs: Self) -> Self { unsafe { Self(_mm_or_si128(self.0, rhs.0), PhantomData) } } }
impl BitXor for SimdVec<u8> { type Output = Self; fn bitxor(self, rhs: Self) -> Self { unsafe { Self(_mm_xor_si128(self.0, rhs.0), PhantomData) } } }
impl Not for SimdVec<u8> { type Output = Self; fn not(self) -> Self { unsafe { Self(_mm_xor_si128(self.0, _mm_set1_epi32(-1)), PhantomData) } } }
impl Shl<i32> for SimdVec<u8> { type Output = Self; fn shl(self, _rhs: i32) -> Self { unimplemented!("u8 shl") } }
impl Shr<i32> for SimdVec<u8> { type Output = Self; fn shr(self, _rhs: i32) -> Self { unimplemented!("u8 shr") } }

impl BatchArithmetic<u8> for SimdVec<u8> {
    fn select(self, if_true: Self, if_false: Self) -> Self {
         unsafe {
            let masked_true = _mm_and_si128(self.0, if_true.0);
            let masked_false = _mm_andnot_si128(self.0, if_false.0);
            Self(_mm_or_si128(masked_true, masked_false), PhantomData)
        }
    }
    fn gather(base: &[u8], indices: Self) -> Self {
        unsafe {
             let mut idx = [0u8; 16];
             _mm_storeu_si128(idx.as_mut_ptr() as *mut _, indices.0);
             let mut res = [0u8; 16];
             for i in 0..16 { res[i] = base[idx[i] as usize]; }
             Self::load(&res)
        }
    }

    fn min(self, other: Self) -> Self { unsafe { Self(_mm_min_epu8(self.0, other.0), PhantomData) } }
    fn max(self, other: Self) -> Self { unsafe { Self(_mm_max_epu8(self.0, other.0), PhantomData) } }
    fn saturating_add(self, other: Self) -> Self { unsafe { Self(_mm_adds_epu8(self.0, other.0), PhantomData) } }
    fn saturating_sub(self, other: Self) -> Self { unsafe { Self(_mm_subs_epu8(self.0, other.0), PhantomData) } }

    fn cmp_eq(self, other: Self) -> Self { unsafe { Self(_mm_cmpeq_epi8(self.0, other.0), PhantomData) } }
    fn cmp_ne(self, other: Self) -> Self {
         unsafe {
             let eq = _mm_cmpeq_epi8(self.0, other.0);
             Self(_mm_xor_si128(eq, _mm_set1_epi32(-1)), PhantomData)
         }
    }
    fn cmp_lt(self, other: Self) -> Self { unsafe { Self(_mm_cmplt_epi8(self.0, other.0), PhantomData) } }
    fn cmp_le(self, other: Self) -> Self {
         let lt = self.cmp_lt(other);
         let eq = self.cmp_eq(other);
         lt | eq
    }
    fn cmp_gt(self, other: Self) -> Self {
         unsafe { Self(_mm_cmpgt_epi8(self.0, other.0), PhantomData) }
    }
    fn cmp_ge(self, other: Self) -> Self {
         let gt = self.cmp_gt(other);
         let eq = self.cmp_eq(other);
         gt | eq
    }
}

// ============================================================================
// f32 Implementation
// ============================================================================

impl Add for SimdVec<f32> { type Output = Self; fn add(self, rhs: Self) -> Self { unsafe { Self(cast_from_ps(_mm_add_ps(cast_to_ps(self.0), cast_to_ps(rhs.0))), PhantomData) } } }
impl Sub for SimdVec<f32> { type Output = Self; fn sub(self, rhs: Self) -> Self { unsafe { Self(cast_from_ps(_mm_sub_ps(cast_to_ps(self.0), cast_to_ps(rhs.0))), PhantomData) } } }
impl Mul for SimdVec<f32> { type Output = Self; fn mul(self, rhs: Self) -> Self { unsafe { Self(cast_from_ps(_mm_mul_ps(cast_to_ps(self.0), cast_to_ps(rhs.0))), PhantomData) } } }
impl Div for SimdVec<f32> { type Output = Self; fn div(self, rhs: Self) -> Self { unsafe { Self(cast_from_ps(_mm_div_ps(cast_to_ps(self.0), cast_to_ps(rhs.0))), PhantomData) } } }
impl BitAnd for SimdVec<f32> { type Output = Self; fn bitand(self, rhs: Self) -> Self { unsafe { Self(_mm_and_si128(self.0, rhs.0), PhantomData) } } }
impl BitOr for SimdVec<f32> { type Output = Self; fn bitor(self, rhs: Self) -> Self { unsafe { Self(_mm_or_si128(self.0, rhs.0), PhantomData) } } }
impl BitXor for SimdVec<f32> { type Output = Self; fn bitxor(self, rhs: Self) -> Self { unsafe { Self(_mm_xor_si128(self.0, rhs.0), PhantomData) } } }
impl Not for SimdVec<f32> { type Output = Self; fn not(self) -> Self { unsafe { Self(_mm_xor_si128(self.0, _mm_set1_epi32(-1)), PhantomData) } } }
impl Shl<i32> for SimdVec<f32> { type Output = Self; fn shl(self, _rhs: i32) -> Self { unimplemented!() } }
impl Shr<i32> for SimdVec<f32> { type Output = Self; fn shr(self, _rhs: i32) -> Self { unimplemented!() } }

impl BatchArithmetic<f32> for SimdVec<f32> {
    fn select(self, if_true: Self, if_false: Self) -> Self {
         unsafe {
            let masked_true = _mm_and_si128(self.0, if_true.0);
            let masked_false = _mm_andnot_si128(self.0, if_false.0);
            Self(_mm_or_si128(masked_true, masked_false), PhantomData)
        }
    }

    fn gather(base: &[f32], indices: Self) -> Self { unimplemented!() }

    fn min(self, other: Self) -> Self { unsafe { Self(cast_from_ps(_mm_min_ps(cast_to_ps(self.0), cast_to_ps(other.0))), PhantomData) } }
    fn max(self, other: Self) -> Self { unsafe { Self(cast_from_ps(_mm_max_ps(cast_to_ps(self.0), cast_to_ps(other.0))), PhantomData) } }
    fn saturating_add(self, other: Self) -> Self { self + other }
    fn saturating_sub(self, other: Self) -> Self { self - other }

    fn cmp_eq(self, other: Self) -> Self { unsafe { Self(cast_from_ps(_mm_cmpeq_ps(cast_to_ps(self.0), cast_to_ps(other.0))), PhantomData) } }
    fn cmp_ne(self, other: Self) -> Self { unsafe { Self(cast_from_ps(_mm_cmpneq_ps(cast_to_ps(self.0), cast_to_ps(other.0))), PhantomData) } }
    fn cmp_lt(self, other: Self) -> Self { unsafe { Self(cast_from_ps(_mm_cmplt_ps(cast_to_ps(self.0), cast_to_ps(other.0))), PhantomData) } }
    fn cmp_le(self, other: Self) -> Self { unsafe { Self(cast_from_ps(_mm_cmple_ps(cast_to_ps(self.0), cast_to_ps(other.0))), PhantomData) } }
    fn cmp_gt(self, other: Self) -> Self { unsafe { Self(cast_from_ps(_mm_cmpgt_ps(cast_to_ps(self.0), cast_to_ps(other.0))), PhantomData) } }
    fn cmp_ge(self, other: Self) -> Self { unsafe { Self(cast_from_ps(_mm_cmpge_ps(cast_to_ps(self.0), cast_to_ps(other.0))), PhantomData) } }
}

impl FloatBatchOps for SimdVec<f32> {
    #[inline(always)]
    fn sqrt(self) -> Self {
        unsafe { Self(cast_from_ps(_mm_sqrt_ps(cast_to_ps(self.0))), PhantomData) }
    }

    #[inline(always)]
    fn abs(self) -> Self {
        unsafe {
            // Clear sign bit: AND with 0x7FFFFFFF
            let sign_mask = _mm_set1_epi32(0x7FFFFFFF_u32 as i32);
            Self(_mm_and_si128(self.0, sign_mask), PhantomData)
        }
    }
}
