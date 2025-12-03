//! AVX512 Backend (16 lanes for u32/f32).

use crate::backend::{Backend, BatchArithmetic, FloatBatchOps, SimdBatch};
use core::any;
use core::arch::x86_64::*;
use core::fmt::{Debug, Formatter};
use core::marker::PhantomData;
use core::ops::*;

/// AVX512 Backend (16 lanes).
#[derive(Copy, Clone, Debug, Default)]
pub struct Avx512;

impl Backend for Avx512 {
    const LANES: usize = 16;
    type Batch<T: Copy + Debug + Default + Send + Sync + 'static> = SimdVec<T>;

    #[inline(always)]
    fn downcast_u32_to_u8(b: SimdVec<u32>) -> SimdVec<u8> {
        unsafe {
            let val_128 = _mm512_cvtepi32_epi8(b.0.i);
            let val_512 = _mm512_castsi128_si512(val_128);
            SimdVec(Avx512Reg { i: val_512 }, PhantomData)
        }
    }

    #[inline(always)]
    fn upcast_u8_to_u32(b: SimdVec<u8>) -> SimdVec<u32> {
        unsafe {
            let val_128 = _mm512_castsi512_si128(b.0.i);
            let val_512 = _mm512_cvtepu8_epi32(val_128);
            SimdVec(Avx512Reg { i: val_512 }, PhantomData)
        }
    }

    #[inline(always)]
    fn u32_to_f32(b: SimdVec<u32>) -> SimdVec<f32> {
        unsafe {
            SimdVec(
                Avx512Reg {
                    f: _mm512_cvtepi32_ps(b.0.i),
                },
                PhantomData,
            )
        }
    }

    #[inline(always)]
    fn f32_to_u32(b: SimdVec<f32>) -> SimdVec<u32> {
        unsafe {
            SimdVec(
                Avx512Reg {
                    i: _mm512_cvttps_epi32(b.0.f),
                },
                PhantomData,
            )
        }
    }

    #[inline(always)]
    fn transmute_u32_to_f32(b: SimdVec<u32>) -> SimdVec<f32> {
        unsafe {
            SimdVec(
                Avx512Reg {
                    f: _mm512_castsi512_ps(b.0.i),
                },
                PhantomData,
            )
        }
    }

    #[inline(always)]
    fn transmute_f32_to_u32(b: SimdVec<f32>) -> SimdVec<u32> {
        unsafe {
            SimdVec(
                Avx512Reg {
                    i: _mm512_castps_si512(b.0.f),
                },
                PhantomData,
            )
        }
    }
}

#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct SimdVec<T: Copy>(pub(crate) Avx512Reg<T>, PhantomData<T>);

#[derive(Copy, Clone)]
#[repr(C)]
pub union Avx512Reg<T> {
    pub i: __m512i,
    pub f: __m512,
    _marker: PhantomData<T>,
}

impl<T: Copy> Default for SimdVec<T> {
    fn default() -> Self {
        unsafe {
            Self(
                Avx512Reg {
                    i: _mm512_setzero_si512(),
                },
                PhantomData,
            )
        }
    }
}

impl<T: Copy> Debug for SimdVec<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        write!(f, "SimdVec({:?})", any::type_name::<T>())
    }
}

impl<T: Copy> SimdVec<T> {
    #[inline(always)]
    pub unsafe fn transmute<U: Copy>(self) -> SimdVec<U> {
        unsafe { core::mem::transmute_copy(&self) }
    }

    #[inline(always)]
    pub fn cast<U: Copy>(self) -> SimdVec<U> {
        unsafe { self.transmute() }
    }

    #[inline(always)]
    pub unsafe fn store(self, ptr: *mut T) {
        unsafe {
            if core::mem::size_of::<T>() == 4 {
                _mm512_storeu_si512(ptr as *mut _, self.0.i);
            } else if core::mem::size_of::<T>() == 2 {
                let half = _mm512_castsi512_si256(self.0.i);
                _mm256_storeu_si256(ptr as *mut _, half);
            } else if core::mem::size_of::<T>() == 1 {
                let quarter = _mm512_castsi512_si128(self.0.i);
                _mm_storeu_si128(ptr as *mut _, quarter);
            }
        }
    }
}

impl SimdVec<u32> {
    #[inline(always)]
    pub fn splat(val: u32) -> Self {
        unsafe {
            Self(
                Avx512Reg {
                    i: _mm512_set1_epi32(val as i32),
                },
                PhantomData,
            )
        }
    }

    #[inline(always)]
    pub fn to_array(self) -> [u32; 16] {
        unsafe { core::mem::transmute(self.0.i) }
    }

    #[inline(always)]
    pub fn from_array(arr: [u32; 16]) -> Self {
         unsafe {
             Self(
                 Avx512Reg {
                     i: _mm512_loadu_si512(arr.as_ptr() as *const _),
                 },
                 PhantomData,
             )
         }
    }
}

impl SimdVec<u16> {
    #[inline(always)]
    pub fn splat(val: u16) -> Self {
        unsafe {
            Self(
                Avx512Reg {
                    i: _mm512_set1_epi16(val as i16),
                },
                PhantomData,
            )
        }
    }
}

impl SimdVec<u8> {
    #[inline(always)]
    pub fn splat(val: u8) -> Self {
        unsafe {
            Self(
                Avx512Reg {
                    i: _mm512_set1_epi8(val as i8),
                },
                PhantomData,
            )
        }
    }
}

impl SimdVec<f32> {
    #[inline(always)]
    pub fn splat(val: f32) -> Self {
        unsafe {
            Self(
                Avx512Reg {
                    f: _mm512_set1_ps(val),
                },
                PhantomData,
            )
        }
    }
}

// ============================================================================
// u32 Implementation
// ============================================================================

impl Add for SimdVec<u32> {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        unsafe {
            Self(
                Avx512Reg {
                    i: _mm512_add_epi32(self.0.i, rhs.0.i),
                },
                PhantomData,
            )
        }
    }
}

impl Sub for SimdVec<u32> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        unsafe {
            Self(
                Avx512Reg {
                    i: _mm512_sub_epi32(self.0.i, rhs.0.i),
                },
                PhantomData,
            )
        }
    }
}

impl Mul for SimdVec<u32> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        unsafe {
            Self(
                Avx512Reg {
                    i: _mm512_mullo_epi32(self.0.i, rhs.0.i),
                },
                PhantomData,
            )
        }
    }
}

impl Div for SimdVec<u32> {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        unsafe {
            let mut a = [0u32; 16];
            let mut b = [0u32; 16];
            _mm512_storeu_si512(a.as_mut_ptr() as *mut _, self.0.i);
            _mm512_storeu_si512(b.as_mut_ptr() as *mut _, rhs.0.i);

            let mut res = [0u32; 16];
            for i in 0..16 {
                res[i] = if b[i] != 0 { a[i] / b[i] } else { 0 };
            }

            Self(
                Avx512Reg {
                    i: _mm512_loadu_si512(res.as_ptr() as *const _),
                },
                PhantomData,
            )
        }
    }
}

impl BitAnd for SimdVec<u32> {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        unsafe {
            Self(
                Avx512Reg {
                    i: _mm512_and_si512(self.0.i, rhs.0.i),
                },
                PhantomData,
            )
        }
    }
}

impl BitOr for SimdVec<u32> {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        unsafe {
            Self(
                Avx512Reg {
                    i: _mm512_or_si512(self.0.i, rhs.0.i),
                },
                PhantomData,
            )
        }
    }
}

impl BitXor for SimdVec<u32> {
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self {
        unsafe {
            Self(
                Avx512Reg {
                    i: _mm512_xor_si512(self.0.i, rhs.0.i),
                },
                PhantomData,
            )
        }
    }
}

impl Not for SimdVec<u32> {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        unsafe {
            let all_ones = _mm512_set1_epi32(-1);
            Self(
                Avx512Reg {
                    i: _mm512_xor_si512(self.0.i, all_ones),
                },
                PhantomData,
            )
        }
    }
}

impl Shl<i32> for SimdVec<u32> {
    type Output = Self;
    #[inline(always)]
    fn shl(self, rhs: i32) -> Self {
        unsafe {
            let count = _mm_cvtsi32_si128(rhs);
            Self(
                Avx512Reg {
                    i: _mm512_sll_epi32(self.0.i, count),
                },
                PhantomData,
            )
        }
    }
}

impl Shr<i32> for SimdVec<u32> {
    type Output = Self;
    #[inline(always)]
    fn shr(self, rhs: i32) -> Self {
        unsafe {
            let count = _mm_cvtsi32_si128(rhs);
            Self(
                Avx512Reg {
                    i: _mm512_srl_epi32(self.0.i, count),
                },
                PhantomData,
            )
        }
    }
}

// Generic SimdBatch implementation
impl<T: Copy + Debug + Default + Send + Sync + 'static> SimdBatch<T> for SimdVec<T> {
    fn splat(val: T) -> Self {
        unsafe {
            if core::mem::size_of::<T>() == 4 {
                let v: u32 = core::mem::transmute_copy(&val);
                Self(
                    Avx512Reg {
                        i: _mm512_set1_epi32(v as i32),
                    },
                    PhantomData,
                )
            } else if core::mem::size_of::<T>() == 2 {
                let v: u16 = core::mem::transmute_copy(&val);
                Self(
                    Avx512Reg {
                        i: _mm512_set1_epi16(v as i16),
                    },
                    PhantomData,
                )
            } else if core::mem::size_of::<T>() == 1 {
                let v: u8 = core::mem::transmute_copy(&val);
                Self(
                    Avx512Reg {
                        i: _mm512_set1_epi8(v as i8),
                    },
                    PhantomData,
                )
            } else {
                Self::default()
            }
        }
    }

    fn sequential_from(start: T) -> Self {
        unsafe {
            if core::mem::size_of::<T>() == 4 {
                let s: u32 = core::mem::transmute_copy(&start);
                let base = _mm512_set1_epi32(s as i32);
                let seq = _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
                Self(
                    Avx512Reg {
                        i: _mm512_add_epi32(base, seq),
                    },
                    PhantomData,
                )
            } else if core::mem::size_of::<T>() == 1 {
                let s: u8 = core::mem::transmute_copy(&start);
                let mut arr = [0u8; 16];
                for i in 0..16 {
                    arr[i] = s.wrapping_add(i as u8);
                }
                let val128 = _mm_loadu_si128(arr.as_ptr() as *const _);
                let val512 = _mm512_castsi128_si512(val128);
                 Self(
                    Avx512Reg {
                        i: val512,
                    },
                    PhantomData,
                )
            } else {
                Self::default()
            }
        }
    }

    #[inline(always)]
    fn load(slice: &[T]) -> Self {
        unsafe {
            if core::mem::size_of::<T>() == 4 {
                Self(
                    Avx512Reg {
                        i: _mm512_loadu_si512(slice.as_ptr() as *const _),
                    },
                    PhantomData,
                )
            } else if core::mem::size_of::<T>() == 2 {
                let ptr = slice.as_ptr() as *const _;
                let val256 = _mm256_loadu_si256(ptr);
                Self(
                    Avx512Reg {
                        i: _mm512_castsi256_si512(val256),
                    },
                    PhantomData,
                )
            } else if core::mem::size_of::<T>() == 1 {
                let ptr = slice.as_ptr() as *const _;
                let val128 = _mm_loadu_si128(ptr);
                Self(
                    Avx512Reg {
                        i: _mm512_castsi128_si512(val128),
                    },
                    PhantomData,
                )
            } else {
                Self::default()
            }
        }
    }

    #[inline(always)]
    fn store(&self, slice: &mut [T]) {
        unsafe {
            SimdVec::store(*self, slice.as_mut_ptr());
        }
    }

    fn first(&self) -> T {
         unsafe {
            if core::mem::size_of::<T>() == 4 {
                let val128 = _mm512_castsi512_si128(self.0.i);
                let v = _mm_cvtsi128_si32(val128);
                core::mem::transmute_copy(&v)
            } else if core::mem::size_of::<T>() == 2 {
                let val128 = _mm512_castsi512_si128(self.0.i);
                let v = _mm_extract_epi16(val128, 0);
                core::mem::transmute_copy(&(v as u16))
            } else if core::mem::size_of::<T>() == 1 {
                let val128 = _mm512_castsi512_si128(self.0.i);
                let v = _mm_extract_epi8(val128, 0);
                core::mem::transmute_copy(&(v as u8))
            } else {
                T::default()
            }
        }
    }
}

impl BatchArithmetic<u32> for SimdVec<u32> {
    fn select(self, if_true: Self, if_false: Self) -> Self {
        unsafe {
            Self(
                Avx512Reg {
                    i: _mm512_or_si512(
                        _mm512_and_si512(self.0.i, if_true.0.i),
                        _mm512_andnot_si512(self.0.i, if_false.0.i)
                    )
                },
                PhantomData
            )
        }
    }

    fn gather(base: &[u32], indices: Self) -> Self {
        unsafe {
            let res = _mm512_i32gather_epi32(indices.0.i, base.as_ptr() as *const _, 4);
            Self(Avx512Reg { i: res }, PhantomData)
        }
    }

    fn gather_u8(base: &[u8], indices: Self) -> Self {
         unsafe {
             let mut idx = [0u32; 16];
             _mm512_storeu_si512(idx.as_mut_ptr() as *mut _, indices.0.i);
             let mut res = [0u32; 16];
             for i in 0..16 {
                 res[i] = base.get(idx[i] as usize).copied().unwrap_or(0) as u32;
             }
             Self(
                 Avx512Reg {
                     i: _mm512_loadu_si512(res.as_ptr() as *const _),
                 },
                 PhantomData,
             )
         }
    }

    #[inline(always)]
    fn min(self, other: Self) -> Self {
        unsafe {
            Self(
                Avx512Reg {
                    i: _mm512_min_epu32(self.0.i, other.0.i),
                },
                PhantomData,
            )
        }
    }

    #[inline(always)]
    fn max(self, other: Self) -> Self {
        unsafe {
            Self(
                Avx512Reg {
                    i: _mm512_max_epu32(self.0.i, other.0.i),
                },
                PhantomData,
            )
        }
    }

    #[inline(always)]
    fn saturating_add(self, other: Self) -> Self {
         unsafe {
             let sum = _mm512_add_epi32(self.0.i, other.0.i);
             let mask = _mm512_cmp_epu32_mask(sum, self.0.i, _MM_CMPINT_LT);
             let all_ones = _mm512_set1_epi32(-1);
             let res = _mm512_mask_blend_epi32(mask, sum, all_ones);
             Self(Avx512Reg{ i: res }, PhantomData)
         }
    }

    #[inline(always)]
    fn saturating_sub(self, other: Self) -> Self {
        unsafe {
            let m = _mm512_max_epu32(self.0.i, other.0.i);
            Self(Avx512Reg { i: _mm512_sub_epi32(m, other.0.i) }, PhantomData)
        }
    }

    #[inline(always)]
    fn cmp_eq(self, other: Self) -> Self {
        unsafe {
            let k = _mm512_cmpeq_epi32_mask(self.0.i, other.0.i);
            Self(Avx512Reg { i: _mm512_movm_epi32(k) }, PhantomData)
        }
    }

    #[inline(always)]
    fn cmp_ne(self, other: Self) -> Self {
        unsafe {
             let k = _mm512_cmpneq_epi32_mask(self.0.i, other.0.i);
             Self(Avx512Reg { i: _mm512_movm_epi32(k) }, PhantomData)
        }
    }

    #[inline(always)]
    fn cmp_lt(self, other: Self) -> Self {
        unsafe {
            let k = _mm512_cmp_epu32_mask(self.0.i, other.0.i, _MM_CMPINT_LT);
            Self(Avx512Reg { i: _mm512_movm_epi32(k) }, PhantomData)
        }
    }

    #[inline(always)]
    fn cmp_le(self, other: Self) -> Self {
        unsafe {
            let k = _mm512_cmp_epu32_mask(self.0.i, other.0.i, _MM_CMPINT_LE);
            Self(Avx512Reg { i: _mm512_movm_epi32(k) }, PhantomData)
        }
    }

    #[inline(always)]
    fn cmp_gt(self, other: Self) -> Self {
        unsafe {
            let k = _mm512_cmp_epu32_mask(other.0.i, self.0.i, _MM_CMPINT_LT);
            Self(Avx512Reg { i: _mm512_movm_epi32(k) }, PhantomData)
        }
    }

    #[inline(always)]
    fn cmp_ge(self, other: Self) -> Self {
        unsafe {
            let k = _mm512_cmp_epu32_mask(other.0.i, self.0.i, _MM_CMPINT_LE);
            Self(Avx512Reg { i: _mm512_movm_epi32(k) }, PhantomData)
        }
    }
}

// ============================================================================
// u16 Implementation
// ============================================================================

impl Add for SimdVec<u16> {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        unsafe {
            Self(
                Avx512Reg {
                    i: _mm512_add_epi16(self.0.i, rhs.0.i),
                },
                PhantomData,
            )
        }
    }
}

impl Sub for SimdVec<u16> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        unsafe {
            Self(
                Avx512Reg {
                    i: _mm512_sub_epi16(self.0.i, rhs.0.i),
                },
                PhantomData,
            )
        }
    }
}

impl Mul for SimdVec<u16> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        unsafe {
            Self(
                Avx512Reg {
                    i: _mm512_mullo_epi16(self.0.i, rhs.0.i),
                },
                PhantomData,
            )
        }
    }
}

impl Div for SimdVec<u16> {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        unsafe {
            let val256_a = _mm512_castsi512_si256(self.0.i);
            let val256_b = _mm512_castsi512_si256(rhs.0.i);

            let mut arr_a = [0u16; 16];
            let mut arr_b = [0u16; 16];
            _mm256_storeu_si256(arr_a.as_mut_ptr() as *mut _, val256_a);
            _mm256_storeu_si256(arr_b.as_mut_ptr() as *mut _, val256_b);

            let mut res = [0u16; 16];
            for i in 0..16 {
                res[i] = if arr_b[i] != 0 { arr_a[i] / arr_b[i] } else { 0 };
            }

            let res256 = _mm256_loadu_si256(res.as_ptr() as *const _);
            Self(Avx512Reg { i: _mm512_castsi256_si512(res256) }, PhantomData)
        }
    }
}

impl BitAnd for SimdVec<u16> {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        unsafe {
            Self(
                Avx512Reg {
                    i: _mm512_and_si512(self.0.i, rhs.0.i),
                },
                PhantomData,
            )
        }
    }
}

impl BitOr for SimdVec<u16> {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        unsafe {
            Self(
                Avx512Reg {
                    i: _mm512_or_si512(self.0.i, rhs.0.i),
                },
                PhantomData,
            )
        }
    }
}

impl BitXor for SimdVec<u16> {
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self {
        unsafe {
            Self(
                Avx512Reg {
                    i: _mm512_xor_si512(self.0.i, rhs.0.i),
                },
                PhantomData,
            )
        }
    }
}

impl Not for SimdVec<u16> {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        unsafe {
             let all_ones = _mm512_set1_epi16(-1);
             Self(
                 Avx512Reg {
                     i: _mm512_xor_si512(self.0.i, all_ones),
                 },
                 PhantomData,
             )
        }
    }
}

impl Shl<i32> for SimdVec<u16> {
    type Output = Self;
    #[inline(always)]
    fn shl(self, rhs: i32) -> Self {
        unsafe {
            let count = _mm_cvtsi32_si128(rhs);
            Self(
                Avx512Reg {
                    i: _mm512_sll_epi16(self.0.i, count),
                },
                PhantomData,
            )
        }
    }
}

impl Shr<i32> for SimdVec<u16> {
    type Output = Self;
    #[inline(always)]
    fn shr(self, rhs: i32) -> Self {
        unsafe {
            let count = _mm_cvtsi32_si128(rhs);
            Self(
                Avx512Reg {
                    i: _mm512_srl_epi16(self.0.i, count),
                },
                PhantomData,
            )
        }
    }
}

impl BatchArithmetic<u16> for SimdVec<u16> {
    fn select(self, if_true: Self, if_false: Self) -> Self {
        unsafe {
            Self(
                Avx512Reg {
                    i: _mm512_or_si512(
                        _mm512_and_si512(self.0.i, if_true.0.i),
                        _mm512_andnot_si512(self.0.i, if_false.0.i)
                    )
                },
                PhantomData
            )
        }
    }

    fn gather(_base: &[u16], _indices: Self) -> Self {
        unsafe {
            // Scalar fallback
            let val256 = _mm512_castsi512_si256(_indices.0.i);
            let mut idx = [0u16; 16];
            _mm256_storeu_si256(idx.as_mut_ptr() as *mut _, val256);

            let mut res = [0u16; 16];
            for i in 0..16 {
                res[i] = _base.get(idx[i] as usize).copied().unwrap_or(0);
            }

            let res256 = _mm256_loadu_si256(res.as_ptr() as *const _);
            Self(Avx512Reg { i: _mm512_castsi256_si512(res256) }, PhantomData)
        }
    }

    fn min(self, other: Self) -> Self {
        unsafe {
            Self(
                Avx512Reg {
                    i: _mm512_min_epu16(self.0.i, other.0.i),
                },
                PhantomData,
            )
        }
    }

    fn max(self, other: Self) -> Self {
        unsafe {
            Self(
                Avx512Reg {
                    i: _mm512_max_epu16(self.0.i, other.0.i),
                },
                PhantomData,
            )
        }
    }

    fn saturating_add(self, other: Self) -> Self {
        unsafe {
            Self(
                Avx512Reg {
                    i: _mm512_adds_epu16(self.0.i, other.0.i),
                },
                PhantomData,
            )
        }
    }

    fn saturating_sub(self, other: Self) -> Self {
        unsafe {
            Self(
                Avx512Reg {
                    i: _mm512_subs_epu16(self.0.i, other.0.i),
                },
                PhantomData,
            )
        }
    }

    fn cmp_eq(self, other: Self) -> Self {
        unsafe {
            let k = _mm512_cmpeq_epi16_mask(self.0.i, other.0.i);
            Self(Avx512Reg { i: _mm512_movm_epi16(k) }, PhantomData)
        }
    }

    fn cmp_ne(self, other: Self) -> Self {
        unsafe {
            let k = _mm512_cmpneq_epi16_mask(self.0.i, other.0.i);
            Self(Avx512Reg { i: _mm512_movm_epi16(k) }, PhantomData)
        }
    }

    fn cmp_lt(self, other: Self) -> Self {
        unsafe {
            let k = _mm512_cmp_epu16_mask(self.0.i, other.0.i, _MM_CMPINT_LT);
            Self(Avx512Reg { i: _mm512_movm_epi16(k) }, PhantomData)
        }
    }

    fn cmp_le(self, other: Self) -> Self {
        unsafe {
            let k = _mm512_cmp_epu16_mask(self.0.i, other.0.i, _MM_CMPINT_LE);
            Self(Avx512Reg { i: _mm512_movm_epi16(k) }, PhantomData)
        }
    }

    fn cmp_gt(self, other: Self) -> Self {
        unsafe {
            let k = _mm512_cmp_epu16_mask(other.0.i, self.0.i, _MM_CMPINT_LT);
            Self(Avx512Reg { i: _mm512_movm_epi16(k) }, PhantomData)
        }
    }

    fn cmp_ge(self, other: Self) -> Self {
        unsafe {
            let k = _mm512_cmp_epu16_mask(other.0.i, self.0.i, _MM_CMPINT_LE);
            Self(Avx512Reg { i: _mm512_movm_epi16(k) }, PhantomData)
        }
    }
}

// ============================================================================
// u8 Implementation
// ============================================================================

impl Add for SimdVec<u8> {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        unsafe {
            Self(
                Avx512Reg {
                    i: _mm512_add_epi8(self.0.i, rhs.0.i),
                },
                PhantomData,
            )
        }
    }
}

impl Sub for SimdVec<u8> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        unsafe {
            Self(
                Avx512Reg {
                    i: _mm512_sub_epi8(self.0.i, rhs.0.i),
                },
                PhantomData,
            )
        }
    }
}

impl Mul for SimdVec<u8> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        unsafe {
             let self128 = _mm512_castsi512_si128(self.0.i);
             let rhs128 = _mm512_castsi512_si128(rhs.0.i);
             let self256 = _mm256_castsi128_si256(self128);
             let rhs256 = _mm256_castsi128_si256(rhs128);

             let a_512 = _mm512_cvtepu8_epi16(self256);
             let b_512 = _mm512_cvtepu8_epi16(rhs256);

             let res_512 = _mm512_mullo_epi16(a_512, b_512);

             let res_256_out = _mm512_cvtepi16_epi8(res_512);

             let res_128 = _mm256_castsi256_si128(res_256_out);

             Self(Avx512Reg { i: _mm512_castsi128_si512(res_128) }, PhantomData)
        }
    }
}

impl Div for SimdVec<u8> {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        unsafe {
            let val128_a = _mm512_castsi512_si128(self.0.i);
            let val128_b = _mm512_castsi512_si128(rhs.0.i);

            let mut arr_a = [0u8; 16];
            let mut arr_b = [0u8; 16];
            _mm_storeu_si128(arr_a.as_mut_ptr() as *mut _, val128_a);
            _mm_storeu_si128(arr_b.as_mut_ptr() as *mut _, val128_b);

            let mut res = [0u8; 16];
            for i in 0..16 {
                res[i] = if arr_b[i] != 0 { arr_a[i] / arr_b[i] } else { 0 };
            }

            let res128 = _mm_loadu_si128(res.as_ptr() as *const _);
            Self(Avx512Reg { i: _mm512_castsi128_si512(res128) }, PhantomData)
        }
    }
}

impl BitAnd for SimdVec<u8> {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        unsafe {
            Self(
                Avx512Reg {
                    i: _mm512_and_si512(self.0.i, rhs.0.i),
                },
                PhantomData,
            )
        }
    }
}

impl BitOr for SimdVec<u8> {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        unsafe {
            Self(
                Avx512Reg {
                    i: _mm512_or_si512(self.0.i, rhs.0.i),
                },
                PhantomData,
            )
        }
    }
}

impl BitXor for SimdVec<u8> {
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self {
        unsafe {
            Self(
                Avx512Reg {
                    i: _mm512_xor_si512(self.0.i, rhs.0.i),
                },
                PhantomData,
            )
        }
    }
}

impl Not for SimdVec<u8> {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        unsafe {
            let all_ones = _mm512_set1_epi8(-1);
            Self(
                Avx512Reg {
                    i: _mm512_xor_si512(self.0.i, all_ones),
                },
                PhantomData,
            )
        }
    }
}

impl Shl<i32> for SimdVec<u8> {
    type Output = Self;
    fn shl(self, rhs: i32) -> Self {
        unsafe {
            let val128 = _mm512_castsi512_si128(self.0.i);
            let mut arr = [0u8; 16];
            _mm_storeu_si128(arr.as_mut_ptr() as *mut _, val128);
            for i in 0..16 {
                arr[i] = arr[i].wrapping_shl(rhs as u32);
            }
            let res128 = _mm_loadu_si128(arr.as_ptr() as *const _);
            Self(Avx512Reg { i: _mm512_castsi128_si512(res128) }, PhantomData)
        }
    }
}

impl Shr<i32> for SimdVec<u8> {
    type Output = Self;
    fn shr(self, rhs: i32) -> Self {
        unsafe {
            let val128 = _mm512_castsi512_si128(self.0.i);
            let mut arr = [0u8; 16];
            _mm_storeu_si128(arr.as_mut_ptr() as *mut _, val128);
            for i in 0..16 {
                arr[i] = arr[i].wrapping_shr(rhs as u32);
            }
            let res128 = _mm_loadu_si128(arr.as_ptr() as *const _);
            Self(Avx512Reg { i: _mm512_castsi128_si512(res128) }, PhantomData)
        }
    }
}

impl BatchArithmetic<u8> for SimdVec<u8> {
    fn select(self, if_true: Self, if_false: Self) -> Self {
        unsafe {
            Self(
                Avx512Reg {
                    i: _mm512_or_si512(
                        _mm512_and_si512(self.0.i, if_true.0.i),
                        _mm512_andnot_si512(self.0.i, if_false.0.i)
                    )
                },
                PhantomData
            )
        }
    }

    fn gather(base: &[u8], indices: Self) -> Self {
         unsafe {
             let mut idx = [0u8; 16];
             // Store low 16 bytes
             let val128 = _mm512_castsi512_si128(indices.0.i);
             _mm_storeu_si128(idx.as_mut_ptr() as *mut _, val128);

             let mut res = [0u8; 16];
             for i in 0..16 {
                 res[i] = base.get(idx[i] as usize).copied().unwrap_or(0);
             }

             let res128 = _mm_loadu_si128(res.as_ptr() as *const _);
             Self(
                 Avx512Reg {
                     i: _mm512_castsi128_si512(res128),
                 },
                 PhantomData,
             )
         }
    }

    fn min(self, other: Self) -> Self {
        unsafe {
            Self(
                Avx512Reg {
                    i: _mm512_min_epu8(self.0.i, other.0.i),
                },
                PhantomData,
            )
        }
    }

    fn max(self, other: Self) -> Self {
        unsafe {
            Self(
                Avx512Reg {
                    i: _mm512_max_epu8(self.0.i, other.0.i),
                },
                PhantomData,
            )
        }
    }

    fn saturating_add(self, other: Self) -> Self {
        unsafe {
            Self(
                Avx512Reg {
                    i: _mm512_adds_epu8(self.0.i, other.0.i),
                },
                PhantomData,
            )
        }
    }

    fn saturating_sub(self, other: Self) -> Self {
        unsafe {
            Self(
                Avx512Reg {
                    i: _mm512_subs_epu8(self.0.i, other.0.i),
                },
                PhantomData,
            )
        }
    }

    fn cmp_eq(self, other: Self) -> Self {
        unsafe {
            // Compare all 64 bytes
            let k = _mm512_cmpeq_epi8_mask(self.0.i, other.0.i);
            // Move mask back to bytes (0 or FF)
            Self(Avx512Reg { i: _mm512_movm_epi8(k) }, PhantomData)
        }
    }

    fn cmp_ne(self, other: Self) -> Self {
        unsafe {
            let k = _mm512_cmpneq_epi8_mask(self.0.i, other.0.i);
            Self(Avx512Reg { i: _mm512_movm_epi8(k) }, PhantomData)
        }
    }

    fn cmp_lt(self, other: Self) -> Self {
        unsafe {
            let k = _mm512_cmp_epu8_mask(self.0.i, other.0.i, _MM_CMPINT_LT);
            Self(Avx512Reg { i: _mm512_movm_epi8(k) }, PhantomData)
        }
    }

    fn cmp_le(self, other: Self) -> Self {
        unsafe {
            let k = _mm512_cmp_epu8_mask(self.0.i, other.0.i, _MM_CMPINT_LE);
            Self(Avx512Reg { i: _mm512_movm_epi8(k) }, PhantomData)
        }
    }

    fn cmp_gt(self, other: Self) -> Self {
        unsafe {
            let k = _mm512_cmp_epu8_mask(other.0.i, self.0.i, _MM_CMPINT_LT);
            Self(Avx512Reg { i: _mm512_movm_epi8(k) }, PhantomData)
        }
    }

    fn cmp_ge(self, other: Self) -> Self {
        unsafe {
            let k = _mm512_cmp_epu8_mask(other.0.i, self.0.i, _MM_CMPINT_LE);
            Self(Avx512Reg { i: _mm512_movm_epi8(k) }, PhantomData)
        }
    }
}

// ============================================================================
// f32 Implementation
// ============================================================================

impl Add for SimdVec<f32> {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        unsafe {
            Self(
                Avx512Reg {
                    f: _mm512_add_ps(self.0.f, rhs.0.f),
                },
                PhantomData,
            )
        }
    }
}

impl Sub for SimdVec<f32> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        unsafe {
            Self(
                Avx512Reg {
                    f: _mm512_sub_ps(self.0.f, rhs.0.f),
                },
                PhantomData,
            )
        }
    }
}

impl Mul for SimdVec<f32> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        unsafe {
            Self(
                Avx512Reg {
                    f: _mm512_mul_ps(self.0.f, rhs.0.f),
                },
                PhantomData,
            )
        }
    }
}

impl Div for SimdVec<f32> {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        unsafe {
            Self(
                Avx512Reg {
                    f: _mm512_div_ps(self.0.f, rhs.0.f),
                },
                PhantomData,
            )
        }
    }
}

impl BitAnd for SimdVec<f32> {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        unsafe {
            let a = _mm512_castps_si512(self.0.f);
            let b = _mm512_castps_si512(rhs.0.f);
            Self(
                Avx512Reg {
                    f: _mm512_castsi512_ps(_mm512_and_si512(a, b)),
                },
                PhantomData,
            )
        }
    }
}

impl BitOr for SimdVec<f32> {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        unsafe {
            let a = _mm512_castps_si512(self.0.f);
            let b = _mm512_castps_si512(rhs.0.f);
            Self(
                Avx512Reg {
                    f: _mm512_castsi512_ps(_mm512_or_si512(a, b)),
                },
                PhantomData,
            )
        }
    }
}

impl BitXor for SimdVec<f32> {
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self {
        unsafe {
            let a = _mm512_castps_si512(self.0.f);
            let b = _mm512_castps_si512(rhs.0.f);
            Self(
                Avx512Reg {
                    f: _mm512_castsi512_ps(_mm512_xor_si512(a, b)),
                },
                PhantomData,
            )
        }
    }
}

impl Not for SimdVec<f32> {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        unsafe {
            let a = _mm512_castps_si512(self.0.f);
            let all_ones = _mm512_set1_epi32(-1);
            Self(
                Avx512Reg {
                    f: _mm512_castsi512_ps(_mm512_xor_si512(a, all_ones)),
                },
                PhantomData,
            )
        }
    }
}

impl Shl<i32> for SimdVec<f32> {
    type Output = Self;
    fn shl(self, _rhs: i32) -> Self {
        unimplemented!()
    }
}

impl Shr<i32> for SimdVec<f32> {
    type Output = Self;
    fn shr(self, _rhs: i32) -> Self {
        unimplemented!()
    }
}

impl BatchArithmetic<f32> for SimdVec<f32> {
    fn select(self, if_true: Self, if_false: Self) -> Self {
        unsafe {
            let mask = _mm512_castps_si512(self.0.f);
            let t = _mm512_castps_si512(if_true.0.f);
            let f = _mm512_castps_si512(if_false.0.f);
            let res = _mm512_or_si512(
                _mm512_and_si512(mask, t),
                _mm512_andnot_si512(mask, f)
            );
            Self(
                Avx512Reg {
                    f: _mm512_castsi512_ps(res),
                },
                PhantomData,
            )
        }
    }

    fn gather(_base: &[f32], _indices: Self) -> Self {
        unsafe {
            // Convert indices (f32) to i32 (trunc)
            let idx = _mm512_cvttps_epi32(_indices.0.f);
            let res = _mm512_i32gather_ps(idx, _base.as_ptr() as *const _, 4);
            Self(Avx512Reg { f: res }, PhantomData)
        }
    }

    fn min(self, other: Self) -> Self {
        unsafe {
            Self(
                Avx512Reg {
                    f: _mm512_min_ps(self.0.f, other.0.f),
                },
                PhantomData,
            )
        }
    }

    fn max(self, other: Self) -> Self {
        unsafe {
            Self(
                Avx512Reg {
                    f: _mm512_max_ps(self.0.f, other.0.f),
                },
                PhantomData,
            )
        }
    }

    fn saturating_add(self, other: Self) -> Self {
        self + other
    }
    fn saturating_sub(self, other: Self) -> Self {
        self - other
    }

    fn cmp_eq(self, other: Self) -> Self {
        unsafe {
            let k = _mm512_cmp_ps_mask(self.0.f, other.0.f, _CMP_EQ_OQ);
            let m = _mm512_movm_epi32(k);
            Self(
                Avx512Reg {
                    f: _mm512_castsi512_ps(m),
                },
                PhantomData,
            )
        }
    }

    fn cmp_ne(self, other: Self) -> Self {
        unsafe {
            let k = _mm512_cmp_ps_mask(self.0.f, other.0.f, _CMP_NEQ_OQ);
            let m = _mm512_movm_epi32(k);
            Self(
                Avx512Reg {
                    f: _mm512_castsi512_ps(m),
                },
                PhantomData,
            )
        }
    }

    fn cmp_lt(self, other: Self) -> Self {
        unsafe {
            let k = _mm512_cmp_ps_mask(self.0.f, other.0.f, _CMP_LT_OQ);
            let m = _mm512_movm_epi32(k);
            Self(
                Avx512Reg {
                    f: _mm512_castsi512_ps(m),
                },
                PhantomData,
            )
        }
    }

    fn cmp_le(self, other: Self) -> Self {
        unsafe {
            let k = _mm512_cmp_ps_mask(self.0.f, other.0.f, _CMP_LE_OQ);
            let m = _mm512_movm_epi32(k);
            Self(
                Avx512Reg {
                    f: _mm512_castsi512_ps(m),
                },
                PhantomData,
            )
        }
    }

    fn cmp_gt(self, other: Self) -> Self {
        unsafe {
            let k = _mm512_cmp_ps_mask(self.0.f, other.0.f, _CMP_GT_OQ);
            let m = _mm512_movm_epi32(k);
            Self(
                Avx512Reg {
                    f: _mm512_castsi512_ps(m),
                },
                PhantomData,
            )
        }
    }

    fn cmp_ge(self, other: Self) -> Self {
        unsafe {
            let k = _mm512_cmp_ps_mask(self.0.f, other.0.f, _CMP_GE_OQ);
            let m = _mm512_movm_epi32(k);
            Self(
                Avx512Reg {
                    f: _mm512_castsi512_ps(m),
                },
                PhantomData,
            )
        }
    }
}

impl FloatBatchOps for SimdVec<f32> {
    #[inline(always)]
    fn sqrt(self) -> Self {
        unsafe {
            Self(
                Avx512Reg {
                    f: _mm512_sqrt_ps(self.0.f),
                },
                PhantomData,
            )
        }
    }

    #[inline(always)]
    fn abs(self) -> Self {
        unsafe {
            Self(
                Avx512Reg {
                    f: _mm512_abs_ps(self.0.f),
                },
                PhantomData,
            )
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use crate::backend::SimdBatch;

    #[test]
    fn test_u32_ops() {
        if !std::is_x86_feature_detected!("avx512f") { return; }
        let a = SimdVec::<u32>::splat(100);
        let b = SimdVec::<u32>::splat(50);
        let c = a + b;
        let d = a - b;
        let e = a * b;
        let f = a / b;

        let mut out_add = [0u32; 16];
        let mut out_sub = [0u32; 16];
        let mut out_mul = [0u32; 16];
        let mut out_div = [0u32; 16];
        SimdBatch::store(&c, &mut out_add);
        SimdBatch::store(&d, &mut out_sub);
        SimdBatch::store(&e, &mut out_mul);
        SimdBatch::store(&f, &mut out_div);
        assert_eq!(out_add, [150; 16]);
        assert_eq!(out_sub, [50; 16]);
        assert_eq!(out_mul, [5000; 16]);
        assert_eq!(out_div, [2; 16]);
    }

    #[test]
    fn test_f32_ops() {
        if !std::is_x86_feature_detected!("avx512f") { return; }
        let a = SimdVec::<f32>::splat(4.0);
        let b = a.sqrt();
        let c = a + b; // 4+2=6
        let d = a / b; // 4/2=2

        let mut out_sqrt = [0.0f32; 16];
        let mut out_add = [0.0f32; 16];
        let mut out_div = [0.0f32; 16];
        SimdBatch::store(&b, &mut out_sqrt);
        SimdBatch::store(&c, &mut out_add);
        SimdBatch::store(&d, &mut out_div);
        assert_eq!(out_sqrt, [2.0; 16]);
        assert_eq!(out_add, [6.0; 16]);
        assert_eq!(out_div, [2.0; 16]);
    }

    #[test]
    fn test_u16_ops() {
        if !std::is_x86_feature_detected!("avx512f") { return; }
        let a = SimdVec::<u16>::splat(20);
        let b = SimdVec::<u16>::splat(10);
        let c = a + b;
        let d = a - b;
        let e = a * b;
        let f = a / b;
        let g = a << 1;
        let h = a >> 1;

        let mut out = [0u16; 16];
        SimdBatch::store(&c, &mut out); assert_eq!(out, [30; 16]);
        SimdBatch::store(&d, &mut out); assert_eq!(out, [10; 16]);
        SimdBatch::store(&e, &mut out); assert_eq!(out, [200; 16]);
        SimdBatch::store(&f, &mut out); assert_eq!(out, [2; 16]);
        SimdBatch::store(&g, &mut out); assert_eq!(out, [40; 16]);
        SimdBatch::store(&h, &mut out); assert_eq!(out, [10; 16]);
    }

    #[test]
    fn test_u8_ops() {
        if !std::is_x86_feature_detected!("avx512f") { return; }
        let a = SimdVec::<u8>::splat(20);
        let b = SimdVec::<u8>::splat(10);
        let c = a + b;
        let d = a - b;
        let e = a * b;
        let f = a / b;
        let g = a << 1;
        let h = a >> 1;

        let mut out = [0u8; 16];
        SimdBatch::store(&c, &mut out); assert_eq!(out, [30; 16]);
        SimdBatch::store(&d, &mut out); assert_eq!(out, [10; 16]);
        SimdBatch::store(&e, &mut out); assert_eq!(out, [200; 16]);
        SimdBatch::store(&f, &mut out); assert_eq!(out, [2; 16]);
        SimdBatch::store(&g, &mut out); assert_eq!(out, [40; 16]);
        SimdBatch::store(&h, &mut out); assert_eq!(out, [10; 16]);
    }

    #[test]
    fn test_gather_u32() {
        if !std::is_x86_feature_detected!("avx512f") { return; }
        let data = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160];
        let idx = <SimdVec<u32> as SimdBatch<u32>>::sequential_from(0);
        let res = <SimdVec<u32> as BatchArithmetic<u32>>::gather(&data, idx);
        let mut out = [0u32; 16];
        SimdBatch::store(&res, &mut out);
        assert_eq!(out, data);
    }

    #[test]
    fn test_sequential_from() {
        if !std::is_x86_feature_detected!("avx512f") { return; }
        let seq = <SimdVec<u32> as SimdBatch<u32>>::sequential_from(10);
        let mut output = [0u32; 16];
        SimdBatch::store(&seq, &mut output);
        assert_eq!(output, [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]);
    }

    #[test]
    fn test_u8_sequential_from() {
        if !std::is_x86_feature_detected!("avx512f") { return; }
        let seq = <SimdVec<u8> as SimdBatch<u8>>::sequential_from(10);
        let mut output = [0u8; 16];
        SimdBatch::store(&seq, &mut output);
        assert_eq!(output, [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]);
    }

    #[test]
    fn test_cmp_lt_u32() {
        if !std::is_x86_feature_detected!("avx512f") { return; }
        let a = <SimdVec<u32> as SimdBatch<u32>>::sequential_from(0);
        let b = SimdVec::<u32>::splat(10);
        let c = a.cmp_lt(b);

        let mut output = [0u32; 16];
        SimdBatch::store(&c, &mut output);
        for i in 0..16 {
            if i < 10 {
                assert_eq!(output[i], 0xFFFFFFFF, "index {}", i);
            } else {
                assert_eq!(output[i], 0, "index {}", i);
            }
        }
    }
}
