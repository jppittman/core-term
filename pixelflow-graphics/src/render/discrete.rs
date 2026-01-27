//! Discrete SIMD batch of packed pixels.

use pixelflow_core::Field;
use pixelflow_core::backend::{Backend, SimdOps, SimdU32Ops};
use pixelflow_core::backend;

// Selection logic matching pixelflow-core
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
type NativeU32Simd = <backend::x86::Avx512 as Backend>::U32;

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    not(target_feature = "avx512f")
))]
type NativeU32Simd = <backend::x86::Avx2 as Backend>::U32;

#[cfg(all(
    target_arch = "x86_64",
    not(target_feature = "avx512f"),
    not(target_feature = "avx2")
))]
type NativeU32Simd = <backend::x86::Sse2 as Backend>::U32;

#[cfg(target_arch = "aarch64")]
type NativeU32Simd = <backend::arm::Neon as Backend>::U32;

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
type NativeU32Simd = <backend::scalar::Scalar as Backend>::U32;

/// SIMD batch of packed RGBA pixels.
#[derive(Copy, Clone, Debug, Default)]
#[repr(transparent)]
pub struct Discrete(pub(crate) NativeU32Simd);

impl Discrete {
    /// Store packed pixels to a slice.
    #[inline(always)]
    pub fn store(&self, out: &mut [u32]) {
        self.0.store(out)
    }

    /// Pack 4 Fields (RGBA, 0.0-1.0) into packed u32 pixels.
    #[inline(always)]
    pub fn pack(r: Field, g: Field, b: Field, a: Field) -> Self {
        let r_u32 = NativeU32Simd::from_f32_scaled(r.0);
        let g_u32 = NativeU32Simd::from_f32_scaled(g.0);
        let b_u32 = NativeU32Simd::from_f32_scaled(b.0);
        let a_u32 = NativeU32Simd::from_f32_scaled(a.0);

        use core::ops::{BitOr, Shl};
        let g_sh = g_u32.shl(8);
        let b_sh = b_u32.shl(16);
        let a_sh = a_u32.shl(24);

        let packed = r_u32.bitor(g_sh).bitor(b_sh).bitor(a_sh);
        Self(packed)
    }

    /// Select between two discrete batches based on a mask Field.
    #[inline(always)]
    pub fn select(mask: Field, if_true: Self, if_false: Self) -> Self {
        <Self as pixelflow_core::Selectable>::select_raw(mask, if_true, if_false)
    }
}

impl core::ops::BitAnd for Discrete {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self { Self(self.0.bitand(rhs.0)) }
}

impl core::ops::BitOr for Discrete {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self { Self(self.0.bitor(rhs.0)) }
}

impl core::ops::Not for Discrete {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self { Self(self.0.not()) }
}

impl pixelflow_core::Computational for Discrete {
    #[inline(always)]
    fn from_f32(val: f32) -> Self {
        Self(NativeU32Simd::splat(val as u32))
    }

    #[inline(always)]
    fn sequential(start: f32) -> Self {
        Self::from_f32(start)
    }
}

impl pixelflow_core::Selectable for Discrete {
    #[inline(always)]
    fn select_raw(mask: Field, if_true: Self, if_false: Self) -> Self {
        use core::ops::{BitAnd, BitOr, Not};
        let mask_bits: NativeU32Simd = unsafe { core::mem::transmute(mask.0) };
        let t = if_true.0.bitand(mask_bits);
        let f = if_false.0.bitand(mask_bits.not());
        Self(t.bitor(f))
    }
}

/// Materialize a discrete color manifold into packed u32 pixels.
#[inline(always)]
pub fn materialize_discrete<M>(m: &M, x: f32, y: f32, out: &mut [u32])
where
    M: pixelflow_core::ManifoldCompat<Field, Output = Discrete> + ?Sized,
{
    let xs = Field::sequential(x);
    let discrete = m.eval_raw(xs, Field::from(y), Field::from(0.0), Field::from(0.0));
    discrete.store(out);
}