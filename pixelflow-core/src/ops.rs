use crate::TensorView;
use crate::simd::Simd;
use crate::pipe::Surface;

// --- 1. Sources ---

/// A source surface that samples from a texture atlas.
#[derive(Copy, Clone)]
pub struct SampleAtlas<'a> {
    /// The source texture atlas.
    pub atlas: TensorView<'a, u8>,
    /// The horizontal step size in 16.16 fixed point format.
    pub step_x_fp: u32,
    /// The vertical step size in 16.16 fixed point format.
    pub step_y_fp: u32,
}

impl<'a, V: Simd> Surface<V, u8> for SampleAtlas<'a> {
    #[inline(always)]
    fn eval(&self, x: V::Cast<u32>, y: V::Cast<u32>) -> V::Cast<u8> {
        let u = x * <V::Cast<u32> as Simd>::splat(self.step_x_fp);
        let v = y * <V::Cast<u32> as Simd>::splat(self.step_y_fp);
        let res = unsafe { self.atlas.sample_4bit_bilinear::<V>(u, v) };
        // Force cast u32 -> u8 (truncation/packing logic handled by Simd implementation or we assume result fits)
        unsafe {
            let tmp = res.cast::<u8>();
            core::ptr::read(&tmp as *const _ as *const V::Cast<u8>)
        }
    }
}

// --- 2. Transformers ---

#[derive(Copy, Clone)]
pub struct Offset<S> {
    pub source: S,
    pub dx: i32,
    pub dy: i32,
}

impl<V: Simd, T: Copy, S: Surface<V, T>> Surface<V, T> for Offset<S> {
    #[inline(always)]
    fn eval(&self, x: V::Cast<u32>, y: V::Cast<u32>) -> V::Cast<T> {
        let ox = <V::Cast<u32> as Simd>::splat(self.dx as u32);
        let oy = <V::Cast<u32> as Simd>::splat(self.dy as u32);
        self.source.eval(x + ox, y + oy)
    }
}

#[derive(Copy, Clone)]
pub struct Skew<S> {
    pub source: S,
    pub shear: i32,
}

impl<V: Simd, S: Surface<V, u8>> Surface<V, u8> for Skew<S> {
    #[inline(always)]
    fn eval(&self, x: V::Cast<u32>, y: V::Cast<u32>) -> V::Cast<u8> {
        let offset = (y * <V::Cast<u32> as Simd>::splat(self.shear as u32)) >> 8;
        self.source.eval(x.saturating_sub(offset), y)
    }
}

#[derive(Copy, Clone)]
pub struct Max<A, B>(pub A, pub B);

impl<V: Simd, T: Copy, A: Surface<V, T>, B: Surface<V, T>> Surface<V, T> for Max<A, B> {
    #[inline(always)]
    fn eval(&self, x: V::Cast<u32>, y: V::Cast<u32>) -> V::Cast<T> {
        self.0.eval(x, y).max(self.1.eval(x, y))
    }
}

// --- 3. Finalizers (Blend) ---

#[derive(Copy, Clone)]
pub struct Over<M, F, B> {
    pub mask: M,
    pub fg: F,
    pub bg: B,
}

#[inline(always)]
fn blend_math<V: Simd>(fg: V::Cast<u32>, bg: V::Cast<u32>, alpha: V::Cast<u32>) -> V::Cast<u32> {
    let inv_alpha = <V::Cast<u32> as Simd>::splat(256) - alpha;
    ((fg * alpha) + (bg * inv_alpha)) >> 8
}

impl<V: Simd, M, F, B> Surface<V, u32> for Over<M, F, B>
where
    M: Surface<V, u8>,
    F: Surface<V, u32>,
    B: Surface<V, u32>,
{
    #[inline(always)]
    fn eval(&self, x: V::Cast<u32>, y: V::Cast<u32>) -> V::Cast<u32> {
        let alpha_val = self.mask.eval(x, y);
        let a = unsafe {
            let tmp = alpha_val.cast::<u32>();
            core::ptr::read(&tmp as *const _ as *const V::Cast<u32>)
        };

        let alpha_broadcast = a * <V::Cast<u32> as Simd>::splat(0x01010101);

        let fg = self.fg.eval(x, y);
        let bg = self.bg.eval(x, y);

        let mask_8 = <V::Cast<u32> as Simd>::splat(0xFF);

        let r = blend_math::<V>(fg & mask_8, bg & mask_8, alpha_broadcast & mask_8);
        let g = blend_math::<V>(
            (fg >> 8) & mask_8,
            (bg >> 8) & mask_8,
            (alpha_broadcast >> 8) & mask_8,
        );
        let b = blend_math::<V>(
            (fg >> 16) & mask_8,
            (bg >> 16) & mask_8,
            (alpha_broadcast >> 16) & mask_8,
        );
        let a = blend_math::<V>(
            (fg >> 24) & mask_8,
            (bg >> 24) & mask_8,
            (alpha_broadcast >> 24) & mask_8,
        );

        r | (g << 8) | (b << 16) | (a << 24)
    }
}
