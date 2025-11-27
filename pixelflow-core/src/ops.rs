use crate::pipe::Surface;
use crate::batch::{Batch, SimdOps, SimdVec}; // Corrected import
use crate::TensorView;

// --- 1. Sources ---

#[derive(Copy, Clone)]
pub struct SampleAtlas<'a> {
    pub atlas: TensorView<'a, u8>,
}

impl<'a> Surface<u8> for SampleAtlas<'a> {
    #[inline(always)]
    fn eval(&self, x: Batch<u32>, y: Batch<u32>) -> Batch<u8> {
        // Direct bilinear sample
        // Convert integer coordinates to 16.16 fixed point format expected by sample_4bit_bilinear.
        unsafe { self.atlas.sample_4bit_bilinear(x << 16, y << 16).cast() }
    }
}

// --- 2. Transformers ---

#[derive(Copy, Clone)]
pub struct Offset<S> {
    pub source: S,
    pub dx: i32,
    pub dy: i32,
}

impl<T: Copy, S: Surface<T>> Surface<T> for Offset<S> {
    #[inline(always)]
    fn eval(&self, x: Batch<u32>, y: Batch<u32>) -> Batch<T> {
        let ox = Batch::splat(self.dx as u32);
        let oy = Batch::splat(self.dy as u32);
        
        self.source.eval(x + ox, y + oy)
    }
}

#[derive(Copy, Clone)]
pub struct Skew<S> {
    pub source: S,
    pub shear: i32,
}

impl<S: Surface<u8>> Surface<u8> for Skew<S> {
    #[inline(always)]
    fn eval(&self, x: Batch<u32>, y: Batch<u32>) -> Batch<u8> {
        let offset = (y * Batch::splat(self.shear as u32)) >> 8;
        self.source.eval(x.saturating_sub(offset), y)
    }
}

#[derive(Copy, Clone)]
pub struct Max<A, B>(pub A, pub B);

// Fix: Added trait bound for SimdOps so that Batch<T>::max is available
impl<T: Copy, A: Surface<T>, B: Surface<T>> Surface<T> for Max<A, B> 
where SimdVec<T>: SimdOps<T>
{
    #[inline(always)]
    fn eval(&self, x: Batch<u32>, y: Batch<u32>) -> Batch<T> {
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
fn blend_math(fg: Batch<u32>, bg: Batch<u32>, alpha: Batch<u32>) -> Batch<u32> {
    let inv_alpha = Batch::splat(256) - alpha;
    ((fg * alpha) + (bg * inv_alpha)) >> 8
}

impl<M, F, B> Surface<u32> for Over<M, F, B>
where
    M: Surface<u8>,
    F: Surface<u32>,
    B: Surface<u32>,
{
    #[inline(always)]
    fn eval(&self, x: Batch<u32>, y: Batch<u32>) -> Batch<u32> {
        let alpha_val = self.mask.eval(x, y);
        let a = alpha_val.cast::<u32>(); 
        let alpha_broadcast = a * Batch::splat(0x01010101);

        let fg = self.fg.eval(x, y);
        let bg = self.bg.eval(x, y);

        let mask_8 = Batch::splat(0xFF);
        
        let r = blend_math(fg & mask_8, bg & mask_8, alpha_broadcast & mask_8);
        let g = blend_math((fg >> 8) & mask_8, (bg >> 8) & mask_8, (alpha_broadcast >> 8) & mask_8);
        let b = blend_math((fg >> 16) & mask_8, (bg >> 16) & mask_8, (alpha_broadcast >> 16) & mask_8);
        let a = blend_math((fg >> 24) & mask_8, (bg >> 24) & mask_8, (alpha_broadcast >> 24) & mask_8);

        r | (g << 8) | (b << 16) | (a << 24)
    }
}
