use crate::TensorView;
use crate::backend::{Backend, SimdBatch, BatchArithmetic};
use crate::pipe::Surface;
use crate::pixel::Pixel;
use crate::batch::{NativeBackend, LANES};
use alloc::boxed::Box;
use alloc::vec;
use core::marker::PhantomData;
use core::fmt::Debug;

// --- 1. Sources ---

#[derive(Copy, Clone)]
pub struct SampleAtlas<'a> {
    pub atlas: TensorView<'a, u8>,
    pub step_x_fp: u32,
    pub step_y_fp: u32,
}

impl<'a> Surface<u8> for SampleAtlas<'a> {
    #[inline(always)]
    fn eval<B: Backend>(&self, x: B::Batch<u32>, y: B::Batch<u32>) -> B::Batch<u8>
    where B::Batch<u32>: BatchArithmetic<u32>
    {
        let u = x * <B::Batch<u32> as SimdBatch<u32>>::splat(self.step_x_fp);
        let v = y * <B::Batch<u32> as SimdBatch<u32>>::splat(self.step_y_fp);
        unsafe {
            let res = self.atlas.sample_4bit_bilinear::<B>(u, v);
            core::mem::transmute_copy(&res)
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

impl<T, S> Surface<T> for Offset<S>
where T: Copy + Debug + Default + Send + Sync + 'static,
      S: Surface<T>
{
    #[inline(always)]
    fn eval<B: Backend>(&self, x: B::Batch<u32>, y: B::Batch<u32>) -> B::Batch<T>
    where B::Batch<u32>: BatchArithmetic<u32>
    {
        let ox = <B::Batch<u32> as SimdBatch<u32>>::splat(self.dx as u32);
        let oy = <B::Batch<u32> as SimdBatch<u32>>::splat(self.dy as u32);
        self.source.eval::<B>(x + ox, y + oy)
    }
}

#[derive(Copy, Clone)]
pub struct Scale<S> {
    pub source: S,
    pub inv_scale_fp: u32,
}

impl<S> Scale<S> {
    #[inline]
    pub fn new(source: S, scale_factor: f64) -> Self {
        let inv_scale_fp = ((1.0 / scale_factor) * 65536.0) as u32;
        Self { source, inv_scale_fp }
    }
}

impl<T, S> Surface<T> for Scale<S>
where T: Copy + Debug + Default + Send + Sync + 'static,
      S: Surface<T>
{
    #[inline(always)]
    fn eval<B: Backend>(&self, x: B::Batch<u32>, y: B::Batch<u32>) -> B::Batch<T>
    where B::Batch<u32>: BatchArithmetic<u32>
    {
        let inv = <B::Batch<u32> as SimdBatch<u32>>::splat(self.inv_scale_fp);
        let lx = (x * inv) >> 16;
        let ly = (y * inv) >> 16;
        self.source.eval::<B>(lx, ly)
    }
}

#[derive(Copy, Clone)]
pub struct Skew<S> {
    pub source: S,
    pub shear: i32,
}

impl<S: Surface<u8>> Surface<u8> for Skew<S> {
    #[inline(always)]
    fn eval<B: Backend>(&self, x: B::Batch<u32>, y: B::Batch<u32>) -> B::Batch<u8>
    where B::Batch<u32>: BatchArithmetic<u32>
    {
        let offset = (y * <B::Batch<u32> as SimdBatch<u32>>::splat(self.shear as u32)) >> 8;
        self.source.eval::<B>(x.saturating_sub(offset), y)
    }
}

#[derive(Copy, Clone)]
pub struct Max<A, B>(pub A, pub B);

macro_rules! impl_max_surface {
    ($($t:ty),*) => {
        $(
            impl<A, B> Surface<$t> for Max<A, B>
            where A: Surface<$t>, B: Surface<$t>
            {
                #[inline(always)]
                fn eval<Back: Backend>(&self, x: Back::Batch<u32>, y: Back::Batch<u32>) -> Back::Batch<$t>
                where Back::Batch<u32>: BatchArithmetic<u32>
                {
                    let a = self.0.eval::<Back>(x, y);
                    let b = self.1.eval::<Back>(x, y);
                    unimplemented!("Max not fully supported in this refactor step due to trait limits")
                }
            }
        )*
    }
}

impl_max_surface!(u32, u8, f32);

// --- 3. Finalizers (Blend) ---

#[derive(Copy, Clone)]
pub struct Over<P, M, F, B> {
    pub mask: M,
    pub fg: F,
    pub bg: B,
    pub _pixel: PhantomData<P>,
}

impl<P, M, F, B> Over<P, M, F, B> {
    #[inline]
    pub fn new(mask: M, fg: F, bg: B) -> Self {
        Self {
            mask,
            fg,
            bg,
            _pixel: PhantomData,
        }
    }
}

#[inline(always)]
fn blend_math<B: Backend>(fg: B::Batch<u32>, bg: B::Batch<u32>, alpha: B::Batch<u32>) -> B::Batch<u32>
where B::Batch<u32>: BatchArithmetic<u32>
{
    let inv_alpha = <B::Batch<u32> as SimdBatch<u32>>::splat(256) - alpha;
    ((fg * alpha) + (bg * inv_alpha)) >> 8
}

impl<P, M, F, Back> Surface<P> for Over<P, M, F, Back>
where
    P: Pixel + Copy,
    M: Surface<u8>,
    F: Surface<P>,
    Back: Surface<P>,
{
    #[inline(always)]
    fn eval<B: Backend>(&self, x: B::Batch<u32>, y: B::Batch<u32>) -> B::Batch<P>
    where B::Batch<u32>: BatchArithmetic<u32>
    {
        let alpha_val = self.mask.eval::<B>(x, y);
        let alpha = unsafe { core::mem::transmute_copy(&alpha_val) };

        let fg_batch = self.fg.eval::<B>(x, y);
        let bg_batch = self.bg.eval::<B>(x, y);

        let fg: B::Batch<u32> = unsafe { core::mem::transmute_copy(&fg_batch) };
        let bg: B::Batch<u32> = unsafe { core::mem::transmute_copy(&bg_batch) };

        let fg_r = P::batch_red::<B>(fg);
        let fg_g = P::batch_green::<B>(fg);
        let fg_b = P::batch_blue::<B>(fg);
        let fg_a = P::batch_alpha::<B>(fg);

        let bg_r = P::batch_red::<B>(bg);
        let bg_g = P::batch_green::<B>(bg);
        let bg_b = P::batch_blue::<B>(bg);
        let bg_a = P::batch_alpha::<B>(bg);

        let r = blend_math::<B>(fg_r, bg_r, alpha);
        let g = blend_math::<B>(fg_g, bg_g, alpha);
        let b = blend_math::<B>(fg_b, bg_b, alpha);
        let a = blend_math::<B>(fg_a, bg_a, alpha);

        let result = P::batch_from_channels::<B>(r, g, b, a);
        unsafe { core::mem::transmute_copy(&result) }
    }
}

#[derive(Copy, Clone)]
pub struct Mul<M, C> {
    pub mask: M,
    pub color: C,
}

impl<P, M, C> Surface<P> for Mul<M, C>
where
    P: Pixel + Copy,
    M: Surface<u8>,
    C: Surface<P>,
{
    #[inline(always)]
    fn eval<B: Backend>(&self, x: B::Batch<u32>, y: B::Batch<u32>) -> B::Batch<P>
    where B::Batch<u32>: BatchArithmetic<u32>
    {
        let alpha_val = self.mask.eval::<B>(x, y);
        let alpha = unsafe { core::mem::transmute_copy(&alpha_val) };

        let color_batch = self.color.eval::<B>(x, y);
        let color: B::Batch<u32> = unsafe { core::mem::transmute_copy(&color_batch) };

        let r = P::batch_red::<B>(color);
        let g = P::batch_green::<B>(color);
        let b = P::batch_blue::<B>(color);
        let a = P::batch_alpha::<B>(color);

        let r = (r * alpha) >> 8;
        let g = (g * alpha) >> 8;
        let b = (b * alpha) >> 8;
        let a = (a * alpha) >> 8;

        let result = P::batch_from_channels::<B>(r, g, b, a);
        unsafe { core::mem::transmute_copy(&result) }
    }
}

// --- 4. Memoizers ---

#[derive(Clone)]
pub struct Baked<P: Pixel> {
    data: Box<[P]>,
    width: u32,
    height: u32,
}

impl<P: Pixel> Baked<P> {
    pub fn new<S: Surface<P>>(source: &S, width: u32, height: u32) -> Self {
        let mut data = vec![P::default(); (width as usize) * (height as usize)].into_boxed_slice();
        crate::execute(source, &mut data, width as usize, height as usize);
        Self {
            data,
            width,
            height,
        }
    }

    #[inline]
    pub fn width(&self) -> u32 { self.width }
    #[inline]
    pub fn height(&self) -> u32 { self.height }
    #[inline]
    pub fn data(&self) -> &[P] { &self.data }
    #[inline]
    pub fn data_mut(&mut self) -> &mut [P] { &mut self.data }
}

impl<P: Pixel> Surface<P> for Baked<P> {
    #[inline(always)]
    fn eval<B: Backend>(&self, x: B::Batch<u32>, y: B::Batch<u32>) -> B::Batch<P>
    where B::Batch<u32>: BatchArithmetic<u32>
    {
        let w = self.width as usize;
        let h = self.height as usize;

        let w_batch = <B::Batch<u32> as SimdBatch<u32>>::splat(w as u32);
        let h_batch = <B::Batch<u32> as SimdBatch<u32>>::splat(h as u32);

        let x_mod = x - (x / w_batch) * w_batch;
        let y_mod = y - (y / h_batch) * h_batch;

        let idx = (y_mod * w_batch) + x_mod;

        let len = self.data.len();
        let u32_data = unsafe { core::slice::from_raw_parts(self.data.as_ptr() as *const u32, len) };

        // Use BatchArithmetic::gather
        let gathered = <B::Batch<u32> as BatchArithmetic<u32>>::gather(u32_data, idx);
        unsafe { core::mem::transmute_copy(&gathered) }
    }
}

impl<'a, P: Pixel> Surface<P> for &'a Baked<P> {
    #[inline(always)]
    fn eval<B: Backend>(&self, x: B::Batch<u32>, y: B::Batch<u32>) -> B::Batch<P>
    where B::Batch<u32>: BatchArithmetic<u32>
    {
        (*self).eval::<B>(x, y)
    }
}
