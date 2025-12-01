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
    where
        B::Batch<u32>: BatchArithmetic<u32>,
        B::Batch<f32>: BatchArithmetic<f32>,
        B::Batch<u8>: BatchArithmetic<u8>
    {
        let u = x * <B::Batch<u32> as SimdBatch<u32>>::splat(self.step_x_fp);
        let v = y * <B::Batch<u32> as SimdBatch<u32>>::splat(self.step_y_fp);
        unsafe {
            let res = self.atlas.sample_4bit_bilinear::<B>(u, v);
            B::downcast_u32_to_u8(res)
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
    where
        B::Batch<u32>: BatchArithmetic<u32>,
        B::Batch<f32>: BatchArithmetic<f32>,
        B::Batch<u8>: BatchArithmetic<u8>
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
    where
        B::Batch<u32>: BatchArithmetic<u32>,
        B::Batch<f32>: BatchArithmetic<f32>,
        B::Batch<u8>: BatchArithmetic<u8>
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
    where
        B::Batch<u32>: BatchArithmetic<u32>,
        B::Batch<f32>: BatchArithmetic<f32>,
        B::Batch<u8>: BatchArithmetic<u8>
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
                where
                    Back::Batch<u32>: BatchArithmetic<u32>,
                    Back::Batch<f32>: BatchArithmetic<f32>,
                    Back::Batch<u8>: BatchArithmetic<u8>
                {
                    let a = self.0.eval::<Back>(x, y);
                    let b = self.1.eval::<Back>(x, y);
                    a.max(b)
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
    where
        B::Batch<u32>: BatchArithmetic<u32>,
        B::Batch<f32>: BatchArithmetic<f32>,
        B::Batch<u8>: BatchArithmetic<u8>
    {
        let alpha_val = self.mask.eval::<B>(x, y);
        let alpha = B::upcast_u8_to_u32(alpha_val);

        let fg_batch = self.fg.eval::<B>(x, y);
        let bg_batch = self.bg.eval::<B>(x, y);

        // Convert batch of pixels P to batch of u32 (packed)
        let fg = P::batch_to_u32::<B>(fg_batch);
        let bg = P::batch_to_u32::<B>(bg_batch);

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
        P::batch_from_u32::<B>(result)
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
    where
        B::Batch<u32>: BatchArithmetic<u32>,
        B::Batch<f32>: BatchArithmetic<f32>,
        B::Batch<u8>: BatchArithmetic<u8>
    {
        let alpha_val = self.mask.eval::<B>(x, y);
        let alpha = B::upcast_u8_to_u32(alpha_val);

        let color_batch = self.color.eval::<B>(x, y);
        let color = P::batch_to_u32::<B>(color_batch);

        let r = P::batch_red::<B>(color);
        let g = P::batch_green::<B>(color);
        let b = P::batch_blue::<B>(color);
        let a = P::batch_alpha::<B>(color);

        let r = (r * alpha) >> 8;
        let g = (g * alpha) >> 8;
        let b = (b * alpha) >> 8;
        let a = (a * alpha) >> 8;

        let result = P::batch_from_channels::<B>(r, g, b, a);
        P::batch_from_u32::<B>(result)
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
    where
        B::Batch<u32>: BatchArithmetic<u32>,
        B::Batch<f32>: BatchArithmetic<f32>,
        B::Batch<u8>: BatchArithmetic<u8>
    {
        let w = self.width as usize;
        let h = self.height as usize;

        let w_batch = <B::Batch<u32> as SimdBatch<u32>>::splat(w as u32);
        let h_batch = <B::Batch<u32> as SimdBatch<u32>>::splat(h as u32);

        let x_mod = x - (x / w_batch) * w_batch;
        let y_mod = y - (y / h_batch) * h_batch;

        let idx = (y_mod * w_batch) + x_mod;

        P::batch_gather::<B>(&self.data, idx)
    }
}

impl<'a, P: Pixel> Surface<P> for &'a Baked<P> {
    #[inline(always)]
    fn eval<B: Backend>(&self, x: B::Batch<u32>, y: B::Batch<u32>) -> B::Batch<P>
    where
        B::Batch<u32>: BatchArithmetic<u32>,
        B::Batch<f32>: BatchArithmetic<f32>,
        B::Batch<u8>: BatchArithmetic<u8>
    {
        (*self).eval::<B>(x, y)
    }
}

// Allow using a reference to a Baked surface in the DSL.
// This enables `glyph.over(fg, bg)` without cloning.
impl<P: Pixel> Surface<P> for &Baked<P> {
    #[inline(always)]
    fn eval(&self, x: Batch<u32>, y: Batch<u32>) -> Batch<P> {
        (*self).eval(x, y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_baked_u8_stores_correctly() {
        // Surface returns values in u32 lanes (natural SIMD layout)
        struct TestSurface;
        impl Surface<u8> for TestSurface {
            fn eval(&self, x: Batch<u32>, _y: Batch<u32>) -> Batch<u8> {
                // Values in u32 lanes - just cast
                x.cast::<u8>()
            }
        }

        let baked = Baked::new(&TestSurface, 8, 2);

        // Row 0 should be: [0, 1, 2, 3, 4, 5, 6, 7]
        assert_eq!(baked.data()[0], 0, "data[0] should be 0");
        assert_eq!(baked.data()[1], 1, "data[1] should be 1");
        assert_eq!(baked.data()[2], 2, "data[2] should be 2");
        assert_eq!(baked.data()[3], 3, "data[3] should be 3");
        assert_eq!(baked.data()[4], 4, "data[4] should be 4");
        assert_eq!(baked.data()[5], 5, "data[5] should be 5");
        assert_eq!(baked.data()[6], 6, "data[6] should be 6");
        assert_eq!(baked.data()[7], 7, "data[7] should be 7");

        // Check second row
        assert_eq!(baked.data()[8], 0, "data[8] (row 1, col 0) should be 0");
        assert_eq!(baked.data()[9], 1, "data[9] (row 1, col 1) should be 1");
    }

    #[test]
    fn test_over_compositor_blends_correctly() {
        use crate::dsl::MaskExt;
        use crate::pixel::Pixel;

        // A simple Pixel implementation for testing (RGBA format)
        #[derive(Copy, Clone, Default, Debug, PartialEq)]
        #[repr(transparent)]
        struct TestRgba(u32);

        impl Pixel for TestRgba {
            fn from_u32(v: u32) -> Self { Self(v) }
            fn to_u32(self) -> u32 { self.0 }
            fn batch_red(batch: Batch<u32>) -> Batch<u32> { batch & Batch::splat(0xFF) }
            fn batch_green(batch: Batch<u32>) -> Batch<u32> { (batch >> 8) & Batch::splat(0xFF) }
            fn batch_blue(batch: Batch<u32>) -> Batch<u32> { (batch >> 16) & Batch::splat(0xFF) }
            fn batch_alpha(batch: Batch<u32>) -> Batch<u32> { batch >> 24 }
            fn batch_from_channels(r: Batch<u32>, g: Batch<u32>, b: Batch<u32>, a: Batch<u32>) -> Batch<u32> {
                r | (g << 8) | (b << 16) | (a << 24)
            }
        }

        impl Surface<TestRgba> for TestRgba {
            fn eval(&self, _x: Batch<u32>, _y: Batch<u32>) -> Batch<TestRgba> {
                Batch::splat(self.0).transmute()
            }
        }

        // Constant mask surface (128 = 50% alpha)
        struct ConstMask(u8);
        impl Surface<u8> for ConstMask {
            fn eval(&self, _x: Batch<u32>, _y: Batch<u32>) -> Batch<u8> {
                Batch::splat(self.0 as u32).transmute()
            }
        }

        let mask = ConstMask(128); // 50% alpha
        let fg = TestRgba(0xFF_00_00_FF); // Red (RGBA: R=255, G=0, B=0, A=255)
        let bg = TestRgba(0xFF_FF_00_00); // Blue (RGBA: R=0, G=0, B=255, A=255)

        // Compose: mask.over(fg, bg)
        let composed = mask.over::<TestRgba, _, _>(fg, bg);

        // Evaluate at (0, 0)
        let result = composed.eval(Batch::splat(0), Batch::splat(0));
        let result_u32: Batch<u32> = result.transmute();
        let pixel = result_u32.to_array_usize()[0] as u32;

        // Extract channels
        let r = pixel & 0xFF;
        let g = (pixel >> 8) & 0xFF;
        let b = (pixel >> 16) & 0xFF;
        let a = (pixel >> 24) & 0xFF;

        // 50% blend of red (255) and blue (0) for R channel: ~127-128
        // 50% blend of blue (0) and red (255) for B channel: ~127-128
        assert!(r > 120 && r < 136, "Red channel should be ~128, got {}", r);
        assert_eq!(g, 0, "Green should be 0");
        assert!(b > 120 && b < 136, "Blue channel should be ~128, got {}", b);
        assert!(a > 250, "Alpha should be ~255, got {}", a);
    }
}
