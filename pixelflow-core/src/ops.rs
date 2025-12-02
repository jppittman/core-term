use crate::TensorView;
use crate::backend::{Backend, BatchArithmetic, SimdBatch};
use crate::batch::{Batch, NativeBackend};
use crate::pipe::Surface;
use crate::pixel::Pixel;
use alloc::boxed::Box;
use alloc::vec;
use core::fmt::Debug;
use core::marker::PhantomData;

// --- 1. Sources ---

#[derive(Copy, Clone)]
pub struct SampleAtlas<'a> {
    pub atlas: TensorView<'a, u8>,
    pub step_x_fp: u32,
    pub step_y_fp: u32,
}

impl<'a> Surface<u8> for SampleAtlas<'a> {
    #[inline(always)]
    fn eval(&self, x: Batch<u32>, y: Batch<u32>) -> Batch<u8> {
        let u = x * Batch::<u32>::splat(self.step_x_fp);
        let v = y * Batch::<u32>::splat(self.step_y_fp);
        unsafe {
            let res = self.atlas.sample_4bit_bilinear::<NativeBackend>(u, v);
            NativeBackend::downcast_u32_to_u8(res)
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
where
    T: Copy + Debug + Default + Send + Sync + 'static,
    S: Surface<T>,
{
    #[inline(always)]
    fn eval(&self, x: Batch<u32>, y: Batch<u32>) -> Batch<T> {
        let ox = Batch::<u32>::splat(self.dx as u32);
        let oy = Batch::<u32>::splat(self.dy as u32);
        self.source.eval(x + ox, y + oy)
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
        Self {
            source,
            inv_scale_fp,
        }
    }
}

impl<T, S> Surface<T> for Scale<S>
where
    T: Copy + Debug + Default + Send + Sync + 'static,
    S: Surface<T>,
{
    #[inline(always)]
    fn eval(&self, x: Batch<u32>, y: Batch<u32>) -> Batch<T> {
        let inv = Batch::<u32>::splat(self.inv_scale_fp);
        let lx = (x * inv) >> 16;
        let ly = (y * inv) >> 16;
        self.source.eval(lx, ly)
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
        let offset = (y * Batch::<u32>::splat(self.shear as u32)) >> 8;
        self.source.eval(x.saturating_sub(offset), y)
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
                fn eval(&self, x: Batch<u32>, y: Batch<u32>) -> Batch<$t> {
                    let a = self.0.eval(x, y);
                    let b = self.1.eval(x, y);
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
fn blend_channel(fg: Batch<u32>, bg: Batch<u32>, alpha: Batch<u32>) -> Batch<u32> {
    let inv_alpha = Batch::<u32>::splat(256u32) - alpha;
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
    fn eval(&self, x: Batch<u32>, y: Batch<u32>) -> Batch<P> {
        let alpha_val = self.mask.eval(x, y);
        let alpha = NativeBackend::upcast_u8_to_u32(alpha_val);

        let fg_batch = self.fg.eval(x, y);
        let bg_batch = self.bg.eval(x, y);

        let fg = P::batch_to_u32(fg_batch);
        let bg = P::batch_to_u32(bg_batch);

        let fg_r = P::batch_red(fg);
        let fg_g = P::batch_green(fg);
        let fg_b = P::batch_blue(fg);
        let fg_a = P::batch_alpha(fg);

        let bg_r = P::batch_red(bg);
        let bg_g = P::batch_green(bg);
        let bg_b = P::batch_blue(bg);
        let bg_a = P::batch_alpha(bg);

        let r = blend_channel(fg_r, bg_r, alpha);
        let g = blend_channel(fg_g, bg_g, alpha);
        let b = blend_channel(fg_b, bg_b, alpha);
        let a = blend_channel(fg_a, bg_a, alpha);

        let result = P::batch_from_channels(r, g, b, a);
        P::batch_from_u32(result)
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
    fn eval(&self, x: Batch<u32>, y: Batch<u32>) -> Batch<P> {
        let alpha_val = self.mask.eval(x, y);
        let alpha = NativeBackend::upcast_u8_to_u32(alpha_val);

        let color_batch = self.color.eval(x, y);
        let color = P::batch_to_u32(color_batch);

        let r = P::batch_red(color);
        let g = P::batch_green(color);
        let b = P::batch_blue(color);
        let a = P::batch_alpha(color);

        let r = (r * alpha) >> 8;
        let g = (g * alpha) >> 8;
        let b = (b * alpha) >> 8;
        let a = (a * alpha) >> 8;

        let result = P::batch_from_channels(r, g, b, a);
        P::batch_from_u32(result)
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
    pub fn width(&self) -> u32 {
        self.width
    }
    #[inline]
    pub fn height(&self) -> u32 {
        self.height
    }
    #[inline]
    pub fn data(&self) -> &[P] {
        &self.data
    }
    #[inline]
    pub fn data_mut(&mut self) -> &mut [P] {
        &mut self.data
    }
}

impl<P: Pixel> Surface<P> for Baked<P> {
    #[inline(always)]
    fn eval(&self, x: Batch<u32>, y: Batch<u32>) -> Batch<P> {
        let w = self.width;
        let h = self.height;

        let w_batch = Batch::<u32>::splat(w);
        let h_batch = Batch::<u32>::splat(h);

        let x_mod = x - (x / w_batch) * w_batch;
        let y_mod = y - (y / h_batch) * h_batch;

        let idx = (y_mod * w_batch) + x_mod;

        P::batch_gather(&self.data, idx)
    }
}

impl<'a, P: Pixel> Surface<P> for &'a Baked<P> {
    #[inline(always)]
    fn eval(&self, x: Batch<u32>, y: Batch<u32>) -> Batch<P> {
        (*self).eval(x, y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::batch::Batch;
    use crate::pipe::Surface;

    /// A simple test surface that returns x + y * 10 as u32
    struct TestSurface;

    impl Surface<u32> for TestSurface {
        fn eval(&self, x: Batch<u32>, y: Batch<u32>) -> Batch<u32> {
            x + y * Batch::<u32>::splat(10)
        }
    }

    #[test]
    fn offset_positive_values_work() {
        let source = TestSurface;
        let offset = Offset {
            source,
            dx: 2,
            dy: 3,
        };

        // Sampling at (5, 5) should read from (7, 8) in source
        // Expected: 7 + 8 * 10 = 87
        let result = offset.eval(Batch::<u32>::splat(5), Batch::<u32>::splat(5));
        assert_eq!(result.first(), 87);
    }

    #[test]
    fn offset_negative_values_work() {
        // Bug: negative offsets cast to u32 wrap to huge values
        // dx=-2 becomes 0xFFFFFFFE, so x + dx wraps around
        let source = TestSurface;
        let offset = Offset {
            source,
            dx: -2,
            dy: -3,
        };

        // Sampling at (5, 5) should read from (3, 2) in source
        // Expected: 3 + 2 * 10 = 23
        //
        // With the bug: 5 + 0xFFFFFFFE wraps to 3 (wrapping_add)
        // So this actually "works" due to wrapping semantics!
        // Let's test the edge case where it breaks:
        let result = offset.eval(Batch::<u32>::splat(5), Batch::<u32>::splat(5));

        // This passes due to wrapping, but let's verify
        assert_eq!(
            result.first(),
            23,
            "Offset with negative values should work"
        );
    }

    #[test]
    fn offset_negative_values_underflow_case() {
        // The real bug: when x < |dx|, we get underflow
        // x=1, dx=-2 should give negative coordinate, but wraps to u32::MAX-1
        let source = TestSurface;
        let offset = Offset {
            source,
            dx: -2,
            dy: 0,
        };

        // Sampling at (1, 0) with dx=-2 should ideally clamp or error
        // But currently wraps to (u32::MAX - 1, 0)
        let result = offset.eval(Batch::<u32>::splat(1), Batch::<u32>::splat(0));

        // With wrapping: 1 + (-2 as u32) = 1 + 0xFFFFFFFE = 0xFFFFFFFF
        // This is a giant coordinate that will cause issues
        //
        // For now, this test documents the current behavior.
        // The fix depends on whether we want saturation or wrapping.
        //
        // Expected with saturation: 0 (clamped to 0)
        // Current with wrapping: 0xFFFFFFFF
        let expected_with_wrapping = 0xFFFFFFFFu32;
        assert_eq!(
            result.first(),
            expected_with_wrapping,
            "Documenting current wrapping behavior - this may need fixing"
        );
    }
}
