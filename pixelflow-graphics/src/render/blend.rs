// pixelflow-graphics/src/render/blend.rs

use crate::render::pixel::Pixel; // Local usage in graphics
use core::fmt::Debug;
use core::marker::PhantomData;
use pixelflow_core::batch::Batch;
use pixelflow_core::traits::Manifold;

/// Computes the maximum value of two surfaces.
#[derive(Copy, Clone)]
pub struct Max<A, B>(pub A, pub B);

macro_rules! impl_max_manifold {
    ($($t:ty),*) => {
        $(
            impl<A, B, C> Manifold<$t, C> for Max<A, B>
            where
                A: Manifold<$t, C>,
                B: Manifold<$t, C>,
                C: Copy + Debug + Default + PartialEq + Send + Sync + 'static,
            {
                #[inline(always)]
                fn eval_raw(&self, x: Batch<C>, y: Batch<C>, z: Batch<C>, w: Batch<C>) -> Batch<$t> {
                    let a = self.0.eval_raw(x, y, z, w);
                    let b = self.1.eval_raw(x, y, z, w);
                    a.max(b)
                }
            }
        )*
    }
}

impl_max_manifold!(u32, u8, f32);

/// Composites a foreground surface over a background surface using a mask.
#[derive(Copy, Clone)]
pub struct Over<P, M, F, B> {
    /// The alpha mask surface.
    pub mask: M,
    /// The foreground surface.
    pub fg: F,
    /// The background surface.
    pub bg: B,
    /// Phantom data for pixel type.
    pub _pixel: PhantomData<P>,
}

impl<P, M, F, B> Over<P, M, F, B> {
    /// Creates a new `Over` compositing surface.
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
pub(crate) fn blend_channel(fg: Batch<u32>, bg: Batch<u32>, alpha: Batch<u32>) -> Batch<u32> {
    let inv_alpha = Batch::<u32>::splat(256u32) - alpha;
    ((fg * alpha) + (bg * inv_alpha)) >> 8
}

impl<P, M, F, Back, C> Manifold<P, C> for Over<P, M, F, Back>
where
    P: Pixel + Copy + PartialEq,
    M: Manifold<P, C>,
    F: Manifold<P, C>,
    Back: Manifold<P, C>,
    C: Copy + Debug + Default + PartialEq + Send + Sync + 'static,
{
    #[inline(always)]
    fn eval_raw(&self, x: Batch<C>, y: Batch<C>, z: Batch<C>, w: Batch<C>) -> Batch<P> {
        // Extract alpha from the mask pixel (coverage in alpha channel)
        let mask_pixel = P::batch_to_u32(self.mask.eval_raw(x, y, z, w));
        let alpha = P::batch_alpha(mask_pixel);

        let fg_batch = self.fg.eval_raw(x, y, z, w);
        let bg_batch = self.bg.eval_raw(x, y, z, w);

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

/// Multiplies a color surface by a mask (alpha).
#[derive(Copy, Clone)]
pub struct Mul<M, C> {
    /// The mask surface.
    pub mask: M,
    /// The color surface.
    pub color: C,
}

impl<P, M, Col, Coord> Manifold<P, Coord> for Mul<M, Col>
where
    P: Pixel + Copy + PartialEq,
    M: Manifold<P, Coord>,
    Col: Manifold<P, Coord>,
    Coord: Copy + Debug + Default + PartialEq + Send + Sync + 'static,
{
    #[inline(always)]
    fn eval_raw(&self, x: Batch<Coord>, y: Batch<Coord>, z: Batch<Coord>, w: Batch<Coord>) -> Batch<P> {
        // Extract alpha from the mask pixel (coverage in alpha channel)
        let mask_pixel = P::batch_to_u32(self.mask.eval_raw(x, y, z, w));
        let alpha = P::batch_alpha(mask_pixel);

        let color_batch = self.color.eval_raw(x, y, z, w);
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
