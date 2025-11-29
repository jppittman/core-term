use crate::TensorView;
use crate::batch::{Batch, SimdOps, SimdVec};
use crate::pipe::Surface;
use crate::pixel::Pixel;
use core::marker::PhantomData;

// --- 1. Sources ---

/// A source surface that samples from a texture atlas.
///
/// This uses bilinear interpolation to sample from the underlying tensor.
#[derive(Copy, Clone)]
pub struct SampleAtlas<'a> {
    /// The source texture atlas.
    pub atlas: TensorView<'a, u8>,
    /// The horizontal step size in 16.16 fixed point format.
    pub step_x_fp: u32,
    /// The vertical step size in 16.16 fixed point format.
    pub step_y_fp: u32,
}

impl<'a> Surface<u8> for SampleAtlas<'a> {
    /// Evaluates the surface at the given coordinates.
    ///
    /// # Parameters
    /// * `x` - X coordinates.
    /// * `y` - Y coordinates.
    ///
    /// # Returns
    /// * A batch of sampled values.
    #[inline(always)]
    fn eval(&self, x: Batch<u32>, y: Batch<u32>) -> Batch<u8> {
        // Direct bilinear sample
        // Convert integer coordinates to 16.16 fixed point format expected by sample_4bit_bilinear.
        let u = x * Batch::splat(self.step_x_fp);
        let v = y * Batch::splat(self.step_y_fp);
        unsafe { self.atlas.sample_4bit_bilinear(u, v).cast() }
    }
}

// --- 2. Transformers ---

/// A transformer that applies a translation offset to the coordinate space.
#[derive(Copy, Clone)]
pub struct Offset<S> {
    /// The source surface.
    pub source: S,
    /// Horizontal offset.
    pub dx: i32,
    /// Vertical offset.
    pub dy: i32,
}

impl<T: Copy, S: Surface<T>> Surface<T> for Offset<S> {
    /// Evaluates the surface with the offset applied.
    ///
    /// # Parameters
    /// * `x` - X coordinates.
    /// * `y` - Y coordinates.
    ///
    /// # Returns
    /// * A batch of values from the offset source.
    #[inline(always)]
    fn eval(&self, x: Batch<u32>, y: Batch<u32>) -> Batch<T> {
        let ox = Batch::splat(self.dx as u32);
        let oy = Batch::splat(self.dy as u32);

        self.source.eval(x + ox, y + oy)
    }
}

/// A transformer that applies a skew (shear) transformation.
#[derive(Copy, Clone)]
pub struct Skew<S> {
    /// The source surface.
    pub source: S,
    /// The shear factor (fixed point?). Code suggests `y * shear >> 8`.
    /// So shear is 8.8 fixed point? Or just an integer slope scaled by 256.
    pub shear: i32,
}

impl<S: Surface<u8>> Surface<u8> for Skew<S> {
    #[inline(always)]
    fn eval(&self, x: Batch<u32>, y: Batch<u32>) -> Batch<u8> {
        let offset = (y * Batch::splat(self.shear as u32)) >> 8;
        self.source.eval(x.saturating_sub(offset), y)
    }
}

/// A composite surface that takes the maximum of two surfaces.
#[derive(Copy, Clone)]
pub struct Max<A, B>(pub A, pub B);

// Fix: Added trait bound for SimdOps so that Batch<T>::max is available
impl<T: Copy, A: Surface<T>, B: Surface<T>> Surface<T> for Max<A, B>
where
    SimdVec<T>: SimdOps<T>,
{
    #[inline(always)]
    fn eval(&self, x: Batch<u32>, y: Batch<u32>) -> Batch<T> {
        self.0.eval(x, y).max(self.1.eval(x, y))
    }
}

// --- 3. Finalizers (Blend) ---

/// A compositing operation that blends a foreground over a background using a mask.
///
/// Generic over pixel format `P`, allowing format-aware channel extraction and
/// reconstruction. This enables zero-cost format abstraction - the channel
/// operations are inlined and monomorphized for each pixel format.
#[derive(Copy, Clone)]
pub struct Over<P, M, F, B> {
    /// The alpha mask.
    pub mask: M,
    /// The foreground surface.
    pub fg: F,
    /// The background surface.
    pub bg: B,
    /// Phantom data for the pixel format.
    pub _pixel: PhantomData<P>,
}

impl<P, M, F, B> Over<P, M, F, B> {
    /// Creates a new Over combinator.
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

/// Helper function to perform alpha blending math.
///
/// Computes `(fg * alpha + bg * (256 - alpha)) / 256`.
#[inline(always)]
fn blend_math(fg: Batch<u32>, bg: Batch<u32>, alpha: Batch<u32>) -> Batch<u32> {
    let inv_alpha = Batch::splat(256) - alpha;
    ((fg * alpha) + (bg * inv_alpha)) >> 8
}

impl<P, M, F, B> Surface<u32> for Over<P, M, F, B>
where
    P: Pixel,
    M: Surface<u8>,
    F: Surface<u32>,
    B: Surface<u32>,
{
    #[inline(always)]
    fn eval(&self, x: Batch<u32>, y: Batch<u32>) -> Batch<u32> {
        let alpha_val = self.mask.eval(x, y);
        let alpha = alpha_val.cast::<u32>();

        let fg = self.fg.eval(x, y);
        let bg = self.bg.eval(x, y);

        // Extract channels using Pixel trait (format-aware)
        let fg_r = P::batch_red(fg);
        let fg_g = P::batch_green(fg);
        let fg_b = P::batch_blue(fg);
        let fg_a = P::batch_alpha(fg);

        let bg_r = P::batch_red(bg);
        let bg_g = P::batch_green(bg);
        let bg_b = P::batch_blue(bg);
        let bg_a = P::batch_alpha(bg);

        // Blend each channel: result = fg * alpha + bg * (256 - alpha)
        let r = blend_math(fg_r, bg_r, alpha);
        let g = blend_math(fg_g, bg_g, alpha);
        let b = blend_math(fg_b, bg_b, alpha);
        let a = blend_math(fg_a, bg_a, alpha);

        // Reconstruct in target format
        P::batch_from_channels(r, g, b, a)
    }
}
