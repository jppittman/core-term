use crate::TensorView;
use crate::batch::{Batch, SimdOps, SimdVec};
use crate::pipe::Surface;
use crate::pixel::Pixel;
use alloc::boxed::Box;
use alloc::vec;
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

/// A transformer that scales a surface by dividing coordinates.
///
/// Used to upscale a logical-resolution surface to physical resolution.
/// For example, with scale=2, coordinate (100, 100) samples source at (50, 50).
///
/// Uses 16.16 fixed-point internally for sub-pixel precision.
#[derive(Copy, Clone)]
pub struct Scale<S> {
    /// The source surface.
    pub source: S,
    /// Inverse scale in 16.16 fixed point (65536 / scale_factor).
    /// For scale=2.0, this is 32768.
    pub inv_scale_fp: u32,
}

impl<S> Scale<S> {
    /// Creates a new Scale transformer.
    ///
    /// # Parameters
    /// * `source` - The source surface (at logical resolution)
    /// * `scale_factor` - The upscale factor (e.g., 2.0 for HiDPI)
    #[inline]
    pub fn new(source: S, scale_factor: f64) -> Self {
        // inv_scale = 1/scale in 16.16 fixed point
        let inv_scale_fp = ((1.0 / scale_factor) * 65536.0) as u32;
        Self { source, inv_scale_fp }
    }
}

impl<T: Copy, S: Surface<T>> Surface<T> for Scale<S> {
    #[inline(always)]
    fn eval(&self, x: Batch<u32>, y: Batch<u32>) -> Batch<T> {
        // Convert physical coords to logical: logical = physical / scale
        // Using fixed point: (x * inv_scale) >> 16
        let inv = Batch::splat(self.inv_scale_fp);
        let lx = (x * inv) >> 16;
        let ly = (y * inv) >> 16;
        self.source.eval(lx, ly)
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

impl<P, M, F, B> Surface<P> for Over<P, M, F, B>
where
    P: Pixel + Copy,
    M: Surface<u8>,
    F: Surface<P>,
    B: Surface<P>,
{
    #[inline(always)]
    fn eval(&self, x: Batch<u32>, y: Batch<u32>) -> Batch<P> {
        let alpha_val = self.mask.eval(x, y);
        let alpha = alpha_val.cast::<u32>();

        // Evaluate children (returns Batch<P>)
        let fg_batch = self.fg.eval(x, y);
        let bg_batch = self.bg.eval(x, y);

        // Transmute to u32 for SIMD math (P is repr(transparent) u32)
        let fg: Batch<u32> = fg_batch.transmute();
        let bg: Batch<u32> = bg_batch.transmute();

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

        // Reconstruct in target format (P determines byte order)
        let result = P::batch_from_channels(r, g, b, a);

        // Transmute back to Batch<P>
        result.transmute()
    }
}

/// A colorizing operation that multiplies a color surface by a mask (alpha).
///
/// This effectively masks the color surface. The result is premultiplied alpha.
#[derive(Copy, Clone)]
pub struct Mul<M, C> {
    /// The mask surface (coverage).
    pub mask: M,
    /// The color surface.
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
        let alpha = alpha_val.cast::<u32>();

        let color_batch = self.color.eval(x, y);
        let color: Batch<u32> = color_batch.transmute();

        let r = P::batch_red(color);
        let g = P::batch_green(color);
        let b = P::batch_blue(color);
        let a = P::batch_alpha(color);

        // Scale each channel by alpha: (c * alpha) >> 8
        let r = (r * alpha) >> 8;
        let g = (g * alpha) >> 8;
        let b = (b * alpha) >> 8;
        let a = (a * alpha) >> 8;

        let result = P::batch_from_channels(r, g, b, a);
        result.transmute()
    }
}

// --- 4. Memoizers ---

/// A memoized surface that caches the result of evaluating another surface.
///
/// `Baked` materializes a lazy `Surface` into a pixel buffer, then serves
/// as a `Surface` itself with wrap-around out-of-bounds behavior. This is
/// the "checkpoint" combinator - use it to cache expensive surface graphs.
#[derive(Clone)]
pub struct Baked<P: Pixel> {
    /// The baked pixel data.
    data: Box<[P]>,
    /// Width in pixels.
    width: u32,
    /// Height in pixels.
    height: u32,
}

impl<P: Pixel> Baked<P> {
    /// Bakes a surface into a memoized buffer.
    ///
    /// Evaluates the source surface at every pixel coordinate and stores
    /// the results. The returned `Baked` is itself a `Surface`.
    ///
    /// # Parameters
    /// * `source` - The surface to bake.
    /// * `width` - Width of the baked region.
    /// * `height` - Height of the baked region.
    pub fn new<S: Surface<P>>(source: &S, width: u32, height: u32) -> Self {
        let mut data = vec![P::default(); (width as usize) * (height as usize)].into_boxed_slice();

        const LANES: usize = 4;
        let w = width as usize;
        let h = height as usize;

        for y in 0..h {
            let row_start = y * w;
            let y_batch = Batch::splat(y as u32);

            // SIMD hot path: 4 pixels at a time
            let mut x = 0;
            while x + LANES <= w {
                let x_batch = Batch::new(x as u32, (x + 1) as u32, (x + 2) as u32, (x + 3) as u32);

                let result: Batch<P> = source.eval(x_batch, y_batch);
                let result_u32: Batch<u32> = result.transmute();

                // Store 4 pixels at once
                if core::mem::size_of::<P>() == 4 {
                    // u32: direct SIMD store
                    unsafe {
                        let ptr = data.as_mut_ptr().add(row_start + x) as *mut u32;
                        result_u32.store(ptr);
                    }
                } else {
                    // u8: extract low byte of each lane and store as 4 bytes
                    let bytes = result_u32.to_bytes_packed();
                    unsafe {
                        let ptr = data.as_mut_ptr().add(row_start + x) as *mut [u8; 4];
                        *ptr = bytes;
                    }
                }

                x += LANES;
            }

            // Scalar cold path: remaining pixels
            while x < w {
                let x_batch = Batch::splat(x as u32);
                let result: Batch<P> = source.eval(x_batch, y_batch);
                let result_u32: Batch<u32> = result.transmute();
                data[row_start + x] = P::from_u32(result_u32.to_array_usize()[0] as u32);
                x += 1;
            }
        }

        Self {
            data,
            width,
            height,
        }
    }

    /// Returns the width of the baked surface.
    #[inline]
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Returns the height of the baked surface.
    #[inline]
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Returns the raw pixel data.
    #[inline]
    pub fn data(&self) -> &[P] {
        &self.data
    }

    /// Returns mutable access to the raw pixel data.
    #[inline]
    pub fn data_mut(&mut self) -> &mut [P] {
        &mut self.data
    }
}

impl<P: Pixel> Surface<P> for Baked<P> {
    /// Samples from the baked buffer with wrap-around.
    #[inline(always)]
    fn eval(&self, x: Batch<u32>, y: Batch<u32>) -> Batch<P> {
        let w = self.width as usize;
        let h = self.height as usize;

        // Extract coordinates and compute wrap-around element-wise
        let x_arr = x.to_array_usize();
        let y_arr = y.to_array_usize();

        // Calculate wrapped indices
        let idx0 = (y_arr[0] % h) * w + (x_arr[0] % w);
        let idx1 = (y_arr[1] % h) * w + (x_arr[1] % w);
        let idx2 = (y_arr[2] % h) * w + (x_arr[2] % w);
        let idx3 = (y_arr[3] % h) * w + (x_arr[3] % w);

        // Gather pixels
        let p0 = self.data[idx0].to_u32();
        let p1 = self.data[idx1].to_u32();
        let p2 = self.data[idx2].to_u32();
        let p3 = self.data[idx3].to_u32();

        Batch::new(p0, p1, p2, p3).transmute()
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
