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
    use crate::backend::SimdBatch;
    use crate::batch::Batch;
    use crate::pipe::Surface;

    /// A simple test surface that returns x + y * 10 as u32.
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

    // =========================================================================
    // Over combinator tests - alpha blending correctness
    // =========================================================================

    /// Constant u8 mask surface for testing alpha blending.
    struct ConstMask(u8);

    impl Surface<u8> for ConstMask {
        fn eval(&self, _x: Batch<u32>, _y: Batch<u32>) -> Batch<u8> {
            Batch::<u8>::splat(self.0)
        }
    }

    /// Constant u32 color surface for testing.
    struct ConstColor(u32);

    impl Surface<u32> for ConstColor {
        fn eval(&self, _x: Batch<u32>, _y: Batch<u32>) -> Batch<u32> {
            Batch::<u32>::splat(self.0)
        }
    }

    /// Extracts RGBA channels from a packed u32 (little-endian: 0xAABBGGRR).
    fn unpack_rgba(packed: u32) -> (u8, u8, u8, u8) {
        let r = (packed & 0xFF) as u8;
        let g = ((packed >> 8) & 0xFF) as u8;
        let b = ((packed >> 16) & 0xFF) as u8;
        let a = ((packed >> 24) & 0xFF) as u8;
        (r, g, b, a)
    }

    /// Packs RGBA channels into u32 (little-endian: 0xAABBGGRR).
    fn pack_rgba(r: u8, g: u8, b: u8, a: u8) -> u32 {
        (r as u32) | ((g as u32) << 8) | ((b as u32) << 16) | ((a as u32) << 24)
    }

    #[test]
    fn over_alpha_zero_returns_background() {
        let fg = ConstColor(pack_rgba(255, 0, 0, 255)); // Red
        let bg = ConstColor(pack_rgba(0, 0, 255, 255)); // Blue
        let mask = ConstMask(0); // Zero alpha

        let over: Over<u32, _, _, _> = Over::new(mask, fg, bg);
        let result = over.eval(Batch::<u32>::splat(0), Batch::<u32>::splat(0));
        let (r, g, b, _a) = unpack_rgba(result.first());

        assert_eq!(r, 0, "With alpha=0, red channel should be 0 (pure bg)");
        assert_eq!(g, 0, "With alpha=0, green channel should be 0");
        assert_eq!(b, 255, "With alpha=0, blue channel should be 255 (pure bg)");
    }

    #[test]
    fn over_alpha_full_returns_foreground() {
        let fg = ConstColor(pack_rgba(255, 0, 0, 255)); // Red
        let bg = ConstColor(pack_rgba(0, 0, 255, 255)); // Blue
        let mask = ConstMask(255); // Full alpha

        let over: Over<u32, _, _, _> = Over::new(mask, fg, bg);
        let result = over.eval(Batch::<u32>::splat(0), Batch::<u32>::splat(0));
        let (r, g, b, _a) = unpack_rgba(result.first());

        // blend_channel formula: (fg * alpha + bg * (256 - alpha)) >> 8
        // With alpha=255: (255 * 255 + 0 * 1) >> 8 = 65025 >> 8 = 254
        assert!(
            r >= 254,
            "With alpha=255, red should be ~255, got {}",
            r
        );
        assert_eq!(g, 0, "With alpha=255, green should be 0");
        assert!(
            b <= 1,
            "With alpha=255, blue should be ~0, got {}",
            b
        );
    }

    #[test]
    fn over_alpha_half_blends_evenly() {
        let fg = ConstColor(pack_rgba(255, 0, 0, 255)); // Red
        let bg = ConstColor(pack_rgba(0, 0, 255, 255)); // Blue
        let mask = ConstMask(128); // Half alpha

        let over: Over<u32, _, _, _> = Over::new(mask, fg, bg);
        let result = over.eval(Batch::<u32>::splat(0), Batch::<u32>::splat(0));
        let (r, g, b, _a) = unpack_rgba(result.first());

        // With alpha=128: (255 * 128 + 0 * 128) >> 8 = 32640 >> 8 = 127
        // Background blue: (0 * 128 + 255 * 128) >> 8 = 32640 >> 8 = 127
        assert!(
            r > 100 && r < 150,
            "With alpha=128, red should be ~127, got {}",
            r
        );
        assert_eq!(g, 0, "With alpha=128, green should be 0");
        assert!(
            b > 100 && b < 150,
            "With alpha=128, blue should be ~127, got {}",
            b
        );
    }

    #[test]
    fn over_preserves_all_channels_independently() {
        // Test that R, G, B, A channels blend independently
        let fg = ConstColor(pack_rgba(200, 100, 50, 255));
        let bg = ConstColor(pack_rgba(50, 150, 200, 128));
        let mask = ConstMask(128);

        let over: Over<u32, _, _, _> = Over::new(mask, fg, bg);
        let result = over.eval(Batch::<u32>::splat(0), Batch::<u32>::splat(0));
        let (r, g, b, a) = unpack_rgba(result.first());

        // Each channel should be ~midpoint between fg and bg
        // R: (200 * 128 + 50 * 128) >> 8 = 32000 >> 8 = 125
        // G: (100 * 128 + 150 * 128) >> 8 = 32000 >> 8 = 125
        // B: (50 * 128 + 200 * 128) >> 8 = 32000 >> 8 = 125
        // A: (255 * 128 + 128 * 128) >> 8 = 49024 >> 8 = 191
        assert!(
            r > 115 && r < 135,
            "Red should be ~125, got {}",
            r
        );
        assert!(
            g > 115 && g < 135,
            "Green should be ~125, got {}",
            g
        );
        assert!(
            b > 115 && b < 135,
            "Blue should be ~125, got {}",
            b
        );
        assert!(
            a > 180 && a < 200,
            "Alpha should be ~191, got {}",
            a
        );
    }

    #[test]
    fn blend_channel_math_is_correct() {
        // Direct test of the blend formula: (fg * alpha + bg * (256 - alpha)) >> 8
        let fg = Batch::<u32>::splat(255);
        let bg = Batch::<u32>::splat(0);

        // Alpha = 0: should return bg
        let alpha_0 = Batch::<u32>::splat(0);
        let result_0 = blend_channel(fg, bg, alpha_0);
        assert_eq!(result_0.first(), 0, "blend(255, 0, alpha=0) should be 0");

        // Alpha = 255: (255 * 255 + 0 * 1) >> 8 = 65025 >> 8 = 254
        let alpha_255 = Batch::<u32>::splat(255);
        let result_255 = blend_channel(fg, bg, alpha_255);
        assert_eq!(result_255.first(), 254, "blend(255, 0, alpha=255) should be 254");

        // Alpha = 128: should be midpoint
        let alpha_128 = Batch::<u32>::splat(128);
        let result_128 = blend_channel(fg, bg, alpha_128);
        // (255 * 128 + 0 * 128) >> 8 = 32640 >> 8 = 127
        assert_eq!(result_128.first(), 127, "blend(255, 0, alpha=128) should be 127");
    }

    #[test]
    fn pack_unpack_rgba_roundtrips() {
        // Verify our test helper functions work correctly
        let packed = pack_rgba(255, 128, 64, 200);
        let (r, g, b, a) = unpack_rgba(packed);
        assert_eq!((r, g, b, a), (255, 128, 64, 200));

        // Verify byte layout: little-endian 0xAABBGGRR
        assert_eq!(packed, 0xC8_40_80_FF);
    }

    #[test]
    fn u32_pixel_channel_extraction_matches_pack_unpack() {
        // Verify the Pixel trait's channel extraction matches our test helpers
        use crate::Pixel;

        let packed = pack_rgba(200, 100, 50, 255);
        let batch = Batch::<u32>::splat(packed);

        let r = <u32 as Pixel>::batch_red(batch).first();
        let g = <u32 as Pixel>::batch_green(batch).first();
        let b = <u32 as Pixel>::batch_blue(batch).first();
        let a = <u32 as Pixel>::batch_alpha(batch).first();

        assert_eq!(r, 200, "batch_red should extract R channel");
        assert_eq!(g, 100, "batch_green should extract G channel");
        assert_eq!(b, 50, "batch_blue should extract B channel");
        assert_eq!(a, 255, "batch_alpha should extract A channel");
    }

    #[test]
    fn u32_pixel_channel_reconstruction_matches_pack() {
        // Verify batch_from_channels produces the same result as pack_rgba
        use crate::Pixel;

        let r = Batch::<u32>::splat(200);
        let g = Batch::<u32>::splat(100);
        let b = Batch::<u32>::splat(50);
        let a = Batch::<u32>::splat(255);

        let reconstructed = <u32 as Pixel>::batch_from_channels(r, g, b, a).first();
        let expected = pack_rgba(200, 100, 50, 255);

        assert_eq!(reconstructed, expected, "batch_from_channels should match pack_rgba");
    }

    #[test]
    fn const_color_surface_returns_correct_value() {
        // Verify ConstColor surface returns the packed value correctly
        let packed = pack_rgba(255, 0, 0, 255);
        let color = ConstColor(packed);
        let result = color.eval(Batch::<u32>::splat(0), Batch::<u32>::splat(0));

        assert_eq!(result.first(), packed, "ConstColor should return the packed value");
    }

    #[test]
    fn const_mask_surface_returns_correct_value() {
        // Verify ConstMask surface returns the alpha value correctly
        let mask = ConstMask(128);
        let result = mask.eval(Batch::<u32>::splat(0), Batch::<u32>::splat(0));

        assert_eq!(result.first(), 128, "ConstMask should return the mask value");
    }

    #[test]
    fn upcast_u8_to_u32_preserves_value() {
        // Verify upcast preserves the byte value in the low bits
        let u8_batch = Batch::<u8>::splat(255);
        let u32_batch = NativeBackend::upcast_u8_to_u32(u8_batch);
        assert_eq!(u32_batch.first(), 255, "upcast_u8_to_u32(255) should be 255");

        let u8_batch = Batch::<u8>::splat(0);
        let u32_batch = NativeBackend::upcast_u8_to_u32(u8_batch);
        assert_eq!(u32_batch.first(), 0, "upcast_u8_to_u32(0) should be 0");

        let u8_batch = Batch::<u8>::splat(128);
        let u32_batch = NativeBackend::upcast_u8_to_u32(u8_batch);
        assert_eq!(u32_batch.first(), 128, "upcast_u8_to_u32(128) should be 128");
    }

    #[test]
    fn over_step_by_step_with_full_alpha() {
        // Trace through Over::eval step by step to isolate the bug
        use crate::Pixel;

        let fg_packed = pack_rgba(255, 0, 0, 255); // Red
        let bg_packed = pack_rgba(0, 0, 255, 255); // Blue
        let alpha_val: u8 = 255;

        // Step 1: Mask eval
        let mask = ConstMask(alpha_val);
        let alpha_u8 = mask.eval(Batch::<u32>::splat(0), Batch::<u32>::splat(0));
        assert_eq!(alpha_u8.first(), 255, "Step 1: Mask should return 255");

        // Step 2: Upcast alpha
        let alpha = NativeBackend::upcast_u8_to_u32(alpha_u8);
        assert_eq!(alpha.first(), 255, "Step 2: Upcast alpha should be 255");

        // Step 3: fg.eval
        let fg_surface = ConstColor(fg_packed);
        let fg_batch = fg_surface.eval(Batch::<u32>::splat(0), Batch::<u32>::splat(0));
        assert_eq!(fg_batch.first(), fg_packed, "Step 3: fg should be red packed");

        // Step 4: P::batch_to_u32 (identity for u32)
        let fg = <u32 as Pixel>::batch_to_u32(fg_batch);
        assert_eq!(fg.first(), fg_packed, "Step 4: batch_to_u32 should be identity");

        // Step 5: Extract channels
        let fg_r = <u32 as Pixel>::batch_red(fg);
        let fg_g = <u32 as Pixel>::batch_green(fg);
        let fg_b = <u32 as Pixel>::batch_blue(fg);
        assert_eq!(fg_r.first(), 255, "Step 5a: fg red should be 255");
        assert_eq!(fg_g.first(), 0, "Step 5b: fg green should be 0");
        assert_eq!(fg_b.first(), 0, "Step 5c: fg blue should be 0");

        // Step 6: bg.eval and extract
        let bg_surface = ConstColor(bg_packed);
        let bg_batch = bg_surface.eval(Batch::<u32>::splat(0), Batch::<u32>::splat(0));
        let bg = <u32 as Pixel>::batch_to_u32(bg_batch);
        let bg_r = <u32 as Pixel>::batch_red(bg);
        let bg_b = <u32 as Pixel>::batch_blue(bg);
        assert_eq!(bg_r.first(), 0, "Step 6a: bg red should be 0");
        assert_eq!(bg_b.first(), 255, "Step 6b: bg blue should be 255");

        // Step 7: Blend red channel
        let result_r = blend_channel(fg_r, bg_r, alpha);
        // (255 * 255 + 0 * 1) >> 8 = 254
        assert_eq!(result_r.first(), 254, "Step 7: blended red should be 254");

        // Step 8: Blend blue channel
        let result_b = blend_channel(fg_b, bg_b, alpha);
        // (0 * 255 + 255 * 1) >> 8 = 0
        assert_eq!(result_b.first(), 0, "Step 8: blended blue should be 0");

        // Step 9: Reconstruct
        let fg_a = <u32 as Pixel>::batch_alpha(fg);
        let bg_a = <u32 as Pixel>::batch_alpha(bg);
        let result_g = blend_channel(fg_g, <u32 as Pixel>::batch_green(bg), alpha);
        let result_a = blend_channel(fg_a, bg_a, alpha);

        let reconstructed = <u32 as Pixel>::batch_from_channels(result_r, result_g, result_b, result_a);
        let (r, g, b, _a) = unpack_rgba(reconstructed.first());

        assert_eq!(r, 254, "Step 9a: final red should be 254");
        assert_eq!(g, 0, "Step 9b: final green should be 0");
        assert_eq!(b, 0, "Step 9c: final blue should be 0");
    }
}
