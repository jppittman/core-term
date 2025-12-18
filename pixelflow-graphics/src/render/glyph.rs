//! Embedded font access and subpixel rendering for pixelflow-render.
//!
//! The pixelflow way: get the font, call `font.glyph(char, size)` to get a `Surface<u32>`,
//! then compose it however you need using standard Surface combinators.
//! Glyphs output white pixels with coverage in the alpha channel (R=G=B=255, A=coverage).
//!
//! For subpixel antialiasing, use [`subpixel`] to wrap a glyph mask.

use crate::fonts::Font;
use pixelflow_core::backend::{Backend, BatchArithmetic, FloatBatchOps, SimdBatch};
use pixelflow_core::batch::{Batch, NativeBackend};
use pixelflow_core::pixel::Pixel;
use pixelflow_core::traits::Manifold;
use std::sync::OnceLock;

// Gamma constants
const GAMMA: f32 = 2.2;
const INV_GAMMA: f32 = 1.0 / 2.2;

// Embedded font data (Noto Sans Mono)
static FONT_DATA: &[u8] = include_bytes!("../../assets/NotoSansMono-Regular.ttf");

static FONT: OnceLock<Font<'static>> = OnceLock::new();

/// Get the embedded monospace font.
///
/// # Example
/// ```ignore
/// use pixelflow_render::font;
/// use pixelflow_core::dsl::MaskExt;
///
/// let f = font();
/// let glyph = f.glyph('A', 24.0).unwrap();  // Surface<u8>
/// let blended = glyph.over(fg_color, bg_surface);
/// execute(blended, &mut target);
/// ```
pub fn font() -> &'static Font<'static> {
    FONT.get_or_init(|| Font::from_bytes(FONT_DATA).expect("Failed to parse embedded font"))
}

/// Maps a 3x-width mask to RGBA by sampling adjacent pixels as R, G, B.
///
/// This combinator takes a mask rendered at 3x horizontal resolution and
/// packs three adjacent samples into the R, G, B channels of the output.
/// This produces LCD subpixel antialiasing.
#[derive(Copy, Clone)]
pub struct SubpixelMap<S> {
    source: S,
}

impl<S> SubpixelMap<S> {
    /// Creates a new subpixel mapping combinator.
    pub fn new(source: S) -> Self {
        Self { source }
    }
}

// Implement for Continuous Source (f32) -> Discrete Output (u32)
impl<S> Manifold<u32> for SubpixelMap<S>
where
    S: Manifold<u32>,
{
    #[inline(always)]
    fn eval(&self, x: Batch<u32>, y: Batch<u32>, z: Batch<u32>, w: Batch<u32>) -> Batch<u32> {
        // Convert integer coordinates to float pixel centers
        let x_f = NativeBackend::u32_to_f32(x) + Batch::<f32>::splat(0.5);
        let y_f = NativeBackend::u32_to_f32(y) + Batch::<f32>::splat(0.5);

        let third = Batch::<f32>::splat(1.0 / 3.0);

        // Sample at subpixel centers: -1/3, 0, +1/3 relative to pixel center
        // (Corresponds to 1/6, 3/6, 5/6 in [0, 1] space)
        let r_pos_f = x_f - third;
        let g_pos_f = x_f;
        let b_pos_f = x_f + third;

        // Convert back to u32 coordinates for eval
        let r_pos = NativeBackend::f32_to_u32(r_pos_f);
        let g_pos = NativeBackend::f32_to_u32(g_pos_f);
        let b_pos = NativeBackend::f32_to_u32(b_pos_f);
        let y_u32 = NativeBackend::f32_to_u32(y_f);

        // Source pixels have coverage in alpha channel
        let r_pixel = self.source.eval(r_pos, y_u32, z, w);
        let g_pixel = self.source.eval(g_pos, y_u32, z, w);
        let b_pixel = self.source.eval(b_pos, y_u32, z, w);

        // Extract alpha (coverage) from each pixel
        let r = <u32 as Pixel>::batch_alpha(r_pixel);
        let g = <u32 as Pixel>::batch_alpha(g_pixel);
        let b = <u32 as Pixel>::batch_alpha(b_pixel);

        <u32 as Pixel>::batch_from_channels(r, g, b, Batch::<u32>::splat(255))
    }
}

/// Wraps a glyph mask for subpixel antialiasing.
///
/// This evaluates the continuous glyph surface at subpixel offsets (-1/3, 0, +1/3)
/// and packs the coverage into R, G, B channels.
///
/// # Example
/// ```ignore
/// let glyph = font.glyph('A', 16.0).unwrap();
/// let subpixel_glyph = subpixel(glyph);  // Surface<u32>
/// ```
pub fn subpixel<S>(source: S) -> SubpixelMap<S> {
    SubpixelMap::new(source)
}

/// Decodes a packed RGBA pixel from sRGB to linear color space.
///
/// Applies gamma expansion (pow 2.2) to RGB channels. Alpha is preserved.
/// Use this as a Map function: `Map::new(background, gamma_decode)`
#[inline(always)]
pub fn gamma_decode(pixel: Batch<u32>) -> Batch<u32> {
    let scale = Batch::<f32>::splat(1.0 / 255.0);
    let unscale = Batch::<f32>::splat(255.0);
    let gamma = Batch::<f32>::splat(GAMMA);

    // Extract channels as u32
    let r_u32 = <u32 as Pixel>::batch_red(pixel);
    let g_u32 = <u32 as Pixel>::batch_green(pixel);
    let b_u32 = <u32 as Pixel>::batch_blue(pixel);
    let a_u32 = <u32 as Pixel>::batch_alpha(pixel);

    // Convert to float [0, 1]
    let r_f = NativeBackend::u32_to_f32(r_u32) * scale;
    let g_f = NativeBackend::u32_to_f32(g_u32) * scale;
    let b_f = NativeBackend::u32_to_f32(b_u32) * scale;

    // Apply gamma expansion (sRGB -> linear)
    let r_lin = r_f.pow(gamma);
    let g_lin = g_f.pow(gamma);
    let b_lin = b_f.pow(gamma);

    // Convert back to u32 [0, 255]
    let r_out = NativeBackend::f32_to_u32(r_lin * unscale);
    let g_out = NativeBackend::f32_to_u32(g_lin * unscale);
    let b_out = NativeBackend::f32_to_u32(b_lin * unscale);

    <u32 as Pixel>::batch_from_channels(r_out, g_out, b_out, a_u32)
}

/// Encodes a packed RGBA pixel from linear to sRGB color space.
///
/// Applies gamma compression (pow 1/2.2) to RGB channels. Alpha is preserved.
/// Use this as a Map function: `Map::new(blended, gamma_encode)`
#[inline(always)]
pub fn gamma_encode(pixel: Batch<u32>) -> Batch<u32> {
    let scale = Batch::<f32>::splat(1.0 / 255.0);
    let unscale = Batch::<f32>::splat(255.0);
    let inv_gamma = Batch::<f32>::splat(INV_GAMMA);

    // Extract channels as u32
    let r_u32 = <u32 as Pixel>::batch_red(pixel);
    let g_u32 = <u32 as Pixel>::batch_green(pixel);
    let b_u32 = <u32 as Pixel>::batch_blue(pixel);
    let a_u32 = <u32 as Pixel>::batch_alpha(pixel);

    // Convert to float [0, 1]
    let r_f = NativeBackend::u32_to_f32(r_u32) * scale;
    let g_f = NativeBackend::u32_to_f32(g_u32) * scale;
    let b_f = NativeBackend::u32_to_f32(b_u32) * scale;

    // Apply gamma compression (linear -> sRGB)
    let r_srgb = r_f.pow(inv_gamma);
    let g_srgb = g_f.pow(inv_gamma);
    let b_srgb = b_f.pow(inv_gamma);

    // Convert back to u32 [0, 255]
    let r_out = NativeBackend::f32_to_u32(r_srgb * unscale);
    let g_out = NativeBackend::f32_to_u32(g_srgb * unscale);
    let b_out = NativeBackend::f32_to_u32(b_srgb * unscale);

    <u32 as Pixel>::batch_from_channels(r_out, g_out, b_out, a_u32)
}

/// Blends foreground with background using per-channel (subpixel) alpha values.
///
/// The mask is a `Manifold<P>` at 3x horizontal resolution. The alpha channel
/// of each mask pixel provides the coverage value. This combinator samples
/// three adjacent mask pixels and uses their alphas for R, G, B blending.
///
/// For gamma-correct rendering, wrap the background with `Map::new(bg, gamma_decode)`
/// and the output with `Map::new(blend, gamma_encode)`.
#[derive(Copy, Clone)]
pub struct SubpixelBlend<P, M, B> {
    /// The 3x-width mask (alpha channel is coverage).
    mask: M,
    /// The foreground color.
    fg: P,
    /// The background surface.
    bg: B,
}

impl<P, M, B> SubpixelBlend<P, M, B> {
    /// Creates a new subpixel blend combinator.
    ///
    /// # Arguments
    /// * `mask` - A surface at 3x horizontal resolution (alpha = coverage)
    /// * `fg` - Foreground color
    /// * `bg` - Background surface
    pub fn new(mask: M, fg: P, bg: B) -> Self {
        Self { mask, fg, bg }
    }
}

#[inline(always)]
fn blend_channel(fg: Batch<u32>, bg: Batch<u32>, alpha: Batch<u32>) -> Batch<u32> {
    let inv_alpha = Batch::<u32>::splat(256) - alpha;
    ((fg * alpha) + (bg * inv_alpha)) >> 8
}

impl<P, M, B> Manifold<P> for SubpixelBlend<P, M, B>
where
    P: Pixel,
    M: Manifold<P>, // This expects u32 coordinates?
    B: Manifold<P>,
{
    #[inline(always)]
    fn eval(&self, x: Batch<u32>, y: Batch<u32>, z: Batch<u32>, w: Batch<u32>) -> Batch<P> {
        let x3 = x * 3;

        // Sample mask at 3 subpixel positions, extract alpha channel
        let alpha_r = P::batch_alpha(P::batch_to_u32(self.mask.eval(x3, y, z, w)));
        let alpha_g = P::batch_alpha(P::batch_to_u32(self.mask.eval(x3 + 1, y, z, w)));
        let alpha_b = P::batch_alpha(P::batch_to_u32(self.mask.eval(x3 + 2, y, z, w)));

        // Extract foreground channels
        let fg_packed = P::batch_to_u32(Batch::<P>::splat(self.fg));
        let fg_r = P::batch_red(fg_packed);
        let fg_g = P::batch_green(fg_packed);
        let fg_b = P::batch_blue(fg_packed);
        let fg_a = P::batch_alpha(fg_packed);

        // Sample and extract background channels
        let bg_packed = P::batch_to_u32(self.bg.eval(x, y, z, w));
        let bg_r = P::batch_red(bg_packed);
        let bg_g = P::batch_green(bg_packed);
        let bg_b = P::batch_blue(bg_packed);
        let bg_a = P::batch_alpha(bg_packed);

        // Blend each channel with its own alpha
        let r = blend_channel(fg_r, bg_r, alpha_r);
        let g = blend_channel(fg_g, bg_g, alpha_g);
        let b = blend_channel(fg_b, bg_b, alpha_b);

        // Alpha: use max of the three subpixel alphas
        let alpha_max = alpha_r.max(alpha_g).max(alpha_b);
        let a = blend_channel(fg_a, bg_a, alpha_max);

        P::batch_from_u32(P::batch_from_channels(r, g, b, a))
    }
}
