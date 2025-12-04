//! Embedded font access and subpixel rendering for pixelflow-render.
//!
//! The pixelflow way: get the font, call `font.glyph(char, size)` to get a `Surface<u8>`,
//! then compose it however you need using standard Surface combinators.
//!
//! For subpixel antialiasing, use [`subpixel`] to wrap a glyph mask.

use pixelflow_core::backend::Backend;
use pixelflow_core::batch::{Batch, NativeBackend};
use pixelflow_core::ops::Scale;
use pixelflow_core::pipe::Surface;
use pixelflow_core::pixel::Pixel;
use pixelflow_fonts::Font;
use std::sync::OnceLock;

// Embedded font data (Noto Sans Mono)
static FONT_DATA: &[u8] = include_bytes!("../assets/NotoSansMono-Regular.ttf");

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

impl<S: Surface<u8>> Surface<u32> for SubpixelMap<S> {
    #[inline(always)]
    fn eval(&self, x: Batch<u32>, y: Batch<u32>) -> Batch<u32> {
        let three = Batch::<u32>::splat(3);
        let one = Batch::<u32>::splat(1);
        let two = Batch::<u32>::splat(2);

        let x3 = x * three;
        let r = NativeBackend::upcast_u8_to_u32(self.source.eval(x3, y));
        let g = NativeBackend::upcast_u8_to_u32(self.source.eval(x3 + one, y));
        let b = NativeBackend::upcast_u8_to_u32(self.source.eval(x3 + two, y));
        let a = Batch::<u32>::splat(255);

        <u32 as Pixel>::batch_from_channels(r, g, b, a)
    }
}

/// Wraps a glyph mask for subpixel antialiasing.
///
/// This scales the mask 3x horizontally and samples adjacent pixels
/// as R, G, B channels, producing LCD subpixel rendering.
///
/// # Example
/// ```ignore
/// let glyph = font.glyph('A', 16.0).unwrap();
/// let subpixel_glyph = subpixel(glyph);  // Surface<u32>
/// ```
pub fn subpixel<S: Surface<u8>>(source: S) -> SubpixelMap<Scale<S>> {
    SubpixelMap::new(Scale::new(source, 3.0, 1.0))
}
