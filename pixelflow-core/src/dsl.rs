use crate::surfaces::{Max, Mul, Offset, Over, Skew};
use crate::traits::Surface;
use crate::pixel::Pixel;
use core::fmt::Debug;

/// Extensions for any surface (Coordinate transforms).
pub trait SurfaceExt<T>: Surface<T> + Sized
where
    T: Copy + Debug + Default + PartialEq + Send + Sync + 'static,
{
    /// Applies a translation offset to the surface.
    fn offset(self, dx: i32, dy: i32) -> Offset<Self> {
        Offset {
            source: self,
            dx,
            dy,
        }
    }

    /// Applies a skew (shear) transformation to the surface.
    fn skew(self, shear: i32) -> Skew<Self> {
        Skew {
            source: self,
            shear,
        }
    }

    /// Computes the maximum of this surface and another.
    fn max<O>(self, other: O) -> Max<Self, O>
    where
        O: Surface<T>,
    {
        Max(self, other)
    }
}

/// Extensions for pixel Masks (Blending).
///
/// Masks are now `Surface<P>` where coverage is in the alpha channel.
/// This aligns with SIMD batch sizes and eliminates u8/u32 conversions.
pub trait MaskExt<P: Pixel>: Surface<P> + Sized {
    /// Composites a foreground over a background using this surface as a mask.
    ///
    /// The mask's alpha channel is used as coverage for the blend.
    fn over<F, B>(self, fg: F, bg: B) -> Over<P, Self, F, B>
    where
        F: Surface<P>,
        B: Surface<P>,
    {
        Over::new(self, fg, bg)
    }

    /// Multiplies a color surface by this mask's alpha channel.
    fn mul<C>(self, color: C) -> Mul<Self, C>
    where
        C: Surface<P>,
    {
        Mul { mask: self, color }
    }
}

// Blanket implementations
impl<T, S: Surface<T>> SurfaceExt<T> for S where T: Copy + Debug + Default + PartialEq + Send + Sync + 'static {}
impl<P: Pixel, S: Surface<P>> MaskExt<P> for S {}
