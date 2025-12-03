use crate::ops::{Max, Mul, Offset, Over, Skew};
use crate::pipe::Surface;
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

/// Extensions for 8-bit Masks (Blending).
pub trait MaskExt: Surface<u8> + Sized {
    /// Composites a foreground over a background using this surface as a mask (alpha).
    fn over<P, F, B>(self, fg: F, bg: B) -> Over<P, Self, F, B>
    where
        P: Pixel,
        F: Surface<P>,
        B: Surface<P>,
    {
        Over::new(self, fg, bg)
    }

    /// Multiplies a color surface by this mask.
    fn mul<P, C>(self, color: C) -> Mul<Self, C>
    where
        P: Pixel + Copy,
        C: Surface<P>,
    {
        Mul { mask: self, color }
    }
}

// Blanket implementations
impl<T, S: Surface<T>> SurfaceExt<T> for S where T: Copy + Debug + Default + PartialEq + Send + Sync + 'static {}
impl<S: Surface<u8>> MaskExt for S {}
