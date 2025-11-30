use crate::batch::{SimdOps, SimdVec};
use crate::ops::{Max, Mul, Offset, Over, Skew};
use crate::pipe::Surface;
use crate::pixel::Pixel;

/// Extensions for any surface (Coordinate transforms).
pub trait SurfaceExt<T: Copy>: Surface<T> + Sized {
    /// Applies a translation offset to the surface.
    ///
    /// # Parameters
    /// * `dx` - Horizontal offset.
    /// * `dy` - Vertical offset.
    ///
    /// # Returns
    /// * An `Offset` wrapper around this surface.
    fn offset(self, dx: i32, dy: i32) -> Offset<Self> {
        Offset {
            source: self,
            dx,
            dy,
        }
    }

    /// Applies a skew (shear) transformation to the surface.
    ///
    /// # Parameters
    /// * `shear` - The shear factor.
    ///
    /// # Returns
    /// * A `Skew` wrapper around this surface.
    fn skew(self, shear: i32) -> Skew<Self> {
        Skew {
            source: self,
            shear,
        }
    }

    /// Computes the maximum of this surface and another.
    ///
    /// # Parameters
    /// * `other` - The other surface.
    ///
    /// # Returns
    /// * A `Max` composite surface.
    fn max<O>(self, other: O) -> Max<Self, O>
    where
        O: Surface<T>,
        SimdVec<T>: SimdOps<T>,
    {
        Max(self, other)
    }
}

/// Extensions for 8-bit Masks (Blending).
pub trait MaskExt: Surface<u8> + Sized {
    /// Composites a foreground over a background using this surface as a mask (alpha).
    ///
    /// Generic over pixel format `P` for format-aware channel operations.
    /// Both fg and bg must be `Surface<P>` - strongly typed throughout.
    ///
    /// # Type Parameters
    /// * `P` - The pixel format type (e.g., `Rgba`, `Bgra`).
    ///
    /// # Parameters
    /// * `fg` - The foreground surface (must be `Surface<P>`).
    /// * `bg` - The background surface (must be `Surface<P>`).
    ///
    /// # Returns
    /// * An `Over` compositing operation that outputs `Surface<P>`.
    fn over<P, F, B>(self, fg: F, bg: B) -> Over<P, Self, F, B>
    where
        P: Pixel,
        F: Surface<P>,
        B: Surface<P>,
    {
        Over::new(self, fg, bg)
    }

    /// Multiplies a color surface by this mask.
    ///
    /// # Parameters
    /// * `color` - The color surface (must be `Surface<P>`).
    ///
    /// # Returns
    /// * A `Mul` operation that outputs `Surface<P>`.
    fn mul<P, C>(self, color: C) -> Mul<Self, C>
    where
        P: Pixel + Copy,
        C: Surface<P>,
    {
        Mul {
            mask: self,
            color,
        }
    }
}

// Blanket implementations
impl<T: Copy, S: Surface<T>> SurfaceExt<T> for S {}
impl<S: Surface<u8>> MaskExt for S {}
