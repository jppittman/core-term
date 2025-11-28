use crate::ops::{Offset, Over, Skew};
use crate::pipe::Surface;
use crate::batch::Batch256;

/// Extensions for any surface (Coordinate transforms).
pub trait SurfaceExt<T: Copy>: Sized {
    /// Applies a translation offset to the surface.
    ///
    /// # Parameters
    /// * `dx` - Horizontal offset.
    /// * `dy` - Vertical offset.
    ///
    /// # Returns
    /// * An `Offset` wrapper around this surface.
    fn offset(self, dx: i32, dy: i32) -> Offset<Self>;

    /// Applies a skew (shear) transformation to the surface.
    ///
    /// # Parameters
    /// * `shear` - The shear factor.
    ///
    /// # Returns
    /// * A `Skew` wrapper around this surface.
    fn skew(self, shear: i32) -> Skew<Self>;
}

impl<T: Copy, S> SurfaceExt<T> for S
where
    S: Surface<Batch256<u32>, T>,
{
    fn offset(self, dx: i32, dy: i32) -> Offset<Self> {
        Offset {
            source: self,
            dx,
            dy,
        }
    }

    fn skew(self, shear: i32) -> Skew<Self> {
        Skew {
            source: self,
            shear,
        }
    }
}

/// Extensions for 8-bit Masks (Blending).
pub trait MaskExt: Sized {
    /// Composites a foreground over a background using this surface as a mask (alpha).
    ///
    /// # Parameters
    /// * `fg` - The foreground surface.
    /// * `bg` - The background surface.
    ///
    /// # Returns
    /// * An `Over` compositing operation.
    fn over<F, B>(self, fg: F, bg: B) -> Over<Self, F, B>
    where
        F: Surface<Batch256<u32>, u32>,
        B: Surface<Batch256<u32>, u32>;
}

impl<S> MaskExt for S
where
    S: Surface<Batch256<u32>, u8>,
{
    fn over<F, B>(self, fg: F, bg: B) -> Over<Self, F, B>
    where
        F: Surface<Batch256<u32>, u32>,
        B: Surface<Batch256<u32>, u32>,
    {
        Over { mask: self, fg, bg }
    }
}
