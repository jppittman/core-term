use crate::pipe::Surface;
use crate::ops::{Offset, Skew, Over};
// Removed invalid `Const` import

/// Extensions for any surface (Coordinate transforms).
pub trait SurfaceExt<T: Copy>: Surface<T> + Sized {
    fn offset(self, dx: i32, dy: i32) -> Offset<Self> {
        Offset { source: self, dx, dy }
    }

    fn skew(self, shear: i32) -> Skew<Self> {
        Skew { source: self, shear }
    }
}

/// Extensions for 8-bit Masks (Blending).
pub trait MaskExt: Surface<u8> + Sized {
    fn over<F, B>(self, fg: F, bg: B) -> Over<Self, F, B> 
    where F: Surface<u32>, B: Surface<u32>
    {
        Over { mask: self, fg, bg }
    }
}

// Blanket implementations
impl<T: Copy, S: Surface<T>> SurfaceExt<T> for S {}
impl<S: Surface<u8>> MaskExt for S {}
