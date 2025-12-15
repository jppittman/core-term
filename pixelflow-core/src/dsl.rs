use crate::batch::Batch;
use crate::pixel::Pixel;
use crate::surfaces::{Map, Max, Mul, Offset, Over, Skew};
use crate::traits::Surface;
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

    /// Warps the coordinate space of the surface.
    ///
    /// This is the `Warp` eigenshader.
    fn warp<F, C>(self, mapping: F) -> crate::surfaces::Warp<Self, F, C>
    where
        F: Fn(Batch<C>, Batch<C>, Batch<C>, Batch<C>) -> (Batch<C>, Batch<C>, Batch<C>, Batch<C>)
            + Send
            + Sync,
        C: Copy + Debug + Default + PartialEq + Send + Sync + 'static,
    {
        crate::surfaces::Warp::new(self, mapping)
    }

    /// Applies a linear transform to the values of the surface.
    ///
    /// This is the `Grade` eigenshader: `val * slope + bias`.
    fn grade<M, B>(self, slope: M, bias: B) -> crate::surfaces::Grade<Self, M, B> {
        crate::surfaces::Grade::new(self, slope, bias)
    }

    /// Linearly interpolates to another surface.
    ///
    /// This is the `Lerp` eigenshader: `self + t * (other - self)`.
    /// Note: `self` is the 'start' (a), `other` is the 'end' (b).
    fn lerp<Param, Other>(
        self,
        t: Param,
        other: Other,
    ) -> crate::surfaces::Lerp<Param, Self, Other> {
        crate::surfaces::Lerp::new(t, self, other)
    }

    /// Clips the surface using a mask.
    ///
    /// This is a convenience for `Select` where the false branch is an empty/zero surface.
    /// For T=P (Pixel), "empty" usually means transparent black.
    ///
    /// Note: This implementation currently uses `Select` but requires the false branch
    /// to be provided or inferred. Since we don't have a generic "Empty" surface yet,
    /// we'll leave this as a TODO or implement a simple version if possible.
    ///
    /// Actually, let's implement `clip` to take a default value or "zero".
    /// Or better, define `clip` on `SurfaceExt` to just wrap in `Select` with a default.
    /// But generic T doesn't imply `Pixel`.
    /// So for now, let's omit `clip` here or make it require `Default`.

    /*
    fn clip<M>(self, mask: M) -> crate::surfaces::Select<M, Self, Empty<T>> ...
    We don't have Empty<T> yet.
    */

    /// Computes the maximum of this surface and another.
    fn max<O>(self, other: O) -> Max<Self, O>
    where
        O: Surface<T>,
    {
        Max(self, other)
    }

    /// Applies a transformation function to surface output.
    ///
    /// Use for gamma correction, color adjustments, or any per-pixel transform.
    fn map<F>(self, transform: F) -> Map<Self, F>
    where
        F: Fn(Batch<T>) -> Batch<T> + Send + Sync,
    {
        Map::new(self, transform)
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
impl<T, S: Surface<T>> SurfaceExt<T> for S where
    T: Copy + Debug + Default + PartialEq + Send + Sync + 'static
{
}
impl<P: Pixel, S: Surface<P>> MaskExt<P> for S {}
