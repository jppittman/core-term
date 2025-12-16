use crate::batch::Batch;
use crate::surfaces::{Map, Max, Mul, Offset, Select, Skew};
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
    /// Returns `self` where `mask` is true, and `Default::default()` (Zero) where `mask` is false.
    ///
    /// This relies on `Select`.
    fn clip<M>(self, mask: M) -> Select<M, Self, T>
    where
        M: Surface<bool>, // Mask must be boolean
    {
        // We use T's default value (impl Surface for T) as the "Empty" surface
        Select {
            mask,
            if_true: self,
            if_false: T::default(),
        }
    }

    /// Computes the maximum of this surface and another.
    fn max<O>(self, other: O) -> Max<Self, O>
    where
        O: Surface<T>,
    {
        Max(self, other)
    }

    /// Multiplies this surface by another (lane-wise).
    fn mul<O>(self, other: O) -> Mul<Self, O>
    where
        O: Surface<T>,
    {
        Mul { a: self, b: other }
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

// Blanket implementations
impl<T, S: Surface<T>> SurfaceExt<T> for S where
    T: Copy + Debug + Default + PartialEq + Send + Sync + 'static
{
}
