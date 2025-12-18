use crate::batch::Batch;
use crate::bitwise::Bitwise;
use crate::traits::Manifold;
use core::fmt::Debug;
use core::marker::PhantomData;

/// A surface defined by a closure.
///
/// This is one of the Six Eigenshaders: `Compute`.
/// It serves as an escape hatch for arbitrary logic.
#[derive(Copy, Clone)]
pub struct Compute<F, T> {
    pub func: F,
    pub _marker: PhantomData<T>,
}

impl<F, T> Compute<F, T> {
    pub fn new(func: F) -> Self {
        Self {
            func,
            _marker: PhantomData,
        }
    }
}

impl<F, T, C> Manifold<T, C> for Compute<F, T>
where
    T: Copy + Debug + Default + PartialEq + Send + Sync + 'static,
    C: Copy + Debug + Default + PartialEq + Send + Sync + 'static,
    F: Fn(Batch<C>, Batch<C>, Batch<C>, Batch<C>) -> Batch<T> + Send + Sync,
{
    #[inline(always)]
    fn eval(&self, x: Batch<C>, y: Batch<C>, z: Batch<C>, w: Batch<C>) -> Batch<T> {
        (self.func)(x, y, z, w)
    }
}

/// Type alias for `Compute` - deprecated.
///
/// This type alias exists for backward compatibility. New code should use `Compute` directly.
#[deprecated(since = "0.1.0", note = "Use Compute instead")]
pub type FnSurface<T, F> = Compute<F, T>;

/// A surface that is backed by a buffer of data.
///
/// This is typically used for textures or cached surfaces.
#[derive(Clone)]
pub struct Baked<T: Bitwise> {
    data: alloc::sync::Arc<[T]>,
    width: u32,
    height: u32,
}

impl<T: Bitwise> Baked<T>
where
    T: Copy + Default + Debug + PartialEq + Send + Sync + 'static,
{
    /// Creates a new `Baked` surface from an existing buffer of data.
    pub fn from_data(data: alloc::sync::Arc<[T]>, width: u32, height: u32) -> Self {
        Self {
            data,
            width,
            height,
        }
    }

    /// Returns the width of the baked surface.
    #[inline]
    pub fn width(&self) -> u32 {
        self.width
    }
    /// Returns the height of the baked surface.
    #[inline]
    pub fn height(&self) -> u32 {
        self.height
    }
    /// Returns the raw pixel data.
    #[inline]
    pub fn data(&self) -> &[T] {
        &self.data
    }
    /// Returns a mutable reference to the raw pixel data.
    ///
    /// # Panics
    /// Panics if there is more than one reference to the internal data.
    #[inline]
    pub fn data_mut(&mut self) -> &mut [T] {
        alloc::sync::Arc::get_mut(&mut self.data).unwrap()
    }
}

impl<T: Bitwise> Manifold<T> for Baked<T>
where
    T: Copy + Default + Debug + PartialEq + Send + Sync + 'static,
{
    #[inline(always)]
    fn eval(&self, x: Batch<u32>, y: Batch<u32>, _z: Batch<u32>, _w: Batch<u32>) -> Batch<T> {
        let w = self.width;
        let h = self.height;

        let w_batch = Batch::<u32>::splat(w);
        let h_batch = Batch::<u32>::splat(h);

        let x_mod = x - (x / w_batch) * w_batch;
        let y_mod = y - (y / h_batch) * h_batch;

        let idx = (y_mod * w_batch) + x_mod;

        // Use generic gather from Bitwise
        T::batch_gather(&self.data, idx)
    }
}

impl<T: Bitwise> Manifold<T> for &Baked<T>
where
    T: Copy + Default + Debug + PartialEq + Send + Sync + 'static,
{
    #[inline(always)]
    fn eval(&self, x: Batch<u32>, y: Batch<u32>, z: Batch<u32>, w: Batch<u32>) -> Batch<T> {
        (*self).eval(x, y, z, w)
    }
}
