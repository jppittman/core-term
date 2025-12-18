use crate::batch::Batch;
use crate::traits::Manifold;

/// A constant surface that returns the same value everywhere.
///
/// This is useful for representing solid colors, constant values, or any
/// data that doesn't vary with position. The value is stored as a scalar
/// and automatically promoted to a batch when evaluated.
///
/// # Example
/// ```ignore
/// use pixelflow_core::surfaces::Constant;
///
/// // A solid red color (RGBA)
/// let red = Constant::new(0xFF0000FF_u32);
///
/// // A constant distance field offset
/// let offset = Constant::new(5.0_f32);
/// ```
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct Constant<T> {
    value: T,
}

impl<T> Constant<T> {
    /// Creates a new constant surface with the given value.
    #[inline(always)]
    pub const fn new(value: T) -> Self {
        Self { value }
    }

    /// Gets the constant value.
    #[inline(always)]
    pub const fn get(&self) -> T
    where
        T: Copy,
    {
        self.value
    }
}

impl<T, C> Manifold<T, C> for Constant<T>
where
    T: Copy + core::fmt::Debug + Default + PartialEq + Send + Sync + 'static,
    C: Copy + core::fmt::Debug + Default + PartialEq + Send + Sync + 'static,
    Batch<T>: crate::backend::SimdBatch<T>,
{
    #[inline(always)]
    fn eval(&self, _x: Batch<C>, _y: Batch<C>, _z: Batch<C>, _w: Batch<C>) -> Batch<T> {
        // Use splat to convert scalar to batch
        use crate::backend::SimdBatch;
        Batch::<T>::splat(self.value)
    }
}
