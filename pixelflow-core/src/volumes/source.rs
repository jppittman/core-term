use crate::batch::Batch;
use crate::traits::{Surface, Volume};
use core::fmt::Debug;

/// Wraps a closure as a Volume.
///
/// Similar to FnSurface but for 3D coordinates.
///
/// # Example
/// ```
/// use pixelflow_core::volumes::FnVolume;
/// use pixelflow_core::batch::Batch;
///
/// // 3D checkerboard pattern
/// let volume = FnVolume::new(|x: Batch<u32>, y: Batch<u32>, z: Batch<u32>| {
///     ((x ^ y ^ z) & Batch::splat(1)).cmp_eq(Batch::splat(0))
/// });
/// ```
pub struct FnVolume<F, T, C = u32>
where
    T: Copy + Debug + Default + PartialEq + Send + Sync + 'static,
    C: Copy + Debug + Default + PartialEq + Send + Sync + 'static,
    F: Fn(Batch<C>, Batch<C>, Batch<C>) -> Batch<T> + Send + Sync,
{
    func: F,
    _phantom_t: core::marker::PhantomData<T>,
    _phantom_c: core::marker::PhantomData<C>,
}

impl<F, T, C> FnVolume<F, T, C>
where
    T: Copy + Debug + Default + PartialEq + Send + Sync + 'static,
    C: Copy + Debug + Default + PartialEq + Send + Sync + 'static,
    F: Fn(Batch<C>, Batch<C>, Batch<C>) -> Batch<T> + Send + Sync,
{
    /// Create a new FnVolume from a closure.
    pub fn new(func: F) -> Self {
        Self {
            func,
            _phantom_t: core::marker::PhantomData,
            _phantom_c: core::marker::PhantomData,
        }
    }
}

// Volume implementation - 3D evaluation
impl<F, T, C> Volume<T, C> for FnVolume<F, T, C>
where
    T: Copy + Debug + Default + PartialEq + Send + Sync + 'static,
    C: Copy + Debug + Default + PartialEq + Send + Sync + 'static,
    F: Fn(Batch<C>, Batch<C>, Batch<C>) -> Batch<T> + Send + Sync,
{
    #[inline(always)]
    fn eval(&self, x: Batch<C>, y: Batch<C>, z: Batch<C>) -> Batch<T> {
        (self.func)(x, y, z)
    }
}

// Surface implementation - 2D projection at z=0 (u32 coordinates only)
impl<F, T> Surface<T, u32> for FnVolume<F, T, u32>
where
    T: Copy + Debug + Default + PartialEq + Send + Sync + 'static,
    F: Fn(Batch<u32>, Batch<u32>, Batch<u32>) -> Batch<T> + Send + Sync,
{
    #[inline(always)]
    fn eval(&self, x: Batch<u32>, y: Batch<u32>) -> Batch<T> {
        // Project 3D -> 2D by evaluating at z=0
        let z = Batch::<u32>::splat(0);
        (self.func)(x, y, z)
    }
}

// Surface implementation - 2D projection at z=0.0 (f32 coordinates)
impl<F, T> Surface<T, f32> for FnVolume<F, T, f32>
where
    T: Copy + Debug + Default + PartialEq + Send + Sync + 'static,
    F: Fn(Batch<f32>, Batch<f32>, Batch<f32>) -> Batch<T> + Send + Sync,
{
    #[inline(always)]
    fn eval(&self, x: Batch<f32>, y: Batch<f32>) -> Batch<T> {
        // Project 3D -> 2D by evaluating at z=0.0
        let z = Batch::<f32>::splat(0.0);
        (self.func)(x, y, z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{BatchArithmetic, SimdBatch};

    #[test]
    fn test_fn_volume_3d_checkerboard() {
        let volume = FnVolume::new(|x: Batch<u32>, y: Batch<u32>, z: Batch<u32>| {
            use crate::backend::BatchArithmetic;
            ((x ^ y ^ z) & Batch::<u32>::splat(1)).cmp_eq(Batch::<u32>::splat(0))
        });

        // Test 3D evaluation (using Volume trait - 3 parameters)
        let x = Batch::<u32>::splat(0);
        let y = Batch::<u32>::splat(0);
        let z = Batch::<u32>::splat(0);
        let result = Volume::eval(&volume, x, y, z);

        // (0,0,0) -> 0^0^0 = 0 -> (0 & 1) = 0 -> 0 == 0 -> true
        assert!(result.all());
    }

    #[test]
    fn test_fn_volume_as_surface() {
        let volume = FnVolume::new(|x: Batch<u32>, y: Batch<u32>, z: Batch<u32>| {
            x + y + z
        });

        // Test 2D projection (z=0)
        let x = Batch::<u32>::splat(5);
        let y = Batch::<u32>::splat(3);
        let result = <FnVolume<_, _> as Surface<u32, u32>>::eval(&volume, x, y);

        // At z=0: 5 + 3 + 0 = 8
        let mut output = [0u32; 4];
        SimdBatch::store(&result, &mut output);
        assert_eq!(output, [8, 8, 8, 8]);
    }

    #[test]
    fn test_volume_trait_from_fn() {
        let vol = FnVolume::new(|x: Batch<u32>, y: Batch<u32>, z: Batch<u32>| {
            x * y * z
        });

        let x = Batch::<u32>::splat(2);
        let y = Batch::<u32>::splat(3);
        let z = Batch::<u32>::splat(4);
        let result = Volume::eval(&vol, x, y, z);

        let mut output = [0u32; 4];
        SimdBatch::store(&result, &mut output);
        assert_eq!(output, [24, 24, 24, 24]);
    }
}
