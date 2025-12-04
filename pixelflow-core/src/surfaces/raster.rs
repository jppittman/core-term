use crate::backend::{Backend, SimdBatch};
use crate::batch::{Batch, NativeBackend};
use crate::traits::Surface;
use core::fmt::Debug;

/// Adapts a Continuous Surface (f32) to a Discrete Surface (u32).
/// Samples at the pixel center (x + 0.5, y + 0.5).
#[derive(Copy, Clone)]
pub struct Rasterize<S>(pub S);

impl<S, P> Surface<P, u32> for Rasterize<S>
where
    S: Surface<P, f32>,
    P: Copy + Debug + Default + PartialEq + Send + Sync + 'static,
{
    #[inline(always)]
    fn eval(&self, x: Batch<u32>, y: Batch<u32>) -> Batch<P> {
        // Bridge: Convert Integer Grid -> Continuous Space
        // Offset by 0.5 to sample pixel centers.
        let fx = NativeBackend::u32_to_f32(x) + Batch::<f32>::splat(0.5);
        let fy = NativeBackend::u32_to_f32(y) + Batch::<f32>::splat(0.5);

        self.0.eval(fx, fy)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{Backend, BatchArithmetic};

    struct Circle {
        radius_sq: f32,
    }

    impl Surface<u32, f32> for Circle {
        fn eval(&self, x: Batch<f32>, y: Batch<f32>) -> Batch<u32> {
            let r2 = Batch::<f32>::splat(self.radius_sq);
            let d2 = x * x + y * y;
            // Compare d2 < r2.
            let mask = d2.cmp_lt(r2);
            // Select 1.0 (inside) or 0.0 (outside) then convert to u32
            let one = Batch::<f32>::splat(1.0);
            let zero = Batch::<f32>::splat(0.0);

            let res = mask.select(one, zero);
            NativeBackend::f32_to_u32(res)
        }
    }

    #[test]
    fn test_rasterize_circle() {
        let circle = Circle { radius_sq: 100.0 }; // r=10
        let rasterized = Rasterize(circle);

        // Test center (0,0) -> (0.5, 0.5) -> dist^2 = 0.5 -> < 100 -> 1
        let val_center = rasterized.eval_one(0, 0);
        assert_eq!(val_center, 1);

        // Test outside (10, 10) -> (10.5, 10.5) -> dist^2 = ~220 -> > 100 -> 0
        let val_outside = rasterized.eval_one(10, 10);
        assert_eq!(val_outside, 0);

        // Test near boundary
        // (9, 0) -> (9.5, 0.5) -> 90.25 + 0.25 = 90.5 < 100 -> 1
        assert_eq!(rasterized.eval_one(9, 0), 1);
    }
}
