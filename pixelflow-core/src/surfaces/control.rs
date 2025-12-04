use crate::backend::SimdBatch;
use crate::batch::Batch;
use crate::pixel::Pixel;
use crate::traits::Surface;
use core::marker::PhantomData;

/// Selects between two surfaces based on a mask.
#[derive(Copy, Clone)]
pub struct Select<M, T, F> {
    /// The mask surface.
    pub mask: M,
    /// The surface to evaluate if the mask is true.
    pub if_true: T,
    /// The surface to evaluate if the mask is false.
    pub if_false: F,
}

impl<P, M, T, F> Surface<P> for Select<M, T, F>
where
    P: Pixel + PartialEq,
    M: Surface<u32>, // Mask must return u32 (0 or !0 usually, but select uses bitwise)
    T: Surface<P>,
    F: Surface<P>,
{
    #[inline(always)]
    fn eval(&self, x: Batch<u32>, y: Batch<u32>) -> Batch<P> {
        let m = self.mask.eval(x, y);
        if m.all() {
            return self.if_true.eval(x, y);
        }
        if !m.any() {
            return self.if_false.eval(x, y);
        }

        // Use church combinator for lazy evaluation / short-circuiting
        let res_u32 = m.church(
            || P::batch_to_u32(self.if_true.eval(x, y)),
            || P::batch_to_u32(self.if_false.eval(x, y)),
        );
        P::batch_from_u32(res_u32)
    }
}

/// A fixed-point combinator for recursive surfaces.
pub struct Fix<P, F> {
    func: F,
    _marker: PhantomData<P>,
}

impl<P, F> Fix<P, F> {
    /// Creates a new `Fix` combinator.
    pub fn new(func: F) -> Self {
        Self {
            func,
            _marker: PhantomData,
        }
    }
}

impl<P, F> Surface<P> for Fix<P, F>
where
    P: Pixel + PartialEq,
    F: Fn(&dyn Surface<P>, Batch<u32>, Batch<u32>) -> Batch<P> + Send + Sync,
{
    #[inline(always)]
    fn eval(&self, x: Batch<u32>, y: Batch<u32>) -> Batch<P> {
        (self.func)(self, x, y)
    }
}
