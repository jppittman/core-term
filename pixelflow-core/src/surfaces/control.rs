use crate::backend::SimdBatch;
use crate::batch::Batch;
use crate::traits::Manifold;
use crate::pixel::Pixel;
use core::marker::PhantomData;

#[derive(Copy, Clone)]
pub struct Select<M, T, F> {
    pub mask: M,
    pub if_true: T,
    pub if_false: F,
}

impl<P, M, T, F> Manifold<P> for Select<M, T, F>
where
    P: Pixel + PartialEq,
    M: Manifold<u32>, // Mask must return u32 (0 or !0 usually, but select uses bitwise)
    T: Manifold<P>,
    F: Manifold<P>,
{
    #[inline(always)]
    fn eval(&self, x: Batch<u32>, y: Batch<u32>, z: Batch<u32>, w: Batch<u32>) -> Batch<P> {
        let m = self.mask.eval(x, y, z, w);
        if m.all() {
            return self.if_true.eval(x, y, z, w)
        }
        if !m.any() {
            return self.if_false.eval(x, y, z, w)
        }

        // Use church combinator for lazy evaluation / short-circuiting
        let res_u32 = m.church(
            || P::batch_to_u32(self.if_true.eval(x, y, z, w)),
            || P::batch_to_u32(self.if_false.eval(x, y, z, w)),
        );
        P::batch_from_u32(res_u32)
    }
}

pub struct Fix<P, F> {
    func: F,
    _marker: PhantomData<P>,
}

impl<P, F> Fix<P, F> {
    pub fn new(func: F) -> Self {
        Self {
            func,
            _marker: PhantomData,
        }
    }
}

impl<P, F> Manifold<P> for Fix<P, F>
where
    P: Pixel + PartialEq,
    F: Fn(&dyn Manifold<P>, Batch<u32>, Batch<u32>, Batch<u32>, Batch<u32>) -> Batch<P> + Send + Sync,
{
    #[inline(always)]
    fn eval(&self, x: Batch<u32>, y: Batch<u32>, z: Batch<u32>, w: Batch<u32>) -> Batch<P> {
        (self.func)(self, x, y, z, w)
    }
}
