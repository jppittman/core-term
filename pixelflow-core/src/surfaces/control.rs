use crate::backend::{BatchArithmetic, SimdBatch};
use crate::batch::Batch;
use crate::bitwise::Bitwise;
use crate::traits::Manifold;
use core::fmt::Debug;
use core::marker::PhantomData;

#[derive(Copy, Clone)]
pub struct Select<M, T, F> {
    pub mask: M,
    pub if_true: T,
    pub if_false: F,
}

impl<Val, M, T, F> Manifold<Val> for Select<M, T, F>
where
    Val: Bitwise + Copy + Debug + Default + PartialEq + Send + Sync + 'static,
    M: Manifold<u32>,
    T: Manifold<Val>,
    F: Manifold<Val>,
    Batch<u32>: BatchArithmetic<u32>,
{
    #[inline(always)]
    fn eval(&self, x: Batch<u32>, y: Batch<u32>, z: Batch<u32>, w: Batch<u32>) -> Batch<Val> {
        let m = self.mask.eval(x, y, z, w);

        // Short-circuit optimization
        if m.all() {
            return self.if_true.eval(x, y, z, w);
        }
        if !m.any() {
            return self.if_false.eval(x, y, z, w);
        }

        let t = self.if_true.eval(x, y, z, w);
        let f = self.if_false.eval(x, y, z, w);

        let t_bits = Val::batch_to_bits(t);
        let f_bits = Val::batch_to_bits(f);

        let res_bits = m.select(t_bits, f_bits);

        Val::batch_from_bits(res_bits)
    }
}

pub struct Fix<Val, F> {
    func: F,
    _marker: PhantomData<Val>,
}

impl<Val, F> Fix<Val, F> {
    pub fn new(func: F) -> Self {
        Self {
            func,
            _marker: PhantomData,
        }
    }
}

impl<Val, F> Manifold<Val> for Fix<Val, F>
where
    Val: Copy + Debug + Default + PartialEq + Send + Sync + 'static,
    F: Fn(&dyn Manifold<Val>, Batch<u32>, Batch<u32>, Batch<u32>, Batch<u32>) -> Batch<Val>
        + Send
        + Sync,
{
    #[inline(always)]
    fn eval(&self, x: Batch<u32>, y: Batch<u32>, z: Batch<u32>, w: Batch<u32>) -> Batch<Val> {
        (self.func)(self, x, y, z, w)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::surfaces::Compute;
    use core::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn test_select_short_circuits_all_true() {
        static TRUE_CALLS: AtomicUsize = AtomicUsize::new(0);
        static FALSE_CALLS: AtomicUsize = AtomicUsize::new(0);

        let s = Select {
            mask: Compute::new(|_, _, _, _| Batch::<u32>::splat(0xFFFFFFFF)), // Always true
            if_true: Compute::new(|_, _, _, _| {
                TRUE_CALLS.fetch_add(1, Ordering::SeqCst);
                Batch::<u32>::splat(1u32)
            }),
            if_false: Compute::new(|_, _, _, _| {
                FALSE_CALLS.fetch_add(1, Ordering::SeqCst);
                Batch::<u32>::splat(0u32)
            }),
        };

        let x = Batch::<u32>::splat(0);
        let y = Batch::<u32>::splat(0);
        let z = Batch::<u32>::splat(0);
        let w = Batch::<u32>::splat(0);

        let res = s.eval(x, y, z, w);
        assert_eq!(res.first(), 1);
        assert_eq!(TRUE_CALLS.load(Ordering::SeqCst), 1);
        assert_eq!(FALSE_CALLS.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn test_select_short_circuits_all_false() {
        static TRUE_CALLS_2: AtomicUsize = AtomicUsize::new(0);
        static FALSE_CALLS_2: AtomicUsize = AtomicUsize::new(0);

        let s = Select {
            mask: Compute::new(|_, _, _, _| Batch::<u32>::splat(0x00000000)), // Always false
            if_true: Compute::new(|_, _, _, _| {
                TRUE_CALLS_2.fetch_add(1, Ordering::SeqCst);
                Batch::<u32>::splat(1u32)
            }),
            if_false: Compute::new(|_, _, _, _| {
                FALSE_CALLS_2.fetch_add(1, Ordering::SeqCst);
                Batch::<u32>::splat(0u32)
            }),
        };

        let x = Batch::<u32>::splat(0);
        let y = Batch::<u32>::splat(0);
        let z = Batch::<u32>::splat(0);
        let w = Batch::<u32>::splat(0);

        let res = s.eval(x, y, z, w);
        assert_eq!(res.first(), 0);
        assert_eq!(TRUE_CALLS_2.load(Ordering::SeqCst), 0);
        assert_eq!(FALSE_CALLS_2.load(Ordering::SeqCst), 1);
    }
}
