use crate::backend::SimdBatch;
use crate::batch::Batch;
use crate::traits::{Manifold, BatchSelect};
use core::fmt::Debug;
use core::marker::PhantomData;

/// The `Select` eigenshader: branchless conditional.
///
/// `if mask { if_true } else { if_false }`
#[derive(Copy, Clone)]
pub struct Select<M, T, F> {
    pub mask: M,
    pub if_true: T,
    pub if_false: F,
}

impl<Val, M, T, F, Coord> Manifold<Val, Coord> for Select<M, T, F>
where
    Val: Copy + Debug + Default + PartialEq + Send + Sync + 'static,
    Coord: Copy + Debug + Default + PartialEq + Send + Sync + 'static,
    M: Manifold<bool, Coord>,
    T: Manifold<Val, Coord>,
    F: Manifold<Val, Coord>,
    Batch<bool>: BatchSelect<Batch<Val>>, // Require batch mask selection
{
    #[inline(always)]
    fn eval(&self, x: Batch<Coord>, y: Batch<Coord>, z: Batch<Coord>, w: Batch<Coord>) -> Batch<Val> {
        let mask = self.mask.eval(x, y, z, w);

        // Short-circuit optimization
        if mask.all() {
            return self.if_true.eval(x, y, z, w);
        }
        if !mask.any() {
            return self.if_false.eval(x, y, z, w);
        }

        let t_val = self.if_true.eval(x, y, z, w);
        let f_val = self.if_false.eval(x, y, z, w);

        // Use the trait to perform selection
        mask.select(t_val, f_val)
    }
}

/// The `Fix` eigenshader: Iteration is a Dimension.
///
/// Constructs a Volume (here Manifold) where the Z-axis represents iteration count.
/// W-axis is ignored (or treated as invariant).
///
/// `z` must be integer-like.
#[derive(Copy, Clone)]
pub struct Fix<Val, Step, Coord> {
    pub seed: Val,
    pub step: Step,
    _marker: PhantomData<Coord>,
}

impl<Val, Step, Coord> Fix<Val, Step, Coord> {
    pub fn new(seed: Val, step: Step) -> Self {
        Self {
            seed,
            step,
            _marker: PhantomData,
        }
    }
}

// Specialization for u32 coordinates where we can actually loop
impl<Val, Step> Manifold<Val, u32> for Fix<Val, Step, u32>
where
    Val: Copy + Debug + Default + PartialEq + Send + Sync + 'static,
    Step: Fn(Batch<Val>, Batch<u32>, Batch<u32>) -> Batch<Val> + Send + Sync,
{
    #[inline(always)]
    fn eval(&self, x: Batch<u32>, y: Batch<u32>, z: Batch<u32>, _w: Batch<u32>) -> Batch<Val> {
        let mut state = Batch::<Val>::splat(self.seed);

        // Fast path: take first lane iteration count from Z.
        let iterations = z.extract_lane(0);

        for _ in 0..iterations {
            state = (self.step)(state, x, y);
        }

        state
    }
}

// Specialization for f32 coordinates
impl<Val, Step> Manifold<Val, f32> for Fix<Val, Step, f32>
where
    Val: Copy + Debug + Default + PartialEq + Send + Sync + 'static,
    Step: Fn(Batch<Val>, Batch<f32>, Batch<f32>) -> Batch<Val> + Send + Sync,
{
    #[inline(always)]
    fn eval(&self, x: Batch<f32>, y: Batch<f32>, z: Batch<f32>, _w: Batch<f32>) -> Batch<Val> {
        let mut state = Batch::<Val>::splat(self.seed);
        // Cast float z to int count
        let iterations = z.extract_lane(0) as u32;

        for _ in 0..iterations {
            state = (self.step)(state, x, y);
        }

        state
    }
}
