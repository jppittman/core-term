use crate::TensorView;
use crate::backend::{Backend, SimdBatch};
use crate::batch::{Batch, NativeBackend};
use crate::pixel::Pixel;
use crate::traits::Manifold;
use alloc::vec;
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

pub type FnSurface<F, T> = Compute<F, T>;

#[derive(Copy, Clone)]
pub struct SampleAtlas<'a> {
    /// The source texture atlas.
    pub atlas: TensorView<'a, u8>,
    /// The horizontal step size in fixed-point format (16.16).
    pub step_x_fp: u32,
    /// The vertical step size in fixed-point format (16.16).
    pub step_y_fp: u32,
}

impl<'a> Manifold<u8> for SampleAtlas<'a> {
    #[inline(always)]
    fn eval(&self, x: Batch<u32>, y: Batch<u32>, _z: Batch<u32>, _w: Batch<u32>) -> Batch<u8> {
        let u = x * Batch::<u32>::splat(self.step_x_fp);
        let v = y * Batch::<u32>::splat(self.step_y_fp);
        unsafe {
            let res = self.atlas.sample_4bit_bilinear::<NativeBackend>(u, v);
            NativeBackend::downcast_u32_to_u8(res)
        }
    }
}

/// A surface that is pre-rendered (baked) into a buffer.
#[derive(Clone)]
pub struct Baked<P: Pixel> {
    data: alloc::sync::Arc<[P]>,
    width: u32,
    height: u32,
}

impl<P: Pixel> Baked<P> {
    /// Creates a new `Baked` surface by rasterizing the source.
    /// Note: source must be a Manifold (or Surface via blanket).
    pub fn new<S: Manifold<P>>(source: &S, width: u32, height: u32) -> Self {
        let mut data = vec![P::default(); (width as usize) * (height as usize)].into_boxed_slice();
        // Execute expects a Surface, which S implements via blanket impl
        crate::execute(source, &mut data, width as usize, height as usize);
        Self {
            data: alloc::sync::Arc::from(data),
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
    pub fn data(&self) -> &[P] {
        &self.data
    }
    /// Returns a mutable reference to the raw pixel data.
    #[inline]
    pub fn data_mut(&mut self) -> &mut [P] {
        alloc::sync::Arc::get_mut(&mut self.data).unwrap()
    }
}

impl<P: Pixel> Manifold<P> for Baked<P> {
    #[inline(always)]
    fn eval(&self, x: Batch<u32>, y: Batch<u32>, _z: Batch<u32>, _w: Batch<u32>) -> Batch<P> {
        let w = self.width;
        let h = self.height;

        let w_batch = Batch::<u32>::splat(w);
        let h_batch = Batch::<u32>::splat(h);

        let x_mod = x - (x / w_batch) * w_batch;
        let y_mod = y - (y / h_batch) * h_batch;

        let idx = (y_mod * w_batch) + x_mod;

        P::batch_gather(&self.data, idx)
    }
}

impl<P: Pixel> Manifold<P> for &Baked<P> {
    #[inline(always)]
    fn eval(&self, x: Batch<u32>, y: Batch<u32>, z: Batch<u32>, w: Batch<u32>) -> Batch<P> {
        (*self).eval(x, y, z, w)
    }
}
