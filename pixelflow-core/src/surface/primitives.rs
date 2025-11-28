use super::Surface;
use crate::batch::Batch;

/// The X coordinate as a surface.
#[derive(Copy, Clone)]
pub struct X;

impl Surface for X {
    type Output = Batch<f32>;

    #[inline(always)]
    fn sample(&self, u: Batch<f32>, _v: Batch<f32>) -> Batch<f32> {
        u
    }
}

/// The Y coordinate as a surface.
#[derive(Copy, Clone)]
pub struct Y;

impl Surface for Y {
    type Output = Batch<f32>;

    #[inline(always)]
    fn sample(&self, _u: Batch<f32>, v: Batch<f32>) -> Batch<f32> {
        v
    }
}

/// Any `Batch<T>` is itself a surface (ignoring coordinates).
impl<T: Copy + Send + Sync> Surface for Batch<T> {
    type Output = Batch<T>;

    #[inline(always)]
    fn sample(&self, _u: Batch<f32>, _v: Batch<f32>) -> Batch<T> {
        *self
    }
}
