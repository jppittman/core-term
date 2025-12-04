use crate::backend::{Backend, BatchArithmetic, SimdBatch};
use crate::batch::{Batch, NativeBackend};
use crate::traits::Surface;

/// A rectangular mask (returns 0 or !0).
#[derive(Copy, Clone, Debug)]
pub struct Rect {
    pub x: i32,
    pub y: i32,
    pub w: u32,
    pub h: u32,
}

impl Rect {
    pub fn new(x: i32, y: i32, w: u32, h: u32) -> Self {
        Self { x, y, w, h }
    }
}

impl Surface<u32> for Rect {
    #[inline(always)]
    fn eval(&self, x: Batch<u32>, y: Batch<u32>) -> Batch<u32> {
        let left = Batch::<u32>::splat(self.x as u32);
        let right = Batch::<u32>::splat((self.x + self.w as i32) as u32);
        let top = Batch::<u32>::splat(self.y as u32);
        let bottom = Batch::<u32>::splat((self.y + self.h as i32) as u32);

        // Logic: (x >= left) & (x < right) & (y >= top) & (y < bottom)
        x.cmp_ge(left) & x.cmp_lt(right) & y.cmp_ge(top) & y.cmp_lt(bottom)
    }
}

impl Surface<u8> for Rect {
    #[inline(always)]
    fn eval(&self, x: Batch<u32>, y: Batch<u32>) -> Batch<u8> {
        let mask = <Self as Surface<u32>>::eval(self, x, y);
        NativeBackend::downcast_u32_to_u8(mask)
    }
}
