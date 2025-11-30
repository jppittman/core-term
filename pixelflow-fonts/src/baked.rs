use pixelflow_core::batch::Batch;
use pixelflow_core::pipe::Surface;
use crate::lazy::Lazy;
use std::sync::Arc;

#[derive(Clone)]
pub struct BakedMask {
    data: Box<[u8]>,
    width: u32,
    height: u32,
}

impl BakedMask {
    pub fn new<S: Surface<u8>>(source: &S, width: u32, height: u32) -> Self {
        let mut data = vec![0u8; (width as usize) * (height as usize)].into_boxed_slice();

        const LANES: usize = 4;
        let w = width as usize;
        let h = height as usize;

        for y in 0..h {
            let row_start = y * w;
            let y_batch = Batch::splat(y as u32);

            let mut x = 0;
            while x + LANES <= w {
                let x_batch = Batch::new(x as u32, (x + 1) as u32, (x + 2) as u32, (x + 3) as u32);
                let result: Batch<u8> = source.eval(x_batch, y_batch);

                let bytes = result.cast::<u32>().to_bytes_packed();

                data[row_start + x] = bytes[0];
                data[row_start + x + 1] = bytes[1];
                data[row_start + x + 2] = bytes[2];
                data[row_start + x + 3] = bytes[3];

                x += LANES;
            }

            while x < w {
                let x_batch = Batch::splat(x as u32);
                let result: Batch<u8> = source.eval(x_batch, y_batch);
                let val = result.cast::<u32>().extract(0) as u8;
                data[row_start + x] = val;
                x += 1;
            }
        }

        Self { data, width, height }
    }
}

impl Surface<u8> for BakedMask {
    fn eval(&self, x: Batch<u32>, y: Batch<u32>) -> Batch<u8> {
        let w = self.width as usize;
        let h = self.height as usize;
        let x_arr = x.to_array_usize();
        let y_arr = y.to_array_usize();

        let idx0 = (y_arr[0] % h) * w + (x_arr[0] % w);
        let idx1 = (y_arr[1] % h) * w + (x_arr[1] % w);
        let idx2 = (y_arr[2] % h) * w + (x_arr[2] % w);
        let idx3 = (y_arr[3] % h) * w + (x_arr[3] % w);

        let p0 = self.data[idx0] as u32;
        let p1 = self.data[idx1] as u32;
        let p2 = self.data[idx2] as u32;
        let p3 = self.data[idx3] as u32;

        Batch::new(p0, p1, p2, p3).cast()
    }
}

pub type BakedFactory<'a> = Arc<dyn Fn() -> BakedMask + Send + Sync + 'a>;

pub trait BakedExt<'a>: Surface<u8> + Sized + Clone + Send + Sync + 'a {
    fn baked(self, width: u32, height: u32) -> Lazy<BakedFactory<'a>, BakedMask> {
        let source = self.clone();
        let factory: BakedFactory<'a> = Arc::new(move || {
            BakedMask::new(&source, width, height)
        });
        Lazy::new(factory)
    }
}

impl<'a, S> BakedExt<'a> for S where S: Surface<u8> + Clone + Send + Sync + 'a {}
