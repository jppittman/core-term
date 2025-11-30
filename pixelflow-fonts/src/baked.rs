use pixelflow_core::batch::Batch;
use pixelflow_core::pipe::Surface;
use pixelflow_core::Pixel;
use std::sync::{Arc, OnceLock};

#[derive(Clone)]
pub struct LazyBaked<S, P> {
    source: S,
    width: u32,
    height: u32,
    cache: Arc<OnceLock<Box<[P]>>>,
}

impl<S, P> LazyBaked<S, P> {
    pub fn new(source: S, width: u32, height: u32) -> Self {
        Self {
            source,
            width,
            height,
            cache: Arc::new(OnceLock::new()),
        }
    }
}

impl<S, P> Surface<P> for LazyBaked<S, P>
where
    S: Surface<P>,
    P: Pixel + Default + Copy + Send + Sync,
{
    fn eval(&self, x: Batch<u32>, y: Batch<u32>) -> Batch<P> {
        let data = self.cache.get_or_init(|| {
            let width = self.width;
            let height = self.height;
            let mut data = vec![P::default(); (width as usize) * (height as usize)].into_boxed_slice();

            const LANES: usize = 4;
            let w = width as usize;
            let h = height as usize;

            for y in 0..h {
                let row_start = y * w;
                let y_batch = Batch::splat(y as u32);

                let mut x = 0;
                while x + LANES <= w {
                    let x_batch = Batch::new(x as u32, (x + 1) as u32, (x + 2) as u32, (x + 3) as u32);
                    let result: Batch<P> = self.source.eval(x_batch, y_batch);
                    let result_u32: Batch<u32> = result.transmute();

                    unsafe {
                        let ptr = data.as_mut_ptr().add(row_start + x) as *mut u32;
                        result_u32.store(ptr);
                    }
                    x += LANES;
                }

                while x < w {
                    let x_batch = Batch::splat(x as u32);
                    let result: Batch<P> = self.source.eval(x_batch, y_batch);
                    let result_u32: Batch<u32> = result.transmute();
                    data[row_start + x] = P::from_u32(result_u32.to_array_usize()[0] as u32);
                    x += 1;
                }
            }
            data
        });

        // Sample
        let w = self.width as usize;
        let h = self.height as usize;
        let x_arr = x.to_array_usize();
        let y_arr = y.to_array_usize();

        let idx0 = (y_arr[0] % h) * w + (x_arr[0] % w);
        let idx1 = (y_arr[1] % h) * w + (x_arr[1] % w);
        let idx2 = (y_arr[2] % h) * w + (x_arr[2] % w);
        let idx3 = (y_arr[3] % h) * w + (x_arr[3] % w);

        let p0 = data[idx0].to_u32();
        let p1 = data[idx1].to_u32();
        let p2 = data[idx2].to_u32();
        let p3 = data[idx3].to_u32();

        Batch::new(p0, p1, p2, p3).transmute()
    }
}

pub trait BakedExt<P: Copy>: Surface<P> + Sized {
    fn baked(self, width: u32, height: u32) -> LazyBaked<Self, P>;
}

impl<S, P> BakedExt<P> for S
where
    S: Surface<P>,
    P: Pixel,
{
    fn baked(self, width: u32, height: u32) -> LazyBaked<Self, P> {
        LazyBaked::new(self, width, height)
    }
}
