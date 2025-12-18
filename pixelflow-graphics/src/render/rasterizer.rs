use super::color::Pixel;
use super::frame::Frame;
use core::fmt::Debug;
use pixelflow_core::backend::{Backend, SimdBatch};
use pixelflow_core::batch::{Batch, NativeBackend, LANES};
use pixelflow_core::traits::{Manifold, Surface};

/// Dimensions and memory layout of a 2D tensor.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct TensorShape {
    /// Width of the tensor.
    pub width: usize,
    /// Height of the tensor.
    pub height: usize,
    /// Stride (elements per row).
    pub stride: usize,
}

impl TensorShape {
    /// Creates a new `TensorShape`.
    pub const fn new(width: usize, height: usize, stride: usize) -> Self {
        Self {
            width,
            height,
            stride,
        }
    }
}

/// A horizontal stripe of rows for parallel rendering.
pub struct Stripe {
    pub start_y: usize,
    pub end_y: usize,
}

/// Render a surface into a Frame.
pub fn render<P, S>(surface: &S, frame: &mut Frame<P>)
where
    P: Pixel,
    S: Surface<P> + ?Sized,
{
    let shape = TensorShape::new(
        frame.width as usize,
        frame.height as usize,
        frame.width as usize,
    );
    execute(surface, frame.as_slice_mut(), shape);
}

/// Render a surface into a typed pixel buffer.
pub fn render_to_buffer<P, S>(surface: &S, buffer: &mut [P], shape: TensorShape)
where
    P: Pixel,
    S: Surface<P> + ?Sized,
{
    execute(surface, buffer, shape);
}

/// Render a Surface<P> into a u32 buffer.
pub fn render_pixel<P, S>(surface: &S, buffer: &mut [u32], shape: TensorShape)
where
    P: Pixel,
    S: Surface<P> + ?Sized,
{
    // SAFETY: P is repr(transparent) over u32
    let typed_buffer: &mut [P] =
        unsafe { core::slice::from_raw_parts_mut(buffer.as_mut_ptr() as *mut P, buffer.len()) };
    execute(surface, typed_buffer, shape);
}

/// Render a Surface<u32> directly into a u32 buffer.
pub fn render_u32<S>(surface: &S, buffer: &mut [u32], shape: TensorShape)
where
    S: Surface<u32> + ?Sized,
{
    execute::<u32, S>(surface, buffer, shape);
}

/// Software rasterizer entry point.
pub fn execute<P, S>(surface: &S, target: &mut [P], shape: TensorShape)
where
    P: Pixel,
    S: Surface<P> + ?Sized,
{
    if shape.width == 0 || shape.height == 0 {
        return;
    }

    render_stripe(surface, target, shape.width, 0, shape.height);
}

/// Render a horizontal stripe of rows [start_y, end_y)
#[inline(always)]
fn render_stripe<P, S>(surface: &S, target: &mut [P], width: usize, start_y: usize, end_y: usize)
where
    P: Pixel,
    S: Surface<P> + ?Sized,
{
    for (row_idx, y) in (start_y..end_y).enumerate() {
        let row_start = row_idx * width;
        let y_batch = Batch::<u32>::splat(y as u32);

        let mut x = 0;
        while x + LANES <= width {
            let x_batch = Batch::<u32>::sequential_from(x as u32);
            let result = surface.eval(x_batch, y_batch);

            P::batch_store(result, &mut target[row_start + x..row_start + x + LANES]);

            x += LANES;
        }

        while x < width {
            target[row_start + x] = surface.eval_one(x as u32, y as u32);
            x += 1;
        }
    }
}

/// Adapts a Continuous Surface (f32) to a Discrete Surface (u32).
/// Samples at the pixel center (x + 0.5, y + 0.5).
#[derive(Copy, Clone)]
pub struct Rasterize<S>(pub S);

impl<S, P> Manifold<P, u32> for Rasterize<S>
where
    S: Manifold<P, f32>,
    P: Copy + Debug + Default + PartialEq + Send + Sync + 'static,
{
    #[inline(always)]
    fn eval(&self, x: Batch<u32>, y: Batch<u32>, z: Batch<u32>, w: Batch<u32>) -> Batch<P> {
        // Bridge: Convert Integer Grid -> Continuous Space
        // Offset by 0.5 to sample pixel centers.
        let fx = NativeBackend::u32_to_f32(x) + Batch::<f32>::splat(0.5);
        let fy = NativeBackend::u32_to_f32(y) + Batch::<f32>::splat(0.5);
        let fz = NativeBackend::u32_to_f32(z) + Batch::<f32>::splat(0.5);
        let fw = NativeBackend::u32_to_f32(w) + Batch::<f32>::splat(0.5);

        self.0.eval(fx, fy, fz, fw)
    }
}

/// Helper to render a surface into a `Baked` buffer.
pub fn bake<P, S>(surface: &S, width: u32, height: u32) -> pixelflow_core::surfaces::Baked<P>
where
    P: Pixel + pixelflow_core::bitwise::Bitwise,
    S: Surface<P> + ?Sized,
{
    let mut data = vec![P::default(); (width * height) as usize];
    let shape = TensorShape::new(width as usize, height as usize, width as usize);
    execute(surface, &mut data, shape);
    pixelflow_core::surfaces::Baked::from_data(data.into(), width, height)
}

/// Render a specific row range into the target buffer.
pub fn execute_stripe<P, S>(surface: &S, target: &mut [P], width: usize, stripe: Stripe)
where
    P: Pixel,
    S: Surface<P> + ?Sized,
{
    render_stripe(surface, target, width, stripe.start_y, stripe.end_y);
}
