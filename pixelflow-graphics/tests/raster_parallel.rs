use pixelflow_core::{traits::Surface, Batch};
use pixelflow_graphics::render::rasterizer::{execute, execute_stripe, Stripe, TensorShape};
use std::thread;

struct GradientSurface;
impl Surface<u32, u32> for GradientSurface {
    fn eval(&self, x: Batch<u32>, y: Batch<u32>) -> Batch<u32> {
        x + y
    }
}

#[test]
fn test_parallel_rasterization_matches_sequential() {
    let width = 100;
    let height = 100;
    let surface = GradientSurface;

    let mut seq_buffer = vec![0u32; width * height];
    let shape = TensorShape::new(width, height, width);
    execute(&surface, &mut seq_buffer, shape);

    let mut par_buffer = vec![0u32; width * height];

    // Split into 4 stripes
    let mid_y = height / 2;

    let (top, bottom) = par_buffer.split_at_mut(mid_y * width);

    thread::scope(|s| {
        s.spawn(|| {
            execute_stripe(
                &surface,
                top,
                width,
                Stripe {
                    start_y: 0,
                    end_y: mid_y,
                },
            );
        });
        s.spawn(|| {
            execute_stripe(
                &surface,
                bottom,
                width,
                Stripe {
                    start_y: mid_y,
                    end_y: height,
                },
            );
        });
    });

    assert_eq!(seq_buffer, par_buffer);
}
