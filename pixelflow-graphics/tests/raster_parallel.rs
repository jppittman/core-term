//! Tests for parallel rasterization.

use pixelflow_graphics::render::color::{Color, NamedColor, Rgba8};
use pixelflow_graphics::render::rasterizer::{rasterize, execute_stripe, Stripe, TensorShape};
use std::thread;

#[test]
fn test_parallel_rasterization_matches_sequential() {
    let width = 100;
    let height = 100;
    let color = Color::Named(NamedColor::Green);

    let mut seq_buffer = vec![Rgba8::default(); width * height];
    let shape = TensorShape::new(width, height);
    rasterize(&color, &mut seq_buffer, shape, 1);

    let mut par_buffer = vec![Rgba8::default(); width * height];

    // Split into 2 stripes
    let mid_y = height / 2;

    let (top, bottom) = par_buffer.split_at_mut(mid_y * width);

    thread::scope(|s| {
        s.spawn(|| {
            execute_stripe(
                &color,
                top,
                Stripe {
                    start_y: 0,
                    end_y: mid_y,
                    width,
                },
            );
        });
        s.spawn(|| {
            execute_stripe(
                &color,
                bottom,
                Stripe {
                    start_y: mid_y,
                    end_y: height,
                    width,
                },
            );
        });
    });

    assert_eq!(seq_buffer, par_buffer);
}
