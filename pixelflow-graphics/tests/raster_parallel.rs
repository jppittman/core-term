//! Tests for parallel rasterization.

use pixelflow_graphics::render::color::{Color, NamedColor, Rgba8};
use pixelflow_graphics::render::rasterizer::{rasterize, RenderOptions, TensorShape};

#[test]
fn test_parallel_rasterization_matches_sequential() {
    let width = 100;
    let height = 100;
    let color = Color::Named(NamedColor::Green);

    // Sequential rendering
    let mut seq_buffer = vec![Rgba8::default(); width * height];
    let shape = TensorShape::new(width, height);
    rasterize(&color, &mut seq_buffer, shape, RenderOptions::default());

    // Parallel rendering with 4 threads
    let mut par_buffer = vec![Rgba8::default(); width * height];
    rasterize(
        &color,
        &mut par_buffer,
        shape,
        RenderOptions { num_threads: 4 },
    );

    // Results should be identical
    assert_eq!(seq_buffer, par_buffer);
}
