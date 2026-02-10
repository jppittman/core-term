//! Tests for parallel rasterization public API.

use pixelflow_graphics::render::color::{Color, NamedColor, Rgba8};
use pixelflow_graphics::render::frame::Frame;
use pixelflow_graphics::render::rasterizer::rasterize;

const WIDTH: u32 = 100;
const HEIGHT: u32 = 100;
const THREADS_SEQ: usize = 1;
const THREADS_PAR: usize = 4;

#[test]
fn rasterize_multithreaded_produces_identical_output_to_singlethreaded() {
    let color = Color::Named(NamedColor::Green);

    // Sequential rendering
    let mut seq_frame = Frame::<Rgba8>::new(WIDTH, HEIGHT);
    rasterize(&color, &mut seq_frame, THREADS_SEQ);

    // Parallel rendering
    let mut par_frame = Frame::<Rgba8>::new(WIDTH, HEIGHT);
    rasterize(&color, &mut par_frame, THREADS_PAR);

    // Results should be identical
    assert_eq!(seq_frame.data, par_frame.data);
}
