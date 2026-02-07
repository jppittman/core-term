//! Tests for the rendering pipeline.
//!
//! TODO: These tests need updating for the new Color manifold system.

use pixelflow_graphics::render::color::Rgba8;
use pixelflow_graphics::render::frame::Frame;
use pixelflow_graphics::render::rasterizer::rasterize;

#[test]
fn color_manifold_should_render_solid_color() {
    use pixelflow_graphics::render::{Color, NamedColor};
    // A solid red color manifold
    let red = Color::Named(NamedColor::Red);

    let mut frame = Frame::<Rgba8>::new(4, 4);

    rasterize(&red, &mut frame, 1);

    // All pixels should be red
    for pixel in &frame.data {
        assert_eq!(pixel.r(), 205); // ANSI Red
        assert_eq!(pixel.g(), 0);
        assert_eq!(pixel.b(), 0);
        assert_eq!(pixel.a(), 255);
    }
}

#[test]
fn named_color_manifold_should_render_correct_values() {
    use pixelflow_graphics::render::NamedColor;
    let blue = NamedColor::Blue;

    let mut frame = Frame::<Rgba8>::new(2, 2);

    rasterize(&blue, &mut frame, 1);

    for pixel in &frame.data {
        assert_eq!(pixel.r(), 0);
        assert_eq!(pixel.g(), 0);
        assert_eq!(pixel.b(), 238); // ANSI Blue
        assert_eq!(pixel.a(), 255);
    }
}
