//! Tests for the rendering pipeline.
//!
//! TODO: These tests need updating for the new Color manifold system.

use pixelflow_graphics::render::color::Rgba8;
use pixelflow_graphics::render::frame::Frame;
use pixelflow_graphics::render::rasterizer::rasterize;

#[test]
fn verify_color_manifold_renders() {
    use pixelflow_graphics::render::{Color, NamedColor};
    // A solid red color manifold
    let red = Color::Named(NamedColor::Red);

    let mut frame = Frame::<Rgba8>::new(4, 4);

    rasterize(&red, &mut frame, 1);

    // All pixels should be red
    let (r, g, b) = NamedColor::Red.to_rgb();
    for pixel in &frame.data {
        assert_eq!(pixel.r(), r);
        assert_eq!(pixel.g(), g);
        assert_eq!(pixel.b(), b);
        assert_eq!(pixel.a(), 255);
    }
}

#[test]
fn verify_named_color_manifold_renders() {
    use pixelflow_graphics::render::NamedColor;
    let blue = NamedColor::Blue;

    let mut frame = Frame::<Rgba8>::new(2, 2);

    rasterize(&blue, &mut frame, 1);

    let (r, g, b) = blue.to_rgb();
    for pixel in &frame.data {
        assert_eq!(pixel.r(), r);
        assert_eq!(pixel.g(), g);
        assert_eq!(pixel.b(), b);
        assert_eq!(pixel.a(), 255);
    }
}
