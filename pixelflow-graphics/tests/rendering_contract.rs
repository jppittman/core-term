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
    for pixel in &frame.data {
        // Red is (205, 49, 49) in NamedColor
        // Wait, NamedColor::Red in to_rgb() is (205, 49, 49)
        // Original assertion: r=205, g=0, b=0?
        // Let's match the definition.
        // NamedColor::Red => (205, 49, 49)
        assert_eq!(pixel.r(), 205);
        assert_eq!(pixel.g(), 49);
        assert_eq!(pixel.b(), 49);
        assert_eq!(pixel.a(), 255);
    }
}

#[test]
fn verify_named_color_manifold_renders() {
    use pixelflow_graphics::render::{Color, NamedColor};
    // NamedColor itself is not a Manifold, wrap it in Color
    let blue = Color::Named(NamedColor::Blue);

    let mut frame = Frame::<Rgba8>::new(2, 2);

    rasterize(&blue, &mut frame, 1);

    for pixel in &frame.data {
        // NamedColor::Blue => (36, 114, 200)
        assert_eq!(pixel.r(), 36);
        assert_eq!(pixel.g(), 114);
        assert_eq!(pixel.b(), 200);
        assert_eq!(pixel.a(), 255);
    }
}
