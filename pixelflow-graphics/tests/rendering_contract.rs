//! Tests for the rendering pipeline.
//!
//! TODO: These tests need updating for the new Color manifold system.

use pixelflow_graphics::render::color::{Color, NamedColor, Rgba8};
use pixelflow_graphics::render::rasterizer::{rasterize, RenderOptions, TensorShape};

#[test]
fn verify_color_manifold_renders() {
    // A solid red color manifold
    let red = Color::Named(NamedColor::Red);

    let mut target = vec![Rgba8::default(); 4 * 4];
    let shape = TensorShape::new(4, 4);

    rasterize(&red, &mut target, shape, RenderOptions::default());

    // All pixels should be red
    for pixel in &target {
        assert_eq!(pixel.r(), 205); // ANSI Red
        assert_eq!(pixel.g(), 0);
        assert_eq!(pixel.b(), 0);
        assert_eq!(pixel.a(), 255);
    }
}

#[test]
fn verify_named_color_manifold_renders() {
    let blue = NamedColor::Blue;

    let mut target = vec![Rgba8::default(); 2 * 2];
    let shape = TensorShape::new(2, 2);

    rasterize(&blue, &mut target, shape, RenderOptions::default());

    for pixel in &target {
        assert_eq!(pixel.r(), 0);
        assert_eq!(pixel.g(), 0);
        assert_eq!(pixel.b(), 238); // ANSI Blue
        assert_eq!(pixel.a(), 255);
    }
}
