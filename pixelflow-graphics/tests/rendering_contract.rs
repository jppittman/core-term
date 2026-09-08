//! A semantic colour reaches the frame as the bytes it names.
//!
//! `Color` and `NamedColor` used to be `Manifold`s that a rasterizer sampled
//! per SIMD batch; S4a deleted that lane, and a colour is now what it always
//! denoted — four numbers in `[0, 1]` — compiled into a scene's four constant
//! channel kernels and packed by the frame's own byte order. The contract
//! under test is unchanged: `Color::Named(Red)` puts ANSI red in every pixel.

use pixelflow_graphics::render::color::Rgba8;
use pixelflow_graphics::render::frame::Frame;
use pixelflow_graphics::render::scene::constant_scene_for;
use pixelflow_graphics::render::{Color, NamedColor};

/// Render `color` over a `size × size` frame and return its first pixel's
/// channels, having asserted every pixel is that one.
fn solid(color: Color, size: u32) -> (u8, u8, u8, u8) {
    let (r, g, b, a) = color.to_f32_rgba();
    let scene = constant_scene_for::<Rgba8>([r, g, b, a], [size, size]);
    let mut frame = Frame::<Rgba8>::new(size, size);
    scene.render(&mut frame, 1);
    let first = frame.data[0];
    assert!(
        frame.data.iter().all(|p| *p == first),
        "a constant colour must be constant across the frame"
    );
    (first.r(), first.g(), first.b(), first.a())
}

#[test]
fn verify_color_manifold_renders() {
    assert_eq!(solid(Color::Named(NamedColor::Red), 4), (205, 0, 0, 255));
}

#[test]
fn verify_named_color_manifold_renders() {
    assert_eq!(solid(Color::Named(NamedColor::Blue), 2), (0, 0, 238, 255));
}
