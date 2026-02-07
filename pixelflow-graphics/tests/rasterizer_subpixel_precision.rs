//! Test to verify rasterizer subpixel precision.

use pixelflow_graphics::render::color::{Pixel, Rgba8};
use pixelflow_graphics::render::frame::Frame;
use pixelflow_graphics::render::{rasterize, Color, NamedColor};
use pixelflow_core::{Field, X};
use pixelflow_graphics::ManifoldExt;

#[test]
fn rasterizer_should_sample_at_pixel_centers() {
    // Manifold: if x > 0.4 then White else Black.
    //
    // The rasterizer should sample at pixel centers (x + 0.5).
    // At x=0, sample coordinate is 0.5.
    // 0.5 > 0.4 is TRUE.
    // Expected result: White.
    //
    // If the rasterizer incorrectly samples at integer coordinates (x=0.0):
    // 0.0 > 0.4 is FALSE.
    // Incorrect result: Black.

    let white = Color::Named(NamedColor::White);
    let black = Color::Named(NamedColor::Black);

    // Use ManifoldExt::select for cleaner syntax
    let manifold = X.gt(Field::from(0.4)).select(white, black);

    // Render a 1x1 frame
    let mut frame = Frame::<Rgba8>::new(1, 1);
    rasterize(&manifold, &mut frame, 1);

    let pixel = frame.data[0];

    // White is (229, 229, 229)
    assert_eq!(pixel.r(), 229, "Pixel at x=0 should be white (sampled at 0.5 > 0.4)");
    assert_eq!(pixel.g(), 229);
    assert_eq!(pixel.b(), 229);
}

#[test]
fn rasterizer_should_sample_second_pixel_correctly() {
    // Manifold: if x > 1.4 then White else Black.
    // At x=1, sample coordinate is 1.5.
    // 1.5 > 1.4 is TRUE -> White.

    let white = Color::Named(NamedColor::White);
    let black = Color::Named(NamedColor::Black);

    let manifold = X.gt(Field::from(1.4)).select(white, black);

    // Render a 2x1 frame
    let mut frame = Frame::<Rgba8>::new(2, 1);
    rasterize(&manifold, &mut frame, 1);

    // Pixel 0 (x=0, sample=0.5): 0.5 > 1.4 -> False -> Black
    let p0 = frame.data[0];
    assert_eq!(p0.r(), 0, "Pixel at x=0 should be black");

    // Pixel 1 (x=1, sample=1.5): 1.5 > 1.4 -> True -> White
    let p1 = frame.data[1];
    assert_eq!(p1.r(), 229, "Pixel at x=1 should be white (sampled at 1.5 > 1.4)");
}
