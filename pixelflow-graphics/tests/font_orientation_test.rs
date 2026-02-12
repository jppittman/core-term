//! Test that glyphs render with correct vertical orientation.
//!
//! This is a regression test for the font Y-axis orientation bug where
//! glyphs were rendering upside-down.
//!
//! Note: We use `DejaVuSansMono-Fallback.ttf` because the main font is an LFS pointer.
//! This fallback font renders "tofu" (rectangles) for ASCII characters, so we
//! can only verify that content is rendered and has consistent dimensions,
//! not specific shapes like 'A' or 'V'.

use pixelflow_graphics::fonts::{text, Font};
use pixelflow_graphics::render::color::{Grayscale, Rgba8};
use pixelflow_graphics::render::frame::Frame;
use pixelflow_graphics::render::rasterizer::rasterize;

const FONT_BYTES: &[u8] = include_bytes!("../assets/DejaVuSansMono-Fallback.ttf");

/// Measure the horizontal extent of rendered pixels at a given Y row.
/// Returns (leftmost_x, rightmost_x) of pixels above the threshold, or None if row is empty.
fn measure_row_extent(
    frame: &Frame<Rgba8>,
    y: usize,
    threshold: u8,
) -> Option<(usize, usize)> {
    let width = frame.width;
    let row_start = y * width;
    let row = &frame.data[row_start..row_start + width];

    let left = row.iter().position(|p| p.r() > threshold)?;
    let right = row.iter().rposition(|p| p.r() > threshold)?;

    Some((left, right))
}

/// Calculate the width of rendered content at a given Y row.
fn row_width(frame: &Frame<Rgba8>, y: usize, threshold: u8) -> usize {
    match measure_row_extent(frame, y, threshold) {
        Some((left, right)) => right - left + 1,
        None => 0,
    }
}

#[test]
fn fallback_font_renders_content() {
    // Verify that the fallback font renders *something* (likely a tofu box).
    // This ensures the rasterizer pipeline is working and coordinate transforms
    // place the glyph on screen.

    let font = Font::parse(FONT_BYTES).expect("Failed to parse font");
    let glyph = text(&font, "A", 48.0);
    let color_manifold = Grayscale(glyph);

    let width = 60;
    let height = 70;
    let mut frame = Frame::<Rgba8>::new(width as u32, height as u32);
    rasterize(&color_manifold, &mut frame, 1);

    // Find the vertical bounds of the rendered glyph
    let threshold = 32;
    let mut top_row = None;
    let mut bottom_row = None;
    for y in 0..height {
        if row_width(&frame, y, threshold) > 0 {
            if top_row.is_none() {
                top_row = Some(y);
            }
            bottom_row = Some(y);
        }
    }

    let top_row = top_row.expect("Glyph should have rendered content");
    let bottom_row = bottom_row.expect("Glyph should have rendered content");
    let glyph_height = bottom_row - top_row + 1;

    println!(
        "\nGlyph bounds: top={}, bottom={}, height={}",
        top_row, bottom_row, glyph_height
    );

    assert!(
        glyph_height > 10,
        "Glyph should be tall enough to measure (got {} rows)",
        glyph_height
    );

    // Verify it's a block (consistent width)
    let top_quarter_y = top_row + glyph_height / 4;
    let bottom_quarter_y = bottom_row - glyph_height / 4;

    let top_width = row_width(&frame, top_quarter_y, threshold);
    let bottom_width = row_width(&frame, bottom_quarter_y, threshold);

    println!("Top quarter width: {}", top_width);
    println!("Bottom quarter width: {}", bottom_width);

    // Allow some small variation due to AA, but basically equal
    assert!(
        (top_width as i32 - bottom_width as i32).abs() <= 2,
        "Fallback font should render a uniform block (tofu)"
    );
}
