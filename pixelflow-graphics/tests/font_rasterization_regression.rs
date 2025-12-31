//! Regression tests for font rasterization.
//!
//! These tests ensure the font rasterization system works correctly and
//! catches bugs like:
//! - Missing Y-offset for ascent in glyph_scaled
//! - Loop-Blinn triangulation errors

use pixelflow_graphics::fonts::{Font, Text};
use pixelflow_graphics::render::color::{Grayscale, Rgba8};
use pixelflow_graphics::render::{execute, TensorShape};

const FONT_BYTES: &[u8] = include_bytes!("../assets/NotoSansMono-Regular.ttf");

// =============================================================================
// Regression: glyph_scaled must include Y-offset for ascent
// =============================================================================

/// Test that glyphs are rendered within the visible area.
///
/// Without the Y-offset fix, glyphs would render above y=0 (outside visible area).
#[test]
fn regression_glyph_ascent_offset() {
    let font = Font::parse(FONT_BYTES).expect("Failed to parse font");

    // Render 'A' at size 100
    let _glyph = font.glyph_scaled('A', 100.0).expect("No glyph 'A'");
    let text = Text::new(&font, "A", 100.0);
    let lifted = Grayscale(text);

    // Create a framebuffer
    let width = 80;
    let height = 120;
    let mut pixels: Vec<Rgba8> = vec![Rgba8::default(); width * height];

    execute(&lifted, &mut pixels, TensorShape::new(width, height));

    // Count non-black pixels (with AA, we have smooth gradients)
    let white_pixels = pixels.iter().filter(|p| p.r() > 0).count();

    // There should be a significant number of non-black pixels (glyph area)
    // A typical 'A' at 100px would cover at least 500 pixels
    assert!(
        white_pixels > 100,
        "Expected at least 100 non-black pixels, got {} (glyph may be outside visible area)",
        white_pixels
    );

    // There should also be many black pixels (background)
    let black_pixels = pixels.iter().filter(|p| p.r() == 0).count();
    assert!(
        black_pixels > 1000,
        "Expected at least 1000 black pixels for background, got {}",
        black_pixels
    );
}

// =============================================================================
// Full pipeline regression tests
// =============================================================================

/// Test that the full text rendering pipeline produces visible output.
#[test]
fn regression_text_rendering_pipeline() {
    let font = Font::parse(FONT_BYTES).expect("Failed to parse font");

    let text = Text::new(&font, "HELLO", 20.0);
    let lifted = Grayscale(text);

    let width = 100;
    let height = 30;
    let mut pixels: Vec<Rgba8> = vec![Rgba8::default(); width * height];

    execute(&lifted, &mut pixels, TensorShape::new(width, height));

    // Count pixels with any coverage
    let covered_count = pixels.iter().filter(|p| p.r() > 0).count();
    let uncovered_count = pixels.iter().filter(|p| p.r() == 0).count();

    // Text should take up some space but not fill the entire buffer
    assert!(
        covered_count > 20,
        "Expected at least 20 covered pixels for 'HELLO', got {}",
        covered_count
    );
    assert!(
        uncovered_count > 500,
        "Expected at least 500 uncovered pixels for background, got {}",
        uncovered_count
    );
}

/// Test that all printable ASCII characters can be rendered.
#[test]
fn regression_all_printable_ascii_render() {
    let font = Font::parse(FONT_BYTES).expect("Failed to parse font");

    for ch in ' '..='~' {
        let glyph = font.glyph_scaled(ch, 16.0);
        assert!(
            glyph.is_some(),
            "Character '{}' (0x{:02X}) should have a scaled glyph",
            ch,
            ch as u32
        );

        // Also verify we can get advance width
        let advance = font.advance_scaled(ch, 16.0);
        assert!(
            advance.is_some(),
            "Character '{}' should have advance width",
            ch
        );
    }
}

/// Test that glyph metrics are reasonable.
#[test]
fn regression_font_metrics() {
    let font = Font::parse(FONT_BYTES).expect("Failed to parse font");

    // NotoSansMono should have these approximate values (fields are public on Font)
    assert!(font.units_per_em >= 1000, "units_per_em should be >= 1000");
    assert!(font.ascent > 0, "ascent should be positive");
    assert!(font.descent < 0, "descent should be negative");
}

/// Test that the advance width is consistent for monospace font.
#[test]
fn regression_monospace_advance() {
    let font = Font::parse(FONT_BYTES).expect("Failed to parse font");

    let advance_a = font.advance_scaled('A', 16.0).unwrap();
    let advance_m = font.advance_scaled('M', 16.0).unwrap();
    let advance_i = font.advance_scaled('i', 16.0).unwrap();

    // For a monospace font, all advances should be equal
    assert!(
        (advance_a - advance_m).abs() < 0.01,
        "Monospace font should have equal advances: A={}, M={}",
        advance_a,
        advance_m
    );
    assert!(
        (advance_a - advance_i).abs() < 0.01,
        "Monospace font should have equal advances: A={}, i={}",
        advance_a,
        advance_i
    );
}

/// Test Loop-Blinn triangle rendering produces consistent output.
#[test]
fn regression_loop_blinn_triangle() {
    use pixelflow_core::{materialize_discrete, PARALLELISM};
    use pixelflow_graphics::fonts::ttf::LoopBlinnTriangle;
    use pixelflow_graphics::render::color::Grayscale;

    // Create a simple solid triangle
    let tri = LoopBlinnTriangle::solid([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]]);
    let lifted = Grayscale(tri);

    // Test point inside the triangle
    let mut pixels = [0u32; PARALLELISM];
    materialize_discrete(&lifted, 0.5, 0.3, &mut pixels);
    let val = pixels[0] & 0xFF;
    assert!(
        val > 128,
        "Point inside triangle should have coverage > 128, got {}",
        val
    );

    // Test point outside the triangle
    materialize_discrete(&lifted, 2.0, 0.3, &mut pixels);
    let val_out = pixels[0] & 0xFF;
    assert!(
        val_out < 128,
        "Point outside triangle should have coverage < 128, got {}",
        val_out
    );
}
