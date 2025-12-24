//! Regression tests for font rasterization.
//!
//! These tests ensure the font rasterization system works correctly and
//! catches bugs like:
//! - Mask AND using `*` instead of `&` (SIMD mask multiplication gives NaN)
//! - Missing Y-offset for ascent in glyph_scaled
//! - Winding number calculation errors

use pixelflow_core::{materialize_discrete, PARALLELISM};
use pixelflow_graphics::fonts::{Font, Text};
use pixelflow_graphics::fonts::ttf::{Curve, Glyph, Line, Segment, Sum};
use pixelflow_graphics::render::color::{Lift, Rgba8};
use pixelflow_graphics::render::{execute, TensorShape};
use std::sync::Arc;

const FONT_BYTES: &[u8] = include_bytes!("../assets/NotoSansMono-Regular.ttf");

// =============================================================================
// Regression: SIMD mask AND must use `&` not `*`
// =============================================================================

/// Test that SIMD mask AND works correctly (bug: using `*` gave NaN).
///
/// This test creates a simple square and verifies that points inside
/// have coverage=1 and points outside have coverage=0.
#[test]
fn regression_mask_and_not_multiply() {
    // Create a 400x400 square from (100,100) to (500,500)
    let segments: Vec<Segment> = vec![
        Segment::Line(Curve([[100.0, 100.0], [500.0, 100.0]])), // bottom
        Segment::Line(Curve([[500.0, 100.0], [500.0, 500.0]])), // right
        Segment::Line(Curve([[500.0, 500.0], [100.0, 500.0]])), // top
        Segment::Line(Curve([[100.0, 500.0], [100.0, 100.0]])), // left
    ];
    let glyph = Glyph::Simple(Sum(Arc::from(segments.into_boxed_slice())));
    let lifted = Lift(glyph);

    // Test center (should be inside, coverage = 255)
    let mut center_pixels = [0u32; PARALLELISM];
    materialize_discrete(&lifted, 300.0, 300.0, &mut center_pixels);
    let center_coverage = center_pixels[0] & 0xFF;
    assert_eq!(center_coverage, 255, "Center of square should be inside (coverage=255)");

    // Test outside left (should be outside, coverage = 0)
    let mut left_pixels = [0u32; PARALLELISM];
    materialize_discrete(&lifted, 50.0, 300.0, &mut left_pixels);
    let left_coverage = left_pixels[0] & 0xFF;
    assert_eq!(left_coverage, 0, "Left of square should be outside (coverage=0)");

    // Test outside right
    let mut right_pixels = [0u32; PARALLELISM];
    materialize_discrete(&lifted, 600.0, 300.0, &mut right_pixels);
    let right_coverage = right_pixels[0] & 0xFF;
    assert_eq!(right_coverage, 0, "Right of square should be outside (coverage=0)");

    // Test outside above
    let mut above_pixels = [0u32; PARALLELISM];
    materialize_discrete(&lifted, 300.0, 50.0, &mut above_pixels);
    let above_coverage = above_pixels[0] & 0xFF;
    assert_eq!(above_coverage, 0, "Above square should be outside (coverage=0)");

    // Test outside below
    let mut below_pixels = [0u32; PARALLELISM];
    materialize_discrete(&lifted, 300.0, 600.0, &mut below_pixels);
    let below_coverage = below_pixels[0] & 0xFF;
    assert_eq!(below_coverage, 0, "Below square should be outside (coverage=0)");
}

/// Test that line segment winding calculation correctly handles the x < x_intersection test.
#[test]
fn regression_line_x_intersection_test() {
    // Vertical line at x=500, going from (500,100) to (500,500)
    let line: Line = Curve([[500.0, 100.0], [500.0, 500.0]]);
    let lifted = Lift(line);

    // Points to the left (x < 500) should contribute +1 (winding direction)
    let mut left_pixels = [0u32; PARALLELISM];
    materialize_discrete(&lifted, 100.0, 300.0, &mut left_pixels);
    let left_value = left_pixels[0] & 0xFF;
    assert_eq!(left_value, 255, "Point left of line should get +1 contribution");

    // Points on or to the right (x >= 500) should contribute 0
    let mut on_pixels = [0u32; PARALLELISM];
    materialize_discrete(&lifted, 500.0, 300.0, &mut on_pixels);
    let on_value = on_pixels[0] & 0xFF;
    assert_eq!(on_value, 0, "Point on line should get 0 contribution");

    let mut right_pixels = [0u32; PARALLELISM];
    materialize_discrete(&lifted, 600.0, 300.0, &mut right_pixels);
    let right_value = right_pixels[0] & 0xFF;
    assert_eq!(right_value, 0, "Point right of line should get 0 contribution");
}

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
    let lifted = Lift(text);

    // Create a framebuffer
    let width = 80;
    let height = 120;
    let mut pixels: Vec<Rgba8> = vec![Rgba8::default(); width * height];

    execute(&lifted, &mut pixels, TensorShape::new(width, height));

    // Count non-black pixels
    let white_pixels = pixels.iter().filter(|p| p.r() > 0).count();

    // There should be a significant number of white pixels (glyph area)
    // A typical 'A' at 100px would cover at least 1000 pixels
    assert!(
        white_pixels > 500,
        "Expected at least 500 white pixels, got {} (glyph may be outside visible area)",
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
    let lifted = Lift(text);

    let width = 100;
    let height = 30;
    let mut pixels: Vec<Rgba8> = vec![Rgba8::default(); width * height];

    execute(&lifted, &mut pixels, TensorShape::new(width, height));

    // Count white (inside glyph) and black (outside) pixels
    let white_count = pixels.iter().filter(|p| p.r() == 255).count();
    let black_count = pixels.iter().filter(|p| p.r() == 0).count();
    let total = width * height;

    // Text should take up some space but not fill the entire buffer
    assert!(white_count > 100, "Expected at least 100 white pixels for 'HELLO', got {}", white_count);
    assert!(black_count > 500, "Expected at least 500 black pixels for background, got {}", black_count);
    assert_eq!(white_count + black_count, total, "All pixels should be either 0 or 255");
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
            ch, ch as u32
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
    let metrics = font.metrics();

    // NotoSansMono should have these approximate values
    assert!(metrics.units_per_em >= 1000, "units_per_em should be >= 1000");
    assert!(metrics.ascent > 0, "ascent should be positive");
    assert!(metrics.descent < 0, "descent should be negative");
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
        advance_a, advance_m
    );
    assert!(
        (advance_a - advance_i).abs() < 0.01,
        "Monospace font should have equal advances: A={}, i={}",
        advance_a, advance_i
    );
}
