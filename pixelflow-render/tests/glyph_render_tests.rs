//! Integration tests for glyph rendering with real font data.

#![cfg(feature = "fonts")]

use pixelflow_render::glyph::{render_glyph, render_glyph_natural};

/// Test rendering a simple ASCII character.
#[test]
fn test_render_ascii_glyph() {
    let glyph = render_glyph_natural('A', 24, false, false);

    // Should have non-zero dimensions
    assert!(glyph.width > 0, "Glyph width should be positive");
    assert!(glyph.height > 0, "Glyph height should be positive");

    // Data should match dimensions
    assert_eq!(
        glyph.data.len(),
        glyph.width * glyph.height,
        "Data length should match width × height"
    );

    // Should have some non-zero pixels (not empty)
    let non_zero_count = glyph.data.iter().filter(|&&px| px > 0).count();
    assert!(
        non_zero_count > 0,
        "Glyph should have at least some visible pixels"
    );

    // Should not be all white (that would indicate a rendering bug)
    let all_white = glyph.data.iter().all(|&px| px == 255);
    assert!(!all_white, "Glyph should not be entirely white");
}

/// Test that bold glyphs are different from normal glyphs.
#[test]
fn test_bold_vs_normal() {
    let normal = render_glyph_natural('B', 24, false, false);
    let bold = render_glyph_natural('B', 24, true, false);

    // Dimensions should be the same
    assert_eq!(normal.width, bold.width, "Bold and normal should have same width");
    assert_eq!(normal.height, bold.height, "Bold and normal should have same height");

    // Pixel data should be different
    assert_ne!(
        normal.data, bold.data,
        "Bold rendering should differ from normal"
    );

    // Bold should have more non-zero pixels (thicker)
    let normal_coverage: usize = normal.data.iter().map(|&px| px as usize).sum();
    let bold_coverage: usize = bold.data.iter().map(|&px| px as usize).sum();

    assert!(
        bold_coverage > normal_coverage,
        "Bold should have more ink coverage than normal (bold={}, normal={})",
        bold_coverage,
        normal_coverage
    );
}

/// Test that italic glyphs are different from normal glyphs.
#[test]
fn test_italic_vs_normal() {
    let normal = render_glyph_natural('I', 24, false, false);
    let italic = render_glyph_natural('I', 24, false, true);

    // Dimensions should be the same
    assert_eq!(normal.width, italic.width);
    assert_eq!(normal.height, italic.height);

    // Pixel data should be different
    assert_ne!(
        normal.data, italic.data,
        "Italic rendering should differ from normal"
    );
}

/// Test rendering at different sizes.
#[test]
fn test_different_sizes() {
    let small = render_glyph_natural('O', 12, false, false);
    let medium = render_glyph_natural('O', 24, false, false);
    let large = render_glyph_natural('O', 48, false, false);

    // Larger sizes should have larger dimensions
    assert!(small.width < medium.width, "Larger size should have larger width");
    assert!(medium.width < large.width, "Larger size should have larger width");
    assert!(small.height < medium.height, "Larger size should have larger height");
    assert!(medium.height < large.height, "Larger size should have larger height");
}

/// Test rendering a character that doesn't exist.
#[test]
fn test_missing_glyph() {
    // Try to render a character that's unlikely to be in the font
    let glyph = render_glyph_natural('\u{FFFF}', 24, false, false);

    // Should return a minimal empty glyph
    assert_eq!(glyph.width, 1, "Missing glyph should have width 1");
    assert_eq!(glyph.height, 1, "Missing glyph should have height 1");
    assert_eq!(glyph.data.len(), 1, "Missing glyph should have 1 byte");
}

/// Test that the same character rendered twice gives the same result.
#[test]
fn test_deterministic_rendering() {
    let first = render_glyph_natural('M', 24, false, false);
    let second = render_glyph_natural('M', 24, false, false);

    assert_eq!(first.width, second.width);
    assert_eq!(first.height, second.height);
    assert_eq!(first.data, second.data, "Rendering should be deterministic");
}

/// Test bearing values are reasonable.
#[test]
fn test_bearing_values() {
    let glyph = render_glyph_natural('A', 24, false, false);

    // Bearings should be reasonable relative to glyph size
    assert!(
        glyph.bearing_x.abs() < 100,
        "bearing_x should be reasonable, got {}",
        glyph.bearing_x
    );
    assert!(
        glyph.bearing_y.abs() < 100,
        "bearing_y should be reasonable, got {}",
        glyph.bearing_y
    );
}

/// Test legacy render_glyph function.
#[test]
fn test_legacy_render_glyph() {
    let data = render_glyph('T', 12, 24, false, false);

    // Should have some data
    assert!(!data.is_empty(), "Should return non-empty data");

    // Should have some visible pixels
    let visible = data.iter().filter(|&&px| px > 0).count();
    assert!(visible > 0, "Should have visible pixels");
}

/// Test that bold + italic works.
#[test]
fn test_bold_italic_combination() {
    let normal = render_glyph_natural('S', 24, false, false);
    let bold_italic = render_glyph_natural('S', 24, true, true);

    // Should have same dimensions
    assert_eq!(normal.width, bold_italic.width);
    assert_eq!(normal.height, bold_italic.height);

    // Should be different
    assert_ne!(
        normal.data, bold_italic.data,
        "Bold+italic should differ from normal"
    );
}

/// Test common punctuation characters.
#[test]
fn test_punctuation_rendering() {
    for ch in &['.', ',', '!', '?', ':', ';', '-', '_', '(', ')', '[', ']'] {
        let glyph = render_glyph_natural(*ch, 24, false, false);

        assert!(glyph.width > 0, "Character '{}' should have positive width", ch);
        assert!(glyph.height > 0, "Character '{}' should have positive height", ch);

        let visible = glyph.data.iter().filter(|&&px| px > 0).count();
        assert!(
            visible > 0,
            "Character '{}' should have visible pixels",
            ch
        );
    }
}

/// Test digits rendering.
#[test]
fn test_digit_rendering() {
    for digit in '0'..='9' {
        let glyph = render_glyph_natural(digit, 24, false, false);

        assert!(glyph.width > 0, "Digit '{}' should have positive width", digit);
        assert!(glyph.height > 0, "Digit '{}' should have positive height", digit);

        let visible = glyph.data.iter().filter(|&&px| px > 0).count();
        assert!(visible > 0, "Digit '{}' should have visible pixels", digit);
    }
}

/// Test that no glyph produces garbage (all values should be 0-255).
#[test]
fn test_no_invalid_values() {
    let glyph = render_glyph_natural('X', 24, false, false);

    // All values should be valid grayscale (0-255)
    // This is automatically true for u8, but let's check distribution
    let min = glyph.data.iter().min().unwrap();
    let max = glyph.data.iter().max().unwrap();

    assert_eq!(*min, 0, "Minimum should be 0 (background)");
    assert!(*max > 0, "Maximum should be > 0 (foreground)");
}

/// Test upscaling (small atlas to large output).
#[test]
fn test_upscaling() {
    // Render at a very large size to force upscaling
    let large = render_glyph_natural('W', 96, false, false);

    assert!(large.width > 20, "Large glyph should have substantial width");
    assert!(large.height > 20, "Large glyph should have substantial height");

    // Should still have visible pixels
    let visible = large.data.iter().filter(|&&px| px > 0).count();
    assert!(visible > 0, "Upscaled glyph should have visible pixels");
}

/// Test downscaling (atlas to tiny output).
#[test]
fn test_downscaling() {
    // Render at a very small size to force downscaling
    let tiny = render_glyph_natural('W', 6, false, false);

    assert!(tiny.width > 0, "Tiny glyph should have positive width");
    assert!(tiny.height > 0, "Tiny glyph should have positive height");

    // Should still have visible pixels even at small size
    let visible = tiny.data.iter().filter(|&&px| px > 0).count();
    assert!(visible > 0, "Downscaled glyph should have visible pixels");
}
