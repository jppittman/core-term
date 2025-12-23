//! Tests for the TTF parser and glyph rendering.

use pixelflow_core::{Field, Manifold};
use pixelflow_graphics::fonts::{CurveSurface, Font};

const FONT_BYTES: &[u8] = include_bytes!("../assets/NotoSansMono-Regular.ttf");

#[test]
fn parse_font_and_get_glyph() {
    let font = Font::from_bytes(FONT_BYTES).expect("Failed to parse font");

    // Test metrics
    let metrics = font.metrics();
    assert!(metrics.units_per_em > 0, "Font should have units_per_em");
    assert!(metrics.ascent > 0, "Font should have positive ascent");

    // Test getting glyphs
    let glyph_a = font.glyph('A', 64.0).expect("Glyph 'A' not found");
    assert!(glyph_a.advance > 0.0, "Glyph should have positive advance");

    // Test that we can get curves
    let curves = glyph_a.curves();
    assert!(!curves.is_empty(), "Glyph 'A' should have curves");

    println!("Glyph 'A' has {} curve segments", curves.len());
    println!("Glyph advance: {}", glyph_a.advance);
    println!("Glyph bounds: {:?}", glyph_a.bounds());
}

#[test]
fn glyph_is_manifold() {
    let font = Font::from_bytes(FONT_BYTES).expect("Failed to parse font");
    let glyph = font.glyph('A', 64.0).expect("Glyph 'A' not found");

    // Verify the glyph implements Manifold by evaluating it
    // We can't extract the values, but we can verify it doesn't panic
    let _val = glyph.eval_raw(
        Field::from(32.0),
        Field::from(32.0),
        Field::from(0.0),
        Field::from(0.0),
    );

    // Test evaluation at various points
    for y in 0..64 {
        for x in 0..64 {
            let _val = glyph.eval_raw(
                Field::from(x as f32 + 0.5),
                Field::from(y as f32 + 0.5),
                Field::from(0.0),
                Field::from(0.0),
            );
        }
    }

    println!("Successfully evaluated glyph at 64x64 points");
}

#[test]
fn all_printable_ascii_glyphs_exist() {
    let font = Font::from_bytes(FONT_BYTES).expect("Failed to parse font");

    for ch in ' '..='~' {
        let glyph = font.glyph(ch, 16.0);
        assert!(
            glyph.is_some(),
            "Printable ASCII character '{}' (0x{:02X}) should exist",
            ch,
            ch as u32
        );
    }

    println!("All printable ASCII characters found in font");
}

#[test]
fn advance_and_kern() {
    let font = Font::from_bytes(FONT_BYTES).expect("Failed to parse font");

    let advance_a = font.advance('A', 16.0);
    let advance_w = font.advance('W', 16.0);

    assert!(advance_a > 0.0, "Advance for 'A' should be positive");
    assert!(advance_w > 0.0, "Advance for 'W' should be positive");

    // In a monospace font, all advances should be equal
    assert!(
        (advance_a - advance_w).abs() < 0.01,
        "Monospace font should have equal advances"
    );

    // Kerning (currently returns 0.0 as TODO)
    let kern = font.kern('A', 'V', 16.0);
    assert_eq!(kern, 0.0, "Kerning returns 0.0 (not yet implemented)");
}
