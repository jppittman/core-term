//! Debug font rendering

use pixelflow_graphics::fonts::Font;
use pixelflow_core::{Field, Manifold};

const FONT_BYTES: &[u8] = include_bytes!("../assets/NotoSansMono-Regular.ttf");

#[test]
fn debug_glyph_coverage() {
    let font = Font::parse(FONT_BYTES).unwrap();

    // Get a scaled glyph (now uses glyph_scaled instead of glyph with size)
    let glyph = font.glyph_scaled('A', 32.0).unwrap();

    println!("\nTesting glyph 'A' at size 32:");
    println!("Font ascent={}, descent={}, units_per_em={}",
             font.ascent, font.descent, font.units_per_em);

    // Just verify we can evaluate the glyph without panicking
    // Field is SIMD so we can't easily extract individual values
    let _coverage = glyph.eval_raw(
        Field::from(16.0),
        Field::from(16.0),
        Field::from(0.0),
        Field::from(0.0)
    );

    println!("Glyph evaluation successful");
}
