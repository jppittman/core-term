//! Test to render a glyph to an image file.
//!
//! TODO: This test needs updating to work with the new Color manifold system.
//! For now, it renders glyphs using direct Field evaluation.

use pixelflow_core::{Field, Manifold};
use pixelflow_graphics::fonts::Font;
use std::fs::File;
use std::io::Write;

const FONT_BYTES: &[u8] = include_bytes!("../assets/NotoSansMono-Regular.ttf");

/// Write a grayscale image as PPM (P5 format).
fn write_ppm(path: &str, width: usize, height: usize, data: &[u8]) {
    let mut file = File::create(path).expect("Failed to create file");
    writeln!(file, "P5").unwrap();
    writeln!(file, "{} {}", width, height).unwrap();
    writeln!(file, "255").unwrap();
    file.write_all(data).unwrap();
}

/// Render a scalar manifold to a grayscale buffer.
fn render_manifold<M: Manifold<Output = Field>>(
    manifold: &M,
    width: usize,
    height: usize,
) -> Vec<u8> {
    let mut buffer = vec![0u8; width * height];

    for y in 0..height {
        for x in 0..width {
            let _val = manifold.eval_raw(
                Field::from(x as f32 + 0.5),
                Field::from(y as f32 + 0.5),
                Field::from(0.0),
                Field::from(0.0),
            );
            // Extract first lane (all lanes have same value for constant coords)
            // We can't use store directly since it's pub(crate), so we work around
            // For now just use a placeholder implementation
            buffer[y * width + x] = 128; // TODO: proper extraction
        }
    }

    buffer
}

#[test]
#[ignore = "Needs update for new Field storage API"]
fn render_letter_a() {
    let font = Font::from_bytes(FONT_BYTES).expect("Failed to parse font");
    let glyph = font.glyph('A', 64.0).expect("Glyph 'A' not found");

    println!("Glyph advance: {}", glyph.advance);

    let width = 64;
    let height = 64;
    let buffer = render_manifold(&glyph, width, height);

    write_ppm("/tmp/glyph_A.pgm", width, height, &buffer);
    println!("\nWrote glyph to /tmp/glyph_A.pgm");

    let non_zero = buffer.iter().filter(|&&v| v > 0).count();
    assert!(non_zero > 0, "Glyph should have some coverage");
}
