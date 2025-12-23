//! Test to render a glyph to an image file.

use pixelflow_core::{materialize, Field, Manifold, PARALLELISM};
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

/// Render a manifold to a grayscale buffer.
fn render_manifold<M: Manifold<Output = Field>>(
    manifold: &M,
    width: usize,
    height: usize,
) -> Vec<u8> {
    let mut buffer = vec![0u8; width * height];
    let mut batch = vec![0.0f32; PARALLELISM];

    for y in 0..height {
        let mut x = 0usize;
        while x < width {
            let remaining = width - x;
            let chunk_size = remaining.min(PARALLELISM);

            materialize(manifold, x as f32, y as f32, &mut batch[..PARALLELISM]);

            for i in 0..chunk_size {
                let val = batch[i].clamp(0.0, 1.0);
                buffer[y * width + x + i] = (val * 255.0) as u8;
            }

            x += chunk_size;
        }
    }

    buffer
}

#[test]
fn render_letter_a() {
    let font = Font::from_bytes(FONT_BYTES).expect("Failed to parse font");
    // Glyph is now inherently antialiased and normalized.
    let glyph = font.glyph('A', 64.0).expect("Glyph 'A' not found");

    println!("Glyph advance: {}", glyph.advance);

    let width = 64;
    let height = 64;
    let buffer = render_manifold(&glyph, width, height);

    write_ppm("/tmp/glyph_A.pgm", width, height, &buffer);
    println!("\nWrote glyph to /tmp/glyph_A.pgm");

    let non_zero = buffer.iter().filter(|&&v| v > 0).count();
    assert!(non_zero > 0, "Glyph should have some coverage");

    println!("\nGlyph 'A':");
    let density = " .:-=+*#%@";
    for y in 0..height {
        for x in 0..width {
            let val = buffer[y * width + x];
            let idx = (val as usize * (density.len() - 1)) / 255;
            print!("{}", density.chars().nth(idx).unwrap());
        }
        println!();
    }
}
