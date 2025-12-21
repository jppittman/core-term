//! Test to render a glyph to an image file.

use pixelflow_core::{materialize, Field, Manifold, PARALLELISM};
use pixelflow_graphics::fonts::loopblinn::SmoothStepExt;
use pixelflow_graphics::fonts::Font;
use std::fs::File;
use std::io::Write;

const FONT_BYTES: &[u8] = include_bytes!("../assets/NotoSansMono-Regular.ttf");

/// Write a grayscale image as PPM (P5 format).
fn write_ppm(path: &str, width: usize, height: usize, data: &[u8]) {
    let mut file = File::create(path).expect("Failed to create file");
    // PGM header (grayscale)
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

    // Get glyph for 'A' at 64px
    let glyph = font.glyph('A', 64.0).expect("Glyph 'A' not found");

    println!("Glyph bounds: {:?}", glyph.bounds);
    println!("Glyph segments: {}", glyph.segments.len());
    println!("Glyph bounds: {:?}", glyph.bounds);
    println!("Glyph segments: {}", glyph.segments.len());

    // Debug: evaluate at a few sample points
    use pixelflow_core::Manifold;
    let zero = Field::from(0.0);

    // Sample at corners and center
    let samples = [(0.0, 0.0), (32.0, 32.0), (10.0, 10.0), (20.0, 40.0)];

    for (sx, sy) in samples {
        let x = Field::from(sx);
        let y = Field::from(sy);
        let result = glyph.eval_raw(x, y, zero, zero);
        println!("Sample ({}, {}): {:?}", sx, sy, result);
    }

    // Render 64x64 image
    let width = 64;
    let height = 64;

    let buffer = render_manifold(&glyph, width, height);

    // Write to file
    write_ppm("/tmp/glyph_A.pgm", width, height, &buffer);

    println!("\nWrote glyph to /tmp/glyph_A.pgm");

    // Basic sanity check: should have some non-zero pixels
    let non_zero = buffer.iter().filter(|&&v| v > 0).count();
    let totally_white = buffer.iter().filter(|&&v| v == 255).count();
    println!(
        "Non-zero pixels: {}, Fully white: {}",
        non_zero, totally_white
    );
    assert!(non_zero > 0, "Glyph should have some coverage");
}
