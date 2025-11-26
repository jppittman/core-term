//! Render text to an actual image file for visual inspection.

#![cfg(feature = "fonts")]

use pixelflow_render::glyph::render_glyph_natural;
use std::fs::File;
use std::io::Write;

/// Write a PGM (grayscale) image file.
fn write_pgm(path: &str, data: &[u8], width: usize, height: usize) {
    let mut file = File::create(path).unwrap();
    writeln!(file, "P5").unwrap();
    writeln!(file, "{} {}", width, height).unwrap();
    writeln!(file, "255").unwrap();
    file.write_all(data).unwrap();
}

#[test]
fn render_alphabet_to_image() {
    let chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    let cell_width = 16;
    let cell_height = 24;
    let cols = 9; // 9 characters per row
    let rows = (chars.len() + cols - 1) / cols;

    let img_width = cols * cell_width;
    let img_height = rows * cell_height;

    let mut image = vec![0u8; img_width * img_height];

    for (i, ch) in chars.chars().enumerate() {
        let col = i % cols;
        let row = i / cols;
        let x_offset = col * cell_width;
        let y_offset = row * cell_height;

        let glyph = render_glyph_natural(ch, cell_height, false, false);

        // Blit glyph to image
        for y in 0..glyph.height.min(cell_height) {
            for x in 0..glyph.width.min(cell_width) {
                let src_idx = y * glyph.width + x;
                let dst_x = x_offset + x;
                let dst_y = y_offset + y;
                let dst_idx = dst_y * img_width + dst_x;

                if dst_idx < image.len() && src_idx < glyph.data.len() {
                    image[dst_idx] = glyph.data[src_idx];
                }
            }
        }
    }

    write_pgm("/tmp/alphabet.pgm", &image, img_width, img_height);
    println!("\n✓ Rendered alphabet to /tmp/alphabet.pgm");
    println!("View with: open /tmp/alphabet.pgm");
    println!("Or convert to PNG: convert /tmp/alphabet.pgm /tmp/alphabet.png");
}

#[test]
fn render_single_large_char() {
    let ch = 'A';
    let glyph = render_glyph_natural(ch, 96, false, false); // Large 96pt

    write_pgm("/tmp/char_A_large.pgm", &glyph.data, glyph.width, glyph.height);
    println!("\n✓ Rendered large 'A' to /tmp/char_A_large.pgm");
    println!("Dimensions: {}x{}", glyph.width, glyph.height);
}

#[test]
fn render_hello_world() {
    let text = "Hello, World!";
    let cell_width = 14;
    let cell_height = 24;

    let img_width = text.len() * cell_width;
    let img_height = cell_height;

    let mut image = vec![0u8; img_width * img_height];

    for (i, ch) in text.chars().enumerate() {
        let x_offset = i * cell_width;
        let glyph = render_glyph_natural(ch, cell_height, false, false);

        for y in 0..glyph.height.min(cell_height) {
            for x in 0..glyph.width.min(cell_width) {
                let src_idx = y * glyph.width + x;
                let dst_x = x_offset + x;
                let dst_y = y;
                let dst_idx = dst_y * img_width + dst_x;

                if dst_idx < image.len() && src_idx < glyph.data.len() {
                    image[dst_idx] = glyph.data[src_idx];
                }
            }
        }
    }

    write_pgm("/tmp/hello_world.pgm", &image, img_width, img_height);
    println!("\n✓ Rendered 'Hello, World!' to /tmp/hello_world.pgm");
    println!("View with: open /tmp/hello_world.pgm");
}

#[test]
fn render_bold_vs_normal() {
    let ch = 'M';
    let height = 48;

    let normal = render_glyph_natural(ch, height, false, false);
    let bold = render_glyph_natural(ch, height, true, false);

    // Side-by-side comparison
    let padding = 4;
    let img_width = normal.width + padding + bold.width;
    let img_height = normal.height.max(bold.height);

    let mut image = vec![0u8; img_width * img_height];

    // Normal on left
    for y in 0..normal.height {
        for x in 0..normal.width {
            let src_idx = y * normal.width + x;
            let dst_idx = y * img_width + x;
            image[dst_idx] = normal.data[src_idx];
        }
    }

    // Bold on right
    let x_offset = normal.width + padding;
    for y in 0..bold.height {
        for x in 0..bold.width {
            let src_idx = y * bold.width + x;
            let dst_x = x_offset + x;
            let dst_idx = y * img_width + dst_x;
            image[dst_idx] = bold.data[src_idx];
        }
    }

    write_pgm("/tmp/bold_comparison.pgm", &image, img_width, img_height);
    println!("\n✓ Rendered bold comparison to /tmp/bold_comparison.pgm");
    println!("Left: normal, Right: bold");
}
