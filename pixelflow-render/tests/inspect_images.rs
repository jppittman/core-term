//! Inspect the rendered images to verify correctness.

use std::fs;

fn visualize_pgm_ascii(path: &str) {
    let data = fs::read(path).unwrap();

    // Skip PGM header (P5\n<width> <height>\n255\n)
    let mut offset = 0;
    while offset < data.len() && data[offset] != b'\n' { offset += 1; } // Skip "P5"
    offset += 1;
    while offset < data.len() && data[offset] != b'\n' { offset += 1; } // Skip dimensions
    offset += 1;
    while offset < data.len() && data[offset] != b'\n' { offset += 1; } // Skip "255"
    offset += 1;

    let pixels = &data[offset..];

    // Find dimensions from header
    let header = String::from_utf8_lossy(&data[..offset]);
    let lines: Vec<&str> = header.lines().collect();
    let dims: Vec<usize> = lines[1].split_whitespace()
        .map(|s| s.parse().unwrap())
        .collect();
    let width = dims[0];
    let height = dims[1];

    println!("\nImage: {} ({}x{})", path, width, height);

    // Print first few rows as ASCII
    let rows_to_show = height.min(20);
    for y in 0..rows_to_show {
        for x in 0..width {
            let idx = y * width + x;
            if idx < pixels.len() {
                let val = pixels[idx];
                let ch = if val < 32 {
                    ' '
                } else if val < 96 {
                    '.'
                } else if val < 160 {
                    '+'
                } else if val < 224 {
                    '#'
                } else {
                    '█'
                };
                print!("{}", ch);
            }
        }
        println!();
    }
}

#[test]
fn inspect_hello_world() {
    visualize_pgm_ascii("/tmp/hello_world.pgm");
}

#[test]
fn inspect_alphabet() {
    visualize_pgm_ascii("/tmp/alphabet.pgm");
}

#[test]
fn inspect_large_a() {
    visualize_pgm_ascii("/tmp/char_A_large.pgm");
}

#[test]
fn check_image_not_all_black() {
    let data = fs::read("/tmp/hello_world.pgm").unwrap();

    // Skip header
    let mut offset = 0;
    while offset < data.len() && data[offset] != b'\n' { offset += 1; }
    offset += 1;
    while offset < data.len() && data[offset] != b'\n' { offset += 1; }
    offset += 1;
    while offset < data.len() && data[offset] != b'\n' { offset += 1; }
    offset += 1;

    let pixels = &data[offset..];

    let non_zero = pixels.iter().filter(|&&x| x > 0).count();
    let all_255 = pixels.iter().filter(|&&x| x == 255).count();

    println!("Non-zero pixels: {} / {} ({:.1}%)",
        non_zero, pixels.len(),
        100.0 * non_zero as f32 / pixels.len() as f32);
    println!("All-255 pixels: {} ({:.1}%)",
        all_255,
        100.0 * all_255 as f32 / pixels.len() as f32);

    assert!(non_zero > 0, "Image should not be all black");
    assert!(all_255 < pixels.len(), "Image should not be all white");
    assert!(all_255 < pixels.len() / 2, "Image should not be mostly white (blocks)");
}
