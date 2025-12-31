//! Demo: Rasterize font glyphs to stdout as ASCII art.
//!
//! Run with: cargo test -p pixelflow-graphics font_stdout_demo -- --nocapture

use pixelflow_graphics::fonts::{Font, Text};
use pixelflow_graphics::render::color::{Grayscale, Rgba8};
use pixelflow_graphics::render::{execute, TensorShape};

const FONT_BYTES: &[u8] = include_bytes!("../assets/NotoSansMono-Regular.ttf");

#[test]
fn demo_single_glyph_rasterization() {
    let font = Font::parse(FONT_BYTES).expect("Failed to parse font");

    println!("\n========================================");
    println!("  Font Rasterizer Demo - Single Glyph");
    println!("========================================\n");

    // Print font metrics (fields are public on Font)
    println!("Font metrics:");
    println!("  units_per_em: {}", font.units_per_em);
    println!("  ascent: {}", font.ascent);
    println!("  descent: {}", font.descent);
    println!("  line_gap: {}", font.line_gap);
    println!();

    // Render 'A' at size 32
    let text = Text::new(&font, "A", 32.0);
    let color_manifold = Grayscale(text);

    let width = 40;
    let height = 45;
    let mut pixels: Vec<Rgba8> = vec![Rgba8::default(); width * height];

    execute(
        &color_manifold,
        &mut pixels,
        TensorShape::new(width, height),
    );

    println!("ASCII render of 'A' ({}x{}):", width, height);
    println!();

    let chars = [' ', '.', ':', '+', '#', '@'];

    for y in 0..height {
        let mut line = String::new();
        for x in 0..width {
            let pixel = pixels[y * width + x];
            let intensity = pixel.r() as usize;
            let idx = (intensity * (chars.len() - 1)) / 255;
            line.push(chars[idx]);
        }
        println!("{}", line);
    }
}

#[test]
fn demo_text_rasterization_with_frame() {
    let font = Font::parse(FONT_BYTES).expect("Failed to parse font");

    println!("\n========================================");
    println!("  Font Rasterizer Demo - Full Pipeline");
    println!("========================================\n");

    // Create text
    let text = Text::new(&font, "HELLO", 20.0);

    // Wrap in Grayscale to convert coverage (Field) to grayscale pixels (Discrete)
    let color_manifold = Grayscale(text);

    // Create a framebuffer
    let width = 100;
    let height = 30;
    let mut pixels: Vec<Rgba8> = vec![Rgba8::default(); width * height];

    // Rasterize!
    execute(
        &color_manifold,
        &mut pixels,
        TensorShape::new(width, height),
    );

    // Print as ASCII art
    println!("Rasterized 'HELLO' ({}x{}):", width, height);
    println!();

    let chars = [' ', '.', ':', '-', '=', '+', '*', '#', '%', '@'];

    for y in 0..height {
        let mut line = String::new();
        for x in 0..width {
            let pixel = pixels[y * width + x];
            // Use red channel as intensity (grayscale)
            let intensity = pixel.r() as usize;
            let idx = (intensity * (chars.len() - 1)) / 255;
            line.push(chars[idx]);
        }
        println!("{}", line);
    }

    println!();
    println!("Done!");
}

#[test]
fn demo_alphabet_rasterization() {
    let font = Font::parse(FONT_BYTES).expect("Failed to parse font");

    println!("\n========================================");
    println!("  Font Rasterizer Demo - Alphabet");
    println!("========================================\n");

    // Create text with all uppercase letters
    let text = Text::new(&font, "ABCDEFGHIJKLM", 16.0);
    let color_manifold = Grayscale(text);

    let width = 180;
    let height = 24;
    let mut pixels: Vec<Rgba8> = vec![Rgba8::default(); width * height];

    execute(
        &color_manifold,
        &mut pixels,
        TensorShape::new(width, height),
    );

    let chars = [' ', '.', ':', '+', '#', '@'];

    println!("ABCDEFGHIJKLM:");
    for y in 0..height {
        let mut line = String::new();
        for x in 0..width {
            let pixel = pixels[y * width + x];
            let intensity = pixel.r() as usize;
            let idx = (intensity * (chars.len() - 1)) / 255;
            line.push(chars[idx]);
        }
        println!("{}", line);
    }
    println!();

    // Second row
    let text2 = Text::new(&font, "NOPQRSTUVWXYZ", 16.0);
    let color_manifold2 = Grayscale(text2);

    let mut pixels2: Vec<Rgba8> = vec![Rgba8::default(); width * height];
    execute(
        &color_manifold2,
        &mut pixels2,
        TensorShape::new(width, height),
    );

    println!("NOPQRSTUVWXYZ:");
    for y in 0..height {
        let mut line = String::new();
        for x in 0..width {
            let pixel = pixels2[y * width + x];
            let intensity = pixel.r() as usize;
            let idx = (intensity * (chars.len() - 1)) / 255;
            line.push(chars[idx]);
        }
        println!("{}", line);
    }
}
