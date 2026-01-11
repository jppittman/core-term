//! Demo: Rasterize font glyphs to stdout as ASCII art.
//!
//! Run with: cargo test -p pixelflow-graphics font_stdout_demo -- --nocapture

use pixelflow_graphics::fonts::{text, Font};
use pixelflow_graphics::render::color::{Grayscale, Rgba8};
use pixelflow_graphics::render::frame::Frame;
use pixelflow_graphics::render::rasterizer::rasterize;

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
    let glyph = text(&font, "A", 32.0);
    let color_manifold = Grayscale(glyph);

    let width = 40;
    let height = 45;
    let mut frame = Frame::<Rgba8>::new(width as u32, height as u32);

    rasterize(
        &color_manifold,
        &mut frame,
        1,
    );
    
    let pixels = frame.data;

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
    let glyph = text(&font, "HELLO", 20.0);

    // Wrap in Grayscale to convert coverage (Field) to grayscale pixels (Discrete)
    let color_manifold = Grayscale(glyph);

    // Create a framebuffer
    let width = 100;
    let height = 30;
    let mut frame = Frame::<Rgba8>::new(width as u32, height as u32);

    // Rasterize!
    rasterize(
        &color_manifold,
        &mut frame,
        1,
    );
    let pixels = frame.data;

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
    let glyph = text(&font, "ABCDEFGHIJKLM", 16.0);
    let color_manifold = Grayscale(glyph);

    let width = 180;
    let height = 24;
    let mut frame = Frame::<Rgba8>::new(width as u32, height as u32);

    rasterize(
        &color_manifold,
        &mut frame,
        1,
    );
    let pixels = frame.data;

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
    let glyph2 = text(&font, "NOPQRSTUVWXYZ", 16.0);
    let color_manifold2 = Grayscale(glyph2);

    let mut frame2 = Frame::<Rgba8>::new(width as u32, height as u32);
    rasterize(
        &color_manifold2,
        &mut frame2,
        1,
    );
    let pixels2 = frame2.data;

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
