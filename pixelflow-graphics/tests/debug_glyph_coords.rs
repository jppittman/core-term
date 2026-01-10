//! Debug test to understand glyph geometry

use pixelflow_graphics::fonts::{text, Font};
use pixelflow_graphics::render::color::{Grayscale, Rgba8};
use pixelflow_graphics::render::{execute, TensorShape};

const FONT_BYTES: &[u8] = include_bytes!("../assets/NotoSansMono-Regular.ttf");

#[test]
fn debug_font_parsing() {
    let font = Font::parse(FONT_BYTES).expect("Failed to parse font");

    println!("\n=== Font Metrics ===");
    println!("units_per_em: {}", font.units_per_em);
    println!("ascent: {} (top of em square, Y-up)", font.ascent);
    println!("descent: {} (bottom of em square, Y-up)", font.descent);

    // In TTF, y_min should be the BOTTOM of the glyph (lower y value)
    // and y_max should be the TOP (higher y value) since Y increases upward
    println!("\nExpected for 'A' in a y-up coordinate system:");
    println!("- y_min should be ~0 (sits on baseline)");
    println!("- y_max should be ~cap_height (top of letter)");

    // Let's check what the raw glyph looks like
    let glyph = font.glyph('A');
    println!("\nRaw glyph 'A': {:?}", glyph);

    // Also check the scaled glyph
    let scaled = font.glyph_scaled('A', 48.0);
    println!("\nScaled glyph 'A' (size 48): {:?}", scaled);
}

#[test]
fn debug_coverage_at_points() {
    let font = Font::parse(FONT_BYTES).expect("Failed to parse font");

    // Create text 'A' at size 32
    let glyph = text(&font, "A", 32.0);
    let color_manifold = Grayscale(glyph);

    // For size 32:
    // scale = 32/1000 = 0.032
    // y_offset = ascent * scale = 1069 * 0.032 = 34.2
    let scale = 32.0 / font.units_per_em as f32;
    let y_offset = font.ascent as f32 * scale;

    println!("\n=== Coverage Debug ===");
    println!("scale: {}", scale);
    println!("y_offset (baseline in screen coords): {}", y_offset);

    // Render to a 40x45 buffer
    let width = 40;
    let height = 45;
    let mut pixels: Vec<Rgba8> = vec![Rgba8::default(); width * height];
    execute(
        &color_manifold,
        &mut pixels,
        TensorShape::new(width, height),
    );

    // Print cross-sections at different Y values
    println!("\nQuerying coverage at x=15 for various screen Y values:");
    println!("Screen Y | Font Y   | Expected region            | Intensity");
    println!("---------|----------|----------------------------|----------");

    let x = 15;
    for screen_y in [10, 15, 20, 25, 30, 35] {
        let font_y = (y_offset - screen_y as f32) / scale;
        let region = if font_y > 600.0 {
            "apex (should be narrow)    "
        } else if font_y > 300.0 {
            "upper body (crossbar area) "
        } else {
            "legs (should be wide)      "
        };

        let intensity = if screen_y < height {
            pixels[screen_y * width + x].r()
        } else {
            0
        };
        println!(
            "{:8} | {:8.1} | {} | {:3}",
            screen_y, font_y, region, intensity
        );
    }

    // Print X cross-sections at different Y values
    println!(
        "\nX cross-section at screen Y=15 (font Y={:.0}, should be NARROW near apex):",
        (y_offset - 15.0) / scale
    );
    print!("  ");
    for x in 0..40 {
        let intensity = pixels[15 * width + x].r();
        let ch = if intensity > 128 {
            '#'
        } else if intensity > 32 {
            '.'
        } else {
            ' '
        };
        print!("{}", ch);
    }
    println!();

    println!(
        "\nX cross-section at screen Y=22 (font Y={:.0}, crossbar area):",
        (y_offset - 22.0) / scale
    );
    print!("  ");
    for x in 0..40 {
        let intensity = pixels[22 * width + x].r();
        let ch = if intensity > 128 {
            '#'
        } else if intensity > 32 {
            '.'
        } else {
            ' '
        };
        print!("{}", ch);
    }
    println!();

    println!(
        "\nX cross-section at screen Y=30 (font Y={:.0}, should be WIDE at legs):",
        (y_offset - 30.0) / scale
    );
    print!("  ");
    for x in 0..40 {
        let intensity = pixels[30 * width + x].r();
        let ch = if intensity > 128 {
            '#'
        } else if intensity > 32 {
            '.'
        } else {
            ' '
        };
        print!("{}", ch);
    }
    println!();

    // Also print the full 'A' for reference
    println!("\nFull 'A' rendering (annotated with font Y on left):");
    for y in 0..height {
        let font_y = (y_offset - y as f32) / scale;
        print!("{:6.0} | ", font_y);
        for x in 0..width {
            let intensity = pixels[y * width + x].r();
            let ch = if intensity > 128 {
                '#'
            } else if intensity > 32 {
                '.'
            } else {
                ' '
            };
            print!("{}", ch);
        }
        println!();
    }
}
