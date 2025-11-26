//! Diagnostic tests to understand why characters appear as blocks.

#![cfg(feature = "fonts")]

use pixelflow_render::glyph::render_glyph_natural;

/// Check if atlas data is actually variable (not all 0xFF).
#[test]
fn test_glyph_has_variation() {
    let glyph = render_glyph_natural('A', 24, false, false);

    println!("Glyph 'A' dimensions: {}x{}", glyph.width, glyph.height);
    println!("Glyph 'A' data length: {}", glyph.data.len());

    // Check for variation in values
    let min = *glyph.data.iter().min().unwrap();
    let max = *glyph.data.iter().max().unwrap();
    let avg: u32 = glyph.data.iter().map(|&x| x as u32).sum::<u32>() / glyph.data.len() as u32;

    println!("Min: {}, Max: {}, Avg: {}", min, max, avg);

    // Count pixels in different ranges
    let black_count = glyph.data.iter().filter(|&&x| x < 32).count();
    let gray_count = glyph.data.iter().filter(|&&x| x >= 32 && x < 224).count();
    let white_count = glyph.data.iter().filter(|&&x| x >= 224).count();

    println!("Black (<32): {}, Gray (32-223): {}, White (≥224): {}",
        black_count, gray_count, white_count);

    // Visualize the glyph as ASCII art
    println!("\nGlyph 'A' visualization:");
    for y in 0..glyph.height {
        for x in 0..glyph.width {
            let val = glyph.data[y * glyph.width + x];
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
        println!();
    }

    // A proper 'A' glyph should NOT be all white or all black
    assert!(max > 0, "Glyph should have some visible pixels");
    assert!(min < 255, "Glyph should have some background");
    assert!(max - min > 64, "Glyph should have contrast, got range {}-{}", min, max);

    // Should have mostly black (background) with some white (foreground)
    assert!(
        black_count > white_count,
        "Background should dominate, got black={} white={}",
        black_count,
        white_count
    );
}

/// Test multiple characters to see if pattern is consistent.
#[test]
fn test_multiple_characters() {
    for ch in ['A', 'B', 'O', 'I', 'W'] {
        let glyph = render_glyph_natural(ch, 24, false, false);

        let min = *glyph.data.iter().min().unwrap();
        let max = *glyph.data.iter().max().unwrap();
        let white_count = glyph.data.iter().filter(|&&x| x >= 224).count();
        let total = glyph.data.len();
        let white_pct = (white_count * 100) / total;

        println!("'{}': {}x{}, range {}-{}, white: {}%",
            ch, glyph.width, glyph.height, min, max, white_pct);

        // If all characters are mostly white, that's the bug
        if white_pct > 80 {
            panic!("Character '{}' is {}% white - appears as block!", ch, white_pct);
        }
    }
}

/// Test that the 4-bit atlas data itself has variation.
#[test]
fn test_atlas_data_has_variation() {
    // This tests the raw GLYPH_DATA to ensure it's not corrupted
    use pixelflow_render::glyph::render_glyph_natural;

    // Render a known complex glyph
    let glyph = render_glyph_natural('W', 24, false, false);

    println!("\nGlyph 'W' (complex character):");
    println!("Dimensions: {}x{}", glyph.width, glyph.height);

    let min = *glyph.data.iter().min().unwrap();
    let max = *glyph.data.iter().max().unwrap();

    // W should have lots of detail
    assert!(
        max - min > 128,
        "Complex glyph 'W' should have high contrast, got {}-{}",
        min,
        max
    );
}

/// Test scaling to see if the problem is in the shader or the atlas.
#[test]
fn test_different_sizes_for_blocks() {
    // If blocks appear at all sizes, it's probably the atlas or shader
    // If blocks only appear at certain sizes, it's the scaling

    for height in [12, 24, 48] {
        let glyph = render_glyph_natural('A', height, false, false);

        let white_count = glyph.data.iter().filter(|&&x| x >= 224).count();
        let white_pct = (white_count * 100) / glyph.data.len();

        println!("'A' at {}pt: {}% white", height, white_pct);

        assert!(
            white_pct < 50,
            "At {}pt, glyph is {}% white (block)",
            height,
            white_pct
        );
    }
}
