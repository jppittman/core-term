//! Regression test for the alpha channel broadcast bug.
//!
//! Bug: rasterizer.rs was using [255, 255, 255, a0] for alpha pixels,
//! which forced RGB channels to 100% opacity, resulting in solid blocks
//! instead of antialiased glyphs.
//!
//! Fix: Broadcast alpha to all channels: [a0, a0, a0, a0]

#![cfg(feature = "fonts")]

use pixelflow_render::commands::Op;
use pixelflow_render::rasterizer::process_frame;
use pixelflow_render::types::{Color, NamedColor};

#[test]
fn test_rasterizer_alpha_broadcast() {
    // Create a framebuffer
    let width = 32;
    let height = 32;
    let cell_width = 16;
    let cell_height = 24;
    let mut fb = vec![0x00_00_00_00u32; width * height];

    // Draw white 'O' on black background
    let commands: Vec<Op<Vec<u8>>> = vec![
        Op::Clear { color: Color::Named(NamedColor::Black) },
        Op::Text {
            ch: 'O',
            x: 8, // Center horizontally
            y: 4, // Center vertically
            fg: Color::Named(NamedColor::BrightWhite),
            bg: Color::Named(NamedColor::Black),
        },
    ];

    process_frame(&mut fb, width, height, cell_width, cell_height, &commands);

    // Analyze the framebuffer to verify antialiasing exists
    let mut fully_opaque_pixels = 0;   // RGB all 255
    let mut fully_transparent = 0;     // RGB all 0
    let mut antialiased_pixels = 0;    // RGB between 0 and 255

    for &pixel in &fb {
        let r = (pixel >> 16) & 0xFF;
        let g = (pixel >> 8) & 0xFF;
        let b = pixel & 0xFF;

        if r == 255 && g == 255 && b == 255 {
            fully_opaque_pixels += 1;
        } else if r == 0 && g == 0 && b == 0 {
            fully_transparent += 1;
        } else {
            antialiased_pixels += 1;
        }
    }

    println!("Pixel analysis for 'O' glyph:");
    println!("  Fully opaque (255,255,255): {}", fully_opaque_pixels);
    println!("  Fully transparent (0,0,0): {}", fully_transparent);
    println!("  Antialiased (intermediate): {}", antialiased_pixels);

    // The bug would cause all pixels to be either fully opaque or transparent,
    // with NO antialiasing (intermediate values).
    //
    // With the fix, we should have plenty of antialiased pixels around the edges
    // of the 'O' shape.
    assert!(
        antialiased_pixels > 10,
        "Expected antialiased pixels for 'O' glyph, but got {}. \
         This indicates the alpha broadcast bug has regressed. \
         Fully opaque: {}, Fully transparent: {}",
        antialiased_pixels,
        fully_opaque_pixels,
        fully_transparent
    );

    // Additional check: the ratio of antialiased to solid pixels should be reasonable
    // for a well-rendered 'O'
    let total_visible = fully_opaque_pixels + antialiased_pixels;
    if total_visible > 0 {
        let aa_ratio = (antialiased_pixels as f32) / (total_visible as f32);
        assert!(
            aa_ratio > 0.1,
            "Antialiased pixel ratio too low: {:.1}%. Expected >10% for smooth rendering.",
            aa_ratio * 100.0
        );
    }
}

#[test]
fn test_rasterizer_blending_correctness() {
    // Test that alpha blending produces mathematically correct results
    // This catches the bug where RGB channels were forced to 255

    let width = 32;
    let height = 32;
    let cell_width = 16;
    let cell_height = 24;

    // Start with gray background
    let mut fb = vec![0x80_80_80_80u32; width * height];

    // Draw a character that we know has half-alpha pixels
    let commands: Vec<Op<Vec<u8>>> = vec![
        Op::Text {
            ch: 'O',
            x: 8,
            y: 4,
            fg: Color::Rgb(255, 255, 255),   // White
            bg: Color::Rgb(128, 128, 128),   // Gray
        },
    ];

    process_frame(&mut fb, width, height, cell_width, cell_height, &commands);

    // Find pixels with intermediate alpha values
    // These should exist if antialiasing is working
    let mut intermediate_pixels = 0;
    let mut all_white = 0;
    let mut all_gray = 0;

    for &pixel in &fb {
        let r = (pixel >> 16) & 0xFF;
        let g = (pixel >> 8) & 0xFF;
        let b = pixel & 0xFF;

        if r == 255 && g == 255 && b == 255 {
            all_white += 1;
        } else if r == 128 && g == 128 && b == 128 {
            all_gray += 1;
        } else if r > 128 && r < 255 {
            intermediate_pixels += 1;
        }
    }

    println!("Blending test results:");
    println!("  All white (255,255,255): {}", all_white);
    println!("  All gray (128,128,128): {}", all_gray);
    println!("  Intermediate (128<R<255): {}", intermediate_pixels);

    // With the bug, RGB was forced to 255, so we'd have NO intermediate values
    // With the fix, antialiased edges should produce intermediate values
    assert!(
        intermediate_pixels > 5,
        "Expected intermediate-alpha pixels from antialiasing, but got {}. \
         This indicates the alpha broadcast bug has regressed.",
        intermediate_pixels
    );
}
