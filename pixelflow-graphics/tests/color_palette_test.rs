//! Tests for the color palette generation logic.
//!
//! Verifies that `Color::Indexed` correctly maps to the xterm-256 color palette.
//! This ensures that the constants `CUBE_SCALE_FACTOR`, `CUBE_BASE_OFFSET`,
//! `GRAYSCALE_STEP`, and `GRAYSCALE_BASE` remain correct and match standard behavior.

use pixelflow_graphics::render::color::Color;

#[test]
fn color_indexed_should_match_xterm_standard() {
    // Helper to check RGBA values (alpha is always 255)
    fn check(idx: u8, expected_r: u8, expected_g: u8, expected_b: u8) {
        let color = Color::Indexed(idx);
        let rgba = color.to_rgba8();
        assert_eq!(
            rgba.r(), expected_r,
            "Red mismatch for index {}. Expected {}, got {}", idx, expected_r, rgba.r()
        );
        assert_eq!(
            rgba.g(), expected_g,
            "Green mismatch for index {}. Expected {}, got {}", idx, expected_g, rgba.g()
        );
        assert_eq!(
            rgba.b(), expected_b,
            "Blue mismatch for index {}. Expected {}, got {}", idx, expected_b, rgba.b()
        );
        assert_eq!(rgba.a(), 255, "Alpha mismatch for index {}", idx);
    }

    // 1. ANSI Colors (0-15) - Basic check to ensure they are not black (except 0)
    // We trust the lookup table for these, but good to check one.
    check(1, 205, 0, 0); // Red (Index 1)

    // 2. Color Cube (16-231)
    // Logic: val = if comp == 0 { 0 } else { comp * 40 + 55 }

    // Index 16: (0, 0, 0) - Black
    // r=0, g=0, b=0
    check(16, 0, 0, 0);

    // Index 17: (0, 0, 1) -> (0, 0, 95)
    // r=0, g=0, b=1 -> b = 1*40 + 55 = 95
    check(17, 0, 0, 95);

    // Index 18: (0, 0, 2) -> (0, 0, 135)
    // b = 2*40 + 55 = 135
    check(18, 0, 0, 135);

    // Index 21: (0, 0, 5) -> (0, 0, 255)
    // b = 5*40 + 55 = 255
    check(21, 0, 0, 255);

    // Index 22: (0, 1, 0) -> (0, 95, 0)
    // cube_idx = 22-16 = 6.
    // b_comp = 6 % 6 = 0. g_comp = (6/6)%6 = 1. r_comp = 0.
    check(22, 0, 95, 0);

    // Index 196: (5, 0, 0) -> (255, 0, 0)
    // cube_idx = 196-16 = 180.
    // r_comp = 180 / 36 = 5. g_comp = 0. b_comp = 0.
    check(196, 255, 0, 0);

    // Index 231: (5, 5, 5) -> (255, 255, 255) - White
    check(231, 255, 255, 255);

    // 3. Grayscale Ramp (232-255)
    // Logic: val = idx * 10 + 8

    // Index 232: Start of ramp
    // gray_idx = 0 -> 8
    check(232, 8, 8, 8);

    // Index 233: Next step
    // gray_idx = 1 -> 18
    check(233, 18, 18, 18);

    // Index 255: End of ramp
    // gray_idx = 23 -> 238
    check(255, 238, 238, 238);
}
