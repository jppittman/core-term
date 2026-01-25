use pixelflow_graphics::render::color::{Color, NamedColor};

#[test]
fn palette_entries_should_match_named_colors_definitions() {
    for i in 0..16 {
        let named = NamedColor::from_index(i);
        let from_palette: u32 = Color::Named(named).into();

        let (r, g, b) = named.to_rgb();
        let expected = u32::from_le_bytes([r, g, b, 255]);

        assert_eq!(
            from_palette, expected,
            "Palette mismatch for NamedColor::{:?} (index {})",
            named, i
        );
    }
}

#[test]
fn palette_entries_should_form_grayscale_ramp() {
    for i in 232..256 {
        let color = Color::Indexed(i as u8);
        let from_palette: u32 = color.into();

        // Logic from generate_palette:
        // let gray_idx = idx - GRAYSCALE_OFFSET (232);
        // let level = gray_idx * 10 + 8;
        let gray_idx = i - 232;
        let level = (gray_idx * 10 + 8) as u8;
        let expected = u32::from_le_bytes([level, level, level, 255]);

        assert_eq!(
            from_palette, expected,
            "Palette mismatch for grayscale index {}",
            i
        );
    }
}
