#[cfg(feature = "fonts")]
#[test]
fn test_glyph_scaling_bug() {
    use pixelflow_render::glyph::{
        render_glyph_direct, GlyphRenderCoords, GlyphStyleOverrides, RenderTarget,
    };

    // 1. Render 'I' at 24px (scale 1.0)
    let width_1x = 24;
    let height_1x = 24;
    let mut buffer_1x = vec![0u32; width_1x * height_1x];
    let target_1x = RenderTarget {
        dest: &mut buffer_1x,
        stride: width_1x,
    };

    let coords_1x = GlyphRenderCoords {
        x_px: 0,
        y_px: 0,
        cell_height: 24,
    };

    let style = GlyphStyleOverrides {
        fg: 0xFFFFFFFF, // Opaque White
        bg: 0x00000000, // Transparent
        bold: false,
        italic: false,
    };

    let metrics_1x = render_glyph_direct('I', target_1x, coords_1x, style);

    assert!(
        metrics_1x.width > 0,
        "Glyph 'I' has 0 width, cannot run regression test."
    );

    // 2. Render 'I' at 48px (scale 2.0)
    let width_2x = 48;
    let height_2x = 48;
    let mut buffer_2x = vec![0u32; width_2x * height_2x];
    let target_2x = RenderTarget {
        dest: &mut buffer_2x,
        stride: width_2x,
    };

    let coords_2x = GlyphRenderCoords {
        x_px: 0,
        y_px: 0,
        cell_height: 48,
    };

    render_glyph_direct('I', target_2x, coords_2x, style);

    // Pick a row that exists in both and has content in 1x
    let check_row = metrics_1x.height / 2;

    // Collect alpha values for comparison
    let row_data_1x: Vec<u8> = (0..metrics_1x.width)
        .map(|x| (buffer_1x[check_row * width_1x + x] >> 24) as u8)
        .collect();
    // For 2x, we look at the SAME physical row index (check_row), not the scaled one.
    // Because we want to prove that it is reading the same atlas data.
    let row_data_2x: Vec<u8> = (0..metrics_1x.width)
        .map(|x| (buffer_2x[check_row * width_2x + x] >> 24) as u8)
        .collect();

    let is_identical = row_data_1x == row_data_2x;

    assert!(
        !is_identical,
        "Regression: Scaling is ignored (1x and 2x outputs match at absolute coordinates)."
    );
}
