//! Basic shader pipeline tests.

use pixelflow_core::{TensorView, TensorViewMut};
use pixelflow_render::shader::{self, FontWeight, GlyphParams, GlyphStyle, Projection};

/// Test that shader doesn't crash with minimal input.
#[test]
fn test_shader_smoke_test() {
    // 2x2 atlas with one white pixel
    let atlas_packed = [0xF0, 0x00];
    let atlas = TensorView::new(&atlas_packed, 2, 2, 1);

    let mut output = vec![0u32; 4];
    let mut dst = TensorViewMut::new(&mut output, 2, 2, 2);

    let params = GlyphParams {
        style: GlyphStyle {
            fg: 0xFF_FF_FF_FF,
            bg: 0x00_00_00_00,
            weight: FontWeight::Normal,
        },
        x_proj: Projection::scale(2, 2),
        y_proj: Projection::scale(2, 2),
    };

    shader::render_glyph(&mut dst, &atlas, params);

    // Should produce some non-zero output
    assert!(output.iter().any(|&px| px != 0), "Should have some visible pixels");
}

/// Test that empty atlas produces black output.
#[test]
fn test_shader_empty_atlas() {
    let atlas_packed = [0x00, 0x00];
    let atlas = TensorView::new(&atlas_packed, 2, 2, 1);

    let mut output = vec![0u32; 4];
    let mut dst = TensorViewMut::new(&mut output, 2, 2, 2);

    let params = GlyphParams {
        style: GlyphStyle {
            fg: 0xFF_FF_FF_FF,
            bg: 0x00_00_00_00,
            weight: FontWeight::Normal,
        },
        x_proj: Projection::scale(2, 2),
        y_proj: Projection::scale(2, 2),
    };

    shader::render_glyph(&mut dst, &atlas, params);

    // All pixels should be black (background)
    for &px in &output {
        let r = (px & 0xFF) as u8;
        assert_eq!(r, 0, "Empty atlas should produce black pixels");
    }
}

/// Test that full atlas produces white output.
#[test]
fn test_shader_full_atlas() {
    let atlas_packed = [0xFF, 0xFF];
    let atlas = TensorView::new(&atlas_packed, 2, 2, 1);

    let mut output = vec![0u32; 4];
    let mut dst = TensorViewMut::new(&mut output, 2, 2, 2);

    let params = GlyphParams {
        style: GlyphStyle {
            fg: 0xFF_FF_FF_FF,
            bg: 0x00_00_00_00,
            weight: FontWeight::Normal,
        },
        x_proj: Projection::scale(2, 2),
        y_proj: Projection::scale(2, 2),
    };

    shader::render_glyph(&mut dst, &atlas, params);

    // All pixels should be white (foreground), allow ±1 for rounding
    for &px in &output {
        let r = (px & 0xFF) as u8;
        assert!(r >= 254, "Full atlas should produce white pixels, got {}", r);
    }
}

/// Test downscaling from 4x4 to 2x2.
#[test]
fn test_shader_downscale() {
    // 4x4 checkerboard
    let atlas_packed = [0xF0, 0xF0, 0x0F, 0x0F, 0xF0, 0xF0, 0x0F, 0x0F];
    let atlas = TensorView::new(&atlas_packed, 4, 4, 2);

    let mut output = vec![0u32; 4];
    let mut dst = TensorViewMut::new(&mut output, 2, 2, 2);

    let params = GlyphParams {
        style: GlyphStyle {
            fg: 0xFF_FF_FF_FF,
            bg: 0x00_00_00_00,
            weight: FontWeight::Normal,
        },
        x_proj: Projection::scale(4, 2),
        y_proj: Projection::scale(4, 2),
    };

    shader::render_glyph(&mut dst, &atlas, params);

    // Should produce gray pixels (average of black and white)
    for &px in &output {
        let r = (px & 0xFF) as u8;
        assert!(r > 64 && r < 192, "Downscaled checkerboard should be gray-ish, got {}", r);
    }
}

/// Test that bold makes pixels brighter/thicker.
#[test]
fn test_shader_bold_synthesis() {
    // Single vertical stripe
    let atlas_packed = [0x0F, 0x00, 0x0F, 0x00];
    let atlas = TensorView::new(&atlas_packed, 3, 2, 2);

    // Render normal
    let mut output_normal = vec![0u32; 6];
    let mut dst_normal = TensorViewMut::new(&mut output_normal, 3, 2, 3);

    let params_normal = GlyphParams {
        style: GlyphStyle {
            fg: 0xFF_FF_FF_FF,
            bg: 0x00_00_00_00,
            weight: FontWeight::Normal,
        },
        x_proj: Projection::scale(3, 3),
        y_proj: Projection::scale(2, 2),
    };

    shader::render_glyph(&mut dst_normal, &atlas, params_normal);

    // Render bold
    let mut output_bold = vec![0u32; 6];
    let mut dst_bold = TensorViewMut::new(&mut output_bold, 3, 2, 3);

    let params_bold = GlyphParams {
        style: GlyphStyle {
            fg: 0xFF_FF_FF_FF,
            bg: 0x00_00_00_00,
            weight: FontWeight::Bold,
        },
        x_proj: Projection::scale(3, 3),
        y_proj: Projection::scale(2, 2),
    };

    shader::render_glyph(&mut dst_bold, &atlas, params_bold);

    // Bold should have more "ink" (higher sum of pixel values)
    let normal_sum: u32 = output_normal.iter().map(|&px| (px & 0xFF)).sum();
    let bold_sum: u32 = output_bold.iter().map(|&px| (px & 0xFF)).sum();

    assert!(
        bold_sum >= normal_sum,
        "Bold should have at least as much ink as normal (bold={}, normal={})",
        bold_sum,
        normal_sum
    );
}
