//! Debug test to understand shader behavior.

use pixelflow_core::{Batch, TensorView, TensorViewMut};
use pixelflow_render::shader::{self, FontWeight, GlyphParams, GlyphStyle, Projection};

#[test]
fn debug_shader_values() {
    // Simple 2x2 atlas with all white (0xFF)
    let atlas_packed = [0xFFu8, 0xFF];
    let atlas = TensorView::new(&atlas_packed, 2, 2, 1);

    // Manually test gather_4bit
    let x = Batch::<u32>::new(0, 1, 0, 1);
    let y = Batch::<u32>::new(0, 0, 1, 1);

    let gathered = unsafe { atlas.gather_4bit(x, y) };

    println!("Gathered values: [{}, {}, {}, {}]",
        gathered.extract(0),
        gathered.extract(1),
        gathered.extract(2),
        gathered.extract(3)
    );

    // All should be 255
    assert_eq!(gathered.extract(0), 255, "Pixel (0,0) should be 255");
    assert_eq!(gathered.extract(1), 255, "Pixel (1,0) should be 255");
    assert_eq!(gathered.extract(2), 255, "Pixel (0,1) should be 255");
    assert_eq!(gathered.extract(3), 255, "Pixel (1,1) should be 255");
}

#[test]
fn debug_blend_alpha() {
    // Test blend_alpha directly
    let fg = Batch::<u32>::splat(0xFF_FF_FF_FF);
    let bg = Batch::<u32>::splat(0x00_00_00_00);

    // Test with full alpha (replicated across channels)
    let alpha = Batch::<u32>::splat(0xFF_FF_FF_FF);

    let result = fg.blend_alpha(bg, alpha);

    let r = (result.extract(0) & 0xFF) as u8;
    println!("Blend result (R channel): {}", r);

    assert!(r > 200, "With full alpha, should get mostly foreground, got {}", r);
}

#[test]
fn debug_shader_step_by_step() {
    // 2x2 white atlas
    let atlas_packed = [0xFFu8, 0xFF];
    let atlas = TensorView::new(&atlas_packed, 2, 2, 1);

    let mut output = vec![0u32; 1]; // Just one pixel
    let mut dst = TensorViewMut::new(&mut output, 1, 1, 1);

    let params = GlyphParams {
        style: GlyphStyle {
            fg: 0xFF_FF_FF_FF,
            bg: 0x00_00_00_00,
            weight: FontWeight::Normal,
        },
        x_proj: Projection::scale(2, 1), // Sample center of 2x2 atlas
        y_proj: Projection::scale(2, 1),
    };

    shader::render_glyph(&mut dst, &atlas, params);

    let pixel = output[0];
    let r = (pixel & 0xFF) as u8;
    let g = ((pixel >> 8) & 0xFF) as u8;
    let b = ((pixel >> 16) & 0xFF) as u8;
    let a = ((pixel >> 24) & 0xFF) as u8;

    println!("Final pixel ARGB: [{}, {}, {}, {}]", a, r, g, b);

    assert!(r > 200, "R channel should be bright, got {}", r);
}
