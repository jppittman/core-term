use pixelflow_core::{Batch, SimdBatch};
use pixelflow_graphics::render::rasterizer::execute;
use pixelflow_graphics::render::{font, gamma_decode, gamma_encode, Rgba, SubpixelBlend};

#[test]
fn verify_subpixel_blend_contract() {
    let f = font();
    let glyph = f.glyph('A', 20.0).expect("Should find 'A'");

    let bg = Rgba::new(255, 255, 255, 255);
    let fg = Rgba::new(0, 0, 0, 255);

    let b = SubpixelBlend::new(glyph, fg, bg);

    let mut target = vec![Rgba::default(); 20 * 20];
    let width = 20;
    let height = 20;
    let shape = pixelflow_graphics::render::rasterizer::TensorShape::new(width, height, width);
    execute(&b, &mut target, shape);
}

#[test]
fn verify_gamma_correction_contract() {
    let gray = Batch::<u32>::splat(0xFF7F7F7F); // ~0.5 in sRGB

    let decoded = gamma_decode(gray);
    // 0.5^2.2 is ~0.21. 0.21 * 255 is ~55 (0x37)
    assert!(decoded.first() & 0xFF < 0x7F);

    let encoded = gamma_encode(decoded);
    // Should be close to original gray
    let diff = (encoded.first() as i32 & 0xFF) - (gray.first() as i32 & 0xFF);
    assert!(diff.abs() < 5);
}
