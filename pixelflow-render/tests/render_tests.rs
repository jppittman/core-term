use pixelflow_render::{process_frame, Color, NamedColor, Op};

#[test]
fn test_color_conversion() {
    let red = Color::Named(NamedColor::Red);
    let red_u32: u32 = red.into();
    // Red is (205, 0, 0), so 0xFF0000CD (ABGR/RGBA depending on endianness/format)
    // format is 0xAABBGGRR
    // R=205 (CD), G=0, B=0, A=255 (FF)
    // 0xFF0000CD
    assert_eq!(red_u32, 0xFF0000CD);

    let rgb = Color::Rgb(10, 20, 30);
    let rgb_u32: u32 = rgb.into();
    // 0xAABBGGRR -> 0xFF1E140A
    assert_eq!(rgb_u32, 0xFF1E140A);
}

#[test]
fn test_process_frame_clear() {
    let width = 10;
    let height = 10;
    let mut framebuffer = vec![0u32; width * height];

    // Clear to Blue
    // Specify T as &[u8] explicitly
    let ops: Vec<Op<&[u8]>> = vec![Op::Clear {
        color: Color::Named(NamedColor::Blue),
    }];

    process_frame(&mut framebuffer, width, height, 8, 16, &ops);

    // Blue is (0, 0, 238) -> 0xAABBGGRR -> 0xFFEE0000
    let expected_color: u32 = Color::Named(NamedColor::Blue).into();

    for pixel in framebuffer {
        assert_eq!(pixel, expected_color);
    }
}

#[test]
fn test_process_frame_blit() {
    let width = 4;
    let height = 4;
    let mut framebuffer = vec![0u32; width * height];

    // 2x2 red square
    // Red: 0xFF0000CD
    let red_u32: u32 = Color::Named(NamedColor::Red).into();
    let r = 205u8;
    let g = 0u8;
    let b = 0u8;
    let a = 255u8;

    let data = vec![r, g, b, a, r, g, b, a, r, g, b, a, r, g, b, a];

    let ops = vec![Op::Blit {
        data: &data,
        w: 2,
        x: 1,
        y: 1,
    }];

    process_frame(&mut framebuffer, width, height, 4, 4, &ops);

    // Check pixel at (1,1) -> index 1*4 + 1 = 5
    assert_eq!(framebuffer[5], red_u32);
    assert_eq!(framebuffer[6], red_u32);
    assert_eq!(framebuffer[9], red_u32);
    assert_eq!(framebuffer[10], red_u32);

    // Check pixel at (0,0) -> index 0 should be 0 (default init)
    assert_eq!(framebuffer[0], 0);
}
