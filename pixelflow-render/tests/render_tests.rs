use pixelflow_render::{Color, NamedColor};

#[test]
fn test_color_conversion() {
    let red = Color::Named(NamedColor::Red);
    let red_u32: u32 = red.into();
    // Red is (205, 0, 0), format is 0xAABBGGRR
    // R=205 (CD), G=0, B=0, A=255 (FF) -> 0xFF0000CD
    assert_eq!(red_u32, 0xFF0000CD);

    let rgb = Color::Rgb(10, 20, 30);
    let rgb_u32: u32 = rgb.into();
    // 0xAABBGGRR -> 0xFF1E140A
    assert_eq!(rgb_u32, 0xFF1E140A);
}
