#[cfg(test)]
mod tests {
    use crate::render::color::Color;

    #[test]
    fn test_color_indexed_parity() {
        // Test standard colors
        assert_eq!(u32::from(Color::Indexed(0)), u32::from_le_bytes([0, 0, 0, 255])); // Black
        assert_eq!(u32::from(Color::Indexed(1)), u32::from_le_bytes([205, 0, 0, 255])); // Red
        assert_eq!(u32::from(Color::Indexed(15)), u32::from_le_bytes([255, 255, 255, 255])); // Bright White

        // Test color cube (16-231)
        // Index 16: (0,0,0) -> 0x000000FF (RGB) -> LE: 00 00 00 FF
        assert_eq!(u32::from(Color::Indexed(16)), u32::from_le_bytes([0, 0, 0, 255]));

        // Index 21: (0,0,5) -> Blue=255 -> (0, 0, 255)
        // Calculation: 21-16=5. r=0, g=0, b=5. b_val = 5*40+55 = 255.
        assert_eq!(u32::from(Color::Indexed(21)), u32::from_le_bytes([0, 0, 255, 255]));

        // Index 196: Red
        // 196-16 = 180.
        // 180 / 36 = 5 (r=5) -> 255
        // 180 % 36 / 6 = 0 (g=0) -> 0
        // 180 % 6 = 0 (b=0) -> 0
        assert_eq!(u32::from(Color::Indexed(196)), u32::from_le_bytes([255, 0, 0, 255]));

        // Test grayscale (232-255)
        // Index 232: (0) -> 8,8,8
        assert_eq!(u32::from(Color::Indexed(232)), u32::from_le_bytes([8, 8, 8, 255]));
        // Index 255: (23) -> 23*10+8 = 238
        assert_eq!(u32::from(Color::Indexed(255)), u32::from_le_bytes([238, 238, 238, 255]));
    }
}
