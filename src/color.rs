/// Standard ANSI named colors (0-15).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum NamedColor {
    Black = 0,
    Red = 1,
    Green = 2,
    Yellow = 3,
    Blue = 4,
    Magenta = 5,
    Cyan = 6,
    White = 7,       // Also known as Grey
    BrightBlack = 8, // Also known as Dark Grey
    BrightRed = 9,
    BrightGreen = 10,
    BrightYellow = 11,
    BrightBlue = 12,
    BrightMagenta = 13,
    BrightCyan = 14,
    BrightWhite = 15,
    // Special values if needed, though Default is usually handled by Color::Default
    // DefaultForeground = 256, // Example, not standard ANSI index
    // DefaultBackground = 257, // Example
}

impl NamedColor {
    /// Converts a u8 index (0-15) to a `NamedColor`.
    /// Panics if the index is out of range.
    pub fn from_index(idx: u8) -> Self {
        match idx {
            0 => NamedColor::Black,
            1 => NamedColor::Red,
            2 => NamedColor::Green,
            3 => NamedColor::Yellow,
            4 => NamedColor::Blue,
            5 => NamedColor::Magenta,
            6 => NamedColor::Cyan,
            7 => NamedColor::White,
            8 => NamedColor::BrightBlack,
            9 => NamedColor::BrightRed,
            10 => NamedColor::BrightGreen,
            11 => NamedColor::BrightYellow,
            12 => NamedColor::BrightBlue,
            13 => NamedColor::BrightMagenta,
            14 => NamedColor::BrightCyan,
            15 => NamedColor::BrightWhite,
            _ => panic!("Invalid NamedColor index: {}", idx),
        }
    }
}

/// Represents a color value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Color {
    /// Default foreground or background color.
    Default,
    /// A standard named ANSI color (0-15).
    Named(NamedColor),
    /// An indexed color from the 256-color palette (0-255).
    /// `NamedColor` covers indices 0-15; this can be used for 16-255.
    Indexed(u8),
    /// An RGB true color.
    Rgb(u8, u8, u8),
}

impl Default for Color {
    fn default() -> Self {
        Color::Default
    }
}
