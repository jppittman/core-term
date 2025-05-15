//! Defines the `Glyph` type, its attributes, and color representations.

use bitflags::bitflags;
use std::fmt;

/// Represents a single character cell on the screen.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Glyph {
    /// The character displayed.
    pub c: char,
    /// The visual attributes of the character.
    pub attr: Attributes,
}

/// Default glyph: a space with default attributes.
pub const DEFAULT_GLYPH: Glyph = Glyph {
    c: ' ',
    // Default attributes will use NamedColor::DefaultFore/Background if defined,
    // or specific colors like White/Black. Let's use specific common defaults.
    attr: Attributes {
        fg: Color::Default,
        bg: Color::Default,
        flags: AttrFlags::empty(),
    },
};

// REPLACEMENT_CHARACTER is defined in Rust's std::char, but if you need a specific
// glyph for it:
// pub const REPLACEMENT_GLYPH: Glyph = Glyph {
//     c: '\u{FFFD}', // Unicode Replacement Character
//     attr: Attributes { /* default attributes */ },
// };

bitflags! {
    /// Represents text attributes like bold, underline, etc.
    /// Matches definitions in st.h for compatibility where possible.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
    pub struct AttrFlags: u16 {
        const BOLD          = 1 << 0;
        const FAINT         = 1 << 1; // Not always supported or distinct from normal
        const ITALIC        = 1 << 2;
        const UNDERLINE     = 1 << 3;
        const BLINK         = 1 << 4; // Slow blink
        const REVERSE       = 1 << 5; // Swaps foreground and background
        const HIDDEN        = 1 << 6; // aka Conceal
        const STRIKETHROUGH = 1 << 7;
        // const WRAP          = 1 << 8; // Line wrap indicator (if needed in glyph itself)
        // const WIDE          = 1 << 9; // Indicates a wide character occupying two cells
        // const UNDERLINE_DOUBLE = 1 << 10; // TODO
        // const UNDERLINE_CURLY  = 1 << 11; // TODO
    }
}

/// Represents the visual attributes of a glyph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Attributes {
    pub fg: Color,
    pub bg: Color,
    pub flags: AttrFlags,
}

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

// Optional: Implement Display for debugging or other purposes
impl fmt::Display for Glyph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.c)
    }
}
