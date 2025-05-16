//! Defines the `Glyph` type, its attributes, and color representations.

use bitflags::bitflags;
use std::fmt;
use super::color::Color;

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


// Optional: Implement Display for debugging or other purposes
impl fmt::Display for Glyph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.c)
    }
}
