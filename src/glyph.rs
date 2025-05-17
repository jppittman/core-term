// src/glyph.rs

//! Defines the `Glyph` type, its visual attributes (`AttrFlags`, `Attributes`),
//! and related constants.
//!
//! A `Glyph` represents a single character cell on the terminal screen,
//! encapsulating the character itself and all its styling information.
//! Color definitions (`Color`, `NamedColor`) are found in the `crate::color` module.

use bitflags::bitflags; // For creating flag enums like AttrFlags
use std::fmt;
// Removed: `use super::color::Color;` - This was a private re-export causing issues.
// Other modules should now import `Color` directly from `crate::color`.
use crate::color::Color; // Import Color directly for use in Attributes and DEFAULT_GLYPH

/// Represents a single character cell on the screen.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Glyph {
    /// The character displayed in the cell.
    /// A `\0` (null character) often signifies the second part of a wide character.
    pub c: char,
    /// The visual attributes of the character (foreground/background color, flags).
    pub attr: Attributes,
}

/// Default glyph: a space character with default SGR attributes.
/// This is used for initializing new cells or clearing existing ones.
pub const DEFAULT_GLYPH: Glyph = Glyph {
    c: ' ', // Default character is a space
    attr: Attributes {
        fg: Color::Default,        // Default foreground color
        bg: Color::Default,        // Default background color
        flags: AttrFlags::empty(), // No special attribute flags by default
    },
};

bitflags! {
    /// Represents text attribute flags like bold, underline, reverse video, etc.
    /// These flags correspond to common ANSI SGR (Select Graphic Rendition) parameters.
    ///
    /// The `bitflags` macro allows these to be combined (e.g., `AttrFlags::BOLD | AttrFlags::UNDERLINE`).
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
    pub struct AttrFlags: u16 {
        // Basic text styling attributes
        const BOLD          = 1 << 0; // Typically increases intensity or changes font weight.
        const FAINT         = 1 << 1; // Typically decreases intensity (not always supported or distinct).
        const ITALIC        = 1 << 2; // Italicizes text (font-dependent).
        const UNDERLINE     = 1 << 3; // Adds an underline.
        const BLINK         = 1 << 4; // Makes text blink (behavior varies; often slow blink).
        const REVERSE       = 1 << 5; // Swaps foreground and background colors.
        const HIDDEN        = 1 << 6; // Makes text invisible (aka Conceal).
        const STRIKETHROUGH = 1 << 7; // Puts a line through the text.

        // Placeholder for future or less common attributes, if needed.
        // const WRAP          = 1 << 8; // Example: Line wrap indicator (if stored per-glyph).
        // const WIDE          = 1 << 9; // Example: Indicates a wide character occupying two cells.
        // const UNDERLINE_DOUBLE = 1 << 10; // Example: Double underline.
        // const UNDERLINE_CURLY  = 1 << 11; // Example: Curly underline.
    }
}

/// Represents the visual attributes of a glyph, including foreground color,
/// background color, and styling flags.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Attributes {
    /// Foreground color of the glyph.
    pub fg: Color,
    /// Background color of the glyph.
    pub bg: Color,
    /// Styling flags (bold, italic, underline, etc.).
    pub flags: AttrFlags,
}

// Optional: Implement Display for Glyph for easier debugging or direct printing if needed.
// This will print only the character, not its attributes.
impl fmt::Display for Glyph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.c)
    }
}
