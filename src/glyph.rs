// src/glyph.rs

//! Defines the `Glyph` type, its visual attributes (`AttrFlags`, `Attributes`),
//! and related constants.
//!
//! A `Glyph` represents a single character cell on the terminal screen,
//! encapsulating the character itself and all its styling information.
//! Color definitions (`Color`, `NamedColor`) are found in the `crate::color` module.

use crate::color::Color;
use bitflags::bitflags; // For creating flag enums like AttrFlags

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Glyph {
    Single(ContentCell),
    WidePrimary(ContentCell),
    WideSpacer{
        primary_column_on_line: u16,
    },
}

/// Represents a single character cell on the screen.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContentCell {
    /// The character displayed in the cell.
    /// A `\0` (null character) often signifies the second part of a wide character.
    pub c: char,
    /// The visual attributes of the character (foreground/background color, flags).
    pub attr: Attributes,
}

/// Placeholder character for the second cell of a double-width character.
pub const WIDE_CHAR_PLACEHOLDER: char = '\0';

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

        // Flags for wide character handling
        const WIDE_CHAR_PRIMARY = 1 << 14; // Indicates the first cell of a double-width character.
        const WIDE_CHAR_SPACER  = 1 << 15; // Indicates the second cell of a double-width character (placeholder).


        // Placeholder for future or less common attributes, if needed.
        // const WRAP          = 1 << 8; // Example: Line wrap indicator (if stored per-glyph).
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

impl Glyph {
    /// Provides a default Glyph instance, representing a cleared/empty cell.
    pub fn default_cell() -> Self {
        Glyph::Single(ContentCell::default_space())
    }

    /// Helper to get the displayable character of the cell, if applicable.
    /// For spacers, this might be the placeholder.
    pub fn display_char(&self) -> char {
        match self {
            Glyph::Single(cc) | Glyph::WidePrimary(cc) => cc.c,
            Glyph::WideSpacer { .. } => WIDE_CHAR_PLACEHOLDER,
        }
    }
}

impl ContentCell {
    pub fn default_space() -> Self {
        ContentCell {
            c: ' ', 
            attr: Attributes::default(),
        }
    }
}
