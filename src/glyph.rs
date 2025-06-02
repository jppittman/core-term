// src/glyph.rs

//! Defines the `Glyph` type, its visual attributes (`AttrFlags`, `Attributes`),
//! and related constants.
//!
//! A `Glyph` represents a single character cell on the terminal screen,
//! encapsulating the character itself and all its styling information.
//! Color definitions (`Color`, `NamedColor`) are found in the `crate::color` module.

use crate::color::Color;
use bitflags::bitflags; // For creating flag enums like AttrFlags

/// Represents a glyph in a terminal grid cell.
///
/// A `Glyph` can be a single standard-width character, the primary (first cell)
/// of a double-width character, or the spacer (second cell) of a double-width character.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Glyph {
    /// A standard-width character, occupying a single cell.
    Single(ContentCell),
    /// The primary part of a double-width character, occupying the first cell.
    /// The character in `ContentCell` is the wide character itself.
    WidePrimary(ContentCell),
    /// The spacer part of a double-width character, occupying the second cell.
    /// This variant does not store its own character or attributes directly;
    /// it typically inherits appearance from its `WidePrimary` counterpart.
    /// `primary_column_on_line` stores the column index of the `WidePrimary`
    /// glyph this spacer belongs to, which can be useful for context.
    WideSpacer{
        // TODO: Consider if `primary_column_on_line` is truly needed here or if
        // the renderer can manage this context. For now, it's kept from previous design.
        primary_column_on_line: u16,
    },
}

/// Holds the actual character and its attributes for a `Glyph::Single` or `Glyph::WidePrimary`.
///
/// This struct contains the concrete visual information for a cell that displays content.
/// `Glyph::WideSpacer` does not directly contain a `ContentCell` as its appearance is
/// derived from its corresponding `Glyph::WidePrimary`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContentCell {
    /// The character to be displayed. For `Glyph::WidePrimary`, this is the wide character itself.
    pub c: char,
    /// The visual attributes of the character (foreground/background color, flags).
    pub attr: Attributes,
}

/// Placeholder character used by `Glyph::display_char()` for `WideSpacer` variants.
///
/// This helps in scenarios where a single char representation is needed for a cell,
/// indicating that it's part of a wide character but not the primary content-bearing part.
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

        // Flags related to wide character handling, primarily for `ContentCell` within `Glyph::WidePrimary`.
        // The `Glyph` enum variant (`Glyph::WidePrimary` or `Glyph::WideSpacer`) is the primary
        // determinant of a glyph's role in a wide character sequence.
        const WIDE_CHAR_PRIMARY = 1 << 14; // Indicates that the `ContentCell` (within `Glyph::WidePrimary`)
                                           // contains a character that occupies two cells.
        const WIDE_CHAR_SPACER  = 1 << 15; // Historically indicated the second cell of a wide character.
                                           // With the `Glyph::WideSpacer` enum variant, this flag on a
                                           // `ContentCell` is largely redundant or only for specific legacy contexts.
                                           // `Glyph::WideSpacer` itself signifies this role.

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
    /// Creates a default `ContentCell` representing a blank space with default attributes.
    pub fn default_space() -> Self {
        ContentCell {
            c: ' ', 
            attr: Attributes::default(),
        }
    }
}
