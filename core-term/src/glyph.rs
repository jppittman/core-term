// src/glyph.rs

//! Defines the `Glyph` type, its visual attributes (`AttrFlags`, `Attributes`),
//! and related constants.
//!
//! A `Glyph` represents a single character cell on the terminal screen,
//! encapsulating the character itself and all its styling information.
//! Color definitions (`Color`, `NamedColor`) are found in the `crate::color` module.

use crate::color::Color;

// Re-export AttrFlags from pixelflow-render
pub use pixelflow_graphics::render::AttrFlags;

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
    /// it inherits appearance from its `WidePrimary` counterpart in the preceding cell.
    WideSpacer,
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
    #[must_use]
    pub fn default_cell() -> Self {
        Glyph::Single(ContentCell::default_space())
    }

    /// Helper to get the displayable character of the cell, if applicable.
    /// For spacers, this might be the placeholder.
    #[must_use]
    pub fn display_char(&self) -> char {
        match self {
            Glyph::Single(cc) | Glyph::WidePrimary(cc) => cc.c,
            Glyph::WideSpacer => WIDE_CHAR_PLACEHOLDER,
        }
    }
}

impl ContentCell {
    /// Creates a default `ContentCell` representing a blank space with default attributes.
    #[must_use]
    pub fn default_space() -> Self {
        ContentCell {
            c: ' ',
            attr: Attributes::default(),
        }
    }
}
