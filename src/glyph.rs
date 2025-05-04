// src/glyph.rs

use bitflags::bitflags;

/// Unicode replacement character (U+FFFD).
/// Used when encountering invalid UTF-8 sequences.
pub const REPLACEMENT_CHARACTER: char = '\u{FFFD}';

bitflags! {
    /// Represents text attributes like bold, underline, etc.
    /// Matches definitions in st.h for compatibility where possible.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
    pub struct AttrFlags: u16 {
        /// Bold text (SGR 1). Equivalent to ATTR_BOLD in st.h.
        const BOLD          = 1 << 0;
        /// Faint text (SGR 2). Often displayed as less intense color. Equivalent to ATTR_FAINT in st.h.
        const FAINT         = 1 << 1;
        /// Italicized text (SGR 3). Equivalent to ATTR_ITALIC in st.h.
        const ITALIC        = 1 << 2;
        /// Underlined text (SGR 4). Equivalent to ATTR_UNDERLINE in st.h.
        const UNDERLINE     = 1 << 3;
        /// Blinking text (SGR 5 & 6). Equivalent to ATTR_BLINK in st.h.
        const BLINK         = 1 << 4;
        /// Reversed foreground/background colors (SGR 7). Equivalent to ATTR_REVERSE in st.h.
        const REVERSE       = 1 << 5;
        /// Hidden text (concealed) (SGR 8). Equivalent to ATTR_INVISIBLE in st.h.
        const HIDDEN        = 1 << 6;
        /// Text with a line through it (SGR 9). Equivalent to ATTR_STRUCK in st.h.
        const STRIKETHROUGH = 1 << 7;
        /// Character occupies two columns (internal flag). Equivalent to ATTR_WIDE in st.h.
        const WIDE          = 1 << 9; // Skip bit 8 (ATTR_WRAP in st.h, handled differently)
        /// Placeholder for the second column of a wide character (internal flag). Equivalent to ATTR_WDUMMY in st.h.
        const WIDE_DUMMY    = 1 << 10;
        // Note: ATTR_WRAP (1 << 8 in st.h) is handled implicitly by line wrapping logic, not as a persistent flag.
        // Note: ATTR_BOLD_FAINT is a combination, not a separate flag.
    }
}

/// Represents foreground or background color options.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Color {
    /// Use the terminal's default foreground/background color.
    Default,
    /// Indexed color from a palette (typically 0-255).
    Idx(u8),
    /// Truecolor (24-bit RGB).
    Rgb(u8, u8, u8),
}

impl Default for Color {
    /// The default color is `Color::Default`.
    fn default() -> Self {
        Color::Default
    }
}

/// Holds the display attributes for a single character cell.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Attributes {
    pub fg: Color,
    pub bg: Color,
    pub flags: AttrFlags,
}

/// Represents a single character cell on the terminal screen.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Glyph {
    pub c: char,
    pub attr: Attributes,
}

impl Default for Glyph {
    /// The default glyph is a space with default attributes.
    fn default() -> Self {
        Glyph {
            c: ' ',
            attr: Attributes::default(),
        }
    }
}


// --- Unit Tests ---
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_glyph_default() {
        let glyph = Glyph::default();
        assert_eq!(glyph.c, ' ');
        assert_eq!(glyph.attr, Attributes::default());
    }

    #[test]
    fn test_attributes_default() {
        let attrs = Attributes::default();
        assert_eq!(attrs.fg, Color::Default);
        assert_eq!(attrs.bg, Color::Default);
        assert_eq!(attrs.flags, AttrFlags::empty());
    }

    #[test]
    fn test_color_default() {
        let color = Color::default();
        assert_eq!(color, Color::Default);
    }

     #[test]
    fn test_color_variants() {
        assert_eq!(Color::Idx(10), Color::Idx(10));
        assert_eq!(Color::Rgb(10,20,30), Color::Rgb(10,20,30));
        assert_ne!(Color::Idx(1), Color::Default);
        assert_ne!(Color::Rgb(0,0,0), Color::Idx(0));
    }

    #[test]
    fn test_attr_flags() {
        let mut flags = AttrFlags::default();
        assert!(!flags.contains(AttrFlags::BOLD));
        flags |= AttrFlags::BOLD;
        assert!(flags.contains(AttrFlags::BOLD));
        flags |= AttrFlags::ITALIC;
        assert!(flags.contains(AttrFlags::BOLD | AttrFlags::ITALIC));
        flags &= !AttrFlags::BOLD;
        assert!(!flags.contains(AttrFlags::BOLD));
        assert!(flags.contains(AttrFlags::ITALIC));
    }
}
