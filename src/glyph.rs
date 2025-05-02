// src/glyph.rs
// Defines the representation of a terminal cell (Glyph) and its attributes,
// including enums for color representation and color mapping logic.

use bitflags::bitflags;

bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
    pub struct AttrFlags: u16 {
        const BOLD       = 1 << 0;
        const UNDERLINE  = 1 << 1;
        const REVERSE    = 1 << 2;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Color { // Represents the 16 standard ANSI colors by name
    Black = 0, Red = 1, Green = 2, Yellow = 3, Blue = 4, Magenta = 5, Cyan = 6, White = 7,
    BrightBlack = 8, BrightRed = 9, BrightGreen = 10, BrightYellow = 11, BrightBlue = 12,
    BrightMagenta = 13, BrightCyan = 14, BrightWhite = 15,
}

impl Color {
    pub fn index(self) -> u8 { self as u8 }
    pub fn rgb(self) -> (u8, u8, u8) {
        const RGB_TABLE: [(u8, u8, u8); 16] = [
            (0, 0, 0), (170, 0, 0), (0, 170, 0), (170, 85, 0), (0, 0, 170), (170, 0, 170), (0, 170, 170), (170, 170, 170),
            (85, 85, 85), (255, 85, 85), (85, 255, 85), (255, 255, 85), (85, 85, 255), (255, 85, 255), (85, 255, 255), (255, 255, 255),
        ];
        RGB_TABLE[self.index() as usize]
    }
     pub fn from_index(index: u8) -> Option<Self> {
         if index <= 15 { Some(unsafe { std::mem::transmute(index) }) } else { None }
     }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ColorSpec {
    Default,
    Idx(u8),
    Rgb(u8, u8, u8),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Attributes {
    pub fg: ColorSpec,
    pub bg: ColorSpec,
    pub flags: AttrFlags,
}

impl Default for Attributes {
    fn default() -> Self {
        Attributes { fg: ColorSpec::Default, bg: ColorSpec::Default, flags: AttrFlags::default() }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Glyph {
    pub c: char,
    pub attr: Attributes,
}

impl Default for Glyph {
    fn default() -> Self {
        Glyph { c: ' ', attr: Attributes::default() }
    }
}

// --- Color Mapping Logic ---

// Constants for 256-color mapping (indices 16-255)
// See: https://en.wikipedia.org/wiki/ANSI_escape_code#8-bit

// 6x6x6 Color Cube
const CUBE_START_INDEX: u8 = 16;
const CUBE_END_INDEX: u8 = 231;
const CUBE_SIZE: u8 = 6;
const CUBE_LEVELS: [u8; CUBE_SIZE as usize] = [0, 95, 135, 175, 215, 255];

// Grayscale Ramp
const GRAYSCALE_START_INDEX: u8 = 232;
// ** FIX: Use u8::MAX for clarity **
const GRAYSCALE_END_INDEX: u8 = u8::MAX; // 255
const GRAYSCALE_STEP: u8 = 10;
const GRAYSCALE_BASE_LEVEL: u8 = 8;

/// Maps an 8-bit color index (0-255) to its approximate RGB value.
/// Returns None if the index is invalid (only possible if input type changes).
pub fn map_index_to_rgb(idx: u8) -> Option<(u8, u8, u8)> {
    match idx {
        // Standard and Bright ANSI colors (0-15)
        0..=15 => Color::from_index(idx).map(|c| c.rgb()),

        // 6x6x6 Color Cube (16-231)
        CUBE_START_INDEX..=CUBE_END_INDEX => {
            let relative_idx = idx - CUBE_START_INDEX;
            let r_idx = (relative_idx / (CUBE_SIZE * CUBE_SIZE)) % CUBE_SIZE;
            let g_idx = (relative_idx / CUBE_SIZE) % CUBE_SIZE;
            let b_idx = relative_idx % CUBE_SIZE;
            Some((
                CUBE_LEVELS[r_idx as usize],
                CUBE_LEVELS[g_idx as usize],
                CUBE_LEVELS[b_idx as usize],
            ))
        }

        // Grayscale Ramp (232-255)
        // ** FIX: Use constant for end index **
        GRAYSCALE_START_INDEX..=GRAYSCALE_END_INDEX => {
            let gray_idx = idx - GRAYSCALE_START_INDEX;
            // Calculate level, ensuring it doesn't exceed u8::MAX (though current math doesn't)
            let level = (GRAYSCALE_BASE_LEVEL as u16 + gray_idx as u16 * GRAYSCALE_STEP as u16).min(u8::MAX as u16) as u8;
            Some((level, level, level))
        }
        // Note: Match is exhaustive for u8
    }
}

#[cfg(test)]
mod tests {
    use super::*; // Import items from outer module

    #[test]
    fn test_color_index() {
        assert_eq!(Color::Black.index(), 0);
        assert_eq!(Color::Red.index(), 1);
        assert_eq!(Color::BrightWhite.index(), 15);
    }

    #[test]
    fn test_color_rgb() {
        assert_eq!(Color::Black.rgb(), (0, 0, 0));
        assert_eq!(Color::Red.rgb(), (170, 0, 0)); // Standard Red
        assert_eq!(Color::BrightRed.rgb(), (255, 85, 85));
        assert_eq!(Color::White.rgb(), (170, 170, 170));
        assert_eq!(Color::BrightWhite.rgb(), (255, 255, 255));
    }

    #[test]
    fn test_color_from_index() {
        assert_eq!(Color::from_index(0), Some(Color::Black));
        assert_eq!(Color::from_index(7), Some(Color::White));
        assert_eq!(Color::from_index(8), Some(Color::BrightBlack));
        assert_eq!(Color::from_index(15), Some(Color::BrightWhite));
        assert_eq!(Color::from_index(16), None); // Index out of range for NamedColor
    }

    #[test]
    fn test_map_index_to_rgb_standard() {
        assert_eq!(map_index_to_rgb(0), Some((0, 0, 0))); // Black
        assert_eq!(map_index_to_rgb(1), Some((170, 0, 0))); // Red
        assert_eq!(map_index_to_rgb(15), Some((255, 255, 255))); // Bright White
    }

    #[test]
    fn test_map_index_to_rgb_cube() {
        assert_eq!(map_index_to_rgb(16), Some((0, 0, 0)));
        assert_eq!(map_index_to_rgb(19), Some((0, 0, 175)));
        // ** FIX: Correct expected value for index 46 **
        assert_eq!(map_index_to_rgb(46), Some((0, 255, 0))); // Index 5 maps to 255
        assert_eq!(map_index_to_rgb(196), Some((255, 0, 0)));
        assert_eq!(map_index_to_rgb(231), Some((255, 255, 255)));
        assert_eq!(map_index_to_rgb(110), Some((135, 175, 215)));
    }


    #[test]
    fn test_map_index_to_rgb_grayscale() {
        assert_eq!(map_index_to_rgb(232), Some((8, 8, 8))); // First gray
        assert_eq!(map_index_to_rgb(244), Some((128, 128, 128))); // Mid gray (index 12 -> 8 + 12*10 = 128)
        assert_eq!(map_index_to_rgb(255), Some((238, 238, 238))); // Last gray (index 23 -> 8 + 23*10 = 238)
    }

    #[test]
    fn test_attributes_default() {
        let default_attr = Attributes::default();
        assert_eq!(default_attr.fg, ColorSpec::Default);
        assert_eq!(default_attr.bg, ColorSpec::Default);
        assert_eq!(default_attr.flags, AttrFlags::empty());
    }

    #[test]
    fn test_glyph_default() {
        let default_glyph = Glyph::default();
        assert_eq!(default_glyph.c, ' ');
        assert_eq!(default_glyph.attr, Attributes::default());
    }
}

