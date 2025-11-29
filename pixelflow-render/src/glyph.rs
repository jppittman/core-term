//! Embedded font access for pixelflow-render.
//!
//! The pixelflow way: get the font, call `font.glyph(char, size)` to get a `Surface<u8>`,
//! then compose it however you need using standard Surface combinators.

use pixelflow_fonts::Font;
use std::sync::OnceLock;

// Embedded font data (Noto Sans Mono)
static FONT_DATA: &[u8] = include_bytes!("../assets/NotoSansMono-Regular.ttf");

static FONT: OnceLock<Font<'static>> = OnceLock::new();

/// Get the embedded monospace font.
///
/// # Example
/// ```ignore
/// use pixelflow_render::font;
/// use pixelflow_core::dsl::MaskExt;
///
/// let f = font();
/// let glyph = f.glyph('A', 24.0).unwrap();  // Surface<u8>
/// let blended = glyph.over(fg_color, bg_surface);
/// execute(blended, &mut target);
/// ```
pub fn font() -> &'static Font<'static> {
    FONT.get_or_init(|| Font::from_bytes(FONT_DATA).expect("Failed to parse embedded font"))
}
