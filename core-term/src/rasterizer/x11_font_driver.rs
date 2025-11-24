#![cfg(use_x11_display)]

//! X11 font driver using fontconfig and freetype
//!
//! This driver provides glyph rasterization for X11 display using:
//! - fontconfig for font discovery and matching
//! - FreeType for font loading and rendering

use super::font_driver::{FontDriver, FontId};
use anyhow::{anyhow, Context, Result};
use log::{debug, trace, warn};
use std::cell::RefCell;
use std::collections::HashMap;
use std::ptr;
use std::rc::Rc;

// Import fontconfig FFI bindings (yeslogic-fontconfig-sys provides fontconfig_sys module)
use fontconfig_sys::constants::{FC_CHARSET, FC_FILE, FC_SCALABLE};
use fontconfig_sys::{
    FcBool, FcCharSetAddChar, FcCharSetCreate, FcCharSetDestroy, FcConfigSubstitute,
    FcDefaultSubstitute, FcMatchPattern, FcPattern, FcPatternAddBool, FcPatternAddCharSet,
    FcPatternCreate, FcPatternDestroy, FcPatternGetString, FcResult, FcResultMatch,
};

// FreeType bindings
use freetype::freetype::{
    FT_Done_Face, FT_Done_FreeType, FT_Get_Char_Index, FT_Init_FreeType, FT_Library, FT_Load_Glyph,
    FT_New_Face, FT_Set_Char_Size, FT_Face, FT_LOAD_RENDER,
};

// Fontconfig externals
extern "C" {
    fn FcFontMatch(
        config: *mut std::os::raw::c_void,
        pattern: *mut FcPattern,
        result: *mut FcResult,
    ) -> *mut FcPattern;
}

/// X11 font handle wrapping a FreeType face.
/// Must be Clone for FontDriver trait requirement.
#[derive(Clone)]
pub struct X11Font {
    // Use Rc to make the face shareable across clones
    face: Rc<FT_Face>,
    name: String,
    size_pt: f64,
}

// SAFETY: FreeType faces are thread-safe for read-only operations (glyph loading/rendering).
// We never mutate the face after creation, only read from it.
unsafe impl Send for X11Font {}
unsafe impl Sync for X11Font {}

impl Drop for X11Font {
    fn drop(&mut self) {
        // Only close the font when the last reference is dropped
        if Rc::strong_count(&self.face) == 1 {
            unsafe {
                FT_Done_Face(*self.face);
            }
        }
    }
}

/// X11 font driver using FreeType for rendering.
pub struct X11FontDriver {
    // FreeType library instance (owned by driver)
    library: FT_Library,
    cell_width_px: usize,
    cell_height_px: usize,
    /// Internal font cache mapping FontId to X11Font
    fonts: RefCell<HashMap<FontId, X11Font>>,
    /// Next font ID to assign
    next_id: RefCell<FontId>,
}

impl X11FontDriver {
    /// Create a new X11 font driver.
    pub fn new(cell_width_px: usize, cell_height_px: usize) -> Self {
        let mut library: FT_Library = ptr::null_mut();
        unsafe {
            let error = FT_Init_FreeType(&mut library);
            if error != 0 {
                panic!("Failed to initialize FreeType library: error {}", error);
            }
        }
        debug!(
            "X11FontDriver: Initialized FreeType library for {}x{} cells",
            cell_width_px, cell_height_px
        );
        Self {
            library,
            cell_width_px,
            cell_height_px,
            fonts: RefCell::new(HashMap::new()),
            next_id: RefCell::new(0),
        }
    }

    /// Use fontconfig to resolve a font name to a file path.
    fn resolve_font_path(&self, name: &str) -> Result<String> {
        unsafe {
            // Create fontconfig pattern
            let pattern = FcPatternCreate();
            if pattern.is_null() {
                return Err(anyhow!("FcPatternCreate failed for font '{}'", name));
            }

            // Request scalable fonts only
            FcPatternAddBool(pattern, FC_SCALABLE.as_ptr(), 1 as FcBool);

            // Substitute defaults
            FcConfigSubstitute(ptr::null_mut(), pattern, FcMatchPattern);
            FcDefaultSubstitute(pattern);

            // Match font
            let mut result: FcResult = 0;
            let matched = FcFontMatch(ptr::null_mut(), pattern, &mut result);

            // Clean up original pattern
            FcPatternDestroy(pattern);

            if matched.is_null() {
                return Err(anyhow!(
                    "FcFontMatch failed to find font '{}': result={}",
                    name,
                    result
                ));
            }

            // Extract file path from matched pattern
            let mut file_path: *mut u8 = ptr::null_mut();
            let fc_result = FcPatternGetString(matched, FC_FILE.as_ptr(), 0, &mut file_path);

            if fc_result != FcResultMatch || file_path.is_null() {
                FcPatternDestroy(matched);
                return Err(anyhow!(
                    "Failed to get file path for font '{}': result={:?}",
                    name,
                    fc_result
                ));
            }

            // Convert C string to Rust string
            let path_cstr = std::ffi::CStr::from_ptr(file_path as *const i8);
            let path = path_cstr
                .to_str()
                .context("Font path is not valid UTF-8")?
                .to_string();

            FcPatternDestroy(matched);

            debug!("Fontconfig resolved '{}' to '{}'", name, path);
            Ok(path)
        }
    }

    /// Use fontconfig to find a font that supports a specific character.
    fn find_font_for_char(&self, ch: char) -> Result<String> {
        unsafe {
            let pattern = FcPatternCreate();
            if pattern.is_null() {
                return Err(anyhow!("FcPatternCreate failed for char '{}'", ch));
            }

            // Request scalable fonts
            FcPatternAddBool(pattern, FC_SCALABLE.as_ptr(), 1 as FcBool);

            // Create charset containing just this character
            let charset = FcCharSetCreate();
            if charset.is_null() {
                FcPatternDestroy(pattern);
                return Err(anyhow!("FcCharSetCreate failed for char '{}'", ch));
            }

            FcCharSetAddChar(charset, ch as u32);
            FcPatternAddCharSet(pattern, FC_CHARSET.as_ptr(), charset);

            // Substitute defaults
            FcConfigSubstitute(ptr::null_mut(), pattern, FcMatchPattern);
            FcDefaultSubstitute(pattern);

            // Match font
            let mut result: FcResult = 0;
            let matched = FcFontMatch(ptr::null_mut(), pattern, &mut result);

            // Clean up
            FcPatternDestroy(pattern);
            FcCharSetDestroy(charset);

            if matched.is_null() {
                return Err(anyhow!(
                    "FcFontMatch found no fallback font for char U+{:X}",
                    ch as u32
                ));
            }

            // Extract file path
            let mut file_path: *mut u8 = ptr::null_mut();
            let fc_result = FcPatternGetString(matched, FC_FILE.as_ptr(), 0, &mut file_path);

            if fc_result != FcResultMatch || file_path.is_null() {
                FcPatternDestroy(matched);
                return Err(anyhow!(
                    "Failed to get file path for fallback font for char U+{:X}",
                    ch as u32
                ));
            }

            let path_cstr = std::ffi::CStr::from_ptr(file_path as *const i8);
            let path = path_cstr
                .to_str()
                .context("Fallback font path is not valid UTF-8")?
                .to_string();

            FcPatternDestroy(matched);

            debug!("Fontconfig found fallback '{}' for char U+{:X}", path, ch as u32);
            Ok(path)
        }
    }
}

impl Drop for X11FontDriver {
    fn drop(&mut self) {
        unsafe {
            FT_Done_FreeType(self.library);
        }
    }
}

impl FontDriver for X11FontDriver {
    fn load_font(&self, name: &str, size_pt: f64) -> Result<FontId> {
        // Resolve font name to file path using fontconfig
        let font_path = self
            .resolve_font_path(name)
            .with_context(|| format!("Failed to resolve font '{}'", name))?;

        // Load font with FreeType
        let mut face: FT_Face = ptr::null_mut();
        let path_cstr = std::ffi::CString::new(font_path.as_str())?;

        unsafe {
            let error = FT_New_Face(self.library, path_cstr.as_ptr(), 0, &mut face);
            if error != 0 || face.is_null() {
                return Err(anyhow!("Failed to load font from '{}': FT error {}", font_path, error));
            }

            // Set character size - use 72 DPI
            // FreeType uses 1/64th of a point, so multiply by 64
            // Use slightly smaller than cell height to avoid overlap (leave room for line spacing)
            let target_size = (self.cell_height_px as f64 * 0.85).round();
            let error = FT_Set_Char_Size(face, (target_size * 64.0) as i64, 0, 72, 72);
            if error != 0 {
                FT_Done_Face(face);
                return Err(anyhow!("Failed to set char size for font '{}': FT error {}", name, error));
            }
        }

        debug!(
            "Loaded font '{}' from '{}' at {}pt",
            name, font_path, size_pt
        );

        let font = X11Font {
            face: Rc::new(face),
            name: name.to_string(),
            size_pt,
        };

        let id = *self.next_id.borrow();
        *self.next_id.borrow_mut() += 1;
        self.fonts.borrow_mut().insert(id, font);

        Ok(id)
    }

    fn find_glyph(&self, font_id: FontId, ch: char) -> Option<u32> {
        let fonts = self.fonts.borrow();
        let font = fonts.get(&font_id)?;

        unsafe {
            let glyph_index = FT_Get_Char_Index(*font.face, ch as u64);
            if glyph_index != 0 {
                trace!("Found glyph {} for char '{}' in font '{}'", glyph_index, ch, font.name);
                Some(glyph_index)
            } else {
                trace!("No glyph for char '{}' in font '{}'", ch, font.name);
                None
            }
        }
    }

    fn find_fallback_font(&self, ch: char) -> Result<FontId> {
        // Use fontconfig to find a font that contains this character
        let font_path = self
            .find_font_for_char(ch)
            .with_context(|| format!("Failed to find fallback font for char U+{:X}", ch as u32))?;

        // Load the font
        let mut face: FT_Face = ptr::null_mut();
        let path_cstr = std::ffi::CString::new(font_path.as_str())?;

        unsafe {
            let error = FT_New_Face(self.library, path_cstr.as_ptr(), 0, &mut face);
            if error != 0 || face.is_null() {
                return Err(anyhow!("Failed to load fallback font from '{}': FT error {}", font_path, error));
            }

            // Set character size - same as primary fonts
            let target_size = (self.cell_height_px as f64 * 0.85).round();
            let error = FT_Set_Char_Size(face, (target_size * 64.0) as i64, 0, 72, 72);
            if error != 0 {
                FT_Done_Face(face);
                return Err(anyhow!("Failed to set char size for fallback font: FT error {}", error));
            }
        }

        debug!("Loaded fallback font '{}' for char U+{:X}", font_path, ch as u32);

        let font = X11Font {
            face: Rc::new(face),
            name: font_path,
            size_pt: 12.0,
        };

        let id = *self.next_id.borrow();
        *self.next_id.borrow_mut() += 1;
        self.fonts.borrow_mut().insert(id, font);

        Ok(id)
    }

    fn rasterize_glyph(
        &self,
        font_id: FontId,
        glyph_id: u32,
        cell_width_px: usize,
        cell_height_px: usize,
    ) -> Vec<u8> {
        let fonts = self.fonts.borrow();
        let font = match fonts.get(&font_id) {
            Some(f) => f,
            None => return vec![0u8; cell_width_px * cell_height_px * 4],
        };

        // Create output buffer (RGBA)
        let size = cell_width_px * cell_height_px * 4;
        let mut data = vec![0u8; size];

        unsafe {
            // Load and render the glyph
            let error = FT_Load_Glyph(*font.face, glyph_id, FT_LOAD_RENDER as i32);
            if error != 0 {
                warn!("Failed to load glyph {}: FT error {}", glyph_id, error);
                return data; // Return transparent buffer
            }

            let face = *font.face;
            let glyph_slot = (*face).glyph;
            let bitmap = &(*glyph_slot).bitmap;

            // Check if we have bitmap data
            if bitmap.width == 0 || bitmap.rows == 0 {
                trace!("Glyph {} has no bitmap data (likely whitespace)", glyph_id);
                return data; // Return transparent buffer for whitespace
            }

            // Get bitmap buffer (grayscale, 1 byte per pixel)
            let bitmap_buffer = std::slice::from_raw_parts(
                bitmap.buffer,
                (bitmap.rows * bitmap.pitch.abs() as u32) as usize,
            );
            let bitmap_width = bitmap.width as usize;
            let bitmap_rows = bitmap.rows as usize;
            let bitmap_pitch = bitmap.pitch as usize;

            // Get glyph metrics for positioning
            let bearing_x = (*glyph_slot).bitmap_left;
            let bearing_y = (*glyph_slot).bitmap_top;

            // Calculate baseline position (centered vertically in cell)
            let metrics = (*(*face).size).metrics;
            let font_height = (metrics.height / 64) as i32;
            let baseline_y = (cell_height_px as i32 - font_height) / 2 + font_height - bearing_y;

            // Calculate horizontal offset (center glyph in cell)
            let glyph_x = ((cell_width_px as i32 - bitmap_width as i32) / 2).max(0) as usize;

            trace!(
                "Rasterizing glyph {}: bitmap={}x{}, bearing=({},{}), baseline_y={}, glyph_x={}",
                glyph_id, bitmap_width, bitmap_rows, bearing_x, bearing_y, baseline_y, glyph_x
            );

            // Blit bitmap into output buffer
            for src_y in 0..bitmap_rows {
                let dst_y = baseline_y as isize - bearing_y as isize + src_y as isize;

                // Skip if outside cell bounds
                if dst_y < 0 || dst_y >= cell_height_px as isize {
                    continue;
                }

                for src_x in 0..bitmap_width {
                    let dst_x = glyph_x + src_x;

                    // Skip if outside cell bounds
                    if dst_x >= cell_width_px {
                        continue;
                    }

                    // Get grayscale alpha value from bitmap
                    let src_offset = src_y * bitmap_pitch + src_x;
                    let alpha = bitmap_buffer[src_offset];

                    // Skip fully transparent pixels
                    if alpha == 0 {
                        continue;
                    }

                    // Write white pixel with alpha
                    let dst_offset = (dst_y as usize * cell_width_px + dst_x) * 4;
                    data[dst_offset] = 255;     // R
                    data[dst_offset + 1] = 255; // G
                    data[dst_offset + 2] = 255; // B
                    data[dst_offset + 3] = alpha; // A (straight alpha from FreeType)
                }
            }
        }

        data
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_font_success() {
        let driver = X11FontDriver::new(8, 16);
        let result = driver.load_font("monospace", 12.0);
        assert!(result.is_ok(), "Should successfully load monospace font");
    }

    #[test]
    fn test_find_glyph_returns_some_for_ascii() {
        let driver = X11FontDriver::new(8, 16);
        let font = driver.load_font("monospace", 12.0).unwrap();

        assert!(driver.find_glyph(&font, 'A').is_some());
        assert!(driver.find_glyph(&font, 'a').is_some());
        assert!(driver.find_glyph(&font, '0').is_some());
        assert!(driver.find_glyph(&font, ' ').is_some());
    }

    #[test]
    fn test_rasterize_glyph_returns_correct_buffer_size() {
        let driver = X11FontDriver::new(8, 16);
        let font = driver.load_font("monospace", 12.0).unwrap();
        let glyph_id = driver.find_glyph(&font, 'A').unwrap();

        let width = 8;
        let height = 16;
        let pixels = driver.rasterize_glyph(&font, glyph_id, width, height);

        assert_eq!(pixels.len(), width * height * 4);
    }

    #[test]
    fn test_rasterize_glyph_produces_non_empty_output() {
        let driver = X11FontDriver::new(8, 16);
        let font = driver.load_font("monospace", 12.0).unwrap();
        let glyph_id = driver.find_glyph(&font, 'A').unwrap();

        let pixels = driver.rasterize_glyph(&font, glyph_id, 8, 16);
        let has_content = pixels.chunks_exact(4).any(|p| p[3] > 0);

        assert!(
            has_content,
            "Rasterized glyph should have pixels with non-zero alpha"
        );
    }

    #[test]
    fn test_find_fallback_font_for_emoji() {
        let driver = X11FontDriver::new(8, 16);
        let result = driver.find_fallback_font('😀');

        assert!(result.is_ok(), "Should find fallback font for emoji");
    }
}
