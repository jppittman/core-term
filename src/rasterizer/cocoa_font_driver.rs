//! Core Text font driver implementation for macOS.

use super::font_driver::FontDriver;
use anyhow::{anyhow, Result};
use core_foundation::base::TCFType;
use core_foundation::string::CFString;
use core_graphics::base::CGFloat;
use core_graphics::color_space::CGColorSpace;
use core_graphics::context::CGContext;
use core_graphics::geometry::{CGPoint, CGRect, CGSize};
use core_text::font::CTFont;
use core_text::font_descriptor::{kCTFontDefaultOrientation, CTFontDescriptor};
use std::os::raw::c_void;
use std::ptr;

// Default fallback font and size for system font queries
const FALLBACK_FONT_NAME: &str = "Menlo";
const FALLBACK_FONT_SIZE_PT: f64 = 12.0;

// RGBA pixel format
const BYTES_PER_PIXEL: usize = 4;

// FFI declarations for Core Text functions not exposed in the crate
#[link(name = "CoreText", kind = "framework")]
extern "C" {
    fn CTFontGetGlyphsForCharacters(
        font: *const c_void,
        characters: *const u16,
        glyphs: *mut u16,
        count: isize,
    ) -> bool;

    fn CTFontCreateForString(
        font: *const c_void,
        string: *const c_void,
        range: core_foundation::base::CFRange,
    ) -> *const c_void;

    fn CTFontDrawGlyphs(
        font: *const c_void,
        glyphs: *const u16,
        positions: *const CGPoint,
        count: usize,
        context: *mut c_void,
    );
}

/// Core Text font driver for macOS.
///
/// Provides thin wrappers around Core Text APIs for:
/// - Loading fonts by name
/// - Finding glyphs for characters
/// - Querying system for fallback fonts
/// - Rasterizing glyphs to RGBA pixels
pub struct CocoaFontDriver;

impl CocoaFontDriver {
    /// Create a new Cocoa font driver
    pub fn new() -> Self {
        Self
    }
}

impl FontDriver for CocoaFontDriver {
    type Font = CTFont;
    type GlyphId = u16; // CGGlyph is u16

    fn load_font(&self, name: &str, size_pt: f64) -> Result<CTFont> {
        let font = core_text::font::new_from_name(name, size_pt)
            .map_err(|_| anyhow!("Failed to load font '{}'", name))?;
        Ok(font)
    }

    fn find_glyph(&self, font: &CTFont, ch: char) -> Option<u16> {
        // Encode character to UTF-16 (Core Text uses UTF-16)
        let mut chars_utf16 = [0u16; 2];
        let encoded_len = ch.encode_utf16(&mut chars_utf16).len();
        let chars = &chars_utf16[..encoded_len];

        let mut glyphs = vec![0u16; encoded_len];

        let found = unsafe {
            CTFontGetGlyphsForCharacters(
                font.as_concrete_TypeRef() as *const c_void,
                chars.as_ptr(),
                glyphs.as_mut_ptr(),
                encoded_len as isize,
            )
        };

        if found && glyphs[0] != 0 {
            Some(glyphs[0])
        } else {
            None
        }
    }

    fn find_fallback_font(&self, ch: char) -> Result<CTFont> {
        // Use CTFontCreateForString to find a font that supports this character
        let base_font = core_text::font::new_from_name(FALLBACK_FONT_NAME, FALLBACK_FONT_SIZE_PT)
            .map_err(|_| anyhow!("Failed to create base font for fallback"))?;

        // Create a string containing just this character
        let cf_string = CFString::new(&ch.to_string());

        // Query for a font that can display this string
        let fallback = unsafe {
            let fallback_ref = CTFontCreateForString(
                base_font.as_concrete_TypeRef() as *const c_void,
                cf_string.as_concrete_TypeRef() as *const c_void,
                core_foundation::base::CFRange::init(0, 1),
            );

            if fallback_ref.is_null() {
                return Err(anyhow!(
                    "Core Text could not find fallback font for character U+{:X}",
                    ch as u32
                ));
            }

            CTFont::wrap_under_create_rule(fallback_ref as *mut _)
        };

        Ok(fallback)
    }

    fn rasterize_glyph(
        &self,
        font: &CTFont,
        glyph_id: u16,
        cell_width_px: usize,
        cell_height_px: usize,
    ) -> Vec<u8> {
        // Create RGBA bitmap context
        let width = cell_width_px;
        let height = cell_height_px;
        let mut pixels = vec![0u8; width * height * 4];

        let color_space = CGColorSpace::create_device_rgb();
        // Use premultiplied alpha (required by CGContext)
        // Note: colorize_glyph will un-premultiply to get correct RGB values
        let bitmap_info = core_graphics::base::kCGImageAlphaPremultipliedLast
            | core_graphics::base::kCGBitmapByteOrder32Big;

        let mut context = CGContext::create_bitmap_context(
            Some(pixels.as_mut_ptr() as *mut _),
            width,
            height,
            8,                   // bits per component
            width * 4,           // bytes per row
            &color_space,
            bitmap_info,
        );

        // Clear to transparent
        context.set_rgb_fill_color(0.0, 0.0, 0.0, 0.0);
        context.fill_rect(CGRect::new(
            &CGPoint::new(0.0, 0.0),
            &CGSize::new(width as CGFloat, height as CGFloat),
        ));

        // NOTE: We used to flip the coordinate system here for the old rendering path.
        // With CALayer rendering, we DON'T flip here because the layer handles coordinates.
        // The bitmap is stored top-to-bottom, glyphs render correctly without transformation.

        // Set text color to white
        context.set_rgb_fill_color(1.0, 1.0, 1.0, 1.0);

        // Get font metrics to position glyph properly
        let ascent = font.ascent();
        let descent = font.descent();
        let font_height = ascent + descent;

        // Calculate baseline position (centered vertically)
        let baseline_y = ((height as f64) - font_height) / 2.0 + descent;

        // Get glyph bounding box to center horizontally
        let glyphs = [glyph_id];
        let bbox = font.get_bounding_rects_for_glyphs(kCTFontDefaultOrientation, &glyphs);
        let glyph_width = bbox.size.width;

        // Center glyph horizontally
        let x_offset = ((width as f64) - glyph_width) / 2.0;

        // Draw the glyph
        let position = CGPoint::new(x_offset, baseline_y);
        unsafe {
            use core_foundation::base::TCFType;

            // The CGContext struct is a wrapper around the raw CGContextRef pointer.
            // To get the raw pointer, we take a pointer to our wrapper, cast it
            // to a pointer-to-a-pointer, and then dereference it.
            let context_ptr = &context as *const _ as *const *mut std::os::raw::c_void;
            let context_ref = *context_ptr;

            CTFontDrawGlyphs(
                font.as_concrete_TypeRef() as *const c_void,
                glyphs.as_ptr(),
                &position as *const CGPoint,
                1,
                context_ref,
            );
        }

        // CRITICAL: Flush the context to ensure drawing is committed to pixel buffer
        context.flush();

        // Log pixels from different positions to see if glyph is rendered elsewhere
        if pixels.len() >= 256 {
            // Sample: top-left, center, middle-right
            let center_offset = (height / 2) * width * 4 + (width / 2) * 4;
            log::debug!(
                "cocoa_font_driver: Raw glyph (premult) - top-left: [{},{},{},{}] center: [{},{},{},{}] has_any_content: {}",
                pixels[0], pixels[1], pixels[2], pixels[3],
                pixels[center_offset], pixels[center_offset+1], pixels[center_offset+2], pixels[center_offset+3],
                pixels.chunks_exact(4).any(|p| p[3] > 0)
            );
        }

        CocoaFontDriver::unpremultiply_alpha(&mut pixels);

        // Log a few pixels AFTER unpremultiply
        if pixels.len() >= 16 {
            log::debug!(
                "cocoa_font_driver: After unpremultiply - first 4 pixels (RGBA): [{},{},{},{}] [{},{},{},{}] [{},{},{},{}] [{},{},{},{}]",
                pixels[0], pixels[1], pixels[2], pixels[3],
                pixels[4], pixels[5], pixels[6], pixels[7],
                pixels[8], pixels[9], pixels[10], pixels[11],
                pixels[12], pixels[13], pixels[14], pixels[15]
            );
        }

        pixels
    }
}

impl CocoaFontDriver {
    /// Convert premultiplied alpha to straight alpha.
    ///
    /// CGContext outputs premultiplied alpha (RGB values already multiplied by A).
    /// This function un-premultiplies to get the original RGB values.
    fn unpremultiply_alpha(pixels: &mut [u8]) {
        for pixel in pixels.chunks_exact_mut(4) {
            let alpha = pixel[3] as f32 / 255.0;

            // Skip fully transparent pixels (avoid division by zero)
            if pixel[3] == 0 {
                continue;
            }

            // Un-premultiply: original_rgb = premult_rgb / alpha
            if alpha > 0.0 {
                pixel[0] = ((pixel[0] as f32 / alpha).min(255.0)) as u8;
                pixel[1] = ((pixel[1] as f32 / alpha).min(255.0)) as u8;
                pixel[2] = ((pixel[2] as f32 / alpha).min(255.0)) as u8;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_font_success() {
        let driver = CocoaFontDriver::new();
        let result = driver.load_font("Menlo", 12.0);
        assert!(result.is_ok(), "Should successfully load Menlo font");
    }

    #[test]
    fn test_find_glyph_returns_some_for_ascii() {
        let driver = CocoaFontDriver::new();
        let font = driver.load_font("Menlo", 12.0).unwrap();

        assert!(driver.find_glyph(&font, 'A').is_some());
        assert!(driver.find_glyph(&font, 'a').is_some());
        assert!(driver.find_glyph(&font, '0').is_some());
        assert!(driver.find_glyph(&font, ' ').is_some());
    }

    #[test]
    fn test_rasterize_glyph_returns_correct_buffer_size() {
        let driver = CocoaFontDriver::new();
        let font = driver.load_font("Menlo", 12.0).unwrap();
        let glyph_id = driver.find_glyph(&font, 'A').unwrap();

        let width = 8;
        let height = 16;
        let pixels = driver.rasterize_glyph(&font, glyph_id, width, height);

        assert_eq!(pixels.len(), width * height * 4);
    }

    #[test]
    fn test_rasterize_glyph_produces_non_empty_output() {
        let driver = CocoaFontDriver::new();
        let font = driver.load_font("Menlo", 12.0).unwrap();
        let glyph_id = driver.find_glyph(&font, 'A').unwrap();

        let pixels = driver.rasterize_glyph(&font, glyph_id, 8, 16);
        let has_content = pixels.chunks_exact(4).any(|p| p[3] > 0);

        assert!(has_content, "Rasterized glyph should have pixels with non-zero alpha");
    }

    #[test]
    fn test_rasterize_glyph_produces_white_glyph() {
        let driver = CocoaFontDriver::new();
        let font = driver.load_font("Menlo", 12.0).unwrap();
        let glyph_id = driver.find_glyph(&font, 'A').unwrap();

        let pixels = driver.rasterize_glyph(&font, glyph_id, 8, 16);

        let has_white_pixels = pixels.chunks_exact(4)
            .filter(|p| p[3] > 0)
            .any(|p| p[0] >= 250 && p[1] >= 250 && p[2] >= 250);

        assert!(has_white_pixels, "Rasterized glyph should produce white pixels as per contract");
    }

    #[test]
    fn test_find_fallback_font_for_emoji() {
        let driver = CocoaFontDriver::new();
        let result = driver.find_fallback_font('😀');

        assert!(result.is_ok(), "Should find fallback font for emoji");
    }

    #[test]
    fn test_rasterize_glyph_actually_draws_pixels() {
        // This test validates our hypothesis that the coordinate system fix
        // (translate + scale) and context flush make Core Text actually render glyphs.
        //
        // The bug was: rasterized glyphs were all zeros [0,0,0,0] (fully transparent),
        // causing the screen to stay black even though we were "rendering" text.
        //
        // This test ensures that after rendering a simple ASCII character,
        // the pixel buffer contains NON-ZERO values, proving Core Text drew something.

        let driver = CocoaFontDriver::new();
        let font = driver.load_font("Menlo", 12.0).unwrap();
        let glyph_id = driver.find_glyph(&font, 'A').unwrap();

        let width = 8;
        let height = 16;
        let pixels = driver.rasterize_glyph(&font, glyph_id, width, height);

        // Count non-transparent pixels
        let non_transparent_pixels = pixels.chunks_exact(4)
            .filter(|p| p[3] > 0) // alpha > 0
            .count();

        // For the letter 'A' at 8x16px, we expect MANY pixels to be non-transparent
        // (the glyph should occupy a significant portion of the cell)
        assert!(
            non_transparent_pixels > 10,
            "Expected at least 10 non-transparent pixels for 'A', but got {}. \
             This means Core Text is not actually drawing glyphs! \
             Pixel data sample (first 16 bytes): {:?}",
            non_transparent_pixels,
            &pixels[..16.min(pixels.len())]
        );

        // Also verify that we have some WHITE pixels (not just alpha)
        // Since we render with white color (1.0, 1.0, 1.0), we expect R, G, B values > 200
        let white_pixels = pixels.chunks_exact(4)
            .filter(|p| p[3] > 0 && p[0] > 200 && p[1] > 200 && p[2] > 200)
            .count();

        assert!(
            white_pixels > 5,
            "Expected at least 5 white pixels for 'A', but got {}. \
             This means glyphs are being drawn but not with the correct color! \
             Pixel data sample (first 16 bytes): {:?}",
            white_pixels,
            &pixels[..16.min(pixels.len())]
        );
    }
}
