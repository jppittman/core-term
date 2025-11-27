//! Glyph rendering: The Builder.
//!
//! This module constructs zero-copy render pipelines by composing
//! functional operators from `pixelflow-core`.

#[cfg(feature = "fonts")]
include!(concat!(env!("OUT_DIR"), "/baked_font.rs"));

use pixelflow_core::dsl::{MaskExt, SurfaceExt};
use pixelflow_core::ops::{Max, SampleAtlas};
use pixelflow_core::{execute, Batch, TensorView, TensorViewMut};

// ============================================================================
// 1. Clean API Structs (Grouped Arguments)
// ============================================================================

/// The drawing surface (destination).
#[derive(Debug)]
pub struct RenderTarget<'a> {
    /// The destination buffer (pixels in u32 format).
    pub dest: &'a mut [u32],
    /// The stride of the destination buffer (pixels per row).
    pub stride: usize,
}

/// The spatial configuration for the glyph.
#[derive(Debug, Clone, Copy)]
pub struct GlyphRenderCoords {
    /// The X coordinate in pixels.
    pub x_px: usize,
    /// The Y coordinate in pixels.
    pub y_px: usize,
    /// The height of the cell in pixels (used for scaling).
    pub cell_height: usize,
}

/// The stylistic overrides.
#[derive(Debug, Clone, Copy)]
pub struct GlyphStyleOverrides {
    /// Foreground color (u32 RGBA).
    pub fg: u32,
    /// Background color (u32 RGBA).
    pub bg: u32,
    /// Whether to render in bold.
    pub bold: bool,
    /// Whether to render in italic.
    pub italic: bool,
}

/// Layout metrics returned after rendering.
#[derive(Debug, Clone, Copy)]
pub struct GlyphMetrics {
    /// The width of the glyph in pixels.
    pub width: usize,
    /// The height of the glyph in pixels.
    pub height: usize,
    /// Horizontal bearing (offset from origin).
    pub bearing_x: i32,
    /// Vertical bearing (offset from origin).
    pub bearing_y: i32,
}

// ============================================================================
// 2. The Composition Engine
// ============================================================================

const ITALIC_SHEAR: i32 = 50; // ~0.2 fixed point

/// Decompress and render a glyph directly into the provided framebuffer.
///
/// **Zero-Copy / Direct Mode:**
/// 1. Takes a mutable slice of the destination framebuffer.
/// 2. Creates a sub-view for the glyph.
/// 3. Composes a pipeline graph.
/// 4. Compiles and executes the graph.
///
/// # Parameters
/// * `ch` - The character to render.
/// * `target` - The render target (buffer and stride).
/// * `coords` - The coordinates and scaling info.
/// * `style` - Style overrides (color, bold, italic).
///
/// # Returns
/// * The metrics of the rendered glyph.
#[cfg(feature = "fonts")]
pub fn render_glyph_direct(
    ch: char,
    target: RenderTarget,
    coords: GlyphRenderCoords,
    style: GlyphStyleOverrides,
) -> GlyphMetrics {
    // --- 1. Metadata & Layout ---
    let glyph_idx = match GLYPH_METADATA.binary_search_by_key(&ch, |meta| meta.c) {
        Ok(idx) => idx,
        Err(_) => {
            return GlyphMetrics {
                width: 0,
                height: 0,
                bearing_x: 0,
                bearing_y: 0,
            }
        }
    };
    let meta = &GLYPH_METADATA[glyph_idx];

    let scale = coords.cell_height as f32 / 24.0;
    let width = (meta.width as f32 * scale).max(1.0) as usize;
    let height = (meta.height as f32 * scale).max(1.0) as usize;
    let bearing_x = (meta.bearing_x as f32 * scale) as i32;
    let bearing_y = (meta.bearing_y as f32 * scale) as i32;

    // --- 2. View Creation (Zero Copy) ---
    // Wrap the whole screen first to establish stride
    let mut screen_view = TensorViewMut::new(
        target.dest,
        target.stride,
        target.dest.len() / target.stride,
        target.stride,
    );

    // Create the window for this specific glyph (sub-slice)
    // Safety: rasterizer is responsible for bounds checking before calling
    let mut window = unsafe { screen_view.sub_view(coords.x_px, coords.y_px, width, height) };

    // --- 3. Pipeline Construction ---

    let stride = pixelflow_core::packing::compute_stride_4bit(meta.width);
    let packed_len = stride * meta.height;
    let packed = &GLYPH_DATA[meta.offset..meta.offset + packed_len];
    let atlas_view = TensorView::new(packed, meta.width, meta.height, stride);

    // Calculate step size in 16.16 fixed point format
    // step = 1.0 / scale
    let inv_scale = if scale > 0.001 { 1.0 / scale } else { 1.0 };
    let step_fp = (inv_scale * 65536.0) as u32;

    // The Base Signal: Sample from Atlas
    let sampler = SampleAtlas {
        atlas: atlas_view,
        step_x_fp: step_fp,
        step_y_fp: step_fp,
    };

    // Define Colors as Constant Surfaces (The Unification!)
    let fg = Batch::splat(style.fg);
    let bg = Batch::splat(style.bg);

    // Static Dispatch / Monomorphization happens here.
    match (style.bold, style.italic) {
        (false, false) => {
            // Simple: Sampler -> Blend
            // Uses the generalized Over<Mask, FG, BG>
            execute(sampler.over(fg, bg), &mut window);
        }
        (true, false) => {
            // Bold: Max(Sampler, Sampler.offset) -> Blend
            let bold = Max(sampler, sampler.offset(1, 0));
            execute(bold.over(fg, bg), &mut window);
        }
        (false, true) => {
            // Italic: Skew(Sampler) -> Blend
            let italic = sampler.skew(ITALIC_SHEAR);
            execute(italic.over(fg, bg), &mut window);
        }
        (true, true) => {
            // Both: Max(Italic, Italic.offset) -> Blend
            let italic = sampler.skew(ITALIC_SHEAR);
            let bold_italic = Max(italic, italic.offset(1, 0));
            execute(bold_italic.over(fg, bg), &mut window);
        }
    }

    GlyphMetrics {
        width,
        height,
        bearing_x,
        bearing_y,
    }
}

/// Fallback implementation when fonts are disabled.
#[cfg(not(feature = "fonts"))]
pub fn render_glyph_direct(
    _ch: char,
    _target: RenderTarget,
    _coords: GlyphRenderCoords,
    _style: GlyphStyleOverrides,
) -> GlyphMetrics {
    GlyphMetrics {
        width: 0,
        height: 0,
        bearing_x: 0,
        bearing_y: 0,
    }
}

/// Helper for Layout (needed by Rasterizer to calculate positions beforehand).
///
/// # Parameters
/// * `ch` - The character.
/// * `cell_height` - The desired cell height in pixels.
///
/// # Returns
/// * The metrics for the glyph.
#[cfg(feature = "fonts")]
pub fn get_glyph_metrics(ch: char, cell_height: usize) -> GlyphMetrics {
    let glyph_idx = match GLYPH_METADATA.binary_search_by_key(&ch, |meta| meta.c) {
        Ok(idx) => idx,
        Err(_) => {
            return GlyphMetrics {
                width: 0,
                height: 0,
                bearing_x: 0,
                bearing_y: 0,
            }
        }
    };
    let meta = &GLYPH_METADATA[glyph_idx];
    let scale = cell_height as f32 / 24.0;
    let width = (meta.width as f32 * scale).max(1.0) as usize;
    let height = (meta.height as f32 * scale).max(1.0) as usize;
    let bearing_x = (meta.bearing_x as f32 * scale) as i32;
    let bearing_y = (meta.bearing_y as f32 * scale) as i32;

    GlyphMetrics {
        width,
        height,
        bearing_x,
        bearing_y,
    }
}

/// Fallback implementation when fonts are disabled.
#[cfg(not(feature = "fonts"))]
pub fn get_glyph_metrics(_ch: char, _cell_height: usize) -> GlyphMetrics {
    GlyphMetrics {
        width: 0,
        height: 0,
        bearing_x: 0,
        bearing_y: 0,
    }
}
