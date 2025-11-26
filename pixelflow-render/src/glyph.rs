//! Zero-copy glyph rendering using functional pipelines.
//!
//! This module uses pixelflow-core's compile-time compute graphs to render glyphs
//! directly into the framebuffer with zero intermediate allocations.

// Include the generated baked font data
#[cfg(feature = "fonts")]
include!(concat!(env!("OUT_DIR"), "/baked_font.rs"));

use pixelflow_core::{execute, ops::*, dsl::*, TensorView, TensorViewMut, Projection};

/// Glyph metrics for layout calculations.
///
/// Does not own pixel data - used only for positioning glyphs on the baseline.
pub struct GlyphMetrics {
    pub width: usize,
    pub height: usize,
    pub bearing_x: i32,
    pub bearing_y: i32,
}

/// Render target: the destination framebuffer and its stride.
pub struct RenderTarget<'a> {
    pub dest: &'a mut [u32],
    pub stride: usize, // Width of entire framebuffer in pixels
}

/// Glyph rendering coordinates and scale.
pub struct GlyphRenderCoords {
    pub x_px: usize,        // Top-left X in destination pixels
    pub y_px: usize,        // Top-left Y in destination pixels
    pub cell_height: usize, // For calculating scaling
}

/// Glyph style overrides.
pub struct GlyphStyleOverrides {
    pub fg: u32,
    pub bg: u32,
    pub bold: bool,
    pub italic: bool,
}

/// Italic shear constant (8.8 fixed-point, approximately 0.2 radians).
const ITALIC_SHEAR: i32 = 50;

/// Get glyph metrics for layout calculations without rendering.
///
/// Used by the rasterizer to calculate baseline positioning before rendering.
#[cfg(feature = "fonts")]
pub fn get_glyph_metrics_for_layout(ch: char, cell_height: usize) -> GlyphMetrics {
    // Binary search in GLYPH_METADATA
    let glyph_idx = match GLYPH_METADATA.binary_search_by_key(&ch, |meta| meta.c) {
        Ok(idx) => idx,
        Err(_) => {
            // Glyph not found, return minimal metrics
            return GlyphMetrics {
                width: 1,
                height: 1,
                bearing_x: 0,
                bearing_y: 0,
            };
        }
    };

    let meta = &GLYPH_METADATA[glyph_idx];

    // Calculate output dimensions
    let scale = cell_height as f32 / 24.0; // 24pt is our base size
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

/// Render a glyph directly into the framebuffer using functional pipelines.
///
/// This is the **zero-copy, zero-allocation** rendering path.
/// The pipeline is composed at compile-time and fused into a single SIMD loop.
///
/// # Architecture
///
/// This function builds a compute graph based on the style flags:
///
/// - **Normal**: `sampler.over(fg, bg)`
/// - **Bold**: `Max(sampler, sampler.offset(1, 0)).over(fg, bg)`
/// - **Italic**: `sampler.skew(ITALIC_SHEAR).over(fg, bg)`
/// - **Bold+Italic**: `Max(italic, italic.offset(1, 0)).over(fg, bg)`
///
/// The Rust compiler monomorphizes each variant into a specialized function
/// with the entire pipeline inlined.
///
/// # Example
///
/// ```ignore
/// let mut fb = vec![0u32; 1920 * 1080];
/// let target = RenderTarget { dest: &mut fb, stride: 1920 };
/// let coords = GlyphRenderCoords { x_px: 100, y_px: 200, cell_height: 24 };
/// let style = GlyphStyleOverrides { fg: 0xFFFFFFFF, bg: 0x00000000, bold: true, italic: false };
///
/// render_glyph_direct('A', target, coords, style);
/// ```
#[cfg(feature = "fonts")]
pub fn render_glyph_direct(
    ch: char,
    target: RenderTarget,
    coords: GlyphRenderCoords,
    style: GlyphStyleOverrides,
) -> GlyphMetrics {
    // 1. Lookup glyph metadata
    let glyph_idx = match GLYPH_METADATA.binary_search_by_key(&ch, |meta| meta.c) {
        Ok(idx) => idx,
        Err(_) => {
            // Glyph not found, return minimal metrics (could render a fallback glyph)
            return GlyphMetrics {
                width: 1,
                height: 1,
                bearing_x: 0,
                bearing_y: 0,
            };
        }
    };

    let meta = &GLYPH_METADATA[glyph_idx];

    // 2. Calculate output dimensions
    let scale = coords.cell_height as f32 / 24.0; // 24pt is our base size
    let width = (meta.width as f32 * scale).max(1.0) as usize;
    let height = (meta.height as f32 * scale).max(1.0) as usize;
    let bearing_x = (meta.bearing_x as f32 * scale) as i32;
    let bearing_y = (meta.bearing_y as f32 * scale) as i32;

    // 3. Create framebuffer view and glyph window
    let screen_height = target.dest.len() / target.stride;
    let mut screen_view = TensorViewMut::new(target.dest, target.stride, screen_height, target.stride);

    // Create sub-view for this specific glyph (bounds-checked by caller)
    let mut glyph_window = unsafe {
        screen_view.sub_view(coords.x_px, coords.y_px, width, height)
    };

    // 4. Get atlas data (stays 4-bit packed!)
    let packed_len = (meta.width * meta.height + 1) / 2;
    let packed = &GLYPH_DATA[meta.offset..meta.offset + packed_len];
    let atlas_view = TensorView::new(packed, meta.width, meta.height, (meta.width + 1) / 2);

    // 5. Build the base sampler with coordinate transformation
    // We need to wrap SampleAtlas with Scale to apply projection
    // For now, we'll use the projection in a custom sampler
    let x_proj = Projection::scale(meta.width, width);
    let y_proj = Projection::scale(meta.height, height);

    // Create a custom sampler that applies projection
    let sampler = ScaledSampler {
        atlas: atlas_view,
        x_start: x_proj.start,
        x_step: x_proj.step,
        y_start: y_proj.start,
        y_step: y_proj.step,
    };

    // 6. Compose the pipeline based on style flags
    match (style.bold, style.italic) {
        (false, false) => {
            // Normal: sampler → blend
            let pipeline = sampler.over(style.fg, style.bg);
            execute(pipeline, &mut glyph_window);
        }

        (true, false) => {
            // Bold: Max(sampler, sampler.offset) → blend
            let bold = Max(sampler, sampler.offset(1, 0));
            let pipeline = bold.over(style.fg, style.bg);
            execute(pipeline, &mut glyph_window);
        }

        (false, true) => {
            // Italic: sampler.skew → blend
            let italic = sampler.skew(ITALIC_SHEAR);
            let pipeline = italic.over(style.fg, style.bg);
            execute(pipeline, &mut glyph_window);
        }

        (true, true) => {
            // Bold+Italic: skew first, then dilate
            let italic = sampler.skew(ITALIC_SHEAR);
            let bold_italic = Max(italic, italic.offset(1, 0));
            let pipeline = bold_italic.over(style.fg, style.bg);
            execute(pipeline, &mut glyph_window);
        }
    }

    // 7. Return metrics for layout
    GlyphMetrics {
        width,
        height,
        bearing_x,
        bearing_y,
    }
}

/// A sampler that applies coordinate projection (scaling).
///
/// This wraps `SampleAtlas` with the coordinate transformation logic
/// that was previously in the shader.
#[derive(Copy, Clone)]
struct ScaledSampler<'a> {
    atlas: TensorView<'a, u8>,
    x_start: u32,
    x_step: u32,
    y_start: u32,
    y_step: u32,
}

impl<'a> pixelflow_core::pipe::Surface<u8> for ScaledSampler<'a> {
    #[inline(always)]
    fn eval(&self, x: pixelflow_core::Batch<u32>, y: pixelflow_core::Batch<u32>) -> pixelflow_core::Batch<u8> {
        use pixelflow_core::Batch;

        // Apply projection (vertex stage)
        let x_start = Batch::splat(self.x_start);
        let x_step = Batch::splat(self.x_step);
        let y_start = Batch::splat(self.y_start);
        let y_step = Batch::splat(self.y_step);

        let u_fp = x_start + (x * x_step);
        let v_fp = y_start + (y * y_step);

        // Sample atlas (texture stage)
        unsafe { self.atlas.sample_4bit_bilinear(u_fp, v_fp).cast() }
    }
}

/// Legacy API: Render glyph with natural scaling, returning owned data.
///
/// **Note**: This allocates a temporary u32 buffer and extracts alpha.
/// Prefer `render_glyph_direct()` for zero-copy rendering.
#[cfg(feature = "fonts")]
pub struct RenderedGlyph {
    pub data: Vec<u8>,
    pub width: usize,
    pub height: usize,
    pub bearing_x: i32,
    pub bearing_y: i32,
}

#[cfg(feature = "fonts")]
pub fn render_glyph_natural(ch: char, target_cell_height: usize, bold: bool, italic: bool) -> RenderedGlyph {
    // Get metrics
    let metrics = get_glyph_metrics_for_layout(ch, target_cell_height);

    // Allocate temporary framebuffer
    let mut fb = vec![0u32; metrics.width * metrics.height];

    // Render using direct API
    let target = RenderTarget {
        dest: &mut fb,
        stride: metrics.width,
    };
    let coords = GlyphRenderCoords {
        x_px: 0,
        y_px: 0,
        cell_height: target_cell_height,
    };
    let style = GlyphStyleOverrides {
        fg: 0xFFFFFFFF, // White
        bg: 0x00000000, // Transparent
        bold,
        italic,
    };

    render_glyph_direct(ch, target, coords, style);

    // Extract alpha channel from ARGB (all channels contain the same value after blending)
    let data: Vec<u8> = fb.iter().map(|&px| (px & 0xFF) as u8).collect();

    RenderedGlyph {
        data,
        width: metrics.width,
        height: metrics.height,
        bearing_x: metrics.bearing_x,
        bearing_y: metrics.bearing_y,
    }
}

/// Legacy API: Render glyph to owned data.
#[cfg(feature = "fonts")]
pub fn render_glyph(ch: char, _width: usize, height: usize, bold: bool, italic: bool) -> Vec<u8> {
    render_glyph_natural(ch, height, bold, italic).data
}

/// Stub for when fonts feature is not enabled.
#[cfg(not(feature = "fonts"))]
pub fn render_glyph(_ch: char, width: usize, height: usize, _bold: bool, _italic: bool) -> Vec<u8> {
    vec![0u8; width * height]
}

#[cfg(not(feature = "fonts"))]
pub fn get_glyph_metrics_for_layout(_ch: char, _cell_height: usize) -> GlyphMetrics {
    GlyphMetrics {
        width: 1,
        height: 1,
        bearing_x: 0,
        bearing_y: 0,
    }
}

#[cfg(not(feature = "fonts"))]
pub fn render_glyph_direct(
    _ch: char,
    _target: RenderTarget,
    _coords: GlyphRenderCoords,
    _style: GlyphStyleOverrides,
) -> GlyphMetrics {
    GlyphMetrics {
        width: 1,
        height: 1,
        bearing_x: 0,
        bearing_y: 0,
    }
}
