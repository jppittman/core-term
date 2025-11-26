//! GPU-style shader rendering for glyphs.
//!
//! This module provides the **fragment shader** implementation for text rendering.
//! Operations are fused into a single pass through SIMD registers with zero
//! intermediate allocations.
//!
//! ## The Pipeline
//!
//! 1. **Vertex Stage**: Transform screen coords to texture coords (projection)
//! 2. **Texture Stage**: Sample from 4-bit packed atlas with bilinear filtering
//! 3. **Effect Stage**: Apply bold synthesis (dilation filter)
//! 4. **Output Stage**: Alpha blend foreground/background colors
//!
//! All stages execute in SIMD registers - no memory allocations.

use pixelflow_core::{Batch, TensorView, TensorViewMut};
use crate::{FontWeight, GlyphParams};

/// Render glyph using shader pipeline (zero allocations, one pass).
///
/// This is the **fragment shader** - processes 4 pixels at a time,
/// sampling from 4-bit packed atlas and writing to color framebuffer.
///
/// # Pipeline Stages (all SIMD-fused)
/// - **4-bit unpacking**: Fused into texture sampling
/// - **Bilinear resampling**: Tensor-based interpolation
/// - **Bold synthesis**: Max(center, right_neighbor)
/// - **Alpha blending**: Foreground × alpha + Background × (1 - alpha)
///
/// # Arguments
/// - `dst`: Destination framebuffer (ARGB u32 pixels)
/// - `atlas`: Source 4-bit packed texture atlas
/// - `params`: Shader uniforms (colors, projections, effects)
///
/// # Example
/// ```ignore
/// let mut framebuffer = vec![0u32; 640 * 480];
/// let mut dst_view = TensorViewMut::new(&mut framebuffer, 640, 480, 640);
///
/// let params = GlyphParams {
///     style: GlyphStyle {
///         fg: 0xFFFFFFFF,  // White
///         bg: 0x00000000,  // Transparent
///         weight: FontWeight::Bold,
///     },
///     x_proj: Projection::scale(24, 12),  // Scale from 24px to 12px
///     y_proj: Projection::scale(24, 12),
/// };
///
/// render_glyph(&mut dst_view, &atlas_view, params);
/// ```
pub fn render_glyph(
    dst: &mut TensorViewMut<u32>,  // Destination framebuffer (ARGB)
    atlas: &TensorView<u8>,        // Source 4-bit packed atlas
    params: GlyphParams,
) {
    // Upload shader uniforms to SIMD registers (broadcast to all lanes)
    let fg = Batch::splat(params.style.fg);
    let bg = Batch::splat(params.style.bg);
    let x_start = Batch::splat(params.x_proj.start);
    let x_step = Batch::splat(params.x_proj.step);
    let y_start = Batch::splat(params.y_proj.start);
    let y_step = Batch::splat(params.y_proj.step);
    let is_bold = params.style.weight == FontWeight::Bold;

    // The shader kernel (fragment shader)
    dst.map_pixels(|x, y| {
        // === VERTEX STAGE ===
        // Transform screen coords to texture coords (16.16 fixed-point)
        let u_fp = x_start + (x * x_step);
        let v_fp = y_start + (y * y_step);

        // === TEXTURE STAGE ===
        // Sample from 4-bit packed atlas with bilinear filtering
        // (Unpacking is fused into the sampling operation)
        let mut alpha = unsafe { atlas.sample_4bit_bilinear(u_fp, v_fp) };

        // === EFFECT STAGE ===
        // Bold synthesis: Dilation filter (OR with right neighbor)
        if is_bold {
            let u_int = u_fp >> 16;
            let v_int = v_fp >> 16;
            let right_sample = unsafe {
                atlas.sample_4bit_nearest(u_int + Batch::splat(1), v_int)
            };
            alpha = alpha.max(right_sample);
        }

        // === OUTPUT STAGE ===
        // Alpha composite: fg × alpha + bg × (256 - alpha)
        // Replicate grayscale alpha across all ARGB channels
        let alpha_argb = alpha | (alpha << 8) | (alpha << 16) | (alpha << 24);
        fg.blend_alpha(bg, alpha_argb)
    });
}
