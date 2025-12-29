// pixelflow-render/src/color.rs
//! Unified color types for terminal rendering.
//!
//! This module provides:
//! - **Semantic colors**: `Color` enum for high-level specification
//! - **Pixel formats**: `Rgba8`, `Bgra8` for framebuffer storage
//! - **Color manifolds**: `ColorManifold`, `Lift`, `ColorMap` for functional color composition
//!
//! For color manifolds, use `pixelflow_core::{Rgba, Red, Green, Blue, Alpha}`.

use bitflags::bitflags;
use pixelflow_core::{Discrete, Field, Manifold};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

// Re-export the Pixel trait from the local pixel module
pub use super::pixel::Pixel;

// =============================================================================
// Semantic Color Types (The "User Input" tier)
// =============================================================================

/// Standard ANSI named colors (indices 0-15).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[repr(u8)]
pub enum NamedColor {
    /// ANSI Black.
    Black = 0,
    /// ANSI Red.
    Red = 1,
    /// ANSI Green.
    Green = 2,
    /// ANSI Yellow.
    Yellow = 3,
    /// ANSI Blue.
    Blue = 4,
    /// ANSI Magenta.
    Magenta = 5,
    /// ANSI Cyan.
    Cyan = 6,
    /// ANSI White.
    White = 7,
    /// ANSI Bright Black.
    BrightBlack = 8,
    /// ANSI Bright Red.
    BrightRed = 9,
    /// ANSI Bright Green.
    BrightGreen = 10,
    /// ANSI Bright Yellow.
    BrightYellow = 11,
    /// ANSI Bright Blue.
    BrightBlue = 12,
    /// ANSI Bright Magenta.
    BrightMagenta = 13,
    /// ANSI Bright Cyan.
    BrightCyan = 14,
    /// ANSI Bright White.
    BrightWhite = 15,
}

impl NamedColor {
    /// Convert a u8 index (0-15) to a NamedColor.
    pub fn from_index(idx: u8) -> Self {
        assert!(idx < 16, "Invalid NamedColor index: {}. Must be 0-15.", idx);
        unsafe { core::mem::transmute(idx) }
    }

    /// Returns the RGB representation of this named color.
    pub fn to_rgb(self) -> (u8, u8, u8) {
        match self {
            NamedColor::Black => (0, 0, 0),
            NamedColor::Red => (205, 0, 0),
            NamedColor::Green => (0, 205, 0),
            NamedColor::Yellow => (205, 205, 0),
            NamedColor::Blue => (0, 0, 238),
            NamedColor::Magenta => (205, 0, 205),
            NamedColor::Cyan => (0, 205, 205),
            NamedColor::White => (229, 229, 229),
            NamedColor::BrightBlack => (127, 127, 127),
            NamedColor::BrightRed => (255, 0, 0),
            NamedColor::BrightGreen => (0, 255, 0),
            NamedColor::BrightYellow => (255, 255, 0),
            NamedColor::BrightBlue => (92, 92, 255),
            NamedColor::BrightMagenta => (255, 0, 255),
            NamedColor::BrightCyan => (0, 255, 255),
            NamedColor::BrightWhite => (255, 255, 255),
        }
    }
}

// Make NamedColor a manifold - an infinite field of that ANSI color
impl pixelflow_core::Manifold for NamedColor {
    type Output = pixelflow_core::Discrete;

    #[inline(always)]
    fn eval_raw(
        &self,
        _x: pixelflow_core::Field,
        _y: pixelflow_core::Field,
        _z: pixelflow_core::Field,
        _w: pixelflow_core::Field,
    ) -> pixelflow_core::Discrete {
        let (r, g, b) = self.to_rgb();
        pixelflow_core::Discrete::pack(
            pixelflow_core::Field::from(r as f32 / 255.0),
            pixelflow_core::Field::from(g as f32 / 255.0),
            pixelflow_core::Field::from(b as f32 / 255.0),
            pixelflow_core::Field::from(1.0),
        )
    }
}

/// Represents a semantic color value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Color {
    /// Default foreground or background color.
    #[default]
    Default,
    /// A standard named ANSI color (indices 0-15).
    Named(NamedColor),
    /// An indexed color from the 256-color palette (indices 0-255).
    Indexed(u8),
    /// An RGB true color.
    Rgb(u8, u8, u8),
}

impl Color {
    /// Convert to an Rgba8 pixel.
    #[inline]
    pub fn to_rgba8(self) -> Rgba8 {
        Rgba8(u32::from(self))
    }

    /// Convert to a Bgra8 pixel.
    #[inline]
    pub fn to_bgra8(self) -> Bgra8 {
        Bgra8::from(self.to_rgba8())
    }

    /// Convert to normalized f32 RGBA components.
    #[inline]
    pub fn to_f32_rgba(self) -> (f32, f32, f32, f32) {
        let rgba = self.to_rgba8();
        (
            rgba.r() as f32 / 255.0,
            rgba.g() as f32 / 255.0,
            rgba.b() as f32 / 255.0,
            rgba.a() as f32 / 255.0,
        )
    }
}

// Make Color a manifold - an infinite field of that color
impl pixelflow_core::Manifold for Color {
    type Output = pixelflow_core::Discrete;

    #[inline(always)]
    fn eval_raw(
        &self,
        _x: pixelflow_core::Field,
        _y: pixelflow_core::Field,
        _z: pixelflow_core::Field,
        _w: pixelflow_core::Field,
    ) -> pixelflow_core::Discrete {
        let (r, g, b, a) = self.to_f32_rgba();
        pixelflow_core::Discrete::pack(
            pixelflow_core::Field::from(r),
            pixelflow_core::Field::from(g),
            pixelflow_core::Field::from(b),
            pixelflow_core::Field::from(a),
        )
    }
}

// Constants for 256-color palette conversion
const ANSI_NAMED_COLOR_COUNT: u8 = 16;
const COLOR_CUBE_OFFSET: u8 = 16;
const COLOR_CUBE_SIZE: u8 = 6;
const COLOR_CUBE_TOTAL_COLORS: u8 = COLOR_CUBE_SIZE * COLOR_CUBE_SIZE * COLOR_CUBE_SIZE;
const GRAYSCALE_OFFSET: u8 = COLOR_CUBE_OFFSET + COLOR_CUBE_TOTAL_COLORS;

impl From<Color> for u32 {
    /// Convert a Color to a u32 pixel value (RGBA format: 0xAABBGGRR).
    fn from(color: Color) -> u32 {
        let (r, g, b) = match color {
            Color::Default => (0, 0, 0),
            Color::Named(named) => named.to_rgb(),
            Color::Indexed(idx) => {
                if idx < ANSI_NAMED_COLOR_COUNT {
                    NamedColor::from_index(idx).to_rgb()
                } else if idx < GRAYSCALE_OFFSET {
                    // 6x6x6 Color Cube (indices 16-231)
                    let cube_idx = idx - COLOR_CUBE_OFFSET;
                    let r_comp = (cube_idx / (COLOR_CUBE_SIZE * COLOR_CUBE_SIZE)) % COLOR_CUBE_SIZE;
                    let g_comp = (cube_idx / COLOR_CUBE_SIZE) % COLOR_CUBE_SIZE;
                    let b_comp = cube_idx % COLOR_CUBE_SIZE;
                    let r_val = if r_comp == 0 { 0 } else { r_comp * 40 + 55 };
                    let g_val = if g_comp == 0 { 0 } else { g_comp * 40 + 55 };
                    let b_val = if b_comp == 0 { 0 } else { b_comp * 40 + 55 };
                    (r_val, g_val, b_val)
                } else {
                    // Grayscale ramp (indices 232-255)
                    let gray_idx = idx - GRAYSCALE_OFFSET;
                    let level = gray_idx * 10 + 8;
                    (level, level, level)
                }
            }
            Color::Rgb(r, g, b) => (r, g, b),
        };
        u32::from_le_bytes([r, g, b, 255])
    }
}

// =============================================================================
// Text Attributes
// =============================================================================

bitflags! {
    /// Text attribute flags (bold, underline, etc.).
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
    #[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
    pub struct AttrFlags: u16 {
        const BOLD          = 1 << 0;
        const FAINT         = 1 << 1;
        const ITALIC        = 1 << 2;
        const UNDERLINE     = 1 << 3;
        const BLINK         = 1 << 4;
        const REVERSE       = 1 << 5;
        const HIDDEN        = 1 << 6;
        const STRIKETHROUGH = 1 << 7;
    }
}

// =============================================================================
// Pixel Format Types (Storage types)
// =============================================================================

/// Rgba8 pixel: bytes are [R, G, B, A] in memory order.
/// As a u32 on little-endian: 0xAABBGGRR.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(transparent)]
pub struct Rgba8(pub u32);

/// Bgra8 pixel: bytes are [B, G, R, A] in memory order.
/// As a u32 on little-endian: 0xAARRGGBB.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(transparent)]
pub struct Bgra8(pub u32);

impl Rgba8 {
    /// Creates a new RGBA pixel from component values.
    #[inline]
    pub fn new(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self(u32::from_le_bytes([r, g, b, a]))
    }

    #[inline]
    pub fn r(self) -> u8 {
        self.0.to_le_bytes()[0]
    }
    #[inline]
    pub fn g(self) -> u8 {
        self.0.to_le_bytes()[1]
    }
    #[inline]
    pub fn b(self) -> u8 {
        self.0.to_le_bytes()[2]
    }
    #[inline]
    pub fn a(self) -> u8 {
        self.0.to_le_bytes()[3]
    }
}

impl Bgra8 {
    /// Creates a new BGRA pixel from component values.
    #[inline]
    pub fn new(b: u8, g: u8, r: u8, a: u8) -> Self {
        Self(u32::from_le_bytes([b, g, r, a]))
    }

    #[inline]
    pub fn b(self) -> u8 {
        self.0.to_le_bytes()[0]
    }
    #[inline]
    pub fn g(self) -> u8 {
        self.0.to_le_bytes()[1]
    }
    #[inline]
    pub fn r(self) -> u8 {
        self.0.to_le_bytes()[2]
    }
    #[inline]
    pub fn a(self) -> u8 {
        self.0.to_le_bytes()[3]
    }
}

// Swizzle: swap bytes 0 and 2 (R and B)
#[inline]
fn swizzle_rb(v: u32) -> u32 {
    (v & 0xFF00FF00) | ((v >> 16) & 0x000000FF) | ((v & 0x000000FF) << 16)
}

impl From<Bgra8> for Rgba8 {
    #[inline]
    fn from(bgra: Bgra8) -> Rgba8 {
        Rgba8(swizzle_rb(bgra.0))
    }
}

impl From<Rgba8> for Bgra8 {
    #[inline]
    fn from(rgba: Rgba8) -> Bgra8 {
        Bgra8(swizzle_rb(rgba.0))
    }
}

// =============================================================================
// Pixel Trait Implementations
// =============================================================================

impl Pixel for Rgba8 {
    #[inline]
    fn from_u32(v: u32) -> Self {
        Self(v)
    }
    #[inline]
    fn to_u32(self) -> u32 {
        self.0
    }
    #[inline]
    fn from_rgba(r: f32, g: f32, b: f32, a: f32) -> Self {
        let r = (r * 255.0).clamp(0.0, 255.0) as u8;
        let g = (g * 255.0).clamp(0.0, 255.0) as u8;
        let b = (b * 255.0).clamp(0.0, 255.0) as u8;
        let a = (a * 255.0).clamp(0.0, 255.0) as u8;
        Self::new(r, g, b, a)
    }
}

impl Pixel for Bgra8 {
    #[inline]
    fn from_u32(v: u32) -> Self {
        Self(v)
    }
    #[inline]
    fn to_u32(self) -> u32 {
        self.0
    }
    #[inline]
    fn from_rgba(r: f32, g: f32, b: f32, a: f32) -> Self {
        let r = (r * 255.0).clamp(0.0, 255.0) as u8;
        let g = (g * 255.0).clamp(0.0, 255.0) as u8;
        let b = (b * 255.0).clamp(0.0, 255.0) as u8;
        let a = (a * 255.0).clamp(0.0, 255.0) as u8;
        Self::new(b, g, r, a)
    }
}

// =============================================================================
// Platform-specific type aliases
// =============================================================================

/// Pixel format for X11 (XImage with ZPixmap on little-endian).
pub type X11Pixel = Bgra8;

/// Pixel format for Cocoa (CGImage with kCGImageAlphaPremultipliedLast).
pub type CocoaPixel = Rgba8;

/// Pixel format for Web (ImageData).
pub type WebPixel = Rgba8;

// =============================================================================
// Color Manifolds
// =============================================================================

/// A color manifold composed of four independent channel manifolds (R, G, B, A).
///
/// # Overview
///
/// `ColorManifold` is the primary abstraction for building complex color compositions.
/// Unlike the semantic `Color` enum (which represents a single color value),
/// a `ColorManifold` treats each RGBA channel as an independent manifold that can
/// vary across the 2D coordinate space.
///
/// # Evaluation Contract
///
/// When a `ColorManifold` is evaluated at coordinates `(x, y)`:
/// 1. The red channel manifold is evaluated to produce a value in [0.0, 1.0]
/// 2. The green channel manifold is evaluated to produce a value in [0.0, 1.0]
/// 3. The blue channel manifold is evaluated to produce a value in [0.0, 1.0]
/// 4. The alpha channel manifold is evaluated to produce a value in [0.0, 1.0]
/// 5. The four values are packed into a single `Discrete` u32 pixel (RGBA format)
///
/// If any channel evaluates outside [0.0, 1.0], it is automatically clamped.
///
/// # Type Parameters
///
/// - `R`: Manifold producing the red channel
/// - `G`: Manifold producing the green channel
/// - `B`: Manifold producing the blue channel
/// - `A`: Manifold producing the alpha channel
///
/// Each parameter can be:
/// - A constant scalar (f32 promotes to a manifold automatically)
/// - A coordinate manifold like `X`, `Y`, or `X * Y`
/// - A complex expression like `(X / 100.0).abs()`
/// - Any type implementing `Manifold<Output = Field>`
///
/// # Examples
///
/// ## Solid Color
/// ```ignore
/// use pixelflow_graphics::ColorManifold;
///
/// // Solid red everywhere
/// let red = ColorManifold::new(1.0, 0.0, 0.0, 1.0);
/// ```
///
/// ## Horizontal Gradient
/// ```ignore
/// use pixelflow_graphics::ColorManifold;
/// use pixelflow_core::X;
///
/// // Red increases from left to right, green and blue are constant
/// let gradient = ColorManifold::new(
///     X / 255.0,   // Red varies with X
///     0.5f32,      // Green constant
///     0.5f32,      // Blue constant
///     1.0f32,      // Alpha fully opaque
/// );
/// ```
///
/// ## Radial Gradient (Heatmap)
/// ```ignore
/// use pixelflow_graphics::ColorManifold;
/// use pixelflow_core::{X, Y};
///
/// // Distance from center determines heat color
/// let center_x = 128.0;
/// let center_y = 128.0;
/// let distance = ((X - center_x) * (X - center_x) + (Y - center_y) * (Y - center_y)).sqrt();
///
/// let heatmap = ColorManifold::new(
///     1.0 - (distance / 200.0),  // Red (inverse distance)
///     distance / 200.0,           // Green (distance)
///     0.0f32,                     // Blue off
///     1.0f32,
/// );
/// ```
///
/// # Channel Accessors
///
/// You can inspect individual channel manifolds after construction:
///
/// ```ignore
/// let cm = ColorManifold::new(X / 255.0, 0.5, 0.5, 1.0);
/// let red_channel = cm.red();    // Access the red manifold
/// let alpha_channel = cm.alpha(); // Access the alpha manifold
/// ```
///
/// # Performance Notes
///
/// - Channels are evaluated independently; the compiler typically fuses them into a single SIMD kernel
/// - Shared subexpressions across channels are automatically deduplicated by the compiler
/// - No runtime overhead compared to manually packing Field values
#[derive(Clone, Copy, Debug)]
pub struct ColorManifold<R, G, B, A> {
    r: R,
    g: G,
    b: B,
    a: A,
}

impl<R, G, B, A> ColorManifold<R, G, B, A> {
    /// Create a new color manifold from 4 channel manifolds.
    pub fn new(r: R, g: G, b: B, a: A) -> Self {
        Self { r, g, b, a }
    }

    /// Access the red channel manifold.
    pub fn red(&self) -> &R {
        &self.r
    }

    /// Access the green channel manifold.
    pub fn green(&self) -> &G {
        &self.g
    }

    /// Access the blue channel manifold.
    pub fn blue(&self) -> &B {
        &self.b
    }

    /// Access the alpha channel manifold.
    pub fn alpha(&self) -> &A {
        &self.a
    }
}

impl<R, G, B, A> Manifold for ColorManifold<R, G, B, A>
where
    R: Manifold<Output = Field>,
    G: Manifold<Output = Field>,
    B: Manifold<Output = Field>,
    A: Manifold<Output = Field>,
{
    type Output = Discrete;

    #[inline(always)]
    fn eval_raw(&self, x: Field, y: Field, z: Field, w: Field) -> Discrete {
        let r = self.r.eval_raw(x, y, z, w);
        let g = self.g.eval_raw(x, y, z, w);
        let b = self.b.eval_raw(x, y, z, w);
        let a = self.a.eval_raw(x, y, z, w);
        Discrete::pack(r, g, b, a)
    }
}

/// Lifts a scalar manifold to grayscale color by duplicating its value across R, G, B channels.
///
/// # Purpose
///
/// `Lift<M>` is a convenience combinator for converting scalar manifolds (that produce single `Field` values)
/// into color manifolds (that produce `Discrete` RGBA pixels). It's commonly used for:
///
/// - Creating monochromatic gradients
/// - Rendering distance fields or signed distance fields as grayscale
/// - Heatmaps where a single value should map to neutral gray
///
/// # Evaluation Contract
///
/// When `Lift<M>` is evaluated at coordinates `(x, y)`:
/// 1. The inner scalar manifold `M` is evaluated to produce a value in [0.0, 1.0]
/// 2. That value is replicated across all three color channels: R = G = B = value
/// 3. Alpha is set to 1.0 (fully opaque)
/// 4. The four components are packed into a single `Discrete` u32 pixel
///
/// Result: If the scalar evaluates to gray (0.5), the pixel will be RGB(128, 128, 128) with A=255.
///
/// # Examples
///
/// ## Grayscale Gradient
/// ```ignore
/// use pixelflow_graphics::Lift;
/// use pixelflow_core::X;
///
/// // Grayscale gradient from black to white (left to right)
/// let gray_gradient = Lift(X / 256.0);
///
/// // At x=0, produces black (0, 0, 0)
/// // At x=256, produces white (255, 255, 255)
/// ```
///
/// ## Distance Field Visualization
/// ```ignore
/// use pixelflow_graphics::Lift;
/// use pixelflow_core::{X, Y};
///
/// // Visualize distance from center as grayscale brightness
/// let center_x = 128.0;
/// let center_y = 128.0;
/// let distance = ((X - center_x) * (X - center_x) + (Y - center_y) * (Y - center_y)).sqrt();
/// let normalized_distance = 1.0 - (distance / 256.0);  // Invert: center is white
///
/// let distance_field = Lift(normalized_distance.clamp(0.0, 1.0));
/// ```
///
/// # Type Parameter
///
/// `M` must implement `Manifold<Output = Field>` - that is, it produces a single scalar value
/// when evaluated. This includes:
/// - Coordinate manifolds (`X`, `Y`, `Z`, `W`)
/// - Arithmetic expressions (`X + Y`, `(X * Y).sqrt()`)
/// - Any custom manifold producing scalar output
///
/// # Performance Notes
///
/// `Lift` is zero-cost. The inner scalar is evaluated once and replicated three times in the packed output.
/// The compiler fuses this into a single instruction on most targets.
///
/// # Contrast with ColorManifold
///
/// | Feature | Lift | ColorManifold |
/// |---------|------|---------------|
/// | Input | Single scalar manifold | Four independent channel manifolds |
/// | Use case | Grayscale/monochrome | Complex color compositions |
/// | Simplicity | High | More complex (but more flexible) |
#[derive(Clone, Copy, Debug)]
pub struct Lift<M>(pub M);

impl<M: Manifold<Output = Field> + Clone> Manifold for Lift<M> {
    type Output = Discrete;

    #[inline(always)]
    fn eval_raw(&self, x: Field, y: Field, z: Field, w: Field) -> Discrete {
        let v = self.0.eval_raw(x, y, z, w);
        Discrete::pack(v, v, v, Field::from(1.0))
    }
}

/// Maps a scalar function over all four color channels independently before packing.
///
/// # Purpose
///
/// `ColorMap` is a postprocessing combinator for applying effects uniformly across all channels
/// (R, G, B, A) after a `ColorManifold` has been evaluated. Common uses:
///
/// - Brightness/contrast adjustment
/// - Gamma correction or tone mapping
/// - Color space transformations
/// - Clamping or normalization
/// - Threshold effects (convert to black/white)
///
/// # Evaluation Contract
///
/// When `ColorMap` evaluates at coordinates `(x, y)`:
/// 1. The inner `ColorManifold` is evaluated to produce raw R, G, B, A values
/// 2. The function `F` is applied to each channel independently: `f(r)`, `f(g)`, `f(b)`, `f(a)`
/// 3. The transformed values are clamped to [0.0, 1.0]
/// 4. The four components are packed into a single `Discrete` u32 pixel
///
/// # Type Parameters
///
/// - `C`: A `ColorManifold<R, G, B, A>` that produces the base colors
/// - `F`: A function `Fn(Field) -> Field + Send + Sync + Copy` that transforms each channel
///
/// # Examples
///
/// ## Brightness Adjustment
/// ```ignore
/// use pixelflow_graphics::{ColorManifold, ColorMap};
/// use pixelflow_core::X;
///
/// // Create a gradient
/// let gradient = ColorManifold::new(X / 255.0, 0.5, 0.5, 1.0);
///
/// // Brighten by multiplying all channels by 1.5
/// let brighter = ColorMap::new(gradient, |channel| channel * 1.5);
/// ```
///
/// ## Gamma Correction
/// ```ignore
/// use pixelflow_graphics::{ColorManifold, ColorMap};
/// use pixelflow_core::X;
///
/// let gradient = ColorManifold::new(X / 255.0, 0.5, 0.5, 1.0);
///
/// // Apply gamma correction (gamma = 2.2)
/// let gamma_corrected = ColorMap::new(gradient, |channel| {
///     // For proper gamma correction: channel.powf(1.0 / 2.2)
///     // But this requires exposing powf on Field
///     channel * channel  // Simplified: square (approx gamma 2.0)
/// });
/// ```
///
/// ## Threshold Effect (Posterize)
/// ```ignore
/// use pixelflow_graphics::{ColorManifold, ColorMap};
/// use pixelflow_core::X;
///
/// let gradient = ColorManifold::new(X / 255.0, 0.5, 0.5, 1.0);
///
/// // Convert to binary black/white based on threshold
/// let posterized = ColorMap::new(gradient, |channel| {
///     if channel < 0.5 { 0.0 } else { 1.0 }
/// });
/// ```
///
/// # Limitations
///
/// The function `F` must be:
/// - **Deterministic**: Same input always produces same output
/// - **Pure**: No side effects (can be called any number of times)
/// - **Simple algebra**: The compiler must be able to inline and vectorize it
///
/// `F` receives and returns `Field` (the SIMD IR), so you can only use `Computational` operations.
/// You cannot use non-algebraic functions (e.g., random numbers, file I/O, external state).
///
/// # Performance Notes
///
/// - The function is applied four times per pixel (once per channel), but typically inlines to a single SIMD operation
/// - The overhead of the function call is eliminated by monomorphization
/// - Combining multiple `ColorMap`s chains them; the compiler fuses them into one kernel
///
/// # Example: Chaining ColorMaps
///
/// ```ignore
/// use pixelflow_graphics::{ColorManifold, ColorMap};
/// use pixelflow_core::X;
///
/// let gradient = ColorManifold::new(X / 255.0, 0.5, 0.5, 1.0);
/// let brighten = ColorMap::new(gradient, |c| c * 1.2);
/// let saturate = ColorMap::new(brighten, |c| c * c);  // Deepen colors
///
/// // Both transformations are fused by the compiler into a single kernel
/// ```
#[derive(Clone, Copy, Debug)]
pub struct ColorMap<C, F> {
    color: C,
    func: F,
}

impl<R, G, B, A, F> ColorMap<ColorManifold<R, G, B, A>, F>
where
    R: Manifold<Output = Field>,
    G: Manifold<Output = Field>,
    B: Manifold<Output = Field>,
    A: Manifold<Output = Field>,
    F: Fn(Field) -> Field + Send + Sync + Copy,
{
    /// Create a new ColorMap.
    pub fn new(color: ColorManifold<R, G, B, A>, func: F) -> Self {
        Self { color, func }
    }
}

impl<R, G, B, A, F> Manifold for ColorMap<ColorManifold<R, G, B, A>, F>
where
    R: Manifold<Output = Field>,
    G: Manifold<Output = Field>,
    B: Manifold<Output = Field>,
    A: Manifold<Output = Field>,
    F: Fn(Field) -> Field + Send + Sync + Copy,
{
    type Output = Discrete;

    #[inline(always)]
    fn eval_raw(&self, x: Field, y: Field, z: Field, w: Field) -> Discrete {
        let r = (self.func)(self.color.r.eval_raw(x, y, z, w));
        let g = (self.func)(self.color.g.eval_raw(x, y, z, w));
        let b = (self.func)(self.color.b.eval_raw(x, y, z, w));
        let a = (self.func)(self.color.a.eval_raw(x, y, z, w));
        Discrete::pack(r, g, b, a)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rgba8_components() {
        let c = Rgba8::new(0x11, 0x22, 0x33, 0xFF);
        assert_eq!(c.r(), 0x11);
        assert_eq!(c.g(), 0x22);
        assert_eq!(c.b(), 0x33);
        assert_eq!(c.a(), 0xFF);
    }

    #[test]
    fn test_bgra8_components() {
        let c = Bgra8::new(0x33, 0x22, 0x11, 0xFF);
        assert_eq!(c.b(), 0x33);
        assert_eq!(c.g(), 0x22);
        assert_eq!(c.r(), 0x11);
        assert_eq!(c.a(), 0xFF);
    }

    #[test]
    fn test_rgba8_to_bgra8() {
        let rgba = Rgba8::new(0x11, 0x22, 0x33, 0xFF);
        let bgra = Bgra8::from(rgba);
        assert_eq!(bgra.r(), 0x11);
        assert_eq!(bgra.g(), 0x22);
        assert_eq!(bgra.b(), 0x33);
        assert_eq!(bgra.a(), 0xFF);
    }
}
