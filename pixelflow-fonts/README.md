# pixelflow-fonts

High-performance, functional font rasterization for the PixelFlow ecosystem.

`pixelflow-fonts` is a pure Rust library that provides resolution-independent glyph rendering using Signed Distance Fields (SDF) and algebraic curve evaluation (Loop-Blinn). It integrates seamlessly with `pixelflow-core` to treat glyphs as just another `Surface` that can be transformed, masked, and composed.

## Features

- **TTF/OTF Support**: Parses TrueType and OpenType fonts via `ttf-parser`.
- **Infinite Resolution**: Glyphs are rasterized on-demand using mathematical curves, not pre-computed bitmaps.
- **SIMD Accelerated**: Heavily optimized vector rendering using `pixelflow-core`'s batching abstractions.
- **Zero-Copy**: Designed to minimize allocations; glyphs are lightweight handles to shared geometry.
- **Combinators**: Easily apply effects like **Bold**, *Slant* (Italic), and Scale using a functional API.

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
pixelflow-fonts = "0.1.0"
pixelflow-core = "0.1.0"
```

## Usage

### Basic Rendering

The core workflow involves loading a font, retrieving a glyph as a `Surface<u8>`, and compositing it.

```rust,ignore
use pixelflow_fonts::{Font, glyphs};
use pixelflow_core::dsl::MaskExt; // for .over()

// 1. Load the font
let font_data = std::fs::read("assets/fonts/Roboto-Regular.ttf")?;
let font = Font::from_bytes(&font_data)?;

// 2. create a glyph factory (handles caching and baking)
let glyph_factory = glyphs(font, 20, 24); // width, height bounds

// 3. Get a glyph (Lazy<Baked<u8>>)
let glyph = glyph_factory('A');

// 4. Render it over a background
// Glyph is a mask (u8), so we use it to blend a foreground color onto a background.
let fg = 0xFFFFFFFF; // White
let bg = 0xFF000000; // Black
let pixel = glyph.over(fg, bg);
```

### Advanced Combinators

You can modify glyphs before rasterization using the functional combinator API.

```rust,ignore
use pixelflow_fonts::{Font, CurveSurfaceExt};

let font = Font::from_bytes(data)?;
let glyph = font.glyph('B', 64.0).expect("Glyph not found");

// Create a Bold, Slanted version of the glyph
let styled_glyph = glyph
    .bold(2.0)       // Increase weight
    .slant(0.2);     // Apply shear/italic

// Render the styled glyph
// styled_glyph implements Surface<u8>
```

## How It Works

Unlike traditional rasterizers that bake glyphs into bitmaps at specific sizes, `pixelflow-fonts` retains the vector curve data (Lines and Quadratic Béziers).

When `eval(x, y)` is called on a `Glyph` surface:
1.  The pixel coordinates are transformed into glyph space.
2.  The winding number is computed to determine if the pixel is inside or outside the shape.
3.  The signed distance to the nearest curve is calculated using Loop-Blinn math for quadratic segments.
4.  Anti-aliasing is applied based on the distance.

This allows for high-quality rendering at any scale and enables effects like dynamic bolding (by dilating the SDF) without regenerating geometry.
