# Font Rendering Design

**Status**: In Progress / Adapting to PixelFlow 1.0

## 1. Overview

In PixelFlow 1.0, Fonts are just **Surfaces**.
A Font file is a factory that produces `Surface<Coverage>` (or `Surface<Color>`).

## 2. The Model

*   **Glyphs are Surfaces**: A glyph is a function `f(x, y) -> coverage`.
    *   This is typically implemented via the Loop-Blinn formulation for quadratic Bézier curves.
    *   `coverage = (u^2 - v)` inside the curve.
*   **Fonts are Factories**: A `Font` object (loaded from TTF/OTF) provides a method to get a Surface for a specific character/glyph index.
    *   `font.get_glyph(char) -> impl Surface<f32>`

## 3. Font Management

The `FontManager` (conceptually part of `pixelflow-graphics`) is responsible for:
1.  **Discovery**: Finding font files on the system (using `fontconfig` on Linux, CoreText on macOS, etc.).
2.  **Loading**: Parsing the font file.
3.  **Caching**: Keeping loaded fonts available.

## 4. Text Rendering Flow

1.  **Layout**: The application (e.g., `core-term`) calculates the position of characters (Text Layout).
2.  **Composition**: The application creates a `Surface` that represents the text.
    *   This is conceptually a composition of many small glyph surfaces, translated (`Warp`) to their correct positions.
    *   In practice, for a terminal grid, this can be optimized as a `GridSurface` that looks up the correct glyph Surface for each cell coordinate.
3.  **Materialization**: The Engine samples this composite surface.
    *   The glyphs are evaluated mathematically (Loop-Blinn) at sample time.
    *   There is no pre-rasterization of glyphs into a texture atlas (unless explicitly cached for performance via a `Bake` combinator, which is just another Surface).

## 5. Integration

*   **Platform Specifics**: Font discovery logic remains platform-specific (X11/FontConfig, Cocoa, etc.) but produces platform-agnostic `Surface` objects.
*   **Attributes**: Bold/Italic are applied as `Warp` (Shear for Italic) or parameter changes (Stroke width for Bold) on the glyph Surface.

This design aligns with the "Everything is a Surface" philosophy of PixelFlow 1.0.
