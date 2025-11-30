# PixelFlow Render

High-performance, software-based rendering built on `pixelflow-core`.

## Architecture

Everything is a lazy, infinite **Surface** until materialization:

- `Frame<P>` is both a target (write into) AND a Surface (read from)
- Colors (`Rgba`, `Bgra`) are constant Surfaces
- Compose with `Over`, `Offset`, `Skew`, `Max`, `Baked`, etc.
- Materialize via `render()` or `execute()`

## Example

```rust
use pixelflow_render::{Frame, Rgba, render, font};
use pixelflow_core::dsl::MaskExt;

// Get a glyph mask (Surface<u8>)
let glyph = font().glyph('A', 24.0).unwrap();

// Compose: mask blends fg over bg
let fg = Rgba::new(255, 255, 255, 255);
let bg = Rgba::new(0, 0, 0, 255);
let surface = glyph.over::<Rgba>(fg, bg);

// Materialize into a frame
let mut frame = Frame::<Rgba>::new(800, 600);
render(surface, &mut frame);
```

## Modules

- **`color`**: Semantic colors (`Color`, `NamedColor`) and pixel formats (`Rgba`, `Bgra`)
- **`frame`**: `Frame<P>` buffer, implements `Surface<P>`
- **`glyph`**: Embedded font access via `font()`
- **`rasterizer`**: `render()`, `execute()`, `render_u32()`

## Pixel Formats

| Platform | Format | Type Alias |
|----------|--------|------------|
| X11      | BGRA   | `X11Pixel` |
| Cocoa    | RGBA   | `CocoaPixel` |
| Web      | RGBA   | `WebPixel` |
