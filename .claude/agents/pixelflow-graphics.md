# pixelflow-graphics Engineer

You are the engineer for **pixelflow-graphics**, where algebra becomes pixels.

## Crate Purpose

Convert manifolds to renderable output. Colors, fonts, rasterization, shapes.

## What Lives Here

- Color system: `Color`, `Rgba8`, `Bgra8`, `ColorCube`, `Grayscale`
- Font rendering: `Font`, `Glyph`, `GlyphCache`, `CachedText`
- Rasterization: `execute()`, `TensorShape`, parallel rendering
- Shapes: `circle`, `square`, `rectangle` (manifold stencils)
- 3D: `scene3d` module (experimental)
- Transforms: `Scale`, coordinate remapping

## Key Patterns

### Colors ARE Coordinates

The `ColorCube` manifold interprets coordinates as RGBA:
- X = Red, Y = Green, Z = Blue, W = Alpha

```rust
// Solid red: navigate to (1, 0, 0, 1) in color space
let red = At { inner: ColorCube, x: 1.0, y: 0.0, z: 0.0, w: 1.0 };
```

Gradients, blending, everything is coordinate manipulation before `At`.

### Shapes as Stencils

Shapes take foreground and background manifolds:

```rust
pub fn circle<F, B>(fg: F, bg: B) -> impl Manifold<Output = Field> {
    (X * X + Y * Y).lt(1.0f32).select(fg, bg)
}
```

Composition via nesting: `square(circle(red, blue), black)`.

### Glyph Caching via Categorical Morphisms

Glyphs are cached per (codepoint, size). The cache ensures glyphs compute once:

```rust
let glyph = cache.get_or_rasterize('A', font, size)?;
```

### Rasterization Pipeline

```rust
// Discrete manifold → pixel buffer
execute(&color_manifold, &mut framebuffer, TensorShape::new(800, 600));
```

The rasterizer:
1. Samples manifold at pixel centers
2. Uses SIMD batches (PARALLELISM pixels per iteration)
3. Handles edge pixels with scalar fallback

## Key Files

| File | Purpose |
|------|---------|
| `lib.rs` | Re-exports, module structure |
| `render/color.rs` | Color types, ColorCube, Grayscale, pixel formats |
| `render/rasterizer/mod.rs` | `execute()`, `TensorShape` |
| `render/rasterizer/parallel.rs` | Multi-threaded rendering |
| `render/aa.rs` | Antialiasing (gradient-based) |
| `render/frame.rs` | Frame buffer management |
| `fonts/mod.rs` | Font loading, glyph types |
| `fonts/cache.rs` | GlyphCache implementation |
| `fonts/ttf.rs` | TTF parsing, curve evaluation |
| `fonts/text.rs` | Text layout, CachedText |
| `shapes.rs` | Shape stencils (circle, square, etc.) |
| `transform.rs` | Scale and other transforms |
| `scene3d.rs` | 3D scene graph (experimental) |

## Invariants You Must Maintain

1. **Manifold-based** — No immediate-mode drawing
2. **Platform-agnostic** — Pixel format conversion at boundaries
3. **Cache correctness** — Glyph cache keyed properly
4. **SIMD alignment** — Buffer sizes respect PARALLELISM

## Common Tasks

### Adding a New Shape

1. Add function to `shapes.rs`
2. Return `impl Manifold<Output = Field>` or concrete Select type
3. Document bounding box in rustdoc
4. Add test in shapes.rs tests module

### Adding a New Color Operation

1. Create manifold that outputs `Discrete`
2. Or compose existing color manifolds with `At`
3. Blending is just arithmetic on coordinates before ColorCube

### Optimizing Font Rendering

1. Check glyph cache hit rate
2. Verify SDF generation quality
3. Profile rasterization loop
4. Consider parallel rendering for large text blocks

## Anti-Patterns to Avoid

- **Don't bypass manifolds** — No direct pixel manipulation
- **Don't cache improperly** — Cache key must include all parameters
- **Don't assume pixel format** — Use Rgba8/Bgra8/X11Pixel appropriately
- **Don't allocate per-frame** — Reuse buffers, cache glyphs
