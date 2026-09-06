# PixelFlow Graphics

`pixelflow-graphics` turns PixelFlow programs into pixels. It contains color and pixel types,
TTF outline compilation, glyph and text caching, framebuffer rasterization, and experimental
2D/3D scene components.

The crate currently spans two representation layers:

- The font pipeline is arena-native: outlines become fused `Kernel` values, symbolic
  derivatives provide coverage antialiasing, and `Lattice::bake` compiles and caches the
  result.
- Existing color, cached-image, and 3D composition code uses `Manifold` combinators. This is
  still a supported execution substrate while the workspace completes its JIT-first
  migration.

That boundary is intentional and visible; consumers should not manipulate `ExprArena`
directly.

## Pipeline

```text
             continuous programs
         ┌──────────┴───────────┐
         │                      │
   coverage Kernel       color/scene Manifold
         │                      │
    JIT + Lattice          SIMD evaluation
         │                      │
         └──────────┬───────────┘
                    ▼
             sampled pixels
                    ▼
       Frame<Rgba8> / Frame<Bgra8>
```

The principal modules are:

| Module | Role |
|---|---|
| `fonts` | TTF parsing, glyph coverage kernels, text layout, and baked glyph caches |
| `render` | Semantic colors, pixel formats, frames, and rasterization |
| `scene3d` | Ray/surface/material constructors: a 3-D scene as four channel kernels |
| `scene3d_surface` | The same scenes over `Jet3` manifolds — the legacy tier, on its way out |
| `shapes`, `transform`, `image`, `mesh` | Additional graphics primitives and composition utilities |

## Fonts: TTF outlines as kernels

`Font::parse` reads the supported TrueType tables. A glyph is compiled directly into one
coverage `Kernel`: line and quadratic-curve crossing terms are combined under the non-zero
winding rule, bounded, scaled, and translated in the arena.

```rust
use pixelflow_graphics::Font;

let font = Font::parse(font_bytes).expect("invalid or unsupported TTF");
let glyph = font
    .glyph_kernel_scaled('A', 32.0)
    .expect("font has no glyph for A");
```

Coverage antialiasing is intrinsic to these kernels. Curve leaves construct a
gradient-normalized ramp using `DX`/`DY`; those become `Dwrt` expressions and are resolved
when the kernel compiles. Coordinate warps therefore carry the ramp width into screen space
without a separate antialiasing wrapper or jet-valued JIT ABI.

### Glyph cache

Analytically evaluating every outline at every sample is unnecessary after a glyph is stable.
`GlyphCache` quantizes the size and display-density keys, bakes the fused kernel once, and
stores `f32` coverage. Read-back is bilinear so fractional placement interpolates cached
coverage.

```rust
use pixelflow_graphics::{Font, GlyphCache};

let font = Font::parse(font_bytes).expect("invalid or unsupported TTF");
let mut cache = GlyphCache::new();
cache.warm_ascii(&font, 16.0, 2.0);

let glyph = cache
    .get(&font, 'A', 16.0, 2.0)
    .expect("font has no glyph for A");
```

`fonts::text` lays a string out as one analytical `Kernel`. `CachedText` instead composes
cached glyph samplers with advances and kerning. Both represent coverage in `[0, 1]`; mapping
coverage to foreground/background color is a separate operation.

Supported outline coverage currently targets TrueType quadratic glyphs, cmap formats 4 and
12, and horizontal `kern` format 0. This is not a general shaping engine.

## Colors and frames

The render module separates semantic color from platform pixel layout:

- `Color` and `NamedColor` represent ANSI, indexed, and RGB choices.
- `ColorCube`, `Grayscale`, and related manifolds map continuous values to packed color.
- `Rgba8` and `Bgra8` define framebuffer byte layout.
- `Frame<P>` owns a row-major pixel buffer.

`render::rasterize` pulls a color manifold at pixel coordinates into a frame and can divide
the work across threads:

```rust
use pixelflow_graphics::render::{rasterize, Frame, Rgba8};

let mut frame = Frame::<Rgba8>::new(800, 600);
rasterize(&color_manifold, &mut frame, 4);
```

The exact SIMD width and platform pixel alias are backend details. Consumers should use the
pixel types rather than assume a byte order or lane count.

## Ray tracing: a scene is four channel kernels

`scene3d` builds a 3-D scene as `Kernel` values of the screen coordinate, in
three stages:

1. `Ray::through_screen` turns the pixel coordinate into a unit direction. The
   observer is fixed at the origin, so a ray *is* a direction and a reflected
   ray is another direction from the same origin.
2. `Sphere::hit` / `Plane::hit` solve for `t` in closed form — a quadratic and
   a division; there is no march — and return the hit point, the outward
   normal, and the mask saying whether the ray met the surface at all.
3. A material (`checker`, `sky`, `Rgba::opaque_gray`) is four channel kernels
   in `[0, 1]`; `Hit::select` chooses material or background per channel, and
   nesting those selects is occlusion.

Antialiasing is `Kernel::dx()`/`dy()`: `Hit::footprint` is the screen-space
size of a pixel on the surface, differentiated symbolically through the screen
mapping, the intersection and the reflection. There is no jet domain and no
curvature heuristic.

The four channels are compiled together (`render::scene::compile_packed_for`),
so the geometry they share is emitted once — the "mullet" saving the jet tier
got from carrying colour as an opaque `Discrete` is now the compiler's, and is
pinned by `scene3d_test::four_channels_share_one_geometry`.

`scene3d_surface` is the jet-manifold version of the same scenes, kept while
`subdivision_autodiff` (whose geometry has no kernel form) and the S3 gate's
"before" side still need it. No new consumer.

## Materialization boundaries

PixelFlow distinguishes composition from storage:

- Compose analytical font work as `Kernel` values, then bake at a glyph or text-cache boundary.
- Compose cached glyphs, images, and platform-backed values as ordinary manifolds.
- Rasterize the final color manifold into a `Frame<P>` at the application boundary.

Intermediate storage is allowed when it is the requested representation—a glyph cache or a
framebuffer—not hidden as an accidental compiler fallback.

## Status and validation

Font kernel goldens compare JIT execution with the interpreter on the same arena. Additional
tests cover antialiasing behavior, curve parsing, cache coordinates, color conversion, and
rendering. Performance claims are intentionally omitted from this README; run the benchmark
targets on the hardware and revision being evaluated.

```bash
cargo test -p pixelflow-graphics
cargo bench -p pixelflow-graphics
cargo bench -p pixelflow-graphics --bench font_rendering
cargo bench -p pixelflow-graphics --bench kernel_bench
```

The current compiler migration is tracked in
[`2026-07-20-kernel-unification.md`](../docs/plans/2026-07-20-kernel-unification.md).

## License

[Apache License 2.0](../LICENSE.md)
