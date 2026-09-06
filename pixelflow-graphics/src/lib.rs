//! # PixelFlow Graphics
//!
//! Turns a **scene into pixels**: colours, fonts, and the compiled program
//! that puts one in a frame.
//!
//! ```text
//!   four channel kernels in [0, 1]        Frame<P>
//!            │                               ▲
//!            │  compile_packed_for::<P>      │
//!            ▼                               │
//!     PackedManifold  ──bind(buffers)──▶  PackedFrame
//!                                            │
//!                             Scene::render: one collapse call per stripe
//! ```
//!
//! ## Colour is four kernels and a byte order
//!
//! A colour output is four `Kernel`s in `[0, 1]` — red, green, blue, alpha —
//! and the pack that turns them into a `u32` pixel is **inside the compiled
//! kernel**: `×255`, clamp, truncate, shift, or, all of it IR. The byte order
//! comes from the frame's own pixel format ([`Pixel::packed_shifts`]), so the
//! format a kernel packs for is the format the frame stores, by construction.
//!
//! ```ignore
//! use pixelflow_graphics::render::scene::{compile_packed_for, Scene};
//! use pixelflow_graphics::scene3d::Rgba;
//! use pixelflow_graphics::Rgba8;
//!
//! let color = Rgba::from([red, green, blue, Kernel::constant(1.0)]);
//! let scene = Scene::Packed(compile_packed_for::<Rgba8>(&color, [w, h]).bind(&[]));
//! scene.render(&mut frame, threads);
//! ```
//!
//! [`Color`] and [`NamedColor`] are the semantic *input* side — ANSI, indexed
//! palette, true colour — and `Color::to_f32_rgba` is the bridge into the four
//! channels. They are data, not manifolds.
//!
//! ## Fonts
//!
//! Vector glyphs (TTF) become coverage kernels. [`GlyphAtlas`](fonts::GlyphAtlas)
//! bakes one glyph per tile into a single gatherable `f32` buffer — one
//! `Lattice::bake` per glyph, cached — which the cell-grid scene kernel
//! gathers from. `fonts::text` composes an uncached analytical text kernel;
//! [`CachedText`] composes already-baked glyph samplers.
//!
//! ## Scenes
//!
//! [`render::scene::compile_cell_grid_for`] is the terminal's: a geometry that
//! denotes four channel kernels over a cell buffer and a coverage atlas.
//! [`scene3d`] is analytic 3D: `Ray::through_screen`, `Sphere::hit`,
//! `Hit::select`, `checker`, `sky` — plain functions producing `Kernel`s,
//! every derivative from `Kernel::dx()`/`dy()`.
//!
//! ## Pixel formats
//!
//! | Type | Layout | Usage |
//! |------|--------|-------|
//! | [`Rgba8`] | `[R, G, B, A]` | macOS Cocoa, Web |
//! | [`Bgra8`] | `[B, G, R, A]` | Linux X11 |
//!
//! [`PlatformPixel`] is this platform's choice, and the single statement of
//! which byte order every packed kernel here packs for.

pub mod fonts;
pub mod render;
pub mod scene3d;

// Re-export fonts (user-facing types only)
pub use fonts::{CachedGlyph, CachedText, Font, GlyphCache};

// Re-export render
pub use render::color::{
    AttrFlags, Bgra8, CocoaPixel, Color, NamedColor, Pixel, PlatformPixel, Rgba8, WebPixel,
    X11Pixel,
};
pub use render::frame::Frame;

// Re-export core types for convenience.
//
// The per-batch `Manifold` *trait* is deliberately not among them: since
// S4b-1 `Manifold` names the compiled object (`pixelflow_core::Manifold`),
// and a second re-export of the trait under that name at this crate's root
// would make `pixelflow_graphics::Manifold` and `pixelflow_core::Manifold`
// two different things. Nothing in the tree imported it from here.
pub use pixelflow_core::{ManifoldExt, Map, W, X, Y, Z};
