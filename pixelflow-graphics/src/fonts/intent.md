# pixelflow-graphics/src/fonts

**Is.** A glyph as a fold: coverage is `min(|Σ_p σ_p·A_p|, 1)` over monotone
arcs, each piece's area `A_p` of the pixel under ink in closed form (DENOTATION
§3; the integral is deferred, not present). The module is the host that gets
there: TrueType bytes to geometry in the drawn frame, to pieces, to a kernel,
and to the baked tiles a frame reads today.

## Owns
- Font bytes and tables (`Font::parse`: cmap, glyf, loca, hmtx, kern), and
  every affine map in a font (em scale, screen flip, component placement, pen
  position), applied to control points on the host so the kernel is built in
  the frame it is evaluated in.
- The monotone split (`MonotoneQuad::split`), which makes the kernel's
  `max(step, 0)` certificate the glyph's own geometry; the piece row; the snap
  to exactly 0 or 1.

## Never
- No coverage is a ramp on a distance, no term an integral, and no coverages
  are added: a run is one glyph, signed areas summed and clamped once.
- No scale is applied to a finished kernel: the only `.at` on a glyph is the
  `+½` pixel-center translation at a bake (correct under translation only).
- No lane, width, intrinsic or ISA; no terminal vocabulary in the formula.

## Contract
`mod.rs` declares `atlas`, `cache`, `loader`, `loop_blinn`, `outline`, `text`,
`ttf` and private `monotone`; re-exports `Font`, the `FontSource` family,
`Outline`, `Glyph`, `Support`, `FontPrograms`, `GlyphRows`, `text`, `GlyphAtlas`,
`GlyphCache`, `CachedGlyph`, `CachedText`. `loop_blinn::{glyph, run}` build a
`Glyph`; `FontPrograms::draw` is the table-free path (a program per piece count).

## Enforced by
- Types: `MonotoneQuad::split` is its one constructor; `GlyphRows::of` the
  only way to a `GlyphRows`; `Glyph::kernel` the one exit.
- `tests/freetype_oracle.rs`, CI "Check glyph coverage against an external
  rasterizer" (`.github/workflows/rust.yaml`): FreeType shares no code with us.
- `tests/glyph_exact_area.rs`, `tests/glyph_area_edge_cases.rs`: atlas tiles and
  `FontPrograms` against an `f64` exact area.
- `tests/run_is_a_glyph.rs`; the proptests in `monotone.rs`; desloppify
  `no-terminal-logic-in-pixelflow` and `simd-is-codegens` (`pixelflow-*/**`).

## Behind
- `GlyphCache`'s key is `(codepoint, size bucket, density)` with no font
  identity, though `get` takes a `Font`: two fonts share entries. `GlyphAtlas`
  documents its one-font limit; the cache does not.
- The frame draws from `GlyphAtlas`; `CachedGlyph`, `CachedText` and
  `BilinearSampler` read memory. D10 replaces them with `FontPrograms` after
  C2's measurement (the-language-is-kernel, O4); `FontPrograms` has no frame
  caller (`core-term` and `render/cell_grid.rs` use the atlas).
  `CachedText::kernel` sums coverages, so overlaps exceed 1 (BACKLOG C3).
- `loop_blinn::glyph`/`run` fold over a `DiscreteManifold` piece table and round
  counts to powers of two; `loop_blinn.rs` builds `ExprArena`, against D16.
- `mod.rs` documents `render::color::Grayscale`, which does not exist, and calls
  `text()` "a sum of advance-translated glyph kernels"; it is one `run`.

## Decided in
`docs/archive/plans/2026-09-23-a-glyph-is-a-formula.md`, `docs/archive/plans/2026-09-09-a-run-is-a-glyph.md`,
`docs/language/2026-09-25-the-language-is-kernel.md` (§1.7, C1, C2, D10, D16),
`docs/language/2026-10-15-kernels-evaluate-kernels.md` §2.
