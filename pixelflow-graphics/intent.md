# pixelflow-graphics

**Is.** The programs that put a scene in a frame, written in the language and
adding nothing to it ("No colour in the language"): a colour output is four
channel kernels in `[0, 1]` with the pack to a `u32` inside the kernel, a glyph
is a fold of closed-form piece areas (DENOTATION §3), a frame is the lattice
the kernel is collapsed over (§5). `fonts/`, `render/` have their own intent.

## Owns
- Byte order: `Pixel::packed_shifts` is the one statement of which byte holds
  which channel; `PlatformPixel` is the platform's choice.
- The pack (`×255`, clamp, shift, or) as IR over four channels
  (`render/packed.rs`): a choice between colours is one select on words.
- Colour as input data (`Color`) and the buffer `Frame<P>`; glyph coverage
  (`fonts/`); analytic 3-D scenes as `Kernel`s, no march (`scene3d.rs`).
- Rendering a scene: one collapse call per stripe, stripes across threads,
  stores landing in the frame's own memory (`render/scene.rs`).

## Never
- No terminal logic: Cargo.toml names neither `core-term` nor
  `pixelflow-runtime`; no PTY, escape-sequence or scrollback item is in `src/`.
- No lane, width, intrinsic or ISA: no `core::arch`, `target_feature` or
  `jit_vector_bytes` outside tests; `pixelflow-codegen` is a dev-dependency.
- No colour is a `Manifold` and no scene has a per-batch path: `Scene` is
  `Packed` or `CellGrid`, both rendered by `collapse_rows`.
- No caller chooses a byte order: `compile_packed_for::<P>` reads it from `P`,
  and `render` asserts the frame's `P` packs the shifts the kernel compiled for.

## Contract
`src/lib.rs` declares `fonts`, `render`, `scene3d`; re-exports `Font`, `GlyphCache`,
`CachedGlyph`, `CachedText`, `Color`, `NamedColor`, `AttrFlags`, `Pixel`, `Rgba8`,
`Bgra8`, `PlatformPixel`, `Frame`. `render::scene`: `Scene`, `compile_packed_for`,
`compile_cell_grid_for`, `constant_scene_for`. `scene3d`: `Rgba`, `Ray`, `Hit`,
`Sphere`, `Plane`, `sky`, `checker`.

## Enforced by
- Cargo: `core-term` and `pixelflow-runtime` depend on it; a dependency back is a cycle.
- desloppify `no-terminal-logic-in-pixelflow`, `simd-is-codegens` and
  `per-frame-allocation`, each covering `pixelflow-graphics/**`.
- `render::color::tests::packed_shifts_agree_with_pixel_from_rgba`;
  `tests/rendering_contract.rs`; `tests/scene3d_test.rs` and
  `render::packed::tests::four_channels_share_one_geometry` (CI-named in
  `.github/workflows/rust.yaml`); `tests/freetype_oracle.rs`.

## Behind
- Depends on `pixelflow-ir` directly, which D16 forbids: `fonts/loop_blinn.rs`
  builds `ExprArena`, `render/packed.rs` names `BufferDecl`; no CI job checks
  D16 (exprarena-on-dag Stage D).
- `PackedFrame` holds a `BoundManifold`; bind is not a concept (DENOTATION §11,
  kernels-evaluate-kernels §2). `GlyphAtlas`, `GlyphCache`, `CachedGlyph` and
  `CachedText` are live; D10 deletes them after C2 measures (the-language-is-kernel, O4).
- Terminal vocabulary lives here (ANSI palette, `AttrFlags`, `render/cell_grid.rs`); D13 moves only core's half.
- `Scene::render` spawns workers per call (`render_bands`); the stripe schedule
  is the CPU backend's (kernels-evaluate-kernels §3).

## Decided in
`docs/archive/plans/2026-09-06-kernel-with-a-lattice.md` (S3b, S4a), `docs/language/2026-09-25-the-language-is-kernel.md`
(D10, D13, D16), `docs/language/2026-10-15-kernels-evaluate-kernels.md` §2–§3.
