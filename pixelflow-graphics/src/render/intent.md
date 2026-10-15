# pixelflow-graphics/src/render

**Is.** The frame (DENOTATION §5): the lattice a colour kernel is collapsed
over, and that colour output — four channel kernels in `[0, 1]` with the pack
to a `u32` inside, compiled together at the frame's shape. It adds nothing to
the language ("No colour in the language").

## Owns
- Byte order: `Pixel::packed_shifts` says which byte holds which channel;
  `PlatformPixel` is the platform's format.
- The pack as IR (`packed_kernel`): `×255`, clamp, truncate, shift, or; a
  choice of colours is one `Bits::select` on words. `pack_scalar` is its one
  scalar restatement (the cell grid's border).
- Four channels as one kernel: shared geometry is emitted once (CLAUDE.md).
- `Frame<P>` and the stripe schedule filling it (`render_bands`: one
  `collapse_rows` per stripe, stores in the frame's memory, FTZ/DAZ).
- Colour input data (`Color`, `Rgba8`, `Bgra8`); the cell grid as a packed
  program (metric a `UniformBlock`, extent the compile key).

## Never
- No colour is a `Manifold` and no scene has a per-batch path: `Scene` is
  `Packed` or `CellGrid`, both rendered through `render_bands`.
- No caller supplies a byte order: `render_bands` asserts `P::packed_shifts()`
  equals the shifts the kernel packed for.
- No lane, width or ISA is named (`core::arch`, `target_feature`,
  `jit_vector_bytes`); `pixelflow-codegen` appears in tests only.

## Contract
`mod.rs` declares `cell_grid` and `packed` (x86-64, aarch64), `color`, `frame`,
`pixel`, `renderer` (own intent), `scene`. `packed`: `PackedManifold`,
`PackedFrame`. `cell_grid`: `CellGridPackedManifold`, `CellGridPackedFrame`.
`scene`: `Scene`, `compile_packed_for`, `compile_cell_grid_for`, `constant_scene_for`.

## Enforced by
- Types: `Pixel::packed_shifts` is an `Option`, so `compile_packed_for` refuses
  a format with no packed form; `PackedManifold::compile` asserts the shifts
  permute the four byte lanes.
- Tests: `scene.rs` `mismatched_frame_format_is_refused`; `color.rs`
  `packed_shifts_agree_with_pixel_from_rgba`; `tests/rendering_contract.rs`;
  `packed::tests::four_channels_share_one_geometry` (four channels emit under 2x the
  code of one; named in `.github/workflows/rust.yaml`).
- desloppify `simd-is-codegens`, `per-frame-allocation`, `no-terminal-logic-in-pixelflow`.

## Behind
- `packed.rs` imports `pixelflow_ir` (`BufferDecl`, `UniformDecl`, `Bits`,
  `Kernel`), which D16 reserves to compiler crates; no CI job checks it
  (the-language-is-kernel D16; exprarena-on-dag Stage D).
- `PackedFrame` holds a `BoundManifold`; bind is not a concept (DENOTATION §11).
- `frame.rs` still calls `Frame` "both a target (write into via execute) AND a
  Surface (read from)"; the Surface lane went in kernel-with-a-lattice S4a.
- `color.rs` holds the ANSI palette (`NamedColor`, the 256-colour cube) and
  `AttrFlags`, terminal vocabulary used by `core-term`, against "no terminal
  logic in PixelFlow"; `cell_grid.rs` names the terminal as its production
  caller (D13 moves the grid up).

## Decided in
`docs/plans/2026-09-06-kernel-with-a-lattice.md` (S2, S3b, S4a),
`docs/plans/2026-09-25-the-language-is-kernel.md` (D13, D16),
`docs/plans/2026-10-15-kernels-evaluate-kernels.md` §2–§3.
