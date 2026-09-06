//! # The cell grid as a packed program
//!
//! [`pixelflow_core::CellGridGeometry`] denotes four channel kernels over two
//! bound buffers — cell data and a coverage atlas — and knows nothing about
//! bytes or pixel formats. This is the instance that turns them into a frame:
//! the four channels packed to one `u32` pixel in IR ([`PackedProgram`]) for a
//! given byte order, with the two buffers bound per frame.
//!
//! The terminal's text grid is the production caller; sprite grids and
//! tile-based scenes are the same primitive.

use std::sync::Arc;

use pixelflow_core::{CellGridBuffers, CellGridGeometry, CellGridKernels};

use crate::render::packed::{PackedFrame, PackedProgram};

/// A cell grid compiled for one geometry and one pixel byte order.
///
/// Compile once per geometry ([`CellGridPackedProgram::compile`]) — a resize
/// or a density change is a recompile, whose cost is independent of the grid's
/// dimensions — and stamp per-frame data into it with
/// [`CellGridPackedProgram::frame`].
pub struct CellGridPackedProgram {
    geom: CellGridGeometry,
    buffers: CellGridBuffers,
    packed: PackedProgram,
}

impl CellGridPackedProgram {
    /// Compose and JIT-compile the packed program for `geom`, with
    /// `default_bg` (linear RGBA) painted outside the grid and `shifts[c]`
    /// giving the bit position of channel `c` in `(r, g, b, a)` order —
    /// `[0, 8, 16, 24]` for `Rgba8`, `[16, 8, 0, 24]` for `Bgra8`.
    ///
    /// # Panics
    ///
    /// Panics on a degenerate geometry, on shifts that are not a permutation
    /// of the four byte lanes, if the composed kernel's buffer reads do not
    /// merge to exactly one cell slot and one atlas slot, or if compilation
    /// fails.
    #[must_use]
    pub fn compile(geom: CellGridGeometry, default_bg: [f32; 4], shifts: [u32; 4]) -> Self {
        let CellGridKernels { channels, buffers } = geom.channel_kernels(default_bg);
        let packed = PackedProgram::compile(&channels, shifts, [geom.frame_w, geom.frame_h]);
        // Identity-merge across the four channels' splices is load-bearing:
        // all cell reads and atlas taps must land in the same two slots a
        // frame binds. A third slot means splice stopped merging — refuse.
        let slots = packed.buffers();
        assert_eq!(
            slots.iter().filter(|d| d.id == buffers.cells).count(),
            1,
            "packed cell-grid kernel did not merge its cell reads to one slot"
        );
        assert_eq!(
            slots.iter().filter(|d| d.id == buffers.atlas).count(),
            1,
            "packed cell-grid kernel did not merge its atlas taps to one slot"
        );
        Self {
            geom,
            buffers,
            packed,
        }
    }

    /// The geometry this program was compiled for.
    #[must_use]
    pub fn geometry(&self) -> &CellGridGeometry {
        &self.geom
    }

    /// The byte lanes this program's kernel packs into.
    #[must_use]
    pub fn shifts(&self) -> [u32; 4] {
        self.packed.shifts()
    }

    /// The compiled kernel's emitted bytes (research/profiling harness).
    #[must_use]
    pub fn code_bytes(&self) -> &[u8] {
        self.packed.code_bytes()
    }

    /// Bind one frame's data. `cells` is `CELL_STRIDE` `f32`s per cell,
    /// row-major; `atlas` is the coverage atlas, row-major texels. Both are
    /// `Arc`s so a frame in flight keeps its data alive while the caller
    /// prepares the next one.
    ///
    /// # Panics
    ///
    /// Panics if either buffer's length does not match the geometry.
    #[must_use]
    pub fn frame(&self, cells: Arc<Vec<f32>>, atlas: Arc<Vec<f32>>) -> PackedFrame {
        self.packed
            .bind(&[(self.buffers.cells, cells), (self.buffers.atlas, atlas)])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::packed::packed_kernel;
    use pixelflow_core::{CellGridFrame, CellGridProgram, PlaneRegion};
    use pixelflow_ir::arena::BufferIdentity;
    use pixelflow_ir::{ExprArena, ExprId};

    /// A 2×1 grid over a 2-tile atlas: tile A solid coverage 1, tile B a
    /// half-coverage field. Colors chosen so every channel distinguishes
    /// fg, bg, and the default background.
    fn tiny_scene() -> (CellGridProgram, Arc<Vec<f32>>, Arc<Vec<f32>>) {
        let (aw, ah, slot) = (12usize, 6usize, 6usize);
        let mut atlas = vec![0.0f32; aw * ah];
        for r in 0..4 {
            for c in 0..4 {
                atlas[(1 + r) * aw + 1 + c] = 1.0; // tile 0: solid
                atlas[(1 + r) * aw + slot + 1 + c] = 0.5; // tile 1: half
            }
        }
        let geom = CellGridGeometry {
            cols: 2,
            rows: 1,
            cell_w: 4.0,
            cell_h: 4.0,
            density: 1.0,
            atlas_width: aw as u32,
            atlas_height: ah as u32,
            tile_w: 4,
            tile_h: 4,
            frame_w: 12,
            frame_h: 6,
        };
        let program = CellGridProgram::compile(geom, [0.9, 0.8, 0.7, 0.6]);
        // Cell 0 → tile 0, red-on-black; cell 1 → tile 1, white-on-blue.
        let cells = vec![
            1.0, 1.0, /* uv */ 1.0, 0.0, 0.0, 1.0, /* fg */ 0.0, 0.0, 0.0,
            1.0, /* bg */
            7.0, 1.0, /* uv */ 1.0, 1.0, 1.0, 1.0, /* fg */ 0.0, 0.0, 1.0,
            1.0, /* bg */
        ];
        (program, Arc::new(cells), Arc::new(atlas))
    }

    /// Bake one channel over pixel centers and unpad to a dense w×h plane —
    /// the four-channel oracle the pack is checked against.
    fn plane(frame: &CellGridFrame, channel: usize, w: usize, h: usize) -> Vec<f32> {
        let stride = PackedFrame::padded_width(w);
        let mut padded = vec![0.0f32; h * stride];
        frame.bake_channel_rows(
            channel,
            PlaneRegion {
                width: w,
                y0: 0,
                rows: h,
            },
            &mut padded,
        );
        let mut dense = Vec::with_capacity(w * h);
        for row in 0..h {
            dense.extend_from_slice(&padded[row * stride..row * stride + w]);
        }
        dense
    }

    fn reachable_nodes(arena: &ExprArena, root: ExprId) -> usize {
        let mut seen = vec![false; arena.nodes_raw().len()];
        let mut stack = vec![root];
        let mut count = 0;
        while let Some(id) = stack.pop() {
            if std::mem::replace(&mut seen[id.0 as usize], true) {
                continue;
            }
            count += 1;
            stack.extend(arena.children(id));
        }
        count
    }

    /// Research harness: dump the packed kernel's machine code and hot-loop
    /// it in one process, so a sampling profiler's addresses correlate to
    /// the dumped bytes exactly. `PIXELFLOW_CODE_DUMP=<path>` receives the
    /// raw bytes; the base pointer prints to stdout.
    #[test]
    #[ignore = "manual profiling harness"]
    fn profile_dump_packed_kernel() {
        let geom = CellGridGeometry {
            cols: 213,
            rows: 66,
            cell_w: 12.0,
            cell_h: 24.0,
            density: 1.0,
            atlas_width: 64,
            atlas_height: 32,
            tile_w: 12,
            tile_h: 24,
            frame_w: 2560,
            frame_h: 1584,
        };
        let program = CellGridPackedProgram::compile(geom, [0.1, 0.1, 0.1, 1.0], [0, 8, 16, 24]);
        let code = program.code_bytes();
        println!(
            "CODE_BASE=0x{:x} CODE_LEN={}",
            code.as_ptr() as usize,
            code.len()
        );
        if let Ok(path) = std::env::var("PIXELFLOW_CODE_DUMP") {
            std::fs::write(&path, code).expect("dump write failed");
        }
        let mut atlas = vec![0.0f32; geom.atlas_len()];
        for (i, t) in atlas.iter_mut().enumerate() {
            *t = ((i * 7) % 11) as f32 / 10.0;
        }
        let mut cells = vec![0.0f32; geom.cells_len()];
        for (i, c) in cells.iter_mut().enumerate() {
            *c = ((i * 13) % 17) as f32 / 16.0;
        }
        let frame = program.frame(Arc::new(cells), Arc::new(atlas));
        let (w, h) = (2560usize, 1584usize);
        let stride = PackedFrame::padded_width(w);
        let mut out = vec![0u32; stride * h];
        for _ in 0..150 {
            frame.bake_packed_rows(
                PlaneRegion {
                    width: w,
                    y0: 0,
                    rows: h,
                },
                &mut out,
            );
            std::hint::black_box(&out);
        }
    }

    /// `Rgba8`'s byte lanes: little-endian `[r, g, b, a]`.
    const RGBA_SHIFTS: [u32; 4] = [0, 8, 16, 24];
    /// `Bgra8`'s byte lanes: little-endian `[b, g, r, a]`.
    const BGRA_SHIFTS: [u32; 4] = [16, 8, 0, 24];

    /// The scalar pack the kernel must be bit-exact with:
    /// `Pixel::from_rgba`'s per-channel `(x·255).clamp(0, 255) as u8`,
    /// composed with the same shifts.
    fn pack_expected(rgba: [f32; 4], shifts: [u32; 4]) -> u32 {
        rgba.iter()
            .zip(shifts)
            .map(|(&x, s)| u32::from((x * 255.0).clamp(0.0, 255.0) as u8) << s)
            .fold(0, |acc, lane| acc | lane)
    }

    /// Bake packed pixels over pixel centers and unpad to a dense w×h plane.
    fn packed_plane(frame: &PackedFrame, w: usize, h: usize) -> Vec<u32> {
        let stride = PackedFrame::padded_width(w);
        let mut padded = vec![0u32; h * stride];
        frame.bake_packed_rows(
            PlaneRegion {
                width: w,
                y0: 0,
                rows: h,
            },
            &mut padded,
        );
        let mut dense = Vec::with_capacity(w * h);
        for row in 0..h {
            dense.extend_from_slice(&padded[row * stride..row * stride + w]);
        }
        dense
    }

    #[test]
    fn packed_bake_is_bit_exact_with_channel_bakes_under_both_byte_orders() {
        // 12×6 over tiny_scene's 8×4 grid: the right and bottom margins are
        // off-grid, so default_bg flows through the pack too. THE invariant:
        // for every pixel, the packed bake equals the four channel bakes
        // composed with the scalar pack — exact u32 equality, no epsilon.
        let (program, cells, atlas) = tiny_scene();
        let frame = program.frame(cells.clone(), atlas.clone());
        let (w, h) = (12, 6);
        let planes: [Vec<f32>; 4] = core::array::from_fn(|c| plane(&frame, c, w, h));
        for shifts in [RGBA_SHIFTS, BGRA_SHIFTS] {
            let packed =
                CellGridPackedProgram::compile(*program.geometry(), [0.9, 0.8, 0.7, 0.6], shifts);
            let got = packed_plane(&packed.frame(cells.clone(), atlas.clone()), w, h);
            for i in 0..w * h {
                let expected = pack_expected(
                    [planes[0][i], planes[1][i], planes[2][i], planes[3][i]],
                    shifts,
                );
                assert_eq!(
                    got[i], expected,
                    "pixel {i} under shifts {shifts:?}: {:#010x} != {:#010x}",
                    got[i], expected
                );
            }
        }
    }

    #[test]
    fn packed_bake_parity_holds_on_an_asymmetric_grid() {
        // The 1×2 grid from the row-offset test (cols ≠ rows, per-row
        // colors), sampled 6×10 so the right and bottom margins exercise a
        // non-gray default background through every channel's shift.
        let (aw, ah, slot) = (12usize, 6usize, 6usize);
        let mut atlas = vec![0.0f32; aw * ah];
        for r in 0..4 {
            for c in 0..4 {
                atlas[(1 + r) * aw + 1 + c] = 1.0; // tile 0: solid
                atlas[(1 + r) * aw + slot + 1 + c] = 0.5; // tile 1: half
            }
        }
        let geom = CellGridGeometry {
            cols: 1,
            rows: 2,
            cell_w: 4.0,
            cell_h: 4.0,
            density: 1.0,
            atlas_width: aw as u32,
            atlas_height: ah as u32,
            tile_w: 4,
            tile_h: 4,
            frame_w: 6,
            frame_h: 10,
        };
        let default_bg = [0.25, 0.5, 0.75, 1.0];
        let cells = Arc::new(vec![
            1.0, 1.0, /* row0 fg */ 1.0, 0.0, 0.0, 1.0, /* bg */ 0.0, 0.0, 0.0, 1.0, 7.0,
            1.0, /* row1 fg */ 0.0, 0.0, 1.0, 1.0, /* bg */ 1.0, 1.0, 1.0, 1.0,
        ]);
        let atlas = Arc::new(atlas);
        let (w, h) = (6, 10);
        let program = CellGridProgram::compile(geom, default_bg);
        let frame = program.frame(cells.clone(), atlas.clone());
        let planes: [Vec<f32>; 4] = core::array::from_fn(|c| plane(&frame, c, w, h));
        let packed = CellGridPackedProgram::compile(geom, default_bg, RGBA_SHIFTS);
        let got = packed_plane(&packed.frame(cells, atlas), w, h);
        for i in 0..w * h {
            let expected = pack_expected(
                [planes[0][i], planes[1][i], planes[2][i], planes[3][i]],
                RGBA_SHIFTS,
            );
            assert_eq!(
                got[i], expected,
                "pixel {i}: {:#010x} != {:#010x}",
                got[i], expected
            );
        }
    }

    /// The packed kernel splices four channel fragments (each with its own
    /// cell reads and atlas taps), and identity-merge must still land them
    /// all in exactly two slots — one per distinct buffer.
    #[test]
    fn packed_kernel_reads_land_in_exactly_two_slots() {
        let geom = CellGridGeometry {
            cols: 2,
            rows: 1,
            cell_w: 4.0,
            cell_h: 4.0,
            density: 1.0,
            atlas_width: 12,
            atlas_height: 6,
            tile_w: 4,
            tile_h: 4,
            frame_w: 8,
            frame_h: 4,
        };
        let CellGridKernels { channels, buffers } = geom.channel_kernels([0.0; 4]);
        let kernel = packed_kernel(&channels, RGBA_SHIFTS);
        let slots = kernel.parts().0.buffers();
        assert_eq!(
            slots.iter().filter(|d| d.id == buffers.cells).count(),
            1,
            "all four channels' cell reads, one buffer, one slot"
        );
        assert_eq!(
            slots.iter().filter(|d| d.id == buffers.atlas).count(),
            1,
            "all four channels' atlas taps, one slot"
        );
        assert_eq!(slots.len(), 2, "exactly {{Cells, Atlas}}, nothing else");
    }

    /// Pins the packed program's size, in the spirit of
    /// `reads_of_one_buffer_share_a_slot_but_not_nodes`: 623 = 4 × 146 (each
    /// `or` re-splices a whole channel fragment) + 4 × 9 (each channel's pack
    /// chain: const 255, mul, const 0, max, const 255, min, trunc, const
    /// shift, shl) + 3 ors. Node sharing is the e-graph's call once it can
    /// see `Gather`; this number falling is the sign that landed.
    #[test]
    fn packed_kernel_node_count_is_the_channel_kernels_plus_the_pack() {
        let geom = CellGridGeometry {
            cols: 2,
            rows: 1,
            cell_w: 4.0,
            cell_h: 4.0,
            density: 1.0,
            atlas_width: 12,
            atlas_height: 6,
            tile_w: 4,
            tile_h: 4,
            frame_w: 8,
            frame_h: 4,
        };
        let kernel = packed_kernel(&geom.channel_kernels([0.0; 4]).channels, RGBA_SHIFTS);
        let (arena, root) = kernel.parts();
        assert_eq!(
            reachable_nodes(arena, root),
            623,
            "composed packed node count (4 channels of 146, plus the pack)"
        );
    }

    #[test]
    fn packed_row_offset_region_matches_the_full_bake() {
        // Mirrors `baking_a_row_offset_region_samples_the_correct_absolute_row`
        // for the packed path: a bake starting at y0 = 4 must reproduce rows
        // 4..8 of the full bake exactly (u32 equality — these are packed
        // pixels, not floats).
        let (aw, ah, slot) = (12usize, 6usize, 6usize);
        let mut atlas = vec![0.0f32; aw * ah];
        for r in 0..4 {
            for c in 0..4 {
                atlas[(1 + r) * aw + 1 + c] = 1.0; // tile 0: solid
                atlas[(1 + r) * aw + slot + 1 + c] = 0.5; // tile 1: half
            }
        }
        let geom = CellGridGeometry {
            cols: 1,
            rows: 2,
            cell_w: 4.0,
            cell_h: 4.0,
            density: 1.0,
            atlas_width: aw as u32,
            atlas_height: ah as u32,
            tile_w: 4,
            tile_h: 4,
            frame_w: 4,
            frame_h: 8,
        };
        let program = CellGridPackedProgram::compile(geom, [0.0; 4], RGBA_SHIFTS);
        let cells = Arc::new(vec![
            1.0, 1.0, /* row0 fg */ 1.0, 0.0, 0.0, 1.0, /* bg */ 0.0, 0.0, 0.0, 1.0, 7.0,
            1.0, /* row1 fg */ 0.0, 0.0, 1.0, 1.0, /* bg */ 1.0, 1.0, 1.0, 1.0,
        ]);
        let frame = program.frame(cells, Arc::new(atlas));

        let full = packed_plane(&frame, 4, 8);

        let stride = PackedFrame::padded_width(4);
        let mut padded = vec![0u32; 4 * stride];
        frame.bake_packed_rows(
            PlaneRegion {
                width: 4,
                y0: 4,
                rows: 4,
            },
            &mut padded,
        );
        for row in 0..4 {
            for col in 0..4 {
                assert_eq!(
                    padded[row * stride + col],
                    full[(row + 4) * 4 + col],
                    "offset bake row {row} col {col} disagrees with the full bake"
                );
            }
        }
    }

    #[test]
    #[should_panic(expected = "permutation of the byte lanes")]
    fn overlapping_pack_shifts_are_refused() {
        let geom = CellGridGeometry {
            cols: 1,
            rows: 1,
            cell_w: 4.0,
            cell_h: 4.0,
            density: 1.0,
            atlas_width: 12,
            atlas_height: 6,
            tile_w: 4,
            tile_h: 4,
            frame_w: 4,
            frame_h: 4,
        };
        // R and B both at bit 0: the OR would blend two channels' bytes.
        let _refused = CellGridPackedProgram::compile(geom, [0.0; 4], [0, 8, 0, 24]);
    }

    /// Production saturation telemetry, stage 1 of 2
    /// (docs/results/2026-09-01-production-saturation-telemetry.md): write
    /// the packed cell-grid arena core-term actually compiles, for the
    /// geometries it actually compiles it at, so the measurer in
    /// `pixelflow-search/src/runtime.rs` (`production_telemetry`) can replay
    /// `optimize_runtime_arena`'s calls on it and keep the `SaturationResult`
    /// production discards. Lives here because `packed_kernel` and
    /// `GridBuffers::mint` are private, and stays private itself.
    ///
    /// Geometry is core-term's, not a fixture's:
    /// `core-term/src/terminal_app.rs:362-372` builds `CellGridGeometry` from
    /// the snapshot's cell size (config defaults `cell_width_px: 10`,
    /// `cell_height_px: 16`, `config.rs`) times the display density, with
    /// `density: 1.0`, and the atlas extents of
    /// `GlyphAtlas::new(cell_height, density, ATLAS_CAPACITY = 128)`
    /// (`terminal_app.rs:204,243`), whose arithmetic (`atlas.rs:90-98`, PAD = 1)
    /// is restated here and cross-checked against a real `GlyphAtlas` by the
    /// glyph dumper. Shifts are `PlatformColorCube::PACKED_SHIFTS` on macOS
    /// (`RgbaColorCube`, `[0, 8, 16, 24]`); `default_bg` is the default
    /// `ColorScheme.background` (black). Density 1.0 is the startup compile,
    /// 2.0 the Retina recompile after `WindowCreated`; 120x40 is one resize.
    #[test]
    #[ignore = "telemetry dumper: PIXELFLOW_TELEMETRY_DIR=<dir> cargo test -p pixelflow-core --release -- --ignored dump_production_cell_grid_arenas"]
    fn dump_production_cell_grid_arenas() {
        let dir = std::path::PathBuf::from(
            std::env::var("PIXELFLOW_TELEMETRY_DIR").expect("PIXELFLOW_TELEMETRY_DIR must be set"),
        );
        std::fs::create_dir_all(&dir).expect("create dump dir");
        const CELL_W_PT: f32 = 10.0;
        const CELL_H_PT: f32 = 16.0;
        const ATLAS_SLOTS_PER_ROW: u32 = 12; // ceil(sqrt(128))
        const ATLAS_SLOT_ROWS: u32 = 11; // ceil(128 / 12)
        const ATLAS_PAD: u32 = 1;
        const DEFAULT_BG_BLACK: [f32; 4] = [0.0, 0.0, 0.0, 1.0];
        for (label, cols, rows, density) in [
            ("80x24_d1", 80u32, 24u32, 1.0f32),
            ("80x24_d2", 80, 24, 2.0),
            ("120x40_d2", 120, 40, 2.0),
        ] {
            let tile_px = (CELL_H_PT * density).round().max(1.0) as u32;
            let slot_px = tile_px + 2 * ATLAS_PAD;
            let cell_w = CELL_W_PT * density;
            let cell_h = CELL_H_PT * density;
            let geom = CellGridGeometry {
                cols,
                rows,
                cell_w,
                cell_h,
                density: 1.0,
                atlas_width: ATLAS_SLOTS_PER_ROW * slot_px,
                atlas_height: ATLAS_SLOT_ROWS * slot_px,
                tile_w: tile_px,
                tile_h: tile_px,
                frame_w: (cols as f32 * cell_w).round() as u32,
                frame_h: (rows as f32 * cell_h).round() as u32,
            };
            let kernel = packed_kernel(
                &geom.channel_kernels(DEFAULT_BG_BLACK).channels,
                RGBA_SHIFTS,
            );
            let (arena, root) = kernel.parts();
            let name = format!("cellgrid:{label}");
            let path = dir.join(format!("cellgrid_{label}.arena"));
            dump_arena(arena, root, &name, &path);
            println!(
                "{name}: {} reachable nodes -> {}",
                reachable_nodes(arena, root),
                path.display()
            );
        }
    }

    /// Text dump of the subgraph reachable from `root`: nodes in ascending
    /// original id order (children precede parents), ids remapped dense,
    /// constants as bit patterns, buffer identities as dense ordinals. The
    /// loader in `pixelflow-search/src/runtime.rs` is the inverse. Duplicated
    /// verbatim in `pixelflow-graphics/tests/production_glyph_arena_dump.rs`
    /// rather than shared, because the only crate both dumpers can see is
    /// `pixelflow-ir`, which must not grow a test-only serializer.
    fn dump_arena(
        arena: &ExprArena,
        root: pixelflow_ir::ExprId,
        name: &str,
        path: &std::path::Path,
    ) {
        use core::fmt::Write as _;
        use pixelflow_ir::arena::ExprNode;
        let len = arena.nodes_raw().len();
        let mut reachable = vec![false; len];
        let mut stack = vec![root];
        while let Some(id) = stack.pop() {
            if core::mem::replace(&mut reachable[id.0 as usize], true) {
                continue;
            }
            stack.extend(arena.children(id));
        }
        let mut out = std::string::String::new();
        writeln!(out, "# pixelflow arena dump v1").expect("fmt");
        writeln!(out, "name {name}").expect("fmt");
        let mut idents: Vec<BufferIdentity> = Vec::new();
        for decl in arena.buffers() {
            let ord = match idents.iter().position(|i| *i == decl.id) {
                Some(p) => p,
                None => {
                    idents.push(decl.id);
                    idents.len() - 1
                }
            };
            writeln!(out, "buf {ord} {} {}", decl.width, decl.height).expect("fmt");
        }
        let mut dense: Vec<u32> = vec![u32::MAX; len];
        let mut next = 0u32;
        let d = |dense: &[u32], id: pixelflow_ir::ExprId| -> u32 {
            let v = dense[id.0 as usize];
            assert_ne!(v, u32::MAX, "child dumped before parent");
            v
        };
        for idx in 0..len {
            if !reachable[idx] {
                continue;
            }
            let id = pixelflow_ir::ExprId(idx as u32);
            match arena.node(id) {
                ExprNode::Var(i) => writeln!(out, "V {i}"),
                ExprNode::Const(v) => writeln!(out, "C {}", v.to_bits()),
                ExprNode::Buffer(b) => writeln!(out, "B {}", b.0),
                ExprNode::Unary(k, a) => writeln!(out, "U {k:?} {}", d(&dense, *a)),
                ExprNode::Binary(k, a, b) => {
                    writeln!(out, "Bi {k:?} {} {}", d(&dense, *a), d(&dense, *b))
                }
                ExprNode::Ternary(k, a, b, c) => writeln!(
                    out,
                    "T {k:?} {} {} {}",
                    d(&dense, *a),
                    d(&dense, *b),
                    d(&dense, *c)
                ),
                other @ (ExprNode::Param(_) | ExprNode::Nary(..)) => {
                    panic!("{name}: production arena contains {other:?}, which optimize_runtime_arena bails on")
                }
            }
            .expect("fmt");
            dense[idx] = next;
            next += 1;
        }
        writeln!(out, "root {}", d(&dense, root)).expect("fmt");
        std::fs::write(path, out).unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
    }
}
