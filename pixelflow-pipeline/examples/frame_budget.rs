//! O4 of `docs/plans/2026-09-25-the-language-is-kernel.md`: what a terminal
//! frame costs, and what a zoom costs, measured on the path production takes
//! today (C2a.0 — the "today" column; the per-cell column arrives with C2a.3).
//!
//! ```text
//! PIXELFLOW_FRAME_BUDGET_FONT=/path/NotoSansMono-Regular.ttf \
//!   cargo run --release -p pixelflow-pipeline --example frame_budget            # frame matrix
//! PIXELFLOW_FRAME_BUDGET_MODE=zoom \
//!   cargo run --release -p pixelflow-pipeline --example frame_budget            # zoom
//! ```
//!
//! **What is measured.** `terminal_app.rs`'s `build_scene` path: a
//! [`GlyphAtlas`] at the cell height (density 1.0, 128 slots, ASCII warmed),
//! a cell buffer of ten floats per cell (`u0 v0 fg rgba bg rgba`),
//! `compile_platform_cell_grid` for the shape, `params` for the metric,
//! `frame` into a [`Scene::CellGrid`], and [`Scene::render`] into a
//! `Frame::<PlatformPixel>` with `threads` workers.
//!
//! **Frame mode** prints one TSV row per configuration, columns
//! `isa cores path grid cell threads workers content median_ms p90_ms min_ms
//! allocs_per_frame fill_us`:
//!
//! - `grid` is `COLSxROWS` and `cell` is `WIDTHxHEIGHT` in device pixels. The
//!   frame is the grid and nothing else, so `grid × cell` is its size.
//! - `workers` restates `render_bands`' arithmetic
//!   (`pixelflow-graphics/src/render/scene.rs`, `worker_count`) so the row
//!   says how many threads actually ran. A `threads` larger than `cores`
//!   prints `skipped` in the measurement columns rather than silently running
//!   fewer.
//! - `median_ms`, `p90_ms` and `min_ms` are over [`TIMED_RENDERS`] renders,
//!   each an `Instant` around one `Scene::render` after [`WARMUP_RENDERS`]
//!   untimed ones. The warm-ups compile the lazily compiled stripe shapes
//!   (`manifold.rs`, `Codes::with`) so no compile lands in a timed render, and
//!   the run refuses to go on if one did: `jit_cache::entry_count()` must be
//!   the same before and after the timed renders. The median of an even count
//!   is the mean of the middle two; `p90` is the nearest rank.
//! - `allocs_per_frame` is the median over the timed renders of heap
//!   allocations during `Scene::render` alone, counted by a global allocator
//!   across every thread. It is not zero, and the header's `spawn_control`
//!   lines say where it goes: `worker_count` calls `available_parallelism()`
//!   once per render (on Linux it reads the cgroup quota from files, which
//!   allocates), and `render_bands` spawns its workers with
//!   `std::thread::scope` (a spawn allocates). The control is the same call
//!   and the same `workers` count of no-op scoped spawns, so `allocs_per_frame`
//!   minus both is the render path's own. The cell `Vec` of the fill is
//!   today's per-frame debt, built untimed and reported by `fill_us`, not
//!   here; the `Arc::new` around it is built by [`Face::scene`] and is in
//!   neither column.
//! - `fill_us` is the median over [`FILL_SAMPLES`] builds of the cell buffer
//!   (app-actor work, apart from the renderer's). It mirrors `build_scene`'s
//!   loop, with a palette of `[f32; 3]` in place of core-term's `Color`
//!   (whose conversion it therefore does not time).
//!
//! **Content** is what the cells hold:
//!
//! - `dense`: printable ASCII without the space, cycled — every cell inked.
//! - `typical`: this crate's `glyph_compile_report.rs`, hard-wrapped to the
//!   grid and padded with spaces; anything outside printable ASCII becomes a
//!   space, so the atlas stays what core-term warms at startup.
//! - `heaviest`: every cell is the font's glyph with the most pieces
//!   (`GlyphRows::pieces`), found by scanning every code point the cmap maps
//!   (U+0000 through U+10FFFF) at the cell height.
//!
//! Tints come from a four-entry palette by `(row + col) % 4`. Today's path
//! gathers the same four atlas taps for every cell, so content should not move
//! the atlas column much; it is here so the per-cell column (C2a.3) adds rows
//! to the same table rather than columns.
//!
//! **Zoom mode** is one process of three steps in core-term's window: cold
//! build at the base cell, the zoom step core-term takes (`term/layout.rs`:
//! `round(cell × 1.1)`), and a revisit of the base. The window is the base
//! grid (80×24) at the base cell, and it **stays that size**; each step's grid
//! is the one that fits it, `floor(window / cell)` per axis
//! (`Layout::grid_for_window`, as `fit_to_window` calls it after a zoom), so
//! the zoomed step is a 72×21 grid at 11×18 inside 800×384, with a margin the
//! program paints in `default_bg`, and the first frame meets the partial last
//! stripe a tight frame never has. The first frame draws on `threads` workers,
//! `min(render_threads, cores)` as in production. Each step prints `face_ms`
//! (the face build: `GlyphAtlas::new` + warm ASCII), `program_ms`
//! (`compile_platform_cell_grid`), their sum `build_ms`, `first_frame_ms` (the
//! first `Scene::render`, lazily compiled stripe shapes included) and the
//! `jit_cache::entry_count()` delta the step added. The frame mode and zoom
//! mode are separate processes so neither warms the other's JIT or optimizer
//! caches.
//!
//! **Knobs**, all environment variables (`PIXELFLOW_FRAME_BUDGET_*`): `MODE`
//! (`frame`|`zoom`), `PATH` (`atlas`; `cells` is C2a.1's), `FONT` (a TTF
//! path; default core-term's font, and an LFS pointer in its place is refused
//! loudly), and comma lists narrowing the matrix: `GRID` (`80x24,200x60`),
//! `CELL_HEIGHT` (`16,32`), `THREADS` (`1,4`), `CONTENT`
//! (`dense,typical,heaviest`). Zoom takes the first `CELL_HEIGHT` and the
//! first `THREADS` (defaults 16 and `min(12, cores)`).
//!
//! Compare two builds' rows from the same host, never a row against prose.
//! The header comments name the commit, the hash of this file and of the text
//! `typical` wraps (the file is untracked until it lands, so the commit alone
//! does not say what measured), the ISA tier, the CPU and its physical-core
//! count, so a row can be told from another machine's. The commit is read when
//! the process starts, so run through `cargo run`, which rebuilds, rather than
//! an old binary. Rows go to stdout and diagnostics to stderr, so redirecting
//! stdout captures a clean table.

use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::BTreeSet;
use std::process::Command;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Instant;

use pixelflow_codegen::fnv1a64;
use pixelflow_core::{CELL_STRIDE, CellGridMetrics, CellGridShape};
use pixelflow_graphics::fonts::{Font, GlyphAtlas, GlyphRows};
use pixelflow_graphics::render::Frame;
use pixelflow_graphics::render::cell_grid::{CellGridPackedManifold, CellGridPackedParams};
use pixelflow_graphics::render::color::PlatformPixel;
use pixelflow_graphics::render::scene::{Scene, compile_platform_cell_grid};

struct Counting;

static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
/// Whether [`Counting`] counts. Set only inside [`allocations_during`], so the
/// builds and bakes the harness times elsewhere (the zoom steps above all) pay
/// a relaxed load per allocation and not a locked increment production does
/// not have.
static COUNTING: AtomicBool = AtomicBool::new(false);

// SAFETY: forwards every call to `System` unchanged; the counter is the only
// addition and touches no allocator state.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

/// Heap allocations `f` makes on any thread. Threads `f` spawns see the flag
/// through the spawn and are joined before it returns, so relaxed accesses
/// suffice; it is not reentrant, and nothing here runs two at once.
fn allocations_during(f: impl FnOnce()) -> usize {
    ALLOCATIONS.store(0, Ordering::Relaxed);
    COUNTING.store(true, Ordering::Relaxed);
    f();
    COUNTING.store(false, Ordering::Relaxed);
    ALLOCATIONS.load(Ordering::Relaxed)
}

const MODE_VAR: &str = "PIXELFLOW_FRAME_BUDGET_MODE";
const PATH_VAR: &str = "PIXELFLOW_FRAME_BUDGET_PATH";
const FONT_VAR: &str = "PIXELFLOW_FRAME_BUDGET_FONT";
const GRID_VAR: &str = "PIXELFLOW_FRAME_BUDGET_GRID";
const CELL_HEIGHT_VAR: &str = "PIXELFLOW_FRAME_BUDGET_CELL_HEIGHT";
const THREADS_VAR: &str = "PIXELFLOW_FRAME_BUDGET_THREADS";
const CONTENT_VAR: &str = "PIXELFLOW_FRAME_BUDGET_CONTENT";

/// core-term's font asset (`terminal_app.rs`: `FONT_FILENAME`). In a checkout
/// without `git lfs pull` this is a pointer file, which [`read_font`] refuses.
const DEFAULT_FONT_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../pixelflow-graphics/assets/NotoSansMono-Regular.ttf"
);
/// A Git LFS pointer is a few lines of text; no TTF starts like this or is
/// this small.
const LFS_POINTER_PREFIX: &[u8] = b"version https://git-lfs";
const LFS_POINTER_MAX_BYTES: usize = 1024;

/// `GlyphAtlas` slots before the first growth (`terminal_app.rs`:
/// `ATLAS_CAPACITY`).
const ATLAS_CAPACITY: usize = 128;
/// The atlas bakes at density 1.0 until a window reports otherwise; the
/// "Retina at density 1 with tile 32" case is a 20x32 cell, not a density.
const DENSITY: f32 = 1.0;
/// What `GlyphAtlas::warm` bakes on startup.
const WARM_RANGE: std::ops::RangeInclusive<char> = ' '..='~';

/// core-term's shipped cell, `Config::default().appearance`
/// (`core-term/src/config.rs`: `cell_width_px: 10, cell_height_px: 16`). The
/// width of a cell at another height keeps this ratio; change it here if the
/// production ratio changes.
const CELL_ASPECT: CellSize = CellSize { w: 10, h: 16 };
/// `ZOOM_STEP_FACTOR` of `core-term/src/term/layout.rs`.
const ZOOM_STEP_FACTOR: f64 = 1.1;
/// Rows of one stripe, restating `STRIPE_ROWS` of
/// `pixelflow-graphics/src/render/scene.rs`, which is private.
const STRIPE_ROWS: u32 = 8;
/// `MIN_GRID_DIMENSION` of `core-term/src/term/mod.rs`: a window too small for
/// one cell still has a one-cell grid.
const MIN_GRID_DIMENSION: u32 = 1;
/// `PerformanceConfig::default().render_threads`
/// (`pixelflow-runtime/src/config.rs`). The renderer clamps it to the cores
/// (`worker_count`), so production draws on `min(this, cores)`.
const PRODUCTION_RENDER_THREADS: usize = 12;

const WARMUP_RENDERS: usize = 10;
const TIMED_RENDERS: usize = 60;
const FILL_SAMPLES: usize = 60;
/// Runs of the spawn control, after one untimed; their median is printed.
const SPAWN_CONTROL_SAMPLES: usize = 9;
const MS_PER_SECOND: f64 = 1e3;
const US_PER_SECOND: f64 = 1e6;
/// The nearest-rank percentile `p90_ms` reports, as a fraction of the samples.
const P90_NUMERATOR: usize = 9;
const P90_DENOMINATOR: usize = 10;

const DEFAULT_GRIDS: [GridSize; 2] = [
    GridSize { cols: 80, rows: 24 },
    GridSize {
        cols: 200,
        rows: 60,
    },
];
/// The grid core-term opens with (`Config::default().appearance`: `columns:
/// 80, rows: 24`). The window is this grid at the base cell
/// (`core-term/src/main.rs`), and a zoom keeps that window.
const ZOOM_GRID: GridSize = GridSize { cols: 80, rows: 24 };
const DEFAULT_CELL_HEIGHTS: [u32; 2] = [16, 32];
const DEFAULT_THREADS: [usize; 2] = [1, 4];
const DEFAULT_CONTENTS: [Content; 3] = [Content::Dense, Content::Typical, Content::Heaviest];

/// A fixed repo source file for the `typical` content. An example of this
/// crate that nothing else touches, so the text does not drift under the rows.
const TYPICAL_SOURCE: &str = include_str!("glyph_compile_report.rs");
/// This file, for the header's hash. It is untracked until it lands, so the
/// commit alone does not say which harness measured; a binary built from other
/// text carries other text's hash.
const HARNESS_SOURCE: &str = include_str!("frame_budget.rs");

/// The program's `default_bg`: what a cell with no ink shows, and what the
/// margin of a frame larger than the grid is painted. The frame matrix is tight
/// to its grid and has no margin; a zoom step's window does.
const DEFAULT_BG: [f32; 4] = [0.05, 0.05, 0.07, 1.0];

/// A foreground and background pair, linear RGB.
struct Tint {
    fg: [f32; 3],
    bg: [f32; 3],
}

const PALETTE: [Tint; 4] = [
    Tint {
        fg: [0.90, 0.90, 0.90],
        bg: [0.05, 0.05, 0.07],
    },
    Tint {
        fg: [0.30, 0.85, 0.40],
        bg: [0.00, 0.00, 0.00],
    },
    Tint {
        fg: [0.05, 0.05, 0.05],
        bg: [0.90, 0.80, 0.30],
    },
    Tint {
        fg: [0.70, 0.90, 1.00],
        bg: [0.10, 0.15, 0.45],
    },
];

#[derive(Clone, Copy)]
struct GridSize {
    cols: u32,
    rows: u32,
}

impl GridSize {
    fn parse(text: &str) -> Option<Self> {
        let (cols, rows) = text.split_once('x')?;
        let (cols, rows) = (cols.parse().ok()?, rows.parse().ok()?);
        (cols > 0 && rows > 0).then_some(Self { cols, rows })
    }

    fn cells(self) -> usize {
        self.cols as usize * self.rows as usize
    }

    fn label(self) -> String {
        format!("{}x{}", self.cols, self.rows)
    }
}

#[derive(Clone, Copy)]
struct CellSize {
    w: u32,
    h: u32,
}

impl CellSize {
    /// The cell core-term's aspect gives at `height`.
    fn at_height(height: u32) -> Self {
        let width = f64::from(height) * f64::from(CELL_ASPECT.w) / f64::from(CELL_ASPECT.h);
        Self {
            w: width.round().max(1.0) as u32,
            h: height,
        }
    }

    /// The cell one zoom-in step lands on (`Config::zoom`): each side scaled
    /// and rounded on its own.
    fn zoomed(self) -> Self {
        let scaled = |side: u32| ((f64::from(side) * ZOOM_STEP_FACTOR).round() as u32).max(1);
        Self {
            w: scaled(self.w),
            h: scaled(self.h),
        }
    }

    fn label(self) -> String {
        format!("{}x{}", self.w, self.h)
    }
}

/// A grid and the frame it is drawn into, in device pixels.
#[derive(Clone, Copy)]
struct Viewport {
    grid: GridSize,
    frame_px: [u32; 2],
}

impl Viewport {
    /// A frame that is the grid and nothing else.
    fn tight(grid: GridSize, cell: CellSize) -> Self {
        Self {
            grid,
            frame_px: [grid.cols * cell.w, grid.rows * cell.h],
        }
    }

    /// The grid that fits a window of `frame_px` at `cell`, whole cells only
    /// (`Layout::grid_for_window`, then `fit_to_window`'s clamp).
    fn fitted(frame_px: [u32; 2], cell: CellSize) -> Self {
        let fit = |frame: u32, side: u32| (frame / side).max(MIN_GRID_DIMENSION);
        Self {
            grid: GridSize {
                cols: fit(frame_px[0], cell.w),
                rows: fit(frame_px[1], cell.h),
            },
            frame_px,
        }
    }

    fn frame_label(self) -> String {
        format!("{}x{}", self.frame_px[0], self.frame_px[1])
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Content {
    Dense,
    Typical,
    Heaviest,
}

impl Content {
    fn parse(text: &str) -> Option<Self> {
        match text {
            "dense" => Some(Self::Dense),
            "typical" => Some(Self::Typical),
            "heaviest" => Some(Self::Heaviest),
            _ => None,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Dense => "dense",
            Self::Typical => "typical",
            Self::Heaviest => "heaviest",
        }
    }
}

enum Mode {
    Frame,
    Zoom,
}

fn mode() -> Mode {
    match std::env::var(MODE_VAR) {
        Ok(v) if v == "frame" => Mode::Frame,
        Ok(v) if v == "zoom" => Mode::Zoom,
        Ok(v) => panic!("{MODE_VAR}={v:?}: expected \"frame\" or \"zoom\""),
        Err(std::env::VarError::NotPresent) => Mode::Frame,
        Err(e) => panic!("{MODE_VAR}: {e}"),
    }
}

/// The render path under measurement. One exists today.
fn check_path() {
    match std::env::var(PATH_VAR) {
        Ok(v) if v == "atlas" => {}
        Ok(v) if v == "cells" => {
            panic!("{PATH_VAR}=cells: the per-cell path is C2a.1's; only \"atlas\" exists")
        }
        Ok(v) => panic!("{PATH_VAR}={v:?}: expected \"atlas\""),
        Err(std::env::VarError::NotPresent) => {}
        Err(e) => panic!("{PATH_VAR}: {e}"),
    }
}

/// A comma list from `var`, or `default` when it is unset. A malformed entry
/// is a refusal, never a skipped one: a narrowed matrix that quietly dropped
/// a row would read as a complete one.
fn list_knob<T: Clone>(var: &str, default: &[T], parse: fn(&str) -> Option<T>) -> Vec<T> {
    match std::env::var(var) {
        Err(std::env::VarError::NotPresent) => default.to_vec(),
        Err(e) => panic!("{var}: {e}"),
        Ok(list) => list
            .split(',')
            .map(|item| {
                parse(item.trim()).unwrap_or_else(|| panic!("{var}={list:?}: cannot read {item:?}"))
            })
            .collect(),
    }
}

fn parse_height(text: &str) -> Option<u32> {
    text.parse().ok().filter(|&h| h > 0)
}

fn parse_threads(text: &str) -> Option<usize> {
    text.parse().ok().filter(|&t| t > 0)
}

fn cores() -> usize {
    std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get)
}

/// How many workers `render_bands` actually spawns for `threads` over a frame
/// `frame_h` rows tall (`scene.rs`, `worker_count`).
fn workers(threads: usize, frame_h: u32) -> usize {
    let stripes = frame_h.div_ceil(STRIPE_ROWS).max(1) as usize;
    threads.clamp(1, cores()).min(stripes)
}

fn read_font(path: &str) -> Vec<u8> {
    let data = std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    assert!(
        !(data.len() < LFS_POINTER_MAX_BYTES && data.starts_with(LFS_POINTER_PREFIX)),
        "{path} is a Git LFS pointer ({} bytes), not a font: run `git lfs pull`, or point \
         {FONT_VAR} at a real copy",
        data.len()
    );
    data
}

/// The checked-out commit, read when the process starts (so an old binary
/// reports the commit of the checkout it is run from, not the one it was built
/// at) and whether tracked files differ from it. Untracked files, this one
/// included, are invisible to both: the header's source hashes cover them.
fn commit() -> String {
    let in_repo = |args: &[&str]| {
        Command::new("git")
            .args(args)
            .current_dir(env!("CARGO_MANIFEST_DIR"))
            .output()
    };
    let rev = in_repo(&["rev-parse", "--short", "HEAD"])
        .ok()
        .filter(|out| out.status.success())
        .map_or_else(
            || "unknown".to_owned(),
            |out| String::from_utf8_lossy(&out.stdout).trim().to_owned(),
        );
    let tree = match in_repo(&["diff", "--quiet", "HEAD"]).map(|out| out.status.code()) {
        Ok(Some(0)) => "tracked files clean",
        Ok(Some(1)) => "tracked files MODIFIED",
        _ => "tracked files unknown",
    };
    format!("{rev} ({tree})")
}

/// The first `model name` line of `/proc/cpuinfo`, or `unknown` where there
/// is no such file — the failed read is the portable answer.
fn cpu_model() -> String {
    std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|info| {
            info.lines()
                .find(|line| line.starts_with("model name"))
                .and_then(|line| line.split_once(':'))
                .map(|(_, model)| model.trim().to_owned())
        })
        .unwrap_or_else(|| "unknown".to_owned())
}

/// Logical CPUs, and the physical cores they sit on by `/proc/cpuinfo`'s
/// `physical id` and `core id` pairs. `cores` is `available_parallelism`, which
/// counts logical CPUs; a scaling run needs the physical count beside it. In a
/// virtual machine this is the guest's view, which cannot show whether the
/// host runs two of its CPUs on one core.
fn cpu_topology() -> String {
    let Ok(info) = std::fs::read_to_string("/proc/cpuinfo") else {
        return "unknown".to_owned();
    };
    let mut logical = 0_usize;
    let mut physical = BTreeSet::new();
    for block in info.split("\n\n") {
        let field = |key: &str| {
            block.lines().find_map(|line| {
                let (name, value) = line.split_once(':')?;
                (name.trim() == key).then(|| value.trim().to_owned())
            })
        };
        if field("processor").is_none() {
            continue;
        }
        logical += 1;
        if let (Some(package), Some(core)) = (field("physical id"), field("core id")) {
            physical.insert((package, core));
        }
    }
    match physical.len() {
        0 => format!("{logical} logical CPUs, physical cores unknown"),
        cores => format!("{logical} logical CPUs on {cores} physical cores"),
    }
}

/// The piece count of `ch` at `size` pixels, if the font draws it.
fn pieces(font: &Font, ch: char, size: f32) -> Option<usize> {
    let id = font.cmap_lookup(ch).filter(|&id| id != 0)?;
    let outline = font.outline_scaled_by_id(id, size)?;
    Some(GlyphRows::of(&outline).pieces())
}

/// What the piece counts of one font at one size come to.
struct PieceCounts {
    ascii_total: usize,
    heaviest: char,
    heaviest_pieces: usize,
}

/// Every code point the cmap maps to a glyph, U+0000 through U+10FFFF
/// (surrogates are not `char`s). `Font` offers a lookup and no ranges to walk,
/// so this is one lookup per code point — seconds, not milliseconds — and is
/// done once per run, not once per cell height.
fn mapped_chars(font: &Font) -> Vec<char> {
    (0..=u32::from(char::MAX))
        .filter_map(char::from_u32)
        .filter(|&ch| font.cmap_lookup(ch).is_some_and(|id| id != 0))
        .collect()
}

/// The ASCII total and the most-pieces glyph of `mapped` (the first, by code
/// point, on a tie).
fn piece_counts(font: &Font, size: f32, mapped: &[char]) -> PieceCounts {
    let ascii_total = WARM_RANGE
        .filter_map(|ch| pieces(font, ch, size))
        .sum::<usize>();
    let (heaviest, heaviest_pieces) = mapped
        .iter()
        .filter_map(|&ch| pieces(font, ch, size).map(|n| (ch, n)))
        .fold(
            (' ', 0),
            |best, found| {
                if found.1 > best.1 { found } else { best }
            },
        );
    PieceCounts {
        ascii_total,
        heaviest,
        heaviest_pieces,
    }
}

fn is_printable_ascii(ch: char) -> bool {
    WARM_RANGE.contains(&ch)
}

/// The grid's characters, row-major, one per cell.
fn layout(content: Content, grid: GridSize, heaviest: char) -> Vec<char> {
    let cols = grid.cols as usize;
    match content {
        Content::Heaviest => vec![heaviest; grid.cells()],
        Content::Dense => {
            let inked: Vec<char> = ('!'..='~').collect();
            (0..grid.cells()).map(|i| inked[i % inked.len()]).collect()
        }
        Content::Typical => {
            let wrapped: Vec<Vec<char>> = TYPICAL_SOURCE
                .lines()
                .flat_map(|line| {
                    let chars: Vec<char> = line
                        .chars()
                        .map(|ch| if is_printable_ascii(ch) { ch } else { ' ' })
                        .collect();
                    let padded = |part: &[char]| {
                        let mut row = part.to_vec();
                        row.resize(cols, ' ');
                        row
                    };
                    match chars.is_empty() {
                        true => vec![padded(&[])],
                        false => chars.chunks(cols).map(padded).collect(),
                    }
                })
                .collect();
            (0..grid.rows as usize)
                .flat_map(|row| wrapped[row % wrapped.len()].iter().copied())
                .collect()
        }
    }
}

/// An atlas at one cell size, and the font it was baked from.
struct Face<'a> {
    font: &'a Font<'a>,
    atlas: GlyphAtlas,
    cell: CellSize,
}

/// A compiled cell-grid program and its metric, for one viewport on one face.
struct Compiled {
    program: CellGridPackedManifold,
    params: CellGridPackedParams,
    atlas_epoch: u64,
}

impl<'a> Face<'a> {
    /// `terminal_app.rs`'s startup: an atlas at the cell height, ASCII warmed.
    fn build(font: &'a Font<'a>, cell: CellSize) -> Self {
        let mut atlas = GlyphAtlas::new(cell.h as f32, DENSITY, ATLAS_CAPACITY);
        atlas.warm(font, WARM_RANGE);
        Self { font, atlas, cell }
    }

    /// `build_scene`'s `compile_platform_cell_grid` and `params`, for a grid
    /// in its frame.
    fn compile(&self, view: Viewport) -> Compiled {
        let shape = CellGridShape {
            cols: view.grid.cols,
            rows: view.grid.rows,
            atlas_width: self.atlas.width() as u32,
            atlas_height: self.atlas.height() as u32,
            frame_w: view.frame_px[0],
            frame_h: view.frame_px[1],
        };
        let metrics = CellGridMetrics {
            cell_w: self.cell.w as f32,
            cell_h: self.cell.h as f32,
            density: DENSITY,
            tile_w: self.atlas.tile_px() as u32,
            tile_h: self.atlas.tile_px() as u32,
            scale: DENSITY,
        };
        let program = compile_platform_cell_grid(shape, DEFAULT_BG);
        let params = program.params(&metrics);
        Compiled {
            program,
            params,
            atlas_epoch: self.atlas.epoch(),
        }
    }

    /// `build_scene`'s cell loop: ten floats per cell.
    fn fill(&mut self, chars: &[char], grid: GridSize) -> Vec<f32> {
        let mut cells = Vec::with_capacity(grid.cells() * CELL_STRIDE);
        for row in 0..grid.rows {
            for col in 0..grid.cols {
                let ch = chars[(row * grid.cols + col) as usize];
                let tint = &PALETTE[(row + col) as usize % PALETTE.len()];
                let (u, v) = self.atlas.uv(self.font, ch);
                let [fg_r, fg_g, fg_b] = tint.fg;
                let [bg_r, bg_g, bg_b] = tint.bg;
                cells.extend_from_slice(&[u, v, fg_r, fg_g, fg_b, 1.0, bg_r, bg_g, bg_b, 1.0]);
            }
        }
        cells
    }

    /// The scene a frame renders: `compiled` over these cells.
    ///
    /// # Panics
    ///
    /// If the atlas grew since `compiled` was made: its program was compiled
    /// for other extents.
    fn scene(&self, compiled: &Compiled, cells: Vec<f32>) -> Scene {
        assert_eq!(
            self.atlas.epoch(),
            compiled.atlas_epoch,
            "the atlas grew after the program was compiled against it"
        );
        Scene::CellGrid(compiled.program.frame(
            &compiled.params,
            Arc::new(cells),
            self.atlas.buffer(),
        ))
    }
}

fn seconds_since(started: Instant) -> f64 {
    started.elapsed().as_secs_f64()
}

fn sorted(mut samples: Vec<f64>) -> Vec<f64> {
    samples.sort_by(f64::total_cmp);
    samples
}

fn median(sorted: &[f64]) -> f64 {
    let mid = sorted.len() / 2;
    match sorted.len() % 2 {
        0 => (sorted[mid - 1] + sorted[mid]) / 2.0,
        _ => sorted[mid],
    }
}

/// Nearest-rank 90th percentile.
fn p90(sorted: &[f64]) -> f64 {
    sorted[(sorted.len() * P90_NUMERATOR).div_ceil(P90_DENOMINATOR) - 1]
}

/// What [`render_samples`] measured for one scene at one thread count.
struct Measured {
    median_ms: f64,
    p90_ms: f64,
    min_ms: f64,
    allocs_per_frame: usize,
}

/// `Scene::render` into `frame`, [`WARMUP_RENDERS`] untimed and then
/// [`TIMED_RENDERS`] timed.
fn render_samples(scene: &Scene, frame: &mut Frame<PlatformPixel>, threads: usize) -> Measured {
    for _ in 0..WARMUP_RENDERS {
        scene.render(frame, threads);
    }
    assert!(
        frame.data.iter().any(|px| *px != frame.data[0]),
        "the rendered frame is one flat colour: nothing was drawn"
    );
    let mut times = Vec::with_capacity(TIMED_RENDERS);
    let mut allocs = Vec::with_capacity(TIMED_RENDERS);
    let entries_before = pixelflow_codegen::jit_cache::entry_count();
    for _ in 0..TIMED_RENDERS {
        let started = Instant::now();
        let counted = allocations_during(|| scene.render(frame, threads));
        times.push(seconds_since(started) * MS_PER_SECOND);
        allocs.push(counted);
    }
    // A compile of ~10 ms among 60 samples moves neither the median nor the
    // p90, so the warm-up's claim is checked, not trusted.
    assert_eq!(
        pixelflow_codegen::jit_cache::entry_count(),
        entries_before,
        "a kernel compiled during the timed renders: the warm-up did not cover every stripe shape"
    );
    allocs.sort_unstable();
    if allocs[0] != allocs[allocs.len() - 1] {
        eprintln!(
            "# allocations varied across renders: {}..={}",
            allocs[0],
            allocs[allocs.len() - 1]
        );
    }
    let times = sorted(times);
    Measured {
        median_ms: median(&times),
        p90_ms: p90(&times),
        min_ms: times[0],
        allocs_per_frame: allocs[allocs.len() / 2],
    }
}

/// Median microseconds over [`FILL_SAMPLES`] builds of the cell buffer.
fn fill_us(face: &mut Face, chars: &[char], grid: GridSize) -> f64 {
    let times: Vec<f64> = (0..FILL_SAMPLES)
        .map(|_| {
            let started = Instant::now();
            let cells = face.fill(chars, grid);
            let elapsed = seconds_since(started) * US_PER_SECOND;
            drop(std::hint::black_box(cells));
            elapsed
        })
        .collect();
    median(&sorted(times))
}

/// The header comments every mode starts with.
fn print_header(font_path: &str, font_bytes: &[u8]) {
    println!(
        "# frame_budget: today's path (GlyphAtlas + compile_platform_cell_grid + Scene::CellGrid)"
    );
    println!("# commit {}", commit());
    println!(
        "# harness frame_budget.rs fnv1a64 {:016x}, typical text glyph_compile_report.rs fnv1a64 {:016x}",
        fnv1a64(HARNESS_SOURCE.as_bytes()),
        fnv1a64(TYPICAL_SOURCE.as_bytes())
    );
    let isa = pixelflow_codegen::isa::detect();
    println!(
        "# isa {} ({} byte vectors)",
        isa.name(),
        pixelflow_codegen::jit_vector_bytes()
    );
    println!("# cpu {}", cpu_model());
    println!("# cores {} (available_parallelism: logical CPUs)", cores());
    println!("# topology {}", cpu_topology());
    println!(
        "# font {font_path} ({} bytes, fnv1a64 {:016x})",
        font_bytes.len(),
        fnv1a64(font_bytes)
    );
    println!(
        "# cell aspect {}:{} (core-term config.rs), zoom step {ZOOM_STEP_FACTOR} (term/layout.rs)",
        CELL_ASPECT.w, CELL_ASPECT.h
    );
    println!(
        "# warmup {WARMUP_RENDERS} untimed renders, {TIMED_RENDERS} timed; fill_us is the median \
         of {FILL_SAMPLES} builds"
    );
}

/// Piece counts at one cell height, as header comments.
fn print_piece_counts(counts: &PieceCounts, height: u32) {
    println!(
        "# ascii_pieces@{height} {} (sum of GlyphRows::pieces over ' '..='~' at {height} px)",
        counts.ascii_total
    );
    println!(
        "# heaviest@{height} U+{:04X} with {} pieces (scan of every mapped code point, U+0000..=U+10FFFF)",
        counts.heaviest as u32, counts.heaviest_pieces
    );
}

/// How much of the code space the cmap covers, as a header comment.
fn print_cmap(mapped: &[char]) {
    let above_bmp = mapped
        .iter()
        .filter(|&&ch| u32::from(ch) > u32::from(u16::MAX))
        .count();
    println!(
        "# cmap {} mapped code points through U+10FFFF, {above_bmp} above the BMP",
        mapped.len()
    );
}

/// Allocations of `workers` no-op scoped spawns: the thread machinery
/// `render_bands` pays before it draws a stripe. The median of
/// [`SPAWN_CONTROL_SAMPLES`] runs after one untimed.
fn spawn_allocations(workers: usize) -> usize {
    let run = || {
        allocations_during(|| {
            std::thread::scope(|scope| {
                for _ in 0..workers {
                    scope.spawn(|| {});
                }
            });
        })
    };
    run();
    let mut samples: Vec<usize> = (0..SPAWN_CONTROL_SAMPLES).map(|_| run()).collect();
    samples.sort_unstable();
    samples[samples.len() / 2]
}

/// Allocations of one `std::thread::available_parallelism()`, which
/// `worker_count` calls on every render. On Linux it reads the cgroup quota
/// from files each time, and a file read allocates.
fn parallelism_allocations() -> usize {
    allocations_during(|| drop(std::hint::black_box(std::thread::available_parallelism())))
}

/// The `spawn_control` header lines for the worker counts a matrix will run:
/// what `render_bands` allocates for its threads before any stripe is drawn.
fn print_spawn_control(worker_counts: &BTreeSet<usize>) {
    println!(
        "# spawn_control: allocations of that many no-op std::thread::scope spawns, plus one \
         available_parallelism() call (worker_count's, once per render); allocs_per_frame minus \
         their sum is the render path's own"
    );
    println!(
        "# spawn_control available_parallelism {}",
        parallelism_allocations()
    );
    for &count in worker_counts {
        println!(
            "# spawn_control workers={count} {}",
            spawn_allocations(count)
        );
    }
}

/// Every worker count a frame row will run on, skipped rows excepted.
fn matrix_workers(grids: &[GridSize], heights: &[u32], thread_counts: &[usize]) -> BTreeSet<usize> {
    let mut counts = BTreeSet::new();
    for grid in grids {
        for &height in heights {
            for &threads in thread_counts.iter().filter(|&&t| t <= cores()) {
                counts.insert(workers(threads, grid.rows * height));
            }
        }
    }
    counts
}

fn frame_mode(font: &Font, font_path: &str, font_bytes: &[u8]) {
    let grids = list_knob(GRID_VAR, &DEFAULT_GRIDS, GridSize::parse);
    let heights = list_knob(CELL_HEIGHT_VAR, &DEFAULT_CELL_HEIGHTS, parse_height);
    let thread_counts = list_knob(THREADS_VAR, &DEFAULT_THREADS, parse_threads);
    let contents = list_knob(CONTENT_VAR, &DEFAULT_CONTENTS, Content::parse);

    print_header(font_path, font_bytes);
    let started = Instant::now();
    let mapped = mapped_chars(font);
    eprintln!("# scanned the cmap in {:.1} s", seconds_since(started));
    print_cmap(&mapped);
    let counts: Vec<PieceCounts> = heights
        .iter()
        .map(|&height| piece_counts(font, height as f32, &mapped))
        .collect();
    for (counts, &height) in counts.iter().zip(&heights) {
        print_piece_counts(counts, height);
    }
    print_spawn_control(&matrix_workers(&grids, &heights, &thread_counts));
    if contents.contains(&Content::Typical) {
        for &grid in &grids {
            let inked = layout(Content::Typical, grid, ' ')
                .iter()
                .filter(|&&ch| ch != ' ')
                .count();
            println!(
                "# typical@{} {inked} of {} cells inked",
                grid.label(),
                grid.cells()
            );
        }
    }
    println!(
        "isa\tcores\tpath\tgrid\tcell\tthreads\tworkers\tcontent\tmedian_ms\tp90_ms\tmin_ms\tallocs_per_frame\tfill_us"
    );

    let isa = pixelflow_codegen::isa::detect().name();
    let cores = cores();
    for (&height, counts) in heights.iter().zip(&counts) {
        let cell = CellSize::at_height(height);
        eprintln!("# building the {} atlas", cell.label());
        let mut face = Face::build(font, cell);
        // The heaviest glyph may lie outside ASCII: bake it now, so the atlas
        // the programs below compile against is the one every row binds.
        let started = Instant::now();
        face.atlas.warm(font, [counts.heaviest]);
        eprintln!(
            "# baked U+{:04X} ({} pieces) in {:.1} s",
            counts.heaviest as u32,
            counts.heaviest_pieces,
            seconds_since(started)
        );
        assert_ne!(
            face.atlas.uv(font, counts.heaviest),
            face.atlas.blank_uv(),
            "the heaviest glyph baked to the blank tile"
        );
        for &grid in &grids {
            eprintln!("# compiling {} cells of {}", grid.label(), cell.label());
            let view = Viewport::tight(grid, cell);
            let compiled = face.compile(view);
            let mut frame = Frame::<PlatformPixel>::new(view.frame_px[0], view.frame_px[1]);
            for &content in &contents {
                let chars = layout(content, grid, counts.heaviest);
                let fill = fill_us(&mut face, &chars, grid);
                let cells = face.fill(&chars, grid);
                let scene = face.scene(&compiled, cells);
                for &threads in &thread_counts {
                    let key = format!(
                        "{isa}\t{cores}\tatlas\t{}\t{}\t{threads}\t{}\t{}",
                        grid.label(),
                        cell.label(),
                        workers(threads, view.frame_px[1]),
                        content.name(),
                    );
                    if threads > cores {
                        println!("{key}\tskipped\tskipped\tskipped\tskipped\tskipped");
                        continue;
                    }
                    let m = render_samples(&scene, &mut frame, threads);
                    println!(
                        "{key}\t{:.3}\t{:.3}\t{:.3}\t{}\t{fill:.1}",
                        m.median_ms, m.p90_ms, m.min_ms, m.allocs_per_frame
                    );
                }
            }
        }
    }
}

/// What the zoom's steps share: the window they all draw into, which a zoom
/// does not resize, and how many threads draw.
struct ZoomRun<'a> {
    font: &'a Font<'a>,
    window_px: [u32; 2],
    threads: usize,
}

impl ZoomRun<'_> {
    /// One zoom step: build a face at `cell` and the program for the grid
    /// that fits the window at it, then draw the first frame, timing each.
    fn step(&self, name: &str, cell: CellSize) -> String {
        let entries_before = pixelflow_codegen::jit_cache::entry_count();

        let started = Instant::now();
        let mut face = Face::build(self.font, cell);
        let face_ms = seconds_since(started) * MS_PER_SECOND;
        let after_face = pixelflow_codegen::jit_cache::entry_count();

        let view = Viewport::fitted(self.window_px, cell);
        let started = Instant::now();
        let compiled = face.compile(view);
        let program_ms = seconds_since(started) * MS_PER_SECOND;
        let after_program = pixelflow_codegen::jit_cache::entry_count();

        let chars = layout(Content::Dense, view.grid, ' ');
        let cells = face.fill(&chars, view.grid);
        let scene = face.scene(&compiled, cells);
        let mut frame = Frame::<PlatformPixel>::new(view.frame_px[0], view.frame_px[1]);
        let started = Instant::now();
        scene.render(&mut frame, self.threads);
        let first_frame_ms = seconds_since(started) * MS_PER_SECOND;

        let entries = pixelflow_codegen::jit_cache::entry_count();
        eprintln!(
            "# {name}: atlas tile {} px; jit entries added: face +{}, program +{}, first frame +{}",
            face.atlas.tile_px(),
            after_face - entries_before,
            after_program - after_face,
            entries - after_program
        );
        format!(
            "{}\t{}\tatlas\t{name}\t{}\t{}\t{}\t{}\t{}\t{face_ms:.1}\t{program_ms:.1}\t{:.1}\t{first_frame_ms:.1}\t{:.1}\t{}\t{entries}",
            pixelflow_codegen::isa::detect().name(),
            cores(),
            cell.label(),
            view.grid.label(),
            view.frame_label(),
            self.threads,
            workers(self.threads, view.frame_px[1]),
            face_ms + program_ms,
            face_ms + program_ms + first_frame_ms,
            entries - entries_before,
        )
    }
}

fn zoom_mode(font: &Font, font_path: &str, font_bytes: &[u8]) {
    let height = list_knob(CELL_HEIGHT_VAR, &[DEFAULT_CELL_HEIGHTS[0]], parse_height)[0];
    let default_threads = cores().min(PRODUCTION_RENDER_THREADS);
    let threads = list_knob(THREADS_VAR, &[default_threads], parse_threads)[0];
    assert!(
        threads <= cores(),
        "{THREADS_VAR}={threads}: more threads than the {} cores here",
        cores()
    );
    let base = CellSize::at_height(height);
    let zoomed = base.zoomed();
    let run = ZoomRun {
        font,
        window_px: Viewport::tight(ZOOM_GRID, base).frame_px,
        threads,
    };

    print_header(font_path, font_bytes);
    let started = Instant::now();
    let mapped = mapped_chars(font);
    eprintln!("# scanned the cmap in {:.1} s", seconds_since(started));
    print_cmap(&mapped);
    print_piece_counts(&piece_counts(font, height as f32, &mapped), height);
    println!(
        "# zoom {} -> {} -> {} in a {}x{} window (the {} grid at the base cell, held through the \
         zoom); each step's grid is the one that fits it, {threads} thread(s); first_frame_ms \
         includes the lazily compiled stripe shapes",
        base.label(),
        zoomed.label(),
        base.label(),
        run.window_px[0],
        run.window_px[1],
        ZOOM_GRID.label()
    );
    println!(
        "isa\tcores\tpath\tstep\tcell\tgrid\tframe\tthreads\tworkers\tface_ms\tprogram_ms\tbuild_ms\tfirst_frame_ms\ttotal_ms\tjit_entries_delta\tjit_entries"
    );
    for (name, cell) in [("cold", base), ("zoom", zoomed), ("revisit", base)] {
        eprintln!("# zoom step {name}: {}", cell.label());
        println!("{}", run.step(name, cell));
    }
}

fn main() {
    let font_path = std::env::var(FONT_VAR).unwrap_or_else(|_| DEFAULT_FONT_PATH.to_owned());
    let font_bytes = read_font(&font_path);
    let font = Font::parse(&font_bytes)
        .unwrap_or_else(|| panic!("{font_path}: not a font Font::parse accepts"));
    check_path();
    match mode() {
        Mode::Frame => frame_mode(&font, &font_path, &font_bytes),
        Mode::Zoom => zoom_mode(&font, &font_path, &font_bytes),
    }
}
