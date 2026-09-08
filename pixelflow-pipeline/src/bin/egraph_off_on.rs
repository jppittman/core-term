//! What the e-graph buys on the shipped kernels — measured through the real
//! bake path, saturation on against saturation off.
//!
//! docs/plans/2026-09-06-egraph-at-production-scale.md §7 lists "F: no
//! e-graph" as the one column never measured on a shipped kernel. This is
//! that measurement. One process per mode:
//!
//! ```text
//! # built once with the switch: --features pixelflow-search/saturation-switch
//! PIXELFLOW_SATURATION=off egraph_off_on run --out rows/off.jsonl
//! PIXELFLOW_SATURATION=on  egraph_off_on run --out rows/on.jsonl
//!                          egraph_off_on run --out rows/cse.jsonl --variant cse-only
//!                          egraph_off_on run --out rows/sh.jsonl  --variant with-select-hoist
//! egraph_off_on diff --off rows/off.jsonl --on rows/on.jsonl --cse-only rows/cse.jsonl \
//!     --with-select-hoist rows/sh.jsonl \
//!     --out-prefix docs/results/2026-09-07-egraph-off-vs-on-real-shaders
//! ```
//!
//! Every kernel is compiled by the same three calls `jit_cache::compile`
//! makes — `optimize_runtime_arena` (which is where `PIXELFLOW_SATURATION`
//! is honoured), `relink`, `emit::compile` — and, for buffer-free kernels,
//! the bytes are asserted identical to `Manifold::compile`'s, so the
//! instrument *is* the production path rather than a model of it. Rows are
//! appended to the JSONL as each kernel finishes.
//!
//! The corpus is the shipped kernels and nothing else: the chrome sphere
//! (packed, plus its red channel alone), the psychedelic shader (packed, with
//! its clock uniform), the terminal cell grid at a Retina 80×24 geometry, the
//! 190 glyph bakes `GlyphAtlas::warm` performs plus the three `font_rendering`
//! bench glyphs (and `O`@32 at a 640-wide row for the prologue estimate), and
//! the twelve `shader_bench` ShaderToy ports.

use std::collections::BTreeMap;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use clap::{Parser, Subcommand, ValueEnum};
use serde::{Deserialize, Serialize};

use pixelflow_codegen::emit::executable::{ExecutableCode, Point4, TileSlice};
use pixelflow_codegen::emit::{self, CompileResult};
use pixelflow_core::lattice::cell_grid::{CELL_STRIDE, CellGridMetrics};
use pixelflow_core::{Bits, CellGridShape, Kernel, Uniform};
use pixelflow_graphics::fonts::{Font, GlyphAtlas};
use pixelflow_graphics::render::Frame;
use pixelflow_graphics::render::color::Rgba8;
use pixelflow_graphics::render::pixel::Pixel;
use pixelflow_graphics::render::scene::{Scene, compile_cell_grid_for};
use pixelflow_graphics::scene3d::{Hit, Plane, Ray, Rgba, Sphere, checker, sky};
use pixelflow_ir::optimize::{Optimize, Rewritten};
use pixelflow_ir::passes::{ExpandReduce, LowerDwrt};
use pixelflow_ir::{
    BindingTable, ExprArena, ExprId, ExprNode, LatticeShape, eval_scalar, pipeline,
};
use pixelflow_pipeline::collapse_bench::corpus::Trips;
use pixelflow_pipeline::collapse_bench::row::StaticFeatures;
use pixelflow_pipeline::collapse_bench::{self, LANES, features_of};
use pixelflow_pipeline::shader_bench::{SHADERTOY_KERNEL_NAMES, named_shadertoy_kernel};
use pixelflow_search::Saturate;
use pixelflow_search::egraph::{
    Budget, CostModel, EpisodeLabels, KeepJournal, Optimizer, Rewrite, RuleSet, Vocabulary,
    all_rules, insert, reachable_count,
};
use pixelflow_search::math::round2_rules::experimental_rules;

const SCHEMA: &str = "egraph-off-on-v1";
const SCREEN: [u32; 2] = [1920, 1080];
const SHADER_EXTENT: [u32; 2] = [256, 256];
const CELL_HEIGHT_PT: f32 = 16.0;
const CELL_WIDTH_PT: f32 = 10.0;
const ATLAS_CAPACITY: usize = 128;
const DENSITIES: [f32; 2] = [1.0, 2.0];
const WARM_RANGE: std::ops::RangeInclusive<char> = ' '..='~';
const BENCH_CHARS: [(&str, char); 3] =
    [("A_linear", 'A'), ("O_quadratic", 'O'), ("S_complex", 'S')];
const BENCH_PT: f32 = 32.0;
const BENCH_EXTENT: [u32; 2] = [40, 45];
const BENCH_WIDE_EXTENT: [u32; 2] = [640, 45];
const ORACLE_POINTS: usize = 256;
const CLOCK_SAMPLES: usize = 7;
const CLOCK_MIN_SAMPLE_NS: u64 = 2_000_000;
const CLOCK_MAX_CALLS: usize = 20_000;
const SELECT_HOIST_PREFIX: &str = "select-hoist-";

#[derive(Parser)]
#[command(
    name = "egraph_off_on",
    about = "Saturation on vs off on every shipped kernel"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Compile (and time) every real kernel on this build; one JSONL row each.
    Run {
        #[arg(long)]
        out: PathBuf,
        /// An in-harness optimizer instead of the production one
        /// (`PIXELFLOW_SATURATION` must be unset or `on`).
        #[arg(long)]
        variant: Option<Variant>,
        /// Skip the clock (deterministic columns only).
        #[arg(long)]
        no_clock: bool,
        /// Skip the saturation probe (rule fire counts / load-bearing rules).
        #[arg(long)]
        no_probe: bool,
        /// Only kernels whose name contains this substring.
        #[arg(long)]
        filter: Option<String>,
        /// Kernels (exact names) to leave out of this run — the emitter
        /// panics (branch range, frame size) rather than returning an error,
        /// so a kernel one mode cannot emit is excluded here and reported by
        /// `diff` as present in one mode only.
        #[arg(long, num_args = 0..)]
        skip: Vec<String>,
        #[arg(long)]
        font: Option<PathBuf>,
    },
    /// Diff runs into the results documents.
    Diff {
        #[arg(long, num_args = 1..)]
        off: Vec<PathBuf>,
        #[arg(long, num_args = 1..)]
        on: Vec<PathBuf>,
        /// Rows from `--variant with-select-hoist`.
        #[arg(long, num_args = 0..)]
        with_select_hoist: Vec<PathBuf>,
        /// Rows from `--variant cse-only`.
        #[arg(long, num_args = 0..)]
        cse_only: Vec<PathBuf>,
        #[arg(long)]
        out_prefix: PathBuf,
        /// Free-text lines (load, bench_scene numbers) appended verbatim.
        #[arg(long, num_args = 0..)]
        note: Vec<String>,
    },
}

/// The in-harness optimizers, each `Optimizer::production()` with one
/// lever moved — the same pipeline `optimize_runtime_arena_uncached` runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum Variant {
    /// Production's 62 rules plus `SelectHoistUnary` (`select-hoist-neg|abs|sqrt`,
    /// which live in `round2_rules::experimental_rules` and are *not* in
    /// `all_rules()`): does the rule the demand plan indicts ever match a
    /// real shader, and what does it do to the guards when it does.
    WithSelectHoist,
    /// Zero rewrite rounds: insert, extract. What the e-graph's hash-consing
    /// alone buys, separated from what the rules buy.
    CseOnly,
}

impl Variant {
    fn label(self) -> &'static str {
        match self {
            Variant::WithSelectHoist => "with-select-hoist",
            Variant::CseOnly => "cse-only",
        }
    }

    fn optimizer(self, shape: LatticeShape) -> Optimizer {
        let base = Optimizer::production().for_lattice(shape);
        match self {
            Variant::WithSelectHoist => base.rules(RuleSet::new(with_select_hoist_rules())),
            Variant::CseOnly => base.budget(Budget::Explicit {
                iterations: 0,
                classes: usize::MAX,
                applications: Some(0),
            }),
        }
    }
}

// ---------------------------------------------------------------------------
// Rows
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Clone, Debug)]
struct GuardTelemetry {
    schedule: u64,
    selects: u64,
    guarded: u64,
    exclusive: u64,
}

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
struct Oracle {
    points: usize,
    /// Same form: `eval_scalar` of the arena that was emitted vs the JIT.
    same_form_max_abs: f64,
    same_form_max_rel: f64,
    same_form_nan_mismatch: usize,
    /// Cross form: `eval_scalar` of the arena as constructed (after the
    /// legalizing prefix, since `eval_scalar` has no `Dwrt`) vs the JIT of
    /// what was compiled from it — what the rewrites moved, at the sample.
    cross_form_max_abs: f64,
    cross_form_max_rel: f64,
    cross_form_nan_mismatch: usize,
    /// Packed kernels: pixels whose 32-bit pattern differs, and the largest
    /// per-byte (per-channel) delta, same form / cross form.
    packed_mismatch_same: usize,
    packed_max_byte_same: u32,
    packed_mismatch_cross: usize,
    packed_max_byte_cross: u32,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct Clock {
    ns_per_call_median: f64,
    ns_per_call_min: f64,
    ns_per_call_iqr: f64,
    calls_per_sample: usize,
    ns_per_px: f64,
    /// `Scene::render` at one thread, median of 5 frames — cell grid only.
    scene_ns_per_px: Option<f64>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct RuleCount {
    rule: String,
    fired: usize,
    /// `EpisodeLabels::compute_tight` (`derivation_ancestors_tight`).
    load_bearing: usize,
    /// `EpisodeLabels::compute_strict`: the application's own e-node was chosen.
    strict: usize,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct SatProbe {
    applications: u64,
    unions: usize,
    classes: usize,
    iterations: usize,
    stop: String,
    wall_ms: f64,
    /// The three `select-hoist-*` rules, always present (zero if never fired).
    select_hoist: Vec<RuleCount>,
    /// Every rule with at least one load-bearing application, most first.
    load_bearing_rules: Vec<RuleCount>,
    /// Probe extraction compiled to the same bytes as the production path.
    bytes_identical_to_production: Option<bool>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct KernelRow {
    schema: String,
    mode: String,
    tier: String,
    lanes: u32,
    git_sha: String,
    name: String,
    class: String,
    extent: [u32; 2],
    packed: bool,
    input_nodes: usize,
    compiled_nodes: usize,
    dag_cost_input: usize,
    dag_cost: usize,
    bytes: u32,
    spill_slots: u32,
    hoisted: u32,
    statics: StaticFeatures,
    guard: Option<GuardTelemetry>,
    optimize_ms: f64,
    emit_ms: f64,
    bytes_identical_to_manifold_compile: Option<bool>,
    picture_hash: u64,
    oracle: Option<Oracle>,
    clock: Option<Clock>,
    probe: Option<SatProbe>,
}

// ---------------------------------------------------------------------------
// The corpus
// ---------------------------------------------------------------------------

struct CellGridCase {
    shape: CellGridShape,
    metrics: CellGridMetrics,
    cells_id: pixelflow_ir::arena::BufferIdentity,
    cells: Arc<Vec<f32>>,
    atlas: Arc<Vec<f32>>,
}

struct RealKernel {
    name: String,
    class: String,
    kernel: Kernel,
    extent: [u32; 2],
    packed: bool,
    cell_grid: Option<CellGridCase>,
}

fn k(v: f32) -> Kernel {
    Kernel::constant(v)
}

/// `pixelflow_graphics::render::packed::packed_kernel`, which is
/// `pub(crate)`: the byte pack `compile_packed_for` wraps a colour in.
/// `main` asserts the bytes of what this produces equal the production
/// program's, so a drift here is loud.
fn packed_kernel(color: &Rgba, shifts: [u32; 4]) -> Kernel {
    color
        .fold(
            &|channels: &[Kernel; 4]| {
                let byte = |c: usize| {
                    channels[c]
                        .mul(&k(255.0))
                        .clamp(&k(0.0), &k(255.0))
                        .trunc_to_int()
                        .shl(shifts[c])
                };
                byte(0).or(&byte(1)).or(&byte(2)).or(&byte(3))
            },
            &|mask, if_true: Bits, if_false: Bits| Bits::select(mask, &if_true, &if_false),
        )
        .into_kernel()
}

fn rgba8_shifts() -> [u32; 4] {
    <Rgba8 as Pixel>::packed_shifts().expect("Rgba8 packs")
}

fn chrome_color() -> Rgba {
    const CENTER: (f32, f32, f32) = (0.0, 0.0, 4.0);
    const RADIUS: f32 = 1.0;
    const FLOOR: f32 = -1.0;
    fn world(ray: &Ray) -> Rgba {
        let floor = Plane::at_height(k(FLOOR)).hit(ray);
        floor.select(
            &checker(&floor.point()[0], &floor.point()[2], &floor.footprint()),
            &sky(ray),
        )
    }
    let ray = Ray::through_screen(SCREEN[0] as f32, SCREEN[1] as f32);
    let sphere: Hit = Sphere::new([k(CENTER.0), k(CENTER.1), k(CENTER.2)], k(RADIUS)).hit(&ray);
    let mirrored = ray.reflected(sphere.normal());
    sphere.select(&world(&mirrored), &world(&ray))
}

fn psych_channel(y_weight: f32, clock: Uniform) -> Kernel {
    let scale = 2.0 / 1080.0;
    let x = Kernel::x().sub(&k(960.0)).mul(&k(scale));
    let y = k(540.0).sub(&Kernel::y()).mul(&k(scale));
    let time = clock.kernel().add(&k(1.3));
    let r_sq = x.mul(&x).add(&y.mul(&y));
    let radial = r_sq.sub(&k(0.7)).abs();
    let swirl_scale = k(1.0).sub(&radial).mul(&k(5.0));
    let vx = x.mul(&swirl_scale);
    let vy = y.mul(&swirl_scale);
    let phase = time.mul(&k(0.5));
    let sin_w03 = time.mul(&k(0.3)).sin();
    let sin_w20 = time.mul(&k(2.0)).sin();
    let vxp = vx.add(&phase);
    let swirl = vxp
        .sin()
        .add(&k(1.0))
        .mul(&vxp.sub(&vy.add(&phase.mul(&k(0.7)))).abs())
        .mul(&k(0.2))
        .add(&k(0.001));
    let pulse = k(1.0).add(&sin_w20.mul(&k(0.1)));
    let radial_factor = radial.mul(&k(-4.0)).mul(&pulse).exp();
    let raw = y
        .mul(&k(y_weight))
        .add(&sin_w03.mul(&k(0.2)))
        .exp()
        .mul(&radial_factor)
        .div(&swirl);
    raw.div(&raw.abs().add(&k(1.0))).add(&k(1.0)).mul(&k(0.5))
}

fn psychedelic_color() -> Rgba {
    let clock = Uniform::new(0.0);
    Rgba::from([
        psych_channel(1.0, clock),
        psych_channel(-1.0, clock),
        psych_channel(-2.0, clock),
        k(1.0),
    ])
}

fn cell_grid_case() -> (Kernel, CellGridCase) {
    const COLS: u32 = 80;
    const ROWS: u32 = 24;
    const DENSITY: f32 = 2.0;
    const ATLAS_SLOTS_PER_ROW: u32 = 12;
    const ATLAS_SLOT_ROWS: u32 = 11;
    const ATLAS_PAD: u32 = 1;
    let tile_px = (CELL_HEIGHT_PT * DENSITY).round().max(1.0) as u32;
    let slot_px = tile_px + 2 * ATLAS_PAD;
    let cell_w = CELL_WIDTH_PT * DENSITY;
    let cell_h = CELL_HEIGHT_PT * DENSITY;
    let shape = CellGridShape {
        cols: COLS,
        rows: ROWS,
        atlas_width: ATLAS_SLOTS_PER_ROW * slot_px,
        atlas_height: ATLAS_SLOT_ROWS * slot_px,
        frame_w: (COLS as f32 * cell_w).round() as u32,
        frame_h: (ROWS as f32 * cell_h).round() as u32,
    };
    let metrics = CellGridMetrics {
        cell_w,
        cell_h,
        density: DENSITY,
        tile_w: tile_px,
        tile_h: tile_px,
        scale: DENSITY,
    };
    let kernels = shape.channel_kernels();
    let kernel = packed_kernel(&Rgba::from(&kernels.channels), rgba8_shifts());

    let mut cells = Vec::with_capacity((COLS * ROWS) as usize * CELL_STRIDE);
    for i in 0..(COLS * ROWS) as usize {
        let slot = (i % 95) as f32;
        cells.extend_from_slice(&[slot, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    }
    let atlas_len = (shape.atlas_width * shape.atlas_height) as usize;
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    let atlas: Vec<f32> = (0..atlas_len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 40) & 0xFF) as f32 / 255.0
        })
        .collect();
    (
        kernel,
        CellGridCase {
            shape,
            metrics,
            cells_id: kernels.buffers.cells,
            cells: Arc::new(cells),
            atlas: Arc::new(atlas),
        },
    )
}

fn real_kernels(font: &Path, filter: Option<&str>) -> Vec<RealKernel> {
    let mut out = Vec::new();
    let mut push = |name: String, class: &str, kernel: Kernel, extent: [u32; 2], packed: bool| {
        out.push(RealKernel {
            name,
            class: class.to_string(),
            kernel,
            extent,
            packed,
            cell_grid: None,
        });
    };
    let shifts = rgba8_shifts();
    let chrome = chrome_color();
    push(
        "chrome_packed".into(),
        "chrome",
        packed_kernel(&chrome, shifts),
        SCREEN,
        true,
    );
    let red = chrome.fold(
        &|channels: &[Kernel; 4]| channels[0].clone(),
        &|mask, a: Kernel, b: Kernel| mask.select(&a, &b),
    );
    push("chrome_R".into(), "chrome_channel", red, SCREEN, false);
    push(
        "psychedelic_packed".into(),
        "psychedelic",
        packed_kernel(&psychedelic_color(), shifts),
        SCREEN,
        true,
    );

    let (kernel, case) = cell_grid_case();
    let extent = [case.shape.frame_w, case.shape.frame_h];
    push("cellgrid_80x24_d2".into(), "cellgrid", kernel, extent, true);
    let cell_grid = case;

    let data = std::fs::read(font).unwrap_or_else(|e| panic!("read {}: {e}", font.display()));
    let parsed = Font::parse(&data).expect("parse the production font");
    let mut missing = 0usize;
    for density in DENSITIES {
        let atlas = GlyphAtlas::new(CELL_HEIGHT_PT, density, ATLAS_CAPACITY);
        let tile = atlas.tile_px() as u32;
        for ch in WARM_RANGE {
            // The winding sum reads a bound piece table, so a glyph is a
            // kernel plus its binding. This harness measures the arena the
            // optimizer sees, which the kernel alone carries.
            let Some(glyph) = parsed.glyph_kernel_scaled(ch, tile as f32) else {
                missing += 1;
                continue;
            };
            let kernel = glyph.kernel;
            push(
                format!("glyph{tile}_U{:04X}", ch as u32),
                &format!("glyph{tile}"),
                kernel,
                [tile, tile],
                false,
            );
        }
    }
    eprintln!("egraph_off_on: the font has no glyph for {missing} warmed characters");
    for (label, ch) in BENCH_CHARS {
        let kernel = parsed
            .glyph_kernel_scaled(ch, BENCH_PT)
            .unwrap_or_else(|| panic!("no glyph for {ch:?}"))
            .kernel;
        push(
            format!("bench_{label}"),
            "bench",
            kernel.clone(),
            BENCH_EXTENT,
            false,
        );
        if ch == 'O' {
            push(
                format!("bench_{label}_wide"),
                "bench_wide",
                kernel,
                BENCH_WIDE_EXTENT,
                false,
            );
        }
    }

    for name in SHADERTOY_KERNEL_NAMES {
        let (arena, root) = named_shadertoy_kernel(name).expect("registered shader");
        push(
            format!("shader_{name}"),
            "shader",
            Kernel::from_parts(arena, root),
            SHADER_EXTENT,
            false,
        );
    }

    out.iter_mut()
        .find(|r| r.class == "cellgrid")
        .expect("cell grid row")
        .cell_grid = Some(cell_grid);
    if let Some(f) = filter {
        out.retain(|r| r.name.contains(f));
    }
    out
}

// ---------------------------------------------------------------------------
// Compile: the production path's three calls, instrumented
// ---------------------------------------------------------------------------

struct Compiled {
    result: CompileResult,
    linked: ExprArena,
    root: ExprId,
    optimize_ms: f64,
    emit_ms: f64,
    guard: Option<GuardTelemetry>,
}

/// Redirect fd 2 into a file around `f` so `PIXELFLOW_GUARD_TELEMETRY`'s
/// `eprintln!` lands somewhere this process can read it back.
fn capture_stderr<T>(f: impl FnOnce() -> T) -> (T, String) {
    use std::os::unix::io::AsRawFd as _;
    let path = std::env::temp_dir().join(format!("egraph_off_on.{}.stderr", std::process::id()));
    let file = std::fs::File::create(&path).expect("stderr capture file");
    let saved = unsafe { libc::dup(2) };
    assert!(saved >= 0, "dup(2) failed");
    assert!(
        unsafe { libc::dup2(file.as_raw_fd(), 2) } >= 0,
        "dup2 failed"
    );
    let r = f();
    assert!(unsafe { libc::dup2(saved, 2) } >= 0, "dup2 restore failed");
    unsafe { libc::close(saved) };
    drop(file);
    let text = std::fs::read_to_string(&path).expect("read stderr capture");
    std::fs::remove_file(&path).expect("remove stderr capture");
    (r, text)
}

fn parse_guard_telemetry(log: &str) -> Option<GuardTelemetry> {
    let line = log.lines().rfind(|l| l.starts_with("guard-telemetry:"))?;
    let field = |key: &str| -> u64 {
        let needle = format!("{key}=");
        let start = line
            .find(&needle)
            .unwrap_or_else(|| panic!("{key} in {line}"))
            + needle.len();
        line[start..]
            .split(|c: char| !c.is_ascii_digit())
            .next()
            .expect("digits")
            .parse()
            .expect("u64")
    };
    Some(GuardTelemetry {
        schedule: field("schedule"),
        selects: field("selects"),
        guarded: field("guarded"),
        exclusive: field("exclusive"),
    })
}

fn compile_via_production_path(arena: &ExprArena, root: ExprId, shape: LatticeShape) -> Compiled {
    let t = Instant::now();
    let optimized = pixelflow_search::runtime::optimize_runtime_arena(arena, root, shape);
    let optimize_ms = t.elapsed().as_secs_f64() * 1e3;
    let (a, r) = optimized
        .as_deref()
        .map(|(a, r)| (a, *r))
        .unwrap_or((arena, root));
    let (linked, root) = if arena.buffers().is_empty() && arena.uniforms().is_empty() {
        (a.clone(), r)
    } else {
        a.relink(r, arena.buffers(), arena.uniforms())
    };
    let t = Instant::now();
    let (result, log) = capture_stderr(|| emit::compile(&linked, root));
    let emit_ms = t.elapsed().as_secs_f64() * 1e3;
    let result = result.expect("real kernel failed to compile");
    Compiled {
        result,
        linked,
        root,
        optimize_ms,
        emit_ms,
        guard: parse_guard_telemetry(&log),
    }
}

/// A [`Variant`]: the same pipeline `optimize_runtime_arena_uncached`
/// runs, with the variant's optimizer in the saturation slot.
fn compile_via_variant(
    arena: &ExprArena,
    root: ExprId,
    shape: LatticeShape,
    variant: Variant,
) -> Compiled {
    let t = Instant::now();
    let optimizer = variant.optimizer(shape);
    let rewritten = pipeline![
        LowerDwrt,
        ExpandReduce,
        Saturate::with(optimizer, Vocabulary::Runtime)
    ]
    .optimize(arena, root);
    let optimize_ms = t.elapsed().as_secs_f64() * 1e3;
    let (a, r) = match rewritten {
        Rewritten::Changed(a, r) => (a, r),
        Rewritten::Unchanged => (arena.clone(), root),
        Rewritten::Declined => panic!("variant pipeline declined a real kernel"),
    };
    let (linked, root) = if arena.buffers().is_empty() && arena.uniforms().is_empty() {
        (a, r)
    } else {
        a.relink(r, arena.buffers(), arena.uniforms())
    };
    let t = Instant::now();
    let (result, log) = capture_stderr(|| emit::compile(&linked, root));
    let emit_ms = t.elapsed().as_secs_f64() * 1e3;
    Compiled {
        result: result.expect("variant kernel failed to compile"),
        linked,
        root,
        optimize_ms,
        emit_ms,
        guard: parse_guard_telemetry(&log),
    }
}

fn with_select_hoist_rules() -> Vec<Box<dyn Rewrite>> {
    let mut rules = all_rules();
    assert!(
        !rules
            .iter()
            .any(|r| r.name().starts_with(SELECT_HOIST_PREFIX)),
        "select-hoist is in all_rules() now; the with-select-hoist variant is moot"
    );
    let hoist: Vec<Box<dyn Rewrite>> = experimental_rules()
        .into_iter()
        .filter(|r| r.name().starts_with(SELECT_HOIST_PREFIX))
        .collect();
    assert_eq!(hoist.len(), 3, "expected exactly three select-hoist rules");
    rules.extend(hoist);
    rules
}

// ---------------------------------------------------------------------------
// Static columns
// ---------------------------------------------------------------------------

fn reachable(arena: &ExprArena, root: ExprId) -> Vec<ExprId> {
    let len = arena.nodes_raw().len();
    let mut seen = vec![false; len];
    let mut stack = vec![root];
    let mut out = Vec::new();
    while let Some(id) = stack.pop() {
        if std::mem::replace(&mut seen[id.0 as usize], true) {
            continue;
        }
        out.push(id);
        stack.extend(arena.children(id));
    }
    out
}

fn dag_cost(arena: &ExprArena, root: ExprId) -> usize {
    let model = CostModel::latency_prior();
    reachable(arena, root)
        .into_iter()
        .filter_map(|id| match arena.node(id) {
            ExprNode::Unary(k, _) | ExprNode::Binary(k, _, _) | ExprNode::Ternary(k, _, _, _) => {
                Some(model.cost(*k))
            }
            _ => None,
        })
        .sum()
}

// ---------------------------------------------------------------------------
// Running the emitted kernel: full-extent output, oracle, clock
// ---------------------------------------------------------------------------

struct Ctx {
    /// Buffer slots in the linked arena's order, then the uniform block.
    slots: Vec<*const f32>,
    _uniforms: Vec<f32>,
    _buffers: Vec<Arc<Vec<f32>>>,
}

/// One kernel's own tabulations, by the [`BufferIdentity`] each was seeded
/// under — `Kernel::buffer_data()` (`bee7813`, "a kernel carries its own
/// tabulations"). A glyph's winding sum reads a piece table this way: the
/// data travels with `RealKernel::kernel` itself, with nothing separate a
/// caller must gather and keep paired with it.
type Carried = [(pixelflow_ir::arena::BufferIdentity, Arc<[f32]>)];

/// Resolve one declared buffer to real memory: `carried` — the kernel's own
/// tabulation — first, then `case`'s cell-grid buffers for the one kernel
/// (the terminal cell grid) that reads externally-owned per-frame memory
/// instead of a self-contained table. Mirrors `Manifold::bind`'s own
/// precedence (`pixelflow-core/src/lattice/manifold.rs`): a slot the kernel
/// carries data for is already spoken for, so a caller-supplied binding is
/// only ever the fallback. This harness cannot call `Manifold::bind` itself
/// — it drives its own instrumented `ExecutableCode`, compiled outside
/// `Manifold::compile` so it can capture guard telemetry and optimize/emit
/// timings — but the resolution a raw context table needs is the same.
///
/// # Panics
/// If `id` names neither the kernel's own tabulation nor (when given) the
/// cell-grid case's buffers — a kernel that declares a buffer nothing here
/// can bind, which is a corpus bug rather than a shape to paper over.
fn resolve_buffer<'a>(
    id: pixelflow_ir::arena::BufferIdentity,
    carried: &'a Carried,
    case: Option<&'a CellGridCase>,
) -> &'a [f32] {
    if let Some((_, data)) = carried.iter().find(|(cid, _)| *cid == id) {
        return data;
    }
    let case = case.unwrap_or_else(|| {
        panic!("buffer {id:?}: not in the kernel's own tabulation and no cell-grid case was given")
    });
    if id == case.cells_id {
        &case.cells
    } else {
        &case.atlas
    }
}

fn context_for(linked: &ExprArena, carried: &Carried, case: Option<&CellGridCase>) -> Ctx {
    let mut buffers: Vec<Arc<Vec<f32>>> = Vec::new();
    for decl in linked.buffers() {
        let data = resolve_buffer(decl.id, carried, case);
        assert_eq!(
            data.len(),
            (decl.width * decl.height) as usize,
            "buffer extent {}x{} does not match its data",
            decl.width,
            decl.height
        );
        buffers.push(Arc::new(data.to_vec()));
    }
    let uniforms: Vec<f32> = linked.uniforms().iter().map(|u| u.default).collect();
    let mut slots: Vec<*const f32> = buffers.iter().map(|b| b.as_ptr()).collect();
    slots.push(uniforms.as_ptr());
    Ctx {
        slots,
        _uniforms: uniforms,
        _buffers: buffers,
    }
}

fn run_once(code: &ExecutableCode, ctx: &Ctx, trips: Trips, out: &mut [f32]) {
    let mut x0 = [0.0f32; LANES];
    for (i, lane) in x0.iter_mut().enumerate() {
        *lane = 0.5 + i as f32;
    }
    let origin = Point4::new(x0, [0.5f32; LANES], [0.0f32; LANES], [0.0f32; LANES]);
    let tile = TileSlice::contiguous(out.as_mut_ptr(), trips.groups as usize, trips.rows as usize);
    unsafe {
        code.call_collapse(ctx.slots.as_ptr(), tile, origin);
    }
}

fn fnv(out: &[f32]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for v in out {
        for b in v.to_bits().to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
    }
    h
}

struct OracleForms<'a> {
    /// The arena as constructed, legalized (`legalize`).
    input: (&'a ExprArena, ExprId),
    /// The arena that was emitted.
    linked: (&'a ExprArena, ExprId),
    /// The kernel's own tabulations — see [`Carried`].
    carried: &'a Carried,
    case: Option<&'a CellGridCase>,
    packed: bool,
}

fn oracle(forms: &OracleForms<'_>, out: &[f32], trips: Trips) -> Oracle {
    let OracleForms {
        input,
        linked,
        carried,
        case,
        packed,
    } = *forms;
    let bindings_for = |arena: &ExprArena| -> BindingTable<'_> {
        let decls = arena.buffers();
        let table = if decls.is_empty() {
            BindingTable::empty()
        } else {
            let slices: Vec<&[f32]> = decls
                .iter()
                .map(|d| resolve_buffer(d.id, carried, case))
                .collect();
            BindingTable::bind(arena, &slices).expect("bind oracle buffers")
        };
        table
            .bind_uniforms(arena, &[])
            .expect("bind oracle uniforms")
    };
    let b_in = bindings_for(input.0);
    let b_ln = bindings_for(linked.0);
    let width = trips.groups as usize * LANES;
    let pixels = width * trips.rows as usize;
    let stride = (pixels / ORACLE_POINTS).max(1);
    let mut o = Oracle::default();
    let mut px = 0usize;
    while px < pixels && o.points < ORACLE_POINTS {
        let (x, y) = ((px % width) as f32 + 0.5, (px / width) as f32 + 0.5);
        let jit = out[px];
        let same = eval_scalar(linked.0, linked.1, &[x, y], &b_ln);
        let cross = eval_scalar(input.0, input.1, &[x, y], &b_in);
        if packed {
            let byte_delta = |a: f32, b: f32| -> u32 {
                a.to_bits()
                    .to_le_bytes()
                    .iter()
                    .zip(b.to_bits().to_le_bytes())
                    .map(|(&p, q)| u32::from(p.abs_diff(q)))
                    .max()
                    .unwrap_or(0)
            };
            if jit.to_bits() != same.to_bits() {
                o.packed_mismatch_same += 1;
                o.packed_max_byte_same = o.packed_max_byte_same.max(byte_delta(jit, same));
            }
            if jit.to_bits() != cross.to_bits() {
                o.packed_mismatch_cross += 1;
                o.packed_max_byte_cross = o.packed_max_byte_cross.max(byte_delta(jit, cross));
            }
        } else {
            let acc = |reference: f32, max_abs: &mut f64, max_rel: &mut f64, nan: &mut usize| match (
                jit.is_nan(),
                reference.is_nan(),
            ) {
                (true, true) => {}
                (true, false) | (false, true) => *nan += 1,
                (false, false) => {
                    let d = f64::from((jit - reference).abs());
                    *max_abs = max_abs.max(d);
                    if reference.abs() > 1e-3 {
                        *max_rel = max_rel.max(d / f64::from(reference.abs()));
                    }
                }
            };
            acc(
                same,
                &mut o.same_form_max_abs,
                &mut o.same_form_max_rel,
                &mut o.same_form_nan_mismatch,
            );
            acc(
                cross,
                &mut o.cross_form_max_abs,
                &mut o.cross_form_max_rel,
                &mut o.cross_form_nan_mismatch,
            );
        }
        o.points += 1;
        px += stride;
    }
    o
}

fn clock(code: &ExecutableCode, ctx: &Ctx, trips: Trips, out: &mut [f32]) -> Clock {
    let run = |calls: usize, out: &mut [f32]| -> u64 {
        let t = Instant::now();
        for _ in 0..calls {
            run_once(code, ctx, trips, out);
            std::hint::black_box(&out[0]);
        }
        t.elapsed().as_nanos() as u64
    };
    let mut calls = 1usize;
    loop {
        let ns = run(calls, out);
        if ns >= CLOCK_MIN_SAMPLE_NS || calls >= CLOCK_MAX_CALLS {
            break;
        }
        let want = (CLOCK_MIN_SAMPLE_NS as f64 / ns.max(1) as f64).ceil() as usize;
        calls = (calls * want.clamp(2, 16)).min(CLOCK_MAX_CALLS);
    }
    let mut per_call: Vec<f64> = (0..CLOCK_SAMPLES)
        .map(|_| run(calls, out) as f64 / calls as f64)
        .collect();
    per_call.sort_by(f64::total_cmp);
    let pixels = (trips.rows * trips.groups) as f64 * LANES as f64;
    Clock {
        ns_per_call_median: per_call[CLOCK_SAMPLES / 2],
        ns_per_call_min: per_call[0],
        ns_per_call_iqr: per_call[(CLOCK_SAMPLES * 3) / 4] - per_call[CLOCK_SAMPLES / 4],
        calls_per_sample: calls,
        ns_per_px: per_call[CLOCK_SAMPLES / 2] / pixels,
        scene_ns_per_px: None,
    }
}

fn cell_grid_scene_ns_per_px(case: &CellGridCase) -> f64 {
    let program = compile_cell_grid_for::<Rgba8>(case.shape, [0.0, 0.0, 0.0, 1.0]);
    let params = program.params(&case.metrics);
    let scene = Scene::CellGrid(program.frame(&params, case.cells.clone(), case.atlas.clone()));
    let (w, h) = (case.shape.frame_w, case.shape.frame_h);
    let mut frame = Frame::<Rgba8>::new(w, h);
    for _ in 0..2 {
        scene.render(&mut frame, 1);
    }
    let mut samples: Vec<f64> = (0..5)
        .map(|_| {
            let t = Instant::now();
            scene.render(&mut frame, 1);
            std::hint::black_box(&frame.data[0]);
            t.elapsed().as_nanos() as f64
        })
        .collect();
    samples.sort_by(f64::total_cmp);
    samples[2] / f64::from(w * h)
}

// ---------------------------------------------------------------------------
// The saturation probe: what fired, what was load-bearing
// ---------------------------------------------------------------------------

/// The legalizing prefix on its own — what both modes run before the
/// switch (`LowerDwrt`, `ExpandReduce`); `eval_scalar` needs it too.
fn legalize(arena: &ExprArena, root: ExprId) -> (ExprArena, ExprId) {
    match pipeline![LowerDwrt, ExpandReduce].optimize(arena, root) {
        Rewritten::Changed(a, r) => (a, r),
        Rewritten::Unchanged => (arena.clone(), root),
        Rewritten::Declined => panic!("legalizing prefix declined a real kernel"),
    }
}

fn saturation_probe(
    arena: &ExprArena,
    root: ExprId,
    shape: LatticeShape,
    variant: Option<Variant>,
    production_bytes: &[u8],
) -> SatProbe {
    let (la, lr) = legalize(arena, root);
    let optimizer = match variant {
        Some(v) => v.optimizer(shape),
        None => Optimizer::production().for_lattice(shape),
    };
    let mut optimizer = optimizer.observe(Some(Box::new(KeepJournal)));
    let mut egraph = optimizer.egraph();
    let root_class = insert(&la, lr, &mut egraph, Vocabulary::Runtime)
        .unwrap_or_else(|_| panic!("probe: real kernel not representable"));
    let node_count = reachable_count(&la, lr);
    let t = Instant::now();
    let optimized = optimizer.run(&mut egraph, root_class, node_count);
    let wall_ms = t.elapsed().as_secs_f64() * 1e3;
    let labels = EpisodeLabels::compute_tight(&egraph, root_class, &optimized.choices);
    let strict = EpisodeLabels::compute_strict(&egraph, root_class, &optimized.choices);
    let rules = optimizer.rule_set();
    let count = |idx: usize| -> RuleCount {
        let s = labels.rule_stats.get(&idx).copied().unwrap_or_default();
        let t = strict.rule_stats.get(&idx).copied().unwrap_or_default();
        RuleCount {
            rule: rules.label_of(idx).unwrap_or_else(|| format!("#{idx}")),
            fired: s.fired,
            load_bearing: s.load_bearing,
            strict: t.load_bearing,
        }
    };
    let select_hoist: Vec<RuleCount> = (0..rules.len())
        .filter(|&i| {
            rules
                .label_of(i)
                .is_some_and(|l| l.contains(SELECT_HOIST_PREFIX))
        })
        .map(count)
        .collect();
    let mut load_bearing_rules: Vec<RuleCount> = labels
        .rule_stats
        .iter()
        .filter(|(_, s)| s.load_bearing > 0)
        .map(|(&i, _)| count(i))
        .collect();
    load_bearing_rules.sort_by(|a, b| {
        b.load_bearing
            .cmp(&a.load_bearing)
            .then(a.rule.cmp(&b.rule))
    });

    // The probe's own extraction, compiled: it must be the production kernel.
    let (pa, pr) = optimized.to_arena(&egraph, root_class);
    let bytes_identical_to_production = if arena.buffers().is_empty() && arena.uniforms().is_empty()
    {
        let compiled = emit::compile(&pa, pr).expect("probe extraction compiles");
        Some(compiled.code.as_bytes() == production_bytes)
    } else {
        None
    };

    SatProbe {
        applications: optimized.stats.applications,
        unions: optimized.stats.unions,
        classes: optimized.stats.classes,
        iterations: optimized.stats.iterations,
        stop: format!("{:?}", optimized.stats.stop),
        wall_ms,
        select_hoist,
        load_bearing_rules,
        bytes_identical_to_production,
    }
}

// ---------------------------------------------------------------------------
// run
// ---------------------------------------------------------------------------

fn mode_label(variant: Option<Variant>) -> String {
    let env = std::env::var("PIXELFLOW_SATURATION").unwrap_or_else(|_| "on".to_string());
    match (variant, env.as_str()) {
        (Some(v), "on") => v.label().to_string(),
        (Some(_), other) => {
            panic!("--variant needs PIXELFLOW_SATURATION unset or on, got {other:?}")
        }
        (None, "on" | "off") => env,
        (None, other) => panic!("PIXELFLOW_SATURATION must be on or off, got {other:?}"),
    }
}

fn head_sha() -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "--short=8", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| "unknown".to_string())
}

struct RunArgs<'a> {
    out: &'a Path,
    variant: Option<Variant>,
    no_clock: bool,
    no_probe: bool,
    filter: Option<&'a str>,
    skip: &'a [String],
    font: Option<&'a Path>,
}

#[allow(clippy::too_many_lines)]
fn run(args: &RunArgs<'_>) {
    let RunArgs {
        out,
        variant,
        no_clock,
        no_probe,
        filter,
        skip,
        font,
    } = *args;
    assert!(
        std::env::var_os("PIXELFLOW_GUARD_TELEMETRY").is_some(),
        "set PIXELFLOW_GUARD_TELEMETRY=1: the guard columns come from the emitter's own report"
    );
    let mode = mode_label(variant);
    let default_font = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../pixelflow-graphics/assets/DejaVuSansMono-Fallback.ttf");
    let mut kernels = real_kernels(font.unwrap_or(&default_font), filter);
    for name in skip {
        let before = kernels.len();
        kernels.retain(|k| &k.name != name);
        assert_eq!(
            kernels.len() + 1,
            before,
            "--skip {name}: not a kernel of this corpus"
        );
        eprintln!("egraph_off_on: skipping {name}");
    }
    eprintln!(
        "egraph_off_on: mode={mode} tier={} lanes={} kernels={}",
        collapse_bench::tier(),
        LANES,
        kernels.len()
    );
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent).expect("create out dir");
    }
    let mut sink = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(out)
        .unwrap_or_else(|e| panic!("open {}: {e}", out.display()));
    let git_sha = head_sha();

    for (i, rk) in kernels.iter().enumerate() {
        let (arena, root) = rk.kernel.parts();
        // A glyph's winding sum reads a bound piece table (S1a,
        // docs/plans/2026-09-09-glyph-as-a-fold-execution.md): its kernel
        // declares a `Buffer` and carries the table's own data
        // (`Kernel::buffer_data`, `bee7813`), with no separate binding for
        // this harness to gather by hand. `context_for`/`oracle` resolve
        // every declared buffer against this before falling back to the
        // cell-grid case's externally-owned memory (`resolve_buffer`).
        let carried: Vec<(pixelflow_ir::arena::BufferIdentity, Arc<[f32]>)> = rk
            .kernel
            .buffer_data()
            .map(|(id, data)| (id, Arc::clone(data)))
            .collect();
        let shape = LatticeShape::new(rk.extent);
        let trips = Trips::of(rk.extent, LANES as u32);
        let started = Instant::now();

        let compiled = match variant {
            Some(v) => compile_via_variant(arena, root, shape, v),
            None => compile_via_production_path(arena, root, shape),
        };
        let bytes_identical_to_manifold_compile =
            if variant.is_none() && arena.buffers().is_empty() && arena.uniforms().is_empty() {
                let m = pixelflow_core::lattice::manifold::Manifold::compile(&rk.kernel, rk.extent);
                Some(m.code_bytes() == compiled.result.code.as_bytes())
            } else {
                None
            };

        let ctx = context_for(&compiled.linked, &carried, rk.cell_grid.as_ref());
        let mut buffer = vec![0.0f32; (trips.rows * trips.groups) as usize * LANES];
        run_once(&compiled.result.code, &ctx, trips, &mut buffer);
        let picture_hash = fnv(&buffer);
        let (legal, legal_root) = legalize(arena, root);
        let oracle = Some(oracle(
            &OracleForms {
                input: (&legal, legal_root),
                linked: (&compiled.linked, compiled.root),
                carried: &carried,
                case: rk.cell_grid.as_ref(),
                packed: rk.packed,
            },
            &buffer,
            trips,
        ));
        let clock = (!no_clock).then(|| {
            let mut c = clock(&compiled.result.code, &ctx, trips, &mut buffer);
            if let Some(case) = rk.cell_grid.as_ref() {
                c.scene_ns_per_px = Some(cell_grid_scene_ns_per_px(case));
            }
            c
        });
        let probe = (!no_probe && mode != "off").then(|| {
            saturation_probe(arena, root, shape, variant, compiled.result.code.as_bytes())
        });

        let row = KernelRow {
            schema: SCHEMA.into(),
            mode: mode.clone(),
            tier: collapse_bench::tier().into(),
            lanes: LANES as u32,
            git_sha: git_sha.clone(),
            name: rk.name.clone(),
            class: rk.class.clone(),
            extent: rk.extent,
            packed: rk.packed,
            input_nodes: reachable(arena, root).len(),
            compiled_nodes: reachable(&compiled.linked, compiled.root).len(),
            dag_cost_input: dag_cost(arena, root),
            dag_cost: dag_cost(&compiled.linked, compiled.root),
            bytes: compiled.result.code.len() as u32,
            spill_slots: compiled.result.spill_count,
            hoisted: compiled.result.hoisted_values,
            statics: features_of(&compiled.result, trips),
            guard: compiled.guard,
            optimize_ms: compiled.optimize_ms,
            emit_ms: compiled.emit_ms,
            bytes_identical_to_manifold_compile,
            picture_hash,
            oracle,
            clock,
            probe,
        };
        let line = serde_json::to_string(&row).expect("serialize row");
        writeln!(sink, "{line}").expect("append row");
        sink.flush().expect("flush");
        eprintln!(
            "[{}/{}] {} {}B dag={} opt={:.1}ms emit={:.1}ms {:?} ({:.1}s)",
            i + 1,
            kernels.len(),
            rk.name,
            row.bytes,
            row.dag_cost,
            row.optimize_ms,
            row.emit_ms,
            row.clock.as_ref().map(|c| c.ns_per_px),
            started.elapsed().as_secs_f64()
        );
    }
}

// ---------------------------------------------------------------------------
// diff
// ---------------------------------------------------------------------------

fn read_rows(paths: &[PathBuf]) -> Vec<KernelRow> {
    let mut rows = Vec::new();
    for p in paths {
        let text =
            std::fs::read_to_string(p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()));
        for line in text.lines().filter(|l| !l.trim().is_empty()) {
            let row: KernelRow = serde_json::from_str(line).expect("parse row");
            assert_eq!(row.schema, SCHEMA, "{}: schema mismatch", p.display());
            rows.push(row);
        }
    }
    rows
}

/// One row per kernel: deterministic columns from the first file, the
/// clock as the median over every file's measurement.
fn collapse(rows: Vec<KernelRow>) -> BTreeMap<String, KernelRow> {
    let mut by: BTreeMap<String, Vec<KernelRow>> = BTreeMap::new();
    for r in rows {
        by.entry(r.name.clone()).or_default().push(r);
    }
    by.into_iter()
        .map(|(name, mut rs)| {
            for r in &rs[1..] {
                assert_eq!(r.bytes, rs[0].bytes, "{name}: bytes differ between passes");
                assert_eq!(
                    r.picture_hash, rs[0].picture_hash,
                    "{name}: picture differs between passes"
                );
            }
            let mut clocks: Vec<f64> = rs
                .iter()
                .filter_map(|r| r.clock.as_ref().map(|c| c.ns_per_px))
                .collect();
            clocks.sort_by(f64::total_cmp);
            let mut scene: Vec<f64> = rs
                .iter()
                .filter_map(|r| r.clock.as_ref().and_then(|c| c.scene_ns_per_px))
                .collect();
            scene.sort_by(f64::total_cmp);
            let mut opt: Vec<f64> = rs.iter().map(|r| r.optimize_ms).collect();
            opt.sort_by(f64::total_cmp);
            let mut em: Vec<f64> = rs.iter().map(|r| r.emit_ms).collect();
            em.sort_by(f64::total_cmp);
            let mut r = rs.swap_remove(0);
            // The clock may sit in a later file than the deterministic columns
            // (a `--no-clock` census merged with clocked passes): take the
            // first clock any pass carries, then median over all of them.
            let mut clock = r
                .clock
                .take()
                .or_else(|| rs.iter_mut().find_map(|x| x.clock.take()));
            if let Some(c) = clock.as_mut() {
                c.ns_per_px = clocks[clocks.len() / 2];
                c.scene_ns_per_px = (!scene.is_empty()).then(|| scene[scene.len() / 2]);
            }
            r.clock = clock;
            r.optimize_ms = opt[opt.len() / 2];
            r.emit_ms = em[em.len() / 2];
            (name, r)
        })
        .collect()
}

fn pct(on: f64, off: f64) -> f64 {
    if off == 0.0 {
        0.0
    } else {
        (on / off - 1.0) * 100.0
    }
}

fn fmt_pct(p: f64) -> String {
    format!("{p:+.1}%")
}

struct DiffInputs<'a> {
    off: &'a [PathBuf],
    on: &'a [PathBuf],
    with_select_hoist: &'a [PathBuf],
    cse_only: &'a [PathBuf],
}

struct Cmp {
    name: String,
    class: String,
    bytes_pct: f64,
    dag_pct: f64,
    clock_pct: Option<f64>,
    off_ns: Option<f64>,
    on_ns: Option<f64>,
    off_bytes: u32,
    on_bytes: u32,
    off_dag: usize,
    on_dag: usize,
    off_mem: u64,
    on_mem: u64,
    picture_identical: bool,
    rules: Vec<RuleCount>,
    sat_ms: f64,
    total_on_ms: f64,
    on_guard: Option<GuardTelemetry>,
    /// The `with-select-hoist` arm.
    sh_fired: usize,
    sh_lb: usize,
    sh_strict: usize,
    sh_bytes: Option<u32>,
    sh_guard: Option<GuardTelemetry>,
    sh_dag: Option<usize>,
    /// The `cse-only` arm.
    cse_bytes: Option<u32>,
    cse_dag: Option<usize>,
    cse_nodes: Option<usize>,
}

fn rules_cell(rules: &[RuleCount]) -> String {
    let v: Vec<String> = rules
        .iter()
        .take(5)
        .map(|r| format!("`{}` {}/{}/{}", r.rule, r.strict, r.load_bearing, r.fired))
        .collect();
    if v.is_empty() {
        "-".into()
    } else {
        v.join(", ")
    }
}

fn opt_str<T: std::fmt::Display>(v: Option<T>) -> String {
    v.map_or("-".into(), |v| v.to_string())
}

fn opt_f(v: Option<f64>) -> String {
    v.map_or("-".into(), |v| format!("{v:.2}"))
}

fn guard_str(g: &Option<GuardTelemetry>) -> String {
    g.as_ref()
        .map_or("-".to_string(), |g| format!("{}/{}", g.guarded, g.selects))
}

fn sched_str(g: &Option<GuardTelemetry>) -> String {
    g.as_ref()
        .map_or("-".to_string(), |g| g.schedule.to_string())
}

fn probe_str(r: &KernelRow) -> String {
    r.probe.as_ref().map_or("-".into(), |p| {
        format!(
            "{} apps / {} it / {} cls / {}",
            p.applications, p.iterations, p.classes, p.stop
        )
    })
}

#[allow(clippy::too_many_lines)]
fn diff(inputs: &DiffInputs<'_>, out_prefix: &Path, notes: &[String]) {
    let off = collapse(read_rows(inputs.off));
    let on = collapse(read_rows(inputs.on));
    let sh = collapse(read_rows(inputs.with_select_hoist));
    let cse = collapse(read_rows(inputs.cse_only));
    let names: Vec<String> = on
        .keys()
        .filter(|n| off.contains_key(*n))
        .cloned()
        .collect();
    assert!(!names.is_empty(), "no kernel present in both off and on");
    let on_only: Vec<&KernelRow> = on.values().filter(|r| !off.contains_key(&r.name)).collect();

    // ---- CSV -------------------------------------------------------------
    let mut csv = String::from(
        "kernel,class,extent_w,extent_h,packed,input_nodes,off_nodes,on_nodes,cse_nodes,input_dag_cost,off_dag_cost,on_dag_cost,cse_dag_cost,sh_dag_cost,\
         off_bytes,on_bytes,cse_bytes,sh_bytes,off_schedule,on_schedule,off_selects,on_selects,off_guarded,on_guarded,sh_guarded,\
         off_spill,on_spill,off_dyn_mem_ops,on_dyn_mem_ops,off_ns_px,on_ns_px,clock_pct,\
         off_optimize_ms,on_optimize_ms,on_emit_ms,on_applications,on_iterations,on_classes,on_stop,picture_identical,\
         same_form_max_abs_off,same_form_max_abs_on,cross_form_max_abs_on,packed_mismatch_cross_on,\
         select_hoist_fired,select_hoist_load_bearing,select_hoist_strict,top_load_bearing_rule\n",
    );
    let mut json_rows = Vec::new();
    let mut cmps: Vec<Cmp> = Vec::new();
    for name in &names {
        let a = &off[name];
        let b = &on[name];
        let h = sh.get(name);
        let c = cse.get(name);
        let clock_pct = match (a.clock.as_ref(), b.clock.as_ref()) {
            (Some(x), Some(y)) => Some(pct(y.ns_per_px, x.ns_per_px)),
            _ => None,
        };
        let hp = h.and_then(|h| h.probe.as_ref());
        let sh_fired: usize = hp.map_or(0, |p| p.select_hoist.iter().map(|r| r.fired).sum());
        let sh_lb: usize = hp.map_or(0, |p| p.select_hoist.iter().map(|r| r.load_bearing).sum());
        let sh_strict: usize = hp.map_or(0, |p| p.select_hoist.iter().map(|r| r.strict).sum());
        let rules = b
            .probe
            .as_ref()
            .map(|p| p.load_bearing_rules.clone())
            .unwrap_or_default();
        let top_rule = rules
            .first()
            .map(|r| format!("{} ({}/{}/{})", r.rule, r.strict, r.load_bearing, r.fired))
            .unwrap_or_else(|| "-".into());
        let g = |g: &Option<GuardTelemetry>, f: fn(&GuardTelemetry) -> u64| g.as_ref().map_or(0, f);
        let o = |o: &Option<Oracle>| o.clone().unwrap_or_default();
        let pb = b.probe.as_ref();
        csv.push_str(&format!(
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{:.2},{:.2},{:.2},{},{},{},{},{},{:e},{:e},{:e},{},{},{},{},{}\n",
            name, b.class, b.extent[0], b.extent[1], b.packed, b.input_nodes, a.compiled_nodes, b.compiled_nodes,
            opt_str(c.map(|c| c.compiled_nodes)),
            b.dag_cost_input, a.dag_cost, b.dag_cost, opt_str(c.map(|c| c.dag_cost)), opt_str(h.map(|h| h.dag_cost)),
            a.bytes, b.bytes, opt_str(c.map(|c| c.bytes)), opt_str(h.map(|h| h.bytes)),
            g(&a.guard, |g| g.schedule), g(&b.guard, |g| g.schedule),
            g(&a.guard, |g| g.selects), g(&b.guard, |g| g.selects),
            g(&a.guard, |g| g.guarded), g(&b.guard, |g| g.guarded), opt_str(h.map(|h| g(&h.guard, |g| g.guarded))),
            a.spill_slots, b.spill_slots, a.statics.dyn_memory_ops, b.statics.dyn_memory_ops,
            a.clock.as_ref().map_or(String::from("-"), |c| format!("{:.3}", c.ns_per_px)),
            b.clock.as_ref().map_or(String::from("-"), |c| format!("{:.3}", c.ns_per_px)),
            clock_pct.map_or(String::from("-"), |p| format!("{p:.1}")),
            a.optimize_ms, b.optimize_ms, b.emit_ms,
            opt_str(pb.map(|p| p.applications)), opt_str(pb.map(|p| p.iterations)), opt_str(pb.map(|p| p.classes)),
            pb.map_or("-".into(), |p| p.stop.clone()),
            a.picture_hash == b.picture_hash,
            o(&a.oracle).same_form_max_abs, o(&b.oracle).same_form_max_abs, o(&b.oracle).cross_form_max_abs,
            o(&b.oracle).packed_mismatch_cross,
            sh_fired, sh_lb, sh_strict,
            top_rule.replace(',', ";"),
        ));
        json_rows.push(serde_json::json!({
            "kernel": name, "class": b.class, "extent": b.extent, "packed": b.packed,
            "off": a, "on": b, "with_select_hoist": h, "cse_only": c,
        }));
        cmps.push(Cmp {
            name: name.clone(),
            class: b.class.clone(),
            bytes_pct: pct(f64::from(b.bytes), f64::from(a.bytes)),
            dag_pct: pct(b.dag_cost as f64, a.dag_cost as f64),
            clock_pct,
            off_ns: a.clock.as_ref().map(|c| c.ns_per_px),
            on_ns: b.clock.as_ref().map(|c| c.ns_per_px),
            off_bytes: a.bytes,
            on_bytes: b.bytes,
            off_dag: a.dag_cost,
            on_dag: b.dag_cost,
            off_mem: a.statics.dyn_memory_ops,
            on_mem: b.statics.dyn_memory_ops,
            picture_identical: a.picture_hash == b.picture_hash,
            rules,
            sat_ms: b.optimize_ms - a.optimize_ms,
            total_on_ms: b.optimize_ms + b.emit_ms,
            on_guard: b.guard.clone(),
            sh_fired,
            sh_lb,
            sh_strict,
            sh_bytes: h.map(|h| h.bytes),
            sh_guard: h.and_then(|h| h.guard.clone()),
            sh_dag: h.map(|h| h.dag_cost),
            cse_bytes: c.map(|c| c.bytes),
            cse_dag: c.map(|c| c.dag_cost),
            cse_nodes: c.map(|c| c.compiled_nodes),
        });
    }
    std::fs::write(out_prefix.with_extension("csv"), &csv).expect("write csv");

    // ---- per-class verdicts ---------------------------------------------
    let mut classes: BTreeMap<String, Vec<&Cmp>> = BTreeMap::new();
    for c in &cmps {
        classes.entry(c.class.clone()).or_default().push(c);
    }
    let median = |mut v: Vec<f64>| -> Option<f64> {
        if v.is_empty() {
            return None;
        }
        v.sort_by(f64::total_cmp);
        Some(v[v.len() / 2])
    };
    let mut md = String::new();
    md.push_str("# Saturation on vs off, on every shipped kernel\n\n");
    md.push_str(
        "The \"F: no e-graph\" column of [`2026-09-06-egraph-at-production-scale.md`](../plans/2026-09-06-egraph-at-production-scale.md) §7, \
         measured. Every kernel is compiled through the production path (`optimize_runtime_arena` → `relink` → `emit::compile`, the three calls \
         `jit_cache::compile` makes; buffer-free kernels are asserted byte-identical to `Manifold::compile`) twice: `PIXELFLOW_SATURATION=off` \
         runs the `Identity` path (`LowerDwrt`, `ExpandReduce`, no saturation), `on` is production. Two in-harness arms beside them: `cse-only` \
         (the production optimizer at zero rewrite rounds — insert, extract — so hash-consing's share is separated from the rules') and \
         `with-select-hoist` (production's rules plus the three `SelectHoistUnary` rules, which are **not** in `all_rules()`). \
         Deterministic columns are the claim; the clock, when taken, is a sign. Per-kernel rows: the `.csv`/`.json` beside this file.\n\n",
    );
    for n in notes {
        md.push_str(n);
        md.push('\n');
    }
    md.push('\n');
    md.push_str("## Verdict per shader class\n\n");
    md.push_str("`Δ` is on relative to off; negative is saturation winning. `mem ops` is the trip-weighted dynamic memory-op count (`dyn_memory_ops`). \
                 `cse` is the zero-round arm: the gap between `off` and `cse` is hash-consing, between `cse` and `on` is the rules.\n\n");
    md.push_str("| class | n | Σ bytes off → cse → on | Σ dag_cost off → cse → on | Σ mem ops off → on | median clock Δ | picture identical | verdict |\n|---|---:|---:|---:|---:|---:|---:|---|\n");
    let mut json_classes = Vec::new();
    for (class, cs) in &classes {
        let sb: (u64, u64) = cs.iter().fold((0, 0), |a, c| {
            (a.0 + u64::from(c.off_bytes), a.1 + u64::from(c.on_bytes))
        });
        let sd: (usize, usize) = cs
            .iter()
            .fold((0, 0), |a, c| (a.0 + c.off_dag, a.1 + c.on_dag));
        let sm: (u64, u64) = cs
            .iter()
            .fold((0, 0), |a, c| (a.0 + c.off_mem, a.1 + c.on_mem));
        let all_cse = cs.iter().all(|c| c.cse_bytes.is_some());
        let cse_b: u64 = cs.iter().map(|c| u64::from(c.cse_bytes.unwrap_or(0))).sum();
        let cse_d: usize = cs.iter().map(|c| c.cse_dag.unwrap_or(0)).sum();
        let clocks: Vec<f64> = cs.iter().filter_map(|c| c.clock_pct).collect();
        let mclock = median(clocks);
        let identical = cs.iter().filter(|c| c.picture_identical).count();
        let bytes_pct = pct(sb.1 as f64, sb.0 as f64);
        let dag_pct = pct(sd.1 as f64, sd.0 as f64);
        let verdict = match (bytes_pct, dag_pct, mclock) {
            (b, d, Some(c)) if (b < -2.0 || d < -2.0) && c < -10.0 => "helps (bytes+clock)",
            (b, d, _) if b < -2.0 || d < -2.0 => "helps (static)",
            (b, d, Some(c)) if (b > 2.0 || d > 2.0) && c > 10.0 => "HURTS (bytes+clock)",
            (b, d, _) if b > 2.0 || d > 2.0 => "hurts (static)",
            (_, _, Some(c)) if c < -10.0 => "clock says helps; static flat",
            (_, _, Some(c)) if c > 10.0 => "clock says hurts; static flat",
            _ => "nothing (|Δ| ≤ 2% static)",
        };
        let cse_bs = if all_cse {
            format!("{cse_b}")
        } else {
            "-".into()
        };
        let cse_ds = if all_cse {
            format!("{cse_d}")
        } else {
            "-".into()
        };
        md.push_str(&format!(
            "| {class} | {} | {} → {} → {} ({}) | {} → {} → {} ({}) | {} → {} ({}) | {} | {identical}/{} | {verdict} |\n",
            cs.len(), sb.0, cse_bs, sb.1, fmt_pct(bytes_pct), sd.0, cse_ds, sd.1, fmt_pct(dag_pct), sm.0, sm.1, fmt_pct(pct(sm.1 as f64, sm.0 as f64)),
            mclock.map_or("-".into(), fmt_pct), cs.len()
        ));
        json_classes.push(serde_json::json!({
            "class": class, "n": cs.len(), "bytes_off": sb.0, "bytes_on": sb.1, "bytes_pct": bytes_pct,
            "bytes_cse": all_cse.then_some(cse_b), "dag_cse": all_cse.then_some(cse_d),
            "dag_off": sd.0, "dag_on": sd.1, "dag_pct": dag_pct, "mem_off": sm.0, "mem_on": sm.1,
            "median_clock_pct": mclock, "picture_identical": identical, "verdict": verdict,
        }));
    }

    // ---- kernels the off path could not emit ------------------------------
    if !on_only.is_empty() {
        md.push_str("\n## Kernels with no `off` row: the un-saturated arena does not compile\n\n");
        md.push_str("Run with `--skip` in `off` mode because the emitter panics on them (the notes above quote the panic). The columns are the `on` row and, where present, the `cse-only` arm.\n\n");
        md.push_str("| kernel | extent | nodes in → cse → on | dag_cost in → cse → on | bytes cse → on | schedule on | guarded/selects on | spills on | saturation (apps / rounds / classes / stop) | compile on (ms) |\n|---|---|---|---:|---:|---:|---:|---:|---|---:|\n");
        for b in &on_only {
            let c = cse.get(&b.name);
            md.push_str(&format!(
                "| {} | {}×{} | {} → {} → {} | {} → {} → {} | {} → {} | {} | {} | {} | {} | {:.1} |\n",
                b.name, b.extent[0], b.extent[1], b.input_nodes, opt_str(c.map(|c| c.compiled_nodes)), b.compiled_nodes,
                b.dag_cost_input, opt_str(c.map(|c| c.dag_cost)), b.dag_cost, opt_str(c.map(|c| c.bytes)), b.bytes,
                sched_str(&b.guard), guard_str(&b.guard), b.spill_slots, probe_str(b), b.optimize_ms + b.emit_ms
            ));
        }
    }

    // ---- headline kernels ------------------------------------------------
    md.push_str("\n## The headline kernels\n\n| kernel | extent | nodes in → off → cse → on | bytes off → cse → on | dag_cost off → cse → on | schedule off → on | guarded/selects off → on | spills off → on | mem ops off → on | saturation (apps / rounds / classes / stop) | compile off → on (ms; saturation share) |\n|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|\n");
    for c in &cmps {
        if !matches!(
            c.class.as_str(),
            "chrome"
                | "chrome_channel"
                | "psychedelic"
                | "cellgrid"
                | "bench"
                | "bench_wide"
                | "shader"
        ) {
            continue;
        }
        let a = &off[&c.name];
        let b = &on[&c.name];
        let share = if c.total_on_ms > 0.0 {
            c.sat_ms / c.total_on_ms * 100.0
        } else {
            0.0
        };
        md.push_str(&format!(
            "| {} | {}×{} | {} → {} → {} → {} | {} → {} → {} ({}) | {} → {} → {} ({}) | {} → {} | {} → {} | {} → {} | {} → {} ({}) | {} | {:.1} → {:.1} ({:.0}%) |\n",
            c.name, b.extent[0], b.extent[1], b.input_nodes, a.compiled_nodes, opt_str(c.cse_nodes), b.compiled_nodes,
            c.off_bytes, opt_str(c.cse_bytes), c.on_bytes, fmt_pct(c.bytes_pct), c.off_dag, opt_str(c.cse_dag), c.on_dag, fmt_pct(c.dag_pct),
            sched_str(&a.guard), sched_str(&b.guard), guard_str(&a.guard), guard_str(&b.guard), a.spill_slots, b.spill_slots,
            c.off_mem, c.on_mem, fmt_pct(pct(c.on_mem as f64, c.off_mem as f64)),
            probe_str(b),
            a.optimize_ms + a.emit_ms, c.total_on_ms, share
        ));
    }

    // ---- prologue on O@32 -------------------------------------------------
    if let (Some(n), Some(w)) = (
        cmps.iter().find(|c| c.name == "bench_O_quadratic"),
        cmps.iter().find(|c| c.name == "bench_O_quadratic_wide"),
    ) && n.off_ns.is_some()
        && w.off_ns.is_some()
    {
        let prologue = |narrow: Option<f64>, wide: Option<f64>| -> Option<f64> {
            let (tn, tw) = (
                narrow? * f64::from(BENCH_EXTENT[0] * BENCH_EXTENT[1]),
                wide? * f64::from(BENCH_WIDE_EXTENT[0] * BENCH_WIDE_EXTENT[1]),
            );
            let rows = f64::from(BENCH_EXTENT[1]);
            let (gn, gw) = (
                f64::from(BENCH_EXTENT[0]) / LANES as f64,
                f64::from(BENCH_WIDE_EXTENT[0]) / LANES as f64,
            );
            let per_row_n = tn / rows;
            let per_row_w = tw / rows;
            let b = (per_row_w - per_row_n) / (gw - gn);
            Some((per_row_n - b * gn) / 1e3)
        };
        md.push_str(&format!(
                "\n## Row prologue on `O`@32 (from the 40- and 640-wide rows, two-point fit)\n\n| | off | on |\n|---|---:|---:|\n| ns/px at 40×45 | {} | {} |\n| ns/px at 640×45 | {} | {} |\n| per-row prologue (µs) | {} | {} |\n",
                opt_f(n.off_ns), opt_f(n.on_ns), opt_f(w.off_ns), opt_f(w.on_ns),
                opt_f(prologue(n.off_ns, w.off_ns)), opt_f(prologue(n.on_ns, w.on_ns)),
            ));
    }

    // ---- regressions and their rules -------------------------------------
    md.push_str("\n## Where saturation makes the kernel worse, and which rule\n\n");
    md.push_str("Sorted by bytes Δ. Rule column: the on-extraction's rules from the provenance journal, `strict/tight/fired` — strict credits an application only when its own e-node was chosen (`EpisodeLabels::compute_strict`), tight is `derivation_ancestors_tight`.\n\n");
    md.push_str("| kernel | bytes off → cse → on | dag_cost off → cse → on | mem ops Δ | rules (strict/tight/fired) |\n|---|---:|---:|---:|---|\n");
    let mut worse: Vec<&Cmp> = cmps
        .iter()
        .filter(|c| c.bytes_pct > 0.0 || c.dag_pct > 0.0)
        .collect();
    worse.sort_by(|a, b| b.bytes_pct.partial_cmp(&a.bytes_pct).unwrap());
    for c in worse.iter().take(15) {
        md.push_str(&format!(
            "| {} | {} → {} → {} ({}) | {} → {} → {} ({}) | {} | {} |\n",
            c.name,
            c.off_bytes,
            opt_str(c.cse_bytes),
            c.on_bytes,
            fmt_pct(c.bytes_pct),
            c.off_dag,
            opt_str(c.cse_dag),
            c.on_dag,
            fmt_pct(c.dag_pct),
            fmt_pct(pct(c.on_mem as f64, c.off_mem as f64)),
            rules_cell(&c.rules)
        ));
    }
    if worse.is_empty() {
        md.push_str("| (none: no kernel grew in bytes or dag_cost) | | | | |\n");
    }
    let mut better: Vec<&Cmp> = cmps.iter().filter(|c| c.bytes_pct < 0.0).collect();
    better.sort_by(|a, b| a.bytes_pct.partial_cmp(&b.bytes_pct).unwrap());
    md.push_str("\n### The largest wins, for the same rule attribution\n\n| kernel | bytes off → cse → on | dag_cost off → cse → on | rules (strict/tight/fired) |\n|---|---:|---:|---|\n");
    for c in better.iter().take(10) {
        md.push_str(&format!(
            "| {} | {} → {} → {} ({}) | {} → {} → {} ({}) | {} |\n",
            c.name,
            c.off_bytes,
            opt_str(c.cse_bytes),
            c.on_bytes,
            fmt_pct(c.bytes_pct),
            c.off_dag,
            opt_str(c.cse_dag),
            c.on_dag,
            fmt_pct(c.dag_pct),
            rules_cell(&c.rules)
        ));
    }

    // ---- rule census -----------------------------------------------------
    let mut census: BTreeMap<String, (usize, usize, usize, usize)> = BTreeMap::new();
    for c in &cmps {
        for r in &c.rules {
            let e = census.entry(r.rule.clone()).or_default();
            e.0 += r.fired;
            e.1 += r.load_bearing;
            e.2 += r.strict;
            e.3 += 1;
        }
    }
    let mut census: Vec<(String, (usize, usize, usize, usize))> = census.into_iter().collect();
    census.sort_by_key(|(_, (_, _, strict, _))| std::cmp::Reverse(*strict));
    md.push_str("\n## Which rules are load-bearing on real shaders (all kernels, production run)\n\n| rule | kernels where load-bearing | Σ strict | Σ tight | Σ fired |\n|---|---:|---:|---:|---:|\n");
    for (rule, (fired, lb, strict, n)) in census.iter().take(25) {
        md.push_str(&format!("| `{rule}` | {n} | {strict} | {lb} | {fired} |\n"));
    }

    // ---- SelectHoistUnary --------------------------------------------------
    let total_fired: usize = cmps.iter().map(|c| c.sh_fired).sum();
    let total_lb: usize = cmps.iter().map(|c| c.sh_lb).sum();
    let total_strict: usize = cmps.iter().map(|c| c.sh_strict).sum();
    let fired_in: usize = cmps.iter().filter(|c| c.sh_fired > 0).count();
    let lb_in: usize = cmps.iter().filter(|c| c.sh_lb > 0).count();
    let measured = cmps.iter().filter(|c| c.sh_bytes.is_some()).count();
    md.push_str(&format!(
        "\n## `SelectHoistUnary` (`select-hoist-neg|abs|sqrt`)\n\n\
         **Not in production.** The three rules live in `round2_rules::experimental_rules()` and are not part of `all_rules()`, \
         so they fire zero times in every production compile above by construction. The `with-select-hoist` arm adds them to \
         production's rules: fired in **{fired_in} of {measured}** measured kernels ({total_fired} applications); tight-load-bearing in \
         **{lb_in}** ({total_lb}); strict {total_strict}.\n\n",
    ));
    md.push_str("| kernel | fired | tight | strict | bytes on → +hoist | dag_cost on → +hoist | guarded/selects on → +hoist | schedule on → +hoist |\n|---|---:|---:|---:|---:|---:|---:|---:|\n");
    let mut shv: Vec<&Cmp> = cmps
        .iter()
        .filter(|c| c.sh_fired > 0 || c.sh_bytes.is_some_and(|m| m != c.on_bytes))
        .collect();
    shv.sort_by(|a, b| b.sh_lb.cmp(&a.sh_lb).then(b.sh_fired.cmp(&a.sh_fired)));
    for c in shv.iter().take(30) {
        md.push_str(&format!(
            "| {} | {} | {} | {} | {} → {} | {} → {} | {} → {} | {} → {} |\n",
            c.name,
            c.sh_fired,
            c.sh_lb,
            c.sh_strict,
            c.on_bytes,
            opt_str(c.sh_bytes),
            c.on_dag,
            opt_str(c.sh_dag),
            guard_str(&c.on_guard),
            guard_str(&c.sh_guard),
            sched_str(&c.on_guard),
            sched_str(&c.sh_guard),
        ));
    }
    if shv.is_empty() {
        md.push_str(
            "| (never fired on any real shader; bytes identical everywhere) | | | | | | | |\n",
        );
    }
    let sh_changed = cmps
        .iter()
        .filter(|c| c.sh_bytes.is_some_and(|m| m != c.on_bytes))
        .count();
    let guard_delta: i64 = cmps
        .iter()
        .filter_map(|c| {
            Some(
                i64::try_from(c.sh_guard.as_ref()?.guarded).ok()?
                    - i64::try_from(c.on_guard.as_ref()?.guarded).ok()?,
            )
        })
        .sum();
    md.push_str(&format!("\nAdding the rule changed the emitted bytes of **{sh_changed}** kernels; Σ guarded-value delta (+hoist − on) over all measured kernels: **{guard_delta:+}**.\n"));

    // ---- correctness -----------------------------------------------------
    let same_worst = cmps
        .iter()
        .filter_map(|c| {
            on[&c.name].oracle.as_ref().map(|o| {
                (
                    c.name.clone(),
                    o.same_form_max_abs,
                    o.same_form_nan_mismatch,
                )
            })
        })
        .max_by(|a, b| a.1.total_cmp(&b.1));
    let cross_worst = cmps
        .iter()
        .filter_map(|c| {
            on[&c.name].oracle.as_ref().map(|o| {
                (
                    c.name.clone(),
                    o.cross_form_max_abs,
                    o.cross_form_nan_mismatch,
                )
            })
        })
        .max_by(|a, b| a.1.total_cmp(&b.1));
    let nan_same: usize = cmps
        .iter()
        .filter_map(|c| on[&c.name].oracle.as_ref())
        .map(|o| o.same_form_nan_mismatch)
        .sum();
    let nan_same_off: usize = cmps
        .iter()
        .filter_map(|c| off[&c.name].oracle.as_ref())
        .map(|o| o.same_form_nan_mismatch)
        .sum();
    let same_worst_off = cmps
        .iter()
        .filter_map(|c| {
            off[&c.name]
                .oracle
                .as_ref()
                .map(|o| (c.name.clone(), o.same_form_max_abs))
        })
        .max_by(|a, b| a.1.total_cmp(&b.1));
    let packed_cross: Vec<String> = on.values().filter(|r| r.packed).map(|r| {
        let o = r.oracle.clone().unwrap_or_default();
        format!("`{}`: same-form {} mismatching of {} sampled pixels (max byte Δ {}), cross-form {} (max byte Δ {})", r.name, o.packed_mismatch_same, o.points, o.packed_max_byte_same, o.packed_mismatch_cross, o.packed_max_byte_cross)
    }).collect();
    let identical = cmps.iter().filter(|c| c.picture_identical).count();
    md.push_str(&format!(
        "\n## Correctness\n\n\
         Same-form: `eval_scalar` of the emitted arena vs the JIT at {ORACLE_POINTS} sampled pixels (a difference here is a JIT bug). Cross-form: `eval_scalar` of the legalized arena as constructed vs the JIT of the on-extraction (what the rewrites moved; divergence at singularities is the algebraic contract, not a defect).\n\n\
         - same-form NaN mismatches over all kernels: on **{nan_same}**, off **{nan_same_off}**; worst same-form |Δ|: on {}, off {}\n\
         - worst cross-form |Δ| (on): {}\n\
         - full-extent output bit-identical off vs on: **{identical} of {}** kernels\n\
         - packed kernels (on): {}\n",
        same_worst.map_or("-".into(), |(n, d, nan)| format!("`{n}` {d:e} ({nan} NaN)")),
        same_worst_off.map_or("-".into(), |(n, d)| format!("`{n}` {d:e}")),
        cross_worst.map_or("-".into(), |(n, d, nan)| format!("`{n}` {d:e} ({nan} NaN)")),
        cmps.len(),
        packed_cross.join("; "),
    ));
    let manifold_check: Vec<&str> = on
        .keys()
        .filter(|n| {
            on[*n].bytes_identical_to_manifold_compile == Some(false)
                || off
                    .get(*n)
                    .is_some_and(|r| r.bytes_identical_to_manifold_compile == Some(false))
        })
        .map(String::as_str)
        .collect();
    let probe_check: Vec<&str> = on
        .keys()
        .filter(|n| {
            on[*n]
                .probe
                .as_ref()
                .is_some_and(|p| p.bytes_identical_to_production == Some(false))
        })
        .map(String::as_str)
        .collect();
    md.push_str(&format!(
        "- instrument = production path: `Manifold::compile` bytes differed for {} kernels {:?}; probe extraction differed from production for {} kernels {:?}\n",
        manifold_check.len(), manifold_check, probe_check.len(), probe_check
    ));

    // ---- compile-time share ---------------------------------------------
    md.push_str("\n## Saturation's share of compile time\n\n`optimize_ms` is `optimize_runtime_arena` (legalize + saturate + extract); `emit_ms` is `emit::compile`. Wall clock at the load stated above — a ratio, not a number.\n\n| class | Σ compile off (ms) | Σ compile on (ms) | Σ saturation (on − off optimize) | share of on |\n|---|---:|---:|---:|---:|\n");
    for (class, cs) in &classes {
        let off_ms: f64 = cs
            .iter()
            .map(|c| off[&c.name].optimize_ms + off[&c.name].emit_ms)
            .sum();
        let on_ms: f64 = cs.iter().map(|c| c.total_on_ms).sum();
        let sat: f64 = cs.iter().map(|c| c.sat_ms).sum();
        md.push_str(&format!(
            "| {class} | {off_ms:.1} | {on_ms:.1} | {sat:.1} | {:.0}% |\n",
            if on_ms > 0.0 {
                sat / on_ms * 100.0
            } else {
                0.0
            }
        ));
    }
    for b in &on_only {
        md.push_str(&format!(
            "| {} (on only) | - | {:.1} | {:.1} | - |\n",
            b.name,
            b.optimize_ms + b.emit_ms,
            b.optimize_ms
        ));
    }

    std::fs::write(out_prefix.with_extension("md"), &md).expect("write md");
    let json = serde_json::json!({
        "schema": SCHEMA,
        "classes": json_classes,
        "on_only": on_only,
        "select_hoist": { "in_production": false, "fired_total": total_fired, "tight_total": total_lb, "strict_total": total_strict, "kernels_fired": fired_in, "kernels_tight": lb_in, "changed_bytes": sh_changed, "guarded_delta_hoist_minus_on": guard_delta },
        "rule_census": census.iter().map(|(r, (f, lb, strict, n))| serde_json::json!({"rule": r, "fired": f, "tight": lb, "strict": strict, "kernels": n})).collect::<Vec<_>>(),
        "kernels": json_rows,
    });
    std::fs::write(
        out_prefix.with_extension("json"),
        serde_json::to_string_pretty(&json).expect("json"),
    )
    .expect("write json");
    print!("{md}");
}

fn main() {
    match Cli::parse().command {
        Command::Run {
            out,
            variant,
            no_clock,
            no_probe,
            filter,
            skip,
            font,
        } => run(&RunArgs {
            out: &out,
            variant,
            no_clock,
            no_probe,
            filter: filter.as_deref(),
            skip: &skip,
            font: font.as_deref(),
        }),
        Command::Diff {
            off,
            on,
            with_select_hoist,
            cse_only,
            out_prefix,
            note,
        } => diff(
            &DiffInputs {
                off: &off,
                on: &on,
                with_select_hoist: &with_select_hoist,
                cse_only: &cse_only,
            },
            &out_prefix,
            &note,
        ),
    }
}
