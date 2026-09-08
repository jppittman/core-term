//! The shipped-kernel corpus, by family, and the static columns every
//! harness reads off a compiled kernel.
//!
//! One definition, shared by `egraph_off_on` (the F measurement) and
//! `rules_filter` (the rules × nodes filter evaluation): the DEV / HELD-OUT
//! split of docs/plans/2026-09-07-benchmark-correction.md §B is a property
//! of *these* kernels, and two enumerations of them would be two corpora.
//!
//! `real_kernels` enumerates, for one font: the chrome sphere (packed, plus
//! its red channel — HELD-OUT), the psychedelic shader (packed, with its
//! clock uniform), the terminal cell grid at a Retina 80×24 geometry, the
//! 190 glyph bakes `GlyphAtlas::warm` performs plus the three
//! `font_rendering` bench glyphs (and `O`@32 at a 640-wide row), and the
//! twelve `shader_bench` ShaderToy ports.

use std::path::Path;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use pixelflow_core::lattice::cell_grid::{CELL_STRIDE, CellGridMetrics};
use pixelflow_core::{Bits, CellGridShape, Kernel, Uniform};
use pixelflow_graphics::fonts::{Font, GlyphAtlas};
use pixelflow_graphics::render::color::Rgba8;
use pixelflow_graphics::render::pixel::Pixel;
use pixelflow_graphics::scene3d::{Hit, Plane, Ray, Rgba, Sphere, checker, sky};
use pixelflow_ir::optimize::{Optimize, Rewritten};
use pixelflow_ir::passes::{ExpandReduce, LowerDwrt};
use pixelflow_ir::{ExprArena, ExprId, ExprNode, pipeline};
use pixelflow_search::egraph::CostModel;

use crate::shader_bench::{SHADERTOY_KERNEL_NAMES, named_shadertoy_kernel};

pub const SCREEN: [u32; 2] = [1920, 1080];
pub const SHADER_EXTENT: [u32; 2] = [256, 256];
pub const CELL_HEIGHT_PT: f32 = 16.0;
pub const CELL_WIDTH_PT: f32 = 10.0;
pub const ATLAS_CAPACITY: usize = 128;
pub const DENSITIES: [f32; 2] = [1.0, 2.0];
pub const WARM_RANGE: std::ops::RangeInclusive<char> = ' '..='~';
pub const BENCH_CHARS: [(&str, char); 3] =
    [("A_linear", 'A'), ("O_quadratic", 'O'), ("S_complex", 'S')];
pub const BENCH_PT: f32 = 32.0;
pub const BENCH_EXTENT: [u32; 2] = [40, 45];
pub const BENCH_WIDE_EXTENT: [u32; 2] = [640, 45];

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct GuardTelemetry {
    pub schedule: u64,
    pub selects: u64,
    pub guarded: u64,
    pub exclusive: u64,
}

pub struct CellGridCase {
    pub shape: CellGridShape,
    pub metrics: CellGridMetrics,
    pub cells_id: pixelflow_ir::arena::BufferIdentity,
    pub cells: Arc<Vec<f32>>,
    pub atlas: Arc<Vec<f32>>,
}

pub struct RealKernel {
    pub name: String,
    pub class: String,
    pub kernel: Kernel,
    pub extent: [u32; 2],
    pub packed: bool,
    pub cell_grid: Option<CellGridCase>,
}

pub fn k(v: f32) -> Kernel {
    Kernel::constant(v)
}

/// `pixelflow_graphics::render::packed::packed_kernel`, which is
/// `pub(crate)`: the byte pack `compile_packed_for` wraps a colour in.
/// `main` asserts the bytes of what this produces equal the production
/// program's, so a drift here is loud.
pub fn packed_kernel(color: &Rgba, shifts: [u32; 4]) -> Kernel {
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

pub fn rgba8_shifts() -> [u32; 4] {
    <Rgba8 as Pixel>::packed_shifts().expect("Rgba8 packs")
}

pub fn chrome_color() -> Rgba {
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

pub fn psych_channel(y_weight: f32, clock: Uniform) -> Kernel {
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

pub fn psychedelic_color() -> Rgba {
    let clock = Uniform::new(0.0);
    Rgba::from([
        psych_channel(1.0, clock),
        psych_channel(-1.0, clock),
        psych_channel(-2.0, clock),
        k(1.0),
    ])
}

pub fn cell_grid_case() -> (Kernel, CellGridCase) {
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

pub fn real_kernels(font: &Path, filter: Option<&str>) -> Vec<RealKernel> {
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
            let Some(kernel) = parsed.glyph_kernel_scaled(ch, tile as f32) else {
                missing += 1;
                continue;
            };
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
            .unwrap_or_else(|| panic!("no glyph for {ch:?}"));
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

/// Redirect fd 2 into a file around `f` so `PIXELFLOW_GUARD_TELEMETRY`'s
/// `eprintln!` lands somewhere this process can read it back.
pub fn capture_stderr<T>(f: impl FnOnce() -> T) -> (T, String) {
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

pub fn parse_guard_telemetry(log: &str) -> Option<GuardTelemetry> {
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

pub fn reachable(arena: &ExprArena, root: ExprId) -> Vec<ExprId> {
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

pub fn dag_cost(arena: &ExprArena, root: ExprId) -> usize {
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

pub fn legalize(arena: &ExprArena, root: ExprId) -> (ExprArena, ExprId) {
    match pipeline![LowerDwrt, ExpandReduce].optimize(arena, root) {
        Rewritten::Changed(a, r) => (a, r),
        Rewritten::Unchanged => (arena.clone(), root),
        Rewritten::Declined => panic!("legalizing prefix declined a real kernel"),
    }
}
