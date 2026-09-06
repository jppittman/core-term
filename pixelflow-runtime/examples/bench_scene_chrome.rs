//! The S3 gate: the chrome sphere rendered as a `Scene`, both ways.
//!
//! `cargo run --release -p pixelflow-runtime --example bench_scene_chrome`
//!
//! - **surface**: `Scene::Surface` — `scene3d_surface`'s `Manifold<Jet3>`
//!   combinators, evaluated once per SIMD batch through the `Manifold` trait
//!   by the work-stealing rasterizer. The legacy tier S4 deletes.
//! - **packed**: `Scene::Packed` — the same scene written as four channel
//!   `Kernel`s with `scene3d`'s constructors, compiled once at the frame's
//!   lattice shape with the pixel pack inside the kernel, collapsed one call
//!   per stripe.
//!
//! Mirrors `bench_scene_psychedelic.rs` (the S2 gate) so the two stages'
//! numbers are comparable, and like it, the lanes are compared pixel by pixel
//! before either is timed — a fast wrong answer cannot win.
//!
//! ## Reading the agreement rows
//!
//! The two lanes draw the same geometry and the same colours, and disagree
//! about three things — two of them antialiasing:
//!
//! - The jet tier seeded its derivatives **after** the pixel-to-screen remap,
//!   so its filter width is one *normalized screen* unit — half the frame
//!   height, 540 pixels here — where [`Hit::footprint`] is one pixel, which
//!   is what a footprint means.
//! - Through a reflection the jet tier did not trust its dual numbers and
//!   scaled the normal's derivatives by a hand-tuned `2/|cos θ|`;
//!   `Kernel::dx()` differentiates the reflection exactly and needs no such
//!   factor, and no scale reproduces it.
//! - And one that is not antialiasing at all: the jet tier's **normal is not
//!   a unit vector**. It normalizes the tangent frame's cross product with
//!   `n_len_sq.sqrt().rsqrt()`, which is `|n|^-½` rather than `|n|^-1`, so
//!   `D − 2(D·N)N` is a reflection only where the screen-to-surface map has
//!   unit area scale. The two lanes therefore reflect differently over the
//!   whole sphere, and the packed one is the reflection
//!   (`scene3d::tests::a_reflection_off_a_sphere_is_a_unit_ray`).
//!
//! No scaling reconciles any of that, so the row that checks the *rest* of
//! the scene takes the checker out of the picture: **matte (no checker)** is
//! sphere, floor and sky in flat grey — silhouette, horizon, sky and pack,
//! with no filter width anywhere — and it is the one that should agree to
//! the goldens' own 2/255 on essentially every pixel. **mirror over sky**
//! isolates the reflection (no checker either), so its disagreement is the
//! non-unit normal above, bounded by the sphere's screen coverage, which is
//! printed beside it. **chrome, as shipped** is the whole scene, whose
//! disagreement is the antialiasing and is meant to be large.

use std::sync::Arc;
use std::time::Instant;

use pixelflow_compiler::ManifoldExpr;
use pixelflow_core::combinators::At;
use pixelflow_core::jet::Jet3;
use pixelflow_core::{Discrete, Field, Kernel, Manifold, ManifoldCompat};
use pixelflow_graphics::render::color::{Rgba8, RgbaColorCube};
use pixelflow_graphics::render::scene::{compile_packed_for, Scene};
use pixelflow_graphics::render::Frame;
use pixelflow_graphics::scene3d::{checker, sky, Hit, Plane, Ray, Rgba, Sphere};
use pixelflow_graphics::scene3d_surface::{
    plane, ColorChecker, ColorReflect, ColorScreenToDir, ColorSky, ColorSurface,
};

const WIDTH: usize = 1920;
const HEIGHT: usize = 1080;

/// Sphere centre and radius, shared by both lanes.
const CENTER: (f32, f32, f32) = (0.0, 0.0, 4.0);
const RADIUS: f32 = 1.0;
/// The checker floor's height.
const FLOOR: f32 = -1.0;
/// The matte sphere's grey.
const MATTE: f32 = 0.5;

type Field4 = (Field, Field, Field, Field);
type Jet3_4 = (Jet3, Jet3, Jet3, Jet3);

// ── the scene as jet combinators (the surface lane) ──

#[derive(Clone, Copy, ManifoldExpr)]
struct SphereAt {
    center: (f32, f32, f32),
    radius: f32,
}

impl Manifold<Jet3_4> for SphereAt {
    type Output = Jet3;

    #[inline]
    fn eval(&self, p: Jet3_4) -> Jet3 {
        let (rx, ry, rz, _w) = p;
        let cx = Jet3::constant(Field::from(self.center.0));
        let cy = Jet3::constant(Field::from(self.center.1));
        let cz = Jet3::constant(Field::from(self.center.2));

        let d_dot_c = rx * cx + ry * cy + rz * cz;
        let c_sq = cx * cx + cy * cy + cz * cz;
        let r_sq = Jet3::constant(Field::from(self.radius * self.radius));
        let discriminant = d_dot_c * d_dot_c - (c_sq - r_sq);

        let epsilon_sq = Jet3::constant(Field::from(0.0001));
        d_dot_c - (discriminant + epsilon_sq).sqrt()
    }
}

/// A flat opaque grey, as the jet tier spells it.
#[derive(Clone, Copy, ManifoldExpr)]
struct SolidGray;

impl Manifold<Jet3_4> for SolidGray {
    type Output = Discrete;

    fn eval(&self, _p: Jet3_4) -> Discrete {
        let gray = Field::from(MATTE);
        Discrete::pack(gray, gray, gray, Field::from(1.0))
    }
}

/// Pixel coordinates to the normalized screen the jet tier's rays start from.
#[derive(Clone, Copy, ManifoldExpr)]
struct ColorScreenRemap<M> {
    inner: M,
    width: f32,
    height: f32,
}

impl<M: ManifoldCompat<Field, Output = Discrete>> Manifold<Field4> for ColorScreenRemap<M> {
    type Output = Discrete;

    fn eval(&self, p: Field4) -> Discrete {
        let (px, py, z, w) = p;
        let width = Field::from(self.width);
        let height = Field::from(self.height);
        let scale = Field::from(2.0) / height;
        let x = (px - width * Field::from(0.5)) * scale.clone();
        let y = (height * Field::from(0.5) - py) * scale;
        At {
            inner: &self.inner,
            x,
            y,
            z,
            w,
        }
        .collapse()
    }
}

fn surface_world() -> ColorSurface<
    pixelflow_graphics::scene3d_surface::PlaneKernel,
    ColorChecker<RgbaColorCube>,
    ColorSky<RgbaColorCube>,
> {
    ColorSurface {
        geometry: plane(FLOOR),
        material: ColorChecker::<RgbaColorCube>::default(),
        background: ColorSky::<RgbaColorCube>::default(),
    }
}

fn surface_sphere() -> SphereAt {
    SphereAt {
        center: CENTER,
        radius: RADIUS,
    }
}

fn on_screen<M>(inner: M) -> Scene
where
    M: ManifoldCompat<Jet3, Output = Discrete> + Send + Sync + 'static,
{
    Scene::Surface(Arc::new(ColorScreenRemap {
        inner: ColorScreenToDir { inner },
        width: WIDTH as f32,
        height: HEIGHT as f32,
    }))
}

/// The sphere reflecting the sky and nothing else: the reflection, alone.
fn surface_mirror() -> Scene {
    on_screen(ColorSurface {
        geometry: surface_sphere(),
        material: ColorReflect {
            inner: ColorSky::<RgbaColorCube>::default(),
        },
        background: ColorSky::<RgbaColorCube>::default(),
    })
}

fn surface_chrome() -> Scene {
    let world = surface_world();
    on_screen(ColorSurface {
        geometry: surface_sphere(),
        material: ColorReflect { inner: world },
        background: world,
    })
}

/// The same geometry in flat grey — sphere, floor and sky, no checker, so no
/// filter width enters the comparison at all.
fn surface_matte() -> Scene {
    on_screen(ColorSurface {
        geometry: surface_sphere(),
        material: SolidGray,
        background: ColorSurface {
            geometry: plane(FLOOR),
            material: SolidGray,
            background: ColorSky::<RgbaColorCube>::default(),
        },
    })
}

// ── the same scenes as four channel kernels (the packed lane) ──

fn k(v: f32) -> Kernel {
    Kernel::constant(v)
}

/// How wide a hit's material filters its edges. The scenes take this as a
/// parameter only because the gate compares against the jet tier's
/// convention; every other caller passes [`Hit::footprint`].
type Filter = dyn Fn(&Hit) -> Kernel;

/// One pixel wide — what a screen-space footprint is.
fn per_pixel(hit: &Hit) -> Kernel {
    hit.footprint()
}

fn ray() -> Ray {
    Ray::through_screen(WIDTH as f32, HEIGHT as f32)
}

fn sphere(ray: &Ray) -> Hit {
    Sphere::new([k(CENTER.0), k(CENTER.1), k(CENTER.2)], k(RADIUS)).hit(ray)
}

/// The checker floor under the sky, seen along `ray`.
fn world(ray: &Ray, filter: &Filter) -> Rgba {
    let floor = Plane::at_height(k(FLOOR)).hit(ray);
    floor.select(
        &checker(&floor.point()[0], &floor.point()[2], &filter(&floor)),
        &sky(ray),
    )
}

fn packed_chrome(filter: &Filter) -> Rgba {
    let ray = ray();
    let sphere = sphere(&ray);
    let mirrored = ray.reflected(sphere.normal());
    sphere.select(&world(&mirrored, filter), &world(&ray, filter))
}

fn packed_matte() -> Rgba {
    let ray = ray();
    let floor = Plane::at_height(k(FLOOR))
        .hit(&ray)
        .select(&Rgba::opaque_gray(MATTE), &sky(&ray));
    sphere(&ray).select(&Rgba::opaque_gray(MATTE), &floor)
}

fn packed_mirror() -> Rgba {
    let ray = ray();
    let sphere = sphere(&ray);
    let mirrored = ray.reflected(sphere.normal());
    sphere.select(&sky(&mirrored), &sky(&ray))
}

fn packed_scene(color: &Rgba) -> Scene {
    Scene::Packed(compile_packed_for::<Rgba8>(color, [WIDTH as u32, HEIGHT as u32]).bind(&[]))
}

/// What fraction of the frame the sphere covers — the area the reflection's
/// antialiasing can disagree over.
fn sphere_coverage() -> f64 {
    let ray = ray();
    let mask = sphere(&ray).mask().select(&k(1.0), &k(0.0));
    let scene = packed_scene(&Rgba::from([mask.clone(), mask.clone(), mask, k(1.0)]));
    let mut frame = Frame::<Rgba8>::new(WIDTH as u32, HEIGHT as u32);
    scene.render(&mut frame, 4);
    let hits = frame.data.iter().filter(|p| p.r() > 0).count();
    hits as f64 / (WIDTH * HEIGHT) as f64
}

// ── measurement ──

/// Median ns/pixel over `runs` frames, after `warm` warm-up frames.
fn time_ns_per_px(scene: &Scene, threads: usize, warm: usize, runs: usize) -> f64 {
    let mut frame = Frame::<Rgba8>::new(WIDTH as u32, HEIGHT as u32);
    for _ in 0..warm {
        scene.render(&mut frame, threads);
        std::hint::black_box(&frame.data[0]);
    }
    let mut samples = Vec::with_capacity(runs);
    for _ in 0..runs {
        let t = Instant::now();
        scene.render(&mut frame, threads);
        samples.push(t.elapsed().as_nanos());
        std::hint::black_box(&frame.data[0]);
    }
    samples.sort_unstable();
    samples[samples.len() / 2] as f64 / (WIDTH * HEIGHT) as f64
}

/// Largest per-channel disagreement, and the fraction of pixels where any
/// channel differs by more than `slack`.
fn agreement(a: &Scene, b: &Scene, slack: u8) -> (u8, f64) {
    let mut fa = Frame::<Rgba8>::new(WIDTH as u32, HEIGHT as u32);
    let mut fb = Frame::<Rgba8>::new(WIDTH as u32, HEIGHT as u32);
    a.render(&mut fa, 1);
    b.render(&mut fb, 1);
    let (mut worst, mut over) = (0u8, 0usize);
    for (pa, pb) in fa.data.iter().zip(fb.data.iter()) {
        let mut d = 0u8;
        for (x, y) in [
            (pa.r(), pb.r()),
            (pa.g(), pb.g()),
            (pa.b(), pb.b()),
            (pa.a(), pb.a()),
        ] {
            d = d.max(x.abs_diff(y));
        }
        worst = worst.max(d);
        over += usize::from(d > slack);
    }
    (worst, over as f64 / (WIDTH * HEIGHT) as f64)
}

fn main() {
    const WARM: usize = 3;
    const RUNS: usize = 9;
    /// `PerformanceConfig::default().render_threads`.
    const RUNTIME_THREADS: usize = 12;
    /// The goldens' own per-channel tolerance.
    const SLACK: u8 = 2;

    // Compile cost: building the scene and JIT-compiling it, once.
    let built = Instant::now();
    let color = packed_chrome(&per_pixel);
    let build_time = built.elapsed();
    let nodes: usize = color.fold(
        &|channels: &[Kernel; 4]| {
            channels
                .iter()
                .map(|c| {
                    let (arena, root) = c.parts();
                    arena.node_count_subtree(root)
                })
                .sum()
        },
        &|mask: &Kernel, if_true: usize, if_false: usize| {
            let (arena, root) = mask.parts();
            arena.node_count_subtree(root) + if_true + if_false
        },
    );
    let compiled = Instant::now();
    let program = compile_packed_for::<Rgba8>(&color, [WIDTH as u32, HEIGHT as u32]);
    let compile_time = compiled.elapsed();
    let code = program.code_bytes().len();

    let surface = surface_chrome();
    let packed = Scene::Packed(program.bind(&[]));

    println!("chrome sphere as a Scene, {WIDTH}x{HEIGHT}, median of {RUNS} frames (ns/pixel)\n");
    println!(
        "  compile: {nodes} arena nodes over four channels, built in {build_time:?}, \
         compiled in {compile_time:?} to {code} bytes of code"
    );
    println!(
        "  the sphere covers {:.2}% of the frame\n",
        sphere_coverage() * 100.0
    );

    for (what, lhs, rhs) in [
        (
            "matte (no checker)",
            surface_matte(),
            packed_scene(&packed_matte()),
        ),
        (
            "mirror over sky",
            surface_mirror(),
            packed_scene(&packed_mirror()),
        ),
        ("chrome, as shipped", surface_chrome(), packed.clone()),
    ] {
        let (worst, over) = agreement(&lhs, &rhs, SLACK);
        println!(
            "  agreement ({what:>22}): max channel delta {worst:>3}, \
             {:>6.3}% of pixels over {SLACK}/255",
            over * 100.0
        );
    }

    let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
    println!("\n  threads   surface (Scene::Surface)   packed (Scene::Packed)   speedup");
    for threads in [1, cores, RUNTIME_THREADS] {
        let s = time_ns_per_px(&surface, threads, WARM, RUNS);
        let p = time_ns_per_px(&packed, threads, WARM, RUNS);
        println!("  {threads:>7}   {s:>22.2}   {p:>20.2}   {:>6.2}x", s / p);
    }
}
