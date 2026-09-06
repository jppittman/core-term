//! Work-stealing rasterizer throughput vs. single-threaded, at 1080p on a
//! reflective raymarched scene.
//!
//! Postsubmit-only (docs/POSTSUBMIT.md): `cargo bench --workspace --benches`
//! is never invoked presubmit, so this never runs on a PR. Split out of
//! `tests/scene3d_test.rs`'s `work_stealing_benchmark`, which mixed this
//! throughput comparison with a correctness assertion — the correctness half
//! (parallel output matches single-threaded) stays a fast `#[test]` there as
//! `work_stealing_matches_single_threaded`, at 64x64 instead of 1080p.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use pixelflow_compiler::ManifoldExpr;
use pixelflow_core::combinators::At;
use pixelflow_core::jet::Jet3;
use pixelflow_core::{Discrete, Field, Manifold, ManifoldCompat};
use pixelflow_graphics::render::color::{Rgba8, RgbaColorCube};
use pixelflow_graphics::render::frame::Frame;
use pixelflow_graphics::render::rasterizer::rasterize;
use pixelflow_graphics::scene3d_surface::{
    plane, ColorChecker, ColorReflect, ColorScreenToDir, ColorSky, ColorSurface,
};

type Field4 = (Field, Field, Field, Field);
type Jet3_4 = (Jet3, Jet3, Jet3, Jet3);

const W: usize = 1920;
const H: usize = 1080;

/// Sphere at given center with radius (mirrors `tests/scene3d_test.rs`).
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

fn renderable() -> impl Manifold<Field4, Output = Discrete> {
    let world = ColorSurface {
        geometry: plane(-1.0),
        material: ColorChecker::<RgbaColorCube>::default(),
        background: ColorSky::<RgbaColorCube>::default(),
    };

    let scene = ColorSurface {
        geometry: SphereAt {
            center: (0.0, 0.0, 4.0),
            radius: 1.0,
        },
        material: ColorReflect { inner: world },
        background: world,
    };

    ColorScreenRemap {
        inner: ColorScreenToDir { inner: scene },
        width: W as f32,
        height: H as f32,
    }
}

fn bench_work_stealing(c: &mut Criterion) {
    let renderable = renderable();
    let mut group = c.benchmark_group("work_stealing_1080p");
    // One full-frame render per sample at 1080p is already expensive;
    // Criterion's default 100 samples would run for a very long time.
    group.sample_size(10);

    group.bench_function("single_threaded", |b| {
        b.iter(|| {
            let mut frame = Frame::<Rgba8>::new(W as u32, H as u32);
            rasterize(black_box(&renderable), &mut frame, 1);
            black_box(frame);
        });
    });

    group.bench_function("work_stealing_12", |b| {
        b.iter(|| {
            let mut frame = Frame::<Rgba8>::new(W as u32, H as u32);
            rasterize(black_box(&renderable), &mut frame, 12);
            black_box(frame);
        });
    });

    group.finish();
}

criterion_group!(work_stealing_benches, bench_work_stealing);
criterion_main!(work_stealing_benches);
