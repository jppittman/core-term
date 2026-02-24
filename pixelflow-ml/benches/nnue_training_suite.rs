//! NNUE Training Data Collection Suite
//!
//! This benchmark generates training data for NNUE by running diverse kernels
//! and collecting (Expr features, actual runtime) pairs.
//!
//! Run with: cargo bench -p pixelflow-ml --bench nnue_training_suite -- --save-baseline training
//!
//! The results can be processed to create NNUE training data.

use criterion::{Criterion, BenchmarkId, black_box, criterion_group, criterion_main};
use pixelflow_core::{Field, Manifold, ManifoldExt, X, Y, Z, W};
use pixelflow_ml::nnue::{Expr, OpType, extract_features};
use pixelflow_ml::evaluator::{extract_expr_features, default_expr_weights};

// ============================================================================
// Macro for generating matched Expr+Manifold kernel pairs
// ============================================================================

/// A kernel with matched Expr representation for feature extraction
struct KernelPair {
    name: &'static str,
    expr: Expr,
    /// HCE cost for reference
    hce_cost: i32,
    /// Critical path for reference
    critical_path: i32,
}

impl KernelPair {
    fn new(name: &'static str, expr: Expr) -> Self {
        let features = extract_expr_features(&expr);
        let hce = default_expr_weights();
        Self {
            name,
            expr,
            hce_cost: hce.evaluate_linear(&features),
            critical_path: features.critical_path,
        }
    }
}

// ============================================================================
// Helper functions for building Expr trees
// ============================================================================

fn var(i: u8) -> Expr { Expr::Var(i) }
fn cnst(v: f32) -> Expr { Expr::Const(v) }
fn add(a: Expr, b: Expr) -> Expr { Expr::Binary(OpType::Add, Box::new(a), Box::new(b)) }
fn sub(a: Expr, b: Expr) -> Expr { Expr::Binary(OpType::Sub, Box::new(a), Box::new(b)) }
fn mul(a: Expr, b: Expr) -> Expr { Expr::Binary(OpType::Mul, Box::new(a), Box::new(b)) }
fn div(a: Expr, b: Expr) -> Expr { Expr::Binary(OpType::Div, Box::new(a), Box::new(b)) }
fn sqrt(a: Expr) -> Expr { Expr::Unary(OpType::Sqrt, Box::new(a)) }
fn neg(a: Expr) -> Expr { Expr::Unary(OpType::Neg, Box::new(a)) }
fn abs(a: Expr) -> Expr { Expr::Unary(OpType::Abs, Box::new(a)) }
fn min(a: Expr, b: Expr) -> Expr { Expr::Binary(OpType::Min, Box::new(a), Box::new(b)) }
fn max(a: Expr, b: Expr) -> Expr { Expr::Binary(OpType::Max, Box::new(a), Box::new(b)) }

// Shorthand for x, y, z, w
fn x() -> Expr { var(0) }
fn y() -> Expr { var(1) }
fn z() -> Expr { var(2) }
fn w() -> Expr { var(3) }

// ============================================================================
// Category 1: Basic Arithmetic (varying op counts)
// ============================================================================

fn basic_arithmetic_kernels() -> Vec<KernelPair> {
    vec![
        // Simple operations
        KernelPair::new("add_xy", add(x(), y())),
        KernelPair::new("mul_xy", mul(x(), y())),
        KernelPair::new("sub_xy", sub(x(), y())),
        KernelPair::new("div_xy", div(x(), y())),

        // Two operations
        KernelPair::new("add_mul", mul(add(x(), y()), z())),
        KernelPair::new("mul_add", add(mul(x(), y()), z())),
        KernelPair::new("sub_mul", mul(sub(x(), y()), z())),
        KernelPair::new("div_add", add(div(x(), y()), z())),

        // Three operations - linear chain
        KernelPair::new("chain3_add", add(add(x(), y()), z())),
        KernelPair::new("chain3_mul", mul(mul(x(), y()), z())),
        KernelPair::new("chain3_mix", add(mul(x(), y()), z())),

        // Four operations
        KernelPair::new("chain4_add", add(add(add(x(), y()), z()), w())),
        KernelPair::new("chain4_mul", mul(mul(mul(x(), y()), z()), w())),
        KernelPair::new("chain4_mix", mul(add(mul(x(), y()), z()), w())),

        // Wide trees (parallel friendly)
        KernelPair::new("wide2_add", add(add(x(), y()), add(z(), w()))),
        KernelPair::new("wide2_mul", add(mul(x(), y()), mul(z(), w()))),
        KernelPair::new("wide2_mix", mul(add(x(), y()), add(z(), w()))),
    ]
}

// ============================================================================
// Category 2: Expensive Operations (sqrt, div chains)
// ============================================================================

fn expensive_op_kernels() -> Vec<KernelPair> {
    vec![
        // Single expensive ops
        KernelPair::new("sqrt_x", sqrt(x())),
        KernelPair::new("sqrt_xy", sqrt(add(x(), y()))),
        KernelPair::new("div_single", div(x(), y())),

        // Multiple sqrts - parallel
        KernelPair::new("sqrt2_wide", add(sqrt(x()), sqrt(y()))),
        KernelPair::new("sqrt3_wide", add(add(sqrt(x()), sqrt(y())), sqrt(z()))),
        KernelPair::new("sqrt4_wide", add(add(sqrt(x()), sqrt(y())), add(sqrt(z()), sqrt(w())))),

        // Multiple sqrts - sequential
        KernelPair::new("sqrt2_deep", sqrt(add(sqrt(x()), cnst(1.0)))),
        KernelPair::new("sqrt3_deep", sqrt(add(sqrt(add(sqrt(x()), cnst(1.0))), cnst(1.0)))),

        // Multiple divs - parallel
        KernelPair::new("div2_wide", add(div(x(), y()), div(z(), w()))),

        // Multiple divs - sequential
        KernelPair::new("div2_deep", div(div(x(), y()), z())),
        KernelPair::new("div3_deep", div(div(div(x(), y()), z()), w())),

        // Mixed expensive
        KernelPair::new("sqrt_div_wide", add(sqrt(x()), div(y(), z()))),
        KernelPair::new("sqrt_div_deep", div(sqrt(x()), sqrt(y()))),
    ]
}

// ============================================================================
// Category 3: Distance Functions (common patterns)
// ============================================================================

fn distance_kernels() -> Vec<KernelPair> {
    vec![
        // 2D distance
        KernelPair::new("dist2d", sqrt(add(mul(x(), x()), mul(y(), y())))),

        // 3D distance
        KernelPair::new("dist3d", sqrt(add(add(mul(x(), x()), mul(y(), y())), mul(z(), z())))),

        // 4D distance
        KernelPair::new("dist4d", sqrt(add(
            add(mul(x(), x()), mul(y(), y())),
            add(mul(z(), z()), mul(w(), w()))
        ))),

        // 2D distance squared (no sqrt)
        KernelPair::new("dist2d_sq", add(mul(x(), x()), mul(y(), y()))),

        // 3D distance squared
        KernelPair::new("dist3d_sq", add(add(mul(x(), x()), mul(y(), y())), mul(z(), z()))),

        // Circle SDF: sqrt(x² + y²) - r
        KernelPair::new("circle_sdf", sub(sqrt(add(mul(x(), x()), mul(y(), y()))), cnst(1.0))),

        // Sphere SDF: sqrt(x² + y² + z²) - r
        KernelPair::new("sphere_sdf", sub(
            sqrt(add(add(mul(x(), x()), mul(y(), y())), mul(z(), z()))),
            cnst(1.0)
        )),

        // Box SDF (approximate): max(|x|-w, |y|-h)
        KernelPair::new("box2d_sdf", max(sub(abs(x()), cnst(1.0)), sub(abs(y()), cnst(1.0)))),

        // Normalize: x / sqrt(x² + y²)
        KernelPair::new("normalize_x", div(x(), sqrt(add(mul(x(), x()), mul(y(), y()))))),
    ]
}

// ============================================================================
// Category 4: Polynomial Expressions
// ============================================================================

fn polynomial_kernels() -> Vec<KernelPair> {
    vec![
        // Linear: ax + b
        KernelPair::new("linear", add(mul(cnst(2.0), x()), cnst(1.0))),

        // Quadratic: ax² + bx + c
        KernelPair::new("quadratic", add(add(mul(cnst(2.0), mul(x(), x())), mul(cnst(3.0), x())), cnst(1.0))),

        // Cubic: x³
        KernelPair::new("cubic", mul(mul(x(), x()), x())),

        // Quartic: x⁴
        KernelPair::new("quartic", mul(mul(mul(x(), x()), x()), x())),

        // 2-var quadratic: x² + y²
        KernelPair::new("quad2v", add(mul(x(), x()), mul(y(), y()))),

        // 2-var cubic: x³ + y³
        KernelPair::new("cubic2v", add(mul(mul(x(), x()), x()), mul(mul(y(), y()), y()))),

        // Cross term: x*y
        KernelPair::new("cross_xy", mul(x(), y())),

        // Cross term: x*y + y*z
        KernelPair::new("cross_xyz", add(mul(x(), y()), mul(y(), z()))),

        // Full quadratic 2D: ax² + bxy + cy²
        KernelPair::new("full_quad2d", add(
            add(mul(cnst(2.0), mul(x(), x())), mul(cnst(3.0), mul(x(), y()))),
            mul(cnst(4.0), mul(y(), y()))
        )),
    ]
}

// ============================================================================
// Category 5: Deep vs Wide Comparisons
// ============================================================================

fn depth_vs_width_kernels() -> Vec<KernelPair> {
    vec![
        // Depth 2, width 4
        KernelPair::new("d2w4", add(add(x(), y()), add(z(), w()))),

        // Depth 3, width 2
        KernelPair::new("d3w2_left", add(add(add(x(), y()), z()), w())),
        KernelPair::new("d3w2_right", add(x(), add(y(), add(z(), w())))),

        // Depth 4, width 1 (fully sequential)
        KernelPair::new("d4w1", add(add(add(add(x(), cnst(1.0)), cnst(2.0)), cnst(3.0)), cnst(4.0))),

        // Wide with expensive ops
        KernelPair::new("wide_sqrt4", add(
            add(sqrt(x()), sqrt(y())),
            add(sqrt(z()), sqrt(w()))
        )),

        // Deep with expensive ops
        KernelPair::new("deep_sqrt3", sqrt(add(sqrt(add(sqrt(x()), cnst(1.0))), cnst(1.0)))),

        // Wide with divs
        KernelPair::new("wide_div2", add(div(x(), y()), div(z(), w()))),

        // Deep with divs
        KernelPair::new("deep_div3", div(div(div(x(), y()), z()), w())),
    ]
}

// ============================================================================
// Category 6: Min/Max Combinations
// ============================================================================

fn minmax_kernels() -> Vec<KernelPair> {
    vec![
        // Single min/max
        KernelPair::new("min_xy", min(x(), y())),
        KernelPair::new("max_xy", max(x(), y())),

        // Clamp: max(lo, min(hi, x))
        KernelPair::new("clamp", max(cnst(0.0), min(cnst(1.0), x()))),

        // Abs via max: max(x, -x)
        KernelPair::new("abs_via_max", max(x(), neg(x()))),

        // SDF union: min(sdf1, sdf2)
        KernelPair::new("sdf_union", min(
            sub(sqrt(add(mul(x(), x()), mul(y(), y()))), cnst(1.0)),
            sub(sqrt(add(mul(sub(x(), cnst(3.0)), sub(x(), cnst(3.0))), mul(y(), y()))), cnst(1.0))
        )),

        // SDF intersection: max(sdf1, sdf2)
        KernelPair::new("sdf_intersect", max(
            sub(sqrt(add(mul(x(), x()), mul(y(), y()))), cnst(2.0)),
            sub(abs(x()), cnst(1.0))
        )),
    ]
}

// ============================================================================
// Collect all kernel pairs
// ============================================================================

fn all_kernel_pairs() -> Vec<KernelPair> {
    let mut all = Vec::new();
    all.extend(basic_arithmetic_kernels());
    all.extend(expensive_op_kernels());
    all.extend(distance_kernels());
    all.extend(polynomial_kernels());
    all.extend(depth_vs_width_kernels());
    all.extend(minmax_kernels());
    all
}

// ============================================================================
// Benchmark runner using Expr interpreter
// ============================================================================

fn bench_expr_interpreter(c: &mut Criterion) {
    let mut group = c.benchmark_group("nnue_training_expr");
    group.sample_size(200);

    let vars = [1.5f32, 2.5, 3.5, 0.5];
    let kernels = all_kernel_pairs();

    // Print kernel info
    println!("\n=== NNUE Training Suite: {} kernels ===", kernels.len());
    for kp in &kernels {
        let features = extract_features(&kp.expr);
        println!("  {}: HCE={}, crit_path={}, nodes={}, halfep_features={}",
            kp.name, kp.hce_cost, kp.critical_path, kp.expr.node_count(), features.len());
    }
    println!();

    for kp in &kernels {
        group.bench_with_input(
            BenchmarkId::new("expr", kp.name),
            &kp.expr,
            |b, expr| b.iter(|| black_box(expr.eval(black_box(&vars)))),
        );
    }

    group.finish();
}

// ============================================================================
// Benchmark runner using compiled Manifold
// ============================================================================

fn bench_manifold_compiled(c: &mut Criterion) {
    let mut group = c.benchmark_group("nnue_training_manifold");
    group.sample_size(200);

    let xf = Field::sequential(1.0);
    let yf = Field::from(2.0);
    let zf = Field::from(3.0);
    let wf = Field::from(0.5);

    // Category 1: Basic arithmetic
    {
        let m = X + Y;
        group.bench_function("add_xy", |b| {
            b.iter(|| black_box(m.eval((black_box(xf), black_box(yf), zf, wf))))
        });
    }
    {
        let m = X * Y;
        group.bench_function("mul_xy", |b| {
            b.iter(|| black_box(m.eval((black_box(xf), black_box(yf), zf, wf))))
        });
    }
    {
        let m = X - Y;
        group.bench_function("sub_xy", |b| {
            b.iter(|| black_box(m.eval((black_box(xf), black_box(yf), zf, wf))))
        });
    }
    {
        let m = X / Y;
        group.bench_function("div_xy", |b| {
            b.iter(|| black_box(m.eval((black_box(xf), black_box(yf), zf, wf))))
        });
    }
    {
        let m = (X + Y) * Z;
        group.bench_function("add_mul", |b| {
            b.iter(|| black_box(m.eval((black_box(xf), black_box(yf), black_box(zf), wf))))
        });
    }
    {
        let m = X * Y + Z;
        group.bench_function("mul_add", |b| {
            b.iter(|| black_box(m.eval((black_box(xf), black_box(yf), black_box(zf), wf))))
        });
    }
    {
        let m = (X - Y) * Z;
        group.bench_function("sub_mul", |b| {
            b.iter(|| black_box(m.eval((black_box(xf), black_box(yf), black_box(zf), wf))))
        });
    }
    {
        let m = X / Y + Z;
        group.bench_function("div_add", |b| {
            b.iter(|| black_box(m.eval((black_box(xf), black_box(yf), black_box(zf), wf))))
        });
    }
    {
        let m = X + Y + Z;
        group.bench_function("chain3_add", |b| {
            b.iter(|| black_box(m.eval((black_box(xf), black_box(yf), black_box(zf), wf))))
        });
    }
    {
        let m = X * Y * Z;
        group.bench_function("chain3_mul", |b| {
            b.iter(|| black_box(m.eval((black_box(xf), black_box(yf), black_box(zf), wf))))
        });
    }
    {
        let m = X * Y + Z;
        group.bench_function("chain3_mix", |b| {
            b.iter(|| black_box(m.eval((black_box(xf), black_box(yf), black_box(zf), wf))))
        });
    }
    {
        let m = X + Y + Z + W;
        group.bench_function("chain4_add", |b| {
            b.iter(|| black_box(m.eval((black_box(xf), black_box(yf), black_box(zf), black_box(wf)))))
        });
    }
    {
        let m = X * Y * Z * W;
        group.bench_function("chain4_mul", |b| {
            b.iter(|| black_box(m.eval((black_box(xf), black_box(yf), black_box(zf), black_box(wf)))))
        });
    }
    {
        let m = (X * Y + Z) * W;
        group.bench_function("chain4_mix", |b| {
            b.iter(|| black_box(m.eval((black_box(xf), black_box(yf), black_box(zf), black_box(wf)))))
        });
    }
    {
        let m = (X + Y) + (Z + W);
        group.bench_function("wide2_add", |b| {
            b.iter(|| black_box(m.eval((black_box(xf), black_box(yf), black_box(zf), black_box(wf)))))
        });
    }
    {
        let m = X * Y + Z * W;
        group.bench_function("wide2_mul", |b| {
            b.iter(|| black_box(m.eval((black_box(xf), black_box(yf), black_box(zf), black_box(wf)))))
        });
    }
    {
        let m = (X + Y) * (Z + W);
        group.bench_function("wide2_mix", |b| {
            b.iter(|| black_box(m.eval((black_box(xf), black_box(yf), black_box(zf), black_box(wf)))))
        });
    }

    // Category 2: Expensive operations
    {
        let m = X.sqrt();
        group.bench_function("sqrt_x", |b| {
            b.iter(|| black_box(m.eval((black_box(xf), yf, zf, wf))))
        });
    }
    {
        let m = (X + Y).sqrt();
        group.bench_function("sqrt_xy", |b| {
            b.iter(|| black_box(m.eval((black_box(xf), black_box(yf), zf, wf))))
        });
    }
    {
        let m = X.sqrt() + Y.sqrt();
        group.bench_function("sqrt2_wide", |b| {
            b.iter(|| black_box(m.eval((black_box(xf), black_box(yf), zf, wf))))
        });
    }
    {
        let m = X.sqrt() + Y.sqrt() + Z.sqrt();
        group.bench_function("sqrt3_wide", |b| {
            b.iter(|| black_box(m.eval((black_box(xf), black_box(yf), black_box(zf), wf))))
        });
    }
    {
        let m = X.sqrt() + Y.sqrt() + Z.sqrt() + W.sqrt();
        group.bench_function("sqrt4_wide", |b| {
            b.iter(|| black_box(m.eval((black_box(xf), black_box(yf), black_box(zf), black_box(wf)))))
        });
    }
    {
        let m = (X.sqrt() + 1.0f32).sqrt();
        group.bench_function("sqrt2_deep", |b| {
            b.iter(|| black_box(m.eval((black_box(xf), yf, zf, wf))))
        });
    }
    {
        let m = ((X.sqrt() + 1.0f32).sqrt() + 1.0f32).sqrt();
        group.bench_function("sqrt3_deep", |b| {
            b.iter(|| black_box(m.eval((black_box(xf), yf, zf, wf))))
        });
    }
    {
        let m = X / Y + Z / W;
        group.bench_function("div2_wide", |b| {
            b.iter(|| black_box(m.eval((black_box(xf), black_box(yf), black_box(zf), black_box(wf)))))
        });
    }
    {
        let m = X / Y / Z;
        group.bench_function("div2_deep", |b| {
            b.iter(|| black_box(m.eval((black_box(xf), black_box(yf), black_box(zf), wf))))
        });
    }
    {
        let m = X / Y / Z / W;
        group.bench_function("div3_deep", |b| {
            b.iter(|| black_box(m.eval((black_box(xf), black_box(yf), black_box(zf), black_box(wf)))))
        });
    }
    {
        let m = X.sqrt() + Y / Z;
        group.bench_function("sqrt_div_wide", |b| {
            b.iter(|| black_box(m.eval((black_box(xf), black_box(yf), black_box(zf), wf))))
        });
    }
    {
        let m = X.sqrt() / Y.sqrt();
        group.bench_function("sqrt_div_deep", |b| {
            b.iter(|| black_box(m.eval((black_box(xf), black_box(yf), zf, wf))))
        });
    }

    // Category 3: Distance functions
    {
        let m = (X * X + Y * Y).sqrt();
        group.bench_function("dist2d", |b| {
            b.iter(|| black_box(m.eval((black_box(xf), black_box(yf), zf, wf))))
        });
    }
    {
        let m = (X * X + Y * Y + Z * Z).sqrt();
        group.bench_function("dist3d", |b| {
            b.iter(|| black_box(m.eval((black_box(xf), black_box(yf), black_box(zf), wf))))
        });
    }
    {
        let m = (X * X + Y * Y + Z * Z + W * W).sqrt();
        group.bench_function("dist4d", |b| {
            b.iter(|| black_box(m.eval((black_box(xf), black_box(yf), black_box(zf), black_box(wf)))))
        });
    }
    {
        let m = X * X + Y * Y;
        group.bench_function("dist2d_sq", |b| {
            b.iter(|| black_box(m.eval((black_box(xf), black_box(yf), zf, wf))))
        });
    }
    {
        let m = X * X + Y * Y + Z * Z;
        group.bench_function("dist3d_sq", |b| {
            b.iter(|| black_box(m.eval((black_box(xf), black_box(yf), black_box(zf), wf))))
        });
    }
    {
        let r = 1.0f32;
        let m = (X * X + Y * Y).sqrt() - r;
        group.bench_function("circle_sdf", |b| {
            b.iter(|| black_box(m.eval((black_box(xf), black_box(yf), zf, wf))))
        });
    }
    {
        let r = 1.0f32;
        let m = (X * X + Y * Y + Z * Z).sqrt() - r;
        group.bench_function("sphere_sdf", |b| {
            b.iter(|| black_box(m.eval((black_box(xf), black_box(yf), black_box(zf), wf))))
        });
    }
    {
        let m = X.abs().max(Y.abs());
        group.bench_function("box2d_sdf", |b| {
            b.iter(|| black_box(m.eval((black_box(xf), black_box(yf), zf, wf))))
        });
    }
    {
        let m = X / (X * X + Y * Y).sqrt();
        group.bench_function("normalize_x", |b| {
            b.iter(|| black_box(m.eval((black_box(xf), black_box(yf), zf, wf))))
        });
    }

    // Category 4: Polynomials
    {
        let m = X * 2.0f32 + 1.0f32;
        group.bench_function("linear", |b| {
            b.iter(|| black_box(m.eval((black_box(xf), yf, zf, wf))))
        });
    }
    {
        let m = X * X * 2.0f32 + X * 3.0f32 + 1.0f32;
        group.bench_function("quadratic", |b| {
            b.iter(|| black_box(m.eval((black_box(xf), yf, zf, wf))))
        });
    }
    {
        let m = X * X * X;
        group.bench_function("cubic", |b| {
            b.iter(|| black_box(m.eval((black_box(xf), yf, zf, wf))))
        });
    }
    {
        let m = X * X * X * X;
        group.bench_function("quartic", |b| {
            b.iter(|| black_box(m.eval((black_box(xf), yf, zf, wf))))
        });
    }
    {
        let m = X * X + Y * Y;
        group.bench_function("quad2v", |b| {
            b.iter(|| black_box(m.eval((black_box(xf), black_box(yf), zf, wf))))
        });
    }
    {
        let m = X * X * X + Y * Y * Y;
        group.bench_function("cubic2v", |b| {
            b.iter(|| black_box(m.eval((black_box(xf), black_box(yf), zf, wf))))
        });
    }
    {
        let m = X * Y;
        group.bench_function("cross_xy", |b| {
            b.iter(|| black_box(m.eval((black_box(xf), black_box(yf), zf, wf))))
        });
    }
    {
        let m = X * Y + Y * Z;
        group.bench_function("cross_xyz", |b| {
            b.iter(|| black_box(m.eval((black_box(xf), black_box(yf), black_box(zf), wf))))
        });
    }
    {
        let m = X * X * 2.0f32 + X * Y * 3.0f32 + Y * Y * 4.0f32;
        group.bench_function("full_quad2d", |b| {
            b.iter(|| black_box(m.eval((black_box(xf), black_box(yf), zf, wf))))
        });
    }

    // Category 5: Depth vs width
    {
        let m = (X + Y) + (Z + W);
        group.bench_function("d2w4", |b| {
            b.iter(|| black_box(m.eval((black_box(xf), black_box(yf), black_box(zf), black_box(wf)))))
        });
    }
    {
        let m = ((X + Y) + Z) + W;
        group.bench_function("d3w2_left", |b| {
            b.iter(|| black_box(m.eval((black_box(xf), black_box(yf), black_box(zf), black_box(wf)))))
        });
    }
    {
        let m = X + (Y + (Z + W));
        group.bench_function("d3w2_right", |b| {
            b.iter(|| black_box(m.eval((black_box(xf), black_box(yf), black_box(zf), black_box(wf)))))
        });
    }
    {
        let m = (((X + 1.0f32) + 2.0f32) + 3.0f32) + 4.0f32;
        group.bench_function("d4w1", |b| {
            b.iter(|| black_box(m.eval((black_box(xf), yf, zf, wf))))
        });
    }
    {
        let m = (X.sqrt() + Y.sqrt()) + (Z.sqrt() + W.sqrt());
        group.bench_function("wide_sqrt4", |b| {
            b.iter(|| black_box(m.eval((black_box(xf), black_box(yf), black_box(zf), black_box(wf)))))
        });
    }
    {
        let m = ((X.sqrt() + 1.0f32).sqrt() + 1.0f32).sqrt();
        group.bench_function("deep_sqrt3", |b| {
            b.iter(|| black_box(m.eval((black_box(xf), yf, zf, wf))))
        });
    }
    {
        let m = X / Y + Z / W;
        group.bench_function("wide_div2", |b| {
            b.iter(|| black_box(m.eval((black_box(xf), black_box(yf), black_box(zf), black_box(wf)))))
        });
    }
    {
        let m = ((X / Y) / Z) / W;
        group.bench_function("deep_div3", |b| {
            b.iter(|| black_box(m.eval((black_box(xf), black_box(yf), black_box(zf), black_box(wf)))))
        });
    }

    // Category 6: Min/Max
    {
        let m = X.min(Y);
        group.bench_function("min_xy", |b| {
            b.iter(|| black_box(m.eval((black_box(xf), black_box(yf), zf, wf))))
        });
    }
    {
        let m = X.max(Y);
        group.bench_function("max_xy", |b| {
            b.iter(|| black_box(m.eval((black_box(xf), black_box(yf), zf, wf))))
        });
    }
    {
        // clamp: max(lo, min(x, hi))
        let m = X.min(1.0f32).max(0.0f32);
        group.bench_function("clamp", |b| {
            b.iter(|| black_box(m.eval((black_box(xf), yf, zf, wf))))
        });
    }
    {
        // abs_via_max: max(x, -x) using multiplication by -1
        let neg_x = X * -1.0f32;
        let m = X.max(neg_x);
        group.bench_function("abs_via_max", |b| {
            b.iter(|| black_box(m.eval((black_box(xf), yf, zf, wf))))
        });
    }
    {
        let c1 = (X * X + Y * Y).sqrt() - 1.0f32;
        let c2 = ((X - 3.0f32) * (X - 3.0f32) + Y * Y).sqrt() - 1.0f32;
        let m = c1.min(c2);
        group.bench_function("sdf_union", |b| {
            b.iter(|| black_box(m.eval((black_box(xf), black_box(yf), zf, wf))))
        });
    }
    {
        let c1 = (X * X + Y * Y).sqrt() - 2.0f32;
        let box_sdf = X.abs() - 1.0f32;
        let m = c1.max(box_sdf);
        group.bench_function("sdf_intersect", |b| {
            b.iter(|| black_box(m.eval((black_box(xf), black_box(yf), zf, wf))))
        });
    }

    group.finish();
}

// ============================================================================
// Print training data summary
// ============================================================================

fn print_training_summary(_c: &mut Criterion) {
    let kernels = all_kernel_pairs();

    println!("\n=== Training Data Summary ===");
    println!("Total kernels: {}", kernels.len());

    // Categorize by HCE cost
    let mut by_hce: Vec<_> = kernels.iter().map(|k| (k.hce_cost, k.name)).collect();
    by_hce.sort_by_key(|(cost, _)| *cost);

    println!("\nKernels by HCE cost:");
    for (cost, name) in &by_hce {
        println!("  HCE={:4}: {}", cost, name);
    }

    println!("\n=== Run benchmarks and compare Expr vs Manifold rankings ===");
}

criterion_group!(
    name = nnue_training;
    config = Criterion::default().sample_size(200);
    targets = bench_expr_interpreter, bench_manifold_compiled, print_training_summary,
);

criterion_main!(nnue_training);
