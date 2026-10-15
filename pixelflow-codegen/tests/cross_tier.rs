//! AVX2 and AVX-512 agree on one kernel, within tolerance, at well-conditioned
//! points (`docs/plans/2026-10-13-tests-check-the-as-if-contract.md`, "The
//! suite this points at": *across tiers*).
//!
//! The tier is decided once per process from CPUID, with `PIXELFLOW_ISA` only
//! a downward override, so one process cannot run both. The test therefore
//! re-executes its own binary with `PIXELFLOW_ISA=avx2` and a marker; the
//! child evaluates the kernels below on a 64x4 lattice and prints the lanes as
//! hex `f32` bits, and the parent compares them against its own AVX-512
//! evaluation of the same kernels. On a host without AVX-512 there is nothing
//! to compare and the test says so and returns.
//!
//! # What "agree" means
//!
//! Never bit-identical (item 4 of the note: the tiers' estimates differ by
//! design, and the optimizer fuses and reassociates). A lane is compared only
//! where an `f64` oracle written here, sharing no code with the compiler, is
//! well-conditioned: evaluated at `(x, y)` and at `(x, y)(1 ± ε)` it moves by
//! no more than a bounded multiple of `ε` times the magnitude of its terms.
//! Floor, round, truncation, an `If` mask and a kink's neighbours fail that
//! test and are left out of agreement; they are still held to **range**: where
//! the oracle is finite, neither tier may return NaN.
//!
//! The tolerance is each row's, built from the documented accuracy of the ops
//! it holds:
//!
//! - arithmetic only: both tiers round `f32` operations in an order the
//!   optimizer chose (fused or not), so the difference is a few ulps of the
//!   sum of the terms' magnitudes. `1e-5` of it; `1e-4` for the two kernels
//!   that sum thousands of terms;
//! - `Recip` and `Rsqrt` (CLAUDE.md, "Floating point at the edges"): `rcpps`
//!   and `rsqrtps` are good to `1.5 * 2^-12`, `vrcp14ps` and `vrsqrt14ps` to
//!   `2^-14`, so two tiers differ by at most `4.3e-4` of a term: `5e-4`;
//! - `sin`/`cos`: the degree-11 Chebyshev polynomial is accurate to `6e-7`
//!   absolute (`SIN_CHEB` in `pixelflow-ir`'s `passes`), so two tiers differ
//!   by at most about `1.2e-6` plus rounding: `4e-6` absolute;
//! - `exp`/`ln`: the degree-5 Taylor polynomial for `2^f` carries a relative
//!   error under `1.5e-4`; the tiers run the same expansion, so `1e-4`
//!   relative bounds what the difference could be if they ever ran different
//!   ones.
//!
//! Not in the set: `memory`, the one `sibling_rows` kernel that reads a
//! buffer. A gather's index is a truncation, discontinuous in `x` at every
//! lane of an integer lattice, so the conditioning criterion would exclude all
//! of it.

#![cfg(target_arch = "x86_64")]

use std::process::Command;

use pixelflow_codegen::{jit_cache, jit_vector_bytes};
use pixelflow_ir::{Kernel, LatticeShape};

#[allow(dead_code)]
mod sibling_rows {
    include!("support/sibling_rows.rs");
}

/// Set in the child so the same test body knows to report instead of compare.
const CHILD_MARKER: &str = "PIXELFLOW_CROSS_TIER_CHILD";
/// The tier the child is pinned to.
const CHILD_ISA: &str = "avx2";
const WIDE_VECTOR_BYTES: usize = 64;
const NARROW_VECTOR_BYTES: usize = 32;

const COLUMNS: usize = 64;
const ROWS: usize = 4;
/// The lattice's first sample. Positive and off the integers, so `sqrt`,
/// `recip` and `ln` are inside their domains and a rounding operand is not
/// pinned to a tie on every lane.
const ORIGIN: [f32; 2] = [1.1, 1.3];

/// Relative size of the perturbation the conditioning criterion applies:
/// far above `f32` rounding noise (6e-8), far below anything a kernel here
/// varies on.
const EPSILON: f64 = 1e-5;
/// How many multiples of `EPSILON` times the terms' magnitude a conditioned
/// point may move. Noise of a few `f32` ulps (`< 1e-6` of a term) times this
/// stays well inside every tolerance below.
const CONDITION_BOUND: f64 = 64.0;
/// A row in which fewer lanes than this fraction are conditioned compares
/// nothing; that is a defect of the row, not a pass.
const MIN_CONDITIONED_FRACTION: f64 = 0.25;

const ARITHMETIC: Tolerance = Tolerance {
    rel: 1e-5,
    abs: 0.0,
};
const LONG_SUM: Tolerance = Tolerance {
    rel: 1e-4,
    abs: 0.0,
};
const ESTIMATE: Tolerance = Tolerance {
    rel: 5e-4,
    abs: 0.0,
};
const SINUSOID: Tolerance = Tolerance {
    rel: 0.0,
    abs: 4e-6,
};
const POLYNOMIAL: Tolerance = Tolerance {
    rel: 1e-4,
    abs: 0.0,
};

/// `|a - b| <= rel * scale + abs`, with `scale` the oracle's sum of its terms'
/// magnitudes.
#[derive(Clone, Copy)]
struct Tolerance {
    rel: f64,
    abs: f64,
}

/// The oracle's answer at a point: the value, and the magnitude of the terms
/// it was built from.
type Oracle = fn(f64, f64) -> (f64, f64);

struct Row {
    name: &'static str,
    kernel: fn() -> Kernel,
    oracle: Oracle,
    tolerance: Tolerance,
}

/// A constant as the kernel holds it.
fn c(v: f32) -> f64 {
    f64::from(v)
}

fn sibling(build: fn() -> (pixelflow_ir::ExprArena, pixelflow_ir::ExprId)) -> Kernel {
    let (arena, root) = build();
    Kernel::from_parts(arena, root)
}

fn glyph_like() -> Kernel {
    sibling(sibling_rows::glyph_like)
}
fn two_sibling_folds() -> Kernel {
    sibling(sibling_rows::two_sibling_folds)
}
fn parked_roots() -> Kernel {
    sibling(|| sibling_rows::parked_roots(sibling_rows::PARKED_TERMS))
}
fn guarded_if_in_fold() -> Kernel {
    sibling(sibling_rows::guarded_if_in_fold)
}
fn unary_ops() -> Kernel {
    sibling(sibling_rows::unary_ops)
}
fn binary_ops() -> Kernel {
    sibling(sibling_rows::binary_ops)
}
fn shift_muladd_blend() -> Kernel {
    sibling(sibling_rows::shift_muladd_blend)
}
fn deep_frame() -> Kernel {
    sibling(|| sibling_rows::deep_frame(sibling_rows::DEEP_FRAME_TERMS))
}

/// `a·x + b·y`, the argument the transcendental rows share.
fn affine(a: f32, b: f32) -> Kernel {
    Kernel::x()
        .mul(&Kernel::constant(a))
        .add(&Kernel::y().mul(&Kernel::constant(b)))
}
fn sin_row() -> Kernel {
    affine(0.37, 0.11).sin()
}
fn cos_row() -> Kernel {
    affine(0.29, -0.13).cos()
}
fn exp_row() -> Kernel {
    affine(0.05, -0.1).exp()
}
fn ln_row() -> Kernel {
    affine(0.5, 1.0).add(&Kernel::constant(1.0)).ln()
}
fn recip_row() -> Kernel {
    Kernel::x().add(&Kernel::y()).recip()
}
fn rsqrt_row() -> Kernel {
    Kernel::x().add(&Kernel::y()).rsqrt()
}

/// `min(|Σ_i smoothstep(clamp(x/4 + y/16 + 5/16 - i/2))|, 1)`
fn glyph_like_oracle(x: f64, y: f64) -> (f64, f64) {
    let edge = x * c(0.25) + (y * c(0.0625) + c(0.3125));
    let total: f64 = (0..5)
        .map(|i| {
            let t = (edge - f64::from(i) * c(0.5)).clamp(0.0, 1.0);
            t * t * (3.0 - 2.0 * t)
        })
        .sum();
    (total.abs().min(1.0), 1.0)
}

/// `Σ_{i<6} (x/2 + i)·(y/8) + min_{i<4} max(x - 3i/4, -2)`
fn two_sibling_folds_oracle(x: f64, y: f64) -> (f64, f64) {
    let first: f64 = (0..6)
        .map(|i| (x * c(0.5) + f64::from(i)) * (y * c(0.125)))
        .sum();
    let second = (0..4)
        .map(|i| (x - f64::from(i) * c(0.75)).max(c(-2.0)))
        .fold(f64::INFINITY, f64::min);
    (first + second, first.abs() + second.abs())
}

/// `Σ_{i<3} Σ_r min(y·c_r, x + i + c_r)`
fn parked_roots_oracle(x: f64, y: f64) -> (f64, f64) {
    let (mut total, mut scale) = (0.0, 0.0);
    for i in 0..3 {
        for r in 0..sibling_rows::PARKED_TERMS {
            let cr = c(0.001 * (r + 1) as f32);
            let term = (y * cr).min(x + f64::from(i) + cr);
            total += term;
            scale += term.abs();
        }
    }
    (total, scale)
}

/// `Σ_{i<4} if t < 3/2 then exp(sin(1.7·t)/10) else sqrt(t² + 1/4)`, `t = x/5 + 3i/10`
fn guarded_if_in_fold_oracle(x: f64, _y: f64) -> (f64, f64) {
    let (mut total, mut scale) = (0.0, 0.0);
    for i in 0..4 {
        let t = x * c(0.2) + f64::from(i) * c(0.3);
        let term = if t < c(1.5) {
            ((c(1.7) * t).sin() * c(0.1)).exp()
        } else {
            (t * t + c(0.25)).sqrt()
        };
        total += term;
        scale += term.abs();
    }
    (total, scale)
}

/// `Σ_op op(x·k_op + y)`, `k_op = 0.5 + index`, the conversions as `to_float(to_int(·))`.
fn unary_ops_oracle(x: f64, y: f64) -> (f64, f64) {
    let operand = |k: f64| x * c(k as f32) + y;
    let terms = [
        -operand(0.5),
        operand(1.5).sqrt(),
        1.0 / operand(2.5).sqrt(),
        operand(3.5).abs(),
        1.0 / operand(4.5),
        operand(5.5).floor(),
        operand(6.5).ceil(),
        operand(7.5).round(),
        operand(9.5).trunc(),
    ];
    (terms.iter().sum(), terms.iter().map(|t| t.abs()).sum())
}

/// With `p = 3x/2 + y`, `q = 5y/2 + x`: the mask
/// `((p < q & p >= x) | (p <= y | p > q)) & (p == x & p != y)` selects
/// `min(p + q, p - q) · max(p, q) / q` over
/// `to_float((to_int(p) + to_int(q)) & 255 | 1)`.
fn binary_ops_oracle(x: f64, y: f64) -> (f64, f64) {
    let p = c(1.5) * x + y;
    let q = c(2.5) * y + x;
    let mask = ((p < q && p >= x) || (p <= y || p > q)) && (p == x && p != y);
    let value = if mask {
        (p + q).min(p - q) * p.max(q) / q
    } else {
        let bits = ((p.trunc() as i64 + q.trunc() as i64) & 255) | 1;
        bits as f64
    };
    (value, value.abs())
}

/// `to_float(to_int(p) << 3 >> 2) · y + x`, plus `x·2` if `x < y` else `y + 1`
fn shift_muladd_blend_oracle(x: f64, y: f64) -> (f64, f64) {
    let p = c(1.5) * x + y;
    let shifted = ((p.trunc() as i64) << 3 >> 2) as f64;
    let fused = shifted * y + x;
    let blend = if x < y { x * 2.0 } else { y + 1.0 };
    (fused + blend, fused.abs() + blend.abs())
}

/// `(Σ tₖ)²`, `tₖ = x·c_{k mod s} + y·d_{k div s}`
fn deep_frame_oracle(x: f64, y: f64) -> (f64, f64) {
    let side = sibling_rows::DEEP_FRAME_TERMS.isqrt() + 1;
    let sum: f64 = (0..sibling_rows::DEEP_FRAME_TERMS)
        .map(|k| {
            let cx = c(0.25 + (k % side) as f32 * 0.125);
            let cy = c(0.5 + (k / side) as f32 * 0.0625);
            x * cx + y * cy
        })
        .sum();
    (sum * sum, sum * sum)
}

fn sin_oracle(x: f64, y: f64) -> (f64, f64) {
    ((x * c(0.37) + y * c(0.11)).sin(), 1.0)
}
fn cos_oracle(x: f64, y: f64) -> (f64, f64) {
    ((x * c(0.29) + y * c(-0.13)).cos(), 1.0)
}
fn exp_oracle(x: f64, y: f64) -> (f64, f64) {
    let v = (x * c(0.05) + y * c(-0.1)).exp();
    (v, v)
}
fn ln_oracle(x: f64, y: f64) -> (f64, f64) {
    let v = (x * c(0.5) + y * c(1.0) + 1.0).ln();
    (v, v.abs())
}
fn recip_oracle(x: f64, y: f64) -> (f64, f64) {
    let v = 1.0 / (x + y);
    (v, v)
}
fn rsqrt_oracle(x: f64, y: f64) -> (f64, f64) {
    let v = 1.0 / (x + y).sqrt();
    (v, v)
}

const ROW_TABLE: [Row; 14] = [
    Row {
        name: "glyph_like",
        kernel: glyph_like,
        oracle: glyph_like_oracle,
        tolerance: ARITHMETIC,
    },
    Row {
        name: "two_sibling_folds",
        kernel: two_sibling_folds,
        oracle: two_sibling_folds_oracle,
        tolerance: ARITHMETIC,
    },
    Row {
        name: "parked_roots",
        kernel: parked_roots,
        oracle: parked_roots_oracle,
        tolerance: LONG_SUM,
    },
    Row {
        name: "guarded_if_in_fold",
        kernel: guarded_if_in_fold,
        oracle: guarded_if_in_fold_oracle,
        tolerance: POLYNOMIAL,
    },
    Row {
        name: "unary_ops",
        kernel: unary_ops,
        oracle: unary_ops_oracle,
        tolerance: ESTIMATE,
    },
    Row {
        name: "binary_ops",
        kernel: binary_ops,
        oracle: binary_ops_oracle,
        tolerance: ARITHMETIC,
    },
    Row {
        name: "shift_muladd_blend",
        kernel: shift_muladd_blend,
        oracle: shift_muladd_blend_oracle,
        tolerance: ARITHMETIC,
    },
    Row {
        name: "deep_frame",
        kernel: deep_frame,
        oracle: deep_frame_oracle,
        tolerance: LONG_SUM,
    },
    Row {
        name: "sin",
        kernel: sin_row,
        oracle: sin_oracle,
        tolerance: SINUSOID,
    },
    Row {
        name: "cos",
        kernel: cos_row,
        oracle: cos_oracle,
        tolerance: SINUSOID,
    },
    Row {
        name: "exp",
        kernel: exp_row,
        oracle: exp_oracle,
        tolerance: POLYNOMIAL,
    },
    Row {
        name: "ln",
        kernel: ln_row,
        oracle: ln_oracle,
        tolerance: POLYNOMIAL,
    },
    Row {
        name: "recip",
        kernel: recip_row,
        oracle: recip_oracle,
        tolerance: ESTIMATE,
    },
    Row {
        name: "rsqrt",
        kernel: rsqrt_row,
        oracle: rsqrt_oracle,
        tolerance: ESTIMATE,
    },
];

/// The row's lanes on this process's tier, row-major, `COLUMNS` per lattice row.
fn evaluate(kernel: &Kernel) -> Vec<f32> {
    let shape = LatticeShape::new([COLUMNS as u32, ROWS as u32]);
    let compiled = jit_cache::compile(kernel, shape)
        .expect("a kernel in the table compiles")
        .kernel;
    let mut out = vec![f32::NAN; COLUMNS * ROWS];
    let origin = ORIGIN;
    // SAFETY: no kernel in the table declares a buffer or a uniform, so
    // `ctx[0]`, the uniform block's slot, is unread; `ctx[1]` is the origin
    // block of two `f32`s, live for the call; `out` holds the whole lattice at
    // a pitch of `COLUMNS`.
    unsafe {
        let ctx: [*const f32; 2] = [core::ptr::null(), origin.as_ptr()];
        compiled.call(ctx.as_ptr(), out.as_mut_ptr(), COLUMNS);
    }
    out
}

/// The coordinates of lane `index`, as the lattice defines them.
fn coordinates(index: usize) -> (f64, f64) {
    let (column, row) = (index % COLUMNS, index / COLUMNS);
    (
        f64::from(ORIGIN[0] + column as f32),
        f64::from(ORIGIN[1] + row as f32),
    )
}

/// Whether the oracle is well-conditioned at a point: finite, and moved by
/// `(x, y)(1 ± ε)` no more than `CONDITION_BOUND · ε` of its terms' magnitude.
fn well_conditioned(oracle: Oracle, x: f64, y: f64) -> bool {
    let (value, scale) = oracle(x, y);
    let (up, _) = oracle(x * (1.0 + EPSILON), y * (1.0 + EPSILON));
    let (down, _) = oracle(x * (1.0 - EPSILON), y * (1.0 - EPSILON));
    [value, up, down].iter().all(|v| v.is_finite())
        && (up - down).abs() <= 2.0 * CONDITION_BOUND * EPSILON * scale
        && (up - value).abs() <= CONDITION_BOUND * EPSILON * scale
        && (value - down).abs() <= CONDITION_BOUND * EPSILON * scale
}

/// The child's half: print every row's lanes as hex bits.
fn report_lanes() {
    // A leading newline: libtest has left `test name ... ` unterminated.
    println!("\nWIDTH {}", jit_vector_bytes());
    for row in &ROW_TABLE {
        let lanes: Vec<String> = evaluate(&(row.kernel)())
            .iter()
            .map(|v| format!("{:08x}", v.to_bits()))
            .collect();
        println!("LANES {} {}", row.name, lanes.join(" "));
    }
}

/// What the child printed: its vector width, and each row's lanes by name.
fn run_child() -> (usize, Vec<(String, Vec<f32>)>) {
    let exe = std::env::current_exe().expect("the test binary has a path");
    let output = Command::new(exe)
        .args([
            "--exact",
            "avx2_and_avx512_agree",
            "--nocapture",
            "--test-threads=1",
        ])
        .env("PIXELFLOW_ISA", CHILD_ISA)
        .env(CHILD_MARKER, "1")
        .output()
        .expect("the test binary re-executes");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "the {CHILD_ISA} child failed: {}\n{stdout}",
        String::from_utf8_lossy(&output.stderr)
    );
    let mut width = None;
    let mut rows = Vec::new();
    for line in stdout.lines() {
        let mut words = line.split_whitespace();
        match words.next() {
            Some("WIDTH") => width = words.next().and_then(|w| w.parse().ok()),
            Some("LANES") => {
                let name = words.next().expect("a row name").to_string();
                let lanes = words
                    .map(|w| f32::from_bits(u32::from_str_radix(w, 16).expect("hex bits")))
                    .collect();
                rows.push((name, lanes));
            }
            _ => {}
        }
    }
    (width.expect("the child reports its width"), rows)
}

#[test]
fn avx2_and_avx512_agree() {
    if std::env::var_os(CHILD_MARKER).is_some() {
        report_lanes();
        return;
    }
    if !(is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512dq")) {
        println!("skipped: this host has no AVX-512 (avx512f + avx512dq); nothing to compare");
        return;
    }
    if jit_vector_bytes() != WIDE_VECTOR_BYTES {
        println!(
            "skipped: this process is pinned to a {}-byte tier, not AVX-512",
            jit_vector_bytes()
        );
        return;
    }

    let (child_width, child_rows) = run_child();
    assert_eq!(
        child_width, NARROW_VECTOR_BYTES,
        "PIXELFLOW_ISA={CHILD_ISA} must select the {NARROW_VECTOR_BYTES}-byte tier"
    );
    assert_eq!(
        child_rows.len(),
        ROW_TABLE.len(),
        "the child reports every row"
    );

    for (row, (child_name, narrow)) in ROW_TABLE.iter().zip(&child_rows) {
        assert_eq!(
            row.name, child_name,
            "the child reports rows in table order"
        );
        let wide = evaluate(&(row.kernel)());
        assert_eq!(wide.len(), narrow.len(), "{}: lane counts differ", row.name);

        let mut conditioned = 0usize;
        for (index, (&w, &n)) in wide.iter().zip(narrow).enumerate() {
            let (x, y) = coordinates(index);
            let (value, scale) = (row.oracle)(x, y);

            // Range holds everywhere the oracle has an answer.
            if value.is_finite() {
                assert!(
                    !w.is_nan() && !n.is_nan(),
                    "{} at ({x}, {y}): AVX-512 gave {w}, AVX2 gave {n}; the oracle's answer is {value}",
                    row.name
                );
            }
            if !well_conditioned(row.oracle, x, y) {
                continue;
            }
            conditioned += 1;
            let bound = row.tolerance.rel * scale + row.tolerance.abs;
            assert!(
                (f64::from(w) - f64::from(n)).abs() <= bound,
                "{} at ({x}, {y}): AVX-512 gave {w}, AVX2 gave {n}, differ by more than {bound:e} (oracle {value})",
                row.name
            );
        }
        let fraction = conditioned as f64 / wide.len() as f64;
        println!(
            "{}: {conditioned}/{} lanes conditioned",
            row.name,
            wide.len()
        );
        assert!(
            fraction >= MIN_CONDITIONED_FRACTION,
            "{}: only {conditioned} of {} lanes are well-conditioned; the row compares too little",
            row.name,
            wide.len()
        );
    }
}
