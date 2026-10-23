//! A fold whose bound is a value runs, on every tier, and computes its fold or
//! NaN: never a hang, never a trip count converted to an integer, never a read
//! past a buffer (docs/language/2026-10-20-a-bound-is-a-value.md, L2 and L5).
//!
//! For a bound `v` at run time, `v` is a count when `0 <= v <= 2^24` and
//! `floor(v) == v`. Then the fold is `⊕_{k ∈ [lo, hi)}`, the monoid's identity
//! over a range that runs backwards; and for any other `v` — a fraction, a
//! negative, NaN, an infinity, `2^24 + 2` — the fold is NaN. The guard that
//! says so is `passes::legalize`'s last pass, and the oracles here are
//! explicit folds in scalar Rust, from the monoid's identity.
//!
//! Every count is an integer below `2^24` and every sample is an integer, so
//! each sum is exact in any order of summation (the optimizer may reassociate)
//! and the answer is compared bit for bit where it is finite.
//!
//! A kernel runs two ways. `run_raw` compiles the arena as written, through
//! `emit::compile`: the loops are the ones written, and the guard and the
//! allocator are what is under test. `run_linked` goes through
//! `jit_cache::compile`, the optimizer first: it may peel a fold or factor an
//! invariant operand out of it, and the answer must not change.

#![cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]

use std::sync::Arc;
use std::sync::mpsc::{self, RecvTimeoutError};
use std::time::Duration;

use pixelflow_codegen::emit::{self, EmitTraffic, Trips};
use pixelflow_codegen::isa::{self, Isa};
use pixelflow_codegen::jit_cache;
use pixelflow_ir::fold::{Binder, Bound, Fold, Monoid};
use pixelflow_ir::{ExprArena, ExprId, Kernel, LatticeShape, OpKind, Uniform};

/// Wide enough for a main column fold at every lane count (4, 8 and 16), and
/// several rows so the row fold is a loop around the columns.
const SHAPE: LatticeShape = LatticeShape::new([64, 4]);

/// A width that leaves a remainder at every lane count, so the column fold is
/// strip-mined and the same `Write` is stored by a main fold and a remainder.
const RAGGED: LatticeShape = LatticeShape::new([37, 3]);

/// Where the lattice's sample `(0, 0)` lies: integers, so every sample is.
const ORIGIN: [f32; 2] = [1.0, 2.0];

/// The most a test waits for a kernel before it calls it a hang.
const HANG: Duration = Duration::from_secs(60);

/// `2^24`, the largest count: the last integer to which an `f32` names every
/// integer.
const TOP: f32 = 16_777_216.0;

/// A per-call count: a uniform holding `v`.
fn count(v: f32) -> Kernel {
    Uniform::new(v).kernel()
}

/// `0` as a bound.
fn zero() -> Kernel {
    Kernel::constant(0.0)
}

/// Run `f` on a thread and wait for it a minute: a fold that hangs is a failed
/// assertion, not a CI timeout. A panic in `f` is the caller's panic, with its
/// message.
fn within<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> T {
    let (tx, rx) = mpsc::channel();
    let worker = std::thread::spawn(move || {
        if let Err(gone) = tx.send(f()) {
            panic!("the test stopped listening before its kernel returned: {gone}");
        }
    });
    match rx.recv_timeout(HANG) {
        Ok(value) => value,
        Err(RecvTimeoutError::Timeout) => {
            panic!("a fold whose bound is outside its domain never hangs: no answer in {HANG:?}")
        }
        Err(RecvTimeoutError::Disconnected) => match worker.join() {
            Err(panic) => std::panic::resume_unwind(panic),
            Ok(()) => unreachable!("a worker that sent its answer is not disconnected"),
        },
    }
}

/// `arena` compiled as written at `shape`, called with `block` as its uniform
/// block, and the plane it fills with the traffic the compile counted.
fn run_arena(
    arena: &ExprArena,
    root: ExprId,
    shape: LatticeShape,
    block: &[f32],
) -> (Vec<f32>, EmitTraffic) {
    let compiled = emit::compile(arena, root, shape).expect("the kernel compiles");
    (call(&compiled.code, block, shape), compiled.traffic)
}

/// The plane `code` fills at `shape`, its uniform block `block`.
fn call(code: &emit::CompiledKernel, block: &[f32], shape: LatticeShape) -> Vec<f32> {
    let [width, height] = shape.extent().map(|n| n as usize);
    let mut out = vec![f32::NAN; width * height];
    // No buffers: the uniform block, then the origin's.
    let ctx: [*const f32; 2] = [block.as_ptr(), ORIGIN.as_ptr()];
    // SAFETY: the kernels here declare no buffer, `ctx` holds the uniform
    // block they were linked against and the origin block, each live for the
    // call, and `out` is the whole plane of the extent they were compiled at.
    unsafe { code.call(ctx.as_ptr(), out.as_mut_ptr(), width) };
    out
}

/// `kernel` as written, with the uniforms at their defaults.
fn run_raw(kernel: &Kernel, shape: LatticeShape) -> (Vec<f32>, EmitTraffic) {
    let (arena, root) = kernel.parts();
    let block: Vec<f32> = arena.uniforms().iter().map(|d| d.default).collect();
    run_arena(arena, root, shape, &block)
}

/// `kernel` through the optimizer and the shape-keyed cache, with the uniforms
/// at their defaults in the link's order.
fn run_linked(kernel: &Kernel, shape: LatticeShape) -> Vec<f32> {
    let linked = jit_cache::compile(kernel, shape).expect("the kernel compiles");
    let block: Vec<f32> = linked.uniforms.iter().map(|d| d.default).collect();
    call(&linked.kernel, &block, shape)
}

/// What a sample's `(x, y)` is, at `(col, row)`.
fn at(shape: LatticeShape, i: usize) -> (f32, f32) {
    let width = shape.extent()[0] as usize;
    (
        ORIGIN[0] + (i % width) as f32,
        ORIGIN[1] + (i / width) as f32,
    )
}

/// A built kernel's answer on both paths, labelled, each inside the watchdog.
fn both(
    shape: LatticeShape,
    build: impl Fn() -> Kernel + Send + Copy + 'static,
) -> [(&'static str, Vec<f32>); 2] {
    [
        ("as written", within(move || run_raw(&build(), shape).0)),
        ("optimized", within(move || run_linked(&build(), shape))),
    ]
}

/// Every sample of both paths is `want(x, y)`, bit for bit; a NaN is only a
/// NaN.
#[track_caller]
fn assert_plane<const PATHS: usize>(
    shape: LatticeShape,
    planes: [(&'static str, Vec<f32>); PATHS],
    context: &str,
    want: impl Fn(f32, f32) -> f32,
) {
    for (path, plane) in planes {
        assert_eq!(
            plane.len(),
            (shape.extent()[0] * shape.extent()[1]) as usize
        );
        for (i, &got) in plane.iter().enumerate() {
            let (x, y) = at(shape, i);
            let want = want(x, y);
            let same = match want.is_nan() {
                true => got.is_nan(),
                false => got.to_bits() == want.to_bits(),
            };
            assert!(
                same,
                "{context} ({path}) at ({x}, {y}): got {got} ({:#010x}), want {want} ({:#010x})",
                got.to_bits(),
                want.to_bits()
            );
        }
    }
}

/// `Σ_{k<n} (X + k)`, one program for every `n`.
fn sum_of_x_plus_k(n: f32) -> Kernel {
    Kernel::over_values(Monoid::SUM, &zero(), &count(n), |k| Kernel::x().add(k))
}

/// The oracle for [`sum_of_x_plus_k`] and its relatives: an explicit fold from
/// the monoid's identity over `lo..hi`, of `term(k)`.
fn fold_of(monoid: Monoid, lo: u64, hi: u64, term: impl Fn(f32) -> f32) -> f32 {
    let identity = f32::from_bits(monoid.identity().bits());
    let combine = |a: f32, b: f32| match monoid {
        Monoid::SUM => a + b,
        Monoid::PRODUCT => a * b,
        Monoid::MIN => a.min(b),
        Monoid::MAX => a.max(b),
        _ => unreachable!("a value bound takes SUM, PRODUCT, MIN or MAX"),
    };
    (lo..hi).fold(identity, |acc, k| combine(acc, term(k as f32)))
}

/// A count in the domain runs its fold: `n` trips, `n = 0` and `-0.0` the
/// identity.
#[test]
fn a_count_in_the_domain_runs_its_fold() {
    for n in [0.0f32, -0.0, 1.0, 7.0, 1000.0] {
        let planes = both(SHAPE, move || sum_of_x_plus_k(n));
        assert_plane(SHAPE, planes, &format!("Σ_{{k<{n}}} (X + k)"), |x, _| {
            fold_of(Monoid::SUM, 0, n as u64, |k| x + k)
        });
    }
}

/// The column fold's remainder stores the same value as its main fold, and
/// both read the one guard.
#[test]
fn a_ragged_row_runs_the_guard_in_both_column_folds() {
    for n in [0.0f32, 7.0, f32::NAN] {
        let planes = both(RAGGED, move || sum_of_x_plus_k(n));
        assert_plane(
            RAGGED,
            planes,
            &format!("ragged, n = {n}"),
            |x, _| match n.is_nan() {
                true => f32::NAN,
                false => fold_of(Monoid::SUM, 0, n as u64, |k| x + k),
            },
        );
    }
}

/// `2^24` is the last count: the counter reaches it exactly, and the latch's
/// compare fires there. The body is constant, so the loop runs once per call.
#[test]
fn a_count_at_the_exact_bound_runs_every_trip() {
    let build =
        || Kernel::over_values(Monoid::SUM, &zero(), &count(TOP), |_| Kernel::constant(1.0));
    let planes = both(LatticeShape::POINT, build);
    assert_plane(LatticeShape::POINT, planes, "Σ_{k<2^24} 1", |_, _| TOP);
}

/// Outside the domain the fold is NaN, and the loop is not entered: `+∞` and
/// `2^24 + 2` would stall an `f32` counter at `2^24` for ever, and NaN never
/// satisfies an ordered compare. The watchdog turns a hang into a failure.
#[test]
fn a_count_outside_the_domain_is_nan_and_never_hangs() {
    // No subnormal: whether one reads as 0 depends on the thread's denormal
    // mode, not on the guard.
    let outside = [
        TOP + 2.0,
        f32::NAN,
        2.5,
        -1.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        1e30,
    ];
    for n in outside {
        let planes = both(SHAPE, move || sum_of_x_plus_k(n));
        assert_plane(SHAPE, planes, &format!("n = {n}"), |_, _| f32::NAN);
    }
}

/// `2^24 + 1` is not a case: no `f32` names it, and `16777217.0` rounds to
/// `2^24`, which is a count.
#[test]
fn the_f32_nearest_two_to_the_24_plus_one_is_two_to_the_24() {
    assert_eq!(16_777_217.0f32, TOP);
}

/// A bound a rule or a hand made from a real that names no count is kept a
/// value, not refused (the arena's `push_reduce`), and its guard makes the
/// fold NaN rather than a panic.
#[test]
fn a_literal_a_rule_made_outside_the_domain_is_nan() {
    let mut a = ExprArena::new();
    let binder = Binder::from_slot(0).expect("slot 0 exists");
    let (x, k) = (a.push_var(0), a.push_var(binder.var()));
    let body = a.push_binary(OpKind::Add, x, k);
    let hi = a.push_const(2.5);
    let root = a.push_reduce(
        Fold::new(Monoid::SUM, binder),
        Bound::Lit(0),
        Bound::Value(hi),
        body,
    );
    let (plane, _) = within(move || run_arena(&a, root, SHAPE, &[]));
    assert!(plane.iter().all(|v| v.is_nan()), "{plane:?}");
}

/// A pair that runs backwards at run time is the monoid's identity, for each
/// monoid a value bound takes.
#[test]
fn a_pair_that_runs_backwards_is_the_identity() {
    for (monoid, identity) in [
        (Monoid::SUM, 0.0f32),
        (Monoid::PRODUCT, 1.0),
        (Monoid::MIN, f32::INFINITY),
        (Monoid::MAX, f32::NEG_INFINITY),
    ] {
        for (lo, hi) in [(5.0f32, 3.0f32), (4.0, 4.0)] {
            let planes = both(SHAPE, move || {
                Kernel::over_values(monoid, &count(lo), &count(hi), |k| Kernel::x().add(k))
            });
            assert_plane(
                SHAPE,
                planes,
                &format!("{monoid:?} [{lo}, {hi})"),
                |_, _| identity,
            );
        }
    }
}

/// A value `lo` starts the fold there: the counter is seeded with its lane.
#[test]
fn a_value_lo_starts_the_fold() {
    let planes = both(SHAPE, || {
        Kernel::over_values(Monoid::SUM, &count(3.0), &count(7.0), |k| {
            Kernel::x().add(k)
        })
    });
    assert_plane(SHAPE, planes, "Σ_{k=3}^{6} (X + k)", |x, _| {
        fold_of(Monoid::SUM, 3, 7, |k| x + k)
    });
}

/// `Σ_{i<8} Σ_{j<i} (X + j)`: the inner `hi` is the outer fold's index, so
/// the guard runs once per outer trip. As written the inner bound stays a
/// value; optimized, the outer fold may peel and make it known (L8).
#[test]
fn a_triangular_sum_runs_each_inner_fold_to_its_outer_index() {
    let build = || {
        Kernel::sum_over(8, |i| {
            Kernel::over_values(Monoid::SUM, &zero(), i, |j| Kernel::x().add(j))
        })
    };
    let planes = both(SHAPE, build);
    assert_plane(SHAPE, planes, "Σ_i Σ_{j<i} (X + j)", |x, _| {
        fold_of(Monoid::SUM, 0, 8, |i| {
            fold_of(Monoid::SUM, 0, i as u64, |j| x + j)
        })
    });
}

/// The same under a per-call count for the outer bound: `n = 0`, `5`, NaN.
#[test]
fn a_triangular_sum_under_a_count() {
    for n in [0.0f32, 5.0, f32::NAN] {
        let planes = both(SHAPE, move || {
            Kernel::over_values(Monoid::SUM, &zero(), &count(n), |i| {
                Kernel::over_values(Monoid::SUM, &zero(), i, |j| Kernel::x().add(j))
            })
        });
        assert_plane(SHAPE, planes, &format!("outer n = {n}"), |x, _| {
            match n.is_nan() {
                true => f32::NAN,
                false => fold_of(Monoid::SUM, 0, n as u64, |i| {
                    fold_of(Monoid::SUM, 0, i as u64, |j| x + j)
                }),
            }
        });
    }
}

/// Two folds each over a count of its own, the inner one inside the outer's
/// body: the inner guard is the outer loop's, run once per outer trip, and
/// either count outside the domain makes the whole NaN.
#[test]
fn two_counts_nest() {
    let nan = f32::NAN;
    for (n, m) in [
        (0.0f32, 3.0f32),
        (3.0, 0.0),
        (3.0, 4.0),
        (nan, 4.0),
        (3.0, nan),
    ] {
        let planes = both(SHAPE, move || {
            Kernel::over_values(Monoid::SUM, &zero(), &count(n), |i| {
                Kernel::over_values(Monoid::SUM, &zero(), &count(m), |j| {
                    Kernel::x().add(i).add(j)
                })
            })
        });
        assert_plane(SHAPE, planes, &format!("n = {n}, m = {m}"), |x, _| {
            match n.is_nan() || m.is_nan() {
                true => f32::NAN,
                false => fold_of(Monoid::SUM, 0, n as u64, |i| {
                    fold_of(Monoid::SUM, 0, m as u64, |j| x + i + j)
                }),
            }
        });
    }
}

/// A bound the body reads as well: `Σ_{k<n} k·n`, one uniform, two readers.
#[test]
fn a_bound_the_body_reads_too() {
    for n in [0.0f32, 1.0, 7.0, 2.5] {
        let planes = both(SHAPE, move || {
            let n = Uniform::new(n);
            Kernel::over_values(Monoid::SUM, &zero(), &n.kernel(), |k| k.mul(&n.kernel()))
        });
        assert_plane(SHAPE, planes, &format!("n = {n}"), |_, _| {
            match n.fract() != 0.0 {
                true => f32::NAN,
                false => fold_of(Monoid::SUM, 0, n as u64, |k| k * n),
            }
        });
    }
}

/// A literal `lo` beside a value `hi`: only the value is in the domain test.
#[test]
fn a_literal_lo_beside_a_value_hi() {
    for n in [0.0f32, 2.0, 3.0, 9.0, -1.0] {
        let planes = both(SHAPE, move || {
            Kernel::over_values(Monoid::SUM, &Kernel::constant(2.0), &count(n), |k| {
                Kernel::x().add(k)
            })
        });
        assert_plane(SHAPE, planes, &format!("[2, {n})"), |x, _| match n < 0.0 {
            true => f32::NAN,
            false => fold_of(Monoid::SUM, 2, n as u64, |k| x + k),
        });
    }
}

/// `Σ_{i<8} Σ_{j∈[i,8)} (X + j)`: a value `lo` that is an enclosing index.
/// The inner counter is seeded with the outer loop's own counter, which is
/// live after the jump (the outer latch steps it), so the allocator copies it
/// rather than handing its lease over. No other kernel gives a loop's
/// parameter as another loop's entry argument.
#[test]
fn an_inner_fold_starting_at_the_outer_index() {
    let build = || {
        Kernel::sum_over(8, |i| {
            Kernel::over_values(Monoid::SUM, i, &Kernel::constant(8.0), |j| {
                Kernel::x().add(j)
            })
        })
    };
    let planes = both(SHAPE, build);
    assert_plane(SHAPE, planes, "Σ_i Σ_{j∈[i,8)} (X + j)", |x, _| {
        fold_of(Monoid::SUM, 0, 8, |i| {
            fold_of(Monoid::SUM, i as u64, 8, |j| x + j)
        })
    });
}

/// `X + Σ_{k<n} k`: the fold reads no coordinate, so its guard and its loop
/// are call-invariant and the carve gives them the body, under the guard's
/// arms. The loop runs once per call, not once per sample.
#[test]
fn a_fold_that_reads_no_coordinate_is_guarded_once_per_call() {
    for n in [0.0f32, 7.0, f32::NAN] {
        let (plane, traffic) = within(move || {
            let sum = Kernel::over_values(Monoid::SUM, &zero(), &count(n), |k| k.clone());
            run_raw(&Kernel::x().add(&sum), SHAPE)
        });
        let want = |x: f32| match n.is_nan() {
            true => f32::NAN,
            false => x + fold_of(Monoid::SUM, 0, n as u64, |k| k),
        };
        assert_plane(
            SHAPE,
            [("as written", plane)],
            &format!("n = {n}"),
            |x, _| want(x),
        );

        let estimates: Vec<_> = traffic
            .trips
            .iter()
            .filter(|t| matches!(t, Trips::Estimated(_)))
            .collect();
        assert_eq!(
            estimates,
            [&Trips::Estimated(Fold::ESTIMATED_TRIPS)],
            "the one loop with a bound that is a value runs once per call: {:?}",
            traffic.trips
        );
        assert_eq!(traffic.dynamic_memory_ops(), None);
    }
}

/// More values live at once than any tier's vector file holds.
const WALL: u32 = 48;

/// `Σ_m (k + X)·(m + 1)` summed in both orders: [`WALL`] products all live
/// until the last is added, so every vector register is reused inside a loop
/// whose body this is.
fn a_wall_of_values(k: &Kernel) -> Kernel {
    let terms: Vec<Kernel> = (0..WALL)
        .map(|m| Kernel::x().add(k).mul(&Kernel::constant((m + 1) as f32)))
        .collect();
    let chain = |terms: &[Kernel]| {
        terms[1..]
            .iter()
            .fold(terms[0].clone(), |acc, term| acc.add(term))
    };
    let reversed: Vec<Kernel> = terms.iter().rev().cloned().collect();
    chain(&terms).add(&chain(&reversed))
}

/// A wall of values live at once, so `hi`, which the body does not read,
/// survives to the latch only if its interval spans the loop.
///
/// `Σ_{k<n} Σ_m (X + k)·(m + 1)`, twice over.
#[test]
fn a_bound_read_only_by_the_loop_reaches_its_latch() {
    for n in [0.0f32, 1.0, 7.0, f32::NAN] {
        let (plane, _) = within(move || {
            run_raw(
                &Kernel::over_values(Monoid::SUM, &zero(), &count(n), a_wall_of_values),
                SHAPE,
            )
        });
        let weight: f32 = (1..=WALL).map(|m| m as f32).sum();
        assert_plane(
            SHAPE,
            [("as written", plane)],
            &format!("n = {n}"),
            |x, _| match n.is_nan() {
                true => f32::NAN,
                false => fold_of(Monoid::SUM, 0, n as u64, |k| 2.0 * weight * (x + k)),
            },
        );
    }
}

/// What the compile counted for `kernel` as written at [`SHAPE`], and the
/// frame it needed.
fn traffic_of(kernel: &Kernel) -> (EmitTraffic, u64) {
    let (arena, root) = kernel.parts();
    let compiled = emit::compile(arena, root, SHAPE).expect("the kernel compiles");
    (compiled.traffic, compiled.frame_bytes)
}

/// The wall over `hi` trips when `hi` is a count, and over a literal seven
/// when it is `None`: the kernel a value bound is compared with.
fn the_wall_over(hi: Option<f32>) -> Kernel {
    match hi {
        Some(n) => Kernel::over_values(Monoid::SUM, &zero(), &count(n), a_wall_of_values),
        None => Kernel::sum_over(7, a_wall_of_values),
    }
}

/// A value `hi` that the allocator does not keep in a register across the
/// loop is reloaded at every latch: the latch's compare reads a register, so
/// it is one load in the fold's scope per trip. A literal `hi` is a constant
/// the allocator places again at its read, and costs none.
#[test]
fn a_bound_that_is_not_carried_is_reloaded_at_every_latch() {
    let (value, _) = traffic_of(&the_wall_over(Some(7.0)));
    let (literal, _) = traffic_of(&the_wall_over(None));
    let fold = |t: &EmitTraffic| *t.scopes.last().expect("a nest has a body");
    assert_eq!(
        fold(&value).loads,
        fold(&literal).loads + 1,
        "the fold's scope reloads its end once more than a literal's"
    );
    assert_eq!(
        value.trips.last().map(|t| t.known()),
        Some(None),
        "the loop's trips are an estimate: {:?}",
        value.trips
    );
}

/// A guard whose masks are the call's but whose fold is the column's: its two
/// masks are computed once, live across every batch, and are read again by
/// the guard's two `If`s in each, on every tier. Where they live between
/// batches is the tier's: the opmask file (AVX-512) has no register to keep
/// them in, so each batch reloads them and the frame grows by the slots they
/// live in, narrow ones among them; NEON's vector file keeps them resident.
#[test]
fn a_guard_computed_once_per_call_is_read_again_in_every_batch() {
    let over = |hi: Option<f32>| match hi {
        Some(n) => Kernel::x().mul(&sum_of_x_plus_k(n)),
        None => Kernel::x().mul(&Kernel::sum_over(7, |k| Kernel::x().add(k))),
    };
    let (value, value_frame) = traffic_of(&over(Some(7.0)));
    let (literal, literal_frame) = traffic_of(&over(None));
    let column = |t: &EmitTraffic| t.scopes[2];
    assert_eq!(
        column(&value).instructions,
        column(&literal).instructions + 2,
        "the column fold's every batch runs the guard's two selects, one per mask: \
         {:?} against {:?}",
        column(&value),
        column(&literal)
    );
    if isa::detect() != Isa::Avx512 {
        return;
    }
    assert!(
        column(&value).loads > column(&literal).loads,
        "each batch reads the opmasks back: {:?} against {:?}",
        column(&value),
        column(&literal)
    );
    assert!(
        value_frame > literal_frame,
        "the opmasks are slots in the frame: {value_frame} against {literal_frame}"
    );
}

/// One program and one code pointer for every `n`: a fresh uniform per count,
/// the cache keyed on structure, and each kernel computing its own `n`.
#[test]
fn every_count_is_one_program() {
    let linked: Vec<_> = [0.0f32, 1.0, 7.0, 1000.0]
        .into_iter()
        .map(|n| {
            let linked = within(move || {
                jit_cache::compile(&sum_of_x_plus_k(n), SHAPE).expect("the kernel compiles")
            });
            (n, linked)
        })
        .collect();
    let (_, first) = &linked[0];
    for (n, each) in &linked {
        assert!(
            Arc::ptr_eq(&first.kernel, &each.kernel),
            "n = {n} compiled to a program of its own"
        );
        let block: Vec<f32> = each.uniforms.iter().map(|d| d.default).collect();
        let code = Arc::clone(&each.kernel);
        let plane = within(move || call(&code, &block, SHAPE));
        let planes = [("shared code", plane)];
        assert_plane(SHAPE, planes, &format!("n = {n}"), |x, _| {
            fold_of(Monoid::SUM, 0, *n as u64, |k| x + k)
        });
    }
}

/// The other three monoids a value bound takes, under the guard. At `n = 0`
/// the fold is the monoid's identity, `−∞` for `MAX` and `+∞` for `MIN`, not
/// the formula that holds for `n ≥ 1`.
#[test]
fn each_monoid_runs_under_its_guard() {
    for n in [0.0f32, 1.0, 7.0, 2.5] {
        let outside = n.fract() != 0.0;
        let trips = n as u64;

        let planes = both(SHAPE, move || {
            Kernel::over_values(Monoid::MAX, &zero(), &count(n), |k| Kernel::x().sub(k))
        });
        assert_plane(SHAPE, planes, &format!("MAX, n = {n}"), |x, _| {
            match (outside, n == 0.0) {
                (true, _) => f32::NAN,
                (false, true) => f32::NEG_INFINITY,
                (false, false) => x,
            }
        });

        let planes = both(SHAPE, move || {
            Kernel::over_values(Monoid::MIN, &zero(), &count(n), |k| Kernel::x().sub(k))
        });
        assert_plane(SHAPE, planes, &format!("MIN, n = {n}"), |x, _| {
            match (outside, n == 0.0) {
                (true, _) => f32::NAN,
                (false, true) => f32::INFINITY,
                (false, false) => x - (n - 1.0),
            }
        });

        let planes = both(SHAPE, move || {
            Kernel::over_values(Monoid::PRODUCT, &zero(), &count(n), |k| {
                k.add(&Kernel::constant(1.0))
            })
        });
        assert_plane(
            SHAPE,
            planes,
            &format!("PRODUCT, n = {n}"),
            |_, _| match outside {
                true => f32::NAN,
                false => fold_of(Monoid::PRODUCT, 0, trips, |k| k + 1.0),
            },
        );
    }
}

/// A bound that is a mask and a number is a count: `3` where the mask holds
/// and `0` where it does not. No tier holds that in a file a loop's seed
/// cannot be in, so it compiles and runs on all of them, `lo` and `hi` alike.
#[test]
fn a_mask_and_a_number_is_a_count_on_every_tier() {
    for (u, trips) in [(5.0f32, 3u64), (-1.0, 0)] {
        let mut a = ExprArena::new();
        let slot = a.declare_uniform(Uniform::new(u).decl());
        let uniform = a.push_uniform(slot);
        let (zero, three) = (a.push_const(0.0), a.push_const(3.0));
        let positive = a.push_binary(OpKind::Lt, zero, uniform);
        let number = a.push_binary(OpKind::BitAnd, positive, three);
        let binder = Binder::from_slot(0).expect("slot 0 exists");
        let (x, k) = (a.push_var(0), a.push_var(binder.var()));
        let body = a.push_binary(OpKind::Add, x, k);

        let ends = a.push_reduce(
            Fold::new(Monoid::SUM, binder),
            Bound::Lit(0),
            Bound::Value(number),
            body,
        );
        let starts = a.push_reduce(
            Fold::new(Monoid::SUM, binder),
            Bound::Value(number),
            Bound::Lit(3),
            body,
        );
        let root = a.push_binary(OpKind::Add, ends, starts);
        let (plane, _) = within(move || run_arena(&a, root, SHAPE, &[u]));
        let planes = [("as written", plane)];
        assert_plane(SHAPE, planes, &format!("u = {u}"), |x, _| {
            fold_of(Monoid::SUM, 0, trips, |k| x + k) + fold_of(Monoid::SUM, trips, 3, |k| x + k)
        });
    }
}

/// A mask is refused as a bound by name, on every tier alike: AVX-512 holds
/// one in an opmask, which no loop parameter may be, and the others would
/// read its bits as a count.
#[test]
#[should_panic(expected = "BoundSort")]
fn a_bound_that_is_a_mask_is_refused_by_name() {
    let mut a = ExprArena::new();
    let slot = a.declare_uniform(Uniform::new(5.0).decl());
    let uniform = a.push_uniform(slot);
    let zero = a.push_const(0.0);
    let positive = a.push_binary(OpKind::Lt, zero, uniform);
    let binder = Binder::from_slot(0).expect("slot 0 exists");
    let (x, k) = (a.push_var(0), a.push_var(binder.var()));
    let body = a.push_binary(OpKind::Add, x, k);
    let root = a.push_reduce(
        Fold::new(Monoid::SUM, binder),
        Bound::Value(positive),
        Bound::Lit(3),
        body,
    );
    let _refused = emit::compile(&a, root, SHAPE);
}
