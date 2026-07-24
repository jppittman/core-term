//! The reduction binder `⊕_{i ∈ D}` on the `Kernel` surface.
//!
//! One monoid-parametrized fold over a bounded discrete domain covers the
//! programs the language was missing: contraction (matmul), softmax's
//! stabilizer and normalizer, spherical-harmonic projection, and the ambient
//! -occlusion sample sum. The tests below build those shapes and check them
//! against closed forms, plus the two evaluation tiers (IR interpreter and
//! JIT) against each other.

use pixelflow_ir::binding::BindingTable;
use pixelflow_ir::{Kernel, eval_scalar};

/// Evaluate through the IR interpreter — the language's reference semantics.
fn interp(k: &Kernel, x: f32, y: f32) -> f32 {
    let (arena, root) = k.parts();
    eval_scalar(arena, root, &[x, y, 0.0, 0.0], &BindingTable::empty())
}

#[test]
fn sum_over_folds_a_bounded_domain() {
    // Σ_{i<5} i = 10, independent of the coordinates.
    let k = Kernel::sum_over(5, |i| i.clone());
    assert_eq!(interp(&k, 0.0, 0.0), 10.0);

    // Σ_{i<4} (i + X) = 6 + 4X — the body closes over outer coordinates.
    let k = Kernel::sum_over(4, |i| i.add(&Kernel::x()));
    assert_eq!(interp(&k, 2.0, 0.0), 6.0 + 8.0);

    // An empty domain folds to the monoid identity.
    assert_eq!(interp(&Kernel::sum_over(0, |i| i.clone()), 0.0, 0.0), 0.0);
}

#[test]
fn monoids_beyond_sum() {
    // Π_{i<4} (i + 1) = 24
    let one = Kernel::constant(1.0);
    assert_eq!(
        interp(&Kernel::product_over(4, |i| i.add(&one)), 0.0, 0.0),
        24.0
    );

    // max_{i<5} (i · X) with X = -1 is 0 (attained at i = 0).
    let k = Kernel::max_over(5, |i| i.mul(&Kernel::x()));
    assert_eq!(interp(&k, -1.0, 0.0), 0.0);
    assert_eq!(interp(&k, 2.0, 0.0), 8.0);

    // min over the same body is symmetric.
    let k = Kernel::min_over(5, |i| i.mul(&Kernel::x()));
    assert_eq!(interp(&k, -1.0, 0.0), -4.0);

    // Empty domains fold to each monoid's identity.
    assert_eq!(
        interp(&Kernel::product_over(0, |i| i.clone()), 0.0, 0.0),
        1.0
    );
    assert_eq!(
        interp(&Kernel::max_over(0, |i| i.clone()), 0.0, 0.0),
        f32::NEG_INFINITY
    );
}

#[test]
fn nested_binders_do_not_capture() {
    // Σ_{i<3} Σ_{j<4} (i·10 + j): the inner index must stay distinct from the
    // outer one. Σ_i Σ_j (10i + j) = 4·10·(0+1+2) + 3·(0+1+2+3) = 120 + 18.
    let ten = Kernel::constant(10.0);
    let k = Kernel::sum_over(3, |i| {
        let scaled = i.mul(&ten);
        Kernel::sum_over(4, move |j| scaled.add(j))
    });
    assert_eq!(interp(&k, 0.0, 0.0), 138.0);

    // Three deep, with the innermost body reading all three indices:
    // Σ_{a<2} Σ_{b<2} Σ_{c<2} (4a + 2b + c) = 4·(0..7 summed) = 28.
    let k = Kernel::sum_over(2, |a| {
        let a4 = a.mul(&Kernel::constant(4.0));
        Kernel::sum_over(2, move |b| {
            let ab = a4.add(&b.mul(&Kernel::constant(2.0)));
            Kernel::sum_over(2, move |c| ab.add(c))
        })
    });
    assert_eq!(interp(&k, 0.0, 0.0), 28.0);
}

#[test]
fn contraction_is_a_sum_over_a_shared_index() {
    // A matmul row·column: Σ_d q(d)·k(d) with q(d) = d + X, k(d) = 2d.
    // With X = 1: Σ_{d<4} (d+1)(2d) = 2(0 + 2 + 6 + 12) = 40.
    let two = Kernel::constant(2.0);
    let dot = Kernel::sum_over(4, move |d| {
        let q = d.add(&Kernel::x());
        let kv = d.mul(&two);
        q.mul(&kv)
    });
    assert_eq!(interp(&dot, 1.0, 0.0), 40.0);
}

#[test]
fn softmax_shape_normalizes() {
    // The softmax of a bounded score row, evaluated at one position: the
    // max-stabilizer and the sum-normalizer are both this binder, and the
    // whole thing is a pure expression of the coordinates.
    //
    // scores(j) = j · X. p(Y) = exp(s(Y) - m) / Σ_j exp(s(j) - m).
    const N: u32 = 4;
    let scores = |j: &Kernel, x: &Kernel| j.mul(x);

    let x = Kernel::x();
    let m = Kernel::max_over(N, {
        let x = x.clone();
        move |j| scores(j, &x)
    });
    let denom = Kernel::sum_over(N, {
        let (x, m) = (x.clone(), m.clone());
        move |j| scores(j, &x).sub(&m).exp()
    });
    let numer = scores(&Kernel::y(), &x).sub(&m).exp();
    let p = numer.div(&denom);

    // Probabilities over the row must sum to 1 and be ordered by score.
    let total: f32 = (0..N).map(|j| interp(&p, 1.0, j as f32)).sum();
    assert!((total - 1.0).abs() < 1e-5, "softmax row sums to {total}");
    for j in 1..N {
        assert!(
            interp(&p, 1.0, j as f32) > interp(&p, 1.0, (j - 1) as f32),
            "softmax must increase with score at j={j}"
        );
    }
    // Uniform scores (X = 0) give the uniform distribution.
    assert!((interp(&p, 0.0, 2.0) - 0.25).abs() < 1e-5);
}

#[test]
fn occlusion_shaped_accumulation_over_samples() {
    // The AO shape: average a bounded set of samples of a field, each taken at
    // an offset coordinate — `at` warps inside the binder's body.
    // field(x, y) = x, sampled at x + i  =>  (1/N) Σ_{i<N} (X + i) = X + 1.5
    const N: u32 = 4;
    let field = Kernel::x();
    let ao = Kernel::sum_over(N, move |i| {
        field.at(&Kernel::x().add(i), &Kernel::y(), &Kernel::z(), &Kernel::w())
    })
    .div(&Kernel::constant(N as f32));

    assert_eq!(interp(&ao, 10.0, 0.0), 11.5);
}

/// The JIT unrolls reductions (`expand_reduce`) while the interpreter folds
/// them natively; both must denote the same function.
#[cfg(all(target_arch = "x86_64", not(target_feature = "avx512f")))]
mod jit {
    use super::*;
    use pixelflow_ir::backend::emit::compile_arena_dag;
    use pixelflow_ir::jit_manifold::JitManifold;

    fn jit_eval(k: &Kernel, x: f32, y: f32) -> f32 {
        use core::arch::x86_64::*;
        let (arena, root) = k.parts();
        let compiled = compile_arena_dag(arena, root).expect("reduction JIT compile");
        let jit = JitManifold::new(compiled.code);
        unsafe {
            _mm_cvtss_f32(jit.call(
                _mm_set1_ps(x),
                _mm_set1_ps(y),
                _mm_set1_ps(0.0),
                _mm_set1_ps(0.0),
            ))
        }
    }

    fn assert_tiers_agree(k: &Kernel, label: &str) {
        for &(x, y) in &[(0.0f32, 0.0f32), (1.0, 2.0), (-1.5, 0.5), (3.0, 1.0)] {
            let (want, got) = (interp(k, x, y), jit_eval(k, x, y));
            assert!(
                (want - got).abs() < 1e-4,
                "{label}: JIT {got} != interpreter {want} at ({x}, {y})"
            );
        }
    }

    #[test]
    fn unrolled_jit_matches_the_interpreter() {
        assert_tiers_agree(&Kernel::sum_over(6, |i| i.mul(&Kernel::x())), "sum");
        assert_tiers_agree(
            &Kernel::max_over(5, |i| i.sub(&Kernel::y())),
            "max",
        );
        assert_tiers_agree(
            &Kernel::product_over(3, |i| i.add(&Kernel::constant(1.0))),
            "product",
        );

        // Nested: Σ_{i<3} Σ_{j<3} (i·X + j·Y)
        let nested = Kernel::sum_over(3, |i| {
            let ix = i.mul(&Kernel::x());
            Kernel::sum_over(3, move |j| ix.add(&j.mul(&Kernel::y())))
        });
        assert_tiers_agree(&nested, "nested");
    }
}
