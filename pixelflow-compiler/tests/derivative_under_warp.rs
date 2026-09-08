//! A macro-built derivative composed under a coordinate warp.
//!
//! # This file pins a bug. The asserted numbers are WRONG.
//!
//! `Kernel::dx` is the symbolic derivative with respect to the *sampling*
//! coordinate, so a kernel warped by [`Kernel::at`] must differentiate through
//! the warp — that is the chain rule, and it is the whole reason the font
//! antialiasing ramp is one *screen* pixel wide at any glyph scale (CLAUDE.md,
//! "Antialiasing: symbolic derivatives"; `fonts/ttf.rs`'s module doc).
//!
//! It does not. Both macros resolve `DX` at expansion time, and the later warp
//! then substitutes into the *already-resolved* derivative instead of
//! re-differentiating through it:
//!
//! ```text
//!   d/dx (x²)                 = 2x         → 6 at x=3   (correct, unwarped)
//!   d/dx ((2x)²) = d/dx (4x²) = 8x         → 24 at x=3  (chain rule, TRUTH)
//!   what we compute: (2·u) with u := 2x    → 12 at x=3  (WRONG)
//! ```
//!
//! The production glyph kernels are correct today only by coincidence: all
//! three carry a `&` mask, which makes the e-graph decline the whole arena, so
//! their `Dwrt` nodes survive expansion and are resolved later — after the
//! warp — by the runtime tier. Remove the mask and they would silently take
//! the wrong branch. A flagship kernel whose correctness rests on an unrelated
//! operator is exactly the "convention written in a comment" CLAUDE.md warns
//! about, so it is written down here as an executable record instead.
//!
//! **When the bug is fixed, this file fails.** That is its job. Replace each
//! `WRONG_*` expectation with the `TRUTH_*` value beside it and delete this
//! notice.

use pixelflow_compiler::{kernel, kernel_raw};
use pixelflow_core::{Kernel, Lattice};

/// Warp `k`'s sampling coordinate by `x -> 2x`, leaving `y` alone.
fn warped_by_double_x(k: &Kernel) -> Kernel {
    k.at(&Kernel::x().mul(&Kernel::constant(2.0)), &Kernel::y())
}

/// `d/dx (x²) = 2x`. Unwarped, both macros already agree with the truth —
/// which is what makes the warped case below a warp bug and not a `DX` bug.
#[test]
fn an_unwarped_derivative_is_correct_in_both_macros() {
    const TRUTH_AT_3: f32 = 6.0;

    assert_eq!(
        Lattice::eval_at(&kernel!(|| DX(X * X)), 3.0, 0.0),
        TRUTH_AT_3
    );
    assert_eq!(
        Lattice::eval_at(&kernel_raw!(|| DX(X * X)), 3.0, 0.0),
        TRUTH_AT_3
    );
}

/// The chain rule through [`Kernel::at`], in `kernel!`.
#[test]
fn a_warped_derivative_ignores_the_warp_in_kernel() {
    const TRUTH_AT_3: f32 = 24.0; // d/dx (2x)² = 8x
    const WRONG_AT_3: f32 = 12.0; // 2·(2x), the pre-warp derivative resampled

    let got = Lattice::eval_at(&warped_by_double_x(&kernel!(|| DX(X * X))), 3.0, 0.0);

    assert_eq!(
        got, WRONG_AT_3,
        "pinned bug changed shape: expected the known-wrong {WRONG_AT_3}, \
         got {got} (chain-rule truth is {TRUTH_AT_3})"
    );
}

/// Same, in `kernel_raw!` — the bug is not caused by the e-graph tier, so
/// removing that tier must not be blamed for it and must not be expected to
/// fix it.
#[test]
fn a_warped_derivative_ignores_the_warp_in_kernel_raw() {
    const TRUTH_AT_3: f32 = 24.0;
    const WRONG_AT_3: f32 = 12.0;

    let got = Lattice::eval_at(&warped_by_double_x(&kernel_raw!(|| DX(X * X))), 3.0, 0.0);

    assert_eq!(
        got, WRONG_AT_3,
        "pinned bug changed shape: expected the known-wrong {WRONG_AT_3}, \
         got {got} (chain-rule truth is {TRUTH_AT_3})"
    );
}

/// The composition that *is* correct today, and the reason nobody noticed:
/// a kernel built by composing `Kernel` values differentiates through the warp
/// properly, because nothing resolved the derivative early.
///
/// This is the control. If it ever starts failing, the bug has spread from the
/// macro tier into `Kernel` composition itself.
#[test]
fn a_composed_derivative_does_follow_the_warp() {
    const TRUTH_AT_3: f32 = 24.0;

    let x_squared = Kernel::x().mul(&Kernel::x());
    let got = Lattice::eval_at(&warped_by_double_x(&x_squared).dx(), 3.0, 0.0);

    assert_eq!(
        got, TRUTH_AT_3,
        "composing Kernel values must differentiate through the warp"
    );
}
