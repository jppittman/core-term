//! Headless proof that the two tiers agree on a frame's worth of pixels: the
//! same three channel expressions, once as `kernel!` combinator trees and once
//! as `kernel_jit!` arena kernels, tabulated over the frame's own pixel-center
//! lattice.
//!
//! Neither tier has a per-batch entry any more. The arena tier never did —
//! `kernel_jit!` produces a `Kernel` and a `Kernel` becomes numbers through
//! `Lattice::bake`, which hands the compiler the whole loop nest — and since
//! S4a the combinator tier's only way out is `Lattice::collapse`, the
//! rasterizer that used to sample it a SIMD batch at a time being gone. So
//! both sides here are planes over the same lattice, compared at the 8-bit
//! precision a frame would have shown.
//!
//! The colour pack is not part of the comparison and never was the claim: a
//! scene's channels are packed inside the compiled kernel now
//! (`render::scene::compile_packed_for`), and `pict_color_tests` is what pins
//! that pack against the scalar one.

use pixelflow_compiler::{kernel, kernel_jit};
use pixelflow_core::Lattice;

const W: u32 = 64;
const H: u32 = 64;

/// The pack's own conversion: clamp to `[0, 1]`, scale by 255, truncate
/// toward zero (`cvttps`/`FCVTZS`). Comparing at this precision keeps the
/// tolerance below about drift in the *shader*, not about rounding
/// conventions.
fn to_u8(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0) as u8
}

#[test]
fn baked_jit_channels_match_the_combinator_render() {
    // Channel expressions are pure functions of the pixel coordinates (X, Y),
    // so each compiles once — exactly the compile-on-resize model, no per-frame
    // recompilation.
    //   red   = X * 0.015            (horizontal ramp)
    //   green = Y * 0.015            (vertical ramp)
    //   blue  = sqrt(X*X + Y*Y) * 0.011  (radial)
    //
    // These use only arithmetic and sqrt, which the combinator and arena paths
    // both evaluate accurately, so the combinator side is a faithful
    // reference. (Transcendentals like `sin` are validated numerically against
    // the analytic result elsewhere; here the goal is that the two tiers carry
    // the same picture, and the combinator's own low-degree `sin` is too
    // coarse to serve as a reference.)
    //
    // The lattice is the frame's: pixel (x, y) sampled at (x + ½, y + ½).
    let lattice = Lattice {
        extent: [W, H, 1, 1],
        origin: [0.5, 0.5, 0.0, 0.0],
    };

    let combo = [
        lattice.collapse(&kernel!(|| X * 0.015)()),
        lattice.collapse(&kernel!(|| Y * 0.015)()),
        lattice.collapse(&kernel!(|| (X * X + Y * Y).sqrt() * 0.011)()),
    ];
    let baked = [
        lattice.bake(&kernel_jit!(|| X * 0.015)),
        lattice.bake(&kernel_jit!(|| Y * 0.015)),
        lattice.bake(&kernel_jit!(|| (X * X + Y * Y).sqrt() * 0.011)),
    ];

    // Compare every pixel of every channel. The arena backend lowers sqrt
    // through rsqrt-class estimates, so a few 8-bit quantization levels of
    // drift are expected; a real failure (wrong scene, wrong lattice, wrong
    // wiring) would be off by tens or more.
    let mut max_diff = 0u8;
    let mut worst = (0usize, 0u8, 0u8);
    for (combo, baked) in combo.iter().zip(baked.iter()) {
        for (i, (c, b)) in combo.buffer().iter().zip(baked.buffer()).enumerate() {
            let (c, b) = (to_u8(*c), to_u8(*b));
            let d = c.abs_diff(b);
            if d > max_diff {
                max_diff = d;
                worst = (i, c, b);
            }
        }
    }

    eprintln!(
        "[jit-render] {W}x{H} colored scene: max 8-bit channel diff between the \
         combinator planes and the baked arena planes = {max_diff} \
         (worst pixel {} combo={} baked={})",
        worst.0, worst.1, worst.2
    );

    assert!(
        max_diff <= 2,
        "baked arena planes diverged from the combinator planes by {max_diff} levels (>2)"
    );

    // Sanity: the scene is non-trivial (not all one color), so the test is real.
    let last = (W * H - 1) as usize;
    assert!(
        baked[0].buffer()[0] != baked[0].buffer()[last]
            || baked[2].buffer()[0] != baked[2].buffer()[last],
        "scene should vary across the frame; got uniform output"
    );
}
