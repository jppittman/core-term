//! Headless proof that a colored scene's channels, compiled by the arena
//! backend and **baked over the frame lattice**, match the combinator render
//! of the same expressions pixel for pixel.
//!
//! The arena tier has no per-batch entry: `kernel_jit!` produces a `Kernel`,
//! and a `Kernel` becomes numbers only through `Lattice::bake`, which hands
//! the compiler the whole loop nest. So the JIT side of this comparison is
//! three baked channel planes over the frame's pixel-center lattice, and the
//! reference side is the unchanged `rasterize()` path over a `ColorCube` of
//! combinator channels.
//!
//! (Rendering the *arena* side through the engine is S2's subject — a scene as
//! a packed kernel over the frame lattice. Until then this test pins the
//! numerics of the two tiers on a real frame.)

use pixelflow_compiler::{kernel, kernel_jit};
use pixelflow_core::Lattice;
use pixelflow_graphics::render::color::PlatformPixel;
use pixelflow_graphics::render::frame::Frame;
use pixelflow_graphics::render::rasterizer::rasterize;
use pixelflow_runtime::platform::ColorCube;

const W: u32 = 64;
const H: u32 = 64;

/// The pack's own conversion: clamp to `[0, 1]`, scale by 255, truncate
/// toward zero (`cvttps`/`FCVTZS`). Matching it exactly keeps the tolerance
/// below about drift in the *shader*, not about rounding conventions.
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
    // both evaluate accurately, so the combinator render is a faithful
    // reference. (Transcendentals like `sin` are validated numerically against
    // the analytic result elsewhere; here the goal is that a baked plane and a
    // rendered frame carry the same picture, and the combinator's own
    // low-degree `sin` is too coarse to serve as a reference.)

    // Reference: channels as combinator trees, rendered by the work-stealing
    // rasterizer, which samples pixel (x, y) at (x + 0.5, y + 0.5). The frame's
    // pixel type must be the platform's, since `ColorCube` is: the cube's byte
    // order and the pixel's accessors are the same platform choice, and mixing
    // them silently reads a different channel than it wrote.
    let combo = ColorCube::default().at(
        kernel!(|| X * 0.015)(),
        kernel!(|| Y * 0.015)(),
        kernel!(|| (X * X + Y * Y).sqrt() * 0.011)(),
        1.0,
    );
    let mut frame = Frame::<PlatformPixel>::new(W, H);
    rasterize(&combo, &mut frame, 1);

    // Arena tier: one bake per channel over the same lattice — the frame's
    // pixel centers.
    let lattice = Lattice {
        extent: [W, H, 1, 1],
        origin: [0.5, 0.5, 0.0, 0.0],
    };
    let red = lattice.bake(&kernel_jit!(|| X * 0.015));
    let green = lattice.bake(&kernel_jit!(|| Y * 0.015));
    let blue = lattice.bake(&kernel_jit!(|| (X * X + Y * Y).sqrt() * 0.011));

    // Compare every pixel. The arena backend lowers sqrt through rsqrt-class
    // estimates, so a few 8-bit quantization levels of drift are expected; a
    // real failure (wrong scene, wrong lattice, wrong wiring) would be off by
    // tens or more.
    let mut max_diff = 0u8;
    let mut worst = (0usize, 0u8, 0u8);
    for (i, px) in frame.data.iter().enumerate() {
        let baked = [
            to_u8(red.buffer()[i]),
            to_u8(green.buffer()[i]),
            to_u8(blue.buffer()[i]),
        ];
        for (rendered, baked) in [(px.r(), baked[0]), (px.g(), baked[1]), (px.b(), baked[2])] {
            let d = rendered.abs_diff(baked);
            if d > max_diff {
                max_diff = d;
                worst = (i, rendered, baked);
            }
        }
    }

    eprintln!(
        "[jit-render] {W}x{H} colored scene: max 8-bit channel diff between the \
         combinator render and the baked arena planes = {max_diff} \
         (worst pixel {} combo={} baked={})",
        worst.0, worst.1, worst.2
    );

    assert!(
        max_diff <= 2,
        "baked arena planes diverged from the combinator render by {max_diff} levels (>2)"
    );

    // Sanity: the scene is non-trivial (not all one color), so the test is real.
    let last = (W * H - 1) as usize;
    assert!(
        red.buffer()[0] != red.buffer()[last] || blue.buffer()[0] != blue.buffer()[last],
        "scene should vary across the frame; got uniform output"
    );
}
