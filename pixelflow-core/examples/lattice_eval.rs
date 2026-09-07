//! The three steps, end to end: bake a kernel over a lattice, then read the
//! resulting buffer back by coordinate — `index(collapse(f)) = f`.
//!
//! `cargo run -p pixelflow-core --example lattice_eval`

use pixelflow_core::lattice::{DiscreteManifold, Lattice};
use pixelflow_core::{Kernel, Manifold};

/// Read a buffer at one coordinate: its nearest-neighbour gather compiled at
/// a one-sample lattice, with the buffer bound, then collapsed.
fn read(dm: &DiscreteManifold, x: f32, y: f32) -> f32 {
    let bound = Manifold::compile(&dm.kernel(), [1, 1]).bind(&[dm.binding()]);
    bound.eval_at(x, y)
}

fn main() {
    // A 2x2 buffer, read back at its four integer coordinates.
    let dm = DiscreteManifold::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2);

    println!("DiscreteManifold read back by coordinate:");
    println!("{:?} {:?}", read(&dm, 0.0, 0.0), read(&dm, 1.0, 0.0));
    println!("{:?} {:?}", read(&dm, 0.0, 1.0), read(&dm, 1.0, 1.0));

    // Bake a kernel over a 4x4 frame.
    let lattice = Lattice::frame(4, 4);
    let baked = lattice.bake(&Kernel::x().add(&Kernel::y().mul(&Kernel::constant(10.0))));
    println!("\nBaked 4x4 `X + 10*Y`: {:?}", baked.buffer());
}
