use pixelflow_core::lattice::{DiscreteManifold, Lattice};
use pixelflow_core::{Field, Kernel, Manifold};

fn main() {
    // 2x2 grid built from a DiscreteManifold
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let dm = DiscreteManifold::new(data, 2, 2);

    // Test evaluating the DiscreteManifold as a Manifold
    let p00 = dm.eval((
        Field::from(0.0),
        Field::from(0.0),
        Field::from(0.0),
        Field::from(0.0),
    ));
    let p10 = dm.eval((
        Field::from(1.0),
        Field::from(0.0),
        Field::from(0.0),
        Field::from(0.0),
    ));
    let p01 = dm.eval((
        Field::from(0.0),
        Field::from(1.0),
        Field::from(0.0),
        Field::from(0.0),
    ));
    let p11 = dm.eval((
        Field::from(1.0),
        Field::from(1.0),
        Field::from(0.0),
        Field::from(0.0),
    ));

    println!("DiscreteManifold evaluated at integer coordinates:");
    println!("{:?} {:?}", p00, p10);
    println!("{:?} {:?}", p01, p11);

    // Bake a constant kernel over a 4x4 frame
    let lattice = Lattice::frame(4, 4, 0.0);
    let baked = lattice.bake(&Kernel::constant(1.5));
    println!("\nBaked 4x4 constant kernel (1.5): {:?}", baked.buffer());
}
