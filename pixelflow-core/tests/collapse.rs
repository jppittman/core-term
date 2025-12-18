use pixelflow_core::{
    Batch, SimdBatch,
    traits::{Manifold, Surface, Volume},
};

struct AddAllManifold;

impl Manifold<f32, f32> for AddAllManifold {
    fn eval(&self, x: Batch<f32>, y: Batch<f32>, z: Batch<f32>, w: Batch<f32>) -> Batch<f32> {
        x + y + z + w
    }
}

#[test]
fn test_manifold_collapse_to_volume() {
    let m = AddAllManifold;
    let x = Batch::<f32>::splat(10.0);
    let y = Batch::<f32>::splat(20.0);
    let z = Batch::<f32>::splat(30.0);

    // Explicitly using the Volume trait method via blanket impl: Volume eval calls Manifold eval with w=0
    let res = <AddAllManifold as Volume<f32, f32>>::eval(&m, x, y, z);

    assert_eq!(res.first(), 60.0); // 10 + 20 + 30 + 0
}

#[test]
fn test_volume_collapse_to_surface() {
    let m = AddAllManifold;
    let x = Batch::<f32>::splat(10.0);
    let y = Batch::<f32>::splat(20.0);

    // Explicitly using the Surface trait method via blanket impl: Surface eval calls Volume eval with z=0 (which calls Manifold with w=0)
    let res = <AddAllManifold as Surface<f32, f32>>::eval(&m, x, y);

    assert_eq!(res.first(), 30.0); // 10 + 20 + 0 + 0
}
