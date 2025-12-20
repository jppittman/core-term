use pixelflow_core::{
    field::Field,
    traits::{Manifold, Surface, Volume},
};

struct AddAllManifold;

// Implement for Field -> Field (SIMD)
impl Manifold<Field, Field> for AddAllManifold {
    fn eval(&self, x: Field, y: Field, z: Field, w: Field) -> Field {
        x + y + z + w
    }
}

#[test]
fn test_manifold_collapse_to_volume() {
    let m = AddAllManifold;
    let x = Field::splat(10.0);
    let y = Field::splat(20.0);
    let z = Field::splat(30.0);

    // Explicitly using the Volume trait method via blanket impl: Volume eval calls Manifold eval with w=0
    let res = <AddAllManifold as Volume<Field, Field>>::eval(&m, x, y, z);

    assert_eq!(res, 60.0); // 10 + 20 + 30 + 0
}

#[test]
fn test_volume_collapse_to_surface() {
    let m = AddAllManifold;
    let x = Field::splat(10.0);
    let y = Field::splat(20.0);

    // Explicitly using the Surface trait method via blanket impl: Surface eval calls Volume eval with z=0 (which calls Manifold with w=0)
    let res = <AddAllManifold as Surface<Field, Field>>::eval(&m, x, y);

    assert_eq!(res, 30.0); // 10 + 20 + 0 + 0
}
