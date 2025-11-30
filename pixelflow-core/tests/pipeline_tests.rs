use pixelflow_core::dsl::SurfaceExt;
use pixelflow_core::pipe::Surface;
use pixelflow_core::{Batch, TensorViewMut, execute_into_tensor};

#[test]
fn test_constant_surface() {
    let surface = Batch::<u32>::splat(42);
    let x = Batch::<u32>::splat(0);
    let y = Batch::<u32>::splat(0);

    let res = surface.eval(x, y);
    assert_eq!(res.to_array_usize(), [42, 42, 42, 42]);
}

#[test]
fn test_offset_surface() {
    // A surface that returns x coordinate
    let identity_x = |x: Batch<u32>, _y: Batch<u32>| x;

    let offset_surface = identity_x.offset(10, 0);

    let x = Batch::<u32>::splat(0);
    let y = Batch::<u32>::splat(0);

    let res = offset_surface.eval(x, y);
    // Should eval at x+10, so returns 10
    assert_eq!(res.to_array_usize(), [10, 10, 10, 10]);
}

#[test]
fn test_execute_pipeline() {
    let mut data = [0u32; 16];
    let mut target = TensorViewMut::new(&mut data, 4, 4, 4);

    // Simple fill pipeline
    let fill = Batch::<u32>::splat(0xFF);

    execute_into_tensor(fill, &mut target);

    for px in data.iter() {
        assert_eq!(*px, 0xFF);
    }
}
