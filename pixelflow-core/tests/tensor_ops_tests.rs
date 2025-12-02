use pixelflow_core::{Batch, TensorView, TensorViewMut, SimdOps, SimdBatch};
use pixelflow_core::batch::NativeBackend;

trait BatchTestExt {
    fn to_array_usize(&self) -> [usize; 4];
}

impl BatchTestExt for Batch<u32> {
    fn to_array_usize(&self) -> [usize; 4] {
        let mut arr = [0u32; 4];
        self.store(&mut arr);
        [arr[0] as usize, arr[1] as usize, arr[2] as usize, arr[3] as usize]
    }
}

#[test]
fn test_tensor_view_gather_clamping() {
    let data: [u32; 4] = [10, 20, 30, 40];
    // 2x2 image, stride 2
    let view = TensorView::new(&data, 2, 2, 2);

    // Coordinates way out of bounds
    let x = Batch::<u32>::new(100, 200, 0, 1);
    let y = Batch::<u32>::new(100, 200, 0, 0);

    // Expected behavior: index = y * stride + x
    // Clamped to data.len() - 1 (index 3, value 40)

    // Lane 0: (100, 100) -> huge index -> clamp to 3 -> 40
    // Lane 1: (200, 200) -> huge index -> clamp to 3 -> 40
    // Lane 2: (0, 0) -> index 0 -> 10
    // Lane 3: (1, 0) -> index 1 -> 20

    let gathered: Batch<u32> = unsafe { view.gather_2d::<NativeBackend>(x, y) };
    let res = gathered.to_array_usize();

    assert_eq!(res, [40, 40, 10, 20]);
}
