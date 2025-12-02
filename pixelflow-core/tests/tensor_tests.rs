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
fn test_tensor_view_gather() {
    let data: [u32; 8] = [10, 20, 30, 40, 50, 60, 70, 80];
    // 4x2 image, stride 4
    let view = TensorView::new(&data, 4, 2, 4);

    let x = Batch::<u32>::new(0, 1, 2, 3);
    let y = Batch::<u32>::splat(0);

    let gathered: Batch<u32> = unsafe { view.gather_2d::<NativeBackend>(x, y) };
    assert_eq!(gathered.to_array_usize(), [10, 20, 30, 40]);

    let y_row1 = Batch::<u32>::splat(1);
    let gathered_row1: Batch<u32> = unsafe { view.gather_2d::<NativeBackend>(x, y_row1) };
    assert_eq!(gathered_row1.to_array_usize(), [50, 60, 70, 80]);
}
