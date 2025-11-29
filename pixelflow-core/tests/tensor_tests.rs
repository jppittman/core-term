use pixelflow_core::{Batch, MapPixels, TensorView, TensorViewMut};

#[test]
fn test_tensor_view_gather() {
    let data: [u32; 8] = [10, 20, 30, 40, 50, 60, 70, 80];
    // 4x2 image, stride 4
    let view = TensorView::new(&data, 4, 2, 4);

    let x = Batch::<u32>::new(0, 1, 2, 3);
    let y = Batch::<u32>::splat(0);

    let gathered = unsafe { view.gather_2d(x, y) };
    assert_eq!(gathered.to_array_usize(), [10, 20, 30, 40]);

    let y_row1 = Batch::<u32>::splat(1);
    let gathered_row1 = unsafe { view.gather_2d(x, y_row1) };
    assert_eq!(gathered_row1.to_array_usize(), [50, 60, 70, 80]);
}

#[test]
fn test_tensor_view_mut_map_pixels() {
    let mut data = [0u32; 16];
    // 4x4 image
    let mut view = TensorViewMut::new(&mut data, 4, 4, 4);

    // Map pixels to set them to coordinate index: y * width + x
    view.map_pixels(|x, y| (y * Batch::splat(4)) + x);

    for i in 0..16 {
        assert_eq!(data[i], i as u32);
    }
}
