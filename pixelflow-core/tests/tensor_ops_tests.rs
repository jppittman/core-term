use pixelflow_core::{Batch, MapPixels, TensorView, TensorViewMut};

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

    let gathered = unsafe { view.gather_2d(x, y) };
    let res = gathered.to_array_usize();

    assert_eq!(res, [40, 40, 10, 20]);
}

#[test]
fn test_tensor_view_mut_map_pixels_odd_width() {
    // Width 5, Height 2. Stride 5. Total 10 pixels.
    let mut data = [0u32; 10];
    let mut view = TensorViewMut::new(&mut data, 5, 2, 5);

    // Map pixels to coordinate sum x + y
    view.map_pixels(|x, y| x + y);

    // Row 0: 0, 1, 2, 3, 4
    // Row 1: 1, 2, 3, 4, 5
    let expected = [0, 1, 2, 3, 4, 1, 2, 3, 4, 5];

    assert_eq!(data, expected);
}

#[test]
fn test_tensor_view_mut_map_pixels_narrow() {
    // Width 1 (narrower than SIMD lane width 4)
    let mut data = [0u32; 2];
    let mut view = TensorViewMut::new(&mut data, 1, 2, 1);

    view.map_pixels(|x, _y| x);

    assert_eq!(data, [0, 0]);
}

#[test]
fn test_tensor_view_sub_view() {
    // Explicitly u32
    let mut data: [u32; 12] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]; // 4x3

    let mut view = TensorViewMut::new(&mut data, 4, 3, 4);

    // Subview at (1, 1) size 2x2
    // Corresponds to indices:
    // (1,1)->5, (2,1)->6
    // (1,2)->9, (2,2)->10

    let mut sub = unsafe { view.sub_view(1, 1, 2, 2) };

    assert_eq!(sub.width, 2);
    assert_eq!(sub.height, 2);
    assert_eq!(sub.stride, 4); // Stride remains parent stride

    // Modify subview
    sub.map_pixels(|_x, _y| Batch::<u32>::splat(99));

    // Check original data
    // 5, 6, 9, 10 should be 99
    let expected = [0, 1, 2, 3, 4, 99, 99, 7, 8, 99, 99, 11];

    assert_eq!(data, expected);
}
