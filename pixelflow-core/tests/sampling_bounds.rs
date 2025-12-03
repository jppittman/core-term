use pixelflow_core::TensorView;
use pixelflow_core::backend::{Backend, BatchArithmetic, SimdBatch};
use pixelflow_core::batch::{Batch, NativeBackend};

#[test]
fn test_tensor_view_wrapping_corruption() {
    // Create a 2x2 tensor:
    // [ 10, 20 ]
    // [ 30, 40 ]
    let data = [10u32, 20, 30, 40];
    let view = TensorView::new(&data, 2, 2, 2); // width=2, height=2, stride=2

    // Case 1: Sample at (-1, 0) via wrapping.
    // x = -1 cast to u32 = u32::MAX
    let x_neg = Batch::<u32>::splat(u32::MAX);
    let y_zero = Batch::<u32>::splat(0);

    // Current behavior prediction:
    // idx = 0 * 2 + u32::MAX = u32::MAX
    // clamped = min(u32::MAX, 3) = 3
    // gather(3) = 40 (bottom-right pixel)
    //
    // Desired behavior: 0 (out of bounds)

    let result_neg = unsafe { view.gather_2d::<NativeBackend>(x_neg, y_zero) };
    let val_neg = result_neg.first();

    println!("Sampled at (-1, 0) [wrapped]: {}", val_neg);

    // Case 2: Sample at (2, 0) - just to the right of first row
    // x = 2. width = 2.
    // idx = 0 * 2 + 2 = 2.
    // 2 is a valid index in the slice (it's the first pixel of the second row, value 30).
    // But geometrically (2, 0) is outside the 2x2 image (0..2, 0..2).
    // This is "wrapping" behavior (row overflow).
    //
    // Desired behavior: 0 (strict 2D bounds)

    let x_overflow = Batch::<u32>::splat(2);
    let result_overflow = unsafe { view.gather_2d::<NativeBackend>(x_overflow, y_zero) };
    let val_overflow = result_overflow.first();

    println!("Sampled at (2, 0) [row overflow]: {}", val_overflow);

    // Assertions to fail if bug is present
    // We expect these to be 0 for correct "Border" sampling
    assert_eq!(
        val_neg, 0,
        "Negative x should return 0 (border), got corruption/clamping"
    );
    assert_eq!(
        val_overflow, 0,
        "Row overflow x should return 0 (border), got next row"
    );
}
