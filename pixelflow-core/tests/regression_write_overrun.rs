use pixelflow_core::batch::{Batch, NativeBackend};
use pixelflow_core::pipe::Surface;
use pixelflow_core::backend::{Backend, SimdBatch};

#[test]
fn test_u8_execution_overrun() {
    // Setup a buffer with guard zones
    let mut buffer = vec![0xFFu8; 32];
    let target_start = 8;
    let width = 4;
    let height = 1;
    let target_end = target_start + width * height; // 12

    // Create a dummy surface that returns 0xAA
    struct CountSurface;
    impl Surface<u8> for CountSurface {
        fn eval(&self, _x: Batch<u32>, _y: Batch<u32>) -> Batch<u8> {
             // Return constant splat 0xAA
             // We use u32::splat(0xAA) then downcast, because Batch<u8>::splat might be what we are testing too?
             // But simpler:
             <NativeBackend as Backend>::Batch::<u8>::splat(0xAA)
        }
    }

    // Execute into the middle of the buffer (slice length 4)
    let slice = &mut buffer[target_start..target_end];
    assert_eq!(slice.len(), 4);

    // We expect this to execute one batch of 4 pixels.
    // If the implementation writes 16 bytes (128 bits), it will overwrite the guard zone.
    pixelflow_core::execute(&CountSurface, slice, width, height);

    // Check target area
    assert_eq!(&buffer[target_start..target_end], &[0xAA, 0xAA, 0xAA, 0xAA], "Target area verification");

    // Check guards
    // Before
    assert_eq!(&buffer[0..target_start], &[0xFF; 8], "Pre-guard verification");

    // After - THIS WILL FAIL if 16 bytes are written
    let post_guard = &buffer[target_end..24]; // Next 12 bytes
    assert_eq!(post_guard, &[0xFF; 12], "Post-guard verification (Overrun detected!)");
}
