use pixelflow_core::backend::{BatchArithmetic, SimdBatch};
use pixelflow_core::backends::scalar::Scalar;
use pixelflow_core::pipe::Surface;
use pixelflow_core::platform::{PixelFormat, Platform};

#[test]
fn test_platform_scalar() {
    let platform = Platform::<Scalar>::new()
        .with_scale(1)
        .with_format(PixelFormat::Rgba);

    // Simple constant surface (u32 implements Surface<u32>)
    let color: u32 = 0xFF0000FF; // Red

    let buffer = platform.materialize(&color, 10, 10);

    assert_eq!(buffer.width, 10);
    assert_eq!(buffer.height, 10);
    // Check first pixel
    assert_eq!(buffer.data[0], 0xFF0000FF);
}

#[test]
fn test_platform_scalar_checkboard() {
    // Custom struct surface
    struct Checker;
    impl Surface<u32> for Checker {
        fn eval<B: pixelflow_core::backend::Backend>(
            &self,
            x: B::Batch<u32>,
            y: B::Batch<u32>,
        ) -> B::Batch<u32>
        where
            B::Batch<u32>: BatchArithmetic<u32>,
        {
            // (x ^ y) & 1
            let one = <B::Batch<u32> as SimdBatch<u32>>::splat(1);
            let x_bit = x & one;
            let y_bit = y & one;
            let check = x_bit ^ y_bit;

            // If 1 then White, else Black
            let white = <B::Batch<u32> as SimdBatch<u32>>::splat(0xFFFFFFFF);
            let black = <B::Batch<u32> as SimdBatch<u32>>::splat(0x000000FF);

            // BatchOps::select takes (mask, if_true, if_false).
            let mask = check.cmp_eq(one);
            mask.select(white, black)
        }
    }

    let platform = Platform::<Scalar>::new();
    let buffer = platform.materialize(&Checker, 4, 4);

    // (0,0) -> 0^0=0 -> Black
    assert_eq!(buffer.data[0], 0x000000FF);
    // (1,0) -> 1^0=1 -> White
    assert_eq!(buffer.data[1], 0xFFFFFFFF);
}
