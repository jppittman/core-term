
use pixelflow_core::{TensorView, Batch};
use pixelflow_core::ops::SampleAtlas;
use pixelflow_core::pipe::Surface;

#[test]
fn test_sample_atlas_integer_coords_bug() {
    // Create a simple 2x2 atlas
    // 10, 20
    // 30, 40
    // Packed 4-bit:
    // Row 0: 0x12 (1=0x1, 2=0x2) -> 17, 34
    // Row 1: 0x34 (3=0x3, 4=0x4) -> 51, 68
    // Note: gather_4bit uses high nibble for even x, low nibble for odd x.
    // x=0 -> high nibble (1) -> 17
    // x=1 -> low nibble (2) -> 34

    let data = [0x12, 0x34];
    // width=2, height=2, stride=1 (byte)
    let view = TensorView::new(&data, 2, 2, 1);
    let sampler = SampleAtlas { atlas: view };

    // Sample at (1, 0)
    // We expect value from x=1, y=0 -> 34 (0x22)
    let x = Batch::splat(1);
    let y = Batch::splat(0);

    let result = sampler.eval(x, y);

    // Cast to u32 to extract. SampleAtlas returns Batch<u8> which is a bitcast of Batch<u32>
    // containing 0x000000VV. So casting back to u32 recovers the value.
    let result_u32: Batch<u32> = result.cast();
    let val = result_u32.extract(0);

    // If bug exists (interpreting 1 as fixed point 1/65536):
    // integer part = 0. fractional part = 0.
    // It samples (0, 0) -> 17.

    // If fixed (interpreting 1 as pixel 1):
    // It samples (1, 0) -> 34.

    println!("Sampled value: {}", val);

    assert_eq!(val, 34, "Expected 34 (pixel at 1,0), got {}. Likely sampled (0,0) due to coordinate interpretation bug.", val);
}
