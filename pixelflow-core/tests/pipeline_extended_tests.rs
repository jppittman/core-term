use pixelflow_core::Batch;
use pixelflow_core::dsl::{MaskExt, SurfaceExt};
use pixelflow_core::ops::Max;
use pixelflow_core::pipe::Surface;
use pixelflow_core::Pixel;

// A simple test surface that returns x coordinate as u8
#[derive(Copy, Clone)]
struct XSurface;
impl Surface<u8> for XSurface {
    fn eval(&self, x: Batch<u32>, _y: Batch<u32>) -> Batch<u8> {
        (x & Batch::splat(0xFF)).cast()
    }
}

// Returns a constant value
#[derive(Copy, Clone)]
struct Constant(u32);
impl Surface<u32> for Constant {
    fn eval(&self, _x: Batch<u32>, _y: Batch<u32>) -> Batch<u32> {
        Batch::splat(self.0)
    }
}

// Simple RGBA pixel type for testing
// Byte order: [R, G, B, A] in memory = 0xAABBGGRR as u32 (little endian)
#[derive(Copy, Clone, Default)]
struct TestRgba;

impl Pixel for TestRgba {
    fn from_u32(_v: u32) -> Self { Self }
    fn to_u32(self) -> u32 { 0 }

    #[inline(always)]
    fn batch_red(batch: Batch<u32>) -> Batch<u32> {
        batch & Batch::splat(0xFF)
    }

    #[inline(always)]
    fn batch_green(batch: Batch<u32>) -> Batch<u32> {
        (batch >> 8) & Batch::splat(0xFF)
    }

    #[inline(always)]
    fn batch_blue(batch: Batch<u32>) -> Batch<u32> {
        (batch >> 16) & Batch::splat(0xFF)
    }

    #[inline(always)]
    fn batch_alpha(batch: Batch<u32>) -> Batch<u32> {
        batch >> 24
    }

    #[inline(always)]
    fn batch_from_channels(
        r: Batch<u32>,
        g: Batch<u32>,
        b: Batch<u32>,
        a: Batch<u32>,
    ) -> Batch<u32> {
        r | (g << 8) | (b << 16) | (a << 24)
    }
}

#[test]
fn test_pipeline_max() {
    let s1 = Constant(10);
    let s2 = Constant(20);
    let max_surf = Max(s1, s2);

    let res = max_surf.eval(Batch::splat(0), Batch::splat(0));
    assert_eq!(res.to_array_usize(), [20, 20, 20, 20]);
}

#[test]
fn test_pipeline_offset_skew_over() {
    // Test composition: (XSurface skewed) over (Red) on (Blue)
    // Mask = XSurface (alpha).

    let mask = XSurface; // Alpha ramps up with X
    let red = Constant(0xFF0000FF); // Red (ABGR/RGBA depending on interpretation)
    let blue = Constant(0xFFFF0000); // Blue

    // Test Over with TestRgba pixel format
    let blend = mask.over::<TestRgba, _, _>(red, blue);

    // At x=0, alpha=0, should be blue
    let x0 = Batch::splat(0);
    let y = Batch::splat(0);
    let res0 = blend.eval(x0, y);
    // res0 should be blue
    assert_eq!(res0.to_array_usize(), [0xFFFF0000; 4]);

    // At x=255, alpha=255.
    // Red Channel: FG=FF, BG=00. Blend(FF, 00, FF) -> 254 (0xFE) due to (255*255)/256 approximation.
    // Alpha Channel: FG=FF, BG=FF. Blend(FF, FF, FF) -> 255 (0xFF) exact.
    // Result: 0xFF0000FE.
    let x255 = Batch::splat(255);
    let res255 = blend.eval(x255, y);

    assert_eq!(res255.to_array_usize(), [0xFF0000FE; 4]);

    // Test Skew on mask
    // Skew shears X based on Y.
    // Skew struct: offset = (y * shear) >> 8; eval(x - offset, y)
    // If shear=256 (1.0), offset = y.

    let skewed_mask = mask.skew(256);
    let skewed_blend = skewed_mask.over::<TestRgba, _, _>(red, blue);

    // At x=10, y=10. Offset = 10. Sample mask at x-offset = 0.
    // So alpha should be 0 -> Blue.
    let x10 = Batch::splat(10);
    let y10 = Batch::splat(10);
    let res_skew = skewed_blend.eval(x10, y10);
    assert_eq!(res_skew.to_array_usize(), [0xFFFF0000; 4]);
}
