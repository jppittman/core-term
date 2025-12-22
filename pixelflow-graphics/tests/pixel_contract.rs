use pixelflow_core::{Batch, SimdBatch};
use pixelflow_graphics::render::{Bgra, Pixel, Rgba};

#[test]
fn test_rgba_batch_consistency() {
    let r = Batch::<u32>::splat(0xAA);
    let g = Batch::<u32>::splat(0xBB);
    let b = Batch::<u32>::splat(0xCC);
    let a = Batch::<u32>::splat(0xDD);

    let packed = Rgba::batch_from_channels(r, g, b, a);

    let expected_u32 = 0xDDCCBBAA;
    assert_eq!(packed.first(), expected_u32);

    assert_eq!(Rgba::batch_red(packed).first(), 0xAA);
    assert_eq!(Rgba::batch_green(packed).first(), 0xBB);
    assert_eq!(Rgba::batch_blue(packed).first(), 0xCC);
    assert_eq!(Rgba::batch_alpha(packed).first(), 0xDD);
}

#[test]
fn test_bgra_batch_consistency() {
    let r = Batch::<u32>::splat(0xAA);
    let g = Batch::<u32>::splat(0xBB);
    let b = Batch::<u32>::splat(0xCC);
    let a = Batch::<u32>::splat(0xDD);

    let packed = Bgra::batch_from_channels(r, g, b, a);

    // Bgra byte order in memory: B(CC), G(BB), R(AA), A(DD)
    // Little-endian value: 0xDD AABB CC
    let expected_u32 = 0xDDAABBCC;
    assert_eq!(packed.first(), expected_u32);

    assert_eq!(Bgra::batch_red(packed).first(), 0xAA);
    assert_eq!(Bgra::batch_green(packed).first(), 0xBB);
    assert_eq!(Bgra::batch_blue(packed).first(), 0xCC);
    assert_eq!(Bgra::batch_alpha(packed).first(), 0xDD);
}

#[test]
fn test_swizzle_correctness() {
    let rgba = Rgba::new(0x11, 0x22, 0x33, 0x44);
    let bgra = Bgra::from(rgba);

    assert_eq!(bgra.r(), 0x11);
    assert_eq!(bgra.g(), 0x22);
    assert_eq!(bgra.b(), 0x33);
    assert_eq!(bgra.a(), 0x44);

    let back_to_rgba = Rgba::from(bgra);
    assert_eq!(rgba, back_to_rgba);
}
