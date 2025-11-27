use pixelflow_core::Batch;

#[test]
fn test_batch_u32_arithmetic() {
    let a = Batch::<u32>::splat(10);
    let b = Batch::<u32>::splat(20);

    let sum = a + b;
    assert_eq!(sum.to_array_usize(), [30, 30, 30, 30]);

    let diff = b - a;
    assert_eq!(diff.to_array_usize(), [10, 10, 10, 10]);

    let prod = a * b;
    assert_eq!(prod.to_array_usize(), [200, 200, 200, 200]);
}

#[test]
fn test_batch_u32_bitwise() {
    let a = Batch::<u32>::splat(0x0F0F0F0F);
    let b = Batch::<u32>::splat(0xF0F0F0F0);

    let and_res = a & b;
    assert_eq!(and_res.to_array_usize(), [0; 4]);

    let or_res = a | b;
    assert_eq!(or_res.to_array_usize(), [0xFFFFFFFF; 4]);

    let not_res = !a;
    assert_eq!(not_res.to_array_usize(), [0xF0F0F0F0; 4]);
}

#[test]
fn test_batch_load_store() {
    let data = [1, 2, 3, 4, 5, 6, 7, 8];

    // Load first 4 elements
    let batch = Batch::<u32>::load_from_slice(&data[0..4]);
    assert_eq!(batch.to_array_usize(), [1, 2, 3, 4]);

    // Store back
    let mut dest = [0u32; 4];
    batch.store_into_slice(&mut dest);
    assert_eq!(dest, [1, 2, 3, 4]);
}

#[test]
fn test_batch_casting() {
    let a = Batch::<u32>::splat(0x12345678);
    // When cast to u16, we expect to see the lower and upper 16 bits
    // This depends on endianness, but assuming LE for x86/ARM
    let b: Batch<u16> = a.cast();

    // We can't easily extract from Batch<u16> without more helper methods exposed or unsafe code,
    // but we can cast back or check if it compiles and runs.
    // Let's cast back to u32 and see if we get the same thing (round trip).
    let c: Batch<u32> = b.cast();
    assert_eq!(c.to_array_usize(), [0x12345678; 4]);
}

#[test]
fn test_batch_min_max() {
    let a = Batch::<u32>::new(1, 10, 5, 20);
    let b = Batch::<u32>::new(10, 1, 20, 5);

    let min = a.min(b);
    assert_eq!(min.to_array_usize(), [1, 1, 5, 5]);

    let max = a.max(b);
    assert_eq!(max.to_array_usize(), [10, 10, 20, 20]);
}

#[test]
fn test_batch_select() {
    let a = Batch::<u32>::splat(1);
    let b = Batch::<u32>::splat(2);
    let mask = Batch::<u32>::new(0xFFFFFFFF, 0, 0xFFFFFFFF, 0);

    let res = a.select(b, mask);
    assert_eq!(res.to_array_usize(), [1, 2, 1, 2]);
}
