use pixelflow_core::{Batch, SimdOps, SimdBatch};

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
fn test_batch_u32_saturating_add() {
    let a = Batch::<u32>::new(100, u32::MAX, u32::MAX - 10, 0);
    let b = Batch::<u32>::new(200, 1, 11, 0);

    let res = a.saturating_add(b);
    let out = res.to_array_usize();

    assert_eq!(out[0], 300);
    assert_eq!(out[1], u32::MAX as usize);
    assert_eq!(out[2], u32::MAX as usize);
    assert_eq!(out[3], 0);
}

#[test]
fn test_batch_u32_saturating_sub() {
    let a = Batch::<u32>::new(100, 0, 10, u32::MAX);
    let b = Batch::<u32>::new(50, 1, 20, 0);

    let res = a.saturating_sub(b);
    let out = res.to_array_usize();

    assert_eq!(out[0], 50);
    assert_eq!(out[1], 0); // 0 - 1 saturated
    assert_eq!(out[2], 0); // 10 - 20 saturated
    assert_eq!(out[3], u32::MAX as usize);
}

#[test]
fn test_batch_u32_min_max() {
    let a = Batch::<u32>::new(10, 20, 30, 40);
    let b = Batch::<u32>::new(40, 30, 20, 10);

    let min_res = a.min(b);
    assert_eq!(min_res.to_array_usize(), [10, 20, 20, 10]);

    let max_res = a.max(b);
    assert_eq!(max_res.to_array_usize(), [40, 30, 30, 40]);
}

#[test]
fn test_batch_u32_shifts() {
    let a = Batch::<u32>::splat(0xF0); // 240

    let shl = a << 4;
    assert_eq!(shl.to_array_usize(), [0xF00, 0xF00, 0xF00, 0xF00]);

    let shr = a >> 4;
    assert_eq!(shr.to_array_usize(), [0xF, 0xF, 0xF, 0xF]);
}

#[test]
fn test_batch_u32_shifts_large() {
    let a = Batch::<u32>::splat(1);

    // Shift by >= 32 behavior is platform dependent in Rust scalar,
    // but SIMD usually masks the shift count.
    // However, let's test within reasonable bounds that might be edge cases.
    let shl = a << 31;
    assert_eq!(
        shl.to_array_usize(),
        [0x80000000, 0x80000000, 0x80000000, 0x80000000]
    );

    let shr = shl >> 31;
    assert_eq!(shr.to_array_usize(), [1, 1, 1, 1]);
}

#[test]
fn test_batch_select_masking() {
    let a = Batch::<u32>::splat(10);
    let b = Batch::<u32>::splat(20);

    // Mask: All ones (true) or all zeros (false)
    let mask_true = Batch::<u32>::splat(u32::MAX);
    let mask_false = Batch::<u32>::splat(0);
    let mask_mixed = Batch::<u32>::new(u32::MAX, 0, u32::MAX, 0);

    // Usage: mask.select(if_true, if_false)
    assert_eq!(mask_true.select(a, b).to_array_usize(), [10, 10, 10, 10]);
    assert_eq!(mask_false.select(a, b).to_array_usize(), [20, 20, 20, 20]);
    assert_eq!(mask_mixed.select(a, b).to_array_usize(), [10, 20, 10, 20]);
}

#[test]
fn test_batch_bitwise_logic() {
    let a = Batch::<u32>::splat(0xAAAA_AAAA);
    let b = Batch::<u32>::splat(0x5555_5555);

    let and = a & b;
    assert_eq!(and.to_array_usize(), [0, 0, 0, 0]);

    let or = a | b;
    assert_eq!(or.to_array_usize(), [u32::MAX as usize; 4]);

    let not_a = !a;
    assert_eq!(not_a.to_array_usize(), [0x5555_5555; 4]);
}
