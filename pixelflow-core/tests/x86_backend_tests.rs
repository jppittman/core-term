#[cfg(target_arch = "x86_64")]
#[cfg(test)]
mod tests {
    extern crate std;
    use pixelflow_core::backend::SimdOps;
    use pixelflow_core::backend::x86::F32x4;
    use std::prelude::v1::*;

    #[test]
    fn test_sse2_arithmetic() {
        let a = F32x4::splat(2.0);
        let b = F32x4::splat(3.0);

        let sum = a + b;
        let mut out = [0.0; 4];
        sum.store(&mut out);
        assert_eq!(out, [5.0, 5.0, 5.0, 5.0]);

        let diff = b - a;
        diff.store(&mut out);
        assert_eq!(out, [1.0, 1.0, 1.0, 1.0]);

        let prod = a * b;
        prod.store(&mut out);
        assert_eq!(out, [6.0, 6.0, 6.0, 6.0]);

        let quot = b / a;
        quot.store(&mut out);
        assert_eq!(out, [1.5, 1.5, 1.5, 1.5]);
    }

    #[test]
    fn test_sse2_sequential() {
        let seq = F32x4::sequential(10.0);
        let mut out = [0.0; 4];
        seq.store(&mut out);
        assert_eq!(out, [10.0, 11.0, 12.0, 13.0]);
    }

    #[test]
    fn test_sse2_logic() {
        let a = F32x4::splat(1.0);
        let b = F32x4::splat(2.0);

        let lt = a.cmp_lt(b);
        assert!(lt.all());

        // Check select
        let t = F32x4::splat(10.0);
        let f = F32x4::splat(20.0);
        let sel = F32x4::select(lt, t, f);
        let mut out = [0.0; 4];
        sel.store(&mut out);
        assert_eq!(out, [10.0, 10.0, 10.0, 10.0]);

        let gt = a.cmp_gt(b);
        assert!(!gt.any());
        let sel2 = F32x4::select(gt, t, f);
        sel2.store(&mut out);
        assert_eq!(out, [20.0, 20.0, 20.0, 20.0]);
    }

    #[test]
    fn test_sse2_bitwise() {
        let a = F32x4::splat(1.0); // 1.0 is 0x3f800000
        let b = F32x4::splat(2.0); // 2.0 is 0x40000000
        let c = a & b;
        let mut out = [0.0; 4];
        c.store(&mut out);
        assert_eq!(out, [0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_sse2_math() {
        let a = F32x4::splat(4.0);
        let sqrt = a.sqrt();
        let mut out = [0.0; 4];
        sqrt.store(&mut out);
        assert_eq!(out, [2.0, 2.0, 2.0, 2.0]);

        let b = F32x4::splat(-2.0);
        let abs = b.abs();
        abs.store(&mut out);
        assert_eq!(out, [2.0, 2.0, 2.0, 2.0]);

        let min = a.min(b);
        min.store(&mut out);
        assert_eq!(out, [-2.0, -2.0, -2.0, -2.0]);
    }

    #[test]
    fn test_sse2_any_all() {
        let zero = F32x4::splat(0.0);
        assert!(!zero.any());
        assert!(!zero.all());

        let one = F32x4::splat(1.0);
        assert!(one.any());
        assert!(one.all());

        let mixed = F32x4::sequential(0.0); // 0, 1, 2, 3
        assert!(mixed.any());
        assert!(!mixed.all());
    }

    #[test]
    #[should_panic]
    fn test_sse2_store_panic() {
        let a = F32x4::default();
        let mut out = [0.0; 3]; // Too small
        a.store(&mut out);
    }
}
