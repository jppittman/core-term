#[cfg(target_arch = "x86_64")]
#[cfg(test)]
mod tests {
    extern crate std;
    use pixelflow_core::backend::x86::F32x4;
    use pixelflow_core::backend::{MaskOps, SimdOps};
    use std::prelude::v1::*;

    #[test]
    fn sse2_arithmetic() {
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
    fn sse2_sequential() {
        let seq = F32x4::sequential(10.0);
        let mut out = [0.0; 4];
        seq.store(&mut out);
        assert_eq!(out, [10.0, 11.0, 12.0, 13.0]);
    }

    #[test]
    fn sse2_logic() {
        let a = F32x4::splat(1.0);
        let b = F32x4::splat(2.0);

        // cmp_lt returns native Mask4
        let lt = a.cmp_lt(b);
        assert!(lt.all());

        // Check select with native mask
        let t = F32x4::splat(10.0);
        let f = F32x4::splat(20.0);
        let sel = F32x4::simd_select(lt, t, f);
        let mut out = [0.0; 4];
        sel.store(&mut out);
        assert_eq!(out, [10.0, 10.0, 10.0, 10.0]);

        let gt = a.cmp_gt(b);
        assert!(!gt.any());
        let sel2 = F32x4::simd_select(gt, t, f);
        sel2.store(&mut out);
        assert_eq!(out, [20.0, 20.0, 20.0, 20.0]);
    }

    #[test]
    fn sse2_bitwise() {
        let a = F32x4::splat(1.0); // 1.0 is 0x3f800000
        let b = F32x4::splat(2.0); // 2.0 is 0x40000000
        let c = a & b;
        let mut out = [0.0; 4];
        c.store(&mut out);
        assert_eq!(out, [0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn sse2_math() {
        let a = F32x4::splat(4.0);
        let sqrt = a.simd_sqrt();
        let mut out = [0.0; 4];
        sqrt.store(&mut out);
        assert_eq!(out, [2.0, 2.0, 2.0, 2.0]);

        let b = F32x4::splat(-2.0);
        let abs = b.simd_abs();
        abs.store(&mut out);
        assert_eq!(out, [2.0, 2.0, 2.0, 2.0]);

        let min = a.simd_min(b);
        min.store(&mut out);
        assert_eq!(out, [-2.0, -2.0, -2.0, -2.0]);
    }

    #[test]
    fn sse2_mask_any_all() {
        // Test MaskOps methods directly on masks
        let zero = F32x4::splat(0.0);
        let zero_mask = zero.float_to_mask();
        assert!(!zero_mask.any());
        assert!(!zero_mask.all());

        let all_true = F32x4::splat(1.0).cmp_gt(F32x4::splat(0.0));
        assert!(all_true.any());
        assert!(all_true.all());

        // Mixed: first lane is false, rest are true (0 > 0 is false, 1 > 0, 2 > 0, 3 > 0 are true)
        let mixed = F32x4::sequential(0.0).cmp_gt(F32x4::splat(0.0));
        assert!(mixed.any());
        assert!(!mixed.all());
    }

    #[test]
    #[should_panic]
    fn sse2_store_panic() {
        let a = F32x4::default();
        let mut out = [0.0; 3]; // Too small
        a.store(&mut out);
    }

    // SimdOps's *provided* methods (exp, ln, log10, pow, hypot, mul_rsqrt,
    // ceil, round, fract, clamp) are default trait-method bodies defined once
    // in `backend::mod` and inherited by every backend (SSE2, AVX2, AVX-512,
    // NEON) rather than reimplemented per-ISA — none of the concrete
    // `SimdOps` impls override them. Exercising them through `F32x4` here
    // covers the shared formula all backends actually run.

    #[test]
    fn exp_matches_the_natural_exponential() {
        let mut out = [0.0; 4];
        F32x4::splat(1.0).exp().store(&mut out);
        for x in out {
            assert!((x - std::f32::consts::E).abs() < 1e-3, "exp(1) got {x}");
        }
    }

    #[test]
    fn ln_matches_the_natural_logarithm() {
        let mut out = [0.0; 4];
        F32x4::splat(std::f32::consts::E).ln().store(&mut out);
        for x in out {
            assert!((x - 1.0).abs() < 1e-3, "ln(e) got {x}");
        }
    }

    #[test]
    fn log10_matches_the_base_ten_logarithm() {
        let mut out = [0.0; 4];
        F32x4::splat(100.0).log10().store(&mut out);
        for x in out {
            assert!((x - 2.0).abs() < 1e-3, "log10(100) got {x}");
        }
    }

    #[test]
    fn pow_raises_a_positive_base_to_a_real_exponent() {
        // 4^3 = 64. Chosen so `*`->`+`/`/` mutants in `exp * self.log2()`
        // land on 16/32 respectively instead of coincidentally hitting 64.
        let mut out = [0.0; 4];
        F32x4::splat(4.0).pow(F32x4::splat(3.0)).store(&mut out);
        for x in out {
            assert!((x - 64.0).abs() < 5e-2, "4^3 got {x}");
        }
    }

    #[test]
    fn hypot_computes_the_euclidean_norm_of_two_lanes() {
        // The 3-4-5 triangle: every `+`/`*` mutation in `self*self + y*y`
        // lands on a different wrong value (17, 22, 144, ...), none of them 5.
        let mut out = [0.0; 4];
        F32x4::splat(3.0).hypot(F32x4::splat(4.0)).store(&mut out);
        for x in out {
            assert!((x - 5.0).abs() < 1e-4, "hypot(3, 4) got {x}");
        }
    }

    #[test]
    fn mul_rsqrt_divides_by_the_square_root_of_the_other_operand() {
        // 6 * rsqrt(4) ~= 6 / 2 = 3.
        let mut out = [0.0; 4];
        F32x4::splat(6.0)
            .mul_rsqrt(F32x4::splat(4.0))
            .store(&mut out);
        for x in out {
            assert!((x - 3.0).abs() < 1e-2, "6 * rsqrt(4) got {x}");
        }
    }

    #[test]
    fn ceil_rounds_toward_positive_infinity() {
        let mut out = [0.0; 4];
        F32x4::splat(2.3).ceil().store(&mut out);
        assert_eq!(out, [3.0, 3.0, 3.0, 3.0]);

        F32x4::splat(-2.3).ceil().store(&mut out);
        assert_eq!(out, [-2.0, -2.0, -2.0, -2.0]);
    }

    #[test]
    fn round_currently_implements_x_plus_half_floor_not_ieee_rounding() {
        // Pins the *current* formula `(self + 0.5).floor()`, which CLAUDE.md
        // documents as a known non-IEEE bug (round(-1.5) == -1 here, -2 under
        // both x86 ties-to-even and aarch64 ties-away) rather than a
        // guaranteed contract — this test exists to catch accidental further
        // drift, not to certify the formula as correct.
        let mut out = [0.0; 4];
        F32x4::splat(2.4).round().store(&mut out);
        assert_eq!(out, [2.0, 2.0, 2.0, 2.0]);

        F32x4::splat(2.6).round().store(&mut out);
        assert_eq!(out, [3.0, 3.0, 3.0, 3.0]);

        F32x4::splat(-1.5).round().store(&mut out);
        assert_eq!(out, [-1.0, -1.0, -1.0, -1.0]);
    }

    #[test]
    fn fract_returns_the_value_minus_its_floor() {
        let mut out = [0.0; 4];
        F32x4::splat(3.75).fract().store(&mut out);
        for x in out {
            assert!((x - 0.75).abs() < 1e-6, "fract(3.75) got {x}");
        }

        F32x4::splat(-1.25).fract().store(&mut out);
        for x in out {
            assert!((x - 0.75).abs() < 1e-6, "fract(-1.25) got {x}");
        }
    }

    #[test]
    fn clamp_bounds_a_value_between_lo_and_hi() {
        let lo = F32x4::splat(0.0);
        let hi = F32x4::splat(3.0);
        let mut out = [0.0; 4];

        F32x4::splat(5.0).clamp(lo, hi).store(&mut out);
        assert_eq!(out, [3.0, 3.0, 3.0, 3.0]);

        F32x4::splat(-5.0).clamp(lo, hi).store(&mut out);
        assert_eq!(out, [0.0, 0.0, 0.0, 0.0]);

        F32x4::splat(1.5).clamp(lo, hi).store(&mut out);
        assert_eq!(out, [1.5, 1.5, 1.5, 1.5]);
    }

    #[test]
    fn sse2_reciprocal_math() {
        let a = F32x4::splat(4.0);
        let mut out = [0.0; 4];

        // Reciprocal
        let recip = a.recip();
        recip.store(&mut out);
        // recip is approximate, tolerance 1e-3
        for x in out.iter() {
            assert!(
                (x - 0.25).abs() < 1e-3,
                "recip(4.0) should be 0.25, got {}",
                x
            );
        }

        // Reciprocal Square Root
        let rsqrt = a.simd_rsqrt();
        rsqrt.store(&mut out);
        // rsqrt is approximate, tolerance 1e-3
        for x in out.iter() {
            assert!(
                (x - 0.5).abs() < 1e-3,
                "rsqrt(4.0) should be 0.5, got {}",
                x
            );
        }
    }
}
