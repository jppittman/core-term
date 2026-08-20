#[cfg(target_arch = "x86_64")]
#[cfg(test)]
mod tests {
    extern crate std;
    use pixelflow_core::backend::x86::{F32x4, U32x4};
    use pixelflow_core::backend::{MaskOps, SimdOps, SimdU32Ops};
    use std::prelude::v1::*;

    #[test]
    fn add_sub_mul_and_div_operate_lanewise_on_all_four_lanes() {
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
    fn sequential_produces_four_consecutive_values_from_the_given_start() {
        let seq = F32x4::sequential(10.0);
        let mut out = [0.0; 4];
        seq.store(&mut out);
        assert_eq!(out, [10.0, 11.0, 12.0, 13.0]);
    }

    #[test]
    fn cmp_lt_and_simd_select_choose_lanes_based_on_the_comparison_result() {
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
    fn bitand_computes_the_lanewise_bitwise_and() {
        let a = F32x4::splat(1.0); // 1.0 is 0x3f800000
        let b = F32x4::splat(2.0); // 2.0 is 0x40000000
        let c = a & b;
        let mut out = [0.0; 4];
        c.store(&mut out);
        assert_eq!(out, [0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn bitor_computes_the_lanewise_bitwise_or() {
        // 1.0 = 0x3f800000, 2.0 = 0x40000000; OR = 0x7f800000 (+inf).
        let a = F32x4::splat(1.0);
        let b = F32x4::splat(2.0);
        let got: [u32; 4] = lanes(a | b).map(f32::to_bits);
        assert_eq!(got, [0x7f800000; 4]);
    }

    #[test]
    fn not_flips_every_bit_in_each_lane() {
        let a = F32x4::splat(0.0);
        let got: [u32; 4] = lanes(!a).map(f32::to_bits);
        assert_eq!(got, [u32::MAX; 4]);
    }

    #[test]
    fn simd_sqrt_simd_abs_and_simd_min_compute_correct_lane_results() {
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
    fn any_and_all_report_whether_any_or_every_lane_is_true() {
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
    fn store_panics_when_the_output_slice_is_shorter_than_four_lanes() {
        let a = F32x4::default();
        let mut out = [0.0; 3]; // Too small
        a.store(&mut out);
    }

    #[test]
    fn recip_and_simd_rsqrt_approximate_the_reciprocal_and_inverse_sqrt() {
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

    // ── SimdOps provided methods ──────────────────────────────────────────
    //
    // Everything above exercises a `SimdOps` *required* method — the per-ISA
    // primitive. These exercise the *provided* methods, the uniform
    // expansions built on top of them, which had no direct coverage at all:
    // a "replace the whole function with `Default::default()`" mutant
    // survived for every one of them.

    fn lanes(v: F32x4) -> [f32; 4] {
        let mut out = [0.0; 4];
        v.store(&mut out);
        out
    }

    #[test]
    fn simd_exp_matches_the_scalar_exponential() {
        let got = lanes(F32x4::splat(3.0).exp());
        let want = 3.0f32.exp();
        for x in got {
            assert!((x - want).abs() < 1e-2, "exp(3.0) = {x}, want {want}");
        }
    }

    #[test]
    fn simd_ln_matches_the_scalar_natural_log() {
        let got = lanes(F32x4::splat(10.0).ln());
        let want = 10.0f32.ln();
        for x in got {
            assert!((x - want).abs() < 1e-2, "ln(10.0) = {x}, want {want}");
        }
    }

    #[test]
    fn simd_log10_matches_the_scalar_base_10_log() {
        let got = lanes(F32x4::splat(100.0).log10());
        let want = 100.0f32.log10();
        for x in got {
            assert!((x - want).abs() < 1e-2, "log10(100.0) = {x}, want {want}");
        }
    }

    #[test]
    fn simd_pow_matches_the_scalar_power_function() {
        // self=4.0, exp=0.5: log2(self) = 2, distinguishing `exp * log2(self)`
        // from `exp + log2(self)` and `exp / log2(self)`, which a self value
        // with log2(self) == 1 would not.
        let got = lanes(F32x4::splat(4.0).pow(F32x4::splat(0.5)));
        let want = 4.0f32.powf(0.5);
        for x in got {
            assert!((x - want).abs() < 1e-2, "pow(4.0, 0.5) = {x}, want {want}");
        }
    }

    #[test]
    fn simd_hypot_computes_the_euclidean_norm() {
        let got = lanes(F32x4::splat(3.0).hypot(F32x4::splat(4.0)));
        for x in got {
            assert!((x - 5.0).abs() < 1e-2, "hypot(3.0, 4.0) = {x}, want 5.0");
        }
    }

    #[test]
    fn simd_mul_rsqrt_divides_by_the_square_root() {
        let got = lanes(F32x4::splat(10.0).mul_rsqrt(F32x4::splat(4.0)));
        for x in got {
            assert!(
                (x - 5.0).abs() < 1e-2,
                "mul_rsqrt(10.0, 4.0) = {x}, want 5.0 (10.0 / sqrt(4.0))"
            );
        }
    }

    #[test]
    fn simd_ceil_rounds_toward_positive_infinity() {
        assert_eq!(lanes(F32x4::splat(1.2).ceil()), [2.0; 4]);
        assert_eq!(lanes(F32x4::splat(-1.2).ceil()), [-1.0; 4]);
        assert_eq!(lanes(F32x4::splat(3.0).ceil()), [3.0; 4]);
    }

    #[test]
    fn simd_round_rounds_to_the_nearest_integer() {
        assert_eq!(lanes(F32x4::splat(2.7).round()), [3.0; 4]);
        assert_eq!(lanes(F32x4::splat(2.2).round()), [2.0; 4]);
    }

    #[test]
    fn simd_fract_returns_the_value_past_the_decimal_point() {
        let got = lanes(F32x4::splat(2.75).fract());
        for x in got {
            assert!((x - 0.75).abs() < 1e-5, "fract(2.75) = {x}, want 0.75");
        }
    }

    #[test]
    fn simd_clamp_bounds_a_value_to_the_given_range() {
        assert_eq!(
            lanes(F32x4::splat(5.0).clamp(F32x4::splat(0.0), F32x4::splat(10.0))),
            [5.0; 4]
        );
        assert_eq!(
            lanes(F32x4::splat(-5.0).clamp(F32x4::splat(0.0), F32x4::splat(10.0))),
            [0.0; 4]
        );
        assert_eq!(
            lanes(F32x4::splat(15.0).clamp(F32x4::splat(0.0), F32x4::splat(10.0))),
            [10.0; 4]
        );
    }

    // ── Debug, gather, bit manipulation, and U32x4 ─────────────────────────
    //
    // Closes the remaining `cargo-mutants` gaps in the always-compiled SSE2
    // region of `backend/x86.rs`: `Debug` impls with nothing checking their
    // output, `gather`'s index-clamping arithmetic, the bit-manipulation
    // helpers backing `log2`/`exp2`, and `U32x4` (packed-pixel path), none of
    // which had any direct coverage.

    fn u32_lanes(v: U32x4) -> [u32; 4] {
        let mut out = [0u32; 4];
        v.store(&mut out);
        out
    }

    #[test]
    fn f32x4_debug_format_lists_all_four_lane_values() {
        let v = F32x4::sequential(1.0);
        assert_eq!(format!("{v:?}"), "F32x4([1.0, 2.0, 3.0, 4.0])");
    }

    #[test]
    fn mask4_debug_format_shows_which_lanes_are_true() {
        let all_true = F32x4::splat(1.0).cmp_gt(F32x4::splat(0.0));
        assert_eq!(format!("{all_true:?}"), "Mask4(1111)");

        let all_false = F32x4::splat(0.0).cmp_gt(F32x4::splat(0.0));
        assert_eq!(format!("{all_false:?}"), "Mask4(0000)");
    }

    #[test]
    fn gather_reads_the_slice_element_at_each_lanes_index() {
        let slice = [10.0f32, 20.0, 30.0, 40.0, 50.0];
        let idx = F32x4::splat(2.0);
        assert_eq!(lanes(F32x4::gather(&slice, idx)), [30.0; 4]);
    }

    #[test]
    fn gather_clamps_an_out_of_range_index_to_the_last_slice_element() {
        // len=5, so the valid upper bound is `len - 1 = 4`. `+` would clamp to
        // `len + 1 = 6` and `/` would clamp to `len / 1 = 5` — both index past
        // the slice and panic, distinguishing them from the correct `- 1`.
        let slice = [10.0f32, 20.0, 30.0, 40.0, 50.0];
        let idx = F32x4::splat(99.0);
        assert_eq!(lanes(F32x4::gather(&slice, idx)), [50.0; 4]);
    }

    #[test]
    fn add_masked_adds_the_value_only_where_the_mask_is_true() {
        let base = F32x4::splat(10.0);
        let val = F32x4::splat(5.0);
        // [false, true, true, true]
        let mask = F32x4::sequential(0.0).cmp_gt(F32x4::splat(0.0));
        assert_eq!(lanes(base.add_masked(val, mask)), [10.0, 15.0, 15.0, 15.0]);
    }

    #[test]
    fn from_u32_bits_reinterprets_the_bit_pattern_as_a_float() {
        let got: [u32; 4] = lanes(F32x4::from_u32_bits(0x3f80_0000)).map(f32::to_bits);
        assert_eq!(got, [0x3f80_0000; 4]);
    }

    #[test]
    fn shr_u32_shifts_the_bit_pattern_right_by_the_given_count() {
        let v = F32x4::from_u32_bits(0xff00_0000);
        let got: [u32; 4] = lanes(v.shr_u32(24)).map(f32::to_bits);
        assert_eq!(got, [0x0000_00ff; 4]);
    }

    #[test]
    fn i32_to_f32_converts_the_bit_pattern_as_a_signed_integer_to_a_float() {
        let v = F32x4::from_u32_bits(5);
        assert_eq!(lanes(v.i32_to_f32()), [5.0; 4]);
    }

    #[test]
    fn u32x4_debug_format_lists_all_four_lane_values() {
        let v = U32x4::splat(42);
        assert_eq!(format!("{v:?}"), "U32x4([42, 42, 42, 42])");
    }

    #[test]
    fn u32x4_store_writes_all_four_lanes_to_the_output_slice() {
        let v = U32x4::splat(7);
        assert_eq!(u32_lanes(v), [7, 7, 7, 7]);
    }

    #[test]
    fn u32x4_bitand_computes_the_lanewise_bitwise_and() {
        let a = U32x4::splat(0b1100);
        let b = U32x4::splat(0b1010);
        assert_eq!(u32_lanes(a & b), [0b1000; 4]);
    }

    #[test]
    fn u32x4_bitor_computes_the_lanewise_bitwise_or() {
        let a = U32x4::splat(0b1100);
        let b = U32x4::splat(0b1010);
        assert_eq!(u32_lanes(a | b), [0b1110; 4]);
    }

    #[test]
    fn u32x4_not_flips_every_bit_in_each_lane() {
        let a = U32x4::splat(0);
        assert_eq!(u32_lanes(!a), [u32::MAX; 4]);
    }

    #[test]
    fn u32x4_shl_shifts_bits_left_by_the_given_count() {
        let a = U32x4::splat(1);
        assert_eq!(u32_lanes(a << 4), [16; 4]);
    }

    #[test]
    fn u32x4_shr_shifts_bits_right_by_the_given_count() {
        let a = U32x4::splat(16);
        assert_eq!(u32_lanes(a >> 4), [1; 4]);
    }

    #[test]
    fn pack_rgba_packs_four_float_channels_into_one_u32_per_pixel() {
        let r = F32x4::splat(1.0);
        let g = F32x4::splat(0.0);
        let b = F32x4::splat(0.5);
        let a = F32x4::splat(1.0);
        // r=255 (0xFF), g=0, b=127 (truncate(0.5*255)), a=255;
        // packed as R | (G << 8) | (B << 16) | (A << 24).
        assert_eq!(u32_lanes(U32x4::pack_rgba(r, g, b, a)), [0xff7f_00ff; 4]);
    }
}
