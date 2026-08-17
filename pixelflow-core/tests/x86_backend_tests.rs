#[cfg(target_arch = "x86_64")]
#[cfg(test)]
mod tests {
    extern crate std;
    use pixelflow_core::backend::x86::{F32x4, U32x4};
    use pixelflow_core::backend::{MaskOps, SimdOps, SimdU32Ops};
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

    // ── SimdOps required primitives with no direct coverage ────────────────
    //
    // 2026-08-17 mutation sweep of the whole file (`cargo mutants -p
    // pixelflow-core --file .../backend/x86.rs`, no `--lib` restriction —
    // scoping to `--lib` hides this integration-test file and silently
    // reports every mutant here as caught vacuously). These are `SimdOps`
    // *required* per-ISA primitives that had zero coverage anywhere, so a
    // "replace the whole function with `Default::default()`" (or an
    // arithmetic-operator swap inside it) mutant survived for each.

    #[test]
    fn gather_clamps_out_of_range_indices_to_the_last_valid_slot() {
        // idx=99 is out of range for a 4-element slice either way, so this
        // also kills the `len - 1` -> `len + 1` / `len / 1` mutants: both
        // would clamp to an index at or past `slice.len()` and panic on the
        // out-of-bounds scalar load, rather than silently returning 40.0.
        let slice = [10.0f32, 20.0, 30.0, 40.0];
        let got = lanes(F32x4::gather(&slice, F32x4::splat(99.0)));
        assert_eq!(got, [40.0; 4]);
    }

    #[test]
    fn add_masked_adds_only_where_the_mask_is_true() {
        let base = F32x4::splat(1.0);
        let addend = F32x4::splat(2.0);
        let all_true = F32x4::splat(1.0).cmp_gt(F32x4::splat(0.0));
        let all_false = F32x4::splat(0.0).cmp_gt(F32x4::splat(0.0));
        assert_eq!(lanes(base.add_masked(addend, all_true)), [3.0; 4]);
        assert_eq!(lanes(base.add_masked(addend, all_false)), [1.0; 4]);
    }

    #[test]
    fn from_u32_bits_reinterprets_the_bit_pattern_as_f32() {
        let got = lanes(F32x4::from_u32_bits(2.0f32.to_bits()));
        assert_eq!(got, [2.0; 4]);
    }

    #[test]
    fn shr_u32_performs_a_logical_right_shift_on_the_bit_pattern() {
        let shifted = F32x4::from_u32_bits(8).shr_u32(2);
        for x in lanes(shifted) {
            assert_eq!(x.to_bits(), 2);
        }
    }

    #[test]
    fn i32_to_f32_converts_the_reinterpreted_bits_as_a_signed_integer() {
        let got = lanes(F32x4::from_u32_bits(5).i32_to_f32());
        assert_eq!(got, [5.0; 4]);
    }

    #[test]
    fn f32x4_bitor_combines_lane_bits() {
        let a = F32x4::from_u32_bits(0b0110);
        let b = F32x4::from_u32_bits(0b1001);
        for x in lanes(a | b) {
            assert_eq!(x.to_bits(), 0b1111);
        }
    }

    #[test]
    fn f32x4_not_flips_every_bit() {
        for x in lanes(!F32x4::from_u32_bits(0)) {
            assert_eq!(x.to_bits(), u32::MAX);
        }
    }

    // ── U32x4 (packed RGBA lanes) ────────────────────────────────────────

    #[test]
    fn u32x4_debug_output_shows_each_lane_value() {
        // U32x4::to_array is private and reachable only through Debug, so
        // this is the only public-API way to observe it; also kills the
        // Debug::fmt -> Ok(Default::default()) mutant, since a mutated fmt
        // writes nothing and the formatted string comes back empty.
        let v = U32x4::splat(7);
        assert_eq!(format!("{v:?}"), "U32x4([7, 7, 7, 7])");
    }

    #[test]
    fn u32x4_store_writes_every_lane_into_the_output_slice() {
        let mut out = [0u32; 4];
        U32x4::splat(42).store(&mut out);
        assert_eq!(out, [42; 4]);
    }

    #[test]
    fn u32x4_bitand_combines_lane_bits() {
        let a = U32x4::splat(0b0110);
        let b = U32x4::splat(0b0011);
        let mut out = [0u32; 4];
        (a & b).store(&mut out);
        assert_eq!(out, [0b0010; 4]);
    }

    #[test]
    fn u32x4_bitor_combines_lane_bits() {
        let a = U32x4::splat(0b0110);
        let b = U32x4::splat(0b1001);
        let mut out = [0u32; 4];
        (a | b).store(&mut out);
        assert_eq!(out, [0b1111; 4]);
    }

    #[test]
    fn u32x4_not_flips_every_bit() {
        // Also kills the `_mm_set1_epi32(-1)` -> `_mm_set1_epi32(1)` mutant
        // (delete `-`): that would XOR with 1 instead of all-ones, leaving
        // every lane at 1 rather than u32::MAX.
        let mut out = [0u32; 4];
        (!U32x4::splat(0)).store(&mut out);
        assert_eq!(out, [u32::MAX; 4]);
    }

    #[test]
    fn u32x4_shl_shifts_bits_left() {
        let mut out = [0u32; 4];
        (U32x4::splat(1) << 4).store(&mut out);
        assert_eq!(out, [16; 4]);
    }

    #[test]
    fn u32x4_shr_shifts_bits_right() {
        let mut out = [0u32; 4];
        (U32x4::splat(16) >> 4).store(&mut out);
        assert_eq!(out, [1; 4]);
    }

    #[test]
    fn pack_rgba_packs_channels_as_r_or_g_shl8_or_b_shl16_or_a_shl24() {
        let r = F32x4::splat(1.0);
        let g = F32x4::splat(0.0);
        let b = F32x4::splat(0.0);
        let a = F32x4::splat(1.0);
        let mut out = [0u32; 4];
        U32x4::pack_rgba(r, g, b, a).store(&mut out);
        assert_eq!(out, [0xFF0000FF; 4]);
    }
}
