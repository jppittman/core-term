//! Small measurement-statistics helpers shared by every binary that reports
//! a held-out Spearman rho against measured JIT cost, or that needs a run
//! noise floor from repeated probe measurements.
//!
//! Moved out of `bootstrap_extraction_head` and `bin/mint_variant_sets`
//! (Aug 2026, round2b) so `bin/train_contrastive` did not restate them: "one
//! definition, imported, not restated" (CLAUDE.md, "Denote before you
//! build" corollary) — these had exactly one correct definition each and no
//! reason to fork.

use pixelflow_ir::{ExprArena, ExprId, OpKind};

/// Average (mid-)ranks of `values`, 1-based, ties resolved by the mean rank
/// of the tied block (so `spearman_rho` computes the exact tie-corrected
/// coefficient rather than the biased `6*sum(d^2)/(n*(n^2-1))` shortcut).
///
/// # Panics
///
/// Panics on a NaN value — ranking is undefined for it, and silently sorting
/// NaN to one end would fabricate a rank no upstream computation intended.
pub fn average_ranks(values: &[f32]) -> Vec<f64> {
    let n = values.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&i, &j| {
        values[i].partial_cmp(&values[j]).unwrap_or_else(|| {
            panic!(
                "average_ranks: non-comparable (NaN) values at indices {i} ({}) and {j} ({})",
                values[i], values[j]
            )
        })
    });
    let mut ranks = vec![0.0f64; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j + 1 < n && values[idx[j + 1]] == values[idx[i]] {
            j += 1;
        }
        // Average of 1-based ranks i+1 ..= j+1.
        let avg = (i + j) as f64 / 2.0 + 1.0;
        for &k in &idx[i..=j] {
            ranks[k] = avg;
        }
        i = j + 1;
    }
    ranks
}

/// Spearman rank correlation: Pearson correlation of average ranks (exact
/// under ties, unlike the 6Σd²/n(n²−1) shortcut).
///
/// Returns `None` when either side has zero rank variance (all values tied) —
/// the coefficient is undefined there, and the caller must report that
/// explicitly rather than receive a fabricated number.
///
/// # Panics
///
/// Panics on length mismatch, fewer than 2 samples, or NaN values.
pub fn spearman_rho(a: &[f32], b: &[f32]) -> Option<f64> {
    assert_eq!(
        a.len(),
        b.len(),
        "spearman_rho: length mismatch ({} vs {})",
        a.len(),
        b.len()
    );
    assert!(
        a.len() >= 2,
        "spearman_rho: need at least 2 samples, got {}",
        a.len()
    );
    let ra = average_ranks(a);
    let rb = average_ranks(b);
    // With average ranks the rank sum is always n(n+1)/2, so the mean rank
    // is (n+1)/2 regardless of ties.
    let mean = (a.len() as f64 + 1.0) / 2.0;
    let mut cov = 0.0f64;
    let mut var_a = 0.0f64;
    let mut var_b = 0.0f64;
    for (x, y) in ra.iter().zip(&rb) {
        let dx = x - mean;
        let dy = y - mean;
        cov += dx * dy;
        var_a += dx * dx;
        var_b += dy * dy;
    }
    if var_a == 0.0 || var_b == 0.0 {
        return None;
    }
    Some(cov / (var_a.sqrt() * var_b.sqrt()))
}

// ── Noise-floor helpers (moved from `bin/mint_variant_sets`, Round 2b) ──────

/// A moderate, fixed reference kernel, re-measured repeatedly to estimate a
/// run's measurement noise floor independent of `BenchSession`'s own
/// internal drift correction (which such a measurement is itself subject
/// to, same as every other label). Shape/size class matches what
/// `VariantSet` candidates are typically drawn from.
#[must_use]
pub fn noise_probe_arena() -> (ExprArena, ExprId) {
    let mut arena = ExprArena::new();
    let x = arena.push_var(0);
    let y = arena.push_var(1);
    let mut acc = x;
    for i in 0..6 {
        let k = arena.push_const(0.5 + i as f32 * 0.1);
        let mul = arena.push_binary(OpKind::Mul, acc, y);
        let add = arena.push_binary(OpKind::Add, mul, k);
        acc = arena.push_unary(OpKind::Sqrt, add);
    }
    (arena, acc)
}

/// Median of a non-empty slice (copies and sorts).
///
/// # Panics
///
/// Panics if `xs` is empty, or contains a value that cannot be ordered
/// (NaN) — a median over either is not a number this function can return.
#[must_use]
pub fn median(mut xs: Vec<f64>) -> f64 {
    assert!(!xs.is_empty(), "median of empty slice");
    xs.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    xs[xs.len() / 2]
}

/// `p`-th percentile (`p` in `[0, 1]`) of a non-empty slice, nearest-rank.
///
/// # Panics
///
/// Panics if `xs` is empty, or contains a value that cannot be ordered.
#[must_use]
pub fn percentile(mut xs: Vec<f64>, p: f64) -> f64 {
    assert!(!xs.is_empty(), "percentile of empty slice");
    xs.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    let idx = ((xs.len() - 1) as f64 * p).round() as usize;
    xs[idx]
}

/// Noise floor from repeated measurements of one fixed expression (A/A
/// style, `bench_extraction_3way`'s method): `max(|geomean ratio - 1|,
/// MAD-derived dispersion)`, as a percent. `None` when fewer than 2 probe
/// samples were taken (not enough to form a ratio).
#[must_use]
pub fn noise_floor_pct(probe_ns: &[f64]) -> Option<f64> {
    if probe_ns.len() < 2 {
        return None;
    }
    let ratios: Vec<f64> = probe_ns.windows(2).map(|w| w[1] / w[0]).collect();
    let log_ratios: Vec<f64> = ratios.iter().map(|r| r.ln()).collect();
    let geomean = (log_ratios.iter().sum::<f64>() / log_ratios.len() as f64).exp();
    let center = median(log_ratios.clone());
    let deviations: Vec<f64> = log_ratios.iter().map(|l| (l - center).abs()).collect();
    let mad = median(deviations);
    let dispersion = 1.4826 * mad / (ratios.len() as f64).sqrt();
    Some((geomean - 1.0).abs().max(dispersion) * 100.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perfect_agreement_is_rho_one() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let b = [10.0f32, 20.0, 30.0, 40.0, 50.0];
        assert!((spearman_rho(&a, &b).expect("defined") - 1.0).abs() < 1e-9);
    }

    #[test]
    fn perfect_inversion_is_rho_minus_one() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let b = [5.0f32, 4.0, 3.0, 2.0, 1.0];
        assert!((spearman_rho(&a, &b).expect("defined") + 1.0).abs() < 1e-9);
    }

    #[test]
    fn all_tied_is_undefined() {
        let a = [1.0f32, 1.0, 1.0];
        let b = [1.0f32, 2.0, 3.0];
        assert_eq!(spearman_rho(&a, &b), None);
    }

    #[test]
    fn ties_use_average_rank() {
        // a has a tie at positions 0,1 (both value 1.0); average rank 1.5.
        let a = [1.0f32, 1.0, 2.0];
        let ranks = average_ranks(&a);
        assert_eq!(ranks, vec![1.5, 1.5, 3.0]);
    }
}
