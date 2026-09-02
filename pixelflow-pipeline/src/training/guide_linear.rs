//! Shared feature encoding + forward pass for the cold-start linear
//! saturation Guide (`docs/plans/2026-08-31-guide-design-revision.md` §5).
//!
//! Split out of `train_guide` (2026-09-02, design doc §5 task 2) so the
//! trainer and the mandatory train/deploy skew test
//! (`bin/skew_test_linear_guide.rs`) share exactly one definition of "what
//! a strict-label JSONL row means as a feature vector" and "what the linear
//! model's forward pass computes" — the same anti-drift discipline
//! `pixelflow_search::egraph::candidate`'s own module doc states ("there is
//! exactly one place a caller can get this wrong"). Two independent
//! re-derivations of this encoding were exactly the failure mode the skew
//! test exists to catch; keeping both binaries importing the same `Sample`/
//! `Model` means the only thing left free to diverge between "trainer
//! forward" and "deployed impl" is the actual code path each one calls
//! through — `Model::logit` here vs
//! `pixelflow_search::nnue::guide::linear::LinearCandidateGuide` — which is
//! the divergence the test is supposed to measure.

use std::collections::HashMap;

use pixelflow_ir::OpKind;
use serde::Deserialize;

/// One strict-label JSONL row, field-for-field what `gen_strict_labels`
/// writes (`pixelflow-pipeline/src/bin/gen_strict_labels.rs`).
#[derive(Deserialize)]
pub struct Record {
    pub expr_name: String,
    pub family_band: u32,
    pub family_seed: u64,
    pub expr_node_count: usize,
    pub rule_idx: usize,
    pub rule_name: String,
    pub budget_fraction: f32,
    pub match_class_node_count: usize,
    pub neighborhood_op_count: usize,
    pub neighborhood_op_hist: std::collections::BTreeMap<String, usize>,
    /// `true` when this application re-fired a `CandidateKey` an earlier
    /// application in the same episode already resolved.
    ///
    /// The live `GuidedSaturation` loop removes seen keys BEFORE scoring, so
    /// a repeat row is a candidate no deployed guide will ever be asked to
    /// rank. Consumers that train on or evaluate against these rows are
    /// measuring a population the deployment never sees — 93.4% of the
    /// committed TRAIN split, overwhelmingly negative. Kept in the schema
    /// (rather than filtered at mint time) so the dedup rate stays
    /// measurable; every consumer must decide about it explicitly.
    pub dedup_repeat: bool,
    pub label_positive: bool,
}

/// One training/eval sample's encoded features plus its label. Sparse in the
/// op dimension (`op_feats`) since a matched class's neighborhood rarely
/// touches more than a handful of distinct `OpKind`s even though the full op
/// table has ~50 entries.
pub struct Sample {
    pub rule_idx: usize,
    /// `(op_index, log1p(count))` pairs, one per distinct op present.
    pub op_feats: Vec<(usize, f32)>,
    pub budget_fraction: f32,
    pub log_match_class: f32,
    pub log_neighborhood: f32,
    pub log_expr_size: f32,
    pub label: f32,
}

/// Build the `OpKind` name -> dense-index table, in `OpKind::all()`'s own
/// order. `gen_strict_labels` wrote histogram keys as `format!("{op:?}")`
/// (a bare variant name, e.g. `"Abs"`), so `format!("{:?}", op)` here reads
/// back exactly the same strings — one definition of the op-name spelling
/// (`OpKind`'s own `Debug` impl), not restated.
#[must_use]
pub fn op_index_table() -> (Vec<String>, HashMap<String, usize>) {
    let names: Vec<String> = OpKind::all().map(|op| format!("{op:?}")).collect();
    let index: HashMap<String, usize> = names
        .iter()
        .enumerate()
        .map(|(i, n)| (n.clone(), i))
        .collect();
    (names, index)
}

/// Encode one JSONL [`Record`] into a [`Sample`] against a fixed op-name
/// index (see [`op_index_table`]).
#[must_use]
pub fn to_sample(record: &Record, op_index: &HashMap<String, usize>) -> Sample {
    let mut op_feats: Vec<(usize, f32)> = record
        .neighborhood_op_hist
        .iter()
        .map(|(name, &count)| {
            let idx = *op_index.get(name).unwrap_or_else(|| {
                panic!(
                    "guide_linear::to_sample: neighborhood_op_hist names op {name:?}, which is \
                     not in OpKind::all() — the strict-label dataset was minted against a \
                     different OpKind table than this binary was built with; regenerate the \
                     dataset"
                )
            });
            (idx, (count as f32 + 1.0).ln())
        })
        .collect();
    op_feats.sort_by_key(|&(i, _)| i);
    Sample {
        rule_idx: record.rule_idx,
        op_feats,
        budget_fraction: record.budget_fraction,
        log_match_class: (record.match_class_node_count as f32 + 1.0).ln(),
        log_neighborhood: (record.neighborhood_op_count as f32 + 1.0).ln(),
        log_expr_size: (record.expr_node_count as f32 + 1.0).ln(),
        label: if record.label_positive { 1.0 } else { 0.0 },
    }
}

/// Model: per-rule bias + bag-of-ops linear term + scalar features.
///
/// `logit = bias + w_rule[rule_idx] + sum_op(hist[op] * w_op[op]) +
/// w_budget*budget_fraction + w_match*log_match_class +
/// w_neigh*log_neighborhood + w_size*log_expr_size`.
///
/// This is the exact formula
/// [`pixelflow_search::nnue::guide::linear::LinearCandidateGuide`] must
/// reproduce; the skew test compares [`Model::logit`] against that type's
/// `score_candidates` on the same checkpoint weights and the same DEV rows.
pub struct Model {
    pub bias: f32,
    pub w_rule: Vec<f32>,
    pub w_op: Vec<f32>,
    pub w_budget: f32,
    pub w_match_class: f32,
    pub w_neighborhood: f32,
    pub w_expr_size: f32,
}

impl Model {
    /// Cold start: every weight at zero (design doc §5, "no warm-start from
    /// any prior guide/mask-head weights").
    #[must_use]
    pub fn zero(num_rules: usize, num_ops: usize) -> Self {
        Self {
            bias: 0.0,
            w_rule: vec![0.0; num_rules],
            w_op: vec![0.0; num_ops],
            w_budget: 0.0,
            w_match_class: 0.0,
            w_neighborhood: 0.0,
            w_expr_size: 0.0,
        }
    }

    /// See `pixelflow_search::nnue::guide::linear::LinearCandidateGuide::logit`'s
    /// doc for why every partial sum here goes through `black_box`: this
    /// function and that one are the two independent sides of the mandatory
    /// train/deploy skew test, and the barrier is what keeps their release-
    /// build FMA-contraction choices from silently diverging by a few ULPs.
    #[must_use]
    pub fn logit(&self, s: &Sample) -> f32 {
        let mut z = std::hint::black_box(self.bias);
        z = std::hint::black_box(z + self.w_rule[s.rule_idx]);
        z = std::hint::black_box(z + self.w_budget * s.budget_fraction);
        z = std::hint::black_box(z + self.w_match_class * s.log_match_class);
        z = std::hint::black_box(z + self.w_neighborhood * s.log_neighborhood);
        z = std::hint::black_box(z + self.w_expr_size * s.log_expr_size);
        for &(idx, count) in &s.op_feats {
            z = std::hint::black_box(z + self.w_op[idx] * count);
        }
        z
    }

    /// One online-SGD step for sample `s`, given the loss gradient w.r.t.
    /// the logit (`grad_z`, already clipped by the caller).
    ///
    /// The data term is sparse — only the features this sample carries have
    /// a gradient. The L2 penalty is NOT: its gradient is `l2 * w` for every
    /// weight on every step. Decaying only the weights a sample happened to
    /// touch made the effective penalty a function of feature frequency, so
    /// a rule firing once per epoch was regularized orders of magnitude less
    /// than `commutative` — which is a distortion of exactly the learned
    /// per-rule priorities this model exists to produce. The tables are ~60
    /// rules and ~50 ops, so decaying all of them per step is cheaper than
    /// the bookkeeping a lazy scheme would need (and exact, which a lazy
    /// scheme is not under this trainer's per-epoch learning-rate decay).
    ///
    /// The bias is exempt by convention: a bias term has no "large weight"
    /// failure mode L2 exists to guard against.
    pub fn sgd_step(&mut self, s: &Sample, grad_z: f32, lr: f32, l2: f32) {
        // Decay first, then the data gradient: `w * (1 - lr*l2) - lr*g` is
        // the same step as `w - lr*(g + l2*w)`, just written so the decay
        // can cover every weight uniformly.
        let keep = 1.0 - lr * l2;
        for w in self.w_rule.iter_mut().chain(self.w_op.iter_mut()) {
            *w *= keep;
        }
        self.w_budget *= keep;
        self.w_match_class *= keep;
        self.w_neighborhood *= keep;
        self.w_expr_size *= keep;

        self.bias -= lr * grad_z;
        self.w_rule[s.rule_idx] -= lr * grad_z;
        for &(idx, count) in &s.op_feats {
            self.w_op[idx] -= lr * grad_z * count;
        }
        self.w_budget -= lr * grad_z * s.budget_fraction;
        self.w_match_class -= lr * grad_z * s.log_match_class;
        self.w_neighborhood -= lr * grad_z * s.log_neighborhood;
        self.w_expr_size -= lr * grad_z * s.log_expr_size;
    }
}

// ── Ranking metrics ──────────────────────────────────────────────────────────
//
// Shared by `train_guide` (reports the linear model's DEV AUC/PR-AUC) and
// `eval_control_guides` (reports the same two metrics for the
// [`crate::training::guide_linear`]-free `PerRuleRateGuide` control) — one
// definition of each metric, so a side-by-side comparison between the two
// guides is comparing like with like.

/// Rank-based AUC-ROC (Mann-Whitney U, average-rank tie handling). `None` if
/// either class is absent — undefined, not zero.
#[must_use]
pub fn auc_roc(scores: &[f32], labels: &[f32]) -> Option<f64> {
    let n = scores.len();
    let n_pos = labels.iter().filter(|&&y| y > 0.5).count();
    let n_neg = n - n_pos;
    if n_pos == 0 || n_neg == 0 {
        return None;
    }
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| {
        scores[a]
            .partial_cmp(&scores[b])
            .unwrap_or_else(|| panic!("auc_roc: NaN score"))
    });
    let mut ranks = vec![0.0f64; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j + 1 < n && scores[idx[j + 1]] == scores[idx[i]] {
            j += 1;
        }
        let avg_rank = ((i + 1) + (j + 1)) as f64 / 2.0;
        for &k in &idx[i..=j] {
            ranks[k] = avg_rank;
        }
        i = j + 1;
    }
    let sum_ranks_pos: f64 = (0..n).filter(|&i| labels[i] > 0.5).map(|i| ranks[i]).sum();
    Some(
        (sum_ranks_pos - n_pos as f64 * (n_pos as f64 + 1.0) / 2.0) / (n_pos as f64 * n_neg as f64),
    )
}

/// One AUC per decision set, macro-averaged over the sets that contain both
/// classes — the ranking question a deployed guide is actually asked.
///
/// Returns `(mean_auc, groups_scored)`, or `None` if no group has both
/// classes.
///
/// A pooled AUC over every record lets a model gain credit from differences
/// BETWEEN expressions and saturation rounds. Two of the model's features
/// cannot possibly produce that credit at deploy time: `expr_node_count` is
/// constant for a whole episode and `budget_fraction` is shared by every
/// candidate in one live scoring round, so neither can reorder a single
/// round's candidates. Whenever positive prevalence varies with expression
/// size or progress, the pooled number reports ranking signal the guide
/// cannot use for move ordering; grouping removes the between-group
/// variation before the metric sees it.
#[must_use]
pub fn auc_roc_within_groups<K: Ord + Clone>(
    scores: &[f32],
    labels: &[f32],
    groups: &[K],
) -> Option<(f64, usize)> {
    assert_eq!(
        scores.len(),
        labels.len(),
        "auc_roc_within_groups: scores/labels length mismatch"
    );
    assert_eq!(
        scores.len(),
        groups.len(),
        "auc_roc_within_groups: scores/groups length mismatch"
    );
    let mut by_group: std::collections::BTreeMap<K, (Vec<f32>, Vec<f32>)> =
        std::collections::BTreeMap::new();
    for i in 0..scores.len() {
        let e = by_group
            .entry(groups[i].clone())
            .or_insert_with(|| (Vec::new(), Vec::new()));
        e.0.push(scores[i]);
        e.1.push(labels[i]);
    }
    let mut sum = 0.0f64;
    let mut n = 0usize;
    for (s, l) in by_group.values() {
        if let Some(a) = auc_roc(s, l) {
            sum += a;
            n += 1;
        }
    }
    (n > 0).then(|| (sum / n as f64, n))
}

/// Average precision (area under the precision-recall step function,
/// sklearn's `average_precision_score` convention). `None` if there are no
/// positives.
#[must_use]
pub fn average_precision(scores: &[f32], labels: &[f32]) -> Option<f64> {
    let n_pos = labels.iter().filter(|&&y| y > 0.5).count();
    if n_pos == 0 {
        return None;
    }
    let mut idx: Vec<usize> = (0..scores.len()).collect();
    idx.sort_by(|&a, &b| {
        scores[b]
            .partial_cmp(&scores[a])
            .unwrap_or_else(|| panic!("average_precision: NaN score"))
    });
    // sklearn's convention evaluates one threshold per DISTINCT score, so
    // every example tied at that score is on the positive side of it
    // together. Walking tied examples one at a time instead makes the result
    // depend on the input's record order — and the per-rule control guide,
    // which assigns one score per rule, is nothing but large tie groups.
    let mut tp = 0usize;
    let mut fp = 0usize;
    let mut ap = 0.0f64;
    let mut i = 0usize;
    while i < idx.len() {
        let mut j = i;
        while j + 1 < idx.len() && scores[idx[j + 1]] == scores[idx[i]] {
            j += 1;
        }
        let mut group_pos = 0usize;
        for &k in &idx[i..=j] {
            if labels[k] > 0.5 {
                group_pos += 1;
            } else {
                fp += 1;
            }
        }
        tp += group_pos;
        if group_pos > 0 {
            let precision = tp as f64 / (tp + fp) as f64;
            ap += precision * (group_pos as f64 / n_pos as f64);
        }
        i = j + 1;
    }
    Some(ap)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn l2_decays_every_weight_not_only_the_ones_a_sample_touched() {
        let mut m = Model::zero(3, 2);
        m.w_rule = vec![1.0, 1.0, 1.0];
        m.w_op = vec![1.0, 1.0];
        let s = Sample {
            rule_idx: 0,
            op_feats: vec![(0, 1.0)],
            budget_fraction: 0.0,
            log_match_class: 0.0,
            log_neighborhood: 0.0,
            log_expr_size: 0.0,
            label: 1.0,
        };
        // Zero gradient: the only thing that can move a weight is the L2 term.
        m.sgd_step(&s, 0.0, 0.1, 0.5);
        let keep = 1.0 - 0.1 * 0.5;
        for (i, w) in m.w_rule.iter().enumerate() {
            assert!(
                (w - keep).abs() < 1e-6,
                "w_rule[{i}] = {w}, expected every rule weight decayed to {keep} — a rule \
                 that did not fire this step must still pay the penalty"
            );
        }
        for (i, w) in m.w_op.iter().enumerate() {
            assert!((w - keep).abs() < 1e-6, "w_op[{i}] = {w}, expected {keep}");
        }
    }

    #[test]
    fn average_precision_groups_tied_scores_at_one_threshold() {
        // Two positives and two negatives, all tied at one score: sklearn's
        // average_precision_score is 0.5 (the single threshold's precision).
        let scores = vec![1.0f32; 4];
        let labels = vec![1.0f32, 0.0, 1.0, 0.0];
        let ap = average_precision(&scores, &labels).unwrap();
        assert!((ap - 0.5).abs() < 1e-9, "ap={ap}");
    }

    #[test]
    fn average_precision_of_a_tied_block_is_independent_of_record_order() {
        let scores = vec![1.0f32; 6];
        let a = average_precision(&scores, &[1.0, 1.0, 0.0, 0.0, 0.0, 0.0]).unwrap();
        let b = average_precision(&scores, &[0.0, 0.0, 0.0, 1.0, 0.0, 1.0]).unwrap();
        assert!(
            (a - b).abs() < 1e-12,
            "a tie group must not be ranked by JSONL order: {a} vs {b}"
        );
    }

    #[test]
    fn within_group_auc_ignores_between_group_score_offsets() {
        // Inside each group the ranking is perfect; the groups sit at
        // completely different score levels. Pooled AUC is dragged down by
        // the between-group offset, the grouped one is not.
        let scores = vec![0.1f32, 0.2, 0.8, 0.9];
        let labels = vec![0.0f32, 1.0, 0.0, 1.0];
        let groups = ["a", "a", "b", "b"];
        let (mean, n) = auc_roc_within_groups(&scores, &labels, &groups).unwrap();
        assert_eq!(n, 2);
        assert!((mean - 1.0).abs() < 1e-9, "mean={mean}");
    }

    #[test]
    fn within_group_auc_is_none_when_no_group_holds_both_classes() {
        let scores = vec![0.1f32, 0.2];
        let labels = vec![0.0f32, 0.0];
        let groups = ["a", "b"];
        assert!(auc_roc_within_groups(&scores, &labels, &groups).is_none());
    }

    #[test]
    fn auc_roc_is_one_for_perfectly_separated_scores() {
        let scores = vec![0.1, 0.2, 0.8, 0.9];
        let labels = vec![0.0, 0.0, 1.0, 1.0];
        assert!((auc_roc(&scores, &labels).unwrap() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn auc_roc_is_zero_for_perfectly_inverted_scores() {
        let scores = vec![0.9, 0.8, 0.2, 0.1];
        let labels = vec![0.0, 0.0, 1.0, 1.0];
        assert!(auc_roc(&scores, &labels).unwrap().abs() < 1e-9);
    }

    #[test]
    fn auc_roc_is_none_when_a_class_is_absent() {
        let scores = vec![0.1, 0.2, 0.3];
        let labels = vec![0.0, 0.0, 0.0];
        assert!(auc_roc(&scores, &labels).is_none());
    }

    #[test]
    fn average_precision_is_one_for_perfectly_separated_scores() {
        let scores = vec![0.1, 0.2, 0.8, 0.9];
        let labels = vec![0.0, 0.0, 1.0, 1.0];
        assert!((average_precision(&scores, &labels).unwrap() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn average_precision_is_none_with_no_positives() {
        let scores = vec![0.1, 0.2, 0.3];
        let labels = vec![0.0, 0.0, 0.0];
        assert!(average_precision(&scores, &labels).is_none());
    }

    #[test]
    fn op_index_table_has_one_entry_per_opkind_and_no_duplicate_names() {
        let (names, index) = op_index_table();
        assert_eq!(names.len(), index.len(), "duplicate OpKind Debug names");
        assert!(!names.is_empty());
        for (i, n) in names.iter().enumerate() {
            assert_eq!(index[n], i);
        }
    }

    #[test]
    fn model_zero_start_scores_every_sample_at_the_bias_only() {
        let model = Model::zero(5, 3);
        let s = Sample {
            rule_idx: 2,
            op_feats: vec![(0, 1.5), (2, 0.5)],
            budget_fraction: 0.3,
            log_match_class: 1.0,
            log_neighborhood: 1.0,
            log_expr_size: 1.0,
            label: 1.0,
        };
        assert_eq!(
            model.logit(&s),
            0.0,
            "a zero-initialized model must score zero everywhere"
        );
    }

    #[test]
    fn sgd_step_moves_a_positive_samples_logit_upward() {
        let mut model = Model::zero(3, 2);
        let s = Sample {
            rule_idx: 1,
            op_feats: vec![(0, 1.0)],
            budget_fraction: 0.5,
            log_match_class: 0.5,
            log_neighborhood: 0.5,
            log_expr_size: 0.5,
            label: 1.0,
        };
        let before = model.logit(&s);
        // A positive gradient step (as if pos_weight*(p-1) with p<1, i.e.
        // negative grad_z) should raise the logit; use a concrete negative
        // grad_z directly since sigmoid/loss live in the trainer binary.
        model.sgd_step(&s, -1.0, 0.1, 0.0);
        let after = model.logit(&s);
        assert!(after > before, "before={before} after={after}");
    }
}
