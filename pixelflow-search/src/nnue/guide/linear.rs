//! Deployment of the cold-start linear Guide, and its zero-candidate-local-
//! information control arm (`docs/plans/2026-08-31-guide-design-revision.md`
//! §5, task 2 of the 2026-09-02 cold-start-training round).
//!
//! # Why this module exists: the train/deploy skew it closes
//!
//! `pixelflow-pipeline`'s `train_guide` binary trains a transparent linear
//! model (`logit = bias + w_rule[rule_idx] + Σ_op(hist[op]·w_op[op]) +
//! w_budget·budget_fraction + w_match_class·log1p(match_class_node_count) +
//! w_neighborhood·log1p(neighborhood_op_count) +
//! w_expr_size·log1p(expr_node_count)`) directly against
//! [`crate::egraph::candidate::CandidateFeatures`]'s own field set, *not*
//! through [`super::SaturationGuide`] — a deliberate scope choice (see that
//! binary's module doc), but one that leaves the trained weights with no
//! path into a live guided-saturation loop unless something implements the
//! trait identically to what was trained. [`LinearCandidateGuide`] is that
//! something: it loads the same JSON checkpoint `train_guide` writes and
//! reproduces its `Model::logit` formula field-for-field. The mandatory
//! skew test (`pixelflow-pipeline`'s `skew_test_linear_guide` binary/test)
//! checks this module's score against `train_guide`'s own forward pass on
//! held-out DEV records and requires bit-exact (≤1e-6) agreement — the same
//! discipline the extraction-head program applied to its own train/deploy
//! boundary, applied here before this weight ever reaches a real saturation
//! loop.
//!
//! # The control arm: [`PerRuleRateGuide`]
//!
//! The Phase 3 registration doc's pre-flight (§8) warns that rule-
//! granularity oracle filtering barely moves the classical-band curve
//! (94.9% → 85.4% median regret at B=100) — almost all of the theoretical
//! headroom is candidate-local, not rule-local. [`PerRuleRateGuide`] is the
//! control that makes that distinction measurable for a *trained* Guide, not
//! just an oracle one: it scores a candidate using only its `rule_idx`,
//! looked up against the TRAIN-measured strict-positive rate for that rule —
//! zero candidate-local information (no neighborhood ops, no budget state,
//! no matched-class size). If [`LinearCandidateGuide`]'s held-out ranking
//! quality is close to this control's, the linear model learned little
//! beyond a per-rule base rate; a real gap is evidence the candidate-local
//! features (§4 of the design doc) earned their place.

use std::collections::BTreeMap;
use std::path::Path;

use pixelflow_ir::OpKind;
use serde_json::Value;

use super::{CandidateSummary, SaturationGuide};

/// One malformed-checkpoint error, named per field so a bad JSON file fails
/// loud with an actionable message rather than a generic parse panic (NO
/// SILENT FAILURES).
#[derive(Debug)]
pub struct CheckpointError(String);

impl std::fmt::Display for CheckpointError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}
impl std::error::Error for CheckpointError {}

fn field<'a>(v: &'a Value, path: &str, name: &str) -> Result<&'a Value, CheckpointError> {
    v.get(name).ok_or_else(|| {
        CheckpointError(format!(
            "checkpoint {path}: missing required field {name:?} — this is not a \
             `train_guide`-written checkpoint, or its schema has drifted"
        ))
    })
}

fn as_f32(v: &Value, path: &str, name: &str) -> Result<f32, CheckpointError> {
    field(v, path, name)?
        .as_f64()
        .map(|x| x as f32)
        .ok_or_else(|| {
            CheckpointError(format!("checkpoint {path}: field {name:?} is not a number"))
        })
}

fn as_f32_vec(v: &Value, path: &str, name: &str) -> Result<Vec<f32>, CheckpointError> {
    field(v, path, name)?
        .as_array()
        .ok_or_else(|| {
            CheckpointError(format!("checkpoint {path}: field {name:?} is not an array"))
        })?
        .iter()
        .enumerate()
        .map(|(i, e)| {
            e.as_f64().map(|x| x as f32).ok_or_else(|| {
                CheckpointError(format!(
                    "checkpoint {path}: field {name:?}[{i}] is not a number"
                ))
            })
        })
        .collect()
}

fn as_string_vec(v: &Value, path: &str, name: &str) -> Result<Vec<String>, CheckpointError> {
    field(v, path, name)?
        .as_array()
        .ok_or_else(|| {
            CheckpointError(format!("checkpoint {path}: field {name:?} is not an array"))
        })?
        .iter()
        .enumerate()
        .map(|(i, e)| {
            e.as_str().map(str::to_string).ok_or_else(|| {
                CheckpointError(format!(
                    "checkpoint {path}: field {name:?}[{i}] is not a string"
                ))
            })
        })
        .collect()
}

fn as_str(v: &Value, path: &str, name: &str) -> Result<String, CheckpointError> {
    field(v, path, name)?
        .as_str()
        .map(str::to_string)
        .ok_or_else(|| {
            CheckpointError(format!("checkpoint {path}: field {name:?} is not a string"))
        })
}

fn as_usize(v: &Value, path: &str, name: &str) -> Result<usize, CheckpointError> {
    field(v, path, name)?
        .as_u64()
        .map(|x| x as usize)
        .ok_or_else(|| {
            CheckpointError(format!(
                "checkpoint {path}: field {name:?} is not a non-negative integer"
            ))
        })
}

/// FNV-1a 64, hex — `pixelflow-pipeline`'s `schema::fnv1a64_hex` recomputed
/// here rather than imported because the dependency runs the other way
/// (that crate depends on this one). The two are held together by
/// `checkpoint_hash_agrees_with_the_writer` in `train_guide`'s own tests: a
/// checkpoint it writes must load here, so a drift in either function is a
/// test failure rather than a silent mis-validation.
fn fnv1a64_hex(bytes: &[u8]) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{h:016x}")
}

/// `train_guide::GuideCheckpoint::weights_fingerprint`, recomputed from the
/// parsed JSON: FNV-1a 64 over the weight-bearing fields in a fixed order at
/// a fixed precision. Anything that edits a weight without rewriting the
/// checkpoint — a hand-patched file, a partial copy, a merge of two
/// checkpoints — changes this and not the recorded hash.
fn weights_fingerprint(
    bias: f32,
    w_rule: &[f32],
    w_op: &[f32],
    w_budget: f32,
    w_match_class: f32,
    w_neighborhood: f32,
    w_expr_size: f32,
) -> String {
    let mut buf = String::new();
    buf.push_str(&format!("{bias:.9}\n"));
    for w in w_rule {
        buf.push_str(&format!("{w:.9}\n"));
    }
    for w in w_op {
        buf.push_str(&format!("{w:.9}\n"));
    }
    buf.push_str(&format!(
        "{w_budget:.9}\n{w_match_class:.9}\n{w_neighborhood:.9}\n{w_expr_size:.9}\n"
    ));
    fnv1a64_hex(buf.as_bytes())
}

fn read_json(path: &Path) -> Result<Value, CheckpointError> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| CheckpointError(format!("cannot read {}: {e}", path.display())))?;
    serde_json::from_str(&text)
        .map_err(|e| CheckpointError(format!("cannot parse {} as JSON: {e}", path.display())))
}

/// One distinct op's `(op-histogram-index, log1p(occurrence-count))` pair,
/// computed exactly as `train_guide`'s `to_sample` builds `Sample::op_feats`
/// from `neighborhood_op_hist` — grouped by [`OpKind`]'s own `Debug` name
/// (the same string `gen_strict_labels` wrote as a JSONL histogram key),
/// counted, then `(count + 1.0).ln()` (NOT `ln_1p` — the trainer's literal
/// formula, matched exactly for the skew test's bit-exactness bar).
fn op_features(
    neighborhood_ops: &[OpKind],
    op_index: &BTreeMap<String, usize>,
) -> Vec<(usize, f32)> {
    let mut counts: BTreeMap<usize, usize> = BTreeMap::new();
    for op in neighborhood_ops {
        let name = format!("{op:?}");
        let idx = *op_index.get(&name).unwrap_or_else(|| {
            panic!(
                "LinearCandidateGuide: neighborhood op {name:?} has no entry in this \
                 checkpoint's op_names table — the checkpoint was trained against a \
                 different OpKind table than this binary was built with; retrain or \
                 rebuild against a matching pixelflow-ir revision"
            )
        });
        *counts.entry(idx).or_insert(0) += 1;
    }
    counts
        .into_iter()
        .map(|(idx, count)| (idx, (count as f32 + 1.0).ln()))
        .collect()
}

/// Deploys `train_guide`'s cold-start linear model as a [`SaturationGuide`]
/// — see the module doc for why this must reproduce `Model::logit` exactly
/// rather than approximate it.
///
/// Scores are raw logits (no sigmoid): [`SaturationGuide::score_candidates`]
/// only needs a move-ordering rank, sigmoid is monotonic and so preserves
/// it, and skipping it avoids a transcendental per candidate. The mandatory
/// skew test compares this module's logit directly against
/// `train_guide::Model::logit`, not against the trainer's reported
/// (post-sigmoid) DEV probabilities.
pub struct LinearCandidateGuide {
    bias: f32,
    w_rule: Vec<f32>,
    w_op: Vec<f32>,
    w_budget: f32,
    w_match_class: f32,
    w_neighborhood: f32,
    w_expr_size: f32,
    /// `OpKind` Debug name -> index into `w_op` — built from the
    /// checkpoint's own `op_names` array, so this guide's op indexing can
    /// never silently disagree with the weights it was loaded next to (see
    /// module doc: the checkpoint, not `OpKind::all()`'s current order, is
    /// the source of truth for which index a weight lives at).
    op_index: BTreeMap<String, usize>,
}

impl LinearCandidateGuide {
    /// Load a checkpoint written by `pixelflow-pipeline`'s `train_guide`
    /// (`GuideCheckpoint::write`) and bind it to the rule table it will
    /// score against. Fails loud (`Err`, never a default) on a missing file,
    /// malformed JSON, a checkpoint missing a field this model needs, a
    /// checkpoint whose recorded content hash does not match its own
    /// weights, or a rule table that disagrees with the one the weights were
    /// trained against — a Guide silently scoring with an all-zero weight it
    /// couldn't find, or with `w_rule[i]` naming a different rewrite than it
    /// did at training time, would corrupt saturation's move ordering
    /// without saying so.
    ///
    /// `rules` is the caller's live rule list in `rule_idx` order — the same
    /// `all_rules()` the e-graph is built with. `w_rule` is indexed by
    /// `rule_idx` and nothing about the index space is self-describing, so
    /// the checkpoint's `rule_names` is the only thing that can catch a
    /// reordered rule table; it is a required argument rather than an
    /// optional `verify` call because a guide constructed without the check
    /// is exactly the object that must not exist.
    pub fn load(path: &Path, rules: &[&str]) -> Result<Self, CheckpointError> {
        let p = path.display().to_string();
        let v = read_json(path)?;

        let bias = as_f32(&v, &p, "bias")?;
        let w_rule = as_f32_vec(&v, &p, "w_rule")?;
        let w_op = as_f32_vec(&v, &p, "w_op")?;
        let w_budget = as_f32(&v, &p, "w_budget")?;
        let w_match_class = as_f32(&v, &p, "w_match_class")?;
        let w_neighborhood = as_f32(&v, &p, "w_neighborhood")?;
        let w_expr_size = as_f32(&v, &p, "w_expr_size")?;
        let op_names = as_string_vec(&v, &p, "op_names")?;
        let rule_names = as_string_vec(&v, &p, "rule_names")?;
        let num_rules = as_usize(&v, &p, "num_rules")?;
        let num_ops = as_usize(&v, &p, "num_ops")?;
        let recorded_hash = as_str(&v, &p, "weights_fnv64")?;
        // Read so a checkpoint without one is refused; the identity's
        // *value* is computed by `pixelflow-pipeline`'s `SchemaIdentity`,
        // which this crate is below in the dependency graph and so cannot
        // evaluate. The content hash below is the check that has teeth here.
        let _schema_identity = as_str(&v, &p, "schema_identity")?;

        let actual_hash = weights_fingerprint(
            bias,
            &w_rule,
            &w_op,
            w_budget,
            w_match_class,
            w_neighborhood,
            w_expr_size,
        );
        if recorded_hash != actual_hash {
            return Err(CheckpointError(format!(
                "checkpoint {p}: recorded weights_fnv64 {recorded_hash} but the weights \
                 in the file hash to {actual_hash} — the file was edited, partially \
                 copied, or merged after `train_guide` wrote it; retrain rather than \
                 deploy weights whose provenance is broken"
            )));
        }

        if num_ops != w_op.len() || op_names.len() != w_op.len() {
            return Err(CheckpointError(format!(
                "checkpoint {p}: num_ops={num_ops}, op_names has {} entries, w_op has {} \
                 — the checkpoint's own op-array lengths disagree, cannot build a \
                 trustworthy op index",
                op_names.len(),
                w_op.len()
            )));
        }
        if num_rules != w_rule.len() || rule_names.len() != w_rule.len() {
            return Err(CheckpointError(format!(
                "checkpoint {p}: num_rules={num_rules}, rule_names has {} entries, w_rule \
                 has {} — the checkpoint's own rule-array lengths disagree",
                rule_names.len(),
                w_rule.len()
            )));
        }
        if rules.len() < w_rule.len() {
            return Err(CheckpointError(format!(
                "checkpoint {p}: trained against {} rules but this caller's rule table has \
                 only {} — every trained `rule_idx` must name a rule that still exists",
                w_rule.len(),
                rules.len()
            )));
        }
        // `""` marks a `rule_idx` never observed in TRAIN or DEV, so the
        // checkpoint has no name to check against — its weight is the
        // untrained zero either way.
        for (idx, trained) in rule_names.iter().enumerate() {
            if trained.is_empty() || trained == rules[idx] {
                continue;
            }
            return Err(CheckpointError(format!(
                "checkpoint {p}: rule_idx {idx} was trained against rule {trained:?} but \
                 this caller's rule table has {:?} there — the rule table was reordered or \
                 edited since training, so every w_rule weight past this index is attached \
                 to the wrong rewrite; retrain against the current table",
                rules[idx]
            )));
        }
        let op_index: BTreeMap<String, usize> = op_names
            .into_iter()
            .enumerate()
            .map(|(i, name)| (name, i))
            .collect();

        Ok(Self {
            bias,
            w_rule,
            w_op,
            w_budget,
            w_match_class,
            w_neighborhood,
            w_expr_size,
            op_index,
        })
    }

    /// One candidate's logit — `train_guide::Model::logit`'s exact formula,
    /// the single computation the mandatory skew test checks bit-exactness
    /// against.
    ///
    /// # Why every step goes through `black_box`
    ///
    /// This function and `train_guide::Model::logit` are two independent
    /// implementations of the identical formula, by design (see this
    /// module's doc — a shared helper would make the skew test check
    /// nothing). Written the obvious way, both agree bit-for-bit in an
    /// unoptimized build but can disagree by a handful of ULPs under
    /// release optimization: LLVM is free to fuse a `w*x + acc` pattern
    /// into one rounding (an FMA) in one function's compiled form and not
    /// the other's, purely as a function of each call site's own inlining
    /// and register allocation — the same single-vs-two-rounding
    /// `MulAdd` divergence this codebase's own floating-point doctrine
    /// documents (`CLAUDE.md`, "Floating point at the edges"), just
    /// surfacing between two host functions instead of between two ISA
    /// targets. `std::hint::black_box` after every partial sum forces that
    /// value to be a fully-rounded, materialized `f32` before the next
    /// term is added, which removes LLVM's opportunity to contract across
    /// the boundary — restoring the same two-rounding behavior an
    /// unoptimized build already gives for free, in release builds too.
    fn logit(&self, c: &CandidateSummary) -> f32 {
        let w_rule = *self.w_rule.get(c.rule_idx).unwrap_or_else(|| {
            panic!(
                "LinearCandidateGuide: rule_idx {} has no entry in this checkpoint's \
                 w_rule table ({} entries) — the checkpoint was trained against a \
                 different, smaller rule set than this candidate came from",
                c.rule_idx,
                self.w_rule.len()
            )
        });
        let log_match_class = (c.match_class_node_count as f32 + 1.0).ln();
        let log_neighborhood = (c.neighborhood_ops.len() as f32 + 1.0).ln();
        let log_expr_size = (c.expr_node_count as f32 + 1.0).ln();

        let mut z = std::hint::black_box(self.bias);
        z = std::hint::black_box(z + w_rule);
        z = std::hint::black_box(z + self.w_budget * c.budget_fraction);
        z = std::hint::black_box(z + self.w_match_class * log_match_class);
        z = std::hint::black_box(z + self.w_neighborhood * log_neighborhood);
        z = std::hint::black_box(z + self.w_expr_size * log_expr_size);
        for (idx, count) in op_features(&c.neighborhood_ops, &self.op_index) {
            z = std::hint::black_box(z + self.w_op[idx] * count);
        }
        z
    }
}

impl SaturationGuide for LinearCandidateGuide {
    fn score_candidates(&self, candidates: &[CandidateSummary]) -> Vec<f32> {
        candidates.iter().map(|c| self.logit(c)).collect()
    }
}

/// The zero-candidate-local-information control arm — see module doc.
/// Scores every candidate of a given rule with that rule's TRAIN-measured
/// strict-positive rate, ignoring everything else about the candidate
/// (neighborhood, budget state, matched-class size).
pub struct PerRuleRateGuide {
    /// `rule_idx -> TRAIN-measured strict-positive rate`, dense (indexable
    /// directly), `0.0` for a rule index that never fired in TRAIN.
    rate: Vec<f32>,
}

impl PerRuleRateGuide {
    /// Build directly from a dense `rule_idx -> rate` table — the seam a
    /// caller with its own measurement (e.g. computed straight from a TRAIN
    /// JSONL split, bypassing any checkpoint at all) uses.
    #[must_use]
    pub fn new(rate: Vec<f32>) -> Self {
        Self { rate }
    }

    /// Load from `train_guide`'s `--report-json` output (`Report::per_rule`,
    /// each row's `rule_idx`/`train_positive_rate`) — the same TRAIN-
    /// measured table [`LinearCandidateGuide`]'s checkpoint was trained
    /// alongside, so a side-by-side comparison of the two guides is reading
    /// two views of the exact same training run rather than two different
    /// ones.
    pub fn from_train_guide_report(path: &Path) -> Result<Self, CheckpointError> {
        let p = path.display().to_string();
        let v = read_json(path)?;
        let rows = field(&v, &p, "per_rule")?.as_array().ok_or_else(|| {
            CheckpointError(format!(
                "checkpoint {p}: field \"per_rule\" is not an array"
            ))
        })?;

        let mut rate: BTreeMap<usize, f32> = BTreeMap::new();
        for (i, row) in rows.iter().enumerate() {
            let rule_idx = row.get("rule_idx").and_then(Value::as_u64).ok_or_else(|| {
                CheckpointError(format!(
                    "report {p}: per_rule[{i}] missing integer field \"rule_idx\""
                ))
            })? as usize;
            let train_rate = row
                .get("train_positive_rate")
                .and_then(Value::as_f64)
                .ok_or_else(|| {
                    CheckpointError(format!(
                        "report {p}: per_rule[{i}] missing numeric field \
                         \"train_positive_rate\""
                    ))
                })? as f32;
            rate.insert(rule_idx, train_rate);
        }

        let max_idx = rate.keys().copied().max().unwrap_or(0);
        let mut dense = vec![0.0f32; max_idx + 1];
        for (idx, r) in rate {
            dense[idx] = r;
        }
        Ok(Self::new(dense))
    }
}

impl SaturationGuide for PerRuleRateGuide {
    fn score_candidates(&self, candidates: &[CandidateSummary]) -> Vec<f32> {
        candidates
            .iter()
            .map(|c| {
                *self.rate.get(c.rule_idx).unwrap_or_else(|| {
                    panic!(
                        "PerRuleRateGuide: rule_idx {} has no entry in this table ({} \
                         entries) — a candidate came from a rule set larger than the one \
                         this table was measured against",
                        c.rule_idx,
                        self.rate.len()
                    )
                })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nnue::factored::EMBED_DIM;

    const TEST_RULES: [&str; 3] = ["r0", "r1", "r2"];

    fn write_checkpoint(dir: &Path) -> std::path::PathBuf {
        let (bias, w_budget, w_match_class, w_neighborhood, w_expr_size) =
            (0.1f32, 2.0f32, 0.3f32, -0.2f32, 0.05f32);
        let w_rule = [1.0f32, -1.0, 0.0];
        let w_op = [0.5f32, -0.25];
        let path = dir.join("checkpoint.json");
        let json = serde_json::json!({
            "schema_identity": "0000000000000000",
            "num_rules": 3,
            "num_ops": 2,
            "rule_names": TEST_RULES,
            "bias": bias,
            "w_rule": w_rule,
            "w_op": w_op,
            "w_budget": w_budget,
            "w_match_class": w_match_class,
            "w_neighborhood": w_neighborhood,
            "w_expr_size": w_expr_size,
            "op_names": ["Add", "Mul"],
            "weights_fnv64": weights_fingerprint(
                bias, &w_rule, &w_op, w_budget, w_match_class, w_neighborhood, w_expr_size,
            ),
        });
        std::fs::write(&path, serde_json::to_string_pretty(&json).unwrap()).unwrap();
        path
    }

    fn candidate(rule_idx: usize, ops: Vec<OpKind>) -> CandidateSummary {
        CandidateSummary {
            rule_embed: [0.0; EMBED_DIM],
            neighborhood_ops: ops,
            budget_fraction: 0.4,
            rule_idx,
            match_class_node_count: 3,
            expr_node_count: 10,
        }
    }

    #[test]
    fn load_reads_every_field_and_scores_a_hand_computed_logit() {
        let dir = std::env::temp_dir().join(format!("linear_guide_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = write_checkpoint(&dir);

        let guide = LinearCandidateGuide::load(&path, &TEST_RULES).unwrap();
        let c = candidate(0, vec![OpKind::Add, OpKind::Add, OpKind::Mul]);
        let scores = guide.score_candidates(&[c]);

        // bias + w_rule[0] + w_budget*0.4 + w_match_class*ln(4) +
        // w_neighborhood*ln(4) + w_expr_size*ln(11)
        //   + w_op[Add]*ln(3) [2 occurrences] + w_op[Mul]*ln(2) [1 occurrence]
        let expected = 0.1
            + 1.0
            + 2.0 * 0.4
            + 0.3 * (4.0f32).ln()
            + (-0.2) * (4.0f32).ln()
            + 0.05 * (11.0f32).ln()
            + 0.5 * (3.0f32).ln()
            + (-0.25) * (2.0f32).ln();
        assert!(
            (scores[0] - expected).abs() < 1e-5,
            "got {}, expected {expected}",
            scores[0]
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn load_refuses_a_missing_file_loudly() {
        let result =
            LinearCandidateGuide::load(Path::new("/nonexistent/checkpoint.json"), &TEST_RULES);
        assert!(result.is_err());
    }

    #[test]
    fn load_refuses_a_checkpoint_whose_recorded_hash_does_not_match_its_weights() {
        let dir = std::env::temp_dir().join(format!("linear_guide_tamper_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = write_checkpoint(&dir);
        let text = std::fs::read_to_string(&path).unwrap();
        let mut v: Value = serde_json::from_str(&text).unwrap();
        v["w_rule"][0] = serde_json::json!(9.0);
        std::fs::write(&path, serde_json::to_string_pretty(&v).unwrap()).unwrap();

        let Err(err) = LinearCandidateGuide::load(&path, &TEST_RULES) else {
            panic!("a hand-edited weight must not deploy");
        };
        assert!(
            err.to_string().contains("weights_fnv64"),
            "error should name the hash it checked: {err}"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn load_refuses_a_rule_table_that_disagrees_with_the_checkpoint() {
        let dir = std::env::temp_dir().join(format!("linear_guide_rules_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = write_checkpoint(&dir);

        let Err(err) = LinearCandidateGuide::load(&path, &["r0", "REORDERED", "r2"]) else {
            panic!("a reordered rule table attaches every weight to the wrong rewrite");
        };
        assert!(
            err.to_string().contains("rule_idx 1"),
            "error should name the index that disagrees: {err}"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    #[should_panic(expected = "no entry in this checkpoint's w_rule table")]
    fn logit_panics_loudly_on_an_out_of_range_rule_idx() {
        let dir = std::env::temp_dir().join(format!("linear_guide_oob_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = write_checkpoint(&dir);
        let guide = LinearCandidateGuide::load(&path, &TEST_RULES).unwrap();
        let _ = guide.score_candidates(&[candidate(99, vec![])]);
    }

    #[test]
    fn per_rule_rate_guide_ignores_everything_but_rule_idx() {
        let guide = PerRuleRateGuide::new(vec![0.1, 0.9]);
        let a = candidate(1, vec![OpKind::Add, OpKind::Add, OpKind::Add]);
        let b = candidate(1, vec![OpKind::Sqrt]);
        let scores = guide.score_candidates(&[a, b]);
        assert!((scores[0] - 0.9).abs() < 1e-9);
        assert!(
            (scores[0] - scores[1]).abs() < 1e-9,
            "rule_idx alone determines the score"
        );
    }

    #[test]
    fn per_rule_rate_guide_from_train_guide_report_reads_rule_idx_and_train_rate() {
        let dir = std::env::temp_dir().join(format!("per_rule_report_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("report.json");
        let json = serde_json::json!({
            "per_rule": [
                {"rule_idx": 0, "rule_name": "a", "train_positive_rate": 0.02},
                {"rule_idx": 2, "rule_name": "b", "train_positive_rate": 0.15},
            ]
        });
        std::fs::write(&path, serde_json::to_string_pretty(&json).unwrap()).unwrap();

        let guide = PerRuleRateGuide::from_train_guide_report(&path).unwrap();
        let scores = guide.score_candidates(&[candidate(0, vec![]), candidate(2, vec![])]);
        assert!((scores[0] - 0.02).abs() < 1e-9);
        assert!((scores[1] - 0.15).abs() < 1e-9);

        std::fs::remove_dir_all(&dir).ok();
    }
}
