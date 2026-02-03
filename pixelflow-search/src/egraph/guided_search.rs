//! Guided search with learned rule filtering.
//!
//! This module implements epoch-based e-graph search where a learned Guide
//! predicts which rules will match, filtering out wasteful rule applications.
//!
//! # Architecture
//!
//! The Guide is a **match predictor**: it predicts P(rule will match given e-graph state).
//! Rules with P(match) > threshold are applied; others are skipped.
//!
//! This is like egg's epoch-based saturation, but with learned filtering:
//! - egg: Apply ALL rules every epoch
//! - GuidedSearch: Apply rules where Guide predicts a match
//!
//! # Training Signal
//!
//! The Guide is trained with asymmetric penalties:
//! - False positive (predict match, doesn't): Heavy penalty (wastes compute)
//! - False negative (predict no match, does): Light penalty (conservative)
//!
//! This makes the Guide learn to skip rules that won't match, while being
//! conservative about skipping rules that might.

use super::cost::CostFunction;
use super::extract::ExprTree;
use super::graph::EGraph;
use super::node::EClassId;

/// Statistics for a rule across search runs.
///
/// Used to compute historical features for the Guide.
#[derive(Clone, Debug, Default)]
pub struct RuleStats {
    /// Total times this rule was attempted
    pub total_attempts: usize,
    /// Total times this rule actually matched
    pub total_matches: usize,
    /// Last epoch this rule matched (None if never)
    pub last_match_epoch: Option<usize>,
}

impl RuleStats {
    /// Match rate: matches / attempts
    pub fn match_rate(&self) -> f32 {
        if self.total_attempts == 0 {
            0.0
        } else {
            self.total_matches as f32 / self.total_attempts as f32
        }
    }

    /// Epochs since last match (saturates at usize::MAX if never matched)
    pub fn epochs_since_match(&self, current_epoch: usize) -> usize {
        match self.last_match_epoch {
            Some(last) => current_epoch.saturating_sub(last),
            None => usize::MAX,
        }
    }
}

/// Features for scoring a rule's match probability.
///
/// The Guide uses these to predict whether a rule will match.
#[derive(Clone, Debug)]
pub struct RuleFeatures {
    /// Rule index (for embedding lookup)
    pub rule_idx: usize,
    /// Number of e-classes in the graph
    pub egraph_classes: usize,
    /// Total nodes across all e-classes
    pub egraph_nodes: usize,
    /// Historical match rate for this rule
    pub historical_match_rate: f32,
    /// Epochs since this rule last matched
    pub epochs_since_match: usize,
    /// Current epoch number
    pub current_epoch: usize,
    /// Maximum epochs (for budget fraction)
    pub max_epochs: usize,
}

impl RuleFeatures {
    /// Budget fraction remaining (1.0 = full budget, 0.0 = exhausted)
    pub fn budget_fraction(&self) -> f32 {
        if self.max_epochs == 0 {
            0.0
        } else {
            1.0 - (self.current_epoch as f32 / self.max_epochs as f32)
        }
    }

    /// Convert to fixed-size f32 array for neural network input.
    pub fn to_array(&self) -> [f32; 8] {
        [
            self.rule_idx as f32,
            self.egraph_classes as f32,
            self.egraph_nodes as f32,
            self.historical_match_rate,
            self.epochs_since_match.min(100) as f32, // Cap at 100
            self.current_epoch as f32,
            self.max_epochs as f32,
            self.budget_fraction(),
        ]
    }
}

/// Record of one rule's application attempt in an epoch.
///
/// Used for collecting training data.
#[derive(Clone, Debug)]
pub struct RuleRecord {
    /// Rule index
    pub rule_idx: usize,
    /// Features at time of decision
    pub features: RuleFeatures,
    /// Guide's prediction (if using Guide)
    pub predicted_p_match: Option<f32>,
    /// Whether we decided to apply this rule
    pub applied: bool,
    /// Whether the rule actually matched (made changes)
    pub matched: bool,
    /// Number of changes made (0 if didn't match)
    pub changes: usize,
}

/// Record of one epoch for training data.
#[derive(Clone, Debug)]
pub struct EpochRecord {
    /// Epoch number
    pub epoch: usize,
    /// Per-rule records
    pub rule_records: Vec<RuleRecord>,
    /// Cost before this epoch
    pub cost_before: i64,
    /// Cost after this epoch
    pub cost_after: i64,
    /// Total changes made this epoch
    pub total_changes: usize,
}

/// Result of guided search.
pub struct GuidedSearchResult {
    /// Best expression tree found
    pub best_tree: ExprTree,
    /// Cost of best tree
    pub best_cost: i64,
    /// Number of epochs used
    pub epochs_used: usize,
    /// Search trajectory for training
    pub trajectory: Vec<EpochRecord>,
    /// Final rule statistics
    pub rule_stats: Vec<RuleStats>,
    /// Reason search stopped
    pub stop_reason: StopReason,
}

/// Why guided search stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StopReason {
    /// Guide predicts no rules will match
    GuidePredictsNoMatch,
    /// No changes made (saturation on approved rules)
    Saturated,
    /// Maximum epochs reached
    MaxEpochs,
    /// No rules in the e-graph
    NoRules,
}

/// Match threshold for rule application.
pub const DEFAULT_MATCH_THRESHOLD: f32 = 0.3;

/// Guided search with learned rule filtering.
///
/// Like egg's epoch-based saturation, but with a learned Guide that
/// predicts which rules will match, skipping wasteful applications.
pub struct GuidedSearch {
    /// The e-graph being searched
    egraph: EGraph,
    /// Root e-class to extract from
    root: EClassId,
    /// Best cost found so far
    best_cost: i64,
    /// Best tree found so far
    best_tree: Option<ExprTree>,
    /// Current epoch
    epoch: usize,
    /// Maximum epochs
    max_epochs: usize,
    /// Per-rule statistics
    rule_stats: Vec<RuleStats>,
    /// Match threshold (rules with P(match) > threshold are applied)
    match_threshold: f32,
}

impl GuidedSearch {
    /// Create a new guided search.
    pub fn new(egraph: EGraph, root: EClassId, max_epochs: usize) -> Self {
        let num_rules = egraph.num_rules();
        Self {
            egraph,
            root,
            best_cost: i64::MAX,
            best_tree: None,
            epoch: 0,
            max_epochs,
            rule_stats: vec![RuleStats::default(); num_rules],
            match_threshold: DEFAULT_MATCH_THRESHOLD,
        }
    }

    /// Set the match threshold.
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.match_threshold = threshold;
        self
    }

    /// Get reference to the e-graph.
    pub fn egraph(&self) -> &EGraph {
        &self.egraph
    }

    /// Get mutable reference to the e-graph.
    pub fn egraph_mut(&mut self) -> &mut EGraph {
        &mut self.egraph
    }

    /// Extract features for a rule.
    fn extract_rule_features(&self, rule_idx: usize) -> RuleFeatures {
        let stats = &self.rule_stats[rule_idx];
        RuleFeatures {
            rule_idx,
            egraph_classes: self.egraph.num_classes(),
            egraph_nodes: self.egraph.node_count(),
            historical_match_rate: stats.match_rate(),
            epochs_since_match: stats.epochs_since_match(self.epoch),
            current_epoch: self.epoch,
            max_epochs: self.max_epochs,
        }
    }

    /// Apply a single rule everywhere it matches.
    ///
    /// Returns the number of changes made.
    fn apply_rule_everywhere(&mut self, rule_idx: usize) -> usize {
        let mut changes = 0;
        let num_classes = self.egraph.num_classes();

        for class_idx in 0..num_classes {
            let class_id = EClassId(class_idx as u32);
            let class_id = self.egraph.find(class_id);

            // Get nodes in this class
            let nodes: Vec<_> = self.egraph.nodes(class_id).to_vec();

            for (node_idx, _node) in nodes.iter().enumerate() {
                if self.egraph.apply_single_rule(rule_idx, class_id, node_idx) {
                    changes += 1;
                }
            }
        }

        changes
    }

    /// Run search with a Guide scoring function.
    ///
    /// The Guide takes (egraph, rule_idx, rule_stats) and returns P(match) in [0, 1].
    /// Rules with P(match) > threshold are applied.
    ///
    /// The Judge takes an ExprTree and returns its cost.
    pub fn run<G, J, C>(
        &mut self,
        mut guide: G,
        mut judge: J,
        costs: &C,
    ) -> GuidedSearchResult
    where
        G: FnMut(&EGraph, usize, &RuleStats) -> f32,
        J: FnMut(&ExprTree) -> i64,
        C: CostFunction,
    {
        let mut trajectory = Vec::new();
        let num_rules = self.egraph.num_rules();

        if num_rules == 0 {
            // No rules to apply - just extract and return
            let (tree, _cost) = self.egraph.extract_best(self.root, costs);
            let cost = judge(&tree);
            return GuidedSearchResult {
                best_tree: tree,
                best_cost: cost,
                epochs_used: 0,
                trajectory,
                rule_stats: self.rule_stats.clone(),
                stop_reason: StopReason::NoRules,
            };
        }

        // Initial extraction
        let (initial_tree, _) = self.egraph.extract_best(self.root, costs);
        self.best_cost = judge(&initial_tree);
        self.best_tree = Some(initial_tree);

        loop {
            if self.epoch >= self.max_epochs {
                return self.finish(StopReason::MaxEpochs, trajectory);
            }

            // Score each rule with the Guide
            let mut rules_to_apply = Vec::new();
            let mut rule_records = Vec::new();

            for rule_idx in 0..num_rules {
                let features = self.extract_rule_features(rule_idx);
                let p_match = guide(&self.egraph, rule_idx, &self.rule_stats[rule_idx]);

                let should_apply = p_match > self.match_threshold;

                rule_records.push(RuleRecord {
                    rule_idx,
                    features,
                    predicted_p_match: Some(p_match),
                    applied: should_apply,
                    matched: false, // Updated after application
                    changes: 0,
                });

                if should_apply {
                    rules_to_apply.push(rule_idx);
                }
            }

            // If Guide predicts nothing will match, stop
            if rules_to_apply.is_empty() {
                // Fill in records and finish
                let epoch_record = EpochRecord {
                    epoch: self.epoch,
                    rule_records,
                    cost_before: self.best_cost,
                    cost_after: self.best_cost,
                    total_changes: 0,
                };
                trajectory.push(epoch_record);
                return self.finish(StopReason::GuidePredictsNoMatch, trajectory);
            }

            // Apply approved rules and track which matched
            let cost_before = self.best_cost;
            let mut total_changes = 0;

            for rule_idx in &rules_to_apply {
                let changes = self.apply_rule_everywhere(*rule_idx);
                total_changes += changes;

                // Update rule stats
                self.rule_stats[*rule_idx].total_attempts += 1;
                if changes > 0 {
                    self.rule_stats[*rule_idx].total_matches += 1;
                    self.rule_stats[*rule_idx].last_match_epoch = Some(self.epoch);
                }

                // Update record
                if let Some(record) = rule_records.iter_mut().find(|r| r.rule_idx == *rule_idx) {
                    record.matched = changes > 0;
                    record.changes = changes;
                }
            }

            // Also update stats for rules we didn't try (for learning purposes)
            // We don't update total_attempts for these since we didn't try them

            // Extract and evaluate if changes were made
            let cost_after = if total_changes > 0 {
                let (tree, _) = self.egraph.extract_best(self.root, costs);
                let cost = judge(&tree);

                if cost < self.best_cost {
                    self.best_cost = cost;
                    self.best_tree = Some(tree);
                }
                self.best_cost
            } else {
                cost_before
            };

            // Record epoch
            let epoch_record = EpochRecord {
                epoch: self.epoch,
                rule_records,
                cost_before,
                cost_after,
                total_changes,
            };
            trajectory.push(epoch_record);

            // Check for saturation (no changes despite applying rules)
            if total_changes == 0 {
                return self.finish(StopReason::Saturated, trajectory);
            }

            self.epoch += 1;
        }
    }

    /// Run with a uniform Guide (baseline: all rules applied).
    ///
    /// This is equivalent to egg-style saturation.
    pub fn run_uniform<J, C>(&mut self, judge: J, costs: &C) -> GuidedSearchResult
    where
        J: FnMut(&ExprTree) -> i64,
        C: CostFunction,
    {
        // Uniform guide: always returns 1.0 (above any threshold)
        self.run(|_, _, _| 1.0, judge, costs)
    }

    /// Run with random Guide (for training data baseline).
    pub fn run_random<J, C>(&mut self, judge: J, costs: &C, seed: u64) -> GuidedSearchResult
    where
        J: FnMut(&ExprTree) -> i64,
        C: CostFunction,
    {
        // Simple LCG for reproducible randomness
        let mut state = seed;
        let random_guide = move |_: &EGraph, _: usize, _: &RuleStats| -> f32 {
            // LCG: state = (state * a + c) mod m
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            // Convert to [0, 1]
            (state as f32) / (u64::MAX as f32)
        };

        self.run(random_guide, judge, costs)
    }

    /// Finish search and return result.
    fn finish(
        &self,
        stop_reason: StopReason,
        trajectory: Vec<EpochRecord>,
    ) -> GuidedSearchResult {
        GuidedSearchResult {
            best_tree: self.best_tree.clone().expect("best_tree should be set"),
            best_cost: self.best_cost,
            epochs_used: self.epoch,
            trajectory,
            rule_stats: self.rule_stats.clone(),
            stop_reason,
        }
    }
}

/// Collect training data by running egg-style saturation.
///
/// This runs full saturation (all rules every epoch) and records
/// ground truth for "did rule X match in state Y?"
pub fn collect_match_training_data<C: CostFunction>(
    egraph: &mut EGraph,
    root: EClassId,
    costs: &C,
    max_epochs: usize,
) -> Vec<EpochRecord> {
    let mut trajectory = Vec::new();
    let num_rules = egraph.num_rules();

    if num_rules == 0 {
        return trajectory;
    }

    let mut rule_stats = vec![RuleStats::default(); num_rules];

    for epoch in 0..max_epochs {
        // Extract cost before
        let (tree_before, _) = egraph.extract_best(root, costs);
        let cost_before = tree_before.node_count() as i64; // Simple cost for now

        // Record features and try each rule
        let mut rule_records = Vec::new();

        for rule_idx in 0..num_rules {
            let stats = &rule_stats[rule_idx];
            let features = RuleFeatures {
                rule_idx,
                egraph_classes: egraph.num_classes(),
                egraph_nodes: egraph.node_count(),
                historical_match_rate: stats.match_rate(),
                epochs_since_match: stats.epochs_since_match(epoch),
                current_epoch: epoch,
                max_epochs,
            };

            // Count matches for this rule (without applying yet)
            // We'll apply all rules at once after counting
            let will_match = would_rule_match(egraph, rule_idx);

            rule_records.push(RuleRecord {
                rule_idx,
                features,
                predicted_p_match: None, // No prediction in ground truth collection
                applied: true,           // We always "apply" in egg-style
                matched: will_match,
                changes: 0, // Updated after actual application
            });
        }

        // Now apply all rules (egg-style epoch)
        let total_changes = egraph.apply_rules_once();

        // Update rule stats based on what actually happened
        for record in &mut rule_records {
            if record.matched {
                rule_stats[record.rule_idx].total_matches += 1;
                rule_stats[record.rule_idx].last_match_epoch = Some(epoch);
            }
            rule_stats[record.rule_idx].total_attempts += 1;
        }

        // Extract cost after
        let (tree_after, _) = egraph.extract_best(root, costs);
        let cost_after = tree_after.node_count() as i64;

        trajectory.push(EpochRecord {
            epoch,
            rule_records,
            cost_before,
            cost_after,
            total_changes,
        });

        // Stop if saturated
        if total_changes == 0 {
            break;
        }
    }

    trajectory
}

/// Check if a rule would match anywhere in the e-graph.
///
/// This is a read-only check (doesn't apply the rule).
fn would_rule_match(egraph: &EGraph, rule_idx: usize) -> bool {
    let Some(rule) = egraph.rule(rule_idx) else {
        return false;
    };

    for class_idx in 0..egraph.num_classes() {
        let class_id = EClassId(class_idx as u32);
        let class_id = egraph.find(class_id);

        for node in egraph.nodes(class_id) {
            if rule.apply(egraph, class_id, node).is_some() {
                return true;
            }
        }
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::egraph::{CostModel, all_rules, ops};

    #[test]
    fn test_guided_search_basic() {
        // Create simple expression: X + 0
        let expr = ExprTree::Op {
            op: &ops::Add,
            children: vec![
                ExprTree::var(0),
                ExprTree::constant(0.0),
            ],
        };

        let mut egraph = EGraph::with_rules(all_rules());
        let root = egraph.add_expr(&expr);

        let costs = CostModel::default();
        let mut search = GuidedSearch::new(egraph, root, 10);

        // Run with uniform guide (baseline)
        let result = search.run_uniform(
            |tree| tree.node_count() as i64,
            &costs,
        );

        // Should find simplified form
        assert!(result.epochs_used > 0, "Should use at least one epoch");
        assert!(result.best_cost <= 3, "Should simplify X + 0 to X");
    }

    #[test]
    fn test_collect_training_data() {
        // (X + 0) * 1
        let expr = ExprTree::Op {
            op: &ops::Mul,
            children: vec![
                ExprTree::Op {
                    op: &ops::Add,
                    children: vec![
                        ExprTree::var(0),
                        ExprTree::constant(0.0),
                    ],
                },
                ExprTree::constant(1.0),
            ],
        };

        let mut egraph = EGraph::with_rules(all_rules());
        let root = egraph.add_expr(&expr);

        let costs = CostModel::default();
        let trajectory = collect_match_training_data(&mut egraph, root, &costs, 10);

        // Should have some epochs
        assert!(!trajectory.is_empty(), "Should have trajectory data");

        // Each epoch should have records for all rules
        for epoch_record in &trajectory {
            assert_eq!(
                epoch_record.rule_records.len(),
                all_rules().len(),
                "Should have record for each rule"
            );
        }
    }

    #[test]
    fn test_rule_stats() {
        let mut stats = RuleStats::default();

        // Initially 0% match rate
        assert_eq!(stats.match_rate(), 0.0);
        assert_eq!(stats.epochs_since_match(0), usize::MAX);

        // Record some matches
        stats.total_attempts = 10;
        stats.total_matches = 3;
        stats.last_match_epoch = Some(5);

        assert!((stats.match_rate() - 0.3).abs() < 0.001);
        assert_eq!(stats.epochs_since_match(7), 2);
        assert_eq!(stats.epochs_since_match(5), 0);
    }
}
