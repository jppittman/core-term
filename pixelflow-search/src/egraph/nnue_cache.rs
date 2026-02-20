//! Cache of NNUE metadata for e-graph nodes.
//!
//! This module provides efficient caching of neural network embeddings and predictions
//! for e-graph nodes, enabling fast scoring of (expression, rule) pairs during guided search.
//!
//! ## Design
//!
//! The cache is decoupled from the core e-graph for:
//! - Optional usage (e-graph works without NNUE)
//! - Efficient bulk operations (batch computation at epoch start)
//! - Clear ownership semantics
//!
//! ## Usage
//!
//! ```ignore
//! let model = Arc::new(ExprNnue::load("model.bin")?);
//! let mut cache = NnueCache::new(model, &rules);
//!
//! // At each epoch start, refresh all embeddings
//! cache.refresh_all(&egraph);
//!
//! // Score rules for an e-node (O(1) using cached mask_features)
//! let score = cache.score(class_id, node_idx, rule_idx);
//! ```

extern crate alloc;

use alloc::sync::Arc;
use alloc::vec::Vec;

#[cfg(feature = "std")]
use std::collections::HashMap;
#[cfg(not(feature = "std"))]
use hashbrown::HashMap;

use super::node::EClassId;
use super::EGraph;
use super::nnue_adapter::eclass_to_expr;
use crate::nnue::{ExprNnue, EMBED_DIM};

/// Cached NNUE metadata for an e-node.
///
/// Stores precomputed embeddings and predictions to avoid recomputation
/// during rule scoring.
#[derive(Clone)]
pub struct ENodeMetadata {
    /// Expression embedding (24 dims).
    /// Shared representation used by both value and mask heads.
    pub expr_embed: [f32; EMBED_DIM],

    /// Predicted cost of this expression NOW.
    /// Used for beam extraction and as input to mask head.
    pub value_pred: f32,

    /// Mask features (24 dims).
    /// Computed from [expr_embed, value_pred] via mask MLP.
    /// Used for bilinear scoring with rule embeddings.
    pub mask_features: [f32; EMBED_DIM],
}

impl Default for ENodeMetadata {
    fn default() -> Self {
        Self {
            expr_embed: [0.0; EMBED_DIM],
            value_pred: 0.0,
            mask_features: [0.0; EMBED_DIM],
        }
    }
}

/// Cache of NNUE metadata for e-graph nodes.
///
/// Provides efficient batch computation and O(1) scoring for (expression, rule) pairs.
pub struct NnueCache {
    /// Metadata keyed by (canonical class_id, node_idx within class).
    cache: HashMap<(EClassId, usize), ENodeMetadata>,

    /// Model reference for computing new embeddings.
    model: Arc<ExprNnue>,

    /// Pre-computed rule embeddings (constant during search).
    /// Indexed by rule_idx.
    pub rule_embeds: Vec<[f32; EMBED_DIM]>,
}

impl NnueCache {
    /// Create new cache with model.
    ///
    /// Rule embeddings are NOT pre-computed here - call `set_rules()` or
    /// `refresh_all_with_rules()` to initialize them.
    #[must_use]
    pub fn new(model: Arc<ExprNnue>) -> Self {
        Self {
            cache: HashMap::new(),
            model,
            rule_embeds: Vec::new(),
        }
    }

    /// Create cache with pre-computed rule embeddings from hand-crafted features.
    #[must_use]
    pub fn with_rule_embeddings(
        model: Arc<ExprNnue>,
        rule_embeds: Vec<[f32; EMBED_DIM]>,
    ) -> Self {
        Self {
            cache: HashMap::new(),
            model,
            rule_embeds,
        }
    }

    /// Get the underlying model.
    #[must_use]
    pub fn model(&self) -> &ExprNnue {
        &self.model
    }

    /// Clear the cache.
    pub fn clear(&mut self) {
        self.cache.clear();
    }

    /// Number of cached e-nodes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Check if cache is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    /// Refresh cache for all e-nodes in the e-graph.
    ///
    /// This is the batch computation approach (Option D from the plan):
    /// - Clears existing cache
    /// - Iterates all e-classes and e-nodes
    /// - Computes and stores metadata for each
    ///
    /// Call this at the start of each saturation epoch.
    pub fn refresh_all(&mut self, egraph: &EGraph) {
        self.cache.clear();

        // Iterate all e-classes
        for class_id in egraph.class_ids() {
            let canonical = egraph.find(class_id);
            let nodes = egraph.nodes(canonical);

            // For each e-node in the class
            for node_idx in 0..nodes.len() {
                // Convert e-node to expression
                // Note: This uses the first representative for children.
                // For more accurate embeddings, we could recurse and pick best children.
                let expr = eclass_to_expr(egraph, canonical);

                // Compute metadata using the model
                let (expr_embed, value_pred, mask_features) = self.model.compute_metadata(&expr);
                let metadata = ENodeMetadata { expr_embed, value_pred, mask_features };

                self.cache.insert((canonical, node_idx), metadata);
            }
        }
    }

    /// Get cached metadata for an e-node.
    ///
    /// Returns None if not in cache (call `refresh_all` first).
    #[must_use]
    pub fn get(&self, class_id: EClassId, node_idx: usize) -> Option<&ENodeMetadata> {
        self.cache.get(&(class_id, node_idx))
    }

    /// Get cached metadata, computing on cache miss.
    pub fn get_or_compute(&mut self, egraph: &EGraph, class_id: EClassId, node_idx: usize) -> &ENodeMetadata {
        let canonical = egraph.find(class_id);
        let key = (canonical, node_idx);

        self.cache.entry(key).or_insert_with(|| {
            let expr = eclass_to_expr(egraph, canonical);
            let (expr_embed, value_pred, mask_features) = self.model.compute_metadata(&expr);
            ENodeMetadata { expr_embed, value_pred, mask_features }
        })
    }

    /// Score a single (e-node, rule) pair.
    ///
    /// Uses cached mask_features and pre-computed rule_embeds for O(1) scoring.
    ///
    /// # Panics
    ///
    /// Panics if the e-node is not in cache or rule_idx is out of bounds.
    #[must_use]
    pub fn score(&self, class_id: EClassId, node_idx: usize, rule_idx: usize) -> f32 {
        let meta = self.cache.get(&(class_id, node_idx))
            .unwrap_or_else(|| panic!("E-node ({:?}, {}) not in cache. Call refresh_all first.", class_id, node_idx));

        let rule_embed = &self.rule_embeds[rule_idx];

        self.model.bilinear_score(&meta.mask_features, rule_embed, rule_idx)
    }

    /// Score all rules for an e-node.
    ///
    /// Returns a vector of scores indexed by rule_idx.
    #[must_use]
    pub fn score_all_rules(&self, class_id: EClassId, node_idx: usize) -> Vec<f32> {
        let meta = self.cache.get(&(class_id, node_idx))
            .unwrap_or_else(|| panic!("E-node ({:?}, {}) not in cache. Call refresh_all first.", class_id, node_idx));

        self.rule_embeds
            .iter()
            .enumerate()
            .map(|(r, rule_embed)| self.model.bilinear_score(&meta.mask_features, rule_embed, r))
            .collect()
    }

    /// Get the value prediction for an e-node.
    ///
    /// Useful for beam extraction.
    #[must_use]
    pub fn value(&self, class_id: EClassId, node_idx: usize) -> Option<f32> {
        self.cache.get(&(class_id, node_idx)).map(|m| m.value_pred)
    }

    /// Get the expression embedding for an e-node.
    #[must_use]
    pub fn expr_embed(&self, class_id: EClassId, node_idx: usize) -> Option<&[f32; EMBED_DIM]> {
        self.cache.get(&(class_id, node_idx)).map(|m| &m.expr_embed)
    }

    /// Get the mask features for an e-node.
    #[must_use]
    pub fn mask_features(&self, class_id: EClassId, node_idx: usize) -> Option<&[f32; EMBED_DIM]> {
        self.cache.get(&(class_id, node_idx)).map(|m| &m.mask_features)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_basic() {
        let model = Arc::new(ExprNnue::new());
        let cache = NnueCache::new(model);

        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }
}
