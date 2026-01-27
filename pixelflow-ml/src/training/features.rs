//! Feature extraction from ExprTree for NNUE training.
//!
//! This module extracts HalfEP sparse features and dense ILP features
//! from expression trees for NNUE evaluation and training.
//!
//! # Design Principles
//!
//! - **Composable**: Small functions that do one thing well
//! - **Functional**: Pure functions, no hidden state
//! - **Reusable**: Same primitives work for tree features and search features

use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};

use pixelflow_search::egraph::{ExprTree, Leaf, BestFirstContext};
use pixelflow_nnue::{HalfEPFeature, OpType, DenseFeatures, Expr as NnueExpr};

/// Special feature indicating a back-reference to an already-seen subexpression.
///
/// This is CRITICAL for proper CSE awareness:
/// - `let x = expensive(); x + x` -> [expensive, Add, BackRef] (cheap!)
/// - `expensive() + expensive()` -> [expensive, Add, expensive] (expensive!)
pub const BACKREF_FEATURE: HalfEPFeature = HalfEPFeature {
    perspective_op: 15,  // Invalid OpType - signals special feature
    descendant_op: 15,
    depth: 0,
    path: 255,
};

/// Map ExprTree variant to OpType for feature encoding.
pub fn expr_tree_to_op_type(tree: &ExprTree) -> OpType {
    match tree {
        ExprTree::Leaf(Leaf::Var(_)) => OpType::Var,
        ExprTree::Leaf(Leaf::Const(_)) => OpType::Const,
        ExprTree::Op { op, .. } => {
            match op.name() {
                "add" => OpType::Add,
                "sub" => OpType::Sub,
                "mul" => OpType::Mul,
                "div" => OpType::Div,
                "neg" => OpType::Neg,
                "sqrt" => OpType::Sqrt,
                "rsqrt" => OpType::Rsqrt,
                "abs" => OpType::Abs,
                "min" => OpType::Min,
                "max" => OpType::Max,
                "mul_add" | "fma" => OpType::MulAdd,
                "recip" => OpType::Div,
                "floor" | "ceil" | "round" | "fract" => OpType::Abs,
                "sin" | "cos" | "tan" | "asin" | "acos" | "atan"
                | "exp" | "exp2" | "ln" | "log2" | "log10" => OpType::Sqrt,
                "atan2" | "pow" | "hypot" => OpType::MulRsqrt,
                _ => OpType::Add,
            }
        }
    }
}

/// Convert ExprTree to NnueExpr for dense feature extraction.
pub fn expr_tree_to_nnue_expr(tree: &ExprTree) -> NnueExpr {
    match tree {
        ExprTree::Leaf(Leaf::Var(i)) => NnueExpr::Var(*i),
        ExprTree::Leaf(Leaf::Const(c)) => NnueExpr::Const(*c),
        ExprTree::Op { op, children } => {
            let name = op.name();
            match children.len() {
                0 => NnueExpr::Const(0.0),
                1 => {
                    let a = Box::new(expr_tree_to_nnue_expr(&children[0]));
                    let op_type = match name {
                        "neg" => OpType::Neg,
                        "sqrt" => OpType::Sqrt,
                        "rsqrt" => OpType::Rsqrt,
                        "abs" => OpType::Abs,
                        "recip" => OpType::Rsqrt,
                        _ => OpType::Sqrt,
                    };
                    NnueExpr::Unary(op_type, a)
                }
                2 => {
                    let a = Box::new(expr_tree_to_nnue_expr(&children[0]));
                    let b = Box::new(expr_tree_to_nnue_expr(&children[1]));
                    let op_type = match name {
                        "add" => OpType::Add,
                        "sub" => OpType::Sub,
                        "mul" => OpType::Mul,
                        "div" => OpType::Div,
                        "min" => OpType::Min,
                        "max" => OpType::Max,
                        "atan2" | "pow" | "hypot" => OpType::MulRsqrt,
                        _ => OpType::Add,
                    };
                    NnueExpr::Binary(op_type, a, b)
                }
                _ => {
                    let a = Box::new(expr_tree_to_nnue_expr(&children[0]));
                    let b = Box::new(expr_tree_to_nnue_expr(&children[1]));
                    let c = Box::new(expr_tree_to_nnue_expr(&children[2]));
                    NnueExpr::Ternary(OpType::MulAdd, a, b, c)
                }
            }
        }
    }
}

/// Compute a structural hash for an ExprTree subtree.
pub fn structural_hash(tree: &ExprTree) -> u64 {
    use std::collections::hash_map::DefaultHasher;

    let mut hasher = DefaultHasher::new();

    match tree {
        ExprTree::Leaf(Leaf::Var(v)) => {
            0u8.hash(&mut hasher);
            v.hash(&mut hasher);
        }
        ExprTree::Leaf(Leaf::Const(c)) => {
            1u8.hash(&mut hasher);
            c.to_bits().hash(&mut hasher);
        }
        ExprTree::Op { op, children } => {
            2u8.hash(&mut hasher);
            op.name().hash(&mut hasher);
            children.len().hash(&mut hasher);
            for child in children {
                structural_hash(child).hash(&mut hasher);
            }
        }
    }
    hasher.finish()
}

/// Extract HalfEP features from an ExprTree with CSE awareness.
///
/// Uses a DAG-walk instead of tree-walk:
/// - First visit to a subtree: extract full features
/// - Subsequent visits to structurally-identical subtrees: emit BackRef token
pub fn extract_tree_features(tree: &ExprTree) -> Vec<HalfEPFeature> {
    let mut features = Vec::new();
    let mut visited = HashSet::new();
    extract_tree_features_dag(tree, &mut features, &mut visited);
    features
}

fn extract_tree_features_dag(
    tree: &ExprTree,
    features: &mut Vec<HalfEPFeature>,
    visited: &mut HashSet<u64>,
) {
    let hash = structural_hash(tree);

    if !visited.insert(hash) {
        features.push(BACKREF_FEATURE);
        return;
    }

    let root_op = expr_tree_to_op_type(tree);
    add_tree_descendant_features(tree, features, root_op.index() as u8, 0, 0);

    match tree {
        ExprTree::Leaf(_) => {}
        ExprTree::Op { children, .. } => {
            for child in children {
                extract_tree_features_dag(child, features, visited);
            }
        }
    }
}

fn add_tree_descendant_features(
    tree: &ExprTree,
    features: &mut Vec<HalfEPFeature>,
    perspective_op: u8,
    depth: u8,
    path: u8,
) {
    const MAX_DEPTH: u8 = 8;
    if depth >= MAX_DEPTH {
        return;
    }

    let descendant_op = expr_tree_to_op_type(tree).index() as u8;
    features.push(HalfEPFeature {
        perspective_op,
        descendant_op,
        depth,
        path,
    });

    match tree {
        ExprTree::Leaf(_) => {}
        ExprTree::Op { children, .. } => {
            match children.len() {
                1 => {
                    add_tree_descendant_features(&children[0], features, perspective_op, depth + 1, path << 1);
                }
                2 => {
                    add_tree_descendant_features(&children[0], features, perspective_op, depth + 1, path << 1);
                    add_tree_descendant_features(&children[1], features, perspective_op, depth + 1, (path << 1) | 1);
                }
                3 => {
                    add_tree_descendant_features(&children[0], features, perspective_op, depth + 1, path << 2);
                    add_tree_descendant_features(&children[1], features, perspective_op, depth + 1, (path << 2) | 1);
                    add_tree_descendant_features(&children[2], features, perspective_op, depth + 1, (path << 2) | 2);
                }
                _ => {
                    for (i, child) in children.iter().enumerate() {
                        let child_path = path.wrapping_shl(4) | (i as u8 & 0x0F);
                        add_tree_descendant_features(child, features, perspective_op, depth + 1, child_path);
                    }
                }
            }
        }
    }
}

// ============================================================================
// Search Context Features (for BestFirstPlanner)
// ============================================================================

/// Marker byte indicating this is a search context feature, not a tree feature.
const SEARCH_CONTEXT_MARKER: u8 = 128;

/// Create a search context feature with the given semantic slots.
///
/// Uses path=128+ to distinguish from tree features (path < 128).
/// This allows the NNUE to learn separate weights for structural vs temporal features.
#[inline]
fn search_feature(slot: u8, primary: u8, secondary: u8) -> HalfEPFeature {
    HalfEPFeature {
        perspective_op: primary,
        descendant_op: secondary,
        depth: slot,
        path: SEARCH_CONTEXT_MARKER,
    }
}

/// Bucketize a value into 0..max_bucket range using linear scaling.
#[inline]
fn bucket_linear(value: usize, divisor: usize, max_bucket: u8) -> u8 {
    (value / divisor.max(1)).min(max_bucket as usize) as u8
}

/// Bucketize a ratio (0.0 to 1.0) into 0..max_bucket range.
#[inline]
fn bucket_ratio(ratio: f64, max_bucket: u8) -> u8 {
    (ratio * max_bucket as f64).round().min(max_bucket as f64) as u8
}

/// Extract features from BestFirstPlanner search context.
///
/// Combines tree features with search progress features:
/// - Slot 0: Cost bucket (how expensive is this tree?)
/// - Slot 1: Depth bucket (how deep in search are we?)
/// - Slot 2: Progress bucket (what fraction of budget used?)
/// - Slot 3: Improvement bucket (how much have we improved?)
/// - Slot 4: Frontier pressure (is the queue crowded?)
///
/// The NNUE learns patterns like:
/// - "Prefer exploitation late in search (high progress)"
/// - "Explore more when improvement has stalled"
/// - "Be greedy when frontier is crowded"
pub fn extract_search_features(ctx: &BestFirstContext<'_>) -> Vec<HalfEPFeature> {
    // Start with tree structure features
    let mut features = extract_tree_features(ctx.tree);

    // Add search context features using dedicated slots
    features.extend([
        // Slot 0: Cost bucket - how expensive is this candidate?
        search_feature(0, bucket_linear(ctx.tree_cost, 10, 13), 0),

        // Slot 1: Depth bucket - how deep in the search tree?
        search_feature(1, bucket_linear(ctx.depth, 1, 13), 0),

        // Slot 2: Expansion progress - fraction of budget used
        // (Note: caller should pass max_expansions if available)
        search_feature(2, bucket_linear(ctx.expansions, 50, 13), 0),

        // Slot 3: Improvement ratio - how much better than initial?
        search_feature(3, bucket_ratio(ctx.improvement_ratio(), 13), 0),

        // Slot 4: Frontier size bucket - queue pressure
        search_feature(4, bucket_linear(ctx.frontier_size, 100, 13), 0),

        // Slot 5: Visited count bucket - exploration breadth
        search_feature(5, bucket_linear(ctx.visited_count, 100, 13), 0),

        // Slot 6: Cost vs best - is this candidate competitive?
        search_feature(6, bucket_linear(
            ctx.tree_cost.saturating_sub(ctx.best_cost),
            5,
            13
        ), 0),
    ]);

    features
}

/// Extract features with explicit max_expansions for accurate progress tracking.
pub fn extract_search_features_with_budget(
    ctx: &BestFirstContext<'_>,
    max_expansions: usize,
) -> Vec<HalfEPFeature> {
    let mut features = extract_tree_features(ctx.tree);

    let progress = ctx.search_progress(max_expansions);

    features.extend([
        search_feature(0, bucket_linear(ctx.tree_cost, 10, 13), 0),
        search_feature(1, bucket_linear(ctx.depth, 1, 13), 0),
        search_feature(2, bucket_ratio(progress, 13), 0),
        search_feature(3, bucket_ratio(ctx.improvement_ratio(), 13), 0),
        search_feature(4, bucket_linear(ctx.frontier_size, 100, 13), 0),
        search_feature(5, bucket_linear(ctx.visited_count, 100, 13), 0),
        search_feature(6, bucket_linear(
            ctx.tree_cost.saturating_sub(ctx.best_cost),
            5,
            13
        ), 0),
    ]);

    features
}

/// Convert op_counts from benchmark data to DenseFeatures.
pub fn op_counts_to_dense(op_counts: &HashMap<String, usize>, node_count: usize) -> DenseFeatures {
    let mut dense = DenseFeatures::default();

    dense.values[0] = *op_counts.get("add").unwrap_or(&0) as i32;
    dense.values[1] = *op_counts.get("sub").unwrap_or(&0) as i32;
    dense.values[2] = *op_counts.get("mul").unwrap_or(&0) as i32;
    dense.values[3] = *op_counts.get("div").unwrap_or(&0) as i32;
    dense.values[4] = *op_counts.get("neg").unwrap_or(&0) as i32;
    dense.values[5] = *op_counts.get("sqrt").unwrap_or(&0) as i32;
    dense.values[6] = *op_counts.get("rsqrt").unwrap_or(&0) as i32;
    dense.values[7] = *op_counts.get("abs").unwrap_or(&0) as i32;
    dense.values[8] = *op_counts.get("min").unwrap_or(&0) as i32;
    dense.values[9] = *op_counts.get("max").unwrap_or(&0) as i32;
    dense.values[10] = *op_counts.get("mul_add").unwrap_or(&0) as i32;
    dense.values[11] = *op_counts.get("recip").unwrap_or(&0) as i32;

    dense.values[12] = node_count as i32;
    dense.values[13] = (node_count / 2).max(1) as i32;
    dense.values[14] = 4;

    dense
}

/// Extract dense features from an ExprTree.
pub fn extract_dense_from_tree(tree: &ExprTree) -> DenseFeatures {
    let nnue_expr = expr_tree_to_nnue_expr(tree);
    pixelflow_nnue::extract_dense_features(&nnue_expr)
}
