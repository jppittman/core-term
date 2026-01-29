//! Feature extraction for NNUE.
//!
//! Defines `HalfEPFeature` (sparse) and `DenseFeatures` (ILP-aware).
//! Includes DAG-aware feature extraction to handle common subexpressions.

use crate::kind::OpKind;
use crate::expr::Expr;
use alloc::vec::Vec;
use core::hash::{Hash, Hasher};
use libm::fabsf;

/// Maximum depth we encode in features.
pub const MAX_DEPTH: usize = 8;

/// Special feature indicating a back-reference to an already-seen subexpression.
pub const BACKREF_FEATURE: HalfEPFeature = HalfEPFeature {
    perspective_op: 15,
    descendant_op: 15,
    depth: 0,
    path: 255,
};

/// A HalfEP feature: (perspective_op, descendant_op, depth, child_path).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct HalfEPFeature {
    pub perspective_op: u8,
    pub descendant_op: u8,
    pub depth: u8,
    pub path: u8,
}

impl HalfEPFeature {
    pub const COUNT: usize = OpKind::COUNT * OpKind::COUNT * MAX_DEPTH * 256;

    pub fn to_index(self) -> usize {
        let p = self.perspective_op as usize;
        let d = self.descendant_op as usize;
        let depth = self.depth as usize;
        let path = self.path as usize;
        ((p * OpKind::COUNT + d) * MAX_DEPTH + depth) * 256 + path
    }
}

/// Simple hasher for no_std environments (FNV-1a like).
struct SimpleHasher(u64);

impl SimpleHasher {
    fn new() -> Self {
        Self(0xcbf29ce484222325)
    }
}

impl Hasher for SimpleHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 ^= b as u64;
            self.0 = self.0.wrapping_mul(0x100000001b3);
        }
    }
}

/// Compute a structural hash for an Expr subtree.
pub fn structural_hash(tree: &Expr) -> u64 {
    let mut hasher = SimpleHasher::new();
    match tree {
        Expr::Var(v) => {
            0u8.hash(&mut hasher);
            v.hash(&mut hasher);
        }
        Expr::Const(c) => {
            1u8.hash(&mut hasher);
            c.to_bits().hash(&mut hasher);
        }
        Expr::Unary(op, a) => {
            2u8.hash(&mut hasher);
            op.name().hash(&mut hasher);
            1usize.hash(&mut hasher);
            structural_hash(a).hash(&mut hasher);
        }
        Expr::Binary(op, a, b) => {
            2u8.hash(&mut hasher);
            op.name().hash(&mut hasher);
            2usize.hash(&mut hasher);
            structural_hash(a).hash(&mut hasher);
            structural_hash(b).hash(&mut hasher);
        }
        Expr::Ternary(op, a, b, c) => {
            2u8.hash(&mut hasher);
            op.name().hash(&mut hasher);
            3usize.hash(&mut hasher);
            structural_hash(a).hash(&mut hasher);
            structural_hash(b).hash(&mut hasher);
            structural_hash(c).hash(&mut hasher);
        }
        Expr::Nary(op, children) => {
            2u8.hash(&mut hasher);
            op.name().hash(&mut hasher);
            children.len().hash(&mut hasher);
            for c in children {
                structural_hash(c).hash(&mut hasher);
            }
        }
    }
    hasher.finish()
}

/// Extract HalfEP features with CSE awareness (DAG walk).
pub fn extract_features(expr: &Expr) -> Vec<HalfEPFeature> {
    let mut features = Vec::new();
    // Simple visited set (linear scan for no_std, O(N^2) but N is small < 100)
    let mut visited = Vec::new();
    extract_features_dag(expr, &mut features, &mut visited);
    features
}

fn extract_features_dag(
    tree: &Expr,
    features: &mut Vec<HalfEPFeature>,
    visited: &mut Vec<u64>,
) {
    let hash = structural_hash(tree);
    if visited.contains(&hash) {
        features.push(BACKREF_FEATURE);
        return;
    }
    visited.push(hash);

    let root_op = tree.kind();
    add_descendant_features(tree, features, root_op as u8, 0, 0);

    match tree {
        Expr::Var(_) | Expr::Const(_) => {}
        Expr::Unary(_, a) => extract_features_dag(a, features, visited),
        Expr::Binary(_, a, b) => {
            extract_features_dag(a, features, visited);
            extract_features_dag(b, features, visited);
        }
        Expr::Ternary(_, a, b, c) => {
            extract_features_dag(a, features, visited);
            extract_features_dag(b, features, visited);
            extract_features_dag(c, features, visited);
        }
        Expr::Nary(_, children) => {
            for c in children {
                extract_features_dag(c, features, visited);
            }
        }
    }
}

fn add_descendant_features(
    expr: &Expr,
    features: &mut Vec<HalfEPFeature>,
    perspective_op: u8,
    depth: u8,
    path: u8,
) {
    if depth as usize >= MAX_DEPTH {
        return;
    }

    features.push(HalfEPFeature {
        perspective_op,
        descendant_op: expr.kind() as u8,
        depth,
        path,
    });

    match expr {
        Expr::Var(_) | Expr::Const(_) => {}
        Expr::Unary(_, a) => {
            add_descendant_features(a, features, perspective_op, depth + 1, path << 1);
        }
        Expr::Binary(_, a, b) => {
            add_descendant_features(a, features, perspective_op, depth + 1, path << 1);
            add_descendant_features(b, features, perspective_op, depth + 1, (path << 1) | 1);
        }
        Expr::Ternary(_, a, b, c) => {
            add_descendant_features(a, features, perspective_op, depth + 1, path << 2);
            add_descendant_features(b, features, perspective_op, depth + 1, (path << 2) | 1);
            add_descendant_features(c, features, perspective_op, depth + 1, (path << 2) | 2);
        }
        Expr::Nary(_, children) => {
            // Nary path logic? Fallback to sequential index bits?
            // For now, treat as opaque path 0 for children > 2 to stay consistent with ternary
            for (i, c) in children.iter().enumerate() {
                // If we assume max N is small, we could encode.
                // Just use 0 for now.
                add_descendant_features(c, features, perspective_op, depth + 1, path);
            }
        }
    }
}

// ============================================================================
// Dense Features
// ============================================================================

/// Dense features extracted from an expression for ILP-aware evaluation.
#[derive(Clone, Debug, Default)]
pub struct DenseFeatures {
    /// Dense feature values (21 features).
    pub values: [i32; Self::COUNT],
}

impl DenseFeatures {
    pub const COUNT: usize = 21;

    pub const ADD: usize = 0;
    pub const SUB: usize = 1;
    pub const MUL: usize = 2;
    pub const DIV: usize = 3;
    pub const NEG: usize = 4;
    pub const SQRT: usize = 5;
    pub const RSQRT: usize = 6;
    pub const ABS: usize = 7;
    pub const MIN: usize = 8;
    pub const MAX: usize = 9;
    pub const FMA: usize = 10;
    pub const MUL_RSQRT: usize = 11;
    pub const NODE_COUNT: usize = 12;
    pub const DEPTH: usize = 13;
    pub const VAR_COUNT: usize = 14;
    pub const CONST_COUNT: usize = 15;
    pub const HAS_IDENTITY: usize = 16;
    pub const HAS_SELF_CANCEL: usize = 17;
    pub const HAS_FUSABLE: usize = 18;
    pub const CRITICAL_PATH: usize = 19;
    pub const MAX_WIDTH: usize = 20;

    #[inline]
    pub fn get(&self, i: usize) -> i32 {
        self.values.get(i).copied().unwrap_or(0)
    }

    #[inline]
    pub fn set(&mut self, i: usize, v: i32) {
        if i < Self::COUNT {
            self.values[i] = v;
        }
    }
}

/// Extract dense features from an expression (ILP-aware).
pub fn extract_dense_features(expr: &Expr) -> DenseFeatures {
    let mut features = DenseFeatures::default();
    let mut width_at_depth = Vec::new();
    let critical_path = extract_dense_recursive_impl(expr, &mut features, 0, &mut width_at_depth);
    features.set(DenseFeatures::CRITICAL_PATH, critical_path);
    features.set(DenseFeatures::MAX_WIDTH, width_at_depth.iter().copied().max().unwrap_or(0));
    features
}

fn extract_dense_recursive_impl(
    expr: &Expr,
    features: &mut DenseFeatures,
    depth: usize,
    width_at_depth: &mut Vec<i32>,
) -> i32 {
    features.values[DenseFeatures::NODE_COUNT] += 1;
    let current_depth = features.values[DenseFeatures::DEPTH];
    features.values[DenseFeatures::DEPTH] = current_depth.max(depth as i32 + 1);

    if depth >= width_at_depth.len() {
        width_at_depth.resize(depth + 1, 0);
    }
    width_at_depth[depth] += 1;

    match expr {
        Expr::Var(_) => {
            features.values[DenseFeatures::VAR_COUNT] += 1;
            0
        }
        Expr::Const(_) => {
            features.values[DenseFeatures::CONST_COUNT] += 1;
            0
        }
        Expr::Unary(op, a) => {
            let op_cost = match op {
                OpKind::Neg => { features.values[DenseFeatures::NEG] += 1; 1 }
                OpKind::Sqrt => { features.values[DenseFeatures::SQRT] += 1; 15 }
                OpKind::Rsqrt => { features.values[DenseFeatures::RSQRT] += 1; 5 }
                OpKind::Abs => { features.values[DenseFeatures::ABS] += 1; 1 }
                _ => 5,
            };
            let child_critical = extract_dense_recursive_impl(a, features, depth + 1, width_at_depth);
            op_cost + child_critical
        }
        Expr::Binary(op, a, b) => {
            let op_cost = match op {
                OpKind::Add => {
                    features.values[DenseFeatures::ADD] += 1;
                    if matches!(a.kind(), OpKind::Mul) {
                        features.values[DenseFeatures::HAS_FUSABLE] += 1;
                    }
                    if is_const_zero(b) || is_const_zero(a) {
                        features.values[DenseFeatures::HAS_IDENTITY] += 1;
                    }
                    4
                }
                OpKind::Sub => {
                    features.values[DenseFeatures::SUB] += 1;
                    4
                }
                OpKind::Mul => {
                    features.values[DenseFeatures::MUL] += 1;
                    if is_const_one(b) || is_const_one(a) {
                        features.values[DenseFeatures::HAS_IDENTITY] += 1;
                    }
                    if matches!(b.kind(), OpKind::Rsqrt) || matches!(a.kind(), OpKind::Rsqrt) {
                        features.values[DenseFeatures::HAS_FUSABLE] += 1;
                    }
                    5
                }
                OpKind::Div => {
                    features.values[DenseFeatures::DIV] += 1;
                    15
                }
                OpKind::Min => { features.values[DenseFeatures::MIN] += 1; 4 }
                OpKind::Max => { features.values[DenseFeatures::MAX] += 1; 4 }
                OpKind::MulRsqrt => { features.values[DenseFeatures::MUL_RSQRT] += 1; 6 }
                _ => 5,
            };
            let crit_a = extract_dense_recursive_impl(a, features, depth + 1, width_at_depth);
            let crit_b = extract_dense_recursive_impl(b, features, depth + 1, width_at_depth);
            op_cost + crit_a.max(crit_b)
        }
        Expr::Ternary(op, a, b, c) => {
            let op_cost = match op {
                OpKind::MulAdd => { features.values[DenseFeatures::FMA] += 1; 5 }
                _ => 10,
            };
            let crit_a = extract_dense_recursive_impl(a, features, depth + 1, width_at_depth);
            let crit_b = extract_dense_recursive_impl(b, features, depth + 1, width_at_depth);
            let crit_c = extract_dense_recursive_impl(c, features, depth + 1, width_at_depth);
            op_cost + crit_a.max(crit_b).max(crit_c)
        }
        Expr::Nary(_, children) => {
            let mut max_crit = 0;
            for c in children {
                let crit = extract_dense_recursive_impl(c, features, depth + 1, width_at_depth);
                max_crit = max_crit.max(crit);
            }
            max_crit
        }
    }
}

fn is_const_zero(expr: &Expr) -> bool {
    match expr {
        Expr::Const(c) => fabsf(*c) < 1e-10,
        _ => false,
    }
}

fn is_const_one(expr: &Expr) -> bool {
    match expr {
        Expr::Const(c) => fabsf(*c - 1.0) < 1e-10,
        _ => false,
    }
}
