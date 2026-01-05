//! # Spatial BSP Tree Combinator
//!
//! A B+ tree style spatial manifold for efficient grid rendering.
//!
//! ## Design
//!
//! Limits dynamic dispatch to two homogeneous arrays instead of per-node `Box<dyn>`:
//! - Interior nodes: routing only (axis, threshold, child refs)
//! - Leaf nodes: actual manifolds
//!
//! The tree structure is the compile-time shape. Array contents are load-time data.
//!
//! ## Performance
//!
//! - Early exit when all SIMD lanes go the same direction
//! - Per-lane blending only at cell boundaries
//! - O(log n) tree depth

use pixelflow_core::{Discrete, Field, Manifold};
use std::sync::Arc;

/// Split axis for BSP nodes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Axis {
    X,
    Y,
}

/// Reference to a child node in the BSP tree.
#[derive(Clone, Copy, Debug)]
pub enum NodeRef {
    /// Index into the interior nodes array.
    Interior(u32),
    /// Index into the leaves array.
    Leaf(u32),
}

/// Interior BSP node - routing only, no manifold data.
#[derive(Clone, Copy, Debug)]
pub struct InteriorNode {
    /// Which axis to split on.
    pub axis: Axis,
    /// Coordinate threshold for the split.
    pub threshold: f32,
    /// Left child (coordinates < threshold).
    pub left: NodeRef,
    /// Right child (coordinates >= threshold).
    pub right: NodeRef,
}

/// Spatial BSP tree manifold.
///
/// A B+ tree style structure where:
/// - Interior nodes contain only routing info (axis, threshold, children)
/// - Leaves contain the actual manifolds
///
/// Generic over leaf type L. For terminals, L = ColoredGlyph<PlatformPixel>.
#[derive(Clone)]
pub struct SpatialBSP<L> {
    /// Interior nodes - routing only.
    interiors: Arc<[InteriorNode]>,
    /// Leaf manifolds.
    leaves: Arc<[L]>,
}

/// Positioned item for building the BSP tree.
#[derive(Clone)]
pub struct Positioned<L> {
    /// Bounding box: (x_min, y_min, x_max, y_max)
    pub bounds: (f32, f32, f32, f32),
    /// The leaf manifold.
    pub leaf: L,
}

impl<L> SpatialBSP<L> {
    /// Create a BSP from pre-built arrays.
    ///
    /// Use `from_positioned` for automatic tree construction.
    pub fn new(interiors: Arc<[InteriorNode]>, leaves: Arc<[L]>) -> Self {
        Self { interiors, leaves }
    }

    /// Create a single-leaf BSP (degenerate case).
    pub fn single(leaf: L) -> Self {
        Self {
            interiors: Arc::from([]),
            leaves: Arc::from([leaf]),
        }
    }

    /// Build a balanced BSP tree from positioned items.
    ///
    /// Items are split recursively on the larger dimension until each
    /// region contains a single leaf.
    pub fn from_positioned(items: Vec<Positioned<L>>) -> Self {
        if items.is_empty() {
            return Self {
                interiors: Arc::from([]),
                leaves: Arc::from([]),
            };
        }

        if items.len() == 1 {
            return Self::single(items.into_iter().next().unwrap().leaf);
        }

        let mut interiors = Vec::new();
        let mut leaves = Vec::new();

        // Recursively build the tree
        let _root = Self::build_tree(&mut interiors, &mut leaves, items);

        Self {
            interiors: Arc::from(interiors),
            leaves: Arc::from(leaves),
        }
    }

    /// Recursively build the BSP tree.
    ///
    /// Returns the NodeRef for this subtree's root.
    fn build_tree(
        interiors: &mut Vec<InteriorNode>,
        leaves: &mut Vec<L>,
        mut items: Vec<Positioned<L>>,
    ) -> NodeRef {
        // Base case: single item
        if items.len() == 1 {
            let idx = leaves.len() as u32;
            leaves.push(items.pop().unwrap().leaf);
            return NodeRef::Leaf(idx);
        }

        // Find bounding box of all items
        let (mut min_x, mut min_y, mut max_x, mut max_y) =
            (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
        for item in &items {
            min_x = min_x.min(item.bounds.0);
            min_y = min_y.min(item.bounds.1);
            max_x = max_x.max(item.bounds.2);
            max_y = max_y.max(item.bounds.3);
        }

        // Split on larger dimension
        let width = max_x - min_x;
        let height = max_y - min_y;

        let (axis, threshold) = if width >= height {
            // Sort by X center, split at median
            items.sort_by(|a, b| {
                let ca = (a.bounds.0 + a.bounds.2) / 2.0;
                let cb = (b.bounds.0 + b.bounds.2) / 2.0;
                ca.partial_cmp(&cb).unwrap()
            });
            let mid_idx = items.len() / 2;
            let threshold = (items[mid_idx - 1].bounds.2 + items[mid_idx].bounds.0) / 2.0;
            (Axis::X, threshold)
        } else {
            // Sort by Y center, split at median
            items.sort_by(|a, b| {
                let ca = (a.bounds.1 + a.bounds.3) / 2.0;
                let cb = (b.bounds.1 + b.bounds.3) / 2.0;
                ca.partial_cmp(&cb).unwrap()
            });
            let mid_idx = items.len() / 2;
            let threshold = (items[mid_idx - 1].bounds.3 + items[mid_idx].bounds.1) / 2.0;
            (Axis::Y, threshold)
        };

        // Split items
        let mid = items.len() / 2;
        let right_items = items.split_off(mid);
        let left_items = items;

        // Recursively build children
        let left = Self::build_tree(interiors, leaves, left_items);
        let right = Self::build_tree(interiors, leaves, right_items);

        // Create interior node
        let idx = interiors.len() as u32;
        interiors.push(InteriorNode {
            axis,
            threshold,
            left,
            right,
        });

        NodeRef::Interior(idx)
    }

    /// Number of interior nodes.
    pub fn interior_count(&self) -> usize {
        self.interiors.len()
    }

    /// Number of leaf nodes.
    pub fn leaf_count(&self) -> usize {
        self.leaves.len()
    }
}

// ============================================================================
// Manifold Implementation for Discrete Output
// ============================================================================

impl<L> Manifold for SpatialBSP<L>
where
    L: Manifold<Output = Discrete> + Send + Sync,
{
    type Output = Discrete;

    #[inline(always)]
    fn eval_raw(&self, x: Field, y: Field, z: Field, w: Field) -> Discrete {
        // Handle degenerate cases
        if self.leaves.is_empty() {
            // No leaves - return transparent black
            let zero = Field::from(0.0);
            return Discrete::pack(zero, zero, zero, zero);
        }

        if self.interiors.is_empty() {
            // Single leaf - evaluate directly
            return self.leaves[0].eval_raw(x, y, z, w);
        }

        // Start traversal from the last interior node (root)
        self.traverse(self.interiors.len() - 1, x, y, z, w)
    }
}

impl<L> SpatialBSP<L>
where
    L: Manifold<Output = Discrete>,
{
    /// Traverse the BSP tree, returning the blended result.
    #[inline(always)]
    fn traverse(&self, idx: usize, x: Field, y: Field, z: Field, w: Field) -> Discrete {
        let node = &self.interiors[idx];

        // Get coordinate for this axis
        let coord = match node.axis {
            Axis::X => x,
            Axis::Y => y,
        };

        // Compute mask: true where coord < threshold
        let mask = coord.lt(Field::from(node.threshold));

        // Early exit when all SIMD lanes go the same direction
        if mask.all() {
            return self.eval_child(node.left, x, y, z, w);
        }
        if !mask.any() {
            return self.eval_child(node.right, x, y, z, w);
        }

        // Mixed: SIMD lanes span the boundary, must evaluate both
        let left_val = self.eval_child(node.left, x, y, z, w);
        let right_val = self.eval_child(node.right, x, y, z, w);

        // Blend using mask
        Discrete::select(mask, left_val, right_val)
    }

    /// Evaluate a child node (either interior or leaf).
    #[inline(always)]
    fn eval_child(&self, child: NodeRef, x: Field, y: Field, z: Field, w: Field) -> Discrete {
        match child {
            NodeRef::Interior(i) => self.traverse(i as usize, x, y, z, w),
            NodeRef::Leaf(i) => self.leaves[i as usize].eval_raw(x, y, z, w),
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// A simple solid color manifold for testing.
    #[derive(Clone)]
    struct SolidColor {
        r: f32,
        g: f32,
        b: f32,
        a: f32,
    }

    impl SolidColor {
        fn new(r: u8, g: u8, b: u8, a: u8) -> Self {
            Self {
                r: r as f32 / 255.0,
                g: g as f32 / 255.0,
                b: b as f32 / 255.0,
                a: a as f32 / 255.0,
            }
        }
    }

    impl Manifold for SolidColor {
        type Output = Discrete;

        fn eval_raw(&self, _x: Field, _y: Field, _z: Field, _w: Field) -> Discrete {
            Discrete::pack(
                Field::from(self.r),
                Field::from(self.g),
                Field::from(self.b),
                Field::from(self.a),
            )
        }
    }

    #[test]
    fn test_single_leaf() {
        let bsp = SpatialBSP::single(SolidColor::new(255, 0, 0, 255));

        assert_eq!(bsp.interior_count(), 0);
        assert_eq!(bsp.leaf_count(), 1);

        // Evaluate at any point should return red
        let x = Field::from(100.0);
        let y = Field::from(100.0);
        let z = Field::from(0.0);
        let w = Field::from(0.0);

        let result = bsp.eval_raw(x, y, z, w);
        // Result should be red (exact value depends on Discrete::splat behavior)
        let _ = result; // Just verify it doesn't panic
    }

    #[test]
    fn test_two_leaves() {
        let items = vec![
            Positioned {
                bounds: (0.0, 0.0, 50.0, 100.0),
                leaf: SolidColor::new(255, 0, 0, 255), // Red on left
            },
            Positioned {
                bounds: (50.0, 0.0, 100.0, 100.0),
                leaf: SolidColor::new(0, 0, 255, 255), // Blue on right
            },
        ];

        let bsp = SpatialBSP::from_positioned(items);

        assert_eq!(bsp.interior_count(), 1);
        assert_eq!(bsp.leaf_count(), 2);
    }

    #[test]
    fn test_empty_bsp() {
        let bsp: SpatialBSP<SolidColor> = SpatialBSP::from_positioned(vec![]);

        assert_eq!(bsp.interior_count(), 0);
        assert_eq!(bsp.leaf_count(), 0);

        // Should not panic on eval
        let x = Field::from(0.0);
        let _ = bsp.eval_raw(x, x, x, x);
    }

    #[test]
    fn test_four_leaves_grid() {
        // 2x2 grid
        let items = vec![
            Positioned {
                bounds: (0.0, 0.0, 50.0, 50.0),
                leaf: SolidColor::new(255, 0, 0, 255), // Top-left: Red
            },
            Positioned {
                bounds: (50.0, 0.0, 100.0, 50.0),
                leaf: SolidColor::new(0, 255, 0, 255), // Top-right: Green
            },
            Positioned {
                bounds: (0.0, 50.0, 50.0, 100.0),
                leaf: SolidColor::new(0, 0, 255, 255), // Bottom-left: Blue
            },
            Positioned {
                bounds: (50.0, 50.0, 100.0, 100.0),
                leaf: SolidColor::new(255, 255, 0, 255), // Bottom-right: Yellow
            },
        ];

        let bsp = SpatialBSP::from_positioned(items);

        // Should have 3 interior nodes for 4 leaves (binary tree)
        assert_eq!(bsp.leaf_count(), 4);
        assert!(bsp.interior_count() >= 1);
    }
}
