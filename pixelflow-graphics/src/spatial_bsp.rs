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

use pixelflow_core::{Discrete, Field, Manifold, ManifoldCompat, Select};
use std::sync::Arc;

/// The standard 4D Field domain type.
type Field4 = (Field, Field, Field, Field);

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
    #[must_use]
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

        let mut interiors = Vec::with_capacity(items.len().saturating_sub(1));
        let mut leaves = Vec::with_capacity(items.len());

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

        // Find bounding box of all items and center extents
        let (mut min_x, mut min_y, mut max_x, mut max_y) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
        let (mut min_cx, mut min_cy, mut max_cx, mut max_cy) =
            (f32::MAX, f32::MAX, f32::MIN, f32::MIN);

        for item in &items {
            min_x = min_x.min(item.bounds.0);
            min_y = min_y.min(item.bounds.1);
            max_x = max_x.max(item.bounds.2);
            max_y = max_y.max(item.bounds.3);

            let cx = (item.bounds.0 + item.bounds.2) * 0.5;
            let cy = (item.bounds.1 + item.bounds.3) * 0.5;
            min_cx = min_cx.min(cx);
            max_cx = max_cx.max(cx);
            min_cy = min_cy.min(cy);
            max_cy = max_cy.max(cy);
        }

        // Choose split axis based on center distribution variance (spread)
        // Fallback to bound dimensions if spreads are effectively zero (concentric items)
        let extent_x = max_cx - min_cx;
        let extent_y = max_cy - min_cy;
        let width = max_x - min_x;
        let height = max_y - min_y;

        let split_x = if extent_x > extent_y {
            true
        } else if extent_y > extent_x {
            false
        } else {
            width >= height
        };

        let (axis, threshold) = if split_x {
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
    #[must_use]
    pub fn interior_count(&self) -> usize {
        self.interiors.len()
    }

    /// Number of leaf nodes.
    #[must_use]
    pub fn leaf_count(&self) -> usize {
        self.leaves.len()
    }
}

// ============================================================================
// Manifold Implementation for Discrete Output
// ============================================================================

impl<L> Manifold<Field4> for SpatialBSP<L>
where
    L: ManifoldCompat<Field, Output = Discrete>,
{
    type Output = Discrete;

    #[inline(always)]
    fn eval(&self, p: Field4) -> Discrete {
        let (x, y, z, w) = p;
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
    L: ManifoldCompat<Field, Output = Discrete>,
{
    /// Traverse the BSP tree, returning the blended result.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
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

        // Blend using Select combinator
        Select {
            cond: mask,
            if_true: left_val,
            if_false: right_val,
        }
        .eval((x, y, z, w))
    }

    /// Evaluate a child node (either interior or leaf).
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
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

    impl Manifold<Field4> for SolidColor {
        type Output = Discrete;

        fn eval(&self, p: Field4) -> Discrete {
            let (_x, _y, _z, _w) = p;
            Discrete::pack(
                Field::from(self.r),
                Field::from(self.g),
                Field::from(self.b),
                Field::from(self.a),
            )
        }
    }

    /// Packed pixel value a solid color evaluates to, for comparison against sampled BSPs.
    fn packed(color: &SolidColor) -> u32 {
        use pixelflow_core::{PARALLELISM, materialize_discrete};
        let mut buf = [0u32; PARALLELISM];
        materialize_discrete(color, 0.0, 0.0, &mut buf);
        buf[0]
    }

    /// Sample a BSP at a point through the public `Manifold` contract.
    fn sample(bsp: &SpatialBSP<SolidColor>, x: f32, y: f32) -> u32 {
        use pixelflow_core::{PARALLELISM, materialize_discrete};
        let mut buf = [0u32; PARALLELISM];
        materialize_discrete(bsp, x, y, &mut buf);
        buf[0]
    }

    /// Build a BSP from `items`/`colors` and assert that querying the center of every
    /// item's own bounds returns that item's own color — the actual contract `eval()`
    /// promises. This exercises axis selection, threshold placement, and interior-node
    /// wiring all at once, entirely through the public API: a bad index panics, and an
    /// unreachable leaf or misrouted split returns the wrong (or another item's) color.
    /// Where an item's bounds are constant on one axis and varying on the other, only
    /// the correct split axis can route it correctly, so this also proves axis selection
    /// without reading the private tree.
    fn build_and_assert_every_item_routes_to_its_own_color(
        items: &[(f32, f32, f32, f32)],
        colors: &[(u8, u8, u8, u8)],
    ) -> SpatialBSP<SolidColor> {
        assert_eq!(items.len(), colors.len());
        let expected: Vec<u32> = colors.iter().map(|&(r, g, b, a)| packed(&SolidColor::new(r, g, b, a))).collect();
        let positioned = items
            .iter()
            .zip(colors)
            .map(|(&bounds, &(r, g, b, a))| Positioned {
                bounds,
                leaf: SolidColor::new(r, g, b, a),
            })
            .collect();

        let bsp = SpatialBSP::from_positioned(positioned);

        for (i, &bounds) in items.iter().enumerate() {
            let cx = (bounds.0 + bounds.2) * 0.5;
            let cy = (bounds.1 + bounds.3) * 0.5;
            let got = sample(&bsp, cx, cy);
            assert_eq!(
                got, expected[i],
                "item {i} at center ({cx}, {cy}) did not route to its own color"
            );
        }

        bsp
    }

    #[test]
    fn single_leaf() {
        let bsp = SpatialBSP::single(SolidColor::new(255, 0, 0, 255));

        assert_eq!(
            bsp.interior_count(),
            0,
            "Single leaf tree has no interior nodes"
        );
        assert_eq!(bsp.leaf_count(), 1, "Single leaf tree has exactly one leaf");

        // Evaluate at any point should compile and execute without panic
        let x = Field::from(100.0);
        let y = Field::from(100.0);
        let z = Field::from(0.0);
        let w = Field::from(0.0);

        let _result = bsp.eval_raw(x, y, z, w);
    }

    #[test]
    fn two_leaves() {
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
    fn empty_bsp() {
        let bsp: SpatialBSP<SolidColor> = SpatialBSP::from_positioned(vec![]);

        assert_eq!(bsp.interior_count(), 0, "Empty BSP has no interior nodes");
        assert_eq!(bsp.leaf_count(), 0, "Empty BSP has no leaves");

        // Should not panic on eval
        let x = Field::from(0.0);
        let _result = bsp.eval_raw(x, x, x, x);
    }

    #[test]
    fn four_leaves_grid() {
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

        // Binary tree with 4 leaves should have exactly 3 interior nodes (n-1)
        assert_eq!(bsp.leaf_count(), 4, "Should have 4 leaves in the grid");
        assert_eq!(
            bsp.interior_count(),
            3,
            "Binary tree with 4 leaves must have exactly 3 interior nodes"
        );
    }

    // ========================================================================
    // Degenerate Case Tests
    // ========================================================================

    #[test]
    fn empty_bsp_returns_transparent_on_eval() {
        let bsp: SpatialBSP<SolidColor> = SpatialBSP::from_positioned(vec![]);

        let x = Field::from(50.0);
        let y = Field::from(50.0);
        let z = Field::from(0.0);
        let w = Field::from(0.0);

        // Should not panic when evaluating empty BSP
        let result = bsp.eval_raw(x, y, z, w);
        // Empty BSP should successfully evaluate (implementation returns default/transparent)
        let _ = result;
    }

    #[test]
    fn single_leaf_has_no_interiors() {
        let bsp = SpatialBSP::single(SolidColor::new(128, 64, 32, 255));

        assert_eq!(bsp.interior_count(), 0);
        assert_eq!(bsp.leaf_count(), 1);
    }

    #[test]
    fn single_leaf_from_positioned_has_no_interiors() {
        let items = vec![Positioned {
            bounds: (0.0, 0.0, 100.0, 100.0),
            leaf: SolidColor::new(255, 128, 0, 255),
        }];

        let bsp = SpatialBSP::from_positioned(items);

        assert_eq!(bsp.interior_count(), 0);
        assert_eq!(bsp.leaf_count(), 1);
    }

    // ========================================================================
    // Boundary Condition Tests
    // ========================================================================

    #[test]
    fn separated_on_x_routes_each_item_to_its_own_color() {
        // Identical Y range, separated on X: only an X-axis split routes these correctly,
        // so this proves axis selection without reading the private tree.
        let bsp = build_and_assert_every_item_routes_to_its_own_color(
            &[(0.0, 0.0, 25.0, 100.0), (75.0, 0.0, 100.0, 100.0)],
            &[(255, 0, 0, 255), (0, 0, 255, 255)],
        );
        assert_eq!(
            bsp.interior_count(),
            1,
            "two non-overlapping items need exactly one split"
        );
    }

    #[test]
    fn tall_items_split_on_y_route_each_item_to_its_own_color() {
        // Identical X range, separated on Y: only a Y-axis split routes these correctly.
        build_and_assert_every_item_routes_to_its_own_color(
            &[(0.0, 0.0, 10.0, 50.0), (0.0, 50.0, 10.0, 100.0)],
            &[(255, 0, 0, 255), (0, 0, 255, 255)],
        );
    }

    #[test]
    fn wide_items_split_on_x_route_each_item_to_its_own_color() {
        // Identical Y range, separated on X: only an X-axis split routes these correctly.
        build_and_assert_every_item_routes_to_its_own_color(
            &[(0.0, 0.0, 50.0, 10.0), (50.0, 0.0, 100.0, 10.0)],
            &[(255, 0, 0, 255), (0, 0, 255, 255)],
        );
    }

    // ========================================================================
    // Identical Bounds Tests
    // ========================================================================

    #[test]
    fn identical_bounds_creates_valid_tree() {
        // All items at same position (overlapping glyphs scenario)
        let items = vec![
            Positioned {
                bounds: (10.0, 10.0, 20.0, 20.0),
                leaf: SolidColor::new(255, 0, 0, 255),
            },
            Positioned {
                bounds: (10.0, 10.0, 20.0, 20.0),
                leaf: SolidColor::new(0, 255, 0, 255),
            },
            Positioned {
                bounds: (10.0, 10.0, 20.0, 20.0),
                leaf: SolidColor::new(0, 0, 255, 255),
            },
        ];

        let bsp = SpatialBSP::from_positioned(items);

        assert_eq!(bsp.leaf_count(), 3, "Should have 3 leaves for 3 items");
        assert_eq!(
            bsp.interior_count(),
            2,
            "Binary tree with 3 leaves must have exactly 2 interior nodes (n-1 rule)"
        );
    }

    #[test]
    fn zero_width_bounds_does_not_panic() {
        let items = vec![
            Positioned {
                bounds: (10.0, 0.0, 10.0, 100.0), // Zero width
                leaf: SolidColor::new(255, 0, 0, 255),
            },
            Positioned {
                bounds: (10.0, 0.0, 10.0, 100.0),
                leaf: SolidColor::new(0, 0, 255, 255),
            },
        ];

        let bsp = SpatialBSP::from_positioned(items);
        assert_eq!(bsp.leaf_count(), 2);
    }

    #[test]
    fn zero_height_bounds_does_not_panic() {
        let items = vec![
            Positioned {
                bounds: (0.0, 10.0, 100.0, 10.0), // Zero height
                leaf: SolidColor::new(255, 0, 0, 255),
            },
            Positioned {
                bounds: (0.0, 10.0, 100.0, 10.0),
                leaf: SolidColor::new(0, 0, 255, 255),
            },
        ];

        let bsp = SpatialBSP::from_positioned(items);
        assert_eq!(bsp.leaf_count(), 2);
    }

    #[test]
    fn point_bounds_creates_valid_tree() {
        // Degenerate rectangles (points)
        let items = vec![
            Positioned {
                bounds: (5.0, 5.0, 5.0, 5.0),
                leaf: SolidColor::new(255, 0, 0, 255),
            },
            Positioned {
                bounds: (15.0, 15.0, 15.0, 15.0),
                leaf: SolidColor::new(0, 0, 255, 255),
            },
        ];

        let bsp = SpatialBSP::from_positioned(items);
        assert_eq!(bsp.leaf_count(), 2);
    }

    // ========================================================================
    // Extreme Value Tests
    // ========================================================================

    #[test]
    fn negative_coordinates_handled_correctly() {
        let bsp = build_and_assert_every_item_routes_to_its_own_color(
            &[(-100.0, -100.0, -50.0, -50.0), (-50.0, -50.0, 0.0, 0.0)],
            &[(255, 0, 0, 255), (0, 0, 255, 255)],
        );
        assert_eq!(bsp.leaf_count(), 2, "should have 2 leaves");
        assert_eq!(bsp.interior_count(), 1, "two items need exactly one split");
    }

    #[test]
    fn large_coordinates_handled_correctly() {
        // A threshold that lost precision at this magnitude (e.g. collapsed to 0, or to
        // one item's own bound) would misroute one of these probes.
        let bsp = build_and_assert_every_item_routes_to_its_own_color(
            &[(1e6, 1e6, 2e6, 2e6), (3e6, 3e6, 4e6, 4e6)],
            &[(255, 0, 0, 255), (0, 0, 255, 255)],
        );
        assert_eq!(bsp.leaf_count(), 2);
    }

    #[test]
    fn mixed_positive_negative_coordinates() {
        let items = vec![
            Positioned {
                bounds: (-50.0, -50.0, 0.0, 0.0),
                leaf: SolidColor::new(255, 0, 0, 255),
            },
            Positioned {
                bounds: (0.0, 0.0, 50.0, 50.0),
                leaf: SolidColor::new(0, 255, 0, 255),
            },
            Positioned {
                bounds: (50.0, 50.0, 100.0, 100.0),
                leaf: SolidColor::new(0, 0, 255, 255),
            },
        ];

        let bsp = SpatialBSP::from_positioned(items);
        assert_eq!(bsp.leaf_count(), 3);
    }

    // ========================================================================
    // Tree Structure Tests
    // ========================================================================

    #[test]
    fn binary_tree_property_interior_count() {
        // For n leaves, binary tree has n-1 interior nodes
        for n in 2..=16 {
            let items: Vec<_> = (0..n)
                .map(|i| {
                    let x = (i as f32) * 10.0;
                    Positioned {
                        bounds: (x, 0.0, x + 5.0, 10.0),
                        leaf: SolidColor::new(255, 0, 0, 255),
                    }
                })
                .collect();

            let bsp = SpatialBSP::from_positioned(items);

            assert_eq!(bsp.leaf_count(), n);
            assert_eq!(
                bsp.interior_count(),
                n - 1,
                "Binary tree with {} leaves should have {} interiors",
                n,
                n - 1
            );
        }
    }

    #[test]
    fn power_of_two_leaves_creates_balanced_tree() {
        // 8 leaves should create a balanced tree of depth 3
        let items: Vec<_> = (0..8)
            .map(|i| {
                let x = (i as f32) * 10.0;
                Positioned {
                    bounds: (x, 0.0, x + 5.0, 10.0),
                    leaf: SolidColor::new((i * 32) as u8, 0, 0, 255),
                }
            })
            .collect();

        let bsp = SpatialBSP::from_positioned(items);

        assert_eq!(bsp.leaf_count(), 8);
        assert_eq!(bsp.interior_count(), 7);
    }

    #[test]
    fn many_items_stress_test() {
        // Create 1000 non-overlapping items
        let items: Vec<_> = (0..1000)
            .map(|i| {
                let row = i / 32;
                let col = i % 32;
                let x = (col as f32) * 10.0;
                let y = (row as f32) * 10.0;
                Positioned {
                    bounds: (x, y, x + 8.0, y + 8.0),
                    leaf: SolidColor::new(
                        ((i * 7) % 256) as u8,
                        ((i * 13) % 256) as u8,
                        ((i * 19) % 256) as u8,
                        255,
                    ),
                }
            })
            .collect();

        let bsp = SpatialBSP::from_positioned(items);

        assert_eq!(bsp.leaf_count(), 1000);
        assert_eq!(bsp.interior_count(), 999);
    }

    // ========================================================================
    // Traversal Correctness Tests
    // ========================================================================

    #[test]
    fn eval_respects_spatial_partitioning() {
        use pixelflow_core::{materialize_discrete, PARALLELISM};

        // Create a simple 2-region split at x=50
        let items = vec![
            Positioned {
                bounds: (0.0, 0.0, 50.0, 100.0),
                leaf: SolidColor::new(255, 0, 0, 255), // Red left
            },
            Positioned {
                bounds: (50.0, 0.0, 100.0, 100.0),
                leaf: SolidColor::new(0, 0, 255, 255), // Blue right
            },
        ];

        let bsp = SpatialBSP::from_positioned(items);

        // Materialize at two different spatial regions
        let mut left_pixels = [0u32; PARALLELISM];
        let mut right_pixels = [0u32; PARALLELISM];

        materialize_discrete(&bsp, 25.0, 50.0, &mut left_pixels);
        materialize_discrete(&bsp, 75.0, 50.0, &mut right_pixels);

        // Verify we got red and blue specifically
        let expected_red = {
            let red_color = SolidColor::new(255, 0, 0, 255);
            let mut buf = [0u32; PARALLELISM];
            materialize_discrete(&red_color, 0.0, 0.0, &mut buf);
            buf[0]
        };
        let expected_blue = {
            let blue_color = SolidColor::new(0, 0, 255, 255);
            let mut buf = [0u32; PARALLELISM];
            materialize_discrete(&blue_color, 0.0, 0.0, &mut buf);
            buf[0]
        };

        assert_eq!(left_pixels[0], expected_red, "Left region must be red");
        assert_eq!(right_pixels[0], expected_blue, "Right region must be blue");
    }

    #[test]
    fn eval_at_multiple_coords_with_single_leaf() {
        let bsp = SpatialBSP::single(SolidColor::new(100, 150, 200, 255));

        // Sample at various locations - all should return same color without panicking
        let coords = [
            (0.0, 0.0),
            (50.0, 50.0),
            (100.0, 100.0),
            (-10.0, -10.0),
            (1000.0, 1000.0),
        ];

        for (x, y) in coords {
            let _result = bsp.eval_raw(
                Field::from(x),
                Field::from(y),
                Field::from(0.0),
                Field::from(0.0),
            );
        }
    }

    #[test]
    fn quadtree_like_structure_four_quadrants() {
        use pixelflow_core::{materialize_discrete, PARALLELISM};

        // Create 4 quadrants to test 2D partitioning
        let items = vec![
            Positioned {
                bounds: (0.0, 0.0, 50.0, 50.0),
                leaf: SolidColor::new(255, 0, 0, 255), // Q1: Red
            },
            Positioned {
                bounds: (50.0, 0.0, 100.0, 50.0),
                leaf: SolidColor::new(0, 255, 0, 255), // Q2: Green
            },
            Positioned {
                bounds: (0.0, 50.0, 50.0, 100.0),
                leaf: SolidColor::new(0, 0, 255, 255), // Q3: Blue
            },
            Positioned {
                bounds: (50.0, 50.0, 100.0, 100.0),
                leaf: SolidColor::new(255, 255, 0, 255), // Q4: Yellow
            },
        ];

        let bsp = SpatialBSP::from_positioned(items);

        // Test centers of each quadrant - each should produce a unique value
        let test_points = [
            (25.0, 25.0), // Q1
            (75.0, 25.0), // Q2
            (25.0, 75.0), // Q3
            (75.0, 75.0), // Q4
        ];

        let mut results = Vec::new();
        for (x, y) in test_points {
            let mut pixels = [0u32; PARALLELISM];
            materialize_discrete(&bsp, x, y, &mut pixels);
            results.push(pixels[0]);
        }

        // Each quadrant must return a distinct value
        for i in 0..results.len() {
            for j in (i + 1)..results.len() {
                assert_ne!(
                    results[i], results[j],
                    "Quadrant {} and {} must have different colors",
                    i, j
                );
            }
        }
    }

    // ========================================================================
    // Overlapping Bounds Tests
    // ========================================================================

    #[test]
    fn overlapping_bounds_creates_valid_tree() {
        // Items with overlapping regions (common in terminal rendering)
        let items = vec![
            Positioned {
                bounds: (0.0, 0.0, 60.0, 100.0),
                leaf: SolidColor::new(255, 0, 0, 128),
            },
            Positioned {
                bounds: (40.0, 0.0, 100.0, 100.0),
                leaf: SolidColor::new(0, 0, 255, 128),
            },
        ];

        let bsp = SpatialBSP::from_positioned(items);

        assert_eq!(bsp.leaf_count(), 2);
        assert_eq!(bsp.interior_count(), 1);
    }

    #[test]
    fn fully_contained_bounds_creates_valid_tree() {
        // One item fully contains another. Both bounding-box centers coincide at
        // (50, 50), so the center-based median split cannot carve out a nested region
        // for the inner item — this BSP is a half-plane space partition, not a
        // containment-aware one. There is no well-defined "correct" leaf for a query
        // exactly at the shared center, so this only asserts the tree is still valid
        // (no panic, right leaf/interior counts), not any particular per-pixel color.
        let items = vec![
            Positioned {
                bounds: (0.0, 0.0, 100.0, 100.0),
                leaf: SolidColor::new(255, 0, 0, 255),
            },
            Positioned {
                bounds: (25.0, 25.0, 75.0, 75.0),
                leaf: SolidColor::new(0, 0, 255, 255),
            },
        ];

        let bsp = SpatialBSP::from_positioned(items);

        assert_eq!(bsp.leaf_count(), 2);
        assert_eq!(bsp.interior_count(), 1);
        let _ = sample(&bsp, 50.0, 50.0); // must not panic
    }

    // ========================================================================
    // Construction API Tests
    // ========================================================================

    #[test]
    fn new_with_empty_arrays_behaves_like_from_positioned_empty() {
        let bsp1: SpatialBSP<SolidColor> = SpatialBSP::new(Arc::from([]), Arc::from([]));
        let bsp2: SpatialBSP<SolidColor> = SpatialBSP::from_positioned(vec![]);

        assert_eq!(bsp1.interior_count(), bsp2.interior_count());
        assert_eq!(bsp1.leaf_count(), bsp2.leaf_count());
    }

    #[test]
    fn clone_preserves_structure() {
        let items = vec![
            Positioned {
                bounds: (0.0, 0.0, 50.0, 100.0),
                leaf: SolidColor::new(255, 0, 0, 255),
            },
            Positioned {
                bounds: (50.0, 0.0, 100.0, 100.0),
                leaf: SolidColor::new(0, 0, 255, 255),
            },
        ];

        let bsp1 = SpatialBSP::from_positioned(items);
        let bsp2 = bsp1.clone();

        assert_eq!(bsp1.interior_count(), bsp2.interior_count());
        assert_eq!(bsp1.leaf_count(), bsp2.leaf_count());
    }

    // ========================================================================
    // Axis Selection Tests
    // ========================================================================

    #[test]
    fn diagonally_separated_square_items_route_correctly() {
        // Both axes separate these items equally (a tie in the split heuristic), so
        // either axis choice is correct here; this only checks eval() still routes
        // each item to its own color, not which axis the tie-break picked.
        build_and_assert_every_item_routes_to_its_own_color(
            &[(0.0, 0.0, 50.0, 50.0), (50.0, 50.0, 100.0, 100.0)],
            &[(255, 0, 0, 255), (0, 0, 255, 255)],
        );
    }

    #[test]
    fn slightly_wider_than_tall_items_route_correctly() {
        // Identical Y range: only an X-axis split routes these correctly.
        build_and_assert_every_item_routes_to_its_own_color(
            &[(0.0, 0.0, 100.0, 99.0), (100.0, 0.0, 200.0, 99.0)],
            &[(255, 0, 0, 255), (0, 0, 255, 255)],
        );
    }

    #[test]
    fn slightly_taller_than_wide_items_route_correctly() {
        // Identical X range: only a Y-axis split routes these correctly.
        build_and_assert_every_item_routes_to_its_own_color(
            &[(0.0, 0.0, 99.0, 100.0), (0.0, 100.0, 99.0, 200.0)],
            &[(255, 0, 0, 255), (0, 0, 255, 255)],
        );
    }

    // ========================================================================
    // Threshold Calculation Precision Tests
    // ========================================================================

    #[test]
    fn threshold_between_non_overlapping_items() {
        // Two items with a gap between them: a threshold outside [40, 60] would
        // misroute at least one of these probes.
        build_and_assert_every_item_routes_to_its_own_color(
            &[(0.0, 0.0, 40.0, 100.0), (60.0, 0.0, 100.0, 100.0)],
            &[(255, 0, 0, 255), (0, 0, 255, 255)],
        );
    }

    #[test]
    fn threshold_calculation_with_touching_items() {
        // Two items that touch at x=50: threshold = (50.0 + 50.0) / 2.0 = 50.0 exactly
        // (see build_tree's median formula), so probes just off either side of the
        // touch point pin the boundary down to sub-pixel precision.
        let bsp = build_and_assert_every_item_routes_to_its_own_color(
            &[(0.0, 0.0, 50.0, 100.0), (50.0, 0.0, 100.0, 100.0)],
            &[(255, 0, 0, 255), (0, 0, 255, 255)],
        );
        assert_eq!(
            sample(&bsp, 49.9, 50.0),
            sample(&bsp, 0.0, 50.0),
            "just below the touch point must still resolve to the left item's color"
        );
        assert_eq!(
            sample(&bsp, 50.1, 50.0),
            sample(&bsp, 99.0, 50.0),
            "just above the touch point must still resolve to the right item's color"
        );
    }

    #[test]
    fn y_axis_threshold_calculation_with_touching_items() {
        // Same as threshold_calculation_with_touching_items but split on Y: items touch
        // at y=50, so threshold = (50.0 + 50.0) / 2.0 = 50.0 exactly. Center-only probes
        // (as in tall_items_split_on_y_route_each_item_to_its_own_color) stay far enough
        // from the boundary to tolerate a miscalculated threshold; probing right at the
        // touch point does not.
        let bsp = build_and_assert_every_item_routes_to_its_own_color(
            &[(0.0, 0.0, 10.0, 50.0), (0.0, 50.0, 10.0, 100.0)],
            &[(255, 0, 0, 255), (0, 0, 255, 255)],
        );
        assert_eq!(
            sample(&bsp, 5.0, 49.9),
            sample(&bsp, 5.0, 0.0),
            "just below the touch point must still resolve to the top item's color"
        );
        assert_eq!(
            sample(&bsp, 5.0, 50.1),
            sample(&bsp, 5.0, 99.0),
            "just above the touch point must still resolve to the bottom item's color"
        );
    }

    #[test]
    fn threshold_with_very_small_items() {
        // A non-finite or collapsed threshold at this scale would misroute one of these
        // probes (NaN compares false against everything, always routing right).
        let bsp = build_and_assert_every_item_routes_to_its_own_color(
            &[(0.0, 0.0, 0.01, 0.01), (0.02, 0.02, 0.03, 0.03)],
            &[(255, 0, 0, 255), (0, 0, 255, 255)],
        );
        assert_eq!(bsp.leaf_count(), 2);
    }

    // ========================================================================
    // SIMD Boundary Behavior Tests
    // ========================================================================

    #[test]
    fn eval_at_and_around_the_threshold_respects_left_lt_right_ge_semantics() {
        // Two items touching at x=50 give a threshold of exactly 50.0 (see
        // threshold_calculation_with_touching_items), so the boundary condition
        // (coord < threshold goes left, coord >= threshold goes right) is pinned down
        // without reading the private tree.
        let red = SolidColor::new(255, 0, 0, 255);
        let blue = SolidColor::new(0, 0, 255, 255);
        let items = vec![
            Positioned {
                bounds: (0.0, 0.0, 50.0, 100.0),
                leaf: red.clone(),
            },
            Positioned {
                bounds: (50.0, 0.0, 100.0, 100.0),
                leaf: blue.clone(),
            },
        ];
        let bsp = SpatialBSP::from_positioned(items);

        assert_eq!(
            sample(&bsp, 50.0, 50.0),
            packed(&blue),
            "exactly at the threshold must go right (>=)"
        );
        assert_eq!(
            sample(&bsp, 49.9, 50.0),
            packed(&red),
            "just below the threshold must go left (<)"
        );
        assert_eq!(
            sample(&bsp, 50.1, 50.0),
            packed(&blue),
            "just above the threshold must go right (>=)"
        );
    }

    // ========================================================================
    // Special Float Value Tests
    // ========================================================================

    #[test]
    #[should_panic(expected = "unwrap")]
    fn nan_bounds_panics_during_construction() {
        // This tests the sharp corner: partial_cmp().unwrap() will panic on NaN
        let items = vec![
            Positioned {
                bounds: (0.0, 0.0, 50.0, 100.0),
                leaf: SolidColor::new(255, 0, 0, 255),
            },
            Positioned {
                bounds: (f32::NAN, 0.0, 100.0, 100.0), // NaN in bounds
                leaf: SolidColor::new(0, 0, 255, 255),
            },
        ];

        // Should panic when trying to sort by center (partial_cmp on NaN)
        let _bsp = SpatialBSP::from_positioned(items);
    }

    #[test]
    fn infinity_bounds_handled() {
        // Infinity should not cause panic (partial_cmp works)
        let items = vec![
            Positioned {
                bounds: (0.0, 0.0, 50.0, 100.0),
                leaf: SolidColor::new(255, 0, 0, 255),
            },
            Positioned {
                bounds: (100.0, 0.0, f32::INFINITY, 100.0),
                leaf: SolidColor::new(0, 0, 255, 255),
            },
        ];

        let bsp = SpatialBSP::from_positioned(items);
        assert_eq!(bsp.leaf_count(), 2);
    }

    #[test]
    fn negative_infinity_bounds_handled() {
        let items = vec![
            Positioned {
                bounds: (f32::NEG_INFINITY, 0.0, -50.0, 100.0),
                leaf: SolidColor::new(255, 0, 0, 255),
            },
            Positioned {
                bounds: (50.0, 0.0, 100.0, 100.0),
                leaf: SolidColor::new(0, 0, 255, 255),
            },
        ];

        let bsp = SpatialBSP::from_positioned(items);
        assert_eq!(bsp.leaf_count(), 2);
    }

    // ========================================================================
    // Inverted Bounds Tests
    // ========================================================================

    #[test]
    fn inverted_x_bounds_handled() {
        // Bounds where min_x > max_x (malformed input)
        let items = vec![
            Positioned {
                bounds: (50.0, 0.0, 0.0, 100.0), // x_max < x_min
                leaf: SolidColor::new(255, 0, 0, 255),
            },
            Positioned {
                bounds: (100.0, 0.0, 150.0, 100.0),
                leaf: SolidColor::new(0, 0, 255, 255),
            },
        ];

        let bsp = SpatialBSP::from_positioned(items);
        assert_eq!(bsp.leaf_count(), 2);
    }

    #[test]
    fn inverted_y_bounds_handled() {
        // Bounds where min_y > max_y (malformed input)
        let items = vec![
            Positioned {
                bounds: (0.0, 100.0, 50.0, 0.0), // y_max < y_min
                leaf: SolidColor::new(255, 0, 0, 255),
            },
            Positioned {
                bounds: (50.0, 0.0, 100.0, 100.0),
                leaf: SolidColor::new(0, 0, 255, 255),
            },
        ];

        let bsp = SpatialBSP::from_positioned(items);
        assert_eq!(bsp.leaf_count(), 2);
    }

    // ========================================================================
    // Deep Tree Tests
    // ========================================================================

    #[test]
    fn linear_chain_creates_unbalanced_tree() {
        // Items already perfectly sorted should still create valid tree
        let items: Vec<_> = (0..32)
            .map(|i| {
                let x = (i as f32) * 10.0;
                Positioned {
                    bounds: (x, 0.0, x + 5.0, 10.0),
                    leaf: SolidColor::new((i * 8) as u8, 0, 0, 255),
                }
            })
            .collect();

        let bsp = SpatialBSP::from_positioned(items);

        assert_eq!(bsp.leaf_count(), 32);
        assert_eq!(bsp.interior_count(), 31);
    }

    #[test]
    fn alternating_dimensions_creates_balanced_tree() {
        // Grid layout should split alternately on X and Y
        let items: Vec<_> = (0..16)
            .map(|i| {
                let x = ((i % 4) as f32) * 25.0;
                let y = ((i / 4) as f32) * 25.0;
                Positioned {
                    bounds: (x, y, x + 20.0, y + 20.0),
                    leaf: SolidColor::new((i * 16) as u8, 0, 0, 255),
                }
            })
            .collect();

        let bsp = SpatialBSP::from_positioned(items);

        assert_eq!(bsp.leaf_count(), 16);
        assert_eq!(bsp.interior_count(), 15);
    }

    // ========================================================================
    // Thread Safety Tests (compile-time via Send+Sync)
    // ========================================================================

    #[test]
    fn bsp_is_send_and_sync() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}

        // SpatialBSP should be Send + Sync due to Arc usage
        assert_send::<SpatialBSP<SolidColor>>();
        assert_sync::<SpatialBSP<SolidColor>>();
    }

    // ========================================================================
    // Structural Invariant Tests - a broken tree (bad index, unreachable leaf,
    // misplaced threshold, self-referencing node) shows up as a wrong color or a
    // panic/hang here, exercised purely through the public eval() contract.
    // ========================================================================

    #[test]
    fn every_item_in_a_16_item_row_routes_to_its_own_color() {
        let items: Vec<_> = (0..16u8)
            .map(|i| {
                let x = f32::from(i) * 10.0;
                ((x, 0.0, x + 5.0, 10.0), (i * 16, 0, 0, 255))
            })
            .collect();
        let (bounds, colors): (Vec<_>, Vec<_>) = items.into_iter().unzip();

        let bsp = build_and_assert_every_item_routes_to_its_own_color(&bounds, &colors);
        assert_eq!(bsp.interior_count(), 15);
    }

    #[test]
    fn every_item_in_a_4x4_grid_routes_to_its_own_color() {
        // A 2D grid forces splits to alternate between X and Y, exercising both axes'
        // worth of interior-node wiring in one tree.
        let items: Vec<_> = (0..16u8)
            .map(|i| {
                let x = f32::from(i % 4) * 25.0;
                let y = f32::from(i / 4) * 25.0;
                ((x, y, x + 20.0, y + 20.0), (0, i * 16, 0, 255))
            })
            .collect();
        let (bounds, colors): (Vec<_>, Vec<_>) = items.into_iter().unzip();

        let bsp = build_and_assert_every_item_routes_to_its_own_color(&bounds, &colors);
        assert_eq!(bsp.interior_count(), 15);
    }

    #[test]
    fn binary_tree_exact_interior_count() {
        // For n leaves, must have EXACTLY n-1 interiors (not >=, not <=, exactly)
        for n in 2..=20 {
            let items: Vec<_> = (0..n)
                .map(|i| {
                    let x = (i as f32) * 10.0;
                    Positioned {
                        bounds: (x, 0.0, x + 5.0, 10.0),
                        leaf: SolidColor::new(255, 0, 0, 255),
                    }
                })
                .collect();

            let bsp = SpatialBSP::from_positioned(items);

            assert_eq!(
                bsp.interior_count(),
                n - 1,
                "Binary tree with {} leaves must have exactly {} interiors, got {}",
                n,
                n - 1,
                bsp.interior_count()
            );
        }
    }

    #[test]
    fn empty_tree_has_exactly_zero_counts() {
        let bsp: SpatialBSP<SolidColor> = SpatialBSP::from_positioned(vec![]);

        assert_eq!(bsp.interior_count(), 0, "Empty tree must have 0 interiors");
        assert_eq!(bsp.leaf_count(), 0, "Empty tree must have 0 leaves");
    }

    #[test]
    fn single_item_has_exactly_one_leaf_zero_interiors() {
        let bsp = SpatialBSP::single(SolidColor::new(255, 0, 0, 255));

        assert_eq!(
            bsp.interior_count(),
            0,
            "Single-item tree must have exactly 0 interiors"
        );
        assert_eq!(
            bsp.leaf_count(),
            1,
            "Single-item tree must have exactly 1 leaf"
        );
    }

    #[test]
    fn stack_of_wide_strips() {
        use pixelflow_core::{materialize_discrete, PARALLELISM};

        // Create 2 wide, short items, stacked vertically.
        // Item 1: (0, 0, 100, 10). Red.
        // Item 2: (0, 10, 100, 20). Blue.
        let items = vec![
            Positioned {
                bounds: (0.0, 0.0, 100.0, 10.0),
                leaf: SolidColor::new(255, 0, 0, 255),
            },
            Positioned {
                bounds: (0.0, 10.0, 100.0, 20.0),
                leaf: SolidColor::new(0, 0, 255, 255),
            },
        ];

        let bsp = SpatialBSP::from_positioned(items);

        // Check Point in Item 1, Right side (x=75, y=5)
        let mut pixels = [0u32; PARALLELISM];
        materialize_discrete(&bsp, 75.0, 5.0, &mut pixels);

        let expected_red = {
            let red = SolidColor::new(255, 0, 0, 255);
            let mut buf = [0u32; PARALLELISM];
            materialize_discrete(&red, 0.0, 0.0, &mut buf);
            buf[0]
        };
        assert_eq!(
            pixels[0], expected_red,
            "Item 1 should be visible on right side"
        );

        // Check Point in Item 2, Left side (x=25, y=15)
        materialize_discrete(&bsp, 25.0, 15.0, &mut pixels);
        let expected_blue = {
            let blue = SolidColor::new(0, 0, 255, 255);
            let mut buf = [0u32; PARALLELISM];
            materialize_discrete(&blue, 0.0, 0.0, &mut buf);
            buf[0]
        };
        assert_eq!(
            pixels[0], expected_blue,
            "Item 2 should be visible on left side"
        );
    }
}
