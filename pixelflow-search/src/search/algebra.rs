//! Algebraic definitions for state space search.
//!
//! We frame the search problem as navigating a Category where:
//! - **Objects** are States (points in the search space).
//! - **Morphisms** are Transitions (moves/rewrites).
//! - **Composition** represents a sequence of moves.

use std::fmt::Debug;
use std::hash::Hash;

/// A Morphism represents a transformation between States.
/// In a game, this is a "Move". In optimization, a "Rewrite".
pub trait Morphism: Clone + Debug + PartialEq + Eq + Hash {
    /// The domain/codomain type (the State).
    type Object;

    /// Apply the morphism to an object to obtain the codomain object.
    /// This is the arrow mapping: f: A -> B
    fn apply(&self, source: &Self::Object) -> Self::Object;
}

/// A Category defines the structure of the search space.
/// It provides the objects and the morphisms available from a given object.
pub trait Category {
    type Object: Clone + Debug + PartialEq + Eq + Hash;
    type Morphism: Morphism<Object = Self::Object>;

    /// The set of morphisms (arrows) originating from a given object.
    /// hom(obj, _)
    fn hom(&self, object: &Self::Object) -> Vec<Self::Morphism>;

    /// A predicate determining if an object is "terminal" (no outgoing morphisms of interest,
    /// or a fixed point).
    fn is_terminal(&self, object: &Self::Object) -> bool;
}

/// A Functor that maps the category of States to a category of Values (Scores).
/// This represents the objective function or heuristic evaluation.
pub trait Valuation<C: Category> {
    type Value: PartialOrd + Copy + Debug;

    /// Map an object to a value.
    fn eval(&self, object: &C::Object) -> Self::Value;
}

/// A Differential Valuation allows computing V(B) given V(A) and the morphism f: A -> B.
/// This enables incremental updates (like NNUE).
pub trait DifferentialValuation<C: Category>: Valuation<C> {
    /// Compute eval(apply(m, s)) given eval(s) and m, potentially faster than full eval.
    fn eval_diff(&self, current_val: &Self::Value, morphism: &C::Morphism) -> Self::Value;
}
