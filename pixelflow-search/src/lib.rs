extern crate alloc;

pub mod egraph;
pub mod search;

// Re-export core algebraic types
pub use search::algebra::{Category, Morphism, Valuation};
pub use search::approx::{Network, Manifold, Dim256, Context, Features};
