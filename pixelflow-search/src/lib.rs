extern crate alloc;

pub mod egraph;
pub mod search;

// Training data generation and evaluation
pub mod eval;
pub mod features;

// Re-export core algebraic types
pub use search::algebra::{Category, Morphism, Valuation};
pub use search::approx::{Network, Manifold, Dim256, Context, Features};

// Re-export training types for convenience
pub use eval::{DataGenerator, DataGenConfig, TrainingSample, DatasetStats, BenchConfig};
pub use features::{HalfEPFeature, OpType, extract_features};