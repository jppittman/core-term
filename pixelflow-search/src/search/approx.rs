//! Functional Approximation of Value Functions.
//!
//! Implements Efficiently Updatable Neural Networks (NNUE) as a differential
//! mapping from the Feature space to the Value space.

use crate::search::algebra::{Morphism};
use std::marker::PhantomData;

/// Type-level constant for dimensions to enforce shape safety.
pub trait Dim {
    const SIZE: usize;
}

/// Standard NNUE Hidden Dimension (256).
pub struct Dim256;
impl Dim for Dim256 { const SIZE: usize = 256; }

/// A dense vector in the accumulator space (the hidden layer).
/// We wrap this to provide algebraic operations.
#[derive(Debug)]
pub struct Manifold<D: Dim> {
    data: Vec<i16>, // Fixed size D::SIZE, specialized handling
    _dim: PhantomData<D>,
}

impl<D: Dim> Clone for Manifold<D> {
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            _dim: PhantomData,
        }
    }
}

impl<D: Dim> Default for Manifold<D> {
    fn default() -> Self {
        Self {
            data: vec![0; D::SIZE],
            _dim: PhantomData,
        }
    }
}

impl<D: Dim> Manifold<D> {
    pub fn new(bias: &[i16]) -> Self {
        assert_eq!(bias.len(), D::SIZE);
        Self {
            data: bias.to_vec(),
            _dim: PhantomData,
        }
    }

    /// Pure functional update: f(v, delta) -> v'
    /// "Addition" in the feature space corresponds to accumulating weights.
    pub fn add(&self, weights: &[i16]) -> Self {
        let mut next = self.clone();
        for (i, &w) in weights.iter().enumerate().take(D::SIZE) {
            next.data[i] = next.data[i].saturating_add(w);
        }
        next
    }

    /// Pure functional inverse update.
    pub fn sub(&self, weights: &[i16]) -> Self {
        let mut next = self.clone();
        for (i, &w) in weights.iter().enumerate().take(D::SIZE) {
            next.data[i] = next.data[i].saturating_sub(w);
        }
        next
    }
}

/// A Network is a static function representing the learned parameters.
/// It maps a (sparse) State to a Score.
pub struct Network<const INPUT: usize, Hidden: Dim> {
    // Flattened feature weights: [INPUT * Hidden::SIZE]
    feature_weights: Vec<i16>,
    feature_bias: Vec<i16>,
    output_weights: Vec<i16>,
    output_bias: i16,
    _hidden: PhantomData<Hidden>,
}

impl<const I: usize, H: Dim> Network<I, H> {
    pub fn new() -> Self {
        Self {
            feature_weights: vec![0; I * H::SIZE],
            feature_bias: vec![0; H::SIZE],
            output_weights: vec![0; H::SIZE],
            output_bias: 0,
            _hidden: PhantomData,
        }
    }

    pub fn get_weights(&self, feature_idx: usize) -> &[i16] {
        let start = feature_idx * H::SIZE;
        &self.feature_weights[start..start + H::SIZE]
    }

    /// Forward pass: Project the manifold point to a scalar value (score).
    /// Uses clipped ReLU activation.
    pub fn project(&self, manifold: &Manifold<H>) -> f32 {
        let mut sum: i32 = self.output_bias as i32;
        const QA: i32 = 255;
        const QB: i32 = 64;

        for (i, &val) in manifold.data.iter().enumerate() {
            // Activation: Clipped ReLU [0, 127] (usually) or simple max(0, x)
            // Here we assume standard NNUE quantization.
            let activation = val.max(0) as i32; 
            sum += activation * (self.output_weights[i] as i32);
        }

        (sum as f32) / (QA as f32 * QB as f32)
    }
}

/// To use the Network as a Valuation, the Category's Object must implement `Features`.
/// This connects the abstract Category to the concrete Vector Space.
pub trait Features {
    /// Returns indices of active features.
    fn active_features(&self) -> Vec<usize>;
    
    /// Returns the difference in features relative to a previous state.
    /// (Added, Removed)
    fn feature_diff(&self, other: &Self) -> (Vec<usize>, Vec<usize>);
}

/// Differential Valuation Implementation.
/// Note: This requires the State (Object) to hold the Accumulator (Manifold) 
/// or for us to track it.
/// 
/// In a pure functional design, `eval` is stateless. But `eval_diff` implies
/// we know the previous value.
/// 
/// We define a `Context` wrapper that pairs a State with its Manifold.
#[derive(Clone, Debug)]
pub struct Context<O, H: Dim> {
    pub object: O,
    pub manifold: Manifold<H>,
}

impl<const I: usize, H: Dim> Network<I, H> {
    /// Bootstraps a context from a raw object.
    pub fn bootstrap<O: Features>(&self, object: O) -> Context<O, H> {
        let mut manifold = Manifold::<H>::new(&self.feature_bias);
        for feat in object.active_features() {
            if feat < I {
                manifold = manifold.add(self.get_weights(feat));
            }
        }
        Context { object, manifold }
    }

    /// Differential update: C' = C + m
    pub fn apply_diff<O, M>(&self, ctx: &Context<O, H>, morphism: &M) -> Context<O, H>
    where
        O: Features + Clone,
        M: Morphism<Object = O>,
    {
        let next_object = morphism.apply(&ctx.object);
        
        // Calculate feature diffs (Added, Removed)
        // This relies on the domain knowledge that M implies a specific change.
        // For efficiency, we'd want M to directly provide the feature delta,
        // but here we derive it from the objects for generality.
        let (added, removed) = next_object.feature_diff(&ctx.object);

        let mut next_manifold = ctx.manifold.clone();

        for feat in removed {
            if feat < I {
                next_manifold = next_manifold.sub(self.get_weights(feat));
            }
        }
        for feat in added {
            if feat < I {
                next_manifold = next_manifold.add(self.get_weights(feat));
            }
        }

        Context {
            object: next_object,
            manifold: next_manifold,
        }
    }
}
