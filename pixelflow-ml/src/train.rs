//! Training machinery and loss functions.

use alloc::vec::Vec;

pub enum Loss {
    MeanSquaredError,
}

impl Loss {
    pub fn compute(&self, pred: f32, target: f32) -> f32 {
        match self {
            Loss::MeanSquaredError => {
                let diff = pred - target;
                diff * diff
            }
        }
    }

    pub fn derivative(&self, pred: f32, target: f32) -> f32 {
        match self {
            Loss::MeanSquaredError => 2.0 * (pred - target),
        }
    }
}

pub struct Trainer {
    pub learning_rate: f32,
}

impl Trainer {
    pub fn new(lr: f32) -> Self {
        Self { learning_rate: lr }
    }
}
