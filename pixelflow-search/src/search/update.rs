//! WeightUpdate implementations.

use crate::nnue::ExprNnue;
use super::{WeightUpdate, Gradients};

/// SGD weight update.
pub struct SgdUpdate<'a> {
    model: &'a mut ExprNnue,
}

impl<'a> SgdUpdate<'a> {
    pub fn new(model: &'a mut ExprNnue) -> Self {
        Self { model }
    }
}

impl<'a> WeightUpdate for SgdUpdate<'a> {
    fn apply(&mut self, grads: &Gradients, lr: f32) {
        // Apply mask MLP gradients
        if let Some(ref mask_grad) = grads.mask_mlp_grad {
            for i in 0..25 {
                for j in 0..16 {
                    self.model.mask_mlp_w1[i][j] -= lr * mask_grad.w1[i][j];
                }
            }
            for j in 0..16 {
                self.model.mask_mlp_b1[j] -= lr * mask_grad.b1[j];
            }
            for j in 0..16 {
                for k in 0..24 {
                    self.model.mask_mlp_w2[j][k] -= lr * mask_grad.w2[j][k];
                }
            }
            for k in 0..24 {
                self.model.mask_mlp_b2[k] -= lr * mask_grad.b2[k];
            }
        }

        // Apply interaction gradients
        if let Some(ref int_grad) = grads.interaction_grad {
            for i in 0..24 {
                for j in 0..24 {
                    self.model.interaction[i][j] -= lr * int_grad[i][j];
                }
            }
        }

        // Apply rule bias gradients
        if let Some(ref bias_grad) = grads.rule_bias_grad {
            for (i, &g) in bias_grad.iter().enumerate() {
                if i < self.model.mask_rule_bias.len() {
                    self.model.mask_rule_bias[i] -= lr * g;
                }
            }
        }

        // Apply value MLP gradients
        if let Some(ref value_grad) = grads.value_mlp_grad {
            for i in 0..24 {
                for j in 0..16 {
                    self.model.value_mlp_w1[i][j] -= lr * value_grad.w1[i][j];
                }
            }
            for j in 0..16 {
                self.model.value_mlp_b1[j] -= lr * value_grad.b1[j];
            }
            for j in 0..16 {
                self.model.value_mlp_w2[j] -= lr * value_grad.w2[j];
            }
            self.model.value_mlp_b2 -= lr * value_grad.b2;
        }
    }
}

/// No-op weight update (for inference).
pub struct NoUpdate;

impl WeightUpdate for NoUpdate {
    fn apply(&mut self, _grads: &Gradients, _lr: f32) {
        // Do nothing
    }
}
